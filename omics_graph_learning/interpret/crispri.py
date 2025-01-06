#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to evaluate in-silico perturbations matching CRISPRi experiments in
K562.

ANALYSES TO PERFORM
# 1 - we expect the tuples with TRUE to have a higher magnitude of change than
random perturbations
# 2 - for each tuple, we expect those with TRUE to affect prediction at a higher
magnitude than FALSE
# 3 - for each tuple, we expect those with TRUE to negatively affect prediction
(recall)
# 4 - for the above, compare the change that randomly chosen FALSE would
postively or negatively affect the prediction
"""


import argparse
from collections import deque
import json
import logging
import math
from pathlib import Path
import pickle
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd
import pybedtools
from scipy import stats  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import from_networkx  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.architecture_builder import build_gnn_architecture
from omics_graph_learning.combination_loss import CombinationLoss


class GNNTrainer:
    """Class to handle GNN training and evaluation.

    Methods
    --------
    train:
        Train GNN model on graph data via minibatching
    evaluate:
        Evaluate GNN model on validation or test set
    inference_all_neighbors:
        Evaluate GNN model on test set using all neighbors
    training_loop:
        Execute training loop for GNN model
    log_tensorboard_data:
        Log data to tensorboard on the last batch of an epoch.

    Examples:
    --------
    # instantiate trainer
    >>> trainer = GNNTrainer(
            model=model,
            device=device,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            tb_logger=tb_logger,
        )

    # train model
    >>> model, _, early_stop = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            model_dir=run_dir,
            args=args,
            min_epochs=min_epochs,
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        data: torch_geometric.data.Data,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]] = None,
    ) -> None:
        """Initialize model trainer."""
        self.model = model
        self.device = device
        self.data = data
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = CombinationLoss(alpha=0.8)

    def _forward_pass(
        self,
        data: torch_geometric.data.Data,
        mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Perform forward pass and compute losses and outputs."""
        data = data.to(self.device)

        regression_out, logits = self.model(
            x=data.x,
            edge_index=data.edge_index,
            mask=mask,
        )

        loss, regression_loss, classification_loss = self.criterion(
            regression_output=regression_out,
            regression_target=data.y,
            classification_output=logits,
            classification_target=data.class_labels,
            mask=mask,
        )

        # collect masked outputs and labels
        regression_out_masked = self._ensure_tensor_dim(regression_out[mask])
        labels_masked = self._ensure_tensor_dim(data.y[mask])

        classification_out_masked = self._ensure_tensor_dim(logits[mask])
        class_labels_masked = self._ensure_tensor_dim(data.class_labels[mask])

        return (
            loss,
            regression_loss,
            classification_loss,
            regression_out_masked,
            labels_masked,
            classification_out_masked,
            class_labels_masked,
        )

    def _train_single_batch(
        self,
        data: torch_geometric.data.Data,
        epoch: int,
        batch_idx: int,
        total_batches: int,
    ) -> Tuple[float, float, float]:
        """Train a single batch."""
        self.optimizer.zero_grad()

        # forward pass
        mask = data.train_mask_loss

        (
            loss,
            regression_loss,
            classification_loss,
            _,
            _,
            _,
            _,
        ) = self._forward_pass(data, mask)

        # backpropagation
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # check for NaN gradients
        self._check_for_nan_gradients()
        self.optimizer.step()

        # step warmup schedulers, if applicable
        if isinstance(self.scheduler, LRScheduler) and not isinstance(
            self.scheduler, ReduceLROnPlateau
        ):
            self.scheduler.step()

        batch_size_mask = int(mask.sum())
        return (
            loss.item() * batch_size_mask,
            regression_loss.item() * batch_size_mask,
            classification_loss.item() * batch_size_mask,
        )

    def _evaluate_single_batch(
        self,
        data: torch_geometric.data.Data,
        mask: str,
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a single batch."""
        mask = getattr(data, f"{mask}_mask_loss")
        if mask.sum() == 0:
            return (
                0.0,
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
            )
        # forward pass
        (
            loss,
            _,
            _,
            regression_out_masked,
            labels_masked,
            classification_out_masked,
            class_labels_masked,
        ) = self._forward_pass(data, mask)

        batch_size_mask = int(mask.sum())
        return (
            loss.item() * batch_size_mask,
            regression_out_masked.cpu(),
            labels_masked.cpu(),
            classification_out_masked.cpu(),
            class_labels_masked.cpu(),
        )

    def train(
        self,
        train_loader: torch_geometric.data.DataLoader,
        epoch: int,
        subset_batches: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Train GNN model on graph data"""
        self.model.train()
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(
            f"\nTraining {self.model.__class__.__name__} model @ epoch: {epoch} - "
        )

        total_loss = total_regression = total_classification = float(0)
        total_examples = 0
        total_batches = len(train_loader)
        for batch_idx, data in enumerate(train_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            loss, regression_loss, classification_loss = self._train_single_batch(
                data=data,
                epoch=epoch,
                batch_idx=batch_idx,
                total_batches=total_batches,
            )
            total_loss += loss
            total_regression += regression_loss
            total_classification += classification_loss
            total_examples += int(data.train_mask_loss.sum())
            pbar.update(1)

        pbar.close()
        final_loss = total_loss / total_examples if total_examples > 0 else 0.0
        final_regression = (
            total_regression / total_examples if total_examples > 0 else 0.0
        )
        final_classification = (
            total_classification / total_examples if total_examples > 0 else 0.0
        )
        return final_loss, final_regression, final_classification

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: torch_geometric.data.DataLoader,
        epoch: int,
        mask: str,
        subset_batches: Optional[int] = None,
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor, float, float]:
        """Base function for model evaluation or inference."""
        self.model.eval()
        pbar = tqdm(total=len(data_loader))
        pbar.set_description(
            f"\nEvaluating {self.model.__class__.__name__} model @ epoch: {epoch}"
        )

        total_loss = float(0)
        total_examples = 0
        regression_outs, regression_labels = [], []
        classification_outs, classification_labels = [], []

        for batch_idx, data in enumerate(data_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            loss, reg_out, reg_label, cls_out, cls_label = self._evaluate_single_batch(
                data=data,
                mask=mask,
            )
            total_loss += loss
            total_examples += int(getattr(data, f"{mask}_mask_loss").sum())

            regression_outs.append(reg_out)
            regression_labels.append(reg_label)
            classification_outs.append(cls_out)
            classification_labels.append(cls_label)

            pbar.update(1)

        pbar.close()
        average_loss = total_loss / total_examples if total_examples > 0 else 0.0

        return (
            torch.cat(regression_outs),
            torch.cat(regression_labels),
        )

    @torch.no_grad()
    def evaluate_single(
        self,
        data: torch_geometric.data.Data,
        mask: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a single data object."""
        self.model.eval()

        mask = getattr(data, f"{mask}_mask_loss")
        if mask.sum() == 0:
            return torch.tensor([]), torch.tensor([])

        data = data.to(self.device)
        # forward pass
        regression_out, logits = self.model(
            x=data.x,
            edge_index=data.edge_index,
            mask=mask,
        )

        # collect masked outputs and labels
        regression_out_masked = self._ensure_tensor_dim(regression_out[mask])
        labels_masked = self._ensure_tensor_dim(data.y[mask])

        return regression_out_masked.cpu(), labels_masked.cpu()

    @staticmethod
    def _compute_regression_metrics(
        regression_outs: List[torch.Tensor],
        regression_labels: List[torch.Tensor],
    ) -> Tuple[float, float]:
        """Compute RMSE and Pearson's R for regression task."""
        if not regression_outs or not regression_labels:
            return 0.0, 0.0

        predictions = torch.cat(regression_outs).squeeze()
        targets = torch.cat(regression_labels).squeeze()
        mse = F.mse_loss(predictions, targets)
        rmse = torch.sqrt(mse).item()
        pearson_r, _ = pearsonr(predictions.numpy(), targets.numpy())

        return rmse, pearson_r

    @staticmethod
    def _compute_classification_metrics(
        classification_outs: List[torch.Tensor],
        classification_labels: List[torch.Tensor],
    ) -> float:
        """Compute accuracy for classification task."""
        if not classification_outs or not classification_labels:
            return 0.0

        logits = torch.cat(classification_outs).squeeze()
        labels = torch.cat(classification_labels).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        return (preds == labels).float().mean().item()

    @staticmethod
    def _ensure_tensor_dim(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has the correct dimensions for evaluation."""
        tensor = tensor.squeeze()
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor


def load_model(
    checkpoint_file: str,
    map_location: torch.device,
    device: torch.device,
) -> nn.Module:
    """Load the model from a checkpoint.

    Args:
        checkpoint_file (str): Path to the model checkpoint file.
        map_location (str): Map location for loading the model.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The loaded model.
    """
    model = build_gnn_architecture(
        model="GAT",
        activation="gelu",
        in_size=44,
        embedding_size=200,
        out_channels=1,
        gnn_layers=2,
        shared_mlp_layers=2,
        heads=4,
        dropout_rate=0.4,
        residual="distinct_source",
        attention_task_head=False,
        train_dataset=None,
    )
    model = model.to(device)

    # load the model
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def load_graph(
    graph_path: str, graph_idxs_path: str, pyg_graph: str
) -> Tuple[Data, Dict[str, int]]:
    """Load the graph and its indices from pickle files.

    Args:
        graph_path (str): Path to the graph data file.
        graph_idxs_path (str): Path to the graph indices file.

    Returns:
        Tuple[Data, Dict[str, int]]: The graph data and index mapping.
    """
    with open(graph_path, "rb") as file:
        graph = pickle.load(file)

    with open(graph_idxs_path, "rb") as file:
        graph_idxs = pickle.load(file)

    with open(pyg_graph, "rb") as file:
        pyg = pickle.load(file)

    return graph, graph_idxs, pyg


def calculate_percent_fold_change(
    baseline_prediction: float, perturbation_prediction: float
) -> float:
    """Convert regression values back to TPM and calculate the percent fold
    change.
    """
    baseline_tpm = 2 ** (baseline_prediction - 0.25)
    perturbation_tpm = 2 ** (perturbation_prediction - 0.25)

    # calculate fold change
    fold_change = perturbation_tpm / baseline_tpm

    return (fold_change - 1) * 100


def calculate_log2_fold_change(
    baseline_prediction: float, perturbation_prediction: float
) -> float:
    """Calculate the log2 fold change from log2-transformed values."""
    log2_fold_change = perturbation_prediction - baseline_prediction
    return (2**log2_fold_change - 1) * 100


def get_subgraph(
    data: Data, graph: nx.Graph, target_node: int
) -> Tuple[Data, Dict[int, int]]:
    """Extracts the entire connected component containing the given target node
    from a PyG Data object. Requires the networkX representation of the graph for
    faster traversal.

    Returns:
        Data: A new PyG Data object containing the subgraph for the connected
        component.
    """
    # find the connected component containing the target node
    for component in nx.connected_components(graph):
        if target_node in component:
            subgraph_nodes = list(component)
            break
    else:
        raise ValueError(f"Target node {target_node} not found in the graph.")

    # create mapping from original node indices to subgraph node indices
    mapping = {node: i for i, node in enumerate(subgraph_nodes)}

    # create subgraph with relabeled nodes
    subgraph = graph.subgraph(subgraph_nodes).copy()
    subgraph = nx.relabel_nodes(subgraph, mapping)

    # convert to PyG Data object
    sub_data = from_networkx(subgraph)

    # create a tensor of original node indices in subgraph node index order
    n_id = torch.tensor(subgraph_nodes, dtype=torch.long)

    # copy features and labels from original data using n_id
    sub_data.x = data.x[n_id]
    sub_data.y = data.y[n_id]
    sub_data.class_labels = data.class_labels[n_id]

    # create mask for the target node
    target_node_subgraph_idx = mapping[target_node]
    sub_data.test_mask_loss = torch.zeros(sub_data.num_nodes, dtype=torch.bool)
    sub_data.test_mask_loss[target_node_subgraph_idx] = True

    return sub_data, mapping


def delete_node(data: Data, node_idx: int) -> Data:
    """
    Deletes a node from the PyG Data object.

    Args:
        data (Data): The PyG Data object.
        node_idx (int): The index of the node to delete in the subgraph.

    Returns:
        Data: A new PyG Data object with the specified node removed.
    """
    # ensure mask is on the same device as edge_index
    mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.edge_index.device)
    mask[node_idx] = False

    # filter nodes
    new_x = data.x[mask]
    new_y = data.y[mask]
    new_class_labels = data.class_labels[mask]

    # filter edges: keep edges where both nodes are not deleted
    edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    new_edge_index = data.edge_index[:, edge_mask]

    # reindex edges to account for removed nodes
    # create a mapping from old node indices to new node indices
    mapping = torch.full(
        (data.num_nodes,), -1, dtype=torch.long, device=data.edge_index.device
    )
    mapping[mask] = torch.arange(new_x.size(0), device=data.edge_index.device)

    # update edge_index with the new node indices
    new_edge_index = mapping[new_edge_index].to(torch.long)

    # remove any edges that may have invalid indices after mapping
    valid_edge_mask = (new_edge_index[0] >= 0) & (new_edge_index[1] >= 0)
    new_edge_index = new_edge_index[:, valid_edge_mask]

    # update the mask for the target node
    new_test_mask_loss = data.test_mask_loss[mask]

    return Data(
        x=new_x,
        edge_index=new_edge_index,
        y=new_y,
        class_labels=new_class_labels,
        test_mask_loss=new_test_mask_loss,
    )


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode to gene symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def _rename_tuple(
    input_tuple: Tuple[str, str, str],
    idxs_dict: Dict[str, int],
    gencode_lookup: Dict[str, str],
    tissue: Optional[str] = "k562",
) -> Union[Tuple[int, int, str], None]:
    """Rename a tuple by replacing gene symbol with corresponding Gencode ID and
    index from idxs_dict.

    Returns:
        (Tuple[int, int, str]) of the form (enhancer, gencode, flag)
    """
    enhancer, gene_symbol, flag = input_tuple

    # find the gencode ID for the gene symbol
    gencode_id = gencode_lookup.get(gene_symbol)
    if gencode_id is None:
        print(f"Gene symbol {gene_symbol} not found in Gencode lookup.")
        return None

    # construct the keys for lookup in the idxs_dict
    enhancer_key = f"{enhancer}_{tissue}"
    gencode_key = f"{gencode_id}_{tissue}"

    # fetch the indices from the idxs_dict
    enhancer_idx = idxs_dict.get(enhancer_key)
    gencode_idx = idxs_dict.get(gencode_key)

    if enhancer_idx is None or gencode_idx is None:
        print(f"Indices for {enhancer_key} or {gencode_key} not found in idxs_dict.")
        return None

    # return the new tuple format
    return (enhancer_idx, gencode_idx, flag)


def rename_tuples(
    tuple_list: List[Tuple[str, str, str]],
    idxs_dict: Dict[str, int],
    gencode_lookup: Dict[str, str],
) -> List[Tuple[int, int, str]]:
    """Rename a list of tuples."""
    return [_rename_tuple(tup, idxs_dict, gencode_lookup) for tup in tuple_list]


def load_crispri(
    crispr_benchmarks: str,
    enhancer_catalogue: str,
    tissue: str = "k562",
) -> Set[Tuple[str, str, str]]:
    """Take an intersect of the regulatory catalogue with the CRISPRi
    benchmarks, replacing the CRISPRi enhancers with those in the regulatory
    catalogue that overlap.

    Returns:
        Set[Tuple[str, str, str]]: Set of tuples containing the enhancer, gene,
        and regulation bool

    """

    def reorder(feature: Any) -> Any:
        """Reorder the features."""
        chrom = feature[6]
        start = feature[7]
        end = feature[8]
        name = feature[3]
        gene = feature[4]
        regulation = feature[5]
        return pybedtools.create_interval_from_list(
            [chrom, start, end, gene, regulation, name]
        )

    # load the file as a bedtool
    # _add_hash_if_missing(crispr_benchmarks)
    links = pybedtools.BedTool(crispr_benchmarks).cut([0, 1, 2, 3, 8, 19])

    # intersect with enhancer catalogue, but only keep the enhancer, gene, and
    # regulation bool
    links = links.intersect(enhancer_catalogue, wa=True, wb=True).each(reorder).saveas()

    return {(f"{link[0]}_{link[1]}_enhancer", link[3], link[4]) for link in links}


def filter_links_for_present_nodes(
    graph: Data, links: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """Filter CRISPRi links to only include those with nodes present in the
    graph.
    """
    num_nodes = graph.num_nodes

    # convert links to tensors
    links_tensor = torch.tensor([(tup[0], tup[1]) for tup in links], dtype=torch.long)

    # check which links have both nodes in the valid range
    valid_links_mask = (links_tensor[:, 0] < num_nodes) & (
        links_tensor[:, 1] < num_nodes
    )

    # apply mask
    valid_links_tensor = links_tensor[valid_links_mask]

    # convert back to a list of tuples with the original third element
    filtered_links = []
    valid_idx = 0
    for i, link in enumerate(links):
        if valid_links_mask[i]:
            filtered_links.append(
                (
                    int(valid_links_tensor[valid_idx][0]),
                    int(valid_links_tensor[valid_idx][1]),
                    link[2],
                )
            )
            valid_idx += 1

    return filtered_links


def create_crispri_dict(
    crispri_links: List[Tuple[int, int, str]]
) -> Dict[int, List[Tuple[int, int]]]:
    """Save all crispr links in a dictionary so we only do subgraph extraction
    once per gene."""
    result = {}
    for link in crispri_links:
        guide = link[1]
        if guide not in result:
            result[guide] = []
        result[guide].append((link[0], link[2]))
    return result


def create_full_neighbor_loader(
    data: Data, target_node: int, enhancer_nodes: List[int], num_layers: int = 10
) -> NeighborLoader:
    """Creates a NeighborLoader to sample neighbors, ensuring the target node
    and all associated enhancers are included in a single batch.

    Args:
        data (Data): The PyG Data object.
        target_node (int): The index of the target gene node.
        enhancer_nodes (List[int]): List of enhancer node indices associated with the gene.
        num_layers (int): Number of layers to sample.

    Returns:
        NeighborLoader: Configured NeighborLoader instance.
    """
    input_nodes = torch.tensor([target_node] + enhancer_nodes, dtype=torch.long)
    batch_size = len(input_nodes)

    return NeighborLoader(
        data=data,
        num_neighbors=[data.avg_edges] * num_layers,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=False,
    )


def main() -> None:
    """Try and see if we can recpitulate crispr stuff man."""
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(126)

    # load graph
    crispr_benchmarks = "EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    enhancer_catalogue = "enhancers_epimap_screen_overlap.bed"
    idx_file = "regulatory_only_k562_allcontacts_global_full_graph_idxs.pkl"

    # load the pytorch graph
    graph = torch.load("graph.pt")

    # create nx graph
    nx_graph = to_networkx(graph, to_undirected=True)

    # load IDXS
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    # get a dictionary of all enhancer idxs
    enhancer_idxs = {k: v for k, v in idxs.items() if "enhancer" in k}
    enhancer_orig_indices = set(enhancer_idxs.values())  # Set for faster lookup

    # load the model
    model = load_model("GAT_best_model.pt", device, device)

    # load crispr benchmarks
    crispri_links = load_crispri(crispr_benchmarks, enhancer_catalogue)
    gencode_lookup = load_gencode_lookup("gencode_to_genesymbol_lookup_table.txt")

    # rename the crispr links to node idxs
    renamed_crispri_links = rename_tuples(crispri_links, idxs, gencode_lookup)
    renamed_crispri_links = {link for link in renamed_crispri_links if link is not None}

    # filter the links for present nodes in the graph
    crispri_test = filter_links_for_present_nodes(
        graph, renamed_crispri_links
    )  # 11,147 E_G pairs
    crispri_test = create_crispri_dict(crispri_test)  # 1,868 genes

    # initialize dictionaries to store fold changes
    true_fold_changes = {}
    false_fold_changes = {}
    random_fold_changes = {}

    for gene_idx, enhancers in tqdm(crispri_test.items(), desc="Processing Genes"):
        print(f"Processing gene {gene_idx} with {len(enhancers)} enhancers...")

        enhancer_indices = [enhancer[0] for enhancer in enhancers]

        # create NeighborLoader for the gene and all its enhancers
        loader = create_full_neighbor_loader(
            data=graph,
            target_node=gene_idx,
            enhancer_nodes=enhancer_indices,
            num_layers=10,
        )

        # iterate over the loader to get the subgraph (only one batch per gene)
        for batch in loader:
            sub_data = batch.to(device)

            # map original node indices to subgraph node indices
            n_id = batch.n_id
            mapping = {
                int(orig_idx): i for i, orig_idx in enumerate(n_id.cpu().numpy())
            }

            target_subgraph_idx = mapping.get(gene_idx)
            if target_subgraph_idx is None:
                print(
                    f"Target gene {gene_idx} not found in the sampled subgraph. Skipping."
                )
                continue

            # create mask for the target node in the subgraph
            sub_data.test_mask_loss = torch.zeros(
                sub_data.num_nodes, dtype=torch.bool, device=device
            )
            sub_data.test_mask_loss[target_subgraph_idx] = True

            # initialize evaluator
            evaluator = GNNTrainer(
                model=model,
                device=device,
                data=sub_data,
            )

            # get baseline prediction
            out, label = evaluator.evaluate_single(
                data=sub_data,
                mask="test",
            )

            if out.numel() != 1 or label.numel() != 1:
                print(f"Unexpected number of outputs for gene {gene_idx}. Skipping.")
                continue

            baseline_prediction = out.item()
            target_label = label.item()

            # process each enhancer associated with the gene
            for enhancer in enhancers:
                enhancer_idx = enhancer[0]
                flag = enhancer[1]

                # check if enhancer is in subgraph
                enhancer_subgraph_idx = mapping.get(enhancer_idx)
                if enhancer_subgraph_idx is None:
                    print(
                        f"Enhancer {enhancer_idx} not found in subgraph for gene {gene_idx}. Skipping."
                    )
                    continue

                # delete enhancer node
                modified_subgraph = delete_node(sub_data, enhancer_subgraph_idx)

                # run inference on modified subgraph
                modified_evaluator = GNNTrainer(
                    model=model,
                    device=device,
                    data=modified_subgraph,
                )

                modified_out, _ = modified_evaluator.evaluate_single(
                    data=modified_subgraph,
                    mask="test",
                )

                if modified_out.numel() != 1:
                    print(
                        f"Unexpected number of outputs after deleting enhancer {enhancer_idx} for gene {gene_idx}. Skipping."
                    )
                    continue

                perturbation_prediction = modified_out.item()
                fold_change = calculate_log2_fold_change(
                    baseline_prediction, perturbation_prediction
                )

                # store fold change
                if flag == "TRUE":
                    true_fold_changes[(enhancer_idx, gene_idx)] = fold_change
                else:
                    false_fold_changes[(enhancer_idx, gene_idx)] = fold_change

                # proceed to delete a random enhancer (control)
                # identify eligible random enhancers
                enhancers_in_subgraph = [
                    i
                    for i, orig_idx in enumerate(n_id)
                    if orig_idx.item() in enhancer_orig_indices
                ]

                # exclude the target gene node and the current enhancer
                excluded_nodes = {target_subgraph_idx, enhancer_subgraph_idx}

                possible_random_enhancers = [
                    n for n in enhancers_in_subgraph if n not in excluded_nodes
                ]

                if possible_random_enhancers:
                    random_node_subgraph_idx = random.choice(possible_random_enhancers)
                    random_modified_subgraph = delete_node(
                        sub_data, random_node_subgraph_idx
                    )

                    # run inference on randomly modified subgraph
                    random_evaluator = GNNTrainer(
                        model=model,
                        device=device,
                        data=random_modified_subgraph,
                    )

                    random_out, _ = random_evaluator.evaluate_single(
                        data=random_modified_subgraph,
                        mask="test",
                    )

                    if random_out.numel() != 1:
                        print(
                            f"Unexpected number of outputs after deleting random enhancer {random_node_subgraph_idx} for gene {gene_idx}. Skipping."
                        )
                        continue

                    random_prediction = random_out.item()
                    fold_change_random = calculate_log2_fold_change(
                        baseline_prediction, random_prediction
                    )

                    # store fold change
                    random_fold_changes[(random_node_subgraph_idx, gene_idx)] = (
                        fold_change_random
                    )
                else:
                    print(
                        f"No eligible enhancer nodes available for random deletion in subgraph for gene {gene_idx}."
                    )

            break

    # save the fold changes
    with open("true_fc.pkl", "wb") as f:
        pickle.dump(true_fold_changes, f)

    with open("false_fc.pkl", "wb") as f:
        pickle.dump(false_fold_changes, f)

    with open("random_fc.pkl", "wb") as f:
        pickle.dump(random_fold_changes, f)

    print(f"True Fold Changes: {len(true_fold_changes)}")
    print(f"False Fold Changes: {len(false_fold_changes)}")
    print(f"Random Fold Changes: {len(random_fold_changes)}")


# In [4]: len(true_fold_changes)
# Out[4]: 613

# In [5]: len(false_fold_changes)
# Out[5]: 10117

# In [6]: len(random_fold_changes)
# Out[6]: 11002

# In [7]: len(crispri_links)
# Out[7]: 12030


if __name__ == "__main__":
    main()
