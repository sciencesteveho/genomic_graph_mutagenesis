#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to evaluate in-silico perturbations matching CRISPRi experiments in
K562.

ANALYSES TO PERFORM
# 1 - we expect the tuples with TRUE to have a higher magnitude of change than random perturbations
# 2 - for each tuple, we expect those with TRUE to affect prediction at a higher magnitude than FALSE
# 3 - for each tuple, we expect those with TRUE to negatively affect prediction (recall)
# 3 - for the above, compare the change that randomly chosen FALSE would postively or negatively affect the prediction
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
from omics_graph_learning.combination_loss import RMSEandBCELoss


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

        self.criterion = RMSEandBCELoss(alpha=0.8)

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

        # compute regression metrics
        rmse, pearson_r = self._compute_regression_metrics(
            regression_outs, regression_labels
        )

        # compute classification metrics
        accuracy = self._compute_classification_metrics(
            classification_outs, classification_labels
        )

        return (
            average_loss,
            rmse,
            torch.cat(regression_outs),
            torch.cat(regression_labels),
            pearson_r,
            accuracy,
        )

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


def calculate_fold_change(
    baseline_prediction: float, perturbation_prediction: float
) -> float:
    """Convert regression values back to TPM to calculate the fold change
    difference.
    """
    baseline_tpm = 2 ** (baseline_prediction - 0.25)
    perturbation_tpm = 2 ** (perturbation_prediction - 0.25)
    return perturbation_tpm / baseline_tpm


def get_subgraph(data: Data, graph: nx.Graph, target_node: int) -> Data:
    """Extracts the entire connected component containing the given target node
    from a PyG Data object. Requires the networkX representation of the graph for
    faster traversal.

    Returns:
        Data: A new PyG Data object containing the subgraph for the connected
        component.
    """
    # make a copy of graph to avoid modifying the original
    extract_graph = graph.copy()

    # make a mask for the target node
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[target_node] = True
    extract_graph.mask = mask

    # find the connected component containing the target node
    for component in nx.connected_components(extract_graph):
        if target_node in component:
            subgraph_nodes = component
            break

    # extract the subgraph from the original graph
    subgraph = graph.subgraph(subgraph_nodes)

    # convert the subgraph back to a PyG Data object
    sub_data = from_networkx(subgraph)

    # copy node features and labels from the original graph
    sub_data.x = data.x[list(subgraph_nodes)]
    sub_data.y = data.y[list(subgraph_nodes)]

    # inherit mask
    sub_data.test_mask_loss = mask[list(subgraph_nodes)]

    return sub_data


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


def perturb_crispri(
    graph: Data,
    crispri_links: Dict[int, List[Tuple[int, int]]],
    nx_graph: nx.Graph,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Initialize three dictionaries, TRUE, FALSE and RANDOM, to store the fold changes
    for each gene in the CRISPRi benchmarks.
    Default dict because we will be appending to the lists.

    For each gene in the crispr_dict:
        Get the subgraph
        Get the baseline prediction
        For each tuple in the crispr_dict[gene]:
            Check if tup[0] is in the subgraph
            If not, continue
            If true:
                Delete node tup[0] from the subgraph
                Get the prediction
                Calculate the fold change
                If tup[1] is TRUE:
                    Append fold change in the TRUE dict with the gene as key, and (tup[0], fold_change) as value
    """
    raise NotImplementedError


def make_loader(
    data: Data,
) -> NeighborLoader:
    """Make a neighbor loader for the data."""
    return NeighborLoader(
        data,
        num_neighbors=[-1],  # all neighbors at each hop
        batch_size=1,
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

    # Some for testing:
    (487024, 11855, "FALSE"),
    (242639, 106, "TRUE"),
    (486837, 11855, "FALSE"),
    (665979, 4811, "FALSE"),
    (301526, 2864, "FALSE"),
    (486838, 716, "FALSE"),
    (90267, 17948, "FALSE"),
    (412026, 14109, "TRUE"),
    (320736, 10356, "TRUE"),

    # get the subgraph
    # test = get_subgraph(graph, nx_graph, 11855)

    # for each loop,
    # get subgraph
    # run baseline inference on subgraph
    # delete target node
    # run new inference on subgraph
    # calculate fold change
    # delete random node
    # run new inference on subgraph
    # calculate fold change
    # save crispri dictionary with key (element, gene) value fold_change

    # iterate over crispri links
    # remember (idx, gene, flag)
    crispr_link = (242639, 106, "TRUE")
    for crispr_link in crispri_links:

        # get subgraph
        subgraph = get_subgraph(graph, nx_graph, crispr_link[1])

        # init evaluator
        evaluator = GNNTrainer(
            model=model,
            device=device,
            data=subgraph,
        )

        # get loader
        loader = make_loader(subgraph)

        # get baseline prediction
        _, _, out, label, _, _ = evaluator.evaluate(
            data_loader=loader,
            mask="test",
            epoch=0,
        )

        #

    # loss, rmse, outs, labels, pearson_r, accuracy = post_eval_trainer.evaluate(
    #     data_loader=data_loader,
    #     mask="test",
    #     epoch=0,
    # )

    # # save final eval
    # np.save(run_dir / "outs.npy", outs.numpy())
    # np.save(run_dir / "labels.npy", labels.numpy())
    # for step, (out, label) in enumerate(zip(outs, labels)):
    #     tb_logger.writer.add_scalar("Predictions", out.item(), step)
    #     tb_logger.writer.add_scalar("Labels", label.item(), step)

    # # bootstrap evaluation
    # mean_correlation, ci_lower, ci_upper = bootstrap_evaluation(
    #     predictions=outs,
    #     labels=labels,
    # )


if __name__ == "__main__":
    main()
