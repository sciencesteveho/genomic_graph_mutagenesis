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


def calculate_log2_fold_change(
    baseline_prediction: float, perturbation_prediction: float
) -> float:
    """Calculate the log2 fold change from log2-transformed values."""
    log2_fold_change = perturbation_prediction - baseline_prediction
    return 2**log2_fold_change - 1


def create_full_neighbor_loader(
    data: Data, target_node: int, enhancer_nodes: List[int], num_layers: int = 10
) -> NeighborLoader:
    """
    Creates a NeighborLoader to sample neighbors, ensuring the target node
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
    """Main function to perform in-silico perturbations matching CRISPRi experiments in K562."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(126)

    # Load graph
    idx_file = "regulatory_only_k562_allcontacts_global_full_graph_idxs.pkl"

    # Load the pytorch graph
    data = torch.load("graph.pt")
    data = data.to(device)

    # Create nx graph
    nx_graph = to_networkx(data, to_undirected=True)

    # Load IDXS
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    # Get dictionaries of enhancer and gene indices
    enhancer_idxs = {k: v for k, v in idxs.items() if "enhancer" in k}
    gene_dixs = {k: v for k, v in idxs.items() if "ENSG" in k}

    # Map node indices to gene IDs
    node_idx_to_gene_id = {v: k for k, v in gene_dixs.items()}
    gene_indices = list(gene_dixs.values())

    # Load the model
    model = load_model("GAT_best_model.pt", device, device)

    # Perform inference on the entire graph
    with torch.no_grad():
        regression_out, _ = model(x=data.x, edge_index=data.edge_index)

    # Extract predictions and labels for gene nodes
    regression_out_gene = regression_out[gene_indices].squeeze()
    labels_gene = data.y[gene_indices].squeeze()

    # Calculate absolute differences between predictions and true labels
    diff = torch.abs(regression_out_gene - labels_gene)

    # Filter genes with predicted output > 5
    mask = regression_out_gene > 5
    diff_filtered = diff[mask]
    regression_out_gene_filtered = regression_out_gene[mask]
    labels_gene_filtered = labels_gene[mask]
    gene_indices_filtered = torch.tensor(gene_indices)[mask]

    # Select top 25 genes with the smallest difference
    indices_top25 = torch.topk(diff_filtered, 25, largest=False).indices
    top25_gene_indices = gene_indices_filtered[indices_top25]
    top25_gene_ids = [node_idx_to_gene_id[idx.item()] for idx in top25_gene_indices]

    # Store baseline predictions
    baseline_predictions = {
        node_idx_to_gene_id[idx.item()]: regression_out[idx].item()
        for idx in gene_indices
    }

    # (Optional) Save baseline predictions to a file
    with open("baseline_predictions.pkl", "wb") as f:
        pickle.dump(baseline_predictions, f)

    # Task 2: Perturb connected components
    # Dictionary to store fold changes for each gene
    gene_fold_changes = {}

    for gene_node in top25_gene_indices:
        gene_node = gene_node.item()
        gene_id = node_idx_to_gene_id[gene_node]

        # Get connected component containing the gene
        gene_component = nx.node_connected_component(nx_graph, gene_node)
        component_nodes = list(gene_component)

        # Exclude the gene node
        component_nodes_without_gene = [n for n in component_nodes if n != gene_node]
        if not component_nodes_without_gene:
            continue  # Skip if no other nodes in the component

        subset = torch.tensor(component_nodes, dtype=torch.long)
        data_sub = data.subgraph(subset).to(device)

        # Get baseline prediction for the gene in the subgraph
        with torch.no_grad():
            regression_out_sub, _ = model(x=data_sub.x, edge_index=data_sub.edge_index)

        # Get index of gene_node in subset
        idx_in_subset = (subset == gene_node).nonzero(as_tuple=True)[0].item()
        baseline_prediction = regression_out_sub[idx_in_subset].item()

        # Initialize dictionary to store fold changes
        fold_changes = {}

        for i in range(100):
            # Randomly select a node to remove
            node_to_remove = random.choice(component_nodes_without_gene)
            subset_minus_node = subset[subset != node_to_remove]
            data_sub_minus_node = data.subgraph(subset_minus_node).to(device)

            # Check if gene_node is still in the subgraph
            if gene_node not in subset_minus_node:
                continue  # Skip if gene node is not in the subgraph

            # Perform inference on the perturbed subgraph
            with torch.no_grad():
                regression_out_minus_node, _ = model(
                    x=data_sub_minus_node.x, edge_index=data_sub_minus_node.edge_index
                )

            # Get index of gene_node in the perturbed subgraph
            idx_in_subset_minus_node = (
                (subset_minus_node == gene_node).nonzero(as_tuple=True)[0].item()
            )
            perturbation_prediction = regression_out_minus_node[
                idx_in_subset_minus_node
            ].item()

            # Compute fold change
            fold_change = calculate_log2_fold_change(
                baseline_prediction, perturbation_prediction
            )

            # Store fold change
            fold_changes[node_idx_to_gene_id[node_to_remove]] = fold_change

        # Store fold changes for this gene
        gene_fold_changes[gene_id] = fold_changes

    # (Optional) Save the fold changes to a file
    with open("gene_fold_changes.pkl", "wb") as f:
        pickle.dump(gene_fold_changes, f)

    # Task 3: Zero out node features
    # Get baseline predictions on the full graph
    with torch.no_grad():
        regression_out_baseline, _ = model(x=data.x, edge_index=data.edge_index)

    # Dictionary to store average distances for each feature perturbation
    feature_fold_changes = {}

    # List of feature indices to perturb (e.g., first 30 features)
    feature_indices = list(range(30))

    for feature_index in feature_indices:
        # Create a copy of the node features and zero out the feature at feature_index
        x_perturbed = data.x.clone()
        x_perturbed[:, feature_index] = 0

        # Perform inference with perturbed features
        with torch.no_grad():
            regression_out_perturbed, _ = model(
                x=x_perturbed, edge_index=data.edge_index
            )

        # Compute average distance between baseline and perturbed predictions
        avg_distance = torch.mean(
            torch.abs(regression_out_baseline - regression_out_perturbed)
        ).item()

        # Store the result
        feature_fold_changes[feature_index] = avg_distance

    # (Optional) Save the feature fold changes to a file
    with open("feature_fold_changes.pkl", "wb") as f:
        pickle.dump(feature_fold_changes, f)


if __name__ == "__main__":
    main()
