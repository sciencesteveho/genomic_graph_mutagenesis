#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Deletion of lethal genes.
"""


import argparse
from collections import defaultdict
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
from torch_geometric.loader import NeighborSampler  # type: ignore
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Base function for model evaluation or inference.

        Returns:
            Tuple containing:
                - regression_outs: Predictions for regression.
                - regression_labels: True labels for regression.
                - node_indices: Original node indices corresponding to the predictions.
        """
        self.model.eval()
        pbar = tqdm(total=len(data_loader))
        pbar.set_description(
            f"Evaluating {self.model.__class__.__name__} model @ epoch: {epoch}"
        )

        regression_outs, regression_labels = [], []
        node_indices = []

        for batch_idx, data in enumerate(data_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            data = data.to(self.device)
            mask_tensor = getattr(data, f"{mask}_mask_loss")

            # Skip batches with no nodes in the mask
            if mask_tensor.sum() == 0:
                pbar.update(1)
                continue

            # Forward pass
            regression_out, _ = self.model(
                x=data.x,
                edge_index=data.edge_index,
                mask=mask_tensor,
            )

            # Collect masked outputs and labels
            regression_out_masked = regression_out[mask_tensor]
            labels_masked = data.y[mask_tensor]
            batch_node_indices = data.n_id[mask_tensor].cpu()

            # Ensure tensors are at least one-dimensional
            if regression_out_masked.dim() == 0:
                regression_out_masked = regression_out_masked.unsqueeze(0)
                labels_masked = labels_masked.unsqueeze(0)
                batch_node_indices = batch_node_indices.unsqueeze(0)

            regression_outs.append(regression_out_masked.cpu())
            regression_labels.append(labels_masked.cpu())
            node_indices.append(batch_node_indices)

            pbar.update(1)

        pbar.close()

        if regression_outs:
            regression_outs = torch.cat(regression_outs, dim=0)
            regression_labels = torch.cat(regression_labels, dim=0)
            node_indices = torch.cat(node_indices, dim=0)
        else:
            regression_outs = torch.tensor([])
            regression_labels = torch.tensor([])
            node_indices = torch.tensor([])

        return regression_outs, regression_labels, node_indices

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


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode to gene symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def main() -> None:
    """Main function to perform in-silico perturbations matching CRISPRi experiments in K562."""
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(126)

    # gencode to symbol
    lookup_file = "gencode_to_genesymbol_lookup_table.txt"
    gencode_to_symbol = load_gencode_lookup(lookup_file)

    # load lethal genes
    lethal_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/lethal_genes_blomen.txt"
    with open(lethal_file, "r") as f:
        lethal_gene_symbols = [line.strip() for line in f]

    lethal_gencode = [
        gencode_to_symbol[gene]
        for gene in lethal_gene_symbols
        if gene in gencode_to_symbol
    ]

    # Load graph
    idx_file = "regulatory_only_k562_allcontacts_global_full_graph_idxs.pkl"
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    lethal_gencode = [
        f"{gene}_k562" for gene in lethal_gencode if f"{gene}_k562" in idxs
    ]
    lethal_idxs = [idxs[gene] for gene in lethal_gencode if gene in idxs]

    # Load the PyTorch graph
    data = torch.load("graph.pt")
    data = data.to(device)

    # Get dictionaries of gene indices
    gene_idxs = {k: v for k, v in idxs.items() if "ENSG" in k}

    # Load the model
    model = load_model("GAT_best_model.pt", device, device)

    # Initialize GNNTrainer
    trainer = GNNTrainer(
        model=model,
        device=device,
        data=data,
    )
    # Combine masks for all nodes
    data.all_mask = data.test_mask | data.train_mask | data.val_mask
    data.all_mask_loss = data.test_mask_loss | data.train_mask_loss | data.val_mask_loss

    # Create the all_loader
    all_loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data, "all_mask"),
        shuffle=False,
    )

    # Evaluate the model on the all_loader to get baseline predictions
    regression_outs, regression_labels, node_indices = trainer.evaluate(
        data_loader=all_loader,
        epoch=0,
        mask="all",
    )

    # Ensure tensors are one-dimensional
    regression_outs = regression_outs.squeeze()
    regression_labels = regression_labels.squeeze()
    node_indices = node_indices.squeeze()

    # Create a DataFrame to keep track of node indices and predictions
    baseline_df = pd.DataFrame(
        {
            "node_idx": node_indices.cpu().numpy(),
            "prediction": regression_outs.cpu().numpy(),
        }
    )

    # Get the list of non-essential genes
    all_gene_ids = list(gene_idxs.keys())
    non_essential_gene_ids = list(set(all_gene_ids) - set(lethal_gencode))

    # Run experiments for essential genes
    essential_fold_changes = []

    for run in range(20):
        print(f"Essential genes experiment run {run+1}")
        # Randomly select 100 essential genes
        selected_essential_node_indices = random.sample(lethal_idxs, 100)

        # Create a copy of data.x
        x_perturbed = data.x.clone()

        # Zero out the node features of the selected essential genes
        x_perturbed[selected_essential_node_indices] = 0

        # Create data_perturbed
        data_perturbed = data.clone()
        data_perturbed.x = x_perturbed

        # Ensure masks are carried over
        data_perturbed.all_mask = data.all_mask
        data_perturbed.all_mask_loss = data.all_mask_loss

        # Create perturbed_loader
        perturbed_loader = NeighborLoader(
            data_perturbed,
            num_neighbors=[data.avg_edges] * 2,
            batch_size=64,
            input_nodes=getattr(data_perturbed, "all_mask"),
            shuffle=False,
        )

        # Evaluate the model on the perturbed data
        regression_outs_perturbed, _, node_indices_perturbed = trainer.evaluate(
            data_loader=perturbed_loader,
            epoch=0,
            mask="all",
        )

        # Create perturbed DataFrame
        perturbed_df = pd.DataFrame(
            {
                "node_idx": node_indices_perturbed.cpu().numpy(),
                "prediction_perturbed": regression_outs_perturbed.cpu().numpy(),
            }
        )

        # Merge baseline and perturbed DataFrames
        merged_df = baseline_df.merge(perturbed_df, on="node_idx")

        # Compute difference
        merged_df["diff"] = merged_df["prediction"] - merged_df["prediction_perturbed"]

        # Compute average fold change
        average_fold_change = merged_df["diff"].mean()

        essential_fold_changes.append(average_fold_change)

    # Save the fold change results
    with open("essential_fold_changes.pkl", "wb") as f:
        pickle.dump(essential_fold_changes, f)

    # Run experiments for non-essential genes
    non_essential_fold_changes = []

    for run in range(20):
        print(f"Non-essential genes experiment run {run+1}")
        # Randomly select 100 non-essential genes
        selected_non_essential_genes = random.sample(non_essential_gene_ids, 100)
        # Get their node indices
        selected_non_essential_node_indices = [
            gene_idxs[gene_id]
            for gene_id in selected_non_essential_genes
            if gene_id in gene_idxs
        ]

        # Create a copy of data.x
        x_perturbed = data.x.clone()

        # Zero out the node features of the selected non-essential genes
        x_perturbed[selected_non_essential_node_indices] = 0

        # Create data_perturbed
        data_perturbed = data.clone()
        data_perturbed.x = x_perturbed

        # Ensure masks are carried over
        data_perturbed.all_mask = data.all_mask
        data_perturbed.all_mask_loss = data.all_mask_loss

        # Create perturbed_loader
        perturbed_loader = NeighborLoader(
            data_perturbed,
            num_neighbors=[data.avg_edges] * 2,
            batch_size=64,
            input_nodes=getattr(data_perturbed, "all_mask"),
            shuffle=False,
        )

        # Evaluate the model on the perturbed data
        regression_outs_perturbed, _, node_indices_perturbed = trainer.evaluate(
            data_loader=perturbed_loader,
            epoch=0,
            mask="all",
        )

        # Create perturbed DataFrame
        perturbed_df = pd.DataFrame(
            {
                "node_idx": node_indices_perturbed.cpu().numpy(),
                "prediction_perturbed": regression_outs_perturbed.cpu().numpy(),
            }
        )

        # Merge baseline and perturbed DataFrames
        merged_df = baseline_df.merge(perturbed_df, on="node_idx")

        # Compute difference
        merged_df["diff"] = merged_df["prediction"] - merged_df["prediction_perturbed"]

        # Compute average fold change
        average_fold_change = merged_df["diff"].mean()

        non_essential_fold_changes.append(average_fold_change)

    with open("non_essential_fold_changes.pkl", "wb") as f:
        pickle.dump(non_essential_fold_changes, f)

    print(f"Essential fold changes: {essential_fold_changes}")
    print(f"Non-essential fold changes: {non_essential_fold_changes}")
