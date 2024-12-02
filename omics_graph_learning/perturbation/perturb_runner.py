#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Inference runner for perturbation experiments."""


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.architecture_builder import build_gnn_architecture
from omics_graph_learning.combination_loss import CombinationLoss


class PerturbationRunner:
    """Class to handle GNN model inference for perturbation experiments.

    Methods
    --------
    evaluate_single(data, mask):
        Evaluate a single data object.
    evaluate(data_loader, epoch, mask, subset_batches):
        Base function for model evaluation or inference.

    Examples:
    --------
    # instantiate trainer
    >>> exp_runner = PerturbationRunner(
        model=model,
        device=device,
        data=data,
    )

    # evaluate model on the test set
    >>> regression_outs, regression_labels, node_indices = exp_runner.evaluate(
        data_loader=test_loader,
        epoch=epoch,
        mask="all",
    )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        data: torch_geometric.data.Data,
    ) -> None:
        """Initialize model trainer."""
        self.model = model
        self.device = device
        self.data = data

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
                - node_indices: Original node indices corresponding to the
                  predictions.
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

    @staticmethod
    def calculate_log2_fold_change(
        baseline_prediction: float, perturbation_prediction: float
    ) -> float:
        """Calculate the log2 fold change from log2-transformed values."""
        log2_fold_change = perturbation_prediction - baseline_prediction
        return 2**log2_fold_change - 1

    @staticmethod
    def _ensure_tensor_dim(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has the correct dimensions for evaluation."""
        tensor = tensor.squeeze()
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor
