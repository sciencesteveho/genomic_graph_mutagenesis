#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Inference runner for perturbation experiments."""


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.architecture_builder import build_gnn_architecture
from omics_graph_learning.combination_loss import CombinationLoss


class PerturbRunner:
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
        regression_outs, regression_labels = [], []
        node_indices = []

        pbar = tqdm(total=len(data_loader))
        pbar.set_description(
            f"Evaluating {self.model.__class__.__name__} model @ epoch: {epoch}"
        )

        for batch_idx, data in enumerate(data_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            data = data.to(self.device)
            mask_tensor = getattr(data, f"{mask}_mask_loss")

            # skip batches with no nodes in the mask
            if mask_tensor.sum() == 0:
                pbar.update(1)
                continue

            # forward pass
            regression_out, _ = self.model(
                x=data.x,
                edge_index=data.edge_index,
                mask=mask_tensor,
            )

            # collect masked outputs and labels
            regression_out_masked = regression_out[mask_tensor]
            labels_masked = data.y[mask_tensor]
            batch_node_indices = data.n_id[mask_tensor].cpu()

            # ensure tensors are at least one-dimensional
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

    @torch.no_grad()
    def infer_subgraph(
        self,
        sub_data: torch_geometric.data.Data,
        mask_attr: str,
    ) -> torch.Tensor:
        """Perform inference on a subgraph."""
        self.model.eval()
        sub_data = sub_data.to(self.device)
        # mask = getattr(sub_data, f"{mask_attr}_mask_loss")
        mask = sub_data.all_mask_loss

        regression_out, _ = self.model(
            x=sub_data.x,
            edge_index=sub_data.edge_index,
            mask=mask,
        )
        print(f"Regression output without mask: {regression_out}")
        return regression_out

    @torch.no_grad()
    def infer_perturbed_subgraph(
        self,
        sub_data: torch_geometric.data.Data,
        node_to_remove_idx: int,
        mask_attr: str,
        gene_node: int,  # Add gene_node as a parameter
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Perform inference on a perturbed subgraph with a node removed."""
        # Mask = all nodes except the node to remove
        self.model.eval()
        mask_nodes = (
            torch.arange(sub_data.num_nodes, device=self.device) != node_to_remove_idx
        )

        # Get subgraph with node removed
        perturbed_edge_index, _, _ = subgraph(
            subset=mask_nodes,
            edge_index=sub_data.edge_index,
            relabel_nodes=True,
            num_nodes=sub_data.num_nodes,
            return_edge_mask=True,
        )

        # Copy features and mask to perturbed subgraph
        perturbed_x = sub_data.x[mask_nodes]
        perturbed_mask = sub_data.all_mask_loss[mask_nodes]
        perturbed_n_id = sub_data.n_id[mask_nodes]

        # Ensure gene_node is still in the perturbed subgraph
        if (perturbed_n_id == gene_node).sum() == 0:
            print(
                f"Gene node not in perturbed subgraph after removing node {sub_data.n_id[node_to_remove_idx].item()}"
            )
            return None, None

        # Find the new index of the gene node after reindexing
        idx_in_perturbed = (
            (perturbed_n_id == gene_node).nonzero(as_tuple=True)[0].item()
        )

        # Inference
        regression_out, _ = self.model(
            x=perturbed_x,
            edge_index=perturbed_edge_index,
            mask=perturbed_mask,
        )
        return regression_out, idx_in_perturbed

    @staticmethod
    def load_model(
        checkpoint_file: str,
        map_location: torch.device,
        model: str,
        activation: str,
        in_size: int,
        embedding_size: int,
        gnn_layers: int,
        shared_mlp_layers: int,
        heads: int,
        dropout_rate: float,
        residual: str,
        attention_task_head: bool,
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
            model=model,
            activation=activation,
            in_size=in_size,
            embedding_size=embedding_size,
            out_channels=1,
            gnn_layers=gnn_layers,
            shared_mlp_layers=shared_mlp_layers,
            heads=heads,
            dropout_rate=dropout_rate,
            residual=residual,
            attention_task_head=attention_task_head,
            train_dataset=None,
        )
        model = model.to(map_location)

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
