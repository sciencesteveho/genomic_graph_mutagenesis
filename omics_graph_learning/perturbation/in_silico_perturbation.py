#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturbation of connected components."""


from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore


class InSilicoPerturbation:
    """Class to handle in silico perturbations."""

    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize the ModelEvaluator.

        Args:
            model (nn.Module): The trained GNN model.
            device (torch.device): The device to run the model on.
        """
        self.model = model
        self.device = device

    @torch.no_grad()
    def inference_on_component(
        self,
        data: Data,
        regression_mask: torch.Tensor,
    ) -> Dict[int, float]:
        """Perform inference on a single connected component.

        Args:
            data (Data): The graph data of the connected component.
            regression_mask (torch.Tensor): Mask indicating nodes for
            regression.

        Returns:
            Dict[int, float]: Dictionary mapping node indices to predicted
            expression values.
        """
        self.model.eval()

        data = data.to(self.device)
        out = self.model(
            x=data.x,
            edge_index=data.edge_index,
        )

        # extract predictions for target nodes
        out_masked = out[regression_mask].squeeze().cpu()
        node_ids = data.n_id[regression_mask].cpu()

        return {
            int(node_id): float(pred) for node_id, pred in zip(node_ids, out_masked)
        }
