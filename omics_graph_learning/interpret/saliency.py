#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Use gradient-based saliency maps to explain feature importance."""


from typing import Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.data import Data  # type: ignore


def compute_gradient_saliency(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    mask: torch.Tensor,
    regression_loss_type: str = "rmse",
    alpha: float = 0.85,
) -> torch.Tensor:
    """Compute gradient-based saliency for node features."""
    data.x.requires_grad = True  # track gradients on node features

    regression_out, logits = model(x=data.x, edge_index=data.edge_index, mask=mask)
    regression_masked = regression_out[mask]
    labels_masked = data.y[mask]

    classification_masked = logits[mask]
    class_labels_masked = data.class_labels[mask].float()

    # regression part of custom loss
    if mask.sum() == 0:
        regression_loss = torch.tensor(0.0, device=device)
    elif regression_loss_type == "rmse":
        mse = F.mse_loss(regression_masked.squeeze(), labels_masked.squeeze())
        regression_loss = torch.sqrt(mse)
    elif regression_loss_type == "smooth_l1":
        regression_loss = F.smooth_l1_loss(
            regression_masked.squeeze(), labels_masked.squeeze()
        )
    else:
        raise ValueError("regression_loss_type must be 'rmse' or 'smooth_l1'.")

    # classification part of custom loss
    if mask.sum() == 0:
        classification_loss = torch.tensor(0.0, device=device)
    else:
        classification_loss = F.binary_cross_entropy_with_logits(
            classification_masked.squeeze(),
            class_labels_masked.squeeze(),
        )

    # combine losses
    loss = alpha * regression_loss + (1 - alpha) * classification_loss

    # backpropagation
    model.zero_grad()
    loss.backward()

    saliency_map = data.x.grad.detach().cpu()
    data.x.requires_grad = False

    return saliency_map
