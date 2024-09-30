#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Custom weighted combination loss function."""


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSEandBCELoss(nn.Module):
    """Custom loss function combining Root Mean Squared Error (RMSE) for
    regression and Binary Cross-Entropy Loss (BCE) for classification tasks for
    OGL.

    Attributes:
        alpha: (float) weighting factor for the primary task (default: 0.8)

    Methods
    --------
    forward::
        Compute the combined loss

    Examples:
    --------
    # Initialize custom loss
    >>> criterion = RMSEandBCELoss(alpha=0.8)

    # Compute the loss and backpropagate
    >>> loss = criterion(
        regression_output=regression_output,
        regression_target=regression_target,
        classification_output=classification_output,
        classification_target=classification_target,
        mask=mask,
        )
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(self, alpha=0.8) -> None:
        """Initialize the custom loss function."""
        super(RMSEandBCELoss, self).__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha

    def forward(
        self,
        regression_output: torch.Tensor,
        regression_target: torch.Tensor,
        classification_output: torch.Tensor,
        classification_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the weighted combination loss via weighted sum of RMSE and
        BCE.
        """
        regression_loss = self.compute_regression_loss(
            regression_output=regression_output,
            regression_target=regression_target,
            mask=mask,
        )
        classification_loss = self.compute_classification_loss(
            classification_output=classification_output,
            classification_target=classification_target,
            mask=mask,
        )
        combined_loss = self.compute_combined_loss(
            regression_loss=regression_loss, classification_loss=classification_loss
        )

        return combined_loss, regression_loss, classification_loss

    def compute_combined_loss(
        self, regression_loss: torch.Tensor, classification_loss: torch.Tensor
    ) -> torch.Tensor:
        """Compute the combined loss."""
        return self.alpha * regression_loss + (1 - self.alpha) * classification_loss

    @staticmethod
    def compute_regression_loss(
        regression_output: torch.Tensor,
        regression_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the regression loss (root mean-square error)."""
        if mask.sum() <= 0:
            return torch.tensor(0.0, device=regression_output.device)

        mse = F.mse_loss(
            regression_output[mask].squeeze(), regression_target[mask].squeeze()
        )
        return torch.sqrt(mse)

    @staticmethod
    def compute_classification_loss(
        classification_output: torch.Tensor,
        classification_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the classification loss (binary cross-entropy)."""
        if mask.sum() <= 0:
            return torch.tensor(0.0, device=classification_output.device)

        return F.binary_cross_entropy_with_logits(
            classification_output[mask].squeeze(),
            classification_target[mask].float().squeeze(),
        )
