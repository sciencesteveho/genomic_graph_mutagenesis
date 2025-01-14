#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Custom weighted combination loss function."""


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinationLoss(nn.Module):
    """Custom loss function combining eithing root mean squared error (RMSE) or
    smooth L1 loss for regression and binary cross-entropy loss (BCE) for
    classification.

    Attributes:
        alpha (float): weighting factor for the primary task (default: 0.90)
        regression_loss_type (str): type of regression loss (default: 'rmse')

    Methods
    --------
    forward::
        Compute the combined loss

    Examples:
    --------
    # Initialize custom loss >>> criterion = RMSEandBCELoss(alpha=alpha)

    # Compute the loss and backpropagate >>> loss = criterion(
        regression_output=regression_output,
        regression_target=regression_target,
        classification_output=classification_output,
        classification_target=classification_target, mask=mask, )
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(self, alpha=0.90, regression_loss_type: str = "rmse") -> None:
        """Initialize the custom loss function."""
        super(CombinationLoss, self).__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha

        if regression_loss_type in {"rmse", "smooth_l1"}:
            self.regression_loss_type = regression_loss_type
        else:
            raise ValueError(
                "regression_loss_type must be either 'rmse' or 'smooth_l1'."
            )

    def forward(
        self,
        regression_output: torch.Tensor,
        regression_target: torch.Tensor,
        classification_output: torch.Tensor,
        classification_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computed the weighted combination loss via weighted sum of the
        regression and classification tasks.
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

    def compute_regression_loss(
        self,
        regression_output: torch.Tensor,
        regression_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the regression loss (root mean-square error or smooth l1)."""
        if mask.sum() <= 0:
            return torch.tensor(0.0, device=regression_output.device)

        masked_output = regression_output[mask].squeeze()
        masked_target = regression_target[mask].squeeze()

        if self.regression_loss_type == "rmse":
            mse = F.mse_loss(masked_output, masked_target)
            return torch.sqrt(mse)
        elif self.regression_loss_type == "smooth_l1":
            return F.smooth_l1_loss(masked_output, masked_target)
        else:
            raise ValueError(
                f"Unsupported regression type: {self.regression_loss_type}"
            )

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
