#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Make plots related to GNN training."""


from pathlib import Path
from typing import Dict, List, Union

import matplotlib  # type: ignore
from matplotlib.figure import Figure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)  # type: ignore
import torch

from omics_graph_learning.visualization import set_matplotlib_publication_parameters


def _load_tb_loss(tensorboard_log: Union[str, Path]) -> Dict[str, List[float]]:
    """Load tensorboard log and return a dataframe with the loss values."""
    event_acc = EventAccumulator(str(tensorboard_log))
    event_acc.Reload()

    # get the loss values
    training_loss = event_acc.Scalars("Training loss")
    val_rmse = event_acc.Scalars("Validation RMSE")
    test_rmse = event_acc.Scalars("Test RMSE")

    return {
        "Epoch": [loss.step for loss in training_loss],
        "Training loss": [loss.value for loss in training_loss],
        "Validation RMSE": [rmse.value for rmse in val_rmse],
        "Test RMSE": [rmse.value for rmse in test_rmse],
    }


def plot_training_losses(tensorboard_log: Union[str, Path]) -> Figure:
    """Plots training losses. Get values from tensorboard log."""
    losses = _load_tb_loss(tensorboard_log)
    df = pd.DataFrame(losses)
    set_matplotlib_publication_parameters()

    fig, ax = plt.subplots(figsize=(3.125, 2.25))
    sns.lineplot(x="Epoch", y="Training loss", data=df, label="Training Loss", ax=ax)
    sns.lineplot(
        x="Epoch", y="Validation RMSE", data=df, label="Validation RMSE", ax=ax
    )
    sns.lineplot(x="Epoch", y="Test RMSE", data=df, label="Test RMSE", ax=ax)
    ax.margins(x=0)
    ax.set_xlabel("Epoch", fontsize=7)
    ax.set_ylabel("Loss / RMSE", fontsize=7)
    ax.set_title("Training and Validation Metrics", fontsize=7)
    fig.tight_layout()
    return fig


def plot_predicted_versus_expected(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    rmse: torch.Tensor,
) -> Figure:
    """Plots predicted versus expected values for a given model"""
    set_matplotlib_publication_parameters()

    fig, ax = plt.subplots(figsize=(3.15, 2.95))
    sns.regplot(
        x=expected,
        y=predicted,
        scatter_kws={"s": 1.5, "alpha": 1},
        ax=ax,
        line_kws={"color": "#D3D3D3"},
        fit_reg=False,
    )
    ax.margins(x=0)
    ax.set_xlabel("Expected Log2 TPM", fontsize=7)
    ax.set_ylabel("Predicted Log2 TPM", fontsize=7)
    ax.set_title(
        f"Expected versus predicted TPM\n"
        f"RMSE: {rmse}\n"
        f"Spearman's R: {stats.spearmanr(expected, predicted)[0]:.4f}\n"
        f"Pearson: {stats.pearsonr(expected, predicted)[0]:.4f}",
        wrap=True,
        fontsize=7,
    )
    fig.tight_layout()
    return fig
