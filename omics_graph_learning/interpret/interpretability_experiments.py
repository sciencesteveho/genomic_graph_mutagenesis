#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run interpretability experiments on a trained model.
1 - PGExplainer subgraph explanations
2 - Gradient-based saliency maps
3 - Attention weights, if applicable (only if model is GATv2)
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import nn  # type: ignore
from torch_geometric.data import Data  # type: ignore

from omics_graph_learning.interpret.attention_weights import get_attention_weights
from omics_graph_learning.interpret.explainer import build_explainer
from omics_graph_learning.interpret.explainer import train_explainer
from omics_graph_learning.interpret.perturb_runner import PerturbRunner
from omics_graph_learning.interpret.saliency import compute_gradient_saliency
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.constants import RANDOM_SEEDS


def load_data_and_model(
    graph_file: str,
    model_file: str,
    device: torch.device,
) -> Tuple[
    Data,
    nn.Module,
]:
    """Load data and model for perturbation experiments."""
    # load graph data
    data = torch.load(graph_file)
    data = data.to(device)

    # instantiate
    model = PerturbRunner.load_model(
        checkpoint_file=model_file,
        map_location=device,
        model="GAT",
        activation="gelu",
        in_size=44,
        embedding_size=200,
        gnn_layers=2,
        shared_mlp_layers=2,
        heads=2,
        dropout_rate=0.3,
        residual="distinct_source",
        attention_task_head=False,
    )

    return (
        data,
        model,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run interpretability experiments")
    parser.add_argument("--lookup_file", type=str, help="Path to gencode lookup file")
    parser.add_argument("--graph_file", type=str, help="Path to PyG Data graph file")
    parser.add_argument("--idx_file", type=str, help="Path to gene index file")
    parser.add_argument(
        "--model_file", type=str, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="interpret_results",
        help="Directory to store interpretability outputs",
    )
    parser.add_argument("--sample", type=str, default="k562", help="Sample name")
    parser.add_argument("--run", type=int, default=1, help="Run number")
    return parser.parse_args()


def main() -> None:
    """Run experiments!"""
    # parse command-line arguments
    args = parse_args()

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set seed
    run = args.run
    seed = RANDOM_SEEDS[run]
    torch.manual_seed(seed)

    (
        data,
        model,
    ) = load_data_and_model(
        graph_file="data/gene_graph.pt",
        model_file="models/gatv2.pt",
        device=device,
    )

    # pgexplainer
    pg_explainer = build_explainer(model=model, epochs=30, lr=0.003)
    explanations = train_explainer(pg_explainer, data)
    torch.save(explanations, Path(args.output_dir) / "pgexplainer_explanations.pt")

    # saliency maps
    gene_mask = data.train_mask_loss | data.val_mask_loss | data.test_mask_loss
    saliency_map = compute_gradient_saliency(
        model=model,
        data=data,
        device=device,
        mask=gene_mask,
        regression_loss_type="rmse",
        alpha=0.85,
    )
    torch.save(saliency_map, Path(args.output_dir) / "saliency_map.pt")

    # attention weights
    attention_weights = get_attention_weights(model, data, gene_mask)
    torch.save(attention_weights, Path(args.output_dir) / "attention_weights.pt")


if __name__ == "__main__":
    main()
