#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturb essential genes and measure impact on overall fold-change."""


import os
import random
from typing import Dict, List, Set

import pandas as pd
import torch
from torch import device
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore

from omics_graph_learning.interpret.interpret_utils import combine_masks
from omics_graph_learning.interpret.perturb_runner import PerturbRunner


def get_baseline_predictions(
    data: Data,
    runner: PerturbRunner,
    mask: str,
    device: torch.device,
) -> pd.DataFrame:
    """Obtain baseline predictions.

    Returns:
        pd.DataFrame: columns are ['node_idx', 'prediction']
    """
    data = combine_masks(data).to(device)

    # create loader for baseline
    loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data, f"{mask}_mask"),
        shuffle=False,
    )

    preds, _, node_indices = runner.evaluate(
        data_loader=loader,
        epoch=0,
        mask=mask,
    )
    preds = preds.squeeze()
    node_indices = node_indices.squeeze()

    return pd.DataFrame(
        {
            "node_idx": node_indices.cpu().numpy(),
            "prediction": preds.cpu().numpy(),
        }
    )


def perturb_and_evaluate(
    data: Data,
    runner: PerturbRunner,
    mask: str,
    selected_nodes: List[int],
    baseline_df: pd.DataFrame,
) -> float:
    """Zero out the features of the specified nodes, then evaluate and compute
    mean difference.

    Returns:
        float: average fold change or difference after perturbation.
    """
    # create a perturbed copy
    x_perturbed = data.x.clone()
    x_perturbed[selected_nodes] = 0

    data_perturbed = data.clone()
    data_perturbed.x = x_perturbed

    loader_perturbed = NeighborLoader(
        data_perturbed,
        num_neighbors=[data_perturbed.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data_perturbed, f"{mask}_mask"),
        shuffle=False,
    )

    perturbed_preds, _, perturbed_node_indices = runner.evaluate(
        data_loader=loader_perturbed,
        epoch=0,
        mask=mask,
    )
    perturbed_df = pd.DataFrame(
        {
            "node_idx": perturbed_node_indices.cpu().numpy(),
            "prediction_perturbed": perturbed_preds.cpu().numpy(),
        }
    )

    merged = baseline_df.merge(perturbed_df, on="node_idx", how="inner")
    merged["diff"] = merged["prediction"] - merged["prediction_perturbed"]
    return merged["diff"].mean()


def perturb_specific_node_group(
    data: Data,
    runner: PerturbRunner,
    node_indices_to_perturb: List[int],
    device: torch.device,
    mask: str = "all",
    num_runs: int = 20,
) -> List[float]:
    """Randomly pick a subset of node_indices_to_perturb, zero out their
    features, and measure changes.

    Args:
        data: PyG Data object of the graph
        runner: PerturbRunner object holding loaded model
        node_indices_to_perturb: idxs of nodes to perturb
        device: torch device
        mask: which mask to use
        num_runs: how many perturbation experiments to run

    Returns:
        list[float]: average fold changes across runs.
    """
    fold_changes = []

    # get the baseline predictions once
    baseline_df = get_baseline_predictions(data, runner, mask, device)

    for run_idx in range(num_runs):
        print(f"[perturb_specific_node_group] run {run_idx + 1}/{num_runs}")

        # randomly pick up to 100 nodes
        num_to_select = min(100, len(node_indices_to_perturb))
        selected_nodes = random.sample(node_indices_to_perturb, num_to_select)

        # evaluate new predictions vs baseline
        avg_fold_change = perturb_and_evaluate(
            data=data,
            runner=runner,
            mask=mask,
            device=device,
            selected_nodes=selected_nodes,
            baseline_df=baseline_df,
        )
        fold_changes.append(avg_fold_change)

    return fold_changes


def load_lethal_symbols(lethal_file: str) -> List[str]:
    """Load gene symbols from Blomen et al. 2015."""
    if not os.path.isfile(lethal_file):
        raise FileNotFoundError(f"lethal file not found: {lethal_file}")
    with open(lethal_file, "r") as f:
        return [line.strip() for line in f]


def get_lethal_node_indices(
    lethal_symbols: List[str],
    gencode_to_symbol: Dict[str, str],
    idxs: Dict[str, int],
    sample: str,
) -> Set[int]:
    """Convert lethal gene symbols to node indices."""
    # convert symbols to gencode IDs
    lethal_gencode = [
        gencode_to_symbol[sym] for sym in lethal_symbols if sym in gencode_to_symbol
    ]
    # append sample suffix
    lethal_gencode_sampled = [
        f"{g}_{sample}" for g in lethal_gencode if f"{g}_{sample}" in idxs
    ]
    # map to node indices
    return {idxs[g] for g in lethal_gencode_sampled if g in idxs}


def get_all_gene_indices(idxs: Dict[str, int]) -> List[int]:
    """Return node indices for genes."""
    return [v for k, v in idxs.items() if "ENSG" in k]


def get_essential_or_nonessential_idxs(
    idxs: Dict[str, int],
    gencode_to_symbol: Dict[str, str],
    lethal_file: str,
    sample: str,
    mode: str = "essential",
) -> List[int]:
    """Get node idxs for either essential or non-essential genes.

    Args:
        idxs: {node name: node idx} mapping dictionary
        gencode_to_symbol: {gencode: symbol} mapping dictionary
        lethal_file: path to file with lethal gene symbols
        sample: str for graph sample, e.g. "k562"
        mode: 'essential' or 'nonessential'

    Returns:
        list[int]: node indices of either lethal or nonlethal genes
    """
    lethal_symbols = load_lethal_symbols(lethal_file)
    lethal_idxs = get_lethal_node_indices(
        lethal_symbols, gencode_to_symbol, idxs, sample
    )
    if mode == "essential":
        return list(lethal_idxs)
    elif mode == "nonessential":
        all_genes = get_all_gene_indices(idxs)
        return list(set(all_genes) - lethal_idxs)
    else:
        raise ValueError(f"unknown mode: {mode}")


def essential_gene_perturbation(
    data: Data,
    runner: PerturbRunner,
    idxs: Dict[str, int],
    gencode_to_symbol: Dict[str, str],
    num_runs: int = 20,
    sample: str = "k562",
    lethal_file: str = "lethal_genes.txt",
    mask: str = "all",
    device: torch.device = torch.device("cpu"),
    essential: bool = True,
) -> List[float]:
    """Perform perturbation on essential (lethal) genes.

    Args:
        data: full graph data
        runner: PerturbRunner object with loaded model
        idxs: {node name: node idx} mapping dictionary
        gencode_to_symbol: {gencode: symbol} mapping dictionary
        num_runs: number of experiments to run
        sample: sample name
        lethal_file: file with lethal genes
        mask: mask name
        device: torch device
        essential: bool indicating whether to perturb essential or non-essential
        genes

    Returns:
        list[float]: average fold changes across runs
    """
    lethal_idxs = get_essential_or_nonessential_idxs(
        idxs=idxs,
        gencode_to_symbol=gencode_to_symbol,
        lethal_file=lethal_file,
        sample=sample,
        mode="essential" if essential else "nonessential",
    )
    return perturb_specific_node_group(
        data=data,
        runner=runner,
        node_indices_to_perturb=lethal_idxs,
        device=device,
        mask=mask,
        num_runs=num_runs,
    )
