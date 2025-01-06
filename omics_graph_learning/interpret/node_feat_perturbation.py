#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturb node features and measure impact on model output."""


from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.perturb_runner import PerturbRunner


def get_test_loader(
    data: Data,
    mask: str,
    batch_size: int = 64,
) -> NeighborLoader:
    """Create a neighborloader for test/validation/evaluation nodes."""
    return NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=batch_size,
        input_nodes=getattr(data, f"{mask}_mask"),
        shuffle=False,
    )


def compute_baseline_output(
    runner: PerturbRunner,
    batch: Data,
    mask_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute baseline model output for comparison."""
    with torch.no_grad():
        baseline_out, _ = runner.model(
            x=batch.x,
            edge_index=batch.edge_index,
            mask=mask_tensor,
        )
    return baseline_out[mask_tensor].cpu()


def compute_feature_perturbation(
    runner: PerturbRunner,
    batch: Data,
    mask_tensor: torch.Tensor,
    baseline_output: torch.Tensor,
    feature_indices: List[int],
) -> Dict[int, Dict[int, List[float]]]:
    """For each feature index, zero out that feature in the batch and compute
    the difference in model output from the baseline.

    Returns:
        {feature_index: {node_index: [differences]}}
    """
    feature_node_differences = defaultdict(lambda: defaultdict(list))
    node_indices = batch.n_id[mask_tensor].cpu()

    for feat_idx in feature_indices:
        x_perturbed = batch.x.clone()
        x_perturbed[:, feat_idx] = 0  # zero out selected feature

        with torch.no_grad():
            perturbed_out, _ = runner.model(
                x=x_perturbed,
                edge_index=batch.edge_index,
                mask=mask_tensor,
            )
        perturbed_out = perturbed_out[mask_tensor].cpu()

        diff = baseline_output - perturbed_out
        for idx, d in zip(node_indices, diff):
            feature_node_differences[feat_idx][idx.item()].append(d.item())

    return feature_node_differences


def compute_average_node_differences(
    feature_node_differences: Dict[int, Dict[int, List[float]]]
) -> Dict[int, Dict[int, float]]:
    """Compute the average difference for each node, for each feature index.

    Args:
        feature_node_differences: {feature_index: node_index: [differences]}

    Returns:
        {feature_index: node_index: average difference}
    """
    return {
        feat_idx: {node: sum(vals) / len(vals) for node, vals in node_diffs.items()}
        for feat_idx, node_diffs in feature_node_differences.items()
    }


def compute_fold_changes(avg_diffs: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """Compute the overall average difference (fold change) per feature index.

    Args:
        avg_diffs: {feature_index: node_index: average difference}

    Returns:
        {feature_index: overall average difference}
    """
    feature_fold_changes = {}
    for feat_idx, node_dict in avg_diffs.items():
        total_diff = sum(node_dict.values())
        num_nodes = len(node_dict)
        avg_diff = total_diff / num_nodes if num_nodes else 0
        feature_fold_changes[feat_idx] = avg_diff
    return feature_fold_changes


def get_top_n_nodes(
    avg_diffs_for_feature: Dict[int, float],
    n: int = 100,
) -> List[Tuple[int, float]]:
    """Get the top N nodes by absolute difference.

    Args:
        avg_diffs_for_feature: {node_idx: difference}
        n: Number of top nodes to retrieve.

    Returns:
        List of (node_idx, diff) sorted in descending order of absolute
        difference.
    """
    sorted_node_diffs = sorted(
        avg_diffs_for_feature.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return sorted_node_diffs[:n]


def map_nodes_to_symbols(
    node_diff_list: List[Tuple[int, float]],
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str],
) -> List[Tuple[str, float]]:
    """Map node indices to gene symbols for pairs of (node_idx, difference).

    Args:
        node_diff_list: (node_idx, difference) pairs.
        node_idx_to_gene_id: maps node index to gene id
        gencode_to_symbol: maps gencode ids to gene symbols.

    Returns:
        List of (gene_symbol_or_node_idx, difference)
    """
    result = []
    for node_idx, diff_value in node_diff_list:
        gene_id = node_idx_to_gene_id.get(node_idx)
        if gene_id and "ENSG" in gene_id:
            gencode_id = gene_id.split("_")[0]
            symbol = gencode_to_symbol.get(gencode_id, gene_id)
            result.append((symbol, diff_value))
        else:
            result.append((str(node_idx), diff_value))
    return result


def get_top_feature_genes(
    feature_indices: List[int],
    avg_diffs: Dict[int, Dict[int, float]],
    top_n: int,
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str],
) -> Dict[int, List[Tuple[str, float]]]:
    """Get top nodes by average difference for each feature index and map them
    to gene symbols.
    """
    feature_top_genes = {}
    for feat_idx in feature_indices:
        top_nodes = get_top_n_nodes(avg_diffs[feat_idx], n=top_n)
        top_genes = map_nodes_to_symbols(
            top_nodes,
            node_idx_to_gene_id=node_idx_to_gene_id,
            gencode_to_symbol=gencode_to_symbol,
        )
        feature_top_genes[feat_idx] = top_genes

    return feature_top_genes


def perturb_node_features(
    data: Data,
    runner: PerturbRunner,
    feature_indices: List[int],
    mask: str,
    device: torch.device,
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str],
    top_n: int = 100,
) -> Tuple[Dict[int, float], Dict[int, List[Tuple[str, float]]]]:
    """Zero out specified node features and measure the impact on model output.

    Computes the output per batch, zeroes out a selected feature, and computes
    the difference. Differneces are then averaged per node.

    Args:
        data: torch_geometric Data object
        runner: Runner holding the model to test
        feature_indices: feature indices to zero out
        mask: Mask name (e.g., 'test', 'val', etc.)
        device: CPU or GPU device.
        node_idx_to_gene_id: maps node index to gene id
        output_prefix: Where to write output files
        sample (str): Sample name for file naming
        top_n (int): Number of top genes to save

    Returns:
        ({feature_index: overall average diff}, {feature_index: list of top gene diffs})
    """
    test_loader = get_test_loader(data, mask)
    global_differences = defaultdict(lambda: defaultdict(list))

    for batch in tqdm(test_loader, desc="Node Feature Perturbation"):
        batch = batch.to(device)
        mask_tensor = getattr(batch, f"{mask}_mask_loss")

        # skip if no gene nodes present
        if mask_tensor.sum() == 0:
            continue

        baseline_out = compute_baseline_output(runner, batch, mask_tensor)

        # compute differences for each feature index
        batch_differences = compute_feature_perturbation(
            runner=runner,
            batch=batch,
            mask_tensor=mask_tensor,
            baseline_out=baseline_out,
            feature_indices=feature_indices,
            device=device,
        )

        # aggregate differences across batches
        for feat_idx, node_dict in batch_differences.items():
            for node_idx, diffs in node_dict.items():
                global_differences[feat_idx][node_idx].extend(diffs)

    # compute avg difference and fold changes
    avg_diffs = compute_average_node_differences(global_differences)
    feature_fold_changes = compute_fold_changes(avg_diffs)

    # get top nodes and map to gene symbols
    feature_top_genes = get_top_feature_genes(
        feature_indices=feature_indices,
        avg_diffs=avg_diffs,
        top_n=top_n,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=gencode_to_symbol,
    )

    return feature_fold_changes, feature_top_genes
