#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturb coessential gene pairs and measure impact expression prediction."""


import random
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.perturb_runner import PerturbRunner


def load_coessential_pairs(
    pos_file: str, neg_file: str, idxs: Dict[str, int]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Load positive and negative coessential gene pairs, map them to node
    indices, and return two lists.
    """

    def read_pairs(file_path: str, expected_label: str) -> List[Tuple[int, int]]:
        """Read gene pairs from a file with the expected label and map them to
        indices.
        """
        pairs = []
        with open(file_path, "r") as f:
            for line in f:
                gene1, gene2, label = line.strip().split("\t")
                if label == expected_label and gene1 in idxs and gene2 in idxs:
                    pairs.append((idxs[gene1], idxs[gene2]))
        return pairs

    pos_pairs = read_pairs(pos_file, "pos")
    neg_pairs = read_pairs(neg_file, "neg")

    return pos_pairs, neg_pairs


def get_subgraph_around_gene(
    data: Data,
    gene_idx: int,
    num_hops: int,
    device: torch.device,
) -> Optional[Data]:
    """Fetch a subgraph centered on gene_idx using a neighborloader."""
    loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * num_hops,
        batch_size=1,
        input_nodes=torch.tensor([gene_idx], dtype=torch.long, device=device),
        shuffle=False,
    )
    try:
        return next(iter(loader)).to(device)
    except StopIteration:
        return None


def compute_baseline_expression(
    sub_data: Data,
    gene_idx: int,
    runner: PerturbRunner,
) -> float:
    """Compute baseline expression for gene_idx in sub_data."""
    local_idx = (sub_data.n_id == gene_idx).nonzero(as_tuple=True)
    if len(local_idx[0]) == 0:
        return float("nan")
    local_idx = local_idx[0].item()

    with torch.no_grad():
        out_sub, _ = runner.model(
            x=sub_data.x,
            edge_index=sub_data.edge_index,
            mask=sub_data.all_mask_loss,
        )
    return out_sub[local_idx].item()


def remove_node_and_get_expression(
    sub_data: Data,
    node_local_idx: int,
    gene_idx: int,
    runner: PerturbRunner,
    device: torch.device,
) -> Optional[float]:
    """Remove a node by local_idx from sub_data, then measure gene node's
    expression.

    Args:
        sub_data: subbatch data object node_local_idx: index of the node to
        remove. gene_idx: index of the gene node runner: PerturbRunner object
        with loaded model device: torch.device object
    """
    mask_nodes = torch.arange(sub_data.num_nodes, device=device) != node_local_idx
    edge_idx, _, _ = subgraph(
        subset=mask_nodes,
        edge_index=sub_data.edge_index,
        relabel_nodes=True,
        num_nodes=sub_data.num_nodes,
        return_edge_mask=True,
    )
    x_perturbed = sub_data.x[mask_nodes]
    mask_perturbed = sub_data.all_mask_loss[mask_nodes]
    n_id_perturbed = sub_data.n_id[mask_nodes]

    if (n_id_perturbed == gene_idx).sum() == 0:
        return None

    idx_gene_local = (n_id_perturbed == gene_idx).nonzero(as_tuple=True)[0].item()
    with torch.no_grad():
        out_perturbed, _ = runner.model(
            x=x_perturbed,
            edge_index=edge_idx,
            mask=mask_perturbed,
        )
    return out_perturbed[idx_gene_local].item()


def paired_gene_perturbation(
    data: Data,
    runner: PerturbRunner,
    pairs: List[Tuple[int, int]],
    num_hops: int = 6,
    random_comparison: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[float], List[float]]:
    """Measure expression changes when removing gene_2 from gene_1's subgraph.

    Args:
        data: PyG Data object of the graph
        runner: PerturbRunner object with loaded model
        _pairs (list of (gene_1_idx, gene_2_idx)): pairs of nodes
        num_hops: subgraph hop size
        random_comparison: Flag to remove a random node for comparison
        device: torch.device object

    Returns:
        Two lists of expression changes ([paired], [random]).
    """
    data = data.to(device)
    paired_changes, random_changes = [], []

    for gene1_idx, gene2_idx in tqdm(pairs, desc="Paired gene perturbation"):
        sub_data = get_subgraph_around_gene(data, gene1_idx, num_hops, device)
        if not sub_data:
            continue

        # check if gene2 is in the subgraph
        if (sub_data.n_id == gene2_idx).sum() == 0:
            continue

        # compute baseline
        baseline_expr = compute_baseline_expression(sub_data, gene1_idx, runner)
        if baseline_expr != baseline_expr:  # NaN check
            continue

        # remove gene2
        g2_local = (sub_data.n_id == gene2_idx).nonzero(as_tuple=True)[0].item()
        expr_after_removal = remove_node_and_get_expression(
            sub_data, g2_local, gene1_idx, runner, device
        )
        if expr_after_removal is not None:
            paired_changes.append(baseline_expr - expr_after_removal)

        # optional random removal for comparison
        if random_comparison:
            g1_local = (sub_data.n_id == gene1_idx).nonzero(as_tuple=True)[0].item()
            other_nodes = [
                i for i in range(sub_data.num_nodes) if i not in [g1_local, g2_local]
            ]
            if not other_nodes:
                continue
            rand_local = random.choice(other_nodes)
            expr_rand = remove_node_and_get_expression(
                sub_data, rand_local, gene1_idx, runner, device
            )
            if expr_rand is not None:
                random_changes.append(baseline_expr - expr_rand)

    return paired_changes, random_changes
