#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturb connected components and measure impact on model output."""


import random
from typing import Any, Dict, List, Optional

import networkx as nx  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.perturb_runner import PerturbRunner


def create_subgraph_loader(
    data: Data,
    gene_node: int,
    num_hops: int,
    batch_size: int = 1,
) -> NeighborLoader:
    """Create a NeighborLoader to fetches subgraph around a specific gene node."""
    return NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * num_hops,
        batch_size=batch_size,
        input_nodes=torch.tensor([gene_node], dtype=torch.long),
        shuffle=False,
    )


def compute_baseline_prediction(
    runner: PerturbRunner,
    sub_data: Data,
    gene_node: int,
    mask_attr: str = "all",
) -> float:
    """Compute the baseline prediction for a given gene node in the subgraph.

    Args:
        runner: PerturbRunner object containing the loaded model
        sub_data: Data subgraph batch
        gene_node: idx of the gene of interest
        mask_attr: Attribute name for the mask

    Returns:
        float: The baseline prediction for the gene node.
    """
    idx_in_subgraph = (sub_data.n_id == gene_node).nonzero(as_tuple=True)[0].item()
    mask_tensor = getattr(sub_data, f"{mask_attr}_mask_loss")

    with torch.no_grad():
        baseline_out, _ = runner.model(
            x=sub_data.x,
            edge_index=sub_data.edge_index,
            mask=mask_tensor,
        )
    return baseline_out[idx_in_subgraph].item()


def get_nodes_to_perturb(
    sub_data: Data,
    gene_node: int,
    max_nodes_to_perturb: int,
) -> List[int]:
    """Get list of candidate nodes to remove (excludes the gene_node itself). If
    more nodes are specified than are available, randomly sample from the
    available nodes.

    Args:
        sub_data: Data subgraph batch
        gene_node: idx of the gene of interest
        max_nodes_to_perturb: Max number of nodes to remove at once

    Returns:
        List of node indices to remove
    """
    # exclude the gene node itself
    nodes_to_perturb = sub_data.n_id[sub_data.n_id != gene_node]

    if len(nodes_to_perturb) == 0:
        return []

    return (
        random.sample(nodes_to_perturb.tolist(), max_nodes_to_perturb)
        if len(nodes_to_perturb) > max_nodes_to_perturb
        else nodes_to_perturb.tolist()
    )


def compute_hop_distances(sub_data: Data, gene_node: int) -> Dict[int, int]:
    """Compute hop distances from the gene_node to every other node in the
    subgraph.

    Returns:
        {node: hop_distance from gene_node}
    """
    # convert to CPU if necessary
    sub_data_cpu = sub_data.clone().cpu()

    # construct a networkx subgraph
    subgraph_nx = to_networkx(sub_data_cpu, to_undirected=True)
    mapping_nx = {i: sub_data_cpu.n_id[i].item() for i in range(sub_data_cpu.num_nodes)}
    subgraph_nx = nx.relabel_nodes(subgraph_nx, mapping_nx)

    return nx.single_source_shortest_path_length(subgraph_nx, gene_node)


def remove_node_and_predict(
    runner: PerturbRunner,
    sub_data: Data,
    node_to_remove: int,
    gene_node: int,
    device: torch.device,
    mask_attr: str = "all",
) -> Optional[float]:
    """Remove the specified node from the subgraph, then compute the model's
    prediction for the gene node in the perturbed subgraph.

    Args:
        runner: PerturbRunner object containing the loaded model.
        sub_data: Subgraph batch.
        node_to_remove: Node to remove from the subgraph.
        gene_node: Gene node to predict.
        device: Device to run the model on.
        mask_attr: Attribute name for the mask.

    Returns:
        Optional[float]: The perturbation prediction for the gene node, or None
        if the gene_node is missing in the perturbed subgraph.
    """
    mask_tensor = getattr(sub_data, f"{mask_attr}_mask_loss")

    try:
        idx_to_remove = (
            (sub_data.n_id == node_to_remove).nonzero(as_tuple=True)[0].item()
        )
    except IndexError:
        return None  # node not found in subgraph

    mask_nodes = torch.arange(sub_data.num_nodes, device=device) != idx_to_remove
    perturbed_edge_idx, _, _ = subgraph(
        subset=mask_nodes,
        edge_index=sub_data.edge_index,
        relabel_nodes=True,
        num_nodes=sub_data.num_nodes,
        return_edge_mask=True,
    )

    perturbed_x = sub_data.x[mask_nodes]
    perturbed_mask = mask_tensor[mask_nodes]
    perturbed_n_id = sub_data.n_id[mask_nodes]

    # ensure gene_node has not been removed
    if (perturbed_n_id == gene_node).sum() == 0:
        return None

    idx_in_perturbed = (perturbed_n_id == gene_node).nonzero(as_tuple=True)[0].item()
    with torch.no_grad():
        perturbed_out, _ = runner.model(
            x=perturbed_x,
            edge_index=perturbed_edge_idx,
            mask=perturbed_mask,
        )
    return perturbed_out[idx_in_perturbed].item()


def perturb_connected_components(
    data: Data,
    device: torch.device,
    runner: PerturbRunner,
    top_gene_nodes: List[int],
    idxs_inv: Dict[int, str],
    num_hops: int = 6,
    max_nodes_to_perturb: int = 100,
    mask_attr: str = "all",
) -> Dict[str, Dict[str, Any]]:
    """Compute the impact of removing nodes from connected components on model output.

    Computes baseline prediction for the node, selects candidate nodes to
    remove, then removes each node computing the log2 fold-change from baseline.

    Args:
        data: PyG Data object containing the graph
        device: Device to run the model on.
        runner: PerturbRunner object containing the loaded model.
        top_gene_nodes: List of top gene nodes to analyze.
        idxs_inv: Mapping from node index: gene identifier.
        num_hops: Number of hops to fetch when building subgraphs.
        max_nodes_to_perturb: Max number of nodes to remove per gene.
        mask_attr: Mask attribute name.

    Returns:
        {gene_id: node_removed: {fold_change, hop_distance}}
    """
    gene_fold_changes: Dict[str, Dict[str, Any]] = {}

    for gene_node in tqdm(top_gene_nodes, desc="Connected Component Perturbation"):
        gene_id = idxs_inv.get(gene_node, str(gene_node))

        loader = create_subgraph_loader(data, gene_node, num_hops)
        sub_data = next(iter(loader)).to(device)

        # get baseline
        baseline_prediction = compute_baseline_prediction(
            runner, sub_data, gene_node, device, mask_attr
        )

        # get nodes to perturb
        selected_nodes = get_nodes_to_perturb(sub_data, gene_node, max_nodes_to_perturb)
        if not selected_nodes:
            continue

        lengths = compute_hop_distances(sub_data, gene_node)

        # compute the new prediction & fold-change for each node to perturb
        fold_changes = {}
        for node_remove in selected_nodes:
            perturb_prediction = remove_node_and_predict(
                runner, sub_data, node_remove, gene_node, device, mask_attr
            )
            if perturb_prediction is None:
                continue

            fold_change = runner.calculate_log2_fold_change(
                baseline_prediction=baseline_prediction,
                perturbation_prediction=perturb_prediction,
            )
            node_name = idxs_inv.get(node_remove, str(node_remove))
            hop_distance = lengths.get(node_remove, -1)
            fold_changes[node_name] = {
                "fold_change": fold_change,
                "hop_distance": hop_distance,
            }

        gene_fold_changes[gene_id] = fold_changes

    return gene_fold_changes
