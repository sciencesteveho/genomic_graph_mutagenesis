#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Helper functions to checking connected components of a graph.

Our model design revolves around easy perturbations of an input graph. Because
GNNs train by successive hops, only their connected components are important. We
implement helper functions to derive connected components, to return
dictionaries of connected node types, and to create masks for perturbation
analysis."""


import pickle
from typing import Dict, List, Set, Tuple

import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import connected_components  # type: ignore


def reverse_mapping_index(graph_idxs: Dict[str, int]) -> Dict[int, str]:
    """Create a reverse mapping from node indices to node identifiers."""
    return {index: identifier for identifier, index in graph_idxs.items()}


def get_connected_components(data: Data) -> List[Set[int]]:
    """Get connected components from the graph data.

    Returns:
        List[Set[int]]: A list where each element is a set of node indices
        forming a connected component.
    """
    return connected_components(data.edge_index, num_nodes=data.num_nodes)


def get_component_subgraph(data: Data, node_indices: Set[int]) -> Data:
    """Extract a subgraph corresponding to a connected component.

    Returns:
        Data: The subgraph data of the connected component.
    """
    # create a mask for the nodes in the component
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask_indices = list(node_indices)
    mask[mask_indices] = True

    return data.subgraph(mask)


def get_gene_and_cre_associations(
    component_nodes: Set[int],
    reverse_graph_idxs: Dict[int, str],
) -> Tuple[Set[int], Set[int]]:
    """Get sets of gene nodes and CRE nodes within a connected component.

    Args:
        component_nodes (Set[int]): Set of node indices in the connected
        component.
        reverse_graph_idxs (Dict[int, str]): Mapping from node
        indices to node identifiers.

    Returns:
        Tuple[Set[int], Set[int]]: Sets of gene node indices and CRE node
        indices.
    """
    gene_nodes = set()
    cre_nodes = set()
    for node_idx in component_nodes:
        node_identifier = reverse_graph_idxs.get(node_idx, "")
        if "ENSG" in node_identifier:
            gene_nodes.add(node_idx)
        elif any(
            keyword in node_identifier.lower()
            for keyword in ["enhancer", "dyadic", "promoter"]
        ):
            cre_nodes.add(node_idx)
    return gene_nodes, cre_nodes


def load_connected_components(
    data: Data,
    reverse_graph_idxs: Dict[int, str],
) -> Tuple[List[Data], List[Set[int]], List[Set[int]]]:
    """
    Load connected components and their gene and CRE associations.

    Args:
        data (Data): The original graph data.
        reverse_graph_idxs (Dict[int, str]): Reverse mapping from node indices to node identifiers.

    Returns:
        Tuple[List[Data], List[Set[int]], List[Set[int]]]:
            - List of subgraphs for each connected component.
            - List of gene node sets for each connected component.
            - List of CRE node sets for each connected component.
    """
    components = get_connected_components(data)
    subgraphs = []
    gene_associations = []
    cre_associations = []

    for component_nodes in components:
        subgraph = get_component_subgraph(data, component_nodes)
        subgraphs.append(subgraph)

        gene_nodes, cre_nodes = get_gene_and_cre_associations(
            component_nodes, reverse_graph_idxs
        )
        gene_associations.append(gene_nodes)
        cre_associations.append(cre_nodes)

    return subgraphs, gene_associations, cre_associations
