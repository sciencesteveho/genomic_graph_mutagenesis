#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to handle perturbations of the graph data."""


from dataclasses import dataclass
from typing import List, Optional

import networkx as nx  # type: ignore
import numpy as np
from torch_geometric.data import Data  # type: ignore

from omics_graph_learning.utils.constants import NodePerturbation


@dataclass
class PerturbationConfig:
    """Data class to store the perturbation configuration."""

    node_perturbation: Optional[str] = None
    edge_perturbation: Optional[str] = None
    node_remove_edges: Optional[List[str]] = None
    total_random_edges: Optional[int] = None
    remove_node: Optional[str] = None
    # single_gene: Optional[str] = None
    # percentile_cutoff: Optional[float] = None


def get_node_perturbation(
    perturbation_name: Optional[str],
) -> Optional[NodePerturbation]:
    """Return the perturbation enum value for the provided perturbation name."""
    if perturbation_name is not None:
        return NodePerturbation[perturbation_name]
    return None


def perturb_node_features(
    perturbation: NodePerturbation, node_features: np.ndarray
) -> np.ndarray:
    """Perturb node features according to the provided perturbation."""
    if perturbation.value >= 0:
        node_features[:, perturbation.value] = 0
        return node_features
    elif perturbation == NodePerturbation.zero_node_feats:
        return np.zeros(node_features.shape)
    elif perturbation == NodePerturbation.randomize_node_feats:
        return np.random.rand(*node_features.shape)
    elif perturbation == NodePerturbation.randomize_node_feat_order:
        return np.apply_along_axis(np.random.permutation, 1, node_features)
    else:
        raise ValueError(
            f"Invalid perturbation provided. Valid perturbations are: {', '.join([p.name for p in NodePerturbation])}"
        )


def remove_specific_edges(edge_index: np.ndarray, node_idxs: List[str]) -> np.ndarray:
    """Remove specific edges from the edge index."""
    return np.array(
        [
            np.delete(edge_index[0], np.array(node_idxs)),
            np.delete(edge_index[1], np.array(node_idxs)),
        ]
    )


def randomize_edges(
    edge_index: np.ndarray, total_random_edges: Optional[int]
) -> np.ndarray:
    """Randomize the edges in the edge index."""
    total_range = max(np.ptp(edge_index[0]), np.ptp(edge_index[1]))
    total_edges = total_random_edges or len(edge_index[0])
    return np.random.randint(0, total_range, (2, total_edges))


def perturb_edge_index(
    edge_perturbation: str,
    edge_index: np.ndarray,
    node_idxs: Optional[List[str]],
    total_random_edges: Optional[int],
) -> np.ndarray:
    """Perturb edge index according to the provided perturbation."""
    if edge_perturbation == "randomize_edges":
        return randomize_edges(
            edge_index=edge_index, total_random_edges=total_random_edges
        )
    elif edge_perturbation == "remove_all_edges":
        return np.array([], dtype=int).reshape(2, 0)
    elif edge_perturbation == "remove_specific_edges":
        if node_idxs:
            return remove_specific_edges(edge_index=edge_index, node_idxs=node_idxs)
        else:
            raise ValueError("node_idxs must be provided when removing specific edges")
    else:
        raise ValueError(
            f"Invalid edge perturbation provided. \
            Valid perturbations are: `remove_all_edges`,`remove_specific_edges`,`randomize_edge`."
        )


# def perturb_single_gene(
#     perturbation_config: PerturbationConfig, num_nodes: int
# ) -> torch.Tensor:
#     """Create a mask to remove a single gene from the graph data."""
#     gene_idx = torch.tensor([perturbation_config.single_gene], dtype=torch.long)
#     gene_mask = torch.zeros(num_nodes, dtype=torch.bool)
#     gene_mask[gene_idx] = True
#     return gene_mask
