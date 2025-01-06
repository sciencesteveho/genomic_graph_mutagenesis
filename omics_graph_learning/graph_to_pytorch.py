#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Convert graphs from np tensors to pytorch geometric Data objects.

Graphs are partitioned into train, test, and validation sets based on the
provided split. Data are prepared as pytorch tensors and stored in a pytorch
geometric Data object. Two masks are produced:

    "split" masks are used for the dataloaders, ensuring that all nodes for the
    loader are referenced
    "split_loss" masks are a subset of the "split" masks containing only genes
    that passed the preceding filter steps for loss calculation (i.e. loss is
    only calculated over these gene nodes)
    
Additionally code is provided to subset the chromosomes for hyperparameter
tuning. List of chrs is explictly defined in constants.py."""


from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data  # type: ignore

from omics_graph_learning.interpret.perturb_graph import get_node_perturbation
from omics_graph_learning.interpret.perturb_graph import perturb_edge_index
from omics_graph_learning.interpret.perturb_graph import perturb_node_features
from omics_graph_learning.interpret.perturb_graph import PerturbationConfig
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.constants import NodePerturbation
from omics_graph_learning.utils.constants import SUBSET_CHROMOSOMES
from omics_graph_learning.utils.constants import TARGET_FILE
from omics_graph_learning.utils.constants import TARGET_FILE_SCALED
from omics_graph_learning.utils.constants import TRAINING_SPLIT_FILE


class GraphToPytorch:
    """Class to handle the conversion of graphs stored as np.arrays to tensors
    and ultimately to Pytorch Geometric Data objects.

    Uses class PerturbationConfig to handle potential graph perturbations.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        split_name: str,
        regression_target: str,
        positional_encoding: Optional[bool],
        scaled_targets: bool = False,
        perturbation_config: Optional[PerturbationConfig] = None,
    ) -> None:
        """Initialize the GraphToPytorch object."""
        self.experiment_config = experiment_config
        self.split_name = split_name
        self.regression_target = regression_target
        self.scaled_targets = scaled_targets
        self.perturbation_config = perturbation_config

        if positional_encoding is not None:
            self.train_positional_encoding = positional_encoding
        else:
            self.train_positional_encoding = experiment_config.train_positional_encoding

        self.graph_indexes, self.split, self.graph_data, self.targets = (
            self._load_data_object_prerequisites(
                experiment_config=experiment_config,
                graph_type=experiment_config.graph_type,
                split_name=split_name,
                scaled_targets=scaled_targets,
            )
        )

    def generate_edge_index(self) -> torch.Tensor:
        """Create the edge index tensor for the graph."""
        edge_perturbation = None
        total_random_edges = None
        node_remove_edges = None

        if self.perturbation_config is not None:
            edge_perturbation = self.perturbation_config.edge_perturbation
            total_random_edges = self.perturbation_config.total_random_edges
            node_remove_edges = self.perturbation_config.node_remove_edges

        return create_edge_index(
            graph_data=self.graph_data,
            edge_perturbation=edge_perturbation,
            total_random_edges=total_random_edges,
            node_idxs=node_remove_edges,
        )

    def create_node_tensors(self) -> torch.Tensor:
        """Create the node tensors for the graph."""
        if self.perturbation_config is not None:
            node_perturbation = get_node_perturbation(
                self.perturbation_config.node_perturbation
            )
        else:
            node_perturbation = None

        return create_node_tensors(
            graph_data=self.graph_data,
            perturbation=node_perturbation,
            positional_encoding=self.train_positional_encoding,
        )

    def instantiate_data_object(self) -> Data:
        """Create the PyG Data object."""
        edge_index = self.generate_edge_index()
        node_tensors = self.create_node_tensors()
        return Data(x=node_tensors.contiguous(), edge_index=edge_index.contiguous())

    def create_masks(self, data: Data) -> Dict[str, torch.Tensor]:
        """Create masks for the Data object."""
        # get mask indexes for loss calculation
        train_loss_idxs, test_loss_idxs, val_loss_idxs, all_loss_idxs = _get_mask_idxs(
            graph_index=self.graph_indexes, split=self.split
        )

        # get mask indexes for training (masks include all nodes that fall on the chr)
        train_idxs, test_idxs, val_idxs, optimization_idxs = _assign_nodes_to_split(
            graph_data=self.graph_data,
            test_chrs=self.experiment_config.test_chrs,
            val_chrs=self.experiment_config.val_chrs,
        )

        return {
            "train_mask_loss": create_mask(data.num_nodes, train_loss_idxs),
            "test_mask_loss": create_mask(data.num_nodes, test_loss_idxs),
            "val_mask_loss": create_mask(data.num_nodes, val_loss_idxs),
            "all_mask_loss": create_mask(data.num_nodes, all_loss_idxs),
            "train_mask": create_mask(data.num_nodes, train_idxs),
            "test_mask": create_mask(data.num_nodes, test_idxs),
            "val_mask": create_mask(data.num_nodes, val_idxs),
            "optimization_mask": create_mask(data.num_nodes, optimization_idxs),
        }

    def create_target_tensor(self, data: Data) -> torch.Tensor:
        """Create the target values tensor for the Data object."""
        # get values for the specified target type
        target_values = _get_target_values_for_mask(targets=self.targets)
        indices = _get_target_indices(self.regression_target)

        # align target value with graph index
        remapped = {
            self.graph_indexes[target]: target_values[target]
            for target in target_values
            if target in self.graph_indexes
        }

        # get list of masked tensors, one tensor for each target index
        y_tensors = [_get_masked_tensor(data.num_nodes) for _ in indices]

        # populate the masked tensors with the target values
        for idx, tensor in enumerate(y_tensors):
            for key, values in remapped.items():
                tensor[key] = values[indices[idx]]

        return torch.stack(y_tensors).view(len(y_tensors), -1)

    def save_average_edges(self, data: Data) -> None:
        """Save average edges for the graph as part of the data object."""
        data.avg_edges = torch.tensor(
            round(self.graph_data["avg_edges"]), dtype=torch.int
        )

    def make_data_object(self) -> Data:
        """Create the PyG Data object."""
        # create the data object with node features and edge index
        data = self.instantiate_data_object()

        # check for NaN or infinite values in node features
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            print("Warning: NaN or infinite values found in node features")

        masks = self.create_masks(data=data)
        targets = self.create_target_tensor(data=data)

        # check for NaN or infinite values in targets
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN or infinite values found in target values")

        # add masks to the data object
        for mask_name, mask in masks.items():
            setattr(data, mask_name, mask.contiguous())

        # add regression targets to the data object
        data.y = targets.T.contiguous()

        # add class labels to the data object
        # class 1 if log2 TPM >= 0, else Class 0
        data.class_labels = (data.y >= 0).long()

        # save average edges for the graph
        self.save_average_edges(data=data)

        return data

    @staticmethod
    def _load_data_object_prerequisites(
        experiment_config: ExperimentConfig,
        graph_type: str,
        split_name: str,
        scaled_targets: bool = False,
    ) -> Tuple[Any, Any, Any, Dict[str, Dict[str, np.ndarray]]]:
        """Prepare specific variables needed to make the PyG Data object."""

        def load_pickle(file_path: Path) -> Any:
            """Load pickle!"""
            with open(file_path, "rb") as f:
                return pickle.load(f)

        experiment_name = experiment_config.experiment_name
        experiment_dir = experiment_config.graph_dir / split_name

        file_paths = {
            "index": experiment_dir / f"{experiment_name}_{graph_type}_graph_idxs.pkl",
            "split": experiment_dir / TRAINING_SPLIT_FILE,
            "target": experiment_dir
            / (TARGET_FILE_SCALED if scaled_targets else TARGET_FILE),
            "graph_data": experiment_dir / f"{experiment_name}_{graph_type}_scaled.pkl",
        }

        return (
            load_pickle(file_paths["index"]),
            load_pickle(file_paths["split"]),
            load_pickle(file_paths["graph_data"]),
            load_pickle(file_paths["target"]),
        )


def get_subset_chromosomes(num_chrs: int = 8) -> List[str]:
    """Return the subset of chromosomes to use for hyperparameter
    optimization.
    """
    return SUBSET_CHROMOSOMES[:num_chrs]


def _assign_nodes_to_split(
    graph_data: Dict[str, Any],
    test_chrs: List[str],
    val_chrs: List[str],
    num_optimization_chrs: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign nodes to train, test, and validation sets according to their chr.

    Args:
        graph (nx.Graph): The graph object.
        test_chrs (List[str]): The list of test chromosomes.
        val_chrs (List[str]): The list of validation chromosomes.
        num_optimization_chrs (int): The number of chromosomes to use
        hyperparameter tuning. Suggested are 8, 10, or 12, which are roughly
        40/50/60% of the training data.
    """
    coordinates = graph_data["node_coordinates"]
    train, test, validation, train_subset = [], [], [], []

    subset_train_chrs = get_subset_chromosomes(num_chrs=num_optimization_chrs)

    for node in range(graph_data["num_nodes"]):
        chromosome = coordinates[node][0]
        if chromosome in test_chrs:
            test.append(node)
        elif chromosome in val_chrs:
            validation.append(node)
        elif chromosome in subset_train_chrs:
            train_subset.append(node)
            train.append(node)
        else:
            train.append(node)

    return (
        torch.tensor(train, dtype=torch.long),
        torch.tensor(test, dtype=torch.long),
        torch.tensor(validation, dtype=torch.long),
        torch.tensor(train_subset, dtype=torch.long),
    )


def _get_mask_idxs(
    graph_index: Dict[str, int],
    split: Dict[str, List[str]],
    percentile_cutoff: Optional[int] = None,
    cutoff_file: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the mask indexes for train, test, and validation sets.

    Args:
        index (str): The path to the index file.
        split (Dict[str, List[str]]): The split dictionary containing train,
        test, and validation gene lists.
        percentile_cutoff (int, optional): The percentile cutoff for test genes.
        Default is None.
        cutoff_file (str, optional): The path to the cutoff file. Default is
        None.
    """

    def get_tensor_for_genes(gene_list: List[str]) -> torch.Tensor:
        """Return a tensor of gene indices for the provided gene list."""
        return torch.tensor(
            [graph_index[gene] for gene in gene_list if gene in graph_index],
            dtype=torch.long,
        )

    # masks used for calculating loss, only contain gene nodes
    all_genes = split["train"] + split["test"] + split["validation"]
    train_mask_loss = get_tensor_for_genes(split["train"])
    val_mask_loss = get_tensor_for_genes(split["validation"])
    all_genes_mask = get_tensor_for_genes(all_genes)

    if percentile_cutoff:
        with open(
            f"{cutoff_file}_{percentile_cutoff}.pkl",
            "rb",
        ) as f:
            test_genes = pickle.load(f)
        test_genes = list(test_genes.keys())
        test_mask_loss = torch.tensor(
            [graph_index.get(gene, -1) for gene in test_genes],
            dtype=torch.long,
        )
    else:
        test_mask_loss = get_tensor_for_genes(split["test"])

    return (
        train_mask_loss,
        test_mask_loss,
        val_mask_loss,
        all_genes_mask,
    )


def _get_target_values_for_mask(
    targets: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Returns a collapsed dictionary of the targets for each split"""
    # load graph indexes
    return {
        k: v
        for split in ["train", "test", "validation"]
        for k, v in targets[split].items()
    }


def _get_masked_tensor(
    num_nodes: int,
    fill_value: int = -1,
) -> torch.Tensor:
    """Returns a tensor of size (num_nodes) with fill_value (default -1)"""
    return torch.full((num_nodes,), fill_value, dtype=torch.float)


def create_edge_index(
    graph_data: dict,
    edge_perturbation: Optional[str],
    total_random_edges: Optional[int],
    node_idxs: Optional[List[str]],
) -> torch.Tensor:
    """Create the edge index tensor for the graph, reversing the edge index to
    ensure an undirected graph.

    Additionally, function will perturb if a perturbation type is provided.

    Args:
        graph_data (dict): The graph data dictionary.
        edge_perturbation (Optional[str]): The type of edge perturbation to apply.
        total_random_edges (Optional[int]): The total number of random edges to
        create. If not specified, amount of edges is equal to the amount of
        edges in the graph.
        node_idxs (Optional[List[str]]): The node indices to remove if
        perturbation type is `remove_specific_edges`.
    """
    if edge_perturbation:
        edge_index = torch.tensor(
            perturb_edge_index(
                edge_index=graph_data["edge_index"],
                edge_perturbation=edge_perturbation,
                node_idxs=node_idxs,
                total_random_edges=total_random_edges,
            ),
            dtype=torch.long,
        )
    else:
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)

    # add reverse edges
    edge_index_reverse = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_reverse], dim=1)
    edge_index = edge_index.unique(dim=1)  # remove duplicate edges

    # validate edge indices
    num_nodes = graph_data["num_nodes"]
    if edge_index.max() >= num_nodes or edge_index.min() < 0:
        print("Warning: Invalid edge indices found")
    return edge_index


def create_node_tensors(
    graph_data: dict,
    perturbation: Optional[NodePerturbation] = None,
    positional_encoding: bool = False,
) -> torch.Tensor:
    """Create the node tensors for the graph, perturbing if necessary according
    to input args.

    Args:
        graph_data (dict): The graph data dictionary.
        node_perturbation (Optional[str]): The type of node perturbation to
        apply. Default is None.
        zero_node_feats (bool): Flag indicating whether to zero out the node
        features. Default is False.
        randomize_feats (bool): Flag indicating whether to randomize the node
        features. Default is False.
    """
    if positional_encoding:
        return torch.tensor(
            np.concatenate(
                (graph_data["node_feat"], graph_data["node_positional_encoding"]),
                axis=1,
            ),
            dtype=torch.float,
        )
    elif perturbation is not None:
        return torch.tensor(
            perturb_node_features(perturbation, graph_data["node_feat"]),
            dtype=torch.float,
        )
    else:
        return torch.tensor(graph_data["node_feat"], dtype=torch.float)


def create_mask(num_nodes: int, indices: torch.Tensor) -> torch.Tensor:
    """Create a boolean mask tensor for the provided indices.

    Args:
        num_nodes (int): The number of nodes.
        indices (List[int]): The indices to set to True in the mask.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    indices = indices.to(torch.long)
    mask[indices] = True
    return mask


def _get_target_indices(regression_target: str) -> List[int]:
    """Return the indices of the target values to use for the regression
    target.
    """
    target_indices = {
        "expression_median_only": [0],
        "expression_median_and_foldchange": [0, 1],
        "difference_from_average": [2],
        "foldchange_from_average": [3],
        "protein_targets": [0, 1, 2, 3, 4, 5],
        "rna_seq": [0],
    }
    if regression_target not in target_indices:
        raise ValueError("Invalid regression_target provided.")
    return target_indices[regression_target]
