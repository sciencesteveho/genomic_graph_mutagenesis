#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Concatenate multiple graphs into a single graph and save their respective
indexes"""


import argparse
from pathlib import Path
import pickle
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np

from omics_graph_learning.utils.common import check_and_symlink
from omics_graph_learning.utils.common import NumpyGraphChecker
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.constants import TARGET_FILE
from omics_graph_learning.utils.constants import TRAINING_SPLIT_FILE

logger = setup_logging()


def _load_graph_and_idxs(
    tissue_graph_path: str,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Open graph and idxs from file."""
    with open(f"{tissue_graph_path}.pkl", "rb") as graph_file, open(
        f"{tissue_graph_path}_idxs.pkl", "rb"
    ) as idxs_file:
        graph = pickle.load(graph_file)
        idxs = pickle.load(idxs_file)
    return graph, idxs


def _reindex_idxs(idxs: Dict[str, int], new_start_idx: int) -> Dict[str, int]:
    """Reindex nodes to integers by adding the start_idx to each node."""
    return {key: value + new_start_idx for key, value in idxs.items()}


def _reindex_edges(edges: np.ndarray, new_start_idx: int) -> np.ndarray:
    """Reindex edges to integers by adding the start_idx to each node."""
    edges += new_start_idx
    return edges


def combine_splits(tissues: List[str], split_directory: Path) -> Dict[str, List[str]]:
    """Concatenate the splits for each tissue into a single split."""
    if len(tissues) <= 1:
        logger.info(f"Only one tissue provided: {tissues[0]}")
        split = pickle.load(
            open(split_directory / f"training_split_{tissues[0]}.pkl", "rb")
        )
        for key in ["train", "test", "validation"]:
            split[key] = [f"{gene}_{tissues[0]}" for gene in split[key]]
        return split

    logger.info(f"Concatenating splits for tissues: {tissues}")
    result: Dict[str, List[str]] = {"train": [], "test": [], "validation": []}
    for tissue in tissues:
        split = pickle.load(
            open(split_directory / f"training_split_{tissue}.pkl", "rb")
        )
        for key in ["train", "test", "validation"]:
            result[key] += [f"{gene}_{tissue}" for gene in split[key]]
    return result


def combine_targets(
    tissues: List[str], target_directory: Path
) -> Dict[str, Dict[str, np.ndarray]]:
    """Takes the targets from different tissues and concatenates them to create
    one target dictionary with train / test / val keys.."""
    if len(tissues) <= 1:
        logger.info(f"Only one tissue provided: {tissues[0]}")
        return pickle.load(
            open(target_directory / f"training_targets_{tissues[0]}.pkl", "rb")
        )

    logger.info(f"Concatenating targets for tissues: {tissues}")
    result: Dict[str, Dict[str, np.ndarray]] = {
        "train": {},
        "test": {},
        "validation": {},
    }
    for tissue in tissues:
        with open(target_directory / f"training_targets_{tissue}.pkl", "rb") as f:
            targets = pickle.load(f)
        for key in result:
            for subkey, value in targets[key].items():
                result[key][subkey] = value
    return result


def _concatenate_graphs(
    tissues: List[str], base_graph_path: str, experiment_graph_directory: Path
) -> None:
    """Combine the graphs for multiple tissues into a single graph."""
    logger.info(f"Concatenating graphs for tissues: {tissues}")
    concat_graph, concat_idxs = _load_graph_and_idxs(base_graph_path)
    concat_edges = concat_graph["edge_index"]
    concat_edge_feat = concat_graph["edge_feat"]
    concat_node_feat = concat_graph["node_feat"]
    concat_node_positional_encoding = concat_graph["node_positional_encoding"]
    concat_node_coordinates = concat_graph["node_coordinates"]
    concat_num_nodes = concat_graph["num_nodes"]
    concat_num_edges = concat_graph["num_edges"]

    for tissue in tissues[1:]:
        base_graph_path = f"{experiment_graph_directory}_{tissue}"
        graph, idxs = _load_graph_and_idxs(base_graph_path)
        end_idx = max(concat_idxs.values()) + 1
        concat_idxs.update(_reindex_idxs(idxs=idxs, new_start_idx=end_idx))
        concat_edges = np.hstack(
            (
                concat_edges,
                _reindex_edges(edges=graph["edge_index"], new_start_idx=end_idx),
            )
        )
        concat_edge_feat = np.concatenate((concat_edge_feat, graph["edge_feat"]))
        concat_node_feat = np.concatenate((concat_node_feat, graph["node_feat"]))
        concat_node_positional_encoding = np.concatenate(
            (concat_node_positional_encoding, graph["node_positional_encoding"])
        )
        concat_node_coordinates = np.concatenate(
            (concat_node_coordinates, graph["node_coordinates"])
        )
        concat_num_nodes += graph["num_nodes"]
        concat_num_edges += graph["num_edges"]

    with open(f"{experiment_graph_directory}.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": concat_edges,
                "node_feat": concat_node_feat,
                "node_positional_encoding": concat_node_positional_encoding,
                "node_coordinates": concat_node_coordinates,
                "edge_feat": concat_edge_feat,
                "num_nodes": concat_num_nodes,
                "num_edges": concat_num_edges,
                "avg_edges": concat_num_edges / concat_num_nodes,
            },
            output,
            protocol=4,
        )

    with open(f"{experiment_graph_directory}_idxs.pkl", "wb") as output:
        pickle.dump(concat_idxs, output, protocol=4)


def combine_graphs(
    experiment_graph_directory: Path,
    tissues: List[str],
) -> None:
    """Concatenate multiple graphs into a single graph. The first tissue in the
    list is used as the base and all subsequent graphs are reindexed then
    concatenated.

    Args:
        experiment_graph_directory (str): The experiment_graph_directory of the
        graph file names.
        tissues (List[str]): The list of tissue names.
    """
    base_graph_path = f"{experiment_graph_directory}_{tissues[0]}"

    if len(tissues) > 1:
        _concatenate_graphs(tissues, base_graph_path, experiment_graph_directory)
    else:
        # symlink if only one tissue
        for suffix in [".pkl", "_idxs.pkl"]:
            check_and_symlink(
                f"{base_graph_path}{suffix}",
                f"{experiment_graph_directory}{suffix}",
            )


def main() -> None:
    """Pipeline to concatenate tissue graphs"""
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
        required=True,
    )
    parser.add_argument(
        "--split_name",
        type=str,
        help="Name of the split to be concatenated",
        required=True,
    )
    args = parser.parse_args()
    params = ExperimentConfig.from_yaml(args.experiment_config)
    graph_type = params.graph_type

    # set up dirs
    split_dir = params.graph_dir / args.split_name
    experiment_graph_directory = (
        split_dir / f"{params.experiment_name}_{graph_type}_graph"
    )

    # combine training splits
    splits = combine_splits(tissues=params.tissues, split_directory=split_dir)
    with open(split_dir / TRAINING_SPLIT_FILE, "wb") as output:
        pickle.dump(splits, output, protocol=4)

    # combine all targets
    targets = combine_targets(tissues=params.tissues, target_directory=split_dir)
    with open(split_dir / TARGET_FILE, "wb") as output:
        pickle.dump(targets, output, protocol=4)

    # concat all graphs
    combine_graphs(
        experiment_graph_directory=experiment_graph_directory, tissues=params.tissues
    )

    # sanity check graph data
    NumpyGraphChecker.check_numpy_graph_data(
        pickle.load(open(f"{experiment_graph_directory}.pkl", "rb"))
    )


if __name__ == "__main__":
    main()
