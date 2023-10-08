#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Metadata handler for graph creation, and inputs metadata into each step to
perform part 1 of the pipeline. Takes config to tell the next 3 step which
arguments to use."""

import argparse
from typing import Dict, List, Union

from edge_parser import EdgeParser
from local_context_parser import LocalContextParser
from prepare_bedfiles import GenomeDataPreprocessor
from utils import _listdir_isfile_wrapper
from utils import dir_check_make
from utils import LOOPFILES
from utils import parse_yaml
from utils import POSSIBLE_NODES

NODES = [
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
]


def preprocess_bedfiles(
    experiment_params: Dict[str, Union[str, list]],
    tissue_params: Dict[str, Union[str, list]],
    nodes: List[str],
) -> None:
    """Directory set-up, bedfile symlinking and preprocessing

    Args:
        experiment_params (Dict[str, Union[str, list]]): experiment config
        tissue_params (Dict[str, Union[str, list]]): tissie-specific configs
        nodes (List[str]): nodes to include in graph
    """
    preprocessObject = GenomeDataPreprocessor(
        experiment_name=experiment_params["experiment_name"],
        interaction_types=experiment_params["interaction_types"],
        nodes=nodes,
        working_directory=experiment_params["working_directory"],
        params=tissue_params,
    )

    preprocessObject.prepare_data_files()
    print("Bedfile preprocessing complete!")


def parse_edges(experiment_params, tissue_params):
    baseloop_directory = experiment_params["baseloop_directory"]
    baseloops = experiment_params["baseloops"]

    edgeparserObject = EdgeParser(
        experiment_name=experiment_params["experiment_name"],
        interaction_types=experiment_params["interaction_types"],
        working_directory=experiment_params["working_directory"],
        loop_file=f"{baseloop_directory}/{baseloops}/{LOOPFILES[baseloops][tissue_params['resources']['tissue']]}",
        params=tissue_params,
    )

    edgeparserObject.parse_edges()
    print("Edges parsed!")


def parse_local_context(experiment_params, tissue_params):
    nodes = experiment_params.get("nodes", []) + NODES
    experiment_name = experiment_params["experiment_name"]
    interaction_types = experiment_params["interaction_types"]
    working_directory = experiment_params["working_directory"]

    remove_nodes = [node for node in POSSIBLE_NODES if node not in nodes]

    local_dir = f"{working_directory}/{experiment_name}/{tissue_params['resources']['tissue']}/local"
    bedfiles = _listdir_isfile_wrapper(dir=local_dir)

    adjusted_bedfiles = [
        bed for bed in bedfiles if not any(node in bed for node in remove_nodes)
    ]

    localparseObject = LocalContextParser(
        experiment_name=experiment_name,
        interaction_types=interaction_types,
        nodes=nodes,
        working_directory=working_directory,
        bedfiles=adjusted_bedfiles,
        params=tissue_params,
    )

    localparseObject.parse_context_data()
    print(f"Local context parser complete!")


def main() -> None:
    """Pipeline to generate jobs for creating graphs"""

    # Parse arguments for type of graphs to produce
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    parser.add_argument(
        "--tissue_config", type=str, help="Path to .yaml file with filenames"
    )
    args = parser.parse_args()
    experiment_params = parse_yaml(args.experiment_config)
    tissue_params = parse_yaml(args.tissue_config)
    nodes = experiment_params.get("nodes", []) + NODES
    print(f"Starting pipeline for {experiment_params['experiment_name']}!")

    preprocess_bedfiles(
        experiment_params=experiment_params,
        tissue_params=tissue_params,
        nodes=nodes,
    )
    parse_edges(experiment_params=experiment_params, tissue_params=tissue_params)
    parse_local_context(experiment_params, tissue_params)


if __name__ == "__main__":
    main()
