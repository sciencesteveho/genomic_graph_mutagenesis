#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Test the accuracy of features aggregated per node by the
LocalContextParser, which form the node features of the graphs."""


# load node
# make a dictionary to links to noderef files, which are be accessed with a path variable
# check that chr, start, and end are correct
# pick 100 random nodes. Using just that single node, make a pybedtool file and manually perform the intersect and check if the derived values are the same.


import os
from pathlib import Path
import pickle
import random

from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.config_handlers import TissueConfig
from omics_graph_learning.local_context_parser import LocalContextParser
import pytest


# TODO replace with mock config files mock_config.yaml
@pytest.fixture
def configs():
    experiment_config = ExperimentConfig.from_yaml("/path/to/experiment_config.yaml")
    tissue_config = TissueConfig.from_yaml("/path/to/tissue_config.yaml")
    return experiment_config, tissue_config


@pytest.fixture
def bedfiles(configs):
    experiment_config, tissue_config = configs
    return [
        f"{node_type}_{tissue_config.resources['tissue']}.bed"
        for node_type in experiment_config.nodes
    ]


@pytest.fixture
def local_context_parser(configs, bedfiles):
    experiment_config, tissue_config = configs
    return LocalContextParser(experiment_config, tissue_config, bedfiles)


def load_attribute_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip().split("\t") for line in f]


def test_save_node_attributes_real_data(local_context_parser):
    # Process all node types
    for node_type in local_context_parser.nodes + ["basenodes"]:
        # Run the method to ensure the reference file is created
        local_context_parser._save_node_attributes(node_type)

        # Check if the output file exists
        output_file = local_context_parser.attribute_dir / f"{node_type}_reference.pkl"
        assert os.path.exists(
            output_file
        ), f"Reference file for {node_type} does not exist"

        # Load the pickled data
        with open(output_file, "rb") as f:
            stored_attributes = pickle.load(f)

        # Load original attribute files for comparison
        attribute_files = {}
        for attribute in LocalContextParser.NODE_FEATS:
            file_path = (
                local_context_parser.attribute_dir
                / attribute
                / f"{node_type}_{attribute}_percentage"
            )
            if os.path.exists(file_path):
                attribute_files[attribute] = load_attribute_file(file_path)

        # Check a random selection of 100 nodes (or all if less than 100)
        node_names = list(stored_attributes.keys())
        selected_nodes = random.sample(node_names, min(100, len(node_names)))

        for node in selected_nodes:
            print(f"Checking node: {node}")
            assert (
                node.startswith(f"{node_type}_") or node_type == "basenodes"
            ), f"Incorrect node type prefix for {node}"
            assert node.endswith(
                f"_{local_context_parser.tissue}"
            ), f"Node {node} doesn't end with correct tissue suffix"

            node_data = stored_attributes[node]

            # Check if all required keys are present
            assert "coordinates" in node_data, f"Missing 'coordinates' in {node}"
            assert "size" in node_data, f"Missing 'size' in {node}"
            assert "gc" in node_data, f"Missing 'gc' in {node}"

            # Check coordinate structure
            assert (
                "chr" in node_data["coordinates"]
            ), f"Missing 'chr' in coordinates of {node}"
            assert (
                "start" in node_data["coordinates"]
            ), f"Missing 'start' in coordinates of {node}"
            assert (
                "end" in node_data["coordinates"]
            ), f"Missing 'end' in coordinates of {node}"

            # Check data types
            assert isinstance(
                node_data["coordinates"]["chr"], str
            ), f"Incorrect type for chr in {node}"
            assert isinstance(
                node_data["coordinates"]["start"], float
            ), f"Incorrect type for start in {node}"
            assert isinstance(
                node_data["coordinates"]["end"], float
            ), f"Incorrect type for end in {node}"
            assert isinstance(
                node_data["size"], float
            ), f"Incorrect type for size in {node}"
            assert isinstance(
                node_data["gc"], float
            ), f"Incorrect type for gc in {node}"

            # Check value ranges
            assert 0 <= node_data["gc"] <= 1, f"GC content out of range for {node}"
            assert node_data["size"] > 0, f"Invalid size for {node}"
            assert (
                node_data["coordinates"]["start"] < node_data["coordinates"]["end"]
            ), f"Invalid coordinate range for {node}"

            # Check other attributes
            for attr in LocalContextParser.NODE_FEATS:
                if attr not in ["start", "end", "size"]:
                    assert attr in node_data, f"Missing attribute {attr} in {node}"
                    assert isinstance(
                        node_data[attr], (int, float)
                    ), f"Incorrect type for {attr} in {node}"

            # Compare with original attribute files
            node_id = node.rsplit("_", 1)[0]  # Remove tissue suffix
            for attr, attr_data in attribute_files.items():
                original_value = next(
                    (float(line[-1]) for line in attr_data if line[3] == node_id), None
                )
                if original_value is not None:
                    if attr == "gc":
                        assert (
                            abs(node_data[attr] - original_value) < 1e-6
                        ), f"Mismatch in {attr} for {node}"
                    else:
                        assert (
                            node_data[attr] == original_value
                        ), f"Mismatch in {attr} for {node}"

        print(f"All checks passed for {node_type}")
