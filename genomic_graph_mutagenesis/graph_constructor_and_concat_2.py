#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] remove hardcoded files and add in the proper area, either the yaml or the processing fxn
#

"""Create graphs from parsed edges

This script will create one graph per tissue from the parsed edges. All edges
are added before being filtered to only keep edges that can traverse back to a
base node. Attributes are then added for each node.
"""

import csv
from itertools import repeat
from multiprocessing import Pool
import pickle
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from utils import dir_check_make
from utils import NODES
from utils import ONEHOT_EDGETYPE
from utils import time_decorator
from utils import TISSUES

CORES = len(TISSUES) + 1  # one process per tissue


@time_decorator(print_args=True)
def graph_constructor(
    tissue: str,
    root_dir: str,
) -> nx.Graph:
    """_summary_

    Args:
        tissue (str): _description_
        root_dir (str): _description_
        graph_dir (str): _description_
        interaction_dir (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        nx.Graph: _description_
    """
    # housekeeping
    interaction_dir = f"{root_dir}/{tissue}/interaction"
    parse_dir = f"{root_dir}/{tissue}/parsing"
    graph_dir = f"{root_dir}/graphs/{tissue}"
    dir_check_make(graph_dir)

    def _base_graph(edges: List[str]):
        """Create a graph from list of edges"""
        G = nx.Graph()
        for tup in edges:
            G.add_edges_from([(tup[0], tup[1], {"edge_type": ONEHOT_EDGETYPE[tup[2]]})])
        return G

    @time_decorator(print_args=True)
    def _get_edges(
        edge_file: str,
        edge_type: str,
        add_tissue: bool = False,
    ) -> List[str]:
        """Get edges from file"""
        if edge_type == "base":
            if add_tissue:
                return [
                    (f"{tup[0]}_{tissue}", f"{tup[1]}_{tissue}", tup[3])
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
            else:
                return [
                    (tup[0], tup[1], tup[3])
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
        if edge_type == "local":
            if add_tissue:
                return [
                    (f"{tup[3]}_{tissue}", f"{tup[7]}_{tissue}", "local")
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
            else:
                return [
                    (tup[3], tup[7], "local")
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
        if edge_type not in ("base", "local"):
            raise ValueError("Edge type must be 'base' or 'local'")

    @time_decorator(print_args=False)
    def _prepare_reference_attributes() -> Dict[str, Dict[str, Any]]:
        """Base_node attr are hard coded in as the first type to load. There are
        duplicate keys in preprocessing but they have the same attributes so
        they'll overrwrite without issue.

        Returns:
            Dict[str, Dict[str, Any]]: nested dict of attributes for each node
        """
        ref = pickle.load(open(f"{parse_dir}/attributes/basenodes_reference.pkl", "rb"))
        for node in NODES:
            ref_for_concat = pickle.load(
                open(f"{parse_dir}/attributes/{node}_reference.pkl", "rb")
            )
            ref.update(ref_for_concat)

        for key in ref:
            if "_tf" in key:
                ref[key]["is_gene"] = 0
                ref[key]["is_tf"] = 1
            elif "ENSG" in key:
                ref[key]["is_gene"] = 1
                ref[key]["is_tf"] = 0
            else:
                ref[key]["is_gene"] = 0
                ref[key]["is_tf"] = 0

        return ref

    # get edges
    base_edges = _get_edges(
        edge_file=f"{interaction_dir}/interaction_edges.txt",
        edge_type="base",
        add_tissue=True,
    )

    local_context_edges = _get_edges(
        edge_file=f"{parse_dir}/edges/all_concat_sorted.bed",
        edge_type="local",
        add_tissue=True,
    )

    # get attribute reference dictionary
    ref = _prepare_reference_attributes()

    # create graphs
    base_graph = _base_graph(edges=base_edges)
    graph = _base_graph(edges=base_edges)

    # add local context edges to full graph
    for tup in local_context_edges:
        graph.add_edges_from([(tup[0], tup[1], {"edge_type": ONEHOT_EDGETYPE[tup[2]]})])

    # add attributes
    for g in [base_graph, graph]:
        nx.set_node_attributes(g, ref)

    return g


@time_decorator(print_args=True)
def _nx_to_tensors(graph_dir: str, graph: nx.Graph, graph_type: str) -> None:
    """Save graphs as np tensors, additionally saves a dictionary to map
    nodes to new integer labels

    Args:
        graph (nx.Graph)
    """
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    edges = nx.to_edgelist(graph)
    nodes = sorted(graph.nodes)

    with open(f"{graph_dir}/all_tissue_{graph_type}.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": np.array(
                    [[edge[0] for edge in edges], [edge[1] for edge in edges]]
                ),
                "node_feat": np.array(
                    [[val for val in graph.nodes[node].values()] for node in nodes]
                ),
                "edge_feat": np.array([edge[2]["edge_type"] for edge in edges]),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "avg_edges": graph.number_of_edges() / graph.number_of_nodes(),
            },
            output,
            protocol=4,
        )


def main(root_dir: str, graph_type: str) -> None:
    """Pipeline to generate individual graphs"""
    graph_dir = f"{root_dir}/graphs"

    # instantiate objects and process graphs in parallel
    # pool = Pool(processes=CORES)
    # graphs = pool.starmap(graph_constructor, zip(TISSUES, repeat(root_dir)))
    # pool.close()
    # graphs = [graph_constructor(tissue=tissue, root_dir=root_dir) for tissue in TISSUES]

    # tmp save so we dont have to do this again
    # with open(f"{graph_dir}/all_tissue_{graph_type}_graphs_raw.pkl", "wb") as output:
    #     pickle.dump(graphs, output)

    with open(f"{graph_dir}/all_tissue_{graph_type}_graphs_raw.pkl", "rb") as file:
        graphs = pickle.load(file)

    # concat all
    graph = nx.compose_all(graphs)

    # save indexes before renaming to integers
    with open(f"{graph_dir}/all_tissue_{graph_type}_graph_idxs.pkl", "wb") as output:
        pickle.dump(
            {node: idx for idx, node in enumerate(sorted(graph.nodes))},
            output,
        )

    # save idxs and write to tensors
    _nx_to_tensors(graph_dir=graph_dir, graph=graph, graph_type="full")


if __name__ == "__main__":
    main(
        root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
        graph_type="full",
    )
