#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""This script will create one graph per tissue from the parsed edges. All edges
are added before being filtered to only keep edges that can traverse back to a
base node. Attributes are then added for each node as node features and parsed
as a series of numpy arrays."""


from pathlib import Path
import pickle
from typing import Any, Dict, List, Set, Tuple, Union

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd

from omics_graph_learning.utils.common import dir_check_make
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.common import time_decorator

logger = setup_logging()


class GraphConstructor:
    """Class to handle constructing graphs from parsed edges and local
    context."""

    # hardcoded helpers, column idxs
    source_idx = 0
    target_idx = 1
    edge_attr_idx = 2
    first_node_idx = 3
    second_node_idx = 7

    def __init__(
        self,
        tissue: str,
        interaction_dir: Path,
        parse_dir: Path,
        graph_type: str,
        nodes: List[str],
        genes: List[str],
    ) -> None:
        """Initialize a GraphConstructor object."""
        self.tissue = tissue
        self.interaction_dir = interaction_dir
        self.parse_dir = parse_dir
        self.graph_type = graph_type
        self.nodes = nodes
        self.genes = genes

    @time_decorator(print_args=False)
    def construct_graph(self) -> nx.Graph:
        """Create and process the graph based on the given parameters."""
        # make base graph
        graph = self._create_base_graph()

        # keep only nodes that eventually hop to a regression target, given max
        # hops
        graph = self._prune_nodes_without_gene_connections(graph)

        # populate graph with features
        graph = self._add_node_attributes(graph)

        # remove nodes in blacklist regions
        graph = self._remove_blacklist_nodes(graph)
        logger.info("Removed nodes in blacklist regions.")

        # remove self loops
        graph = self._remove_self_loops(graph)

        # remove isolated nodes
        graph = self._remove_isolated_nodes(graph)

        self._log_gene_nodes(graph)
        return graph

    def _log_gene_nodes(self, graph: nx.Graph) -> None:
        """Log the number of gene nodes in the graph."""
        gene_nodes = set({node for node in graph.nodes if "ENSG" in node})
        logger.info(f"Gene nodes: {len(gene_nodes)}")

    def _create_base_graph(self) -> nx.Graph:
        """Create the initial graph from edge data."""
        edges = self._get_edges()
        graph = nx.from_pandas_edgelist(
            edges,
            source=self.source_idx,
            target=self.target_idx,
            edge_attr=self.edge_attr_idx,
        )
        logger.info(
            f"Base graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )
        return graph

    def _get_edges(self) -> pd.DataFrame:
        """Parse the edge data from the interaction and local context files."""
        if self.graph_type == "full":
            return pd.concat(
                [
                    self._get_edge_data(
                        self.interaction_dir / "interaction_edges.txt",
                        add_tissue=True,
                    ),
                    self._get_edge_data(
                        self.parse_dir / "edges" / "all_concat_sorted.bed",
                        local=True,
                        add_tissue=True,
                    ),
                ]
            )
        else:
            return self._get_edge_data(
                self.interaction_dir / "interaction_edges.txt", add_tissue=True
            )

    @time_decorator(print_args=False)
    def _prepare_reference_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Base_node attr are hard coded in as the first type to load. There are
        duplicate keys in preprocessing but they have the same attributes so they'll
        overwrite without issue. The hardcoded removed_keys are either chr, which we
        don't want the model or learn, or they contain sub dictionaries. Coordinates
        which are not a learning feature and positional encodings are added at a
        different step of model training.
        """
        # load basenodes
        with open(f"{self.parse_dir}/attributes/basenodes_reference.pkl", "rb") as file:
            ref = pickle.load(file)

        # add attributes for all nodes
        for node in self.nodes:
            with open(
                f"{self.parse_dir}/attributes/{node}_reference.pkl", "rb"
            ) as file:
                ref_for_concat = pickle.load(file)
                ref.update(ref_for_concat)

        return ref

    def _add_node_attributes(self, graph: nx.Graph) -> nx.Graph:
        """Add attributes to graph nodes."""
        ref = self._prepare_reference_attributes()
        nx.set_node_attributes(graph, ref)
        logger.info(
            f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )
        return graph

    def _prune_nodes_without_gene_connections(self, graph: nx.Graph) -> nx.Graph:
        """Remove nodes from an undirected graph if they are not in the same
        connected component as any node in gene_nodes.
        """
        connected_to_gene = set()
        for component in nx.connected_components(graph):
            if any(gene_node in component for gene_node in self.genes):
                connected_to_gene.update(component)

        nodes_to_remove = set(graph) - connected_to_gene
        graph.remove_nodes_from(nodes_to_remove)
        logger.info(
            f"Removed {len(nodes_to_remove)} nodes from graph that do not hop to genes."
        )
        return graph

    def _remove_isolated_nodes(self, graph: nx.Graph) -> nx.Graph:
        """Remove nodes from the graph w/o edges."""
        isolated_nodes = list(nx.isolates(graph))
        graph.remove_nodes_from(isolated_nodes)
        logger.info(f"Removed {len(isolated_nodes)} isolated nodes from the graph.")
        return graph

    @time_decorator(print_args=False)
    def _get_edge_data(
        self,
        edge_file: str,
        local: bool = False,
        add_tissue: bool = False,
    ) -> pd.DataFrame:
        """Load edge data from file and process it."""
        if local:
            df = pd.read_csv(
                edge_file,
                sep="\t",
                header=None,
                usecols=[self.first_node_idx, self.second_node_idx],
            ).rename(columns={self.first_node_idx: 0, self.second_node_idx: 1})
            df[2] = "local"
        else:
            df = pd.read_csv(edge_file, sep="\t", header=None)

        suffix = f"_{self.tissue}" if add_tissue else ""

        # avoid adding suffix multiple times
        df[0] = df[0].apply(lambda x: x if x.endswith(suffix) else f"{x}{suffix}")
        df[1] = df[1].apply(lambda x: x if x.endswith(suffix) else f"{x}{suffix}")

        return df

    @staticmethod
    def _remove_self_loops(graph: nx.Graph) -> nx.Graph:
        """Remove self loops from the graph."""
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    @staticmethod
    def _remove_blacklist_nodes(graph: nx.Graph) -> nx.Graph:
        """Remove nodes from graph if they have no attributes."""
        blacklist = [node for node, attrs in graph.nodes(data=True) if len(attrs) == 0]
        graph.remove_nodes_from(blacklist)
        logger.info(
            f"Removed {len(blacklist)} nodes from graph because they overlapped blacklist / have no attributes."
        )
        return graph


class GraphSerializer:
    """Class to handle serialization of graphs into np.arrays."""

    def __init__(
        self,
        graph: nx.Graph,
        graph_dir: Path,
        graph_type: str,
        prefix: str,
        tissue: str,
        build_positional_encoding: bool = False,
    ) -> None:
        """Initialize a GraphSerializer object."""
        self.graph = graph
        self.graph_dir = graph_dir
        self.graph_type = graph_type
        self.prefix = prefix
        self.tissue = tissue
        self.build_positional_encoding = build_positional_encoding

    def serialize(self) -> None:
        """Serialize the graph to numpy tensors and save to file."""
        # save node mapping dictionary: node name -> index
        rename = self._create_node_mapping()
        self._save_node_mapping(rename)

        # convert graph to numpy tensors
        self._nx_to_tensors(rename)

    def _create_node_mapping(self) -> Dict[str, int]:
        """Create a mapping of node names to integer indices."""
        return {node: idx for idx, node in enumerate(sorted(self.graph.nodes))}

    def _save_node_mapping(self, rename: Dict[str, int]):
        """Save the node mapping to a file."""
        with open(
            self.graph_dir
            / f"{self.prefix}_{self.graph_type}_graph_{self.tissue}_idxs.pkl",
            "wb",
        ) as output:
            pickle.dump(rename, output)

    @time_decorator(print_args=False)
    def _nx_to_tensors(self, rename: Dict[str, int]):
        """Convert the networkx graph to numpy tensors and save to file."""
        # relabel nodes to integers, according to determined mapping with
        # `rename`
        graph = nx.relabel_nodes(self.graph, mapping=rename)

        # node specific data
        coordinates, positional_encodings, node_features = self._extract_node_data(
            graph, self.build_positional_encoding
        )

        # edge specific data
        edges, edge_features = self._extract_edge_data(graph, rename)

        # save graph to file
        self._dump_graph_data(
            coordinates=coordinates,
            positional_encodings=positional_encodings,
            node_features=node_features,
            edges=edges,
            edge_features=edge_features,
            graph=graph,
        )

    def _dump_graph_data(
        self,
        coordinates: List[List[Union[str, float]]],
        positional_encodings: List[np.ndarray],
        node_features: List[Union[int, float]],
        edges: np.ndarray,
        edge_features: List[str],
        graph,
    ):
        """Save the extracted graph data to a file."""
        with open(
            self.graph_dir / f"{self.prefix}_{self.graph_type}_graph_{self.tissue}.pkl",
            "wb",
        ) as output:
            pickle.dump(
                {
                    "edge_index": edges,
                    "node_feat": np.array(node_features),
                    "node_positional_encoding": np.array(positional_encodings),
                    "node_coordinates": np.array(coordinates),
                    "edge_feat": edge_features,
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "avg_edges": graph.number_of_edges() / graph.number_of_nodes(),
                },
                output,
                protocol=4,
            )

    @staticmethod
    def _extract_node_data(
        graph: nx.Graph,
        build_positional_encoding: bool = False,
    ) -> Tuple[List[List[Union[str, float]]], List[np.ndarray], List[Any]]:
        """Extract node-specific data from the graph."""
        # prepare lists to store node data
        coordinates: List[List[Union[str, float]]] = []
        positional_encodings: List[np.ndarray] = []
        node_features: List[Any] = []

        # extraction, not featuring chris hemsworth
        for i in range(len(graph.nodes)):
            node_data = graph.nodes[i]
            coordinates.append(list(node_data["coordinates"].values()))

            if build_positional_encoding:
                positional_encodings.append(node_data["positional_encoding"].flatten())

            node_features.append(
                [
                    value
                    for key, value in node_data.items()
                    if key not in ["coordinates", "positional_encoding"]
                ]
            )
        return coordinates, positional_encodings, node_features

    @staticmethod
    def _extract_edge_data(
        graph: nx.Graph, rename: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract edge-specific data from the graph."""
        edges = np.array(
            [
                [edge[0], edge[1]]
                for edge in nx.to_edgelist(graph, nodelist=list(rename.values()))
            ]
        ).T
        edge_features = [graph[u][v][2] for u, v in edges.T]
        return edges, edge_features


def check_missing_target_genes(
    graph: nx.Graph, target_genes: List[str], tissue: str
) -> Set[str]:
    """Check for missing target genes in the graph and log the results.

    Targets are chosen via TPM filter, but they might not necessarily remain in
    the graph as we prune nodes that are self isolated. Nodes may or may not be
    isolated depending on the resolution of the 3d chromatin data used for graph
    construction. Thus, not all target genes may be present in the final graph
    due to the design - this function serves more as an information check than
    an error check.
    """
    graph_nodes = set(graph.nodes())
    gene_nodes = {node for node in graph_nodes if "ENSG" in node}

    if missing_genes := set(target_genes) - gene_nodes:
        logger.warning(
            "The following target genes are missing from the final graph: "
            f"{missing_genes}"
        )
        logger.warning(f"Number of missing genes: {len(missing_genes)}")
    else:
        logger.info("All target genes are present in the final graph.")

    # return present genes
    return gene_nodes


def construct_tissue_graph(
    nodes: List[str],
    experiment_name: str,
    working_directory: Path,
    split_name: str,
    graph_type: str,
    tissue: str,
    target_genes: List[str],
    build_positional_encoding: bool = False,
) -> None:
    """Pipeline to generate per-sample graphs."""

    # set directories
    split_dir = working_directory / "graphs" / split_name
    interaction_dir = working_directory / tissue / "interaction"
    parse_dir = working_directory / tissue / "parsing"
    dir_check_make(split_dir)

    # construct graph
    graph = GraphConstructor(
        tissue=tissue,
        interaction_dir=interaction_dir,
        parse_dir=parse_dir,
        graph_type=graph_type,
        nodes=nodes,
        genes=target_genes,
    ).construct_graph()

    # check for missing target genes
    present_genes = check_missing_target_genes(
        graph=graph, target_genes=target_genes, tissue=tissue
    )
    with open(split_dir / f"{experiment_name}_{tissue}_genes.pkl", "wb") as output:
        pickle.dump(present_genes, output)

    # serialize to arrays
    serializer = GraphSerializer(
        graph=graph,
        graph_dir=split_dir,
        graph_type=graph_type,
        prefix=experiment_name,
        tissue=tissue,
        build_positional_encoding=build_positional_encoding,
    )
    serializer.serialize()
