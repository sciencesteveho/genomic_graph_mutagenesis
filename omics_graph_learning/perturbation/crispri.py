#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to evaluate in-silico perturbations matching CRISPRi experiments in
K562.

ANALYSES TO PERFORM
# 1 - we expect the tuples with TRUE to have a higher magnitude of change than random perturbations
# 2 - for each tuple, we expect those with TRUE to affect prediction at a higher magnitude than FALSE
# 3 - for each tuple, we expect those with TRUE to negatively affect prediction (recall)
# 3 - for the above, compare the change that randomly chosen FALSE would postively or negatively affect the prediction
"""


import pickle
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx  # type: ignore
import numpy as np
import pybedtools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import from_networkx  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.graph_to_pytorch import graph_to_pytorch
from omics_graph_learning.perturbation.in_silico_perturbation import (
    InSilicoPerturbation,
)
from omics_graph_learning.utils.common import _add_hash_if_missing


def calculate_fold_change(
    baseline_prediction: float, perturbation_prediction: float
) -> float:
    """Convert regression values back to TPM to calculate the fold change
    difference.
    """
    baseline_tpm = 2 ** (baseline_prediction - 0.25)
    perturbation_tpm = 2 ** (perturbation_prediction - 0.25)
    return perturbation_tpm / baseline_tpm


def get_subgraph(data: Data, graph: nx.Graph, target_node: int) -> Data:
    """Extracts the entire connected component containing the given target node
    from a PyG Data object. Requires the networkX representation of the graph for
    faster traversal.

    Returns:
        Data: A new PyG Data object containing the subgraph for the connected
        component.
    """
    # find the connected component containing the target node
    for component in nx.connected_components(graph):
        if target_node in component:
            subgraph_nodes = component
            break

    # extract the subgraph from the original graph
    subgraph = graph.subgraph(subgraph_nodes)

    # convert the subgraph back to a PyG Data object
    sub_data = from_networkx(subgraph)

    # copy node features and labels from the original graph
    sub_data.x = data.x[list(subgraph_nodes)]
    sub_data.y = data.y[list(subgraph_nodes)]

    return sub_data


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode to gene symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def _rename_tuple(
    input_tuple: Tuple[str, str, str],
    idxs_dict: Dict[str, int],
    gencode_lookup: Dict[str, str],
    tissue: Optional[str] = "k562",
) -> Union[Tuple[int, int, str], None]:
    """Rename a tuple by replacing gene symbol with corresponding Gencode ID and
    index from idxs_dict.

    Returns:
        (Tuple[int, int, str]) of the form (enhancer, gencode, flag)
    """
    enhancer, gene_symbol, flag = input_tuple

    # find the gencode ID for the gene symbol
    gencode_id = gencode_lookup.get(gene_symbol)
    if gencode_id is None:
        print(f"Gene symbol {gene_symbol} not found in Gencode lookup.")
        return None

    # construct the keys for lookup in the idxs_dict
    enhancer_key = f"{enhancer}_{tissue}"
    gencode_key = f"{gencode_id}_{tissue}"

    # fetch the indices from the idxs_dict
    enhancer_idx = idxs_dict.get(enhancer_key)
    gencode_idx = idxs_dict.get(gencode_key)

    if enhancer_idx is None or gencode_idx is None:
        print(f"Indices for {enhancer_key} or {gencode_key} not found in idxs_dict.")
        return None

    # return the new tuple format
    return (enhancer_idx, gencode_idx, flag)


def rename_tuples(
    tuple_list: List[Tuple[str, str, str]],
    idxs_dict: Dict[str, int],
    gencode_lookup: Dict[str, str],
) -> List[Tuple[int, int, str]]:
    """Rename a list of tuples."""
    return [_rename_tuple(tup, idxs_dict, gencode_lookup) for tup in tuple_list]


def load_crispri(
    crispr_benchmarks: str,
    enhancer_catalogue: str,
    tissue: str = "k562",
) -> Set[Tuple[str, str, str]]:
    """Take an intersect of the regulatory catalogue with the CRISPRi
    benchmarks, replacing the CRISPRi enhancers with those in the regulatory
    catalogue that overlap.

    Returns:
        Set[Tuple[str, str, str]]: Set of tuples containing the enhancer, gene,
        and regulation bool

    """

    def reorder(feature: Any) -> Any:
        """Reorder the features."""
        chrom = feature[6]
        start = feature[7]
        end = feature[8]
        name = feature[3]
        gene = feature[4]
        regulation = feature[5]
        return pybedtools.create_interval_from_list(
            [chrom, start, end, gene, regulation, name]
        )

    # load the file as a bedtool
    # _add_hash_if_missing(crispr_benchmarks)
    links = pybedtools.BedTool(crispr_benchmarks).cut([0, 1, 2, 3, 8, 19])

    # intersect with enhancer catalogue, but only keep the enhancer, gene, and
    # regulation bool
    links = links.intersect(enhancer_catalogue, wa=True, wb=True).each(reorder).saveas()

    return {(f"{link[0]}_{link[1]}_enhancer", link[3], link[4]) for link in links}


def filter_links_for_present_nodes(
    graph: Data, links: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """Filter CRISPRi links to only include those with nodes present in the
    graph.
    """
    num_nodes = graph.num_nodes

    # convert links to tensors
    links_tensor = torch.tensor([(tup[0], tup[1]) for tup in links], dtype=torch.long)

    # check which links have both nodes in the valid range
    valid_links_mask = (links_tensor[:, 0] < num_nodes) & (
        links_tensor[:, 1] < num_nodes
    )

    # apply mask
    valid_links_tensor = links_tensor[valid_links_mask]

    # convert back to a list of tuples with the original third element
    filtered_links = []
    valid_idx = 0
    for i, link in enumerate(links):
        if valid_links_mask[i]:
            filtered_links.append(
                (
                    int(valid_links_tensor[valid_idx][0]),
                    int(valid_links_tensor[valid_idx][1]),
                    link[2],
                )
            )
            valid_idx += 1

    return filtered_links


def create_crispri_dict(
    crispri_links: List[Tuple[int, int, str]]
) -> Dict[int, List[Tuple[int, int]]]:
    """Save all crispr links in a dictionary so we only do subgraph extraction
    once per gene."""
    result = {}
    for link in crispri_links:
        guide = link[1]
        if guide not in result:
            result[guide] = []
        result[guide].append((link[0], link[2]))
    return result


def perturb_crispri(
    graph: Data,
    crispri_links: Dict[int, List[Tuple[int, int]]],
    nx_graph: nx.Graph,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Initialize three dictionaries, TRUE, FALSE and RANDOM, to store the fold changes
    for each gene in the CRISPRi benchmarks.
    Default dict because we will be appending to the lists.

    For each gene in the crispr_dict:
        Get the subgraph
        Get the baseline prediction
        For each tuple in the crispr_dict[gene]:
            Check if tup[0] is in the subgraph
            If not, continue
            If true:
                Delete node tup[0] from the subgraph
                Get the prediction
                Calculate the fold change
                If tup[1] is TRUE:
                    Append fold change in the TRUE dict with the gene as key, and (tup[0], fold_change) as value
    """
    raise NotImplementedError


def main() -> None:
    # load graph
    crispr_benchmarks = "EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    enhancer_catalogue = "enhancers_epimap_screen_overlap.bed"

    graph = torch.load("graph.pt")

    # create nx graph
    nx_graph = to_networkx(graph, to_undirected=True)

    # load IDXS
    with open(
        "../regulatory_only_k562_combinedloopcallers_full_graph_idxs.pkl", "rb"
    ) as f:
        idxs = pickle.load(f)

    # load crispr benchmarks
    crispri_links = load_crispri(crispr_benchmarks, enhancer_catalogue)
    gencode_lookup = load_gencode_lookup("gencode_to_genesymbol_lookup_table.txt")

    # rename the crispr links to node idxs
    renamed_crispri_links = rename_tuples(crispri_links, idxs, gencode_lookup)
    renamed_crispri_links = {link for link in renamed_crispri_links if link is not None}

    # filter the links for present nodes in the graph
    crispri_links = filter_links_for_present_nodes(graph, renamed_crispri_links)
    crispri_links = create_crispri_dict(crispri_links)

    # Some for testing:
    (487024, 11855, "FALSE"),
    (242639, 106, "TRUE"),
    (486837, 11855, "FALSE"),
    (665979, 4811, "FALSE"),
    (301526, 2864, "FALSE"),
    (486838, 716, "FALSE"),
    (90267, 17948, "FALSE"),
    (412026, 14109, "TRUE"),
    (320736, 10356, "TRUE"),

    # get the subgraph
    subgraph = get_subgraph(graph, nx_graph, 11855)

    # start loop for eval
    # load the model
    model = load_model(checkpoint_file, map_location, device)

    # iterate over crispri links
    for crispr_link in crispri_links:
        # get the subgraph
        # run a function given

        # get the data object
        data = Data(
            x=subgraph.x,
            edge_index=subgraph.edge_index,
            edge_attr=subgraph.edge_attr,
            y=subgraph.y,
        )

        # get the loader
        loader = NeighborLoader(data, size=100, num_workers=4)

        # get the prediction
        prediction = evaluate(model, loader, device)

        # save the prediction
        save_prediction(prediction, crispr_link, output_file)


if __name__ == "__main__":
    main()
