#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturbation of connected components."""


import argparse
import contextlib
import csv
import math
import pickle
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pybedtools
from scipy.stats import ttest_ind  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import connected_components  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.graph_to_pytorch import graph_to_pytorch
from omics_graph_learning.utils.common import _add_hash_if_missing

# from omics_graph_learning.perturbation import _device_check
# from omics_graph_learning.perturbation import _load_GAT_model_for_inference


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
    _add_hash_if_missing(crispr_benchmarks)
    links = pybedtools.BedTool(crispr_benchmarks).cut([0, 1, 2, 3, 8, 19])

    # intersect with enhancer catalogue, but only keep the enhancer, gene, and
    # regulation bool
    links = links.intersect(enhancer_catalogue, wa=True, wb=True).each(reorder).saveas()

    return {(f"{link[0]}_{link[1]}_enhancer", link[3], link[4]) for link in links}


"""
crispr_benchmarks = "EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
enhancer_catalogue = "enhancers_epimap_screen_overlap.bed"
crispri_links = load_crispri(crispr_benchmarks, enhancer_catalogue)
with open("../regulatory_only_k562_allcontacts_unimp_full_graph_idxs.pkl", "rb") as f:
    idxs = pickle.load(f)
gencode_lookup = load_gencode_lookup("gencode_to_genesymbol_lookup_table.txt")
renamed_crispri_links = rename_tuples(crispri_links, idxs, gencode_lookup)
renamed_crispri_links = {link for link in renamed_crispri_links if link is not None}
"""

# which analysis to perform?
# 1 - we expect the tuples with TRUE to have a higher magnitude of change than random perturbations
# 2 - for each tuple, we expect those with TRUE to affect prediction at a higher magnitude than FALSE
# 3 - for each tuple, we expect those with TRUE to negatively affect prediction (recall)
# 3 - for the above, compare the change that randomly chosen FALSE would postively or negatively affect the prediction


# for each tuple, we extract the connected component for tuple[0]
# we check if the connected component contains node for tuple[1]
# if it does, we perturb the connected component by removing the node for tuple[1]
# we report the change in the prediction for the node for tuple[1]
# we compare this to a random perturbation of the same size


class ModelEvaluator:
    """Class for evaluating the GNN model.

    We
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize the ModelEvaluator.

        Args:
            model (nn.Module): The trained GNN model.
            device (torch.device): The device to run the model on.
        """
        self.model = model
        self.device = device

    @torch.no_grad()
    def inference_on_component(
        self,
        data: Data,
        regression_mask: torch.Tensor,
    ) -> Dict[int, float]:
        """Perform inference on a single connected component.

        Args:
            data (Data): The graph data of the connected component.
            regression_mask (torch.Tensor): Mask indicating target nodes for
            regression.

        Returns:
            Dict[int, float]: Dictionary mapping node indices to predicted
            expression values.
        """
        self.model.eval()

        data = data.to(self.device)
        out = self.model(
            x=data.x,
            edge_index=data.edge_index,
        )

        # extract predictions for target nodes
        out_masked = out[regression_mask].squeeze().cpu()
        node_ids = data.n_id[regression_mask].cpu()

        return {
            int(node_id): float(pred) for node_id, pred in zip(node_ids, out_masked)
        }


def perform_perturbation_analysis(
    evaluator: ModelEvaluator,
    subgraph: Data,
    gene_nodes: Set[int],
    cre_nodes: Set[int],
    perturbation_type: str,
    perturbation_targets: List[int],
) -> Dict[int, float]:
    """Perform perturbation analysis on a connected component subgraph.

    Args:
        evaluator (ModelEvaluator): The model evaluator.
        subgraph (Data): The subgraph data.
        gene_nodes (Set[int]): Set of gene node indices.
        cre_nodes (Set[int]): Set of CRE node indices.
        perturbation_type (str): Type of perturbation ('gene' or 'cre').
        perturbation_targets (List[int]): List of node indices to perturb.

    Returns:
        Dict[int, float]: Dictionary mapping node indices to predicted
        expression values after perturbation.
    """
    # create a copy of the subgraph to avoid modifying the original
    perturbed_subgraph = subgraph.clone()

    # remove edges associated with the perturbation targets
    if perturbation_type in ("gene", "cre"):
        # map perturbation targets to subgraph node indices
        node_id_map = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(perturbed_subgraph.n_id.tolist())
        }
        perturbation_target_indices = [
            node_id_map.get(target)
            for target in perturbation_targets
            if target in node_id_map
        ]

        if perturbation_target_indices:
            # Identify edges to keep (edges not connected to perturbation targets)
            edge_index = perturbed_subgraph.edge_index
            mask = ~(
                torch.isin(edge_index[0], torch.tensor(perturbation_target_indices))
                | torch.isin(edge_index[1], torch.tensor(perturbation_target_indices))
            )
            perturbed_subgraph.edge_index = edge_index[:, mask]
        else:
            # No perturbation targets in this subgraph
            pass
    else:
        raise ValueError("Invalid perturbation_type. Must be 'gene' or 'cre'.")

    # Create a regression mask for the gene nodes
    regression_mask = torch.zeros(perturbed_subgraph.num_nodes, dtype=torch.bool)
    node_id_map = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(perturbed_subgraph.n_id.tolist())
    }
    for gene_node in gene_nodes:
        if gene_node in node_id_map:
            idx_in_subgraph = node_id_map[gene_node]
            regression_mask[idx_in_subgraph] = True

    # Perform inference on the perturbed subgraph
    predictions = evaluator.inference_on_component(perturbed_subgraph, regression_mask)
    return predictions


def main() -> None:
    """Main function orchestrating the workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Refactored Recapitulation Script with Perturbations"
    )
    parser.add_argument(
        "--tissue",
        "-t",
        type=str,
        required=True,
        help="Name of the tissue to process",
    )
    args = parser.parse_args()

    # Define paths (update these paths accordingly)
    graph_path = "/path/to/full_graph_scaled.pkl"
    graph_idxs_path = "/path/to/full_graph_idxs.pkl"
    checkpoint_file = "/path/to/model_checkpoint.pt"
    positive_coessential_genes = "/path/to/coessential_gencode_named_pos.txt"
    negative_coessential_genes = "/path/to/coessential_gencode_named_neg.txt"
    lethal_genes_path = "/path/to/lethal_genes.txt"
    crispri_data_path = "/path/to/crispri_data.txt"
    enhancer_pairs_path = "/path/to/enhancer_pairs.txt"
    savedir = "/path/to/save_directory"
    config = "/path/to/config.yaml"

    # Load configuration
    params = utils.parse_yaml(config)

    # Load graph and indices
    graph, graph_idxs = load_graph(graph_path, graph_idxs_path)

    # Create reverse index mapping
    reverse_graph_idxs = create_reverse_index_mapping(graph_idxs)

    # Check device
    device, map_location = _device_check()

    # Load model
    model = load_model(checkpoint_file, map_location, device)

    # Load data
    data = load_data(params)

    # Initialize evaluator
    evaluator = ModelEvaluator(model=model, device=device)

    # Load connected components and associations
    subgraphs, gene_associations, cre_associations = load_connected_components(
        data, reverse_graph_idxs
    )

    # Load perturbation data (e.g., coessential genes, lethal genes)
    # For example, load coessential gene pairs
    coessential_pairs = load_coessential_gene_pairs(
        positive_coessential_genes, graph_idxs, args.tissue
    )

    # Iterate over connected components
    for idx, (subgraph, gene_nodes, cre_nodes) in enumerate(
        zip(subgraphs, gene_associations, cre_associations)
    ):
        print(f"Processing connected component {idx+1}/{len(subgraphs)}")

        # Identify perturbation targets within this component
        component_node_ids = set(subgraph.n_id.tolist())
        perturbation_targets = component_node_ids.intersection(coessential_pairs)

        if not perturbation_targets:
            continue  # Skip if no perturbation targets in this component

        # Perform baseline inference on the unperturbed component
        regression_mask = torch.zeros(subgraph.num_nodes, dtype=torch.bool)
        node_id_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(subgraph.n_id.tolist())
        }
        for gene_node in gene_nodes:
            if gene_node in node_id_map:
                idx_in_subgraph = node_id_map[gene_node]
                regression_mask[idx_in_subgraph] = True

        baseline_predictions = evaluator.inference_on_component(
            subgraph, regression_mask
        )

        # Perform perturbation analysis
        perturbed_predictions = perform_perturbation_analysis(
            evaluator=evaluator,
            subgraph=subgraph,
            gene_nodes=gene_nodes,
            cre_nodes=cre_nodes,
            perturbation_type="gene",  # or 'cre' depending on the perturbation
            perturbation_targets=list(perturbation_targets),
        )

        # Compare baseline and perturbed predictions
        differences = {}
        for node_id in baseline_predictions:
            if node_id in perturbed_predictions:
                diff = abs(
                    baseline_predictions[node_id] - perturbed_predictions[node_id]
                )
                differences[node_id] = diff

        # Collect results for statistical analysis
        # For example, compare to random perturbations
        possible_random_targets = component_node_ids - perturbation_targets - gene_nodes
        if len(possible_random_targets) >= len(perturbation_targets):
            random_targets = random.sample(
                possible_random_targets, len(perturbation_targets)
            )
        else:
            random_targets = list(possible_random_targets)

        random_perturbed_predictions = perform_perturbation_analysis(
            evaluator=evaluator,
            subgraph=subgraph,
            gene_nodes=gene_nodes,
            cre_nodes=cre_nodes,
            perturbation_type="gene",
            perturbation_targets=random_targets,
        )

        random_differences = {}
        for node_id in baseline_predictions:
            if node_id in random_perturbed_predictions:
                diff = abs(
                    baseline_predictions[node_id]
                    - random_perturbed_predictions[node_id]
                )
                random_differences[node_id] = diff

        # Perform statistical test
        diffs = list(differences.values())
        random_diffs = list(random_differences.values())
        t_stat, p_value = ttest_ind(diffs, random_diffs, equal_var=False)

        print(f"Component {idx+1}: t-statistic={t_stat}, p-value={p_value}")

        # Save or aggregate results as needed
        # ...

    # Additional analyses for other perturbation types can be added similarly
    # ...


def load_graph(graph_path: str, graph_idxs_path: str) -> Tuple[Data, Dict[str, int]]:
    """Load the graph and its indices from pickle files.

    Args:
        graph_path (str): Path to the graph data file.
        graph_idxs_path (str): Path to the graph indices file.

    Returns:
        Tuple[Data, Dict[str, int]]: The graph data and index mapping.
    """
    with open(graph_path, "rb") as file:
        graph = pickle.load(file)

    with open(graph_idxs_path, "rb") as file:
        graph_idxs = pickle.load(file)

    return graph, graph_idxs


def load_model(
    checkpoint_file: str, map_location: str, device: torch.device
) -> nn.Module:
    """Load the GAT model from a checkpoint.

    Args:
        checkpoint_file (str): Path to the model checkpoint file.
        map_location (str): Map location for loading the model.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The loaded model.
    """
    model = _load_GAT_model_for_inference(
        in_size=41,
        embedding_size=256,
        num_layers=2,
        checkpoint=checkpoint_file,
        map_location=map_location,
        device=device,
    )
    return model


def load_data(params: Dict[str, Any]) -> Data:
    """Load the graph data using graph_to_pytorch.

    Args:
        params (Dict[str, Any]): Configuration parameters.

    Returns:
        Data: The loaded graph data.
    """
    root_dir = f"{params['working_directory']}/{params['experiment_name']}"
    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type="full",
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
    )
    return data


def load_coessential_gene_pairs(
    coessential_genes_path: str,
    graph_idxs: Dict[str, int],
    tissue: str,
) -> Set[int]:
    """Load coessential gene pairs and return a set of gene indices.

    Args:
        coessential_genes_path (str): Path to the coessential genes file.
        graph_idxs (Dict[str, int]): Mapping from gene names to node indices.
        tissue (str): The tissue type.

    Returns:
        Set[int]: Set of gene node indices that are coessential.
    """
    coessential_pairs = set()
    with open(coessential_genes_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for gene1, gene2 in reader:
            gene1_key = f"{gene1}_{tissue}"
            gene2_key = f"{gene2}_{tissue}"
            if gene1_key in graph_idxs:
                coessential_pairs.add(graph_idxs[gene1_key])
            if gene2_key in graph_idxs:
                coessential_pairs.add(graph_idxs[gene2_key])
    return coessential_pairs


if __name__ == "__main__":
    main()
