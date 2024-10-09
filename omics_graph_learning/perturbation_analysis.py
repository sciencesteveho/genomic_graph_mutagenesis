#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Analyze model via perturbations of the graph data.

To assess the model, we perform a variety of graph perturbations derived from
experimental data:
    1. Aggregated CRISPRi enhancer screens released by Gschwind et al., 2024.
       For experimentally verified CRISPRi validated CRE gene links, if we see
       if the CREs in our graph overlap with the validated CREs, we can delete
       the CREs and see if the expression of the linked genes change more than a
       deletion of a random CRE in the same connected component, assuming that
       the gene is in the same connected component.
    2. Co-essential gene pairs from Wainberg et al, Nature, 2021. Assumption:
       given coessential modules, if we delete the paired coessential gene, we
       should see a larger change in expression compared to a gene picked at
       random from the same connected component.
    3. Lethal genes from Blomen at al, Nature, 2015. Assumption: if we delete a
       lethal gene, the surrounding genes should have a larger change in
       expression compared to a gene picked at random from the same connected
       component.
    4. Single cell enhancer-enhancer associations from Ziyani, Delaneau, &
       Ribeiro, Communications Biology, 2024. For a given enhancer-enhancer
       pair, we first filter for a list of pairs for which both enhancers are in
       the same connected component. We then assume that deletion of an enhancer
       in the pair will have a larger effect on the expression of the linked
       genes than a random enhancer in the same connected component.
            **These are for GM12878
"""


import argparse
import contextlib
import csv
import math
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.graph_to_pytorch import graph_to_pytorch  # type: ignore
from omics_graph_learning.perturb_graph import _device_check  # type: ignore
from omics_graph_learning.perturb_graph import (
    _load_GAT_model_for_inference,
)  # type: ignore


@torch.no_grad()
def all_inference(
    model: nn.Module, device: torch.device, data_loader: NeighborLoader
) -> Tuple[float, List[Any], List[Any], Dict[int, List[float]]]:
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description("Performing inference")

    mse, outs, labels = [], [], []
    expression: Dict[int, List[float]] = {}
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # calculate loss
        outs.extend(out[data.all_mask])
        labels.extend(data.y[data.all_mask])
        mse.append(F.mse_loss(out[data.all_mask], data.y[data.all_mask]).cpu())
        loss = torch.stack(mse)

        # get idxs
        array_index_only = data.n_id.cpu().numpy() * data.all_mask.cpu().numpy()
        array_index_only = array_index_only[array_index_only != 0]
        outdata = out[data.all_mask].cpu().detach().numpy()
        for idx, _ in enumerate(array_index_only):
            try:
                expression[array_index_only[idx]].append(outdata[idx][0])
            except KeyError:
                expression[array_index_only[idx]] = [outdata[idx][0]]
        pbar.update(1)

    pbar.close()
    return math.sqrt(float(loss.mean())), outs, labels, expression


def _pre_nest_tissue_dict(tissues: List[str]) -> Dict[str, Dict]:
    return {tissue: {} for tissue in tissues}


def _flatten_inference_array(input_list: List[List[float]]) -> List[float]:
    result_list = []

    for arr in input_list:
        if len(arr) > 1:
            mean_value = np.mean(arr)
            result_list.append(mean_value)
        else:
            result_list.append(arr[0])

    return result_list


def _get_idxs_for_coessential_pairs(
    positive_coessential_genes: str,
    negative_coessential_genes: str,
    graph_idxs: Dict[str, str],
) -> Dict[str, Dict[str, Dict[int, List[int]]]]:
    """_summary_ of function

    Create a dictionary of coessential genes for each tissue of the following
    format:
        coessential_genes: {
            'tissue_1': {
                positive: {idx1: [], idx2: []},
                negative: {idx1: [], idx2: []},
            },
            'tissue_2': {
                positive: {idx1: [], idx2: []},
                negative: {idx1: [], idx2: []},
            },
        }
    """

    def _prepopulate_dict_with_keys(
        pairs: List[Tuple[str, str]], tissue: str
    ) -> Dict[int, List[int]]:
        all_genes = [tup[0] for tup in pairs] + [tup[1] for tup in pairs]
        return {
            graph_idxs[f"{gene}_{tissue}"]: []
            for gene in all_genes
            if f"{gene}_{tissue}" in graph_idxs
        }

    def _populate_dict_with_co_pairs(
        pairs: List[Tuple[str, str]], tissue: str, pre_dict: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        for pair in pairs:
            with contextlib.suppress(KeyError):
                idx1 = graph_idxs[f"{pair[0]}_{tissue}"]
                idx2 = graph_idxs[f"{pair[1]}_{tissue}"]
                if idx2 < idx1:
                    pre_dict[idx1].append(idx2)
        return {key: value for key, value in pre_dict.items() if value}

    positive_pairs = [
        (line[0], line[1])
        for line in csv.reader(
            open(positive_coessential_genes, newline=""), delimiter="\t"
        )
    ]

    negative_pairs = [
        (line[0], line[1])
        for line in csv.reader(
            open(negative_coessential_genes, newline=""), delimiter="\t"
        )
    ]

    coessential_genes = _pre_nest_tissue_dict(TISSUES_early_testing)
    for key in coessential_genes.keys():
        coessential_genes[key] = {
            "positive": _populate_dict_with_co_pairs(
                pairs=positive_pairs,
                tissue=key,
                pre_dict=_prepopulate_dict_with_keys(pairs=positive_pairs, tissue=key),
            ),
            "negative": _populate_dict_with_co_pairs(
                pairs=negative_pairs,
                tissue=key,
                pre_dict=_prepopulate_dict_with_keys(pairs=negative_pairs, tissue=key),
            ),
        }

    return coessential_genes


def _random_gene_pairs(
    coessential_genes: Dict[str, Dict[str, Dict[int, List[int]]]],
    graph_idxs: Dict[str, str],
    model_dir: str = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/alldata_combinedloops",
) -> Dict[str, Dict[str, Dict[int, List[int]]]]:
    """_summary_ of function"""

    def _get_number_of_pairs(subdict: Dict[int, List[int]]) -> int:
        total = sum(len(value) + 1 for value in subdict.values())
        return total

    def _gencode_to_idx_list(tissue: str, genes: List[str]) -> List[int]:
        gencode_keys = {f"{gene}_{tissue}" for gene in genes}
        return [graph_idxs[key] for key in gencode_keys if key in graph_idxs]

    def _generate_random_pairs(
        tissue_genes: List[str], total_pairs: int
    ) -> List[Tuple[int, int]]:
        gencode_idxs = _gencode_to_idx_list(tissue=tissue, genes=tissue_genes)
        random_pairs = []

        for _ in range(total_pairs):
            idx1 = random.choice(gencode_idxs)
            idx2 = random.choice(gencode_idxs)
            if idx2 < idx1:
                random_pairs.append((idx1, idx2))

        return random_pairs

    # initialize the dict
    random_copairs = _pre_nest_tissue_dict(TISSUES_early_testing)

    # get number of pairs to emulate, use first tissue as reference
    first_tissue = list(coessential_genes.keys())[0]
    total_positive_pairs = _get_number_of_pairs(
        coessential_genes[first_tissue]["positive"]
    )
    total_negative_pairs = _get_number_of_pairs(
        coessential_genes[first_tissue]["negative"]
    )

    for tissue, subdict in random_copairs.items():
        subdict["positive"] = {}
        subdict["negative"] = {}
        tissue_genes = utils.filtered_genes_from_bed(
            tpm_filtered_genes=f"{model_dir}/{tissue}/tpm_filtered_genes.bed",
        )
        positive_pairs = _generate_random_pairs(tissue_genes, total_positive_pairs)
        negative_pairs = _generate_random_pairs(tissue_genes, total_negative_pairs)

        for idx1, idx2 in positive_pairs:
            if idx1 not in subdict["positive"]:
                subdict["positive"][idx1] = []
            subdict["positive"][idx1].append(idx2)

        for idx1, idx2 in negative_pairs:
            if idx1 not in subdict["negative"]:
                subdict["negative"][idx1] = []
            subdict["negative"][idx1].append(idx2)

    return random_copairs


def _average_values_across_dict(dictionary: Dict[Any, List[float]]) -> Dict[Any, float]:
    """Given a dictionary where each value is a list of floats, returns the same
    dictionary but replaces the list of floats with the average"""
    return {key: np.mean(value) for key, value in dictionary.items()}


def _calculate_difference(
    baseline_dict: Dict[int, float],
    perturbed_dict: Dict[int, float],
    key1: int,
    key2: int,
) -> float:
    baseline = baseline_dict[key1]
    perturbed = perturbed_dict[key2]
    return abs(baseline - perturbed)


def generate_unique_tuples(
    input_list: List[int], existing_tuples: List[Tuple[int, int]], num_tuples: int
) -> List[Tuple[int, int]]:
    unique_tuples = []

    while len(unique_tuples) < num_tuples:
        value1 = random.choice(input_list)
        value2 = random.choice(input_list)

        # Ensure values are not the same
        if value1 != value2:
            new_tuple = (value1, value2)

            # Check if the tuple already exists in the existing_tuples list
            if new_tuple not in existing_tuples:
                unique_tuples.append(new_tuple)

    return unique_tuples


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tissue",
        "-t",
        type=str,
    )
    args = parser.parse_args()

    # get big ugly names out the way!
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_idxs.pkl"
    checkpoint_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_256_0.0001_batch32_neighbor_full_targetnoscale_idx_expression_only/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_256_0.0001_batch32_neighbor_full_targetnoscale_idx_expression_only_mse_1.843210432337007.pt"
    positive_coessential_genes = "/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_pos.txt"
    negative_coessential_genes = "/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_neg.txt"
    savedir = "/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential"

    # parse yaml for params, used to load data
    config = "/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs/ablation_experiments/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm.yaml"
    # parser = argparse.ArgumentParser()
    params = utils.parse_yaml(config)

    # open graph
    with open(graph, "rb") as file:
        graph = pickle.load(file)

    # open idxs
    with open(graph_idxs, "rb") as file:
        graph_idxs = pickle.load(file)

    # check for device
    device, map_location = _device_check()

    # initialize model and load checkpoint weights
    model = _load_GAT_model_for_inference(
        in_size=41,
        embedding_size=256,
        num_layers=2,
        checkpoint=checkpoint_file,
        map_location=map_location,
        device=device,
    )

    working_directory = params["working_directory"]
    root_dir = f"{working_directory}/{params['experiment_name']}"

    # load data
    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type="full",
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
    )

    # set up loaders for inference
    batch_size = 32
    all_loader = NeighborLoader(
        data,
        num_neighbors=[5, 5, 5, 5, 5, 3],
        batch_size=2048,
        input_nodes=data.all_mask,
    )

    # perform inference for the three splits
    _, outs, labels, expression = all_inference(
        model=model,
        device=device,
        data_loader=all_loader,
    )

    # average values across the dictionary
    # each key is an idx
    baseline_expression = _average_values_across_dict(expression)

    # coessentiality
    # prepare IDXs for different perturbations
    coessential_genes = _get_idxs_for_coessential_pairs(
        positive_coessential_genes=positive_coessential_genes,
        negative_coessential_genes=negative_coessential_genes,
        graph_idxs=graph_idxs,
    )

    random_co_idxs = _random_gene_pairs(
        coessential_genes=coessential_genes,
        graph_idxs=graph_idxs,
    )

    batch_size = 2048
    all_positive = []
    positive = coessential_genes[args.tissue]["positive"]
    positive = [(key, val) for key, values in positive.items() for val in values]
    positive = [
        tup
        for tup in positive
        if tup[0] in baseline_expression.keys() and tup[1] in baseline_expression.keys()
    ]
    all_positive.extend(positive)

    positive = coessential_genes[args.tissue]["positive"]
    # negative = coessential_genes[key]["negative"]
    positive = [(key, val) for key, values in positive.items() for val in values]
    # negative = [(key, val) for key, values in negative.items() for val in values]
    positive = [
        tup
        for tup in positive
        if tup[0] in baseline_expression.keys() and tup[1] in baseline_expression.keys()
    ]
    positive = random.sample(positive, 250)

    # convert random to a list of tuples as well
    # random_co_testing = random_co_idxs[key]["positive"]
    # random_co_flat = [(key, val) for key, values in random_co_testing.items() for val in values]

    # Perturb graphs for coessential pairs. Save the difference in expression
    # coessential_perturbed_expression = []
    # for tup in positive:
    #     data = graph_to_pytorch(
    #         experiment_name=params["experiment_name"],
    #         graph_type='full',
    #         root_dir=root_dir,
    #         targets_types=params["training_targets"]["targets_types"],
    #         test_chrs=params["training_targets"]["test_chrs"],
    #         val_chrs=params["training_targets"]["val_chrs"],
    #         node_remove_edges=[tup[0]]
    #     )
    #     perturb_loader = NeighborLoader(
    #         data,
    #         num_neighbors=[5, 5, 5, 5, 5, 3],
    #         batch_size=batch_size,
    #         input_nodes=data.all_mask,
    #     )
    #     _, _, _, perturbed_expression = all_inference(
    #         model=model,
    #         device=device,
    #         data_loader=perturb_loader,
    #     )
    #     coessential_perturbed_expression.append(
    #         _calculate_difference(baseline_expression, perturbed_expression, tup[0], tup[1])
    #     )

    # coessential_perturbed_expression = _flatten_inference_array(coessential_perturbed_expression)
    # with open(f"{savedir}/{args.tissue}_perturbed.pkl", "wb") as file:
    #     pickle.dump(coessential_perturbed_expression, file)

    # # Perturb graphs for random pairs
    # total_comp = len(perturbed_expression)
    # random_co_idxs_for_testing = [
    #     tup for tup
    #     in random_co_flat
    #     if tup[0] in baseline_expression.keys()
    #     and tup[1] in baseline_expression.keys()
    # ]
    # random_co_idxs_for_testing = [
    #     tup for tup
    #     in random_co_idxs_for_testing
    #     if tup not in positive
    # ]
    # random_co_idxs_for_testing = random.sample(random_co_flat, total_comp)
    random_co_idxs_for_testing = generate_unique_tuples(
        input_list=list(baseline_expression.keys()),
        existing_tuples=all_positive,
        num_tuples=250,
    )
    random_perturbed_expression = []
    for key, value in random_co_idxs_for_testing:
        data = graph_to_pytorch(
            experiment_name=params["experiment_name"],
            graph_type="full",
            root_dir=root_dir,
            targets_types=params["training_targets"]["targets_types"],
            test_chrs=params["training_targets"]["test_chrs"],
            val_chrs=params["training_targets"]["val_chrs"],
            node_remove_edges=[key],
        )
        perturb_loader = NeighborLoader(
            data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=batch_size,
            input_nodes=data.all_mask,
        )
        _, _, _, perturbed_expression = all_inference(
            model=model,
            device=device,
            data_loader=perturb_loader,
        )
        random_perturbed_expression.append(
            _calculate_difference(baseline_expression, perturbed_expression, key, value)
        )

    random_perturbed_expression = _flatten_inference_array(random_perturbed_expression)
    with open(f"{savedir}/{args.tissue}_random.pkl", "wb") as file:
        pickle.dump(random_perturbed_expression, file)

    # print(f"tissue: {key}")
    # print(stats.ttest_ind(perturbed_expression, random_perturbed_expression))


if __name__ == "__main__":
    main()
