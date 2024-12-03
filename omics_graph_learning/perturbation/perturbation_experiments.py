#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run a series of different perturbation experiments:

1.
2.
3.

"""


from collections import defaultdict
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx  # type: ignore
import pandas as pd
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.perturbation.perturb_runner import PerturbRunner


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode to gene symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def combine_masks(data: Data) -> Data:
    """Combine masks for all nodes."""
    data.all_mask = data.test_mask | data.train_mask | data.val_mask
    return data


def get_gene_idx_mapping(idxs: Dict[str, int]) -> Tuple[Dict[int, str], List[int]]:
    """Map node indices to gene IDs."""
    gene_idxs = {k: v for k, v in idxs.items() if "ENSG" in k}
    return {v: k for k, v in gene_idxs.items()}, list(gene_idxs.values())


def load_data_and_model(
    lookup_file: str,
    graph_file: str,
    idx_file: str,
    model_file: str,
    device: torch.device,
) -> Tuple[
    Data, PerturbRunner, Dict[int, str], List[int], Dict[int, str], Dict[str, int]
]:
    """Load data and model for perturbation experiments."""
    # load gencode to symbol lookup, pytorch graph, and indices
    gencode_to_symbol = load_gencode_lookup(lookup_file)
    data = torch.load(graph_file)
    data = data.to(device)
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    # get gene indices
    node_idx_to_gene_id, gene_indices = get_gene_idx_mapping(idxs)

    # inverse mapping from node indices to node names
    idxs_inv = {v: k for k, v in idxs.items()}

    # instantiate
    model = PerturbRunner.load_model(model_file, device, device)
    runner = PerturbRunner(
        model=model,
        device=device,
        data=data,
    )

    return data, runner, node_idx_to_gene_id, gene_indices, idxs_inv, idxs


def get_baseline_predictions(
    data: Data,
    mask: str,
    runner: PerturbRunner,
) -> pd.DataFrame:
    """Evaluate the model on the test set and return predictions DataFrame."""
    # combine masks for all nodes
    data = combine_masks(data)

    # create NeighborLoader for the test set data
    test_loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data, f"{mask}_mask"),
        shuffle=False,
    )

    # evaluate the model on the test set
    regression_outs, regression_labels, node_indices = runner.evaluate(
        data_loader=test_loader,
        epoch=0,
        mask=mask,
    )

    # ensure tensors are one-dimensional
    regression_outs = regression_outs.squeeze()
    regression_labels = regression_labels.squeeze()
    node_indices = node_indices.squeeze()

    # verify that the lengths match
    assert (
        regression_outs.shape[0] == regression_labels.shape[0] == node_indices.shape[0]
    ), "Mismatch in tensor lengths."

    return pd.DataFrame(
        {
            "node_idx": node_indices.cpu().numpy(),
            "prediction": regression_outs.cpu().numpy(),
            "label": regression_labels.cpu().numpy(),
        }
    )


def get_best_predictions(
    df: pd.DataFrame,
    node_idx_to_gene_id: Dict[int, str],
    gene_indices: List[int],
    topk: int = 100,
    prediction_threshold: float = 5.0,
    output_prefix: str = "",
) -> List[int]:
    """Compute absolute differences and save a specific top k of best
    predictions past a TPM threshold.
    """
    # filter for gene nodes
    gene_node_indices = set(gene_indices)
    df_genes = df[df["node_idx"].isin(gene_node_indices)].copy()

    # map node indices to gene IDs
    df_genes["gene_id"] = df_genes["node_idx"].map(node_idx_to_gene_id)

    # compute absolute differences
    df_genes["diff"] = (df_genes["prediction"] - df_genes["label"]).abs()

    # filter genes with predicted output > prediction_threshold
    df_genes_filtered = df_genes[df_genes["prediction"] > prediction_threshold]

    # check if there are enough genes after filtering
    if df_genes_filtered.empty:
        print(f"No gene predictions greater than {prediction_threshold} found.")
        return []

    # select topk genes with the smallest difference
    topk = min(topk, len(df_genes_filtered))
    df_topk = df_genes_filtered.nsmallest(topk, "diff")

    # get topk gene IDs
    topk_gene_ids = df_topk["gene_id"].tolist()
    topk_node_indices = df_topk["node_idx"].tolist()

    # save baseline predictions to a file
    baseline_predictions = dict(zip(df_genes["gene_id"], df_genes["prediction"]))
    with open(f"{output_prefix}baseline_predictions.pkl", "wb") as f:
        pickle.dump(baseline_predictions, f)

    # save topk gene IDs to a file
    with open(f"{output_prefix}top{topk}_gene_ids.pkl", "wb") as f:
        pickle.dump(topk_gene_ids, f)

    # save dataframe
    df_topk.to_csv(f"{output_prefix}top{topk}_gene_predictions.csv", index=False)

    return topk_node_indices


def perturb_graph_features(
    data: Data,
    runner: PerturbRunner,
    feature_indices: List[int],
    mask: str,
    device: torch.device,
    output_prefix: str = "",
) -> None:
    """Perform absolute feature perturbation and compute differences."""
    feature_cumulative_differences = defaultdict(list)

    # use NeighborLoader to load the test data in batches
    test_loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data, f"{mask}_mask"),
        shuffle=False,
    )

    # iterate over the test batches
    for batch in tqdm(
        test_loader, desc="Processing Test Batches for Feature Perturbation"
    ):
        batch = batch.to(device)
        mask_tensor = getattr(batch, f"{mask}_mask_loss")

        if mask_tensor.sum() == 0:
            continue

        # get baseline predictions for this batch
        with torch.no_grad():
            regression_out_baseline, _ = runner.model(
                x=batch.x,
                edge_index=batch.edge_index,
                mask=mask_tensor,
            )
            regression_out_baseline = regression_out_baseline[mask_tensor].cpu()

        # for each feature to perturb
        for feature_index in feature_indices:
            # create a copy of the node features and zero out the feature at feature_index
            x_perturbed = batch.x.clone()
            x_perturbed[:, feature_index] = 0

            # perform inference with perturbed features
            with torch.no_grad():
                regression_out_perturbed, _ = runner.model(
                    x=x_perturbed,
                    edge_index=batch.edge_index,
                    mask=mask_tensor,
                )
                regression_out_perturbed = regression_out_perturbed[mask_tensor].cpu()

            # compute difference between baseline and perturbed predictions for test nodes
            diff = torch.abs(regression_out_baseline - regression_out_perturbed)

            # accumulate differences
            feature_cumulative_differences[feature_index].append(diff)

    # after processing all batches, compute average differences for each feature
    feature_fold_changes = {}
    for feature_index in feature_indices:
        diffs = torch.cat(feature_cumulative_differences[feature_index])
        avg_distance = torch.mean(diffs).item()
        feature_fold_changes[feature_index] = avg_distance

    # save the feature fold changes to a file
    with open(f"{output_prefix}feature_fold_changes.pkl", "wb") as f:
        pickle.dump(feature_fold_changes, f)


def perturb_connected_components(
    data: Data,
    runner: PerturbRunner,
    top_gene_nodes: List[int],
    idxs_inv: Dict[int, str],
    num_hops: int = 6,
    max_nodes_to_perturb: int = 100,
    mask_attr: str = "all",
) -> Dict[str, Dict[str, Any]]:
    """Perturb connected components node-by-node and compute fold changes."""
    gene_fold_changes = {}

    for gene_node in tqdm(top_gene_nodes, desc="Processing Genes for Perturbation"):
        gene_id = idxs_inv.get(gene_node, str(gene_node))

        # use NeighborLoader to get a subgraph around the gene node
        num_neighbors = [13] * num_hops
        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=1,
            input_nodes=torch.tensor([gene_node], dtype=torch.long),
            shuffle=False,
        )

        # get the subgraph (only one batch since batch_size=1 and one input node)
        sub_data = next(iter(loader))
        sub_data = sub_data.to(runner.device)

        # get index of gene_node in subgraph
        idx_in_subgraph = (sub_data.n_id == gene_node).nonzero(as_tuple=True)[0].item()

        # get baseline prediction for the gene in the subgraph
        regression_out_sub = runner.infer_subgraph(sub_data, mask_attr)
        baseline_prediction = regression_out_sub[idx_in_subgraph].item()
        print(f"Baseline prediction for gene {gene_id}: {baseline_prediction}")

        # get nodes to perturb (excluding the gene node)
        nodes_to_perturb = sub_data.n_id[sub_data.n_id != gene_node]

        num_nodes_to_perturb = len(nodes_to_perturb)
        if num_nodes_to_perturb == 0:
            print(f"No other nodes in the subgraph of gene {gene_id}. Skipping.")
            continue

        if num_nodes_to_perturb > max_nodes_to_perturb:
            selected_nodes = random.sample(
                nodes_to_perturb.tolist(), max_nodes_to_perturb
            )
        else:
            selected_nodes = nodes_to_perturb.tolist()

        # create a copy of sub_data on CPU for NetworkX
        sub_data_cpu = sub_data.clone().cpu()

        # create a NetworkX graph from sub_data_cpu to compute shortest paths
        subgraph_nx = to_networkx(sub_data_cpu, to_undirected=True)

        # map subgraph node indices to original node indices
        mapping_nx = {
            i: sub_data_cpu.n_id[i].item() for i in range(len(sub_data_cpu.n_id))
        }
        subgraph_nx = nx.relabel_nodes(subgraph_nx, mapping_nx)

        # compute shortest path lengths from gene_node to all other nodes in subgraph
        lengths = nx.single_source_shortest_path_length(subgraph_nx, gene_node)

        # initialize dictionary to store fold changes
        fold_changes = {}

        for node_to_remove in selected_nodes:
            try:
                # get local index of node_to_remove in subgraph
                idx_to_remove = (
                    (sub_data.n_id == node_to_remove).nonzero(as_tuple=True)[0].item()
                )

                # perform inference on perturbed subgraph
                result = runner.infer_perturbed_subgraph(
                    sub_data, idx_to_remove, mask_attr
                )

                if result is None:
                    continue  # skip if gene node is not in the subgraph
                regression_out_perturbed, idx_in_perturbed = result

                perturbation_prediction = regression_out_perturbed[
                    idx_in_perturbed
                ].item()

                # compute fold change
                fold_change = runner.calculate_log2_fold_change(
                    baseline_prediction, perturbation_prediction
                )

                # get the actual name of the node being removed
                node_name = idxs_inv.get(node_to_remove, str(node_to_remove))

                # get hop distance from gene_node to node_to_remove
                hop_distance = lengths.get(node_to_remove, -1)

                # store fold change with additional info
                fold_changes[node_name] = {
                    "fold_change": fold_change,
                    "hop_distance": hop_distance,
                }

            except Exception as e:
                print(f"An error occurred while processing node {node_to_remove}: {e}")
                continue  # skip this node and continue with the next one

        # store fold changes for this gene
        gene_fold_changes[gene_id] = fold_changes

    return gene_fold_changes


def main() -> None:
    """Run perturbation experiments."""
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(126)

    # specify files and parameters
    lookup_file = "../gencode_to_genesymbol_lookup_table.txt"
    graph_file = "h1.pt"
    idx_file = "regulatory_only_h1_esc_allcontacts_global_full_graph_idxs.pkl"
    model_file = "GAT_best_model.pt"
    mask = "all"
    output_prefix = "all_"
    topk = 100
    prediction_threshold = 5.0
    feature_indices = list(range(5, 42))
    num_hops = 6
    max_nodes_to_perturb = 100

    # load data and model
    data, runner, node_idx_to_gene_id, gene_indices, idxs_inv, idxs = (
        load_data_and_model(lookup_file, graph_file, idx_file, model_file, device)
    )

    # evaluate the model
    df = get_baseline_predictions(data, runner, mask, device)

    # process predictions and save results
    top_gene_nodes = get_best_predictions(
        df, node_idx_to_gene_id, gene_indices, topk, prediction_threshold, output_prefix
    )

    if not top_gene_nodes:
        print("No top genes found. Exiting.")
        return

    # run perturbation Experiment One: perturb features
    perturb_graph_features(data, runner, feature_indices, mask, device, output_prefix)

    # run perturbation Experiment Two: perturb connected components
    gene_fold_changes = perturb_connected_components(
        data,
        runner,
        top_gene_nodes,
        idxs,
        idxs_inv,
        num_hops=num_hops,
        max_nodes_to_perturb=max_nodes_to_perturb,
        mask_attr=mask,
    )

    # save the fold changes to a file
    with open(f"{output_prefix}gene_fold_changes.pkl", "wb") as f:
        pickle.dump(gene_fold_changes, f)


if __name__ == "__main__":
    main()
