#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run a series of different perturbation experiments:

1.
2.
3.

"""

import argparse
from collections import defaultdict
import pickle
import random
from typing import Any, Dict, List, Tuple

import networkx as nx  # type: ignore
from omics_graph_learning.perturbation.perturb_runner import PerturbRunner
import pandas as pd
from scipy.stats import ttest_ind  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore


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
    Data,
    PerturbRunner,
    Dict[int, str],
    List[int],
    Dict[int, str],
    Dict[str, int],
    Dict[str, str],
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
    model = PerturbRunner.load_model(
        checkpoint_file=model_file,
        map_location=device,
        model="GAT",
        activation="gelu",
        in_size=44,
        embedding_size=200,
        gnn_layers=2,
        shared_mlp_layers=2,
        heads=2,
        dropout_rate=0.3,
        residual="distinct_source",
        attention_task_head=False,
    )
    runner = PerturbRunner(
        model=model,
        device=device,
        data=data,
    )

    return (
        data,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        gencode_to_symbol,
    )


def load_coessential_pairs(
    pos_file: str, neg_file: str, idxs: Dict[str, int]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Load positive and negative coessential gene pairs and map to node indices."""
    pos_pairs = []
    neg_pairs = []

    # Load positive pairs
    with open(pos_file, "r") as f:
        for line in f:
            gene1, gene2, label = line.strip().split("\t")
            if label != "pos":
                continue
            # Map to indices
            if gene1 in idxs and gene2 in idxs:
                pos_pairs.append((idxs[gene1], idxs[gene2]))

    # Load negative pairs
    with open(neg_file, "r") as f:
        for line in f:
            gene1, gene2, label = line.strip().split("\t")
            if label != "neg":
                continue
            # Map to indices
            if gene1 in idxs and gene2 in idxs:
                neg_pairs.append((idxs[gene1], idxs[gene2]))

    return pos_pairs, neg_pairs


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
    gencode_to_symbol: Dict[str, str] = None,
    sample: str = "k562",
) -> Tuple[List[int], pd.DataFrame]:
    """compute differences and save a specific top k of best
    predictions past a TPM threshold.
    """
    if gencode_to_symbol is None:
        gencode_to_symbol = {}

    # filter for gene nodes
    gene_node_indices = set(gene_indices)
    df_genes = df[df["node_idx"].isin(gene_node_indices)].copy()

    # map node indices to gene IDs
    df_genes["gene_id"] = df_genes["node_idx"].map(node_idx_to_gene_id)

    # compute differences without absolute value
    df_genes["diff"] = df_genes["prediction"] - df_genes["label"]

    # filter genes with predicted output > prediction_threshold
    df_genes_filtered = df_genes[df_genes["prediction"] > prediction_threshold]

    # check if there are enough genes after filtering
    if df_genes_filtered.empty:
        print(f"No gene predictions greater than {prediction_threshold} found.")
        return []

    # select topk genes with the smallest absolute difference
    topk = min(topk, len(df_genes_filtered))
    df_topk = df_genes_filtered.reindex(
        df_genes_filtered["diff"].abs().sort_values(ascending=True).index
    ).head(topk)

    # get topk gene IDs and their corresponding node indices
    topk_gene_ids = df_topk["gene_id"].tolist()
    topk_node_indices = df_topk["node_idx"].tolist()

    # map node indices to gene symbols
    topk_gene_names = []
    for node_idx in topk_node_indices:
        gene_id = node_idx_to_gene_id.get(node_idx)
        if gene_id and "ENSG" in gene_id:
            if gene_symbol := gencode_to_symbol.get(gene_id.split("_")[0]):
                topk_gene_names.append(gene_symbol)
            else:
                topk_gene_names.append(gene_id)
        else:
            topk_gene_names.append(str(node_idx))

    # save topk gene names to a file
    with open(f"{output_prefix}/{sample}_top{topk}_gene_names.txt", "w") as f:
        for gene in topk_gene_names:
            f.write(f"{gene}\n")

    return topk_node_indices, df_topk


def perturb_node_features(
    data: Data,
    runner: PerturbRunner,
    feature_indices: List[int],
    mask: str,
    device: torch.device,
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str],
    output_prefix: str = "",
    sample: str = "k562",
) -> None:
    """Perform feature perturbation, compute differences, and save top affected genes."""
    feature_node_differences = defaultdict(lambda: defaultdict(list))

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

        node_indices = batch.n_id[mask_tensor].cpu()

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
            diff = regression_out_baseline - regression_out_perturbed  # preserve sign

            # accumulate differences per node
            for idx, d in zip(node_indices, diff):
                feature_node_differences[feature_index][idx.item()].append(d.item())

    # after processing all batches, compute average differences per node and per feature
    feature_node_avg_differences = {}
    feature_fold_changes = {}
    for feature_index in feature_indices:
        node_diffs = feature_node_differences[feature_index]
        # Compute average difference per node
        avg_diffs_per_node = {
            node_idx: sum(diffs) / len(diffs) for node_idx, diffs in node_diffs.items()
        }
        feature_node_avg_differences[feature_index] = avg_diffs_per_node

        # Compute average difference per feature across all nodes
        total_diff = sum(avg_diffs_per_node.values())
        num_nodes = len(avg_diffs_per_node)
        avg_diff = total_diff / num_nodes
        feature_fold_changes[feature_index] = avg_diff

        # get top 100 genes most affected by the feature ablation
        sorted_node_diffs = sorted(
            avg_diffs_per_node.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_100 = sorted_node_diffs[:100]
        top_100_gene_names = []
        for node_idx, diff_value in top_100:
            # map node_idx to gene_id
            gene_id = node_idx_to_gene_id.get(node_idx)
            if gene_id and "ENSG" in gene_id:
                gene_symbol = gencode_to_symbol.get(gene_id.split("_")[0])
                if gene_symbol:
                    top_100_gene_names.append((gene_symbol, diff_value))
                else:
                    top_100_gene_names.append((gene_id, diff_value))
            else:
                top_100_gene_names.append((str(node_idx), diff_value))

        # save top 100 gene names for this feature
        with open(
            f"{output_prefix}/{sample}_{feature_index}_ablated_top100_genes.txt", "w"
        ) as f:
            for gene, diff_value in top_100_gene_names:
                f.write(f"{gene}\t{diff_value}\n")

    # save the feature fold changes to a file
    with open(f"{output_prefix}/{sample}_feature_fold_changes.pkl", "wb") as f:
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
        num_neighbors = [data.avg_edges] * num_hops
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
        with torch.no_grad():
            regression_out_sub, __annotations__ = runner.model(
                x=sub_data.x,
                edge_index=sub_data.edge_index,
                mask=sub_data.all_mask_loss,
            )
        print(f"regression_out_sub: {regression_out_sub}")
        baseline_prediction = regression_out_sub[idx_in_subgraph].item()
        print(f"Baseline prediction for gene {gene_id}: {baseline_prediction}")

        # get nodes to perturb (excluding the gene node)
        gene_node_tensor = torch.tensor(
            [gene_node], dtype=sub_data.n_id.dtype, device=sub_data.n_id.device
        )

        nodes_to_perturb = sub_data.n_id[sub_data.n_id != gene_node_tensor]

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
        assert (
            gene_node not in selected_nodes
        ), "Gene node should not be in selected_nodes"
        print(f"Gene node: {gene_node}")
        print(f"Selected nodes: {selected_nodes}")

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
                print(f"idx to remove is {idx_to_remove}")

                # mask for nodes to keep (exclude node_to_remove)
                mask_nodes = (
                    torch.arange(sub_data.num_nodes, device=device) != idx_to_remove
                )
                nodes_to_keep = torch.arange(sub_data.num_nodes, device=device)[
                    mask_nodes
                ]

                perturbed_edge_index, _, mapping = subgraph(
                    subset=mask_nodes,  # Use 'subset' instead of 'nodes'
                    edge_index=sub_data.edge_index,
                    relabel_nodes=True,
                    num_nodes=sub_data.num_nodes,
                    return_edge_mask=True,
                )

                # Get perturbed node features and other attributes
                perturbed_x = sub_data.x[mask_nodes]
                perturbed_mask = sub_data.all_mask_loss[mask_nodes]
                perturbed_n_id = sub_data.n_id[mask_nodes]

                # Check if gene_node is still in the perturbed subgraph
                if (perturbed_n_id == gene_node).sum() == 0:
                    continue  # Skip if gene node is not in the subgraph

                # Find the new index of the gene_node after reindexing
                idx_in_perturbed = (
                    (perturbed_n_id == gene_node).nonzero(as_tuple=True)[0].item()
                )

                # perform inference on perturbed subgraph
                with torch.no_grad():
                    regression_out_perturbed, __annotations__ = runner.model(
                        x=perturbed_x,
                        edge_index=perturbed_edge_index,
                        mask=perturbed_mask,
                    )

                perturbation_prediction = regression_out_perturbed[
                    idx_in_perturbed
                ].item()

                # compute fold change
                fold_change = runner.calculate_log2_fold_change(
                    baseline_prediction=baseline_prediction,
                    perturbation_prediction=perturbation_prediction,
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
                continue  # Skip this node and continue with the next one

        # store fold changes for this gene
        gene_fold_changes[gene_id] = fold_changes

    return gene_fold_changes


# def perturb_specific_nodes(
#     data: Data,
#     runner: PerturbRunner,
#     node_indices_to_perturb: List[int],
#     device: torch.device,
#     num_runs: int = 20,
#     output_prefix: str = "",
# ) -> List[float]:
#     """Zero out the node features of specified nodes and compute average fold
#     change.
#     """
#     fold_changes = []

#     # all node mask
#     data = data.to(device)
#     data = combine_masks(data)

#     # all node loader
#     all_loader = NeighborLoader(
#         data,
#         num_neighbors=[data.avg_edges] * 2,
#         batch_size=64,
#         input_nodes=getattr(data, "all_mask_loss"),
#         shuffle=False,
#     )

#     # evaluate the model on the all_loader to get baseline predictions
#     regression_outs, regression_labels, node_indices = runner.evaluate(
#         data_loader=all_loader,
#         epoch=0,
#         mask="all",
#     )

#     # ensure tensors are one-dimensional
#     regression_outs = regression_outs.squeeze()
#     node_indices = node_indices.squeeze()

#     # create a DataFrame to keep track of node indices and predictions
#     baseline_df = pd.DataFrame(
#         {
#             "node_idx": node_indices.cpu().numpy(),
#             "prediction": regression_outs.cpu().numpy(),
#         }
#     )

#     for run in range(num_runs):
#         print(f"Perturbation experiment run {run+1}")
#         # randomly select 100 nodes from node_indices_to_perturb
#         selected_node_indices = random.sample(node_indices_to_perturb, 100)

#         # create a copy of data.x
#         x_perturbed = data.x.clone()

#         # zero out the node features of the selected nodes
#         x_perturbed[selected_node_indices] = 0

#         # create data_perturbed
#         data_perturbed = data.clone()
#         data_perturbed.x = x_perturbed

#         # ensure masks are carried over
#         data_perturbed.all_mask = data.all_mask
#         data_perturbed.all_mask_loss = data.all_mask_loss

#         # create perturbed_loader
#         perturbed_loader = NeighborLoader(
#             data_perturbed,
#             num_neighbors=[data.avg_edges] * 2,
#             batch_size=64,
#             input_nodes=getattr(data_perturbed, "all_mask"),
#             shuffle=False,
#         )

#         # evaluate the model on the perturbed data
#         regression_outs_perturbed, _, node_indices_perturbed = runner.evaluate(
#             data_loader=perturbed_loader,
#             epoch=0,
#             mask="all",
#         )

#         # create perturbed DataFrame
#         perturbed_df = pd.DataFrame(
#             {
#                 "node_idx": node_indices_perturbed.cpu().numpy(),
#                 "prediction_perturbed": regression_outs_perturbed.cpu().numpy(),
#             }
#         )

#         # merge baseline and perturbed DataFrames
#         merged_df = baseline_df.merge(perturbed_df, on="node_idx")

#         # compute difference
#         merged_df["diff"] = merged_df["prediction"] - merged_df["prediction_perturbed"]

#         # compute average fold change
#         average_fold_change = merged_df["diff"].mean()

#         fold_changes.append(average_fold_change)

#     # save the fold change results
#     with open(f"{output_prefix}fold_changes.pkl", "wb") as f:
#         pickle.dump(fold_changes, f)

#     return fold_changes


# def essential_gene_perturbation(
#     data: Data,
#     runner: PerturbRunner,
#     idxs: Dict[str, int],
#     gencode_to_symbol: Dict[str, str],
#     output_prefix: str = "",
#     num_runs: int = 20,
#     sample: str = "k562",
# ) -> List[float]:
#     """Perform perturbation experiments on essential genes."""
#     # load lethal genes
#     lethal_file = "/path/to/lethal_genes.txt"
#     with open(lethal_file, "r") as f:
#         lethal_gene_symbols = [line.strip() for line in f]

#     # map gene symbols to Gencode IDs
#     lethal_gencode = [
#         gencode_to_symbol[gene]
#         for gene in lethal_gene_symbols
#         if gene in gencode_to_symbol
#     ]

#     # append cell type suffix and get indices
#     lethal_gencode = [
#         f"{gene}_{sample}" for gene in lethal_gencode if f"{gene}_{sample}" in idxs
#     ]
#     lethal_idxs = [idxs[gene] for gene in lethal_gencode if gene in idxs]

#     return perturb_specific_nodes(
#         data=data,
#         runner=runner,
#         node_indices_to_perturb=lethal_idxs,
#         mask="all",
#         device=runner.device,
#         num_runs=num_runs,
#         output_prefix=f"{output_prefix}essential_",
#     )


# def nonessential_gene_perturbation(
#     data: Data,
#     runner: PerturbRunner,
#     idxs: Dict[str, int],
#     gencode_to_symbol: Dict[str, str],
#     output_prefix: str = "",
#     num_runs: int = 20,
#     sample: str = "k562",
# ) -> List[float]:
#     """Perform perturbation experiments on non-essential genes."""
#     # load lethal genes
#     lethal_file = "/path/to/lethal_genes.txt"
#     with open(lethal_file, "r") as f:
#         lethal_gene_symbols = [line.strip() for line in f]

#     # map gene symbols to Gencode IDs
#     lethal_gencode = [
#         gencode_to_symbol[gene]
#         for gene in lethal_gene_symbols
#         if gene in gencode_to_symbol
#     ]

#     # append cell type suffix and get indices
#     lethal_gencode = [
#         f"{gene}_{sample}" for gene in lethal_gencode if f"{gene}_{sample}" in idxs
#     ]

#     # get all gene indices
#     gene_idxs = {k: v for k, v in idxs.items() if "ENSG" in k}
#     all_gene_ids = list(gene_idxs.keys())

#     # get non-essential gene IDs and indices
#     non_essential_gene_ids = list(set(all_gene_ids) - set(lethal_gencode))
#     non_essential_idxs = [idxs[gene] for gene in non_essential_gene_ids if gene in idxs]

#     return perturb_specific_nodes(
#         data=data,
#         runner=runner,
#         node_indices_to_perturb=non_essential_idxs,
#         mask="all",
#         device=runner.device,
#         num_runs=num_runs,
#         output_prefix=f"{output_prefix}nonessential_",
#     )


# def perturb_coessential_pairs(
#     data: Data,
#     runner: PerturbRunner,
#     coessential_pairs: List[Tuple[int, int]],
#     idxs_inv: Dict[int, str],
#     mask_attr: str = "all",
#     num_hops: int = 6,
#     output_prefix: str = "",
# ) -> List[float]:
#     """Perturb coessential pairs and compare with random gene perturbations."""
#     coessential_changes = []
#     random_changes = []

#     for gene1_idx, gene2_idx in tqdm(
#         coessential_pairs, desc="Processing Coessential Pairs"
#     ):
#         gene1_id = idxs_inv.get(gene1_idx, str(gene1_idx))
#         gene2_id = idxs_inv.get(gene2_idx, str(gene2_idx))

#         # Use NeighborLoader to get a subgraph around gene1
#         num_neighbors = [13] * num_hops
#         loader = NeighborLoader(
#             data,
#             num_neighbors=num_neighbors,
#             batch_size=1,
#             input_nodes=torch.tensor([gene1_idx], dtype=torch.long),
#             shuffle=False,
#         )

#         # Get the subgraph (only one batch)
#         sub_data = next(iter(loader))
#         sub_data = sub_data.to(runner.device)

#         # Check if gene2 is in the subgraph
#         if (sub_data.n_id == gene2_idx).sum() == 0:
#             continue  # Skip if gene2 is not in subgraph

#         # Get index of gene1 in subgraph
#         idx_gene1_in_subgraph = (
#             (sub_data.n_id == gene1_idx).nonzero(as_tuple=True)[0].item()
#         )
#         idx_gene2_in_subgraph = (
#             (sub_data.n_id == gene2_idx).nonzero(as_tuple=True)[0].item()
#         )

#         # Get baseline prediction for gene1
#         regression_out_sub = runner.infer_subgraph(sub_data, mask_attr)
#         baseline_prediction = regression_out_sub[idx_gene1_in_subgraph].item()

#         # Perturbation: Delete gene2
#         result = runner.infer_perturbed_subgraph(
#             sub_data, idx_gene2_in_subgraph, mask_attr
#         )
#         if result[0] is None:
#             continue  # Skip if gene1 is not in perturbed subgraph
#         regression_out_perturbed, idx_in_perturbed = result
#         perturbation_prediction = regression_out_perturbed[idx_in_perturbed].item()

#         # Compute change in expression
#         change = baseline_prediction - perturbation_prediction
#         coessential_changes.append(change)

#         # Perturbation: Delete a random gene (not gene1 or gene2)
#         other_nodes = [
#             i
#             for i in range(sub_data.num_nodes)
#             if i not in [idx_gene1_in_subgraph, idx_gene2_in_subgraph]
#         ]
#         if not other_nodes:
#             continue  # No other nodes to perturb

#         random_node_idx = random.choice(other_nodes)
#         result = runner.infer_perturbed_subgraph(sub_data, random_node_idx, mask_attr)
#         if result[0] is None:
#             continue  # Skip if gene1 is not in perturbed subgraph
#         regression_out_random_perturbed, idx_in_random_perturbed = result
#         perturbation_prediction_random = regression_out_random_perturbed[
#             idx_in_random_perturbed
#         ].item()

#         # Compute change in expression
#         random_change = baseline_prediction - perturbation_prediction_random
#         random_changes.append(random_change)

#     # Save changes to files
#     with open(f"{output_prefix}coessential_changes.pkl", "wb") as f:
#         pickle.dump(coessential_changes, f)

#     with open(f"{output_prefix}random_changes.pkl", "wb") as f:
#         pickle.dump(random_changes, f)

#     return coessential_changes, random_changes


def main() -> None:
    """Run perturbation experiments."""
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="k562", help="Sample name")
    parser.add_argument("--run", type=int, default=1, help="Run number")
    args = parser.parse_args()

    # user specified
    sample = args.sample
    run = args.run
    if run == 1:
        seed = 42
    elif run == 2:
        seed = 84
    else:
        seed = 126
    torch.manual_seed(seed)

    if sample == "k562":
        model_file = f"/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/regulatory_k562_allcontacts-global_gat_2layers_dim_2attnheads/run_{run}/GAT_best_model.pt"
    else:
        model_file = f"/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/regulatory_{sample}_allcontacts-global_gat_2layers_200dim_2attnheads/run_{run}/GAT_best_model.pt"
    idx_file = f"/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/experiments/regulatory_only_{sample}_allcontacts_global/graphs/tpm_0.5_samples_0.1_test_8-9_val_10_rna_seq/regulatory_only_{sample}_allcontacts_global_full_graph_idxs.pkl"
    graph_file = f"/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/pyg_graphs/{sample}_pyggraph.pt"

    # other files and params
    lookup_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/crispri/gencode_to_genesymbol_lookup_table.txt"
    outpath = "/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/exp"
    mask = "all"
    topk = 100
    prediction_threshold = 5.0
    feature_indices = list(range(5, 42))
    num_hops = 6
    max_nodes_to_perturb = 100
    num_runs = 10  # number of runs for essential/non-essential perturbation

    # load data and model
    (
        data,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        gencode_to_symbol,
    ) = load_data_and_model(lookup_file, graph_file, idx_file, model_file, device)

    # evaluate the model
    df = get_baseline_predictions(
        data=data,
        mask=mask,
        runner=runner,
    )

    # save some best predictions for analysis
    # save top K gene nodes past a TPM threshold
    top_gene_nodes, topk_df = get_best_predictions(
        df=df,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gene_indices=gene_indices,
        topk=100,
        prediction_threshold=5.0,
        gencode_to_symbol=gencode_to_symbol,
        output_prefix=outpath,
        sample=sample,
    )

    # # run experiment 1: node feature perturbation
    perturb_node_features(
        data=data,
        runner=runner,
        feature_indices=feature_indices,
        mask=mask,
        device=device,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=gencode_to_symbol,
        output_prefix=outpath,
        sample=sample,
    )

    # run experiment 2: connected component perturbation
    component_fold_changes = perturb_connected_components(
        data=data,
        runner=runner,
        top_gene_nodes=top_gene_nodes,
        idxs_inv=idxs_inv,
    )

    # save the fold changes to a file
    with open(f"{outpath}/{sample}_connected_components_perturbation.pkl", "wb") as f:
        pickle.dump(component_fold_changes, f)

    # # Run Experiment 3A: Essential gene perturbation
    # essential_fold_changes = essential_gene_perturbation(
    #     data,
    #     runner,
    #     idxs,
    #     gencode_to_symbol,
    #     output_prefix=outpath,
    #     num_runs=num_runs,
    # )

    # # Run Experiment 3B: Non-essential gene perturbation
    # nonessential_fold_changes = nonessential_gene_perturbation(
    #     data,
    #     runner,
    #     idxs,
    #     gencode_to_symbol,
    #     output_prefix=outpath,
    #     num_runs=num_runs,
    # )

    # print(f"Essential fold changes: {essential_fold_changes}")
    # print(f"Non-essential fold changes: {nonessential_fold_changes}")

    # # Run Experiment 4: Coessential pair perturbation
    # pos_pairs_file = "/path/to/coessential_gencode_named_pos.txt"
    # neg_pairs_file = "/path/to/coessential_gencode_named_neg.txt"

    # pos_pairs, neg_pairs = load_coessential_pairs(pos_pairs_file, neg_pairs_file, idxs)

    # # Perturb positive coessential pairs
    # coessential_changes, random_changes = perturb_coessential_pairs(
    #     data,
    #     runner,
    #     pos_pairs,
    #     idxs_inv,
    #     mask_attr=mask,
    #     num_hops=num_hops,
    #     output_prefix=f"{output_prefix}coessential_pos_",
    # )

    # t_stat, p_value = ttest_ind(coessential_changes, random_changes, equal_var=False)
    # print("T-test results for positive coessential pairs vs random perturbations:")
    # print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # # You can repeat the same for negative coessential pairs if needed

    # # Optionally, save the statistical results
    # with open(f"{output_prefix}coessential_stats.txt", "w") as f:
    #     f.write(f"T-statistic: {t_stat}, P-value: {p_value}\n")


if __name__ == "__main__":
    main()
