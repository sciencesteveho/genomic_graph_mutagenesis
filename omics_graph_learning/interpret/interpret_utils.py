#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Common utility functions for interpretability and perturbation
experiments.
"""


import pickle
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import pearsonr  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore

from omics_graph_learning.interpret.perturb_runner import PerturbRunner


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode-to-gene-symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def combine_masks(data: Data) -> Data:
    """Combine test/train/val masks into one."""
    data.all_mask = data.test_mask | data.train_mask | data.val_mask
    return data


def get_gene_idx_mapping(idxs: Dict[str, int]) -> Tuple[Dict[int, str], List[int]]:
    """Map 'ENSG...' nodes to a dict of node_idx->gene_id and a list of gene_indices."""
    gene_idxs = {k: v for k, v in idxs.items() if "ENSG" in k}
    node_idx_to_gene_id = {v: k for k, v in gene_idxs.items()}
    gene_indices = list(gene_idxs.values())
    return node_idx_to_gene_id, gene_indices


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
    """Load model, Data object, and index mappings."""
    # load gencode: symbol table
    gencode_to_symbol = load_gencode_lookup(lookup_file)

    # load PyG data
    data = torch.load(graph_file).to(device)

    # load node index dictionary
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)
    node_idx_to_gene_id, gene_indices = get_gene_idx_mapping(idxs)

    idxs_inv = {v: k for k, v in idxs.items()}  # inverse mapping

    # load model via PerturbRunner
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
    runner = PerturbRunner(model=model, device=device, data=data)

    return (
        data,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        gencode_to_symbol,
    )


def get_baseline_predictions(
    data: Data, mask: str, runner: PerturbRunner
) -> pd.DataFrame:
    """Evaluate the model on a given mask and return predictions as a
    DataFrame.
    """
    data = combine_masks(data)
    test_loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data, f"{mask}_mask"),
        shuffle=False,
    )

    (
        regression_outs,
        regression_labels,
        node_indices,
        classification_outs,
        classification_labels,
    ) = runner.evaluate(data_loader=test_loader, epoch=0, mask=mask)

    # ensure shape alignment
    regression_outs = regression_outs.squeeze()
    regression_labels = regression_labels.squeeze()
    node_indices = node_indices.squeeze()
    classification_outs = classification_outs.squeeze()
    classification_labels = classification_labels.squeeze()

    assert (
        regression_outs.shape[0]
        == regression_labels.shape[0]
        == node_indices.shape[0]
        == classification_outs.shape[0]
        == classification_labels.shape[0]
    ), "Mismatch in tensor shapes."

    return pd.DataFrame(
        {
            "node_idx": node_indices.cpu().numpy(),
            "prediction": regression_outs.cpu().numpy(),
            "label": regression_labels.cpu().numpy(),
            "class_logits": classification_outs.cpu().numpy(),
            "class_label": classification_labels.cpu().numpy(),
        }
    )


def get_best_predictions(
    df: pd.DataFrame,
    gene_indices: List[int],
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str] = None,
    output_prefix: str = "",
    sample: str = "k562",
    max_low_genes: int = 500,
    min_pearson_r: float = 0.80,
) -> Tuple[List[int], pd.DataFrame]:
    """Get predictions for genes given a certain threshold.

    1. We first compute Pearosn R for each gene across all data points
    2. We bin by mean label:
        - high if >=5
        - medium if >= 1 and < 5
        - low if > 0 and < 1
    3. We keep all the genes in high and medium, but cap the number of low
    5. Write the final gene symbols to a file for reference
    """
    if gencode_to_symbol is None:
        gencode_to_symbol = {}

    # filter for gene nodes
    df_genes = df[df["node_idx"].isin(gene_indices)].copy()

    # map node indices to gene IDs
    df_genes["gene_id"] = df_genes["node_idx"].map(node_idx_to_gene_id)

    # compute differences without absolute value
    df_genes["diff"] = df_genes["prediction"] - df_genes["label"]

    # filter genes with predicted output > prediction_threshold
    df_filtered = df_genes[df_genes["prediction"] > prediction_threshold]

    # check if there are enough genes after filtering
    if df_filtered.empty:
        print(f"No gene predictions greater than {prediction_threshold} found.")
        return []

    # select topk genes with the smallest absolute difference
    df_topk = df_filtered.reindex(
        df_filtered["diff"].abs().sort_values(ascending=True).index
    ).head(topk)

    # get topk gene IDs and their corresponding node indices
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
