# sourcery skip: avoid-global-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Systematically perturb the graph by removing nodes and evaluating the model.
"""


import argparse
from collections import defaultdict
from collections import deque
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd
import pybedtools
from scipy import stats  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.loader import NeighborSampler  # type: ignore
from torch_geometric.utils import from_networkx  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.architecture_builder import build_gnn_architecture


def load_model(
    checkpoint_file: str,
    map_location: torch.device,
    device: torch.device,
) -> nn.Module:
    """Load the model from a checkpoint.

    Args:
        checkpoint_file (str): Path to the model checkpoint file.
        map_location (str): Map location for loading the model.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The loaded model.
    """
    model = build_gnn_architecture(
        model="GAT",
        activation="gelu",
        in_size=44,
        embedding_size=200,
        out_channels=1,
        gnn_layers=2,
        shared_mlp_layers=2,
        heads=4,
        dropout_rate=0.4,
        residual="distinct_source",
        attention_task_head=False,
        train_dataset=None,
    )
    model = model.to(device)

    # load the model
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def calculate_log2_fold_change(
    baseline_prediction: float, perturbation_prediction: float
) -> float:
    """Calculate the log2 fold change from log2-transformed values."""
    log2_fold_change = perturbation_prediction - baseline_prediction
    return 2**log2_fold_change - 1


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(126)

# Load graph
idx_file = "regulatory_only_k562_allcontacts_global_full_graph_idxs.pkl"

# Load the PyTorch graph
data = torch.load("graph.pt")

# Create NetworkX graph
nx_graph = to_networkx(data, to_undirected=True)

# Load indices
with open(idx_file, "rb") as f:
    idxs = pickle.load(f)

# Get dictionaries of gene indices
gene_idxs = {k: v for k, v in idxs.items() if "ENSG" in k}

# Map node indices to gene IDs
node_idx_to_gene_id = {v: k for k, v in gene_idxs.items()}
gene_indices = list(gene_idxs.values())

# Load the model
model = load_model("GAT_best_model.pt", device, device)

df_top100 = pd.read_csv("top100_gene_predictions.csv")

# Task 3: Perturb connected components (was Task 2)
# Create inverse mapping from node indices to node names
top25_node_indices = df_top100["node_idx"].tolist()
idxs_inv = {v: k for k, v in idxs.items()}

# Dictionary to store fold changes for each gene
gene_fold_changes = {}

# Do not move data to device yet
# data = data.to(device)  # Comment out or remove this line

# Adjust the number of hops
num_hops = 6  # 5-hop neighborhood

for gene_node in tqdm(top25_node_indices, desc="Processing Genes for Task 3"):
    gene_id = node_idx_to_gene_id[gene_node]

    # Use NeighborLoader to get a subgraph around the gene node
    num_neighbors = [13] * num_hops
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=1,
        input_nodes=torch.tensor([gene_node], dtype=torch.long),
        shuffle=False,
    )

    # Get the subgraph (only one batch since batch_size=1 and one input node)
    sub_data = next(iter(loader))
    sub_data = sub_data.to(device)  # Move sub_data to GPU

    # Get index of gene_node in subgraph
    idx_in_subgraph = (sub_data.n_id == gene_node).nonzero(as_tuple=True)[0].item()

    # Get baseline prediction for the gene in the subgraph
    with torch.no_grad():
        regression_out_sub, _ = model(
            x=sub_data.x,
            edge_index=sub_data.edge_index,
            mask=sub_data.test_mask_loss,
        )
    baseline_prediction = regression_out_sub[idx_in_subgraph].item()
    print(f"Baseline prediction for gene {gene_id}: {baseline_prediction}")

    # Get nodes to perturb (excluding the gene node)
    nodes_to_perturb = sub_data.n_id[sub_data.n_id != gene_node]

    num_nodes_to_perturb = len(nodes_to_perturb)
    if num_nodes_to_perturb == 0:
        print(f"No other nodes in the subgraph of gene {gene_id}. Skipping.")
        continue

    if num_nodes_to_perturb > 100:
        selected_nodes = random.sample(nodes_to_perturb.tolist(), 100)
    else:
        selected_nodes = nodes_to_perturb.tolist()

    # Create a copy of sub_data on CPU for NetworkX
    sub_data_cpu = sub_data.clone().cpu()

    # Create a NetworkX graph from sub_data_cpu to compute shortest paths
    subgraph_nx = to_networkx(sub_data_cpu, to_undirected=True)

    # Map subgraph node indices to original node indices
    mapping_nx = {i: sub_data_cpu.n_id[i].item() for i in range(len(sub_data_cpu.n_id))}
    subgraph_nx = nx.relabel_nodes(subgraph_nx, mapping_nx)

    # Compute shortest path lengths from gene_node to all other nodes in subgraph
    lengths = nx.single_source_shortest_path_length(subgraph_nx, gene_node)

    # Initialize dictionary to store fold changes
    fold_changes = {}

    for node_to_remove in selected_nodes:
        try:
            # Get local index of node_to_remove in subgraph
            idx_to_remove = (
                (sub_data.n_id == node_to_remove).nonzero(as_tuple=True)[0].item()
            )

            # Mask for nodes to keep (exclude the node to remove)
            mask_nodes = (
                torch.arange(sub_data.num_nodes, device=device) != idx_to_remove
            )
            nodes_to_keep = torch.arange(sub_data.num_nodes, device=device)[mask_nodes]

            perturbed_edge_index, _, mapping = subgraph(
                subset=mask_nodes,  # Use 'subset' instead of 'nodes'
                edge_index=sub_data.edge_index,
                relabel_nodes=True,
                num_nodes=sub_data.num_nodes,
                return_edge_mask=True,
            )

            # Get perturbed node features and other attributes
            perturbed_x = sub_data.x[mask_nodes]
            perturbed_y = sub_data.y[mask_nodes]
            perturbed_mask = sub_data.test_mask_loss[mask_nodes]
            perturbed_n_id = sub_data.n_id[mask_nodes]

            # Check if gene_node is still in the perturbed subgraph
            if (perturbed_n_id == gene_node).sum() == 0:
                continue  # Skip if gene node is not in the subgraph

            # Find the new index of the gene_node after reindexing
            idx_in_perturbed = (
                (perturbed_n_id == gene_node).nonzero(as_tuple=True)[0].item()
            )

            # Perform inference on the perturbed subgraph
            with torch.no_grad():
                regression_out_perturbed, _ = model(
                    x=perturbed_x,
                    edge_index=perturbed_edge_index,
                    mask=perturbed_mask,
                )

            perturbation_prediction = regression_out_perturbed[idx_in_perturbed].item()

            # Compute fold change
            fold_change = calculate_log2_fold_change(
                baseline_prediction, perturbation_prediction
            )

            # Get the actual name of the node being removed
            node_name = idxs_inv.get(node_to_remove, str(node_to_remove))

            # Get hop distance from gene_node to node_to_remove
            hop_distance = lengths.get(node_to_remove, -1)

            # Store fold change with additional info
            fold_changes[node_name] = {
                "fold_change": fold_change,
                "hop_distance": hop_distance,
            }

        except Exception as e:
            print(f"An error occurred while processing node {node_to_remove}: {e}")
            continue  # Skip this node and continue with the next one

    # Store fold changes for this gene
    gene_fold_changes[gene_id] = fold_changes

# Save the fold changes to a file
with open("gene_fold_changes.pkl", "wb") as f:
    pickle.dump(gene_fold_changes, f)
