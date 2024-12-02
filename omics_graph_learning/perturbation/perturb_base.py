#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to evaluate in-silico perturbations matching CRISPRi experiments in
K562."""


from collections import defaultdict
import pickle
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.perturb_runner import PerturbRunner


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


def main() -> None:
    """Main function to perform in-silico perturbations matching CRISPRi experiments in K562."""
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(126)

    # load gencode to symbol lookup
    lookup_file = "../gencode_to_genesymbol_lookup_table.txt"
    gencode_to_symbol = load_gencode_lookup(lookup_file)

    # load graph
    idx_file = "regulatory_only_h1_esc_allcontacts_global_full_graph_idxs.pkl"

    # load the PyTorch graph
    data = torch.load("h1.pt")
    data = data.to(device)

    # load indices
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    # get gene indices
    node_idx_to_gene_id, gene_indices = get_gene_idx_mapping(idxs)

    # create NetworkX graph copy
    nx_graph = to_networkx(data, to_undirected=True)

    # instantiate perturb_runner
    model = PerturbRunner.load_model("GAT_best_model.pt", device, device)
    runner = PerturbRunner(
        model=model,
        device=device,
        data=data,
    )

    # combine masks for all nodes
    data = combine_masks(data)

    # create NeighborLoader for the test set data
    test_loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data, "all_mask"),
        shuffle=False,
    )

    # evaluate the model on the test set
    regression_outs, regression_labels, node_indices = runner.evaluate(
        data_loader=test_loader,
        epoch=0,
        mask="all",
    )

    # Ensure tensors are one-dimensional
    regression_outs = regression_outs.squeeze()
    regression_labels = regression_labels.squeeze()
    node_indices = node_indices.squeeze()

    # Verify that the lengths match
    assert (
        regression_outs.shape[0] == regression_labels.shape[0] == node_indices.shape[0]
    ), "Mismatch in tensor lengths."

    # Create a DataFrame to keep track of node indices, predictions, and labels
    df = pd.DataFrame(
        {
            "node_idx": node_indices.cpu().numpy(),
            "prediction": regression_outs.cpu().numpy(),
            "label": regression_labels.cpu().numpy(),
        }
    )

    # Filter for gene nodes
    gene_node_indices = set(gene_indices)
    df_genes = df[df["node_idx"].isin(gene_node_indices)].copy()

    # Map node indices to gene IDs
    df_genes["gene_id"] = df_genes["node_idx"].map(node_idx_to_gene_id)

    # Compute absolute differences
    df_genes["diff"] = (df_genes["prediction"] - df_genes["label"]).abs()

    # Filter genes with predicted output > 5
    df_genes_filtered = df_genes[df_genes["prediction"] > 5]

    # Check if there are enough genes after filtering
    if df_genes_filtered.empty:
        print("No gene predictions greater than 5 found.")

    # Select top 25 genes with the smallest difference
    topk = min(100, len(df_genes_filtered))
    df_top100 = df_genes_filtered.nsmallest(topk, "diff")

    # Get top 25 gene IDs
    top100_gene_ids = df_top100["gene_id"].tolist()

    # Save baseline predictions to a file
    baseline_predictions = dict(zip(df_genes["gene_id"], df_genes["prediction"]))
    with open("baseline_predictions_all.pkl", "wb") as f:
        pickle.dump(baseline_predictions, f)

    # Save top 25 gene IDs to a file
    with open("top100_gene_ids_all.pkl", "wb") as f:
        pickle.dump(top100_gene_ids, f)

    # save dataframe
    df_top100.to_csv("top100_gene_predictions_all.csv", index=False)

    # Task 2: Zero out node features (moved up)
    # Get baseline predictions on the test set
    feature_cumulative_differences = defaultdict(list)
    feature_indices = list(range(5, 42))

    # Use NeighborLoader to load the test data in batches
    test_loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=64,
        input_nodes=getattr(data, "all_mask"),
        shuffle=False,
    )

    # Iterate over the test batches
    for batch in tqdm(test_loader, desc="Processing Test Batches for Task 2"):
        batch = batch.to(device)
        mask = batch.all_mask_loss

        if mask.sum() == 0:
            continue

        # Get baseline predictions for this batch
        with torch.no_grad():
            regression_out_baseline, _ = model(
                x=batch.x,
                edge_index=batch.edge_index,
                mask=mask,
            )
            regression_out_baseline = regression_out_baseline[mask].cpu()

        # For each feature to perturb
        for feature_index in feature_indices:
            # Create a copy of the node features and zero out the feature at feature_index
            x_perturbed = batch.x.clone()
            x_perturbed[:, feature_index] = 0

            # Perform inference with perturbed features
            with torch.no_grad():
                regression_out_perturbed, _ = model(
                    x=x_perturbed,
                    edge_index=batch.edge_index,
                    mask=mask,
                )
                regression_out_perturbed = regression_out_perturbed[mask].cpu()

            # Compute difference between baseline and perturbed predictions for test nodes
            diff = torch.abs(regression_out_baseline - regression_out_perturbed)

            # Accumulate differences
            feature_cumulative_differences[feature_index].append(diff)

    # After processing all batches, compute average differences for each feature
    feature_fold_changes = {}
    for feature_index in feature_indices:
        diffs = torch.cat(feature_cumulative_differences[feature_index])
        avg_distance = torch.mean(diffs).item()
        feature_fold_changes[feature_index] = avg_distance

    # Save the feature fold changes to a file
    with open("feature_fold_changes_all.pkl", "wb") as f:
        pickle.dump(feature_fold_changes, f)


if __name__ == "__main__":
    main()
