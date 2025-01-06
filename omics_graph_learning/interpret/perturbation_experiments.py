#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run graph perturbation experiments."""


import argparse

from scipy.stats import ttest_ind  # type: ignore
import torch

from omics_graph_learning.interpret.coessential_pair_perturbation import (
    load_coessential_pairs,
)
from omics_graph_learning.interpret.coessential_pair_perturbation import (
    paired_gene_perturbation,
)
from omics_graph_learning.interpret.connected_component_perturbation import (
    connected_component_perturbation,
)
from omics_graph_learning.interpret.essential_gene_perturbation import (
    essential_gene_perturbation,
)
from omics_graph_learning.interpret.interpret_utils import get_baseline_predictions
from omics_graph_learning.interpret.interpret_utils import get_best_predictions
from omics_graph_learning.interpret.interpret_utils import load_data_and_model
from omics_graph_learning.interpret.node_feat_perturbation import perturb_node_features
from omics_graph_learning.utils.constants import RANDOM_SEEDS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="k562", help="Sample name")
    parser.add_argument("--run", type=int, default=1, help="Run number")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        help="Which experiment(s) to run: node_features, connected_components, essential, nonessential, coessential, all.",
    )
    parser.add_argument(
        "--lethal_file",
        type=str,
        default="/lethal_genes.txt",
        help="Path to lethal_genes.txt for essential gene perturbation.",
    )
    parser.add_argument(
        "--pos_pairs_file",
        type=str,
        default="/coessential_pos.txt",
        help="Path to positive coessential pairs file.",
    )
    parser.add_argument(
        "--neg_pairs_file",
        type=str,
        default="/coessential_neg.txt",
        help="Path to negative coessential pairs file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run graph perturbation experiments."""
    args = parse_args()
    sample = args.sample
    run = args.run
    experiment = args.experiment.lower()
    lethal_file = args.lethal_file
    pos_pairs_file = args.pos_pairs_file
    neg_pairs_file = args.neg_pairs_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = RANDOM_SEEDS[run]
    torch.manual_seed(seed)

    # Example file paths (adjust to your environment)
    if sample == "k562":
        model_file = (
            f"/models/regulatory_k562_allcontacts-global_gat_2layers_dim_2attnheads/"
            f"run_{run}/GAT_best_model.pt"
        )
    else:
        model_file = (
            f"/models/regulatory_{sample}_allcontacts-global_gat_2layers_200dim_2attnheads/"
            f"run_{run}/GAT_best_model.pt"
        )

    idx_file = f"/{sample}_graph_idxs.pkl"
    graph_file = f"/{sample}_pyggraph.pt"
    lookup_file = "/gencode_to_genesymbol_lookup_table.txt"

    outpath = "/output/"
    mask = "all"

    # load data
    (
        data,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        gencode_to_symbol,
    ) = load_data_and_model(lookup_file, graph_file, idx_file, model_file, device)

    # get baseline predictions
    df = get_baseline_predictions(data, mask, runner)

    # get best predictions from model
    top_gene_nodes, df_topk = get_best_predictions(
        df=df,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gene_indices=gene_indices,
        topk=100,
        prediction_threshold=5.0,
        output_prefix=outpath,
        gencode_to_symbol=gencode_to_symbol,
        sample=sample,
    )

    # run experiments
    print("Running Node Feature Perturbation...")
    perturb_node_features(
        data=data,
        runner=runner,
        feature_indices=list(range(5, 42)),
        mask=mask,
        device=device,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=gencode_to_symbol,
    )

    print("Running Connected Component Perturbation...")
    connected_component_perturbation(
        data=data,
        device=device,
        runner=runner,
        top_gene_nodes=top_gene_nodes,
        idxs_inv=idxs_inv,
        num_hops=6,
        max_nodes_to_perturb=100,
        mask_attr=mask,
    )

    print("Running Essential Gene Perturbation...")
    essential_fold_changes = essential_gene_perturbation(
        data=data,
        runner=runner,
        idxs=idxs,
        gencode_to_symbol=gencode_to_symbol,
        output_prefix=outpath,
        sample=sample,
        lethal_file=lethal_file,
        mask=mask,
        device=device,
        essential=True,
    )

    print("Running Non-Essential Gene Perturbation...")
    nonessential_fold_changes = essential_gene_perturbation(
        data=data,
        runner=runner,
        idxs=idxs,
        gencode_to_symbol=gencode_to_symbol,
        output_prefix=outpath,
        sample=sample,
        lethal_file=lethal_file,
        mask=mask,
        device=device,
        essential=False,
    )

    print("Running Coessential Pair Perturbation...")
    pos_pairs, neg_pairs = load_coessential_pairs(pos_pairs_file, neg_pairs_file, idxs)
    coessential_changes, random_changes = paired_gene_perturbation(
        data=data,
        runner=runner,
        pairs=pos_pairs,  # or neg_pairs
        num_hops=6,
        device=device,
        random_comparison=True,
    )

    # simple t-test for coessential vs random perturbations
    t_stat, p_value = ttest_ind(coessential_changes, random_changes, equal_var=False)
    print("T-test results for positive coessential pairs vs random perturbations:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    print("All experiments complete.")


if __name__ == "__main__":
    main()
