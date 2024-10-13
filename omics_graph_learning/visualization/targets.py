#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Plot the distribution of regression targets."""


import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

from omics_graph_learning.utils.common import _load_pickle
from omics_graph_learning.visualization import set_matplotlib_publication_parameters


def flatten_targets(targets: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten the targets into a single array."""
    return np.array([targets[target] for target in targets]).flatten()


def get_target_values(
    targets: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the targets into individual lists."""
    return (
        flatten_targets(targets["train"]),
        flatten_targets(targets["validation"]),
        flatten_targets(targets["test"]),
    )


def get_targets(
    target_file: Union[str, Path], present_genes: Union[str, Path]
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """Load the targets and return a filtered version based on present genes."""
    targets = _load_pickle(target_file)
    genes = _load_pickle(present_genes)
    filtered_targets = {
        key: {
            target: targets[key][target] for target in targets[key] if target in genes
        }
        for key in targets
    }
    return targets, filtered_targets


def get_gene_list_from_targets(targets: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
    """Get a list of genes from the targets."""
    genes = (
        list(targets["train"].keys())
        + list(targets["validation"].keys())
        + list(targets["test"].keys())
    )
    return [gene.split("_")[0] for gene in genes]


def gene_chr_pairs(
    gtf: str,
) -> Dict[str, str]:
    """Get gene: chromosome dict from a gencode gtf file."""
    with open(gtf, newline="") as file:
        return {
            line[3]: line[0]
            for line in csv.reader(file, delimiter="\t")
            if line[0] not in ["chrX", "chrY", "chrM"]
        }


def get_gene_chr_pairs(
    gene_to_chr: Dict[str, str], target_genes: List[str]
) -> pd.DataFrame:
    """Given a list of genes, get a DataFrame of gene: chromosome pairs."""
    return pd.DataFrame(
        data={"gene": target_genes, "chr": [gene_to_chr[gene] for gene in target_genes]}
    )


def plot_target_distribution(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    save_path: Path,
    filtered: bool = False,
) -> None:
    """Plot distribution of targets."""
    set_matplotlib_publication_parameters()
    title = (
        "Distribution of Unfiltered Regression Targets"
        if filtered
        else "Distribution of Regression Targets"
    )
    save_file = (
        "unfiltered_regression_target_distribution.png"
        if filtered
        else "regression_target_distribution.png"
    )
    sns.kdeplot(data=[train, validation, test], common_norm=False, fill=True)
    plt.xlabel("TPM (log transformed)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(labels=["Train", "Validation", "Test"])
    plt.tight_layout()
    plt.savefig(save_path / save_file, dpi=450, bbox_inches="tight")


def combine_gene_distributions(
    genes: pd.DataFrame,
    genes_filtered: pd.DataFrame,
) -> pd.DataFrame:
    """Get the count of genes per chr and combine into a single df."""
    genes_count = (
        genes["chr"].value_counts().sort_index().reset_index(name="Unfiltered")
    )
    genes_filtered_count = (
        genes_filtered["chr"].value_counts().sort_index().reset_index(name="Filtered")
    )
    plot_data = pd.merge(
        genes_count, genes_filtered_count, on="index", how="outer"
    ).fillna(0)
    plot_data = plot_data.melt(id_vars=["index"], var_name="Filter", value_name="Count")
    return plot_data


def gene_barplots_per_chromosome(
    genes: pd.DataFrame, genes_filtered: pd.DataFrame, save_path: Path
) -> None:
    """Barplot of potential targets, stratified by chromosome. Two bars, the
    first showing unfiltered genes, the second showing the genes after filtering
    during graph construction.
    """
    set_matplotlib_publication_parameters()

    # merge data to plot together
    plot_data = combine_gene_distributions(genes, genes_filtered)
    plot_data["index"] = plot_data["index"].str.replace("chr", "")

    # plot!
    ax = sns.barplot(
        data=plot_data,
        x="index",
        y="Count",
        hue="Filter",
        palette=["blue", "red"],
        alpha=0.7,
    )

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["Before construction", "After graph construction"],
        title="",
        frameon=True,
    )

    plt.xlabel("Chromosome")
    plt.ylabel("Number of Targets")
    plt.title("Number of Targets per Chromosome")
    plt.tight_layout()
    plt.savefig(
        save_path / "regression_target_counts.png", dpi=450, bbox_inches="tight"
    )


def generate_all_target_plots(
    target_file: Union[str, Path],
    gene_file: Union[str, Path],
    gencode_gtf: str,
    save_path: Path,
) -> None:
    """Plot the density of chromatin contacts across the genome."""
    targets, filtered_targets = get_targets(target_file, gene_file)
    train, validation, test = get_target_values(targets)
    gene_to_chr = gene_chr_pairs(gencode_gtf)

    # plot possible targets
    genes = get_gene_chr_pairs(gene_to_chr, get_gene_list_from_targets(targets))
    genes_filtered = get_gene_chr_pairs(
        gene_to_chr, get_gene_list_from_targets(filtered_targets)
    )
    gene_barplots_per_chromosome(genes, genes_filtered, save_path)

    # plot target values
    train_filtered, validation_filtered, test_filtered = get_target_values(
        filtered_targets
    )
    plot_target_distribution(train, validation, test, save_path)
    plot_target_distribution(
        train_filtered, validation_filtered, test_filtered, save_path
    )
