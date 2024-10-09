#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Plot contacts across a chromosome to visualize the density of the loops that
form the graph basis."""


from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import seaborn as sns  # type: ignore

from omics_graph_learning.visualization import set_matplotlib_publication_parameters

CHROM_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chrX": 156040895,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr11": 135086622,
    "chr10": 133797422,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr20": 64444167,
    "chr19": 58617616,
    "chrY": 57227415,
    "chr22": 50818468,
    "chr21": 46709983,
}


def read_bedpe_file(file: Union[str, Path]) -> pd.DataFrame:
    """Read the BEDPE file and return a df."""
    columns_to_keep = [0, 1, 2, 3, 4, 5]
    column_names = ["chr1", "start1", "end1", "chr2", "start2", "end2"]

    df = pd.read_csv(file, sep="\t", header=None)
    df = df.iloc[:, columns_to_keep]
    df.columns = column_names  # type: ignore

    return df


def calculate_contact_size(df: pd.DataFrame) -> pd.DataFrame:
    """Get the size of each chromatin contact."""
    df["size"] = df.apply(
        lambda row: max(row["end1"], row["end2"]) - min(row["start1"], row["start2"]),
        axis=1,
    )
    return df


def subsample_contacts(df: pd.DataFrame, sample_size: int = 500000) -> pd.DataFrame:
    """Subsample the number of contacts to a reasonable plotting number."""
    return df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df


def plot_contact_size_distribution(
    df: pd.DataFrame, save_path: Path, max_size: int = 2000000
) -> None:
    """Plot distribution of chromatin contact sizes."""
    set_matplotlib_publication_parameters()
    sns.histplot(data=df, x="size", bins=200, kde=True)
    plt.xlim(0, max_size)
    plt.xlabel("Contact Size (bp)")
    plt.ylabel("Count")
    plt.title("Distribution of Chromatin Contact Sizes")
    plt.tight_layout()
    plt.savefig(
        save_path / "contact_size_distribution.png", dpi=300, bbox_inches="tight"
    )


def generate_chromatin_contact_density_plot(
    file: Union[str, Path], save_path: Path
) -> None:
    """Plot the density of chromatin contacts across the genome."""
    df = read_bedpe_file(file)
    df = calculate_contact_size(df)
    df = subsample_contacts(df)
    plot_contact_size_distribution(df, save_path)
