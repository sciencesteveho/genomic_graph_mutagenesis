#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Class to handle the splitting of genes into training, test, and validation
sets. The class can split genes based based on given chrs (whole chromosome holdouts)
or it can assign them randomly based on percentage."""

import csv
from typing import Dict, List, Optional, Tuple

import numpy as np

from omics_graph_learning.utils.common import time_decorator
from omics_graph_learning.utils.config_handlers import ExperimentConfig


class GeneTrainTestSplitter:
    """Train test splitter for genes.

    Attributes:
        target_genes (List[str]): List of genes to split.

    Methods
    --------
    genes_train_test_val_split(
        tissues: Optional[List[str]] = None,
        test_chrs: Optional[List[int]] = None,
        val_chrs: Optional[List[int]] = None,
        tissue_append: Optional[bool] = True,
    ) -> Dict[str, List[str]]:
        Creates training, test, and validation splits for genes based on
        chromosome. Adds tissues to each label to differentiate targets between
        graphs. If no test or val chrs are provided, then 80% of genes are for
        training. The remaining 20% are split into test and validation.

    Examples:
    --------
    >>> splitter = GeneTrainTestSplitter(
        target_genes
    )

    >>> split = splitter.train_test_val_split(
        target_genes,
        experiment_config,
        tissue_append=True
    )

    Split will be a dictionary with keys "train", "test", and "validation" with
    values {gene}_{tissue}
    """

    def __init__(self, target_genes: List[str]) -> None:
        """Initialize the gene splitter."""
        self.target_genes = target_genes

    @time_decorator(print_args=False)
    def train_test_val_split(
        self,
        experiment_config: ExperimentConfig,
        tissue_append: Optional[bool] = True,
    ) -> Dict[str, List[str]]:
        """Creates training, test, and validation splits for genes based on
        chromosome. Adds tissues to each label to differentiate targets between
        graphs. If no test or val chrs are provided, then 80% of genes are for
        training. The remaining 20% are split into test and validation.

        Args:
            target_genes (List[str]): List of genes to split.
            tissues (Optional[List[str]], optional): List of tissues to append to
            gene names. Defaults to None.
            test_chrs (Optional[List[int]], optional): List of chromosome numbers
            to use for test set. Defaults to None.
            val_chrs (Optional[List[int]], optional): List of chromosome numbers
            to use for validation set. Defaults to None.
            tissue_append (Optional[bool], optional): Whether to append tissues to
            gene names. Defaults to True.
        Returns:
            Dict[str, List[str]]: Dictionary of genes split into train, test, and
            validation.
        """
        test_chrs = experiment_config.test_chrs
        val_chrs = experiment_config.val_chrs
        # tissues = experiment_config.tissues
        self._validate_chrs(test_chrs, val_chrs)

        target_genes_only = [gene.split("_")[0] for gene in self.target_genes]

        # get dict of filtered genes as {gene: chr}
        gene_chr_pairs = self._gtf_gene_chr_pairing(
            gtf=experiment_config.gencode_gtf,
            target_genes=target_genes_only,
        )
        all_genes = list(gene_chr_pairs.keys())

        # split genes based on input chromosome holdouts
        if test_chrs and val_chrs:
            train_genes, test_genes, val_genes = self._manual_chromosome_split(
                all_genes=all_genes,
                gene_chr_pairs=gene_chr_pairs,
                test_chrs=test_chrs,
                val_chrs=val_chrs,
            )
        else:
            # split genes randomly
            train_genes, test_genes, val_genes = self._random_split(all_genes)

        print(f"Number of genes in training set: {len(train_genes)}")
        print(f"Number of genes in test set: {len(test_genes)}")
        print(f"Number of genes in validation set: {len(val_genes)}")

        # append tissues to gene names for identification
        return {
            "train": list(train_genes),
            "test": list(test_genes),
            "validation": list(val_genes),
        }

    def _manual_chromosome_split(
        self,
        all_genes: List[str],
        gene_chr_pairs: Dict[str, str],
        test_chrs: List[str],
        val_chrs: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Manually split genes by assigning to lists based on chromosome
        number. Any genes not assigned to test or validation are assigned to
        training."""
        test_genes, val_genes = self._get_test_val_genes(
            gene_chr_pairs=gene_chr_pairs, test_chrs=test_chrs, val_chrs=val_chrs
        )
        train_genes = list(set(all_genes) - set(test_genes) - set(val_genes))
        return train_genes, test_genes, val_genes

    @staticmethod
    def _validate_chrs(test_chrs, val_chrs):
        """Validate that both `test_chrs` and `val_chrs` are either both `None`
        or both not `None`."""
        if bool(test_chrs) != bool(val_chrs):
            raise ValueError(
                "Both test_chrs and val_chrs must be provided together, or both must be empty."
            )

    @staticmethod
    def _get_test_val_genes(
        gene_chr_pairs: Dict[str, str],
        test_chrs: List[str],
        val_chrs: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Get test and validation genes based on chromosome number"""
        return [
            gene for gene, chr_num in gene_chr_pairs.items() if chr_num in test_chrs
        ], [gene for gene, chr_num in gene_chr_pairs.items() if chr_num in val_chrs]

    @staticmethod
    def _random_split(genes: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Randomly split genes into training, test, and validation sets. The
        default split is 80% training, 10% test, and 10% validation."""
        random_split_genes = np.split(
            np.random.permutation(genes),
            [int(0.8 * len(genes)), int(0.9 * len(genes))],
        )
        return (
            list(random_split_genes[0]),
            list(random_split_genes[1]),
            list(random_split_genes[2]),
        )

    @staticmethod
    def _gtf_gene_chr_pairing(
        gtf: str,
        target_genes: List[str],
    ) -> Dict[str, str]:
        """Get gene: chromosome dict from a gencode gtf file."""
        with open(gtf, newline="") as file:
            return {
                line[3]: line[0]
                for line in csv.reader(file, delimiter="\t")
                if line[0] not in ["chrX", "chrY", "chrM"] and line[3] in target_genes
            }

    @staticmethod
    def _append_tissues(genes: List[str], tissues: List[str]) -> List[str]:
        """Append tissue names to the gene split"""
        return [f"{gene}_{tissue}" for gene in genes for tissue in tissues]
