#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to handle filtering of genes by TPM values."""

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from pybedtools import BedTool  # type: ignore

from omics_graph_learning.utils.common import time_decorator
from omics_graph_learning.utils.config_handlers import TissueConfig


def read_encode_rna_seq_data(
    rna_seq_file: str,
) -> pd.DataFrame:
    """Read an ENCODE rna-seq tsv, keep only ENSG genes"""
    df = pd.read_table(rna_seq_file, index_col=0, header=[0])
    return df[df.index.str.contains("ENSG")]


class TPMFilter:
    """Data preprocessor for dealing with differences in bed files.

    Attributes:
        tissue_config (TissueConfig): Configuration object for tissue data.
        split_path (Path): Path to save split data.
        percent_of_samples_filter (float): Percentage of samples filter.
        tpm_filter (Union[float, int]): TPM filter value.

    Methods
    --------
    filter_genes(tissue: str) -> List[str]:
        Filter genes based on the samples within that tissue
    filtered_genes_from_encode_rna_data(tissues: List[str]) -> None:
        Filter genes based on rna-seq instead of gtex matrices

    Examples:
    --------
    >>> TPMFilterObj = TPMFilter(
        tissue_config=tissue_config,
        split_path=split_dir,
        tpm_filter=args.tpm_filter,
        percent_of_samples_filter=args.percent_of_samples_filter,
    )

    # Filter genes "within" only the tissue
    # Specify the TPM file to be the tissue-specific tpm file
    >>> TPMFilterObj.filter_genes(tissue=tissue, tpm_file=tpm_file)

    # Filter genes "across" all samples
    # Specify the TPM file to be the entire gtex expression matrix
    >>> TPMFilterObj.filter_genes(
        tissue=tissue,
        tpm_file=gtex_tpm_file,
        filter_mode="across"
    )

    # Filter genes based on encode rna-seq data
    # Does not require instantiation of the TPMFilter class
    >>> TPMFilter.filtered_genes_from_encode_rna_data(
        rna_seq_file=rna_seq_file
    )
    """

    def __init__(
        self,
        tissue_config: TissueConfig,
        split_path: Path,
        percent_of_samples_filter: float,
        tpm_filter: Union[float, int],
        local_dir: Path,
        filter_mode: str = "within",
    ):
        """Initialize the data preprocessor."""
        self.tissue_config = tissue_config
        self.split_path = split_path
        self.tpm_filter = tpm_filter
        self.percent_of_samples_filter = percent_of_samples_filter
        self.local_dir = local_dir
        self.filter_mode = filter_mode
        self.rna_file = tissue_config.resources["rna"]

    @time_decorator(print_args=True)
    def filter_genes(self, tissue: str, tpm_file: str) -> List[str]:
        """Filter genes for all tissues based on TPM values.

        Args:
            config_dir (Path): Directory containing tissue configuration files.
            gencode_gtf (str): Path to the Gencode GTF file.
            percent_of_samples_filter (float): Percentage of samples filter.
            split_path (Path): Path to save filtered gene files.
            tissues (List[str]): List of tissue n
            ames.
            tpm_filter (Union[float, int]): TPM filter value.
        """
        genes = self.split_path / f"{tissue}_tpm_filtered_genes.bed"

        if not genes.exists():
            gencode_filtered = self._filter_genes_by_tpm(
                gencode=self.local_dir / self.tissue_config.local["gencode"],
                tpm_file=tpm_file,
                tpm_filter=self.tpm_filter,
                percent_of_samples_filter=self.percent_of_samples_filter,
            )
            gencode_filtered.saveas(genes)
        else:
            gencode_filtered = BedTool(str(genes))
            self.filtered_genes = [x[3] for x in gencode_filtered]
        return self.filtered_genes

    def _filter_genes_by_tpm(
        self,
        gencode: Path,
        tpm_file: str,
        tpm_filter: Union[float, int],
        percent_of_samples_filter: float,
        autosomes_only: bool = True,
    ) -> BedTool:
        """
        Filter out genes in a GTEx tissue with less than (tpm) tpm across (%) of
        samples in the given dataframe. Additionally, we exclude analysis of sex
        chromosomes.

        Returns:
            pybedtools object with +/- <window> windows around that gene
        """
        gencode_bed = BedTool(gencode)
        df = self._load_gtex_tpm_df(tpm_file=tpm_file, filter_mode=self.filter_mode)

        # Get list of filtered genes
        self.filtered_genes = self._filter_gtex_dataframe_by_tpm(
            df=df,
            tpm_filter=tpm_filter,
            percent_of_samples_filter=percent_of_samples_filter,
        )

        # Use list to filter the gencode gtf
        if autosomes_only:
            gencode_filtered = self._filter_bedtool_by_autosomes(
                self._filter_bedtool_by_genes(gencode_bed, self.filtered_genes)
            )
        else:
            gencode_filtered = self._filter_bedtool_by_genes(
                gencode_bed, self.filtered_genes
            )

        return gencode_filtered.sort()

    @staticmethod
    def filtered_genes_from_encode_rna_data(
        gencode_bed: Path,
    ) -> List[str]:
        """Filter rna_seq data by TPM"""
        gencode = BedTool(gencode_bed)
        return [
            feature[3]
            for feature in gencode
            if "protein_coding" in str(feature)
            and feature[0] not in ["chrX", "chrY", "chrM"]
        ]

        # rna_seq_file: str,
        # tpm_filter: Union[float, int],
        # df = read_encode_rna_seq_data(rna_seq_file)
        # filtered_df = df[df["TPM"] >= tpm_filter]
        # return list(filtered_df.index)
        # temp change - use all protein coding genes instead of TPM filter
        # def _get_protein_coding_genes(self) -> List[str]:
        # """Retrieve protein coding genes from the gencode gtf file."""
        # gencode_bed = BedTool(self.local_dir / self.tissue_config.local["gencode"])

    @staticmethod
    def _filter_bedtool_by_genes(
        bed: BedTool,
        genes: List[str],
    ) -> BedTool:
        """Filter a pybedtools object by a list of genes."""
        filter_term = lambda x: x[3] in genes
        return bed.filter(filter_term).saveas()

    @staticmethod
    def _filter_bedtool_by_autosomes(
        bed: BedTool,
    ) -> BedTool:
        """Filter a pybedtools object by autosomes."""
        filter_term = lambda x: x[0] not in ["chrX", "chrY", "chrM"]
        return bed.filter(filter_term).saveas()

    @staticmethod
    def _filter_gtex_dataframe_by_tpm(
        df: pd.DataFrame,
        tpm_filter: Union[float, int],
        percent_of_samples_filter: float,
    ) -> List[str]:
        """Filter out genes in a GTEx dataframe."""
        sample_n = len(df.columns)
        sufficient_tpm = df.select_dtypes("number").ge(tpm_filter).sum(axis=1)
        return list(
            df.loc[sufficient_tpm >= (percent_of_samples_filter * sample_n)].index
        )

    @staticmethod
    def _load_gtex_tpm_df(
        tpm_file: str,
        filter_mode: str,
    ) -> pd.DataFrame:
        """Load the GTEx TPM dataframe."""
        if filter_mode == "across":
            return pd.read_table(tpm_file, index_col=0, header=[2])
        elif filter_mode == "within":
            df = pd.read_table(tpm_file, index_col=1, header=[2])
            return df.drop("id", axis=1)
        else:
            raise ValueError("Invalid filter schema. Must be 'across' or 'within'.")
