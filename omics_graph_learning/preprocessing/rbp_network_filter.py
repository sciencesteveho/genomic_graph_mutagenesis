#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to filter active RNA binding proteins in a tissue based off a TPM
filter. The class will store the list of active RBPs as well as the filtered
network (in edges) to be used in the edge_parser module

The RBP network is downloaded from POSTAR 3 and is a list of the
    RBP --> Gene
    Where the binding site for the RBP intersects the gene body.
    
The TPM values are from ENCODE RNA-seq data. If they pass a TPM threshold, we
consider them active, and keep the RBP --> Gene edge.
"""

import csv
from typing import Dict, List, Tuple

import pandas as pd
from pybedtools import BedTool  # type: ignore


class RBPNetworkFilter:
    """Class to handle filtering of RNA binding proteins based on TPM values."""

    def __init__(
        self,
        rbp_proteins: str,
        gencode: str,
        network_file: str,
        tpm_filter: int,
        rna_seq_file: str,
    ) -> None:
        """Initialize the RBPNetworkFilter class."""
        self.rbp_proteins = rbp_proteins
        self.gencode = gencode
        self.network_file = network_file
        self.tpm_filter = tpm_filter
        self.rna_seq_file = rna_seq_file

    def filter_rbp_network(self) -> None:
        """_summary_"""
        # set up gene lists
        genes = self.genes_from_gencode(BedTool(self.gencode))
        rbp_genes = [
            line[0] for line in csv.reader(open(self.rbp_proteins), delimiter="\t")
        ]

        # map rbps and network to genesymbols
        common_genes = set(genes) & set(rbp_genes)
        rbp_genesymbol = {gene: genes[gene] for gene in common_genes}
        rbp_network = self._network_with_genesymbols(rbp_genesymbol=rbp_genesymbol)

        # filter rbp network based on tpm
        rna_seq_data = self._read_encode_rna_seq_data()
        rbp_df = rna_seq_data[rna_seq_data.index.isin(rbp_genesymbol.values())]
        filtered_rbp = rbp_df[rbp_df["TPM"] >= self.tpm_filter]

        # final TPM filtered network and attributes
        self.active_rbps = filtered_rbp.index.tolist()
        self.filtered_network = [
            edge for edge in rbp_network if edge[0] in self.active_rbps
        ]

    def _read_encode_rna_seq_data(
        self,
    ) -> pd.DataFrame:
        """Read an ENCODE rna-seq tsv, keep only ENSG genes"""
        df = pd.read_table(self.rna_seq_file, index_col=0, header=[0])
        return df[df.index.str.contains("ENSG")]

    def _network_with_genesymbols(
        self, rbp_genesymbol: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """Convert rbp -> target edges to rbp_genesymbol -> target_genesymbol edges"""
        reader = csv.reader(open(self.network_file), delimiter="\t")
        return [
            (rbp_genesymbol[line[0]], line[1])
            for line in reader
            if line[0] in rbp_genesymbol
        ]

    @staticmethod
    def genes_from_gencode(gencode_ref: BedTool) -> Dict[str, str]:
        """Returns a dict of gencode v26 genes, their ids and associated gene
        symbols
        """
        return {
            line[9].split(";")[3].split('"')[1]: line[3]
            for line in gencode_ref
            if line[0] not in ["chrX", "chrY", "chrM"]
        }


# RBP_PROTEINS = "/ocean/projects/bio210019p/stevesho/data/data_preparse/rbp_proteins.txt"
# GENCODE = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/shared_data/references/gencode_v26_genes_only_with_GTEx_targets.bed"
# NETWORK = "/ocean/projects/bio210019p/stevesho/data/data_preparse/rbp_network.txt"
# tpm_test = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/shared_data/targets/tpm/ENCFF019KLP.tsv"
# TPM_FILTER = 10
