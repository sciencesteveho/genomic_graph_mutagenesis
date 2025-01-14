#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to filter active RNA binding proteins in a tissue based off a TPM
filter. The class will store the list of active RBPs as well as the filtered
network (in edges) to be used in the edge_parser module.

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
    """Class to handle filtering of RNA binding proteins based on TPM values.

    First, we load the rbp network, to keep a list of rbp GENCODE
    ids. For each rbp, we go through the expression TSV and filter them based
    on the attribute tpm_filter. We keep those passing as a list of "active"
    rbps for the sample.

    We then load the RBP --> gene network derived from POSTAR3. First, we check
    that a reference rbp exists as one of the edges. If so, we check to see if
    the rbp is in the list of active rbps. If it is, we keep the edge,
    otherwise we discard it.

    Arguments:
        gencode: gencode v26 bed file
        network_file: rbp gene network file
        rna_seq_file: ENCODE RNA-seq TPM file
        rbp_proteins: mirbase derived reference
        tpm_filter: TPM threshold to filter RBPs
    """

    def __init__(
        self,
        network_file: str,
        rna_seq_file: str,
        tpm_filter: int = 2,
    ) -> None:
        """Initialize the RBPNetworkFilter class."""
        self.network_file = network_file
        self.tpm_filter = tpm_filter
        self.rna_seq_file = rna_seq_file

        # load reference rbps
        self.ref_rbp = {
            line[0] for line in csv.reader(open(network_file), delimiter="\t")
        }
        print(f"Number of reference RBPs: {len(self.ref_rbp)}")

    def filter_rbp_network(self) -> None:
        """Get RBP network."""

        # read rna-seq data and filter based on TPM
        rna_exp = self._read_encode_rna_seq_data(self.rna_seq_file)
        rbp_df = rna_exp[rna_exp.index.isin(self.ref_rbp)]
        active_rbp = rbp_df[rbp_df["TPM"] >= self.tpm_filter]
        print(f"Number of active RBPs: {len(active_rbp)}")

        # filter network based on active rbps
        rbp_network = [
            [line[0], line[1]]
            for line in csv.reader(open(self.network_file), delimiter="\t")
        ]
        print(f"Number of edges in network: {len(rbp_network)}")

        # final TPM filtered network and attributes
        self.filtered_network = [edge for edge in rbp_network if edge[0] in active_rbp]
        print(f"Number of edges in filtered network: {len(self.filtered_network)}")

    def _read_encode_rna_seq_data(
        self,
        rna_seq_file: str,
    ) -> pd.DataFrame:
        """Read an ENCODE rna-seq tsv, keep only ENSG genes"""
        df = pd.read_table(rna_seq_file, index_col=0, header=[0])
        return df[df.index.str.contains("ENSG")]
