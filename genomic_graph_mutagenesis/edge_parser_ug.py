#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import argparse
import csv
import itertools
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pybedtools

from utils import (
    genes_from_gencode,
    parse_yaml,
    time_decorator,
)


class EdgeParser:
    """Object to construct tensor based graphs from parsed bedfiles

    The baseline graph structure is build from the following in order:
        Curated protein-protein interactions from the integrated interactions
        database V 2021-05
        TF-gene circuits from Marbach et al.
        TF-gene interactions from TFMarker. We keep "TF" and "I Marker" type relationships
        Enhancer-gene networks from FENRIR
        Enhancer-enhancer networks from FENRIR

        Alternative polyadenylation targets from APAatlas

    Args:
        params: configuration vals from yaml

    Methods
    ----------
    _genes_from_gencode:
        Lorem
    _base_graph:
        Lorem
    _iid_ppi:
        Lorem
    _mirna_targets:
        Lorem
    _tf_markers:
        Lorem
    _marchbach_regulatory_circuits:
        Lorem
    _enhancer_index:
        Lorem
    _format_enhancer:
        Lorem
    _process_graph_edges:
        Lorem
    """

    PROTEIN_TISSUES = [
        "amygdala",
        "bone",
        "bone marrow",
        "brain",
        "heart",
        "hypothalamus",
        "kidney",
        "liver",
        "lung",
        "lymph nodes",
        "mammary gland",
        "ovary",
        "pancreas",
        "pituitary gland",
        "placenta",
        "prostate",
        "salivary gland",
        "skeletal muscle",
        "small intestine",
        "spleen",
        "stomach",
        "testes",
        "uterus",
        "synovial macrophages",
        "chondrocytes",
        "synovial membrane",
        "articular cartilage",
    ]

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
    ):
        """Initialize the class"""
        self.tissue_name = "universalgenome"

        self.gencode = params["shared"]["gencode"]
        self.interaction_files = params["interaction"]
        self.tissue = params["resources"]["tissue"]
        self.tissue_specific = params["tissue_specific"]

        self.root_dir = params["dirs"]["root_dir"]
        self.circuit_dir = params["dirs"]["circuit_dir"]
        self.shared_dir = f"{self.root_dir}/shared_data"
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
        self.parse_dir = f"{self.tissue_dir}/parsing"
        self.interaction_dir = f"{self.tissue_dir}/interaction"
        self.shared_interaction_dir = f"{self.shared_dir}/interaction"

        self.gencode_ref = pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        self.genesymbol_to_gencode = genes_from_gencode(gencode_ref=self.gencode_ref)
        self.gencode_attr_ref = self._blind_read_file(
            f"{self.tissue_dir}/local/gencode_v26_node_attr.bed"
        )
        self.e_indexes = self._enhancer_index(
            e_index=f"{self.shared_interaction_dir}/enhancer_indexes.txt",
            e_index_unlifted=f"{self.shared_interaction_dir}/enhancer_indexes_unlifted.txt",
        )

    def _blind_read_file(self, file: str) -> List[str]:
        """Blindly reads a file into csv reader and stores file as a list of
        lists
        """
        return [line for line in csv.reader(open(file, newline=""), delimiter="\t")]

    @time_decorator(print_args=True)
    def _iid_ppi(
        self,
        interaction_file: str,
        tissue: str,
    ) -> List[Tuple[str, str, float, str]]:
        """Protein-protein interactions from the Integrated Interactions
        Database v 2021-05"""
        df = pd.read_csv(interaction_file, delimiter="\t")
        df = df[["symbol1", "symbol2", "evidence_type", "n_methods", tissue]]
        t_spec_filtered = df[
            (df[tissue] > 0)
            & (df["n_methods"] >= 4)
            & (df["evidence_type"].str.contains("exp"))
        ]
        edges = list(
            zip(
                *map(t_spec_filtered.get, ["symbol1", "symbol2"]),
                itertools.repeat(-1),
                itertools.repeat("ppi"),
            )
        )
        return [
            (
                f"{self.genesymbol_to_gencode[edge[0]]}",
                f"{self.genesymbol_to_gencode[edge[1]]}",
                edge[2],
                edge[3],
            )
            for edge in edges
            if edge[0] in self.genesymbol_to_gencode.keys()
            and edge[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _mirna_targets(
        self,
        target_list: str,
    ) -> List[Tuple[str, str]]:
        """Filters all miRNA -> target interactions from miRTarBase and only
        keeps the miRNAs that are active in the given tissue from mirDIP.
        """
        return [
            (
                line[0],
                self.genesymbol_to_gencode[line[1]],
                -1,
                "mirna",
            )
            for line in csv.reader(open(target_list, newline=""), delimiter="\t")
            if line[1] in self.genesymbol_to_gencode.keys()
        ]
        
    @time_decorator(print_args=True)
    def _tf_markers(self, interaction_file: str) -> List[Tuple[str, str]]:
        tf_keep = ["TF", "I Marker", "TFMarker"]
        tf_markers = []
        with open(interaction_file, newline="") as file:
            file_reader = csv.reader(file, delimiter="\t")
            next(file_reader)
            for line in file_reader:
                if line[2] in tf_keep:
                    try:
                        if ";" in line[10]:
                            genes = line[10].split(";")
                            for gene in genes:
                                if line[2] == 'I Marker':
                                    tf_markers.append((gene, line[1]))
                                else:
                                    tf_markers.append((line[1], gene))
                        else:
                            if line[2] == 'I Marker':
                                tf_markers.append((line[10], line[1]))
                            else:
                                tf_markers.append((line[1], line[10]))
                    except IndexError: 
                        pass 

        return [
            (
                f'{self.genesymbol_to_gencode[tup[0]]}_tf',
                self.genesymbol_to_gencode[tup[1]],
                -1,
                "tf_marker",
            )
            for tup in tf_markers
            if tup[0] in self.genesymbol_to_gencode.keys()
            and tup[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _marbach_regulatory_circuits(
        self,
        interaction_file: str,
        score_filter: int,
        ) -> List[Tuple[str, str, float, str]]:
        """Regulatory circuits from Marbach et al., Nature Methods, 2016. Each
        network is in the following format:
            col_1   TF
            col_2   Target gene
            col_3   Edge weight 
        """
        tf_g, scores = [], []
        with open(interaction_file, newline = '') as file:
            file_reader = csv.reader(file, delimiter='\t')
            for line in file_reader:
                scores.append(float(line[2]))
                if line[0] in self.genesymbol_to_gencode.keys() and line[1] in self.genesymbol_to_gencode.keys():
                    tf_g.append((line[0], line[1], float(line[2])))
        
        cutoff = np.percentile(scores, score_filter)

        return [
            (f'{self.genesymbol_to_gencode[line[0]]}_tf',
            self.genesymbol_to_gencode[line[1]],
            line[2],
            'circuits',)
            for line in tf_g
            if line[2] >= cutoff
            ]
        
    def _load_tss(self) -> pybedtools.BedTool:
        """Load TSS file and ignore any TSS that do not have a gene target.

        Returns:
            pybedtools.BedTool - TSS w/ target genes
        """
        tss = pybedtools.BedTool(f"{self.tss}")
        return tss.filter(lambda x: x[3].split("_")[3] != "").saveas()

    @time_decorator(print_args=True)
    def get_loop_edges(
        self,
        chromatin_loops: str,
        feat_1: str,
        feat_2: str,
        edge_type: str,
        tss: bool = False,
    ) -> List[Tuple[str, str, float, str]]:
        """Connects nodes if they are linked by chromatin loops. Can specify if
        the loops should only be done for direct overlaps or if they should
        be within 2mb of a loop anchor for TSS. If using TSS, make sure to
        specify the TSS as the second feature!

        Args:
            feat_1 (str): _description_
            feat_2 (str): _description_
            type (str): _description_

        Returns:
            List[Tuple[str, str, float, str]]: _description_
        """

        def _split_chromatin_loops(
            chromatin_loops: str,
        ) -> Tuple[pybedtools.BedTool, pybedtools.BedTool]:
            """_summary_
            Returns:
                Tuple[pybedtools.BedTool, pybedtools.BedTool]: _description_
            """
            first_anchor = pybedtools.BedTool(chromatin_loops)
            second_anchor = pybedtools.BedTool(
                "\n".join(
                    [
                        "\t".join([x[3], x[4], x[5], x[0], x[1], x[2]])
                        for x in first_anchor
                    ]
                ),
                from_string=True,
            )
            return first_anchor.cut([0, 1, 2, 3, 4, 5]), second_anchor.cut(
                [0, 1, 2, 3, 4, 5]
            )

        def _loop_direct_overlap(
            loops: pybedtools.BedTool, features: pybedtools.BedTool
        ) -> pybedtools.BedTool:
            """Get features that directly overlap with loop anchor"""
            return loops.intersect(features, wo=True)

        def _loop_within_distance(
            loops: pybedtools.BedTool,
            features: pybedtools.BedTool,
            distance: int,
        ) -> pybedtools.BedTool:
            """Get features 2kb within loop anchor

            Args:
                loops (pybedtools.BedTool): _description_
                features (pybedtools.BedTool): _description_
                distance (int): _description_
            """
            return loops.window(features, w=distance)

        def _flatten_anchors(*beds: pybedtools.BedTool) -> Dict[str, List[str]]:
            """Creates a dict to store each anchor and its overlaps. Adds the feature by
            ignoring the first 7 columns of the bed file and adding whatever is left."""
            anchor = {}
            for bed in beds:
                for feature in bed:
                    anchor.setdefault("_".join(feature[0:3]), []).append(
                        "_".join([feature[6], feature[7], feature[9]])
                    )
            return anchor

        def _loop_edges(
            loops: pybedtools.BedTool,
            first_anchor_edges: Dict[str, List[str]],
            second_anchor_edges: Dict[str, List[str]],
        ) -> List[Any]:
            """Return a list of edges that are connected by their overlap over chromatin
            loop anchors by matching the anchor names across dicts"""
            edges = []
            for loop in loops:
                first_anchor = "_".join(loop[0:3])
                second_anchor = "_".join(loop[3:6])
                try:
                    uniq_edges = list(
                        itertools.product(
                            first_anchor_edges[first_anchor],
                            second_anchor_edges[second_anchor],
                        )
                    )
                    edges.extend(uniq_edges)
                except KeyError:
                    continue
            return edges

        def _check_tss_gene_in_gencode(tss: str) -> bool:
            """Simple check to see if gene has gencode ID"""
            gene = tss.split("_")[5]
            if gene in self.genesymbol_to_gencode.keys():
                return self.genesymbol_to_gencode[gene]
            else:
                return False

        # split loops into anchors
        first_anchor, second_anchor = _split_chromatin_loops(chromatin_loops)

        if tss:
            first_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(first_anchor, feat_1),
                _loop_within_distance(first_anchor, feat_2, 2000),
            )
            second_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(second_anchor, feat_1),
                _loop_within_distance(second_anchor, feat_2, 2000),
            )
            return_edges = []
            for edge in list(
                set(_loop_edges(first_anchor, first_anchor_edges, second_anchor_edges))
            ):
                if "tss" in edge[0] and "tss" in edge[1]:
                    if _check_tss_gene_in_gencode(
                        edge[0]
                    ) and _check_tss_gene_in_gencode(edge[1]):
                        return_edges.append(
                            (
                                _check_tss_gene_in_gencode(edge[0]),
                                _check_tss_gene_in_gencode(edge[1]),
                                -1,
                                "g_g",
                            )
                        )
                    else:
                        pass
                elif "tss" in edge[0] and "tss" not in edge[1]:
                    if _check_tss_gene_in_gencode(edge[0]):
                        return_edges.append(
                            (
                                _check_tss_gene_in_gencode(edge[0]),
                                edge[1],
                                -1,
                                edge_type,
                            )
                        )
                    else:
                        pass
                elif "tss" not in edge[0] and "tss" in edge[1]:
                    if _check_tss_gene_in_gencode(edge[1]):
                        return_edges.append(
                            (
                                edge[0],
                                _check_tss_gene_in_gencode(edge[1]),
                                -1,
                                edge_type,
                            )
                        )
                    else:
                        pass
                else:
                    return_edges.append(
                        (
                            edge[0],
                            edge[1],
                            -1,
                            edge_type,
                        )
                    )
            return return_edges
        else:
            first_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(first_anchor, feat_1),
                _loop_direct_overlap(first_anchor, feat_2),
            )
            second_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(second_anchor, feat_1),
                _loop_direct_overlap(second_anchor, feat_2),
            )
            return list(
                set(_loop_edges(first_anchor, first_anchor_edges, second_anchor_edges))
            )
    
    @time_decorator(print_args=True)
    def _process_graph_edges(self) -> None:
        """_summary_ of function"""
        ppi_edges = [
            self.iid_ppi(
                f"{self.interaction_dir}/{self.interaction_files['ppis']}", tissue
            )
            for tissue in self.PROTEIN_TISSUES
        ]
        mirna_targets = self._mirna_targets(target_list=f"{self.interaction_dir}/{self.interaction_files['mirnatargets']}")
        tf_markers = self._tf_markers(
            interaction_file=f"{self.interaction_dir}/{self.interaction_files['tf_marker']}",
        )
        circuit_edges = [self._marbach_regulatory_circuits(interaction_file=f"{self.circuit_dir}/{file}", score_filter=80,) for file in os.listdir(self.circuit_dir)]
        

def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
