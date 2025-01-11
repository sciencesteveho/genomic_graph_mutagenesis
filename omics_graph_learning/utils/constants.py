#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Constants for omics graph learning modules."""


from enum import Enum
from typing import Dict, List


class NodePerturbation(Enum):
    """Enum class to handle node perturbation types."""

    size = 0
    gc = 1
    atac = 2
    cnv = 3
    cpg = 4
    ctcf = 5
    dnase = 6
    h3k27ac = 7
    h3k27me3 = 8
    h3k36me3 = 9
    h3k4me1 = 10
    h3k4me2 = 11
    h3k4me3 = 12
    h3k79me2 = 13
    h3k9ac = 14
    h3k9me3 = 15
    indels = 16
    line = 17
    ltr = 18
    microsatellites = 19
    phastcons = 20
    polr2a = 21
    polyasites = 22
    rad21 = 23
    rbpbindingsites = 24
    recombination = 25
    repg1b = 26
    repg2 = 27
    reps1 = 28
    reps2 = 29
    reps3 = 30
    reps4 = 31
    rnarepeat = 32
    simplerepeats = 33
    sine = 34
    smc3 = 35
    snp = 36
    zero_node_feats = -1
    randomize_node_feats = -2
    randomize_node_feat_order = -3


# integer constants
ANCHOR_GRACE = 1500
EARLY_STOP_PATIENCE = 12
HG38_TOTAL_BASEPAIRS = 3088269227
MAX_FEAT_LEN = 39  # 37 node features + is_gene + is_tf
N_TRIALS = 200
RANDOM_SEEDS = [42, 84, 168]


# filename constants
TARGET_FILE = "targets_combined.pkl"
TARGET_FILE_SCALED = "targets_combined_scaled.pkl"
TRAINING_SPLIT_FILE = "training_split_combined.pkl"


# graph construction related helpers
ATTRIBUTES: List[str] = [
    "gc",
    "atac",
    "cnv",
    "cpg",
    "ctcf",
    "dnase",
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
    "h3k79me2",
    "h3k9ac",
    "h3k9me3",
    "indels",
    "line",
    "ltr",
    "microsatellites",
    "phastcons",
    "polr2a",
    "polyasites",
    "rad21",
    "rbpsiteclusters",
    "recombination",
    "repg1b",
    "repg2",
    "reps1",
    "reps2",
    "reps3",
    "reps4",
    "rnarepeat",
    "simplerepeats",
    "sine",
    "smc3",
    "snp",
]

NODE_FEAT_IDXS: Dict[str, int] = {
    "size": 0,
    "gc": 1,
    "atac": 2,
    "cnv": 3,
    "cpg": 4,
    "ctcf": 5,
    "dnase": 6,
    "h3k27ac": 7,
    "h3k27me3": 8,
    "h3k36me3": 9,
    "h3k4me1": 10,
    "h3k4me2": 11,
    "h3k4me3": 12,
    "h3k79me2": 13,
    "h3k9ac": 14,
    "h3k9me3": 15,
    "indels": 16,
    "line": 17,
    "ltr": 18,
    "microsatellites": 19,
    "phastcons": 20,
    "polr2a": 21,
    "polyasites": 22,
    "rad21": 23,
    "rbpbindingsites": 24,
    "recombination": 25,
    "repg1b": 26,
    "repg2": 27,
    "reps1": 28,
    "reps2": 29,
    "reps3": 30,
    "reps4": 31,
    "rnarepeat": 32,
    "simplerepeats": 33,
    "sine": 34,
    "smc3": 35,
    "snp": 36,
}


POSSIBLE_NODES: List[str] = [
    "cpgislands",
    "crms",
    "ctcfccre",
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
    "superenhancers",
    "tads",
    "tfbindingsites",
    "tss",
]

TISSUES: List[str] = [
    "aorta",
    "hippocampus",
    "left_ventricle",
    "liver",
    "lung",
    "mammary",
    "pancreas",
    "skeletal_muscle",
    "skin",
    "small_intestine",
    # "hela",
    # "k562",
    # "npc",
]

REGULATORY_ELEMENTS: Dict[str, Dict[str, str]] = {
    "intersect": {
        "enhancer": "enhancer_epimap_screen_overlap.bed",
        "promoter": "promoter_epimap_screen_overlap.bed",
        "dyadic": "dyadic_epimap_screen_overlap.bed",
    },
    "union": {
        "enhancer": "enhancer_all_union_hg38.bed",
        "promoter": "promoter_all_union_hg38.bed",
        "dyadic": "DYADIC_masterlist_locations.lifted_hg38.bed",
    },
    "epimap": {
        "enhancer": "ENH_masterlist_locations.lifted_hg38.bed",
        "promoter": "PROM_masterlist_locations.lifted_hg38.bed",
        "dyadic": "DYADIC_masterlist_locations.lifted_hg38.bed",
    },
    "encode": {
        "enhancer": "GRCh38-ELS.bed",
        "promoter": "GRCh38-PLS.bed",
    },
}


# optimization helpers
SUBSET_CHROMOSOMES = [
    "chr1",  # large chromosome, very high gene density (4924 genes)
    "chr2",  # large chromosome, high gene density (3787 genes)
    "chr15",  # medium to small chromosome, moderate gene density (1988 genes)
    "chr11",  # medium chromosome, high gene density (3046 genes)
    "chr4",  # medium chromosome, moderate gene density (2418 genes)
    "chr17",  # medium chromosome, high gene density (2763 genes)
    "chr3",  # large chromosome, moderate gene density (2862 genes)
    "chr12",  # medium chromosome, moderate gene density (2773 genes)
    "chr5",  # medium to large chromosome, moderate gene density (2735 genes)
    "chr6",  # medium chromosome, moderate gene density (2738 genes)
    "chr14",  # medium chromosome, moderate gene density (2044 genes)
    "chr7",  # medium chromosome, moderate gene density (2704 genes)
]


"""
Gene density reference
    4924 chr1
    2095 chr10
    3046 chr11
    2773 chr12
    1239 chr13
    2044 chr14
    1988 chr15
    2317 chr16
    2763 chr17
    1113 chr18
    2626 chr19
    3787 chr2
    1299 chr20
    791 chr21
    1224 chr22
    2862 chr3
    2418 chr4
    2735 chr5
    2738 chr6
    2704 chr7
    2226 chr8
    2102 chr9
"""
