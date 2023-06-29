#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] add index for loop in edge feat
# - [ ] adjust the edges output to resembl that from edge_parser.py
# - [ ] figure out why product returns key error. need to check for empty keys


"""Create a base graph from chromatin loop data.

We filter for distal ELSs and link them to other enhancers and promoters (loop
overlap) or genes (within 2kb of a loop anchor)."""

import csv
import itertools
import re
from typing import Dict, List, Tuple

import pybedtools


def _load_tss(tss_path: str) -> pybedtools.BedTool:
    """Load TSS file and ignore any TSS that do not have a gene target.
    Returns:
        pybedtools.BedTool - TSS w/ target genes
    """
    tss = pybedtools.BedTool(tss_path)
    return tss.filter(lambda x: x[3].split("_")[3] != "").saveas()


def _split_chromatin_loops(
    loop_path: str,
    caller: str = "yue",
    cutoff: int = 0,
) -> Tuple[pybedtools.BedTool, pybedtools.BedTool]:
    """_summary_

    Args:
        loop_path (str): _description_

    Returns:
        Tuple[pybedtools.BedTool, pybedtools.BedTool]: _description_
    """
    if caller == "peakachu":
        loop_path = loop_path
    elif caller == "deeploop":
        file_reader = csv.reader(open(loop_path), delimiter="\t")
        anchors = [
            (re.split(":|-", line[0] + ":" + line[1])) + [line[2]]
            for line in file_reader
        ]
        try:
            sorted_anchors = sorted(
                [anchor for idx, anchor in enumerate(anchors) if idx > cutoff],
                key=lambda x: float(x[6]),
                reverse=True,
            )  # sort by score and keep n top anchors
            loop_path = sorted_anchors
        except IndexError:
            print("Deeploop calls have associated scores, which are missing. Is this a loop file from Peakachu?")
    else:
        raise ValueError("Caller must be either 'peakachu' or 'deeploop'")
    first_anchor = pybedtools.BedTool(loop_path)
    second_anchor = pybedtools.BedTool(
        "\n".join(
            ["\t".join([x[3], x[4], x[5], x[0], x[1], x[2]]) for x in first_anchor]
        ),
        from_string=True,
    )
    return first_anchor, second_anchor


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
            anchor.setdefault("_".join(feature[0:3]), []).append("_".join(feature[7:]))
    return anchor


def _loop_edges(
    loops: pybedtools.BedTool,
    first_anchor_edges: Dict[str, List[str]],
    second_anchor_edges: Dict[str, List[str]],
) -> None:
    """Return a list of edges that are connected by their overlap over chromatin
    loop anchors by matching the anchor names across dicts"""
    edges = []
    for loop in loops:
        first_anchor = "_".join(loop[0:3])
        second_anchor = "_".join(loop[3:6])
        try:
            uniq_edges = list(
                itertools.product(
                    first_anchor_edges[first_anchor], second_anchor_edges[second_anchor]
                )
            )
            edges.extend(uniq_edges)
        except KeyError:
            continue
    return edges


def get_loop_edges(
    enhancers,
    tss,
    loop_path,
    caller,
):
    first_anchor, second_anchor = _split_chromatin_loops(
        loop_path=loop_path,
        caller=caller,
    )
    
    first_anchor_edges = _flatten_anchors(
        _loop_direct_overlap(first_anchor, enhancers),
        _loop_within_distance(first_anchor, tss, 2000),
    )
    second_anchor_edges = _flatten_anchors(
        _loop_direct_overlap(second_anchor, enhancers),
        _loop_within_distance(second_anchor, tss, 2000),
    )
    
    return _loop_edges(first_anchor, first_anchor_edges, second_anchor_edges)


def main(
    enhancer_path: str,
    loop_path: str,
    tss_path: str,   
) -> None:
    """Main function"""
    enhancers = pybedtools.BedTool(enhancer_path)
    tss = _load_tss(tss_path)
    edges = get_loop_edges(
        enhancers=enhancers
    )
    len(set([x for edge in edges for x in edge]))


if __name__ == "__main__":
    main(
        enhancer_path = "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/epimap/BSS00511_LVR.LIVER_hg38_enhancer.bed",
        loop_path = "GSE167200_Liver.top300K.txt",
        tss_path = "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/reftss/reftss_annotated.bed",
    )


# enhancer_path = (
#     "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/GRCh38-ELS.bed"
# )
# enhancer_path = (
#     "/ocean/projects/bio210019p/stevesho/data/preprocess/v4_graphs/hippocampus/local/enhancers_lifted_hippocampus.bed_noalt"
# )
# enhancer_path = (
#     "/ocean/projects/bio210019p/stevesho/data/preprocess/v4_graphs/liver/local/enhancers_lifted_liver.bed_noalt"
# # )
# loop_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/hippocampus/Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops"
    

# liver
# all els - 137720
# fenrir specific - 16539
# epimap - 20149

# hippocampus
# all els - 110449
# fenrir specific - 12515