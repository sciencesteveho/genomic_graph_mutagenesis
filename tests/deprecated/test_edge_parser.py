#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Test the EdgeParser class which handles the basis of graph construction."""


# make a fake small chr loop file
# make a fake small enhancer file
# make a fake small promoter file
# make a fake small gene file
# do the overlaps and make sure the edges are actually parsed


from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.config_handlers import TissueConfig
from omics_graph_learning.edge_parser import EdgeParser
import pandas as pd
import pybedtools  # type: ignore
import pytest


# TODO replace mock config with a mock_config.yaml
@pytest.fixture
def mock_configs():
    experiment_config = MagicMock(spec=ExperimentConfig)
    experiment_config.experiment_name = "test_experiment"
    experiment_config.interaction_types = ["mirna", "tf_marker", "tfbinding"]
    experiment_config.gene_gene = False
    experiment_config.root_dir = Path("/root")
    experiment_config.attribute_references = {
        "gencode": "/path/to/gencode",
        "regulatory_elements": "/path/to/regulatory",
        "super_enhancers": "/path/to/super_enhancers",
        "mirna": "/path/to/mirna",
        "tf_motifs": "/path/to/tf_motifs",
    }
    experiment_config.regulatory_schema = "encode"
    experiment_config.working_directory = Path("/working")
    experiment_config.rbp_proteins = ["RBPX", "RBPY"]
    experiment_config.rbp_network = "/path/to/rbp_network"

    tissue_config = MagicMock(spec=TissueConfig)
    tissue_config.local = {"gencode": "gencode.bed", "tss": "tss.bed"}
    tissue_config.interaction = {
        "mirdip": "mirdip.txt",
        "tf_marker": "tf_marker.txt",
        "tfbinding": "tfbinding.txt",
    }
    tissue_config.resources = {
        "blacklist": "/path/to/blacklist",
        "tissue": "test_tissue",
        "chromfile": "/path/to/chromfile",
        "marker_name": "test_marker",
        "ppi_tissue": "test_ppi_tissue",
        "rna": "/path/to/rna_seq",
    }

    return experiment_config, tissue_config


@pytest.fixture
def edge_parser(mock_configs):
    experiment_config, tissue_config = mock_configs
    return EdgeParser(experiment_config, tissue_config)


def test_initialization(edge_parser, mock_configs):
    experiment_config, tissue_config = mock_configs
    assert edge_parser.experiment_name == "test_experiment"
    assert edge_parser.interaction_types == ["mirna", "tf_marker", "tfbinding"]
    assert edge_parser.gene_gene == False
    assert edge_parser.root_dir == Path("/root")
    assert edge_parser.tissue == "test_tissue"


@patch("edge_parser.pybedtools.BedTool")
def test_remove_blacklist(mock_bedtool, edge_parser):
    mock_bed = MagicMock()
    mock_bed.intersect.return_value = "filtered_bed"
    edge_parser.blacklist = mock_bed

    result = edge_parser._remove_blacklist(mock_bedtool)

    mock_bed.intersect.assert_called_once_with(mock_bedtool, v=True, sorted=True)
    assert result == "filtered_bed"


@patch("edge_parser.pybedtools.BedTool")
@patch("os.path.exists")
def test_create_bedtool(mock_exists, mock_bedtool, edge_parser):
    mock_exists.return_value = True
    edge_parser._remove_blacklist = MagicMock(return_value="filtered_bed")

    result = edge_parser._create_bedtool(Path("/test/path"))

    mock_bedtool.assert_called_once_with(Path("/test/path"))
    edge_parser._remove_blacklist.assert_called_once()
    assert result == "filtered_bed"


def test_prepare_regulatory_elements(edge_parser):
    edge_parser._create_bedtool = MagicMock(
        side_effect=["enhancer_bed", "promoter_bed", "dyadic_bed"]
    )

    enhancer, promoter, dyadic = edge_parser._prepare_regulatory_elements()

    assert enhancer == "enhancer_bed"
    assert promoter == "promoter_bed"
    assert dyadic == "dyadic_bed"


@patch("edge_parser.open")
@patch("edge_parser.csv.reader")
def test_mirna_targets(mock_csv_reader, mock_open, edge_parser):
    mock_csv_reader.side_effect = [
        [["miRNA1"], ["miRNA2"]],  # active_mirna
        [["miRNA1", "Gene1"], ["miRNA2", "Gene2"], ["miRNA3", "Gene3"]],  # target_list
    ]
    edge_parser.genesymbol_to_gencode = {"Gene1": "ENSG1", "Gene2": "ENSG2"}

    result = list(edge_parser._mirna_targets("target_list.txt", "active_mirna.txt"))

    assert result == [("miRNA1", "ENSG1", "mirna"), ("miRNA2", "ENSG2", "mirna")]


@patch("edge_parser.open")
@patch("edge_parser.csv.reader")
def test_tf_markers(mock_csv_reader, mock_open, edge_parser):
    mock_csv_reader.return_value = iter(
        [
            ["header"],
            ["Gene1", "TF1", "TF", "", "", "test_marker", "", "", "", "", "Target1"],
            [
                "Gene2",
                "TF2",
                "I Marker",
                "",
                "",
                "test_marker",
                "",
                "",
                "",
                "",
                "Target1;Target2",
            ],
        ]
    )
    edge_parser.genesymbol_to_gencode = {
        "TF1": "ENSG_TF1",
        "TF2": "ENSG_TF2",
        "Target1": "ENSG_T1",
        "Target2": "ENSG_T2",
    }
    edge_parser.tf_extension = "_tf"

    result = list(edge_parser._tf_markers("tf_markers.txt"))

    assert result == [
        ("ENSG_TF1_tf", "ENSG_T1", "tf_marker"),
        ("ENSG_T1", "ENSG_TF2_tf", "tf_marker"),
        ("ENSG_T2", "ENSG_TF2_tf", "tf_marker"),
    ]


@patch("edge_parser.pybedtools.BedTool")
def test_tfbinding_footprints(mock_bedtool, edge_parser):
    mock_tf_binding = MagicMock()
    mock_tf_binding.intersect.return_value = [
        ["chr1", "100", "200", "TF1", "1", "+", "chr1", "150", "160", "FP1"],
        ["chr2", "300", "400", "TF2", "1", "-", "chr2", "350", "360", "FP2"],
    ]
    edge_parser._remove_blacklist = MagicMock(return_value=mock_tf_binding)
    edge_parser.genesymbol_to_gencode = {"TF1": "ENSG_TF1", "TF2": "ENSG_TF2"}

    result = list(edge_parser._tfbinding_footprints("tfbinding.txt", "footprints.txt"))

    assert result == [
        ("ENSG_TF1_tf", "chr1_150_FP1", "tf_binding_footprint"),
        ("ENSG_TF2_tf", "chr2_350_FP2", "tf_binding_footprint"),
    ]


@patch("edge_parser.pd.DataFrame")
def test_generate_edge_combinations(mock_dataframe, edge_parser):
    df1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [100, 200],
            "end": [150, 250],
            "name": ["A", "B"],
            "score": [1, 2],
            "strand": ["+", "-"],
            "thickStart_x": ["Gene1,Gene2", "Gene3"],
        }
    )
    df2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [100, 200],
            "end": [150, 250],
            "name": ["A", "B"],
            "score": [1, 2],
            "strand": ["+", "-"],
            "thickStart_y": ["Gene4", "Gene5,Gene6"],
        }
    )

    result = edge_parser._generate_edge_combinations(df1, df2, "test_edge")

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "edge_0": ["Gene1", "Gene2", "Gene3"],
                "edge_1": ["Gene4", "Gene4", "Gene5"],
                "type": ["test_edge", "test_edge", "test_edge"],
            }
        ),
    )


@patch("edge_parser.pybedtools.BedTool")
def test_split_chromatin_loops(mock_bedtool, edge_parser):
    mock_bedtool.return_value.sort.return_value = MagicMock()
    mock_bedtool.return_value.sort.return_value.cut.return_value = "second_anchor"

    first_anchor, second_anchor = EdgeParser._split_chromatin_loops("loops.txt")

    assert isinstance(first_anchor, MagicMock)
    assert second_anchor == "second_anchor"


def test_reverse_anchors(edge_parser):
    mock_bed = MagicMock()
    mock_bed.cut.return_value = "reversed_bed"

    result = EdgeParser._reverse_anchors(mock_bed)

    mock_bed.cut.assert_called_once_with([3, 4, 5, 0, 1, 2, 6])
    assert result == "reversed_bed"


def test_add_feat_names(edge_parser):
    feature = ["chr1", "100", "200", "name", "1", "+", "old_name", "7", "8", "new_name"]

    result = EdgeParser._add_feat_names(feature)

    assert result[6] == "new_name"


@patch("edge_parser.pybedtools.BedTool")
def test_loop_direct_overlap(mock_bedtool, edge_parser):
    mock_loops = MagicMock()
    mock_features = MagicMock()
    mock_loops.intersect.return_value = "overlapped_bed"

    result = EdgeParser._loop_direct_overlap(mock_loops, mock_features)

    mock_loops.intersect.assert_called_once_with(mock_features, wo=True, stream=True)
    assert result == "overlapped_bed"


if __name__ == "__main__":
    pytest.main()
