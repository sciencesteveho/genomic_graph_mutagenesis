#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Handler for and pipeline execution for OGL preprocessing steps."""


import argparse
import contextlib
from typing import List

from omics_graph_learning.graph.construct_graphs import construct_tissue_graph
from omics_graph_learning.preprocessing.data_preprocessor import GenomeDataPreprocessor
from omics_graph_learning.preprocessing.edge_parser import EdgeParser
from omics_graph_learning.preprocessing.local_context_parser import LocalContextParser
from omics_graph_learning.split.target_consolidator import TrainingTargetConsolidator
from omics_graph_learning.utils.common import _get_files_in_directory
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.config_handlers import TissueConfig
from omics_graph_learning.utils.constants import ATTRIBUTES
from omics_graph_learning.visualization.targets import generate_all_target_plots

logger = setup_logging()


def _check_for_existing_data(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> bool:
    """Check if the pre-split data has already been parsed by looking for the
    output of linear context parser for a given experiment.
    """
    tissue = tissue_config.resources["tissue"]
    edge_dir = experiment_config.working_directory / tissue / "parsing" / "edges"
    file = edge_dir / LocalContextParser.sorted_concatenated_file
    exists = file.exists()
    logger.info(f"Edges exists: {exists}")
    return exists


def preprocess_bedfiles(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> None:
    """Directory set-up, bedfile symlinking and preprocessing"""
    preprocessObject = GenomeDataPreprocessor(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
    )

    preprocessObject.prepare_data_files()
    logger.info("Bedfile preprocessing complete!")


def parse_edges(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> None:
    """Parse nodes and edges to create base graph and for local context
    augmentation
    """
    edgeparserObject = EdgeParser(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
    )

    edgeparserObject.parse_edges()
    logger.info("Edges parsed!")


def parse_local_context(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> None:
    """Add local context edges based on basenode input"""

    def _get_config_filetypes(
        experiment_config: ExperimentConfig,
        tissue_config: TissueConfig,
    ) -> List[str]:
        """Only keep bedfiles that are relevant to the nodes in the graph,
        deriving from the constants and the configs."""
        local_dir = (
            experiment_config.working_directory
            / tissue_config.resources["tissue"]
            / "local"
        )
        keep_files = experiment_config.nodes + ATTRIBUTES + ["basenodes"]
        bedfiles = _get_files_in_directory(dir=local_dir)

        return [
            bedfile
            for bedfile in bedfiles
            if bedfile.split("_")[0].casefold() in keep_files
        ]

    bedfiles_for_parsing = _get_config_filetypes(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
    )

    # instantiate object
    localparseObject = LocalContextParser(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
        bedfiles=bedfiles_for_parsing,
    )

    localparseObject.parse_context_data()
    logger.info("Local context parser complete!")


def training_target_consolidator(
    experiment_config: ExperimentConfig,
    tissue_config: TissueConfig,
    tpm_filter: float,
    percent_of_samples_filter: float,
    filter_mode: str,
    split_name: str,
    target: str,
) -> List[str]:
    """Assemble tpm-filtered training targets for the sample."""
    consolidator = TrainingTargetConsolidator(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
        tpm_filter=tpm_filter,
        percent_of_samples_filter=percent_of_samples_filter,
        filter_mode=filter_mode,
        split_name=split_name,
        target=target,
    )

    return consolidator.consolidate_training_targets()


def create_tissue_graph(
    experiment_config: ExperimentConfig,
    tissue_config: TissueConfig,
    split_name: str,
    target_genes: List[str],
) -> None:
    """Creates a graph for the individual tissue. Concatting is dealt with after
    this initial pipeline.
    """
    tissue = tissue_config.resources["tissue"]

    construct_tissue_graph(
        nodes=experiment_config.nodes,
        experiment_name=experiment_config.experiment_name,
        working_directory=experiment_config.working_directory,
        split_name=split_name,
        graph_type=experiment_config.graph_type,
        tissue=tissue,
        target_genes=target_genes,
        build_positional_encoding=experiment_config.build_positional_encoding,
    )
    logger.info(f"Graph for {tissue} created!")


def main() -> None:
    """Pipeline to generate jobs for creating graphs"""
    # Parse arguments for type of graphs to produce
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    parser.add_argument(
        "--tissue_config", type=str, help="Path to .yaml file with filenames"
    )
    parser.add_argument("--tpm_filter", type=float, default=1.0)
    parser.add_argument("--percent_of_samples_filter", type=float, default=0.2)
    parser.add_argument(
        "--filter_mode",
        type=str,
        help="Mode to filter genes, specifying within the target tissue or across all possible gtex tissues (e.g. `within` or `across`). This is required if the target type is not `rna_seq`",
        default="within",
    )
    parser.add_argument("--split_name", type=str, help="Name of the split")
    parser.add_argument(
        "--target",
        type=str,
        help="Type of target to generate for training.",
    )
    parser.add_argument(
        "--positional_encoding",
        action="store_true",
        help="Whether to generate positional encodings.",
    )
    args = parser.parse_args()

    # Load in the experiment and tissue config files
    experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
    tissue_config = TissueConfig.from_yaml(args.tissue_config)

    # remove dyadic if encode regulatory schema
    if experiment_config.regulatory_schema == "encode":
        with contextlib.suppress(ValueError):
            experiment_config.nodes.remove("dyadic")

    # set up some common variables
    split_dir = experiment_config.working_directory / "graphs" / args.split_name
    experiment_name = experiment_config.experiment_name
    tissue = tissue_config.resources["tissue"]

    logger.info(
        f"Starting pipeline for {experiment_config.experiment_name}! "
        "Checking to see if pre-split data has been parsed..."
    )
    if _check_for_existing_data(
        experiment_config=experiment_config, tissue_config=tissue_config
    ):
        logger.info(
            "Pre-split data has already been parsed. Continuing to graph target consolidation onwards..."
        )
    else:
        logger.info(
            "Pre-split data has not been parsed yet. Starting pipeline to parse data..."
        )
        # pipeline!
        preprocess_bedfiles(
            experiment_config=experiment_config,
            tissue_config=tissue_config,
        )
        parse_edges(
            experiment_config=experiment_config,
            tissue_config=tissue_config,
        )
        parse_local_context(
            experiment_config=experiment_config,
            tissue_config=tissue_config,
        )

    target_genes = training_target_consolidator(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
        tpm_filter=args.tpm_filter,
        percent_of_samples_filter=args.percent_of_samples_filter,
        filter_mode=args.filter_mode,
        split_name=args.split_name,
        target=args.target,
    )
    create_tissue_graph(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
        split_name=args.split_name,
        target_genes=target_genes,
    )

    # plot the targets
    generate_all_target_plots(
        target_file=split_dir / f"training_targets_{tissue}.pkl",
        gene_file=split_dir / f"{experiment_name}_{tissue}_genes.pkl",
        gencode_gtf=experiment_config.reference_dir / tissue_config.local["gencode"],
        save_path=split_dir,
    )


if __name__ == "__main__":
    main()
