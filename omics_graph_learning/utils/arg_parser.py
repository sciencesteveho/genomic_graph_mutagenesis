#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Shared argument parser across OGL modules."""


import argparse
import contextlib
import sys
from typing import List

from omics_graph_learning.utils.constants import NodePerturbation


class OGLCLIParser:
    """Class for parsing command-line arguments for the OGL pipeline.

    Methods:
    --------
    parse_args:
        Parse the command-line arguments and validate.
    add_gnn_training_args:
        Add arguments related to train_gnn.py
    add_perturbation_arguments:
        Add arguments for graph perturbation experiments.

    Examples:
    --------
    # Import the parser
    >>> from omics_graph_learning.utils.arg_parser import OGLCLIParser

    # Call the base parser
    >>> parser = OGLCLIParser()
    >>> args = parser.parse_args()

    # Add arguments for training GNN
    >>> parser = OGLCLIParser()
    >>> parser.add_gnn_training_args()
    >>> args = parser.parse_args()
    """

    def __init__(self):
        """Initialize the argument parser."""
        self.parser = argparse.ArgumentParser(
            description="Omics Graph Learning Pipeline"
        )
        self._add_base_arguments()

    def parse_args(self) -> argparse.Namespace:
        """Parse the command-line arguments and validate."""
        args = self.parser.parse_args()
        self._validate_args(args)
        self._replace_none_with_none(args)
        return args

    def _replace_none_with_none(self, args: argparse.Namespace) -> None:
        """Replace string "None" with None in the parsed arguments."""
        for arg, value in vars(args).items():
            if value == "None":
                setattr(args, arg, None)

    def _add_base_arguments(self) -> None:
        """Add base arguments to the parser."""
        self.add_configuration_arguments()
        self.add_model_arguments()
        self.add_boolean_flags()

    def add_configuration_arguments(self) -> None:
        """Add required arguments to the parser."""
        self.parser.add_argument(
            "--experiment_yaml",
            type=str,
            help="Path to experiment YAML file",
            required=True,
        )
        self.parser.add_argument(
            "--partition",
            type=str,
            choices=["RM", "EM"],
            help="Partition for SLURM scheduling",
            default="RM",
        )
        self.parser.add_argument(
            "--tpm_filter",
            type=float,
            default=0.5,
        )
        self.parser.add_argument(
            "--percent_of_samples_filter",
            type=float,
            default=0.1,
        )
        self.parser.add_argument(
            "--filter_mode",
            type=str,
            default="within",
            help="Mode to filter genes, specifying within the target tissue or across all possible gtex tissues (e.g. `within` or `across`). This is required if the target type is not `rna_seq`",
        )
        self.parser.add_argument(
            "--clean-up",
            action="store_true",
            help="Remove intermediate files in tissue-specific directories",
            default=False,
        )
        self.parser.add_argument("--n_gpus", type=int)
        self.parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="Alternative model name.",
        )

    def add_model_arguments(self) -> None:
        """Add arguments related to GNN model configuration or training."""
        self.parser.add_argument(
            "--model",
            type=str,
            default="GCN",
            choices=[
                "GCN",
                "GraphSAGE",
                "PNA",
                "GAT",
                "UniMPTransformer",
                "DeeperGCN",
                "MLP",
            ],
        )
        self.parser.add_argument(
            "--target",
            type=str,
            default="expression_median_only",
            choices=[
                "expression_median_only",
                "expression_media_and_foldchange",
                "difference_from_average",
                "foldchange_from_average",
                "protein_targets",
                "rna_seq",
            ],
        )
        self.parser.add_argument("--gnn_layers", type=int, default=2)
        self.parser.add_argument("--linear_layers", type=int, default=3)
        self.parser.add_argument(
            "--activation",
            type=str,
            default="relu",
            choices=["relu", "leakyrelu", "gelu"],
        )
        self.parser.add_argument("--dimensions", type=int, default=256)
        self.parser.add_argument(
            "--residual",
            type=str,
            default=None,
            choices=["shared_source", "distinct_source", "None"],
        )
        self.parser.add_argument("--epochs", type=int, default=60)
        self.parser.add_argument("--batch_size", type=int, default=256)
        self.parser.add_argument("--learning_rate", type=float, default=1e-4)
        self.parser.add_argument(
            "--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"]
        )
        self.parser.add_argument(
            "--scheduler",
            type=str,
            default="plateau",
            choices=["plateau", "cosine", "linear_warmup"],
        )
        self.parser.add_argument(
            "--regression_loss_type",
            type=str,
            default="rmse",
            choices=["rmse", "smooth_l1"],
        )
        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--heads", type=int, default=None)
        self.parser.add_argument("--n_trials", type=int)
        self.parser.add_argument(
            "--run_number",
            type=int,
            help="Run number to specify for GNN training. If not specified, the pipeline will instead submit three jobs [0, 1, 2] and train three models across three seeds.",
        )

    def add_boolean_flags(self) -> None:
        """Add boolean flags to the parser."""
        self.parser.add_argument(
            "--attention_task_head", action="store_true", default=False
        )
        self.parser.add_argument("--positional_encoding", action="store_true")
        self.parser.add_argument("--early_stop", action="store_true", default=True)
        self.parser.add_argument("--gene_only_loader", action="store_true")
        self.parser.add_argument("--optimize_params", action="store_true")

    def add_perturbation_arguments(self) -> None:
        """Add perturbation arguments to the parser."""
        perturbation_choices: List[str] = [
            perturbation.name for perturbation in NodePerturbation
        ]
        self.parser.add_argument(
            "--node_perturbation",
            type=str,
            default=None,
            choices=perturbation_choices,
            help="Type of node based perturbation to apply. Choose from either `zero_node_feats`, `randomize_node_feats`, `randomize_node_feat_order`, or pick the name of a specific feat to perturb",
        )
        self.parser.add_argument(
            "--edge_perturbation",
            type=str,
            default=None,
            choices=["randomize_edges", "remove_all_edges", "remove_specific_edges"],
            help="Type of node based perturbation to apply. Choose from either `zero_node_feats`, `randomize_node_feats`, `randomize_node_feat_order`, or pick the name of a specific feat to perturb",
        )
        self.parser.add_argument("--total_random_edges", type=int, default=None)

    def add_gnn_training_args(self) -> None:
        """Add arguments related to train_gnn.py"""
        self.parser.add_argument("--split_name", type=str, required=True)
        self.parser.add_argument(
            "--seed", type=int, default=42, help="random seed to use (default: 42)"
        )
        self.parser.add_argument(
            "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
        )
        self.parser.add_argument("--alpha", type=float, default=0.85)

    @staticmethod
    def _validate_args(args: argparse.Namespace) -> None:
        """Helper function to validate CLI arguments that have dependencies."""
        if args.target != "rna_seq" and args.filter_mode is None:
            print("Error: if target type is not `rna_seq`, --filter_mode is required")
            sys.exit(1)

        if args.model in ["GAT", "UniMPTransformer"] and args.heads is None:
            print(f"Error: --heads is required when model is {args.model}")
            sys.exit(1)

        if args.optimize_params and args.n_gpus is None:
            print(
                "Error: specifying --n_gpus is required when --optimize_params is set."
            )
            sys.exit(1)

        # only validate if perturbation arguments exist
        with contextlib.suppress(AttributeError):
            if args.total_random_edges and args.edge_perturbation != "randomize_edges":
                print(
                    "Error: if --total_random_edges is set, --edge_perturbation must be `randomize_edges`"
                )
                sys.exit(1)
