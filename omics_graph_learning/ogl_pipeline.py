#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Main script for running the Omics Graph Learning pipeline, built for running
on a HPC via the SLURM scheduler."""


import argparse
import contextlib
import os
from pathlib import Path
import shutil
from typing import List, Optional, Tuple

import pytest

from omics_graph_learning.utils.arg_parser import OGLCLIParser
from omics_graph_learning.utils.common import _dataset_split_name
from omics_graph_learning.utils.common import _run_command
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.common import submit_slurm_job
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.constants import N_TRIALS

logger = setup_logging()


class PipelineRunner:
    """Class for handling the entire pipeline, from data parsing to graph
    construction and GNN training. The specific implementation is designed for
    running on a SLURM cluster and there are accompanying SLURM scripts for each
    step.

    Attributes:
        config (ExperimentConfig): Experiment configuration
        args (argparse.Namespace): Parsed CLI arguments specific to the GNN
        experiment

    Methods:
    --------
    run_pipeline: Run the entire pipeline

    Examples:
    --------
    # Run the pipeline and train a model according to default params
    >>> python ogl.py \
        --partition [PARTITION] 
        --experiment_yaml [EXPERIMENT_YAML_PATH] \

    # Run the pipeline and train a model, overriding default params
    >>> python ogl.py \
        --partition [PARTITION] \
        --experiment_yaml [EXPERIMENT_YAML_PATH] \
        --model GAT \
        --heads 2

    # Run the pipeline and optimize hyperparameters instead of training a full
    model
    >>> python ogl.py \
        --partition [PARTITION] \
        --experiment_yaml [EXPERIMENT_YAML_PATH] \
        --optimize_params
    """

    def __init__(self, config: ExperimentConfig, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.graph_dir = self.config.graph_dir
        self.tissues = self.config.tissues

    def clean_up(self) -> None:
        """Remove intermediate files in tissue-specific directories."""

        def _remove_files_in_dir(directory: Path) -> None:
            """Remove files in a directory, avoiding removing symlinks
            recursively!
            """
            for item in directory.iterdir():
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                elif item.is_file() or (
                    item.is_symlink() and not item.resolve().is_dir()
                ):
                    item.unlink()

        directories_to_clean: List[Path] = []
        for tissue in self.tissues:
            tissue_dir = self.config.working_directory / tissue
            directories_to_clean += [
                tissue_dir / "local",
                tissue_dir / "parsing" / "attributes",
                tissue_dir / "parsing" / "intermediate",
                tissue_dir / "interaction",
                tissue_dir / "unprocessed",
            ]

            # files to preserve
            interaction_edges = tissue_dir / "interaction" / "interaction_edges.txt"
            basenodes_hg38 = (
                tissue_dir / "parsing" / "attributes" / "basenodes_hg38.txt"
            )

            if interaction_edges.exists():
                shutil.move(
                    str(interaction_edges), str(tissue_dir / "interaction_edges.txt")
                )

            if basenodes_hg38.exists():
                shutil.move(str(basenodes_hg38), str(tissue_dir / "basenodes_hg38.txt"))

        # remove all files in the directories; do not follow symlinks
        for directory in directories_to_clean:
            if directory.is_dir():
                for item in directory.iterdir():
                    if item.is_dir() and not item.is_symlink():
                        shutil.rmtree(item)
                    elif item.is_file() or (
                        item.is_symlink() and not item.resolve().is_dir()
                    ):
                        item.unlink()

                # Remove the empty directory itself
                if directory.exists():
                    directory.rmdir()

    def _get_file_paths(self, split_name: str) -> Tuple[str, str]:
        """Construct file paths for graphs to check if files exist"""
        experiment_name = self.config.experiment_name
        final_graph = os.path.join(
            self.graph_dir,
            split_name,
            f"{experiment_name}_{self.config.graph_type}_scaled.pkl",
        )
        intermediate_graph = os.path.join(
            self.graph_dir,
            split_name,
            f"{experiment_name}_{self.config.graph_type}_graph.pkl",
        )
        return final_graph, intermediate_graph

    def _check_all_intermediates(self, split_name: str) -> bool:
        """Check if all intermediate graphs are present."""
        for tissue in self.tissues:
            intermediate_graph = os.path.join(
                self.graph_dir,
                split_name,
                f"{tissue}_{self.config.graph_type}_graph.pkl",
            )
            if not os.path.isfile(intermediate_graph):
                return False
        return True

    def get_train_test_val_split(self, slurm_dependency: str, split_name: str) -> str:
        """Submit a SLURM job to get splits."""
        sbatch_command = f"sbatch --parsable --dependency=afterok:{slurm_dependency} training_targets.sh {self.args.experiment_yaml} {self.args.tpm_filter} {self.args.percent_of_samples_filter} {split_name}"
        if self.args.target == "rna_seq":
            sbatch_command += " --rna_seq"
        return _run_command(command=sbatch_command, get_output=True) or ""

    def run_node_and_edge_generation(self, split_name: str) -> List[str]:
        """Run node and edge generation jobs."""
        partition_specific_script = (
            "pipeline_node_and_edge_generation_mem.sh"
            if self.args.partition == "EM"
            else "pipeline_node_and_edge_generation.sh"
        )

        slurmids = []
        for tissue in self.tissues:
            args = f"{self.args.experiment_yaml} \
                {self.config.sample_config_dir}/{tissue}.yaml \
                {self.args.tpm_filter} \
                {self.args.percent_of_samples_filter} \
                {self.args.filter_mode} \
                {split_name} \
                {self.args.target}"

            if self.args.positional_encoding:
                args += " --positional_encoding"

            job_id = submit_slurm_job(
                job_script=partition_specific_script,
                args=args,
                dependency=None,
            )
            slurmids.append(job_id)
        return slurmids

    def run_graph_concatenation(
        self,
        split_name: str,
        slurmids: Optional[List[str]],
    ) -> str:
        """Run graph concatenation job."""
        constructor = "concat.sh"
        return submit_slurm_job(
            job_script=constructor,
            args=f"{self.args.experiment_yaml} \
                {split_name}",
            dependency=":".join(slurmids) if slurmids else None,
        )

    def scale_node_features(self, slurmids: List[str], split_name: str) -> str:
        """Run node feature scaling job."""
        return submit_slurm_job(
            job_script="scale_features.sh",
            args=f"{self.args.experiment_yaml} \
                {split_name}",
            dependency=":".join(slurmids),
        )

    def prepare_gnn_training_args(
        self, args: argparse.Namespace, split_name: str, run_number: int
    ) -> str:
        """Prepare arguments for GNN training."""
        bool_flags = " ".join(
            [
                f"--{flag}"
                for flag in [
                    "early_stop",
                    "attention_task_head",
                    "positional_encoding",
                    "gene_only_loader",
                    "optimize_params",
                ]
                if getattr(args, flag)
            ]
        )

        train_args = (
            f"--experiment_yaml {args.experiment_yaml} "
            f"--model {args.model} "
            f"--target {args.target} "
            f"--gnn_layers {args.gnn_layers} "
            f"--linear_layers {args.linear_layers} "
            f"--activation {args.activation} "
            f"--dimensions {args.dimensions} "
            f"--residual {args.residual} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--learning_rate {args.learning_rate} "
            f"--optimizer {args.optimizer} "
            f"--scheduler {args.scheduler} "
            f"--regression_loss_type {args.regression_loss_type} "
            f"--dropout {args.dropout} "
            f"--split_name {split_name} "
            f"--run_number {run_number} "
            f"--alpha {args.alpha} "
            f"--model_name {args.model_name} "
        )

        if args.heads:
            train_args += f" --heads {args.heads} "

        with contextlib.suppress(AttributeError):
            if args.total_random_edges:
                if args.edge_perturbation == "randomize_edges":
                    train_args += f" --total_random_edges {args.total_random_edges} "
                else:
                    raise ValueError(
                        "`total_random_edges` should only be set when `randomize_edges` is True"
                    )
            if args.node_perturbation:
                train_args += f" --node_perturbation {args.node_perturbation} "
            if args.edge_perturbation:
                train_args += f" --edge_perturbation {args.edge_perturbation} "
            if args.randomize_node_feature_idx:
                train_args += (
                    f" --randomize_node_feature_idx {args.randomize_node_feature_idx} "
                )
        return train_args + bool_flags

    def submit_gnn_job(self, split_name: str, dependency: Optional[str]) -> None:
        """Submit GNN training job."""
        if self.args.optimize_params:
            self.submit_optimization(split_name, dependency)
        elif self.args.run_number:
            train_args = self.prepare_gnn_training_args(
                self.args, split_name, self.args.run_number
            )
            submit_slurm_job(
                job_script="train_gnn.sh", args=train_args, dependency=dependency
            )
            logger.info("GNN training job submitted.")
        else:
            for run_number in range(1, 4):
                train_args = self.prepare_gnn_training_args(
                    self.args, split_name, run_number
                )
                submit_slurm_job(
                    job_script="train_gnn.sh",
                    args=train_args,
                    dependency=dependency,
                )
                logger.info(f"GNN training job submitted for run {run_number}.")

    def submit_optimization(self, split_name: str, dependency: Optional[str]) -> None:
        """Submit hyperparameter optimization jobs."""
        if not self.args.n_gpus:
            raise ValueError(
                "Number of GPUs must be specified when optimizing hyperparameters."
            )
        slurm_ids = []
        num_trials = (
            calculate_trials(self.args.n_gpus, self.args.n_trials)
            if self.args.n_trials
            else calculate_trials(self.args.n_gpus, N_TRIALS)
        )
        for _ in range(self.args.n_gpus):
            job_id = submit_slurm_job(
                job_script="optimize_params.sh",
                args=f"{self.args.experiment_yaml} {self.args.target} {split_name} {num_trials}",
                dependency=dependency,
            )
            slurm_ids.append(job_id)
        logger.info(
            f"{self.args.n_gpus} hyperparameter optimization jobs submitted for {num_trials} each."
        )
        self.plot_importances(slurm_ids)

    def plot_importances(self, slurmids: List[str]) -> None:
        """Submit job to plot feature importances."""
        submit_slurm_job(
            job_script="importances.sh",
            args=f"{self.args.experiment_yaml}",
            dependency=":".join(slurmids),
        )
        logger.info("Feature importances plot job submitted.")

    def all_pipeline_jobs(self, intermediate_graph: str, split_name: str) -> None:
        """Submit all pipeline jobs if a final graph is not found."""
        if not os.path.isfile(intermediate_graph):
            logger.info("No intermediates found. Running entire pipeline!")
            slurm_id = self.graph_construction_jobs(split_name)
        else:
            slurm_id = self.post_split_jobs(split_name)
        scale_id = self.scale_node_features([slurm_id], split_name)
        logger.info("Node feature scaling job submitted.")

        self.submit_gnn_job(split_name, scale_id)

    def graph_construction_jobs(self, split_name: str) -> str:
        """Submit jobs for node and edge generation, local context parsing, and
        graph construction."""
        # node and edge generation
        slurmids = self.run_node_and_edge_generation(split_name=split_name)
        logger.info("Node and edge generation job submitted.")

        # concatenate graphs
        construct_id = self.run_graph_concatenation(
            slurmids=slurmids, split_name=split_name
        )
        logger.info("Graph concatenation job submitted.")

        return construct_id

    def post_split_jobs(self, split_name: str) -> str:
        """Submit jobs after splitting the data, from construction onwards."""
        logger.info(
            "Intermediate graph found. Checking if all other tissues are done..."
        )
        if not self._check_all_intermediates(split_name):
            logger.info(
                "Not all intermediates found. Re-running pipeline (with built in check, so edges won't be reparsed if they are done)."
            )
            return self.graph_construction_jobs(split_name)

        logger.info("All intermediates found. Concatenating graphs.")
        result = self.run_graph_concatenation(split_name=split_name, slurmids=None)
        logger.info("Graph concatenation job submitted.")

        return result

    def run_pipeline(self) -> None:
        """Run the pipeline! Check for existing files and submit jobs as needed."""
        # get split name
        split_name = _dataset_split_name(
            test_chrs=self.config.test_chrs,
            val_chrs=self.config.val_chrs,
            tpm_filter=self.args.tpm_filter,
            percent_of_samples_filter=self.args.percent_of_samples_filter,
        )
        if self.args.target == "rna_seq":
            split_name += "_rna_seq"

        # get final and intermediate graph paths
        logger.info(f"Starting process for: {split_name}")
        final_graph, intermediate_graph = self._get_file_paths(split_name=split_name)

        # decide which jobs to submit
        logger.info(f"Checking for final graph: {final_graph}")
        if not os.path.isfile(final_graph):
            logger.info(
                f"Final graph not found. \
                Checking for intermediate graph: {intermediate_graph}"
            )
            self.all_pipeline_jobs(intermediate_graph, split_name)
        else:
            logger.info(
                "Final graph found. \
                Going straight to GNN training."
            )
            self.submit_gnn_job(split_name, None)

        if self.args.clean_up:
            self.clean_up()


def run_tests() -> bool:
    """Run selected unit tests to ensure pipeline code is functioning correctly.

    Returns:
        bool: `True` if all tests pass, `False` otherwise
    """
    exit_code = pytest.main([])
    return exit_code == 0  # pytest.ExitCode.OK is 0


def calculate_trials(n_gpus: int, n_trials: int) -> int:
    """Calculate the number of trials to run for hyperparameter optimization
    (per gpu).
    """
    return n_trials // n_gpus


def main() -> None:
    """Run OGL pipeline, from data parsing to graph constructuion to GNN
    training with checks to avoid redundant computation.
    """
    args = OGLCLIParser().parse_args()

    # run OGL pipeline
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)
    pipe_runner = PipelineRunner(
        config=experiment_config,
        args=args,
    )
    pipe_runner.run_pipeline()


if __name__ == "__main__":
    main()
