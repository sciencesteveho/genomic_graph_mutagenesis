#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Utilities for omics graph learning modules."""


import argparse
from collections import defaultdict
from contextlib import suppress
import csv
from datetime import timedelta
import functools
import inspect
import logging
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import psutil  # type: ignore
from pybedtools import BedTool  # type: ignore
from scipy import stats  # type: ignore
from scipy.stats import spearmanr  # type: ignore
import seaborn as sns  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
import yaml  # type: ignore

from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.config_handlers import TissueConfig


# decorator to track execution time
def time_decorator(print_args: bool = False, display_arg: str = "") -> Callable:
    """Decorator to time functions.

    Args:
        print_args (bool, optional): Whether to print the function arguments.
        Defaults to False. display_arg (str, optional): The argument to display
        in the print statement. Defaults to "".

    Returns:
        Callable: The decorated function.
    """

    def _time_decorator_func(function: Callable) -> Callable:
        @functools.wraps(function)
        def _execute(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            fxn_args = inspect.signature(function).bind(*args, **kwargs).arguments
            result = function(*args, **kwargs)
            end_time = time.monotonic()
            args_to_print = list(fxn_args.values()) if print_args else display_arg
            print(
                f"Finished {function.__name__} {args_to_print} - \
                Time: {timedelta(seconds=end_time - start_time)}"
            )
            return result

        return _execute

    return _time_decorator_func


# logging setup
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Prepare a logger that prints to stderr with date, time, and module name.
    Optionally writes to a file if a filepath is specified.

    Args:
        log_file (Optional[str]): Path to the log file. If None, logging only
        occurs to stderr.
    """
    # get name of module calling logger
    script_name = os.path.basename(sys.argv[0])
    module_name = os.path.splitext(script_name)[0]

    # create logger
    logger = logging.getLogger(module_name)

    # clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # set date, time, and module name format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create file handler if additional log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# file and directory operations
def dir_check_make(dir: Union[str, Path]) -> None:
    """Utility to make directories only if they do not already exist"""
    with suppress(FileExistsError):
        os.makedirs(dir)


def check_and_symlink(src: Union[str, Path], dst: Path, boolean: bool = False) -> None:
    """Create a symlink from src to dst if it doesn't exist. If boolean is True,
    also check that src exists before creating the symlink.
    """
    if boolean and not os.path.exists(src):
        return
    if not os.path.exists(dst):
        with suppress(FileExistsError):
            os.symlink(src, dst)


def _get_files_in_directory(dir: Path) -> List[str]:
    """Return a list of files within the directory."""
    return [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]


def _add_hash_if_missing(file: str) -> None:
    """Add a hash to the first line of a file if it is missing. This is for
    reading in a file as a bedfile to pybedtools.
    """
    with open(file, "r") as f:
        lines = f.readlines()

    # add hash if missing
    if not lines[0].startswith("#"):
        lines[0] = "#" + lines[0]

    # write back to file
    with open(file, "w") as f:
        f.writelines(lines)


# data loading and saving
def _load_pickle(file_path: Union[str, Path]) -> Any:
    """Wrapper to load a pkl"""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def _save_pickle(data: object, file_path: Union[str, Path]) -> None:
    """Wrapper to save a pkl"""
    with open(file_path, "wb") as output:
        pickle.dump(data, output)


def parse_yaml(config_file: str) -> Dict[str, Any]:
    """Load yaml for parsing"""
    with open(config_file, "r") as stream:
        return yaml.safe_load(stream)


# command line operations
def _run_command(command: str, get_output: bool = False) -> Optional[str]:
    """Run a shell command and optionally return its output."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
            shell=True,
        )
        if get_output:
            return result.stdout.strip()
        else:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running command: {command}")
        print(e.output)
        sys.exit(1)
    return None


def _chk_file_and_run(file: str, cmd: str) -> None:
    """Check that a file does not exist before calling subprocess"""
    if not os.path.isfile(file) or os.path.getsize(file) == 0:
        subprocess.run(cmd, stdout=None, shell=True)


def get_remaining_walltime(logger: logging.Logger) -> Union[int, None]:
    """Get remaining walltime in seconds."""
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", "$SLURM_JOB_ID"], capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "TimeLeft=" in line:
                time_left = line.split("TimeLeft=")[1].split()[0]
                days, time = (
                    time_left.split("-") if "-" in time_left else ("0", time_left)
                )
                hours, mins, secs = time.split(":")
                return (
                    int(days) * 86400 + int(hours) * 3600 + int(mins) * 60 + int(secs)
                )
    except Exception as e:
        logger.warning(f"Failed to get remaining walltime: {e}")
    return None


# slurm operations
def submit_slurm_job(job_script: str, args: str, dependency: Optional[str]) -> str:
    """Submits a SLURM job and returns its job ID."""
    dependency_arg = f"--dependency=afterok:{dependency}" if dependency else ""
    command = f"sbatch {dependency_arg} {job_script} {args}"
    job_id = _run_command(command, get_output=True)
    assert job_id is not None
    return job_id.split()[-1]  # extract job ID


# system operations
def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    return psutil.cpu_count(logical=False) - 1


def check_cuda_env() -> None:
    """Set cuda blocking to debug potential errors across GPUs"""
    cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING")
    torch_use_cuda_dsa = os.environ.get("TORCH_USE_CUDA_DSA")

    print(f"CUDA_LAUNCH_BLOCKING is set to: {cuda_launch_blocking}")
    print(f"TORCH_USE_CUDA_DSA is set to: {torch_use_cuda_dsa}")

    if cuda_launch_blocking != "1" or torch_use_cuda_dsa != "1":
        print("Warning: CUDA debugging environment variables are not set to '1'.")
        print("This may affect error reporting and debugging capabilities.")


# statistics tests
def calculate_spearman_r(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate the Spearman correlation coefficient from GNN output."""
    r, _ = spearmanr(predictions, targets)
    return float(r)


# graph data and gnn operations
class NumpyGraphChecker:
    """Class to check and print statistics for concatenated graph data in numpy
    format.
    """

    def __init__(self, graph_data: Dict[str, Any]):
        self.graph_data = graph_data

    def print_array_stats(self, array: np.ndarray, name: str) -> None:
        """Prints statistics for a numpy array."""
        print(f"\n{name} statistics:")
        print(f"Mean: {array.mean(axis=0)[:5]}...")
        print(f"Std: {array.std(axis=0)[:5]}...")
        print(f"Min: {array.min(axis=0)[:5]}...")
        print(f"Max: {array.max(axis=0)[:5]}...")
        print(f"Contains NaN: {np.isnan(array).any()}")
        print(f"Contains Inf: {np.isinf(array).any()}")

    def print_graph_summary(self) -> None:
        """Prints a summary of the graph structure."""
        print("Numpy Graph Summary:")
        print(f"Number of nodes: {self.graph_data['num_nodes']}")
        print(f"Number of edges: {self.graph_data['num_edges']}")
        print(f"Average degree: {self.graph_data['avg_edges']:.2f}")

    def print_attribute_shapes(self) -> None:
        """Prints the shapes of all attributes in the graph data."""
        print("\nAttribute shapes:")
        for key, item in self.graph_data.items():
            if isinstance(item, np.ndarray):
                print(f"{key}: {item.shape}")

    def print_edge_index_stats(self) -> None:
        """Prints statistics about the edge index."""
        edge_index = self.graph_data["edge_index"]
        print("\nEdge index statistics:")
        print(f"Min node idx: {edge_index.min()}")
        print(f"Max node idx: {edge_index.max()}")

    def check_edge_index_bounds(self) -> None:
        """Checks if edge_index contains out-of-bounds indices."""
        if self.graph_data["edge_index"].max() >= self.graph_data["num_nodes"]:
            print("\nWARNING: edge_index contains out-of-bounds indices!")

    def check_node_feature_normalization(self) -> None:
        """Checks if node features appear to be normalized."""
        node_feat = self.graph_data["node_feat"]
        if node_feat.min() >= 0 and node_feat.max() <= 1:
            print("\nNode features appear to be normalized (min >= 0, max <= 1)")
        else:
            print("\nWARNING: Node features may not be normalized")

    def check_positional_encoding(self) -> None:
        """Checks in NaN or INF in positional encoding"""
        encodings = self.graph_data["node_positional_encoding"]
        inf_mask = np.isinf(encodings)
        inf_count = np.sum(inf_mask)

        # Check for NaN
        nan_mask = np.isnan(encodings)
        nan_count = np.sum(nan_mask)

        print(f"INF values found: {inf_count}")
        print(f"NaN values found: {nan_count}")

        if inf_count > 0:
            print("Positions of INF values:")
            print(np.where(inf_mask))

        if nan_count > 0:
            print("Positions of NaN values:")
            print(np.where(nan_mask))

    def run_numpy_graph_checks(self) -> None:
        """Runs all checks and prints all statistics for the numpy graph data."""
        self.print_graph_summary()
        self.print_attribute_shapes()
        self.print_array_stats(self.graph_data["node_feat"], "Node features")
        self.check_positional_encoding()
        self.print_edge_index_stats()
        self.check_edge_index_bounds()
        self.check_node_feature_normalization()

    @staticmethod
    def check_numpy_graph_data(graph_data: Dict[str, Any]) -> None:
        """Check the numpy graph data."""
        checker = NumpyGraphChecker(graph_data)
        checker.run_numpy_graph_checks()


class PyGDataChecker:
    """Class to check and print statistics for custom PyG Data objects."""

    def __init__(self, data: Data):
        self.data = data

    def print_tensor_stats(self, tensor: torch.Tensor, name: str) -> None:
        """Prints statistics for a tensor."""
        print(f"\n{name} statistics:")
        print(f"Mean: {tensor.mean(dim=0)[:5]}...")
        print(f"Std: {tensor.std(dim=0)[:5]}...")
        print(f"Min: {tensor.min(dim=0)[0][:5]}...")
        print(f"Max: {tensor.max(dim=0)[0][:5]}...")
        if torch.isnan(tensor).any().item():
            print(f"Contains NaN: {torch.isnan(tensor).any().item()}")
        if torch.isinf(tensor).any().item():
            print(f"Contains Inf: {torch.isinf(tensor).any().item()}")

    def print_graph_summary(self) -> None:
        """Prints a summary of the graph structure."""
        print("PyG Data Object - Graph Summary:")
        print(f"Number of nodes: {self.data.num_nodes}")
        print(f"Number of edges: {self.data.num_edges}")
        print(f"Number of node features: {self.data.num_node_features}")
        print(f"Number of edge features: {self.data.num_edge_features}")
        print(f"Has isolated nodes: {self.data.has_isolated_nodes()}")
        print(f"Has self loops: {self.data.has_self_loops()}")
        print(f"Is directed: {self.data.is_directed()}")

    def print_mask_info(self) -> None:
        """Prints information about the masks and corresponding target variables."""
        print("\nMask information:")
        for key in [
            "train_mask",
            "val_mask",
            "test_mask",
            "train_mask_loss",
            "val_mask_loss",
            "test_mask_loss",
        ]:
            if not hasattr(self.data, key):
                return
            mask = getattr(self.data, key)
            print(f"{key}: {mask.sum().item()} nodes")
            if hasattr(self.data, "y"):
                self._print_mask_info(mask)

    def _print_mask_info(self, mask: torch.Tensor) -> None:
        """Prints information about the masks and corresponding target variables."""
        masked_y = self.data.y[mask]
        print(f" Mean: {masked_y.mean().item():.4f}")
        print(f" Std: {masked_y.std().item():.4f}")
        print(f" Min: {masked_y.min().item():.4f}")
        print(f" Max: {masked_y.max().item():.4f}")
        if torch.isnan(masked_y).any().item():
            print(f" Contains NaN: {torch.isnan(masked_y).any().item()}")
        if torch.isinf(masked_y).any().item():
            print(f" Contains Inf: {torch.isinf(masked_y).any().item()}")

    def run_pyg_data_check(self) -> None:
        """Runs all checks and prints all statistics for the PyG Data object."""
        self.print_graph_summary()
        self.print_tensor_stats(self.data.x, "Node features")
        if hasattr(self.data, "y"):
            self.print_tensor_stats(self.data.y, "Target variable")
        self.print_mask_info()

    @staticmethod
    def check_pyg_data(data: Data) -> None:
        """Check the PyG Data object."""
        checker = PyGDataChecker(data)
        checker.run_pyg_data_check()


def tensor_out_to_array(tensor: torch.Tensor) -> np.ndarray:
    """Convert masked out tensor to numpy array"""
    return tensor.cpu().numpy()


def ensure_mask_fidelity(x: torch.Tensor, regression_mask: torch.Tensor) -> None:
    """Run a series of checks to ensure that the regression mask is valid."""
    if regression_mask.sum() == 0:
        raise ValueError(
            "Regression mask is empty. No targets specified for regression."
        )

    if regression_mask.dtype != torch.bool:
        raise TypeError("Regression mask must be a boolean tensor.")

    if regression_mask.shape != (x.shape[0],):
        raise ValueError(
            f"Regression mask shape {regression_mask.shape} "
            f"does not match input shape {x.shape[0]}"
        )


def _calculate_max_distance_base_graph(bed: List[List[str]]) -> Set[int]:
    """Calculate the max distance between nodes in the base graph. Report the
    max, mean, and median distances for all interaction type data.
    """
    return {
        max(int(line[2]), int(line[6])) - min(int(line[3]), int(line[7]))
        for line in bed
    }


def _combine_and_sort_arrays(edge_index: np.ndarray) -> np.ndarray:
    """Combines stored edge index and returns dedupe'd array of nodes"""
    combined = np.concatenate((edge_index[0], edge_index[1]))
    return np.unique(combined)


def save_error_state(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    error: RuntimeError,
) -> None:
    """Utility to save the model state to troubleshoot CUDA runtime errors."""
    state = {"model_state": model.state_dict(), "inputs": inputs, "error": str(error)}
    torch.save(state, f"error_state_{model.name}.pt")


def count_model_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# genome data utils
class ScalerUtils:
    """Utility class for scaling node features, as the modules for scaling share
    most of the same args"""

    def __init__(self) -> None:
        """Initialize the class."""
        args = self._parse_args()
        self.experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
        self.split_name = args.split_name
        self.onehot_node_feats = self.experiment_config.onehot_node_feats
        self.file_prefix = f"{self.experiment_config.experiment_name}_{self.experiment_config.graph_type}"

        # directories
        self.graph_dir = self.experiment_config.graph_dir
        self.split_dir = self.graph_dir / self.split_name
        self.scaler_dir = self.split_dir / "scalers"

        # files
        self.idxs = self.split_dir / f"{self.file_prefix}_graph_idxs.pkl"
        self.graph = self.split_dir / f"{self.file_prefix}_graph.pkl"
        self.split = self.split_dir / "training_split_combined.pkl"

    def load_from_file(self, file_path: Union[Path, str]) -> Dict[str, Any]:
        """Load data from a file."""
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def load_idxs(self) -> Dict[str, int]:
        """Load indexes from file."""
        return self.load_from_file(self.idxs)

    def load_graph(self) -> Dict[str, Any]:
        """Load graph from file."""
        return self.load_from_file(self.graph)

    def load_split(self) -> Dict[str, Any]:
        """Load split from file."""
        return self.load_from_file(self.split)

    @staticmethod
    def _parse_args():
        """Get arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--experiment_config",
            type=str,
            help="Path to .yaml file with experimental conditions",
        )
        parser.add_argument("--split_name", type=str)
        return parser.parse_args()


def _get_chromatin_loop_file(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> str:
    """Returns the specified loop file. Loop files either end in loops.bedpe or
    contacts.bedpe. There is no redundancy in the naming, so it looks for one
    and if it doesn't find it, it looks for the other.
    """
    if "_" in experiment_config.baseloops:
        method, resolution = experiment_config.baseloops.rsplit("_", 1)
        prefix = f"{experiment_config.baseloop_dir}/{method}/{resolution}/{tissue_config.resources['tissue']}"
    else:
        prefix = f"{experiment_config.baseloop_dir}/{experiment_config.baseloops}/{tissue_config.resources['tissue']}"
    suffixes = ["_loops.bedpe", "_contacts.bedpe"]
    for suffix in suffixes:
        file = f"{prefix}{suffix}"
        if os.path.isfile(file):
            return file
    raise FileNotFoundError(
        f"Loop file not found for {tissue_config.resources['tissue']}"
    )


def _dataset_split_name(
    test_chrs: Optional[List[str]] = None,
    val_chrs: Optional[List[str]] = None,
    tpm_filter: Union[float, int] = 0.1,
    percent_of_samples_filter: float = 0.2,
) -> str:
    """Constructs a name for the dataset based on the chromosome split and gene
    filter arguments.
    """
    chrs = []

    if test_chrs and val_chrs:
        chrs.append(
            f"test_{('-').join(map(str, test_chrs))}_val_{('-').join(map(str, val_chrs))}"
        )
    elif test_chrs:
        chrs.append(f"test_{('-').join(map(str, test_chrs))}")
    elif val_chrs:
        chrs.append(f"val_{('-').join(map(str, val_chrs))}")
    else:
        chrs.append("random_assign")

    return f"tpm_{tpm_filter}_samples_{percent_of_samples_filter}_{''.join(chrs).replace('chr', '')}"


def chunk_genes(
    genes: List[str],
    chunks: int,
) -> Dict[int, List[str]]:
    """Constructs graphs in parallel"""
    for _ in range(5):
        random.shuffle(genes)

    split = lambda l, chunks: [l[n : n + chunks] for n in range(0, len(l), chunks)]
    split_genes = split(genes, chunks)
    return dict(enumerate(split_genes))


def filtered_genes_from_bed(tpm_filtered_genes: str) -> List[str]:
    """Extracts the gene names from a filtered genes BED file."""
    with open(tpm_filtered_genes, newline="") as file:
        return [line[3] for line in csv.reader(file, delimiter="\t")]


def genes_from_gencode(gencode_ref: BedTool) -> Dict[str, str]:
    """Returns a dict of gencode v26 genes, their ids and associated gene
    symbols
    """
    return {
        line[9].split(";")[3].split('"')[1]: line[3]
        for line in gencode_ref
        if line[0] not in ["chrX", "chrY", "chrM"]
    }


def _reftss_cut_cols(file: str) -> None:
    """Cuts the first and eighth columns of the refTSS annotation file to
    produce a file with only the TSS and gene symbol"""
    cmd = f"cut -f1,8 {file} > {file}.cut"
    subprocess.run(cmd, stdout=None, shell=True)


def _tss_to_gene_tuples(file: str) -> List[Tuple[str, str]]:
    """Read refTSS annotation file and return a list of tuples matching TSS to
    each gene symbol."""
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        return [(row[0], col) for row in reader for col in row[1].split(" ") if col]


def _tss_tuples_to_dict(
    tss_tuples: List[Tuple[str, str]],
    genesymbol_to_gencode: Dict[str, str],
) -> Dict[str, List[str]]:
    """Map tuples to a dictionary with the first element as the key"""
    key_mappings = defaultdict(list)
    for key, value in tss_tuples:
        if value in genesymbol_to_gencode:
            key_mappings[key].append(genesymbol_to_gencode[value])
    return key_mappings


def prepare_tss_file(
    tss_file: str, annotation_file: str, gencode_ref: str, savedir: str
) -> None:
    """Prepares parsed TSS file, where each TSS is linked with its gencode gene.
    If multiple genes are associated with the TSS, the tss is split across
    multiple lines.

    Args:
        tss_file (str): The path to the TSS file.
        annotation_file (str): The path to the annotation file.
        gencode_ref (str): The path to the GENCODE reference file.
        savedir (str): The directory to save the output file.

    Returns:
        None

    Examples:
    >>> prepare_tss_file(
            tss_file="refTSS_v4.1_human_coordinate.hg38.bed.txt",
            annotation_file="refTSS_v4.1_human_hg38_annotation.txt",
            gencode_ref="gencode_v26_genes_only_with_GTEx_targets.bed",
            savedir="/reftss",
        )
    """
    _reftss_cut_cols(annotation_file)
    genesymbol_to_gencode = genes_from_gencode(BedTool(gencode_ref))
    tss = BedTool(tss_file)
    maps = _tss_tuples_to_dict(
        _tss_to_gene_tuples(f"{annotation_file}.cut"),
        genesymbol_to_gencode=genesymbol_to_gencode,
    )

    bed: List[List[str]] = []
    for line in tss:
        if line[3] in maps:
            bed.extend(
                [line[0], line[1], line[2], f"tss_{line[3]}_{value}"]
                for value in maps[line[3]]
            )
        else:
            bed.append([line[0], line[1], line[2], f"tss_{line[3]}"])

    bed = BedTool(bed).saveas(f"{savedir}/tss_parsed_hg38.bed")


def _map_genesymbol_to_tss(
    tss_path: str, annotation_path: str
) -> List[tuple[str, str, str, str]]:
    """_summary_

    Args:
        tss_path (str): _description_
        annotation_path (str): _description_

    Returns:
        List[str]: _description_
    """
    genesymbols = {
        line[0]: line[7] for line in csv.reader(open(annotation_path), delimiter="\t")
    }

    return [
        (line[0], line[1], line[2], f"tss_{line[3]}_{genesymbols[line[3]]}")
        for line in csv.reader(open(tss_path), delimiter="\t")
    ]


def _convert_coessential_to_gencode(
    coessential: str,
    genesymbol_to_gencode: Dict[str, str],
):
    """_summary_

    Args:
        dir (str): _description_ coessential (str): _description_ gencode_ref
        (str): _description_
    """
    return [
        (
            genesymbol_to_gencode[line[0]],
            genesymbol_to_gencode[line[1]],
            line[2],
        )
        for line in csv.reader(
            open(coessential, newline="", encoding="utf-8-sig"), delimiter="\t"
        )
        if line[0] in genesymbol_to_gencode and line[1] in genesymbol_to_gencode
    ]


def gene_list_from_graphs(root_dir: str, tissue: str) -> List[str]:
    """Returns a list of genes with constructed graphs, avoiding genes that may
    not have edges in smaller window"""
    directory = f"{root_dir}/{tissue}/parsing/graphs"
    return [gene.split("_")[0] for gene in os.listdir(directory)]


# optuna utils
def custom_sort_pearson(val: float) -> float:
    """Custom sort to move inf values to bottom."""
    return -float("inf") if np.isinf(val) else val
