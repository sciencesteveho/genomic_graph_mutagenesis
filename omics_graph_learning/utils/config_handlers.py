# sourcery skip: snake-case-variable-declarations
#! /usr/bin/env python


"""Class to handle stored and shared data from omics graph learning configs."""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml  # type: ignore


def load_yaml(yaml_file: Union[Path, str]) -> Dict[str, Any]:
    """Load a YAML file and return the contents as a dictionary."""
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)


@dataclass
class ExperimentConfig:
    """Class representing the configuration for an experiment.

    Arguments:
        yaml_file: Path to the YAML file containing the experiment
        configuration.

    Methods
    --------
    from_yaml(yaml_file: Path) -> ExperimentConfig:
        Loads the configuration YAML, resolves directories, updates param
        values, and ensures required lists exist as lists.

    Returns:
        ExperimentConfig: An instance of ExperimentConfig initialized with the
        configuration loaded from the YAML file.

    Examples:
    --------
    >>> from config_handlers import ExperimentConfig
    >>> config = ExperimentConfig.from_yaml("k562_combined_contacts.yaml")
    """

    valid_log_transforms = {"log2", "log1p", "log10"}

    # Params
    attribute_references: Dict[str, str]
    baseloops: str
    blacklist: str
    chromfile: str
    config_dir: Path
    differentiate_tf: bool
    experiment_name: str
    fasta: str
    feat_window: int
    gene_gene: bool
    graph_type: str
    interaction_types: List[str]
    k_fold: int
    liftover: str
    liftover_chain: str
    log_transform: str
    nodes: List[str]
    rbp_network: str
    regulatory_schema: str
    root_dir: Path
    tissues: List[str]

    # Params from positional_encoding
    build_positional_encoding: bool
    train_positional_encoding: bool

    # Params from directories dict
    baseloop_dir: Path
    expression_dir: Path
    interaction_dir: Path
    local_data_dir: Path
    matrix_dir: Path
    raw_data_dir: Path
    reference_dir: Path
    regulatory_dir: Path
    shared_data_dir: Path
    target_dir: Path
    tpm_dir: Path

    # Params from training_targets dict
    average_activity_df: str
    expression_median_across_all: str
    expression_median_matrix: str
    expression_all_matrix: str
    gencode_gtf: str
    protein_abundance_matrix: str
    protein_abundance_medians: str
    test_chrs: List[str]
    val_chrs: List[str]

    # Params from instantiation
    working_directory: Path
    graph_dir: Path
    sample_config_dir: Path

    @classmethod
    def from_yaml(cls, yaml_file: Union[Path, str]) -> "ExperimentConfig":
        """Load the configuration from a yaml and set all configs."""

        # load the yaml file
        params = load_yaml(yaml_file)

        # update the params with the params hidden in dictionaries
        cls._unpack_dictionary_params(params=params, key="training_targets")
        cls._unpack_dictionary_params(params=params, key="positional_encoding")
        cls._add_attribute_references(params=params, key="attribute_references")
        cls._resolve_directories(params=params, root_dir=params["root_dir"])
        cls._ensure_lists(
            params=params,
            keys=["interaction_types", "nodes", "test_chrs", "val_chrs", "tissues"],
        )

        # update str dirs to Path objects
        params["root_dir"] = Path(params["root_dir"])
        params["config_dir"] = Path(params["config_dir"])

        # add the default node types
        cls._update_node_types(params=params)

        # add other common directories
        params["working_directory"] = (
            params["root_dir"] / "experiments" / params["experiment_name"]
        )
        params["graph_dir"] = params["working_directory"] / "graphs"
        params["sample_config_dir"] = params["config_dir"] / "samples"

        # validate log_transform
        params["log_transform"] = cls.validate_log_transform(params["log_transform"])

        return cls(**params)

    @classmethod
    def validate_log_transform(cls, value: str) -> str:
        """Check if the log_transform value is valid"""
        if value not in cls.valid_log_transforms:
            raise ValueError(
                f"log_transform value must be one of {cls.valid_log_transforms}, got '{value}'."
            )
        return value

    @staticmethod
    def _unpack_dictionary_params(params: dict, key: str) -> None:
        training_targets = params.pop(key, {})
        for key, value in training_targets.items():
            params[key] = value

    @staticmethod
    def _add_attribute_references(params: dict, key: str) -> None:
        """Add attribute references by prepending the reference directory to the
        file paths. Additionally, add the regulatory elements attr based on the
        regulatory schema."""
        ref_dir = params[key].pop("ref_dir")
        for subkey, value in params[key].items():
            params[key][subkey] = f"{ref_dir}/{value}"

        # add regulatory ref
        reg_attr_file = (
            f"regulatory_elements_{params['regulatory_schema']}_node_attr.bed"
        )
        params[key]["regulatory_elements"] = f"{ref_dir}/{reg_attr_file}"

    @staticmethod
    def _resolve_directories(params: dict, root_dir: str) -> None:
        """Sets up all required directories as resolved absolute paths, based on a given root directory."""
        root_path = Path(root_dir).resolve()
        directories = params.pop("derived_directories")
        for key, relative_path in directories.items():
            absolute_path = root_path / relative_path
            params[key] = absolute_path.resolve()

    @staticmethod
    def _ensure_lists(params: dict, keys: List[str]) -> None:
        """Ensure the given keys have a list value."""
        for key in keys:
            if params.get(key) is None:
                params[key] = []

    @staticmethod
    def _update_node_types(params: Dict[str, Any]) -> None:
        """Update the node_types by adding the values from the YAML file to the default node_types."""
        default_node_types = ["dyadic", "enhancer", "gencode", "promoter"]
        yaml_node_types = params.get("nodes", [])
        params["nodes"] = list(set(yaml_node_types + default_node_types))


@dataclass
class TissueConfig:
    """Class representing the configuration for an experiment.

    Arguments:
        Lorem: Ipsum

    Methods
    --------

    Returns:
        Lorem: Ipsum

    Examples:
    --------
    >>> from config_handlers import TissueConfig
    >>> config = TissueConfig.from_yaml("left_ventricle.yaml")
    """

    dirs: Dict[str, str]
    features: Dict[str, str]
    interaction: Dict[str, str]
    local: Dict[str, str]
    methylation: Dict[str, Any]
    resources: Dict[str, str]
    tissue_specific_nodes: Dict[str, str]

    @classmethod
    def from_yaml(cls, yaml_file: Union[Path, str]) -> "TissueConfig":
        """Load the configuration from a yaml and set all configs."""

        # load the yaml file
        params = load_yaml(yaml_file)
        return cls(**params)

    @staticmethod
    def unpack_nested_dict(nested: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """A helper function to unpack a nested dictionary into a flat dictionary."""
        unpacked = {}
        for key, value in nested.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    unpacked[f"{subkey}"] = subvalue
            else:
                unpacked[key] = value
        return unpacked
