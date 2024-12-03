# sourcery skip: lambdas-should-be-short
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to handle the assembly of target values for the regression task. Values
are taken from a variety of different matrices and are scaled using
StandardScaler. The class can handle both GTEx TPM data (median regression) and
rna-seq data from ENCODE (also TPM, but at the sample level and not median)."""


from copy import deepcopy
from typing import Dict, List, Tuple, Union

from cmapPy.pandasGEXpress.parse_gct import parse  # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore

from omics_graph_learning.split.gene_filter import read_encode_rna_seq_data
from omics_graph_learning.utils.common import _load_pickle
from omics_graph_learning.utils.common import time_decorator
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.config_handlers import TissueConfig


class TargetAssembler:
    """Train test splitter for genes.

    Attributes:
        experiment_config (ExperimentConfig): Configuration object for the
        experiment.
        split (Dict[str, List[str]]): Split information for genes.

    Methods
    --------
    assemble_matrix_targets:
        Assembles matrix targets based on provided split information.

    scale_targets:
        Scales targets using StandardScaler and returns the scaled targets.

    Examples:
    --------
    >>> assembler = TargetAssembler(
        experiment_config,
        split
    )

    >>> targets = assembler.assemble_matrix_targets()
    >>> targets = assembler.assemble_rna_targets()

    # To scale targets
    >>> scaled_targets = assembler.scale_targets(targets)
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        split: Dict[str, List[str]],
        pseudocount: float = 0.25,
    ):
        """Initialize the gene splitter."""
        self.experiment_config = experiment_config
        self.split = split
        self.pseudocount = pseudocount

        self.sample_config_dir = experiment_config.sample_config_dir
        self.log_transform = experiment_config.log_transform
        self.tissues = experiment_config.tissues

        # set matrices
        self.expression_median_matrix = experiment_config.expression_median_matrix
        self.protein_abundance_matrix = experiment_config.protein_abundance_matrix
        self.protein_abundance_medians = experiment_config.protein_abundance_medians

        self.average_activity = _load_pickle(experiment_config.average_activity_df)
        self.expression_median_across_all = _load_pickle(
            experiment_config.expression_median_across_all
        )

    def assemble_rna_targets(
        self, tissue_config: TissueConfig
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Assemble RNA targets based on provided split information."""
        rna_quantifications = self._get_rna_quantifications(
            rna_matrix=tissue_config.resources["rna"]
        )

        return self._assign_target_to_split(tissue_config, rna_quantifications)

    def assemble_tissue_median_targets(
        self, tissue_config: TissueConfig
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Assemble tissue median targets based on provided split information."""
        quantifications = self._get_tissue_median_quantifications(
            gtex_name=tissue_config.resources["gtex_name"]
        )

        return self._assign_target_to_split(tissue_config, quantifications)

    def _assign_target_to_split(
        self, tissue_config: TissueConfig, quantifications: Dict[str, float]
    ):
        """Assign target values to the split partitions."""
        targets = {}
        for partition in self.split:
            partition_targets = {}
            for gene in self.split[partition]:
                target = f'{gene}_{tissue_config.resources["tissue"]}'
                partition_targets[target] = np.array(
                    [self.match_quantification(gene, quantifications)]
                )
            targets[partition] = partition_targets
        return targets

    @staticmethod
    def match_quantification(
        gene: str, rna_quantifications: Dict[str, float]
    ) -> Union[float, int]:
        """Retrieve the RNA quantification for a given gene.

        If the exact gene identifier is not found, attempt to find a match by
        removing the version suffix and searching for any key that starts with
        the base gene identifier.
        """
        # base gene identifier w/ annotation number (e.g., 'ENSG00000164164.15')
        key = gene.split("_")[0]

        try:
            return rna_quantifications[key]
        except KeyError as e:
            # remove the annotation
            base_key = key.split(".")[0]

            if not (
                possible_matches := [
                    k for k in rna_quantifications if k.startswith(base_key + ".")
                ]
            ):
                print(f"Gene '{gene}' not found in rna_quantifications.")
                return -1
            matched_key = possible_matches[0]
            return rna_quantifications[matched_key]

    def _get_rna_quantifications(self, rna_matrix: str) -> Dict[str, float]:
        """Returns a dictionary of gene: log transformed + pseudocount TPM
        values for an RNA-seq quantification .tsv from ENCODE"""
        df = read_encode_rna_seq_data(rna_matrix)
        return (
            (df["TPM"] + self.pseudocount)
            .apply(
                lambda x: self._apply_log_transform(
                    pd.DataFrame([x]), transform_type=self.log_transform
                )[0][0]
            )
            .to_dict()
        )

    def _get_tissue_median_quantifications(self, gtex_name: str) -> Dict[str, float]:
        """Returns a dictionary of gene: log transformed + pseudocount TPM
        values for tissue median TPMs from GTEx."""
        df = self._load_tpm_median_df()
        return (
            (df[gtex_name] + self.pseudocount)
            .apply(
                lambda x: self._apply_log_transform(
                    pd.DataFrame([x]), transform_type=self.log_transform
                )[0][0]
            )
            .to_dict()
        )

    def assemble_matrix_targets(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Assemble target values for training by collecting the required
        datatypes and creating a dictionary of arrays holding unique target
        values.
        """
        # prepare keywords for data extraction if targets != rna_seq
        self.tissue_keywords = self._prepare_keywords()

        tissue_names = [
            tissue_names[0] for tissue_names in self.tissue_keywords.values()
        ]

        # load necessary data
        tpm_median_df = self._load_tpm_median_df()
        protein_abundance_tissue_matrix = self._load_protein_abundance_tissue_matrix(
            tissue_names
        )
        protein_abundance_median_across_all = (
            self._protein_abundance_median_across_all_tissues()
        )

        # create dataframe with difference from average activity for each tissue
        # formally, this is defined as the average expression in the tissue minus
        # the average expression in all tissues, and is an absolute value. Uses the
        # first name in the keynames tuple, which are gtex names
        diff_from_average_df, foldchange_from_average_df = (
            self._make_difference_from_average_activity_df(tissue_names=tissue_names)
        )

        # create dataframes with target medians and fold change(median in tissue /
        # median across all tissues) for TPM
        tpm_median_and_foldchange_df = self._calculate_foldchange_from_medians(
            median_matrix=tpm_median_df,
            median_across_tissues=self.expression_median_across_all,
            data_type="tpm",
        )

        # create dataframes with target medians and fold change(median in tissue /
        # median across all tissues) for prn abundance
        protein_median_and_foldchange_df = self._calculate_foldchange_from_medians(
            median_matrix=protein_abundance_tissue_matrix,
            median_across_tissues=protein_abundance_median_across_all,
            data_type="protein",
        )

        return {
            dataset: self._get_targets_per_partition(
                partition=dataset,
                tpm_median_and_foldchange_df=tpm_median_and_foldchange_df,
                diff_from_average_df=diff_from_average_df,
                foldchange_from_average_df=foldchange_from_average_df,
                protein_median_and_foldchange_df=protein_median_and_foldchange_df,
            )
            for dataset in self.split
        }

    def _prepare_keywords(self) -> Dict[str, Tuple[str, str]]:
        """Prepare keywords for data extraction from expression dataframes."""
        return {
            tissue: (
                config.resources["key_tpm"],
                config.resources["key_protein_abundance"],
            )
            for tissue in self.tissues
            for config in [
                TissueConfig.from_yaml(self.sample_config_dir / f"{tissue}.yaml")
            ]
        }

    def _load_tpm_median_df(self) -> pd.DataFrame:
        """Load the GTEx median GCT."""
        return parse(
            self.experiment_config.training_targets.expression_median_matrix
        ).data_df

    def _load_protein_abundance_tissue_matrix(
        self, tissue_names: List[str]
    ) -> pd.DataFrame:
        """Load protein abundance matrix from Jiang et al., Cell, 2020. Note *-
        these matrices are pre-processed to work with pandas, the original files
        are in .xlsx.
        """
        return self._raw_protein_abundance_matrix(
            protein_abundance_medians=self.experiment_config.protein_abundance_medians,
            # tissue_names=tissue_names,
            # graph="protein",
        )

    def _protein_abundance_median_across_all_tissues(
        self,
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame with median protein abundance for each gene
        across all samples and tissues. Values are log2, so we inverse log them,
        as we transform them ourselves later.
        """
        return (
            pd.read_csv(
                self.protein_abundance_matrix, sep=",", index_col="gene.id.full"
            )
            .drop(columns=["gene.id"])
            .apply(np.exp2)
            .fillna(0)
            .median(axis=1)
            .rename("all_tissues")
            .to_frame()
        )

    def _raw_protein_abundance_matrix(
        self,
        protein_abundance_medians: str,
        # tissue_names: List[str],
        # graph: str = "tissue",
    ) -> pd.DataFrame:
        """Returns a dataframe containing the protein abundance median for each
        gene specified in the tissue list.
        """
        # usecols = ["gene.id.full"] + tissue_names if graph == "tissue" else None
        df = pd.read_csv(
            protein_abundance_medians,
            sep=",",
            index_col="gene.id.full",
            # usecols=usecols,
        )
        # if graph != "tissue":
        df.rename(columns={x: self._tissue_rename(x) for x in df.columns}, inplace=True)
        return df.apply(np.exp2).fillna(0)

    @time_decorator(print_args=False)
    def _calculate_foldchange_from_medians(
        self,
        median_matrix: pd.DataFrame,
        median_across_tissues: pd.DataFrame,
        data_type: str = "tpm",
    ) -> pd.DataFrame:
        """Calculate fold change between median values and apply the log2
        transformation. Formally, the fold change is calculated by dividing the
        median value in a tissue by the median value across all tissues.

        Args:
            median_matrix (pd.DataFrame): DataFrame containing median values.
            median_across_tissues (pd.DataFrame): DataFrame containing median values
            across tissues.
            pseudocount (float): Pseudocount value added before log2 transformation.
            data_type (str, optional): Type of data, e.g., "tpm". Defaults to "tpm".

        Returns:
            pd.DataFrame: DataFrame with fold change values.
        """
        rename_pairs = {
            x: self._tissue_rename(tissue=x, data_type=data_type)
            for x in median_matrix.columns
        }

        df = pd.concat([median_matrix, median_across_tissues], axis=1)
        df.rename(columns=rename_pairs, inplace=True)
        df += self.pseudocount
        df[[f"{tissue}_foldchange" for tissue in df.columns]] = df[df.columns].div(
            df["all_tissues"], axis=0
        )

        return pd.DataFrame(
            self._apply_log_transform(
                df.drop(columns=["all_tissues"]), transform_type=self.log_transform
            )
        )

    def _make_difference_from_average_activity_df(self, tissue_names):
        """Calculate difference and fold change from average activity."""
        return self._combine_difference_from_average_activity_dfs(
            average_activity_df=self.average_activity,
            log_transform_type=self.log_transform,
            tissues=tissue_names,
        )

    def _tissue_difference_from_average_activity(
        self, tissue: str, average_activity_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Get fold change and difference from average for a given tissue given
        an average activity dataframe."""
        tissue_config = TissueConfig.from_yaml(
            self.sample_config_dir / f"{tissue}.yaml"
        )
        df = pd.read_table(
            tissue_config.resources["tpm"], index_col=1, header=[2]
        ).drop("id", axis=1)
        tissue_average = df.mean(axis=1)

        # difference from average activity
        difference = abs(tissue_average.subtract(average_activity_df["average"]))
        difference.name = f"{tissue}_difference_from_average"

        # fold change from average activity
        fold_change = tissue_average.divide(average_activity_df["average"])
        fold_change.name = f"{tissue}_foldchange_from_average"

        return difference, fold_change

    @time_decorator(print_args=False)
    def _combine_difference_from_average_activity_dfs(
        self,
        average_activity_df: pd.DataFrame,
        log_transform_type: str,
        tissues: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Takes the average activity of the genes in one tissue, and calculates
        the difference from the average activity of the genes in all tissues. In
        this case, the all tissue average includes our tissue of interest.

        Args:
            tpm_dir (str): _description_
            tissues (List[str]): _description_
            average_activity (pd.DataFrame): _description_

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: difference and fold change
            against the average
        """

        differences, fold_changes = zip(
            *(
                self._tissue_difference_from_average_activity(
                    tissue=tissue, average_activity_df=average_activity_df
                )
                for tissue in tissues
            )
        )

        return pd.DataFrame(
            self._apply_log_transform(
                pd.concat(differences, axis=1) + self.pseudocount,
                transform_type=log_transform_type,
            )
        ), pd.DataFrame(
            self._apply_log_transform(
                pd.concat(fold_changes, axis=1) + self.pseudocount,
                transform_type=log_transform_type,
            )
        )

    @time_decorator(print_args=False)
    def _get_targets_per_partition(
        self,
        partition: str,
        diff_from_average_df: pd.DataFrame,
        foldchange_from_average_df: pd.DataFrame,
        tpm_median_and_foldchange_df: pd.DataFrame,
        protein_median_and_foldchange_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Get target values for each tissue based on the provided dataframes
        and split information.
        """
        return {
            target: self._get_target_values(
                gene,
                self.tissue_keywords[tissue][0],
                self.tissue_keywords[tissue][1],
                tpm_median_and_foldchange_df,
                diff_from_average_df,
                foldchange_from_average_df,
                protein_median_and_foldchange_df,
            )
            for tissue in self.tissue_keywords
            for target in self.split[partition]
            if (gene := target.split("_", 1)[0]) and target.split("_", 1)[1] == tissue
        }

    @staticmethod
    def _get_target_values(
        gene: str,
        tpmkey: str,
        prokey: str,
        tpm_median_and_foldchange_df: pd.DataFrame,
        diff_from_average_df: pd.DataFrame,
        foldchange_from_average_df: pd.DataFrame,
        protein_median_and_foldchange_df: pd.DataFrame,
    ) -> np.ndarray:
        """Get target values for a specific gene based on various dataframes.

        Args:
            gene (str): The gene for which target values are calculated.
            tpmkey (str): The key for TPM values in the dataframes.
            prokey (str): The key for protein abundance values in the dataframes.
            tpm_median_and_foldchange_df (pd.DataFrame): DataFrame containing
            TPM median and foldchange values.
            diff_from_average_df (pd.DataFrame): DataFrame containing
            differences from average activity.
            foldchange_from_average_df (pd.DataFrame): DataFrame containing fold
            changes from average activity.
            protein_median_and_foldchange_df (pd.DataFrame): DataFrame
            containing protein abundance median and foldchange values.

        Returns:
            np.ndarray: An array of target values for the gene.
        """
        return np.array(
            [
                tpm_median_and_foldchange_df.loc[gene, tpmkey],
                tpm_median_and_foldchange_df.loc[gene, f"{tpmkey}_foldchange"],
                diff_from_average_df.loc[gene, f"{tpmkey}_difference_from_average"],
                foldchange_from_average_df.loc[
                    gene, f"{tpmkey}_foldchange_from_average"
                ],
                (
                    protein_median_and_foldchange_df.loc[gene, prokey]
                    if gene in protein_median_and_foldchange_df.index
                    else -1
                ),
                (
                    protein_median_and_foldchange_df.loc[gene, f"{prokey}_foldchange"]
                    if gene in protein_median_and_foldchange_df.index
                    else -1
                ),
            ]
        )

    @staticmethod
    @time_decorator(print_args=False)
    def scale_targets(
        targets: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Scale targets using StandardScaler."""
        scaled_targets = deepcopy(targets)
        num_target_types = len(next(iter(targets["train"].values())))
        target_scalers = {i: StandardScaler() for i in range(num_target_types)}

        # Fit scalers for each type of target data
        for i in range(num_target_types):
            data = np.stack(
                [
                    scaled_targets["train"][target][i]
                    for target in scaled_targets["train"]
                ]
            )
            target_scalers[i].fit(data.reshape(-1, 1))

        # Scale the targets in all splits
        for split_targets in scaled_targets.values():
            for target, values in split_targets.items():
                split_targets[target] = np.array(
                    [
                        target_scalers[i].transform(values[i].reshape(-1, 1)).flatten()
                        for i in range(num_target_types)
                    ]
                )
        return scaled_targets

    @staticmethod
    def _tissue_rename(tissue: str, data_type: str = "tpm") -> str:
        """Rename a tissue string for a given data type (TPM or protein).

        Args:
            tissue (str): The original tissue name to be renamed.
            data_type (str): The type of data (e.g., 'tpm' or 'protein').

        Returns:
            str: The renamed and standardized tissue name.

        Raises:
            ValueError if data_type is not 'tpm' or 'protein'.
        """
        if data_type not in ["tpm", "protein"]:
            raise ValueError("data_type must be either `tpm` or `protein`.")

        replacements = (
            {"- ": "", "-": "", "(": "", ")": "", " ": "_"}
            if data_type == "tpm"
            else {" ": "_"}
        )
        for old, new in replacements.items():
            tissue = tissue.replace(old, new)
        return tissue.casefold()

    @staticmethod
    def _apply_log_transform(
        data: pd.DataFrame, transform_type: str = "log2"
    ) -> np.ndarray:
        """Applies the specified log transformation to a DataFrame.

        Args:
            data (pd.DataFrame): The data to apply the transformation to.
            transform_type (str): The type of log transformation: 'log2',
            'log1p', or 'log10'.

        Returns:
            pd.DataFrame: The log-transformed data.
        """
        if transform_type == "log2":
            return np.log2(data)
        elif transform_type == "log1p":
            return np.log1p(data)
        elif transform_type == "log10":
            return np.log10(data)
        else:
            raise ValueError(
                f"Invalid log transformation type: {transform_type}. "
                "Must be `log2`, `log1p`, or `log10`."
            )
