# Separate from automated testing, this script is used to ensure that the
# training targets are accurate

"""Tests that a random subset of targets match the median values in their
respective tissues."""

import numpy as np
import pandas as pd
import pickle
import random

from cmapPy.pandasGEXpress.parse_gct import parse

TISSUES = {
    "aorta": "Artery - Aorta",
    "hippocampus": "Brain - Hippocampus",
    "left_ventricle": "Heart - Left Ventricle",
    "liver": "Liver",
    "lung": "Lung",
    "mammary": "Breast - Mammary Tissue",
    "pancreas": "Pancreas",
    "skeletal_muscle": "Muscle - Skeletal",
    "skin": "Skin - Sun Exposed (Lower leg)",
    "small_intestine": "Small Intestine - Terminal Ileum",
}


def _tissue_rename(
    tissue: str,
    data_type: str,
) -> str:
    """Rename a tissue string for a given data type (TPM or protein).

    Args:
        tissue (str): The original tissue name to be renamed.
        data_type (str): The type of data (e.g., 'tpm' or 'protein').

    Returns:
        str: The renamed and standardized tissue name.
    """
    if data_type == "tpm":
        regex = (("- ", ""), ("-", ""), ("(", ""), (")", ""), (" ", "_"))
    else:
        regex = ((" ", "_"), ("", ""))

    tissue_rename = tissue.casefold()
    for r in regex:
        tissue_rename = tissue_rename.replace(*r)

    return tissue_rename


def test_tpm_median_values(
    targets,
    true_df,
):
    for split in targets:
        test_targets = random.sample(list(targets[split]), 100)
        for entry in test_targets:
            target, tissue = entry.split("_", 1)

            true_median = true_df.loc[target, TISSUES[tissue]]
            # print(f"log2 true median plus 0.25: {np.log2(true_median + 0.25)}")
            # print(targets[split][entry][0])
            assert np.isclose(np.log2(true_median + 0.25), targets[split][entry][0])


def test_tpm_foldchange_values(
    targets,
    true_df,
    median_across_all,
):
    for split in targets:
        test_targets = random.sample(list(targets[split]), 100)
        for entry in test_targets:
            target, tissue = entry.split("_", 1)

            true_median = true_df.loc[target, TISSUES[tissue]]
            true_all_median = median_across_all.loc[target, "all_tissues"]
            true_fold = np.log2((true_median + 0.25) / (true_all_median + 0.25))

            # print(f"true fold: {true_fold}")
            # print(f"target fold: {targets[split][entry][1]}")
            assert np.isclose(true_fold, targets[split][entry][1])


def test_difference_from_average_activity(
    targets,
    true_df,
):
    for split in targets:
        test_targets = random.sample(list(targets[split]), 100)
        for entry in test_targets:
            target, tissue = entry.split("_", 1)

            tissue_rename = _tissue_rename(tissue=TISSUES[tissue], data_type="tpm")
            true_average_diff = true_df.loc[
                target, f"{tissue_rename}_difference_from_average"
            ]

            assert np.isclose(true_average_diff, targets[split][entry][2])


def run_test():
    root_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess"
    shared_data_dir = f"{root_dir}/shared_data"

    # load df with true median values
    true_df = parse(
        f"{shared_data_dir}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    ).data_df

    # load all tissue medians
    median_across_all = pd.read_pickle(
        f"{shared_data_dir}/gtex_tpm_median_across_all_tissues.pkl"
    )

    # load df containing difference from average activity
    average_activity = pd.read_pickle(
        f"{shared_data_dir}/average_differences_all_tissues_log2.pkl"
    )

    targets_dir = f"{root_dir}/graph_processing"
    for targets in [
        "targets_random_assign.pkl",
        "targets_test_chr1.pkl",
        "targets_test_chr8-chr9_val_chr11-chr13.pkl",
    ]:
        with open(targets_dir + "/" + targets, "rb") as f:
            # open targets
            targets = pickle.load(f)

        # run tests!
        # testing tpm medians
        test_tpm_median_values(
            targets=targets,
            true_df=true_df,
        )

        # testing median fold-change
        test_tpm_foldchange_values(
            targets=targets,
            true_df=true_df,
            median_across_all=median_across_all,
        )

        # testing difference from average activity
        test_difference_from_average_activity(
            targets=targets,
            true_df=average_activity,
        )


if __name__ == "__main__":
    run_test()