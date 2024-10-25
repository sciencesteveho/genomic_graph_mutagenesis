#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Baseline predictors to compare against the performance of the GNN model.

Cell line baseline:
    Expression values for cell lines across replicates are log2 transformed
    (+0.25 pseudocount). We then take the average (mean) log2 TPM of each gene.
    As a baseline, we use this average to predict the expression for our K562
    regression targets.

Below is DEPRECATED:
The first predictor is based on a random assignment of activity, from a
distribution of expression activity across all tissues in GTEx. The second is
based on the average activity of that specific gene across all samples adapted
from Schrieber et al., Genome Biology, 2020."""


import os
import pickle
from typing import List, Tuple

from cmapPy.pandasGEXpress.parse_gct import parse  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error  # type: ignore


def _tpm_all_tissue_average(
    expression_gct: str,
) -> pd.DataFrame:
    """Get the average (not median!) expression for each gene in GTEx across all
    samples. Formally, the average activity is the summed expression at each
    gene across all samples, divided by the total number of samples.

    Args:
        expression_gct (str): path to GTEx expression .gct

    Returns:
        np.ndarray: array with average activity for each gene
    """
    df = parse(expression_gct).data_df
    sample_count = df.astype(bool).sum(axis=1)
    summed_activity = pd.Series(df.sum(axis=1), name="all_tissues").to_frame()
    summed_activity["average"] = (
        summed_activity.div(sample_count, axis=0).fillna(0).values
    )
    summed_activity.to_pickle("average_activity_df.pkl")
    return summed_activity["average"]


def _tpm_dataset_specific_tissue_average(
    tpm_dir: str,
) -> pd.DataFrame:
    """_summary_

    Args:
        tpm_dir (str): /path/to/ tissue split TPM tables

    Returns:
        pd.DataFrame
    """
    dfs = [
        pd.read_table(file, index_col=0, header=[2])
        for file in os.listdir(tpm_dir)
        if "tpm.txt" in file
    ]
    df = pd.concat(dfs, axis=1)
    samples = len(df.columns)
    summed = df.sum(axis=1).to_frame()
    summed["average"] = summed.div(samples)
    summed.to_pickle("average_activity_tissues_df.pkl")
    return summed["average"]


def _difference_from_average_activity_per_tissue(
    tpm_dir: str,
    average_activity: pd.DataFrame,
):
    """Lorem"""
    dfs = []
    for file in os.listdir(tpm_dir):
        if "tpm.txt" in file:
            df = pd.read_table(file, index_col=0, header=[2])
            samples = len(df.columns)
            tissue_average = df.sum(axis=1).div(samples)
            difference = tissue_average.subtract(average_activity["average"]).abs()
            difference.name = f'{file.split(".tpm.txt")[0]}_difference_from_average'
            dfs.append(difference)
    return pd.concat(dfs, axis=1)


def _get_targets(
    split: str,
    targets: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        list(targets[split].keys()),
        [x[0] for x in np.array(list(targets[split].values()))],
    )


def _avg_activity_baseline_predictions(labels, s):
    return [s[label.split("_")[0]] for label in labels]


def main(
    expression_gct: str,
) -> None:
    """Main function"""

    # get predictions as the average tpm for that gene across all tissues
    average_activity = _tpm_all_tissue_average(expression_gct)
    with open("average_activity_before_transform.pkl", "wb") as f:
        pickle.dump(average_activity, f)

    # with open("average_activity_before_transform.pkl", "rb") as f:
    #     average_activity = pickle.load(f)

    tissue_average = _tpm_dataset_specific_tissue_average(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/baseline"
    )
    tissue_pred = np.log2(tissue_average + 0.25)  # add 0.01 to avoid log(0)
    tissue_pred_dict = tissue_pred.to_dict()

    # y_pred = np.log2(average_activity + 0.25)  # add 0.01 to avoid log(0)
    # s = y_pred.to_dict()

    with open(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/training_targets.pkl",
        "rb",
    ) as f:
        targets = pickle.load(f)

    train_labels, train_true = _get_targets("train", targets)
    test_labels, test_true = _get_targets("test", targets)
    val_labels, val_true = _get_targets("validation", targets)

    # train_preds = _avg_activity_baseline_predictions(train_labels, s)
    # test_preds = _avg_activity_baseline_predictions(test_labels, s)
    # val_preds = _avg_activity_baseline_predictions(val_labels, s)

    tissue_train_preds = _avg_activity_baseline_predictions(
        train_labels, tissue_pred_dict
    )
    tissue_test_preds = _avg_activity_baseline_predictions(
        test_labels, tissue_pred_dict
    )
    tissue_val_preds = _avg_activity_baseline_predictions(val_labels, tissue_pred_dict)

    # train_error = mean_squared_error(train_true, train_preds, squared=False)
    # test_error = mean_squared_error(test_true, test_preds, squared=False)
    # val_error = mean_squared_error(val_true, val_preds, squared=False)

    tissue_train_error = mean_squared_error(
        train_true, tissue_train_preds, squared=False
    )
    test_train_error = mean_squared_error(test_true, tissue_test_preds, squared=False)
    val_train_error = mean_squared_error(val_true, tissue_val_preds, squared=False)

    # print(f"Train error: {train_error}")
    # print(f"Test error: {test_error}")
    # print(f"Validation error: {val_error}")

    print(
        f"Tissue train error: {tissue_train_error} \nTissue test error: {test_train_error} \nTissue validation error: {val_train_error}"
    )

    # with open("all_tissue_baseline_test_preds.pkl", "wb") as f:
    #     pickle.dump(test_preds, f)

    with open("all_tissue_baseline_test_labels.pkl", "wb") as f:
        pickle.dump(test_true, f)

    with open("dataset_tissue_baseline_test_preds.pkl", "wb") as f:
        pickle.dump(tissue_test_preds, f)


if __name__ == "__main__":
    main(expression_gct="GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct")
