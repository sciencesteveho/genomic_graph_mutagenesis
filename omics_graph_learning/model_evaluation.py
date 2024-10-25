#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Statistically compare performance metrics across multiple models of varying
graph construction, architecture, and hyperparameters."""


import json
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import pingouin as pg  # type: ignore
from scipy.stats import f_oneway  # type: ignore
from scipy.stats import ttest_rel  # type: ignore
from scipy.stats import wilcoxon  # type: ignore
from statsmodels.stats.anova import AnovaRM  # type: ignore
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore


class ModelMetrics:
    """
    Class to store model performance metrics and compute statistics across runs.

    Attributes:
        model_dir: Path to the model directory.
        per_run_df: DataFrame containing metrics for each run.
        summary_df: DataFrame containing mean and std for each metric.
    """

    def __init__(self, model_dir: Union[Path, str]) -> None:
        """Initialize the ModelMetrics object."""
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.model_dir = model_dir

        # Collect metrics from all runs
        self.per_run_df = self._load_metrics()

        # Compute summary statistics
        self.summary_df = self._compute_summary()

    def _load_metrics(self) -> pd.DataFrame:
        """Load metrics from model evaluation for all runs."""
        metrics = []

        for run_number in range(1, 4):
            metrics_file = self.model_dir / f"run_{run_number}" / "eval_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_json = json.load(f)
                    metrics.append(
                        {
                            "Model": self.model_dir.name,
                            "Run": run_number,
                            "CI_Lower": metrics_json.get("CI lower", np.nan),
                            "CI_Upper": metrics_json.get("CI upper", np.nan),
                            "Final_Test_Pearson": metrics_json.get(
                                "Final test pearson", np.nan
                            ),
                            "Final_Test_RMSE": metrics_json.get(
                                "Final test RMSE", np.nan
                            ),
                            "Bootstrap_Pearson": metrics_json.get(
                                "Bootstrap pearson", np.nan
                            ),
                        }
                    )
            else:
                print(f"Metrics file not found: {metrics_file}")

        if not metrics:
            # If no metrics were loaded, return a DataFrame with NaNs
            return pd.DataFrame(
                {
                    "Model": [self.model_dir.name],
                    "Run": [np.nan],
                    "CI_Lower": [np.nan],
                    "CI_Upper": [np.nan],
                    "Final_Test_Pearson": [np.nan],
                    "Final_Test_RMSE": [np.nan],
                    "Bootstrap_Pearson": [np.nan],
                }
            )

        # Create DataFrame from the list of metrics
        return pd.DataFrame(metrics)

    def _compute_summary(self) -> pd.DataFrame:
        """Compute mean and std for each metric."""
        mean_metrics = self.per_run_df.mean(numeric_only=True)
        std_metrics = self.per_run_df.std(numeric_only=True, ddof=1)

        summary_data = {
            "Model": self.model_dir.name,
            "Final_Test_Pearson_Mean": mean_metrics.get("Final_Test_Pearson", np.nan),
            "Final_Test_Pearson_Std": std_metrics.get("Final_Test_Pearson", np.nan),
            "Final_Test_RMSE_Mean": mean_metrics.get("Final_Test_RMSE", np.nan),
            "Final_Test_RMSE_Std": std_metrics.get("Final_Test_RMSE", np.nan),
            "CI_Lower_Mean": mean_metrics.get("CI_Lower", np.nan),
            "CI_Lower_Std": std_metrics.get("CI_Lower", np.nan),
            "CI_Upper_Mean": mean_metrics.get("CI_Upper", np.nan),
            "CI_Upper_Std": std_metrics.get("CI_Upper", np.nan),
            "Bootstrap_Pearson_Mean": mean_metrics.get("Bootstrap_Pearson", np.nan),
            "Bootstrap_Pearson_Std": std_metrics.get("Bootstrap_Pearson", np.nan),
        }

        return pd.DataFrame([summary_data])

    def get_per_run_df(self) -> pd.DataFrame:
        """Return the DataFrame containing metrics for each run."""
        return self.per_run_df

    def get_summary_df(self) -> pd.DataFrame:
        """Return the DataFrame containing the mean and std for the model."""
        return self.summary_df

    def print_results(self) -> None:
        """Print the computed statistics for the model."""
        print(f"Model: {self.summary_df['Model'].iloc[0]}")
        print("Mean Metrics:")
        print(
            self.summary_df.iloc[0][
                [
                    "Final_Test_Pearson_Mean",
                    "Final_Test_RMSE_Mean",
                    "CI_Lower_Mean",
                    "CI_Upper_Mean",
                    "Bootstrap_Pearson_Mean",
                ]
            ]
        )
        print("\nStandard Deviation Metrics:")
        print(
            self.summary_df.iloc[0][
                [
                    "Final_Test_Pearson_Std",
                    "Final_Test_RMSE_Std",
                    "CI_Lower_Std",
                    "CI_Upper_Std",
                    "Bootstrap_Pearson_Std",
                ]
            ]
        )


class ModelComparison:
    """
    Class to compare model performances using statistical tests.

    Attributes:
        combined_metrics_df: DataFrame containing per-run metrics for all models.
        summary_df: DataFrame containing mean and std for each model.
    """

    def __init__(
        self, combined_metrics_df: pd.DataFrame, summary_df: pd.DataFrame
    ) -> None:
        """
        Initialize the ModelComparison class.

        Args:
            combined_metrics_df: DataFrame with per-run metrics for all models.
            summary_df: DataFrame with summary statistics for each model.
        """
        self.combined_metrics_df = combined_metrics_df
        self.summary_df = summary_df

    def perform_anova(self) -> float:
        """
        Perform one-way ANOVA to test if there are any statistically significant differences between the means of the models.

        Returns:
            p-value from the ANOVA test.
        """
        # Extract the data for ANOVA
        groups = [
            group["Final_Test_Pearson"].values
            for name, group in self.combined_metrics_df.groupby("Model")
        ]

        # Perform one-way ANOVA
        f_stat, p_val = f_oneway(*groups)
        print(
            f"One-Way ANOVA Results: F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}"
        )
        return p_val

    def perform_posthoc_ttests(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform post-hoc pairwise t-tests with Bonferroni correction.

        Args:
            alpha: Significance level.

        Returns:
            DataFrame containing pairwise comparisons and their adjusted p-values.
        """
        models = self.combined_metrics_df["Model"].unique()
        results = []

        num_comparisons = len(models) * (len(models) - 1) / 2
        bonferroni_alpha = alpha / num_comparisons

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_i = models[i]
                model_j = models[j]

                data_i = self.combined_metrics_df[
                    self.combined_metrics_df["Model"] == model_i
                ]["Final_Test_Pearson"]
                data_j = self.combined_metrics_df[
                    self.combined_metrics_df["Model"] == model_j
                ]["Final_Test_Pearson"]

                # Ensure paired data if applicable; otherwise, use independent t-test
                # Here, runs are independent between models
                t_stat, p_val = ttest_rel(data_i, data_j)

                # Apply Bonferroni correction
                adjusted_p_val = p_val * num_comparisons
                adjusted_p_val = min(adjusted_p_val, 1.0)

                results.append(
                    {
                        "Model 1": model_i,
                        "Model 2": model_j,
                        "t-statistic": t_stat,
                        "p-value": p_val,
                        "Adjusted p-value (Bonferroni)": adjusted_p_val,
                    }
                )

        results_df = pd.DataFrame(results)
        print("\nPost-Hoc Pairwise T-Test Results with Bonferroni Correction:")
        print(results_df)

        # Save to CSV
        results_df.to_csv("posthoc_pairwise_ttest_results.csv", index=False)

        return results_df

    def visualize_pvalues(self, posthoc_df: pd.DataFrame) -> None:
        """
        Visualize the adjusted p-values as a heatmap.

        Args:
            posthoc_df: DataFrame containing post-hoc test results.
        """
        # Create a pivot table for heatmap
        pivot_df = posthoc_df.pivot(
            index="Model 1", columns="Model 2", values="Adjusted p-value (Bonferroni)"
        )

        # Initialize a square DataFrame for symmetric heatmap
        models = self.combined_metrics_df["Model"].unique()
        heatmap_df = pd.DataFrame(np.nan, index=models, columns=models)

        for _, row in posthoc_df.iterrows():
            heatmap_df.loc[row["Model 1"], row["Model 2"]] = row[
                "Adjusted p-value (Bonferroni)"
            ]
            heatmap_df.loc[row["Model 2"], row["Model 1"]] = row[
                "Adjusted p-value (Bonferroni)"
            ]

        # # Set diagonal to NaN
        # np.fill_diagonal(heatmap_df.values, np.nan)

        # plt.figure(figsize=(12, 10))
        # sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1)
        # plt.title("Post-Hoc Pairwise T-Test Adjusted p-values (Bonferroni Correction)")
        # plt.xticks(rotation=90)
        # plt.yticks(rotation=0)
        # plt.tight_layout()
        # plt.savefig("posthoc_pvalues_heatmap.png")
        # plt.close()

        # print("\nPost-hoc p-values heatmap saved as 'posthoc_pvalues_heatmap.png'.")

    # def plot_performance(self) -> None:
    #     """
    #     Plot the mean Final Test Pearson correlation with error bars representing the standard deviation.
    #     """
    #     plt.figure(figsize=(12, 8))
    #     sns.barplot(
    #         x="Final_Test_Pearson_Mean",
    #         y="Model",
    #         data=self.summary_df,
    #         xerr=self.summary_df["Final_Test_Pearson_Std"],
    #         palette="viridis",
    #     )
    #     plt.xlabel("Final Test Pearson Mean")
    #     plt.ylabel("Model")
    #     plt.title("Model Performance Comparison")
    #     plt.tight_layout()
    #     plt.savefig("model_performance_comparison.png")
    #     plt.show()

    #     print(
    #         "\nModel performance comparison plot saved as 'model_performance_comparison.png'."
    #     )
