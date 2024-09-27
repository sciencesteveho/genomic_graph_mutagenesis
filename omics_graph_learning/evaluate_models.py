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
from scipy.stats import ttest_rel  # type: ignore
from scipy.stats import wilcoxon  # type: ignore
from statsmodels.stats.anova import AnovaRM  # type: ignore


class ModelMetrics:
    """Class to store model performance metrics, and compute the mean and
    standard deviation across the three seeded runs.

    Attributes:
        model_dir: Path to the model directory.
        metrics: Dictionary of performance metrics for each run.

    Methods
    --------
    print_results:
        Print the computed statistics for the model.

    Examples:
    --------
    # Instantiating the object will automatically call metrics calculation
    >>> model_metrics = ModelMetrics(Path("path/to/model_dir"))

    # Print the computed statistics to the console
    >>> model_metrics.print_results()
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

        # compute statistics
        self.metrics = self._load_precomputed_metrics()
        self._compute_statistics()

    def _load_precomputed_metrics(self) -> Dict[str, List[float]]:
        """Load precomputed bootstrap confidence intervals and final test set
        RMSE from model post evaluation.
        """
        metrics: Dict[str, List[float]] = {
            "ci_lower": [],
            "ci_upper": [],
            "final_test_pearson": [],
            "final_test_rmse": [],
            "bootstrap_pearson": [],
        }

        for run_number in range(1, 4):
            metrics_file = self.model_dir / f"run_{run_number}" / "eval_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_json = json.load(f)
                    metrics["ci_lower"].append(metrics_json.get("CI lower", np.nan))
                    metrics["ci_upper"].append(metrics_json.get("CI upper", np.nan))
                    metrics["final_test_pearson"].append(
                        metrics_json.get("Final test pearson", np.nan)
                    )
                    metrics["final_test_rmse"].append(
                        metrics_json.get("Final test RMSE", np.nan)
                    )
                    metrics["bootstrap_pearson"].append(
                        metrics_json.get("Bootstrap pearson", np.nan)
                    )
            else:
                print(f"CI file not found: {metrics_file}")

        return metrics

    def _compute_statistics(self) -> None:
        """Compute means and standard deviations of the performance metrics
        from all runs.
        """
        metrics_to_compute = {
            "final_test_rmse": "final_test_rmse",
            "final_test_pearson": "final_test_pearson",
            "bootstrap_pearson": "bootstrap_pearson",
            "ci_lower": "ci_lower",
            "ci_upper": "ci_upper",
        }

        for metric_key, metric_name in metrics_to_compute.items():
            setattr(self, f"{metric_name}_mean", np.mean(self.metrics[metric_key]))
            setattr(
                self, f"{metric_name}_std", np.std(self.metrics[metric_key], ddof=1)
            )

    def print_results(self) -> None:
        """Print the computed statistics for the model."""
        metrics_to_print = [
            "final_test_pearson_mean",
            "final_test_rmse_mean",
            "final_test_rmse_std",
            "bootstrap_pearson_mean",
            "bootstrap_pearson_std",
            "ci_lower_mean",
            "ci_upper_mean",
        ]
        for metric in metrics_to_print:
            print(f"{metric}: {getattr(self, metric)}")


class ModelComparison:
    """Class to compare model performances and perform statistical analyses.

    Attributes:
        collected_metrics: List of ModelMetrics objects to compare.
        model_names: List of model names.
        performances: Dictionary of model performances.
        p_values_ttest: Pairwise p-values from paired t-tests.
        p_values_wilcoxon: Pairwise p-values from Wilcoxon signed-rank tests.
        effect_sizes: Pairwise Cohen's d effect sizes.
        anova_results: Repeated measures ANOVA results.
        posthoc_results: Post-hoc pairwise t-test results.

    Methods
    --------
    compare_models:
        Perform statistical tests and print results.
    report_results:
        Print model comparison results.

    Examples:
    --------
    # Instantiate the ModelComparison object with a list of ModelMetrics objects
    >>> model_comparison = ModelComparison([model_metrics_1, model_metrics_2])

    # Compare models and print results
    >>> model_comparison.compare_models()

    # Print results again
    >>> model_comparison.report_results()
    """

    def __init__(self, collected_metrics: List[ModelMetrics]) -> None:
        """Initialize the ModelComparison class with a list of ModelMetrics
        objects to compare.
        """
        self.collected_metrics = collected_metrics
        self.model_names = [mm.model_dir.name for mm in collected_metrics]

        if len(self.model_names) < 2:
            raise ValueError(
                "Need at least two ModelMetrics objects to compare performance."
            )

        # collect performances
        self.performances = {
            mm.model_dir.name: mm.metrics["final_test_pearson_mean"]
            for mm in collected_metrics
        }

    def compare_models(self) -> None:
        """Perform statistical tests and print results."""
        self._statistical_tests()
        self.report_results()

    def _statistical_tests(self) -> None:
        """Perform paired t-tests, Wilcoxon tests, calculate effect sizes, and
        repeated measures ANOVA on the test-set Pearson R.
        """
        models = self.model_names
        n_models = len(models)
        self.p_values_ttest = np.full((n_models, n_models), np.nan)
        self.p_values_wilcoxon = np.full((n_models, n_models), np.nan)
        self.effect_sizes = np.full((n_models, n_models), np.nan)

        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_i = models[i]
                model_j = models[j]
                data_i = np.array(self.performances[model_i])
                data_j = np.array(self.performances[model_j])

                # paired t-test
                _, p_val_ttest = ttest_rel(data_i, data_j)
                self.p_values_ttest[i, j] = p_val_ttest
                self.p_values_ttest[j, i] = p_val_ttest  # symmetric

                # wilcoxon signed-rank test
                try:
                    _, p_val_wilcoxon = wilcoxon(data_i, data_j)
                except ValueError:
                    # handle the case where all differences are zero
                    p_val_wilcoxon = 1.0
                self.p_values_wilcoxon[i, j] = p_val_wilcoxon
                self.p_values_wilcoxon[j, i] = p_val_wilcoxon  # symmetric

                # Cohen's d - effect size
                cohen_d = self._cohens_d(data_i, data_j)
                self.effect_sizes[i, j] = cohen_d
                self.effect_sizes[j, i] = cohen_d  # symmetric

        # repeated measures ANOVA
        if n_models > 2:
            self._repeated_measures_anova()
        else:
            self.anova_results = None
            self.posthoc_results = None

    def _cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Cohen's d for paired samples."""
        diff = x - y
        sd_diff = np.std(diff, ddof=1)
        mean_diff = np.mean(diff)
        return mean_diff / sd_diff if sd_diff != 0 else 0.0

    def _repeated_measures_anova(self) -> None:
        """Perform repeated measures ANOVA and post-hoc tests."""
        subjects = np.arange(len(next(iter(self.performances.values()))))
        data: List[Dict[str, Union[str, float]]] = []
        for model in self.model_names:
            data.extend(
                {"Subject": subjects[i], "Model": model, "Performance": perf}
                for i, perf in enumerate(self.performances[model])
            )
        df = pd.DataFrame(data)
        aovrm = AnovaRM(df, "Performance", "Subject", within=["Model"])
        res = aovrm.fit()
        self.anova_results = res

        # check if ANOVA shows significant differences
        p_value = res.anova_table["Pr > F"][0]
        if p_value < 0.05:
            # post-hoc pairwise t-tests with Holm correction
            self.posthoc_results = pg.pairwise_ttests(
                dv="Performance",
                within="Model",
                subject="Subject",
                data=df,
                padjust="holm",
            )
        else:
            self.posthoc_results = None

    def report_results(self) -> None:
        """Print model comparison results."""
        print("Model Performance Metrics:")
        metrics_to_print = [
            "p_values_ttest",
            "p_values_wilcoxon",
            "effect_sizes",
            "anova_results",
            "posthoc_results",
        ]

        for metric in metrics_to_print:
            if getattr(self, metric) is not None:
                print(f"{metric}: {getattr(self, metric)}")
            else:
                print(f"{metric}: not calculated.")

        if self.anova_results is None:
            print(
                "\nANOVA was not performed (less than 3 models) or did not show significant differences."
            )
