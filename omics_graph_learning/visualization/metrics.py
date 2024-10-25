#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Make plots to visualize model performances and metrics."""


from typing import List

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from omics_graph_learning.model_evaluation import ModelComparison
from omics_graph_learning.model_evaluation import ModelMetrics
from omics_graph_learning.visualization import set_matplotlib_publication_parameters


class ModelVisualization:
    """Class to handle visualization of model metrics and comparisons."""

    def __init__(
        self,
        model_metrics: ModelMetrics,
        model_comparison: ModelComparison,
    ):
        self.model_metrics = model_metrics
        self.model_comparison = model_comparison
        set_matplotlib_publication_parameters()

    def plot_performance_with_error_bars(self) -> None:
        """Plot performance with error bars (95% CI) for a ModelMetrics
        object.
        """
        if self.model_metrics is None:
            raise ValueError("ModelMetrics object is required for this plot.")

        means = self.model_metrics.metrics["test_pearson_r"]
        ci_lower = self.model_metrics.metrics["ci_lower"]
        ci_upper = self.model_metrics.metrics["ci_upper"]

        # calculate error bars
        lower_errors = [mean - lower for mean, lower in zip(means, ci_lower)]
        upper_errors = [upper - mean for mean, upper in zip(means, ci_upper)]
        errors = [lower_errors, upper_errors]

        plt.bar(range(len(means)), means, yerr=errors, capsize=5, color="skyblue")
        plt.xticks(range(len(means)), [f"Run {i+1}" for i in range(len(means))])
        plt.ylabel("Performance Metric (Pearson R)")
        plt.title(f"Model Performance (95% CI): {self.model_metrics.model_dir.name}")
        plt.tight_layout()
        plt.show()

    def plot_heatmap_p_values(self) -> None:
        """Plot heatmap of pairwise p-values."""
        models = self._validate_model_comparison()
        sns.heatmap(
            self.model_comparison.p_values_ttest,
            annot=True,
            fmt=".3f",
            xticklabels=models,
            yticklabels=models,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
        )
        self.show_plot("Pairwise p-values (Paired t-test)")

    def plot_heatmap_effect_sizes(self) -> None:
        """Plot heatmap of pairwise effect sizes for a ModelComparison
        object.
        """
        models = self._validate_model_comparison()
        sns.heatmap(
            self.model_comparison.effect_sizes,
            annot=True,
            fmt=".3f",
            xticklabels=models,
            yticklabels=models,
            cmap="viridis",
        )
        self.show_plot("Pairwise Effect Sizes (Cohen's d)")

    def _validate_model_comparison(self) -> List[str]:
        if self.model_comparison is None:
            raise ValueError("ModelComparison object is required for this plot.")
        return self.model_comparison.model_names

    @staticmethod
    def show_plot(title: str) -> None:
        """Show plot with title."""
        plt.title(title)
        plt.tight_layout()
        plt.show()
