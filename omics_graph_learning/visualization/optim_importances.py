#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Plot the results of the Optuna optimization process."""


import argparse
import logging
from pathlib import Path

import numpy as np
import optuna  # type: ignore
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import plotly  # type: ignore

from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.optimize_hyperparameters import set_optim_directory
from omics_graph_learning.utils.common import setup_logging


def custom_sort_pearson(val: float) -> float:
    """Custom sort to move inf values to bottom."""
    return -float("inf") if np.isinf(val) else val


def display_results(
    study: optuna.Study, optuna_dir: Path, logger: logging.Logger
) -> None:
    """Display the results of the Optuna study."""
    logger.info("Study statistics:")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(
        "Number of pruned trials: "
        f"{len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}"
    )
    logger.info(
        "Number of complete trials: "
        f"{len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}"
    )

    best_trial = study.best_trial
    logger.info("Best trial:")
    logger.info(f"Best Pearson's r: {best_trial.value}")
    logger.info("Best params:")
    for key, value in best_trial.params.items():
        logger.info(f"\t{key}: {value}")

    # save to csv
    df = study.trials_dataframe()
    df = df.sort_values(
        "value", key=lambda x: x.map(custom_sort_pearson), ascending=False
    )
    df.to_csv(optuna_dir / "optuna_results.csv", index=False)
    logger.info(f"Ten best runs: {df.head(10)}")


def plot_importances(
    study: optuna.Study, optuna_dir: Path, logger: logging.Logger
) -> None:
    """Plot and save the importances of the hyperparameters."""
    most_important_parameters = optuna.importance.get_param_importances(study)

    # display the most important hyperparameters
    logger.info("\nMost important hyperparameters:")
    for key, value in most_important_parameters.items():
        logger.info("  {}:{}{:.2f}%".format(key, (15 - len(key)) * " ", value * 100))

    # plot and save importances to file
    optuna.visualization.plot_optimization_history(study).write_image(
        f"{optuna_dir}/history.png"
    )
    optuna.visualization.plot_param_importances(study).write_image(
        f"{optuna_dir}/importances.png"
    )
    optuna.visualization.plot_slice(study).write_image(f"{optuna_dir}/slice.png")


def main() -> None:
    """Load finished study and display results."""
    parser = argparse.ArgumentParser(
        description="Plot the results of the Optuna optimization process."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML file",
    )
    args = parser.parse_args()

    # set up logging
    logger = setup_logging()

    # set up the directories
    experiment_config = ExperimentConfig.from_yaml(args.config)
    optuna_dir = set_optim_directory(experiment_config)

    # load the study
    storage_file = optuna_dir / "optuna_journal_storage.log"
    storage = JournalStorage(JournalFileBackend(str(storage_file)))
    study = optuna.load_study(study_name="distributed_optimization", storage=storage)

    # display the results
    display_results(
        study=study,
        optuna_dir=optuna_dir,
        logger=logger,
    )

    # plot the importances
    plot_importances(
        study=study,
        optuna_dir=optuna_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()