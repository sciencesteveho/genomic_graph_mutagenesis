# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to handle automated hyperparameter tuning via Optuna using the resource
conscious Hyperband pruner. The code automatically detects the presence of
multiple CUDA capable CPUs. Given SQL errors with filelocking, and given that
the optimization does not require saving model states, we opt to save
performance metrics in memory before writing to JSON files.

The optimization of the graphs is done on a subset of whole chromosomes. We opt
for 40% (8 chromosomes) but keep the entirety of the validation chromosomes for
the optimization."""


import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import optuna  # type: ignore
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import plotly  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore

from omics_graph_learning.architecture_builder import build_gnn_architecture
from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.graph_to_pytorch import GraphToPytorch
from omics_graph_learning.schedulers import OptimizerSchedulerHandler
from omics_graph_learning.train_gnn import build_gnn_architecture
from omics_graph_learning.train_gnn import GNNTrainer
from omics_graph_learning.train_gnn import prep_loader
from omics_graph_learning.utils.common import check_cuda_env
from omics_graph_learning.utils.common import dir_check_make
from omics_graph_learning.utils.common import PyGDataChecker
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.common import tensor_out_to_array

# helpers for trial
EPOCHS = 20
MIN_RESOURCE = 4
REDUCTION_FACTOR = 3
PATIENCE = 6


def set_optim_directory(experiment_config: ExperimentConfig) -> Path:
    """Ensure an optimization directory exists."""
    model_dir = (
        experiment_config.root_dir / "models" / experiment_config.experiment_name
    )
    optuna_dir = model_dir / "optuna"

    dir_check_make(optuna_dir)
    return optuna_dir


def suggest_embedding_size(
    trial: optuna.Trial, heads: Optional[int] = None, model: Optional[str] = None
) -> int:
    """Suggest embedding size based on model and number of heads, because heads
    increase dimensionality by heads * embedding_size.
    """
    if model == "DeeperGCN":
        return trial.suggest_int("embedding_size", low=64, high=256, step=64)
    if model == "PNA":
        return trial.suggest_int("embedding_size", low=64, high=256, step=64)
    if heads:
        embedding_max = 320
        embedding_high = embedding_max // heads
        return trial.suggest_int("embedding_size", low=64, high=embedding_high, step=64)
    return trial.suggest_int("embedding_size", low=64, high=320, step=64)


def suggest_gnn_layers(trial: optuna.Trial, model: str) -> int:
    """Suggest GNN layers based on the model."""
    if model == "DeeperGCN":
        return trial.suggest_int("gnn_layers", low=6, high=12)
    return trial.suggest_int("gnn_layers", low=2, high=8)


def suggest_hyperparameters(
    trial: optuna.Trial,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Suggest hyperparameters for the Optuna trial.

    Hyperparameters are based off a mix of ancedotal results and from Tönshoff
    et al. (2023) - "Where Did the Gap Go? Reassessing the Long-Range Graph
    Benchmark."
    """
    model = trial.suggest_categorical(
        "model", ["GCN", "GraphSAGE", "PNA", "GAT", "UniMPTransformer", "DeeperGCN"]
    )

    model_params = {
        "model": model,
        "activation": trial.suggest_categorical(
            "activation", ["relu", "leakyrelu", "gelu"]
        ),
        "shared_mlp_layers": trial.suggest_int(
            "shared_mlp_layers", low=1, high=3, step=1
        ),
        "dropout_rate": trial.suggest_float(
            "dropout_rate", low=0.0, high=0.5, step=0.1
        ),
        "attention_task_head": trial.suggest_categorical(
            "attention_task_head", [True, False]
        ),
        "positional_encoding": trial.suggest_categorical(
            "positional_encoding", [True, False]
        ),
        "gnn_layers": suggest_gnn_layers(trial, model),
    }

    train_params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", low=5e-5, high=5e-2, log=True
        ),
        "optimizer_type": trial.suggest_categorical(
            "optimizer_type", ["Adam", "AdamW"]
        ),
        # "scheduler_type": trial.suggest_categorical(
        #     "scheduler_type", ["plateau", "linear_warmup", "cosine"]
        # ),
        "scheduler_type": "plateau",
        "batch_size": trial.suggest_int("batch_size", low=32, high=256, step=32),
    }

    # set heads and embedding size for attention-based models
    if model in ["GAT", "UniMPTransformer"]:
        heads = trial.suggest_int("heads", 1, 4)
        model_params["heads"] = heads
        model_params["embedding_size"] = suggest_embedding_size(
            trial=trial, heads=heads, model=model
        )
    else:
        model_params["embedding_size"] = suggest_embedding_size(
            trial=trial, model=model
        )

    if model != "DeeperGCN":
        model_params["residual"] = trial.suggest_categorical(
            "residual", ["shared_source", "distinct_source", None]
        )

    return model_params, train_params


def train_and_evaluate(
    trial: optuna.Trial,
    trainer: GNNTrainer,
    train_loader: torch_geometric.data.DataLoader,
    val_loader: torch_geometric.data.DataLoader,
    scheduler: LRScheduler,
    logger: logging.Logger,
    min_epochs: int = 0,
) -> Tuple[float, float]:
    """Run training and evaluation."""
    best_loss = float("inf")
    best_pearson_r = -float("inf")
    early_stop_counter = 0

    for epoch in range(EPOCHS + 1):
        train_loss, _, _ = trainer.train(train_loader=train_loader, epoch=epoch)
        logger.info(f"Epoch {epoch}, Loss: {train_loss:.3f}")

        if np.isnan(train_loss):
            logger.info(f"Trial {trial.number} pruned at epoch {epoch} due to NaN loss")
            raise optuna.exceptions.TrialPruned()

        # validation
        (
            val_loss,
            val_rmse,
            _,
            _,
            val_pearson_r,
            val_accuracy,
        ) = trainer.evaluate(
            data_loader=val_loader,
            epoch=epoch,
            mask="val",
        )

        # calculate metrics on validation set
        logger.info(
            f"Validation loss: {val_loss:.3f}, Pearson's R: {val_pearson_r:.3f}, "
            f"RMSE: {val_rmse:.3f}, Accuracy: {val_accuracy:.3f}"
        )

        if np.isnan(val_loss):
            logger.info(f"Trial {trial.number} pruned at epoch {epoch} due to NaN loss")
            raise optuna.exceptions.TrialPruned()

        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_pearson_r = val_pearson_r
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE and epoch >= min_epochs:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # report for pruning
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        trial.report(val_loss, epoch)

        # handle pruning based on the intermediate value.
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

    return best_loss, best_pearson_r


def get_objective_loaders(
    data: torch_geometric.data.Data,
    batch_size: int,
    layers: int,
) -> Tuple[torch_geometric.data.DataLoader, torch_geometric.data.DataLoader]:
    """Get the objective loaders for training and validation."""
    train_loader = prep_loader(
        data=data,
        mask="optimization_mask",
        batch_size=batch_size,
        shuffle=True,
        layers=layers,
    )

    val_loader = prep_loader(
        data=data,
        mask="val_mask",
        batch_size=batch_size,
        layers=layers,
    )

    return train_loader, val_loader


def suggest_and_log_hyperparameters(
    trial: optuna.Trial, logger: logging.Logger
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Suggest hyperparameters and log them."""
    model_params, train_params = suggest_hyperparameters(trial)
    logger.info("=" * 50)
    logger.info(f"Starting Trial {trial.number} with parameters:")
    logger.info(f"Model Params: {model_params}")
    logger.info(f"Train Params: {train_params}")
    logger.info("=" * 50)
    for handler in logger.handlers:
        handler.flush()
    return model_params, train_params


def load_and_validate_data(
    experiment_config: ExperimentConfig,
    args: argparse.Namespace,
    positional_encoding: bool,
) -> torch_geometric.data.Data:
    """Load graph data and validate"""
    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=args.split_name,
        regression_target=args.target,
        positional_encoding=positional_encoding,
    ).make_data_object()

    PyGDataChecker.check_pyg_data(data)  # validate data
    return data


def get_model_and_optimizer(
    data: torch_geometric.data.Data,
    train_loader: torch_geometric.data.DataLoader,
    model_params: Dict[str, Any],
    train_params: Dict[str, Any],
    warmup_steps: int,
    total_steps: int,
) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer, Union[LRScheduler, ReduceLROnPlateau]
]:
    """Get the model, optimizer, and scheduler for each trial."""
    model = build_gnn_architecture(
        in_size=data.x.shape[1],
        out_channels=1,
        train_dataset=train_loader if model_params["model"] == "PNA" else None,
        **model_params,
    )

    optimizer = OptimizerSchedulerHandler.set_optimizer(
        optimizer_type=train_params["optimizer_type"],
        learning_rate=train_params["learning_rate"],
        model_params=model.parameters(),
    )

    scheduler = OptimizerSchedulerHandler.set_scheduler(
        scheduler_type=train_params["scheduler_type"],
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        training_steps=total_steps,
    )
    return model, optimizer, scheduler


def objective(
    trial: optuna.Trial,
    experiment_config: ExperimentConfig,
    args: argparse.Namespace,
    logger: logging.Logger,
    device: torch.device,
) -> float:
    """Objective function to be optimized by Optuna."""
    try:
        # get trial hyperparameters
        model_params, train_params = suggest_and_log_hyperparameters(
            trial=trial, logger=logger
        )
        batch_size = train_params["batch_size"]

        # load graph data
        data = load_and_validate_data(
            experiment_config=experiment_config,
            args=args,
            positional_encoding=model_params["positional_encoding"],
        )

        # set data loaders
        train_loader, val_loader = get_objective_loaders(
            data=data,
            batch_size=batch_size,
            layers=model_params["gnn_layers"],
        )

        # get steps for optimizer and scheduler
        total_steps, warmup_steps = OptimizerSchedulerHandler.calculate_training_steps(
            train_loader=train_loader,
            batch_size=batch_size,
            epochs=100,  # calculate warmup steps as a fraction of 100 epochs
        )

        # define and build model, optimizer, and scheduler
        model, optimizer, scheduler = get_model_and_optimizer(
            data=data,
            train_loader=train_loader,
            model_params=model_params,
            train_params=train_params,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # print model architecture and dimensions
        logger.info(model)
        model = model.to(device)

        # initialize trainer
        trainer = GNNTrainer(
            model=model,
            device=device,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
        )

        # set epoch min for warmup schedulers
        if train_params["scheduler_type"] in ["linear_warmup", "cosine"]:
            min_epochs = 10 + (
                PATIENCE // 3
            )  # 10 epochs for warmup and 2 epochs for patience
        else:
            min_epochs = 0

        # training loop
        best_loss, best_pearson_r = train_and_evaluate(
            trial, trainer, train_loader, val_loader, scheduler, logger, min_epochs
        )

        trial.set_user_attr("best_loss", best_loss)  # save best loss
        trial.set_user_attr("best_pearson_r", best_pearson_r)  # save best pearson r
        return best_loss
    except RuntimeError as e:
        handle_cuda_out_of_memory_error(e=e, trial=trial, logger=logger)


def handle_cuda_out_of_memory_error(
    e: RuntimeError, trial: optuna.Trial, logger: logging.Logger
) -> None:
    """Handle CUDA out of memory error."""
    if "CUDA out of memory" in str(e):
        trial.report(float("inf"), EPOCHS)
        logger.info(f"Trial {trial.number} pruned due to CUDA out of memory")
        raise optuna.exceptions.TrialPruned()
    else:
        logger.error(f"RuntimeError in trial {trial.number}: {str(e)}")
        raise e


def run_optimization(
    args: argparse.Namespace,
    logger: logging.Logger,
    optuna_dir: Path,
    device: torch.device,
    n_trials: int,
) -> optuna.Study:
    """Run the optimization process per GPU."""
    experiment_config = ExperimentConfig.from_yaml(args.config)
    study_name = "distributed_optimization"

    # set up JournalStorage
    storage_file = optuna_dir / "optuna_journal_storage.log"
    storage = JournalStorage(JournalFileBackend(str(storage_file)))

    # create the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=MIN_RESOURCE,
            max_resource=EPOCHS,
            reduction_factor=REDUCTION_FACTOR,
        ),
    )
    logger.info(f"Created new study: {study_name}")

    study.optimize(
        lambda trial: objective(trial, experiment_config, args, logger, device),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    return study


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters with Optuna."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="expression_median_only",
        choices=[
            "expression_median_only",
            "expression_media_and_foldchange",
            "difference_from_average",
            "foldchange_from_average",
            "protein_targets",
            "rna_seq",
        ],
    )
    parser.add_argument(
        "--split_name",
        type=str,
    )
    parser.add_argument(
        "--n_trials", type=int, help="Number of trials to run for this study."
    )
    return parser.parse_args()


def main() -> None:
    """Main function to optimize hyperparameters w/ optuna!"""
    check_cuda_env()
    args = parse_arguments()

    # set up directories
    experiment_config = ExperimentConfig.from_yaml(args.config)
    optuna_dir = set_optim_directory(experiment_config)

    # set up logging
    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    log_file = (
        optuna_dir / f"{experiment_config.experiment_name}_optuna_job_{job_id}.log"
    )
    logger = setup_logging(str(log_file))

    logger.info("Starting optimization process")
    logger.info(f"Configuration: {args}")

    study = run_optimization(
        args=args,
        logger=logger,
        optuna_dir=optuna_dir,
        device=torch.device("cuda"),
        n_trials=args.n_trials,
    )

    # log best trial for this job
    best_trial = study.best_trial
    logger.info(f"Best trial in this job: {best_trial.number}")
    logger.info(f"Best Pearson's r in this job: {best_trial.value}")
    logger.info("Best params in this job:")
    for key, value in best_trial.params.items():
        logger.info(f"\t{key}: {value}")


if __name__ == "__main__":
    main()
