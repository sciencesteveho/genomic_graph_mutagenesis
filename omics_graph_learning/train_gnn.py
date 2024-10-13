#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Data set-up, training loops, evaluation, and automated plotting for GNNs in
the OGL pipeline."""


import argparse
import json
import logging
import math
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import stats  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.architecture_builder import build_gnn_architecture
from omics_graph_learning.combination_loss import RMSEandBCELoss
from omics_graph_learning.graph_to_pytorch import GraphToPytorch
from omics_graph_learning.perturbation.perturb_graph import PerturbationConfig
from omics_graph_learning.schedulers import OptimizerSchedulerHandler
from omics_graph_learning.utils.arg_parser import OGLCLIParser
from omics_graph_learning.utils.common import count_model_parameters
from omics_graph_learning.utils.common import dir_check_make
from omics_graph_learning.utils.common import PyGDataChecker
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.common import tensor_out_to_array
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.constants import EARLY_STOP_PATIENCE
from omics_graph_learning.utils.constants import RANDOM_SEEDS
from omics_graph_learning.utils.tb_logger import TensorBoardLogger
from omics_graph_learning.visualization.training import plot_predicted_versus_expected
from omics_graph_learning.visualization.training import plot_training_losses


class GNNTrainer:
    """Class to handle GNN training and evaluation.

    Methods
    --------
    train:
        Train GNN model on graph data via minibatching
    evaluate:
        Evaluate GNN model on validation or test set
    inference_all_neighbors:
        Evaluate GNN model on test set using all neighbors
    training_loop:
        Execute training loop for GNN model
    log_tensorboard_data:
        Log data to tensorboard on the last batch of an epoch.

    Examples:
    --------
    # instantiate trainer
    >>> trainer = GNNTrainer(
            model=model,
            device=device,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            tb_logger=tb_logger,
        )

    # train model
    >>> model, _, early_stop = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            model_dir=run_dir,
            args=args,
            min_epochs=min_epochs,
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        data: torch_geometric.data.Data,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]] = None,
        logger: Optional[logging.Logger] = None,
        tb_logger: Optional[TensorBoardLogger] = None,
    ) -> None:
        """Initialize model trainer."""
        self.model = model
        self.device = device
        self.data = data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.tb_logger = tb_logger

        self.criterion = RMSEandBCELoss(alpha=0.8)

    def _forward_pass(
        self,
        data: torch_geometric.data.Data,
        mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Perform forward pass and compute losses and outputs."""
        data = data.to(self.device)

        regression_out, logits = self.model(
            x=data.x,
            edge_index=data.edge_index,
            mask=mask,
        )

        loss, regression_loss, classification_loss = self.criterion(
            regression_output=regression_out,
            regression_target=data.y,
            classification_output=logits,
            classification_target=data.class_labels,
            mask=mask,
        )

        # collect masked outputs and labels
        regression_out_masked = self._ensure_tensor_dim(regression_out[mask])
        labels_masked = self._ensure_tensor_dim(data.y[mask])

        classification_out_masked = self._ensure_tensor_dim(logits[mask])
        class_labels_masked = self._ensure_tensor_dim(data.class_labels[mask])

        return (
            loss,
            regression_loss,
            classification_loss,
            regression_out_masked,
            labels_masked,
            classification_out_masked,
            class_labels_masked,
        )

    def _train_single_batch(
        self,
        data: torch_geometric.data.Data,
        epoch: int,
        batch_idx: int,
        total_batches: int,
    ) -> Tuple[float, float, float]:
        """Train a single batch."""
        self.optimizer.zero_grad()

        # forward pass
        mask = data.train_mask_loss

        (
            loss,
            regression_loss,
            classification_loss,
            _,
            _,
            _,
            _,
        ) = self._forward_pass(data, mask)

        # backpropagation
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # log if last batch of epoch
        if self.tb_logger:
            self._log_tensorboard_data(
                epoch=epoch,
                last_batch=batch_idx == total_batches - 1,
            )

        # check for NaN gradients
        self._check_for_nan_gradients()
        self.optimizer.step()

        # step warmup schedulers, if applicable
        if isinstance(self.scheduler, LRScheduler) and not isinstance(
            self.scheduler, ReduceLROnPlateau
        ):
            self.scheduler.step()

        batch_size_mask = int(mask.sum())
        return (
            loss.item() * batch_size_mask,
            regression_loss.item() * batch_size_mask,
            classification_loss.item() * batch_size_mask,
        )

    def _evaluate_single_batch(
        self,
        data: torch_geometric.data.Data,
        mask: str,
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a single batch."""
        mask = getattr(data, f"{mask}_mask_loss")
        if mask.sum() == 0:
            return (
                0.0,
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
            )
        # forward pass
        (
            loss,
            _,
            _,
            regression_out_masked,
            labels_masked,
            classification_out_masked,
            class_labels_masked,
        ) = self._forward_pass(data, mask)

        batch_size_mask = int(mask.sum())
        return (
            loss.item() * batch_size_mask,
            regression_out_masked.cpu(),
            labels_masked.cpu(),
            classification_out_masked.cpu(),
            class_labels_masked.cpu(),
        )

    def train(
        self,
        train_loader: torch_geometric.data.DataLoader,
        epoch: int,
        subset_batches: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Train GNN model on graph data"""
        self.model.train()
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(
            f"\nTraining {self.model.__class__.__name__} model @ epoch: {epoch} - "
        )

        total_loss = total_regression = total_classification = float(0)
        total_examples = 0
        total_batches = len(train_loader)
        for batch_idx, data in enumerate(train_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            loss, regression_loss, classification_loss = self._train_single_batch(
                data=data,
                epoch=epoch,
                batch_idx=batch_idx,
                total_batches=total_batches,
            )
            total_loss += loss
            total_regression += regression_loss
            total_classification += classification_loss
            total_examples += int(data.train_mask_loss.sum())
            pbar.update(1)

        pbar.close()
        final_loss = total_loss / total_examples if total_examples > 0 else 0.0
        final_regression = (
            total_regression / total_examples if total_examples > 0 else 0.0
        )
        final_classification = (
            total_classification / total_examples if total_examples > 0 else 0.0
        )
        return final_loss, final_regression, final_classification

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: torch_geometric.data.DataLoader,
        epoch: int,
        mask: str,
        subset_batches: Optional[int] = None,
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor, float, float]:
        """Base function for model evaluation or inference."""
        self.model.eval()
        pbar = tqdm(total=len(data_loader))
        pbar.set_description(
            f"\nEvaluating {self.model.__class__.__name__} model @ epoch: {epoch}"
        )

        total_loss = float(0)
        total_examples = 0
        regression_outs, regression_labels = [], []
        classification_outs, classification_labels = [], []

        for batch_idx, data in enumerate(data_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            loss, reg_out, reg_label, cls_out, cls_label = self._evaluate_single_batch(
                data=data,
                mask=mask,
            )
            total_loss += loss
            total_examples += int(getattr(data, f"{mask}_mask_loss").sum())

            regression_outs.append(reg_out)
            regression_labels.append(reg_label)
            classification_outs.append(cls_out)
            classification_labels.append(cls_label)

            pbar.update(1)

        pbar.close()
        average_loss = total_loss / total_examples if total_examples > 0 else 0.0

        # compute regression metrics
        rmse, pearson_r = self._compute_regression_metrics(
            regression_outs, regression_labels
        )

        # compute classification metrics
        accuracy = self._compute_classification_metrics(
            classification_outs, classification_labels
        )

        # log metrics
        self.logger.info(
            f"Epoch: {epoch:03d}, Loss: {average_loss:.3f}, RMSE: {rmse:.3f}, "
            f"Pearson's R: {pearson_r:.3f}, Accuracy: {accuracy:.3f}"
        )

        return (
            average_loss,
            rmse,
            torch.cat(regression_outs),
            torch.cat(regression_labels),
            pearson_r,
            accuracy,
        )

    def train_model(
        self,
        train_loader: torch_geometric.data.DataLoader,
        val_loader: torch_geometric.data.DataLoader,
        test_loader: torch_geometric.data.DataLoader,
        epochs: int,
        model_dir: Path,
        args: argparse.Namespace,
        min_epochs: int = 0,
    ) -> Tuple[torch.nn.Module, float, bool]:
        """Execute training loop for GNN models.

        The loop will train and evaluate the model on the validation set with
        minibatching. Afterward, the model is evaluated on the test set
        utilizing all neighbors.

        To account for warm-up schedulers, the function passes min_epochs. Early
        stopping counter starts after min_epochs + early_stop_patience // 2.
        """
        self.tb_logger.log_hyperparameters(vars(args))

        # set up early stopping counter
        best_validation = float("inf")
        stop_counter = 0

        for epoch in range(epochs + 1):
            self.logger.info(f"Epoch: {epoch:03d}")

            # train
            train_loss, train_regression_loss, train_classification_loss = self.train(
                train_loader=train_loader, epoch=epoch
            )

            # validation
            val_loss, val_rmse, _, _, val_r, val_accuracy = self.evaluate(
                data_loader=val_loader, epoch=epoch, mask="val"
            )

            # test
            test_loss, test_rmse, _, _, test_r, test_accuracy = self.evaluate(
                data_loader=test_loader, epoch=epoch, mask="test"
            )

            # log metrics to tensorboard
            metrics = {
                "Training loss": train_loss,
                "Training regression loss": train_regression_loss,
                "Training classification loss": train_classification_loss,
                "Validation loss": val_loss,
                "Validation Pearson's R": val_r,
                "Validation accuracy": val_accuracy,
                "Validation RMSE": val_rmse,
                "Test loss": test_loss,
                "Test Pearson's R": test_r,
                "Test accuracy": test_accuracy,
                "Test RMSE": test_rmse,
            }
            self.tb_logger.log_metrics(metrics, epoch)
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.3f}")

            # step scheduler if normal plateau scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            # early stopping logic
            if args.early_stop:
                if val_loss < best_validation:
                    stop_counter = 0
                    best_validation = val_loss
                    torch.save(
                        self.model.state_dict(),
                        model_dir / f"{args.model}_best_model.pt",
                    )
                else:
                    stop_counter += 1

                # check if minimum epochs have been reached before allowing
                # early stopping
                if epoch >= min_epochs and stop_counter >= EARLY_STOP_PATIENCE:
                    self.logger.info("*********** Early stopping triggered!")
                    break

        return self.model, best_validation, stop_counter == EARLY_STOP_PATIENCE

    def _log_tensorboard_data(
        self,
        epoch: int,
        last_batch: bool = False,
    ) -> None:
        """Log data to tensorboard on the last batch of an epoch."""
        if last_batch:
            self.tb_logger.log_learning_rate(self.optimizer, epoch)
            self.tb_logger.log_summary_statistics(self.model, epoch)
            self.tb_logger.log_aggregate_module_metrics(self.model, epoch)
            self.tb_logger.log_gradient_norms(self.model, epoch)

    def _check_for_nan_gradients(self) -> None:
        """Check for NaN gradients in model parameters."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")

    @staticmethod
    def _compute_regression_metrics(
        regression_outs: List[torch.Tensor],
        regression_labels: List[torch.Tensor],
    ) -> Tuple[float, float]:
        """Compute RMSE and Pearson's R for regression task."""
        if not regression_outs or not regression_labels:
            return 0.0, 0.0

        predictions = torch.cat(regression_outs).squeeze()
        targets = torch.cat(regression_labels).squeeze()
        mse = F.mse_loss(predictions, targets)
        rmse = torch.sqrt(mse).item()
        pearson_r, _ = pearsonr(predictions.numpy(), targets.numpy())

        return rmse, pearson_r

    @staticmethod
    def _compute_classification_metrics(
        classification_outs: List[torch.Tensor],
        classification_labels: List[torch.Tensor],
    ) -> float:
        """Compute accuracy for classification task."""
        if not classification_outs or not classification_labels:
            return 0.0

        logits = torch.cat(classification_outs).squeeze()
        labels = torch.cat(classification_labels).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        return (preds == labels).float().mean().item()

    @staticmethod
    def _ensure_tensor_dim(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has the correct dimensions for evaluation."""
        tensor = tensor.squeeze()
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor


def get_seed(run_number: int) -> int:
    """Get seed for reproducibility. We run models for 3 runs, each of which
    uses a different seed.
    """
    return RANDOM_SEEDS[run_number - 1]


def setup_device(args: argparse.Namespace) -> Tuple[torch.device, int]:
    """Check for GPU and set device accordingly."""
    seed = get_seed(args.run_number)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        return torch.device(f"cuda:{args.device}"), seed
    return torch.device("cpu"), seed


def prep_loader(
    data: torch_geometric.data.Data,
    mask: str,
    batch_size: int,
    shuffle: bool = False,
    layers: int = 2,
    avg_connectivity: bool = True,
) -> torch_geometric.data.DataLoader:
    """Loads data into NeighborLoader for GNN training. Returns a DataLoader
    with randomly sampled neighbors, either by 10 neighbors * layers or by using
    the average connectivity in the graph."""
    if avg_connectivity:
        num_neighbors = [data.avg_edges] * layers
    else:
        num_neighbors = [10] * layers
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=getattr(data, mask),
        shuffle=shuffle,
    )


def bootstrap_evaluation(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Perform bootstrap evaluation by resampling with replacement."""
    n_samples = len(predictions)
    bootstrap_correlations: List[float] = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_pred = predictions[indices].numpy()
        bootstrap_labels = labels[indices].numpy()

        correlation, _ = stats.pearsonr(bootstrap_pred, bootstrap_labels)
        bootstrap_correlations.append(correlation)

    mean_correlation = np.mean(bootstrap_correlations)
    ci_lower, ci_upper = stats.t.interval(
        confidence,
        len(bootstrap_correlations) - 1,
        loc=mean_correlation,
        scale=stats.sem(bootstrap_correlations),
    )

    return mean_correlation, ci_lower, ci_upper


def load_final_model(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    model_dir: Path,
) -> torch.nn.Module:
    """Load final model from checkpoint."""
    checkpoint = torch.load(
        model_dir / f"{model_name}_best_model.pt", map_location=device
    )
    model.load_state_dict(checkpoint, strict=False)
    return model


def make_model_plots(
    outs: torch.Tensor,
    labels: torch.Tensor,
    rmse: float,
    tb_logger: TensorBoardLogger,
    run_dir: Path,
) -> None:
    """Plot training losses and performance"""
    # convert to numpy arrays
    predictions_median = tensor_out_to_array(outs)
    labels_median = tensor_out_to_array(labels)

    # plot performance
    performance = plot_predicted_versus_expected(
        predicted=predictions_median,
        expected=labels_median,
        rmse=rmse,
        save_path=run_dir,
    )
    tb_logger.writer.add_figure("Model performance", performance)

    # plot loss
    loss = plot_training_losses(tensorboard_log=tb_logger.log_dir)
    tb_logger.writer.add_figure("Training loss", loss)


def post_model_evaluation(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    data: torch_geometric.data.Data,
    data_loader: torch_geometric.data.DataLoader,
    tb_logger: TensorBoardLogger,
    logger: logging.Logger,
    run_dir: Path,
    early_stop: bool,
) -> None:
    """Plot training losses and performance"""

    # load early stopping model, otherwise, use the final model
    if early_stop:
        model = load_final_model(
            model_name=model_name,
            model=model,
            device=device,
            model_dir=run_dir,
        )

    if model is not None:
        model.to(device)
    else:
        raise ValueError("Model is None. Cannot plot performance")

    post_eval_trainer = GNNTrainer(
        model=model,
        device=device,
        data=data,
        logger=logger,
        tb_logger=tb_logger,
    )

    loss, rmse, outs, labels, pearson_r, accuracy = post_eval_trainer.evaluate(
        data_loader=data_loader,
        mask="test",
        epoch=0,
    )

    # save final eval
    np.save(run_dir / "outs.npy", outs.numpy())
    np.save(run_dir / "labels.npy", labels.numpy())
    for step, (out, label) in enumerate(zip(outs, labels)):
        tb_logger.writer.add_scalar("Predictions", out.item(), step)
        tb_logger.writer.add_scalar("Labels", label.item(), step)

    # bootstrap evaluation
    mean_correlation, ci_lower, ci_upper = bootstrap_evaluation(
        predictions=outs,
        labels=labels,
    )

    metrics = {
        "Final test loss": loss,
        "Final test pearson": pearson_r,
        "Final test RMSE": rmse,
        "Final test accuracy": accuracy,
        "Bootstrap pearson": mean_correlation,
        "CI lower": ci_lower,
        "CI upper": ci_upper,
    }

    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.3f}")

    tb_logger.log_metrics(metrics, 0)
    with open(run_dir / "eval_metrics.json", "w") as output:
        json.dump(metrics, output, indent=4)

    # make model plots
    make_model_plots(
        outs=outs,
        labels=labels,
        rmse=rmse,
        tb_logger=tb_logger,
        run_dir=run_dir,
    )


def _dump_metadata_json(
    args: argparse.Namespace,
    experiment_config: ExperimentConfig,
    run_dir: Path,
    total_parameters: int,
) -> None:
    """Dump metadata json to run directory."""
    metadata = vars(args)
    metadata["experiment_name"] = experiment_config.experiment_name
    metadata["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["total_parameters"] = total_parameters
    with open(run_dir / "metadata.json", "w") as output:
        json.dump(metadata, output, indent=4)


def _experiment_setup(
    args: argparse.Namespace, experiment_config: ExperimentConfig
) -> Tuple[Path, logging.Logger, TensorBoardLogger]:
    """Prepare directories and set up logging for experiment."""
    # set run directories
    experiment_dir = (
        experiment_config.root_dir / "models" / experiment_config.experiment_name
    )
    run_dir = experiment_dir / f"run_{args.run_number}"
    tb_dir = experiment_dir / "tensorboard"

    for folder in ["logs", "plots"]:
        dir_check_make(run_dir / folder)

    # set up loggers
    logger = setup_logging(log_file=str(run_dir / "logs" / "training_log.txt"))
    tb_logger = TensorBoardLogger(log_dir=tb_dir / f"run_{args.run_number}")

    logger.info("Experiment setup initialized.")
    logger.info(f"Experiment configuration: {experiment_config}")
    logger.info(f"Model directory: {run_dir}")
    return run_dir, logger, tb_logger


def prepare_pertubation_config(
    args: argparse.Namespace,
) -> Optional[PerturbationConfig]:
    """Set up perturbation config, if applicable."""
    if args.node_perturbation or args.edge_perturbation:
        perturbations = {
            "node_perturbation": args.node_perturbation or None,
            "edge_perturbation": args.edge_perturbation or None,
            "total_random_edges": args.total_random_edges or None,
        }
        return PerturbationConfig(**perturbations)
    return None


def calculate_min_epochs(args: argparse.Namespace) -> int:
    """Calculate minimum epochs before early stopping kicks to account for
    warm-up steps.
    """
    if args.scheduler in ("linear_warmup", "cosine"):
        min_epochs = math.ceil(0.1 * args.epochs // 2)
        min_epochs += EARLY_STOP_PATIENCE // 2
        return min_epochs
    return 0


def parse_training_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = OGLCLIParser()
    parser.add_gnn_training_args()
    return parser.parse_args()


def main() -> None:
    """Main function to train GNN on graph data!"""
    args = parse_training_args()
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)
    run_dir, logger, tb_logger = _experiment_setup(
        args=args, experiment_config=experiment_config
    )

    # check for GPU
    device, seed = setup_device(args)
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed set to: {seed}")

    # get graph data
    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=args.split_name,
        regression_target=args.target,
        positional_encoding=args.positional_encoding,
    ).make_data_object()

    # check data integreity
    PyGDataChecker.check_pyg_data(data)

    # set up data loaders
    mask_suffix = "_loss" if args.gene_only_loader else ""
    train_loader = prep_loader(
        data=data,
        mask=f"train_mask{mask_suffix}",
        batch_size=args.batch_size,
        shuffle=True,
        layers=args.gnn_layers,
    )
    val_loader = prep_loader(
        data=data,
        mask=f"val_mask{mask_suffix}",
        batch_size=args.batch_size,
        shuffle=False,
        layers=args.gnn_layers,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1],  # all neighbors at each hop
        batch_size=args.batch_size,
        input_nodes=getattr(data, f"test_mask{mask_suffix}"),
        shuffle=False,
    )

    # CHOOSE YOUR WEAPON
    model = build_gnn_architecture(
        model=args.model,
        activation=args.activation,
        in_size=data.x.shape[1],
        embedding_size=args.dimensions,
        out_channels=1,
        gnn_layers=args.gnn_layers,
        shared_mlp_layers=args.linear_layers,
        heads=args.heads,
        dropout_rate=args.dropout or None,
        residual=args.residual,
        attention_task_head=args.attention_task_head,
        train_dataset=train_loader if args.model == "PNA" else None,
    )
    model = model.to(device)

    # log model graph to tensorboard and json
    tb_logger.log_model_graph(
        model=model,
        device=device,
        feature_size=data.x.shape[1],
    )
    _dump_metadata_json(
        args=args,
        experiment_config=experiment_config,
        run_dir=run_dir,
        total_parameters=count_model_parameters(model),
    )

    # set up optimizer & scheduler
    total_steps, warmup_steps = OptimizerSchedulerHandler.calculate_training_steps(
        train_loader=train_loader,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    optimizer = OptimizerSchedulerHandler.set_optimizer(
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        model_params=model.parameters(),
    )
    scheduler = OptimizerSchedulerHandler.set_scheduler(
        scheduler_type=args.scheduler,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        training_steps=total_steps,
    )

    # get min epochs before early stopping kicks in for warm-up
    min_epochs = calculate_min_epochs(args)

    # start model training and initialize tensorboard utilities
    epochs = args.epochs
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(run_dir / "tensorboard")
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    prof.start()

    # initialize trainer
    trainer = GNNTrainer(
        model=model,
        device=device,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        tb_logger=tb_logger,
    )

    logger.info(f"Training for {epochs} epochs (early stopping)")
    model, _, early_stop = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        model_dir=run_dir,
        args=args,
        min_epochs=min_epochs,
    )

    # close out tensorboard utilities and save final model
    prof.stop()
    torch.save(
        model.state_dict(),
        run_dir / f"{args.model}_final_model.pt",
    )

    # generate loss and prediction plots for best model
    post_model_evaluation(
        model_name=args.model,
        model=model,
        device=device,
        data=data,
        data_loader=test_loader,
        tb_logger=tb_logger,
        logger=logger,
        run_dir=run_dir,
        early_stop=early_stop,
    )


if __name__ == "__main__":
    main()
