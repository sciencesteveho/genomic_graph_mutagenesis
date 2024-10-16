#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""GNN training optimizers and schedulers"""


import math
from typing import Iterator, Tuple, Union

import torch
from torch.nn import Parameter
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore


class OptimizerSchedulerHandler:
    """Organizer class for selecting optimizer, scheduler, and calculating
    training sets for proper warm-up. The class provides a set of static methods
    and should be used as such as opposed to instantiating an object.

    Methods
    --------
    set_scheduler:
        Returns the selected learning rate scheduler.
    set_optimizer:
        Returns the selected gradient descent optimizer.
    calculate_training_steps:
        Calculate the total number of projected warm-up and training steps.

    Examples:
    --------
    # First, calculate the total number of training steps and warm-up steps
    >>> total_steps, warmup_steps = OptimizerSchedulerHandler.calculate_training_steps(
            train_loader=train_loader,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    # Call the optimizer and scheduler
    >>> optimizer = OptimizerSchedulerHandler.set_optimizer(
            optimizer_type=args.optimizer,
            learning_rate=args.learning_rate,
            model_params=model.parameters(),
        )
    >>> scheduler = OptimizerSchedulerHandler.set_scheduler(
            scheduler_type=args.scheduler,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            training_steps=total_steps,
        )
    """

    @classmethod
    def set_scheduler(
        cls,
        scheduler_type: str,
        optimizer: Optimizer,
        warmup_steps: int,
        training_steps: int,
    ) -> Union[LRScheduler, ReduceLROnPlateau]:
        """Set learning rate scheduler"""
        if scheduler_type == "plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )
        if scheduler_type == "cosine":
            return cls._get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=training_steps,
            )
        elif scheduler_type == "linear_warmup":
            return cls._get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=training_steps,
            )
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    @staticmethod
    def set_optimizer(
        optimizer_type: str, learning_rate: float, model_params: Iterator[Parameter]
    ) -> Optimizer:
        """Choose gradient descent optimizer."""
        if optimizer_type == "Adam":
            return torch.optim.Adam(
                model_params,
                lr=learning_rate,
            )
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(
                model_params,
                lr=learning_rate,
                weight_decay=0.02,
            )
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    @staticmethod
    def calculate_training_steps(
        train_loader: torch_geometric.data.DataLoader,
        batch_size: int,
        epochs: int,
        warmup_ratio: float = 0.1,
    ) -> Tuple[int, int]:
        """Calculate the total number of projected training steps to provide a
        percentage of warm-up steps.
        """
        steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        return total_steps, warmup_steps

    @staticmethod
    def _get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 1.0,
        last_epoch: int = -1,
    ) -> LRScheduler:
        """
        Adapted from HuggingFace:
        https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

        Create a scheduler with a learning rate that decreases following the values
        of the cosine function between the initial lr set in the optimizer to 0,
        after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.

        Returns:
            an instance of LambdaLR.
        """

        def lr_lambda(current_step: int) -> float:
            """Compute the learning rate lambda value based on current step."""
            if current_step < num_warmup_steps:
                return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
            )

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_linear_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> LRScheduler:
        """Creates a learning rate scheduler with linear warmup where the learning
        rate increases linearly from 0 to the initial optimizer learning rate over
        `num_warmup_steps`. After completing the warmup, the learning rate will
        decrease from the initial optimizer learning rate to 0 linearly over the
        remaining steps until `num_training_steps`.
        """

        def lr_lambda(current_step: int) -> float:
            """Compute the learning rate lambda value based on current step."""
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                1.0
                - (
                    float(current_step - num_warmup_steps)
                    / float(max(1, num_training_steps - num_warmup_steps))
                ),
            )

        return LambdaLR(optimizer, lr_lambda)
