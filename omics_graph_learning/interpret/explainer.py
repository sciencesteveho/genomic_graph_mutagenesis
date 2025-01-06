#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementation of PGExplainer.

In contrast to perturbations, PGExplainer learns a generalizable strategy on
graph embeddings and applies them to potential subgraphs. The explanations focus
on subgraph structures rather than specific features as with perturbations.
"""

from typing import List

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.explain import Explainer  # type: ignore
from torch_geometric.explain import PGExplainer  # type: ignore
from torch_geometric.explain.config import ExplanationType  # type: ignore
from torch_geometric.explain.config import MaskType  # type: ignore
from torch_geometric.explain.config import ModelMode  # type: ignore


def build_explainer(model: nn.Module, epochs: int = 30, lr: float = 0.003) -> Explainer:
    """Build a PGExplainer instance. We use ExplanationType.model to explain the
    model's behaviors on the regression task, focused on learning feature and
    edge importances.
    """
    return Explainer(
        model=model,
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type=ExplanationType.model,
        model_mode=ModelMode.regression,
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object,
    )


def train_explainer(explainer: Explainer, data: Data) -> List[torch.Tensor]:
    """Train the explainer running it on the nodes that represent genes."""
    # get indices for the genes that we regress
    gene_mask = data.train_mask_loss | data.val_mask_loss | data.test_mask_loss
    gene_indices = gene_mask.nonzero(as_tuple=True)[0]

    # train the explainer
    explanations = []
    for node_idx in gene_indices:
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            node_idx=node_idx,
        )
        explanations.append(explanation)

    return explanations
