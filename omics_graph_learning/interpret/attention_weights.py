# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Module to extract attention weights from a trained GATv2 models. The models
are trained with (return_attention_weights=False), so we:
    1) Set each GATv2Conv's return_attention_weights=True.
    2) Register a forward hook that captures the alpha values.
    3) Perform a forward pass with.
    4) Return a {layer_name: alpha_tensor} dictionary."""


from typing import Dict, Optional

import torch  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.nn import GATv2Conv  # type: ignore


def register_gat_hooks(model: torch.nn.Module) -> None:
    """Registers a forward hook on GAT layers to capture attention weights in
    the return of the forward pass.
    """

    def gat_attention_hook(
        module: GATv2Conv,
        output: torch.Tensor,
    ) -> None:
        """Forward with return_attention_weights=True yields:
        output = (out, (edge_index, alpha))
        """
        if isinstance(output, tuple) and len(output) == 2:
            # output is (out, (edge_index, alpha))
            _, alpha_info = output
            if len(alpha_info) == 2:
                _, alpha = alpha_info
                module.cached_alpha = alpha.detach().cpu()
            else:
                module.cached_alpha = None
        else:
            module.cached_alpha = None

    for _, submodule in model.named_modules():
        if isinstance(submodule, GATv2Conv):
            submodule.register_forward_hook(gat_attention_hook)


def get_attention_weights(
    model: torch.nn.Module,
    data: Data,
    mask: torch.Tensor,
) -> Dict[str, Optional[torch.Tensor]]:
    """Force gat layers to return attention weights and capture them in a
    forward hook.
    """
    # force GAT layers to preturn attention weights
    for __build_class__, submodule in model.named_modules():
        if isinstance(submodule, GATv2Conv):
            submodule.return_attention_weights = True

    # capture attention weights
    register_gat_hooks(model)

    # forward pass
    with torch.no_grad():
        _ = model(
            x=data.x,
            edge_index=data.edge_index,
            mask=mask,
        )

    return {
        name: getattr(submodule, "cached_alpha", None)
        for name, submodule in model.named_modules()
        if isinstance(submodule, GATv2Conv)
    }
