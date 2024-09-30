# sourcery skip: avoid-single-character-names-variables, no-complex-if-expressions, upper-camel-case-classes
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""GNN model architectures for node regression. We adopt a base class for the
GNN architecture space. Specific GNN models inherit from the baseclass to
allow flexible and module GNN design. Model args specific for each convolutional
operator are hardcoded into the class definition.

The models include four different classes of architectures:

(1) Classic message passing neural networks (MPNNs): 
    Graph convolutional networks (GCN)
    GraphSAGE
    Principle neighborhood aggregation (PNA - MPNN w/ aggregators and scalers)

(2) Attention-based models:
    Graph attention networks V2 (GATv2)
    UniMPTransformer (GCN with UniMP transformer operator)

(3) Large-scale models (GNNs designed for deep layers):
    DeeperGCN

(4) Baseline: 3-layer MLP
"""


from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Dropout
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv  # type: ignore
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GENConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import PNAConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models import DeepGCNLayer  # type: ignore


class AttentionTaskHead(nn.Module):
    """A node regression task head augmented with multiheaded attention. A
    single linear layer with residual, dropout, and layer normalization to be
    used after graph convolutions and X linear layers.

    Attributes:
        embedding_size: hidden size
        out_channels: output size (probably 1)
        num_heads: number of attention heads (default: `4`)
        dropout_rate: % neurons to drop out (default: `0.1`)
    """

    def __init__(
        self,
        embedding_size: int,
        out_channels: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize an MLP with multiheaded attention."""
        super().__init__()
        self.attention_layer = nn.MultiheadAttention(
            embedding_size, num_heads, batch_first=True, dropout=dropout_rate
        )
        self.norm = LayerNorm(embedding_size)
        self.linear = Linear(embedding_size, out_channels)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        # compute multiheaded attention
        attn_out, _ = self.attention_layer(x, x, x)

        # residual, normalization, and dropout
        x = self.dropout(self.norm(x + attn_out))

        # output
        return self.linear(x)


class MLP(nn.Module):
    """Simple three layer MLP model architecture.

    Attributes:
        in_size: The input size of the MLP model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        activation: String specifying the activation function between linear
        layers. Defaults to `relu`.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        activation: str = "relu",
    ):
        """Initialize the MLP model."""
        super().__init__()
        self.linears = nn.ModuleList(
            [
                Linear(in_size, embedding_size),
                Linear(embedding_size, embedding_size),
                Linear(embedding_size, out_channels),
            ]
        )
        self.activation = get_activation_function(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))

        return self.linears[-1](x)


class ModularGNN(nn.Module):
    """Highly modular, flexible GNN model architecture.

    We opt for a GNN solution space as opposed to a static architecture. The
    implemented class allows for modular construction of GNN models with
    different convolutional operators, convolutional, linear layers, residual
    connection types, and task heads.

    This model is design to regress log2 values of gene expression (RNA-seq) or
    log2 tpm. Thus, the distribution is bimodal, so the model employs an
    auxiliary task to simultaneously predict the class of each target (classes
    are split via the bimodality of the targets, log2 tpm < 0 and log2 tpm >=
    0).

    Attributes:
        embedding_size: Hidden size / width.
        out_channels: Output size (likely 1).
        dropout_rate: Percentage of neurons to drop out.
        residual: Type of residual connection to use.
        shared_mlp_layers: Number of linear layers, not including the task head.
        attention_task_head: Boolean indicating if the task head should use
        attention.
        activation: The activation function to use, from `relu`, `leakyrelu`,
        `gelu`.
        heads: Number of attention heads, only for attention-based models.
        convs: List of convolutional layers.
        norms: List of normalization layers.
        linears: List of linear layers.
        layer_norms: List of layer normalization layers.
        task_head: The task head for the model.
        linear_projection: Linear projection for residual connection.
    """

    def __init__(
        self,
        activation: str,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        gnn_layers: int,
        shared_mlp_layers: int,
        gnn_operator_config: Dict[str, Any],
        dropout_rate: Optional[float] = None,
        residual: Optional[str] = None,
        attention_task_head: bool = False,
    ):
        """Initialize the model"""
        super().__init__()
        self.embedding_size = embedding_size
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.shared_mlp_layers = shared_mlp_layers
        self.attention_task_head = attention_task_head
        self.activation_name = activation
        self.activation = get_activation_function(activation)

        # add attention heads if specified in config
        self.heads = gnn_operator_config.get("heads")

        # initialize GNN layers and batch normalization
        self.convs = self._create_gnn_layers(
            in_size=in_size,
            embedding_size=embedding_size,
            layers=gnn_layers,
            config=gnn_operator_config,
        )

        # initialize normalization layers
        self.norms = self._get_normalization_layers(
            layers=gnn_layers, embedding_size=embedding_size, heads=self.heads
        )

        # initialize fully connected layers
        self.linears, self.layer_norms = self._get_linear_layers(
            in_size=embedding_size,
            layers=shared_mlp_layers,
            heads=self.heads,
        )

        # initialize task head(s)
        self.regression_task_head: Union[AttentionTaskHead, nn.Linear]
        self.classification_task_head: Union[AttentionTaskHead, nn.Linear]
        if self.attention_task_head:
            self.regression_task_head = AttentionTaskHead(
                embedding_size=embedding_size,
                out_channels=out_channels,
            )
            self.classification_task_head = AttentionTaskHead(
                embedding_size=embedding_size,
                out_channels=1,  # binary classification
            )
        else:
            self.regression_task_head = self._general_task_head(
                in_size=embedding_size, out_size=out_channels
            )
            self.classification_task_head = self._general_task_head(
                in_size=embedding_size, out_size=1
            )

        # create linear projection for residual connections
        self.linear_projection = None
        if self.residual:
            self.linear_projection = self._residuals(
                in_size=in_size,
                embedding_size=(
                    self.heads * embedding_size if self.heads else embedding_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the neural network.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.
            mask: Boolean mask of shape (num_nodes,) indicating which
            nodes should be regressed. True for nodes to be regressed, False
            otherwise.

        Returns:
            torch.Tensor: The output tensor.
        """
        try:
            h1 = x  # save input for residual connections

            # graph convolutions with normalization and optional residual connections.
            for i, (conv, batch_norm) in enumerate(zip(self.convs, self.norms)):
                if self.linear_projection:
                    if isinstance(self.linear_projection, nn.Linear):
                        residual = (
                            self.linear_projection(h1) if conv == self.convs[0] else x
                        )
                        x = self.activation(batch_norm(conv(x, edge_index))) + residual
                    elif isinstance(self.linear_projection, nn.ModuleList):
                        residual = self.linear_projection[i](
                            h1 if conv == self.convs[0] else x
                        )
                        x = self.activation(batch_norm(conv(x, edge_index))) + residual
                else:
                    x = self.activation(batch_norm(conv(x, edge_index)))

                # check for NaN values
                if torch.isnan(x).any():
                    print(f"Warning: NaN detected in layer {i}")

            # fully connected layers
            x = apply_mlp_layers(
                x=x,
                linear_layers=self.linears,
                layer_norms=self.layer_norms,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                training=self.training,
            )

            # apply regression task head
            regression_out = compute_masked_output(
                task_head=self.regression_task_head,
                x=x,
                mask=mask,
            )

            # apply classification task head
            logits = compute_masked_output(
                task_head=self.classification_task_head,
                x=x,
                mask=mask,
            )

            return regression_out, logits
        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(
                    f"CUDA error detected in forward pass for model {self.__class__.__name__}"
                )
                print(f"Input shapes: x={x.shape}, edge_index={edge_index.shape}")
                print(f"Model state: {self.state_dict().keys()}")
            raise e

    def _general_task_head(
        self,
        in_size: int,
        out_size: int,
    ) -> nn.Linear:
        """Create a general task head for the model, a single linear layer that
        regresses to the output size.
        """
        return nn.Linear(in_size, out_size)

    def _residuals(
        self, in_size: int, embedding_size: int
    ) -> Union[nn.ModuleList, nn.Linear]:
        """Create residual connection linear projections. The skip connection
        type can be specificed as shared_source (from the input) or
        distinct_source (from the output of each layer).
        """
        if self.residual == "shared_source":
            return nn.Linear(in_size, embedding_size, bias=False)
        elif self.residual == "distinct_source":
            return nn.ModuleList(
                [
                    nn.Linear(
                        in_size if i == 0 else embedding_size,
                        embedding_size,
                        bias=False,
                    )
                    for i in range(len(self.convs))
                ]
            )
        else:
            raise ValueError(
                "Invalid skip connection type: must be `shared_source` or `distinct_source`."
            )

    def _get_linear_layers(
        self,
        in_size: int,
        layers: int,
        heads: Optional[int] = None,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """Create linear layers for the model along with optional linear
        projection for residual connections. Uses the last size of the linear
        dimensions as the first size will be * heads for attention models.

        Args:
            in_size: The input size of the linear layers.
            out_size: The output size of the final layer.
            gnn_layers: The number of linear layers.

        Returns:
            nn.ModuleList: The module list containing the linear layers.
        """
        sizes = self._linear_layer_dimensions(in_size, layers, heads)
        last_size = sizes[-1]
        return self._linear_module(sizes), nn.ModuleList(
            [LayerNorm(last_size) for _ in sizes]
        )

    @staticmethod
    def _create_gnn_layers(
        in_size: int, embedding_size: int, layers: int, config: dict
    ) -> nn.ModuleList:
        """Create graph convolutional layers and normalization layers for GNN
        models.

        Args:
            in_size (int): Size of the input tensor.
            embedding_size (int): Size of hidden dimensions.
            layers (int): Number of convolutional layers.
            config (dict): Configuration for operator specific params.

        Returns:
            nn.ModuleList: Module list containing the convolutional layers.
        """
        convs = nn.ModuleList()

        operator = config.pop("operator")
        base_config = {
            "out_channels": embedding_size,
            **config,
        }

        for i in range(layers):
            layer_config = {
                "in_channels": (
                    in_size
                    if i == 0
                    else (
                        config["heads"] * embedding_size
                        if "heads" in config
                        else embedding_size
                    )
                ),
            } | base_config
            convs.append(operator(**layer_config))
        return convs

    @staticmethod
    def _get_normalization_layers(
        layers: int,
        embedding_size: int,
        heads: Optional[int] = None,
    ) -> nn.ModuleList:
        """Create GraphNorm normalization layers for the model."""
        embedding_size = embedding_size * heads if heads else embedding_size
        return nn.ModuleList([GraphNorm(embedding_size) for _ in range(layers)])

    @staticmethod
    def _linear_layer_dimensions(
        embedding_size: int,
        linear_layers: int,
        heads: Optional[int] = None,
    ) -> List[int]:
        """Get the sizes of the linear layers for models w/ attention heads."""
        if heads:
            return [heads * embedding_size] + [embedding_size] * linear_layers
        return [embedding_size] * (linear_layers + 1)

    @staticmethod
    def _linear_module(sizes: List[int]) -> nn.ModuleList:
        """Create a linear layer module.

        Args:
            sizes: A list of sizes for the linear layers.
        """
        return nn.ModuleList(
            [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )


class DeeperGCN(nn.Module):
    """DeeperGCN model architecture. Does not inherit from the ModularGNN class
    as it uses the DeepGCNLayer as well as a different initial layer and
    unshared params and activations. residual connections are not coded into the
    forward pass because they're inherent to the architecture and handled by the
    DeepGCNLayer class.

    Args:
        in_size: The input size of the DeeperGCN model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        gnn_layers: The number of convolutional layers.
        linear_layers: The number of linear layers.
        activation: The activation function.
        dropout_rate: The dropout rate.
        task_specific_mlp: Boolean indicating if task specific MLPs are used.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        gnn_layers: int,
        linear_layers: int,
        activation: str,
        attention_task_head: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        """Initialize the model"""
        super().__init__()
        self.activation_name = activation
        self.attention_task_head = attention_task_head
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.activation = get_activation_function(activation)
        self.layer_act = self.nonfunctional_activation(activation)

        # create deeper layers
        self.convs.append(
            self.get_deepergcn_layers(
                in_channels=in_size, out_channels=embedding_size, layer_number=1
            )
        )
        for layer_number in range(2, gnn_layers + 1):
            self.convs.append(
                self.get_deepergcn_layers(
                    in_channels=embedding_size,
                    out_channels=embedding_size,
                    layer_number=layer_number,
                )
            )

        # create linear layers
        self.linears.append(Linear(embedding_size, embedding_size))
        self.layer_norms.append(LayerNorm(embedding_size))

        for _ in range(linear_layers - 2):
            self.linears.append(Linear(embedding_size, embedding_size))
            self.layer_norms.append(LayerNorm(embedding_size))

        # specify task head
        self.regression_task_head: Union[AttentionTaskHead, nn.Linear]
        self.classification_task_head: Union[AttentionTaskHead, nn.Linear]
        if attention_task_head:
            self.regression_task_head = AttentionTaskHead(
                embedding_size=embedding_size,
                out_channels=out_channels,
            )
            self.classification_task_head = AttentionTaskHead(
                embedding_size=embedding_size,
                out_channels=1,  # binary classification
            )
        else:
            self.regression_task_head = Linear(embedding_size, out_channels)
            self.classification_task_head = Linear(embedding_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the neural network.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index)

        # shared linear layers
        x = apply_mlp_layers(
            x=x,
            linear_layers=self.linears,
            layer_norms=self.layer_norms,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )

        # regression task
        regression_out = compute_masked_output(
            task_head=self.regression_task_head, x=x, mask=mask
        )

        # classification task
        logits = compute_masked_output(
            task_head=self.classification_task_head, x=x, mask=mask
        )

        return regression_out, logits

    def get_deepergcn_layers(
        self,
        in_channels: int,
        out_channels: int,
        layer_number: int,
    ) -> DeepGCNLayer:
        """Create DeeperGCN layers"""
        conv = GENConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )
        norm = LayerNorm(out_channels, affine=True)
        act = self.layer_act()
        return DeepGCNLayer(
            conv=conv,
            norm=norm,
            act=act,
            block="res+" if layer_number > 1 else "plain",
            dropout=self.dropout_rate or 0.1,
            ckpt_grad=1 if layer_number == 1 else (layer_number % 3),
        )

    @staticmethod
    def nonfunctional_activation(activation: str) -> Callable:
        """Defines the activation function according to the given string"""
        activations = {
            "gelu": torch.nn.GELU,
            "leakyrelu": torch.nn.LeakyReLU,
            "relu": torch.nn.ReLU,
        }
        try:
            return activations[activation]
        except ValueError as error:
            raise ValueError(
                "Invalid activation function. Supported: relu, leakyrelu, gelu"
            ) from error


class GCN(ModularGNN):
    """Classic Graph Convolutional Network (GCN) model architecture."""

    def __init__(self, *args: Any, **kwargs: Any):
        gcn_config = {
            "operator": GCNConv,
        }
        kwargs["gnn_operator_config"] = {
            **gcn_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class GraphSAGE(ModularGNN):
    """GraphSAGE model architecture."""

    def __init__(self, *args: Any, **kwargs: Any):
        sage_config = {
            "operator": SAGEConv,
            "aggr": "mean",
        }
        kwargs["gnn_operator_config"] = {
            **sage_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class PNA(ModularGNN):
    """GCN architecture with multiple aggregators (Principle Neighborhood
    Aggregation).
    """

    def __init__(self, deg: torch.Tensor, *args: Any, **kwargs: Any):
        self.deg = deg
        pna_config = {
            "operator": PNAConv,
            "aggregators": ["mean", "min", "max"],
            "scalers": ["identity", "amplification", "attenuation"],
            "deg": self.deg,
            "towers": 2,
            "divide_input": False,
        }
        kwargs["gnn_operator_config"] = {
            **pna_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class GATv2(ModularGNN):
    """Graph Attention Network (GAT) model architecture using the V2 operator
    using dynamic attention.

    The dimensions should look like:
        # create graph convolution layers
        for i in range(gnn_layers):
            in_channels = in_size if i == 0 else heads * embedding_size
            self.convs.append(GATv2Conv(in_channels, embedding_size, heads))
            self.norms.append(GraphNorm(heads * embedding_size))
    """

    def __init__(self, heads: int, *args: Any, **kwargs: Any):
        gat_config = {
            "operator": GATv2Conv,
            "heads": heads,
        }
        kwargs["gnn_operator_config"] = {
            **gat_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class UniMPTransformer(ModularGNN):
    """UniMP Transformer model architecture.

    The dimensions should be exactly the same as the GATv2 model, especially as
    we do not pass an argument for concat. It will default to `true` and provide
    and output size of `heads * embedding_size`.
    """

    def __init__(self, heads: int, *args: Any, **kwargs: Any):
        transformer_config = {
            "operator": TransformerConv,
            "heads": heads,
        }
        kwargs["gnn_operator_config"] = {
            **transformer_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


def get_activation_function(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Defines the activation function according to the given string"""
    activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "gelu": F.gelu,
        "leakyrelu": F.leaky_relu,
        "relu": F.relu,
    }
    try:
        return activations[activation]
    except KeyError as error:
        raise KeyError(
            "Invalid activation function. Supported: relu, leakyrelu, gelu"
        ) from error


def compute_masked_output(
    task_head: Union[AttentionTaskHead, nn.Linear],
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the task head to produce the output tensor with task applied only
    to the specified indices.
    """
    # process only task nodes
    indices = indices_tensor(mask).to(x.device)

    # return a tensor of zeros if no regression nodes
    if indices.numel() == 0:
        return torch.zeros(x.size(0), device=x.device)

    out = task_head(x[indices])
    return output_tensor(x=x, out=out, indices=indices)


def output_tensor(
    x: torch.Tensor, out: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Produce the output tensor for the model. Creates a tensor of output shape
    with all zeroes, before filling in values for the indices.
    """
    full_out = torch.zeros(x.size(0), device=x.device)
    out = out.to(x.device)

    # fill in the outputs only for indices that exist
    if indices.numel() > 0:
        full_out[indices] = out.squeeze()

    return full_out


def indices_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Get the indices of the task nodes from the mask."""
    return torch.where(mask)[0]


def apply_mlp_layers(
    x: torch.Tensor,
    linear_layers: nn.ModuleList,
    layer_norms: nn.ModuleList,
    activation: Callable[[torch.Tensor], torch.Tensor],
    training: bool = True,
    dropout_rate: Optional[float] = None,
) -> torch.Tensor:
    """Apply linear layers, normalization, activation, and optional dropout."""
    for linear_layer, layer_norm in zip(linear_layers, layer_norms):
        x = activation(layer_norm(linear_layer(x)))
        if isinstance(dropout_rate, float):
            x = F.dropout(x, p=dropout_rate, training=training)
    return x
