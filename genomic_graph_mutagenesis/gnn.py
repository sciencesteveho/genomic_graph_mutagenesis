#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] fix layer sizes
# - [ ] add option for different architectures
# - [ ] add hyperparamters as parser args that output to a log
# - [ ] implement tensorboard

"""Code to train GNNs on the graph data!"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer
from torch_geometric.explain import GNNExplainer
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch


# Define/Instantiate GNN model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_size, embedding_size, out_channels, num_layers):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_size, embedding_size))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(embedding_size, embedding_size))
        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx != len(self.convs) - 1:
                x = F.dropout(x, p=0.1, training=self.training)
                x = F.relu(x)
                x = self.lin1(x)
                x = F.relu(x)
                x = self.lin2(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_size, embedding_size, out_channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_size, embedding_size))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(embedding_size, embedding_size))
        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx != len(self.convs) - 1:
                x = F.dropout(x, p=0.1, training=self.training)
                x = F.relu(x)
                x = self.lin1(x)
                x = F.relu(x)
                x = self.lin2(x)
        return x


class GATv2(torch.nn.Module):
    def __init__(self, in_size, embedding_size, out_channels, num_layers, heads):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_size, embedding_size, heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(heads * embedding_size, embedding_size, heads))
        self.convs.append(
            GATv2Conv(heads * embedding_size, out_channels, heads, concat=False)
        )
        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx != len(self.convs) - 1:
                x = F.dropout(x, p=0.1, training=self.training)
                x = F.relu(x)
                x = self.lin1(x)
                x = F.relu(x)
                x = self.lin2(x)
        return x


def train(model, device, optimizer, train_loader, epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:04d}")

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index)
        # loss = F.mse_loss(
        #     out[data.train_mask].squeeze(), data.y[data.train_mask].squeeze()
        # )

        # only calculate on values that are not -1
        target_mask = data.train_mask != -1
        out_data = out[data.train_mask]
        truth = data.y[data.train_mask]
        loss = F.mse_loss(out_data[target_mask].squeeze(), truth[target_mask].squeeze())

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(model, device, test_loader, epoch):
    model.eval()

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    val_mse, test_mse = []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # only calculate on values that are not -1
        target_mask = data.val_mask != -1
        out_vals = out[data.val_mask]
        truth = data.y[data.val_mask]
        val_acc = F.mse_loss(
            out_vals[target_mask].squeeze(), truth[target_mask].squeeze()
        )

        target_mask = data.test_mask != -1
        out_test = out[data.test_mask]
        truth = data.y[data.test_mask]
        test_acc = F.mse_loss(
            out_test[target_mask].squeeze(), truth[target_mask].squeeze()
        )
        # total_test_acc += float(test_acc) * int(data.test_mask.sum())
        # total_val_acc += float(val_acc) * int(data.val_mask.sum())
    pbar.close()
    return float(torch.cat(val_mse, dim=0).mean()), float(
        torch.cat(test_mse, dim=0).mean()
    )


def main() -> None:
    """_summary_"""
    # Parse training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default="2",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default="600",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed to use (default: 42)",
    )
    parser.add_argument(
        "--loader",
        type=str,
        defualt="random",
        help="'neighbor' or 'random' node loader (default: 'random')",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset storage.",
        default="/ocean/projects/bio210019p/stevesho/data/preprocess",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="full",
    )
    args = parser.parse_args()

    # check for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # prepare data
    data = graph_to_pytorch(
        root_dir=args.root,
        graph_type=args.graph_type,
    )

    # data loaders
    if args.loader == "random":
        train_loader = RandomNodeLoader(
            data,
            num_parts=50,
            shuffle=True,
            num_workers=5,
        )
        test_loader = RandomNodeLoader(
            data,
            num_parts=50,
            num_workers=5,
        )

    if args.loader == "neighbor":
        train_loader = NeighborLoader(
            data,
            num_neighbors=[30] * 2,
            batch_size=128,
            input_nodes=data.train_mask,
        )
        test_loader = NeighborLoader(
            data,
            num_neighbors=[30] * 2,
            batch_size=128,
            input_nodes=data.val_mask,
        )

    # choose your weapon
    if args.model == "GraphSage":
        model = GraphSAGE(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=4,
            layers=args.layers,
        ).to(device)
    if args.model == "GCN":
        model = GCN(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=4,
            layers=args.layers,
        ).to(device)
    if args.model == "GAT":
        model = GATv2(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=4,
            layers=args.layers,
            heads=2,
        ).to(device)

    # set gradient descent optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 100
    for epoch in range(0, epochs):
        loss = train(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
        )
        val_acc, test_acc = test(
            model=model,
            device=device,
            test_loader=test_loader,
            epoch=epoch,
        )
        print(f"Epoch: {epoch:03d}, Loss: {loss}")
        print(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}, Test: {test_acc:.4f}")

    torch.save(
        model, f"models/{args.model}_{args.layers}_{args.dimensions}_{args.loader}.pt"
    )

    explainer = Explainer(
        model=model,
        alorigthm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression", task_level="node", return_type="log_probs"
        ),
    )

    node_index = 10
    explanation = explainer(data.x, data.edge_index, index=node_index)
    print(f"Generated explanations in {explanation.available_explanations}")

    path = "feature_importance.png"
    explanation.visualize_feature_importance(path, top_k=10)
    print(f"Feature importance plot has been saved to '{path}'")

    path = "subgraph.pdf"
    explanation.visualize_graph(path)
    print(f"Subgraph visualization plot has been saved to '{path}'")


if __name__ == "__main__":
    main()
