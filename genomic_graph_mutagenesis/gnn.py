#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] implement tensorboard
# - [ ] add dropout as arg

"""Code to train GNNs on the graph data!"""

import argparse
from datetime import datetime
import logging
import math
from typing import Any, Dict, Optional

import torch
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer
from torch_geometric.explain import GNNExplainer
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GPSConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.attention import PerformerAttention
import torch_geometric.transforms as T
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch
from utils import _set_matplotlib_publication_parameters
from utils import _tensor_out_to_array
from utils import dir_check_make
from utils import parse_yaml
from utils import plot_predicted_versus_expected
from utils import plot_training_losses


# Define/Instantiate GNN model
class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
        num_layers,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_size, embedding_size, aggr="sum"))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(embedding_size, embedding_size, aggr="sum"))
            self.batch_norms.append(BatchNorm(embedding_size))

        self.lin1 = nn.Linear(embedding_size, embedding_size)
        # self.lin2 = nn.Linear(embedding_size, out_channels)  # if only using 2 linear layers
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.lin3 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
        num_layers,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(in_size, embedding_size))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(embedding_size, embedding_size))
            self.batch_norms.append(BatchNorm(embedding_size))

        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


class GATv2(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
        num_layers,
        heads,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GATv2Conv(in_size, embedding_size, heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(heads * embedding_size, embedding_size, heads))
            # self.batch_norms.append(BatchNorm(heads * embedding_size))
            self.batch_norms.append(GraphNorm(heads * embedding_size))

        self.lin1 = nn.Linear(heads * embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return x
    
    
class GPSTransformer(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        walk_length: int,
        channels: int,
        pe_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.node_emb = nn.Linear(in_size, embedding_size - pe_dim)
        self.pe_lin = nn.Linear(walk_length, pe_dim)
        self.pe_norm = nn.BatchNorm1d(walk_length)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            gcnconv = GCNConv(embedding_size, embedding_size)
            conv = GPSConv(
                channels,
                gcnconv,
                heads=4,
                attn_kwargs={"dropout": 0.5}
            )
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        
        self.redraw_projection = RedrawProjection(
            self.convs,
            None)

    def forward(self, x, pe, edge_index, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)

        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return self.mlp(x)
    
    
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
        

### baseline MLP
class MLP(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
    ):
        super().__init__()

        self.lin1 = nn.Linear(in_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.lin3 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
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

        # calculate loss
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()
    return total_loss / total_examples


@torch.no_grad()
def test(model, device, data_loader, epoch, mask):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse = []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        print(out)
        print(data.y)

        # calculate loss
        if mask == "val":
            idx_mask = data.val_mask
        if mask == "test":
            idx_mask = data.test_mask
        mse.append(F.mse_loss(out[idx_mask], data.y[idx_mask]).cpu())
        loss = torch.stack(mse)

        pbar.update(1)

    pbar.close()
    return math.sqrt(float(loss.mean()))


@torch.no_grad()
def test_with_idxs(model, device, data_loader, epoch, mask):
    # spearman = SpearmanCorrCoef(num_outputs=2)
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse = []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # calculate loss
        if mask == "val":
            idx_mask = data.val_mask
        if mask == "test":
            idx_mask = data.test_mask
        mse.append(F.mse_loss(out[idx_mask], data.y[idx_mask]).cpu())
        # outs.extend(out[idx_mask])
        # labels.extend(data.y[idx_mask])
        loss = torch.stack(mse)

        pbar.update(1)

    pbar.close()
    # print(spearman(torch.stack(outs), torch.stack(labels)))
    return math.sqrt(float(loss.mean()))


@torch.no_grad()
def inference(model, device, data_loader, epoch):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse, outs, labels = [], [], []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # calculate loss
        outs.extend(out[data.test_mask])
        labels.extend(data.y[data.test_mask])
        mse.append(F.mse_loss(out[data.test_mask], data.y[data.test_mask]).cpu())
        loss = torch.stack(mse)

        pbar.update(1)

    pbar.close()
    # print(spearman(torch.stack(outs), torch.stack(labels)))
    return math.sqrt(float(loss.mean())), outs, labels


def main() -> None:
    """_summary_"""
    # Parse training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
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
        default="neighbor",
        help="'neighbor' or 'random' node loader (default: 'random')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--idx",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--zero_nodes",
        type=str,
        default="false",
    )
    parser.add_argument(
        "--randomize_node_feats",
        type=str,
        default="false",
    )
    parser.add_argument(
        "--early_stop",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--expression_only",
        type=str,
        default="false",
    )
    args = parser.parse_args()

    params = parse_yaml(args.experiment_config)

    # set up helper variables
    working_directory = params["working_directory"]
    root_dir = f"{working_directory}/{params['experiment_name']}"
    savestr = f"{params['experiment_name']}_{args.model}_{args.layers}_{args.dimensions}_{args.learning_rate}_batch{args.batch_size}_{args.loader}_{args.graph_type}_targetnoscale_idx"

    # adjust log name
    if args.randomize_node_feats == "true":
        savestr = f"{savestr}_random_node_feats"
    if args.expression_only == "true":
        savestr = f"{savestr}_expression_only"

    # make directories and set up training log
    dir_check_make(f"{working_directory}/models/logs")
    dir_check_make(f"{working_directory}/models/{savestr}")

    logging.basicConfig(
        filename=f"{working_directory}/models/logs/{savestr}.log",
        level=logging.DEBUG,
    )

    # check for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type=args.graph_type,
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        randomize_feats=args.randomize_node_feats,
        zero_node_feats=args.zero_nodes,
    )
    
    if args.model == "GPS":
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        data = transform(data)

    # data loaders
    if args.loader == "random":
        train_loader = RandomNodeLoader(
            data,
            num_parts=250,
            shuffle=True,
            num_workers=5,
        )
        test_loader = RandomNodeLoader(
            data,
            num_parts=250,
            num_workers=5,
        )

    if args.loader == "neighbor":
        train_loader = NeighborLoader(
            data,
            num_neighbors=[15, 10, 5],
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = NeighborLoader(
            data,
            num_neighbors=[15, 10, 5],
            batch_size=args.batch_size,
        )
        if args.idx == "true":
            train_loader = NeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=args.batch_size,
                input_nodes=data.train_mask,
                shuffle=True,
            )
            test_loader = NeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=args.batch_size,
                input_nodes=data.test_mask,
            )
            val_loader = NeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=args.batch_size,
                input_nodes=data.val_mask,
            )

    # CHOOSE YOUR WEAPON
    if args.model == "GraphSAGE":
        model = GraphSAGE(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
            num_layers=args.layers,
        ).to(device)
    if args.model == "GCN":
        model = GCN(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
            num_layers=args.layers,
        ).to(device)
    if args.model == "GAT":
        model = GATv2(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
            num_layers=args.layers,
            heads=2,
        ).to(device)
    if args.model == "MLP":
        model = MLP(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
        ).to(device)
    if args.model == "GPS":
        model = GPSTransformer(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            walk_length=20,
            channels=64,
            pe_dim=8,
            num_layers=args.layers,
        ).to(device)

    # set gradient descent optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
    #                             min_lr=0.00001)


    epochs = 100
    best_validation = stop_counter = 0
    for epoch in range(0, epochs + 1):
        loss = train(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
        )
        print(f"Epoch: {epoch:03d}, Train: {loss}")
        logging.info(f"Epoch: {epoch:03d}, Train: {loss}")

        if args.idx == "true":
            val_acc = test_with_idxs(
                model=model,
                device=device,
                data_loader=val_loader,
                epoch=epoch,
                mask="val",
            )

            test_acc = test_with_idxs(
                model=model,
                device=device,
                data_loader=test_loader,
                epoch=epoch,
                mask="test",
            )
        else:
            val_acc = test(
                model=model,
                device=device,
                data_loader=test_loader,
                epoch=epoch,
                mask="val",
            )

            test_acc = test(
                model=model,
                device=device,
                data_loader=test_loader,
                epoch=epoch,
                mask="test",
            )
            
        # scheduler.step(val_acc)
        if args.early_stop == "true":
            if epoch == 0:
                best_validation = val_acc
            else:
                if val_acc < best_validation:
                    stop_counter = 0
                    best_validation = val_acc
                    torch.save(
                        model.state_dict(),
                        f"{working_directory}/models/{savestr}/{savestr}_early_epoch_{epoch}_mse_{best_validation}.pt",
                    )
                if best_validation < val_acc:
                    stop_counter += 1
                if stop_counter == 15:
                    print("***********Early stopping!")
                    break

        print(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}")
        logging.info(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}")

        print(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        logging.info(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")

    torch.save(
        model.state_dict(),
        f"{working_directory}/models/{savestr}/{savestr}_mse_{best_validation}.pt",
    )

    # set params for plotting
    _set_matplotlib_publication_parameters()

    # calculate and plot spearmann rho, predictions vs. labels
    # first, load checkpoints
    checkpoint = torch.load(
        f"{working_directory}/models/{savestr}/{savestr}_mse_{best_validation}.pt",
        map_location=torch.device("cuda:" + str(0)),)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # get predictions
    rmse, outs, labels = inference(
        model=model, device=device, data_loader=test_loader, epoch=0
    )

    predictions_median = _tensor_out_to_array(outs, 0)
    labels_median = _tensor_out_to_array(labels, 0)

    experiment_name = params["experiment_name"]
    if args.randomize_node_feats == "true":
        experiment_name = f"{experiment_name}_random_node_feats"
    if args.zero_nodes == "true":
        experiment_name = f"{experiment_name}_zero_node_feats"

    # plot performance
    plot_predicted_versus_expected(
        expected=labels_median,
        predicted=predictions_median,
        experiment_name=experiment_name,
        model=args.model,
        layers=args.layers,
        width=args.dimensions,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        outdir=f"{working_directory}/models/plots",
        rmse=rmse,)

    # plot training losses
    plot_training_losses(
        log=f"{working_directory}/models/logs/{savestr}.log",
        experiment_name=experiment_name,
        model=args.model,
        layers=args.layers,
        width=args.dimensions,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        outdir=f"{working_directory}/models/plots",
    )

    # # GNN Explainer!
    # if args.model != "MLP":
    #     with open(
    #         "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/explainer_node_ids.pkl",
    #         "rb",
    #     ) as file:
    #         ids = pickle.load(file)
    #     explain_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/explainer"
    #     explainer = Explainer(
    #         model=model,
    #         algorithm=GNNExplainer(epochs=200),
    #         explanation_type="model",
    #         node_mask_type="attributes",
    #         edge_mask_type="object",
    #         model_config=dict(mode="regression", task_level="node", return_type="raw"),
    #     )

    #     data = data.to(device)
    #     for index in random.sample(ids, 5):
    #         explanation = explainer(data.x, data.edge_index, index=index)

    #         print(f"Generated explanations in {explanation.available_explanations}")

    #         path = f"{explain_path}/feature_importance_{savestr}_{best_validation}.png"
    #         explanation.visualize_feature_importance(path, top_k=10)
    #         print(f"Feature importance plot has been saved to '{path}'")

    #         path = f"{explain_path}/subgraph_{savestr}_{best_validation}.pdf"
    #         explanation.visualize_graph(path)
    #         print(f"Subgraph visualization plot has been saved to '{path}'")


if __name__ == "__main__":
    main()
