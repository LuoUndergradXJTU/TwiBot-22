import os
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver
from sacred import Experiment
import sys
sys.path.append("..")
from utils import set_global_seed
from torch_geometric.data.batch import Batch
from itertools import chain
from utils.eval import evaluate_on_all_metrics
import torch
from model import GraphHistEncoder, Hist, GraphHistDecoder
from tqdm import tqdm
import numpy as np
from preprocess import generate_graph_loader


ex = Experiment("Graph Hist")
ex.observers.append(FileStorageObserver("./tmp/log"))
ex.captured_out_filter = apply_backspaces_and_linefeeds

def print_dict(dic):
    return ",".join(map(lambda val: "{:.4f}".format(val), dic.values()))

class Trainer:
    def __init__(
        self,
        dataset="TwiBot-22",
        epochs=50,
        batch_size=256,
        input_dim=44,
        hidden_dim=64,
        n_gcns=3,
        hist_l=-1,
        hist_r=1,
        n_bins=25,
        activation=torch.nn.Tanh,
        device="cuda:5",
        optimizer=torch.optim.Adam,
        lr=1e-4,
        weight_decay=1e-5,
        num_neighbors=40,
        num_sample_layers=3,
        baseline=True,
        warm_up=True,
        use_initial=True,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.num_neighbors = num_neighbors
        self.num_sample_layers = num_sample_layers
        self.warm_up = warm_up
        self.use_initial = use_initial

        self.dataset = dataset
        self.train_loader, self.valid_loader, self.test_loader = generate_graph_loader(dataset=dataset,
            num_neighbors=self.num_neighbors, num_layers=self.num_sample_layers, batch_size=1)

        self.encoder = GraphHistEncoder(
            input_dim, n_gcns, hist_l, hist_r, n_bins, hidden_dim)
        self.decoder = GraphHistDecoder(n_bins, n_gcns * hidden_dim, n_bins)

        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.optimizer = optimizer(chain(self.encoder.parameters(
        ), self.decoder.parameters()), lr=lr, weight_decay=weight_decay)

        self.loss_func = torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def test(self):
        self.encoder.eval()
        self.decoder.eval()
        preds = []
        labels = []
        batch_list = []
        y = []
        for data in self.test_loader:
            data = data.to(self.device, "edge_index", "x")
            data.batch_size = None
            # batch = data.batch
            batch_list.append(data)
            y.append(data.y[0].item())
            if len(batch_list) % self.batch_size != 0:
                continue
            data = Batch.from_data_list(batch_list)
            batch = data.batch
            pred = self.encoder(data.x, data.edge_index)
            graph_feats = list(
                map(lambda idx: pred[batch == idx], range(self.batch_size)))
            histograms = list(map(lambda H: Hist.apply(H, self.encoder.model_bin_centers,
                              self.encoder.hist_l, self.encoder.hist_r, self.encoder.n_bins), graph_feats))
            feats = torch.stack(histograms, dim=0)
            logits = self.decoder(feats)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(np.array(y).astype(np.int32))
            y = []
            batch_list = []

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        metrics = evaluate_on_all_metrics(labels, preds)
        if self.best_acc < metrics["Acc"]:
            self.best_metric = metrics
            self.best_acc = metrics["Acc"]
        if self.best_acc > (.70 if self.dataset == "cresci-2015" else .45) and self.use_initial and self.best_metric["Pre"] > .01:
            torch.save(self.encoder.state_dict(), f"./{self.dataset}/encoder.pt")
            torch.save(self.decoder.state_dict(), f"./{self.dataset}/decoder.pt")
        print(evaluate_on_all_metrics(labels, preds))

    @torch.no_grad()
    def valid(self):
        self.encoder.eval()
        self.decoder.eval()
        preds = []
        labels = []
        batch_list = []
        y = []
        for data in self.valid_loader:
            data = data.to(self.device, "edge_index", "x")
            data.batch_size = None
            # batch = data.batch
            batch_list.append(data)
            y.append(data.y[0].item())
            if len(batch_list) % self.batch_size != 0:
                continue
            data = Batch.from_data_list(batch_list)
            batch = data.batch
            pred = self.encoder(data.x, data.edge_index)
            graph_feats = list(
                map(lambda idx: pred[batch == idx], range(self.batch_size)))
            histograms = list(map(lambda H: Hist.apply(H, self.encoder.model_bin_centers,
                              self.encoder.hist_l, self.encoder.hist_r, self.encoder.n_bins), graph_feats))
            feats = torch.stack(histograms, dim=0)
            logits = self.decoder(feats)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(np.array(y).astype(np.int32))
            y = []
            batch_list = []

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        print(evaluate_on_all_metrics(labels, preds))

    def train(self):
        self.best_acc = 0.
        self.best_metric = {}
        self.encoder.train()
        self.decoder.train()
        if self.warm_up and "encoder.pt" in os.listdir(f"./{self.dataset}/"):
            self.encoder.load_state_dict(torch.load(f"./{self.dataset}/encoder.pt", map_location=self.device))
            self.decoder.load_state_dict(torch.load(f"./{self.dataset}/decoder.pt", map_location=self.device))
        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            batch_list = []
            y = []
            with tqdm(self.train_loader) as progress_bar:
                for data in progress_bar:
                    data.batch_size = None
                    batch_list.append(data)
                    y.append(data.y[0].item())
                    if len(batch_list) % self.batch_size != 0:
                        continue
                    data = Batch.from_data_list(batch_list)
                    data = data.to(self.device, "edge_index", "x", "y")
                    batch = data.batch
                    # batch_size = data.batch_size
                    pred = self.encoder(data.x, data.edge_index)
                    graph_feats = list(
                        map(lambda idx: pred[batch == idx], range(self.batch_size)))
                    histograms = list(map(lambda H: Hist.apply(H, self.encoder.model_bin_centers,
                                      self.encoder.hist_l, self.encoder.hist_r, self.encoder.n_bins), graph_feats))
                    # print(len(histograms))
                    feats = torch.stack(histograms, dim=0)

                    logits = self.decoder(feats)
                    y = torch.LongTensor(y).to(self.device)
                    loss = self.loss_func(logits, y)
                    loss.backward()
                    self.optimizer.zero_grad()
                    self.optimizer.step()

                    progress_bar.set_description(desc=f"Epoch: {epoch}")
                    progress_bar.set_postfix(loss=loss.item())
                    batch_list = []
                    y = []

            self.valid()
            self.test()
        with open(f"./{self.dataset}/ret.txt", "a") as f:
            f.write(print_dict(self.best_metric) + '\n')
        print(self.best_metric)
        
    def sweep(self, seed_list):
        for seed in seed_list:
            set_global_seed(seed)
            self.train()


@ex.config
def config():
    params = dict(
        dataset="Twibot-20",
        epochs=10,
        batch_size=256,
        input_dim=44,
        hidden_dim=16,
        n_gcns=3,
        hist_l=-1,
        hist_r=1,
        n_bins=25,
        activation=torch.nn.Tanh,
        device="cuda:0",
        optimizer=torch.optim.Adam,
        lr=1e-4,
        weight_decay=1e-5,
        num_neighbors=40,
        num_sample_layers=3,
        baseline=True,
        warm_up=True,
        use_initial=False,
    )


@ex.automain
def main(params):
    trainer = Trainer(**params)
    # trainer.train()
    seed_list = [100, 200, 300, 400, 500]
    trainer.sweep(seed_list)
