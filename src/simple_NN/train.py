import torch
import numpy as np
from preprocess import get_dataloaders
from model import twoLayerMLP
from tqdm import tqdm

import sys
sys.path.append("..")
from utils.eval import evaluate_on_all_metrics

class Trainer:
    def __init__(
        self,
        epochs=50,
        batch_size=128,
        hidden_dim=256,
        activation="relu",
        dataset="cresci-2017",
        server_id="209",
        device="cuda:0",
        optimizer="Adam",
        lr=1e-4,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(batch_size, dataset, server_id)
        self.model = twoLayerMLP(self.train_loader.dataset[0][0].size(0), 2, hidden_dim=hidden_dim, activation=activation).to(device)
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.device = torch.device(device)
        
        
    @torch.no_grad()
    def valid(self):
        preds = []
        gt = []
        
        for data, labels in self.valid_loader:
            data = data.to(self.device)
            pred = self.model(data)
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            gt.append(labels.numpy())
            
        preds = np.concatenate(preds, axis=0)
        gt = np.concatenate(gt, axis=0)
        print(evaluate_on_all_metrics(gt, preds))
        
    @torch.no_grad()
    def test(self):
        preds = []
        gt = []
        for data, labels in self.test_loader:
            data = data.to(self.device)
            pred = self.model(data)
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            gt.append(labels.numpy())
            
        preds = np.concatenate(preds, axis=0)
        gt = np.concatenate(gt, axis=0)
        print(evaluate_on_all_metrics(gt, preds))
        
    def train(self):
        for epoch in range(self.epochs):
            with tqdm(self.train_loader) as progress_bar:
                for data, labels in progress_bar:
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    pred = self.model(data)
                    loss = self.loss_func(pred, labels)
                    
                    loss.backward()
                    self.optimizer.zero_grad()
                    self.optimizer.step()
                    
                    progress_bar.set_description(desc=f"Epoch:{epoch}")
                    progress_bar.set_postfix(loss=loss.item())
                    
            self.valid()
            self.test()
            
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()