import torch
from model import BotRGCN
from preprocess import generate_hetero_graph_loader, generate_homo_graph_loader_only_user
from tqdm import tqdm
import numpy as np
from Dataset import Twibot20

from eval import evaluate_on_all_metrics
import argparse
print(evaluate_on_all_metrics)
class hetero_Trainer:
    def __init__(
        self,
        epochs=50,
        batch_size=128,
        input_dim=768,
        hidden_dim=128,
        # num_relations=3,
        # num_bases=20,
        num_layers=20,
        num_neighbor_users=20,
        num_neighbor_tweets=20,
        activation="relu",
        dataset="Twibot-20",
        server_id="209",
        device="cuda:0",
        optimizer=torch.optim.Adam,
        weight_decay=1e-5,
        lr=1e-4,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.model = BotRGCN(embedding_dimension=hidden_dim)
        self.train_loader, self.valid_loader, self.test_loader = generate_hetero_graph_loader(num_layers, num_neighbor_users, num_neighbor_tweets, batch_size, False, dataset, server_id)
        #twibot20=Twibot20(root='../../data/',device=device,process=False,save=False)
        
        self.device = torch.device(device)
        self.model.to(self.device)
        self.des_tensor=torch.load('../../data/des_tensor.pt').to(self.device)
        self.tweet_tensor=torch.load('../../data/tweets_tensor.pt').to(self.device)
        self.num_prop=torch.load('../../data/num_properties_tensor.pt').to(self.device)
        self.cat_prop=torch.load('../../data/cat_properties_tensor.pt').to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.loss_func = torch.nn.CrossEntropyLoss()
    
    @torch.no_grad()
    def valid(self):
        preds = []
        labels =[]
        
        for data in self.valid_loader:
            data = data.to(self.device, "edge_index", "x")
            batch_size = data.batch_size
            pred = self.model(self.des_tensor,self.tweet_tensor,self.num_prop,self.cat_prop, data.edge_index,data.edge_type)[:batch_size]
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            labels.append(data.y[:batch_size].numpy())
            
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        print(evaluate_on_all_metrics(labels, preds))
        
        
    @torch.no_grad()
    def test(self):
        preds = []
        labels = []
        for data in self.test_loader:
            data = data.to(self.device, "edge_index", "x")
            batch_size = data.batch_size
            pred = self.model(self.des_tensor,self.tweet_tensor,self.num_prop,self.cat_prop, data.edge_index,data.edge_type)[:batch_size]
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            labels.append(data.y[:batch_size].numpy())
            
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        print(evaluate_on_all_metrics(labels, preds))
    
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            with tqdm(self.train_loader) as progress_bar:
                for data in progress_bar:
                    data = data.to(self.device, "edge_index", "x", "y")
                    batch_size = data.batch_size
                    pred = self.model(self.des_tensor,self.tweet_tensor,self.num_prop,self.cat_prop, data.edge_index,data.edge_type)[:batch_size]
                    loss = self.loss_func(pred, data.y[:batch_size])
                    
                    loss.backward()
                    self.optimizer.zero_grad()
                    self.optimizer.step()
                    
                    progress_bar.set_description(desc=f"Epoch:{epoch}")
                    progress_bar.set_postfix(loss=loss.item())
                    
            self.valid()
            self.test()

#if __name__ == "__main__":
    #trainer = hetero_Trainer()
    #trainer.train()
