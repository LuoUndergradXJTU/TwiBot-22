"""
CS7643 Instructions for Chaeyoung/Michael
1. Create a new config file like in './config/1.yaml' and rename is as '2.yaml'.
2. Navigate to ./simple_GNN and run 'python3 train.py --config 2.yaml'
3. You will then see the output of your config appearing in output:
   - A set of training, validation and test metrics.
   - A plot of the training and validation loss.
4. Feel free to add/remove configuration settings that you'd like to change.
"""
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import torch
    import numpy as np
    from preprocess import get_dataloaders
    from model import twoLayerMLP, threeLayerMLP
    from tqdm import tqdm

    import sys
    # from utils.eval import evaluate_on_all_metrics

    from util import null_metrics, calc_metrics, is_better
    import argparse
    import yaml
    import pandas as pd
    import os 
    from datetime import datetime
    import matplotlib.pyplot as plt
    import copy

sys.path.append("..")

parser = argparse.ArgumentParser(description='simple_NN')
parser.add_argument('--config', default='./config/1.yaml')

class Trainer:
    def __init__(
        self,
        epochs=50,
        model="twoLayerMLP",
        batch_size=128,
        hidden_dim=256,
        activation="relu",
        dataset="cresci-2015",
        server_id="209",
        device="cuda:0",
        optimizer="adam",
        lr=1e-4,
        loss_func="cross_entropy",
        weight_decay=1e-5,
        config=None
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(batch_size, dataset, server_id)
        if model == "twoLayerMLP":
            self.model = twoLayerMLP(self.train_loader.dataset[0][0].size(0), 2, hidden_dim=hidden_dim, activation=activation).to(device)
        elif model == "threeLayerMLP":
            self.model = threeLayerMLP(self.train_loader.dataset[0][0].size(0), 2, hidden_dim=hidden_dim, activation=activation).to(device)
        
        self.lr = lr
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # elif optimizer == "something else":
        #   self.optimizer = something else
        if loss_func == "cross_entropy":
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif loss_func == "bce_with_logits":
            # bugged, don't use
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        #else:
        #...
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Retrieve "1" from parser.add_argument('--config', default='./config/1.yaml')
        self.config = os.path.splitext(os.path.basename(config))[0]

        # Create a dataframe that contains epoch, loss, and all the metrics
        self.results = pd.DataFrame(columns=['epoch',
                                             'train_loss',
                                             'train_acc',
                                             'train_pre',
                                             'train_rec',
                                             'train_f1',
                                             'train_auc',
                                             'train_mcc',
                                             'valid_loss',
                                             'valid_acc',
                                             'valid_pre',
                                             'valid_rec',
                                             'valid_f1',
                                             'valid_auc',
                                             'valid_mcc',
                                             'test_acc',
                                             'test_pre',
                                             'test_rec',
                                             'test_f1',
                                             'test_auc',
                                             'test_mcc']
                                    )
        
        
    @torch.no_grad()
    def valid(self, name = "valid"):
        self.model.eval()
        all_label = []
        all_pred = []
        ave_loss = 0.0
        cnt = 0.0
        if name == "valid":
            loader = self.valid_loader
        elif name == "test":
            loader = self.test_loader
        
        for batch in loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            n_batch = len(x)
            out = self.model(x)
            label = y[:n_batch]
            out = out[:n_batch]
            all_label += label.data
            all_pred += out
            loss = self.loss_func(out, label)
            ave_loss += loss.item() * n_batch
            cnt += n_batch
            
        ave_loss /= cnt
        all_label = torch.stack(all_label)
        all_pred = torch.stack(all_pred)
        val_results, plog = calc_metrics(all_label, all_pred)
        return val_results, ave_loss
    
    # @torch.no_grad()
    # def test(self):
    #     preds = []
    #     gt = []
    #     for data, labels in self.test_loader:
    #         data = data.to(self.device)
    #         pred = self.best_model(data)
    #         preds.append(pred.argmax(dim=-1).cpu().numpy())
    #         gt.append(labels.cpu().numpy())
            
    #     preds = np.concatenate(preds, axis=0)
    #     gt = np.concatenate(gt, axis=0)

    #     test_results = evaluate_on_all_metrics(gt, preds)
    #     return test_results
    
    def train(self):
        # self.model.train()
        for epoch in range(self.epochs):
            all_label = []
            all_pred =[]
            ave_loss = 0.0
            cnt = 0.0
            for batch in self.train_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                # print(batch)
                # print(type(batch))
                self.optimizer.zero_grad()
                # batch = batch.to(self.device)
                n_batch = len(batch[0])
                out = self.model(x)
                label = y[:n_batch]
                out = out[:n_batch]

                all_label += label.data
                all_pred += out
                loss = self.loss_func(out, label)
                ave_loss += loss.item()*n_batch
                cnt+=n_batch

                loss.backward()
                self.optimizer.step()
            ave_loss /= cnt
            # ave_loss /= cnt
            all_label = torch.stack(all_label)
            all_pred = torch.stack(all_pred)
            train_results, plog = calc_metrics(all_label, all_pred)
            plog = 'Epoch-{} train loss: {:.6}'.format(epoch, ave_loss) + plog
            train_loss = ave_loss

        
            val_results, val_loss = self.valid("valid")
            
            best = 0
            if val_results['Acc'] > best:
                best = val_results['Acc']
                self.best_model = copy.deepcopy(self.model)
            
            # Run test on the best model
            test_results, _ = self.valid("test")

            one_epoch_results = {'epoch': epoch,
                                'train_loss': train_loss,
                                'train_acc': train_results['Acc'],
                                'train_pre': train_results['Pre'],
                                'train_rec': train_results['Rec'],
                                'train_f1': train_results['MiF'],
                                'train_auc': train_results['AUC'],
                                'train_mcc': train_results['MCC'],
                                'train_pr-auc': train_results['pr-auc'],
                                'valid_loss': val_loss,
                                'valid_acc': val_results['Acc'],
                                'valid_pre': val_results['Pre'],
                                'valid_rec': val_results['Rec'],
                                'valid_f1': val_results['MiF'],
                                'valid_auc': val_results['AUC'],
                                'valid_mcc': val_results['MCC'],
                                'valid_pr-auc': val_results['pr-auc'],
                                'test_acc': test_results['Acc'],
                                'test_pre': test_results['Pre'],
                                'test_rec': test_results['Rec'],
                                'test_f1': test_results['MiF'],
                                'test_auc': test_results['AUC'],
                                'test_mcc': test_results['MCC'],
                                'test_pr-auc': test_results['pr-auc'],
                                }
            
            self.results = self.results.append(one_epoch_results, ignore_index=True)
        
        # print timenow
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results.to_csv(f'./output/{self.config}_{timenow}.csv')

        # Plot a matplotlib chart for training training and validation loss and training and validation AUC
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))

        # Set title of chart as "GNN self.config"
        fig.suptitle(f'Simple_NN Expt {self.config}_{timenow}')
        ax[0].plot(self.results['epoch'], self.results['train_loss'], label='train_loss')
        ax[0].plot(self.results['epoch'], self.results['valid_loss'], label='valid_loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Loss vs. Epochs')
        ax[0].legend()

        ax[1].plot(self.results['epoch'], self.results['train_acc'], label='train_acc')
        ax[1].plot(self.results['epoch'], self.results['valid_acc'], label='valid_acc')

        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_title('Accuracy vs. Epochs')
        ax[1].legend()

        plt.savefig(f'./output/{self.config}_{timenow}.png')

        if args.save_best:
            torch.save(self.best_model.state_dict(),
                    f'./checkpoints/{self.config}_{timenow}.pth')
            
            
if __name__ == "__main__":

    global args
    args = parser.parse_args()
    with open(f"./config/{args.config}") as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    trainer = Trainer(epochs=args.epochs,
                    model=args.model,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    activation=args.activation,
                    optimizer=args.optimizer,
                    lr=args.lr,
                    loss_func=args.loss_func,
                    weight_decay=args.weight_decay,
                    config=args.config)
    trainer.train()