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
    import torch
    from model import GNN_classifier, GNN_classifier_1_layer, GNN_classifier_3_layer
    from preprocess import generate_hetero_graph_loader, generate_homo_graph_loader_only_user
    from tqdm import tqdm
    import numpy as np

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

    parser = argparse.ArgumentParser(description='simple_GNN')
    parser.add_argument('--config', default='./config/1.yaml')

sys.path.append("..")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class homo_Trainer:
    def __init__(
        self,
        epochs=50,
        batch_size=128,
        input_dim=768,
        hidden_dim=256,
        # num_relations=3,
        # num_bases=20,
        num_layers=20,
        num_hidden_layers=2,
        num_neighbors=20,
        activation="relu",
        dataset="cresci-2015",
        server_id="209",
        device="cuda:0",
        optimizer="adam",
        weight_decay=1e-5,
        lr=1e-4,
        loss_func="cross_entropy",
        config=None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.activation = activation
        if num_hidden_layers == 1:
            self.model = GNN_classifier_1_layer(input_dim, 2, hidden_dim, activation)
        elif num_hidden_layers == 3:
            self.model = GNN_classifier_3_layer(input_dim, 2, hidden_dim, activation)
        else:
            self.model = GNN_classifier(input_dim, 2, hidden_dim, activation)
        self.train_loader, self.valid_loader, self.test_loader = generate_homo_graph_loader_only_user(num_layers, num_neighbors, batch_size, True, dataset, server_id)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # elif optimizer == "something else":
        #   self.optimizer = something else
        if loss_func == "cross_entropy":
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif loss_func == "bce_with_logits":
            # bugged, don't use
            self.loss_func = torch.nn.BCEWithLogitsLoss()
        #else:
        #...
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
    def valid(self, name="valid"):
        self.model.eval()
        all_label = []
        all_pred = []
        ave_loss = 0.0
        cnt = 0.0
        # preds = []
        # pred_probs = []
        # labels =[]
        if name == "valid":
            loader = self.valid_loader
        elif name == "test":
            loader = self.test_loader
        for batch in loader:
            batch = batch.to(self.device, "edge_index", "x", "y")
            n_batch = batch.batch_size
            out = self.model(batch.x, batch.edge_index)
            label = batch.y[:n_batch]
            out = out[:n_batch]
            all_label += label.data
            all_pred += out
            loss = self.loss_func(out, label)
            ave_loss += loss.item() * n_batch
            cnt += n_batch
            # preds.append(pred.argmax(dim=-1).cpu().numpy())
            # pred_probs.append(pred.cpu().numpy())
            # labels.append(batch.y[:n_batch].cpu().numpy())
        ave_loss /= cnt
        all_label = torch.stack(all_label)
        all_pred = torch.stack(all_pred)
        val_results, plog = calc_metrics(all_label, all_pred)
        # preds = np.concatenate(preds, axis=0)
        # pred_probs = np.concatenate(pred_probs, axis=0)
        # labels = np.concatenate(labels, axis=0)

        # val_results = evaluate_on_all_metrics(labels, preds)
        # val_loss = self.loss_func(torch.tensor(pred_probs), torch.tensor(labels))
        
        return val_results, ave_loss
        
        
    # @torch.no_grad()
    # def test(self):
    #     preds = []
    #     labels = []
    #     for data in self.test_loader:
    #         data = data.to(self.device, "edge_index", "x")
    #         batch_size = data.batch_size
    #         pred = self.model(data.x, data.edge_index)[:batch_size]
    #         preds.append(pred.argmax(dim=-1).cpu().numpy())
    #         labels.append(data.y[:batch_size].cpu().numpy())
            
    #     preds = np.concatenate(preds, axis=0)
    #     labels = np.concatenate(labels, axis=0)
        
    #     test_results = evaluate_on_all_metrics(labels, preds)
    #     return test_results
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            all_label = []
            all_pred = []
            ave_loss = 0.0
            cnt = 0.0
            for batch in self.train_loader:
                # Create a PyTorch tensor that stores all predictions
                # preds = []
                # labels  = []

                self.optimizer.zero_grad()
                # batch = batch.to(device)
                batch = batch.to(self.device, "edge_index", "x", "y")
                n_batch = batch.batch_size
                out = self.model(batch.x, batch.edge_index)
                label = batch.y[:n_batch]
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
            # preds.append(pred.argmax(dim=-1).cpu().numpy())
            # labels.append(data.y[:batch_size].cpu().numpy())                    
            # print(batch_size)
            # print(pred.shape)
            # print(loss.shape)
            # # print(preds.shape)
            # # print(labels.shape)
            # print(pred[0])
            # # print(loss.item)
            # print(preds[0])
            # print(labels[0])
            # print('\n')
    
            # train_results = evaluate_on_all_metrics(np.concatenate(labels, axis=0), 
                                                    # np.concatenate(preds, axis=0))
            val_results, val_loss = self.valid("valid")

            best = 0
            if val_results['Acc'] > best:
                best = val_results['Acc']
                self.best_model = copy.deepcopy(self.model)

            # Run test on best model (on validation) so far
            test_results, test_loss = self.valid("test")

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
        fig.suptitle(f'GNN Expt {self.config}_{timenow}')
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

    trainer = homo_Trainer(epochs=args.epochs,
                            batch_size=args.batch_size,
                            input_dim=args.input_dim,
                            hidden_dim=args.hidden_dim,
                            num_hidden_layers=args.num_hidden_layers,
                            num_layers=args.num_layers,
                            num_neighbors=args.num_neighbors,
                            activation=args.activation,
                            dataset="cresci-2015",
                            server_id="209",
                            device="cuda:0",
                            optimizer=args.optimizer,
                            weight_decay=args.weight_decay,
                            lr=args.lr,
                            loss_func=args.loss_func,
                            config = args.config)
    trainer.train()
