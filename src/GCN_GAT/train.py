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
    import os
    import argparse
    from utils import null_metrics, calc_metrics, is_better
    import torch
    from dataset import get_train_data
    from torch_geometric.loader import NeighborLoader
    from tqdm import tqdm
    import torch.nn as nn
    from model import BotGAT, BotGCN, BotRGCN

    import yaml
    import pandas as pd
    from datetime import datetime
    import matplotlib.pyplot as plt
    import copy

parser = argparse.ArgumentParser(description='GCN_GAN')
parser.add_argument('--config', default='./config/1.yaml')

if not os.path.exists("checkpoints/"):
    os.mkdir("checkpoints/")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# parser = ArgumentParser()
# parser.add_argument('--mode', type=str, default='GCN')
# parser.add_argument('--visible', type=bool, default=False)
# parser.add_argument('--hidden_dim', type=int, default=128)
# parser.add_argument('--max_epoch', type=int, default=1000)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--no_up', type=int, default=50)
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--weight_decay', type=float, default=1e-5)
# parser.add_argument('--dropout', type=float, default=0.3)

global args
args = parser.parse_args()
with open(f"./config/{args.config}") as f:
    config = yaml.safe_load(f)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)

dataset_name = "cresci-2015"
mode = args.mode
visible = args.visible

assert mode in ['GCN', 'GAT', 'RGCN']
# assert dataset_name in ['cresci-2015']

data = get_train_data(dataset_name)

# hidden_dim = args.hidden_dim
# dropout = args.dropout
# lr = args.lr
# weight_decay = args.weight_decay
# max_epoch = args.max_epoch
# batch_size = args.batch_size
# no_up = args.no_up

# Retrieve "1" from parser.add_argument('--config', default='./config/1.yaml')
config = os.path.splitext(os.path.basename(args.config))[0]

                            
def forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader):
    model.train()
    all_label = []
    all_pred = []
    ave_loss = 0.0
    cnt = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_batch = batch.batch_size
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        all_label += label.data
        all_pred += out
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
        loss.backward()
        optimizer.step()
    ave_loss /= cnt
    # ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    train_results, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} train loss: {:.6}'.format(epoch, ave_loss) + plog
    train_loss = ave_loss
    
    if args.visible:
        print(plog)
    val_loss, val_results = validation(epoch, 'validation', model, loss_fn, val_loader)

    
    return train_loss, train_results, val_loss, val_results


@torch.no_grad()
def validation(epoch, name, model, loss_fn, loader):
    model.eval()
    all_label = []
    all_pred = []
    ave_loss = 0.0
    cnt = 0.0
    for batch in loader:
        batch = batch.to(device)
        n_batch = batch.batch_size
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        all_label += label.data
        all_pred += out
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
    ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} {} loss: {:.6}'.format(epoch, name, ave_loss) + plog
    if args.visible:
        print(plog)
    return ave_loss, metrics


def train(results):
    print(data)
    train_loader = NeighborLoader(data,
                                  num_neighbors=[256] * 4,
                                  batch_size=args.batch_size,
                                  input_nodes=data.train_idx,
                                  shuffle=True)
    val_loader = NeighborLoader(data,
                                num_neighbors=[256] * 4,
                                batch_size=args.batch_size,
                                input_nodes=data.val_idx)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256] * 4,
                                 batch_size=args.batch_size,
                                 input_nodes=data.test_idx)
    if mode == 'GAT':
        model = BotGAT(hidden_dim=args.hidden_dim,
                       dropout=args.dropout,
                       num_prop_size=data.num_property_embedding.shape[-1],
                       cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
    elif mode == 'GCN':
        model = BotGCN(hidden_dim=args.hidden_dim,
                       dropout=args.dropout,
                       skip_connection=args.skip_connection,
                       num_prop_size=data.num_property_embedding.shape[-1],
                       cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
    elif mode == 'RGCN':
        model = BotRGCN(hidden_dim=args.hidden_dim,
                        dropout=args.dropout,
                        num_prop_size=data.num_property_embedding.shape[-1],
                        cat_prop_size=data.cat_property_embedding.shape[-1],
                        num_relations=data.edge_type.max().item() + 1).to(device)
    else:
        raise KeyError
    best_val_results = null_metrics()
    best_state_dict = None
    
    if args.loss_func == "cross_entropy":
        loss_fn =  torch.nn.CrossEntropyLoss()
    elif args.loss_func == "bce_with_logits":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    #else:
    #...
    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # elif optimizer == "something else":
        #   self.optimizer = something else
    pbar = tqdm(range(args.max_epoch), ncols=0)
    cnt = 0
    for epoch in pbar:
        train_loss, train_results, val_loss, val_results = forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader)
        if is_better(val_results, best_val_results):
            best_val_results = val_results
            best_state_dict = model.state_dict()
            cnt = 0
        else:
            cnt += 1
        pbar.set_postfix_str('val acc {} no up cnt {}'.format(val_results['Acc'], cnt))
        if cnt == args.no_up:
            break
            
        model.load_state_dict(best_state_dict)
        _, test_results = validation(args.max_epoch, 'test', model, loss_fn, test_loader)
        print("test results: ")
        print(test_results)

        one_epoch_results = {'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_results['Acc'],
                'train_pre': train_results['Pre'],
                'train_rec': train_results['Rec'],
                'train_f1': train_results['MiF'],
                'train_auc': train_results['AUC'],
                'train_mcc': train_results['MCC'],
                'train_pr_auc': train_results['pr-auc'],
                'valid_loss': val_loss,
                'valid_acc': val_results['Acc'],
                'valid_pre': val_results['Pre'],
                'valid_rec': val_results['Rec'],
                'valid_f1': val_results['MiF'],
                'valid_auc': val_results['AUC'],
                'valid_mcc': val_results['MCC'],
                'valid_pr_auc': val_results['pr-auc'],
                'test_acc': test_results['Acc'],
                'test_pre': test_results['Pre'],
                'test_rec': test_results['Rec'],
                'test_f1': test_results['MiF'],
                'test_auc': test_results['AUC'],
                'test_mcc': test_results['MCC'],
                'test_pr_auc': test_results['pr-auc']
                }

        results = results.append(one_epoch_results, ignore_index=True)

    # print timenow
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    results.to_csv(f'./output/{config}_{timenow}.csv')

    # Plot a matplotlib chart for training training and validation loss and training and validation AUC
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # Set title of chart as "GNN config"
    fig.suptitle(f'GNN Expt {config}_{timenow}')
    ax[0].plot(results['epoch'], results['train_loss'], label='train_loss')
    ax[0].plot(results['epoch'], results['valid_loss'], label='valid_loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].legend()

    ax[1].plot(results['epoch'], results['train_acc'], label='train_acc')
    ax[1].plot(results['epoch'], results['valid_acc'], label='valid_acc')

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs. Epochs')
    ax[1].legend()

    plt.savefig(f'./output/{config}_{timenow}.png')

    torch.save(best_state_dict, f'checkpoints/{config}_{timenow}.pt')
    for key, value in test_results.items():
        print(key, value)
    print("best test_results: ")
    print(test_results)

if __name__ == '__main__':

    # Create a dataframe that contains epoch, loss, and all the metrics
    results = pd.DataFrame(columns=['epoch',
                                            'train_loss',
                                            'train_acc',
                                            'train_pre',
                                            'train_rec',
                                            'train_f1',
                                            'train_auc',
                                            'train_mcc',
                                            'train_pr_auc',
                                            'valid_loss',
                                            'valid_acc',
                                            'valid_pre',
                                            'valid_rec',
                                            'valid_f1',
                                            'valid_auc',
                                            'valid_mcc',
                                            'valid_pr_auc',
                                            'test_acc',
                                            'test_pre',
                                            'test_rec',
                                            'test_f1',
                                            'test_auc',
                                            'test_mcc',
                                            'test_pr_auc'])

    train(results)
