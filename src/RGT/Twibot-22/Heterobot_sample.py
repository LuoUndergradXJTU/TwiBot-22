from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from layer import RGTLayer
import pytorch_lightning as pl
from torch import nn
import torch
# from Dataset import BotDataset
# from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

def load_data(args):
    print("loading features...")
    cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cpu")
    prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cpu")
    tweet_features = torch.load(args.path + "tweets_tensor.pt", map_location="cpu")
    des_features = torch.load(args.path + "des_tensor.pt", map_location="cpu")
    x = torch.cat((cat_features, prop_features, tweet_features, des_features), dim=1)
    
    print("loading edges & label...")
    edge_index = torch.load(args.path + "edge_index.pt", map_location="cpu")
    edge_type = torch.load(args.path + "edge_type.pt", map_location="cpu").unsqueeze(-1)
    label = torch.load(args.path + "label.pt", map_location="cpu")
    data = Data(x=x, edge_index = edge_index, edge_attr=edge_type, y=label)
    
    print("loading index...")
    data.train_idx = torch.load(args.path + "train_idx.pt", map_location="cpu")
    data.valid_idx = torch.load(args.path + "val_idx.pt", map_location="cpu")
    data.test_idx = torch.load(args.path + "test_idx.pt", map_location="cpu")
    return data
    
class RGTDetector(pl.LightningModule):
    def __init__(self, args):
        super(RGTDetector, self).__init__()
    
        self.lr = args.lr
        self.l2_reg = args.l2_reg
    
        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGT_layer1 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        cat_features = train_batch.x[:, :args.cat_num]
        prop_features = train_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
        tweet_features = train_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
        des_features = train_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
        label = train_batch.y
        
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_attr.view(-1)
        
        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
        
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features)
        loss = self.CELoss(pred, label)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            cat_features = val_batch.x[:, :args.cat_num]
            prop_features = val_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            tweet_features = val_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            des_features = val_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            
            label = val_batch.y
        
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_attr.view(-1)
            
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            # print(pred.size())
            pred_binary = torch.argmax(pred, dim=1)
            
            # print(self.label[val_batch].size())

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

            # print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            cat_features = test_batch.x[:, :args.cat_num]
            prop_features = test_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            tweet_features = test_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            des_features = test_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            
            label = test_batch.y[:args.test_batch_size]
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_attr.view(-1)
            
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)[:args.test_batch_size]
            pred_binary = torch.argmax(pred, dim=1)
            
            pred_test.append(pred_binary.squeeze().cpu())
            pred_test_prob.append(pred[:,1].squeeze().cpu())
            label_test.append(label.squeeze().cpu())

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            precision =precision_score(label.cpu(), pred_binary.cpu())
            recall = recall_score(label.cpu(), pred_binary.cpu())
            auc = roc_auc_score(label.cpu(), pred[:,1].cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", auc)

            # print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


parser = argparse.ArgumentParser(description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="/data2/whr/czl/TwiBot22-baselines/src/BotRGCN/data_twi22/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.3, help="description channel")
parser.add_argument("--trans_head", type=int, default=2, help="description channel")
parser.add_argument("--semantic_head", type=int, default=2, help="description channel")
parser.add_argument("--batch_size", type=int, default=128, help="description channel")
parser.add_argument("--epochs", type=int, default=1000, help="description channel")
parser.add_argument("--lr", type=float, default=5e-4, help="description channel")
parser.add_argument("--l2_reg", type=float, default=1e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")
parser.add_argument("--test_batch_size", type=int, default=200, help="random")

if __name__ == "__main__":
    global args, pred_test, pred_test_prob, label_test
    args = parser.parse_args()
    pred_test = []
    pred_test_prob = []
    label_test = []
       
    if args.random_seed != None:
        pl.seed_everything(args.random_seed)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    # train_dataset = BotDataset(name="train")
    # valid_dataset = BotDataset(name="valid")
    # test_dataset = BotDataset(name="test")
    data = load_data(args)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=1)
    # test_loader = DataLoader(test_dataset, batch_size=1)
    print("loading...")
    train_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.train_idx, batch_size=args.batch_size, shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.valid_idx, batch_size=args.batch_size)# , num_workers=1
    test_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.test_idx, batch_size=args.test_batch_size)# , num_workers=1
    
    model = RGTDetector(args)
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1)#, callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_loader)#, valid_loader)

    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])

    best_model = RGTDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)
    
    # label_test= torch.load(args.path + "label.pt", map_location="cpu")[-100000:]
    pred_test = torch.cat(pred_test).cpu()
    pred_test_prob = torch.cat(pred_test_prob).cpu()
    label_test = torch.cat(label_test).cpu()
    
    print(pred_test.size())
    
    acc = accuracy_score(label_test.cpu(), pred_test.cpu())
    f1 = f1_score(label_test.cpu(), pred_test.cpu())
    precision =precision_score(label_test.cpu(), pred_test.cpu())
    recall = recall_score(label_test.cpu(), pred_test.cpu())
    auc = roc_auc_score(label_test.cpu(), pred_test_prob.cpu())
    
    print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))