from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from layer import SimpleHGN
import pytorch_lightning as pl
from torch import nn
import torch
from Dataset import BotDataset
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir


class SHGNDetector(pl.LightningModule):
    def __init__(self, args):
        super(SHGNDetector, self).__init__()
        self.edge_index = torch.load(args.path + "edge_index.pt", map_location="cuda")
        self.edge_type = torch.load(args.path + "edge_type.pt", map_location="cuda").long()
        self.label = torch.load(args.path + "label.pt", map_location="cuda")
  
        self.lr = args.lr
        self.l2_reg = args.l2_reg

        self.cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cuda")
        self.prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cuda")
        self.tweet_features = torch.load(args.path + "tweets_tensor.pt", map_location="cuda")
        self.des_features = torch.load(args.path + "des_tensor.pt", map_location="cuda")

        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)

        self.HGN_layer1 = SimpleHGN(num_edge_type=2, in_channels=args.linear_channels, out_channels=args.out_channel, rel_dim=args.rel_dim, beta=args.beta)
        self.HGN_layer2 = SimpleHGN(num_edge_type=2, in_channels=args.linear_channels, out_channels=args.out_channel, rel_dim=args.rel_dim, beta=args.beta, final_layer=True)

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
        train_batch = train_batch.squeeze(0)

        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
        
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        user_features, alpha = self.HGN_layer1(user_features, self.edge_index, self.edge_type)
        user_features, _ = self.HGN_layer2(user_features, self.edge_index, self.edge_type, alpha)

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features[train_batch])
        loss = self.CELoss(pred, self.label[train_batch])

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            val_batch = val_batch.squeeze(0)

            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features, alpha = self.HGN_layer1(user_features, self.edge_index, self.edge_type)
            user_features, _ = self.HGN_layer2(user_features, self.edge_index, self.edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features[val_batch])
            # print(pred.size())
            pred_binary = torch.argmax(pred, dim=1)
            
            acc = accuracy_score(self.label[val_batch].cpu(), pred_binary.cpu())
            f1 = f1_score(self.label[val_batch].cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc)
            self.log("val_f1", f1)

            # print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            test_batch = test_batch.squeeze(0)
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features, alpha = self.HGN_layer1(user_features, self.edge_index, self.edge_type)
            user_features, _ = self.HGN_layer2(user_features, self.edge_index, self.edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features[test_batch])
            
            pred_binary = torch.argmax(pred, dim=1)

            acc = accuracy_score(self.label[test_batch].cpu(), pred_binary.cpu())
            f1 = f1_score(self.label[test_batch].cpu(), pred_binary.cpu())
            precision =precision_score(self.label[test_batch].cpu(), pred_binary.cpu())
            recall = recall_score(self.label[test_batch].cpu(), pred_binary.cpu())
            auc = roc_auc_score(self.label[test_batch].cpu(), pred[:,1].cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", auc)

            print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))

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
parser.add_argument("--path", type=str, default="/data2/whr/czl/TwiBot22-baselines/src/RGT/cresci-15/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=1, help="catgorical features")
parser.add_argument("--rel_dim", type=int, default=100, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--batch_size", type=int, default=128, help="description channel")
parser.add_argument("--epochs", type=int, default=50, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")
parser.add_argument("--beta", type=float, default=0.05, help="description channel")

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    if args.random_seed != None:
        pl.seed_everything(args.random_seed)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    train_dataset = BotDataset(name="train", args=args)
    valid_dataset = BotDataset(name="valid", args=args)
    test_dataset = BotDataset(name="test", args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = SHGNDetector(args)
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1, callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_loader, valid_loader)

    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])

    best_model = SHGNDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)