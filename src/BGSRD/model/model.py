import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GATConv
from transformers import AutoModel
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# cuda = torch.device('cuda')

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 feat_drop=0.5,
                 negative_slope=0.2,
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers # layer numbers
        self.gat_layers = nn.ModuleList() # model list
        self.activation = nn.LeakyReLU()
        # input projection
        self.gat_layers.append(GATConv(
            in_channels=in_dim,out_channels=num_hidden,heads=heads[0],
            dropout=feat_drop, negative_slope = negative_slope))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(self.activation(GATConv(
                in_channels= num_hidden * heads[l-1], out_channels=num_hidden,heads=heads[l],
                dropout=feat_drop,negative_slope=negative_slope)))
        # output projection
        self.gat_layers.append(GATConv(
            in_channels=num_hidden * heads[-2], out_channels=num_classes, heads=heads[-1],
            dropout=feat_drop, negative_slope=negative_slope))

    def forward(self, x, inputs):
        edge_index = inputs

        for l in range(self.num_layers):
            x = self.gat_layers[l](x, edge_index).flatten(1)
        # output projection
        
        logits = self.gat_layers[-1](x, edge_index)
        return logits

class BERT_GAT(pl.LightningModule):
    def __init__(self, info,pretrained_model='roberta_base', 
    nb_class=2, 
    m=0.7, 
    gcn_layers=2, 
    heads=8, 
    n_hidden=32, 
    dropout=0.5,
    lr=1e-5,
    gpu = 1,
    bert_switch = 'on',
    gcn_weight_dec = 1e-5
    ):
        super().__init__()

        self.ginfo = info
        self.m = m # trade-off between gcn and bert
        self.nb_class = nb_class # output layer
        
        self.activation = nn.LeakyReLU()
        # self.CELoss = F.nll_loss()
        
        self.pretrained = pretrained_model
        self.bert_model = AutoModel.from_pretrained(pretrained_model) # bert
        self.feat_dim = list(self.bert_model.modules())[-2].out_features # input dim 768
        
        self.classifier = nn.Linear(self.feat_dim, nb_class) # mlp for bert_model
        self.linear = nn.Linear(self.feat_dim, self.feat_dim)

        self.gcn = GAT(
                 num_layers=gcn_layers-1, # gat layer number
                 in_dim=self.feat_dim, # input dim
                 num_hidden=n_hidden, # hidden layer dim
                 num_classes=nb_class, # output layer dim
                 heads=[heads] * (gcn_layers-1) + [1], # head for each layer
                 feat_drop=dropout# it is what it is
        ) # gat layer
        self.lr = lr
        self.gpu = gpu

        self.val_score = []
        self.test_score = []
        self.bert_switch = bert_switch
        self.gcn_weight_dec = gcn_weight_dec
        self.test_pred = torch.tensor([])
        self.test_true = torch.tensor([])
    def forward(self, batch, idx):

        if self.bert_switch != "off":
            try:
                self.gpu = 1/(1 / self.gpu)
                input_ids, attention_mask = self.ginfo['input_ids'][idx].cuda(), self.ginfo['attention_mask'][idx].cuda()
                self.cls_feats = self.ginfo['cls_feats'].cuda()
                self.edge_index = self.ginfo['edge_index'].cuda()
            except:
                input_ids, attention_mask = self.ginfo['input_ids'][idx], self.ginfo['attention_mask'][idx]
                self.cls_feats = self.ginfo['cls_feats']
                self.edge_index = self.ginfo['edge_index']

            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0] # output of bert and initial feature of nodes
        else:
            try:
                self.cls_feats = self.ginfo['cls_feats'].cuda().half()
                self.edge_index = self.ginfo['edge_index'].cuda()
                cls_feats = self.ginfo['feature'][idx].cuda().half()
            except:
                self.cls_feats = self.ginfo['cls_feats']
                self.edge_index = self.ginfo['edge_index']
                cls_feats = self.ginfo['feature'][idx].half()

        cls_logit = self.classifier(self.activation(cls_feats))

        cls_pred = nn.Softmax(dim=1)(cls_logit) # out_put of bert module 
        pred = cls_pred
        
        try:
            self.cls_feats[idx] = cls_feats.half()

        except:
            self.cls_feats[idx] = cls_feats

        
        gcn_logit = self.gcn(self.cls_feats,self.edge_index)[idx]
        gcn_pred = nn.Softmax(dim=1)(gcn_logit) # out_put of bert module
        pred = gcn_pred
        
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m) # sum up
        
        pred = torch.log(pred)
        
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.gcn_weight_dec)
        return optimizer

    def training_step(self, batch, batch_idx):
        try:
            self.gpu = 1/(1 / self.gpu)
            (idx, ) = [x.cuda() for x in batch]
        except:
            (idx, ) = [x for x in batch]
        # train_mask = self.ginfo['train'][idx].type(torch.BoolTensor)
        y_pred = self(batch,idx)
        try:
            self.gpu = 1/(1 / self.gpu)
            y_true = self.ginfo['label'][idx].cuda()
        except:
            y_true = self.ginfo['label'][idx]

        loss = F.nll_loss(y_pred, y_true)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            idx = batch[0]

            y_pred = self(batch,idx)
            # print(y_pred)
            try:
                self.gpu = 1/(1 / self.gpu)
                y_true = self.ginfo['label'][idx].cuda()
            except:
                y_true = self.ginfo['label'][idx]

            loss = F.nll_loss(y_pred, y_true)
            predictions = torch.argmax(y_pred, dim=1)
            accuracy = (predictions == y_true).float().mean().cpu()
            predictions = predictions.cpu()
            y_true = y_true.cpu()
            
            print("valid_pred",predictions)
            # f1 = f1_score(y_true, predictions)
            # precision = precision_score(y_true, predictions)
            # recall = recall_score(y_true, predictions)

            # fpr, tpr, thresholds = metrics.roc_curve(y_true, predictions)
            # auc = metrics.auc(fpr, tpr)
            # print([accuracy, precision, recall, f1, auc])
            # self.val_score = [accuracy, precision, recall, f1, auc]
            self.log('val_loss', loss)
            self.log('val_ACC', accuracy)
            # self.log('val_f1', f1)
            # self.log('val_precision', precision)
            # self.log('val_recall', recall)
            # self.log('val_auc', auc)
            # the validation_step method expects a dictionary, which should at least contain the val_loss
            # print(accuracy, f1, precision, recall, auc)
            
            # the validation_step method expects a dictionary, which should at least contain the val_loss
            return {'val_loss': loss, 'val_ACC': accuracy, 'val_predictions': predictions, "val_true":y_true}

    def validation_epoch_end(self, validation_step_outputs):
        # OPTIONAL The second parameter in the validation_epoch_end - we named it validation_step_outputs -
        # contains the outputs of the validation_step, collected for all the batches over the entire epoch.

        # We use it to track progress of the entire epoch, by calculating averages

        avg_loss = torch.stack([x['val_loss']
                                for x in validation_step_outputs]).mean()

        avg_accuracy = torch.stack(
            [x['val_ACC'] for x in validation_step_outputs]).mean()

        all_pred = torch.cat(
            [x['val_predictions'] for x in validation_step_outputs])
        all_true = torch.cat(
            [x['val_true'] for x in validation_step_outputs])

        f1 = f1_score(all_true, all_pred)
        # precision = precision_score(all_true, all_pred)
        # recall = recall_score(all_true, all_pred)

        # fpr, tpr, thresholds = metrics.roc_curve(all_true, all_pred)
        # auc = metrics.auc(fpr, tpr)
        # print([avg_accuracy, precision, recall, f1, auc])
        # self.val_score = [avg_accuracy, precision, recall, f1, auc]
        print([avg_accuracy, f1])
        self.val_score = [avg_accuracy, f1]
        
        
        # print(avg_loss, avg_accuracy)
        
    def test_step(self, batch, batch_idx):
        
        try:
            self.gpu = 1/(1 / self.gpu)
            (idx, ) = [x.cuda() for x in batch]

        except:
            (idx, ) = [x for x in batch]

        
        y_pred = self(batch,idx)
        print(y_pred)

        try:
            self.gpu = 1/(1 / self.gpu)
            y_true = self.ginfo['label'][idx].cuda()
        except:
            y_true = self.ginfo['label'][idx]
        loss = F.nll_loss(y_pred, y_true)
        
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        y_pred1 = torch.cat([self.test_pred, y_pred])
        y_true1 = torch.cat([self.test_true, y_true])
        self.test_pred = y_pred1
        self.test_true = y_true1
        y_true = y_true1
        y_pred = y_pred1
        
        predictions = torch.argmax(y_pred, dim=1)
        accuracy = (predictions == y_true).float().mean().cpu()
        predictions = predictions.cpu()
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        

        f1 = f1_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        
        
        fpr, tpr, thresholds = metrics.roc_curve(y_true, predictions)
        auc = metrics.auc(fpr, tpr)
        
        self.test_score = [accuracy, precision,recall, f1, auc]

        self.log('test_ACC', accuracy)
        self.log('test_f1', f1)
        self.log('test_precision',precision)
        self.log('test_recall',recall)
        self.log('test_auc',auc)
        # the validation_step method expects a dictionary, which should at least contain the val_loss
        self.cls_feats = []
        self.edge_index = []