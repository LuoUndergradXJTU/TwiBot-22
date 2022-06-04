#%%
# so we need a comment?
import os
from argparse import ArgumentParser
import torch

from model import BERT_GAT

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoTokenizer

import numpy as np

import os.path as osp

'''
This code is the final model of proposed method
'''
class BGSRD_dataset(Dataset):
    # 
    def __init__(self, args):
        super().__init__()

        self.args=(args) # store the hyper parameters

        A = AutoTokenizer # for embedding preparation
        self.tokenizer = A.from_pretrained(self.args.pretrained)
    #

    def load_info(self):
        # load label, train_idx, val_idx, test_idx

        meta = torch.load(f"{self.args.source_path}/{self.args.dataset}.pt")
        train_idx = []
        val_idx = []
        test_idx = []
        y = []
        y_test = []
        for m_i in range(len(meta)):
            line = meta[m_i]
            temp = line.replace("\n","").split("\t")
            if temp[1].find('test') != -1:
                test_idx.append(m_i)
                if temp[2].find('bot') != -1:
                    y_test.append(1)
                else:
                    y_test.append(0)
            elif temp[1].find('train') != -1:
                
                if temp[2].find('bot') != -1:
                    y.append(1)
                    train_idx.append(m_i)
                else:
                    y.append(0)
                    if m_i % 1 == 0:
                        train_idx.append(m_i)
            elif temp[1].find('val') != -1:
                val_idx.append(m_i)
                if temp[2].find('bot') != -1:
                    y.append(1)
                else:
                    y.append(0)
        label = torch.cat([torch.tensor(y), torch.tensor(y_test)])
        # train_mask = torch.zeros(len(train_idx)+len(val_idx)+len(test_idx))
        # train_mask[train_idx] = 1
        # val_mask = torch.zeros(len(train_idx)+len(val_idx)+len(test_idx))
        # val_mask[val_idx] = 1
        # test_mask = torch.zeros(len(train_idx)+len(val_idx)+len(test_idx))
        # test_mask[test_idx] = 1
        # info = {'label':label,'train_idx':train_mask.long(),'val_idx':val_mask.long(),'test_idx':test_mask.long()}
        info = {'label':label,'train_idx':torch.LongTensor(train_idx), 'val_idx':torch.LongTensor(val_idx), 'test_idx':torch.LongTensor(test_idx)}

        return info
    
    def setup(self, stage=None):

        self.info = self.load_info()

        if self.args.bert_switch != "off":
            # Load corpus for bert
            sentences = torch.load('./data/'+self.args.dataset+'_description.pt')
            # tokenize corpus
            text_input = self.tokenizer(
                sentences, 
                max_length=self.args.max_length, 
                truncation=True, 
                padding='max_length', 
                return_tensors='pt')
            input_ids, attention_mask = text_input.input_ids, text_input.attention_mask

            self.info['input_ids'] = input_ids
            self.info['attention_mask'] = attention_mask
            feature = torch.zeros(len(input_ids), 768)
            print("Token Finish", feature.size())
        else:
            feature = torch.load(self.args.source_path+self.args.dataset+'_Embedding.pt')
            self.info['input_ids'] = torch.zeros(len(feature), 1)
            self.info['attention_mask'] = torch.zeros(len(feature), 1)
        # torch.save(self.info, self.args.source_path+self.args.dataset+'_info.pt')
        # self.info = torch.load(self.args.source_path+self.args.dataset+'_info.pt')
        feature = torch.zeros(len(self.info['input_ids']), 768)
        feature = torch.cat((feature,self.info['input_ids'],self.info['attention_mask']),dim=1)
        edge_index_dict = torch.load(self.args.source_path+self.args.dataset+'_edge_index.pt')
        cls_feats = torch.cat([feature, torch.zeros(edge_index_dict['word_size'], feature.shape[1])])
        # edge_index = torch.stack([edge_index_dict['edge_index'][1],edge_index_dict['edge_index'][0]])
        edge_index = edge_index_dict['edge_index']
        labels = torch.cat([self.info['label'],torch.LongTensor([-100]*edge_index_dict['word_size'])])

        
        self.data = Data(x=cls_feats, edge_index=edge_index, y = labels, input_ids_dim = self.info['input_ids'].shape[1])
        
        self.data.train_idx = self.info['train_idx']
        self.data.val_idx = self.info['val_idx']
        self.data.test_idx  = self.info['test_idx']
        
        
        print(f'edge_index:{self.data.edge_index.size()},x:{self.data.x.size()},y:{self.data.y.size()},input_ids_dim:{self.data.input_ids_dim}')
        print(f'train_idx:{self.data.train_idx.size()},val_idx:{self.data.val_idx.size()},test:{self.data.test_idx.size()}')

        
        

    def train_dataloader(self):
        print("train_dataloader")
        # data = self.setup()
        
        # return NeighborLoader(
        #     data,  # The training samples.
        #     num_neighbors=[-1]*1, 
        #     input_nodes=data.train_idx, 
        #     batch_size=self.args.batch_size,
        #     directed = True,
        #     shuffle = True)
        return NeighborLoader(
            self.data,  # The training samples.
            num_neighbors=[50]*2, 
            input_nodes=self.data.train_idx, 
            batch_size=self.args.batch_size, 
            directed = True,
            shuffle=True)

    def val_dataloader(self):
        print("val_dataloader")
        # data = self.setup()
        # return NeighborLoader(
        #     data,  # The training samples.
        #     num_neighbors=[30]*2, 
        #     input_nodes=data.val_idx, 
        #     batch_size=self.args.batch_size)
        return NeighborLoader(
            self.data,  # The training samples.
            num_neighbors=[50]*2, 
            input_nodes=self.data.val_idx, 
            batch_size=self.args.batch_size)

    def test_dataloader(self):
        print("test_dataloader")
        # data = self.setup()
        # return NeighborLoader(
        #     data,  # The training samples.
        #     num_neighbors=[-1]*1, 
        #     input_nodes=data.test_idx, 
        #     batch_size=self.args.batch_size)
        return NeighborLoader(
            self.data,  # The training samples.
            num_neighbors=[50]*2,
            input_nodes=self.data.test_idx, 
            batch_size=self.args.batch_size)




#%%
if __name__ == "__main__":

    data_parser = ArgumentParser()

    data_parser.add_argument('--bert_switch', type=str, default='on')
    data_parser.add_argument('--pretrained', type=str, default='roberta-base')
    data_parser.add_argument('--dataset', default='Twibot-22', 
        choices=['Twibot-22','Twibot-20','botometer-feedback-2019','cresci-2015','cresci-2017','cresci-rtbust-2019','cresci-stock-2018','gilani-2017','midterm-2018'])
    data_parser.add_argument('--max_length', type=int, default=128, 
        help='the input length for bert')
    data_parser.add_argument('--batch_size', type=int, default=512)

    
    data_parser.add_argument('--nb_class', type=float, default=2)
    data_parser.add_argument('-m', '--m', type=float, default=0.7, 
        help='the factor balancing BERT and GCN prediction')
    data_parser.add_argument('--gcn_layers', type=int, default=2)
    data_parser.add_argument('--heads', type=int, default=8, 
        help='the number of attentionn heads for gat')
    data_parser.add_argument('--n_hidden', type=int, default=200, 
        help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
    data_parser.add_argument('--dropout', type=float, default=0.5)
    data_parser.add_argument('--lr', type=float, default=1e-4)
    data_parser.add_argument('--m_epochs', type=int, default=50)
    data_parser.add_argument('--gpu',type=int ,default=1)
    
    data_parser.add_argument('--run_s', type=int, default=0)
    data_parser.add_argument('--run_e', type=int, default=5)
    data_parser.add_argument('--ely_stop_p', type=int, default=15)
    
    data_parser.add_argument('--log_v', type=str, default='0')
    data_parser.add_argument('--source_path', type=str, default='data/')
    
    data_parser = pl.Trainer.add_argparse_args(data_parser)
    data_args = data_parser.parse_args()
    print(data_args.dataset)
    print(data_args.gcn_layers)

    data_args.num_nodes=1
    data_args.precision=16

    log_v = data_args.log_v
    source_path = data_args.source_path

    for runtime in range(data_args.run_s,data_args.run_e):

        # start : get training steps
        d = BGSRD_dataset(data_args)
        d.setup()
        
        logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=runtime,
        name=data_args.dataset + f'_lightning_logs_{log_v}')
        data_args.logger = logger
        early_stop_callback = EarlyStopping(monitor="val_ACC", min_delta=0.00, patience=data_args.ely_stop_p, verbose=False, mode="max")    



        trainer = pl.Trainer(gpus=data_args.gpu, 
        num_nodes=data_args.num_nodes, 
        precision=data_args.precision, 
        max_epochs=data_args.m_epochs, 
        callbacks=[early_stop_callback],
        logger=data_args.logger)

        
        if data_args.gcn_layers == 1:
            data_args.n_hidden = data_args.gcn_layers
            
        m = BERT_GAT(info = d.info,
        pretrained_model=data_args.pretrained, 
        nb_class=data_args.nb_class, 
        m=data_args.m, 
        gcn_layers=data_args.gcn_layers, 
        heads=data_args.heads, 
        n_hidden=data_args.n_hidden, 
        dropout=data_args.dropout,
        lr=data_args.lr,
        gpu = data_args.gpu,
        bert_switch = data_args.bert_switch
        )


        trainer.fit(m,d.train_dataloader(), d.val_dataloader())
        with open(data_args.dataset + f'_lightning_logs_{log_v}/'+data_args.dataset+'.txt','a') as f:
            f.write(str(runtime)+'  val '+str(m.val_score)+'\n')
            
        trainer.test(m, d.test_dataloader())
        test_score = np.array(m.test_score)
        with open(data_args.dataset + f'_lightning_logs_{log_v}/'+data_args.dataset+'.txt','a') as f:
            f.write(str(runtime)+' test '+str(test_score)+'\n')
