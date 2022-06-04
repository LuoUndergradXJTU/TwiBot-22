# so we need a comment?
import os
from argparse import ArgumentParser
import torch

from model import BERT_GAT

from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoTokenizer

import numpy as np

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
        try:
            info = torch.load(self.args.source_path+self.args.dataset+'_info1.pt')
        except:
            meta = torch.load(f"data/{self.args.dataset}.pt")
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
            label = torch.cat([torch.LongTensor(y), torch.LongTensor(y_test)])
            info = {'label':label,'train_idx':train_idx,'val_idx':val_idx,'test_idx':test_idx}
            torch.save(info, self.args.source_path+self.args.dataset+'_info.pt')
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
        else:
            self.info['input_ids'] = []
            self.info['attention_mask'] = []
            feature = torch.load(self.args.source_path+self.args.dataset+'_Embedding.pt')
            self.info['feature'] = feature
            
        edge_index_dict = torch.load(self.args.source_path+self.args.dataset+'_edge_index.pt')
        self.info['cls_feats'] = torch.cat([torch.zeros(feature.shape[0],feature.shape[1]), torch.zeros(edge_index_dict['word_size'], feature.shape[1])]).half()
        self.info['edge_index'] = edge_index_dict['edge_index']
        self.data = Data(x=torch.cat([feature, torch.zeros(edge_index_dict['word_size'], feature.shape[1])]), edge_index=edge_index_dict['edge_index'])
        
        self.train_dataset = TensorDataset(torch.tensor(self.info['train_idx']))
        self.val_dataset = TensorDataset(torch.tensor(self.info['val_idx']))
        self.test_dataset = TensorDataset(torch.tensor(self.info['test_idx']))

    def train_dataloader(self):
        print("train_dataloader")
        return DataLoader(
            self.train_dataset,  # The training samples.
            sampler=RandomSampler(
                self.train_dataset),  # Select batches randomly
            batch_size=self.args.batch_size  # Trains with this batch size.
        )

    def val_dataloader(self):
        print("val_dataloader\n\n{}".format(len(self.val_dataset)))
        return DataLoader(
            self.val_dataset,  # The training samples.
            sampler=RandomSampler(self.val_dataset),  # Select batches randomly
            batch_size=self.args.batch_size,  # Trains with this batch size.
            shuffle=False)

    def test_dataloader(self):
        print("test_dataloader\n\n{}".format(len(self.test_dataset)))
        return DataLoader(
            self.test_dataset,  # The training samples.
            sampler=RandomSampler(
                self.test_dataset),
            batch_size=self.args.batch_size,  # Trains with this batch size.
            shuffle=False)



if __name__ == "__main__":

    data_parser = ArgumentParser()

    data_parser.add_argument('--bert_switch', type=str, default='on')
    data_parser.add_argument('--pretrained', type=str, default='roberta-base')
    data_parser.add_argument('--dataset', default='Twibot-22', 
        choices=['Twibot-22','Twibot-20','botometer-feedback-2019','cresci-2015','cresci-2017','cresci-rtbust-2019','cresci-stock-2018','gilani-2017','midterm-2018'])
    data_parser.add_argument('--max_length', type=int, default=128, 
        help='the input length for bert')
    data_parser.add_argument('--batch_size', type=int, default=32)

    
    data_parser.add_argument('--nb_class', type=float, default=2)
    data_parser.add_argument('-m', '--m', type=float, default=0.7, 
        help='the factor balancing BERT and GCN prediction')
    data_parser.add_argument('--gcn_layers', type=int, default=2)
    data_parser.add_argument('--heads', type=int, default=4, 
        help='the number of attentionn heads for gat')
    data_parser.add_argument('--n_hidden', type=int, default=25, 
        help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
    data_parser.add_argument('--dropout', type=float, default=0.5)
    data_parser.add_argument('--lr', type=float, default=1e-4)
    data_parser.add_argument('--m_epochs', type=int, default=50)
    data_parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
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
        try:
            sgpu = 1/(1 / data_args.gpu)
            trainer = pl.Trainer(gpus=data_args.gpu, 
            num_nodes=data_args.num_nodes, 
            precision=data_args.precision, 
            max_epochs=data_args.m_epochs, 
            callbacks=[early_stop_callback],
            logger=data_args.logger)
        except:
            trainer = pl.Trainer(
                gpus = 0,
                max_epochs=data_args.m_epochs, 
                callbacks=[early_stop_callback],
                logger=data_args.logger)
            data_args.gpu = 0
        
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
