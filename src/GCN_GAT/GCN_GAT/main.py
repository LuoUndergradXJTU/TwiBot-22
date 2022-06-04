from model import BotRGCN
from Dataset import Twibot20
import torch
from torch import nn
from utils import accuracy,init_weights
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef

device = 'cuda:6'
embedding_size,dropout,lr,weight_decay=16,0.3,1e-3,5e-3

dataset=Twibot20(device=device,process=False,save=False)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()
print(des_tensor.size(),tweets_tensor.size())
print(num_prop.size(),category_prop.size())
model=BotRGCN(embedding_dimension=embedding_size).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

def train(epoch):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

def test():
    model.eval()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(label[test_idx],output[test_idx])
    mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    auc = roc_auc_score(label[test_idx],  output[test_idx])
    pre = precision_score(label[test_idx],  output[test_idx])
    recall = recall_score(label[test_idx],  output[test_idx])
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "f1_score= {:.4f}".format(f1.item()),
            "mcc= {:.4f}".format(mcc),
            "auc= {:.4f}".format(auc),
            "pre= {:.4f}".format(pre),
            "recall= {:.4f}".format(recall),
            )
    return acc_test,loss_test,f1_score

model.apply(init_weights)

epochs=300
for epoch in range(epochs):
    train(epoch)
    
test()