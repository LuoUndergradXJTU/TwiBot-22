import sys
sys.path.append("..")
from utils.dataset import merge_and_split, simple_vectorize

import torch
from torch.utils.data import Dataset, DataLoader

class simple_dataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return self.data.size(0)
    
def get_dataloaders(batch_size, dataset="botometer-feedback-2019", server_id="209"):
    train, valid, test = merge_and_split(dataset, server_id)
    train_data, train_labels = simple_vectorize(train)
    valid_data, valid_labels = simple_vectorize(valid)
    test_data, test_labels = simple_vectorize(test)
    
    train_loader = DataLoader(simple_dataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(simple_dataset(valid_data, valid_labels), batch_size=batch_size)
    test_loader = DataLoader(simple_dataset(test_data, test_labels), batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader