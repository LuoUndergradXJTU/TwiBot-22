import torch
from torch.utils.data import Dataset

class BotDataset(Dataset):
    def __init__(self, name, path):
        self.name = name
        
        # different features dataset
        # self.cat_features = torch.load(path + "cat_properties_tensor.pt")
        # self.prop_features = torch.load(path + "num_properties_tensor.pt")
        # self.tweet_features = torch.load(path + "tweets_tensor.pt")
        # self.des_features = torch.load(path + "des_tensor.pt")
        self.train_idx = torch.load(path + "train_idx.pt")
        self.val_idx = torch.load(path + "val_idx.pt")
        self.test_idx = torch.load(path + "test_idx.pt")

        if self.name == "train":
            self.len = int(self.train_idx.shape[0])

        else:
            self.len = 1

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        if self.name == "train": 
            return self.train_idx[index]
        elif self.name == "valid":
            return self.val_idx
        elif self.name == "test":
            return self.test_idx