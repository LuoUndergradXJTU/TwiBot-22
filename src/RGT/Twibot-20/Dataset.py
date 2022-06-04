import torch
from torch.utils.data import Dataset

class BotDataset(Dataset):
    def __init__(self, name):
        self.name = name
        
        # different features dataset
        # self.cat_features = torch.load(path + "cat_properties_tensor.pt")
        # self.prop_features = torch.load(path + "num_properties_tensor.pt")
        # self.tweet_features = torch.load(path + "tweets_tensor.pt")
        # self.des_features = torch.load(path + "des_tensor.pt")

        if self.name == "train":
            self.len = 8278

        else:
            self.len = 1

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        if self.name == "train": 
            return index
        elif self.name == "valid":
            return torch.arange(8278, 10643)
        elif self.name == "test":
            return torch.arange(10643, 11826)