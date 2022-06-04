import torch
from torch.utils.data import Dataset

class BotDataset(Dataset):
    def __init__(self, name, args):
        self.name = name
        
        self.train_idx = torch.load(args.path + "train_idx.pt")
        self.val_idx = torch.load(args.path + "val_idx.pt")
        self.test_idx = torch.load(args.path + "test_idx.pt")

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