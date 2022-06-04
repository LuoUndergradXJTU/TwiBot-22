from sklearn.ensemble import RandomForestClassifier
import torch as th
import sys
sys.path.append("..")
sys.path.append("../..")
from utils import set_global_seed
from utils.eval import evaluate_on_all_metrics

def train():
    train_data = th.load("train_data.pt")
    train_label = th.load("train_label.pt")
    valid_data = th.load("valid_data.pt")
    valid_label = th.load("valid_label.pt")
    test_data = th.load("test_data.pt")
    test_label = th.load("test_label.pt")
    for seed in [100, 200, 300, 400, 500]:
        set_global_seed(seed)
        rfc = RandomForestClassifier()
        rfc.fit(train_data, train_label)
        pred = rfc.predict(test_data)
        print(evaluate_on_all_metrics(test_label, pred))
    
if __name__ == "__main__":
    train()