import os
import pandas as pd
from pathlib import Path

root_path = Path("/data2/whr/czl/TwiBot22-baselines/datasets")
datasets = os.listdir(root_path)


def calc_bot_rate(dataset):
    dataset_path = root_path / dataset
    label = pd.read_csv(dataset_path / "label.csv")
    split = pd.read_csv(dataset_path / "split.csv")
    sl = pd.merge(label, split)
    train_rate = len(sl[(sl.split == "train") & (
        sl.label == "bot")]) / len(sl[sl.split == "train"])
    valid_rate = len(sl[(sl.split.str.startswith("val")) & (
        sl.label == "bot")]) / len(sl[sl.split.str.startswith("val")])
    test_rate = len(sl[(sl.split == "test") & (
        sl.label == "bot")]) / len(sl[sl.split == "test"])
    print(f"{dataset}: train:{train_rate:.2f} valid:{valid_rate:.2f} test:{test_rate:.2f}")


for dataset in datasets:
    calc_bot_rate(dataset)
