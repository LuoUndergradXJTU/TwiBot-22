import numpy as np
import json
import pandas as pd
import csv

import sys

sys.path.append('..')

from utils.dataset import merge_and_split, simple_vectorize

def preprocess_dataset(dataset, server_id="209"):
    train, valid, test = merge_and_split(dataset=dataset, server_id=server_id)
    
    train_data, train_labels = simple_vectorize(train)
    valid_data, valid_labels = simple_vectorize(valid)
    test_data, test_labels = simple_vectorize(test)
    
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

if __name__ == "__main__":
    preprocess_dataset("botometer-feedback-2019", "209")