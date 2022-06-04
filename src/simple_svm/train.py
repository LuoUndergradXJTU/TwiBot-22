from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from preprocess import preprocess_dataset

import sys
sys.path.append("..")
from utils.eval import evaluate_on_all_metrics

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, default="botometer-feedback-2019")
    parser.add_argument("-s", "--server_id", type=str, default="209")
    
    args = parser.parse_args()

    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = preprocess_dataset(args.dataset, args.server_id)
    
    svm_classifier = Pipeline([
        ("standardlize", StandardScaler()),
        ("pca", PCA(100)),
        ("svm", svm.SVC())
    ])
    
    svm_classifier.fit(train_data, train_labels)
    
    train_pred = svm_classifier.predict(train_data)
    result = evaluate_on_all_metrics(train_labels, train_pred)
    print(result)
    
    test_pred = svm_classifier.predict(test_data)
    
    result = evaluate_on_all_metrics(test_labels, test_pred)
    print(result)