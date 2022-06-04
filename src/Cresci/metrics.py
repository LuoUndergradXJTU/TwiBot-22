from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


def metrics(y:  list, pred:  list):
    print("ACC: {}".format(accuracy_score(y, pred)))
    print("ROC: {}".format(roc_auc_score(y, pred)))
    print("F1: {}".format(f1_score(y, pred)))
    print("Precision: {}".format(precision_score(y, pred)))
    print("Recall: {}\n".format(recall_score(y, pred)))

if __name__ == '__main__':
    pass
