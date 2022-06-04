from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


metrics = {
    "Acc": accuracy_score,
    "Pre": precision_score,
    "Rec": recall_score,
    "MiF": f1_score,
    "AUC": roc_auc_score,
    # "Auc":
}


def evaluate_on_all_metrics(y_true, y_pred):
    ret = {}
    for name, func in metrics.items():
        ret[name] = func(y_true, y_pred)
        
    return ret