import warnings
with warnings.catch_warnings():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# Do not give warning
warnings.filterwarnings('always')

metrics = {
    "Acc": accuracy_score,
    "Pre": precision_score,
    "Rec": recall_score,
    "MiF": f1_score,
    "AUC": roc_auc_score,
    "MCC": matthews_corrcoef
}


def evaluate_on_all_metrics(y_true, y_pred):
    ret = {}
    for name, func in metrics.items():
        try:
            ret[name] = func(y_true, y_pred)
        except:
            # invalid figure
            ret[name] = 0
        
    return ret