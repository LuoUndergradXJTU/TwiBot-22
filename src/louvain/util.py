from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn.functional as func


def null_metrics():
    return {
        'Acc': 0.0,
        'Pre': 0.0,
        'Rec': 0.0,
        'MiF': 0.0,
        'MCC': 0.0,
        'AUC': 0.0,
        'pr-auc': 0.0
    }


def calc_metrics(y, pred):
    assert y.dim() == 1 and pred.dim() == 2
    if torch.any(torch.isnan(pred)):
        metrics = null_metrics()
        plog = ''
        for key, value in metrics.items():
            plog += ' {}: {:.6}'.format(key, value)
        return metrics, plog
    pred = func.softmax(pred, dim=-1)
    pred_label = torch.argmax(pred, dim=-1)
    pred_score = pred[:, -1]
    y = y.to('cpu').numpy().tolist()
    pred_label = pred_label.to('cpu').tolist()
    pred_score = pred_score.to('cpu').tolist()
    precision, recall, _thresholds = precision_recall_curve(y, pred_score)
    metrics = {
        'Acc': accuracy_score(y, pred_label),
        'Pre': precision_score(y, pred_label),
        'Rec': recall_score(y, pred_label),
        'MiF': f1_score(y, pred_label),
        'AUC': roc_auc_score(y, pred_score),
        'MCC': matthews_corrcoef(y, pred_label),
        'pr-auc': auc(recall, precision)
    }
    plog = ''
    for key in ['Acc', 'MiF', 'MCC']:
        plog += ' {}: {:.6}'.format(key, metrics[key])
    return metrics, plog


def is_better(now, pre):
    if now['Acc'] != pre['Acc']:
        return now['Acc'] > pre['Acc']
    if now['MiF'] != pre['MiF']:
        return now['MiF'] > pre['MiF']
    if now['MCC'] != pre['MCC']:
        return now['MCC'] > pre['MCC']
    if now['pr-auc'] != pre['pr-auc']:
        return now['pr-auc'] > pre['pr-auc']
    if now['AUC'] != pre['AUC']:
        return now['AUC'] > pre['AUC']
    if now['Pre'] != pre['Pre']:
        return now['Pre'] > pre['Pre']
    if now['Rec'] != pre['Rec']:
        return now['Rec'] > pre['Rec']
    return False
