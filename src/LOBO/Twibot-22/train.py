from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
train_features = np.loadtxt('train_features.txt')
train_labels = np.loadtxt('train_labels.txt')
valid_features = np.loadtxt('valid_features.txt')
valid_labels = np.loadtxt('valid_labels.txt')
test_features = np.loadtxt('test_features.txt')
test_labels = np.loadtxt('test_labels.txt')
accuracy = []
precision = []
f1 = []
recall = []
auc = []

for random_seed in tqdm([100,200,300,400,500], 'Training and testing repeatedly for 5 times(about 10 minutes each time)'):
    clf = RandomForestClassifier(max_depth=20, criterion='entropy', random_state=random_seed)
    clf.fit(train_features, train_labels)
    test_predict = clf.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predict)
    accuracy.append(test_accuracy)
    test_precision = precision_score(test_labels, test_predict)
    precision.append(test_precision)
    test_recall = recall_score(test_labels, test_predict)
    recall.append(test_recall)
    test_f1 = f1_score(test_labels, test_predict)
    f1.append(test_f1)
    test_auc = roc_auc_score(test_labels, test_predict)
    auc.append(test_auc)

def mean(metrics):
    return np.mean(np.array(metrics))

def std(metrics):
    return np.std(np.array(metrics), ddof=1)

print('test_accuracy:{:.4f} {:.4f}'.format(mean(accuracy),std(accuracy)))
print('test_precision:{:.4f} {:.4f}'.format(mean(precision),std(precision)))
print('test_recall:{:.4f} {:.4f}'.format(mean(recall),std(recall)))
print('test_f1:{:.4f} {:.4f}'.format(mean(f1),std(f1)))
print('test_auc:{:.4f} {:.4f}'.format(mean(auc),std(auc)))
        
