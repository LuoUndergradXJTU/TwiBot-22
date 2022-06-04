import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def main(argv):
    if argv[1] == '--datasets':
        try:
            name = argv[2]
            return name
        except:
            return "Wrong command!"
    else:
        return "Wrong command!"


if __name__ == '__main__':
    dataset_name = main(sys.argv)
    df = pd.read_csv("{}/features.csv".format(dataset_name))

    train = df[df['split'].isin(['train'])]
    val = df[df['split'].isin(['valid'])] # In some datasets the valid_set is labeled as 'val' not 'valid'
    test = df[df['split'].isin(['test'])]
    '''print(train)
    print(val)
    print(test)
    '''
    X_train, y_train = train.drop(columns=["id", "label", "split"], axis=1), train["label"]
    X_test, y_test = test.drop(columns=["id", "label", "split"], axis=1), test["label"]
    X_val, y_val = val.drop(columns=["id", "label", "split"], axis=1), val["label"]

    rf = RandomForestClassifier(n_estimators=150)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_ = auc(fpr, tpr)
    confusion_matrix = confusion_matrix(y_test, y_pred)

    print('acc:', acc)
    print('precision:', precision)
    print('recall:', recall)
    print('f1score:', f1score)
    print('mcc:', mcc)
    print('auc:', auc_)
    print('confusion_matrix:\n', confusion_matrix)

    f = open('{}/results.txt'.format(dataset_name), 'a')
    f.write(str(acc) + '\n')
    f.write(str(precision) + '\n')
    f.write(str(recall) + '\n')
    f.write(str(f1score) + '\n')
    f.write(str(mcc) + '\n')
    f.write(str(auc_) + '\n\n')
    f.close()
