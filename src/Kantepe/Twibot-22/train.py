import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, auc, roc_curve, \
    roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from tqdm import tqdm
torch.manual_seed(100) #保证每次运行初始化的随机数相同
chooses = ["Twibot-22"]
for choose in chooses:
        for i in range(1):

                path = "/data2/whr/zqy/53/" + choose + "second.json"
                df = pd.read_json(path)
                df.replace([np.inf, -np.inf, "", np.nan], 0, inplace=True)

                # df.fillna('100')

                X = df.drop(columns=["id", "label","index_x","index_y","public_metrics"], axis=1)
                y = df[["split","label"]]

                #sex_mapping = {"False": 0, "True": 1}
                #X = X.map(sex_mapping)



                for u in X.columns:
                        if X[u].dtype == bool:
                                X[u] = X[u].astype('int')

                for i in list(X.columns) :
                        if X[i].dtype!=object:
                                # 获取各个指标的最大值和最小值
                                Max = np.max(X[i])
                                Min = np.min(X[i])
                                X[i] = (X[i] - Min) / (Max - Min+1e-6)
                                X[i] =X[i].astype('float32')

                X_train=X[X.split == "train"]
                X_test = X[X.split == "test"]
                y_train=y[y.split == "train"]
                y_test=y[y.split == "test"]

                X_train = X_train.drop(['split'], axis=1)
                X_test = X_test.drop(['split'], axis=1)
                y_train = y_train.drop(['split'], axis=1)
                y_test = y_test.drop(['split'], axis=1)

                thenum=len(y_train)

###由于训练集不平衡，复制一下负例，等价于给负例的梯度下降更多的权重

                is_hol = y_train['label'] == 1
                print("this")
                print( is_hol)
                df_y = y_train[is_hol]
                df_x = X_train[is_hol]

                X_train=X_train.append([df_x] * 3, ignore_index=True)
                y_train=y_train.append([df_y] * 3, ignore_index=True)

                
                
                #from imblearn.over_sampling import RandomOverSampler
                #ros = RandomOverSampler(random_state=0)
                #X_train,  y_train= ros.fit_resample(X_train,  y_train)
                '''
                from imblearn.over_sampling import SMOTE, ADASYN
                ros = SMOTE(random_state=0)
                X_train,  y_train =ros.fit_resample(X_train,  y_train)
                '''


                y_true = y_test
                print(X_train)
                print(X_train.dtypes)
                print(y_train)
                print(y_train.dtypes)

                Random_Forest = RandomForestClassifier(n_estimators=100)
                Random_Forest.fit(X_train, y_train)

                y_pred = Random_Forest.predict(X_test)
                #y_pred_quant = Random_Forest.predict_proba(X_test)[:, 1]
                print(y_pred)
                print(y_true)




                """
                from sklearn import ensemble
                clf = ensemble.GradientBoostingRegressor(n_estimators=100,
                                                         max_depth=3,
                                                         loss='squared_error')
                gbdt_model = clf.fit(X_train, y_train)
                y_pred = gbdt_model.predict(X_test) # predict
                """

                """
                import sklearn.svm as svm
                model = svm.SVC(C=10,kernel='linear')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)  # predict
                """

                """
                print(y_pred)
                #y_pred =  torch.tensor(y_pred)
                #y_pred = torch.sigmoid(y_pred)
                #y_pred = y_pred.numpy()

                y_pred = np.around(y_pred, 0).astype(int)  # .around()是四舍五入的函数 第二个参数0表示保留0位小数，也就只保留整数！！ .astype(int) 将浮点数转化为int型
                print(y_pred)
                print(y_true)
               """




                ACC = accuracy_score(y_true, y_pred)
                Precision = precision_score(y_true, y_pred)
                Recall = recall_score(y_true, y_pred)
                F1_score = f1_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)  # pos_label='human'
                auc_ = auc(fpr, tpr)
                ROC = roc_auc_score(y_true, y_pred)
                Confusion_Matrix = confusion_matrix(y_true, y_pred,labels=[1,0])

                print('ACC:', ACC)
                print('Precision:', Precision)
                print('Recall:', Recall)
                print('F1_score:', F1_score)
                print('ROC:', ROC)
                print('auc:', auc_)
                print(Confusion_Matrix)




