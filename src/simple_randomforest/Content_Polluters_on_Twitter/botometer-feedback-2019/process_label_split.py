import pandas as pd

label = pd.read_csv('/data2/whr/czl/TwiBot22-baselines/datasets/botometer-feedback-2019/label.csv')
split = pd.read_csv('/data2/whr/czl/TwiBot22-baselines/datasets/botometer-feedback-2019/split.csv')

user = pd.read_csv('./user_feature.csv')
user_list = list(user['id'])

del_list = []
bool_label = []


for i,r in label.iterrows():
    if r[1] == 'human':
        bool_label.append(1)
    else:
        bool_label.append(0)
    if r[0] not in user_list:
        del_list.append(i)

label.insert(loc=len(label.columns), column='labels', value=bool_label)
label.drop(index = del_list,inplace = True)
label.sort_values(by=['id'], ascending=True, inplace=False)
label = label.drop_duplicates(subset='id')
label.to_csv('./new_label.csv',index=False)

split.drop(index = del_list, inplace = True)
split.sort_values(by=['id'], ascending=True, inplace=False)
split = split.drop_duplicates(subset='id')
split.to_csv('./split.csv', index= False)