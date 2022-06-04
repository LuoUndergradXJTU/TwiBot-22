import pandas as pd

label = pd.read_csv('/data2/whr/czl/TwiBot22-baselines/datasets/botwiki-2019/label.csv')

main_user = pd.read_csv('./user_feature.csv')
user = main_user.loc[0:len(main_user)-1]
user_list = list(user['user_id'])

del_list = []
bool_label = []

for i,r in label.iterrows():
    if r[1] == 'human':
        bool_label.append(1)
    else:
        bool_label.append(0)
    if r[0] not in user_list:
        del_list.append(i)

print(len(del_list))

label.insert(loc=len(label.columns), column='labels', value=bool_label)
label.drop(index = del_list,inplace = True)
label.sort_values(by=['id'], ascending=True, inplace=True)
label = label.drop_duplicates(subset=['id'])

label.to_csv('./new_label.csv',index=False)

user = pd.read_csv('./user_feature.csv')
user.sort_values(by=['user_id'], ascending=True, inplace=True)
final = pd.read_csv('./new_label.csv')
if len(user) == len(final):
    user.insert(loc=len(user.columns), column='label', value=list(final.loc[0:len(final)-1]['labels']))
    user.to_csv('./feature.csv',index=False)

