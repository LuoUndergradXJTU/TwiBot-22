import pandas as pd

label = pd.read_csv('/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/label.csv')

main_user = pd.read_csv('./user_feature.csv')
user = main_user.loc[0:len(main_user)-1]
user_list = list(user['user_id'])

del_list = []

for i,r in label.iterrows():
    if r[0] not in user_list:
        del_list.append(i)
print(len(del_list))
label.drop(index = del_list,inplace = True)
label.sort_values(by=['id'], ascending=True, inplace=True)

label.to_csv('./new_label.csv')