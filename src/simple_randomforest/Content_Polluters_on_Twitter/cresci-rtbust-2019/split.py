import pandas as pd

split = pd.read_csv('/data2/whr/czl/TwiBot22-baselines/datasets/cresci-rtbust-2019/split.csv')

user = pd.read_csv('./user_feature.csv')
user_list = list(user['id'])

del_list = []

for i,r in split.iterrows():
    if r[0] not in user_list:
        del_list.append(i)

print(len(del_list))

split.drop(index = del_list, inplace = True)
split.sort_values(by=['id'], ascending=True, inplace=True)
split = split.drop_duplicates(subset='id')
print(len(split))

split.to_csv('./split.csv', index= False)