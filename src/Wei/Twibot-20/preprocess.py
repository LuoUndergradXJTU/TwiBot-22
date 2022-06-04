# user feature generation: Alhosseini

import json
import datetime
import pandas as pd
import numpy as np
import csv

#path="/data2/whr/zqy/327/"
#chooses = ["gilani-2017","cresci-stock-2018","cresci-rtbust-2019","cresci-2015","botometer-feedback-2019","Twibot-20"]
chooses = ["cresci-2015"]



for choose in chooses:
    path = "/data2/whr/TwiBot22-baselines/datasets/" + choose + "/"
    dl = pd.read_csv(path+"label.csv")
    ds = pd.read_csv(path+"split.csv")
    ds = ds[ds.split != "support"]
    ds = pd.merge(ds, dl,  left_on='id', right_on='id')


    de = pd.read_csv(path+'edge.csv')
    de = de[de.relation == "post"]
    de = de[de.source_id.isin(ds.id) ]



    dsde = pd.merge(ds, de,  left_on='id', right_on='source_id')
    del dsde["source_id"]

    data=pd.read_json(path+"node.json")
    data=data[['id','text']]
    out=pd.merge(dsde, data,  left_on='target_id', right_on='id')
    out.dropna(inplace = True)
    out.to_json("./" + choose + "1"+".json")
    #out.to_csv('./a.csv')
"""
f2 = open(path+"node.json", 'r')
dj = json.load(f2)

out=pd.merge(dsde, dj,  left_on='id', right_on='id')

print(out)
"""
"""
file = open(path+"node.json", 'r')
js = file.read()
data_list = json.loads(js)
data_df = pd.DataFrame(data_list, index=[0])
out=pd.merge(dsde, data_df,  left_on='id', right_on='id')
"""

"""
data=pd.read_json(path+"node.json")
out=pd.merge(dsde, data,  left_on='target_id', right_on='id')
print(out)
"""
#         #user_features.append(torch.tensor(feature_temp))
# torch.save(torch.tensor(user_features), 'user_feature1.pt')
# temp = torch.load('user_feature1.pt')
# print(temp.size())
