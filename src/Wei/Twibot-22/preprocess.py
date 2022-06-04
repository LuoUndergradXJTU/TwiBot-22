# user feature generation: Alhosseini
import re
import json
import datetime
from operator import itemgetter
from math import log
import pandas as pd
import numpy as np
import csv
import collections
def read_json_1dict(path):
    """
    读取json，用于一个dict
    :param path: 路径
    :return: 一个dict
    """
    with open(path, "r", encoding='utf-8') as r:
        dic = json.load(r)
    #print(dic)
    return dic
# path="/data2/whr/zqy/327/"
# chooses = ["gilani-2017","cresci-stock-2018","cresci-rtbust-2019","cresci-2015","botometer-feedback-2019"]
chooses = ["Twibot-22"]

for choose in chooses:
    path = "/data2/whr/TwiBot22-baselines/datasets/" + choose + "/"

    dl = pd.read_csv(path + "label.csv")
    ds = pd.read_csv(path + "split.csv")
    ds = ds[ds.split != "support"]
    ds = pd.merge(ds, dl, left_on='id', right_on='id')

    #de = pd.read_csv(path + 'edge.csv')
    #de = de[de.relation == "post"]
    #de = de[de.source_id.isin(ds.id)]

    #dsde = pd.merge(ds, de, left_on='id', right_on='source_id')
    #del dsde["source_id"]
    #print( ds)

    ###大表拼接user和id_tweet
    data = pd.read_json(path + "user.json")
    data2 = read_json_1dict("/data2/whr/TwiBot22-baselines/src/twibot22_Botrgcn_feature/id_tweet.json")
    df = pd.DataFrame.from_dict(data2, orient='index')
    df.transpose()
    df.fillna('null', inplace=True)
    #print(df)



    #df['text'] = [''.join(i + ' ') for i in df.values]
    #df = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], axis=1)
    new_col = ['text0', 'text1', 'text2', 'text3','text4','text5','text6','text7','text8','text9','text10',
                        'text11','text12', 'text13','text14','text15','text16','text17','text18','text19','text20',]
    df.columns = new_col



    #data = pd.concat([data, df], axis=1)
    #print(df)
    #print(data)

    #data=pd.merge(data,df,left_index=True,right_index=True)
    #data=pd.concat([data, df], axis=1, join='inner')
    data=data.reset_index()
    df=df.reset_index()
    #print(data.index)
    #print(df.index)
    data = pd.merge(data, df, left_index=True, right_index=True,how='outer')
    #data=data.insert(df)
    print(data)


###
    dsde = pd.merge(ds, data, left_on='id', right_on='id')








    out2=dsde
    out2 = out2.drop(['created_at', 'description', 'entities', 'location', 'name', 'pinned_tweet_id',
                      'profile_image_url', 'url', 'username', 'withheld'], axis=1)
    out2 = out2.drop(['index_x', 'protected', 'public_metrics','verified', 'index_y'], axis=1)

    print(out2.columns)
    print(out2)

    form0 = out2[['id', 'split', 'label','text0']]
    form0.rename(columns={'text0':'text'}, inplace=True)

    form1 = out2[['id', 'split', 'label', 'text1']]
    form1.rename(columns={'text1': 'text'}, inplace=True)
    formfinal=pd.concat([form0, form1])

    form1 = formfinal
    form1.drop(form1.index[(form1['text'] == 'null')], inplace=True)
    form1 = form1.reset_index()
    form1.to_json("./" + choose + "lstm1" + ".json")

    print(form0.columns)
    print(form0)

    form2 = out2[['id', 'split', 'label', 'text2']]
    form2.rename(columns={'text2': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal,form2])

    form3 = out2[['id', 'split', 'label', 'text3']]
    form3.rename(columns={'text3': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal,form3])

    form4 = out2[['id', 'split', 'label', 'text4']]
    form4.rename(columns={'text4': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal,form4])

    form5 = out2[['id', 'split', 'label', 'text5']]
    form5.rename(columns={'text5': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal,form5])

    form5 = formfinal
    form5.drop(form5.index[(form5['text'] == 'null')], inplace=True)
    form5 = form5.reset_index()
    form5.to_json("./" + choose + "lstm5" + ".json")

    form6 = out2[['id', 'split', 'label', 'text6']]
    form6.rename(columns={'text6': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form6])

    form7 = out2[['id', 'split', 'label', 'text7']]
    form7.rename(columns={'text7': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form7])

    form8 = out2[['id', 'split', 'label', 'text8']]
    form8.rename(columns={'text8': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form8])

    form9 = out2[['id', 'split', 'label', 'text9']]
    form9.rename(columns={'text9': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form9])

    form10 = out2[['id', 'split', 'label', 'text10']]
    form10.rename(columns={'text10': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form10])
    
    form10=formfinal
    form10.drop(form10.index[(form10['text'] == 'null')], inplace=True)
    form10 = form10.reset_index()
    form10.to_json("./" + choose + "lstm10" + ".json")



    form11 = out2[['id', 'split', 'label', 'text11']]
    form11.rename(columns={'text11': 'text'}, inplace=True)
    formfinal = pd.concat([form0, form11])

    form12 = out2[['id', 'split', 'label', 'text12']]
    form12.rename(columns={'text12': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form12])

    form13 = out2[['id', 'split', 'label', 'text13']]
    form13.rename(columns={'text13': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form13])

    form14 = out2[['id', 'split', 'label', 'text14']]
    form14.rename(columns={'text14': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form14])

    form15 = out2[['id', 'split', 'label', 'text15']]
    form15.rename(columns={'text15': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form15])

    form16 = out2[['id', 'split', 'label', 'text16']]
    form16.rename(columns={'text16': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form16])

    form17 = out2[['id', 'split', 'label', 'text17']]
    form17.rename(columns={'text17': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form17])

    form18 = out2[['id', 'split', 'label', 'text18']]
    form18.rename(columns={'text18': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form18])

    form19 = out2[['id', 'split', 'label', 'text19']]
    form19.rename(columns={'text19': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form19])

    form20 = out2[['id', 'split', 'label', 'text20']]
    form20.rename(columns={'text20': 'text'}, inplace=True)
    formfinal = pd.concat([formfinal, form20])

    form0.drop(form0.index[(form0['text'] == 'null')], inplace=True)
    form0 = form0.reset_index()
    form0.to_json("./" + choose + "lstm0" + ".json")

    formfinal.drop(formfinal.index[(formfinal['text'] == 'null')], inplace=True)
    formfinal = formfinal.reset_index()
    print(formfinal.columns)
    print(formfinal)
    formfinal.to_json("./" + choose + "lstm20" + ".json")