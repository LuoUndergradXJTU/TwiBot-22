import pandas as pd
import time
import sys
import os

def main(argv):
    if argv[1] == '--datasets':
        try:
            name = argv[2]
            return name
        except:
            return "Wrong command!"
    else:
        return "Wrong command!"


def is_same(df1, df2, val):
    df1_val = df1[val]
    df2_val = df2[val]
    compare = set(df1_val == df2_val)
    if False in compare:
        return False
    else:
        return True
        

def find_del(part, all, val):
    a = list(part[val])
    b = list(all[val])
    dell = list(set(b) - set(a))
    return dell        


def change(m):
    length = len(m)
    for i in range(length):
        if m[i] == 'bot':
            m[i] = 1
        else:
            m[i] = 0
    return m
    
if __name__ == '__main__':
    start = time.time()
    dataset_name = main(sys.argv)
    val = 'id'
    
    uf = pd.read_csv("{}/user_feature.csv".format(dataset_name))
    uf.sort_values(by=['id'], ascending=True, inplace=True, ignore_index = True)

    cflag = False
    if os.path.exists("{}/content_feature.csv".format(dataset_name)):
        cflag = True
        cf = pd.read_csv("{}/content_feature.csv".format(dataset_name))
        cf.sort_values(by=['id'], ascending=True, inplace=True, ignore_index=True)

    label = pd.read_csv("./datasets/{}/label.csv".format(dataset_name))
    delt = find_del(uf, label, val)
    label = label[~label['id'].isin(delt)]
    label.sort_values(by=['id'], ascending=True, inplace=True, ignore_index=True)
    # label = label.reset_index(drop=True)
    bot_human = list(label['label'])
    bot_human = change(bot_human)
    label['label'] = bot_human

    split = pd.read_csv("./datasets/{}/split.csv".format(dataset_name))
    dels = find_del(uf, split, val)
    split.sort_values(by=['id'], ascending=True, inplace=True, ignore_index = True)
    split = split[~split['id'].isin(dels)]
    split = split.reset_index(drop=True)


    if cflag:
        uc = is_same(uf, cf, val)
    else:
        uc = True  # for those which don't have content feature
    ul = is_same(uf, label, val)
    us = is_same(uf, split, val)
    print(uc, ul, us)

    if (uc and ul and us):
        if cflag:
            cff = cf.drop(columns=["id"], axis=1)
            df = pd.concat([uf, cff, label['label'], split['split']], axis=1)
        else:
            df = pd.concat([uf, label['label'], split['split']], axis=1)
        df.to_csv('{}/features.csv'.format(dataset_name), index=False)
    else:
        print('User id do not match!!!')
     
    
    
    
    
    
    
    
    
    
    