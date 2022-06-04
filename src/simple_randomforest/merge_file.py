import pandas as pd

def get_user_list():
    data = pd.read_csv('./user_post.csv')
    length = len(data)
    all_user = data.loc[0:length-1]
    all_user = list(all_user['id'])
    return all_user
    
if __name__ == '__main__':
    user_list = get_user_list()
    label = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/label.csv")
    del_list = []
    for i,r in label.iterrows():
        if r[0] not in user_list:
            del_list.append(i)
    label.drop(index = del_list,inplace = True)
    header = ['id','label']
    label.to_csv('./new_label.csv')