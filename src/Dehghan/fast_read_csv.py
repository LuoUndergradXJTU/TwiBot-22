import pandas
import numpy as np
from tqdm import tqdm
def read_single_csv(input_path):
    import pandas as pd
    df_chunk=pd.read_csv(input_path,chunksize=1000)
    res_chunk=[]
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df=pd.concat(res_chunk)
    return res_df
if __name__ == '__main__':
    dataset_name='Twibot-22'
    follow_edges=[]
    friend_edges=[]
    id_list=np.load('/data2/whr/lyh/twibot22_baseline/'+dataset_name+'/id.npy')
    id_list=list(id_list)
    data=read_single_csv('/data2/whr/czl/TwiBot22-baselines/datasets/'+dataset_name+'/'+ 'edge.csv')

    id_list=sorted(id_list)
    mark=0
    edges=[]
    for i in range(len(data)):
        if(eval(data.loc[i].source_id[1:])==id_list[0]):
            mark=i
            break
    data=data.loc[mark:]
    
    
    data=data.drop(['relation'],axis=1)
    data=list(data.values)
    for i in tqdm(data):
        edges.append([eval(i[0][1:]),eval(i[1][1:])])
    edges=np.array(edges)
    np.save(dataset_name+'/edges.npy',edges)
    print()