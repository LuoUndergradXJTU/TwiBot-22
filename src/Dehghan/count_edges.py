import csv
import numpy as np
from tqdm import tqdm
import json
dataset_name='Twibot-22'
edges=[]
id_list=np.load('/data2/whr/lyh/twibot22_baseline/'+dataset_name+'/id.npy')
id_list=list(id_list)
f_write=open('/data2/whr/lyh/baseline2/edge_cleaned.txt','w')

count=0

with open('/data2/whr/czl/TwiBot22-baselines/datasets/'+dataset_name+'/'+ 'edge.csv','r') as csvfile:
    lines=csvfile.readlines()
    for row in tqdm(lines):

        row=row.rstrip()
        id1,relation,id2=row.split(',')
      
        if id1[0] == 's':
            continue
        try:
            id1=eval(id1[1:])
            id2=eval(id2[1:])
            #edges.append([id_list.index(id1),id_list.index(id2)])
            
            #edge.append([id1,id2])
            f_write.write(f'{id_list.index(id1)},{id_list.index(id1)}\n')
            #edge.append([id1,id2])
            count=count+1
        except:
            #print("missing")
            pass
print()
f_write.close()
#out_file=open('/data2/whr/lyh/baseline2/edge_cleaned.json','w')
#json.dump(edge,out_file)