import numpy as np
from tqdm import tqdm
id_list=np.load('/data2/whr/lyh/twibot22_baseline/Twibot-22/id.npy')
id_include=(np.load('/data2/whr/lyh/baseline2/Twibot-22'+'/id_include.npy',allow_pickle=True))
id_include=list(id_include.item())
id_list=list(id_list)
train_id=[]
val_id=[]
test_id=[]

# with open(r'/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/split.csv','r') as f:
#     lines=f.readlines()[1:]
#     for line in tqdm(lines):
        
#         line=line.split(',')
#         curr=eval(line[0][1:])
#         curr=id_list.index(curr)
#         try:
#             curr=id_include.index(curr)
#             if(line[1]=='train'):
#                 train_id.append(curr)
#             elif(line[1]=='valid'):
#                 val_id.append(curr)
#             else:
#                 test_id.append(curr)
#         except:
#             pass

# print(f"train_size:{len(train_id)} val_size:{len(val_id)} test_size:{len(test_id)} all:{len(id_include)}")
# np.save('/data2/whr/lyh/baseline2/Twibot-22/'+'train_id.npy',np.array(train_id))
# np.save('/data2/whr/lyh/baseline2/Twibot-22/'+'val_id.npy',np.array(val_id))
# np.save('/data2/whr/lyh/baseline2/Twibot-22/'+'test_id.npy',np.array(test_id))



f_train=open('/data2/whr/lyh/baseline2/Twibot-22/train.txt','w')
f_val=open('/data2/whr/lyh/baseline2/Twibot-22/val.txt','w')
f_test=open('/data2/whr/lyh/baseline2/Twibot-22/test.txt','w')

id_list_dict={}
for i,d in enumerate(tqdm(id_list)):
    id_list_dict[d]=i
id_include_dict={}
for i,d in enumerate(tqdm(id_include)):
    id_include_dict[d]=i

with open(r'/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/split.csv','r') as f:
    lines=f.readlines()[1:]
    for line in tqdm(lines[:700000]):
        
        line=line.split(',')
        curr=eval(line[0][1:])
        curr=id_list_dict[curr]
        if not (curr in id_include):
            continue
        
        curr=id_include_dict[curr]
        
        train_id.append(curr)
    for line in tqdm(lines[700000:900000]):
        line=line.split(',')
        curr=eval(line[0][1:])
        curr=id_list_dict[curr]
        if not (curr in id_include):
            continue
        
        curr=id_include_dict[curr]
        
        val_id.append(curr)
    for line in tqdm(lines[900000:]):
        line=line.split(',')
        curr=eval(line[0][1:])
        curr=id_list_dict[curr]
        if not (curr in id_include):
            continue
        
        curr=id_include_dict[curr]
        
        test_id.append(curr)
        
print(f"train_size:{len(train_id)} val_size:{len(val_id)} test_size:{len(test_id)} all:{len(id_include)}")
np.save('/data2/whr/lyh/baseline2/Twibot-22/'+'train_id.npy',np.array(train_id))
np.save('/data2/whr/lyh/baseline2/Twibot-22/'+'val_id.npy',np.array(val_id))
np.save('/data2/whr/lyh/baseline2/Twibot-22/'+'test_id.npy',np.array(test_id))