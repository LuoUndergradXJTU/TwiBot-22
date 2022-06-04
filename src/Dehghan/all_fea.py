
from re import L
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
#load features
dataset='Twibot-22'
#nlp features 
nlp= np.load('Twibot-22/nlp_22.npy')
id_include=np.load('Twibot-22/id_include.npy',allow_pickle=True)
id_include=list(id_include.item())
#profile features
nlp=nlp
p=np.load('Twibot-22/profile.npy')
p=p
#graph features
gf=np.load('Twibot-22/node_fea.npy').T

#embeddings

#deepwalk
model1=Word2Vec.load(dataset+'/deepwalk.model')
#model.load('Twibot-20/deepwalk.model')
deepwalk=[]
for i in range(len(model1.wv)):
    deepwalk.append(model1.wv[i])
deepwalk=np.array(deepwalk) #(11826, 128)
deepwalk=np.load('Twibot-22/deepwalk_22.npy')
#node2vec
model2=Word2Vec.load('Twibot-22/node2vec.model')
node2vec=[]
for i in range(len(model2.wv)):
    node2vec.append(model2.wv[i])
node2vec=np.load('Twibot-22/node2vec_emb.npy') #(11826, 128)

#role2vec
role2vec=np.load('Twibot-22/role2vec_fea.npy')#(11826, 128)


#roix
roix=np.load('Twibot-22/roles.npy') #(11826,1)

# graph_wave
graph_wave=np.load('Twibot-22/graph_fea_chi.npy') #(11826, 12)

# struct2vec (11826, 128)
struct_emb=np.loadtxt('Twibot-22/struct_22.emb')
# struct=np.zeros((11826,128))
# with open('codes/struc2vec/struct_all.emb','r') as f:
#     txt=f.readlines()[1:]
#     for line in txt:
#         line=line.split(' ')
#         index=eval(line[0])
#         fea=[]
#         for num in line[1:]:
#             fea.append(eval(num))
#         fea=np.array(fea)
#         struct[index]=fea


#emb=np.concatenate((node2vec,graph_wave),1)
#np.save('Twibot-22/emb.npy',emb)
np.save(dataset+'/deepwalk_22.npy',deepwalk)
np.save('Twibot-22/node2vec_22.npy',node2vec)
np.save('Twibot-22/role2vec_22.npy',role2vec)
np.save('Twibot-22/roix_emb_22.npy',roix)
np.save('Twibot-22/graph_wave_22.npy',graph_wave)
np.save('Twibot-22/struct_22.npy',struct_emb)
# print()