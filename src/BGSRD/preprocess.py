# %%
import csv
import json
import torch
from argparse import ArgumentParser
import re
import torch

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
store_path = './data/'
source_path = '../../datasets/'
dataset_name = 'Twibot-22'
#%%


#%%
parser = ArgumentParser()
parser.add_argument('--source_path', default="../../datasets/")
parser.add_argument('--dataset', default="Twibot-22")



args = parser.parse_args()
source_path = args.source_path
'''
These code aiming at transfer the orginal dataset into the format suiting these model, providing source data for build_graph.py
'''
#%%
dataset_name = args.dataset
# %%

label_csv = csv.reader(open(source_path + dataset_name +"/label.csv"))
split_csv = csv.reader(open(source_path + dataset_name +"/split.csv"))
#%%
split_dict = {}
count = 0
print(f'Dataset: {dataset_name}')
for line in split_csv:
    if count == 0:
        count += 1
        continue
    if line[1] != 'support':
        split_dict[line[0]] = [line[1]]
for line in label_csv:
    if line[0] in split_dict:
        split_dict[line[0]].append(line[1])

user_file = 'user'
if dataset_name != "Twibot-22":
    user_file = 'node'
with open(source_path + dataset_name +f'/{user_file}.json') as f:
    users = json.load(f)
    for user in users:
        if user['id'] in split_dict:
            split_dict[user['id']].append(user['description'])


user_label = []
for user in split_dict:
    user_label.append(user + '\t' + split_dict[user][0] + '\t' + split_dict[user][1])



clean_description = []
for user in split_dict:
    try:
        clean_description.append(clean_str(split_dict[user][2]))
    except:
        clean_description.append('')



torch.save(clean_description,store_path + dataset_name +'_description.pt')
torch.save(user_label,store_path+ dataset_name +'.pt')


print('Finish meta data')

if dataset_name == "Twibot-22 ":
    from transformers import pipeline
    import torch
    from transformers import *
    pretrained_weights = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights, model_max_length = 500)
    feature_extractor = pipeline('feature-extraction', model = RobertaModel.from_pretrained(pretrained_weights), tokenizer = tokenizer, device = 0)

    dataset = dataset_name
    SourcePath = './data/'
    corpus_file = SourcePath +dataset+'_description.pt'
    sentences = torch.load(corpus_file)

    count = 0
    SenEmb = [];
    print("Embedding Begin")
    for sentence in sentences:
        count += 1
        try:
            this_sentence = torch.zeros(768)
            sentence_temp = torch.tensor(feature_extractor(sentence))
            sentence_temp = torch.mean(sentence_temp.squeeze(0), 0)
            this_sentence = this_sentence + sentence_temp
            SenEmb.append(this_sentence)
        except:
            SenEmb.append(torch.randn(768))
            print('error')
        if count % 1000 == 0:
            print(count / len(sentences))

    torch.save(torch.stack(SenEmb), SourcePath + f'{dataset}_Embedding.pt')

#%%
import random
count = 50000
clean_description = torch.load(store_path + dataset_name +'_description.pt')
print('build_graph begin')
edge_index = [[],[]]
# build vocab
word_set = set()
for doc_words in clean_description:
    words = doc_words.split()
    for word in words:
        word_set.add(word)

vocab = list(word_set)
vocab_size = len(vocab)
random.shuffle(vocab)

word_edge_dict = {}

word_id_map = {}
for i in range(vocab_size):
    if i < count:
        word_id_map[vocab[i]] = i + len(clean_description)
    else:
        word_id_map[vocab[i]] = False

for i in range(len(clean_description)):
    doc_words = clean_description[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        elif word_id_map[word]:
            edge_index[0].append(i)
            edge_index[1].append(word_id_map[word])
            appeared.add(word)
    appeared = list(appeared)
    for word in appeared:
        for word_co in appeared:
            if word_co == word:
                continue
            else:
                edge_index[0].append(word_id_map[word])
                edge_index[1].append(word_id_map[word_co])
#%%
edge_index = torch.LongTensor(edge_index)




# %%
print(f'Finish edge_index={edge_index.size()}, vocab_size={len(vocab)}, sentences_size={len(clean_description)}')

# %%
edge_index_new = {}
edge_index_new['edge_index'] = edge_index
edge_index_new['word_size'] = min(len(vocab),count)
torch.save(edge_index_new,f'{store_path}{dataset_name}_edge_index.pt')
# %%

