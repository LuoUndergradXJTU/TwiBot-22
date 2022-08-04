### Heterogeneity-aware Twitter Bot Detection with Relational Graph Transformers

---

- **authors**: Shangbin Feng, Zhaoxuan Tan, Rui Li, Minnan Luo

- **link**: https://arxiv.org/abs/2109.02927

- **file structure**: 

```python
├── cresci-15
│   ├── Heterobot.py  # train model on cresci-2015
|   ├── Dataset.py
|   └── layer.py
├── Twibot-20    
│   ├── Heterobot.py  # train model on Twibot-20
|   ├── Dataset.py
|   └── layer.py
└── Twibot-22
    ├── Heterobot_sample.py  # train model on Twibot-22
    └── layer.py
```

- **Requirements**

```
CUDA==10.2
pytorch-lightning==1.4.9
torch==1.9.1
torch-geometric==2.0.2
torch-cluster==1.5.9
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
sklearn
argparse
```

- **implement details**: 

1. The input features of our implementation is the same as BotRGCN, for more details about preprocessing please refer to [BotRGCN](https://github.com/BunsenFeng/BotRGCN).

2. To enable the training on large-scale graph such as Twibot-22, we adopt the [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader) in torch geometric with num_neighbors set to 20 and sample for 2 iterations.

#### How to reproduce:

1. run the preprocess code to get following files
```
cat_properties_tensor.pt
des_tensor.pt
edge_index.pt
edge_type.pt
label.pt
num_properties_tensor.pt
test_idx.pt
train_idx.pt
val_idx.pt
test_idx.pt
```

2. train the model

Please change the path to proprocess file in the following commands

* on Twibot-20 & cresci-2015
```python
CUDA_VISIBLE_DEVICES=0 python Heterobot.py --batch_size 128 --epochs 200 --path /path/to/preprocessed/file
```

* on Twibot-22
```python
CUDA_VISIBLE_DEVICES=0 python Heterobot_sample.py --batch_size 256 --epochs 200 --path /path/to/preprocessed/file
```

#### Results:

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9715 | 0.9638   | 0.9923 | 0.9778 |
| Cresci-2015 | std  | 0.0032 | 0.0059   | 0.0015 | 0.0024 |
| Twibot-20   | mean | 0.8657 | 0.8515   | 0.9106 | 0.8801 |
| Twibot-20   | std  | 0.0041 | 0.0028   | 0.0080 | 0.0041 |
| Twibot-22   | mean | 0.7647 | 0.7503   | 0.3010 | 0.4294 |
| Twibot-22   | std  | 0.0045 | 0.0085   | 0.0017 | 0.0049 |





| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| RGT | 0.7647 | 0.4294 | F T G |`Graph Neural Networks`|

