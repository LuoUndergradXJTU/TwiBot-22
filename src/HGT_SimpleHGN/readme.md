### Hetergenour Graph Transformers
- **authors**: Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun
- **link**: https://arxiv.org/abs/2003.01332


### Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks
- **authors**: Qingsong Lv, Ming Ding, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, Jie Tang
- **link**: https://arxiv.org/abs/2112.14936

---

- **file structure**: 

```python
├── cresci-2015
│   ├── HGT.py  # train HGT on cresci-2015
│   ├── SimpleHGN.py # train SimpleHGN on cresci-2015
│   ├── Dataset.py
│   └── layer.py
├── Twibot-20    
│   ├── HGT.py  # train HGT on Twibot-20
│   ├── SimpleHGN.py # train SimpleHGN on cresci-2015
│   ├── Dataset.py
│   └── layer.py
└── Twibot-22
    ├── HGT_sample.py  # train HGT on Twibot-22
    ├── SimpleHGN_sample.py # train SimpleHGN on Twibot-22
    └── layer.py
```

- **implement details:**

1. We just consider the graph with user as node and their follow relations (e.g. follower & following) as edge due to the limitation of computation resources.

2. We fix HGT and SimpleHGN's attention head to 1 for simplicity.

3. To enable the training on large-scale graph such as Twibot-22, we adopt the [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader) in torch geometric with num_neighbors set to 20 and sample for 2 iterations.
  
4. Input features for user node is taken from BotRGCN, more details please refer to [BotRGCN imlementation](https://github.com/BunsenFeng/BotRGCN)


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


#### How to reproduce:
Please change the path argument into your own path to preprocess files

* reproduce HGT on cresci-2015 & Twibot-20
```python
CUDA_VISIBLE_DEVICES=0 python HGT.py --batch_size 128 --epochs 200 --path /path/to/preprocess/file
```
* reproduce HGT on Twibot-22
```python
CUDA_VISIBLE_DEVICES=0 python HGT.py --batch_size 128 --epochs 200 --path /path/to/preprocess/file
```

* reproduce SimpleHGN on cresci-2015 & Twibot-2015
```python
CUDA_VISIBLE_DEVICES=0 python SimpleHGN.py --batch_size 128 --epochs 200 --path /path/to/preprocess/file
```

* reproduce SimpleHGN on Twibot-22 dataset
```python
CUDA_VISIBLE_DEVICES=0 python SimpleHGN.py --batch_size 128 --epochs 200 --path /path/to/preprocess/file
```



#### Result:

##### HGT

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9603 | 0.9480   | 0.9911 | 0.9693 |
| Cresci-2015 | std  | 0.0032 | 0.0049   | 0.0012 | 0.0024 |
| Twibot-20   | mean | 0.8691 | 0.8555   | 0.9100 | 0.8819 |
| Twibot-20   | std  | 0.0024 | 0.0031   | 0.0057 | 0.0019 |
| Twibot-22   | mean | 0.7491 | 0.6822   | 0.2803 | 0.3960 |
| Twibot-22   | std  | 0.0013 | 0.0271   | 0.0260 | 0.0211 |

##### SimpleHGN

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9671 | 0.9568   | 0.9929 | 0.9745 |
| Cresci-2015 | std  | 0.0054 | 0.0090   | 0.0040 | 0.0040 |
| Twibot-20   | mean | 0.8674 | 0.8476   | 0.9206 | 0.8825 |
| Twibot-20   | std  | 0.0022 | 0.0046   | 0.0051 | 0.0018 |
| Twibot-22   | mean | 0.7672 | 0.7257   | 0.3290 | 0.4544 |
| Twibot-22   | std  | 0.0027 | 0.0279   | 0.0164 | 0.0042 |





| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| HGT|0.7491|0.3960|F T G|`Graph Neural Networks`|
| SimpleHGN|0.7672|0.4544|F T G|`Graph Neural Networks`|
