# Manually Labeled Data for Twibot-22 

## file structure 

```python
├── Twibot20-anno_study.json # randomly selected users from Twibot-20 dataset
├── Twibot22-anno_study.json # randomly selected users from Twibot-22 dataset
├── annotation_combine.csv # manually annotation for annotation study
└── label_manual.csv # expert annotation for Twibot-22
```

## Annotation Study 
  
In the annotation study, we randomly select 500 users from Twibot-20 and Twibot-22, respectively (```Twibot20-anno_study.json, Twibot220-anno_study.json```). Each user is manually annotated by three different bot detection experts, their annotation results are listed in ```annotation_combine.csv```.

In ```annotation_combine.csv```, label means:
```python
0 = definitely HUMAN
1 = likely HUMAN
2 = Not sure
3 = likely BOT
4 = definitely BOT
```

## Expert Annotation Data

In expert annotation step, we select 1000 users and assign each user to 5 Twitter bot detection experts to identify if it is a bot. ```record_combine.csv``` contains the raw annotation data and the annotation results are listed in ```label.manual.csv```.


## Labeling Functions Ablation Study
We remove labeling functions in annotation process and compare their results with the full annotation model.

|  labeling function   |  bot->bot | bot->human | human->human | human->bot | changed (%)|
|  :----:  | :----:  | :----: | :----: | :----: | :----: |
|w/o  adaboost|77050|49665|869317| 3968 | 5.36|
|w/o random forest|117191|9524|841531|31754| 4.13|
|w/o MLP|109152|17563|796159|77126|9.46|
|w/o GCN|120925|5790|833776|39509|4.54|
|w/o GAT|123397|3318|824436|48849|5.22|
|w/o RGCN|123676|3042|819293|53970|5.70|
|w/o verify|122177|4538|873101|184|0.47|
|w/o keywords|123880|2835|840142|33143|3.60|
