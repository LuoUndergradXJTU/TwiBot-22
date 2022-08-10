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

In expert annotation step, we select 1000 users and assign each user to 5 Twitter bot detection experts to identify if it is a bot. Their annotation results are listed in ```label.manual.csv```