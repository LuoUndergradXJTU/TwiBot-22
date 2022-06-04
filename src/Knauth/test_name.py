import json
train_user=open(r'Twibot-22/train.json','r')
users=json.load(train_user)
all=len(users)
count=0
for user in users:
    if user['profile']['screen_name']=='':
        count=count+1
print(count/all)