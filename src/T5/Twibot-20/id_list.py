import json
import torch
import pandas as pd
from pathlib import Path
path1 = Path('datasets/Twibot-20/node.json')
with open(path1, 'r') as f:
    users = json.loads(f.read())
    users = pd.DataFrame(users)
    users = users['id'][users.id.str.contains('^u')]
    """
    in total (229580)
    """
user_idlist = []
print(users[0])
for user in users:
    try:
        user_idlist.append(user)
    except:
        continue

path2 = Path('src/T5/Twibot-20/id_list.json')
with open(path2, 'w') as f:
    json.dump(user_idlist, f)
# import json
# with open('src/RoBERTa/Twibot-20/id_list.json', 'r') as f:
#     data = json.loads(f.read())
#     print(len(data))