import numpy as np
import torch as th
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def write_list(li):
    formattor = "{:.4f}"
    return ",".join(map(lambda x: formattor.format(x), li)) + "\n"

def training_one_data(data, label):
    data = np.array(data)
    rfc = RandomForestClassifier()
    rfc.fit(data, label)
    return rfc

def transfer():
    acc_r = open("./varol_acc.txt", "a")
    f1_r = open("./varol_f1.txt", "a")
    numbers = range(10)
    users = [th.load(f"./{x}.pt") for x in numbers]
    for user in users:
        rfc = training_one_data(**user)
        ret_acc = []
        ret_f1 = []
        for i in range(10):
            pred = rfc.predict(np.array(users[i]["data"]))
            ret_acc.append(accuracy_score(users[i]["label"], pred))
            ret_f1.append(f1_score(users[i]["label"], pred))
        acc_r.write(write_list(ret_acc))
        f1_r.write(write_list(ret_f1))
        
    acc_r.close()
    f1_r.close()
    
if __name__ == "__main__":
    transfer()