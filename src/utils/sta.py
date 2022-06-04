import os
from pathlib import Path

base = Path("/data2/whr/czl/TwiBot22-baselines/src")

datasets = os.listdir(base)

def has_markdown_file(files):
    for file in files:
        if file.endswith(".md"):
            return True
    return False

excluded = ["utils", "simple_GNN", "simple_NN", "simple_svm", "template_model", "tmp", "TwiBot22-split", "twibot22_Botrgcn_feature", "reduce_feature"]

no_mds = []

for dataset in datasets:
    if dataset in excluded:
        continue
    dataset_path = base / dataset
    files = os.listdir(dataset_path)
    if not has_markdown_file(files):
        no_mds.append(dataset)
        
print(no_mds)