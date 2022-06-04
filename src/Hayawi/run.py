import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='name of the dataset used for training')
args = parser.parse_args()
print('running DeeBotPro on '+args.dataset)

order="python "+args.dataset+".py"
os.system(order)
print('process_1 finished')
order="python des_embedding.py "+ args.dataset
os.system(order)
print('process_2 finished')
print('training')
order="python train.py "+ args.dataset
os.system(order)
print('finished')
