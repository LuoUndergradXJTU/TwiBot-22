import torch
import logging
import csv
from tensorboardX import SummaryWriter
from pathlib import Path


class logger:
    def __init__(self, log_dir, log_filename):
        self.log_dir = log_dir
        self.log_filename = log_filename
        
    def record(self, metric_dict):
        