'''
Author: your name
Date: 2021-04-14 20:05:39
LastEditTime: 2021-04-15 20:58:00
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Homework2_yangxiaocong/dl_2021_hw2/dl_2021_hw2/hw2_start_code/src/setup_seed.py
'''
import torch
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True