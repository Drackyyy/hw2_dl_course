'''
Author: your name
Date: 2021-04-14 22:38:01
LastEditTime: 2021-04-22 15:12:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Homework2_yangxiaocong/dl_2021_hw2/dl_2021_hw2/hw2_start_code/src/models.py
'''
from torchvision import models
import torchvision
import torch.nn as nn
import resnet
from setup_seed import setup_seed
import torch
from adjusted_densenet import adjusted_densenet121
from sklearn.svm import SVC

def model_A(num_classes):
    model_resnet = resnet.resnet50(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


def model_B(num_classes):
    return adjusted_densenet121(num_classes = 10)
   


def model_C(num_classes):
    svm_head = SVC(C = 1, kernel = 'linear',class_weight={0:0.0001,1:0.001,2:0.001,3:1,4:1,5:1,6:1,7:0.01,8:1,9:1})
    return adjusted_densenet121(num_classes = 10), svm_head
    
    
    
    