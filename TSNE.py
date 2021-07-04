'''
Author: your name
Date: 2021-04-11 22:30:12
LastEditTime: 2021-04-16 20:21:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Homework2_yangxiaocong/dl_2021_hw2/dl_2021_hw2/hw2_start_code/src/TSNE.py
'''
from sklearn import manifold
import numpy as np
import resnet_vis
import torch
import matplotlib.pyplot as plt

def t_sne_plot(X, y):
    tsne = manifold.TSNE(n_components=2,init='pca')
    X_tsne = tsne.fit_transform(X.cpu())
    ## data normalization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    y = y.numpy()
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig('tSNE_figure')
    # plt.show()

def tsne (max_step_tsne,loader):
    fc_features = None
    fc_labels = None            
    # get fc features 
    for step, (inputs, labels) in enumerate(loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        model_ft = resnet_vis.ft_net()
        model_ft.eval()
        with torch.no_grad():
            fc_feature = model_ft(inputs,'fc_feature')
            if fc_features != None:
                fc_features = torch.cat((fc_features,fc_feature))  # each row represents a sample
            else :
                fc_features = fc_feature
            
            if fc_labels != None:
                fc_labels = torch.cat((fc_labels,labels.cpu()))  # each row represents a sample
            else :
                fc_labels = labels.cpu()
            
            if step >= max_step_tsne:
                break
    
    t_sne_plot(fc_features,fc_labels)
