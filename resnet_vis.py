'''
Author: your name
Date: 2021-04-15 23:20:58
LastEditTime: 2021-04-16 20:25:14
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Homework2_yangxiaocong/dl_2021_hw2/dl_2021_hw2/hw2_start_code/src/resnet_vis.py
'''
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import TSNE
import data
from PIL import Image


target_dir='.resnet_vis'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)


def draw_features(width,height,x,savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = torch.load('best_model.pt')
        self.model = model_ft

    def forward(self, x, mode, savepath = None):
        if mode == 'cnn_vis':
            
            x = self.model.conv1(x)
            draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))

            x = self.model.bn1(x)
            draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))

            x = self.model.relu(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))

            x = self.model.maxpool(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))

            x = self.model.layer1(x)
            draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))

            x = self.model.layer2(x)
            draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))

            x = self.model.layer3(x)
            draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))

            x = self.model.layer4(x)
            draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))

            x = self.model.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

            # x = x.view(x.size(0), -1)
            # x = self.model.fc(x)
            # plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
            # plt.savefig("{}/f10_fc.png".format(savepath))
            # plt.clf()
            # plt.close()
            
        elif mode == 'fc_feature':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)

        return x
    
    
def get_cnn_layer(input_size):
    model_ft = ft_net()
    model_ft.eval()
    root_dir = "../../../../hw2_dataset/hw2_dataset/test"
    img_category = os.listdir(root_dir)[:5]
    for cate in img_category:
        img_path = os.path.join(os.path.join(root_dir,cate),os.listdir(os.path.join(root_dir,cate))[0]) # take one image per category for first five categories
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
        )

        img = transform(img).cuda()
        img = img.unsqueeze(0)
        with torch.no_grad():
            start = time.time()
            if not os.path.exists(os.path.join(target_dir,cate)):os.mkdir(os.path.join(target_dir,cate))
            model_ft(img, mode='cnn_vis',savepath = os.path.join(target_dir,cate))
            # print("total time:{}".format(time.time()-start))
            # result=out.cpu().numpy()
            # # ind=np.argmax(out.cpu().numpy()
            # ind=np.argsort(result,axis=1)
            # for i in range(5):
            #     print("predict:top {} = cls {} : score {}".format(i+1,ind[0,1000-i-1],result[0,1000-i-1]))
            # print("done")
