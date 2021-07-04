'''
Author: your name
Date: 2021-04-11 16:11:52
LastEditTime: 2021-04-22 15:12:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Homework2_yangxiaocong/dl_2021_hw2/dl_2021_hw2/hw2_start_code/src/data.py
'''
from albumentations import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from torch.utils.data import DistributedSampler
import cv2
import numpy as np
import torch.utils.data as Data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import DatasetFolder
from torch.utils.data.sampler import WeightedRandomSampler

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

## transform for task_B
def strong_aug():
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        Resize(224,224,p=1,always_apply=True),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], 
    p=1,
)



## dataset for task_B
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
class MyFolder(DatasetFolder):
    def __init__(self,root: str,extension:str = "png",transform = None,target_transform = None,loader = None,is_valid_file= None):
        super(MyFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


## sampler for task_C
def get_sampler(dataset):
    trainratio = np.bincount(dataset.targets)
    classcount = trainratio.tolist()
    train_weights = 1./torch.tensor(classcount, dtype=torch.float)
    train_sampleweights = train_weights[dataset.targets]
    sampler = WeightedRandomSampler(weights=train_sampleweights, 
                                 num_samples=len(train_sampleweights))
    return sampler



def load_data(task, data_dir = "../../../../hw2_dataset/hw2_dataset/",input_size = 224 ,batch_size = 32):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size = input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([transforms.RandomRotation(degrees= 30,expand=True), 
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomAffine(degrees=0),
                                     transforms.RandomVerticalFlip(),
            ]),
            transforms.ToTensor(),
            # transforms.RandomApply([transforms.RandomErasing()],p=0.25),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),   ## Color changing is harmful here !!!
            transforms.Resize(size = input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'strong':strong_aug(),
    }
    

    if task == 'task_A':
        image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, '1-Large-Scale', 'train'), data_transforms['train'])
    elif task == 'task_B':
        # image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, '2-Medium-Scale', 'train'), data_transforms['train'])
        image_dataset_train = MyFolder(os.path.join(data_dir, '2-Medium-Scale', 'train'), transform=data_transforms['strong'])
    elif task == 'task_C':
        # image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, '4-Long_Tailed', 'train'), data_transforms['train'])
        image_dataset_train = MyFolder(os.path.join(data_dir, '4-Long-Tailed', 'train'), transform=data_transforms['strong'])
        image_dataset_feature = datasets.ImageFolder(os.path.join(data_dir, '4-Long-Tailed', 'train'),transform=data_transforms['test'])

    image_dataset_valid = datasets.ImageFolder(os.path.join(data_dir,'test'), data_transforms['test'])



    if task == 'task_C':
        train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=False, num_workers=4,sampler=get_sampler(image_dataset_train))
        feature_loader = DataLoader(image_dataset_feature, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)
    
    
    if task == 'task_C':
        return train_loader, valid_loader, feature_loader
    else:
        return train_loader, valid_loader
