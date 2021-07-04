'''
Author: Yang Xiaocong
Date: 2021-04-07 18:27:18
LastEditTime: 2021-06-07 09:51:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Homework2_yangxiaocong\dl_2021_hw2\dl_2021_hw2\hw2_start_code\src\main.py
'''
import sys
sys.path.append('.')
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
import pylab
from setup_seed import setup_seed
import TSNE
import resnet_vis
from confusion_matrix import plot_confusion
import argparse
import dill
from torch.optim import lr_scheduler
from smoothloss import LabelSmoothLoss
from sklearn.metrics import confusion_matrix, accuracy_score


def train_model(task, model,train_loader, valid_loader, criterion, optimizer, max_num_epochs, scheduler=None):

    def train(model, train_loader,optimizer,criterion,scheduler):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            # inputs = inputs.cuda()
            inputs = inputs.cuda().float()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        ## note scheduler is renewed per epoch instead of per iteration as optimizer
        try:
            scheduler.step()
        except:
            pass
        
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        total_labels = None
        total_predictions = None
        for inputs, labels in valid_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

                
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()


    best_acc = 0.0
    training_accuracy = []
    training_loss = []
    test_loss = []
    test_accuracy = []
    
    for epoch in range(max_num_epochs):
        # setup_seed(epoch)       ## don't use in task_B
        print('epoch:{:d}/{:d}'.format(epoch, max_num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion,scheduler)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        
        training_loss.append(train_loss)
        test_loss.append(valid_loss)
        training_accuracy.append(train_acc)
        test_accuracy.append(valid_acc)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            # try: 
            #     torch.save(best_model, f'best_{task}.pt')
            # except:
            #     pass

    ## print loss curves with matplotlib
    fig = plt.figure(figsize = (6,8))
    ax = fig.add_subplot(211)
    ax.plot(training_loss)
    ax.plot(test_loss)
    ax.set_title('loss curves',fontsize=16)
    ax.set(xlabel='epoch',ylabel='loss')
    ax.legend(['training loss','test loss'],loc='best')
    ay = fig.add_subplot(212)
    ay.plot(training_accuracy)
    ay.plot(test_accuracy)
    ay.set_title('accuracy curves',fontsize=16)
    ay.set(xlabel='epoch',ylabel='accuracy',ylim=[0,1])
    ay.legend(['training_accuracy','test accuracy'],loc='best')
    fig.subplots_adjust(top=1.5)
    if not os.path.exists('output'): os.mkdir('output')
    plt.savefig(f'{task}_curves', bbox_inches='tight')

        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## about model
    num_classes = 10

    ## about data
    data_dir = "../../../../hw2_dataset/hw2_dataset/" 
    input_size = 224
    batch_size = 32
    
    ## about training
    max_num_epochs = 120
    lr = 0.001
    
    ## about convolutional layers visualization
    max_step_tsne = 15     ## number of sample in tsne = max_step_tsne * batch_size
    
    ## model initialization for a specific task
    parser = argparse.ArgumentParser(description='choose a task from task_A, task_B and task_C')
    parser.add_argument('--task', type=str, help = 'task name')
    arg = parser.parse_args()
    
    if arg.task == 'task_A':
        model = torch.jit.script(models.model_A(num_classes=num_classes).cuda())
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
    
    elif arg.task == 'task_B':
        model = models.model_B(num_classes=num_classes).cuda()
        optimizer = optim.Adam(model.parameters(), lr= lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor= 0.7, patience = 3)
        criterion = LabelSmoothLoss(smoothing=0.1)

    elif arg.task == 'task_C':
        model , head = models.model_C(num_classes=num_classes)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr= lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor= 0.7, patience = 3)
        criterion = LabelSmoothLoss(smoothing=0.1)
    
    ## data preparation for a specific task
    if arg.task == 'task_C':
        train_loader, valid_loader, feature_loader = data.load_data(task = arg.task, batch_size = batch_size)
    else:
        train_loader, valid_loader = data.load_data(task = arg.task, batch_size = batch_size)
    
    ## Parallel computing (could cause conflict with argpaser since when using parallel computing in terminal,
    ## '-m torch.distributed.launch' must be added but cannot be identified by self-defined argpaser)
    
    # if torch.cuda.device_count() > 1:
    #     torch.distributed.init_process_group(backend='nccl')
    #     model = model.cuda()
    #     model = torch.nn.parallel.DistributedDataParallel(model)
    # elif torch.cuda.device_count() == 1:
    #     model = model.cuda()
    
    ## train model
    train_model(arg.task, model, train_loader, valid_loader, criterion, optimizer, max_num_epochs, scheduler=None)
    
    ## downstream tasks
    if arg.task == 'task_A':
        TSNE.tsne(max_step_tsne = max_step_tsne,loader=train_loader)
        resnet_vis.get_cnn_layer(input_size=input_size)


    if arg.task == 'task_C':
        model = torch.load('best_task_C.pt')
        model.train(False)
    
        ## extracting features of training set. Note here we use feature_loader to input data in training set but with transform in 
        ## test set to ensure same distribution for SVM training and test sets
        
        X_train = None
        y_train = None
        
        for inputs, labels in feature_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            if y_train != None:
                y_train = torch.cat((y_train, labels.detach().cpu()),dim=0)
            else:
                y_train = labels.detach().cpu()
            
            if X_train != None:
                X_train = torch.cat((X_train , outputs.detach().cpu()),dim=0)
            else:
                X_train = outputs.detach().cpu()
        
        X_train = X_train.numpy()
        y_train = y_train.numpy()
        
        ## extracting feature of test set
        X_test = None
        y_test = None
        
        for inputs, labels in valid_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            
            if y_test != None:
                y_test = torch.cat((y_test, labels.detach().cpu()),dim=0)
            else:
                y_test = labels.detach().cpu()
            
            if X_test != None:
                X_test = torch.cat((X_test , outputs.detach().cpu()),dim=0)
            else:
                X_test = outputs.detach().cpu()
        
        X_test = X_test.numpy()
        y_test = y_test.numpy()
        
        ## training head in task_C
        head.fit(X_train, y_train)
        y_pred = head.predict(X_test)
        print(confusion_matrix(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        plot_confusion(y_test, y_pred)
        
