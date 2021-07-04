#!/bin/bash
###
 # @Author: your name
 # @Date: 2021-04-16 22:19:26
 # @LastEditTime: 2021-04-22 15:57:07
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /Homework2_yangxiaocong/dl_2021_hw2/dl_2021_hw2/hw2_start_code/src/homework2.sh
### 

export PATH=/extension/yangxiaocong/anaconda3/bin:$PATH
echo "Excecute task_A, task_B and task_C consecutively. Otherwise, use 'python main.py --task task_X' to excecute single task. "
python main.py --task task_A
python main.py --task task_B
python main.py --task task_C