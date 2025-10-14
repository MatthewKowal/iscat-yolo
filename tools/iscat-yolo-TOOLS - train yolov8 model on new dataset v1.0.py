# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:55:58 2023

RUN THIS SCRIPT IN THE yolo_v8_cpu ENVIRONMENT

@author: Matt
"""
import os
from ultralytics import YOLO

import torch

print("Is cuda available? ", torch.cuda.is_available())
print("Torch Cuda Version", torch.version.cuda)

print("Torch Cude Device Count", torch.cuda.device_count())


print("Torch Cuda Current device", torch.cuda.current_device())


print("torch cuda device", torch.cuda.device(0))


print("Device Name: ", torch.cuda.get_device_name(0))

''' its ok if these fail since we are running this in cpu mode '''


#%%
from ultralytics import YOLO

'''
                IMPORTANT NOTE 
 For now this code hangs before running the first epoch when running on the gpu. 
for that reason, this code should be run in the yolov8_cpu environment

'''


#dataset_path = 'C:/Users/Matt/Desktop/Github/iscat-yolo/pdatasetv5/data.yaml' # this was used to train model 31 which worked well for 50nm and ok for 25nm and 100nm
#dataset_path = 'C:/Users/Matt/Desktop/Github/iscat-yolo/pdatasetv7/data.yaml'  # train model 32
dataset_path = 'C:/Users/Matt/Desktop/Github/iscat-yolo.beta-testing/pdatasetv8/data.yaml'  # train model 37


#os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
model=YOLO('yolov8n.pt')

model.train(data=dataset_path, imgsz=256, epochs=10, batch=64)


# Monitor runs by opening a conda terminal and type the following:
#    tensorboard --logdir C:\Users\Matt\runs\

# Then, in a browser go to:       http://localhost:6006/

# completed runs can be found here: C:\Users\Matt\runs\detect