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




from ultralytics import YOLO

'''
                IMPORTANT NOTE 
 For now this code hangs before running the first epoch when running on the gpu. 
for that reason, this code should be run in the yolov8_cpu environment

Alternatively, it seems if you set workers=0 it will run on the gpu

'''


#dataset_path = 'C:/Users/Matt/Desktop/Github/iscat-yolo/pdatasetv5/data.yaml' # this was used to train model 31 which worked well for 50nm and ok for 25nm and 100nm
#dataset_path = 'C:/Users/Matt/Desktop/Github/iscat-yolo/pdatasetv7/data.yaml'  # train model 32
#dataset_path = 'C:/Users/Matt/Desktop/Github/iscat-yolo.beta-testing/pdatasetv8/data.yaml'  # train model 37
#dataset_path = r"C:/Users/Matt/Desktop/EPD-iSCAT PYTHON SCRIPTS/iscat particle datasets/iSCAT Particle Image Library.v9i.yolov8/data.yaml"
dataset_path = r"C:/Users/Matt/Desktop/EPD-iSCAT PYTHON SCRIPTS/iscat particle datasets/iSCAT Particle Image Library.v10i.yolov8/data.yaml"
dataset_path = r"C:\Users\Matt\Desktop\EPD-iSCAT PYTHON SCRIPTS\iscat particle datasets\Deeptrack Synth Dataset v1/data.yaml" #deeptrack synthesis data
dataset_path = r"C:/Users/Matt/Desktop/iSCAT DEEPTRACK IMAGE SYNTHESIS/hybrid dataset v1/data.yaml"
#os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
model=YOLO('yolov8n.pt')

#quick and dirty model training
#model.train(data=dataset_path, imgsz=256, epochs=10, batch=64, workers=0)

#robust model training
# model.train(
#     data=dataset_path,
#     imgsz=256,
#     epochs=100,
#     batch=32,
#     workers=0,
#     patience=10,
#     optimizer='SGD',
#     lr0=0.01,
#     augment=True,
#     verbose=True
# )

#robust model training with image augmentation
model.train(
    data=dataset_path,
    imgsz=256,
    epochs=100,
    batch=64,
    workers=0,          # must use zero workers or else it hangs when training with GPU
    patience=10,        # early stop if no improvement after 10 epochs
    optimizer='SGD',    # or 'Adam' (faster convergence but slightly riskier overfitting)
    lr0=0.01,           # starting learning rate (default good for SGD)
    warmup_epochs=3,    # warmup to avoid instability early on
    degrees=10,         # random rotation up to ±10 degrees
    translate=0.1,      # random shift 10% of width/height
    scale=0.5,          # random scale 50–150%
    shear=2.0,          # slight shearing
    perspective=0.0005, # very small perspective distortion
    flipud=0.2,         # 20% chance of vertical flip
    fliplr=0.5,         # 50% chance of horizontal flip
    hsv_h=0.01,         # hue augmentation
    hsv_s=0.5,          # saturation augmentation
    hsv_v=0.5           # brightness augmentation
)



#The models saves here:    C:\Users\Matt\runs\detect


# Monitor runs by opening a conda terminal and type the following:
#    tensorboard --logdir C:\Users\Matt\runs\

# Then, in a browser go to:       http://localhost:6006/

# completed runs can be found here: C:\Users\Matt\runs\detect