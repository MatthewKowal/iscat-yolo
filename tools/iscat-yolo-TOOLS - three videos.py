# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:49:18 2023

@author: Matt
"""


#add the current folder to the system directory to look for modules
import sys
import os
import time
import numpy as np


folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path


import iscat_yolo_v1_0_3 as iscat

import torch
print("Is cuda available? ", torch.cuda.is_available())
print("Torch Cuda Version", torch.version.cuda)
print("Torch Cude Device Count", torch.cuda.device_count())
print("Torch Cuda Current device", torch.cuda.current_device())
print("torch cuda device", torch.cuda.device(0))
print("Device Name: ", torch.cuda.get_device_name(0))

def generate_output(pl, constants, tag):
    #data
    iscat.save_pickle_data(pl, constants, tag)
    iscat.save_constants(constants)
    
    #spreadsheet
    iscat.generate_sdcontrast_csv(pl, constants, tag)
    iscat.generate_particle_list_csv(pl, constants, tag)
    iscat.generate_landing_rate_csv(pl, constants, tag)
    
    #image
    iscat.plot_landing_map(pl, constants, tag)
    iscat.plot_landing_rate(pl, constants, tag)
    iscat.plot_sdcontrast_hist(pl, constants, tag)
    iscat.plot_waterfall(pl, constants, tag)


binfile = r"C:/Users/Matt/Desktop/test data/2023-03-16-50nm PS 0.5 nM ITO pH7 Laser 150mW/VIDEOS/2023-03-16_17-38-47_raw_12_256_200.bin"




#%%

v1 = iscat.load_video("C:/Users/Matt/Desktop/test data/2023-03-16-50nm PS 0.5 nM ITO pH7 Laser 150mW/VIDEOS/2023-03-16_17-38-47 yolo_out v1_0_3/2023-03-16_17-38-47_raw_12_256_200raw.mp4")
v2 = iscat.load_video("C:/Users/Matt/Desktop/test data/2023-03-16-50nm PS 0.5 nM ITO pH7 Laser 150mW/VIDEOS/2023-03-16_17-38-47 yolo_out v1_0_3/2023-03-16_17-38-47_raw_12_256_200ratiometric.mp4")
#%%
v3 = iscat.load_video("C:/Users/Matt/Desktop/test data/2023-03-16-50nm PS 0.5 nM ITO pH7 Laser 150mW/VIDEOS/2023-03-16_17-38-47 yolo_out v1_0_3/2023-03-16_17-38-47_raw_12_256_200yolo-color.avi")


#%%

v1 = v1[:5900]
v2 = v2[:5900]
v3 = v3[:5900]
#%%
import PIL


n, x, y = v1.shape

v_out = np.zeros((n, y, 3*x), dtype=np.uint8)


for i in range(n):
    
    new_frame = PIL.Image.fromarray(v_out[i])
    
    i1 = PIL.Image.fromarray(v1[i])
    new_frame.paste(i1, (0,0))
    
    i2 = PIL.Image.fromarray(v2[i])
    new_frame.paste(i2, (255,0))
    
    i3 = PIL.Image.fromarray(v3[i])
    new_frame.paste(i3, (511,0))
    
    
    
    
    draw = PIL.ImageDraw.Draw(new_frame)
    #font = PIL.ImageFont.truetype("arial.ttf", 32)
    #draw.text( (2,2), "50 mW", 255, font=font)
    #draw.text( (257,2), "150 mW", 255, font=font)

    
    
    v_out[i] = np.array(new_frame)
    

constants = {}
constants["output path"] = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS"
constants["output framerate"] = 50
constants["name"] = "three video test"
iscat.save_bw_video(v_out, constants, "test", print_frame_nums = False)