# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:46:14 2023


This script generates images for a roboflow dataset

you can choose from sequential images (good for getting the whole landing)
of you can choose random images to sample evenly over a whole video
@author: Matt
"""

#add the current folder to the system directory to look for modules
import sys
import os
import time


folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path


import iscat_yolo_v2_0_1 as iscat




import numpy as np
from collections import deque
import tqdm
# def ratio_vid_only(images, constants):
#     #DEFINE CONSTANTS
#     bufsize             = constants['bufsize']
#     clipmin             = constants['clipmin']
#     clipmax             = constants['clipmax']
#     start               = time.time()
#     n,x,y               = images.shape    
#     i16                 = np.ndarray([x,y]).astype(np.float16)
#     i8                  = np.ndarray([x,y]).astype(np.uint8)
#     v8                  = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)       # this contains the 8 bit video
#     print("\n\nPerforming Ratiometric Image Process on ", n, " Frames\n")
    
#                # Step 1: autofill the deques
#     d1 = deque(images[:bufsize].astype(np.float16), bufsize)#[]
#     d2 = deque(images[bufsize:(2*bufsize)].astype(np.float16), bufsize)#[]
   
#     # Step 2, ratiometric on rest of images
#     for f in tqdm.tqdm(range(n-2*bufsize)):
#         # Create Ratiometric Frame
#         d1sum = np.sum(d1, axis=0).astype(np.float16)  #sum queue1                  
#         d2sum = np.sum(d2, axis=0).astype(np.float16)
#         #i16 = ( d2sum/np.sum(d2sum) ) / (d1sum/np.sum(d1sum))
#         i16 = d2sum / d1sum
        
#         # Manage Frame Queue
#         d1.append(d2.popleft())
#         d2.append(images[f+2*bufsize])
#         #save frame as an 8-bit image and append it to a video
#         i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
#         v8[f] = i8

#     end = time.time()
#     print(" ")
#     print("\t Resolution: \t\t\t\t\t", x, y, "px")
#     print("\t Total Pixels per frame: \t\t", (x*y), "px")
#     print("\t New number of frames: \t\t\t", n, "frames")
#     print("\t Elapsed time: \t\t\t\t\t", (end-start), " seconds")
#     print("\t Elapsed time: \t\t\t\t\t", ((end-start)/60), " minutes")
#     #print("\t Total Number of Particles: \t", pID, "particles")
#     print("\t Speed (n-fps): \t\t\t\t", (n / (end-start)), " finished frames / sec" )
#     return v8


import PIL
def export_n_frames(images, start, num, skipframes, constants):
    n,x,y               = images.shape 
    
    foldername = "n_frames " + constants["name"]
    if not os.path.exists(os.path.join(constants["output path"], foldername)):
        os.makedirs(os.path.join(constants["output path"], foldername))
    
    # save sequential frames, accounting for skipping frames. its a good idea to skip a few frames
    # when the frame rate is high (30+ fps)
    for c in range( start, start+(num*skipframes) ):
        
        if c%skipframes == 0:
            
            #print(c)
            savefilepath = os.path.join(constants["output path"], foldername, (str(c)+".png"))
            im = PIL.Image.fromarray(images[c])
            im.save(savefilepath)
    
import random
def export_rnd_frames(images, nframes, constants):
    n,x,y               = images.shape 
    #print("length of video: n)
    if nframes > n: nframes = n
        
    foldername = "rnd_frames " + constants["name"]
    if not os.path.exists(os.path.join(constants["output path"], foldername)):
        os.makedirs(os.path.join(constants["output path"], foldername))

    samples = random.sample(range(0,n), nframes)
    for s in samples:
        f = images[s]
        savefilepath = os.path.join(constants["output path"], foldername, (str(s)+".png"))
        im = PIL.Image.fromarray(f)
        im.save(savefilepath)
        #plt.close('all')
    


#25nm
#binfile = r"C:/Users/Matt/Desktop/test data/2023-03-16-25nm PS 1nM ITO pH7 Laser150mW/VIDEOS/2023-03-16_16-22-24_raw_12_256_200.bin"
#sample_name = "25nm test 1nM" 

#50nm
#binfile = r"C:/Users/Matt/Desktop/test data/2023-03-16-50nm PS 0.5 nM ITO pH7 Laser 150mW/VIDEOS/2023-03-16_17-38-47_raw_12_256_200.bin"
# binfile = r"C:/Users/Matt/Desktop/PS ITO DATA wait 1 minute/50 nm/2023-03-23-50nm PS 0.1nM ITO pH7 Laser200mW/VIDEOS/2023-03-23_15-36-29_raw_12_256_200.bin"
# sample_name = "50nm test 0.5nM"
#binfile = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/KINETICS - LP/LP kinetics mdk87 - 1.03mW 75nm pr/2025-04-25_15-47-17_raw_35_256_200.bin"
#sample_name = "50nm ps on ITO 1 mW per 400 um sq"
#binfile = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/KINETICS - LP/LP kinetics mdk87 - 6mW 100nm pr/2025-04-24_18-18-08_raw_35_256_200.bin"
binfile = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/KINETICS - LP 0/LP kinetics mdk87 - 6mW 100nm pr/2025-04-24_18-23-52_raw_35_256_200.bin"
sample_name = "50nm ps on ITO 6 mW per 400 um sq"

#100nm
#binfile = r"C:/Users/Matt/Desktop/test data/2023-03-17-100nm PS 0.01 nM ITO pH7 Laser 150mW/VIDEOS/2023-03-17_18-06-25_raw_12_256_200.bin"
#sample_name = "100nm test 0.01nM"
#binfile = r"C:/Users/Matt/Desktop/PS ITO DATA wait 1 minute/100 nm/2023-03-23-100nm PS 0.01nM PDL pH7 Laser200mW/VIDEOS/2023-03-23_16-43-03_raw_12_256_200.bin"

#200nm
#binfile= r"C:/Users/Matt/Desktop/test data/2023-03-23-200nm PS 0.01nM PDL pH7 Laser200mW/VIDEOS/2023-03-23_16-58-34_raw_12_256_200.bin"
#sample_name = "200nm test 0.01nM"

# binfile=r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/2023-04-12-100nm PS 0.01 nM ITO pH7 Laser150mW/VIDEOS/2023-04-12_22-15-19_raw_12_256_200.bin"
# sample_name = "100nm false positive"

''''IMPORT AND PROCESS BINARY FILE'''
running_for_first_time = True
if running_for_first_time:
    binimages = iscat.load_binfile_into_array(binfile)
    basepath, filename, name, nframes, fov, x, y, fps = iscat.get_bin_metadata(binfile) # get basic info about binfile
    # if not os.path.exists(os.path.join(basepath, "output")):
    #     os.makedirs(os.path.join(basepath, "output"))

nm_per_px     = fov*1000/x
constants = {}
#constants['sample name']        = sample_name
constants['bufsize']            = 30                   # 1/2 the ratiometric buffer size
constants['clipmin']            = 0.97                 #0.95,  # black point clipping of ratiometric image
constants['clipmax']            = 1.03                 #1.05,  # white point clipping of ratiometric image
# constants['bbox']               = 15         # 1/2 the width of a saved particle image. this also defines where to cut off edge particles
# constants['MIN_DIST']           = 10         # minimum distance for new overlapping particles to be considered the same      
# constants['temporal tolerance'] = 10         # maximum number of skipped frames for particle to be considered the same      
# constants['spatial tolerance']  = 10         # maximum distance from particle in a previous frame to be considered the same
# constants['history tolerance']  = 100        # maximum number of frames to look back before giving up the search for a matching particle
constants["video x dim"]        = x
constants["video y dim"]        = y
constants["basepath"]           = basepath
dataset_metastring = f"_dataset_images buf{constants['bufsize']}_dr{((constants['clipmax'] - constants['clipmin'])/2):.2f}"
constants["output path"]        = os.path.join(basepath, (name[:19] + dataset_metastring))
constants["name"]               = name
constants["fps"]                = fps
constants["fov"]                = fov
constants['output framerate']   = 50#fps #200, 24
constants["nframes"]            = nframes

if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"])




''' GENERATE RATIOMETRIC VIDEO (no particle finding)'''
r8 = iscat.ratio_only(binimages, constants)


#iscat.save_bw_video(r8, constants, "ratio-clean", print_frame_nums = True)      # Save the video with frame numbers
#                                                                                # so that we can pinpoint the frame
#                                                                                # sequence of rare particle landings

''' EXPORT FRAMES FOR DATASET'''
#export_n_frames(r8, 4640, 50, 2, constants) # export sequential images
export_rnd_frames(r8, 200, constants)       # export random images
