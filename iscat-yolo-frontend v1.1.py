# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:46:14 2023


!!!!!!!         RUN THIS SCRIPT IN THE "yolo" ENVIRONMENT        !!!!!!!!


This script is the "front end" for the iscat-yolo methods. 

The purpose is the characterize an iscat movie and produce quantitative results
about the particles in the system.



pip install ultralytics

@author: Matt
"""

#add the current folder to the system directory to look for modules
import sys
import os
import time
import numpy as np

import torch
if torch.cuda.is_available():
    print("Is cuda available? ", torch.cuda.is_available())
    print("Torch Cuda Version", torch.version.cuda)
    print("Torch Cude Device Count", torch.cuda.device_count())
    print("Torch Cuda Current device", torch.cuda.current_device())
    print("torch cuda device", torch.cuda.device(0))
    print("Device Name: ", torch.cuda.get_device_name(0))
else: print("Currently running Torch in CPU mode (no GPU acceleration)")

''' initialize current working directory and load iscat-yolo backend '''
script_dir, script_filename = os.path.split(__file__)  # get folder and filename of this script
sys.path.insert(0, script_dir)               # add the current folder to the system path
import iscat_yolo_backend_v1_1 as iscat
iscat.loadscreen()

  
''' PLEASE SPECIFY A .BIN FILE OR AN .MP4 VIDEO FILE '''
'''LOAD_FROM_BINFILE:'''
binfile = os.path.join(script_dir, "example", r'2022-06-30_14-56-58_raw_12_256_68.bin')
binimages = iscat.load_binfile_into_array(binfile)
basepath, filename, name, nframes, fov, x, y, fps, voltfile = iscat.get_bin_metadata(binfile)
# ''' LOAD_FROM_MP4VIDEO:'''
# mp4file = os.path.join(script_dir, "example", r'2023-05-26_16-04-32_raw_12_256_200.mp4')
# binimages = iscat.load_video(mp4file)
# basepath, filename, name, nframes, fov, x, y, fps, voltfile = iscat.get_bin_metadata(mp4file)

    

''' Constants and Experimental Parameters ''' 
constants = {}
constants['sample name']        = "EPD-iSCAT example" #sample_name
constants['bufsize']            = 20                   # 1/2 the ratiometric buffer size
constants['clipmin']            = 0.97                 #0.95,  # black point clipping of ratiometric image
constants['clipmax']            = 1.03                 #1.05,  # white point clipping of ratiometric image
constants['bbox']               = 15         # 1/2 the width of a saved particle image. this also defines where to cut off edge particles
constants['MIN_DIST']           = 10         # minimum distance for new overlapping particles to be considered the same      
constants['temporal tolerance'] = 10         # maximum number of skipped frames for particle to be considered the same      
constants['spatial tolerance']  = 10         # maximum distance from particle in a previous frame to be considered the same
constants['history tolerance']  = 100        # maximum number of frames to look back before giving up the search for a matching particle
constants["video x dim"]        = x
constants["video y dim"]        = y
constants["basepath"]           = basepath
constants["output path"]        = os.path.join(basepath, (name[:19] + " yolo_out " + iscat.getVersion()))
constants["name"]               = name
constants["fps"]                = fps
constants["fov"]                = fov
constants["nm per pixel"]       = fov*1000/x
constants['output framerate']   = 50 #fps #200, 24
constants["nframes"]            = nframes
constants["timestamp"]          = name[:20]
constants["yolo model loc"]     = os.path.join(script_dir, "yolo model", "train37_best.pt")
constants["min confidence"]     = 0.45 # good for contrast accuracy     # 0.3 #good compromise #0.05 # good for exploratory dialing back
constants["voltfile"]           = voltfile
if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"])




''' MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES '''
use_short_video = False
if use_short_video:
    start_frame = 0#30000#350  #2200
    nframes     = 200 #12000  # 12000 frames is about a minute of video
    binimages   = binimages[start_frame:(start_frame+nframes)]
    constants["nframes"] = nframes



'''RATIOMETRIC PARTICLE FINDER'''
#r8, noise = iscat.ratio_only(binimages, constants) #''' Ratiometric video only (no particle finding '''
r8, pl, results_list, noise = iscat.ratio_particle_finder(binimages, constants)
print("Memory Check:")
print("\tsize of bin images:         ", sys.getsizeof(binimages)/1000000000, " gig")
print("\tsize of ratiometric_video:  ", sys.getsizeof(r8)/1000000, " mb")
print("\tsize of particle_list:      ", sys.getsizeof(pl)/1000000, " mb")
print("\tsize of results_list:       ", sys.getsizeof(results_list)/1000000, " mb")



''' POSTPROCESSING: Remove very short lived particles '''
#keep only if longer than 10 frames
tag_ = "yolo-c"+str(constants["min confidence"])+"-trim10" 
pl2 = []
new_id = 1
for p in pl:
    if len(p.f_vec) > 10:
        pl2.append(p)
        pl2[-1].pID = new_id
        new_id += 1
pl = pl2



''' GENERATE OUTPUT '''
tag = tag_
# save data
iscat.save_pickle_data(pl, constants, tag)
iscat.save_constants(constants)

#generate spreadsheets
iscat.generate_sdcontrast_csv(pl, constants, tag)
iscat.generate_particle_list_csv(pl, constants, tag)
iscat.generate_landing_rate_csv(pl, constants, tag)

#generate data plot image
iscat.plot_landing_map(pl, constants, tag)
iscat.plot_landing_rate(pl, constants, tag)
iscat.plot_sdcontrast_hist(pl, constants, tag)
iscat.plot_waterfall(pl, constants, tag)

#generate videos
iscat.save_bw_video(r8, constants, "_archive", print_frame_nums = False)
iscat.save_particle_list_video(r8, pl, constants, tag_)
iscat.save_si_video2(r8, pl, constants, "-si-", offset=0)









