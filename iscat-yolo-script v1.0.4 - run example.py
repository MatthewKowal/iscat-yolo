# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:46:14 2023


!!!!!!!         RUN THIS SCRIPT IN THE "yolo" ENVIRONMENT        !!!!!!!!


This script is the "front end" for the iscat-yolo methods. 

The purpose is the characterize an iscat movie and produce quantitative results
about the particles in the system.

A Chronological Order of the Methods
    Import .bin file
    
    Perform Ratiometric Particle Finding
        For Each Frame
            Generate Ratiometric Image
            Generate a "frame particle list" by finding Particles in the Ratiometric image using YOLO
            


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


import iscat_yolo_v1_0_4 as iscat

''' NOTE!! THIS SHOULD BE RUN IN THE yolo conda environment.
    to start, open a conda terminal and type "conda activate yolo"  '''
   
    
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

#%%
###############################################################################
#               PERFORM ANALYSIS ON RAW .MP4 VIDEO
###############################################################################

''' get the location of the current script'''
script_path      = os.path.abspath(__file__)       # Use __file__ to get the path of the currently running script
script_directory = os.path.dirname(script_path)    # Use os.path.dirname to get the directory containing the script
print("Directory of the currently running script:", script_directory)
basepath = os.path.join(script_directory, "example")


# quit early if we are not doing testing
#TESTING_SCENARIO = False
#import sys
#if TESTING_SCENARIO == False: sys.exit()

#### TESTING CONDITIONS ###
#binfile = r'D:/PS - Kinetics 1/50nm PS npn sq - weird frwquencies/0.1 nM PS 50 data, probably mislabeled/2023-05-25-50nm PS 0.1nM ITO pH7 Laser150mW - sq/VIDEOS/2023-05-26_16-04-32_raw_12_256_200.bin'
binfile = '' #input location of .bin file to process
mp4file = os.path.join(basepath, r'2023-05-26_16-04-32_raw_12_256_200.mp4')

print(mp4file)
                       
#%%
# ''''IMPORT AND PROCESS BINARY FILE'''
# running_for_first_time = True
# if running_for_first_time:
#     binimages = iscat.load_binfile_into_array(binfile)
#     basepath, filename, name, nframes, fov, x, y, fps = iscat.get_bin_metadata(binfile) # get basic info about binfile
#     # if not os.path.exists(os.path.join(basepath, "output")):
#     #     os.makedirs(os.path.join(basepath, "output"))

binimages = iscat.load_video(mp4file)
#%%
#basepath = os.path.split(mp4file)[0]
filename = os.path.split(mp4file)[1]
name = os.path.split(mp4file)[1] 
fps = 1111
fov = 1111
x, y, nframes = binimages.shape

name               = filename.split(".")[0]
metadata           = name.split("_")
print(metadata)

#date = metadata[0]
#time = metadata[1]
fov  = int(metadata[3])
#x    = int(metadata[4])
#y    = int(metadata[4])
fps  = int(metadata[5])


nm_per_px     = fov*1000/x
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
constants['output framerate']   = 50 #fps #200, 24
constants["nframes"]            = nframes
constants["timestamp"]          = name[:20]
constants["frame total"]        = nframes

#constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train31/weights/best.pt'
#constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train32/weights/best.pt'
#constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train37/weights/best.pt'
constants["yolo model loc"]     = os.path.join(script_directory, "yolo model", "train37_best.pt")
constants["min confidence"]         = 0.45 # good for contrast accuracy     # 0.3 #good compromise #0.05 # good for exploratory dialing back
#%%
if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"])

# ########################################################
# # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
# use_short_video = False
# if use_short_video:
#     start_frame = 5300#30000#350  #2200
#     nframes     = 2000 #12000  # 12000 frames is about a minute of video
#     binimages      = binimages[start_frame:(start_frame+nframes)]
#     constants["nframes"] = nframes
# else:
#     # # # OR USE THE FULL FILE
#     images = binimages
# ########################################################

#%%

'''RATIOMETRIC PARTICLE FINDER'''
r8, pl, results_list, noise = iscat.ratio_particle_finder(binimages, constants)
print("Memory Check:")
print("\n size of bin images:         ", sys.getsizeof(binimages)/1000000000, " gig")
print("\n size of ratiometric_video:  ", sys.getsizeof(r8)/1000000, " mb")
print("\n size of particle_list:      ", sys.getsizeof(pl)/1000000, " mb")
print("\n size of results_list:       ", sys.getsizeof(results_list)/1000000, " mb")




''' GENERATE OUTPUT '''
#keep only if longer than 10 frames
tag_ = "yolo-c"+str(constants["min confidence"])+"-trim10" 
pl2 = []
new_id = 1
for p in pl:
    if len(p.f_vec) > 10:
        pl2.append(p)
        pl2[-1].pID = new_id
        new_id += 1
    
iscat.save_particle_list_video(r8, pl2, constants, tag_)
generate_output(pl2, constants, tag_)

#%%
iscat.save_si_video(r8, pl2, constants, "-si-", offset=0)

 











#%% #!!!
#        _____  _                __            __     _      
#       / ___/ (_)____   ____ _ / /___        / /_   (_)____ 
#       \__ \ / // __ \ / __ `// // _ \      / __ \ / // __ \
#      ___/ // // / / // /_/ // //  __/  _  / /_/ // // / / /
#     /____//_//_/ /_/ \__, //_/ \___/  (_)/_.___//_//_/ /_/ 
#                     /____/                                


''' PROCESS A SINGLE .BIN FILE '''
def generate_exp_data(binfile, sample_name):
    st = time.time()
    ###########################################################################
    
    ''''IMPORT AND PROCESS BINARY FILE'''
    running_for_first_time = True
    if running_for_first_time:
        binimages = iscat.load_binfile_into_array(binfile)
        basepath, filename, name, nframes, fov, x, y, fps = iscat.get_bin_metadata(binfile) # get basic info about binfile
        # if not os.path.exists(os.path.join(basepath, "output")):
        #     os.makedirs(os.path.join(basepath, "output"))


        
    nm_per_px     = fov*1000/x
    constants = {}
    constants['sample name']        = sample_name
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
    constants['output framerate']   = 50#fps #200, 24
    constants["nframes"]            = nframes
    constants["timestamp"]          = name[:20]

    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train31/weights/best.pt'
    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train32/weights/best.pt'
    constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train37/weights/best.pt'
    constants["min confidence"]         = 0.45 # good for contrast accuracy     # 0.3 #good compromise #0.05 # good for exploratory dialing back


    if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"]) # if path doesnt exist then create it now

   
   

    # ########################################################
    # # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
    use_short_video = False
    if use_short_video:
        start_frame = 800#30000#350  #2200
        nframes     = 1000 #12000  # 12000 frames is about a minute of video
        binimages      = binimages[start_frame:(start_frame+nframes)]
        constants["nframes"] = nframes
    # else:
    #     # # # OR USE THE FULL FILE
    #     images = binimages
    # ########################################################
       

    '''RATIOMETRIC PARTICLE FINDER'''
    r8, pl, results_list, noise = iscat.ratio_particle_finder(binimages, constants)
    iscat.generate_noisefloor_csv(noise, constants)
    # intitally: at this point (early version), one minute of "dense" particle video (i.e ~20 particles/frame). It takes about 6 minutes (~30it/s). It takes 22 minutes to process a 3 minute video
    # then: somehow this changed and exploded to 90 minutes for a 3 minute video. this happened after i added all the particle bookkeeping stuff, but when i record the time each process takes it seems that the only thing that gets slower over time is the yolo particle finder. so now im not sure what to think about this
    # finally: I now run this program in the conda yolo environment so that it runs on the gpu and its not blazingly (relative) fast (60it/s). A 5 minute video takes around 20 min
    print("Memory Check:")
    print("\n size of bin images:         ", sys.getsizeof(binimages)/1000000000, " gig")
    print("\n size of ratiometric_video:  ", sys.getsizeof(r8)/1000000, " mb")
    print("\n size of particle_list:      ", sys.getsizeof(pl)/1000000, " mb")
    print("\n size of results_list:       ", sys.getsizeof(results_list)/1000000, " mb")


      
    ''' GENERATE OUTPUT '''
    
    #tag_ = "yolo-c45"   
    #iscat.save_particle_list_video(r8, pl, constants, tag_)
    #generate_output(pl, constants, tag_)
    #save_all_particle_images(pl, constants, tag_)
    
    
    
    #confidence threshold 45% keep only if longer than 10 frames
    tag_ = "yolo-c45-trim10" 
    pl2 = []
    new_id = 1
    for p in pl:
        if len(p.f_vec) > 10:
            pl2.append(p)
            pl2[-1].pID = new_id
            new_id += 1
    iscat.save_particle_list_video(r8, pl2, constants, tag_)
    #iscat.save_bw_video(r8, constants, "ratiometric", print_frame_nums=False)
    generate_output(pl2, constants, tag_)
    #save_all_particle_images(pl2, constants, tag_)
    #iscat.save_particle_list_video(r8, pl, constants, tag_)
    #iscat.save_si_video(r8, pl2, constants, "-si-", offset=0)
    
    
    
    ###########################################################################
    et = time.time()
    print("\n", name)
    print("\n\nFINSIHED... Processed ", os.path.getsize(binfile), " bytes in ", ((et-st)/60), " minutes\n\n")
        
 




#%%



#         ____          __         __               
#        / __ ) ____ _ / /_ _____ / /_              
#       / __  |/ __ `// __// ___// __ \             
#      / /_/ // /_/ // /_ / /__ / / / /             
#     /_____/ \__,_/ \__/ \___//_/ /_/              
#         ____                                      
#        / __ \ _____ ____   _____ ___   _____ _____
#       / /_/ // ___// __ \ / ___// _ \ / ___// ___/
#      / ____// /   / /_/ // /__ /  __/(__  )(__  ) 
#     /_/    /_/    \____/ \___/ \___//____//____/  
#         ______        __     __                   
#        / ____/____   / /____/ /___   _____        
#       / /_   / __ \ / // __  // _ \ / ___/        
#      / __/  / /_/ // // /_/ //  __// /            
#     /_/     \____//_/ \__,_/ \___//_/        

''' PROCESS A WHOLE FOLDER OF FOLDERS OF .BINS '''
def batch_process_exp_data(root_dir):
    
    binfiles = []
    folders = []
    sample_names = []

    for f in os.listdir(root_dir):
        
        sample_name = f[11:]
        folder = os.path.join(root_dir, f, 'VIDEOS')
        for filename in os.listdir(folder):
            if filename.endswith('.bin'):
                binfile = os.path.join(folder, filename)
                
                binfiles.append(binfile)
                folders.append(folder)
                sample_names.append(sample_name)
                
                print(binfile, "\n", sample_name, "\n\n")
                
                generate_exp_data(binfile, sample_name)
                
    return binfiles, folders, sample_names







'''#######################################################################
  _________     _______  ______   ______ ____  _      _____  ______ _____  
 |__   __\ \   / /  __ \|  ____| |  ____/ __ \| |    |  __ \|  ____|  __ \ 
    | |   \ \_/ /| |__) | |__    | |__ | |  | | |    | |  | | |__  | |__) |
    | |    \   / |  ___/|  __|   |  __|| |  | | |    | |  | |  __| |  _  / 
    | |     | |  | |    | |____  | |   | |__| | |____| |__| | |____| | \ \ 
  _ |_|     |_|  |_| __ |______| |_|  _ \____/|______|_____/|______|_|  \_\
 | \ | |   /\   |  \/  |  ____| | |  | |  ____|  __ \|  ____|              
 |  \| |  /  \  | \  / | |__    | |__| | |__  | |__) | |__                 
 | . ` | / /\ \ | |\/| |  __|   |  __  |  __| |  _  /|  __|                
 | |\  |/ ____ \| |  | | |____  | |  | | |____| | \ \| |____               
 |_| \_/_/    \_\_|  |_|______| |_|  |_|______|_|  \_\______|     
 
 '''


root_dir = '' # directory containing all iscat videos
strt = time.time()

a, b, c = batch_process_exp_data(root_dir)

endt = time.time()
print("Total Runtime: ", ((endt-strt)/60), " minutes")





