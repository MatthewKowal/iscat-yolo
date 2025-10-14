# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:46:14 2023



    
@author: Matt
"""

import ffmpeg
import os



def find_files_with_string(folder_path, search_string):
    matching_files = []

    # Use os.walk to traverse all directories and subdirectories
    for root, _, files in os.walk(folder_path):
        for filename in files:
            # Check if the search string is in the filename
            if search_string in filename:
                # Append the full path to the matching file
                matching_files.append(os.path.join(root, filename))
    return matching_files

def compress_video(vid_path, file_out, compression):
    #compression could be 10 or 28, maybe

    ffmpeg.input(vid_path).output(
        file_out,
        **{'vcodec': 'libx264', 'crf': str(compression)}  # 'crf' controls quality (lower is better, but larger file)
    ).run()
    
    print("Compression complete!")




# Example usage
#folder_path = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\MDK 55 good"
#folder_path = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\MDK 55 good\2024-11-05-MDK 25 B 200 nm PS 0.01nM 1v1v\VIDEOS\2024-11-05_16-26-43 yolo_out v2_0_1.buf50"
#folder_path = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\mdk 57 B\2024-11-13-mdk 57 B - 50nm psnp 0.05 nM\VIDEOS\2024-11-13_16-47-10 yolo_out v2_0_1.buf50"
folder_path = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\mdk 57 B\2024-11-13-mdk 57 B - 50nm psnp 0.05 nM\VIDEOS\2024-11-13_16-47-10 yolo_out v2_0_1.buf100"
#search_string = '_rawbin.mp4'
#search_string = '_ratio100.mp4'
search_string = "_ - 0 yolo raw output-color.avi"
matching_files = find_files_with_string(folder_path, search_string)



savepath = folder_path
compression = 20

# compres   new/old     quality
#  -sion   file size
#   20       25%          
#   25       10%          ok

print(f"Files containing the string: \n \t {search_string} \n")
for f in matching_files: print(f"{f} \n")






import tqdm


for vf in tqdm.tqdm(matching_files):
    #print(vf[-40:])
    fname = os.path.split(vf)[1].split(".")[0] + f"_compressed-{compression}.mp4"
    #print("\t", fname)
    file_out = os.path.join(savepath, fname)
    print(file_out)
    compress_video(vf, file_out, compression)
    
#%%


#ISCAT SCRIPT BELOW




#add the current folder to the system directory to look for modules
import sys
import os
import time
import numpy as np


folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path


import iscat_yolo_v2_0_1 as iscat

''' NOTE!! THIS SHOULD BE RUN IN THE yolo conda environment.
    to start, open a conda terminal and type "conda activate yolo"  '''
    

import torch
print("Is cuda available? ", torch.cuda.is_available())
print("Torch Cuda Version", torch.version.cuda)
print("Torch Cude Device Count", torch.cuda.device_count())
print("Torch Cuda Current device", torch.cuda.current_device())
print("torch cuda device", torch.cuda.device(0))
print("Device Name: ", torch.cuda.get_device_name(0))



# def generate_output(r8, pl, constants, tag):
#     #data
#     iscat.save_pickle_data(pl, constants, tag)             #save particle list and processing parameters as a pickle file. the pickle file can be opened later with a ratiometric video to reprocess the video in different ways
#     iscat.save_constants(constants)                        #save the processing parameters in a text file
    
#     #spreadsheet
#     iscat.generate_sdcontrast_csv(pl, constants, tag)      #save a list of stddev contrast values
#     iscat.generate_particle_list_csv(pl, constants, tag)   #save a particle list excel file
#     iscat.generate_landing_rate_csv(pl, constants, tag)    #save a count of landings vs time
    
#     #image
#     iscat.plot_landing_map(pl, constants, tag)             #save an image map of where particles landed
#     iscat.plot_landing_rate(pl, constants, tag)            #save an image of the landing rate 
#     iscat.plot_sdcontrast_hist(pl, constants, tag)         #save contrast histogram
#     iscat.plot_waterfall(pl, constants, tag)               #save particle landings over time as a function of mass
    
#     #video
#     iscat.save_bw_video(r8, constants, "ratiometric", print_frame_nums=False)





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

 
    nm_per_px     = 37.5#fov*1000/x
    constants = {}
    constants['sample name']        = "test" #sample_name
    constants['bufsize']            = 50#100#20                   # 1/2 the ratiometric buffer size
    constants['clipmin']            = 0.98#0.97                 #0.95,  # black point clipping of ratiometric image
    constants['clipmax']            = 1.02#1.03                 #1.05,  # white point clipping of ratiometric image

    constants['bbox']               = 15         # 1/2 the width of a saved particle image. this also defines where to cut off edge particles
    constants['MIN_DIST']           = 20#10         # minimum distance for new overlapping particles to be considered the same      
    constants['temporal tolerance'] = 500#10         # maximum number of skipped frames for particle to be considered the same      
    constants['spatial tolerance']  = 20#10         # maximum distance from particle in a previous frame to be considered the same
    constants['history tolerance']  = 500#100        # maximum number of frames to look back before giving up the search for a matching particle
    constants["video x dim"]        = x
    constants["video y dim"]        = y
    constants["basepath"]           = basepath
    constants["output path"]        = os.path.join(basepath, (name[:19] + " yolo_out " + iscat.getVersion() +"buf"+str(constants['bufsize'])))
    constants["name"]               = name
    constants["fps"]                = fps
    constants["fov"]                = fov
    constants['output framerate']   = 50#fps #200, 24
    constants["nframes"]            = nframes #total number of frames of the raw file
    constants["timestamp"]          = name[:19]
    constants["frame total"]        = nframes  #total number of frames of the cropped file
    
    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train31/weights/best.pt'
    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train32/weights/best.pt'
    constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train37/weights/best.pt'
    constants["min confidence"]         = 0.05 #0.25 #0.45 # good for contrast accuracy     # 0.3 #good compromise #0.05 # good for exploratory dialing back
 
    if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"])
    
    if constants["nframes"] < 2*constants["bufsize"]: return
 
    # ########################################################
    # # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
    start_frame=0
    use_short_video = False
    
    if use_short_video:
        start_frame = 6900#30000#350  #2200
        nframes     = 1000 #12000  # 12000 frames is about a minute of video
        binimages      = binimages[start_frame:(start_frame+nframes)]
        constants["nframes"] = nframes
    # else:
    #     # # # OR USE THE FULL FILE
    #     images = binimages
    # ########################################################
     
    ''' RAW BINFILE OUTPUT '''
    iscat.save_bw_video(binimages, constants, "rawbin", print_frame_nums=False)
    iscat.save_constants(constants)
    
    ratio_only = False
    if ratio_only:
        r8 = iscat.ratio_only(binimages, constants)
        iscat.save_bw_video(r8, constants, ("ratio"+str(constants["bufsize"])), print_frame_nums=False)
        iscat.save_si_video2(r8, [], constants, " - voltage video", offset=0, print_fnum=False, print_v=True)
    
    else:
        '''RATIOMETRIC PARTICLE FINDER'''
        tag = " - 0 yolo raw output"
        r8, pl, results_list, noise = iscat.ratio_particle_finder(binimages, constants)
        #STANDARD RATIOMETRIC OUTPUT
        iscat.save_bw_video(r8, constants, "ratio"+str(constants['bufsize']), print_frame_nums=True)
        iscat.generate_noisefloor_csv(noise, constants, "buf"+str(constants['bufsize']))
        #iscat.save_si_video2(r8, [], constants, " - voltage video", offset=0, print_fnum=False, print_v=True)
        
        
        #FIT BUT NO FILTER
        tag = " - 0 yolo raw output"
        pl_fit = iscat.fit_gaussian_to_particle_list(pl, constants)
        #VIDEO
        iscat.save_particle_list_video(r8, pl_fit, constants, tag)
        #CSVS
        iscat.generate_particle_list_csv(pl_fit, constants, tag)
        iscat.generate_landing_rate_csv(pl_fit, constants, tag)
        #PICS
        iscat.plot_contrast(pl_fit, constants, tag)
        iscat.plot_landing_map(pl_fit, constants, tag)
        iscat.plot_landing_map2(pl_fit, constants, tag)
        iscat.plot_landing_rate(pl_fit, constants, tag)
        
        iscat.save_pickle_data(pl_fit, constants, tag)
        #iscat.plot_sdcontrast_hist(pl_fit, constants, tag)
        
        
    
        #pimages
        #iscat.plot_pimages(pl_fit, constants)
        
        
        print("Memory Check:")
        print("\t size of bin images:         ", sys.getsizeof(binimages)/1000000000, " gig")
        print("\t size of ratiometric_video:  ", sys.getsizeof(r8)/1000000, " mb")
        print("\t size of particle_list:      ", sys.getsizeof(pl)/1000000, " mb")
        print("\t size of results_list:       ", sys.getsizeof(results_list)/1000000, " mb")
    
    
        # ''' Generate level 1 output - all detected particles'''
        # ''' REMOVE PARTICLE FOUND IN LESS THAN 'bufsize' FRAMES '''
        # #keep only if longer than 10 frames
        # tag = " - 1 remove short lived particles" 
        # pl2 = []
        # new_id = 1
        # for p in pl:
        #     if len(p.f_vec) > constants["bufsize"]:
        #         pl2.append(p)
        #         pl2[-1].pID = new_id
        #         new_id += 1
                
        # iscat.save_particle_list_video(r8, pl2, constants, tag)
        # iscat.generate_particle_list_csv(pl2, constants, tag)
        # iscat.generate_landing_rate_csv(pl2, constants, tag)
        # iscat.plot_landing_map(pl2, constants, tag)
        # iscat.plot_landing_rate(pl2, constants, tag)
        # iscat.plot_contrast(pl2, constants, tag)
    
    
    
        # ''' Generate level 2 output - fit gaussian and remove false positives'''
        # ''' FIT GAUSSIAN DATA '''
        # tag = " - particle v removal"
        
        # run iscat.particle_v_removal_tool(particle list) 
        #pl2_fit = iscat.fit_gaussian_to_particle_list(pl2, constants)
    
        # iscat.plot_particle_contrast2(pl2_fit, constants)
        # iscat.save_pickle_data(pl2_fit, constants, tag)
        
        # iscat.save_particle_list_video(r8, pl2_fit, constants, tag)
        # iscat.generate_particle_list_csv(pl2_fit, constants, tag)
        # iscat.generate_landing_rate_csv(pl2_fit, constants, tag)
        # iscat.plot_landing_map(pl2_fit, constants, tag)
        # iscat.plot_landing_rate(pl2_fit, constants, tag)
        # iscat.plot_contrast(pl2_fit, constants, tag)
        
        # #unused plotting output functions
        # #iscat.plot_sdcontrast_hist(pl_fit, constants, tag)
        # #iscat.plot_waterfall(pl_fit, constants, tag)
            
    ###########################################################################
    et = time.time()
    print("\n", name)
    print("\n\nFINSIHED... Processed ", os.path.getsize(binfile), " bytes in ", ((et-st)/60), " minutes\n\n")
            
     



# binfile = r"C:/Users/Matt/Desktop/PS ITO DATA wait 1 minute/25 nm/2023-03-23-25nm new ITO 150mW/VIDEOS/2023-03-23_15-09-20_raw_12_256_200.bin"
# sample_name = "test"
# generate_exp_data(binfile, sample_name)


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


root_dir = r"D:/ACS NANO TESTS/7-5 large particle centering tests"
root_dir = r"D:/ACS NANO TESTS/PS - Contrast 3"
root_dir = r"D:/Carraugh/Aug 29"
root_dir = r"D:/_DATA FOLDER_/Teresa"
root_dir = r"D:/_DATA FOLDER_/Haoxin"
root_dir = r"D:/_DATA FOLDER_/Haoxin 2"
root_dir = r"D:/_DATA FOLDER_/extra 0.3 nM data"
root_dir = r"D:/_DATA FOLDER_/haoxin 3"
root_dir = r"D:/_DATA FOLDER_/Haoxin 4"
root_dir = r"D:/_DATA FOLDER_/Carraugh oct 26"
root_dir = r"D:/ADDITIONAL DATA FOR ACS NANO/new AuNP Data 2"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/matt new nov 14"
root_dir = r"D:/ADDITIONAL DATA FOR ACS NANO/matt new nov 14"
root_dir = r"D:/ADDITIONAL DATA FOR ACS NANO/unprocessed"
root_dir = r"D:/ADDITIONAL DATA FOR ACS NANO/new AuNP Data"
root_dir = r"D:/_DATA FOLDER_/Carraugh nov 21"
root_dir = r"D:/_DATA FOLDER_/Carraugh -  BGG2"
root_dir = r"C:/Users/Matt/Desktop/EPD-iSCAT NTA/skitterbugs"
root_dir = r"D:/_DATA FOLDER_/flaminia March22.2024"
root_dir = r"D:/_DATA FOLDER_/flaminia March22.2024/peg"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/recent experiments april 2024"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/Tomas data"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/new protein data"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/new protein data May 2024"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/files with errors"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/Peter"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/Peter/peter redo at 20 bin"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_peter day 2"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_new laser camera"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_reprocess old PSNP data"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/new camera 2"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/new camera 3"
root_dir = r"E:/Best Data"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/new lens"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_peter MPL/peter day 1 data/Data/pre filter data"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_peter MPL/peter day 2 data/MPN filter"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/7-10 iscat demo 50nm psnp"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_teresa/20240716"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_teresa/20240717"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_new lasertack EPD laser power landing rate"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_nanoplastic generation TMS MDK"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/to process"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/25nm partial reflector test"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/2. to process"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/2. to process - ITO test prints for SEM"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/2. to process - ITO test prints 1"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/100 nm psnp print for sem"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_0. pet print unfinished"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/_0. 100 nm psnp print for sem/MDK II 45 A - new gauss code test"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/RDF experiments 1"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/0. successful 100nm and 200 nm prints"

strt = time.time()

# this called the batch_process_exp_data function, which loops through all
# the files in a folder and calls generate_exp_data function on each one.
# you can change processing parameters in the generate_exp_data function.
a, b, c = batch_process_exp_data(root_dir)

endt = time.time()
print("Total Runtime: ", ((endt-strt)/60), " minutes")





