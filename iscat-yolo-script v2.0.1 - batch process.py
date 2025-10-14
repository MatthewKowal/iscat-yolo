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
            

The results produced:
    info
    info
    info
    info
    
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

 
    #nm_per_px     = 37.5#fov*1000/x
    constants = {}
    constants['sample name']        = "test" #sample_name
    constants['bufsize']            = 40# 60#100#20                   # 1/2 the ratiometric buffer size
    constants['clipmin']            = 0.98#0.98#0.97                 #0.95,  # black point clipping of ratiometric image
    constants['clipmax']            = 1.02#1.02#1.03                 #1.05,  # white point clipping of ratiometric image

    constants['bbox']               = 15            # 1/2 the width of a saved particle image. this also defines where to cut off edge particles
    #constants['MIN_DIST']           = 5#20#10      # minimum distance for new overlapping particles to be considered the same      
    constants['spatial tolerance1']  = 10#5#20#10   # maximum distance from particle in the same frame to be considered the same
    constants['spatial tolerance2']  = 10#5#20#10   # maximum d☺istance from particle in a previous frame to be considered the same
    constants['temporal tolerance']  = 100#500#10     # maximum number of skipped frames for particle to be considered the same      
    #constants['history tolerance']  = 500#100      # maximum number of frames to look back before giving up the search for a matching particle
    #constants["frame std cutoff"]   = 1             # do not count particles when the standard deviation of the entire frame (a sign of stage movement) exceeds this value
    constants["video x dim"]        = x
    constants["video y dim"]        = y
    constants["basepath"]           = basepath
    
    constants["name"]               = name
    constants["fps"]                = fps
    constants["fov"]                = fov
    constants['output framerate']   = fps#fps #200, 24
    constants["nframes"]            = nframes #total number of frames of the raw file
    constants["timestamp"]          = name[:19]
    constants["frame total"]        = nframes  #total number of frames of the cropped file
    
    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train31/weights/best.pt'
    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train32/weights/best.pt'
    #constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train37/weights/best.pt'
    constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train47/weights/best.pt'
    constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train49/weights/best.pt'   #latest roboflow trained as of 8/13/2025
    constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train58 - deeptrack db v1/weights/best.pt' #simulation dataset
    constants["yolo model loc"]     = 'C:/Users/Matt/runs/detect/train58 - deeptrack db v1/weights/best.pt' #simulation dataset
    constants["yolo model loc"]     = r"C:/Users/Matt/Desktop/EPD-iSCAT PYTHON SCRIPTS/runs/detect/train2/weights/best.pt" #hybrid dataset model
    
    constants["min confidence"]         = 0.05 #0.05 #0.25 #0.45 # good for contrast accuracy     # 0.3 #good compromise #0.05 # good for exploratory dialing back
 
    constants["output path"]        = os.path.join(basepath, (name[:19] + " yolo_out " + iscat.getVersion() +"buf"+str(constants['bufsize'])+"conf"+str(constants['min confidence'])))
    
    if not os.path.exists(constants["output path"]): os.makedirs(constants["output path"])
    
    if constants["nframes"] < 2*constants["bufsize"]: return
 
    # # ########################################################
    # # # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
    # start_frame=0
    # use_short_video = False
    
    # if use_short_video:
    #     start_frame = 6900#30000#350  #2200
    #     nframes     = 1000 #12000  # 12000 frames is about a minute of video
    #     binimages      = binimages[start_frame:(start_frame+nframes)]
    #     constants["nframes"] = nframes
    # # else:
    # #     # # # OR USE THE FULL FILE
    # #     images = binimages
    # # ########################################################
     
    ''' RAW BINFILE OUTPUT '''
    iscat.save_bw_video(binimages, constants, "rawbin", print_frame_nums=False)
    iscat.save_constants(constants)
    
    ''' Choose the processing modes '''
    trim_video                  = False
    load_pickle                 = False
    ratio_only                  = False
    brightness_only             = False
    processing_level1_stdevonly = True  #usually just use this one
    calculate_gaussian          = False #this takes a while so I ususally keep it False
    save_pickle                 = True #this makes a huge file but it can be useful because you get the whole particle list instead of just whats in the csv file
    
    
    if brightness_only == True:
        iscat.measure_video_brightness(binimages, constants)
        
    if trim_video == True:
        #start_frame=0
        start_frame = 3000#30000#350  #2200
        nframes     = 3000 #12000  # 12000 frames is about a minute of video, 3000 is about 15 seconds
        binimages      = binimages[start_frame:(start_frame+nframes)]
        constants["nframes"] = nframes
    
    if load_pickle == True:
        pickle_path = os.path.join(constants["output path"], (constants["timestamp"]+" - 0 yolo raw output.pkl"))
        pl = iscat.load_pickle_data(pickle_path)
        iscat.plot_pimages(pl, constants)
        
    if ratio_only == True:
        r8 = iscat.ratio_only(binimages, constants)
        iscat.measure_video_brightness(binimages, constants)
        iscat.save_bw_video(r8, constants, ("ratio"+str(constants["bufsize"])), print_frame_nums=False)
        iscat.save_si_video2(r8, [], constants, " - voltage video", offset=0, print_fnum=False, print_v=True)

    if processing_level1_stdevonly == True:#False#else:
        
        '''RATIOMETRIC PARTICLE FINDER'''
        tag = " - 0 yolo raw"
        #RATIOMETRIC
        r8, pl, results_list, noise, screen_std_list = iscat.ratio_particle_finder(binimages, constants)
        iscat.measure_video_brightness(binimages, constants)
        iscat.generate_noisefloor_csv(noise, screen_std_list, constants, "buf"+str(constants['bufsize']))
        iscat.save_bw_video(r8, constants, "ratio"+str(constants['bufsize']), print_frame_nums=False)
        
        #GENERATE OUTPUT LEVEL 1 - YOLO RAW
        iscat.plot_sdcontrast_hist(pl, constants, tag) #this one only plots std contrast
        iscat.plot_landing_map(pl, constants, tag)
        iscat.plot_landing_rate(pl, constants, tag)
        iscat.generate_landing_rate_csv(pl, constants, tag)
        iscat.plot_contrast_kinetics(pl, constants, tag)
        
        #VIDEO
        iscat.save_particle_list_video(r8, pl, constants, tag, skip_gauss=True)
        #iscat.save_si_video2(r8, [], constants, " - voltage video", offset=0, print_fnum=False, print_v=True
        
        #CALCULATE GAUSSIAN
        if calculate_gaussian:
            pl = iscat.fit_gaussian_to_particle_list(pl, constants)
            iscat.plot_landing_map2(pl, constants, tag+"_2") #this one requires gauss
            iscat.plot_contrast(pl, constants, tag) #this one plots gauss and std contrast
        
        #SAVE PARTILE LIST
        iscat.generate_particle_list_csv(pl, constants, tag)
        if save_pickle: iscat.save_pickle_data(pl, constants, tag)
        
    

    def a_list_of_all_processing_methods():
        
        #generate ratiometric video and find particles
        r8, pl, results_list, noise = iscat.ratio_particle_finder(binimages, constants)
        
        #video feed reference data
        iscat.measure_video_brightness(binimages, constants)
        iscat.generate_noisefloor_csv(noise, constants, "buf"+str(constants['bufsize']))
        
        #gaussian fitting
        pl = iscat.fit_gaussian_to_particle_list(pl, constants)

        #database
        iscat.generate_particle_list_csv(pl, constants, tag)
        iscat.save_pickle_data(pl, constants, tag)
        iscat.plot_pimages(pl, constants)
        iscat.save_constants(constants)

        #video
        iscat.save_bw_video(r8, constants, "ratio"+str(constants['bufsize']), print_frame_nums=True)
        iscat.save_particle_list_video(r8, pl, constants, tag)
        iscat.save_si_video(r8, pl, constants, tag, offset=0, print_fnum=False, print_v=False)
        iscat.save_si_video2(r8, [], constants, " - voltage video", offset=0, print_fnum=False, print_v=True)
        
        #location data
        iscat.plot_landing_map(pl, constants, tag)
        iscat.plot_landing_map2(pl, constants, tag)
        
        #kinetics data
        iscat.plot_landing_rate(pl, constants, tag)
        iscat.generate_landing_rate_csv(pl, constants, tag)
        iscat.plot_waterfall(pl, constants, tag)
        
        #contrast data
        iscat.plot_contrast(pl, constants, tag)
        iscat.plot_sdcontrast_hist(pl, constants, tag)
        iscat.plot_particle_contrast2(pl, constants)
        
        #file loading functions
        iscat.load_video("video filepath") #load a raw or ratiometric video WITHOUT annotations. must be BW video
        iscat.load_constants("directory where constants.txt lives")
        iscat.load_pickle_file("pickle filepath")
            
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
    
    
    for f in os.listdir(root_dir):                   #go through the root directory
        print(f)
        line = ''.join(["#"]*60)
        print("\n\n")
        print(line)
        print(line)
        print(line)
        #print(f)
        sample_name = f[11:]
        print(sample_name)
        #print(sample_name)
        folder = os.path.join(root_dir, f)#, 'VIDEOS')
        print(folder)
        for filename in os.listdir(folder):
            print(filename)
            if filename.endswith('.bin'):
                binfile = os.path.join(folder, filename)
                binfiles.append(binfile)
                folders.append(folder)
                sample_names.append(sample_name)
                #print(binfile, "\n", sample_name, "\n\n")
                generate_exp_data(binfile, sample_name)        #generate experiment output for a single file
                
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
root_dir = r"C:[/Users/Matt/Desktop/ISCAT EXPERIMENTS/RDF experiments 1"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/0. successful 100nm and 200 nm prints"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/MDK 55 good"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/MDK 55 bad"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/200 nm ps on PDL"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/mdk 57"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/mdk 57 B"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/0. Biotin Then Step"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/PSF size"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/LD-EPD-iSCAT contrast data feb 2025"
root_dir = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\LD-EPD-iSCAT contrast III"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/LD-EPD - Laser Power Landing Rate II"
root_dir = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\LD-EPD - EPD Landing Rate II"
root_dir = r"E:\LD-EPD-iSCAT DATA\LD-EPD - Laser Power Landing Rate III"
root_dir = r"E:\LD-EPD-iSCAT DATA\Saturday March 15th experiments"
root_dir = r"E:/LD-EPD-iSCAT DATA/LDLR IV"
root_dir = r"E:/LD-EPD-iSCAT DATA/Friday March 21st experiments"
root_dir = r"E:/LD-EPD-iSCAT DATA/LD-EPD - EPD Landing Rate V - 1 mW"
root_dir = r"E:/LD-EPD-iSCAT DATA/PS 50 PS 100"
root_dir = r"E:/LD-EPD-iSCAT DATA/test"
root_dir = r"E:/LD-EPD-iSCAT DATA/LD-EPD - EPD Landing Rate V - 1 mW"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/PS 50 PS 100 II"
root_dir = r"E:/LD-EPD-iSCAT DATA/PS 50 PS 100 - for Ed/PDL III"
root_dir = r"E:\LD-EPD-iSCAT DATA\PS 50 PS 100 - for Ed\ITO II"
root_dir = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\TEST CODE BATCH"
root_dir = r"E:/LD-EPD-iSCAT DATA/test data ii - sticky movers"
root_dir = r"E:/LD-EPD-iSCAT DATA/PS 50 PS 100 - for Ed/PDL III"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/epd prints"
root_dir = r"E:/LD-EPD-iSCAT DATA/LD-EPD - Laser Power Landing Rate I - no bicarb - 1.1VEPD/data"
root_dir = r"E:/LD-EPD-iSCAT DATA/LD-EPD - EPD Landing Rate I - maybe 4 mW/data"
root_dir = r"E:/LD-EPD-iSCAT DATA/test data ii - sticky movers"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/LP kinetics test runs - 1.2v aobd one bubble repeats"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/LP kinetics test runs - 1.2v aobd 6mW one bubble"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/LP EPD kinetics friday"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/KINETICS - LP"
root_dir = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\KINETICS - LP 0"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/KINETICS - LP - Monday 0"
#root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/KINETICS - TEST PARTICLE LINKING"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/KINETICS - EPD - 0.01nM"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/Kinetics friday may 9"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/SYC contrast day 1"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/contrast voltage sweep 1"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/contrast voltage sweep and background"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/serenas experimetns friday june 13"
root_dir = r"E:/LD-EPD-iSCAT DATA/LD-EPD - prints/mdk 45 A"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/MDK friday 6-20"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/MDK friday 6-20 - phosphate contrast voltage"
root_dir = r"E:/LD-EPD-iSCAT DATA/LD-EPD - contrast II"
root_dir = r"E:\LD-EPD-iSCAT DATA\RDF tests"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/2025_07_27 SYC Contrast 2"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/2025-07-30 SYC IPA Experiments"
root_dir = r"E:/LD-EPD-iSCAT DATA/PR contrast dependence for 50 nm PSNP on ITO - DeepTrack DB Test"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/SYC aug 16 ITO kinetics"
root_dir = r"E:/LD-EPD-iSCAT DATA/THESIS Data/some contrast data"
root_dir=r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/SYC sept 12 test RDF"
root_dir = r"C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/SYC sept. 24 0.1 nM PS50 in 1e-6"
strt = time.time()

# this called the batch_process_exp_data function, which loops through all
# the files in a folder and calls generate_exp_data function on each one.
# you can change processing parameters in the generate_exp_data function.
a, b, c = batch_process_exp_data(root_dir)

endt = time.time()
print("Total Runtime: ", ((endt-strt)/60), " minutes")























