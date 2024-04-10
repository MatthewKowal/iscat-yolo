# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:46:14 2023


Text header created using:
    Text to ASCII Art Generator (TAAG)
    http://www.patorjk.com/software/taag/
    Font: Slant
    Char Width: Fitted
    Char Height: Fitted

@author: Matt
"""

# bin file mamangement
import os
import time
import numpy as np
import tqdm

# ratiometric particle finder
from collections import deque
from ultralytics import YOLO
from skimage import color
import random


#particle video
import cv2
import PIL

#particle list management
from math import dist

#spreadsheet generation
import pandas as pd

#plot images
import matplotlib.pyplot as plt


# i/o helper functions
# pickle file io, load video into array, constants io
import pickle 
import ast
import skvideo




'''
###############################################################################
###############################################################################
###############################################################################
         ____   _           ____ _  __                                     
        / __ ) (_)____     / __/(_)/ /___                                  
       / __  |/ // __ \   / /_ / // // _ \                                 
   _  / /_/ // // / / /  / __// // //  __/                                 
  (_)/_____//_//_/ /_/  /_/  /_//_/ \___/                                  
      __  ___                                                           __ 
     /  |/  /____ _ ____   ____ _ ____ _ ___   ____ ___   ___   ____   / /_
    / /|_/ // __ `// __ \ / __ `// __ `// _ \ / __ `__ \ / _ \ / __ \ / __/
   / /  / // /_/ // / / // /_/ // /_/ //  __// / / / / //  __// / / // /_  
  /_/  /_/ \__,_//_/ /_/ \__,_/ \__, / \___//_/ /_/ /_/ \___//_/ /_/ \__/  
                               /____/                                    
###############################################################################
###############################################################################
###############################################################################
'''#!!!

def loadscreen():
    print("  _ ___  ___   _ _____  __   _____  _    ___            ")
    print(" (_) __|/ __| /_\_   _| \ \ / / _ \| |  / _ \           ")
    print(" | \__ \ (__ / _ \| |    \ V / (_) | |_| (_) |          ")
    print(" |_|___/\___/_/_\_\_|   _ |_| \___/|____\___/ _         ")
    print(" | _ \__ _ _ _| |_(_)__| |___  | __(_)_ _  __| |___ _ _ ")
    print(" |  _/ _` | '_|  _| / _| / -_) | _|| | ' \/ _` / -_) '_|")
    print(" |_| \__,_|_|  \__|_\__|_\___| |_| |_|_||_\__,_\___|_|  ") 
    print("                                                        ")
    print("\t\t\t\t\t", getVersion(), "   Matthew Kowal 2023\n\n")

    
    
def getVersion():
    # returns the scripts version number as a string
    #   this is to be used for printing the version
    #   number on the output folder.
    folder, filename = os.path.split(__file__)
    x = filename[:filename.rfind("-")-1]
    return x[x.rfind("v"):]
  
def get_bin_metadata(binfile, printmeta=True):
    
    basepath, filename = os.path.split(binfile)
    name               = filename.split(".")[0]
    filetype           = filename.split(".")[1]
    metadata           = name.split("_")
    print(metadata)
    
    date = metadata[0]
    time = metadata[1]
    fov  = int(metadata[3])
    x    = int(metadata[4])
    y    = int(metadata[4])
    fps  = int(metadata[5])
    
    filesize        = os.path.getsize(binfile)
    
    if filetype=="bin": nframes   = int(filesize / (x * y))
    if filetype=="mp4": nframes   = load_video(binfile).shape[0]
    
    remaining_bytes = filesize%(x * y)
    
    if os.path.exists(os.path.join(basepath, (date+"_"+time+"_"+"voltage.txt"))):
        print("Found a voltfile, ok")
        voltfile = os.path.join(basepath, (date+"_"+time+"_"+"voltage.txt"))
    else:
        print("did not find a volt file")
        voltfile = False
    print(voltfile)
    
    if printmeta:         #print everything out
        print("\nFile Properties")
        print("\tLocation:\t\t\t", basepath)
        print("\tFilename:\t\t\t", filename)
        print("\tSquare FOV (um) :\t", fov)
        print("\tX Resolution : \t\t", x)
        print("\tY Resolution : \t\t", y)
        print("\tFrames per second:  ", fps)
        print("\tFile Size: \t\t\t", filesize)
        print("\tNumber of frames: \t", nframes)
        print("\tRunning time:(s) \t", (nframes/fps), " seconds")
        print("\tRunning time (m): \t", (nframes/fps/60), " minutes")
        print("\tRemaining Bytes: \t", remaining_bytes)

    return basepath, filename, name, nframes, fov, x, y, fps, voltfile

# def get_video_metadata(binfile):
#     #could replace this with a get_video_metadata function (should write one)
#     basepath = os.path.join(script_dir, "example")
#     filename = "filename"
#     nframes  = binimages.shape[0]
#     fov      = 12 #microns
#     x        = binimages.shape[1]
#     y        = binimages.shape[2]
#     fps      = 200
#     filesize = os.path.getsize(mp4file)
#     name     = "date_time_raw"+"_"+str(fov)+"_"+str(x)+"_"+str(fps)
#     print("\nVideo File Properties:")
#     print("\tLocation: \t", os.path.split(mp4file)[0])
#     print("\tFilename: \t", os.path.split(mp4file)[1])
#     print("\tSquare FOV (um): \t", fov)
#     print("\tX Resolution : \t\t", x)
#     print("\tY Resolution : \t\t", y)
#     print("\tFrames per second:  ", fps)
#     print("\tFile Size: \t\t\t", filesize)
#     print("\tNumber of frames: \t", nframes)
#     print("\tRunning time:(s) \t", (nframes/fps), " seconds")
#     print("\tRunning time (m): \t", (nframes/fps/60), " minutes")
#     return basepath, filename, name, nframes, fov, x, y, fps 


def load_binfile_into_array(binfile, print_time=True): #open a binfile and import data into image array
    print("\n\nLoading binfile into memory...")
    if print_time: start=time.time()
    
    #get constants    
    basepath, filename, name, nframes, fov, x, y, fps, voltfile = get_bin_metadata(binfile, printmeta=False) # get basic info about binfile
    
    # import video
    dt = np.uint8                                     # choose an output datatype
    images = np.zeros([nframes, x, y], dtype=dt)      # this will be the output var
    print("\nImporting....")
    for c in tqdm.tqdm(range(nframes)):
        #print(c)
        frame1d = np.fromfile(binfile, dtype=np.uint8, count=(x*y), offset=(c*x*y))                      # read one chunk (a frame) from binary file as a row vector
        frame2d = np.reshape(frame1d, (x,y))                                                             # reshape the row vector as a 2d fram (x,y)
        images[c] = np.array(frame2d, dtype=dt)                                                          # add the frame to the output array of images
        if np.min(images[c] < 0):   print("WARNING, FRAME ", c, " OUT OF RANGE: ", np.min(images[c]))    # warn if minimum < 0 
        if np.max(images[c] > 255): print("WARNING, FRAME ", c, " OUT OF RANGE: ", np.max(images[c]))    # warn if maximum > 255
    #save_vid(images, 24, video_file)
    #printout = "saved " + str(video_file) + " OK!\n\n"
    #print(type(video_file))
    #print(type(printout))
    #return printout
      
    if print_time:
        end=time.time()
        print("\n\n\t Binfile Size:\t\t ", os.path.getsize(binfile) / 1E6, "Megabytes")
        print("\t Elapsed time (s):\t ", (end-start), " seconds")
        print("\t Elapsed time (m):\t ", ((end-start)/60), " minutes")
        print("\t Speed (Mbps):\t\t ", ( os.path.getsize(binfile) / 1E6) / (end-start), "Mb / s" )
        
    return images




'''
###############################################################################
###############################################################################
###############################################################################
        ____          __   _                          __         _      
       / __ \ ____ _ / /_ (_)____   ____ ___   ___   / /_ _____ (_)_____
      / /_/ // __ `// __// // __ \ / __ `__ \ / _ \ / __// ___// // ___/
     / _, _// /_/ // /_ / // /_/ // / / / / //  __// /_ / /   / // /__  
    /_/ |_| \__,_/ \__//_/ \____//_/ /_/ /_/ \___/ \__//_/   /_/ \___/  
        ____                __   _        __                            
       / __ \ ____ _ _____ / /_ (_)_____ / /___                         
      / /_/ // __ `// ___// __// // ___// // _ \                        
     / ____// /_/ // /   / /_ / // /__ / //  __/                        
    /_/     \__,_//_/    \__//_/ \___//_/ \___/                         
        ______ _             __                                         
       / ____/(_)____   ____/ /___   _____                              
      / /_   / // __ \ / __  // _ \ / ___/                              
     / __/  / // / / // /_/ //  __// /                                  
    /_/    /_//_/ /_/ \__,_/ \___//_/                               

###############################################################################
###############################################################################
###############################################################################
'''#!!!

# Particle Class
class Particle:
    def __init__(self, px, py, wx, wy, first_frame_seen, pID, yoloimage, pimage, stddev, mean, conf, new_pimage, new_stddev):
        self.px_vec            = [px]                # particle x position for each frame
        self.py_vec            = [py]                # particle y position for each frame
        self.wx_vec            = [wx]                # particle bbox width for each frame
        self.wy_vec            = [wy]                # particle bbox height for each frame
        
        self.f_vec            = [first_frame_seen]     # first frame the particle was seen
        self.pID              = pID 
        
        self.yoloimage_vec         = [yoloimage]             # 16bit image straight from yolo
        self.pimage_vec            = [pimage]                # 16bit 30x30 particle image
        self.stddev_vec            = [stddev]                # particle bbox height for each frame
        self.mean_vec              = [mean]
        self.conf_vec              = [conf]

        self.new_pimage_vec        = [new_pimage]
        self.new_stddev_vec        = [new_stddev]
        
    def updateParticle(self, newp): # Take in a new particle and use its specs to update yourself
        self.px_vec.append(newp.px_vec[0])
        self.py_vec.append(newp.py_vec[0])
        self.wx_vec.append(newp.wx_vec[0])
        self.wy_vec.append(newp.wy_vec[0])
        
        self.f_vec.append(newp.f_vec[0])
        
        self.yoloimage_vec.append(newp.yoloimage_vec[0])
        self.pimage_vec.append(newp.pimage_vec[0])
        self.stddev_vec.append(newp.stddev_vec[0])
        self.mean_vec.append(newp.mean_vec[0])
        self.conf_vec.append(newp.conf_vec[0])
        
        self.new_pimage_vec.append(newp.new_pimage_vec[0])
        self.new_stddev_vec.append(newp.new_stddev_vec[0])
        

def ratio_only(images, constants, return_v16=False):
    '''STEPS:
        1. Define constants, video dimensions, queues, image & video arrays
        2. Perform ratiometric image processing
        3. Return an 8-bit video
    '''
    start               = time.time()
    #DEFINE CONSTANTS
    bufsize             = constants['bufsize']
    print("bufsize", bufsize)
    clipmin             = constants['clipmin']
    clipmax             = constants['clipmax']
    n,x,y               = images.shape    
    print("n, x, y ", n,x,y)
    i16                 = np.ndarray([x,y]).astype(np.float16)
    i8                  = np.ndarray([x,y]).astype(np.uint8)
    #print(n, bufsize,x,y)
    v8                  = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)       # this contains the 8 bit video
                                  # has enough elements for every frame, but we will only populate it for frames with particles 
    if return_v16: v16 = np.ndarray([n-2*bufsize,x,y]).astype(np.float16)

    noise_floor_list    = [1]*(n-2*bufsize)
    print("\n\nPerforming Ratiometric Image Process on ", n, " Frames")
    print(" \t\t Clip Length: ", (n/constants["fps"] / 60), " minutes.\n")

                # Step 1: autofill the deques with the first 'bufsize x 2' number of frames
    d1 = deque(images[:bufsize].astype(np.float16), bufsize)#[]
    d2 = deque(images[bufsize:(2*bufsize)].astype(np.float16), bufsize)#[]

    for f in tqdm.tqdm(range(n-2*bufsize)):
        # f is frame number
        ''' RATIOMETRIC '''
        # Create Ratiometric Frame
        d1sum = np.sum(d1, axis=0).astype(np.float16)  #sum queue1                  
        d2sum = np.sum(d2, axis=0).astype(np.float16)
        #i16 = ( d2sum/np.sum(d2sum) ) / (d1sum/np.sum(d1sum))
        i16 = d2sum / d1sum
        if return_v16: v16[f] = i16
        
        #if f==0: info = [d1, d2, d1sum, d2sum, i16]
        # Manage Frame Queue
        d1.append(d2.popleft())
        d2.append(images[f+2*bufsize])
        #save frame as an 8-bit image and append it to a video
        i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
        v8[f] = i8
        
        noise_floor_list[f] = get_noise(i16, constants)
        
    end = time.time()
    print("RATIOMETRIC PARTICLE FINDER SPEED REPORT:")
    print("\t Resolution: \t\t\t\t\t", x, y, "px, \t",  n, "frames")
    print("\t Elapsed time: \t\t\t\t\t", (end-start), " seconds")
    print("\t Frames per Second: \t\t\t", (n / (end-start)), " fps" )
    #print(v8.shape, v16.shape, len(noise_floor_list), return_v16)
    if return_v16: return v8, v16, noise_floor_list
    else: return v8, noise_floor_list
      
    
    
        
# This is the main ratiometric particle finder function. 
# other functions are called within this function.
# Input: Raw Binar array for raw video frames
# Method: 
#   Step 1 - Fill the ratiometric buffer with the first 2*bufsize number of frames
#   Step 2 - Generate ratiometric video and find particles in frame.
#   Step 3 - Yolov8 the frame and make an initial set of particles for the frame
#   Step 5 - Remove overlapping particles and edge particles
#   Step 6 - Merge particles between frames
#   Step 7 - Measure image background noise
# Output: 
#                  v8 = ratiometric video (8-bit)
# video_particle_list = a list of particles found and their measured data
#        results_list = original result output from yolo (not sure if i need this)
#              v16roi = 16-bit ratiometric video, but the only frames saved are the ones with particle landings
#    noise_floor_list = a list of minimum measured sd noise for each frame
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import math
import tracemalloc
def ratio_particle_finder(images, constants):
    '''STEPS:
        1. Define constants, video dimensions, queues, image & video arrays
        2. Perform ratiometric image processing
        3. Return an 8-bit video
    '''
    start               = time.time()
    
    
    #DEFINE CONSTANTS
    bufsize             = constants['bufsize']
    clipmin             = constants['clipmin']
    clipmax             = constants['clipmax']
    
    bbox                = constants['bbox']
    
    n,x,y               = images.shape    
    i16                 = np.ndarray([x,y])#.astype(np.float16)
    i8                  = np.ndarray([x,y])#.astype(np.uint8)
    v8                  = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)       # this contains the 8 bit video
    #v16roi              = [0]*(n-2*bufsize)                                    # has enough elements for every frame, but we will only populate it for frames with particles 
    noise_floor_list    = [1]*(n-2*bufsize)                                       # this records the s.d. noise floor for each frame
    
    yolo_model          = YOLO(constants["yolo model loc"])
    results_list        = [0]*(n-2*bufsize)                                    # yolo results for each frame
    
    video_particle_list = []
    #video_particle_list = [0]*100000 
    pID = 1
    #MIN_DIST = 15
    MIN_DIST = constants['MIN_DIST']
    
    print("\n\nPerforming Ratiometric Image Process on ", n, " Frames")
    print(" \t\t Clip Length: ", (n/constants["fps"] / 60), " minutes.\n")


    
    #PERFORM RATIOMETRIC, PARTICLE FINDING, PARTICLE LIBRARY MANAGEMENT

                # Step 1: autofill the deques with the first 'bufsize x 2' number of frames
    d1 = deque(images[:bufsize].astype(np.float16), bufsize)#[]
    d2 = deque(images[bufsize:(2*bufsize)].astype(np.float16), bufsize)#[]


                # Step 2: For the remaining framse:
                #    - Ratiometric process
                #    - Particle finding
                #    - Particle library updating
                
    

    timestamp_array = [0]*(n-2*bufsize)
    TRACEMEMORY = False
    for f in tqdm.tqdm(range(n-2*bufsize)):
        # f is frame number
        
        
        if TRACEMEMORY: tracemalloc.start()
        
        t1 = time.time()
        
        ''' RATIOMETRIC '''
        # Create Ratiometric Frame
        d1sum = np.sum(d1, axis=0).astype(np.float16)  #sum queue1                  
        d2sum = np.sum(d2, axis=0).astype(np.float16)
        #i16 = ( d2sum/np.sum(d2sum) ) / (d1sum/np.sum(d1sum))
        i16 = d2sum / d1sum
        
        #if f==0: info = [d1, d2, d1sum, d2sum, i16]
        # Manage Frame Queue
        d1.append(d2.popleft())
        d2.append(images[f+2*bufsize])
        #save frame as an 8-bit image and append it to a video
        i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
        v8[f] = i8
        
        
        t2 = time.time()
        
        ''' PAD THE IMAGE SO THAT WE DO NOT DETECT EDGE PARTICLES '''
        pillowimage = PIL.Image.fromarray(i8)
        draw = PIL.ImageDraw.Draw(pillowimage)
        x = 255
        y = 255
        e = 15
        draw.rectangle((0, 0, x, e), fill=128, outline=128)
        draw.rectangle((0, y-e, x, y), fill=128, outline=128)
        draw.rectangle((0, 0, e, y), fill=128, outline=128)
        draw.rectangle((x-e, 0, x, y), fill=128, outline=128)
        i8_padded = np.array(pillowimage)
        
        
        ''' Particle Finding - Using YOLO '''
        #use this if you want to keep everything in the function (cleaner)
        rgb_image = color.gray2rgb(i8_padded) # numpy grayscale array to numpy rgb array
        frame_results = yolo_model(rgb_image, conf=constants["min confidence"], verbose=False, save_conf=True)
        results_list[f] = frame_results
        
        #use this if you want to call the function below
        # frame_results = yolo_finder(i8, yolo_model)
        # results_list[f] = frame_results
        
        
        t3 = time.time()
        
        '''Convert Yolo format results into a slim, conventient format (a list of lists)'''
        frame_pl = []
        frame_boxes = frame_results[0]
        for p in frame_boxes.boxes:
            #collect particle data
            px   = int(p.xywh[0][0])
            py   = int(p.xywh[0][1])
            wx   = int(p.xywh[0][2])
            wy   = int(p.xywh[0][3])         
            conf = float(p.conf)
            #add the particle to a list of frame particles
            frame_pl.append([px, py, wx, wy, conf])
        
        
        '''REMOVE OVERLAPPING PARTICLES (in this frame)'''
        temp_frame_pl = []
        for p in frame_pl:      # loop each particle on frame
            
            px   = p[0]
            py   = p[1]
            wx   = p[2]
            wy   = p[3]
            conf = p[4]

            # keep only particles that are not close to an already confirmed frame particle
            matched = False
            for i in temp_frame_pl: 
                # if x is too close to another particle and y is too close to another particle, then we found a match
                if np.absolute(i[0] - px) < MIN_DIST and np.absolute(i[1] - py) < MIN_DIST: matched = True
            # if we never found a match then add the particle to the list
            if not matched:
                temp_frame_pl.append(p)
        frame_pl = temp_frame_pl
        
        
        
        '''REMOVE EDGE PARTICLES'''
        temp_frame_pl = []
        for p in frame_pl:
            px   = p[0]
            py   = p[1]
            wx   = p[2]
            wy   = p[3]
            conf = p[4]
            if px + bbox < x and px - bbox >= 0 and py + bbox < y and py - bbox >= 0: #only add particles where i can save a whole image (bbox*2, bbox*2)
                temp_frame_pl.append(p)
        frame_pl = temp_frame_pl
        
        t4 = time.time()
        
        ''' FRAME PARTICLE LIST BOOKKEEPING  '''
        for p in frame_pl:
            #v16roi[f] = i16 #add the frame to the 16bit roi video. we will use this later when generating particle images

            
            # new particle data
            px   = p[0]
            py   = p[1]
            wx   = p[2]
            wy   = p[3]
            conf = p[4]     # new particle confidence value (float)
            hwx = int(np.floor(wx/2))
            hwy = int(np.floor(wy/2))
            yoloimage = i16[ (py-hwy):(py+hwy), (px-hwx):(px+wx) ]                # original yolo image (full width)
            pimage = i16[ (py-bbox):(py+bbox), (px-bbox):(px+bbox) ]           # new particle 16 bit image
            stddev = np.std(pimage)                                            # new particle standard deviation
            mean   = np.mean(pimage)
            
            ''' find more accurate center of particle and calculate new stddev '''
            yi = yoloimage
            CANNY_SIGMA = 2
            CANNY_LOW_THRESH = 30
            CANNY_HIGH_THRESH = 50
            #HOUGH_RADIUS = 4 # 19, 10, 4
            clipmin = constants["clipmin"]
            clipmax = constants["clipmax"]
            bbox = constants["bbox"]
            yi8 = np.clip( ((yi - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
            yi_edges = canny(yi8, CANNY_SIGMA, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
            h_radius = 4
            h_res_ = hough_circle(yi_edges, [h_radius], normalize = True)
            h_res = h_res_[0]
            a_, x_, y_, r_ = hough_circle_peaks(hspaces = [h_res],
                                                radii = [h_radius],
                                                threshold = 0.4, #HOUGH_THRESH,
                                                num_peaks = 1,
                                                total_num_peaks = 1,
                                                normalize=True)
            #print(x_, y_)
            if x_.size > 0:
                cy = y_.item()
                cx = x_.item()
                
                new_pimage = yi[ (cy-bbox):(cy+bbox), (cx-bbox):(cx+bbox) ]
                new_stddev = np.std(new_pimage)
                new_mean   = np.mean(new_pimage)
                if math.isnan(new_stddev):
                    new_stddev = 0
                    #new_mean   = 0
                    new_pimage = pimage
                #print(" >0", new_mean, new_stddev, "(",cx,cy,")", "(",wx,wy,")")
                
                #print(new_stddev)
            else: # if a circle wasnt found then just use the image and stddev from the center of the yolo image
                new_pimage = pimage
                new_stddev = 0#stddev
                #print("==0", new_stddev)
                #print("no bueno")
            
            
            # construct new particle from data
            new_particle = Particle(px, py, wx, wy, f, pID, yoloimage, pimage, stddev, mean, conf, new_pimage, new_stddev)
            
            # Check to see if we need to update an old particle or add a new particle
            add_new_particle_to_list = False
            #if the video particle_list is empty, add the first particle to it
            if len(video_particle_list) == 0: add_new_particle_to_list = True
                
            #for all other particles, look for a matching particle in the list
            #   if a matching particle was found: update the video particle list
            #   otherwise: create a new particle
            else: #look for a matching particle
                matchID = look_for_matching_previous_particle(new_particle, video_particle_list, constants, f)
                #matchID = look_for_matching_previous_particle(new_particle, video_particle_list, constants, f, pID)
                
                #matchID = False #uncheck this to see all frames as new particles
                if matchID: video_particle_list[matchID-1].updateParticle(new_particle)
                else: add_new_particle_to_list = True
                
            
            #add_new_particle_to_list = True
            if add_new_particle_to_list:
                video_particle_list.append(new_particle)
                #print(len(video_particle_list), pID-1)
                #video_particle_list[pID-1] = new_particle
                pID += 1
        
        t5 = time.time()
        
        # measure the minimum contrast for this framea nd append it to the list
        noise_floor_list[f] = get_noise(i16, constants)
            
        
        t6 = time.time()        
        
        
        particles_found = len(video_particle_list)
        ratio_time = t2-t1
        yolo_time  = t3-t2
        clean_time = t4-t3
        merge_time = t5-t4
        noise_time = t6-t5
        if TRACEMEMORY: memory_usage = tracemalloc.get_traced_memory()
        else: memory_usage = [0,0]
        
        
        #print("Current: %d, Peak %d" % tracemalloc.get_traced_memory())
        #print(type(tracemalloc.get_traced_memory()))
        

        timestamp_array[f] = [particles_found, ratio_time, yolo_time, clean_time, merge_time, noise_time, memory_usage]
        


    # export timestamp array as .csv
    export_timestamp_array(timestamp_array, constants)


    end = time.time()
    print("RATIOMETRIC PARTICLE FINDER SPEED REPORT:")
    print("\t Resolution: \t\t\t\t\t", x, y, "px")
    print("\t Total Pixels per frame: \t\t", (x*y), "px")
    print("\t New number of frames: \t\t\t", n, "frames")
    print("\t Elapsed time: \t\t\t\t\t", (end-start), " seconds")
    print("\t Elapsed time: \t\t\t\t\t", ((end-start)/60), " minutes")
    #print("\t Total Number of Particles: \t", pID, "particles")
    print("\t Speed (n-fps): \t\t\t\t", (n / (end-start)), " finished frames / sec" )
    

    return v8, video_particle_list, results_list, noise_floor_list




# this is called from ratiometric particle finder
# this function will compare a newly found particle to a list of particles
# if it finds that its the same as a particle found on a previous frame, then
# it will return the matching particles ID number. 
# if it doesnt find a matching particle then it must be a new particle. in this
# case it returns False
def look_for_matching_previous_particle(new_particle, particle_list, constants, f):
    # def look_for_matching_previous_particle(new_particle, particle_list_in, constants, f, pID):
    # particle_list = particle_list_in[:pID-1] (#only look through the amount that we've filled in so far)
    
    # Define Constants    

    temporal_tolerance     = constants['temporal tolerance']
    spatial_tolerance = constants['spatial tolerance']
                                                                                ####    IMPORTANT: ONLY LOOK THROUGH THE LAST 1000 OR
       
    matched_pID            = False       # By default, the Particle is not found on the list
                                                            
    # new particle data
    px = new_particle.px_vec[0]       # particle x position
    py = new_particle.py_vec[0]       # particle y position
    pf = new_particle.f_vec[0]       # current frame number
    
    
    # check to see if this particle existed in a previous frame
    # if you've looked back in time longer than (temporal_tolerance)
    # number of frames, then just quit looking
    
    max_l = 100   #max number of previous particles to look through 
    matched_pID_list = []
    if len(particle_list) > max_l: particle_list = particle_list[-max_l:] # if the particle list is longer than the max number of particle to look through, then just look at the last 'max_l' number of particles
    
    # browse the particle list
    for i, lpart in enumerate(reversed(particle_list)): #reversed(particle_list):
        
        # list particle data
        lpx  = lpart.px_vec[-1]
        lpy  = lpart.py_vec[-1]
        lpf  = lpart.f_vec[-1]

        
        # if the last frame the list particle was seen was over 100 frames ago, then just give up
        # this works because we look through the list backwards
        if f - lpf > 100: break
        
        #dbbox = psq+lpsq
        dp = dist((px,py),(lpx,lpy))
        #print(dp, dbbox, pf, lpf, temporal_tolerance)
        #if dp < spatial_tolerance and pf - lpf < temporal_tolerance and pf-lpf > 0:
        #    matched_pID = lpart.pID 
    
        # if the distance between the particles is small
        # and the number of frames between the particles is small
        # and the number of frames between the particles is not zero (not on same frame)
        # THEN, we have found a match
        if dp < spatial_tolerance and pf - lpf < temporal_tolerance and pf-lpf > 0:
            matched_pID = lpart.pID 
            
            
            ''' 
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                RISKY DECISIONS
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            right now i think it keeps going through the list and it 
            just happens to return the last one it finds. this isnt great
            because the last one it finds might not be the right one. instead,
            I should have it add each possible match to a list, then go through
            the list and pick the particle with the greatest standard deviation
            
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            '''
            
            
            matched_pID_list.append(lpart.pID)
            #matched_std_list.append(lpart.pimagestd_vec[-1])
            #print(f, lpart.pID, lpart.pimagestd_vec[-1])
    #print(f, len(particle_list), i) #print frame number, particle list size, number of iterations performed before exit
    if matched_pID is not False:
        #print(np.argmax(matched_std_list))
        #matched_pID = matched_pID_list[np.argmax(matched_std_list)]
        matched_pID = matched_pID_list[0] #for now, just give the first close pID it finds
    
    #returns either False or the matched pID
    return matched_pID





# this function takes in a a 16 bit image from a ratiometric video and
# randomly samples a particle sized sample from a blank frame and measures
# its standard deviation. it will do this multiple times from random places
# in the image then return the lowest value. This function serves to measure
# the noise floor of a particle contrast measurement
def get_noise(img16, constants):
    
    nsamples = 20
    if len(img16.shape) < 2: return 1
    x, y = img16.shape
    l = constants["bbox"]*2
    #l = 30
    
    #whole_std = np.std(img)
    
    min_std = 1
    for s in range(nsamples):
        bx, by = random.randint(50, x-l-1), random.randint(0, y-l-1)
        #print(x, y, bx, by)
    
        sample_img = img16[ (by-l):(by+l), (bx-l):(bx+l) ]
        sample_std = np.std(sample_img)
    
        #print("\t", s, sample_std)
        if sample_std < min_std: min_std = sample_std
    
    #print("\t\t minimum std: ", min_std)
    return min_std




'''
###############################################################################
###############################################################################
###############################################################################

   _____                                __       __                 __       
  / ___/ ____   _____ ___   ____ _ ____/ /_____ / /_   ___   ___   / /_ _____
  \__ \ / __ \ / ___// _ \ / __ `// __  // ___// __ \ / _ \ / _ \ / __// ___/
 ___/ // /_/ // /   /  __// /_/ // /_/ /(__  )/ / / //  __//  __// /_ (__  ) 
/____// .___//_/    \___/ \__,_/ \__,_//____//_/ /_/ \___/ \___/ \__//____/  
     /_/                                                                  
     

###############################################################################
###############################################################################
###############################################################################
'''#!!!



# save a list of elapsed time for each process. this is supposed to be a way
# to find ineffiecient processes and help decide what to improve for speed
# this function is called from the ratiometric particle finder
def export_timestamp_array(timestamp_arr, constants):
    #generate empty dataframes for .csv files
    df1 = pd.DataFrame() 
    
    #print("\nMaking Particle Library Spreadsheets .csv file...") #create files
    csv_filename  = os.path.join(constants["output path"], ("particle finder speed.csv"))

    nframes = len(timestamp_arr)
    
    #generate empty lists to store 
    particles_found  = np.zeros(nframes)
    ratio_time      = np.zeros(nframes)
    yolo_time       = np.zeros(nframes)
    clean_time      = np.zeros(nframes)
    merge_time      = np.zeros(nframes)
    noise_time      = np.zeros(nframes)
    memory_usage_c  = np.zeros(nframes)
    memory_usage_p  = np.zeros(nframes)
    
    


    # iterate through the timestamp_list
    for i, data in enumerate(timestamp_arr):
             
        #assign timstamp data to fill in a row of the .csv
        particles_found[i]   = data[0]
        ratio_time[i]        = data[1]
        yolo_time[i]         = data[2]
        clean_time[i]        = data[3]
        merge_time[i]        = data[4]
        noise_time[i]        = data[5]
        mem = data[6]
        memory_usage_c[i]     = mem[0]
        memory_usage_p[i]     = mem[1]
        

    # Particle ID    
    df1["nParticles"]      = particles_found
    df1["ratio"]           = ratio_time
    df1["yolo"]            = yolo_time 
    df1["clean"]           = clean_time
    df1["merge"]           = merge_time
    df1["noise"]           = noise_time
    df1["current memory"]  = memory_usage_c
    df1["peak memory"]     = memory_usage_p
    

    # Lifetime Info
    df1.to_csv(csv_filename, index=True)

# this generates the main spreadsheet of particle info
def generate_particle_list_csv(particle_list, constants, tag):    
    
    name = constants["name"]
    #generate empty dataframes for .csv files
    df1 = pd.DataFrame() 
    
    print("Making Particle Library Spreadsheets .csv file...") #create files
    csv_filename  = os.path.join(constants["output path"], ("Particle List__"+name+tag+".csv"))

    
    #generate empty lists to store 
    pIDs                   = np.zeros(len(particle_list))
    
    # Lifetime Info
    f_0                    = np.zeros(len(particle_list))
    lifetimes              = np.zeros(len(particle_list))
    frames                 = [0]*(len(particle_list))
    confidence             = [0]*(len(particle_list))
    confidence_low         = np.zeros(len(particle_list))
    confidence_high        = np.zeros(len(particle_list))
    confidence_average     = np.zeros(len(particle_list))
    
    
    # Location Info
    px_vec                  = [0]*(len(particle_list))
    py_vec                  = [0]*(len(particle_list))
    px_0                    = np.zeros(len(particle_list))
    py_0                    = np.zeros(len(particle_list))
    
    
    # bbox info
    wx_vec                  = [0]*(len(particle_list))
    wy_vec                  = [0]*(len(particle_list))
    wx_0                    = np.zeros(len(particle_list))
    wy_0                    = np.zeros(len(particle_list))

    # contrast info    
    ones_counter             = np.zeros(len(particle_list))
    darkest_pixel            = np.zeros(len(particle_list))
    average_darkest_pixels   = np.zeros(len(particle_list))
    brightest_pixel          = np.zeros(len(particle_list))
    average_brightest_pixels = np.zeros(len(particle_list))
    
    mean_vec                = [0]*(len(particle_list))
    mean_mean               = np.zeros(len(particle_list))
    stddev_vec              = [0]*(len(particle_list))
    stddev_max              = np.zeros(len(particle_list))

    new_stddev_max          = np.zeros(len(particle_list))


    # iterate through the particle list
    for c, p in enumerate(particle_list):
             
        #assign particle data to fill in a row of the .csv
        pIDs[c]                = p.pID


        # Lifetime Info
        f_0[c]                 = p.f_vec[0]
        lifetimes[c]           = len(p.f_vec)
        frames[c]              = p.f_vec
        
        #confidence info
        confidence[c]             = p.conf_vec
        confidence_low[c]         = np.min(p.conf_vec)
        confidence_high[c]        = np.max(p.conf_vec)
        confidence_average[c]     = np.average(p.conf_vec)
        
        # Location Info
        px_0[c]                  = p.px_vec[0]
        py_0[c]                  = p.py_vec[0]
        px_vec[c]                = p.px_vec
        py_vec[c]                = p.py_vec
        
        
        wx_0[c]                  = p.wx_vec[0]
        wy_0[c]                  = p.wy_vec[0]
        wx_vec[c]                = p.wx_vec
        wy_vec[c]                = p.wy_vec
        
        
        # set up some stuff to calculate darkest and brightest pixels
        #https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        k = 4 # choose 4 darkest or lightest values
        center_frame = np.argmax(p.stddev_vec)
        center_image = p.pimage_vec[center_frame]
        d_idx = np.argpartition(center_image, k)
        b_idx = np.argpartition(center_image, -k)
        
        #calculate darkest and brightest pixels for the most contrasty image for each particle
        ones_counter                = (center_image == 1).sum() # counts the number of values equal to 1.0
        #print(center_image)
        darkest_pixel[c]            = np.min(center_image)
        average_darkest_pixels[c]   = np.average(center_image[d_idx[:k]])
        
        brightest_pixel[c]          = np.max(center_image)
        average_brightest_pixels[c] = np.average(center_image[b_idx[:-k]]) 
                
        mean_mean[c]             = np.mean(p.mean_vec)
        mean_vec[c]              = p.mean_vec
        stddev_max[c]            = np.max(p.stddev_vec)
        stddev_vec[c]            = p.stddev_vec
        
        new_stddev_max[c]        = np.max(p.new_stddev_vec)
     
    # Particle ID    
    df1["pID"]                 = pIDs
    df1["new stddev"]          = new_stddev_max

    # Lifetime Info
    df1["f0"]                  = f_0
    df1["lifetime"]            = lifetimes
    df1["frames"]              = frames
    
    # Confidence Info
    df1["Conf"]            = confidence
    df1["Conf low"]            = confidence_low
    df1["Conf high"]            = confidence_high
    df1["Conf avg"]            = confidence_average
    
    # Position Info
    df1["px0"]                  = px_0
    df1["py0"]                  = py_0
    df1["px list"]              = px_vec
    df1["py list"]              = py_vec

    # bbox Info
    df1["wx0"]                  = wx_0
    df1["wy0"]                  = wy_0
    df1["wx list"]              = wx_vec
    df1["wy list"]              = wy_vec
    
    #sd contrast info
    df1["ones counter"]         = ones_counter
    df1["dark pxl"]             = darkest_pixel
    df1["avg dark pxl"]         = average_darkest_pixels
    df1["bright pxl"]           = brightest_pixel
    df1["avg bright pxl"]       = average_brightest_pixels
    
    df1["mean mean"]            = mean_mean
    df1["mean vec"]             = mean_vec
    df1["stddev max"]           = stddev_max
    df1["stddev list"]          = stddev_vec

    df1.to_csv(csv_filename, index=True)
    #print("\t Particles list saved as: ", csv_filename)
    
    
    
def generate_sdcontrast_csv(pl_in, constants, tag):#cnn_pl, constants, "-cnn")
    print("\nExporting sd-contrast .csv file...")
    max_contrast = []
    for p in pl_in:
        max_contrast.append(np.max(p.stddev_vec))
    filename_out = os.path.join(constants['output path'], (tag+"__sd-contrast.csv"))
    np.savetxt(filename_out, 
           max_contrast,
           delimiter =", ", 
           fmt ='% s')

def generate_noisefloor_csv(noise, constants):#cnn_pl, constants, "-cnn")
    print("exporting noise floor .csv file...")
    filename_out = os.path.join(constants['output path'], "__noisefloor.csv")
    np.savetxt(filename_out, 
           noise,
           delimiter =", ", 
           fmt ='% s')

def generate_landing_rate_csv(particle_list, constants, tag):
    print("Making .csv file for Landing Rate...")
    n        = constants["nframes"]
    #basepath = constants["basepath"]
    fps      = constants['fps']
    # particles per frame, list
    ppf = np.zeros(n)
    c = 0                                                       # initialize a total particle counter
    for i, f in enumerate(ppf):                                 #loop through each frame of the video     
        pids = [p.pID for p in particle_list if p.f_vec[0] == i]   # returns a list of particle ID's (which are equivalent to particle counts)
                                                                # for all particles that first landed on this particular frame
        if pids: c = pids[-1]                                   # if a list of landed particles was created, then use the largest pID
                                                                # (the last particle on the list), as the new total particle count
        ppf[i] = c                                              # set the particles per frame to be the total number of particles found so far

    # seconds per frame list
    # this converts the x axis from frames to seconds
    spf = np.linspace(0, (n/fps), n)
    #save original particle landing rate data
    csv_filename = os.path.join(constants["output path"], ("Landing Rate__"+ constants["name"]+ tag +".csv"))    
    np.savetxt(csv_filename, np.transpose([spf, ppf]), delimiter=',')
    #print("\t landing rate csv saved as: ", csv_filename)
    


'''
###############################################################################
###############################################################################
###############################################################################
                    _    __ _      __           
                   | |  / /(_)____/ /___   ____ 
                   | | / // // __  // _ \ / __ \
                   | |/ // // /_/ //  __// /_/ /
                   |___//_/ \__,_/ \___/ \____/ 
                       ______ _  __             
                      / ____/(_)/ /___   _____  
                     / /_   / // // _ \ / ___/  
                    / __/  / // //  __/(__  )   
                   /_/    /_//_/ \___//____/                               

###############################################################################
###############################################################################
###############################################################################
'''#!!!

def save_si_video2(r8_in, particle_list_in, constants, tag, offset=0, print_fnum=False, print_v=False):
    
    if not constants["voltfile"]: return
    voltpath = os.path.join(constants["basepath"], (constants["timestamp"] + "voltage.txt")) 
    #print(voltpath)
    voltdata = pd.read_csv(voltpath, header=None, names=["volts", "nan"])
    #print(voltdata.iloc[1, 0])
  
    # initialize useful variables
    n, x, y = r8_in.shape
    #video_RGB = []
    #EDGE = 15
    EDGE = constants['bbox']
    font = PIL.ImageFont.truetype("arial.ttf", 20)


    #print some info
    #print("Total raw video frames:   ", constants["frame total"])
    print("Total raw video frames:   ", constants["nframes"])
    print("Total volt log frames:    ", len(voltdata["volts"]))
    print("Total ratio video frames: ", n)
    offset2 = constants["bufsize"]*2 #this offset accounts for the frame loss due to ratiometric buffer
    print("ratio bufer offset: ", offset2)
        # VIDEO DRAWING

    #first, convert video to RGB
    # print("Converting grayscale video to RGB...")
    # for c, f in enumerate(r8_in):
    #     rgb_image = color.gray2rgb(r8_in[c])
    #     video_RGB.append(rgb_image)
    # video_out = np.array(video_RGB)
    
    video_out = []
    
    # define colos
    tR, tG, tB = 220, 220, 220
    #bR, bG, bB = 221,  28, 200
    bB, bG, bR = 0, 89, 170
    vB, vG, vR = 253, 142, 124
    
    # loop through each frame drawing frame numbers on each one
    # also draw a rectantular bounding box that defines the particle images
    # and also the particle edge cut-off distance
    #print(video_out.shape)
    
    # voltage trace positions
    txp_    = 8                 # trace x position
    span_   = 196                # length of trace (in frames)
    
    typ    = 200  #180              # trace y position
    typ_0    = 200                # trace y position (at zero volts i think)
    
    tyh    = 30#50                 # trace height (this is essentially a multiplier for the voltage)
    typ_m = 30               #verticle type spacing multiplier
    
    typ_vo = -10 #type verticle offset
    
    n_tot = constants["nframes"]                                         # total number of frames
    txp = txp_+span_
    span = 1  
    
    print("Drawing Framenumbers and voltage trace on video...")
    #for fnum in range(len(video_out)):
    
    for fnum in tqdm.tqdm(range(n)):
        # initialize a pillow image and a drawing object for that iamge
        #pillowImage = PIL.Image.fromarray(video_out[fnum])
        #pillowImage = PIL.Image.fromarray(np.array(color.gray2rgb(r8_in[fnum])))
        #pillowImage = PIL.Image.fromarray(color.gray2rgb(r8_in[fnum]))
        
        rgb_image   = color.gray2rgb(r8_in[fnum])
        pillowImage = PIL.Image.fromarray(rgb_image)
        draw        = PIL.ImageDraw.Draw(pillowImage)
        
        '''draw frame numbers'''
        if print_fnum:
            draw.text( (2,2), str(fnum), (tR, tG, tB), font=font)
        
        
        '''print current volts on screen'''
        if print_v:
            volts = voltdata.iloc[(fnum+offset+offset2), 0]
            draw.text((253,2),
                      (str(volts) + " V"),
                      (vR, vG, vB),
                      font=font,
                      anchor="rt",
                      align="right")
        
        #print(fnum, fnum+offset-span-offset2, span, n_tot, span+offset2)
        #if fnum >= span+offset2 and fnum + offset + span + offset2 < n_tot:                         # you will get array errors if you run this on the whole array so this makes sure that doesnt happen

        if fnum + offset + span + offset2 < n_tot:                         # you will get array errors if you run this on the whole array so this makes sure that doesnt happen
            #this accounts for the first few frames when the frame number is less than the voltage trace    
            if txp>txp_: txp-=1
            if span<span_: span+=1
            #print(txp, span)
            
            ''' get data for voltage trace of (span) number of frames '''
            trace = np.array(voltdata.iloc[fnum+offset-span+offset2:(fnum+offset+offset2), 0])   # trace is a snippit of the voltage frame array. trace is what we print on the screen 
            #print(fnum, len(trace), trace[0], trace[-1])
            #print("trace: ", trace)

            ''' Draw the trace onto the frame '''
            last_v = trace[0]      
            # the method works by drawing a line from point n to point n+1. this makes sure the starting point for the line is the same as the first point it will draw to
            for vi, vv in enumerate(trace):                                      # Loop through the whole trace
                ''' draw the voltage trace '''
                for h in range(3):
                    draw.point(( txp+vi, typ-(tyh*vv)+h   ), fill=(vR, vG, vB))      #
                
                ''' draw a tick mark every 200 voltages'''
                voltage_position = fnum+offset+vi
                if voltage_position % 200 == 0 and span==span_:      
                    #print(fnum, voltage_position)
                    for h in range(7):
                        for t in range(3):
                            draw.point(( txp+vi+t, typ-(tyh*vv)-h ), fill=(vR, vG, vB))
                            #draw.point(( txp+vi+t, typ+(tyh*(1-vv))-h ), fill=(vR, vG, vB))
                
                ''' this draws the verticle line that connects when the voltage switches'''
                if vv != last_v:                                                 
                    lx1, ly1 = txp+vi, typ-(tyh*vv)       #get line starting point
                    lx2, ly2 = txp+vi, typ-(tyh*last_v)   #get line ending point
                    draw.line([(lx1, ly1), (lx2, ly2)], fill=(vR, vG, vB), width=3)  #draw line
                last_v = vv 
 
            
            ''' draw text on frame '''
            draw.text( (250,typ_0+typ_vo-30), "1V", (vR, vG, vB), font=font, anchor="rt", align="right")
            draw.text( (250,typ_0+typ_vo), "0V",  (vR, vG, vB), font=font, anchor="rt", align="right")
            draw.text( (250,typ_0+typ_vo+30), "-1V",  (vR, vG, vB), font=font, anchor="rt", align="right")
            
        
        #draw frame boundary 
        #draw.rectangle((EDGE,EDGE,x-EDGE,y-EDGE), fill=None, outline=(240,240,240))
        #Convert the frame to a numpy array
        #video_out[fnum] = np.array(pillowImage, np.uint8)
        video_out.append(np.array(pillowImage, np.uint8))
        # if fnum >= 10000:
        #     n = 10000
        #     break
    
    video_out = np.array(video_out)
    
    
    
    
    # print("Drawing Particles on video...")
    # #draw each particle on the frames that it exists on
    # # for this one it makes the most sense to loop through the by particle rather than by frame
    # for p in tqdm.tqdm(particle_list_in): #go through list of particles
    #     #print("Particle: ", p.pID, " found in frames: ", p.f_vec)
        
    #     for i, fnum in enumerate(p.f_vec): #for this particle, go through the frame vector and draw the particle bounding box on the frame
            
    #         # initialize a pillow image and a drawing object for that iamge
    #         pillowImage = PIL.Image.fromarray(video_out[fnum])
    #         draw = PIL.ImageDraw.Draw(pillowImage)
        
    #         px = p.px_vec[i]
    #         py = p.py_vec[i]
    #         wx = p.wx_vec[i]
    #         wy = p.wy_vec[i]
            
    #         # draw a box around each particle ( x1, y1,  x2,  y2 )
    #         draw.rectangle((px-wx, py-wy, px+wx, py+wy), fill=None, outline=(bR, bG, bB))
            
    #         font = PIL.ImageFont.truetype("arial.ttf", 32)
    #         xloc, yloc = px, py #p.x_vec[cc], p.y_vec[cc]
    #         if xloc > x - 2*EDGE: xloc -= EDGE
    #         if yloc > y - 2*EDGE: yloc -= EDGE
    #         #draw.text( (xloc, yloc), str(p.pID), (tR, tG, tB), font=font) #color was 220, 20, 220
            
    #         image_copy = np.array(pillowImage, np.uint8)
            
    #         video_out[fnum] = image_copy
    
    
    
    # VIDEO SAVING
    # Generate Filename
    name = constants["name"]
    filename = name + tag + "-color.avi"
    save_file_path = os.path.join(constants["output path"], filename)

    # Write and save video
    print("Saving Yolo Particle Video to:   ", save_file_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
                        #use a different 4-byte code for other codecs
                        #https://www.fourcc.org/codecs.php
    videoObject = cv2.VideoWriter(save_file_path, fourcc, constants["output framerate"], (y, x), isColor = True)
    for c in range(n):
        videoObject.write(video_out[c])
    videoObject.release()
    return 



def save_particle_list_video(r8_in, particle_list_in, constants, tag):
    
    
    # initialize useful variables
    n, x, y = r8_in.shape
    video_RGB = []
    #EDGE = 15
    EDGE = constants['bbox']

        # VIDEO DRAWING

    #first, convert video to RGB
    print("Converting grayscale video to RGB...")
    for c, f in enumerate(r8_in):
        rgb_image = color.gray2rgb(r8_in[c])
        video_RGB.append(rgb_image)
    video_out = np.array(video_RGB)
    
    # define colos
    tR, tG, tB = 220, 220, 220
    bR, bG, bB = 221,  28, 200
    
    # loop through each frame drawing frame numbers on each one
    # also draw a rectantular bounding box that defines the particle images
    # and also the particle edge cut-off distance
    #print(video_out.shape)
    print("Drawing Framenumbers and Particle Bounding boxed on video...")
    for fnum in range(len(video_out)):
        # initialize a pillow image and a drawing object for that iamge
        pillowImage = PIL.Image.fromarray(video_out[fnum])
        draw = PIL.ImageDraw.Draw(pillowImage)
        
        #draw frame number
        font = PIL.ImageFont.truetype("arial.ttf", 32)
        draw.text( (2,2), str(fnum), (tR, tG, tB), font=font)
        
        #draw frame boundary 
        draw.rectangle((EDGE,EDGE,x-EDGE,y-EDGE), fill=None, outline=(240,240,240))
        #Convert the frame to a numpy array
        video_out[fnum] = np.array(pillowImage, np.uint8)
        
 
    
    print("Drawing Particles on video...")

    #draw each particle on the frames that it exists on
    # for this one it makes the most sense to loop through the by particle rather than by frame
    for p in particle_list_in: #go through list of particles
        #print("Particle: ", p.pID, " found in frames: ", p.f_vec)
        
        for i, fnum in enumerate(p.f_vec): #for this particle, go through the frame vector and draw the particle bounding box on the frame
            
            # initialize a pillow image and a drawing object for that iamge
            pillowImage = PIL.Image.fromarray(video_out[fnum])
            draw = PIL.ImageDraw.Draw(pillowImage)
        
            px = p.px_vec[i]
            py = p.py_vec[i]
            wx = p.wx_vec[i]
            wy = p.wy_vec[i]
            
            # draw a box around each particle ( x1, y1,  x2,  y2 )
            draw.rectangle((px-wx, py-wy, px+wx, py+wy), fill=None, outline=(bR, bG, bB))
            
            font = PIL.ImageFont.truetype("arial.ttf", 32)
            xloc, yloc = px, py #p.x_vec[cc], p.y_vec[cc]
            if xloc > x - 2*EDGE: xloc -= EDGE
            if yloc > y - 2*EDGE: yloc -= EDGE
            draw.text( (xloc, yloc), str(p.pID), (tR, tG, tB), font=font) #color was 220, 20, 220
            
            image_copy = np.array(pillowImage, np.uint8)
            
            video_out[fnum] = image_copy
    
    # VIDEO SAVING
    # Generate Filename
    name = constants["name"]
    filename = name + tag + "-color.avi"
    save_file_path = os.path.join(constants["output path"], filename)

    # Write and save video
    print("Saving Yolo Particle Video to:   ", save_file_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
                        #use a different 4-byte code for other codecs
                        #https://www.fourcc.org/codecs.php
    videoObject = cv2.VideoWriter(save_file_path, fourcc, constants["output framerate"], (y, x), isColor = True)
    for c in range(n):
        videoObject.write(video_out[c])
    videoObject.release()
    return 
    


#this is for debugging purposes only
# def save_yolo_results_video(r8_in, results_list, constants, tag):
#     # draw results on video
#     n, x, y = r8_in.shape
#     video_RGB = []

#     #first, convert video to RGB
#     print("Converting grayscale video to RGB...")
#     for c, f in enumerate(r8_in):
#         rgb_image = color.gray2rgb(r8_in[c])
#         video_RGB.append(rgb_image)
#     video_out = np.array(video_RGB)
    
#     # define colos
#     tR, tG, tB = 220, 220, 220
#     bR, bG, bB = 221,  28, 50
    
#     #print(video_out.shape)
#     print("Drawing Framenumbers and Particle Bounding boxed on video...")
#     for fnum in range(len(video_out)):
#         # initialize a pillow image and a drawing object for that iamge
#         pillowImage = PIL.Image.fromarray(video_out[fnum])
#         draw = PIL.ImageDraw.Draw(pillowImage)
        
#         #draw frame number
#         font = PIL.ImageFont.truetype("arial.ttf", 32)
#         draw.text( (2,2), str(fnum), (tR, tG, tB), font=font)
        
#         #Draw Particle Bounding Boxes
#         frame_results = results_list[fnum]
#         pl = frame_results[0].boxes.xywh
#         #print(len(pl))
#         for p in pl: 
#             px = int(p[0])
#             py = int(p[1])
#             wx = int(p[2])
#             wy = int(p[3])
#             # draw a box around each particle ( x1, y1,  x2,  y2 )
#             draw.rectangle((px-wx, py-wy, px+wx, py+wy), fill=None, outline=(bR, bG, bB))
            
#         #Convert the frame to a numpy array
#         video_out[fnum] = np.array(pillowImage, np.uint8)
     
#     # Generate Filename
#     #n, x, y, colors = rgb_video.shape
#     name = constants["name"]
#     filename = name + tag + "-color.avi"
#     save_file_path = os.path.join(constants["output path"], filename)

#     # Write and save video
#     print("Saving Yolo Particle Video to:   ", save_file_path)
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#     #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     #fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
#                         #use a different 4-byte code for other codecs
#                         #https://www.fourcc.org/codecs.php
#     videoObject = cv2.VideoWriter(save_file_path, fourcc, constants["output framerate"], (y, x), isColor = True)
#     for c in range(n):
#         videoObject.write(video_out[c])
#     videoObject.release()
#     return 
  
    
def save_bw_video(images, constants, tag, print_frame_nums = False):
    print("Saving Video...")
    filename = constants["name"] + "_"+str(constants["output framerate"])+"fps"+tag + ".mp4"
                                           
    save_file_path = os.path.join(constants["output path"], filename)
    n, x, y = images.shape
    print(n,x,y)
    #fourcc = cv2.VideoWriter_fourcc(*'MPG4V')
    fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
                        #use a different 4-byte code for other codecs
                        #https://www.fourcc.org/codecs.php
    video = cv2.VideoWriter(save_file_path, fourcc, constants["output framerate"], (y, x), 0)
    font = PIL.ImageFont.truetype("arial.ttf", 32)
    for f in range(len(images)):
        # create a PIL image from the current frame
        pillowImage = PIL.Image.fromarray(images[f])
        draw = PIL.ImageDraw.Draw(pillowImage)
        # if you'd like, draw frame numbers on the image
        if print_frame_nums: draw.text( (2,2), str(f), 255, font=font)
        # write the frame to the video
        video.write(np.array(pillowImage, np.uint8))
    video.release()
    print("Video Saved as: \t", save_file_path)  
  

    
'''
###############################################################################
###############################################################################
###############################################################################
        ____   __        __     ____                                    
       / __ \ / /____   / /_   /  _/____ ___   ____ _ ____ _ ___   _____
      / /_/ // // __ \ / __/   / / / __ `__ \ / __ `// __ `// _ \ / ___/
     / ____// // /_/ // /_   _/ / / / / / / // /_/ // /_/ //  __/(__  ) 
    /_/    /_/ \____/ \__/  /___//_/ /_/ /_/ \__,_/ \__, / \___//____/  
                                                   /____/               
                                                                    
###############################################################################
###############################################################################
###############################################################################
'''#!!!

# def save_all_particle_images(pl, constants):
    
#     for p in pl:
#         print(len(p.yoloimage_vec))
              
    

def plot_waterfall(pl, constants, tag):
    print("Drawing Landing Rate vs Potential Plot...")
    
    # some helpful definitions: particles per frame, list
    nframes = constants["nframes"]
    fps     = constants['fps']
    name    = constants["name"]
      

    if constants["voltfile"]:
        # # generate voltage data from file, trim if necessary
        nsplit = constants["name"].split("_")
        voltage_file = os.path.join(constants["basepath"], (nsplit[0] +"_"+nsplit[1]+"_voltage.txt") )
        vdata = np.loadtxt(voltage_file, dtype=str, delimiter=',') 
        vpf = np.array([x[0] for x in vdata]).astype(float)
        if len(vpf) > nframes: vpf = vpf[:nframes]
     
  
    # Generate waterfall data
    frame_nums = [p.f_vec[0] for p in pl]             # first frame a particle was seen
    contrast   = [np.max(p.stddev_vec) for p in pl]   # maximum contrast of that particle
    point_size = [c*500 for c in contrast]
    
    
    plot_sizes = ["x-small", "small", "medium", "large"]
    for size in plot_sizes:
        if size == "x-small":  crange = [0, 0.02]
        if size == "small":    crange = [0, 0.05]
        if size == "medium":   crange = [0, 0.10]
        if size == "large":    crange = [0, 0.50]
       
        # set up the plot
        fig, ax1 = plt.subplots()
        #plt.rcParams['figure.figsize'] = [8, 8]
        #plt.figure(dpi=150)        
        ax2 = ax1.twinx()
         
        # Generate waterfall plot
        ax1.scatter(frame_nums, contrast, s=point_size, color='steelblue', marker='.', linewidths = 0.1, edgecolor="grey",  alpha=0.9)
         
        #plot the voltage data
        if constants["voltfile"]:
            #ax2.scatter(spf, vpf, color='violet', alpha=0.5, s=2)
            ax2.plot(vpf, color='violet', linewidth=3, alpha=0.2)
            ax2.set_ylabel('Applied Potential /Volt')
            ax2.set_ylim((-2,2)) 
            ax1.set_title(constants["sample name"])
         
        # labels and stuff
        ax1.set_xlabel('time, Frame number')
        ax1.set_ylabel('s.d. contrast')
        ax1.set_ylim(crange)
        
        # Save the figure
        filename = "Waterfall " + name + tag + size + ".png"
        save_file_path = os.path.join(constants["output path"], filename)
        print(save_file_path)
        plt.savefig(save_file_path, dpi=150)
        #plt.show()
        plt.clf
        plt.close('all')
        


    
def plot_landing_map(particle_list, constants, tag):
    print("Drawing Particle Landing Map...")
    name = constants["name"]
    mpp = constants["fov"] / constants["video x dim"]
    sample_name = constants["sample name"]
    #basepath = constants["basepath"]
    #n = len(particle_list)
    #x, y = np.zeros(n), np.zeros(n)
    #allx, ally = [], [] #get data into the right shape
    #for c in range(len(particle_list)):
    #    allx += particle_list[c].x_vec
    #    ally += particle_list[c].y_vec
    s_factor=1000
    
    x = [p.px_vec[0]*mpp for p in particle_list]
    y = [p.py_vec[0]*mpp for p in particle_list]
    c = [np.max(p.stddev_vec)*s_factor for p in particle_list]
    #print(c)
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.figure(dpi=150)
    plt.scatter(x, y, s=c, alpha=0.5) # plot data
    plt.xlabel('microns')
    plt.ylabel('microns')
    title = "Particle Landing Locations, " + sample_name
    plt.title(title)
    
    # generate filepath and save figure as .png
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - particle map.png"
    filename = "Particle Map " + name + tag + ".png"
    save_file_path = os.path.join(constants["output path"], filename)
    plt.savefig(save_file_path)
    #plt.show()
    plt.clf()
    plt.close('all')


  
def plot_sdcontrast_hist(particle_list, constants, tag):
    if len(particle_list) == 0: return
    name = constants["name"]
    #basepath = constants["basepath"]
    #sample_name = constants["sample name"]
    #if not os.path.exists(os.path.join(basepath, "output")): os.makedirs(os.path.join(basepath, "output"))
    
    # generate a list of particle contrasts
    contrasts = np.ones_like(particle_list)
    for i, p in enumerate(particle_list):
        #print(p.peak_contrastDoG)
        #contrasts[i] = np.min(p.drkpxl_avg_vec) #p.peak_contrastDoG 
        contrasts[i] = np.max(p.stddev_vec)
        #print(contrasts[i])
    
    print("\n")
    #plot the contrasts as a histogram, plot using multiple sizes of x axis
    sizes = ["x-small", "small", "medium", "large"]
    for size in sizes:
        if size == "x-small":  xrange, bins = [0, 0.02], 50
        if size == "small":    xrange, bins = [0, 0.05], 50
        if size == "medium":   xrange, bins = [0, 0.10], 50
        if size == "large":    xrange, bins = [0, 0.50], 50
        image_filename  = os.path.join(constants["output path"], ("Contrast Histogram " + name + size + tag + ".png"))
        
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.figure(dpi=150)
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 18}
        SMALL_SIZE = 12
        MEDIUM_SIZE=18
        BIGGER_SIZE=18
        #matplotlib.rc('font', **font)
        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE) 
    
        alpha = 0.8
        #bins = 100
    
        
        n, bins, patches = plt.hist(contrasts, bins=bins, alpha=alpha, edgecolor='black', range=xrange)
        #these are incorrect titles
        #print("num-bins    max-bin-height    most-frequent-sdcontrast")
        #print( "  ", np.argmax(n), "\t\t\t", int(np.max(n)), "\t\t", bins[np.argmax(n)])
        
        most_frequent = bins[np.argmax(n)]
        
        plt.xlim(xrange[0], xrange[1])
        
        plt.title("iScat Contrast")
        #plt.yscale("log")
        plt.xlabel("iscat contrast (standard deviation)")
        plt.ylabel("counts")
        #plt.legend(prop={'size': 14})
        txt="min:  " + str(np.min(contrasts)) + "\nmode:" + str(most_frequent) + "\nmax:   " + str(np.max(contrasts)) 
        
        plt.figtext(0.7, 0.7, txt, wrap=True, horizontalalignment='center', fontsize=18)
        
        plt.savefig(image_filename)
        #plt.show()
        plt.clf()
        plt.close('all')









def plot_landing_rate(particle_list, constants, tag):
    print("Drawing Landing Rate vs Potential Plot...")
    
    # some helpful definitions: particles per frame, list
    nframes = constants["nframes"]
    fps     = constants['fps']
    name    = constants["name"]

    ''' initialize a plot area '''
    fig, ax1 = plt.subplots()
    #plt.rcParams['figure.figsize'] = [8, 8]
    #plt.figure(dpi=150)        
    ax2 = ax1.twinx()      
    
    
    ''' x axis '''
    spf = np.linspace(0, (nframes/fps), nframes) #seconds per frame
    total_time = nframes/fps
    minutes = [m*60 for m in range(int((total_time+60)/60))]
    seconds = [m*10 for m in range(int((total_time+10)/10))]
    #if the clip is less than a minute, generate them in a different way
    if total_time < 60:
        minutes = seconds
        seconds = [m*1 for m in range(int((total_time+1)/1))]
        
        
    ''' plot landing rate '''
    ppf = np.zeros(nframes)   # makea  placeholder array to store the number of particles landed per frame
    c = 0
    for i in range(len(ppf)):   # loop through each frame of the video 
        pids = [p.pID for p in particle_list if p.f_vec[0] == i]   # returns a list of particle ID's (which are equivalent to particle counts) for all particles that first landed on this particular frame
        if pids: c = pids[-1]                                   # if a list of landed particles was created, then use the largest pID. (the last particle on the list), as the new total particle count
        ppf[i] = c                                              # set the particles per frame to be the total number of particles found so far
        
    ax1.plot(spf, ppf, color='steelblue', linewidth=3, alpha=0.8)
    
    
    ''' lot voltage data'''
    if constants["voltfile"]:
        nsplit = constants["name"].split("_")
        voltage_file = os.path.join(constants["basepath"], (nsplit[0] +"_"+nsplit[1]+"_voltage.txt") )
        data = np.loadtxt(voltage_file, dtype=str, delimiter=',') 
        vpf = np.array([x[0] for x in data]).astype(float)
        if len(vpf) > nframes: vpf = vpf[:nframes]
        #plot the voltage data
        #ax2.scatter(spf, vpf, color='violet', alpha=0.5, s=2)
        ax2.plot(spf, vpf, color='violet', linewidth=3, alpha=0.8)
        ax2.set_ylabel('Applied Potential /Volt')
        ax2.set_ylim((-2,2))
        # ax2.set_xticks(minutes)
        # ax2.set_xticks(seconds, minor=True)
    


     
    # labels and stuff
    ax1.set_xlabel('Time /s')
    ax1.set_ylabel('Number of Landings')
    ax1.set_xticks(minutes)
    ax1.set_xticks(seconds, minor=True)
    
    # ax1.set_xticks([0,60])#minutes)
    # ax1.set_xticks([0,10,20,30,40,50])#seconds, minor=True)

    ax1.set_title(constants["sample name"])   

    filename = "Landing Rate " + name + tag+".png"
    save_file_path = os.path.join(constants["output path"], filename)
    print(save_file_path)
    plt.savefig(save_file_path, dpi=150)
    
    #plt.show()
    
    
    plt.clf
    plt.close('all')
    
    
'''
###############################################################################
###############################################################################
###############################################################################
             _     __          __           __                  
            (_)  _/_/____     / /_   ___   / /____   ___   _____
           / / _/_/ / __ \   / __ \ / _ \ / // __ \ / _ \ / ___/
          / /_/_/  / /_/ /  / / / //  __// // /_/ //  __// /    
         /_//_/    \____/  /_/ /_/ \___//_// .___/ \___//_/     
                                          /_/                   
             ____                     __   _                    
            / __/__  __ ____   _____ / /_ (_)____   ____   _____
           / /_ / / / // __ \ / ___// __// // __ \ / __ \ / ___/
          / __// /_/ // / / // /__ / /_ / // /_/ // / / /(__  ) 
         /_/   \__,_//_/ /_/ \___/ \__//_/ \____//_/ /_//____/  
                                                                    
###############################################################################
###############################################################################
###############################################################################
'''#!!!

                # SAVING
#saves particle list to a pickle file
def save_pickle_data(particle_list, constants, tag):
    # generate filepath and save figure as .png
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - particle list.pkl"
    #basepath = constants["basepath"]
    filename = constants["name"] + tag + ".pkl"
    save_file_path = os.path.join(constants["output path"], filename)
    pickle.dump(particle_list, open(save_file_path, "wb"))
    return True

# save a text file of the constants used for this experiment
def save_constants(constants):
    #saves constants to a text file
    # use read_constants() to open the text file as a dict again
    c_string = str(constants)
    filepath = os.path.join(constants["output path"], "constants.txt")
    with open(filepath, "w") as f:
        f.write(c_string)
    return 1


                # LOADING
#loads particle list from a pickle file
def load_pickle_data(filepath):
    particle_list = pickle.load( open(filepath, "rb"))
    return particle_list

# loads constants from a previous run
def load_constants(output_path):
    txt_file = os.path.join(output_path, "constants.txt")
    with open(txt_file, "r") as f:
        dict_string = f.read()
    c_dict = ast.literal_eval(dict_string)
    return c_dict

  
import cv2
import numpy as np
def load_video(filepath):
    print("loading video in memory...")
    cap = cv2.VideoCapture(filepath)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret: #check if the video has ended
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)
    
    frames_array = np.array(frames)
    cap.release()
    print("\t\t\t\t...Done!")
    return frames_array
