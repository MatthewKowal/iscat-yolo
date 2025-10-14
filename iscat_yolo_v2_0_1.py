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

'''
#%%
add color to this
from colorama import Fore, Style

Fore.BLACK
Fore.RED
Fore.GREEN
Fore.YELLOW
Fore.BLUE
Fore.MAGENTA
Fore.CYAN
Fore.WHITE
Fore.RESET  # resets to default
print(Fore.RED + "This is red text")
print(Fore.GREEN + "This is green text")
print(Style.RESET_ALL + "Back to normal")


#%%
#https://rich.readthedocs.io/en/stable/appendix/colors.html
from rich import print

print("[red]Red[/red]")
print("[green]Green[/green]")
print("[blue]Blue[/blue]")
print("[magenta]Magenta[/magenta]")
print("[orange1]Orange1[/orange1]")
print("[#ff00ff]Custom Hex Color[/#ff00ff]")

print("[bold italic underline cyan]Stylish![/bold italic underline cyan]")

print("[bold magenta]Hello in magenta![/bold magenta]")
print("[green]Success![/green] [red]Error![/red]")
#%%
'''


# bin file mamangement
import os
import time
import numpy as np
import tqdm

# ratiometric particle finder
#from collections import deque
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



def cprint(text):
    from rich.console import Console
    # hexcolor = ["#00af87", #grey blue
    #              "#008787", #bright blue
    #              "#5f00af", #dark magenta
    #              "#87d700", #blood red
    #              ]
    console = Console()
    console.print(text, style=STYLE)

    # USE IT LIKE THIS IN A FUNCTION
    # global STYLE
    # STYLE = "#00af87"
    # cprint("Loading binfile into memory...")

''' NEW FUCNTIONS '''

def measure_video_brightness(raw_video, constants):
    raw_brightness_values = [np.sum(f)/(constants["video x dim"]*constants["video y dim"]) for f in raw_video]
    import csv
    with open(os.path.join(constants["output path"], "screen brightness.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        for value in raw_brightness_values:
            writer.writerow([value])  # write one number per row
    
    plt.plot(raw_brightness_values)
    plt.xlabel("frame number")
    plt.ylabel("average pixel value")
    plt.savefig(os.path.join(constants["output path"], "screen brightness.png"))
    plt.clf()
    
    
def plot_contrast_kinetics(pl_in, constants, tag):
    import matplotlib
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    ax2 = ax.twinx()
    
    ''' VOLT DATA '''
    # generate voltage data
    nframes = constants["nframes"]
    nsplit = constants["name"].split("_")
    voltage_file = os.path.join(constants["basepath"], (nsplit[0] +"_"+nsplit[1]+"_EPD voltage.txt") )
    data = np.loadtxt(voltage_file, dtype=str, delimiter=',') 
    vpf = np.array([x[0] for x in data]).astype(float)
    if len(vpf) > nframes: vpf = vpf[:nframes]
    voltdata = vpf#get_voltdata()
    
    
    volttime = np.linspace(0, len(voltdata)/constants["fps"], len(voltdata))
    ax2.plot(volttime, voltdata, color=matplotlib.cm.viridis(0.9))
    ax2.set_ylim(-0.2, 2)
    ax2.set_ylabel("volts")

    # ''' Generate Landing rate data '''    
    f = np.arange(0, 120, 0.1)
    f_sparse = [(p.f_vec[0]+200)/constants["fps"] for p in pl_in] #this is a list of the first frame each particle was seen (len(f_sparse) = len(pl))
    # #generate y data that corresponds to each frame number. the total number of deposited particles at any frame is the number of elements in f_sparse that are less than the current frame
    n = np.zeros_like(f)
    for j, fval in enumerate(f):
        n[j] = len([fnum for fnum in f_sparse if fnum <= fval])
    # ax.plot(f, n, linewidth=2, color=matplotlib.cm.viridis(0.1))
    # ax.set_xlabel("time, s")
    # ax.set_label("counts")
    # ax.legend(markerscale=3)

    ''' Generate Contrast Data '''
    p_time     = [(p.f_vec[0]+200)/constants["fps"] for p in pl_in]
    p_contrast = [(p.std_max)**(1/3) for p in pl_in]
    ax.scatter(p_time, p_contrast, s=1)
    
    ''' Plot a Running Average '''
    # Convert to NumPy arrays
    p_time = np.array(p_time)
    p_contrast = np.array(p_contrast)
    
    # Now sort using indices
    sorted_indices = np.argsort(p_time)
    p_time = p_time[sorted_indices]
    p_contrast = p_contrast[sorted_indices]
    # Choose a window size in time units (e.g., 1 second)
    window_size = 5.0
    # Create output arrays
    avg_time = []
    avg_contrast = []
    # Slide the window along time
    for i, t in enumerate(p_time):
        # Find all points within the window centered at t
        in_window = (p_time >= t - window_size/2) & (p_time <= t + window_size/2)
        if np.sum(in_window) > 0:
            avg_time.append(t)
            avg_contrast.append(np.mean(p_contrast[in_window]))
    # Plot running average
    ax.plot(avg_time, avg_contrast, color='red', linewidth=1.5, label=f'{window_size}s running average')

    ax.set_ylim([0.05, 0.2])
    ''' SAVE '''
    #scriptpath, scriptfilename = os.path.split(__file__)
    plt.savefig(os.path.join(constants["output path"], "contrast kinetics" + tag + ".png"))
    
    # ''' SAVE ZOOM '''
    # ax.set_xlim([-2,30])
    # #get max number of particles at t=30
    # ax.set_ylim([-5, n[np.where(f == 30)]])
    # plt.savefig(os.path.join(scriptpath, "FILTERED contrast kinetics -zoom" + tag + ".png"))
    #plt.show()
    
    return


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

def getVersion():
    # returns the scripts version number as a string
    #   this is to be used for printing the version
    #   number on the output folder.
    folder, filename = os.path.split(__file__)
    x = filename[:filename.rfind("-")-1]
    return x[x.rfind("v"):]
  


def get_bin_metadata(binfile, printmeta=False):
    
    basepath, filename = os.path.split(binfile)
    name               = filename.split(".")[0]
    metadata           = name.split("_")
    #print(len(metadata), metadata)
    
    try:
        #date = metadata[0]
        #time = metadata[1]
        fov  = int(metadata[3])
        x    = int(metadata[4])
        y    = int(metadata[4])
        fps  = int(metadata[5])
    
    except ValueError as e:
        raise ValueError("Error in bar(): {}".format(e))
        
    filesize        = os.path.getsize(binfile)
    nframes         = int(filesize / (x * y))
    remaining_bytes = filesize%(x * y)
    

    # if printmeta:         #print everything out
    #     print("\nFile Properties")
    #     print("\tLocation:\t\t\t", basepath)
    #     print("\tFilename:\t\t\t", filename)
    #     print("\tSquare FOV (um) :\t", fov)
    #     print("\tX Resolution : \t\t", x)
    #     print("\tY Resolution : \t\t", y)
    #     print("\tFrames per second:  ", fps)
    #     print("\tFile Size: \t\t\t", filesize)
    #     print("\tNumber of frames: \t", nframes)
    #     print("\tRunning time:(s) \t", (nframes/fps), " seconds")
    #     print("\tRunning time (m): \t", (nframes/fps/60), " minutes")
    #     print("\tRemaining Bytes: \t", remaining_bytes)

    if printmeta:         #print everything out
        cprint("\nFile Properties")
        cprint("\tLocation:\n\t"+basepath)
        cprint("\tFilename:\t\t"+filename)
        cprint("\tSquare FOV (um) :\t" + str(fov))
        cprint("\tX Resolution : \t\t" + str(x))
        cprint("\tY Resolution : \t\t"+ str(y))
        cprint("\tFrames per second:\t"+ str(fps))
        cprint("\tFile Size: \t\t" +  str(filesize))
        cprint("\tNumber of frames: \t" +  str(nframes))
        cprint("\tRunning time:(s) \t" +  str(nframes/fps) + " seconds")
        cprint("\tRunning time (m): \t" + str(nframes/fps/60) + " minutes")
        cprint("\tRemaining Bytes: \t" + str(remaining_bytes))

    return basepath, filename, name, nframes, fov, x, y, fps


def load_binfile_into_array(binfile, print_time=True): #open a binfile and import data into image array
    
    global STYLE
    STYLE = "#00af87"
    cprint("Loading binfile into memory...")


    if print_time: start=time.time()
    
    #get constants    
    try:
        basepath, filename, name, nframes, fov, x, y, fps = get_bin_metadata(binfile, printmeta=True) # get basic info about binfile
    except ValueError as ve:
        raise ValueError("value error: ", ve)
    
    # import video
    dt = np.uint8                                     # choose an output datatype
    images = np.zeros([nframes, x, y], dtype=dt)      # this will be the output var
    cprint("Importing....")
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

def ratio_only(images, constants, return_v16=False):
    '''STEPS:
            1. Define constants, video dimensions, queues, image & video arrays
            2. Perform ratiometric image processing
            3. Return an 8-bit video
    '''
    
    #DEFINE CONSTANTS
    bufsize             = constants['bufsize']
    clipmin             = constants['clipmin']
    clipmax             = constants['clipmax']
    n,x,y               = images.shape    
    v8                  = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)
    v16                 = np.ndarray([n-2*bufsize,x,y])#.astype(np.uint8)
    
                                   
    print("\n\nPerforming Ratiometric Image Process on ", n, " Frames")
    print(" \t\t Clip Length: ", (n/constants["fps"] / 60), " minutes.\n")
    
    # Step 0: Generate buffer images consisting of the sum of all 
    #         images in the first and second half of the buffer.
    q1 = np.sum([i/np.mean(i) for i in images[:bufsize]], axis=0)
    q2 = np.sum([i/np.mean(i) for i in images[bufsize:2*bufsize]], axis=0)
    
    
    for f in tqdm.tqdm(range(n-2*bufsize)):
         # f is frame number
    
         ''' RATIOMETRIC IMAGE ALGO '''
       
         ''' divide the two sums to get a ratiometric iamge'''  
         i16 = (q2/q1)
         
         ''' generate 8-bit image from a clipped version of the 16-bit ratiometric image '''
         i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
         v8[f] = i8
         v16[f] = i16
         
         ''' Manage Frame Queue '''
         if f < (n-2*bufsize)-1:
             qout  = (images[f]                    / np.mean(images[f])                   )
             qmove = (images[f+bufsize+1]          / np.mean(images[f+bufsize+1])         )
             qnew  = (images[f+bufsize+bufsize+1]  / np.mean(images[f+bufsize+bufsize+1]) )
             q1 = q1 + qmove - qout
             q2 = q2 + qnew - qmove
    if return_v16: return v16, v8
    else: return v8





# Particle Class
class Particle:
    def __init__(self, px, py, wx, wy, first_frame_seen, pID, yoloimage, pimage, stddev, mean, conf, screen_stddev):
        # particle x and y and the width of the box surrounding the particle
        self.px_vec            = [px]                # particle x position for each frame
        self.py_vec            = [py]                # particle y position for each frame
        self.wx_vec            = [wx]                # particle bbox width for each frame
        self.wy_vec            = [wy]                # particle bbox height for each frame
        
        #a list of frames the particle was seen in and a particle iD
        self.f_vec            = [first_frame_seen]     # first frame the particle was seen
        self.pID              = pID 
        
        #images of the particle
        self.yoloimage_vec         = [yoloimage]             # 16bit image straight from yolo
        self.pimage_vec            = [pimage]                # 16bit 30x30 particle image
        
        #std dev measurement for the entire frame
        self.screen_stddev_vec         = [screen_stddev]
        #stddev contrast measurements for the particle

        self.mean_vec              = [mean]
        self.conf_vec              = [conf]
        
        #new particle image and stddev contrast after the new center of the particle is found
        #self.new_pimage_vec        = [new_pimage]
        #self.new_stddev_vec        = [new_stddev]
        
        #gaussian fit data

        #self.lo_gauss_vec          = [] #laplacian of gaussian (not good)
        
        #line fit contrast data
        self.std_vec            = [stddev]                # particle bbox height for each frame
        self.std_max = 0
        self.std_fit  = 0
        self.std_trim = 0
        
        #gauss contrast
        self.gauss_a             = []
        self.gauss_max = 0
        self.gauss_fit = 0
        self.gauss_trim = 0

        #gauss position
        self.gauss_x_vec             = []
        self.gauss_y_vec             = []
        self.gauss_wx_vec            = []
        self.gauss_wy_vec            = []
        self.gauss_z_vec             = []
        
        
        #self.log_fit_contrast   = 0 #laplacian of gaussian (not good)
        #self.log_trim_contrast   = 0    #laplacian of gaussian (not good)

        #self.newsd_fit_contrast = 0
        
        #line trim contrast data
        # (this ones weeds out bad data by using lines fit to the contrast timeline)
        # condition: lines must fit with 

        #self.newsd_trim_contrast = 0
        
        
    def updateParticle(self, newp): # Take in a new particle and use its specs to update yourself
        self.px_vec.append(newp.px_vec[0])
        self.py_vec.append(newp.py_vec[0])
        self.wx_vec.append(newp.wx_vec[0])
        self.wy_vec.append(newp.wy_vec[0])
        
        self.f_vec.append(newp.f_vec[0])
        
        self.yoloimage_vec.append(newp.yoloimage_vec[0])
        self.pimage_vec.append(newp.pimage_vec[0])
        self.std_vec.append(newp.std_vec[0])
        self.mean_vec.append(newp.mean_vec[0])
        self.conf_vec.append(newp.conf_vec[0])
        
        self.screen_stddev_vec.append(newp.screen_stddev_vec[0])
        
        
        #self.new_pimage_vec.append(newp.new_pimage_vec[0])
        #self.new_stddev_vec.append(newp.new_stddev_vec[0])
        
        
        
        
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

#from skimage.transform import hough_circle, hough_circle_peaks
#from skimage.feature import canny
#import math
import tracemalloc
def ratio_particle_finder(images, constants):
    global STYLE
    STYLE = "#87d700"
    
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

    v8                  = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)       # this contains the 8 bit video
    #v16roi              = [0]*(n-2*bufsize)                                    # has enough elements for every frame, but we will only populate it for frames with particles 
    noise_floor_list    = [1]*(n-2*bufsize)                                       # this records the s.d. noise floor for each frame
    screen_std_list     = [1]*(n-2*bufsize)                                       # this records the s.d. noise floor for each frame
    yolo_model          = YOLO(constants["yolo model loc"])
    results_list        = [0]*(n-2*bufsize)                                    # yolo results for each frame
    
    video_particle_list = []
    #video_particle_list = [0]*100000 
    pID = 1
    #MIN_DIST       = 15
    MIN_DIST        = constants["spatial tolerance1"] #constants['MIN_DIST']
    timestamp_array = [0]*(n-2*bufsize)
    TRACEMEMORY     = False
    
    
    cprint("\n\nPerforming Ratiometric Image Process on " + str(n) + " Frames")
    cprint(" \t\t Clip Length: " + str(n/constants["fps"] / 60) + " minutes.\n")
    #print(n, bufsize,x,y)
    cprint("nframes, buf size, x, y"+str(n)+" "+str(bufsize)+str(n)+" "+str(x)+str(n)+" "+str(y))



    #######################################
    #i16                 = np.ndarray([x,y]).astype(np.float16)
    #i8                  = np.ndarray([x,y]).astype(np.uint8)
    # Step 0: Generate buffer images consisting of the sum of all 
    #         images in the first and second half of the buffer.
    q1 = np.sum([i/np.mean(i) for i in images[:bufsize]], axis=0)
    q2 = np.sum([i/np.mean(i) for i in images[bufsize:2*bufsize]], axis=0)
     
    for f in tqdm.tqdm(range(n-2*bufsize)):
        t1 = time.time()
        # f is frame number
     
        ''' RATIOMETRIC IMAGE ALGO '''
        
        ''' divide the two sums to get a ratiometric iamge'''  
        i16 = (q2/q1)
         
        ''' generate 8-bit image from a clipped version of the 16-bit ratiometric image '''
        i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
        v8[f] = i8
          
        ''' Manage Frame Queue '''
        if f < (n-2*bufsize)-1:
            qout  = (images[f]                    / np.mean(images[f])                   )
            qmove = (images[f+bufsize+1]          / np.mean(images[f+bufsize+1])         )
            qnew  = (images[f+bufsize+bufsize+1]  / np.mean(images[f+bufsize+bufsize+1]) )
            q1 = q1 + qmove - qout
            q2 = q2 + qnew - qmove
        ####################################
        
        
        t2 = time.time()
        
        ''' PAD THE IMAGE SO THAT WE DO NOT DETECT EDGE PARTICLES '''
        pillowimage = PIL.Image.fromarray(i8)
        draw = PIL.ImageDraw.Draw(pillowimage)
        x = x
        y = x
        e = constants["bbox"]
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
        
        ''' DISREGARD PARTICLE DETECTOR WHEN THE SCREEN HAS TOO HIGH CONTRAST '''
        #if np.std(i16) > 1: frame_pl = []
        #print(f, np.std(i16))
        screen_stddev = np.std(i16)
        
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
        
        ''' PARTICLE LIST BOOKKEEPING  '''
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
            

            new_particle = Particle(px, py, wx, wy, f, pID, yoloimage, pimage, stddev, mean, conf, screen_stddev)
            #new_particle = Particle(px, py, wx, wy, f, pID, yoloimage, pimage, stddev, mean, conf, new_pimage, new_stddev)
            
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
        screen_std_list[f] = screen_stddev
            
        
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
    # print("RATIOMETRIC PARTICLE FINDER SPEED REPORT:")
    # print("\t number of particles found: \t\t", len(video_particle_list))
    # print("\t Resolution: \t\t\t\t\t", x, y, "px")
    # print("\t Total Pixels per frame: \t\t", (x*y), "px")
    # print("\t New number of frames: \t\t\t", n, "frames")
    # print("\t Elapsed time: \t\t\t\t\t", (end-start), " seconds")
    # print("\t Elapsed time: \t\t\t\t\t", ((end-start)/60), " minutes")
    # #print("\t Total Number of Particles: \t", pID, "particles")
    # print("\t Speed (n-fps): \t\t\t\t", (n / (end-start)), " finished frames / sec" )
    
    cprint("RATIOMETRIC PARTICLE FINDER SPEED REPORT:")
    cprint("\t number of particles found: \t" + str(len(video_particle_list)))
    cprint("\t Resolution: \t\t\t" + str(x) + " x " + str(y) + " px")
    cprint("\t Total Pixels per frame: \t" + str(x*y) + "px")
    cprint("\t New number of frames: \t\t" + str(n) + "frames")
    cprint("\t Elapsed time: \t\t\t" + str(end-start) + " seconds")
    cprint("\t Elapsed time: \t\t\t" + str((end-start)/60) + " minutes")
    cprint("\t Speed (n-fps): \t\t" + str(n / (end-start)) + " finished frames / sec" )
    

    return v8, video_particle_list, results_list, noise_floor_list, screen_std_list
# This doesnt need to be separate (I think, but I am keeping it here because of
# some ideas I have about memory usage that i want to test later)
# def yolo_finder(i8, yolo_model):
#     rgb_image = color.gray2rgb(i8) # numpy grayscale array to numpy rgb array
#     frame_results = yolo_model(rgb_image, conf=0.01, verbose=False)
    # return frame_results



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

    temporal_tolerance     = constants["temporal tolerance"]
    spatial_tolerance = constants["spatial tolerance2"]
                                                                                ####    IMPORTANT: ONLY LOOK THROUGH THE LAST 1000 OR
       
    matched_pID            = False       # By default, the Particle is not found on the list
                                                            
    # new particle data
    px = new_particle.px_vec[0]       # particle x position
    py = new_particle.py_vec[0]       # particle y position
    #!!!
    #pf = new_particle.f_vec[0]       # current frame number
    pf = f
    
    # check to see if this particle existed in a previous frame
    # if you've looked back in time longer than (temporal_tolerance)
    # number of frames, then just quit looking
    
    max_l = 100   #max number of previous particles to look through 
    matched_pID_list = []
    
    # if the particle list is longer than the max number of particle to look through, then just look at the last 'max_l' number of particles
    if len(particle_list) > max_l: particle_list = particle_list[-max_l:] 
    
    # browse the particle list
    for i, lpart in enumerate(reversed(particle_list)): #reversed(particle_list):
        
        # list particle data
        lpx  = lpart.px_vec[-1]
        lpy  = lpart.py_vec[-1]
        lpf  = lpart.f_vec[-1]

        
        # if the last frame the list particle was seen was over 100 frames ago, then just give up
        # this works because we look through the list backwards
        #if f - lpf > 100: break
        
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
    
            matched_pID_list.append(lpart.pID)
            #matched_std_list.append(lpart.pimagestd_vec[-1])
            #print(f, lpart.pID, lpart.pimagestd_vec[-1])
        if matched_pID is not False: break #i guess this should stop the loop as soon as we find one.
        
    #print(f, len(particle_list), i) #print frame number, particle list size, number of iterations performed before exit
    if matched_pID is not False:
        #print(np.argmax(matched_std_list))
        #matched_pID = matched_pID_list[np.argmax(matched_std_list)]
        matched_pID = matched_pID_list[0] #for now, just give the first close pID it finds. might be better to use some other criteria like closest or darkest
    
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
    
    print("Making Particle Library Spreadsheets .csv file...") #create files
    csv_filename  = os.path.join(constants["output path"], (constants["timestamp"]+"_particle finder speed.csv"))

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
    
    name = constants["timestamp"]
    #generate empty dataframes for .csv files
    df1 = pd.DataFrame() 
    
    print("Making Particle Library Spreadsheets .csv file...") #create files
    csv_filename  = os.path.join(constants["output path"], (name+"_Particle List__"+tag+".csv"))

    
    #generate empty lists to store 
    pIDs                   = np.zeros(len(particle_list))
    
    calc_x             = np.zeros(len(particle_list))
    calc_y             = np.zeros(len(particle_list))
    
    # Lifetime Info
    f_0                    = np.zeros(len(particle_list))
    lifetimes              = np.zeros(len(particle_list))
    frames                 = [0]*(len(particle_list))
    
    confidence             = [0]*(len(particle_list))
    confidence_low         = np.zeros(len(particle_list))
    confidence_high        = np.zeros(len(particle_list))
    confidence_average     = np.zeros(len(particle_list))
    
    #screen stddev
    screen_stddev          = [0]*(len(particle_list))
    screen_stddev_low      = np.zeros(len(particle_list))
    screen_stddev_high     = np.zeros(len(particle_list))
    screen_stddev_average  = np.zeros(len(particle_list))
    
    # Location Info
    px_vec                  = [0]*(len(particle_list))
    py_vec                  = [0]*(len(particle_list))
    #px_0                    = np.zeros(len(particle_list))
    #py_0                    = np.zeros(len(particle_list))
    
    
    # bbox info
    wx_vec                  = [0]*(len(particle_list))
    wy_vec                  = [0]*(len(particle_list))
    wx_0                    = np.zeros(len(particle_list))
    wy_0                    = np.zeros(len(particle_list))

    # contrast info    
    # ones_counter             = np.zeros(len(particle_list))
    # darkest_pixel            = np.zeros(len(particle_list))
    # #average_darkest_pixels   = np.zeros(len(particle_list))
    # brightest_pixel          = np.zeros(len(particle_list))
    # #average_brightest_pixels = np.zeros(len(particle_list))
    
    # mean_vec                = [0]*(len(particle_list))
    # mean_mean               = np.zeros(len(particle_list))


    #contrast measurements
    # new_stddev_max          = np.zeros(len(particle_list))
    # init_stdev_max          = np.zeros(len(particle_list))
    #lo_gauss_max        = np.zeros(len(particle_list))
    std_vec              = [0]*(len(particle_list))
    std_max              = np.zeros(len(particle_list))
    std_fit              = np.zeros(len(particle_list))
    std_trim             = np.zeros(len(particle_list))
    gauss_a               = [0]*(len(particle_list))
    gauss_max               = np.zeros(len(particle_list))
    gauss_fit              = np.zeros(len(particle_list))
    gauss_trim             = np.zeros(len(particle_list))
    
    gauss_x_vec               = [0]*(len(particle_list))
    gauss_y_vec               = [0]*(len(particle_list))
    gauss_wx_vec              = [0]*(len(particle_list))
    gauss_wy_vec              = [0]*(len(particle_list))
    gauss_z_vec               = [0]*(len(particle_list))
    

    # #linefit contrast data
    # gauss_fit_contrast       = np.zeros(len(particle_list))
    # log_fit_contrast         = np.zeros(len(particle_list))
    # sd_fit_contrast          = np.zeros(len(particle_list))
    # newsd_fit_contrast       = np.zeros(len(particle_list))

    # iterate through the particle list and tablutate data
    for c, p in enumerate(particle_list):
             
        #assign particle data to fill in a row of the .csv
        pIDs[c]                = p.pID

        #calc_x[c]               = np.mean(p.px_vec)+np.mean(p.gauss_x_vec)
        #calc_y[c]               = np.mean(p.py_vec)+np.mean(p.gauss_y_vec)

        # Lifetime Info
        f_0[c]                 = p.f_vec[0]
        lifetimes[c]           = len(p.f_vec)
        frames[c]              = p.f_vec
        
        #confidence info
        confidence[c]             = p.conf_vec
        confidence_low[c]         = np.min(p.conf_vec)
        confidence_high[c]        = np.max(p.conf_vec)
        confidence_average[c]     = np.average(p.conf_vec)
        
        #confidence info
        screen_stddev[c]             = p.screen_stddev_vec
        screen_stddev_low[c]         = np.min(p.screen_stddev_vec)
        screen_stddev_high[c]        = np.max(p.screen_stddev_vec)
        screen_stddev_average[c]     = np.average(p.screen_stddev_vec)
        
        
        
        # Location Info
        #px_0[c]                  = p.px_vec[0]
        #py_0[c]                  = p.py_vec[0]
        px_vec[c]                = p.px_vec
        py_vec[c]                = p.py_vec
        
        
        wx_0[c]                  = p.wx_vec[0]
        wy_0[c]                  = p.wy_vec[0]
        wx_vec[c]                = p.wx_vec
        wy_vec[c]                = p.wy_vec
        
        
        # set up some stuff to calculate darkest and brightest pixels
        #https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        #k = 4 # choose 4 darkest or lightest values
        #center_frame = np.argmax(p.stddev_vec)
        #center_image = p.pimage_vec[center_frame]
        #d_idx = np.argpartition(center_image, k)
        #b_idx = np.argpartition(center_image, -k)
        
        #calculate darkest and brightest pixels for the most contrasty image for each particle
        #ones_counter                = (center_image == 1).sum() # counts the number of values equal to 1.0
        #print(center_image)
        #darkest_pixel[c]            = np.min(center_image)
        #average_darkest_pixels[c]   = np.average(center_image[d_idx[:k]])
        
        #brightest_pixel[c]          = np.max(center_image)
        #average_brightest_pixels[c] = np.average(center_image[b_idx[:-k]]) 
                
        #mean_mean[c]             = np.mean(p.mean_vec)
        #mean_vec[c]              = p.mean_vec
        
        #contrast measurements
        std_vec[c]               = p.std_vec
        std_max[c]               = np.max(p.std_vec)
        std_fit[c]               = p.std_fit
        std_trim[c]              = p.std_trim
        gauss_a[c]             = p.gauss_a
        
        
        try: gauss_max[c]        = np.min(p.gauss_a)
        except ValueError as e: gauss_max[c] = 0
        
        gauss_fit[c]                = p.gauss_fit
        gauss_trim[c]               = p.gauss_trim
        
        gauss_x_vec[c]             = p.gauss_x_vec
        gauss_y_vec[c]             = p.gauss_y_vec
        gauss_wx_vec[c]            = p.gauss_wx_vec
        gauss_wy_vec[c]            = p.gauss_wy_vec
        gauss_z_vec[c]             = p.gauss_z_vec
        
        
        #contrast measurements
        #new_stddev_max[c]        = np.max(p.new_stddev_vec)
        #init_stdev_max[c]       = np.max(p.stddev_vec)
        #lo_gauss_max[c]  = np.max(p.lo_gauss_a)
        
        #line fit contrast
        #gauss_fit_contrast = p.gauss_fit_contrast
        #log_fit_contrast   = p.log_fit_contrast
        #sd_fit_contrast    = p.sd_fit_contrast
        #newsd_contrastfit = p.newsd_fit_contrast
        
        
        
    #write data to the spreadsheet    
    # Particle ID    
    df1["pID"]                 = pIDs
    
    df1["calc x"]                 = calc_x
    df1["calc y"]                 = calc_y    
    
    
    #measures of contrast
    # df1["init stdev"]          = init_stdev_max
    # df1["new stddev"]          = new_stddev_max
    # df1["gauss"]               = gauss_max
    # df1["lo gauss"]      = lo_gauss_max
    
    #line fit contrast
    df1["gauss max"]       = gauss_max
    df1["gauss fit"]       = gauss_fit
    df1["gauss trim"]       = gauss_trim
    
    #df1["log line contrast"]         = log_fit_contrast
    df1["std max"]          = std_max
    df1["std fit"]           = std_fit 
    df1["std trim"]             = std_trim
    #df1["new sd line contrast"]      = newsd_fit_contrast
    
    # Lifetime Info
    df1["f0"]                  = f_0
    df1["lifetime"]            = lifetimes
    df1["frames"]              = frames
    
    # Confidence Info
    df1["Conf"]                = confidence
    df1["Conf low"]            = confidence_low
    df1["Conf high"]           = confidence_high
    df1["Conf avg"]            = confidence_average
    
    # screen stddev Info
    df1["screen std"]                = screen_stddev
    df1["screen std low"]            = screen_stddev_low
    df1["screen std high"]           = screen_stddev_high
    df1["screen std avg"]            = screen_stddev_average
    
    # Position Info
    # df1["px0"]                  = px_0
    # df1["py0"]                  = py_0
    df1["px list"]              = px_vec
    df1["py list"]              = py_vec
    
    df1["g x list"]              = gauss_x_vec
    df1["g y list"]              = gauss_y_vec
    df1["g wx list"]             = gauss_wx_vec
    df1["g wy list"]             = gauss_wy_vec
    df1["g zo list"]             = gauss_z_vec

    # bbox Info
    # df1["wx0"]                  = wx_0
    # df1["wy0"]                  = wy_0
    df1["wx list"]              = wx_vec
    df1["wy list"]              = wy_vec
    
    #sd contrast info
    # df1["ones counter"]         = ones_counter
    # df1["dark pxl"]             = darkest_pixel
    # df1["avg dark pxl"]         = average_darkest_pixels
    # df1["bright pxl"]           = brightest_pixel
    # df1["avg bright pxl"]       = average_brightest_pixels
    
    # df1["mean mean"]            = mean_mean
    # df1["mean vec"]             = mean_vec
    # df1["stddev max"]           = stddev_max
    df1["stddev list"]          = std_vec

    df1.to_csv(csv_filename, index=True)
    #print("\t Particles list saved as: ", csv_filename)
    
    
    
def generate_sdcontrast_csv(pl_in, constants, tag):#cnn_pl, constants, "-cnn")
    print("exporting sd-contrast .csv file...")
    max_contrast = []
    for p in tqdm.tqdm(pl_in):
        max_contrast.append(np.max(p.std_vec))
    filename_out = os.path.join(constants['output path'], (tag+"__sd-contrast.csv"))
    np.savetxt(filename_out, 
           max_contrast,
           delimiter =", ", 
           fmt ='% s')

def generate_noisefloor_csv(noise, screen_std_list, constants, tag):#cnn_pl, constants, "-cnn")
    print("exporting noise floor .csv file...")
    print("minimum noise: ", np.min(noise))
    framenums = np.arange(0, len(noise), 1)
    filename_out = os.path.join(constants['output path'], (constants["timestamp"]+"__noise__"+str(np.min(noise))[:8]+tag+".csv"))
    np.savetxt(filename_out,
           np.transpose([framenums, noise, screen_std_list]),
           header="fnum, min noise, screen stddev",
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
    csv_filename = os.path.join(constants["output path"], (constants["timestamp"]+"_Landing Rate__"+ tag +".csv"))    
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
    
    
    voltpath = os.path.join(constants["basepath"], (constants["timestamp"] + "_EPD voltage.txt")) 
    #print(voltpath)
    voltdata = pd.read_csv(voltpath, header=None, names=["volts", "nan"])
    #print(voltdata.iloc[1, 0])
  
    # initialize useful variables
    n, x, y = r8_in.shape
    #video_RGB = []
    #EDGE = 15
    #EDGE = constants['bbox']
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
    #bB, bG, bR = 0, 89, 170
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
    #typ_m = 30               #verticle type spacing multiplier
    
    typ_vo = -10 #type verticle offset
    
    n_tot = constants["nframes"]                                         # total number of frames
    txp = txp_+span_
    span = 1  
    
    print("Drawing Framenumbers and voltage trace on video...")
    #for fnum in range(len(video_out)):
    
    print(type(voltdata))
    print("voltfile length: ")    
    
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
            
            volts = voltdata.iloc[(fnum+offset+offset2-1), 0]
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
    name = constants["timestamp"]
    filename = name + "_"+tag + "-color.avi"
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



def save_si_video(r8_in, particle_list_in, constants, tag, offset=0, print_fnum=False, print_v=False):
    
    
    voltpath = os.path.join(constants["basepath"], (constants["timestamp"] + "EPD voltage.txt")) 
    #print(voltpath)
    voltdata = pd.read_csv(voltpath, header=None, names=["volts", "nan"])
    #print(voltdata.iloc[1, 0])
  
    # initialize useful variables
    n, x, y = r8_in.shape
    #video_RGB = []
    #EDGE = 15
    #EDGE = constants['bbox']
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
    #bB, bG, bR = 0, 89, 170
    vB, vG, vR = 253, 142, 124
    
    # loop through each frame drawing frame numbers on each one
    # also draw a rectantular bounding box that defines the particle images
    # and also the particle edge cut-off distance
    #print(video_out.shape)
    
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
        
        #draw frame number
        if print_fnum:
            draw.text( (2,2), str(fnum), (tR, tG, tB), font=font)
        
        
        #print current volts on screen
        if print_v:
            volts = voltdata.iloc[(fnum+offset), 0]
            draw.text( (100,20), (str(volts) + " V"), (vR, vG, vB), font=font)
        
   
        # draw the voltage trace
        span   = 196                # length of trace (in frames)
        txp    = 8                 # trace x position
        typ    = 180                # trace y position
        tyh    = 50                 # trace height (this is essentially a multiplier for the voltage)
        #x_off = -20
        #n_tot = constants["frame total"]                                         # total number of frames
        n_tot = constants["nframes"]                                         # total number of frames
        
        
        #print(fnum, fnum+offset-span-offset2, span, n_tot, span+offset2)
        if fnum >= span+offset2 and fnum + offset + span + offset2 < n_tot:                         # you will get array errors if you run this on the whole array so this makes sure that doesnt happen
            
        
            trace = np.array(voltdata.iloc[fnum+offset-span+offset2:(fnum+offset+offset2), 0])   # trace is a snippit of the voltage frame array. trace is what we print on the screen 
            #print(fnum, len(trace), trace[0], trace[-1])
            #print("trace: ", trace)

            last_v = trace[0]                                                     # the method works by drawing a line from point n to point n+1. this makes sure the starting point for the line is the same as the first point it will draw to
            for vi, vv in enumerate(trace):                                      # Loop through the whole trace
                
                # draw a point on the trace line (it actually draws a few points to add thicness)
                draw.point(( txp+vi, typ+(tyh*(1-vv))   ), fill=(vR, vG, vB))      #
                draw.point(( txp+vi, typ+(tyh*(1-vv))+1 ), fill=(vR, vG, vB))
                draw.point(( txp+vi, typ+(tyh*(1-vv))+2 ), fill=(vR, vG, vB))

                # draw a tick mark every 200 voltages
                voltage_position = fnum+offset+vi
                if voltage_position % 200 == 0:      
                    #print(fnum, voltage_position)
                    draw.point(( txp+vi, typ+(tyh*(1-vv))-1 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi, typ+(tyh*(1-vv))-2 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi, typ+(tyh*(1-vv))-3 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi, typ+(tyh*(1-vv))-4 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi, typ+(tyh*(1-vv))-5 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi, typ+(tyh*(1-vv))-6 ), fill=(vR, vG, vB))

                    draw.point(( txp+vi+1, typ+(tyh*(1-vv))-1 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+1, typ+(tyh*(1-vv))-2 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+1, typ+(tyh*(1-vv))-3 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+1, typ+(tyh*(1-vv))-4 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+1, typ+(tyh*(1-vv))-5 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+1, typ+(tyh*(1-vv))-6 ), fill=(vR, vG, vB))
                    
                    draw.point(( txp+vi+2, typ+(tyh*(1-vv))-1 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+2, typ+(tyh*(1-vv))-2 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+2, typ+(tyh*(1-vv))-3 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+2, typ+(tyh*(1-vv))-4 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+2, typ+(tyh*(1-vv))-5 ), fill=(vR, vG, vB))
                    draw.point(( txp+vi+2, typ+(tyh*(1-vv))-6 ), fill=(vR, vG, vB))

                
                
                #print(trace[vi], last_v)
                if vv != last_v:                                                 # this draws the line that connects when it switches from 0 to 1 V
                    #lx1, ly1 = vi+(0.5*x-span)+x_off,    tp+(th*(1-vv))
                    #lx2, ly2 = vi+(0.5*x-span)+x_off,    tp+(th*(1-last_v))
                    
                    lx1, ly1 = txp+vi, typ+(tyh*(1-vv))
                    lx2, ly2 = txp+vi, typ+(tyh*(1-last_v))
                    draw.line([(lx1, ly1), (lx2, ly2)], fill=(vR, vG, vB), width=3)
                last_v = vv  
            
            draw.text( (210,typ-20), "1V", (vR, vG, vB), font=font)
            draw.text( (210,typ+tyh-20), "0V",  (vR, vG, vB), font=font)
            


        
        
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
    name = constants["timestamp"]
    filename = name + "_"+tag + "-color.avi"
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
                    

''' SAVE VIDEO '''
#def save_video_from_pl2(pl_in, r8, TAG):
def save_particle_list_video(r8_in, pl_in, constants, tag, skip_gauss=False):
    global STYLE
    STYLE="deep_pink1"
    cprint("Generating Particle Video...")
    
    t0 = time.time()
    # initialize useful variables
    from skimage import color
    import PIL
    from PIL import ImageDraw, ImageFont
    import tqdm
    
    #particle_list_in = pl3
    
    n, x, y = r8_in.shape
    video_RGB = []
    #EDGE = 15
    EDGE = constants['bbox']
    
        # VIDEO DRAWING
    #first, convert video to RGB
    cprint("Converting grayscale video to RGB...")
    for c, f in enumerate(r8_in):
        rgb_image = color.gray2rgb(r8_in[c])
        video_RGB.append(rgb_image)
    video_out = np.array(video_RGB)
    
    # define colors
    tR, tG, tB = 220, 220, 220 #text color
    bR, bG, bB = 221,  28, 200 #box color
    #pR, pG, pB = 20,   40, 200 #point color
    pR, pG, pB = 221,  28, 200 #point color
    fR, fG, fB = 240, 240, 240 #frame color
    font = ImageFont.truetype("arial.ttf", 32)
    font_small = ImageFont.truetype("arial.ttf", 10)
    
    cprint("Preparing Particle annotations...")
    
    ba = [[] for _ in range(len(video_out))]   #box annotations
    pa = [[] for _ in range(len(video_out))] #point annotations
    pida = [[] for _ in range(len(video_out))] #particle ID annotations
   
    #draw each particle on the frames that it exists on
    # for this one it makes the most sense to loop through the by particle rather than by frame
    for p in tqdm.tqdm(pl_in): #go through list of particles
        #print("Particle: ", p.pID, " found in frames: ", p.f_vec)
        for i, f in enumerate(p.f_vec):#, pfnum in enumerate(p.f_vec): #for this particle, go through the frame vector and draw the particle bounding box on the frame
            px = p.px_vec[i]
            py = p.py_vec[i]
            wx = p.wx_vec[i]
            wy = p.wy_vec[i]
            ba[f].append([px-wx, py-wy, px+wx, py+wy])
            pa[f].append([px-1, py-1, px+1, py+1])

            xloc, yloc = px, py #p.x_vec[cc], p.y_vec[cc]
            if xloc > x - 2*EDGE: xloc -= EDGE
            if yloc > y - 2*EDGE: yloc -= EDGE
            pida[f].append([xloc, yloc, p.pID])

    # loop through each frame drawing frame numbers on each one
    # also draw a rectantular bounding box that defines the particle images
    # and also the particle edge cut-off distance
    #print(video_out.shape)
    cprint("\nAnnotating video frames...")
    for i in tqdm.tqdm(range(len(video_out))):
        pillowImage = PIL.Image.fromarray(video_out[i])
        draw = ImageDraw.Draw(pillowImage)
        
        #draw particle boxes
        # for b in ba[i]:
        #     draw.rectangle((b[0], b[1], b[2], b[3]), fill=None, outline=(bR, bG, bB))
        
        #draw particle points
        for p in pa[i]:
            draw.rectangle((p[0], p[1], p[2], p[3]), fill=None, outline=(pR, pG, pB))
            

        #write particle numbers next to particles
        xloc, yloc = px, py #p.x_vec[cc], p.y_vec[cc]
        if xloc > x - 2*EDGE: xloc -= EDGE
        if yloc > y - 2*EDGE: yloc -= EDGE
        for p in pida[i]:
            draw.text((p[0], p[1]), str(p[2]),  (tR, tG, tB), font=font_small) #color was 220, 20, 220
         

        #draw frame number    
        draw.text( (2,2), str(i), (tR, tG, tB), font=font)
        
        #draw frame boundary 
        draw.rectangle((EDGE,EDGE,x-EDGE,y-EDGE), fill=None, outline=(fR, fG, fB))
        #Convert the frame to a numpy array
        video_out[i] = np.array(pillowImage, np.uint8)
    
    # VIDEO SAVING
    # Generate Filename
    # scriptpath, scriptfilename = os.path.split(__file__)
    # timestamp = constants["timestamp"]
    # filename = "FILTERED " + timestamp + "-color" + tag + ".avi"
    # save_file_path = os.path.join(scriptpath, filename)
    
    # VIDEO SAVING
    # Generate Filename
    name = constants["timestamp"]
    filename = name + "_"+tag + "-color.avi"
    save_file_path = os.path.join(constants["output path"], filename)
    
    
    # Write and save video
    cprint("\nSaving Yolo Particle Video to:")
    print(f"{save_file_path}\n")
    
    # Save with compression settings
    import imageio.v2 as imageio
    imageio.mimwrite(
        save_file_path,
        video_out,
        fps=constants["output framerate"],
        codec="libx264",  # H.264 codec
        ffmpeg_params=["-crf", "22", "-preset", "slow"] #23 is pretty good. lower number is less compression
    )
    t1 = time.time()
    print(f"\ntotal time for video saving 2: {t1-t0} seconds\n")

# def save_particle_list_video(r8_in, particle_list_in, constants, tag, skip_gauss=False):
#     global STYLE
#     STYLE="#5f00af"
#     # initialize useful variables
#     n, x, y = r8_in.shape
#     video_RGB = []
#     #EDGE = 15
#     EDGE = constants['bbox']

#         # VIDEO DRAWING

#     print("\n")
#     cprint("Generating Color Video...")
#     #first, convert video to RGB
#     cprint("Converting grayscale video to RGB...")
#     for c, f in enumerate(r8_in):
#         rgb_image = color.gray2rgb(r8_in[c])
#         video_RGB.append(rgb_image)
#     video_out = np.array(video_RGB)
    
#     # define colos
#     tR, tG, tB = 220, 220, 220
#     bR, bG, bB = 221,  28, 200
    
#     # loop through each frame drawing frame numbers on each one
#     # also draw a rectantular bounding box that defines the particle images
#     # and also the particle edge cut-off distance
#     #print(video_out.shape)
#     cprint("Drawing Framenumbers and Particle Bounding boxed on video...")
#     for fnum in range(len(video_out)):
#         # initialize a pillow image and a drawing object for that iamge
#         pillowImage = PIL.Image.fromarray(video_out[fnum])
#         draw = PIL.ImageDraw.Draw(pillowImage)
        
#         #draw frame number
#         font = PIL.ImageFont.truetype("arial.ttf", 32)
#         draw.text( (2,2), str(fnum), (tR, tG, tB), font=font)
        
#         #draw frame boundary 
#         draw.rectangle((EDGE,EDGE,x-EDGE,y-EDGE), fill=None, outline=(240,240,240))
#         #Convert the frame to a numpy array
#         video_out[fnum] = np.array(pillowImage, np.uint8)
        
 
    
#     cprint("Drawing Particles on video...")

#     #draw each particle on the frames that it exists on
#     # for this one it makes the most sense to loop through the by particle rather than by frame
#     for p in particle_list_in: #go through list of particles
#         #print("Particle: ", p.pID, " found in frames: ", p.f_vec)
        
#         for i, fnum in enumerate(p.f_vec): #for this particle, go through the frame vector and draw the particle bounding box on the frame
            
#             # initialize a pillow image and a drawing object for that iamge
#             pillowImage = PIL.Image.fromarray(video_out[fnum])
#             draw = PIL.ImageDraw.Draw(pillowImage)
        
#             px = p.px_vec[i]
#             py = p.py_vec[i]
#             wx = p.wx_vec[i]
#             wy = p.wy_vec[i]
            
#             # draw a box around each particle ( x1, y1,  x2,  y2 )
#             draw.rectangle((px-wx, py-wy, px+wx, py+wy), fill=None, outline=(bR, bG, bB))
            
            
#             #draw a little square in the yolo middle of the particle
#             draw.rectangle((px-1, py-1, px+1, py+1), fill=None, outline=(20, 40, 200))
            
#             if skip_gauss == False:
#                 #draw a little square in the gaussian middle of the particle
#                 gx = px + p.gauss_x_vec[i] - constants["bbox"]
#                 gy = py + p.gauss_y_vec[i] - constants["bbox"]
#                 draw.rectangle((gx-1, gy-1, gx+1, gy+1), fill=None, outline=(200, 40, 0))
                
            
#             font = PIL.ImageFont.truetype("arial.ttf", 32)
#             xloc, yloc = px, py #p.x_vec[cc], p.y_vec[cc]
#             if xloc > x - 2*EDGE: xloc -= EDGE
#             if yloc > y - 2*EDGE: yloc -= EDGE
#             draw.text( (xloc, yloc), str(p.pID), (tR, tG, tB), font=font) #color was 220, 20, 220
            
#             image_copy = np.array(pillowImage, np.uint8)
            
#             video_out[fnum] = image_copy
    
#     # VIDEO SAVING
#     # Generate Filename
#     name = constants["timestamp"]
#     filename = name + "_"+tag + "-color.avi"
#     save_file_path = os.path.join(constants["output path"], filename)

#     # Write and save video
#     cprint("Saving Color Particle Video to: \n" + save_file_path + "\n")
    
#     # Save with compression settings
#     import imageio.v2 as imageio
#     imageio.mimwrite(
#         save_file_path,
#         video_out,
#         fps=constants["output framerate"],
#         codec="libx264",  # H.264 codec
#         ffmpeg_params=["-crf", "23", "-preset", "slow"]
#     )
    
#     #save video - old method
#     # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     # #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#     # #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     # #fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
#     #                     #use a different 4-byte code for other codecs
#     #                     #https://www.fourcc.org/codecs.php
#     # videoObject = cv2.VideoWriter(save_file_path, fourcc, constants["output framerate"], (y, x), isColor = True)
#     # for c in range(n):
#     #     videoObject.write(video_out[c])
#     # videoObject.release()
    
#     # #compress video
#     # compressed_filename = constants["timestamp"] + "_"+tag + "-compressed.mp4"
#     # compressed_filepath = os.path.join(constants["output path"], compressed_filename)
#     # compress_video(save_file_path, compressed_filepath, crf=23, preset="slow")
    
#     # #delete old video
#     # if os.path.exists(save_file_path):
#     #     os.remove(save_file_path)
#     #     print(f"{save_file_path} deleted successfully.")
#     # else:
#     #     print("File not found.")
#     return 
    


#this is only used for debugging because it displays the raw yolo results before particle bookkeeping
def save_yolo_results_video(r8_in, results_list, constants, tag):
    # draw results on video
    n, x, y = r8_in.shape
    video_RGB = []

    #first, convert video to RGB
    print("Converting grayscale video to RGB...")
    for c, f in enumerate(r8_in):
        rgb_image = color.gray2rgb(r8_in[c])
        video_RGB.append(rgb_image)
    video_out = np.array(video_RGB)
    
    # define colos
    tR, tG, tB = 220, 220, 220
    bR, bG, bB = 221,  28, 50
    
    #print(video_out.shape)
    print("Drawing Framenumbers and Particle Bounding boxed on video...")
    for fnum in range(len(video_out)):
        # initialize a pillow image and a drawing object for that iamge
        pillowImage = PIL.Image.fromarray(video_out[fnum])
        draw = PIL.ImageDraw.Draw(pillowImage)
        
        #draw frame number
        font = PIL.ImageFont.truetype("arial.ttf", 32)
        draw.text( (2,2), str(fnum), (tR, tG, tB), font=font)
        
        #Draw Particle Bounding Boxes
        frame_results = results_list[fnum]
        pl = frame_results[0].boxes.xywh
        #print(len(pl))
        for p in pl: 
            px = int(p[0])
            py = int(p[1])
            wx = int(p[2])
            wy = int(p[3])
            # draw a box around each particle ( x1, y1,  x2,  y2 )
            draw.rectangle((px-wx, py-wy, px+wx, py+wy), fill=None, outline=(bR, bG, bB))
            
        #Convert the frame to a numpy array
        video_out[fnum] = np.array(pillowImage, np.uint8)
     
    # Generate Filename
    #n, x, y, colors = rgb_video.shape
    name = constants["timestamp"]
    filename = name + "_"+tag + "-color.avi"
    save_file_path = os.path.join(constants["output path"], filename)

    # Write and save video
    print("Saving Yolo Particle Video to:   ", save_file_path)
    #fourcc = cv2.VideoWriter_fourcc(*'X264')  # this one makes big files and sometimes doesnt work
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #this one works but the files are larger
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
  
    
def save_bw_video(images, constants, tag, print_frame_nums = False):
    
    global STYLE
    STYLE = "#008787"
    
    cprint("Generating BW Video...")
    filename = constants["timestamp"] + "_"+tag + ".mp4"
    save_file_path = os.path.join(constants["output path"], filename)
    n, x, y = images.shape
    cprint("nframes, x, y: " + str(n)+", "+str(x)+", "+str(y))
    font = PIL.ImageFont.truetype("arial.ttf", 32)
    video_out = np.zeros_like(images)
    for f in tqdm.tqdm(range(len(images))):
        # create a PIL image from the current frame
        pillowImage = PIL.Image.fromarray(images[f])
        draw = PIL.ImageDraw.Draw(pillowImage)
        # if you'd like, draw frame numbers on the image
        if print_frame_nums: draw.text( (2,2), str(f), 255, font=font)
        video_out[f] = np.array(pillowImage, np.uint8)
        
    # Save with compression settings
    import imageio.v2 as imageio
    imageio.mimwrite(
        save_file_path,
        video_out,
        fps=constants["output framerate"],
        codec="libx264",  # H.264 codec
        ffmpeg_params=["-crf", "23", "-preset", "slow"],
        format='ffmpeg'
    )
    cprint("Video Saved as: \n"+save_file_path)  
    
    
    # #save video - old method
    # print("Saving Video...")
    # #fourcc = cv2.VideoWriter_fourcc(*'MPG4V')
    # #fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use H.264 codec for better compression
    # fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
    #                     #use a different 4-byte code for other codecs
    #                     #https://www.fourcc.org/codecs.php
    # video = cv2.VideoWriter(save_file_path, fourcc, constants["output framerate"], (y, x), 0)
    # for f in range(len(images)):
    #     # write the frame to the video
    #     video.write(np.array(pillowImage, np.uint8))
    # video.release()
    # print("Video Saved as: \t", save_file_path)  
    
    # #compress video
    # compressed_filename = constants["timestamp"] + "_"+tag + "-compressed.mp4"
    # compressed_filepath = os.path.join(constants["output path"], compressed_filename)
    # compress_video(save_file_path, compressed_filepath, crf=18, preset="slow")
    
    # #delete old video
    # if os.path.exists(save_file_path):
    #     os.remove(save_file_path)
    #     print(f"{save_file_path} deleted successfully.")
    # else:
    #     print("File not found.")
    return
  
# def compress_video(input_file, output_file, crf=23, preset="medium"):
#     import subprocess
#     """
#     Compress a video using ffmpeg.
    
#     Parameters:
#     - input_file: str, path to the input video
#     - output_file: str, path to the output compressed video
#     - crf: int, Constant Rate Factor (lower = better quality, higher = more compression)
#     - preset: str, compression speed vs. quality (slower presets give better compression)
    
#     CRF values:
#     - 18 = Visually lossless
#     - 23 = Default
#     - 28 = More compression, lower quality
#     """
#     command = [
#         "ffmpeg",
#         "-i", input_file,  # Input file
#         "-vcodec", "libx264",  # Video codec - pretty fast. good compression
#         #"-vcodec", "libx265",  # Video codec - very very slow
#         #"-vcodec", "libaom-av1",  # Video codec
#         "-crf", str(crf),  # Compression level
#         "-preset", preset,  # Speed vs. quality
#         "-an", #remove audio
#         #"-c:a", "aac",  # Audio codec (AAC)
#         #"-b:a", "128k",  # Audio bitrate
#         #"-movflags", "+faststart",  # Optimize for web streaming
#         output_file
#     ]
    
#     subprocess.run(command, check=True)

    
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
    #fps     = constants['fps']
    name    = constants["timestamp"]
      
    

    # # generate voltage data from file, trim if necessary
    nsplit = constants["name"].split("_")
    voltage_file = os.path.join(constants["basepath"], (nsplit[0] +"_"+nsplit[1]+"_EPD voltage.txt") )
    vdata = np.loadtxt(voltage_file, dtype=str, delimiter=',') 
    vpf = np.array([x[0] for x in vdata]).astype(float)
    if len(vpf) > nframes: vpf = vpf[:nframes]
     
    
  
    # Generate waterfall data
    frame_nums = [p.f_vec[0] for p in pl]             # first frame a particle was seen
    contrast   = [np.max(p.std_vec) for p in pl]   # maximum contrast of that particle
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
        #ax2.scatter(spf, vpf, color='violet', alpha=0.5, s=2)
        ax2.plot(vpf, color='violet', linewidth=3, alpha=0.2)
         
         
        # labels and stuff
        ax1.set_xlabel('time, Frame number')
        ax1.set_ylabel('s.d. contrast')
        ax1.set_ylim(crange)
        
        ax2.set_ylabel('Applied Potential /Volt')
        ax2.set_ylim((-2,2)) 
        ax1.set_title(constants["sample name"])   
             
        # Save the figure
        filename = "Waterfall " + name + tag + size + ".png"
        save_file_path = os.path.join(constants["output path"], filename)
        print(save_file_path)
        plt.savefig(save_file_path, dpi=150)
        
        #plt.show()
        
        
        plt.clf
        plt.close('all')
        


def plot_landing_map2(particle_list, constants, tag):
    print("Drawing Particle Landing Map...")
    name = constants["timestamp"]
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
    
    x = [(p.px_vec[-1] + p.gauss_x_vec[-1])*mpp for p in particle_list] #x = bbox(px) + gx (image offset ps)
    y = [(p.py_vec[-1] + p.gauss_y_vec[-1])*mpp for p in particle_list]
    c = [np.max(p.std_vec)*s_factor for p in particle_list]
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
    filename = name + "_Particle Map " + tag + ".png"
    save_file_path = os.path.join(constants["output path"], filename)
    plt.savefig(save_file_path)
    #plt.show()
    plt.clf()
    plt.close('all')

    
def plot_landing_map(particle_list, constants, tag):
    import matplotlib
    print("Drawing Particle Landing Map...")
    name = constants["timestamp"]
    mpp = constants["fov"] / constants["video x dim"]
    sample_name = constants["sample name"]
    #basepath = constants["basepath"]
    #n = len(particle_list)
    #x, y = np.zeros(n), np.zeros(n)
    #allx, ally = [], [] #get data into the right shape
    #for c in range(len(particle_list)):
    #    allx += particle_list[c].x_vec
    #    ally += particle_list[c].y_vec
    contrasts = [p.std_max for p in particle_list]
    max_std = np.max(contrasts)
    
    s_factor=1000
    
    x = [p.px_vec[0]*mpp for p in particle_list]
    y = [p.py_vec[0]*mpp for p in particle_list]
    c = [np.max(p.std_vec)*s_factor for p in particle_list]
    #print(c)
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.figure(dpi=150)
    for i, p in enumerate(particle_list):
        plt.scatter(x[i], y[i], s=c[i], alpha=0.5, color=matplotlib.cm.viridis_r(p.std_max/max_std)) # plot data
    plt.colorbar()
    plt.xlabel('microns')
    plt.ylabel('microns')
    #title = "Particle Landing Locations, " + sample_name
    #plt.title(title)
    
    # generate filepath and save figure as .png
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - particle map.png"
    filename = name + "_Particle Map " + tag + ".png"
    save_file_path = os.path.join(constants["output path"], filename)
    plt.savefig(save_file_path)
    #plt.show()
    plt.clf()
    plt.close('all')

def plot_contrast(pl, constants, tag):
    plt.clf()
    
    if len(pl) == 0: return
    #name = constants["timestamp"]
    #basepath = constants["basepath"]
    #sample_name = constants["sample name"]
    #if not os.path.exists(os.path.join(basepath, "output")): os.makedirs(os.path.join(basepath, "output"))
    
    # generate a list of particle contrasts
    contrasts = np.ones_like(pl)
    for i, p in enumerate(pl):
        #print(p.peak_contrastDoG)
        #contrasts[i] = np.min(p.drkpxl_avg_vec) #p.peak_contrastDoG 
        contrasts[i] = np.max(p.std_vec)
        #print(contrasts[i])
    
    print("\n Plotting Contrast")


    
    #image_filename  = os.path.join(constants["output path"], ("Contrast Histogram.png"))
    
    

    
    ''' MAKE HISTOGRAM COMBINED IMAGE '''
    fig, axs = plt.subplots(2, figsize=(8,10), dpi=300)
    alpha = 0.5
    
    
    ''' use max value for contrast '''
    #overlay max values for contrast in column 0
    g_list     = []
    sd_list    = []
    for p in tqdm.tqdm(pl):
        #print("ok")
        #print(p.gauss_a)
        g_list.append(-1*np.min(p.gauss_a))
        sd_list.append(np.max(p.std_vec))
        
    '''remove outliers'''
    '''######################'''
    def remove_outliers(data):
        Q1 = np.percentile(data, 25)   # Calculate Q1 (25th percentile)
        Q3 = np.percentile(data, 75)   # Calculate Q3 (75th percentile)
        IQR = Q3 - Q1                    # Calculate the IQR
        lower_bound = Q1 - 1.5 * IQR     # Calculate the lower bound
        upper_bound = Q3 + 1.5 * IQR     # Calculate the upper bound
        return [x for x in data if lower_bound <= x <= upper_bound] # Filtered data
        
    g_list = remove_outliers(g_list)
    sd_list = remove_outliers(sd_list)
    '''######################'''
        
    axs[0].hist(g_list,     bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(g_list)))
    axs[1].hist(sd_list,    bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(sd_list)))
    ''' SAVE CONTRAST CSV '''
    #gaussian
    filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-gauss_max.csv"))
    np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    #std dev
    filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-std_max.csv"))
    np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')

    

    
    
    axs[0].set_title("Max gaussian")
    axs[1].set_title("Max std dev")

    
    axs[0].legend()
    axs[1].legend()

    
    plt.tight_layout()
    plt.savefig(os.path.join(constants["output path"], (constants["timestamp"]+"_max_contrast_histogram"+tag+".png")))
    plt.clf()
    plt.close()
    
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-gauss_fit.csv"))
    # np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    # #std dev
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-std_fit.csv"))
    # np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')
    
    
    
    # plt.rcParams['figure.figsize'] = [10, 8]
    # plt.figure(dpi=150)
    # font = {'family' : 'normal',
    #         'weight' : 'normal',
    #         'size'   : 18}
    # SMALL_SIZE = 12
    # MEDIUM_SIZE=18
    # BIGGER_SIZE=18
    # #matplotlib.rc('font', **font)
    # plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE) 

    # alpha = 0.8
    # #bins = 100

    
    # n, bins, patches = plt.hist(contrasts, bins=bins, alpha=alpha, edgecolor='black', range=xrange)
    # #these are incorrect titles
    # #print("num-bins    max-bin-height    most-frequent-sdcontrast")
    # #print( "  ", np.argmax(n), "\t\t\t", int(np.max(n)), "\t\t", bins[np.argmax(n)])
    
    # most_frequent = bins[np.argmax(n)]
    
    # plt.xlim(xrange[0], xrange[1])
    
    # plt.title("iScat Contrast")
    # #plt.yscale("log")
    # plt.xlabel("iscat contrast (standard deviation)")
    # plt.ylabel("counts")
    # #plt.legend(prop={'size': 14})
    # txt="min:  " + str(np.min(contrasts)) + "\nmode:" + str(most_frequent) + "\nmax:   " + str(np.max(contrasts)) 
    
    # plt.figtext(0.7, 0.7, txt, wrap=True, horizontalalignment='center', fontsize=18)
    
    # plt.savefig(image_filename)
    # #plt.show()
    # plt.clf()
    # plt.close('all')

  
def plot_sdcontrast_hist(particle_list, constants, tag):
    if len(particle_list) == 0: return
    name = constants["timestamp"]
    #basepath = constants["basepath"]
    #sample_name = constants["sample name"]
    #if not os.path.exists(os.path.join(basepath, "output")): os.makedirs(os.path.join(basepath, "output"))
    
    # generate a list of particle contrasts
    contrasts = np.ones_like(particle_list)
    for i, p in enumerate(particle_list):
        #print(p.peak_contrastDoG)
        #contrasts[i] = np.min(p.drkpxl_avg_vec) #p.peak_contrastDoG 
        contrasts[i] = np.max(p.std_vec)
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
        # font = {'family' : 'normal',
        #         'weight' : 'normal',
        #         'size'   : 18}
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
    name    = constants["timestamp"]
      
    # generate landing rate data
    ppf = np.zeros(nframes)   # makea  placeholder array to store the number of particles landed per frame
    c = 0
    for i in range(len(ppf)):   # loop through each frame of the video 
        pids = [p.pID for p in particle_list if p.f_vec[0] == i]   # returns a list of particle ID's (which are equivalent to particle counts)
                                                    # for all particles that first landed on this particular frame
                                                    #print(pids)
        if pids: c = pids[-1]                                   # if a list of landed particles was created, then use the largest pID
                                                # (the last particle on the list), as the new total particle count
        ppf[i] = c                                              # set the particles per frame to be the total number of particles found so far
    
    # generate voltage data
    nsplit = constants["name"].split("_")
    voltage_file = os.path.join(constants["basepath"], (nsplit[0] +"_"+nsplit[1]+"_EPD voltage.txt") )
    data = np.loadtxt(voltage_file, dtype=str, delimiter=',') 
    vpf = np.array([x[0] for x in data]).astype(float)
    if len(vpf) > nframes: vpf = vpf[:nframes]
     
    # Generate X-Axis
    spf = np.linspace(0, (nframes/fps), nframes)
     
    
    #generate x axis tick marks for minutes and 10s intervals
    total_time = nframes/fps
    minutes = [m*60 for m in range(int((total_time+60)/60))]
    seconds = [m*10 for m in range(int((total_time+10)/10))]
    #if the clip is less than a minute, generate them in a different way
    if total_time < 60:
        minutes = seconds
        seconds = [m*1 for m in range(int((total_time+1)/1))]
        
    # print(nframes)
    # print(fps)
    # print(minutes)
    # print(seconds)
    # initialize a plot area
    fig, ax1 = plt.subplots()
    #plt.rcParams['figure.figsize'] = [8, 8]
    #plt.figure(dpi=150)        
    ax2 = ax1.twinx()
     
    #plot the voltage data
    #ax2.scatter(spf, vpf, color='violet', alpha=0.5, s=2)
    ax2.plot(spf, vpf, color='violet', linewidth=3, alpha=0.8)
     
     
    #plot the landing data data
    ax1.plot(spf, ppf, color='steelblue', linewidth=3, alpha=0.8)
     
     
    # labels and stuff
    ax1.set_xlabel('Time /s')
    ax1.set_ylabel('Number of Landings')
    ax1.set_xticks(minutes)
    ax1.set_xticks(seconds, minor=True)
    
    # ax1.set_xticks([0,60])#minutes)
    # ax1.set_xticks([0,10,20,30,40,50])#seconds, minor=True)

    
    ax2.set_ylabel('Applied Potential /Volt')
    ax2.set_ylim((-2,2))
    # ax2.set_xticks(minutes)
    # ax2.set_xticks(seconds, minor=True)
    
    ax1.set_title(constants["sample name"])   
     
    
    filename = name + "_Voltage Landing Rate " + tag+".png"
    save_file_path = os.path.join(constants["output path"], filename)
    print(save_file_path)
    plt.savefig(save_file_path, dpi=150)
    
    #plt.show()
    
    
    plt.clf
    plt.close()
    
    
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
    filename = constants["timestamp"] + tag + ".pkl"
    save_file_path = os.path.join(constants["output path"], filename)
    pickle.dump(particle_list, open(save_file_path, "wb"))
    return True

# save a text file of the constants used for this experiment
def save_constants(constants):
    #saves constants to a text file
    # use read_constants() to open the text file as a dict again
    c_string = str(constants)
    filepath = os.path.join(constants["output path"], (constants["timestamp"] +"_constants.txt"))
    with open(filepath, "w") as f:
        f.write(c_string)
    return True


                # LOADING
#loads particle list from a pickle file
def load_pickle_data(filepath):
    particle_list = pickle.load( open(filepath, "rb"))
    return particle_list

# loads constants from a previous run
def load_constants(output_path):
    import glob
    #txt_file = os.path.join(output_path, "constants.txt") #use this if the file is just named constants.txt
    txt_file = glob.glob(os.path.join(output_path, "*constants.txt"))[0] #use this is the constants file has a naming prefix
    with open(txt_file, "r") as f:
        dict_string = f.read()
    c_dict = ast.literal_eval(dict_string)
    return c_dict

# loads a video file into an 8-bit image array
import skvideo.io 
def load_video(filepath):
    ''' this function loads a video file and returns a 3d numpy array (frames, x, y) '''
    ''' it is assumed that the video file to be read in was created by the save_bw_video function in this script '''
    
    #d1 = skvideo.io.vread(filepath, as_grey=False) 
    d2 = skvideo.io.vread(filepath, as_grey=True) 
    #d3 = np.dot(d1[:][:][:],[0.3, 0.3, 0.3]).astype(np.uint8)
    d4 = np.dot(d2,[1]).astype(np.uint8)
    return d4
  
    
  
    
''' 
##############################################################################
##############################################################################
##############################################################################
##############################################################################

GAUSSIAN FITTING FUNcTIONS '''  
''' GAUSSIAN 

##############################################################################
##############################################################################
##############################################################################
##############################################################################
'''
#!!!
#this function finds best fit gaussian parameters for some experimental data
def gaussian(height, center_x, center_y, width_x, width_y, z_offset): #gaussian lamda function generator
    """Returns a gaussian function with the given parameters"""
    # width_x = float(width_x)
    # width_y = float(width_y)
    return lambda x,y: z_offset + height*np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_y)**2)/2)
from scipy import optimize
# def fit_gaussian_parameters(data): #find optimized gaussian fit for a particle
#     #this is a lambda function which defines the 2d gaussian function
#     # def gaussian(height, center_x, center_y, width_x, width_y, z_offset): #gaussian lamda function generator
#     #     """Returns a gaussian function with the given parameters"""
#     #     # width_x = float(width_x)
#     #     # width_y = float(width_y)
#     #     return lambda x,y: z_offset + height*np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_x)**2)/2)

#     #make a good initial guess at the gaussian parameters
#     height   = -0.1
#     x        = data.shape[0]/2
#     y        = data.shape[1]/2
#     width_x  = 3.0
#     width_y  = 3.0
#     z_offset = 1.0
#     params = height, x, y, width_x, width_y, z_offset
#     errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
#     p, success = optimize.leastsq(errorfunction, params)
#     return p

'''new for version 2'''
def fit_gaussian_parameters(data):
    # Initial guess for the Gaussian parameters
    height = -0.1
    x = data.shape[0] / 2
    y = data.shape[1] / 2
    width_x = 3.0
    width_y = 3.0
    z_offset = 1.0
    initial_guess = (height, x, y, width_x, width_y, z_offset)

    # Bounds for the parameters
    bounds = ([-0.5, 0, 0, 0, 0, -np.inf], [0.5, np.inf, np.inf, np.inf, np.inf, np.inf])

    # Define the Gaussian function for curve fitting
    def gaussian_2d(coords, height, center_x, center_y, width_x, width_y, z_offset):
        x, y = coords
        return z_offset + height * np.exp(
            -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2
        ).ravel()

    x, y = np.indices(data.shape)
    coords = np.vstack((x.ravel(), y.ravel()))

    # Fit the data
    try:
            
        params, covariance = optimize.curve_fit(
            gaussian_2d, coords, data.ravel(), p0=initial_guess, bounds=bounds)
    except RuntimeError as e:
        print(f"Error occurred: {e}")
        params = [0,0,0,0,0,0]

    return params #params = (height, center_x, center_y, width_x, width_y, z_offset)







# import math
# def LoG(height, center_x, center_y, sigma):
#     ox = lambda x: center_x - x
#     oy = lambda y: center_y - y
#     return lambda x, y: -(height*1000)/(math.pi*sigma**4)*(1-((ox(x)**2+oy(y)**2)/(2*sigma**2)))*np.exp(-((ox(x)**2+oy(y)**2)/(2*sigma**2)))+1
# def fitLoGaussian(data, pimage_dim): #find optimized gaussian fit for a particle
#     #first guess parameters
#     height   = 0.07
#     center_x = pimage_dim
#     center_y = pimage_dim
#     sigma    = 4.0
#     params = height, center_x, center_y, sigma
#     errorfunction = lambda p: np.ravel(LoG(*p)(*np.indices(data.shape)) - data)
#     p, success = optimize.leastsq(errorfunction, params)
#     return p


#import tqdm
def fit_gaussian_to_particle_list(pl, constants):
    print("Fitting Gaussian to Particles...")
    pl_out         = []
    
    for p in tqdm.tqdm(pl):
        p.gauss_a = []
        #p.lo_gauss_vec = []
        for i, img in enumerate(p.pimage_vec):
    
            
            params_g = fit_gaussian_parameters(img)
            #print(i, img.shape, params_g)
            p.gauss_a.append(params_g[0])
            p.gauss_x_vec.append(params_g[1])
            p.gauss_y_vec.append(params_g[2])
            p.gauss_wx_vec.append(params_g[3])
            p.gauss_wy_vec.append(params_g[4])
            p.gauss_z_vec.append(params_g[5])
            
            # params_log = fitLoGaussian(img, 15)
            # #print(params_log)
            # p.lo_gauss_vec.append(params_log[0])
        pl_out.append(p)
        
        # #plot gaussian fit with pimage
        # print("pID: ",         p.pID)
        # print("px:",           p.px_vec)
        # print("py:",           p.px_vec)
        # print("bbox:",         constants["bbox"])
        # print("pimage.shape:", p.pimage_vec[0].shape)
        
        # print("wx:",        p.wx_vec)
        # print("wy:",        p.wy_vec)
        # print("yoloimage.shape:", p.yoloimage_vec[0].shape)

        
        # print("amp: ",      p.gauss_a)
        # print("x: ",        p.gauss_a)
        # print("y: ",        p.gauss_a)
        # print("wx: ",       p.gauss_a)
        # print("wy: ",       p.gauss_a)
        # print("z-offset: ", p.gauss_a)
        
    return pl_out



def plot_pimages(pl, constants):
    #make pimage folder
    newpath = os.path.join(constants["output path"], "pimages")
    if not os.path.exists(newpath): os.makedirs(newpath)
    
    print_every_image = False
    print_max_sdcontrast = True
    for p in pl:
        if print_every_image == True:
            for f in range(len(p.pimage_vec)):
                gx = str(int(p.gauss_x_vec[f]))
                gy = str(int(p.gauss_y_vec[f]))
                
                pfname = "pimg_"+str(p.pID)+"_"+str(f)+"_"+gx+"_"+gy+".png"
                yfname = "yolo_"+str(p.pID)+"_"+str(f)+".png"
                
                pfpath = os.path.join(newpath, pfname)
                yfpath = os.path.join(newpath, yfname)
                
                plt.imsave(pfpath, p.pimage_vec[f])
                plt.imsave(yfpath, p.yoloimage_vec[f])
                
        if print_max_sdcontrast == True:
            print(p.pID, np.argmax(p.std_vec), np.max(p.std_vec))
            f = np.argmax(p.std_vec)
            
            gx = str(int(p.gauss_x_vec[f]))
            gy = str(int(p.gauss_y_vec[f]))
            
            pfname = "pimg_"+str(p.pID)+"_"+str(f)+"_"+gx+"_"+gy+".png"
            yfname = "yolo_"+str(p.pID)+"_"+str(f)+".png"
            
            pfpath = os.path.join(newpath, pfname)
            yfpath = os.path.join(newpath, yfname)
            
            plt.imsave(pfpath, p.pimage_vec[f])
            plt.imsave(yfpath, p.yoloimage_vec[f])
            
            
                       





#import matplotlib.pyplot as plt
#import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def linear_func(x, m, b):
    return m * x + b


# def gauss1d(x, amp, mean, sigma, offset):
    # return offset * amp*np.exp((x-mean)**2/(2*sigma)**2)
def sine_wave(x, amp, phase, offset):
    return amp * np.sin(np.radians(x) + phase) + offset

from scipy import stats
import matplotlib.patches as patched


##!!!
##!!!

##!!!
##!!!
##!!!

##!!!
##!!!

##!!!
##!!!
##!!!

##!!!
##!!!


def plot_particle_contrast2(pl, constants):
    v_plot_dir = os.path.join(constants["output path"], "v_plots")
    if not os.path.exists(v_plot_dir): os.makedirs(v_plot_dir)
    c = 0
    print("fitting and plotting contrast data")
    
    ''' Make contrast V plots '''
    for p in tqdm.tqdm(pl):
        
        c+=1
        xdata=np.arange(len(p.f_vec))        
        #fig, ax1 = plt.subplots() 
        fig, axs = plt.subplots(2,2, figsize=(10,10))
    
        ''' METHOD 1: GAUSSIAN FIT '''
        ydata=[-g for g in p.gauss_a]
        c1 = "#111199" #data
        c2 = "#111199" #sine wave
        c3 = "#111199" #center marker
        c4 = "#111199" #fit lines
        ''' STEP 0: Plot the contrast data '''
        axs[0,0].scatter(xdata, ydata, color=c1, s=1, label="gauss")
        axs[0,1].scatter(xdata, ydata, color=c1, s=1, label="gauss")
        
        '''STEP 1: Find the "center" of the "V" plot by fitting a sine wave to the data '''    
        initial_guess = [1, 0, 0]
        popt, pcov = curve_fit(sine_wave, xdata, ydata, p0=initial_guess)
        fit_wave = sine_wave(xdata, *popt)
        axs[0,1].plot(xdata, fit_wave, linewidth=1, color=c2)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        #find the center by assuming the maximum value of the sine wave is the center
        center_x = np.argmax(fit_wave)
        max_val = np.max(ydata)
        axs[0,1].plot([center_x, center_x], [max_val+max_val/10, max_val-max_val/10], color=c3, linewidth=1)
        '''STEP 2: Check to see if the two lines meet somewhere near the middle '''
        #print("pID: ", p.pID, "\t frames: ", len(p.f_vec), "\t center: ", center_x)
        MINEDGE = int(constants["bufsize"]/4)
        if center_x>MINEDGE and center_x<len(xdata)-MINEDGE:
            ''' if the center point is at least 3 points from an edge, '''
            ''''fit lines and measure new contrast from the average of '''
            '''where they cross the center '''
            leftx = np.linspace(0,center_x,center_x+1)
            rightx = np.linspace(center_x, len(xdata),(len(xdata)-center_x))
            #plot left side
            slope, intercept, r_value, p_value, std_err = stats.linregress(leftx, ydata[:center_x+1])
            #print("\t r_value: ", r_value)
            leftfit = slope * leftx + intercept
            axs[0,1].plot(leftx, leftfit, color=c4, linewidth=4, alpha=0.5)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
            #plot right side
            slope, intercept, r_value, p_value, std_err = stats.linregress(rightx, ydata[center_x:])
            rightfit = slope * rightx + intercept
            axs[0,1].plot(rightx, rightfit, color=c4, linewidth=4, alpha=0.5)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
            #measure the contrast from the average of the two end points
            g_contrast = (leftfit[-1] + rightfit[0]) / 2
            g_x = leftx[-1]
            g_trim = g_contrast
        else:
            ''' if the center then say the gaussian trim contrast is 0 '''
            #print("\t error: no center found")
            g_contrast = np.max(ydata)
            g_x = center_x
            g_trim = 0
        e_center = (g_x, g_contrast)
        e_width  = 1
        e_height = g_contrast/100
        e_angle = 0
        ellipse = patched.Ellipse(e_center, e_width, e_height, angle=e_angle, edgecolor="black", facecolor=c4)
        axs[0,1].add_patch(ellipse)
        axs[0,0].set_title("Gaussian")
        axs[0,1].set_title("Gaussian")
        
        if g_contrast > 1: g_contrast = 0#print("WARNING: CONTRAST TOO HIGH")
        p.gauss_fit = g_contrast
        p.gauss_trim = g_trim
        
        
        

        


        #ax2 = ax1.twinx()
        ''' METHOD 3: Standard Deviation '''
        ydata = p.std_vec
        c1 = "#994411" #data
        c2 = "#994411" #sine wave
        c3 = "#994411" #center marker
        c4 = "#994411" #fit lines
        ''' STEP 0: Plot the contrast data '''
        axs[1,0].scatter(xdata, ydata, color=c1, s=1, label="gauss")
        axs[1,1].scatter(xdata, ydata, color=c1, s=1, label="gauss")
        '''STEP 1: Find the "center" of the "V" plot by fitting a sine wave to the data '''    
        initial_guess = [1, 0, 0]
        if p.pID == 20:
            print("pID: ", p.pID)
            print(ydata)
        popt, pcov = curve_fit(sine_wave, xdata, ydata, p0=initial_guess)
        fit_wave = sine_wave(xdata, *popt)
        axs[1,1].plot(xdata, fit_wave, linewidth=1, color=c2)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        #find the center by assuming the maximum value of the sine wave is the center
        center_x = np.argmax(fit_wave)
        max_val = np.max(ydata)
        axs[1,1].plot([center_x, center_x], [max_val+max_val/10, max_val-max_val/10], color=c3, linewidth=1)
        '''STEP 2: Check to see if the two lines meet somewhere near the middle '''
        #print("pID: ", p.pID, "\t frames: ", len(p.f_vec), "\t center: ", center_x)
        MINEDGE = int(constants["bufsize"]/4)
        if center_x>MINEDGE and center_x<len(xdata)-MINEDGE:
            leftx = np.linspace(0,center_x,center_x+1)
            rightx = np.linspace(center_x, len(xdata),(len(xdata)-center_x))
            #plot left side
            slope, intercept, r_value, p_value, std_err = stats.linregress(leftx, ydata[:center_x+1])
            #print("\t r_value: ", r_value)
            leftfit = slope * leftx + intercept
            axs[1,1].plot(leftx, leftfit, color=c4, linewidth=4, alpha=0.5)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
            #plot right side
            slope, intercept, r_value, p_value, std_err = stats.linregress(rightx, ydata[center_x:])
            rightfit = slope * rightx + intercept
            axs[1,1].plot(rightx, rightfit, color=c4, linewidth=4, alpha=0.5)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
            #measure the contrast from the average of the two end points
            g_contrast = (leftfit[-1] + rightfit[0]) / 2
            g_x = leftx[-1]
            g_trim = g_contrast
        else:
            #print("\t error: no center found")
            g_contrast = np.max(ydata)
            g_x = center_x
            g_trim = 0
        e_center = (g_x, g_contrast)
        e_width  = 1
        e_height = g_contrast/100
        e_angle = 0
        ellipse = patched.Ellipse(e_center, e_width, e_height, angle=e_angle, edgecolor="black", facecolor=c4)
        axs[1,1].add_patch(ellipse)
        axs[1,0].set_title("std contrast") 
        axs[1,1].set_title("std contrast") 
        
        if g_contrast > 1: g_contrast = 0#print("WARNING: CONTRAST TOO HIGH")
        p.std_fit = g_contrast
        p.std_trim = g_trim

        ''' SAVE V PLOTS '''
        filename = str(p.pID) + " V.png"
        save_file_path = os.path.join(constants["output path"], "v_plots", filename)
        #print(save_file_path)
        plt.savefig(save_file_path, dpi=150)

    

    ''' MAKE HISTOGRAM COMBINED IMAGE '''
    fig, axs = plt.subplots(2,3, figsize=(20,8), dpi=300)
    alpha = 0.5
    
    
    ''' use max value for contrast '''
    #overlay max values for contrast in column 0
    g_list     = []
    sd_list    = []
    for p in tqdm.tqdm(pl):
        g_list.append(-1*np.min(p.gauss_a))
        sd_list.append(np.max(p.std_vec))
        
    axs[0,0].hist(g_list,     bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(g_list)))
    axs[1,0].hist(sd_list,    bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(sd_list)))
    ''' SAVE CONTRAST CSV '''
    #gaussian
    filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-gauss_max.csv"))
    np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    #std dev
    filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-std_max.csv"))
    np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')

    
    #plot line fit contrast in column 1
    g_list     = []
    sd_list    = []
    for p in pl:
        if p.gauss_fit > 0: g_list.append(p.gauss_fit)
        if p.std_fit > 0:    sd_list.append(p.std_fit)

    axs[0,1].hist(g_list,     bins=50, edgecolor="black", alpha=alpha, label="fit "+str(len(g_list)))
    axs[1,1].hist(sd_list,    bins=50, edgecolor="black", alpha=alpha, label="fit "+str(len(sd_list)))
    ''' SAVE FIT CONTRAST CSV '''
    #gaussian
    filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-gauss_fit.csv"))
    np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    #std dev
    filename_out = os.path.join(constants['output path'], (constants['name']+"contrast-std_fit.csv"))
    np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')

    

    #overlay trimmed contrast data in column 2
    g_list     = []
    #log_list   = []
    sd_list    = []
    #newsd_list = []
    for p in tqdm.tqdm(pl):
        if p.gauss_trim > 0: g_list.append(p.gauss_trim)
        if p.std_trim > 0: sd_list.append(p.std_trim)
        
    axs[0,2].hist(g_list,     bins=50, edgecolor="black", alpha=alpha, label="trim "+str(len(g_list)))
    axs[1,2].hist(sd_list,    bins=50, edgecolor="black", alpha=alpha, label="trim "+str(len(sd_list)))
    ''' SAVE CONTRAST CSV '''
    #gaussian
    filename_out = os.path.join(constants['output path'], (constants['timestamp']+"_contrast-gauss_trim.csv"))
    np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    #std dev contrast
    filename_out = os.path.join(constants['output path'], (constants['timestamp']+"_contrast-std_trim.csv"))
    np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')

    
    axs[0,0].set_title("Max gaussian")
    axs[1,0].set_title("Max std dev")
    axs[0,1].set_title("Fit gaussian")
    axs[1,1].set_title("Fit std dev")
    axs[0,2].set_title("Trim gaussian")
    axs[1,2].set_title("Trim std dev")
    
    axs[0,0].legend()
    axs[1,0].legend()
    axs[0,1].legend()
    axs[1,1].legend()
    axs[0,2].legend()
    axs[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(constants["output path"], (constants["timestamp"]+"_fit_histograms.png")))
    plt.clf()
    plt.close()

    return pl




                       

# def plot_particle_contrast(pl, constants):
#     v_plot_dir = os.path.join(constants["output path"], "v_plots")
#     if not os.path.exists(v_plot_dir): os.makedirs(v_plot_dir)
#     c = 0
#     print("fitting and plotting contrast data")
#     for p in tqdm.tqdm(pl):
        
#         c+=1
#         xdata=np.arange(len(p.f_vec))        
#         #fig, ax1 = plt.subplots() 
#         fig, axs = plt.subplots(2,2, figsize=(4,8))
    
#         ''' METHOD 1: GAUSSIAN FIT '''
#         ydata=[-g for g in p.gauss_a]
#         c1 = "#111199" #data
#         c2 = "#111199" #sine wave
#         c3 = "#111199" #center marker
#         c4 = "#111199" #fit lines
#         ''' STEP 0: Plot the contrast data '''
#         axs[0,0].scatter(xdata, ydata, color=c1, s=1, label="gauss")
#         '''STEP 1: Find the "center" of the "V" plot by fitting a sine wave to the data '''    
#         initial_guess = [1, 0, 0]
#         popt, pcov = curve_fit(sine_wave, xdata, ydata, p0=initial_guess)
#         fit_wave = sine_wave(xdata, *popt)
#         axs[0,0].plot(xdata, fit_wave, linewidth=1, color=c2)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#         #find the center by assuming the maximum value of the sine wave is the center
#         center_x = np.argmax(fit_wave)
#         max_val = np.max(ydata)
#         axs[0,0].plot([center_x, center_x], [max_val+max_val/10, max_val-max_val/10], color=c3, linewidth=1)
#         '''STEP 2: Check to see if the two lines meet somewhere near the middle '''
#         #print("pID: ", p.pID, "\t frames: ", len(p.f_vec), "\t center: ", center_x)
#         if center_x>3 and center_x<len(xdata)-3:
#             leftx = np.linspace(0,center_x,center_x+1)
#             rightx = np.linspace(center_x, len(xdata),(len(xdata)-center_x))
#             #plot left side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(leftx, ydata[:center_x+1])
#             #print("\t r_value: ", r_value)
#             leftfit = slope * leftx + intercept
#             axs[0,0].plot(leftx, leftfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #plot right side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(rightx, ydata[center_x:])
#             rightfit = slope * rightx + intercept
#             axs[0,0].plot(rightx, rightfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #measure the contrast from the average of the two end points
#             g_contrast = (leftfit[-1] + rightfit[0]) / 2
#             g_x = leftx[-1]
#             g_trim = g_contrast
#         else:
#             #print("\t error: no center found")
#             g_contrast = np.max(ydata)
#             g_x = center_x
#             g_trim = 0
#         e_center = (g_x, g_contrast)
#         e_width  = 1
#         e_height = g_contrast/100
#         e_angle = 0
#         ellipse = patched.Ellipse(e_center, e_width, e_height, angle=e_angle, edgecolor="black", facecolor=c4)
#         axs[0,0].add_patch(ellipse)
#         axs[0,0].set_title("Gaussian")
#         if g_contrast > 1: g_contrast = 0#print("WARNING: CONTRAST TOO HIGH")
#         p.gauss_fit_contrast = g_contrast
#         p.gauss_trim_contrast = g_trim
        
        
#         ''' METHOD 2: laplacian of Gaussian-fit '''
#         ydata = [g for g in p.lo_gauss_a]
#         c1 = "#449944" #data
#         c2 = "#449944" #sine wave
#         c3 = "#449944" #center marker
#         c4 = "#449944" #fit lines
#         ''' STEP 0: Plot the contrast data '''
#         axs[0,1].scatter(xdata, ydata, color=c1, s=1, label="gauss")
#         '''STEP 1: Find the "center" of the "V" plot by fitting a sine wave to the data '''    
#         initial_guess = [1, 0, 0]
#         popt, pcov = curve_fit(sine_wave, xdata, ydata, p0=initial_guess)
#         fit_wave = sine_wave(xdata, *popt)
#         axs[0,1].plot(xdata, fit_wave, linewidth=1, color=c2)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#         #find the center by assuming the maximum value of the sine wave is the center
#         center_x = np.argmax(fit_wave)
#         max_val = np.max(ydata)
#         axs[0,1].plot([center_x, center_x], [max_val+max_val/10, max_val-max_val/10], color=c3, linewidth=1)
#         '''STEP 2: Check to see if the two lines meet somewhere near the middle '''
#         #print("pID: ", p.pID, "\t frames: ", len(p.f_vec), "\t center: ", center_x)
#         if center_x>3 and center_x<len(xdata)-3:
#             leftx = np.linspace(0,center_x,center_x+1)
#             rightx = np.linspace(center_x, len(xdata),(len(xdata)-center_x))
#             #plot left side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(leftx, ydata[:center_x+1])
#             #print("\t r_value: ", r_value)
#             leftfit = slope * leftx + intercept
#             axs[0,1].plot(leftx, leftfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #plot right side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(rightx, ydata[center_x:])
#             rightfit = slope * rightx + intercept
#             axs[0,1].plot(rightx, rightfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #measure the contrast from the average of the two end points
#             g_contrast = (leftfit[-1] + rightfit[0]) / 2
#             g_x = leftx[-1]
#             g_trim = g_contrast
#         else:
#             #print("\t error: no center found")
#             g_contrast = np.max(ydata)
#             g_x = center_x
#             g_trim = 0
#         e_center = (g_x, g_contrast)
#         e_width  = 1
#         e_height = g_contrast/100
#         e_angle = 0
#         ellipse = patched.Ellipse(e_center, e_width, e_height, angle=e_angle, edgecolor="black", facecolor=c4)
#         axs[0,1].add_patch(ellipse)
#         axs[0,1].set_title("laplacian of gauss")
#         if g_contrast > 1: g_contrast = 0#print("WARNING: CONTRAST TOO HIGH")
#         p.log_fit_contrast = g_contrast
#         p.log_trim_contrast = g_trim

        


#         #ax2 = ax1.twinx()
#         ''' METHOD 3: Standard Deviation '''
#         ydata = p.stddev_vec
#         c1 = "#994411" #data
#         c2 = "#994411" #sine wave
#         c3 = "#994411" #center marker
#         c4 = "#994411" #fit lines
#         ''' STEP 0: Plot the contrast data '''
#         axs[1,0].scatter(xdata, ydata, color=c1, s=1, label="gauss")
#         '''STEP 1: Find the "center" of the "V" plot by fitting a sine wave to the data '''    
#         initial_guess = [1, 0, 0]
#         popt, pcov = curve_fit(sine_wave, xdata, ydata, p0=initial_guess)
#         fit_wave = sine_wave(xdata, *popt)
#         axs[1,0].plot(xdata, fit_wave, linewidth=1, color=c2)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#         #find the center by assuming the maximum value of the sine wave is the center
#         center_x = np.argmax(fit_wave)
#         max_val = np.max(ydata)
#         axs[1,0].plot([center_x, center_x], [max_val+max_val/10, max_val-max_val/10], color=c3, linewidth=1)
#         '''STEP 2: Check to see if the two lines meet somewhere near the middle '''
#         #print("pID: ", p.pID, "\t frames: ", len(p.f_vec), "\t center: ", center_x)
#         if center_x>3 and center_x<len(xdata)-3:
#             leftx = np.linspace(0,center_x,center_x+1)
#             rightx = np.linspace(center_x, len(xdata),(len(xdata)-center_x))
#             #plot left side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(leftx, ydata[:center_x+1])
#             #print("\t r_value: ", r_value)
#             leftfit = slope * leftx + intercept
#             axs[1,0].plot(leftx, leftfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #plot right side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(rightx, ydata[center_x:])
#             rightfit = slope * rightx + intercept
#             axs[1,0].plot(rightx, rightfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #measure the contrast from the average of the two end points
#             g_contrast = (leftfit[-1] + rightfit[0]) / 2
#             g_x = leftx[-1]
#             g_trim = g_contrast
#         else:
#             #print("\t error: no center found")
#             g_contrast = np.max(ydata)
#             g_x = center_x
#             g_trim = 0
#         e_center = (g_x, g_contrast)
#         e_width  = 1
#         e_height = g_contrast/100
#         e_angle = 0
#         ellipse = patched.Ellipse(e_center, e_width, e_height, angle=e_angle, edgecolor="black", facecolor=c4)
#         axs[1,0].add_patch(ellipse)
#         axs[1,0].set_title("sd contrast") 
#         if g_contrast > 1: g_contrast = 0#print("WARNING: CONTRAST TOO HIGH")
#         p.sd_fit_contrast = g_contrast
#         p.sd_trim_contrast = g_trim


#         ''' METHOD 4: re-centered Standard Deviation '''
#         ydata = p.new_stddev_vec
#         c1 = "#bbbb22" #data
#         c2 = "#bbbb22" #sine wave
#         c3 = "#bbbb22" #center marker
#         c4 = "#bbbb22" #fit lines
#         ''' STEP 0: Plot the contrast data '''
#         axs[1,1].scatter(xdata, ydata, color=c1, s=1, label="gauss")
#         '''STEP 1: Find the "center" of the "V" plot by fitting a sine wave to the data '''    
#         initial_guess = [1, 0, 0]
#         popt, pcov = curve_fit(sine_wave, xdata, ydata, p0=initial_guess)
#         fit_wave = sine_wave(xdata, *popt)
#         axs[1,1].plot(xdata, fit_wave, linewidth=1, color=c2)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#         #find the center by assuming the maximum value of the sine wave is the center
#         center_x = np.argmax(fit_wave)
#         max_val = np.max(ydata)
#         axs[1,1].plot([center_x, center_x], [max_val+max_val/10, max_val-max_val/10], color=c3, linewidth=1)
#         '''STEP 2: Check to see if the two lines meet somewhere near the middle '''
#         #print("pID: ", p.pID, "\t frames: ", len(p.f_vec), "\t center: ", center_x)
#         if center_x>3 and center_x<len(xdata)-3:
#             leftx = np.linspace(0,center_x,center_x+1)
#             rightx = np.linspace(center_x, len(xdata),(len(xdata)-center_x))
#             #plot left side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(leftx, ydata[:center_x+1])
#             #print("\t r_value: ", r_value)
#             leftfit = slope * leftx + intercept
#             axs[1,1].plot(leftx, leftfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #plot right side
#             slope, intercept, r_value, p_value, std_err = stats.linregress(rightx, ydata[center_x:])
#             rightfit = slope * rightx + intercept
#             axs[1,1].plot(rightx, rightfit, color=c4, linewidth=4)#, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             #measure the contrast from the average of the two end points
#             g_contrast = (leftfit[-1] + rightfit[0]) / 2
#             g_x = leftx[-1]
#             g_trim = g_contrast
#         else:
#             #print("\t error: no center found")
#             g_contrast = np.max(ydata)
#             g_x = center_x
#             g_trim = 0
#         e_center = (g_x, g_contrast)
#         e_width  = 1
#         e_height = g_contrast/100
#         e_angle = 0
#         ellipse = patched.Ellipse(e_center, e_width, e_height, angle=e_angle, edgecolor="black", facecolor=c4)
#         axs[1,1].add_patch(ellipse)
#         axs[1,1].set_title("new sd contrast")
#         if g_contrast > 1: g_contrast = 0#print("WARNING: CONTRAST TOO HIGH")
#         p.newsd_fit_contrast = g_contrast
#         p.newsd_trim_contrast = g_trim

#         #ax1.legend()
#         #ax2.legend()
#         #plt.show()
#         plt.tight_layout()
#         plt.savefig(os.path.join(constants["output path"], "v_plots", (str(p.pID) + ".png")))
#         plt.clf()

    

    # ''' MAKE HISTOGRAM COMBINED IMAGE '''
    # alpha = 0.5
    # g_list     = []
    # log_list   = []
    # sd_list    = []
    # newsd_list = []
    # for p in pl:
    #     if p.gauss_fit_contrast > 0: g_list.append(p.gauss_fit_contrast)
    #     if p.log_fit_contrast > 0:   log_list.append(p.log_fit_contrast)
    #     if p.sd_fit_contrast > 0:    sd_list.append(p.sd_fit_contrast)
    #     if p.newsd_fit_contrast > 0: newsd_list.append(p.newsd_fit_contrast)
    # fig, axs = plt.subplots(2,2, figsize=(10,8))
    # axs[0,0].hist(g_list,     bins=50, edgecolor="black", alpha=alpha, label="fit "+str(len(g_list)))
    # axs[0,1].hist(log_list,   bins=50, edgecolor="black", alpha=alpha, label="fit "+str(len(log_list)))
    # axs[1,0].hist(sd_list,    bins=50, edgecolor="black", alpha=alpha, label="fit "+str(len(sd_list)))
    # axs[1,1].hist(newsd_list, bins=50, edgecolor="black", alpha=alpha, label="fit "+str(len(newsd_list)))

    # ''' SAVE FIT CONTRAST CSV '''
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--fit_gauss.csv"))
    # np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    # #laplacian of gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--fit_log.csv"))
    # np.savetxt(filename_out, log_list, delimiter =", ", fmt ='% s')
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--fit_sd.csv"))
    # np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--fit_newsd.csv"))
    # np.savetxt(filename_out, newsd_list, delimiter =", ", fmt ='% s')
    
   
    # #overlay original histograms
    # g_list     = []
    # log_list   = []
    # sd_list    = []
    # newsd_list = []
    # for p in tqdm.tqdm(pl):
    #     # print("")
    #     # print(-1*np.min(p.gauss_vec))
    #     # print(np.max(p.lo_gauss_vec))
    #     # print(np.max(p.stddev_vec))
    #     # print(np.max(p.new_stddev_vec))
    #     g_list.append(-1*np.min(p.gauss_vec))
    #     log_list.append(np.max(p.lo_gauss_vec))
    #     sd_list.append(np.max(p.stddev_vec))
    #     newsd_list.append(np.max(p.new_stddev_vec))
        
    # axs[0,0].hist(g_list,     bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(g_list)))
    # axs[0,1].hist(log_list,   bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(log_list)))
    # axs[1,0].hist(sd_list,    bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(sd_list)))
    # axs[1,1].hist(newsd_list, bins=50, edgecolor="black", alpha=alpha, label="max "+str(len(newsd_list)))
    # ''' SAVE CONTRAST CSV '''
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--max_gauss.csv"))
    # np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    # #laplacian of gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--max_log.csv"))
    # np.savetxt(filename_out, log_list, delimiter =", ", fmt ='% s')
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--max_sd.csv"))
    # np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--max_newsd.csv"))
    # np.savetxt(filename_out, newsd_list, delimiter =", ", fmt ='% s')
    
    # #overlay trimmed histograms
    # g_list     = []
    # log_list   = []
    # sd_list    = []
    # newsd_list = []
    # for p in tqdm.tqdm(pl):
    #     # print("")
    #     # print(-1*np.min(p.gauss_vec))
    #     # print(np.max(p.lo_gauss_vec))
    #     # print(np.max(p.stddev_vec))
    #     # print(np.max(p.new_stddev_vec))
    #     if p.gauss_trim_contrast > 0: g_list.append(p.gauss_trim_contrast)
    #     if p.log_trim_contrast > 0: log_list.append(p.log_trim_contrast)
    #     if p.sd_trim_contrast > 0: sd_list.append(p.sd_trim_contrast)
    #     if p.newsd_trim_contrast > 0: newsd_list.append(p.newsd_trim_contrast)
        
    # axs[0,0].hist(g_list,     bins=50, edgecolor="black", alpha=alpha, label="trim "+str(len(g_list)))
    # axs[0,1].hist(log_list,   bins=50, edgecolor="black", alpha=alpha, label="trim "+str(len(log_list)))
    # axs[1,0].hist(sd_list,    bins=50, edgecolor="black", alpha=alpha, label="trim "+str(len(sd_list)))
    # axs[1,1].hist(newsd_list, bins=50, edgecolor="black", alpha=alpha, label="trim "+str(len(newsd_list)))
    # ''' SAVE CONTRAST CSV '''
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--trim_gauss.csv"))
    # np.savetxt(filename_out, g_list, delimiter =", ", fmt ='% s')
    # #laplacian of gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--trim_log.csv"))
    # np.savetxt(filename_out, log_list, delimiter =", ", fmt ='% s')
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--trim_sd.csv"))
    # np.savetxt(filename_out, sd_list, delimiter =", ", fmt ='% s')
    # #gaussian
    # filename_out = os.path.join(constants['output path'], (constants['name']+"contrast--trim_newsd.csv"))
    # np.savetxt(filename_out, newsd_list, delimiter =", ", fmt ='% s')
    
    # axs[0,0].set_title("gaussian")
    # axs[0,1].set_title("laplacian")
    # axs[1,0].set_title("std dev")
    # axs[1,1].set_title("new sd. dev")
    # axs[0,0].legend()
    # axs[0,1].legend()
    # axs[1,0].legend()
    # axs[1,1].legend()
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(constants["output path"], "_fit_histograms.png"))
    # plt.clf()
    

    
    
    # return pl



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# #define a function
# def linear_func(x, a, b, c):
#     return a * np.exp(-b * x) + c
# def fit_line_to_data():
#     #generate some experimental data:
#     xdata = np.linspace(0, 4, 50)
#     y = linear_func(xdata, 2.5, 1.3, 0.5)
#     rng = np.random.default_rng()
#     y_noise = 0.2 * rng.normal(size=xdata.size)
#     ydata = y + y_noise
#     plt.plot(xdata, ydata, 'b-', label='data')
    
    
#     #Fit for the parameters a, b, c of the function func:
#     popt, pcov = curve_fit(linear_func, xdata, ydata)
#     plt.plot(xdata, linear_func(xdata, *popt), 'r-',
#              label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    
#     #Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
#     popt, pcov = curve_fit(linear_func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
#     plt.plot(xdata, linear_func(xdata, *popt), 'g--',
#              label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.show()
    
    
#pl_fit = fit_gaussian_to_particle_list(pl2, constants)

#plot_particle_contrast(pl_fit, constants)
#iscat.generate_particle_list_csv(pl_fit, constants, tag)


#plt.plot(p.gauss_vec)
#plt.show()
    