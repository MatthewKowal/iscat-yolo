# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:48:56 2024

@author: user1
"""

import pandas as pd
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt
import ast

import matplotlib.patches as patches


    # SYSTEM PARAMETERS
    
iscat_hfw = 18.7 #microns
iscat_xdim = 256
iscat_ydim = 256
iscat_ups = 1
iscat_res = iscat_hfw*1000/iscat_xdim

    

    #IMPORT ISCAT DATA

''' GET CURRENT DIRECTORY '''
basepath = os.path.dirname(os.path.abspath(__file__))






#########################  DATASET 1 ##########################################
# print("DATASET 1")
# subfolder = "iscat data"
# samplename = "yolo raw output"
# files = [r"2024-09-30_14-10-21_Particle List__ - 0 yolo raw output.csv",
#           r"2024-09-30_14-13-41_Particle List__ - 0 yolo raw output.csv",
#           r"2024-09-30_14-17-05_Particle List__ - 0 yolo raw output.csv"]
# samplename = "yolo raw output"

# files = [r"2024-09-30_14-10-21_Particle List__ - 1 remove short lived particles.csv",
#           r"2024-09-30_14-13-41_Particle List__ - 1 remove short lived particles.csv",
#           r"2024-09-30_14-17-05_Particle List__ - 1 remove short lived particles.csv"]
# samplename = "remove short lived particles"

# files = [r"2024-09-30_14-10-21_Particle List__ - 2 fit to gauss.csv",
#           r"2024-09-30_14-13-41_Particle List__ - 2 fit to gauss.csv",
#           r"2024-09-30_14-17-05_Particle List__ - 2 fit to gauss.csv"]
# samplename = "fit to gauss"

# files = [os.path.join(basepath, subfolder, f) for f in files] 
###############################################################################

# #########################  DATASET 2 ##########################################
# print("DATASET 2 - 100 nm PS - first SEM deposition")
# subfolder = "iscat data 2"
# samplename = "iscat data 2"

# files = [r"2024-09-30_14-10-21_Particle List__ - 0 yolo raw output.csv",
#          r"2024-09-30_14-13-41_Particle List__ - 0 yolo raw output.csv",
#          r"2024-09-30_14-17-05_Particle List__ - 0 yolo raw output.csv"]
# files = [os.path.join(basepath, subfolder, f) for f in files] 
# ###############################################################################



#########################  DATASET 3 ##########################################
print("DATASET 3 - carraugh flaminia data")
subfolder  = "data"
samplename = "carraugh flaminia"

files = [r"Particle List__2024-10-11_15-39-07_336w_336h_10avg_50fpsgtest.csv"]
files = [os.path.join(basepath, subfolder, f) for f in files] 

# SYSTEM PARAMETERS    
iscat_hfw = 12#18.7 #microns
iscat_xdim = 256
iscat_ydim = 256
iscat_ups = 1
iscat_res = iscat_hfw*1000/iscat_xdim


###############################################################################



def getdata(cellcontents): #convert a string of numbers in a spreadsheet cell into an array
    return ast.literal_eval(cellcontents.strip())

    # GENERATE PARTICLE COORDINATE LIST: iscat_list
print("Generating Particle Data List...")
#make a of iscat particle coordinates by combining (appending)
# all the coordinates from the iscat files
pl = []
for f in files: #go through each file
    df = pd.read_csv(f, index_col=False)
    print("\trows", len(df))
    for i in range(len(df)): # go through each row in the spreadsheet
        PARTICLEOK=True
        
        #scalars
        pid  = df["pID"][i]                 # 0
        life = df["lifetime"][i]            # 5
        sdmax= df["stddev max"][i]          # 8
        gamax= np.max(getdata(df["contrast vec"][i]))       # 9
        
        #vectors
        f    = getdata(df["frames"][i])       # 1
        x    = getdata(df["px list"][i])      # 2
        y    = getdata(df["py list"][i])      # 3
        gavec= getdata(df["contrast vec"][i])        # 6
        sdvec= getdata(df["stddev list"][i])         # 7
        conf = getdata(df["Conf"][i])        # 4
        
        
        
        
        
        # gx   = getdata(df["g x list"][i])   # 4
        # gy   = getdata(df["g y list"][i])   # 5
        # conf = getdata(df["Conf"][i])       # 6
        # life = df["lifetime"][i]            # 7
        # sdmax= df["std max"][i]             # 8
        # gamax= df["gauss max"][i]           # 9
        # sd contrast vector
        # gauss amp
        
        newp = {"pID":   pid,
                "life":  life,
                "sdmax": sdmax,
                "gamax": gamax,
                
                "f":     f,
                "x":     x,
                "y":     y,
                "conf":  conf,
                "sdvec": sdvec,
                "gavec": gavec
                }
        #           0   1  2  3   4      5     6      7      8      9
        pl.append(newp)#[pid, f, x, y, conf, life, sdvec, gavec, sdmax, gamax])#, gx, gy, conf, life, sdmax, gamax]) #append the particle data to a particle list
        
print("\niSCAT info:")
print("\tiSCAT dimensions (x,y):\t", iscat_xdim, iscat_ydim)
print("\tiSCAT hwf: \t\t\t\t", iscat_hfw, "microns")
print("\tiSCAT nm/px: \t\t\t", iscat_res)

#%%

#%% 
''' DATA INSPECTION PLOTS '''
#plot gauss contrast vs SD contrast. This data should track as a line

# 0 pID
# 1 f
# 2 x
# 3 y
# 4 conf
# 5 life
# 6 gavec
# 7 sdvec
# 8 sdmax
# 9 gamax

plt.figure(figsize=(10,10), dpi=300)


life_ = [p["life"] for p in pl]
sdmax_ = [p["sdmax"] for p in pl]
gamax_ = [p["gamax"] for p in pl]
plt.scatter(gamax_, sdmax_, s=1)

print(gamax_)
# m = -0.22
# b = -0.002
# x = np.linspace(-0.1, 0.1, 100)
# y = m * x + b
# plt.plot(x, y, color="red")
# plt.xlim(-0.55, 0.1)
# plt.ylim(0, 0.035)

plt.xlabel("gauss contrast")
plt.ylabel("sd contrast")

#plt.xticks([])

plt.show()




#%%
#plot gauss contrast vs SD contrast - WITH PARTICLE FILTER

plt.figure(figsize=(8,6), dpi=300)


#plot all particles
life_  = [p[7] for p in pl]
sdmax_ = [p[8] for p in pl]
gamax_ = [p[9] for p in pl]
plt.scatter(gamax_, sdmax_, s=1, alpha=0.8, color="blue")

#plot line
m = -0.22
b = -0.002
x = np.linspace(-1, 1, 100)
y = m * x + b
plt.plot(x, y, alpha=0.5, color="red")

#plot filtered particles
pl2 = []
#pl_ = [p for p in pl if p[9] > -0.1] # remove if gauss contrast is < -0.1
pl2 = [p for p in pl if p[8] < m * p[9] + b] # remove if sdcontrast < m * x + b = -0.22 * gausscontrast - 0.002 
life_2  = [p[7] for p in pl2]
sdmax_2 = [p[8] for p in pl2]
gamax_2 = [p[9] for p in pl2]
plt.scatter(gamax_2, sdmax_2, s=1, alpha=0.8, color="orange")

txt = [print(p[0]) for p in pl2]
plt.xlabel("gauss contrast")
plt.ylabel("sd contrast")
plt.xlim(-0.55, 0.1)
plt.ylim(0, 0.035)
plt.show()

''' i could say: if gauss contrast < -0.1, then particle is false '''

#%%


life_ = [p[7] for p in pl]
sdmax_ = [p[8] for p in pl]
gamax_ = [p[9] for p in pl]

plt.figure(figsize=(8,6), dpi=300)
plt.scatter(gamax_, life_, s=1)
plt.xlabel("gauss contrast")
plt.ylabel("lifetime")
plt.show()
''' i could say: if gauss contrast < -0.1, then particle is false '''
#%%

 #yolo confidence vs lifetime
life_ = [p[7] for p in pl]
sdmax_ = [p[8] for p in pl]
gamax_ = [p[9] for p in pl]
cfmax_ = [np.max(p[6]) for p in pl]
                 
plt.figure(figsize=(8,6), dpi=300)
plt.scatter(cfmax_, life_, s=1)
plt.xlabel("yolo confidence max")
plt.ylabel("lifetime")
plt.show()

''' i could say: lifetime thresh = 25 or 50
                 yolo conf thresh = 0.4 '''

#%%


''' APPLY PARTICLE SELECTION FILTER FILTER '''

#keep if gauss contrast > -0.1
pl_ = [p for p in pl if p[9] > -0.1]
pl = pl_

#keep if lifetime > 25 or 50
pl_ = [p for p in pl if p[7] > 25]
pl = pl_

#keep if yolo conf > 0.4
pl_ = [p for p in pl if np.max(p[6]) > 0.4]
pl = pl_

#%%

''' PLOT THE MOST CONFIDENT POSITION '''
#make a list of coordinates of particles positioned during their most confident frame
framemax = [np.argmax(p[6]) for p in pl]

xf_, yf_ = [], []
for i, p in enumerate(pl):
    xf_.append(p[1][framemax[i]])
    yf_.append(p[2][framemax[i]])
plt.figure(figsize=(6,6), dpi=300)    
plt.scatter(xf_, yf_, s=0.1)
plt.show()
#%%
# life_ = [p[7] for p in pl]
# sdmax_ = [p[8] for p in pl]
# gamax_ = [p[9] for p in pl]
# cfmax_ = [np.max(p[6]) for p in pl]
                 
# plt.figure(figsize=(8,6), dpi=300)
# plt.scatter(cfmax_, gamax_, s=1)
# plt.xlabel("yolo confidence max")
# plt.ylabel("lifetime")
# plt.show()



#%%

#plot confidence for every particle

#for p in pl:
#    print(p[6])
    
# fig, ax = plt.subplots(figsize=(6,6), dpi=300)
# for p in pl[:1000]:
#     ax.plot(p[6], linewidth=1)
# plt.show()

#%%

# ANIMATE LIVE PARTICLE POSITIONS DURING RATIOMETRIC DEPOSITION

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example: List of frames, each frame containing a list of [x, y] coordinates
# This is what the data should look like. Replace this with your actual data
# data = [
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]]
#     ]

#now make data from the particle list

#find last frame
flist = [p[1] for p in pl]
flist_ = [item for sublist in flist for item in sublist]
max_frame = np.max(flist_)

frames = [[[0,0]] for f in range(max_frame+1)] #make an empty set of frames containing particle positions

#stuck_particles = frames

for p in pl: #loop through each particle
    
    for i in range(len(p[1])): #loop through each particle frame
        #print(p[1][i], p[2][i], p[3][i], p[4][i], p[5][i])
        f = p[1][i]
        x = p[2][i]
        y = p[3][i]
        frames[f].append([x, y])
    #after the last frame, find the final resting position of the particle
    #and add it to the stuck particle like at this frame
data1 = frames[:200] #trim to some smaller number of frames for testing
#data2 = frames[:200]

# Convert data to numpy arrays for easier manipulation
data1 = [np.array(frame) for frame in data1]
#data2 = [np.array(frame) for frame in data2]

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
scat1 = ax.scatter([], [], color='b', s=1, alpha=0.5, label='Plot 1')
#scat2 = ax.scatter([], [], color='r', s=2, alpha=0.5, label="Plot 2")  # Second scatter plot in red


# Set axis limits (adjust according to your data)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)

# Update function for the animation
def update(frame):
    scat1.set_offsets(frame)
    #scat.set_offsets(frame[0])  # Update first scatter plot data with data from frame[0]
    #scat.set_offsets(frame[1])  # Update second scatter plot data with data from frame[1]
    
    return scat1,

#plot_frames = zip(data1, data2)
plot_frames = data1

# Create the animation
ani = FuncAnimation(fig, update, frames=plot_frames, blit=True)


# Save the animation as a movie file
#ani.save('scatter_animation.mp4', writer='ffmpeg', fps=5)


# Save the animation as a movie file
ani.save(os.path.join(basepath, 'live particles only.mp4'), writer='ffmpeg', fps = 10)


#%%   

# make a particle landing video from plots, including final position of particles

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example: List of frames, each frame containing a list of [x, y] coordinates
# This is what the data should look like. Replace this with your actual data
# data = [
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]],
#     [[1, 2], [2, 3], [3, 1]],
#     [[2, 3], [3, 4], [4, 2]]
#     ]

#now make data from the particle list

#find last frame
flist = [p[1] for p in pl]
flist_ = [item for sublist in flist for item in sublist]
max_frame = np.max(flist_)



    # MAKE DATA FOR LIVE PARTICLES
frames = [[[0,0]] for f in range(max_frame+1)] #make an empty set of frames containing particle positions
for p in pl:    
    for i in range(len(p[1])): #loop through each particle frame
        #print(p[1][i], p[2][i], p[3][i], p[4][i], p[5][i])
        f = p[1][i]
        x = p[2][i]
        y = p[3][i]
        frames[f].append([x, y])

    
    # MAKE DATA FOR DEAD PARTICLES
#MAKE FINAL RESTING SPOT DATA. EACH PARTICLE SHOWS UP FOR ONLY ONE FRAME, ITS LAST
frames_f = [[[0,0]] for f in range(max_frame+1)] #make an empty set of frames containing particle positions
for p in pl:    
    ff = p[1][-1]
    fx = int(np.mean(p[2]))
    fy = int(np.mean(p[3]))
    updating_frame = frames_f[ff] #updating_frame is a list of all f_particles placed in the frame so far
    updating_frame.append([fx,fy]) #here, we add the new f_particle to the list
    frames_f[ff] = updating_frame
#MAKE FINAL RESTING SPOT VIDEO DATA "STREAK"
deadlist = []
frames_ff = []
for i, f in enumerate(frames_f):
    #if f==[[0,0]]: #if the new frame doesnt have new particles
        #frames_ff[i] = deadlist
        #print(i, "\t\t\t\t", frames_ff[i])
    if f!=[[0,0]]:
        for coord in f[1:]:
            deadlist.append(coord)
        #print(i, f[1:], "  \t", deadlist)
    frames_ff.append(np.array(deadlist))
    #print(i, frames_ff[i])
    

    #PLOT THE TWO DATA SETS AS AN ANIMATION
#trim to some smaller number of frames for testing
#flimit=2000 
data1 = frames#[:flimit] 
data2 = frames_ff#[:flimit]


# Convert data to numpy arrays for easier manipulation
data1 = [np.array(frame) for frame in data1]
data2 = [np.array(frame) for frame in data2]

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
scat1 = ax.scatter([], [], facecolor='blue', edgecolor=None, s=20, alpha=0.3, label='alive')
scat2 = ax.scatter([], [], color='r', s=0.1, alpha=0.9, label='dead')  # Second scatter plot in red

# Set axis limits (adjust according to your data)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)

# Update function for the animation
def update(frame):
    #scat1.set_offsets(frame)
    scat1.set_offsets(frame[0])  # Update first scatter plot data with data from frame[0]
    scat2.set_offsets(frame[1])  # Update second scatter plot data with data from frame[1]
    return scat1, scat2

plot_frames = zip(data1, data2)
#plot_frames = data1

# Create and save the animation 
ani = FuncAnimation(fig, update, frames=plot_frames, blit=True)
ani.save(os.path.join(basepath, 'iscat particle plot video.mp4'), writer='ffmpeg', fps = 200)





#%%
    #after the last frame, find the final resting position of the particle
    #and add it to the stuck particle like at this frame
    fx_ = int(np.mean(p[2])) # final x, resting place =  mean value
    fy_ = int(np.mean(p[3])) # final y 
    ff_ = p[1][-1] #final frame
    #frames_f[ff_] = frames_f[ff_].append([fx_,fy_]) #add the final resting spot of the particle to the frames_f list.
                                    #next we will "streak" the plot
                                    
    print(p[0], ff_, fx_, fy_, frames_f[ff_])
    
#%%
stuck_particles = [] #this is a running list of all dead particles                                
#frames_f2       = [[[0,0]] for f in range(max_frame+1)]

for i, f in enumerate(frames_f):
    if f != [[0,0]]: print(f)
    #stuck_particles.append(f) # f = position of a new "final" particle in the frame

    #frames_f2[i] = stuck_particles

#%%
#trim to some smaller number of frames for testing
data = frames[:200] 
data2 = frames_f2[:200]

# Convert data to numpy arrays for easier manipulation
data = [np.array(frame) for frame in data]
data2 = [np.array(frame) for frame in data2]


# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(4,4), dpi=300)

# create scatter plots
scat = ax.scatter([], [], s=10, alpha=0.6, color="orange")
scat2 = ax.scatter([], [], s=50, alpha=0.1, color="blue")

# Set axis limits (adjust according to your data)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)

# Update function for the animation
def update(frame):
    
    
    scat.set_offsets(frame[0])  # Update scatter plot data with new frame
    scat2.set_offsets(frame[1])  # Update scatter plot data with new frame
    
    return scat, scat2

plot_frames = zip(data1, data2) # THIS GIVES A WARNING

# Create the animation
ani = FuncAnimation(fig, update, frames=plot_frames, blit=True)

# Save the animation as a movie file
#ani.save('scatter_animation.mp4', writer='ffmpeg', fps=5)


# Save the animation as a movie file
ani.save(os.path.join(basepath, 'animation.mp4'), writer='ffmpeg', fps = 10)


#%%
#  display iSCAT particles

fig, ax = plt.subplots(figsize=(10, 10*(iscat_ydim/iscat_xdim)), dpi=300)

print("plotting iSCAT particle coordinates")
iscat_px = [p[2] for p in pl] #broken
iscat_py = [p[3] for p in pl] #broken
plt.scatter(iscat_px, iscat_py, s = 1, color="red")

#set bounds
plt.xlim(0, iscat_xdim)
plt.ylim(0, iscat_ydim)

#manage axis stuff
plt.gca().invert_yaxis()
ax.set_xlabel("pixel")
ax.set_ylabel("pixel")

ax2 = ax.twiny()
micron_scale = np.linspace(0,iscat_hfw, 5)#.astype(int)
ax2.set_xticks(micron_scale)
ax2.set_xlabel("micron")

ax3 = ax.twinx()
micron_scale = np.linspace(0,iscat_hfw*iscat_ydim/iscat_xdim, 5)
ax3.set_yticks(micron_scale)
ax3.set_ylabel("micron")

ax.set_title("iSCAT Particle Deposition Positions :: "+samplename)

outputpath = os.path.join(basepath, "iSCAT positions - "+samplename+".png")
print(outputpath)
fig.savefig(outputpath, dpi=300, bbox_inches='tight')
plt.show()

print(np.max(iscat_px))
print(np.max(iscat_py))




#%%


#convert pixel positions to micron positions

iscat_xdim_microns = iscat_xdim*iscat_res/1000
iscat_ydim_microns = iscat_ydim*iscat_res/1000

iscat_px_microns = [x*iscat_res/1000 for x in iscat_px]
iscat_py_microns = [y*iscat_res/1000 for y in iscat_py]




#%%


#  display iSCAT particles with a randomly places square
print("plotting iSCAT particle coordinates + Particle Subset")


USESUBSET = False

# micron plot dimensions
iscat_xdim_microns = iscat_xdim*iscat_res/1000
iscat_ydim_microns = iscat_ydim*iscat_res/1000


#get particle coordinates
iscat_px = [coords[0] for coords in iscat_list]
iscat_py = [coords[1] for coords in iscat_list]

# convert pixel positions to micron positions
iscat_px_microns = [x*iscat_res/1000 for x in iscat_px]
iscat_py_microns = [y*iscat_res/1000 for y in iscat_py]
#iscat_list_microns = 
# plot the data
fig, ax = plt.subplots(figsize=(10, 10*(iscat_ydim/iscat_xdim)), dpi=300)
plt.scatter(iscat_px,  iscat_py,  s = 0.1, color="red")


if USESUBSET == True:
    #get subset of particle coordinates
    # draw a random square and pick particles only 
    # from there (square is defined by rx, ry and bbox)
    bbox = 150
    xmin = 20
    ymin = 20
    rx = np.random.randint(iscat_xdim-bbox-2*xmin)+xmin
    ry = np.random.randint(iscat_ydim-bbox-2*ymin)+ymin
    #make a new particle list from the partiles in the box
    iscat_list2 = [c for c in iscat_list if rx < c[0] < rx+bbox and ry < c[1] < ry+bbox] 
    #generate individual x and y coordinate lists
    iscat_px2 = [coords[0] for coords in iscat_list2]
    iscat_py2 = [coords[1] for coords in iscat_list2]
    iscat_px2_microns = [x*iscat_res/1000 for x in iscat_px2]
    iscat_py2_microns = [y*iscat_res/1000 for y in iscat_py2]
    plt.scatter(iscat_px2, iscat_py2, s = 0.1, color="blue")
    #draw the box
    rect = patches.Rectangle((rx, ry), bbox, bbox, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    print("numer of particles (subset):", len(iscat_list2))


#manage axis stuff and show plot
plt.xlim(0, iscat_xdim)
plt.ylim(0, iscat_ydim)
plt.gca().invert_yaxis()
ax.set_xlabel("pixel")
ax.set_ylabel("pixel")
ax2 = ax.twiny()
micron_scale = np.linspace(0,iscat_hfw, 5)#.astype(int)
ax2.set_xticks(micron_scale)
ax2.set_xlabel("micron")
ax3 = ax.twinx()
micron_scale = np.linspace(0,iscat_hfw*iscat_ydim/iscat_xdim, 5)
ax3.set_yticks(micron_scale)
ax3.set_ylabel("micron")
ax.set_title("iSCAT Particle Deposition Positions :: "+samplename)

outputpath = os.path.join(basepath, "iSCAT positions + box - "+samplename+".png")
print(outputpath)
fig.savefig(outputpath, dpi=300, bbox_inches='tight')

plt.show()



print("numer of particles (total):", len(iscat_list))



# PLOT RDF FOR
#    THE CROPPED DATASET
#    THE WHOLE DATASET
#    THE SEM DATASET



import numpy as np
import matplotlib.pyplot as plt

#RDF ALGORITHM
def calculate_rdf(positions, xdim, ydim, max_dist, num_bins):
    # Calculate pairwise distances
    distances = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1)
    # Remove self-distances
    distances = distances[distances != 0]
    # Create bins
    r = np.linspace(0, max_dist, num_bins + 1)
    bin_centers = 0.5 * (r[1:] + r[:-1])
    # Count the number of pairs in each bin
    counts, _ = np.histogram(distances, bins=r)
    # Normalize to get RDF
    volume = np.pi * (r[1:]**2 - r[:-1]**2)  # Area of each shell in 2D
    
    rdf = counts / (len(positions) * volume)  # Normalization factor: N * area of shell
    return bin_centers, rdf

# PARAMETERS
xdim      = iscat_xdim_microns#100  # Width of the image
ydim      = iscat_ydim_microns#100  # Height of the image
max_dist  = np.sqrt(xdim**2 + ydim**2) / 2  # Maximum distance based on image size

#DATA 1: All iSCAT DATA
# positions should be an array of arrays
positions = np.array(list(zip(iscat_px_microns, iscat_py_microns)))
num_bins  = int(len(positions)/10)  # Number of bins for the RDF
bin_centers, rdf = calculate_rdf(positions, xdim, ydim, max_dist, num_bins)

#DATA 2: Crop Square of ISCAT DATA
# # positions should be an array of arrays
positions2 = np.array(list(zip(iscat_px2_microns, iscat_py2_microns)))
num_bins  = int(len(positions2)/10)  # Number of bins for the RDF
bin_centers2, rdf2 = calculate_rdf(positions2, xdim, ydim, max_dist, num_bins)


#DATA 3: SEM DATA
sempath = r"C:/Users/user1/Desktop/Lab Notebook/PS 100 SEM particle line up/SEM rdf plot data.csv"
semdata = np.genfromtxt(sempath, delimiter=',', skip_header=1)
sem_bin = [d[0] for d in semdata]
sem_rdf = [d[1] for d in semdata]


# Plot the RDF
plt.figure(figsize=(10, 6))

plt.plot(bin_centers, rdf, label='all data', color='r')
plt.plot(bin_centers2, rdf2, label="crop data:"+samplename, color='b')
#plt.plot(sem_bin, sem_rdf, label='sem data: '+samplename, color='g')


plt.xlabel('Distance (r, microns)')
plt.ylabel('g(r)')
plt.xlim(0,2)
plt.title('Radial Distribution Function (RDF) in 2D - CROPPED DATASET')
plt.grid()
plt.legend()

outputpath = os.path.join(basepath, "RDF DATA "+samplename+".png")
print(outputpath)
plt.savefig(outputpath, dpi=300, bbox_inches='tight')


plt.show()





#%%


    # calculate radial distribution function of the whole dataset

# import numpy as np
# import matplotlib.pyplot as plt


# # PARAMETERS
# # positions should be an array of arrays
# positions = np.array(list(zip(iscat_px_microns, iscat_py_microns)))
# N = len(positions)  # Number of particles
# # Define image dimensions
# xdim = iscat_xdim_microns#100  # Width of the image
# ydim = iscat_ydim_microns#100  # Height of the image
# # define max distance and number of bins
# max_distance = np.sqrt(xdim**2 + ydim**2) / 2  # Maximum distance based on image size
# num_bins = 400  # Number of bins for the RDF



# #RDF ALGORITHM
# # Calculate pairwise distances
# distances = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1)
# # Remove self-distances
# distances = distances[distances != 0]
# # Create bins
# r = np.linspace(0, max_distance, num_bins + 1)
# bin_centers = 0.5 * (r[1:] + r[:-1])
# # Count the number of pairs in each bin
# counts, _ = np.histogram(distances, bins=r)
# # Normalize to get RDF
# volume = np.pi * (r[1:]**2 - r[:-1]**2)  # Area of each shell in 2D
# rdf = counts / (N * volume)  # Normalization factor: N * area of shell



# # Plot the RDF
# plt.figure(figsize=(10, 6))
# plt.plot(bin_centers, rdf, label='Radial Distribution Function', color='b')
# plt.xlabel('Distance (r, microns)')
# plt.ylabel('g(r)')
# plt.xlim(0,2)
# plt.title('Radial Distribution Function (RDF) in 2D - WHOLE DATASET')
# plt.grid()
# plt.legend()
# plt.show()






#%%
    # calculate radial distribution function of the cropped dataset

# import numpy as np
# import matplotlib.pyplot as plt

# # PARAMETERS
# # positions should be an array of arrays
# positions = np.array(list(zip(iscat_px2_microns, iscat_py2_microns)))
# N = len(positions)  # Number of particles
# # Define image dimensions
# xdim = iscat_xdim_microns#100  # Width of the image
# ydim = iscat_ydim_microns#100  # Height of the image
# # define max distance and number of bins
# max_distance = np.sqrt(xdim**2 + ydim**2) / 2  # Maximum distance based on image size
# num_bins = 200  # Number of bins for the RDF


# #RDF ALGORITHM

# # Calculate pairwise distances
# distances = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1)
# # Remove self-distances
# distances = distances[distances != 0]
# # Create bins
# r = np.linspace(0, max_distance, num_bins + 1)
# bin_centers = 0.5 * (r[1:] + r[:-1])
# # Count the number of pairs in each bin
# counts, _ = np.histogram(distances, bins=r)
# # Normalize to get RDF
# volume = np.pi * (r[1:]**2 - r[:-1]**2)  # Area of each shell in 2D
# rdf = counts / (N * volume)  # Normalization factor: N * area of shell


# # Plot the RDF
# plt.figure(figsize=(10, 6))
# plt.plot(bin_centers, rdf, label='Radial Distribution Function', color='b')
# plt.xlabel('Distance (r, microns)')
# plt.ylabel('g(r)')
# plt.xlim(0,2)
# plt.title('Radial Distribution Function (RDF) in 2D - CROPPED DATASET')
# plt.grid()
# plt.legend()
# plt.show()


#%%