# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:15:43 2024

Erik Oslen wrote the first itteration of this iSCAT image simulation script. 

Matt Kowal edited code with the following intent:
    1. Simulate iSCAT images which are representative of our experimental ratiometric images
    2. Train a NN which predicts the position of our experimental data

Deeptrack Documentation
https://deeptrack-20.readthedocs.io/en/latest/installation.html

Deeptrack PyPi
https://pypi.org/project/deeptrack/

Deeptrack Github
https://github.com/DeepTrackAI/DeepTrack2


@author: Matt
"""

''' CURRENT QUESTIONS / THINGS TO FIX
1. why does is make 256,256 sized images when i set image_size to 512 ???!?
2. sampletot.update()() doesnt work when using multiple particles
3. what are sample2 and sample3 used for? the comments say labels and masks, but sample3 looks more like a label to me


Things I think I should tweak:
    1. Add a mottled sort of background to match the coverslip roughness
    2. DONExxxAdd some random level of interference waves in the background (see: r8[692])
    3. Tone down the intensity of the rings so they match the milder AOBD pattern
    4. DONETHISSEEMSOKNOWxxxMake the center of the simulated particles more circular

Text header created using:
    Text to ASCII Art Generator (TAAG)
    http://www.patorjk.com/software/taag/
    Font: Slant
    Char Width: Fitted
    Char Height: Fitted
'''

import deeptrack as dt
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import skimage.measure
from skimage.filters import gaussian 



    
    
'''#####################################################

    ____                ______                __  
   / __ \___  ___  ____/_  __/________ ______/ /__
  / / / / _ \/ _ \/ __ \/ / / ___/ __ `/ ___/ //_/
 / /_/ /  __/  __/ /_/ / / / /  / /_/ / /__/ ,<   
/_____/\___/\___/ .___/_/ /_/  _\__,_/\___/_/|_|  
   / ____/_  __/_/_  _____/ /_(_)___  ____  _____ 
  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/ 
 / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )  
/_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/   
                                                 
                                                       
#####################################################'''
#!!!
def zscore(image):
    #zero center and divide by standard deviation
    image = np.real(image)
    
    mean = np.mean(image)
    mean = 1 #shortcut/assumption
    std  = np.std(image.astype(np.float32)) #convert to a floating point for this to avoid precision overflow of 16 bit floating point
    zscore_image = (image-mean)/std
    # print(mean)
    # print(std)
    # print(np.mean(zscore_image))
    # print(np.std(zscore_image))
    return zscore_image
    
def get_positions(image): 
    return np.array(image.get_property("position", get_one=False))

def get_z_positions(image):
    return np.array(image.get_property("z", get_one=False))

def get_size(image):
    return np.array(image.get_property("radius"))#, get_one=False)

# normalization function :: ERIKS CODE, CAN DELETE
# def batch_function(image):
#     return zscore(image)
#     #return np.real((image[0]-1)/np.std(image[0])) # removed abs

def generate_label(image, include_gauss = True):
    #concatenate the images
    #return np.concatenate((np.abs(image[1]-1)/np.std(image[0]), np.abs(image[2])), axis=2)
    if include_gauss:
        return np.concatenate((np.abs(image[1]-1)/np.std(image[0]), np.abs(image[2])), axis=2)
    else:
        return np.abs(image[1]-1)/np.std(image[0])

def gaussian_filter(width):
    def apply_guassian_filter(image):
        NORM_FACTOR = 100
        image = np.pad(image, [(1, 1), (1, 1), (0, 0)])
        image = gaussian(image, width, truncate=2.5) * NORM_FACTOR
        image = image[1:-1, 1:-1]
        return image
    return apply_guassian_filter

def to_real():
    def inner(image):
        return np.real(image)
    return inner

#this is a downsampling function but im going to get rid of it
def ds(factor=2):
    def inner(image):
        return skimage.measure.block_reduce(image, (factor, factor, 1), np.mean)
    return inner

def to_mask(mask_size=2):
    def inner(image):
        new_image=np.zeros(image.shape)
        pos = image.get_property("position")
        x = int(pos[0])
        y = int(pos[1])
        new_image[x-mask_size//2:x+mask_size//2,y-mask_size//2:y+mask_size//2]=1
        return new_image
    return inner

def generate_sine_wave_2D(p):
    """
    Generate a 2D sine wave pattern with adjustable direction.

    Parameters:
    - N: The size of the square image (N x N).
    - frequency: The frequency of the sine wave.
    - direction_degrees: The direction of the wave in degrees.

    Returns:
    - A 2D numpy array representing the sine wave pattern.
    """
    def inner(image):
        N = image.shape[0]
        frequency = np.random.uniform(4, 10)
        direction_degrees = np.random.uniform(0,180) # changed
        warp_factor = np.random.uniform(0, 0.5) # changed
        
        x = np.linspace(-np.pi, np.pi, N)
        y = np.linspace(-np.pi, np.pi, N)

        # Convert direction to radians
        direction_radians = np.radians(direction_degrees)

        # Calculate displacement for both x and y with warping
        warped_x = x * np.cos(direction_radians) + warp_factor * np.sin(direction_radians * x)
        warped_y = y * np.sin(direction_radians) + warp_factor * np.sin(direction_radians * y)

        # Generate 2D sine wave using the warped coordinates
        sine2D = np.sin((warped_x[:, np.newaxis] + warped_y) * frequency)# 128.0 + (127.0 * np.sin((warped_x[:, np.newaxis] + warped_y) * frequency))
        # sine2D = sine2D / 255.0

        #flip or mirror the pattern
        if np.random.rand()>0.5:
            sine2D=np.flip(sine2D,0)
        if np.random.rand()>0.5:
            sine2D=np.flip(sine2D,1)
        if np.random.rand()>0.5:
            sine2D=np.transpose(sine2D)
        image = image + np.expand_dims(sine2D, axis = -1)*p
        return image
    return inner

#PHASE ADDER??
def phase_adder(ph):
    def inner(image):
        image=image-1
        image=image*np.exp(1j*ph)
        image=image+1
        return np.abs(image)
    return inner

#CONJUGATOR??
def conjugater():
    def inner(image):
        if np.random.rand()>0.5:
            image=np.conj(image-1)
            image=image+1
            return image
        else:
            return image
    return inner




'''  ###############################################3
    ____  __      __  __  _                      
   / __ \/ /___  / /_/ /_(_)___  ____ _          
  / /_/ / / __ \/ __/ __/ / __ \/ __ `/          
 / ____/ / /_/ / /_/ /_/ / / / / /_/ /           
/_/   /_/\____/\__/\__/_/_/ /_/\__, /            
    ______                 __ /____/             
   / ____/_  ______  _____/ /_(_)___  ____  _____
  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
 / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  ) 
/_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/  
                                                
######################################################'''
#!!!
def plot_sim_exp(sim, exp, plot_histograms=False, downsize=False):
    r16 = exp
    sample = sim
    
    # compared simulated data to experimental data
    print("\nExperimental Data Specs:\n")
    print("--------------------")
    #fnum = -1*np.random.randint(0,100)      # use this if you watn to pick a random frame of just noise
    #fnum=45#343, 45 #this is a weird one where the stddev overflows
    fnum = np.random.randint(0,len(r16))       # pick a random frame
    imageexp = r16[fnum]    #use that frame as the experimental data
    print("imageexp", imageexp.shape)
    imageexp[0,0] = 1  #this is all weird non-image data from the camera
    imageexp[0,1] = 1  #this is all weird non-image data from the camera
    imageexp[0,2] = 1  #this is all weird non-image data from the camera
    imageexp[0,3] = 1  #this is all weird non-image data from the camera
    imageexp_clip = np.clip(imageexp, clipmin, clipmax)  #make a clipped version
    imageexp_zscore = zscore(imageexp)                   #make a zscore version
    print("Experimental Images length: ", len(r16))
    print("Frame number used: ", fnum)
    print("Frame max value: ", np.max(imageexp))
    print("Frame min value: ", np.min(imageexp))
    #print("sorted image value:\n\t", np.sort(imageexp.flatten()))
    #print(imageexp[0:10, 0:10])
    
    #simulated data
    print("\nSimulated Data Specs:\n")
    print("--------------------")
    im=sample.update()()
    imagesim = np.real(im)
    print("imagesim", imagesim.shape)
    if downsize==False: imagesim = np.reshape(imagesim, [image_size, image_size])
    if downsize==True:  imagesim = np.reshape(imagesim, [image_size//2, image_size//2])
    print("imagesim", imagesim.shape)
    imagesim_clip = np.clip(imagesim, clipmin, clipmax)
    imagesim_clip[0,0] = clipmin
    imagesim_clip[0,1] = clipmax
    imagesim_zscore = zscore(imagesim)
    print("position:       ", im.get_property("position"))
    print("z-position:     ", im.get_property("z"))
    print("radius:         ", image.get_property("radius"))
    print("image stddev:   ", np.std(imagesim))
    print("section stddev: ", np.std(imagesim[0:29, 0:29]))
    print("pixel size:     ", image.get_property("pixel_size"))

    #plotting function
    gx = 2 #grid size
    gy = 3
    fig,ax=plt.subplots(gy,gx,figsize=(10,10), dpi=300)
    plt.rc('font', size=10)
    plt.rc('figure', titlesize=10)
    plt.rc('ytick', labelsize=10)
    q1 = ax[0,0].imshow(imagesim)
    q2 = ax[0,1].imshow(imageexp)
    q3 = ax[1,0].imshow(imagesim_clip)
    q4 = ax[1,1].imshow(imageexp_clip)
    q5 = ax[2,0].imshow(imagesim_zscore)
    q6 = ax[2,1].imshow(imageexp_zscore)
    plt.colorbar(q1, shrink=0.82)
    plt.colorbar(q2, shrink=0.82)
    plt.colorbar(q3, shrink=0.82)
    plt.colorbar(q4, shrink=0.82)
    plt.colorbar(q5, shrink=0.82)
    plt.colorbar(q6, shrink=0.82)
    #Make Titles, Embelishments, and Finishing Touches
    ax[0,0].set_xlabel("Deeptrack Sim")
    ax[0,1].set_xlabel(("iSCAT frame "+str(fnum)))
    ax[1,0].set_xlabel("Deeptrack - Scaled")
    ax[1,1].set_xlabel(("iSCAT - Scaled "+str(fnum)))
    ax[2,0].set_xlabel("Deeptrack - zscore")
    ax[2,1].set_xlabel(("iSCAT - zscore "+str(fnum)))
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[2,0].set_xticks([])
    ax[2,0].set_yticks([])
    ax[2,1].set_xticks([])
    ax[2,1].set_yticks([])
    fig.suptitle("simulated image                                          experimental images")
    plt.tight_layout()
    plt.show()
    
    
    #do you want to plot histograms for the data?
    if plot_histograms:
        #plot histograms
        fig,ax=plt.subplots(gy,gx,figsize=(6,6), dpi=300)
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 2}
        SMALL_SIZE = 12
        MEDIUM_SIZE=18
        BIGGER_SIZE=8
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE) 
        ax[0,0].hist(imageexp)
        ax[0,1].hist(imagesim)
        ax[1,0].hist(imageexp_clip)
        ax[1,1].hist(imagesim_clip)
        ax[2,0].hist(imageexp_zscore)
        ax[2,1].hist(imagesim_zscore)
        ax[0,0].set_yticks([])
        ax[0,1].set_yticks([])
        ax[1,0].set_yticks([])
        ax[1,1].set_yticks([])
        ax[2,0].set_yticks([])
        ax[2,1].set_yticks([])
        fig.suptitle("simulated histograms                                          experimental histograms")
        #plt.tight_layout()
        plt.show()
    return

# check each input next to each other for one sample
def plot_labels(sampletot, length):
    sample_bundle=sampletot
    input_data = sample_bundle.update()()
    # in1 = np.real(input_data[0])
    # in2 = np.real(input_data[1])
    # in3 = np.real(input_data[2])
    fig, axs = plt.subplots(1, length, figsize=(15, 5))
    plt.rc('font', size=18)
    for i in range(length):
        #print(i, input_data_idx)
        in_data = np.real(input_data[i])
        axs[i].imshow(in_data)
        axs[i].axis('off')  # Optional: Turn off axis
    axs[0].set_title("simulation")
    axs[1].set_title("label")
    #axs[2].set_title("mask")
    plt.tight_layout()
    plt.show()







'''#####################################################################
    _  _____  ______ ___   ______   _                              
   (_)/ ___/ / ____//   | /_  __/  (_)____ ___   ____ _ ____ _ ___ 
  / / \__ \ / /    / /| |  / /    / // __ `__ \ / __ `// __ `// _ \
 / / ___/ // /___ / ___ | / /    / // / / / / // /_/ // /_/ //  __/
/_/ /____/ \____//_/  |_|/_/    /_//_/ /_/ /_/ \__,_/ \__, / \___/ 
                           __   __                 _ /____/        
      _____ __  __ ____   / /_ / /_   ___   _____ (_)_____         
     / ___// / / // __ \ / __// __ \ / _ \ / ___// // ___/         
    (__  )/ /_/ // / / // /_ / / / //  __/(__  )/ /(__  )          
   /____/ \__, //_/ /_/ \__//_/ /_/ \___//____//_//____/           
         /____/                                                   
###################################################################'''         
#!!!


''' PARAMETERS FOR THE OPTICAL SIMULATION '''
image_size             = 256#512
''' why does is make 256,256 sized images when i set image_size to 512 ???!? '''
offset_px              = 16

''' Setting up the optical system '''
#horisontal_coma_range = np.array([-100, 100]) 
NA               = 1.28
working_distance = 0.17e-3
wavelength       = 532e-9
resolution       = 140# 0e-9#340e-9 nm/pixel
resolution       *= 1e-9
magnification    = 4

''' noise paramaters '''
noisemin         = 0.000000001 #0.005
noisemax         = 0.000000001 #0.007
noisemin         = 0.005
noisemax         = 0.007


''' Dynamic Range Settings '''
#clipmin, clipmax = constants["clipmin"], constants["clipmax"] 
#print("clipmin, clipmax ", clipmin, clipmax)

''' define an optical system '''
optics=dt.Brightfield(
    NA                  = NA,
    working_distance    = working_distance,
    aberration          = dt.Astigmatism(coefficient=5),
    wavelength          = wavelength,
    resolution          = resolution,
    magnification       = magnification,
    output_region       = (0,0,image_size,image_size),
    padding             = (16,) *4,#(image_size//2,) * 4,
    polarization_angle  = lambda: np.random.rand() * 2 * np.pi,
    return_field        = True,
    backscatter         = True,
    illumination_angle  = np.pi,
    )



''' PARAMETERS FOR THE SIMLATED PARTICLES '''
#particle parameters
#radius_range = np.array([10,20])*1e-9 #use this if you pretty much only want to see noise, note: you may get erros if you pick a size thats too small
#r = 20 #max:100
#radius_range           = np.array([r, r+1])*1e-9 #i thought 20/30-100/160 looked good before but now idk. ive used [42, 150] before...
radius_range           = np.array([20,100])*1e-9
refractive_index_range = np.array([1.35, 1.59])
z_range_px             = np.array([0, 3]) #-1, 4 was ok

#define a particle
particle=dt.MieSphere(
    position                = lambda: (np.random.uniform(offset_px,image_size - offset_px), np.random.uniform(offset_px,image_size - offset_px)),
    radius                  = lambda: np.random.uniform(radius_range[0], radius_range[1]),
    refractive_index        = lambda: np.random.uniform(refractive_index_range[0], refractive_index_range[1]), 
    z                       = lambda: np.random.uniform(z_range_px[0], z_range_px[1])*2,
    #position_objective      = (np.random.uniform(-250,250)*1e-6,np.random.uniform(-250,250)*1e-6, np.random.uniform(-15,15)*1e-6),
    position_unit           = "pixel",
    refractive_index_medium = 1.33,
    #intensity = lambda: np.random.randint(1,100)
    )

particle_z=dt.MieSphere(
    position                = lambda: (np.random.uniform(offset_px,image_size - offset_px), np.random.uniform(offset_px,image_size - offset_px)),
    radius                  = lambda: np.random.uniform(radius_range[0], radius_range[1]),
    refractive_index        = lambda: np.random.uniform(refractive_index_range[0], refractive_index_range[1]), 
    z                       = 0,
    position_objective      = (0, 0, 0),
    position_unit           = "pixel",
    refractive_index_medium = 1.33,
    #intensity = lambda: np.random.randint(1,100)
    )

'''s0 is a microscop object: <deeptrack.optics.Microscope object '''
   
#Main particle
MULTIPARTICLE    =  True
if MULTIPARTICLE == False:
    num_p = 1 #number of particles
    s0 = optics(particle)
if MULTIPARTICLE == True:
    num_p = np.random.randint(1,8) # pick a random number of particles
    s0 = optics(particle^num_p)
    s1 = optics(particle_z^num_p)
print(num_p)


''' OPTICAL FILTERS '''
mask          = dt.Lambda(to_mask, mask_size=3)                # DRAWS THE PARTICLE AS A SINGLE POINT
make_gaussian = dt.Lambda(function=gaussian_filter, width=9)   # gaussian noise
to_real_dt    = dt.Lambda(to_real)                             # use this if not using phadd. phadd results in a real number and so does this and laters filters require a real number


phadd         = dt.Lambda(phase_adder, ph=lambda: np.random.uniform(1.25*np.pi,1.75*np.pi)) # ??
wave          = dt.Lambda(generate_sine_wave_2D, p=lambda: np.random.uniform(0.0001,0.002)) # This creates a wavey background emulating stray interference patterns from the optics
gauss_noise   = dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax))
conj          = dt.Lambda(conjugater)                          # ??
dst           = dt.Lambda(ds, factor=2)                       # downsample image with skimage.measure.block_reduce(image, blocksize, func, cval)

''' sample is a deeptack feature chain: <deeptrack.features.Chain object'''
#s0 = optics(particle)
#eriks version
#sample = s0 >> conj >> phadd >> wave >> conj >> dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#minimal version
#sample = s0 >> dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#epd-iscat replica 1
#sample = s0 >> wave >> dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax)) >> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
#epd-iscat replica 2 - not sure its noticably different than replica 1

#image of multiple particles
#sample = s0 >> phadd >> wave >> gauss_noise #>> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
sample = s0 >> phadd >> conj >> wave >> gauss_noise #>> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))

#image  = sample.resolve()
image = sample(return_properties=True)

''' GET PARTICLE INFO'''
print(f"number of particles requested: {num_p}")
print(f"length of image properties: {len(image.properties)}")
#for i in range(len(image.properties)):
this_particle = image.properties[0]
positions = np.array(image.get_property("position", get_one=False))
zs        = np.array(image.get_property("z", get_one=False))
radii     = np.array(image.get_property("radius", get_one=False))
prtcl_Ns  = np.array(image.get_property("refractive_index", get_one=False))   


PRINT_PARTICLE_INFO = False
if PRINT_PARTICLE_INFO:
    for i in range(num_p):
        print("\t PARTICLE: ", i)
        print(f"Particle position:        {positions[i]}")
        print(f"Particle z:               {zs[i]:.2f}")
        print(f"Particle radius:          {radii[i]*1e9:.2f} nm")
        print(f"Part. refractive index:   {prtcl_Ns[i]:.2f}")
        print("\n")
    

import cv2
TEST_PARTICLE_LABEL=True

if TEST_PARTICLE_LABEL:
    clipmin, clipmax = 0.97, 1.03
    label_image = np.clip( ((image - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
    #label_image = np.array(image[:,:,0])
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)  # convert to RGB for matplotlib
    
    for i in range(num_p):
        print("\t PARTICLE: ", i)
        print(f"Particle position:        {positions[i]}")
        print(f"Particle z:               {zs[i]:.2f}")
        print(f"Particle radius:          {radii[i]*1e9:.2f} nm")
        print(f"Part. refractive index:   {prtcl_Ns[i]:.2f}")
        print("\n")

        bbox=19
        x1 = int(positions[i,1]-bbox)
        y1 = int(positions[i,0]-bbox)
        x2 = int(positions[i,1]+bbox)
        y2 = int(positions[i,0]+bbox)
        
        cv2.rectangle(label_image, (x1, y1), (x2, y2), (200, 0, 200), 1)

    
    plt.imshow(label_image, cmap='gray')
    plt.show()

GENERATE_YOLO_DATABASE = True
if GENERATE_YOLO_DATABASE:
    print("ok")
    
#%%

import os
import cv2
import uuid
import yaml
import numpy as np

# Example inputs:
# image = np.random.rand(256, 256)  # Replace with your 256x256 NumPy image
# positions = np.array([[50, 60], [120, 180]])  # Example: 2 objects
# CLASS_NAMES = ["particle"]

CLASS_NAMES = ["particle"]  # Edit if more classes
IMG_SIZE = 256
BOX_SIZE_PIXELS = 16  # width/height of object boxes

# Create YOLO folder structure
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "new dataset")
os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)

def save_yolo_example(image, positions, dataset_path):
    # Generate a unique filename
    file_id = str(uuid.uuid4())
    img_filename = f"{file_id}.png"
    label_filename = f"{file_id}.txt"

    # Save image (convert to 8-bit if needed)
    #img_8bit = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    clipmin, clipmax = 0.97, 1.03
    image8 = np.clip( ((image - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(dataset_path, "images", img_filename), image8)

    # Convert positions to YOLO format and save
    with open(os.path.join(dataset_path, "labels", label_filename), "w") as f:
        for (x, y) in positions:
            # Convert from pixels to normalized
            x_center = x / IMG_SIZE
            y_center = y / IMG_SIZE
            w_norm = BOX_SIZE_PIXELS / IMG_SIZE
            h_norm = BOX_SIZE_PIXELS / IMG_SIZE

            # YOLO format: class x_center y_center width height
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def create_yaml(dataset_path):
    data = {
        "train": os.path.join(dataset_path, "images"),  # You can split into train/val later
        "val": os.path.join(dataset_path, "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False)

# Example usage:
save_yolo_example(image, positions, dataset_path)
create_yaml(dataset_path)

    
#%%


# #print the particle info
# print(f"number of particles requested: {num_p}")
# print(f"length of image properties: {len(image.properties)}")
# for i in range(len(image.properties)):
#     this_particle = image.properties[i]
#     print(f"Particle position:        {this_particle['position']}")
#     print(f"Particle z:               {this_particle['z']:.2f}")
#     print(f"Particle radius:          {this_particle['radius']*1e9:.2f} nm")
#     print(f"Part. refractive index:   {this_particle['refractive_index']:.2f}")


''' PLOT SIMULATED IMAGE '''
# plt.imshow(image, cmap='gray')
# plt.show()


''' PLOT SIMULATED IMAGE AND HISTOGRAM '''
# pixels = image.ravel()
# fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))

# ax_img.imshow(image, cmap='gray')#, vmin=0, vmax=1)
# ax_img.set_title("Grayscale Image")
# ax_img.axis("off")

# # Plot histogram
# ax_hist.hist(pixels, bins=256, color='black')
# ax_hist.set_title("Pixel Intensity Histogram")
# ax_hist.set_xlabel("Intensity")
# ax_hist.set_ylabel("Count")

# plt.tight_layout()
# plt.show()

''' LOAD EXPERIMENTAL DATA '''
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_loc = os.path.join(script_dir, "sample exp data")
import pickle
with open(os.path.join(sample_data_loc, 'r8.pkl'), 'rb') as f:
    r8 = pickle.load(f)
with open(os.path.join(sample_data_loc, 'constants.pkl'), 'rb') as f:
    constants = pickle.load(f)
with open(os.path.join(sample_data_loc, 'r16.pkl'), 'rb') as f:
    r16 = pickle.load(f)
#remove error pixels at the top of corner of the video
for f in range(len(r16)):
    r16[f, 0, 0:4] = 1


dynamicrange = 0.03
clipmin, clipmax = 1-dynamicrange, 1+dynamicrange
print("clipmin, clipmax ", clipmin, clipmax)

'''  ANALYZE BACKGROUND NOISE '''
EXPLORE_NOISE = False
if EXPLORE_NOISE:
    fig, ((ax_simg, ax_simg_zoom),
          (ax_eimg, ax_eimg_zoom)) = plt.subplots(2, 2, figsize=(6, 6))
    
    # Simulated Image
    ax_simg.imshow(image, cmap='gray')#, vmin=0, vmax=1)
    ax_simg.set_title(f"min:{np.min(image):.4f}, max:{np.max(image):.4f}")
    ax_simg.text(-0.3, 0.3, "Simulation", transform=ax_simg.transAxes, rotation=90)
    ax_simg.text(0.3, 1.2, "full image", transform=ax_simg.transAxes)
    # simulated image zoom
    image_roi = image[1:31, 1:31]
    ax_simg_zoom.imshow(image_roi, cmap='gray')#, vmin=0, vmax=1)
    ax_simg_zoom.set_title(f"min:{np.min(image_roi):.4f}, max:{np.max(image_roi):.4f}")
    ax_simg_zoom.text(0.3, 1.2, "zoomed image", transform=ax_simg_zoom.transAxes)
    print("\nSIM IMAGE")
    print("mean: ", np.mean(image_roi))
    print("std: ", np.std(image_roi))
    
    # Test Image
    frame_num = np.random.randint(0, len(r16))
    #frame_num = 971
    print(f"frame_num: {frame_num}") 
    test_image = r16[frame_num]
    #test_image[0, 0:4] = 1
    ax_eimg.imshow(test_image, cmap='gray')#, vmin=0, vmax=1)
    ax_eimg.set_title(f"min:{np.min(test_image):.4f}, max:{np.max(test_image):.4f}")
    ax_eimg.text(-0.3, 0.3, "Experimental", transform=ax_eimg.transAxes, rotation=90)
    #experimental image zoom
    test_image_roi = test_image[0:30, 0:30]#[1:31, 1:31]
    print("\nTEST IMAGE")
    print("mean: ", np.mean(test_image_roi))
    print("std: ", np.std(test_image_roi))
    
    
    ax_eimg_zoom.imshow(test_image_roi, cmap='gray')#, vmin=0, vmax=1)
    ax_eimg_zoom.set_title(f"min:{np.min(test_image_roi):.4f}, max:{np.max(test_image_roi):.4f}")
    
    
    plt.tight_layout()
    plt.show()


''' COMPARE TO 8-bit test images '''
COMPARE_TO_TEST_IMAGES = True
if COMPARE_TO_TEST_IMAGES:
    fig, ((ax_simg, ax_shist, ax_s8img, ax_s8hist),
          (ax_eimg, ax_ehist, ax_e8img, ax_e8hist)) = plt.subplots(2, 4, figsize=(12, 6))
    
    
    # Simulated Image
    spixvals = image.ravel()
    
    ax_simg.imshow(image, cmap='gray')#, vmin=0, vmax=1)
    ax_simg.set_title(f"min:{np.min(image):.4f}, max:{np.max(image):.4f}")
    ax_simg.text(-0.3, 0.3, "Simulation", transform=ax_simg.transAxes, rotation=90)
    ax_simg.text(1.0, 1.2, "Raw Ratio", transform=ax_simg.transAxes)
    ax_simg.text(3.3, 1.2, "8-bit Ratio", transform=ax_simg.transAxes)
    
    
    ax_shist.hist(spixvals, bins=256, color='black')
    ax_shist.set_yticks([])
    
    
    
    # Simulated Image in 8bit
    image8 = np.clip( ((image - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
    s8pixvals = image8.ravel()
    
    ax_s8img.imshow(image8, cmap='gray')#, vmin=0, vmax=1)
    ax_s8img.set_title(f"min:{np.min(image8):.4f}, max:{np.max(image8):.4f}")
    
    ax_s8hist.hist(s8pixvals, bins=256, color='black')
    ax_s8hist.set_xlim([0,255])
    ax_s8hist.set_yticks([])
    
    
    # Test Image
    frame_num = np.random.randint(0, len(r16))
    #frame_num = 971
    print(f"frame_num: {frame_num}")
    test_image = r16[frame_num]
    
    epixvals = test_image.ravel()
    
    ax_eimg.imshow(test_image, cmap='gray')#, vmin=0, vmax=1)
    ax_eimg.set_title(f"min:{np.min(test_image):.4f}, max:{np.max(test_image):.4f}")
    ax_eimg.text(-0.3, 0.3, "Experimental", transform=ax_eimg.transAxes, rotation=90)
    ax_ehist.hist(epixvals, bins=256, color='black')
    ax_ehist.set_yticks([])
    
    # Test Image in 8 bit
    test_image8 = np.clip( ((test_image - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
    e8pixvals = test_image8.ravel()
    
    ax_e8img.imshow(test_image8, cmap='gray')#, vmin=0, vmax=1)
    ax_e8img.set_title(f"min:{np.min(test_image8):.4f}, max:{np.max(test_image8):.4f}")
    ax_e8hist.hist(e8pixvals, bins=256, color='black')
    ax_e8hist.set_xlim([0,255])
    ax_e8hist.set_yticks([])
    
    plt.tight_layout()
    plt.show()




