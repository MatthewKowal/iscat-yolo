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
#import gc
import os
#import skimage.measure
#from skimage.filters import gaussian 
import cv2
import uuid
import yaml
import tqdm


    
    
'''#####################################################

    ____                    ______                    __  
   / __ \ ___   ___   ____ /_  __/_____ ____ _ _____ / /__
  / / / // _ \ / _ \ / __ \ / /  / ___// __ `// ___// //_/
 / /_/ //  __//  __// /_/ // /  / /   / /_/ // /__ / ,<   
/_____/ \___/ \___// .___//_/  /_/    \__,_/ \___//_/|_|  
    ______        /_/          __   _                     
   / ____/__  __ ____   _____ / /_ (_)____   ____   _____ 
  / /_   / / / // __ \ / ___// __// // __ \ / __ \ / ___/ 
 / __/  / /_/ // / / // /__ / /_ / // /_/ // / / /(__  )  
/_/     \__,_//_/ /_/ \___/ \__//_/ \____//_/ /_//____/   
                                                         
                                                 
                                                       
#####################################################'''
#!!!


def to_real():
    def inner(image):
        return np.real(image)
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

''' OPTICAL FILTERS '''
to_real_dt    = dt.Lambda(to_real)                             # use this if not using phadd. phadd results in a real number and so does this and laters filters require a real number
phadd         = dt.Lambda(phase_adder, ph=lambda: np.random.uniform(1.25*np.pi,1.75*np.pi)) # randomize the phase angle of scattered light. for 0-pi the circle is white. so i choose a value orthogonal to that
wave          = dt.Lambda(generate_sine_wave_2D, p=lambda: np.random.uniform(0.0001,0.002)) # This creates a wavey background emulating stray interference patterns from the optics
gauss_noise   = dt.Gaussian(sigma=lambda: np.random.uniform(noisemin,noisemax))
conj          = dt.Lambda(conjugater)                          # ??





''' ####################################################
               __  __ ____   __    ____                  
               \ \/ // __ \ / /   / __ \                 
                \  // / / // /   / / / /                 
                / // /_/ // /___/ /_/ /                  
               /_/ \____//_____/\____/                   
    ______                     __   _                    
   / ____/__  __ ____   _____ / /_ (_)____   ____   _____
  / /_   / / / // __ \ / ___// __// // __ \ / __ \ / ___/
 / __/  / /_/ // / / // /__ / /_ / // /_/ // / / /(__  ) 
/_/     \__,_//_/ /_/ \___/ \__//_/ \____//_/ /_//____/  
                                                        
######################################################'''

def save_yolo_example(image, positions, dataset_path, dynamic_range, yolo_box_size):
    image_dim = image.shape[0]


    # Generate a unique filename
    file_id = str(uuid.uuid4())
    img_filename = f"{file_id}.png"
    label_filename = f"{file_id}.txt"

    # Save image (convert to 8-bit if needed)
    #img_8bit = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    #clipmin, clipmax = 0.97, 1.03
    
    #dynamicrange = 0.03
    clipmin, clipmax = 1-dynamic_range, 1+dynamic_range
    #print("clipmin, clipmax ", clipmin, clipmax)

    image8 = np.clip( ((image - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(dataset_path, "images", img_filename), image8)

    # Convert positions to YOLO format and save
    with open(os.path.join(dataset_path, "labels", label_filename), "w") as f:
        for (y, x) in positions:
            # Convert from pixels to normalized
            x_center = x / image_dim
            y_center = y / image_dim
            w_norm = yolo_box_size*2 / image_dim
            h_norm = yolo_box_size*2 / image_dim

            # YOLO format: class x_center y_center width height
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def create_yaml(dataset_path):
    #create a yaml file to declare class names, number of classes and the location of the database
    #this could be split into train, test, val now or later. 
    data = {
        "train": "../images",  # You can split into train/val later
        "val": "../images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False)






''' GENERATE YOLO DATABASE FROM DEEPTRACK '''
''' GENERATE YOLO DATABASE FROM DEEPTRACK 

   ______                                 __         
  / ____/___   ____   ___   _____ ____ _ / /_ ___    
 / / __ / _ \ / __ \ / _ \ / ___// __ `// __// _ \   
/ /_/ //  __// / / //  __// /   / /_/ // /_ /  __/   
\____/ \___//_/ /_/ \___//_/    \__,_/ \__/ \___/    
__  __ ____   __    ____            ____             
\ \/ // __ \ / /   / __ \   _   __ ( __ )            
 \  // / / // /   / / / /  | | / // __  |            
 / // /_/ // /___/ /_/ /   | |/ // /_/ /             
/_/ \____//_____/\____/    |___/ \____/              
    ____          __          __                     
   / __ \ ____ _ / /_ ____ _ / /_   ____ _ _____ ___ 
  / / / // __ `// __// __ `// __ \ / __ `// ___// _ \
 / /_/ // /_/ // /_ / /_/ // /_/ // /_/ /(__  )/  __/
/_____/ \__,_/ \__/ \__,_//_.___/ \__,_//____/ \___/ 
                                                    

    GENERATE YOLO DATABASE FROM DEEPTRACK '''
''' GENERATE YOLO DATABASE FROM DEEPTRACK '''

# Define Particle Classes
CLASS_NAMES = ["particle"]  # Edit if more classes
dataset_name = "new_dataset"
# Create YOLO folder structure
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, dataset_name)
os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)
create_yaml(dataset_path)




''' GENERATE YOLO DATABASE '''
database_size = 10000
print("\nGenerating YOLOv8 Database from DeepTrack simulated iSCAT images")
print("Producing ", database_size, " images \n")
for i in tqdm.tqdm(range(database_size)):
    
    ''' Dynamic Range Settings '''
    dynamic_range = 0.03
    clipmin, clipmax = 1-dynamic_range, 1+dynamic_range
    #print("clipmin, clipmax ", clipmin, clipmax)

    
    
    ''' PARAMETERS FOR THE OPTICAL SIMULATION '''
    image_size       = 256#512
    ''' image padding '''
    padding          = 16
    ''' why does is make 256,256 sized images when i set image_size to 512 ???!? '''
    offset_px        = 16
    
    ''' Setting up the optical system '''
    #horisontal_coma_range = np.array([-100, 100]) 
    NA               = 1.32
    working_distance = 0.17e-3
    wavelength       = 532e-9
    resolution       = 140# 0e-9#340e-9 nm/pixel
    resolution       *= 1e-9
    magnification    = 3
    
    ''' noise paramaters '''
    noisemin         = 0.005
    noisemax         = 0.007
    

    ''' define an optical system '''
    optics=dt.Brightfield(
        NA                  = NA,
        working_distance    = working_distance,
        aberration          = dt.Astigmatism(coefficient=5),
        wavelength          = wavelength,
        resolution          = resolution,
        magnification       = magnification,
        output_region       = (0,0,image_size,image_size),
        padding             = (padding,) *4,#(image_size//2,) * 4, #what is padding? is it the same as offset_px?
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
    radius_range           = np.array([50,100])*1e-9
    refractive_index_range = np.array([1.38, 1.5])
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
    
    
    
    ''' Generate the optical environment '''
    '''s0 is a microscop object: <deeptrack.optics.Microscope object '''
    num_p = np.random.randint(1,8) # pick a random number of particles
    s0 = optics(particle^num_p)
    
    ''' sample is a deeptack feature chain: <deeptrack.features.Chain object'''
    ''' This is basically an optical filter stack '''
    sample = s0 >> phadd >> conj >> wave >> gauss_noise #>> dst # dt.Gaussian(sigma=lambda: np.random.uniform(0.002,0.04))
    
    ''' This resolves the image '''
    image = sample(return_properties=True)
    
    
    ADD_PARTICLE_TO_DATABASE=True
    if ADD_PARTICLE_TO_DATABASE:
        positions = np.array(image.get_property("position", get_one=False))
        save_yolo_example(image, positions, dataset_path, dynamic_range, yolo_box_size=offset_px)
         
    
    
    
    SHOW_LABELLED_PARTICLE_IMAGE=False
    if SHOW_LABELLED_PARTICLE_IMAGE:
        #get paricle info
        print(f"number of particles requested: {num_p}")
        #for i in range(len(image.properties)):
        #this_particle = image.properties[0]
        positions = np.array(image.get_property("position", get_one=False))
        zs        = np.array(image.get_property("z", get_one=False))
        radii     = np.array(image.get_property("radius", get_one=False))
        prtcl_Ns  = np.array(image.get_property("refractive_index", get_one=False)) 
        
        #clip image to 8 bit for labelling and saving
        label_image = np.clip( ((image - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
        #label_image = np.array(image[:,:,0])
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)  # convert to RGB for matplotlib
        
        for i in range(num_p):    
            bbox=offset_px
            x1 = int(positions[i,1]-bbox)
            y1 = int(positions[i,0]-bbox)
            x2 = int(positions[i,1]+bbox)
            y2 = int(positions[i,0]+bbox)
            
            cv2.rectangle(label_image, (x1, y1), (x2, y2), (200, 0, 200), 1)
            cv2.putText(label_image, str(i), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 1)
        
        plt.imshow(label_image, cmap='gray')
        plt.show()
        
    PRINT_PARTICLE_INFO=False
    if PRINT_PARTICLE_INFO:
        positions = np.array(image.get_property("position", get_one=False))
        zs        = np.array(image.get_property("z", get_one=False))
        radii     = np.array(image.get_property("radius", get_one=False))
        prtcl_Ns  = np.array(image.get_property("refractive_index", get_one=False)) 
        for i in range(num_p):
            print("\t PARTICLE: ", i)
            print(f"Particle position:        {positions[i]}")
            print(f"Particle z:               {zs[i]:.2f}")
            print(f"Particle radius:          {radii[i]*1e9:.2f} nm")
            print(f"Part. refractive index:   {prtcl_Ns[i]:.2f}")
            print("\n")
    
    


 
