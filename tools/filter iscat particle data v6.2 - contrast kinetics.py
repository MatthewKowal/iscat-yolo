# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 19:29:59 2022

@author: user1
"""




import os
import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ast
import time


''' GENERATE COLOR ARRAY FROM COLORMAP ''' 
def get_custom_color_array(cmap_name, n_colors):
    color_array = False
    if cmap_name == "geodataviz":
        if n_colors==1:  color_array = '#f0746e'
        if n_colors==5:  color_array = ['#089099','#7ccba2','#fcde9c','#f0746e','#dc3977']
        if n_colors==7:  color_array = ['#045275','#089099','#7ccba2','#fcde9c','#f0746e','#dc3977','#7c1d6f']
        if n_colors==10: color_array = ['#024c66', '#045275', '#089099', '#7ccba2', '#fcde9c', '#f49b78', '#f0746e', '#dc3977', '#7c1d6f', '#5a0e52']
    if color_array == False:
        print("Warning: Unrecognized cmap_name. giving viridis instead")
        color_array = get_pyplot_color_array('viridis', n_colors)
    return color_array
def get_pyplot_color_array(cmap_name, n_colors):
    #'viridis'
    cmap = plt.get_cmap(cmap_name)
    color_array = cmap(np.linspace(0, 1, n_colors))
    return color_array
def get_seaborn_color_array(cmap_name, n_colors):
    import seaborn as sns
    # Blues
    # rocket
    # mako
    # flare
    # crest
    # magma
    # RdYlBu
    # RdYlGn
    # Set3
    # Spectral
    # gist_earth
    # inferno
    print(f"seaborn cmap: {cmap_name}")
    colors = sns.color_palette(cmap_name, n_colors=n_colors)
    return colors
#color_array = get_custom_color_array("geodataviz", 4)
#color_array = get_seaborn_color_array("magma", 4)  # get indexed colors from seaborn
#color_array = get_pyplot_color_array("viridis", 4) # get indexed colors form matplotlib

''' ADD COLOR TO THE CONSOLE '''
STYLE="white"
def cprint(text):
    from rich.console import Console
    console = Console()
    console.print(text, style=STYLE)

    # USE IT LIKE THIS IN A FUNCTION
    # global STYLE
    # STYLE = "#00af87"
    # cprint("Loading binfile into memory...")

    # These are the only colors that seem wotk work in spyder
    # hexcolor = ["#00af87", #grey blue
    #              "#008787", #bright blue
    #              "#5f00af", #dark magenta
    #              "#87d700", #blood red
    #              ]

def banner():
    global STYLE
    STYLE="purple"
    cprint("\n\n\t\t\t iSCAT Particle Filter Script v2\n\n")


'''##################################################################
#####################################################################
                IMPORT DATA
#####################################################################
##################################################################'''

''' READ PARTICLE LIST FILE '''
def make_pl_from_pl_file():
    ''' THIS IS EXPECTING THE USER TO RUN THE FILE FROM THE DATA FOLDER
    THIS MEANS THAT IT WILL LOOK FOR THE **Particle List*.csv** FILE 
    FROM THE SAME DIRECTORY WHERE THE SCRIPT IS '''
    import os
    import glob
    import pandas as pd
    import ast
    
    class Prtcl:
        def __init__(self, pID, px_vec, py_vec, f_vec, std_max, conf_vec, wx_vec, wy_vec):
            self.pID      = pID
            self.px_vec   = px_vec
            self.py_vec   = py_vec
            self.f_vec    = f_vec
            self.std_max  = std_max
            self.conf_vec = conf_vec
            self.wx_vec   = wx_vec
            self.wy_vec   = wy_vec
            
            
    scriptpath, scriptfilename = os.path.split(__file__)
    
    plfilepath = glob.glob(os.path.join(scriptpath, "*Particle List__ - 0 yolo raw output*.csv"))[0] #get the Particle List csv
    global STYLE
    STYLE="orange3"
    cprint("Opening Particle List.csv File...")
    print(f"{plfilepath}\n")
    df = pd.read_csv(plfilepath)
    
    #single valued items
    pIDs      = list(df["pID"])
    std_maxs  = list(df["std max"])
    
    #vectors
    x_vecs    = list(df["px list"].apply(ast.literal_eval))
    y_vecs    = list(df["py list"].apply(ast.literal_eval))
    f_vecs    = list(df["frames"].apply(ast.literal_eval))
    conf_vecs = list(df["Conf"].apply(ast.literal_eval))
    
    wx_vecs   = list(df["wx list"].apply(ast.literal_eval))
    wy_vecs   = list(df["wy list"].apply(ast.literal_eval))
    
    pl = [Prtcl(*i) for i in zip(pIDs, x_vecs, y_vecs, f_vecs, std_maxs, conf_vecs, wx_vecs, wy_vecs)]
    return pl






''' READ CONSTANTS FILE ''' 
def read_constants():
    ''' THIS IS EXPECTING THE USER TO RUN THE FILE FROM THE DATA FOLDER
    THIS MEANS THAT IT WILL LOOK FOR THE **constants.txt** FILE 
    FROM THE SAME DIRECTORY WHERE THE SCRIPT IS '''

    scriptpath, scriptfilename = os.path.split(__file__)
    
    txt_file = glob.glob(os.path.join(scriptpath, "*constants.txt"))[0]
    #print("Opening Constants Text File...\n", txt_file, "\n")
    with open(txt_file, "r") as f:
        dict_string = f.read()
    c_dict = ast.literal_eval(dict_string)
    return c_dict





''' READ VOLTAGE DATA'''
def get_voltdata():
    #scriptpath, scriptfilename = os.path.split(__file__)
    scriptpath = os.path.dirname(os.path.abspath(__file__))
    scriptfilename = os.path.basename(__file__)
    #print("SCRIPTPATH:", scriptpath)
    timestamp = os.path.split(scriptpath)[1][:19]
    voltfileloc = os.path.dirname(scriptpath)
    voltfilename = timestamp+"_EPD voltage.txt"
    voltfilepath = os.path.join(voltfileloc, voltfilename)
    #print("VOLT FILE PATH:", voltfilepath, "\n")
    #print("TIMESTAMP:", timestamp)
    #print("VOLT FILE NAME:", voltfilename, "\n")
    voltdata = np.loadtxt(voltfilepath, delimiter=",", usecols=0)
    return voltdata

''' LOAD VIDEO '''
def load_video_cv2():
    global STYLE
    STYLE="salmon1"
    
    import os
    import cv2
    import numpy as np
    
    constants = read_constants()    
    scriptpath, scriptfilename = os.path.split(__file__)
    ratiovidfilename = constants["timestamp"]+"_ratio"+str(constants["bufsize"])+".mp4"
    ratiovidfilepath = os.path.join(scriptpath, ratiovidfilename)
    cprint("Loading Video...")
    print(ratiovidfilepath, "\n")
    
    cap = cv2.VideoCapture(ratiovidfilepath)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return np.array(frames, dtype=np.uint8)





'''
############################################################################
############################################################################
                       FILTERS
#############################################################################
#############################################################################
'''

#FILTER 0.5
def crop_particle_area(pl_in, pixels):
    xdim = 255
    pl_out = []
    for p in pl_in:
        if pixels < np.mean(p.px_vec) < xdim-pixels  and pixels < np.mean(p.py_vec) < xdim-pixels:
            pl_out.append(p)
    return pl_out



#FILTER 1
''' FILTER PARTICLES BASED LIFETIME '''
def filter_pl_by_lifetime(pl, lifetime):
    global STYLE
    STYLE="dark_olive_green3"
    cprint("Removing particles with the following criteria")
    cprint(f"\t Lifetime: \t\t{lifetime}")
    pl_new = [p for p in pl if len(p.f_vec) > lifetime]
    return pl_new


#FILTER 2
''' FILTER PARTICLES BASED ON CONFIDENCE '''
def filter_pl_by_conf(pl, cavg, chigh):
    global STYLE
    STYLE="dark_olive_green3"
    cprint("Removing particles with the following criteria")
    cprint(f"\t Average Confidence: \t{cavg}")
    cprint(f"\t Maximum Confidence: \t{chigh}\n")
    pl_new = [p for p in pl if np.max(p.conf_vec) > chigh and np.mean(p.conf_vec) > cavg]
    return pl_new



#FILTER 3
''' FILTER BY LONG DISTANCE MOVERS, i.e. SKITTERBUGS '''
def remove_long_range_moving_particles(pl_in, max_dist=50):
    global STYLE
    STYLE="light_slate_blue"
    cprint("Removing Skitterbugs...\n")
    
    def calculate_distances(pl_euclidean):
        pdist_out = []
        for p in pl_euclidean:
            #stack each x and y coordinate for this particle
            points = np.column_stack((p.px_vec, p.py_vec ))
            #compute the pairwise differences between consecutive points
            diffs = np.diff(points, axis=0)
            #compute the Euclidean distance and sum them
            distances = np.linalg.norm(diffs, axis=1)
            total_distance = np.sum(distances)
            pdist_out.append(total_distance)
        return pdist_out
    
    ''' GET PL DATA '''
    pIDs_         = [p.pID for p in pl_in]
    plifetime_    = [len(p.f_vec) for p in pl_in]
    pdist_        = calculate_distances(pl_in)
    
    ''' UPPER PLOT '''
    x = pIDs_
    y = pdist_
    labels = pIDs_
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 16), dpi=300)
    ax[0, 0].scatter(x, y, s=1)
    ax[0, 0].set_xlabel("pIDs")
    ax[0, 0].set_ylabel("distance travelled")
    for xi, yi, label in zip(x, y, labels):
        ax[0, 0].text(xi, yi, label, fontsize=10, ha='right', va='bottom')
    
    #draw particle position traces
    for i, p in enumerate(pl_in):
        ax[0, 1].plot(p.px_vec, p.py_vec, color=matplotlib.cm.viridis_r(pdist_[i]/max_dist))
    ax[0, 1].invert_yaxis()
    #add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin=0, vmax=np.max(pdist_)))
    fig.colorbar(sm, ax=ax[0, 1])
    
    ''' MAKE NEW PL DATA ''' 
    pl_out = []
    for i in range(len(pl_in)):
        if pdist_[i] < max_dist: pl_out.append(pl_in[i])
    pIDs_out         = [p.pID for p in pl_out]
    plifetime        = [len(p.f_vec) for p in pl_out]
    pdist_out        = calculate_distances(pl_out)
    
    ''' LOWER PLOT '''   
    x = pIDs_out 
    y = pdist_out
    labels = pIDs_out
    ax[1, 0].scatter(x, y, s=1)
    ax[1, 0].set_xlabel("pID")
    ax[1, 0].set_ylabel("distance travelled")
    for xi, yi, label in zip(x, y, labels):
        ax[1, 0].text(xi, yi, label, fontsize=10, ha='right', va='bottom')
    
    #draw particle position traces
    for i, p in enumerate(pl_out):
        ax[1, 1].plot(p.px_vec, p.py_vec, color=matplotlib.cm.viridis_r(pdist_out[i]/max_dist))
        if pdist_out[i] > 0.5*max_dist: ax[1, 1].text(p.px_vec[-1], p.py_vec[-1], int(p.pID), fontsize=12, ha='right', va='bottom', color=matplotlib.cm.viridis_r(pdist_out[i]/max_dist))
    ax[1, 1].invert_yaxis()
    #add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin=0, vmax=np.max(pdist_out)))
    fig.colorbar(sm, ax=ax[1, 1])
    
    plt.tight_layout()
    
    #SAVE IMAGE
    scriptpath, scriptfilename = os.path.split(__file__)
    plt.savefig(os.path.join(scriptpath, "FILTERED 0 filter 3 - long range" + TAG + ".png"))
    #plt.show()
    
    return pl_out
#pl3 = remove_long_range_moving_particles(pl2, max_dist=50)


#FILTER 4
''' REMOVE CLUSTERS '''
def remove_clusters2(pl_in, space_eps, time_eps):
    global STYLE
    STYLE="sky_blue3"
    cprint("Removing clusters with DBSCAN with the following criteria:")
    cprint(f"\t Spatial Distance (pixels):\t{space_eps}\n\t Temporal Distance (seconds):\t{time_eps}\n")
    
    from sklearn.cluster import DBSCAN
    import numpy as np
    import os
    # space_eps = 2
    # time_eps = 26
    scriptpath, scriptfilename = os.path.split(__file__)
    constants = read_constants()
    # detections = list of (x, y, t) tuples
    #detections = np.array([(px_[i], py_[i], firstframes_[i]) for i in range(len(px_))])  # shape: (N, 3)    
    detections = [(int(np.mean(p.px_vec)), int(np.mean(p.py_vec)), p.f_vec[0]/constants["fps"]) for p in pl_in]

    # Example: detections is a list of (x, y, t) tuples
    detections = np.array(detections)  # shape (N, 3)
    # --- Parameters ---
    #space_eps = 5     # max spatial distance (pixels)
    #time_eps = 10      # max time difference (e.g., seconds or frames)
    # --- Scale time so it’s comparable to space ---
    scaled = detections.astype(float)#.copy()
    scaled[:, 2] *= (space_eps / time_eps)
    # --- Cluster with DBSCAN ---
    db = DBSCAN(eps=space_eps, min_samples=1).fit(scaled)
    labels = db.labels_
    # --- Extract earliest detection in each cluster ---
    filtered = []
    kept_indices = []
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_points = detections[cluster_indices]
        # Get the index of the earliest time in this cluster
        earliest_idx_in_cluster = np.argmin(cluster_points[:, 2])
        original_idx = cluster_indices[earliest_idx_in_cluster]
        kept_indices.append(original_idx)
        filtered.append(detections[original_idx])
    filtered = np.array(filtered)
    #plotting
    fig, axs = plt.subplots(2,1, figsize=(7, 14), dpi=300)
    axs[0].scatter(detections[:,0], detections[:,1], s=1)
    axs[1].scatter(filtered[:,0], filtered[:,1], s=1)
    
    axs[0].set_xticks(np.arange(0, 255, 20))
    axs[0].set_yticks(np.arange(0, 255, 10))
    axs[1].set_xticks(np.arange(0, 255, 20))
    axs[1].set_yticks(np.arange(0, 255, 10))
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    
    axs[0].minorticks_on()
    axs[1].minorticks_on()
    #axs[0].xaxis.set_tick_params(which='minor', bottom=False)
    #axs[1].xaxis.set_tick_params(which='minor', bottom=False)
    
    # axs[0].set_xlabel("pixels")
    # axs[0].set_ylabel("pixels")
    # axs[1].set_xlabel("pixels")
    # axs[1].set_ylabel("pixels")
    plt.tight_layout()
    plt.savefig(os.path.join(scriptpath, "FILTERED 0 filter 4 - decluster.png"))
    #plt.show()
    
    pl_out = list(np.array(pl_in)[kept_indices])
    
    # print("pIDs removed from cluster detection")
    # pl_pids = [p.pID for p in pl_in]
    # print(list(set(pl_pids).symmetric_difference(kept_indices)))
    return pl_out
#pl4 = remove_clusters2(pl3, space_eps=5, time_eps=2)

'''
############################################################################
############################################################################
                     RESULTS
#############################################################################
#############################################################################
'''

''' PLOT DATA '''
def plot_datasheet(pl, pl2, volttime, voltdata, mass_split):
    
    ''' GET INFO FORM PLs '''
    # pl1        
    pIDs_         = [p.pID for p in pl]
    lifetimes_    = [len(p.f_vec) for p in pl]
    contrasts_    = [p.std_max for p in pl]
    arrivaltimes_ = [p.f_vec[0]/constants["fps"] for p in pl] 
    confs_avg_    = [np.mean(p.conf_vec) for p in pl]
    confs_high_   = [np.max(p.conf_vec) for p in pl]
    px_           = [np.mean(p.px_vec) for p in pl]
    py_           = [np.mean(p.py_vec) for p in pl]

    # pl2
    pIDs         = [p.pID for p in pl2]
    lifetimes    = [len(p.f_vec) for p in pl2]
    contrasts    = [p.std_max for p in pl2]
    arrivaltimes = [p.f_vec[0]/constants["fps"] for p in pl2] 
    confs_avg    = [np.mean(p.conf_vec) for p in pl2]
    confs_high   = [np.max(p.conf_vec) for p in pl2]
    px           = [np.mean(p.px_vec) for p in pl2]
    py           = [np.mean(p.py_vec) for p in pl2]


    ''' MAKE PLOTS '''
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 18), dpi=300)
    # Manually plot each subplot

    '''CONFIDENCE'''
    a00 = axes[0, 0].scatter(confs_avg_, confs_high_, c=lifetimes_, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title("Raw Data")
    axes[0, 0].set_xlabel("Conf - average")
    axes[0, 0].set_ylabel("Conf - Max")
    axes[0, 0].grid(True)
    axes[0, 0].set_xlim(0,1)
    axes[0, 0].set_ylim(0,1)
    cbar = fig.colorbar(a00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    #cbar.set_label("lifetime")
    
    a01 = axes[0, 1].scatter(confs_avg, confs_high, c=lifetimes, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title("Filtered Data")
    axes[0, 1].set_xlabel("Conf - average")
    #axes[0, 1].set_ylabel("Conf - Max")
    axes[0, 1].grid(True)
    axes[0, 1].set_xlim(0,1)
    axes[0, 1].set_ylim(0,1)
    axes[0, 1].set_yticklabels([])
    cbar = fig.colorbar(a01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label("lifetime")
    
    
    '''CONTRAST HISTOGRAM'''
    n, bins, _ = axes[1, 0].hist(contrasts_, bins=np.arange(0,0.05, 0.001), color=matplotlib.cm.viridis(0.7))
    axes[1, 0].plot([mass_split]*5, list(np.linspace(0,max(n),5)), color="red", lw=1, ls='--')
    axes[1, 0].set_xlabel("SD Contrast")
    axes[1, 0].set_ylabel("Count")
    
    n, bins, _ = axes[1, 1].hist(contrasts, bins=np.arange(0,0.05, 0.001), color=matplotlib.cm.viridis(0.7))
    axes[1, 1].plot([mass_split]*5, list(np.linspace(0,max(n),5)), color="red", lw=1, ls='--')
    axes[1, 1].set_xlabel("SD Contrast")
    #axes[1, 1].set_ylabel("Count")
    
    
    '''KINETICS'''
    ''' SEPARATE MASS COMPONENTS '''
    mass_split = 0.0075
    # pl1
    ''' SEPARATE MASS COMPONENTS '''
    mass_split = 0.0075
    # pl1
    ps50_ff_  = [0] #arrival time
    ps50_n_   = [0] #particle number
    ps100_ff_ = [0]
    ps100_n_  = [0]
    for i in range(len(contrasts_)):
        if contrasts_[i] < mass_split: #this is the small group
            ps50_ff_.append(arrivaltimes_[i])
        else:
            ps100_ff_.append(arrivaltimes_[i])
    
    #generate a particle count for each particle arrival time 
    ps50_n_ = list(np.arange(1,len(ps50_ff_)+1,1))
    ps100_n_ = list(np.arange(1,len(ps100_ff_)+1,1))        

    #add a final point at the end to make sure each plot goes to the end of the experiment
    maxtime = volttime[-1]
    ps50_n_.append(ps50_n_[-1])   #a new count value, which is equal to the last one
    ps50_ff_.append(maxtime)      #a new time point, which is equal to the final time point in the experiment
    ps100_n_.append(ps100_n_[-1])
    ps100_ff_.append(maxtime)

    # ps50_c_   = [] #contrast
    # ps50_ff_  = [] #arrival time
    # ps50_n_   = [] #particle number
    # ps100_c_  = []
    # ps100_ff_ = []
    # ps100_n_  = []
    # for i in range(len(contrasts_)):
    #     if contrasts_[i] < mass_split: #this is the small group
    #         ps50_c_.append(contrasts_[i])
    #         ps50_ff_.append(arrivaltimes_[i])
    #     else:
    #         ps100_c_.append(contrasts_[i])
    #         ps100_ff_.append(arrivaltimes_[i])
    # ps50_n_ = list(np.arange(1,len(ps50_c_)+1,1))
    # ps100_n_ = list(np.arange(1,len(ps100_c_)+1,1))     
    

    # pl2
    ps50_ff  = [0] #arrival time
    ps50_n   = [0] #particle number
    ps100_ff = [0]
    ps100_n  = [0]
    for i in range(len(contrasts)):
        if contrasts[i] < mass_split: #this is the small group
            ps50_ff.append(arrivaltimes[i])
        else:
            ps100_ff.append(arrivaltimes[i])
    
    #generate a particle count for each particle arrival time 
    ps50_n = list(np.arange(1,len(ps50_ff)+1,1))
    ps100_n = list(np.arange(1,len(ps100_ff)+1,1))        

    #add a final point at the end to make sure each plot goes to the end of the experiment
    maxtime = volttime[-1]
    ps50_n.append(ps50_n[-1])   #a new count value, which is equal to the last one
    ps50_ff.append(maxtime)      #a new time point, which is equal to the final time point in the experiment
    ps100_n.append(ps100_n[-1])
    ps100_ff.append(maxtime)
    # ps50_c   = [] #contrast
    # ps50_ff  = [] #arrival time
    # ps50_n   = [] #particle number
    # ps100_c  = []
    # ps100_ff = []
    # ps100_n  = []
    # for i in range(len(contrasts)):
    #     if contrasts[i] < mass_split: #this is the small group
    #         ps50_c.append(contrasts[i])
    #         ps50_ff.append(arrivaltimes[i])
    #     else:
    #         ps100_c.append(contrasts[i])
    #         ps100_ff.append(arrivaltimes[i])
    # ps50_n = list(np.arange(1,len(ps50_c)+1,1))
    # ps100_n = list(np.arange(1,len(ps100_c)+1,1))        

    ax2 = axes[2, 0].twinx()
    ax2.plot(volttime, voltdata, color=matplotlib.cm.viridis(0.9))
    ax2.set_ylim(-2, 2)
    ax2.set_yticks([])
    axes[2, 0].plot(ps100_ff_, ps100_n_, label="100 nm PS", linewidth=2, color=matplotlib.cm.viridis(0.1))#, color=cmap(j), label="trial "+str(j), s=4)#, color=cmap(i))#colors[i])
    axes[2, 0].plot(ps50_ff_, ps50_n_, label="50 nm PS", linewidth=2, color=matplotlib.cm.viridis(0.8))
    axes[2, 0].set_xlabel("time, s")
    axes[2, 0].set_ylabel("Count")
    axes[2, 0].legend(markerscale=3, loc=4)
    
    ax2 = axes[2, 1].twinx()
    ax2.plot(volttime, voltdata, color=matplotlib.cm.viridis(0.9))
    ax2.set_ylim(-2, 2)
    ax2.set_ylabel("volts")
    axes[2, 1].plot(ps100_ff, ps100_n, label="100 nm PS", linewidth=2, color=matplotlib.cm.viridis(0.1))#, color=cmap(j), label="trial "+str(j), s=4)#, color=cmap(i))#colors[i])
    axes[2, 1].plot(ps50_ff, ps50_n, label="50 nm PS", linewidth=2, color=matplotlib.cm.viridis(0.8))
    #axes[2, 1].set_title("Contrast - Filtered")
    axes[2, 1].set_xlabel("time, s")
    #axes[2, 1].set_ylabel("Count")
    axes[2, 1].legend(markerscale=3, loc=4)
    
    
    ''' LANDING MAP '''
    axes[3, 0].scatter(px_, py_, s=1, color=matplotlib.cm.viridis(0.7))
    axes[3, 1].scatter(px, py, s=1, color=matplotlib.cm.viridis(0.7))
    
    
    ''' CONFIDENSE VS LIFETIME '''
    axes[4, 0].scatter(lifetimes_, confs_high_, s=1, alpha=0.5, color=matplotlib.cm.viridis(0.5))
    axes[4, 0].scatter(lifetimes_, confs_avg_, s=1, alpha=0.5, color=matplotlib.cm.viridis(0.8))
    axes[4, 0].plot([constants["bufsize"]]*5, list(np.linspace(0,1,5)), color="red", lw=1, ls='--', alpha=0.5)
    axes[4, 0].plot([constants["bufsize"]*2]*5, list(np.linspace(0,1,5)), color="red", lw=1, ls='--', alpha=0.5)
    axes[4, 0].set_ylabel("confidence")
    axes[4, 0].set_xlabel("lifetime")
    
    axes[4, 1].scatter(lifetimes, confs_high, s=1, alpha=0.5, color=matplotlib.cm.viridis(0.5))
    axes[4, 1].scatter(lifetimes, confs_avg, s=1, alpha=0.5, color=matplotlib.cm.viridis(0.8))
    axes[4, 1].set_xlabel("lifetime")
    axes[4, 1].set_yticklabels([])
    axes[4, 1].plot([constants["bufsize"]]*5, list(np.linspace(0,1,5)), color="red", lw=1, ls='--', alpha=0.5)
    axes[4, 1].plot([constants["bufsize"]*2]*5, list(np.linspace(0,1,5)), color="red", lw=1, ls='--', alpha=0.5)
    
    #axes[4, 0].scatter(lifetimes_, [h/a for h, a in zip(confs_high_, confs_avg_)], s=1, alpha=0.5, color="orange")
    #axes[4, 1].scatter(lifetimes, [h/a for h, a in zip(confs_high, confs_avg)], s=1, alpha=0.5, color="orange")
    
    # Adjust layout
    #plt.tight_layout()
    scriptpath, scriptfilename = os.path.split(__file__)
    plt.savefig(os.path.join(scriptpath, "FILTERED Datasheet" + TAG + ".png"))
    
    #plt.show()

    return



'''PLOT AND SAVE FILTERED CONTRAST '''
def plot_contrast(pl, mass_split, tag):    
    plt.figure(figsize=(10,6), dpi=300)
    contrasts = [p.std_max for p in pl]
    n, bins, _ = plt.hist(contrasts, bins=np.arange(0,0.05, 0.001), color=matplotlib.cm.viridis(0.7))
    if mass_split!=0: plt.plot([mass_split]*5, list(np.linspace(0,max(n),5)), color="red", lw=1, ls='--')
    plt.xlabel("SD Contrast")
    plt.ylabel("Count")
    scriptpath, scriptfilename = os.path.split(__file__)
    plt.savefig(os.path.join(scriptpath, "FILTERED Contrast" + tag + ".png"))
    #plt.show()
    return


# def plot_kinetics_with_mass_split(pl, mass_split, tag):
#     ''' VOLT DATA '''
#     voltdata = get_voltdata()
#     volttime = np.linspace(0, len(voltdata)/constants["fps"], len(voltdata))
#     ''' GET PL DATA '''
#     pIDs_         = [p.pID for p in pl]
#     lifetimes_    = [len(p.f_vec) for p in pl]
#     contrasts_    = [p.std_max for p in pl]
#     arrivaltimes_ = [p.f_vec[0]/constants["fps"] for p in pl] 
#     confs_avg_    = [np.mean(p.conf_vec) for p in pl]
#     confs_high_   = [np.max(p.conf_vec) for p in pl]
#     px_           = [np.mean(p.px_vec) for p in pl]
#     py_           = [np.mean(p.py_vec) for p in pl]
#     ''' SEPARATE MASS COMPONENTS '''
#     #mass_split = 0.0075
#     # pl1
#     #ps50_c_   = [0] #contrast
#     ps50_ff_  = [0] #arrival time
#     ps50_n_   = [0] #particle number
#     #ps100_c_  = [0]
#     ps100_ff_ = [0]
#     ps100_n_  = [0]
#     for i in range(len(contrasts_)):
#         if contrasts_[i] < mass_split: #this is the small group
#             #ps50_c_.append(contrasts_[i])
#             ps50_ff_.append(arrivaltimes_[i])
#         else:
#             #ps100_c_.append(contrasts_[i])
#             ps100_ff_.append(arrivaltimes_[i])
#     #generate a particle count for each particle arrival time 
#     ps50_n_ = list(np.arange(1,len(ps50_ff_)+1,1))
#     ps100_n_ = list(np.arange(1,len(ps100_ff_)+1,1))        
#     #add a final point at the end to make sure each plot goes to the end of the experiment
#     maxtime = volttime[-1]
#     ps50_n_.append(ps50_n_[-1])   #a new count value, which is equal to the last one
#     ps50_ff_.append(maxtime)      #a new time point, which is equal to the final time point in the experiment
#     ps100_n_.append(ps100_n_[-1])
#     ps100_ff_.append(maxtime)
#     ''' PLOT '''
#     fig, ax = plt.subplots(figsize=(10,8), dpi=300)
#     ax2 = ax.twinx()
#     ax2.plot(volttime, voltdata, color=matplotlib.cm.viridis(0.9))
#     ax2.set_ylim(-2, 2)
#     ax2.set_ylabel("volts")
#     ax.plot(ps100_ff_, ps100_n_, label="100 nm PS", linewidth=2, color=matplotlib.cm.viridis(0.8))#, color=cmap(j), label="trial "+str(j), s=4)#, color=cmap(i))#colors[i])
#     ax.plot(ps50_ff_, ps50_n_, label="50 nm PS", linewidth=2, color=matplotlib.cm.viridis(0.1))
#     ax.set_xlabel("time, s")
#     ax.set_label("counts")
#     ax.legend(markerscale=3)
#     scriptpath, scriptfilename = os.path.split(__file__)
#     plt.savefig(os.path.join(scriptpath, "FILTERED Kinetics" + tag + ".png"))
#     plt.show()
#     return


'''PLOT AND SAVE FILTERED LANDING RATE '''

def plot_kinetics(pl_in, tag):
    
    
    ''' VOLT DATA '''
    voltdata = get_voltdata()
    volttime = np.linspace(0, len(voltdata)/constants["fps"], len(voltdata))
    
    f = np.arange(0, 120, 0.1)
    f_sparse = [(p.f_vec[0]+200)/constants["fps"] for p in pl_in] #this is a list of the first frame each particle was seen (len(f_sparse) = len(pl))
    
    #generate y data that corresponds to each frame number. the total number of deposited particles at any frame is the number of elements in f_sparse that are less than the current frame
    n = np.zeros_like(f)
    for j, fval in enumerate(f):
        n[j] = len([fnum for fnum in f_sparse if fnum <= fval])

    ''' PLOT '''
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    ax2 = ax.twinx()
    ax2.plot(volttime, voltdata, color=matplotlib.cm.viridis(0.9))
    ax2.set_ylim(-2, 2)
    ax2.set_ylabel("volts")
    #ax.plot(ps100_ff_, ps100_n_, label="100 nm PS", linewidth=2, color=matplotlib.cm.viridis(0.8))#, color=cmap(j), label="trial "+str(j), s=4)#, color=cmap(i))#colors[i])
    ax.plot(f, n, linewidth=2, color=matplotlib.cm.viridis(0.1))
    ax.set_xlabel("time, s")
    ax.set_label("counts")
    ax.legend(markerscale=3)
    scriptpath, scriptfilename = os.path.split(__file__)
    
    #SAVE FULL PLOT
    plt.savefig(os.path.join(scriptpath, "FILTERED Kinetics" + tag + ".png"))
    
    # SAVE INIT ZOOM PLOT
    ax.set_xlim([-2,30])
    #get max number of particles at t=30
    ax.set_ylim([-5, n[np.where(f == 30)]])
    plt.savefig(os.path.join(scriptpath, "FILTERED Kinetics -zoom" + tag + ".png"))
    
    #plt.show()
    
    return
#plot_kinetics(pl4, "-pl4-new")



def generate_landing_rate_csv(pl_in, tag, mass_split):
    ''' CONSTANTS '''
    constants = read_constants()   
    
    ''' VOLT DATA '''
    voltdata = get_voltdata()
    volttime = np.linspace(0, len(voltdata)/constants["fps"], len(voltdata))

    n        = constants["nframes"]
    fps      = constants['fps']

    # this converts the x axis from frames to seconds
    spf = np.linspace(0, (n/fps), n)
        
    #mass_split = 0#0.0075
    #m1pf = []
    #m2pf = []
    
    print("Making .csv file for Landing Rate...")
    # particles per frame, list
    m1_ppf = np.zeros(n)
    m2_ppf = np.zeros(n)
    
    c1 = 0
    c2 = 0                                                        # initialize a total particle counter
    for i in range(n):                                 #loop through each frame of the video     
        
        # returns a list of particle ID's for all particles that first landed on this particular frame
        # (which are equivalent to particle counts)
        m1_pids = [p.pID for p in pl_in if p.f_vec[0] == i and p.std_max <= mass_split ]
        m2_pids = [p.pID for p in pl_in if p.f_vec[0] == i and p.std_max > mass_split ]
    
                                                               
        
        # if a list of landed particles was created, then use the largest pID
        # (the last particle on the list), as the new total particle count
        # set the particles per frame to be the total number of particles found so far
        c1 += len(m1_pids)
        c2 += len(m2_pids)
                                                        
        m1_ppf[i] = c1                                              
        m2_ppf[i] = c2

    # seconds per frame list

    #save original particle landing rate data
    scriptpath, scriptfilename = os.path.split(__file__)
    csv_filename = os.path.join(scriptpath, "FILTERED Landing Rate"+ tag +".csv")   
    np.savetxt(csv_filename, np.transpose([volttime, voltdata, m1_ppf, m2_ppf]), delimiter=',', header=f"time, volts, nlandings <= {mass_split} sd contrast, nlandings > {mass_split} sd contrast", fmt="%.8f")
    #print("\t landing rate csv saved as: ", csv_filename)








# def save_pl(pl_in, plnumber):
#     scriptpath, scriptfilename = os.path.split(__file__)
#     plfilepath = glob.glob(os.path.join(scriptpath, "*Particle List*.csv"))[0] #get the Particle List csv
#     # global STYLE
#     # STYLE="orange3"
#     # cprint("Opening Particle List.csv File...")
#     # print(f"{plfilepath}\n")
#     original_pl_df = pd.read_csv(plfilepath)
    
#     #single valued items
#     keep_pIDs    = [p.pID for p in pl_in]
    
#     pl_out = original_pl_df[original_pl_df['pID'].isin(keep_pIDs)]
#     pl_out.to_csv(os.path.join(scriptpath, "FILTERED Particle List"+plnumber+TAG+".csv"), index=False)


def plot_map(pl, tag):
    plt.figure(figsize=(10,10), dpi=300)
    for p in pl:
        plt.scatter(np.mean(p.px_vec), np.mean(p.py_vec))
    scriptpath, scriptfilename = os.path.split(__file__)
    filename = "FILTERED particle map"+tag+".png"
    print(filename)
    savefilepath = os.path.join(scriptpath, filename)
    plt.savefig(savefilepath)
    #plt.show()
    
def export_xy(pl, tag):
    x = [np.mean(p.px_vec) for p in pl]
    y = [np.mean(p.py_vec) for p in pl]
    pos_xy = np.array(list(zip(x, y)))
    scriptpath, scriptfilename = os.path.split(__file__)
    np.savetxt(os.path.join(scriptpath, "results - iSCAT DYNAMIC pos xy.csv"), pos_xy)#, delimiter=",")

def export_xy_um(pl, hfw, tag):
    xdim=256
    scope_res    = hfw/xdim #um/px
    x = [np.mean(p.px_vec)*scope_res for p in pl]
    y = [np.mean(p.py_vec)*scope_res for p in pl]
    pos_xy = np.array(list(zip(x, y)))
    scriptpath, scriptfilename = os.path.split(__file__)
    np.savetxt(os.path.join(scriptpath, "results - iSCAT DYNAMIC pos xy um.csv"), pos_xy)#, delimiter=",")
    
    
    
# hfw       = 18.56 * 2 #micron

# export_xy(pl2_1, "-pl2_1")
# export_xy_um(pl2_1, hfw, "-pl2_1")



# x = [np.mean(p.px_vec) for p in pl2_1]
# y = [np.mean(p.py_vec) for p in pl2_1]
# pos_xy = np.array(list(zip(x, y)))
# scriptpath, scriptfilename = os.path.split(__file__)
# np.savetxt(os.path.join(scriptpath, "results - iSCAT DYNAMIC pos xy.csv"), pos_xy, delimiter=",")



''' SAVE VIDEO '''
def save_video_from_pl(pl_in, r8_in, TAG):
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
    constants = read_constants()
    
    #IMPORT VOLT DATA
    voltdata = get_voltdata()
    #print some info
    print("Total raw video frames:   ", constants["frame total"])
    print("Total raw video frames:   ", constants["nframes"])
    print("Total volt log frames:    ", len(voltdata))
    print("Total ratio video frames: ", len(r8_in))
    offset_r = constants["bufsize"]*2 #this offset accounts for the frame loss due to ratiometric buffer
    print("ratio bufer offset: ", offset_r)
    
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
    vB, vG, vR = 253, 142, 124 #volt color
    font = ImageFont.truetype("arial.ttf", 32)
    font_small = ImageFont.truetype("arial.ttf", 10)
    
    cprint("Preparing Particle annotations...")
    
    ba = [[] for _ in range(len(video_out))]   #box annotations
    pa = [[] for _ in range(len(video_out))]   #point annotations
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
            pida[f].append([xloc, yloc, int(p.pID)])

    # loop through each frame drawing frame numbers on each one
    # also draw a rectantular bounding box that defines the particle images
    # and also the particle edge cut-off distance
    #print(video_out.shape)
    cprint("\nAnnotating video frames...")
    offset_t = 0 #frame offset from a trimmed video
    for i in tqdm.tqdm(range(len(video_out))):
        pillowImage = PIL.Image.fromarray(video_out[i])
        draw = ImageDraw.Draw(pillowImage)
        

   
        # # draw the voltage trace
        # span   = 196                # length of trace (in frames)
        # txp    = 8                  # trace x position
        # typ    = 180                # trace y position
        # tyh    = 50                 # trace height (this is essentially a multiplier for the voltage)
        # #x_off = -20
        # #n_tot = constants["frame total"]                                         # total number of frames
        # n_tot = constants["nframes"]                                         # total number of frames
        
        # #print(fnum, fnum+offset-span-offset2, span, n_tot, span+offset2)
        # if i >= span+offset_r and i + offset_t + span + offset_r < n_tot:                         # you will get array errors if you run this on the whole array so this makes sure that doesnt happen
            
        #     trace = np.array(voltdata[(i+offset_t-span+offset_r):(i+offset_t+offset_r)])   # trace is a snippit of the voltage frame array. trace is what we print on the screen 
        #     #print(fnum, len(trace), trace[0], trace[-1])
        #     #print("trace: ", trace)

        #     last_v = trace[0]                                                     # the method works by drawing a line from point n to point n+1. this makes sure the starting point for the line is the same as the first point it will draw to
        #     for vi, vv in enumerate(trace):                                      # Loop through the whole trace
                
        #         # draw a point on the trace line (it actually draws a few points to add thicness)
        #         draw.point(( txp+vi, typ+(tyh*(1-vv))   ), fill=(vR, vG, vB))      #
        #         draw.point(( txp+vi, typ+(tyh*(1-vv))+1 ), fill=(vR, vG, vB))
        #         draw.point(( txp+vi, typ+(tyh*(1-vv))+2 ), fill=(vR, vG, vB))

        #         # draw a tick mark every 200 voltages
        #         voltage_position = i+offset_t+vi
        #         if voltage_position % 200 == 0:      
        #             #print(fnum, voltage_position)
        #             draw.point(( txp+vi, typ+(tyh*(1-vv))-1 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi, typ+(tyh*(1-vv))-2 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi, typ+(tyh*(1-vv))-3 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi, typ+(tyh*(1-vv))-4 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi, typ+(tyh*(1-vv))-5 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi, typ+(tyh*(1-vv))-6 ), fill=(vR, vG, vB))

        #             draw.point(( txp+vi+1, typ+(tyh*(1-vv))-1 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+1, typ+(tyh*(1-vv))-2 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+1, typ+(tyh*(1-vv))-3 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+1, typ+(tyh*(1-vv))-4 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+1, typ+(tyh*(1-vv))-5 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+1, typ+(tyh*(1-vv))-6 ), fill=(vR, vG, vB))
                    
        #             draw.point(( txp+vi+2, typ+(tyh*(1-vv))-1 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+2, typ+(tyh*(1-vv))-2 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+2, typ+(tyh*(1-vv))-3 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+2, typ+(tyh*(1-vv))-4 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+2, typ+(tyh*(1-vv))-5 ), fill=(vR, vG, vB))
        #             draw.point(( txp+vi+2, typ+(tyh*(1-vv))-6 ), fill=(vR, vG, vB))

        #         #print(trace[vi], last_v)
        #         if vv != last_v:                                                 # this draws the line that connects when it switches from 0 to 1 V
        #             #lx1, ly1 = vi+(0.5*x-span)+x_off,    tp+(th*(1-vv))
        #             #lx2, ly2 = vi+(0.5*x-span)+x_off,    tp+(th*(1-last_v))
                    
        #             lx1, ly1 = txp+vi, typ+(tyh*(1-vv))
        #             lx2, ly2 = txp+vi, typ+(tyh*(1-last_v))
        #             draw.line([(lx1, ly1), (lx2, ly2)], fill=(vR, vG, vB), width=3)
        #         last_v = vv  
            
        #     draw.text( (210,typ-20), "1V", (vR, vG, vB), font=font)
        #     draw.text( (210,typ+tyh-20), "0V",  (vR, vG, vB), font=font)
            




        # #draw particle boxes
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
                    

        #print EPD volts on screen
        volts = voltdata[i+offset_t+offset_r]
        draw.text( (175,2), (str(volts) + " V"), (vR, vG, vB), font=font)
        
        #draw frame number    
        draw.text( (2,2), str(i), (tR, tG, tB), font=font)
        
        #draw frame boundary 
        draw.rectangle((EDGE,EDGE,x-EDGE,y-EDGE), fill=None, outline=(fR, fG, fB))
        
        
        #Convert the frame to a numpy array
        video_out[i] = np.array(pillowImage, np.uint8)
    
    # VIDEO SAVING
    # Generate Filename
    scriptpath, scriptfilename = os.path.split(__file__)
    timestamp = constants["timestamp"]
    filename = "FILTERED " + timestamp + "-color" + TAG + ".avi"
    save_file_path = os.path.join(scriptpath, filename)
    
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
        ffmpeg_params=["-crf", "23", "-preset", "slow"]
    )
    t1 = time.time()
    print(f"\ntotal time for video saving 2: {t1-t0} seconds\n")

def save_pl(pl_in, plnumber):
    #get original particle list data
    scriptpath, scriptfilename = os.path.split(__file__)
    plfilepath = glob.glob(os.path.join(scriptpath, "*Particle List__ - 0 yolo raw output*.csv"))[0] #get the Particle List csv
    # global STYLE
    # STYLE="orange3"
    # cprint("Opening Particle List.csv File...")
    # print(f"{plfilepath}\n")
    original_pl_df = pd.read_csv(plfilepath)
    
    #single valued items
    keep_pIDs    = [p.pID for p in pl_in]
    
    #copy the particles from the original list into a new list if their pIDs match the new list
    pl_out = original_pl_df[original_pl_df['pID'].isin(keep_pIDs)].copy()
    
    #generate missing frame ratio and missing frame count
    missing_frame_ratio = [(p.f_vec[-1]-p.f_vec[0]+1)/len(p.f_vec) for p in pl_in]
    missing_frame_count = [p.f_vec[-1]-p.f_vec[0] for p in pl_in]
    
    #insert missing frame ratio and count into the new particle list
    pl_out["missing ratio"] = missing_frame_ratio
    pl_out["missing count"] = missing_frame_count
    pl_out.to_csv(os.path.join(scriptpath, constants["timestamp"][-8:]+" - FILTERED Particle List"+plnumber+TAG+".csv"), index=False)




def plot_contrast_kinetics(pl_in, tag):
    
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    ax2 = ax.twinx()
    
    ''' VOLT DATA '''
    voltdata = get_voltdata()
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
    scriptpath, scriptfilename = os.path.split(__file__)
    plt.savefig(os.path.join(scriptpath, "FILTERED contrast kinetics" + tag + ".png"))
    
    # ''' SAVE ZOOM '''
    # ax.set_xlim([-2,30])
    # #get max number of particles at t=30
    # ax.set_ylim([-5, n[np.where(f == 30)]])
    # plt.savefig(os.path.join(scriptpath, "FILTERED contrast kinetics -zoom" + tag + ".png"))
    #plt.show()
    
    return
#plot_contrast_kinetics(pl4, "-pl4-new")


#pl_in = pl4
#pltag = "-pl4"
#plot_contrast_kinetics(pl_in, pltag)


def plot_contrast_voltage(pl_in, tag):
    
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    #ax2 = ax.twinx()
    
    ''' VOLT DATA '''
    voltdata = get_voltdata()
    p_time     = [(p.f_vec[0]+200) for p in pl_in]
    
    p_contrast = [(p.std_max)**(1/3) for p in pl_in]
    #p_volts    = [voltdata[pt] for pt in p_time]
    p_group = []
    p_contrast_group = []
    for i, pt in enumerate(p_time):
        if len(p_group) == 0:
            p_group.append(voltdata[pt])
            contrast_group_temp = []
        if p_group[-1] != voltdata[pt]:
            p_contrast_group.append(contrast_group_temp)
            contrast_group_temp = []
            p_group.append(voltdata[pt])
        contrast_group_temp.append(p_contrast[i])
        if i == len(p_time)-1: p_contrast_group.append(contrast_group_temp) 
        
    for i in range(len(p_group)):
        ax.scatter([i]*len(p_contrast_group[i]), p_contrast_group[i], label=p_group[i])
    ax.legend()
    #ax.scatter(p_contrast, p_volts, s=100, alpha=0.1)
    #ax.set_ylim([-0.1, 2.1])
    #plt.show()    
    
    return p_time
#plot_contrast_voltage(pl4, "pl4")

   

''' START SCRIPT '''
scriptpath, scriptfilename = os.path.split(__file__)
TAG = ""    
#program init / open files
banner()
pltag = "-pl0"
pl0 = make_pl_from_pl_file()
constants = read_constants()
voltdata = get_voltdata()
volttime = np.linspace(0, len(voltdata)/constants["fps"], len(voltdata))
save_pl(pl0, "-pl0")


r8 = load_video_cv2()
np.savetxt(os.path.join(scriptpath, "particle count"+pltag+" -- "+str(len(pl0))), [len(pl0)])
#save_video_from_pl(pl0, r8, "-pl0")







def export_data(pl_in, pltag, save_vid=False):
    save_pl(pl_in, pltag)
    plot_map(pl_in, pltag)
    plot_contrast_kinetics(pl_in, pltag)
    plot_contrast(pl_in, 0, pltag)
    plot_kinetics(pl_in, pltag)
    plot_contrast_voltage(pl_in, pltag)
    #if save_vid==True: save_video_from_pl(pl_in, r8, pltag)
    np.savetxt(os.path.join(scriptpath, "particle count"+pltag+" -- "+str(len(pl_in))), [len(pl_in)])

def remove_early_particles(pl_in, fnum)    :
    pl_new = [p for p in pl_in if p.f_vec[0] > fnum]
    return pl_new
    
pl0_5 = remove_early_particles(pl0, 1800)

#filter particles by lifetime
pl1 = filter_pl_by_lifetime(pl0_5, lifetime=int(0.25*constants["bufsize"]))
export_data(pl1, "-pl1")


#crop frame
pl1_5 = crop_particle_area(pl1, 25)
export_data(pl1_5, "-pl1_5")

#filter data by confidence
pl2 = filter_pl_by_conf(pl1_5, cavg=0, chigh=0.20) #cavg .1 or .15
export_data(pl2, "-pl2")

#remove long range particles 
pl3 = remove_long_range_moving_particles(pl2, max_dist=50)
export_data(pl3, "-pl3")

#remove repeated clusters of particles
pl4 = remove_clusters2(pl3, space_eps=2, time_eps=26) 
export_data(pl4, "-pl4-new", save_vid=True)


# # #filter particles if there are many breaks
# pl5 = [p for p in pl4 if (p.f_vec[-1]-p.f_vec[0]) / len(p.f_vec) < 1.2] #1.2 to 1.5 should be good #filter "flashing" particles (ones that are recorded with many skipped frames )
# export_data(pl5, "-pl5")

# # #filter particles if there are many breaks
# pl6 = [p for p in pl5 if (p.f_vec[-1]-p.f_vec[0]) < int(0.5*constants["bufsize"])] #1.2 to 1.5 should be good #filter "flashing" particles (ones that are recorded with many skipped frames )
# export_data(pl6, "-pl6")
