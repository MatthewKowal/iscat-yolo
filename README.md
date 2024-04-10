# iscat-yolo 
![EPD-iSCAT Graphical Abstract](https://github.com/MatthewKowal/iscat-yolo/blob/main/figures/iscat%20graphical%20abstract.png)
iscat-yolo is an automated particle detection, localization, and measurement tool for processing raw video from an iSCAT microscope. Video pre-processing utilizes ratiometric imaging to amplify the weak signal from scattering nanoparticles as small as 5 nm while simultaneously subtracting the background and averaging away noise. Fast and facile particle detection is performed using Ultralytics YOLO ver. 8 trained on a annotated particle image dataset built from experimental data. The dataset is publicly available on Roboflow and the trained model is available in the repository. Output is generated in the form of spreadsheets containing particle data, images containing data plots, and processed video data. This work was developed during my PhD in Chemistry at the University of British Columbia in Canada and has been published in ACS Nano in 2024.

  ACS Nano: https://doi.org/10.1021/acsnano.3c09221
  
  ChemRxiv: [10.26434/chemrxiv-2023-ddzfc-v2](https://doi.org/10.26434/chemrxiv-2023-ddzfc-v2) (open access)
  
  Roboflow: https://universe.roboflow.com/iscat-particle-image-library (open access)
  
  Ultralytics:  https://github.com/ultralytics/ultralytics

  iSCAT wiki:  https://en.wikipedia.org/wiki/Interferometric_scattering_microscopy


Keywords: Intrferometric Scattering Microscopy, Mass Photometry, Ratiometric Video Processing, YOLOv8, Dynamic Nano-Microscopy, Particle Detection, Localization, Measurement

## Processing Raw Data
The iscat-yolo-frontend file is the script that should be run to process data. This file imports and makes calls to the iscat-yolo-backend file which contains all of the input, processing and output functions.

Open and Edit the iscat-yolo-frontend file and specify the name of the file you would like to process. Example data is included in this repository in the 'example' directory in the form of an .mp4 video file and as a .bin file. Additionally, .mp4 video is accompanied by a voltage.txt log file which includes voltage data used during an Electrophoretic Deposition iSCAT (EPD-iSCAT) experiment. You may use your own 8-bit binary file from a digital camera feed of .mp4 video file, though it is recommended to ensure the pixel resolution is 47 nm/pixel to match the annotated dataset.

The general workflow is as follows:
1) Import raw video from an iSCAT microscope (either an 8-bit binary file from a camera stream, or an mp4 video file)
2) Convert raw video to ratiometric video and locate particles on each frame
3) Perform data quality assurance (remove false positives)
4) Generate plots, spreadsheets and output statistics


### Image processing Algorithm
![Image processing steps](https://github.com/MatthewKowal/iscat-yolo/blob/main/figures/image%20processing.png)
Image processing & particle finding are performed sequentially, frame by frame. A) Bright laser light reflecting off of the coverslip drowns out the weak scattering signal of sub-wavelength sized nanoparticles, making the particles impossible to detect in a raw iSCAT video. B) a ratiometric image processing algorithm normalizes and mean-centers the unchanging background reflection while amplifying changes in local image intensity between frames, which occurs when a nanoparticle collides with the in-focus coverslip surface. C) The well-defined, high contrast particle deposition events are detected and localized using a YOLOv8 object detection model trained on a [custom dataset](https://universe.roboflow.com/iscat-particle-image-library/iscat-particle-image-library).

## Example Output
![Contrast Histogram](https://github.com/MatthewKowal/iscat-yolo/blob/main/figures/sample%20output.png)
Above left, A histogram of optical scattering intensity signals measures as polystyrene nanoparticles are deposited electrohoretically onto a charged indium tin oxide coated glass coverslip. Above right, a cumulative count of nanoparticle depositions over time (blue) with a modulating DC electric field (orange).

Example output video showing initiation and extinction of 50 nm Polystyrene Nanoplastics deposition onto a coverslip during an EPD-iSCAT experiment.
[Video 1](https://figshare.com/articles/media/Electrophoretic_Deposition_Interferometric_Scattering_Microscopy_EPD-iSCAT_voltage_controlled_deposition_and_detection_of_50_nm_polystyrene_nanoparticles_/24185811)







