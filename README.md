# iSCAT-Yolo 
### Automated Particle Detection, Localization, and Measurement for iSCAT Microscope Video

Keywords: Intrferometric Scattering Microscopy, Mass Photometry, Ratiometric Video Processing, YOLOv8, Dynamic Nano-Microscopy, Particle Detection, Localization, Measurement

## Overview
This repository contains Python code for processing for Interferometric Scattering Microscopy Video Data. 
The general workflow is as follows:
1) Open 8-bit binary video file
2) Convert raw video to ratiometric video and locate particles on each frame
3) Perform data quality assurance (remove false positives)
4) Generate plots, spreadsheets and output statistics

### Image processing Agorithm
![Image processing steps](https://github.com/MatthewKowal/iscat-yolo/blob/main/figures/image%20processing.png)
Image processing & particle finding are performed sequentially, frame by frame. A) Bright laser light reflecting off of the coverslip drowns out the weak scattering signal of sub-wavelength sized nanoparticles, making the particles impossible to detect in a raw iSCAT video. B) a ratiometric image processing algorithm normalizes and mean-centers the unchanging background reflection while amplifying changes in local image intensity between frames, which occurs when a nanoparticle collides with the in-focus coverslip surface. C) The well-defined, high contrast particle deposition events are detected and localized using a YOLOv8 object detection model trained on a [custom dataset](https://universe.roboflow.com/iscat-particle-image-library/iscat-particle-image-library).

## Example Output
Example output video showing initiation and cessation of 50 nm Polystyrene Nanoplastics deposition onto a coverslip during an EPD-iSCAT experiment.
[Video 1](https://figshare.com/articles/media/Electrophoretic_Deposition_Interferometric_Scattering_Microscopy_EPD-iSCAT_voltage_controlled_deposition_and_detection_of_50_nm_polystyrene_nanoparticles_/24185811)


## Useful Links
iSCAT wiki  https://en.wikipedia.org/wiki/Interferometric_scattering_microscopy

Yolo v8  https://github.com/ultralytics/ultralytics

Custom Dataset  https://universe.roboflow.com/iscat-particle-image-library/iscat-particle-image-library




