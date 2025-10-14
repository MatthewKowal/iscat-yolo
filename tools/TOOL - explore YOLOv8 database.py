# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:08:08 2025

@author: Matt
"""

import os
import random
import cv2
import matplotlib.pyplot as plt

# Paths to your YOLO dataset folders
script_dir = os.path.dirname(os.path.abspath(__file__))
USE_ROBOFLOW_DB=False
if USE_ROBOFLOW_DB:
    image_subdir = r"iSCAT Particle Image Library.v10i.yolov8/train"
    image_subdir = r"iSCAT Particle Image Library.v10i.yolov8/test"
    image_subdir = r"iSCAT Particle Image Library.v10i.yolov8/valid"
USE_DEEPTRACK_DB=True
if USE_DEEPTRACK_DB:
    image_subdir = r"new dataset"

IMAGES_DIR = os.path.join(script_dir, image_subdir, "images")
LABELS_DIR = os.path.join(script_dir, image_subdir, "labels")

# Class names (edit according to your dataset)
CLASS_NAMES = ["particle"]

def load_random_image():
    # Pick a random image file
    image_file = random.choice(os.listdir(IMAGES_DIR))
    image_path = os.path.join(IMAGES_DIR, image_file)
    
    # Corresponding label file (same base name but .txt)
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(LABELS_DIR, label_file)
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB for matplotlib
    
    # Draw bounding boxes if label file exists
    if os.path.exists(label_path):
        h, w, _ = img.shape
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:])
                
                # Convert from normalized to pixel coordinates
                x_center *= w
                y_center *= h
                box_w *= w
                box_h *= h
                
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 0, 200), 1)
                label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                #cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    return img, image_file

# Show the random image with bounding boxes
img, fname = load_random_image()
plt.figure(figsize=(7, 7), dpi=300)
plt.imshow(img)
plt.title(f"Random image: {fname}")
plt.axis("off")
plt.show()
