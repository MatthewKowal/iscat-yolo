# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:56:48 2025

@author: Matt
"""

import os
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
dataset_dir = r"C:/Users/Matt/Desktop/iSCAT DEEPTRACK IMAGE SYNTHESIS/new_dataset"  # folder with images and labels
images_ext = ['.jpg', '.jpeg', '.png']
train_ratio = 0.8
val_ratio = 0.2
test_ratio = 0.0  # set > 0.0 if you want a test split too

random.seed(42)

# --- INPUT PATHS ---
images_path = Path(dataset_dir) / 'images'
labels_path = Path(dataset_dir) / 'labels'

# --- OUTPUT PATHS ---
splits = ['train', 'val'] + (['test'] if test_ratio > 0 else [])
output_paths = {
    'images': {s: images_path / s for s in splits},
    'labels': {s: labels_path / s for s in splits}
}

# --- Ensure output directories exist ---
for split in splits:
    os.makedirs(output_paths['images'][split], exist_ok=True)
    os.makedirs(output_paths['labels'][split], exist_ok=True)

# --- Gather all image files ---
all_images = [f for f in os.listdir(images_path) if Path(f).suffix.lower() in images_ext and not Path(f).parent.name in splits]
random.shuffle(all_images)

# --- Compute split sizes ---
total = len(all_images)
n_train = int(total * train_ratio)
n_val = int(total * val_ratio)
n_test = total - n_train - n_val

split_counts = {
    'train': n_train,
    'val': n_val,
    'test': n_test
}

# --- Split and move files ---
index = 0
for split in splits:
    for _ in range(split_counts[split]):
        img_file = all_images[index]
        label_file = Path(img_file).with_suffix('.txt')
        
        # Copy image
        shutil.copy(images_path / img_file, output_paths['images'][split] / img_file)

        # Copy label if exists
        label_src = labels_path / label_file
        if label_src.exists():
            shutil.copy(label_src, output_paths['labels'][split] / label_file)

        index += 1

print("✅ Dataset split complete.")
