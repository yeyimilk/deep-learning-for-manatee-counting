#!/usr/bin/env python
# coding: utf-8

import json
import os
import cv2 
import numpy as np
import h5py
from tqdm import tqdm
from utils import generate_density_map, make_sure_folder_exists
from pathlib import Path

truncate_t = 4.0
sigma_s = 15
window_size_w = (truncate_t * sigma_s + 0.5) * 2 + 1

def generate_density_maps(img_root, dest_folder, post_fix):
    # get all image of the dataset
    img_paths = []
    for root, dirs, files in os.walk(img_root):
        for img_name in files:
            if img_name.endswith(".jpg"):
                img_paths.append(os.path.join(root, img_name))

    for img_path in tqdm(img_paths):
        gt_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        if not os.path.isfile(gt_path):
            continue
        gt = []
        # read gt line by line
        with open(gt_path, "r") as json_file:
            label_data = json.load(json_file)
            lines = label_data["boxes"]
            num_manatee = label_data["human_num"]
            for line in lines:
                x = float((line["sx"] + line["ex"]) / 2)
                y = float((line["sy"] + line["ey"]) / 2)
                gt.append([x, y])
        # load the image
        image = cv2.imread(img_path)
        # generate the density map
        positions = generate_density_map(
            shape=image.shape, points=np.array(gt), f_sz=window_size_w+1, sigma=sigma_s
        )
        
        if not num_manatee == np.round(positions.sum()):
            print(f"{img_path}, {num_manatee} ")
            continue
        
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder)
            
        # save the density map
        _, tail = os.path.split(img_path)
        dest_file = os.path.join(dest_folder, tail.replace(".jpg", f"_{post_fix}.h5"))
        with h5py.File(dest_file, 'w') as hf:
            hf['density'] = positions

if __name__ == '__main__':
    img_root = Path(__file__).parent.parent.parent / 'dataset' / 'images'
    dest_folder = Path(__file__).parent.parent.parent / 'dataset' / 'ground_truth_dot' #  sigma15
    post_fix = 'sigma15'
    make_sure_folder_exists(dest_folder)
    print(f"Make sure you have the correct path to the dataset: {img_root}")
    generate_density_maps(img_root, dest_folder, post_fix)


