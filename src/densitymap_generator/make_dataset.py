#!/usr/bin/env python
# coding: utf-8

import json
import os
import cv2 
import numpy as np
import h5py
from tqdm import tqdm
from utils import generate_density_map, make_sure_folder_exists, generate_density_map_anisotropy
from pathlib import Path
import math

class DensityMapGenerator:
    def __init__(self) -> None:
        self.truncate_t = 4.0
        self.sigma_s = 15
        self.window_size_w = (self.truncate_t * self.sigma_s + 0.5) * 2 + 1
        self.dataset_root = Path(__file__).parent.parent.parent / 'dataset'
        self.img_root = self.dataset_root / 'images'
    
    def _get_img_paths(self, img_root):
        img_paths = []
        for root, _, files in os.walk(img_root):
            for img_path in files:
                # only jpg image
                if img_path.endswith(".jpg"):
                    img_paths.append(os.path.join(root, img_path))
        return img_paths
    
    def _map_for_dot_dataset(self, lines, img_path):
        gt = []
        for line in lines:
            x = float((line["sx"] + line["ex"]) / 2)
            y = float((line["sy"] + line["ey"]) / 2)
            gt.append([x, y])
        image = cv2.imread(img_path)
        positions = generate_density_map(
            shape=image.shape, points=np.array(gt), f_sz=self.window_size_w+1, sigma=self.sigma_s
        )
        return positions
        
    def generate_dot_dataset(self):
        dest_folder = self.dataset_root / 'ground_truth_dot' #  sigma15
        post_fix = '' #'sigma15'
        self._generate_maps(self._map_for_dot_dataset, dest_folder, post_fix)
    
    def _map_for_line_dataset(self, lines, img_path):
        image = cv2.imread(img_path)
        density_map = np.zeros(image.shape[0:2])
        for line in lines:
            gt = []
            startx = int(line["sx"])
            endx = int(line["ex"])
            starty =line["sy"]
            endy = line["ey"]

            if startx < endx:
                for x in range(startx, endx):
                    y= int( (x-startx) * (endy-starty)/(endx-startx) ) + starty
                    gt.append([x, y])
            elif startx > endx:
                for x in range(endx, startx):
                    y= int( (x-startx) * (endy-starty)/(endx-startx) ) + starty
                    gt.append([x, y])
            else:
                starty = int(min(line["sy"], line["ey"]))
                endy = int(max(line["sy"], line["ey"]))
                for y in range(starty, endy):
                    gt.append([startx, y])

            sub_dmap = generate_density_map(
                shape=image.shape, points=np.array(gt), f_sz=self.window_size_w + 1, sigma=self.sigma_s
            )

            sub_dmap /= sub_dmap.sum()            
            density_map += sub_dmap
        return density_map
            
    
    def generate_line_dataset(self):
        dest_folder = self.dataset_root / 'ground_truth_line' #  sigma15
        post_fix = '' # 'line_s15'
        self._generate_maps(self._map_for_line_dataset, dest_folder, post_fix)
    
    def _map_for_oval_dataset(self, lines, img_path):
        image = cv2.imread(img_path)
        density_map = np.zeros(image.shape[0:2])
        for line in lines:
            line_start = np.array([int(line["sx"]), int(line["sy"])])
            line_end = np.array([int(line["ex"]), int(line["ey"])])

            points = [line_start, line_end]
            p3 = line_end - line_start
            line_length = math.hypot(p3[0], p3[1])
            if (line_length < 10):
                print('warning: too short line label ', img_path)
            
            window_size_w = line_length
            window_size_w = np.around(window_size_w) + 1 if np.around(window_size_w) % 2 == 0 else np.around(
                window_size_w)
            sigma_x = (window_size_w / 2 - 0.5) / self.truncate_t * 2.355
            sigma_y = sigma_x / 4

            sub_dmap = generate_density_map_anisotropy(
                shape=image.shape, points=points, fx_sz=window_size_w, fy_sz=window_size_w, sigma=[sigma_x, sigma_y]
            )

            sub_dmap /= sub_dmap.sum()
            density_map += sub_dmap
            
        return density_map

    
    def generate_oval_dataset(self):
        # Use aninstropic gaussian kernel
        dest_folder = self.dataset_root / 'ground_truth_anisotropy_1_4'
        post_fix = '' # 'anisotropy_1_4'
        self._generate_maps(self._map_for_oval_dataset, dest_folder, post_fix)
    
    def _generate_maps(self, generator, dest_folder, post_fix):
        make_sure_folder_exists(dest_folder)
        
        img_paths = self._get_img_paths(self.img_root)
        for img_path in tqdm(img_paths):
            gt_path = img_path.replace("images", "labels").replace(".jpg", ".json")
            if not os.path.isfile(gt_path):
                continue
            
            positions = None
            with open(gt_path, "r") as json_file:
                label_data = json.load(json_file)
                lines = label_data["boxes"]
                num_manatee = label_data["human_num"]
                positions = generator(lines, img_path)
            
            if positions is None or not num_manatee == np.round(positions.sum()):
                print(f"{img_path}, {num_manatee} ")
                continue
            
            # save the density map
            _, tail = os.path.split(img_path)
            if post_fix != '':
                post_fix = f"_{post_fix}"
            
            tail = tail.replace(".jpg", f"{post_fix}.h5")
            
            dest_file = os.path.join(dest_folder, tail)
            
            with h5py.File(dest_file, 'w') as hf:
                hf['density'] = positions
    
    def generate(self):
        self.generate_dot_dataset()
        self.generate_line_dataset()
        self.generate_oval_dataset()

if __name__ == '__main__':
    generator = DensityMapGenerator()
    generator.generate()
    
    