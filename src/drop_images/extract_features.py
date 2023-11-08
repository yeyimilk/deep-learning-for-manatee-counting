#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pickle
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        outs = []
        out = self.layer1(x)
        out = self.layer2(out)
        outs.append(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        outs.append(out)
        
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        outs.append(out)
        
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        outs.append(out)
        
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        outs.append(out)
        
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
        return out, outs

def save_as_pickle_file(data, name):
    with open('{}.pickle'.format(name), 'wb') as handle:
        pickle.dump(data, handle)

def get_files_from_folder(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and ".DS_Store" not in f]
    return onlyfiles

class ImageFeatureExtraction():
    def __init__(self, image_folder):
        self.folder = image_folder
        self.vgg = VGG16()
        
    def _get_image_path(self, name):
        return "{}/{}".format(self.folder, name) 
        
    def _load_image(self, name):
        path = self._get_image_path(name)
        image = Image.open(path)
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image)
        return image
    
    def _make_sure_dir_exist(self, folder='feature_data'):
        abspath = os.path.abspath('')
        path = abspath + "/feature_data"
        if not os.path.exists(path) or not os.path.isdir(path):
            os.mkdir(path)
        return path
    
    def _save_as_pickle_file(self, data, name):
        path = self._make_sure_dir_exist()
        path = "{}/{}.pickle".format(path, name)
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
        
        
    def extract_features_and_save(self):
        my_image_folder = self.folder
        image_list = get_files_from_folder(my_image_folder)
        image_list.sort()
        
        for i in tqdm(range(len(image_list)), desc = 'Feature Extraction'):
            name = image_list[i]
            image = self._load_image(name)
            # has not found a good way to handle this
            size = list(image.shape)
            size.insert(0, 1)
            image = image.reshape(size) # (3, 720, 1280) => (1, 3, 720, 128)
            _, outputs = self.vgg.forward(image)
            
            # this need to be refactored to avoid hardcoded
            self._save_as_pickle_file(outputs, name.replace(".jpg", ""))

def extract_features_and_save(folder):
    ife = ImageFeatureExtraction(folder)
    ife.extract_features_and_save()
    

if __name__ == "__main__":
    folder = sys.argv[1]
    extract_features_and_save(folder)
