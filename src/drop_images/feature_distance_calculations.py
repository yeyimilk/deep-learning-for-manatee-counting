#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import sys


def get_files_from_folder(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and ".DS_Store" not in f]
    return onlyfiles

class FeatureDistanceCalculations():
    def __init__(self, data_folder, file_name="distance.pickle"):
        self.folder = data_folder
        self.file_name = file_name
        
    def _make_sure_dir_exist(self):
        if not os.path.exists(self.folder) or not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        return self.folder
    
    def _save_as_pickle_file(self, data):
        path = self._make_sure_dir_exist()
        path = "{}/{}".format(path, self.file_name)
        # remove old file to make sure everything is clean
        if os.path.isfile(path): 
            os.remove(path)
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
    
    def __load_data(self, path, name):
        abspath = os.path.join(path, name)
        with open(abspath, 'rb') as handle:
            b = pickle.load(handle)
        return b
        
    def __do_calculate(self, a, b):
        final = 0
        for i in range(len(a)):
            a_i = a[i].detach().numpy()
            b_i = b[i].detach().numpy()

            chw = list(a_i.shape)
            denominator = 1
            for c in chw:
                denominator *= c

            result = a_i - b_i
            result = result * result
            result = np.sum(result)
            result = result / denominator
            final += result
        return final
        
    def __get_dic_key(self, a, b):
        return "{}_{}".format(a.replace(".pickle", ""), b.replace(".pickle", ""))
        
    def calculate_and_save(self, feature_folder):
        data_list = get_files_from_folder(feature_folder)
        data_list.sort()
        
        result = {}
        for i in range(len(data_list)):
            a = self.__load_data(feature_folder, data_list[i])
            for j in range(i+1, len(data_list)):
                b = self.__load_data(feature_folder, data_list[j])
                distance = self.__do_calculate(a, b)
                key = self.__get_dic_key(data_list[i], data_list[j])
                result[key] = distance
        
        self._save_as_pickle_file(result)
        return result

def calculate_and_save(folder):
    result_path = folder + "/distance_results"
    fdc = FeatureDistanceCalculations(result_path)
    fdc.calculate_and_save(folder)


if __name__ == "__main__":
    folder = sys.argv[1]
    calculate_and_save(folder)

