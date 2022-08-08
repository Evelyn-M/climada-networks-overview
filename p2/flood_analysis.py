#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:34:07 2022

@author: evelynm
"""
import os
import pickle


successful_calcs = []
unsuccessful_calcs = []
parent_folder_path = '/cluster/work/climate/evelynm/nw_outputs/' # '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/p2/' #'/cluster/work/climate/evelynm/nw_outputs'
for file in os.listdir(parent_folder_path):
    d = os.path.join(parent_folder_path, file)
    if os.path.isdir(d):
        if os.path.isfile(d+f'/flood_{d[-3:]}.hdf5'):
            successful_calcs.append(d[-3:])
        else:
            unsuccessful_calcs.append(d[-3:])
            
with open(f"{parent_folder_path}success", "wb") as fp:   #Pickling
    pickle.dump(successful_calcs, fp)
 
with open(f"{parent_folder_path}fail", "wb") as fp:   #Pickling
    pickle.dump(unsuccessful_calcs, fp)

with open(f"{parent_folder_path}success", "rb") as fp: 
    success = pickle.load(fp)