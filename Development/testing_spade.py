# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:36:17 2018

@author: Aman Kumar
"""

from spade import *
from sklearn.externals import joblib

dest_addr_spade_folder = 'E:\Desktop'
src_addr = 'F:\\BCI project\\Dataset\\CHB-MIT\\chb04'
#create_dir(dest_addr_spade_folder)
 
 
 #physically put the info files
 # then run these
 
dest_addr = dest_addr_spade_folder + '\\SPADE'
num_of_channels,channels, sampling_freq = fetch_system_info(dest_addr) 
preictal_info = fetch_preictal_info(dest_addr)
interictal_info = fetch_interictal_info(dest_addr)

extract_seizure_data(src = src_addr, des = dest_addr , preictal_info = preictal_info, interictal_info = interictal_info)

createDataset(addr = dest_addr, start_time = 25, end_time = 35, num_of_channels = num_of_channels, sampling_freq = sampling_freq, channels = range(23))



trainModel(addr = dest_addr, start_time = 35, end_time = 40)

model = joblib.load(dest_addr + '\\preictal_35_40_trainedModel.pkl')