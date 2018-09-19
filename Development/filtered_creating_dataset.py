# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:37:03 2018

@author: amankumar
"""

import numpy as np
import pandas as pd
import os


# Loading the Common Spatial Pattern filter
csp_filter_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/csp_filterA_5_10.csv'
csp_filter = (pd.read_csv((csp_filter_addr),header=None)).as_matrix()


################  Loading preictal and interictal dataset(epochs)
num_of_channels = 23
sampling_freq = 256 #in Hz
length_of_time_window = 10 #in seconds
length_of_epoch = sampling_freq*length_of_time_window
# Loading Preictal epochs
preictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/'
folder_names = ['preictal_5_10','preictal_10_15','preictal_15_20','preictal_20_25',
                'preictal_25_30','preictal_30_35','preictal_35_40','preictal_40_45','preictal_45_50']

preictal_files = []
for folder_name in folder_names:
    for file_name in os.listdir(preictal_addr+folder_name):
        if file_name.endswith(".csv"):
            preictal_files.append(preictal_addr+folder_name+ '//' + file_name)

preictal = np.zeros((len(preictal_files),num_of_channels,length_of_epoch))
count = 0
for files in preictal_files:
    preictal[count,:,:] = (pd.read_csv(files,header=None)).as_matrix()
    if(count%50==0):    
        print(count)
    count = count + 1

print 'Done loading the preictal epochs'
# Loading interictal epochs
interictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal'

interictal_files = []
for file_name in os.listdir(interictal_addr):
    if file_name.endswith(".csv"):
       interictal_files.append(interictal_addr + '//' +file_name)

interictal = np.zeros((len(preictal_files),num_of_channels,length_of_epoch))
count = 0
for files in interictal_files[:len(preictal_files)]:
    interictal[count,:,:] = (pd.read_csv(files,header=None)).as_matrix()
    if(count%50==0):
        print(count)
    count = count + 1

print 'Done loading the interictal epochs'
############# Loading the dataset is done ######################




filtered_preictal_feature_data = np.zeros((preictal.shape[0],preictal.shape[2]))
filtered_interictal_feature_data = np.zeros((interictal.shape[0],interictal.shape[2]))

sum_ch = np.zeros((filtered_preictal_feature_data.shape[1]))
for i  in range(0,preictal.shape[0]):    
    filtered_preictal = np.dot(csp_filter, preictal[i])

    sum_ch = 0*sum_ch
    for ch in filtered_preictal:
        sum_ch = sum_ch + ch
    
    averaged_surrogate_channel = np.divide(sum_ch,num_of_channels)
    filtered_preictal_feature_data[i] = averaged_surrogate_channel
    
    
sum_ch = np.zeros((filtered_interictal_feature_data.shape[1]))
for i  in range(0,interictal.shape[0]):    
    filtered_interictal = np.dot(csp_filter, preictal[i])

    sum_ch = 0*sum_ch
    for ch in filtered_interictal:
        sum_ch = sum_ch + ch
    
    averaged_surrogate_channel = np.divide(sum_ch,num_of_channels)
    filtered_interictal_feature_data[i] = averaged_surrogate_channel
    
       
    
np.savetxt(
'/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/filtered_preictal2_5_50.csv',
           filtered_preictal_feature_data,fmt = '%f', delimiter = ',')   
           
           
np.savetxt(
'/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/filtered_interictal2_5_50.csv',
           filtered_interictal_feature_data,fmt = '%f', delimiter = ',')   