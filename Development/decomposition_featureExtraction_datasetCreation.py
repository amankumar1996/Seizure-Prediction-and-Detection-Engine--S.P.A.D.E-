# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 03:05:16 2018

@author: amankumar
"""

import numpy as np
from numpy import mean, std
from scipy.stats import skew, kurtosis
import pandas as pd
import os
from pywt import wavedec, WaveletPacket
import matplotlib.pyplot as pp

def WPD(signal, level):
    wpd = WaveletPacket(data = signal, wavelet='db4', mode = 'symmetric', maxlevel = level)
    node_names = [node.path for node in wpd.get_level(level)]
    
    wpd_coeffs = []
    for names in node_names:
        wpd_coeffs.append(wpd[names].data)
        
    return wpd_coeffs
    
def extract_features(coefs):
    
    feature_set = []
    
    for sub_band in coefs:
        sub_band_mean = mean([abs(ele) for ele in sub_band]) 
        avg_power = mean(sub_band**2)
        sd = std(sub_band)
        skewness = skew(sub_band)
        kurt = kurtosis(sub_band)
        feature_set.extend((sub_band_mean,avg_power,sd,skewness,kurt))
    
    
    return feature_set
    



# Loading the data
preictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/'
folder_names = ['preictal_55_60',]

preictal_files = []
for folder_name in folder_names:
    for file_name in os.listdir(preictal_addr+folder_name):
        if file_name.endswith(".csv"):
            preictal_files.append(preictal_addr+folder_name+ '//' + file_name)

preictal = np.zeros((len(preictal_files),23,2560))
count = 0
for files in preictal_files:
    preictal[count,:,:] = (pd.read_csv(files,header=None)).as_matrix()
    if(count%50==0):    
        print(count)
    count = count + 1

# Performing wavelet Packet decomposition
level = 4
channels = range(0,23) #[1, 2, 3, 7, 11, 13, 14, 15] #channel_num -1 #range(0,23)
number_of_features_extracted = 5
size_of_feature_vector = len(channels)*(2**level)*number_of_features_extracted

#intializing dataset
preictal_feature_data = np.zeros((len(preictal_files), size_of_feature_vector))

#creating dataset
i=0
for epoch in preictal:
    feature_vector = []
    for ch in channels:
        coefficients = WPD(epoch[ch,:],level)
        features = extract_features( coefficients )
        feature_vector.extend(features)
        
    preictal_feature_data[i,:] = feature_vector
    i = i+1
    if (i%20==0):
        print(i)

np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_feature_data_55_60.csv',
           preictal_feature_data,fmt = '%f', delimiter = ',')



interictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal'

interictal_files = []
for file_name in os.listdir(interictal_addr):
    if file_name.endswith(".csv"):
       interictal_files.append(interictal_addr + '//' +file_name)

interictal = np.zeros((len(preictal_files),23,2560))
count = 0
for files in interictal_files[:len(preictal_files)]:
    interictal[count,:,:] = (pd.read_csv(files,header=None)).as_matrix()
    if(count%50==0):
        print(count)
    count = count + 1



#intializing dataset
interictal_feature_data = np.zeros((len(preictal_files), size_of_feature_vector))

#creating dataset
i=0
for epoch in interictal:
    feature_vector = []
    for ch in channels:
        coefficients = WPD(epoch[ch,:],level)
        features = extract_features( coefficients )
        feature_vector.extend(features)
        
    interictal_feature_data[i,:] = feature_vector
    i = i+1
    if (i%20==0):
        print(i)


# saving the dataset      
np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal_feature_data_6000.csv',
           interictal_feature_data,fmt = '%f', delimiter = ',')           
#preictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/Development/preictal_feature_data.csv',
#                                    header = None)).as_matrix()
#
#interictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/Development/interictal_feature_data.csv',
#                                    header = None)).as_matrix()
           
#pp.plot(preictal_feature_data[:,3])
#pp.plot(interictal_feature_data[:,3])

