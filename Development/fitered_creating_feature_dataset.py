# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:53:54 2018

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
preictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/filtered_preictal2_5_50.csv'
#folder_names = ['preictal_55_60',]

#preictal_files = []
#for folder_name in folder_names:
#    for file_name in os.listdir(preictal_addr+folder_name):
#        if file_name.endswith(".csv"):
#            preictal_files.append(preictal_addr+folder_name+ '//' + file_name)
#
#preictal = np.zeros((len(preictal_files),23,2560))
#count = 0
#for files in preictal_files:
#    preictal[count,:,:] = (pd.read_csv(files,header=None)).as_matrix()
#    if(count%50==0):    
#        print(count)
#    count = count + 1
    
filtered_preictal = (pd.read_csv(preictal_addr,header=None)).as_matrix()
# Performing wavelet Packet decomposition
level = 4
number_of_features_extracted = 5
size_of_feature_vector = (2**level)*number_of_features_extracted

#intializing dataset
filtered_preictal_feature_data = np.zeros((filtered_preictal.shape[0], size_of_feature_vector))

#creating dataset
i=0
for epoch in filtered_preictal:
    feature_vector = []

    coefficients = WPD(epoch,level)
    features = extract_features( coefficients )
    feature_vector.extend(features)
    
    filtered_preictal_feature_data[i,:] = feature_vector
    i = i+1
    if (i%20==0):
        print(i)

np.savetxt(
'/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/filtered_preictal_feature_data2_5_50.csv',
           filtered_preictal_feature_data,fmt = '%f', delimiter = ',')



interictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/filtered_interictal2_5_50.csv'

filtered_interictal = (pd.read_csv(interictal_addr,header=None)).as_matrix()


#intializing dataset
filtered_interictal_feature_data = np.zeros((filtered_interictal.shape[0], size_of_feature_vector))

#creating dataset
i=0
for epoch in filtered_interictal:
    feature_vector = []

    coefficients = WPD(epoch,level)
    features = extract_features( coefficients )
    feature_vector.extend(features)
    
    filtered_interictal_feature_data[i,:] = feature_vector
    i = i+1
    if (i%20==0):
        print(i)

np.savetxt(
'/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/filtered_interictal_feature_data2_5_50.csv',
           filtered_interictal_feature_data,fmt = '%f', delimiter = ',')
# saving the dataset      
   