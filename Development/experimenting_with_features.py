# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 03:13:48 2018

@author: amankumar
"""

import numpy as np
from numpy import mean, std
from scipy.stats import skew, kurtosis
import pandas as pd
import os
from pywt import wavedec, WaveletPacket
import matplotlib.pyplot as pp
from pyrem_univariate import *

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
        abs_sub_band = np.abs(sub_band)
        #sub_band_mean = mean(abs_sub_band)         
        sd = std(abs_sub_band)
        avg_power = mean(sub_band**2)
        #skewness = skew(abs_sub_band)
        kurt = kurtosis(abs_sub_band)
        h_fractal_dim = hfd(sub_band, 16)
        a, hjorth_mob, hjorth_complexity = hjorth(sub_band)
        entropy = svd_entropy(sub_band,2,20)
        hurst_exponent = hurst(sub_band)
        feature_set.extend((avg_power,kurt, h_fractal_dim, hjorth_mob, hjorth_complexity, entropy, hurst_exponent))
    
    
    return feature_set
    



# Loading the data
preictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/'
folder_names = ['preictal_5_10','preictal_10_15','preictal_15_20','preictal_20_25',
                'preictal_25_30','preictal_30_35','preictal_35_40','preictal_40_45','preictal_45_50']

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
number_of_features_extracted = 7
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

np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_newFeature_data_5_50.csv',
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
np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal_newFeature_data_5_50.csv',
           interictal_feature_data,fmt = '%f', delimiter = ',')           
#preictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/Development/preictal_feature_data.csv',
#                                    header = None)).as_matrix()
#
#interictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/Development/interictal_feature_data.csv',
#                                    header = None)).as_matrix()
           
#pp.plot(preictal_feature_data[:,3])
#pp.plot(interictal_feature_data[:,3])


feature_set = []
    
sub_band = preictal[0,1,:]
#sub_band_mean = mean([abs(ele) for ele in sub_band])         
#sd = std(sub_band)
avg_power = mean(sub_band**2)
#skewness = skew(sub_band)
kurt = kurtosis(sub_band)
h_fractal_dim = hfd(sub_band, 16)
a , hjorth_mob, hjorth_complexity = hjorth(sub_band)
entropy = svd_entropy(sub_band,2,20)
hurst_exponent = hurst(sub_band)
feature_set.extend((avg_power,kurt, h_fractal_dim, hjorth_mob, hjorth_complexity, entropy, hurst_exponent))

from numpy.linalg import svd, lstsq
from math import log
Y = embed_seq(sub_band, 2, 20)
W = svd(Y, compute_uv = 0)
W /= sum(W) # normalize singular values
t = -1*sum([ (w * log(w,2)) for w in W ])
del pyeeg
del hfd, hjorth, svd_entropy, hurst, embed_seq