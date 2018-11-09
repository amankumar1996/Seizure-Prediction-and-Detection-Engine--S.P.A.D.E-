# -*- coding: utf-8 -*-
"""
Created on Sun Apr 08 22:52:39 2018

@author: Aman Kumar
"""

###################################################################
#                                                                 
#    APPLYING BAND PASS FILTER ON EPOCHED EEG DATA SAMPLES
#                  
#                                                              
###################################################################

import pandas as pd
import numpy as np
from mne.filter import filter_data
import os
import matplotlib.pyplot as plt

source_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal'

os.chdir(source_addr)

eeg_samplefiles = []


for file in os.listdir(source_addr):
    if file.endswith(".csv"):
        eeg_samplefiles.append(file)

low_freq = 1
high_freq = 30
sampling_freq = 256
dest_addr = '/media/amankumar/Pro/BCI project/processed_dataset/band_pass_filtered/chb01/preictal'        
for files in eeg_samplefiles[3:]:
    eeg_sample = pd.read_csv(files,header=None)
    eeg_sample = eeg_sample.as_matrix()

    bpf_eeg_sample = filter_data(data = eeg_sample, sfreq = sampling_freq, l_freq = low_freq, h_freq = high_freq)
    filename = dest_addr + '/' + 'bpf_' + files
    np.savetxt(filename,bpf_eeg_sample,fmt = '%f', delimiter = ',')
                          

#X = range(0,750)
#plt.plot(X,eeg_sample[0,750:1500])
#plt.plot(X,bpf_eeg_sample[0,750:1500])