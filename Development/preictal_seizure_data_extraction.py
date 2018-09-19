# -*- coding: utf-8 -*-
"""
Created on Sat May 19 22:00:40 2018

@author: amankumar
"""

"""
This program is to extract pre-ictal data from the .edf format files to .csv format files of specific
window size

"""


import pyedflib
import numpy as np
import pandas as pd

#Initialize the following variables
count = 0   #CAUTION!! - input this after proper inspection - it will be the file number

source_addr = '/media/amankumar/Pro/BCI project/Dataset/CHB-MIT/chb01/preictal'
dest_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal'

start_time =  1862     #in seconds as provided in patient summary file
duration = 25           #in minutes
sampling_freq = 256



time_window = 10 #in seconds: it is preferred that 60 is divisible by time_window size because otherwise
#[contd.] no of splits won't be an integer, and then code need to be set accordingly

seizure_file_name = 'chb01_26.edf' 
prev_file_name = 'chb01_25.edf'


##############################***************#####################

duration = duration*60   #converting into seconds

if start_time >= duration:
    
    print('-----Previous seizure is not required----')
    print('Reading data from .edf file')
    # Reading data from .edf file
    f = pyedflib.EdfReader(source_addr + '/' + seizure_file_name)
    n = f.signals_in_file
    signal_labels1 = f.getSignalLabels()
    seizure = np.zeros((n, f.getNSamples()[0]))

    for i in np.arange(n):
        seizure[i, :] = f.readSignal(i)  
        
    print('Reading done!!!')
    
    seizure = seizure[:,(start_time-duration)*sampling_freq:start_time*sampling_freq]
    print('Seizure data extracted!!!!')
    
else:
    print('-----Seizure data is in both files----')
    print('Reading data from .edf files')
    # Reading data from .edf files
    f = pyedflib.EdfReader(source_addr + '/' + seizure_file_name)
    n = f.signals_in_file
    signal_labels1 = f.getSignalLabels()
    seizure = np.zeros((n, f.getNSamples()[0]))
    
    for i in np.arange(n):
        seizure[i, :] = f.readSignal(i)  
        
    
    f = pyedflib.EdfReader(source_addr + '/' + prev_file_name)
    n = f.signals_in_file
    signal_labels1 = f.getSignalLabels()
    seizure_prev = np.zeros((n, f.getNSamples()[0]))
    
    for i in np.arange(n):
        seizure_prev[i, :] = f.readSignal(i)  
            
    print('Reading done!!!')
    #extracting component from seizure_prev part
    remaining_dur = duration - start_time
    remaining_dur = remaining_dur*sampling_freq  #converting it into number of columns
    remaining_part = seizure_prev[:,(seizure_prev.shape[1]-remaining_dur):]
    
    #extracting from seizure file
    seizure = seizure[:,0:start_time*sampling_freq]
    
    #concatening both
    
    seizure = np.concatenate((remaining_part,seizure), axis = 1)
    print('Seizure data extracted!!!!')



print('Splitting the seizure data into time windows of ',time_window)
fname_std = dest_addr + '//' + 'preictal_'
    
num_of_splits = (seizure.shape[1])/(256*time_window)
print('Number of splits are coming out to be ',num_of_splits)

#if round(num_of_splits) != num_of_splits:
#    trim_value = time_window*256*(num_of_splits - floor(num_of_splits))
#    chb = chb[:,0:chb.shape[1] - trim_value]
#    num_of_splits = floor(num_of_splits)

split_array = np.hsplit(seizure,num_of_splits)

    
for sample in split_array:
    fname = fname_std + str(count) + '.csv'
    np.savetxt(fname, sample, delimiter = ',', fmt = '%f')
    if count%10 ==0:
        print('-------interictal_',count,'.csv','--------- FORMED!!!\n')
            
    count = count+1
    
f._close()
 