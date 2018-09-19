# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:13:07 2018

@author: amankumar
"""

# -*- coding: utf-8 -*-

import pyedflib
import numpy as np
import pandas as pd
from math import floor
#Initialize the following variables
count = 0   #CAUTION!! - input this after proper inspection - it will be the file number

source_addr = '/media/amankumar/Pro/BCI project/Dataset/CHB-MIT/chb01/preictal'
dest_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_55_60'

start_time =  [2996, 1732, 1720, 327, 1862] #1467 & 1015 tricky ones    #in seconds as provided in patient summary file
seizure_fileno = ['03', '15',  '18', '21', '26']
seizure_prev_fileno = ['02', '14', '17', '20', '25']
prediction_start_time = 60        #how many minutes before the onset of seizure in minutes
#taking duration before end to be 5 minutes
sampling_freq = 256

overlap = 0.75
time_window = 10 #in seconds: it is preferred that 60 is divisible by time_window size because otherwise
#[contd.] no of splits won't be an integer, and then code need to be set accordingly

def splitArray(seizure, length):
    
    num_of_splits = int(floor(seizure.shape[1]/length))
    print('Number of splits are coming out to be ',num_of_splits)
    splits = np.zeros((num_of_splits, seizure.shape[0],int(length)))
    for i in range(0,num_of_splits):
        splits[i,:,:] = seizure[:,i*int(length):i*int(length) + int(length)]
    
    return splits

prediction_start_time *= 60 #coverting into seconds

for k in range(0, len(start_time)):
    if  (start_time[k]-prediction_start_time) >= 0:
        # Reading data from .edf file
        f = pyedflib.EdfReader(source_addr + '/chb01_' + seizure_fileno[k] + '.edf')
        n = f.signals_in_file
        seizure = np.zeros((n, f.getNSamples()[0]))
    
        for i in np.arange(n):
            seizure[i, :] = f.readSignal(i)  
            
        print('Reading done!!!')
        
        seizure = seizure[:,(start_time[k]-prediction_start_time)*sampling_freq:(start_time[k]-prediction_start_time+300)*sampling_freq]
        print('Seizure data extracted!!!!')
        
        
        
    elif (start_time[k]-prediction_start_time + 300) >= 0:
        print('-----Seizure data is in both files----')
        print('Reading data from .edf files')
        # Reading data from .edf files
        f = pyedflib.EdfReader(source_addr + '/chb01_' + seizure_fileno[k] + '.edf')
        n = f.signals_in_file
        seizure = np.zeros((n, f.getNSamples()[0]))
        
        for i in np.arange(n):
            seizure[i, :] = f.readSignal(i)  
            
        
        f = pyedflib.EdfReader(source_addr + '/chb01_' + seizure_prev_fileno[k] + '.edf')
        n = f.signals_in_file
        seizure_prev = np.zeros((n, f.getNSamples()[0]))
        
        for i in np.arange(n):
            seizure_prev[i, :] = f.readSignal(i)  
                
        print('Reading done!!!')
        #extracting component from seizure_prev part
        start_index = seizure_prev.shape[1]- (prediction_start_time - start_time[k])*sampling_freq
        part1 = seizure_prev[:,start_index:]
        end_index = (start_time[k] - prediction_start_time + 300)*sampling_freq
        part2 = seizure[:,0:end_index]
        
        #concatening both
        seizure = np.concatenate((part1,part2), axis = 1)
        print('Seizure data extracted!!!!')
    
        
        
    else:
        # Reading data from .edf file
        f = pyedflib.EdfReader(source_addr + '/chb01_' + seizure_prev_fileno[k] + '.edf')
        n = f.signals_in_file
        seizure = np.zeros((n, f.getNSamples()[0]))
    
        for i in np.arange(n):
            seizure[i, :] = f.readSignal(i)  
            
        print('Reading done!!!')
        start_index = seizure.shape[1] - (prediction_start_time - start_time[k])*sampling_freq
        
        seizure = seizure[:,start_index:start_index + 300*sampling_freq]
        print('Seizure data extracted!!!!')
        
        
    print('Splitting the seizure data into time windows of ',time_window)
    fname_std = dest_addr + '//' + 'preictal_55_60_'
    iters = (1/(1-overlap))-1
    
    for i in range(0,int(iters)):        
        #if round(num_of_splits) != num_of_splits:
        #    trim_value = time_window*256*(num_of_splits - floor(num_of_splits))
        #    chb = chb[:,0:chb.shape[1] - trim_value]
        #    num_of_splits = floor(num_of_splits)
        
        split_array = splitArray(seizure,sampling_freq*time_window)        
        for sample in split_array:
            fname = fname_std + str(count) + '.csv'
            np.savetxt(fname, sample, delimiter = ',', fmt = '%f')
            if count%10 ==0:
                print('-------preictal_',count,'.csv','--------- FORMED!!!\n')
                    
            count = count+1
        
        seizure = seizure[:,int(sampling_freq*time_window*(1-overlap)):]
      
      
    f._close()

##############################***************#####################


