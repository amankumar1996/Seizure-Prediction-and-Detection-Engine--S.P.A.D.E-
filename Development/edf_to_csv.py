# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:06:55 2018

@author: Aman Kumar
"""

###################################################################
#                                                                 
#    Converting .edf files into .csv format of CHB-MIT dataset
#                   AND PERFORMS EPOCHING OF 10 SECONDS
#                                                              
###################################################################


import os
import pyedflib
import numpy as np
import pandas as pd
from copy import deepcopy
from math import floor

source_address = 'F:\BCI project\Dataset\CHB-MIT\chb03'
os.chdir(source_address)

edf_files = []


for file in os.listdir(source_address):
    if file.endswith(".edf"):
        edf_files.append(file)
        
dest_address = "F:\BCI project\processed_dataset\CHB-MIT\chb03"

#dataset = np.ndarray(shape=(23,))
f = pyedflib.EdfReader(edf_files[0])
n = f.signals_in_file
signal_labels1 = f.getSignalLabels()
chb = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
    chb[i, :] = f.readSignal(i)  


count = 0
time_window = 10
fname_std = dest_address + '\\' + 'interictal_'
for file in edf_files[15:]:
    print('Reading - ',file , '.......')
    f = pyedflib.EdfReader(file)
    n = f.signals_in_file
    signal_labels1 = f.getSignalLabels()
    chb = np.zeros((n, f.getNSamples()[0]))

    for i in np.arange(n):
        chb[i, :] = f.readSignal(i)
        
    print('Reading " ',file,' " is successful and complete ')
    
    num_of_splits = (chb.shape[1])/(256*time_window)
    print('Number of splits are coming out to be ',num_of_splits)
    
    if round(num_of_splits) != num_of_splits:
        trim_value = time_window*256*(num_of_splits - floor(num_of_splits))
        chb = chb[:,0:chb.shape[1] - trim_value]
        num_of_splits = floor(num_of_splits)

    split_array = np.hsplit(chb,num_of_splits)
    print('Split array formed of = ',file)
        
    for sample in split_array:
        fname = fname_std + str(count) + '.csv'
        np.savetxt(fname, sample, delimiter = ',', fmt = '%f')
        if count%60 ==0:
            print('-------interictal_',count,'.csv','--------- FORMED!!!\n')
                
        count = count+1
        
    f._close()
 
            
            
            
#else:
#print("round(num_of_splits) == num_of_splits was not equal. That's why break; " )
#break
    


#chb_test = pd.read_csv(fname,header=None)
#fname = dest_address + '\\' + file[:-4] + '.csv'
#np.savetxt(fname, chb, delimiter = ',', fmt = '%d')
#np.concatenate((dataset,chb),axis = 1)
#dataset = deepcopy(chb)