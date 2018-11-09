# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:57:56 2018

@author: Aman Kumar
"""
from sklearn.externals import joblib
from spade import WPD, extract_features
import pyedflib
import numpy as np
from math import floor

channels = None
time_window = 10
sampling_freq = 256
epoch_length = time_window*sampling_freq

model = joblib.load('F:\\BCI project\\Development\\trained_model_TestingModule.pkl')
eeg_data_addr = 'F:\\BCI project\\Dataset\\CHB-MIT\\chb01\\'


# Reading data from .edf file
f = pyedflib.EdfReader(eeg_data_addr + 'chb01_01.edf')
n = f.signals_in_file
chb01_01 = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
    chb01_01[i, :] = f.readSignal(i)  
    
print('Reading done!!!')
f._close()


# Reading data from .edf file
f = pyedflib.EdfReader(eeg_data_addr + 'preictal\\' + 'chb01_02.edf')
n = f.signals_in_file
chb01_02 = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
    chb01_02[i, :] = f.readSignal(i)  
    
print('Reading done!!!')
f._close()


## Reading data from .edf file
#f = pyedflib.EdfReader(eeg_data_addr + 'preictal\\' + 'chb01_03.edf')
#n = f.signals_in_file
#chb01_03 = np.zeros((n, f.getNSamples()[0]))
#
#for i in np.arange(n):
#    chb01_03[i, :] = f.readSignal(i)  
#    
#print('Reading done!!!')
#f._close()

if channels == None:
    channels = range(chb01_01.shape[0])

level = 4
print '--------------- Loaded the trained model --------------'
a = raw_input()

time_to_go = 170 #in minutes
print 'Starting the analysis of EEG signals ', int(time_to_go/60),'hours ',int(time_to_go%60),'minutes before the onset'
a = raw_input()
for t in range(epoch_length,chb01_01.shape[1],epoch_length):
    epoch = chb01_01[channels,(t-epoch_length):t]
    feature_vector = []
    for ch in channels:
        coefficients = WPD(epoch[ch,:],level)
        features = extract_features( coefficients )
        feature_vector.extend(features)
        
    
    if model.predict( (np.array(feature_vector)).reshape(1,-1) ) == 1:
        print '******  Seizure predicted (', floor(time_to_go/60), 'hours ', (time_to_go%60), 'minutes before onset) *****'
    else:
        if (float(t-epoch_length)/(256*60))%2 == 0:
            print 'No seizure predicted -- ', '(', floor(time_to_go/60), 'hours ', (time_to_go%60), 'minutes to go)'
    time_to_go = time_to_go - float(10)/60


time_to_go = 110 #in minutes
print 'Starting the analysis of EEG signals ', int(time_to_go/60),'hours ',int(time_to_go%60),'minutes before the onset'
a = raw_input()
for t in range(epoch_length,chb01_02.shape[1],epoch_length):
    epoch = chb01_02[channels,(t-epoch_length):t]
    feature_vector = []
    for ch in channels:
        coefficients = WPD(epoch[ch,:],level)
        features = extract_features( coefficients )
        feature_vector.extend(features)
        
    
    if model.predict( (np.array(feature_vector)).reshape(1,-1) ) == 1:
        print '******  Seizure predicted (', floor(time_to_go/60), 'hours ', (time_to_go%60), 'minutes before onset) *****'
    else:
        if (float(t-epoch_length)/(256*60))%2 == 0:
            print 'No seizure predicted -- ', '(', floor(time_to_go/60), 'hours ', (time_to_go%60), 'minutes to go)'
    time_to_go = time_to_go - float(10)/60


#decision = np.zeros((chb01_01.shape[1]/epoch_length,2))
#i = 0
#for t in range(epoch_length,chb01_01.shape[1],epoch_length):
#    epoch = chb01_01[channels,(t-epoch_length):t]
#    feature_vector = []
#    for ch in channels:
#        coefficients = WPD(epoch[ch,:],level)
#        features = extract_features( coefficients )
#        feature_vector.extend(features)
#        
#    
#    if model.predict( (np.array(feature_vector)).reshape(1,-1) ) == 1:
#        decision[i,0] = model.decision_function((np.array(feature_vector)).reshape(1,-1))
#        decision[i,1] = 1
#    else:
#        decision[i,0] = model.decision_function((np.array(feature_vector)).reshape(1,-1))
#        decision[i,1] = 0
#        
#    i = i+1
#            
#
#
#def smooth(signal, n):
#    smooth_signal = np.zeros((1,signal.shape[1]-n))
#    
#    for i in range(smooth_signal.shape[1]):
#        smooth_signal[0,i] = sum(signal[0,i:(i+n)])/n
#        
#    return smooth_signal
#
#
#import matplotlib.pyplot as pp
#smoothened_decision = smooth(decision[:,0].reshape(1,-1),30)
#pp.plot(smoothened_decision[0,:])


((170 - float(10)/60)%60)

float(10/60)
