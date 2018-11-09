# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:23:24 2018

@author: Aman Kumar
"""


import scipy.io

mat = scipy.io.loadmat('Dog_5_interictal_segment_0001.mat')

arr = mat['interictal_segment_1']

f = pyedflib.EdfReader("chb01_01.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)



#######################################################################
#Things that worked
      
import numpy as np
import pyedflib
import matplotlib.pyplot as plt

#extracting data from edf file
f = pyedflib.EdfReader("chb01_03.edf")
n = f.signals_in_file
signal_labels1 = f.getSignalLabels()
chb01_03 = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
    chb01_03[i, :] = f.readSignal(i)
    
t = 10
for i in range(0,110,10):
    start_time = 2900 + i
    X = range(0,256*t) 
    #plotting
    plt.plot(X,chb01_03[0,start_time*256:(start_time*256 + t*256)])
    plt.plot(X,chb01_01[0,start_time*256:(start_time*256 + t*256)])

f = pyedflib.EdfReader("chb01_01.edf")
n = f.signals_in_file
signal_labels1 = f.getSignalLabels()
chb01_01 = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
    chb01_01[i, :] = f.readSignal(i)
    
    
fft_output = np.fft.rfft(chb01_03)     # Perform real ff

fft_mag = np.abs(fft_output)    # Take only magnitude of spectrum

     
real_fft_output= fft_output.real
start_time = 100
X = range(0,256*t) 
#plotting
plt.plot(X,real_fft_output[0,start_time*256:(start_time*256 + t*256)])

imag_fft_output= fft_output.imag
t = 20/256
start_time = 0
X = range(0,int(256*t)) 
#plotting
plt.plot(X,imag_fft_output[0,start_time*256:(start_time*256 + int(t*256))])

dt = float(1/256)
n = len(chb01_03)
rfreqs = np.fft.rfftfreq(n, dt)  # Calculatel frequency bins


#trying fourier transform as feature extraction
interictal_sample = chb01_01[12, 12000:(12000 + 256*20)]
ictal_sample = chb01_03[12, 3000*256:3000*256+256*20]
inter_output = np.fft.rfft(interictal_sample)
inctal_output = np.fft.rfft(ictal_sample)
inter_freqs = np.fft.rfftfreq(len(interictal_sample),dt)
ictal_freqs = np.fft.rfftfreq(len(ictal_sample),dt)