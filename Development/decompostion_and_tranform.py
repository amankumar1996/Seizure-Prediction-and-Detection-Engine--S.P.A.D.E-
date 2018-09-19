# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:38:54 2018

@author: amankumar
"""

import numpy as np
import pandas as pd
import os
from pywt import wavedec, WaveletPacket

address = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal'

os.chdir(address)

eeg_signal = (pd.read_csv('preictal_300.csv',header = None)).as_matrix()

channel = eeg_signal[0,:]

#Discrete Wavelet Transform
dwt_coeffs = wavedec(channel, 'db4', level = 6)

#Wavelet Packet Decomposition
wpd = WaveletPacket(data = channel, wavelet='db4', mode = 'symmetric', maxlevel = 4)
node_names = [node.path for node in wpd.get_level(4)]

wpd_coeffs = []
for names in node_names:
    wpd_coeffs.append(wpd[names].data)