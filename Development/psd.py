# -*- coding: utf-8 -*-
"""
Created on Tue May 29 05:29:13 2018

@author: amankumar
"""

from mne.time_frequency import psd_welch 
import pandas as pd
import numpy as np
from matplotlib.pyplot import plot

from scipy.signal import welch

source_addr = '//media//amankumar//Pro//BCI project//processed_dataset//band_pass_filtered//chb01//preictal'
data = pd.read_csv(source_addr + '//bpf_preictal_100.csv',header=None)

data = data.as_matrix()


test_welch = welch(data, fs = 60)

plot(test_welch[1,:)