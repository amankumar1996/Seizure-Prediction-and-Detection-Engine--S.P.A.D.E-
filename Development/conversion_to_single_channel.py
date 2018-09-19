# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 04:31:04 2018

@author: amankumar
"""

import pandas as pd
import numpy as np

preictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_feature_data.csv',
                                    header = None)).as_matrix()

interictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal_feature_data.csv',
                                    header = None)).as_matrix()
                                    

channel_no = 17

preictal_feature_data = preictal_feature_data[:,(channel_no-1)*80:channel_no*80]
interictal_feature_data = interictal_feature_data[:,(channel_no-1)*80:channel_no*80]

#saving the dataset
np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/singleChannel_interictal_feature_data.csv',
           interictal_feature_data,fmt = '%f', delimiter = ',')
           
np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/singleChannel_preictal_feature_data.csv',
           preictal_feature_data,fmt = '%f', delimiter = ',')