# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 04:53:39 2018

@author: amankumar
"""

import numpy as np
import pandas as pd
import os
from time import time
import random
import matplotlib.pyplot as pp
from math import floor
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score

# preictal1 = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_feature_data_50_55.csv',
#                                    header = None)).as_matrix()
#
# preictal2 = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_feature_data_55_60.csv',
#                                    header = None)).as_matrix()

# '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_newFeature_data_5_50.csv'
#  '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal_newFeature_data_5_50.csv'
preictal_addr = 'F:\BCI project\processed_dataset\CHB-MIT\chb01\preictal_newFeature_data_5_50.csv'
interictal_addr = 'F:\BCI project\processed_dataset\CHB-MIT\chb01\interictal_newFeature_data_5_50.csv'
preictal_feature_data = (pd.read_csv(preictal_addr,
                         header = None)).as_matrix()

interictal_feature_data = (pd.read_csv(interictal_addr,
                           header = None)).as_matrix()

# preictal_feature_data = np.concatenate((preictal_feature_data,preictal1, preictal2),axis = 0)
##preictal_feature_data = preictal2
num_of_features_per_channel = 112
folder_names = ['preictal_5_10', 'preictal_10_15', 'preictal_15_20', 'preictal_20_25',
                'preictal_25_30', 'preictal_30_35', 'preictal_35_40', 'preictal_40_45', 'preictal_45_50']

corres_num_of_epochs = [0, 616, 616, 616, 616, 528, 528, 528, 440, 352]
corres_num_of_epochs = np.cumsum(corres_num_of_epochs)
n_RFE_features = [110, 110, 110, 110, 110, 85, 65, 65, 65]
for i in range(corres_num_of_epochs.shape[0] - 1):
    new_preictal_feature_data = preictal_feature_data[corres_num_of_epochs[i]:corres_num_of_epochs[i + 1], :]
    print 'Rows are selected for: ', folder_names[i]
    print 'From ', corres_num_of_epochs[i], ' to ', corres_num_of_epochs[i + 1], ' of length ', new_preictal_feature_data.shape[0]

    random_values = set()
    while True:
        random_values.add(random.randint(0, interictal_feature_data.shape[0] - 1))
        if len(random_values) >= new_preictal_feature_data.shape[0]:
            break

    new_interictal_feature_data = interictal_feature_data[np.array(list(random_values)), :]
    print '\nInterictal dataset also created....'
    zero_col = np.zeros(new_interictal_feature_data.shape[0])
    one_col = np.ones(new_preictal_feature_data.shape[0])

    Y = np.concatenate((zero_col, one_col), axis=0)
    X = np.concatenate((new_interictal_feature_data, new_preictal_feature_data), axis=0)  # concatenate along rows
    #channels = [8]
    #channel_features = []
    #for channel in channels:
    #    channel_features.extend(
    #       range((channel - 1) * num_of_features_per_channel, channel * num_of_features_per_channel))

    #X = X[:, channel_features]
    #print 'Channels: ', channels, ' , are extracted'
    # selector = SelectKBest(score_func = f_classif,k=110)
    # X = selector.fit_transform(X,Y)

    # Fitting the SVM to the Training set

    classifier = SVC(kernel='linear', random_state=0)
    selector = RFE(classifier, n_RFE_features[i])
    print 'RFE has started for ', folder_names[i]
    start_time = time()
    X = selector.fit_transform(X,Y)
    end_time = time()
    print 'Time taken for RFE : ', (end_time - start_time) / 60, ' minutes'

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=5)

    print 'Training of SVM classifier has started....for  ', folder_names[i]
    start_time = time()
    classifier.fit(x_train, y_train)

    # Predicting the test results
    y_pred = classifier.predict(x_test)

    end_time = time()
    print 'Time taken to train : ', (end_time - start_time) / 60, ' minutes'
    # Making the confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    precision = float(tp) / (tp + fp)
    roc_score = roc_auc_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)

    print '----------------  All channels, ', folder_names[i], ', RFE = ', n_RFE_features[i], ' features\n'

    print 'Accuracy = ', acc * 100
    print 'Sensitivity = ', sensitivity * 100
    print 'Specificity = ', specificity * 100
    print 'Precision = ', precision * 100
    print 'auc_roc_score = ', roc_score
    print 'f1_Score = ', f1Score, '\n\n'

    ##################
    #           when KBest feature selection
    #
    selected_features = selector.get_support()
    features = []
    for i in range(1,len(preictal_feature_data[0]) + 1):
        if selected_features[i-1]==True:
            features.append(i)
    print 'features_selected = ',features, '\n'
    #
    #
    corres_channel = {}
    for i in range(1,(preictal_feature_data.shape[1]/num_of_features_per_channel)+1):
       corres_channel[i]=0
    #
    #
    for item in features:
       corres_channel[floor(item/num_of_features_per_channel)+1]+=1

    print 'corres_channel = ',corres_channel, '\n'
    #
    sorted_corres_channel = sorted(corres_channel, key=corres_channel.get)
    sorted_corres_channel = sorted(corres_channel.items(), key = lambda x: x[1], reverse = True)
    print 'sorted_corres_channel = ',sorted_corres_channel, '\n\n'




    # os.chdir('/media/amankumar/Pro/BCI project/Development')
    # selected_features = np.loadtxt('feature_selection_results_support', delimiter = '\n', dtype = 'int16')
    # features = []
    # for i in range(0,len(preictal_feature_data[0])):
    #    if selected_features[i]==1:
    #        features.append(i)
    #
    # interictal_feature_data = interictal_feature_data[:,features]
    # preictal_feature_data = preictal_feature_data[:,features]
    #
    # print features