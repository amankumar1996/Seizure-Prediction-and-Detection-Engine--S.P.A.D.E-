# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 06:26:11 2018

@author: amankumar
"""

import numpy as np
import pandas as pd
import os
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


preictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/band_pass_filtered/chb01/bpf_preictal_feature_data.csv',
                                    header = None)).as_matrix()

interictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/band_pass_filtered/chb01/bpf_interictal_feature_data.csv',
                                    header = None)).as_matrix()



zero_col = np.zeros(interictal_feature_data.shape[0])
one_col = np.ones(preictal_feature_data.shape[0])



Y = np.concatenate((zero_col,one_col), axis = 0)
X = np.concatenate((interictal_feature_data,preictal_feature_data),axis = 0) #concatenate along rows


selector = SelectKBest(score_func = f_classif,k=180)
X = selector.fit_transform(X,Y)

#Fitting the SVM to the Training set

classifier = SVC(kernel = 'linear',random_state=0)

#selector = RFE(classifier, 110)
#X = selector.fit_transform(X,Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state = 5)


classifier.fit(x_train,y_train)


#Predicting the test results
y_pred = classifier.predict(x_test)

#Making the confusion matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test,y_pred)
sensitivity = float(tp)/(tp+fn)
specificity = float(tn)/(tn+fp)
precision = float(tp)/(tp+fp)
roc_score = roc_auc_score(y_test, y_pred)
f1Score = f1_score(y_test,y_pred)

print 'Accuracy = ',acc*100
print 'Sensitivity = ',sensitivity*100
print 'Specificity = ',specificity*100
print 'Precision = ',precision*100
print 'auc_roc_score = ',roc_score
print 'f1_Score = ',f1Score

##################
#           when KBest feature selection

selected_features = selector.get_support()
features = []
for i in range(1,len(preictal_feature_data[0]) + 1):
    if selected_features[i-1]==True:
        features.append(i)
print 'features_selected = ',features

corres_channel = {}       
for i in range(1,(preictal_feature_data.shape[1]/80)+1):
    corres_channel[i]=0


for item in features:
    corres_channel[floor(item/80)+1]+=1 

print 'corres_channel = ',corres_channel    

sorted_corres_channel = sorted(corres_channel, key=corres_channel.get)
sorted_corres_channel = sorted(corres_channel.items(), key = lambda x: x[1], reverse = True)
print 'sorted_corres_channel = ',sorted_corres_channel