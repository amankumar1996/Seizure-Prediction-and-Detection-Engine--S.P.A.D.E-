# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:37:41 2018

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

preictal1 = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_feature_data_50_55.csv',
                                    header = None)).as_matrix()

preictal2 = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_feature_data_55_60.csv',
                                    header = None)).as_matrix()

preictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/preictal_feature_data_5_50.csv',
                                    header = None)).as_matrix()

interictal_feature_data = (pd.read_csv('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal_feature_data_6000.csv',
                                    header = None)).as_matrix()

preictal_feature_data = np.concatenate((preictal_feature_data,preictal1, preictal2),axis = 0)
#preictal_feature_data = preictal2                            
random_values = set()
while True:
  random_values.add(random.randint(0,interictal_feature_data.shape[0]-1))
  if len(random_values) >= preictal_feature_data.shape[0]:
    break
    
interictal_feature_data = interictal_feature_data[np.array(list(random_values)),:]

zero_col = np.zeros(interictal_feature_data.shape[0])
one_col = np.ones(preictal_feature_data.shape[0])



Y = np.concatenate((zero_col,one_col), axis = 0)
X = np.concatenate((interictal_feature_data,preictal_feature_data),axis = 0) #concatenate along rows

#PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(X)
scaling_values = sc.scale_
means = sc.mean_
variances = sc.var_ 

#Understanding standard scaler
#from math import sqrt
#X[0][0]/scaling_values[0]
#(X[0][0] - min(X[:,0]))/(max(X[:,0]) - min(X[:,0]))
#(X[0][0] - means[0])/sqrt(variances[0])
#0.38676457085930155 - scaled
#139.29593 - original
#65.033 - scaling value
#from numpy import mean, var

from sklearn.decomposition import PCA
pca = PCA(n_components = 665)
x_pca = pca.fit_transform(x_scaled)

components = pca.components_ 
explained_variance = pca.explained_variance_ 
explained_variance_ratio = pca.explained_variance_ratio_ 
singular_values = pca.singular_values_ 
n_components = pca.n_components

for i in range(1,len(explained_variance_ratio)):
    if sum(explained_variance_ratio[:i]) >= 0.99:
        print i
        break
    
classifier = SVC(kernel = 'linear',random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x_pca,Y, test_size=0.20, random_state = 5)


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




