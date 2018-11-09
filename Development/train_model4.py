# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:07:17 2018

@author: Aman Kumar
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
from sklearn.externals import joblib


preictal_addr = 'F:\\BCI project\\processed_dataset\\CHB-MIT\\chb01\\'
interictal_addr = 'F:\\BCI project\\processed_dataset\\CHB-MIT\\chb01\\interictal_feature_data_6000.csv'

file_names = ['preictal_feature_data_5_10', 'preictal_feature_data_10_15', 'preictal_feature_data_15_20', 
'preictal_feature_data_20_25', 'preictal_feature_data_25_30', 'preictal_feature_data_30_35', 
'preictal_feature_data_35_40', 'preictal_feature_data_40_45', 'preictal_feature_data_45_50']

preictal_feature_data = (pd.read_csv(preictal_addr +  file_names[0] + '.csv', header = None)).as_matrix()
for file_name in file_names[1:]:
    preictal = (pd.read_csv(preictal_addr +  file_name + '.csv', header = None)).as_matrix()
    preictal_feature_data = np.concatenate((preictal_feature_data,preictal), axis = 0)
    

interictal_feature_data = (pd.read_csv(interictal_addr,
                           header = None)).as_matrix()

random_values = set()
while True:
    random_values.add(random.randint(0, interictal_feature_data.shape[0] - 1))
    if len(random_values) >= preictal_feature_data.shape[0]:
        break

interictal_feature_data = interictal_feature_data[np.array(list(random_values)), :]


zero_col = np.zeros(interictal_feature_data.shape[0])
one_col = np.ones(preictal_feature_data.shape[0])

Y = np.concatenate((zero_col, one_col), axis=0)
X = np.concatenate((interictal_feature_data, preictal_feature_data), axis=0)  # concatenate along rows

# Fitting the SVM to the Training set
classifier = SVC(kernel='linear', random_state=0)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=5)

print 'Training of SVM classifier has started....for  '

classifier.fit(x_train, y_train)

# Predicting the test results
y_pred = classifier.predict(x_test)

# Making the confusion matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
sensitivity = float(tp) / (tp + fn)
specificity = float(tn) / (tn + fp)
precision = float(tp) / (tp + fp)
roc_score = roc_auc_score(y_test, y_pred)
f1Score = f1_score(y_test, y_pred)

print 'Accuracy = ', acc * 100
print 'Sensitivity = ', sensitivity * 100
print 'Specificity = ', specificity * 100
print 'Precision = ', precision * 100
print 'auc_roc_score = ', roc_score
print 'f1_Score = ', f1Score, '\n\n'

#saving model
joblib.dump(classifier, 'F:\\BCI project\\Development\\' + 'trained_model_TestingModule.pkl' )