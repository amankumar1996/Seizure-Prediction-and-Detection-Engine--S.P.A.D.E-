# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 05:03:14 2018

@author: amankumar
"""
import numpy as np
import scipy.linalg as la
import pandas as pd
import os


# Common Spatial Pattern implementation in Python, used to build spatial filters for identifying task-related activity.

# CSP takes any number of arguments, but each argument must be a collection of trials associated with a task
# That is, for N tasks, N arrays are passed to CSP each with dimensionality (# of trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has the same dimensionality,
# otherwise there can be no spatial filtering since the trials cannot be compared
def CSP(*tasks):
	if len(tasks) < 2:
		print "Must have at least 2 tasks for filtering."
		return (None,) * len(tasks)
	else:
		filters = ()
		# CSP algorithm
		# For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
		print('Number of classes: ',len(tasks))
        iterator = range(0,len(tasks))
        for x in iterator:
			# Find Rx
            print('Size of one epoch', tasks[x][0].shape)
            Rx = covarianceMatrix(tasks[x][0])
            for t in range(1,len(tasks[x])):
				Rx += covarianceMatrix(tasks[x][t])
            Rx = Rx / len(tasks[x])
            print 'Covariance matrix of first population is computed!!'
			# Find not_Rx
            count = 0
            not_Rx = Rx * 0
            for not_x in [element for element in iterator if element != x]:
				for t in range(0,len(tasks[not_x])):
					not_Rx += covarianceMatrix(tasks[not_x][t])
					count += 1
            not_Rx = not_Rx / count
            print 'Covariance of second population is computed'
			# Find the spatial filter SFx
            SFx = spatialFilter(Rx,not_Rx)
            print 'One spatial filter is ready!'
            filters += (SFx,)

			# Special case: only two tasks, no need to compute any more mean variances
            if len(tasks) == 2:
				filters += (spatialFilter(not_Rx,Rx),)
				break
        print 'Other spatial filter is also done!'
        return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
    R = Ra + Rb
    E,U = la.eig(R)
   
    # CSP requires the eigenvalues E and eigenvector U be sorted in descending order
    ord = np.argsort(E)
    ord = ord[::-1] # argsort gives ascending order, flip to get descending
    E = E[ord]
    U = U[:,ord]
    
    # Find the whitening transformation matrix
    P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))
    
    # The mean covariance matrices may now be transformed
    Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
    Sb = np.dot(P,np.dot(Rb,np.transpose(P)))
    
    # Find and sort the generalized eigenvalues and eigenvector
    E1,U1 = la.eig(Sa,Sb)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:,ord1]
    
    # The projection matrix (the spatial filter) may now be obtained
    SFa = np.dot(np.transpose(U1),P)
    return SFa.astype(np.float32)



################### main program #############################
num_of_channels = 23
sampling_freq = 256 #in Hz
length_of_time_window = 10 #in seconds
length_of_epoch = sampling_freq*length_of_time_window
# Loading Preictal epochs
preictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/'
folder_names = ['preictal_5_10']#,'preictal_10_15','preictal_15_20','preictal_20_25',
                #'preictal_25_30','preictal_30_35','preictal_35_40','preictal_40_45','preictal_45_50']

preictal_files = []
for folder_name in folder_names:
    for file_name in os.listdir(preictal_addr+folder_name):
        if file_name.endswith(".csv"):
            preictal_files.append(preictal_addr+folder_name+ '//' + file_name)

preictal = np.zeros((len(preictal_files),num_of_channels,length_of_epoch))
count = 0
for files in preictal_files:
    preictal[count,:,:] = (pd.read_csv(files,header=None)).as_matrix()
    if(count%50==0):    
        print(count)
    count = count + 1

# Loading interictal epochs
interictal_addr = '/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/interictal'

interictal_files = []
for file_name in os.listdir(interictal_addr):
    if file_name.endswith(".csv"):
       interictal_files.append(interictal_addr + '//' +file_name)

interictal = np.zeros((len(preictal_files),num_of_channels,length_of_epoch))
count = 0
for files in interictal_files[:len(preictal_files)]:
    interictal[count,:,:] = (pd.read_csv(files,header=None)).as_matrix()
    if(count%50==0):
        print(count)
    count = count + 1

fs = CSP(preictal,interictal)
filterA = fs[0]

filterB = fs[1]

np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/csp_filterA_5_10.csv',
           filterA,fmt = '%f', delimiter = ',')
           
np.savetxt('/media/amankumar/Pro/BCI project/processed_dataset/CHB-MIT/chb01/csp_filterB_5_10.csv',
           filterB,fmt = '%f', delimiter = ',')


##normalize the X
#X_bar = np.divide(preictal[0],np.trace(np.dot(preictal[0],np.transpose(preictal[0]))))
#F = np.dot(filterA,X_bar)
#F_square = F**2
#f=[]
#for i in range(0,F.shape[0]):
#    f.append(sum(F_square[i])/2560)
##normalize the X
#Xb_bar = np.divide(interictal[0],np.trace(np.dot(interictal[0],np.transpose(interictal[0]))))
#Fb = np.dot(filterB,Xb_bar)
#Fb_square = Fb**2
#f_b=[]
#for i in range(0,Fb.shape[0]):
#    f_b.append(sum(Fb_square[i])/2560)
#
#
#import matplotlib.pyplot as pp
#pp.plot(range(1, len(f)+1), f)
#
#
#filtered_preictal = 0*preictal
#filtered_interictal = 0*interictal
#for i  in range(0,len(preictal)):    
#    filtered_preictal[i] = np.dot(filterA, preictal[i])
#    filtered_interictal[i] = np.dot(filterA, interictal[i])
   

    