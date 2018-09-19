# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:09:59 2018

@author: Aman Kumar
"""
# Importing libraries
import os
from os import mkdir

import numpy as np
from numpy import mean, std
from scipy.stats import skew, kurtosis
import pandas as pd

import pyedflib
from pywt import WaveletPacket


import random
from math import floor

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.externals import joblib

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score


def create_dir(des):
    '''
    This function will create folders mentioned in "folders_to_create" and a folder named 
    "info_files" at directory address passed on this function
    
    Parameter - Destination address, where you want to create directories for processed data and 
    trained model
    
    Returns - This function doesn't return anything
    
    '''
    folders_to_create = ['preictal_5_10','preictal_10_15','preictal_15_20','preictal_20_25','preictal_25_30','preictal_30_35','preictal_35_40',
                         'preictal_40_45','preictal_45_50','preictal_50_55','preictal_55_60', 'interictal']

    main_dir = 'SPADE'
    dataset_dir = 'processed_data'

    #creating main directory
    main_path = des + '/' + main_dir
    mkdir(main_path)

    #creating directory for dataset
    dataset_path = main_path + '/' + dataset_dir
    mkdir(dataset_path)

    for folder in folders_to_create:
        folder_path = dataset_path + '/' + folder
        mkdir(folder_path)


    mkdir(main_path + '/' + 'info_files')
    
#def create_system_info_file(des, ): #postponed till futther discussion
    
    
def fetch_system_info(addr):
    '''
    This function extracts the system information from the 'system_info.txt' file
    
    Parameter - 
    Addr: Address of the folder which contains 'info_files' folder 
    
    Returns (in this order only) -
    Number of channels: an integer
    Channel names: a list object
    Sampling frequency: an integer
    '''
    addr = addr + '//info_files//system_info.txt'
    system_info = pd.read_csv(addr, sep = '\t').as_matrix()
    #system_info[0,1] gives the number of channels
    #system_info[1,1] gives the string of channel names separated by commas
    #system_info[2,1] gives sampling frequency
    return int(system_info[0,1]), str(system_info[1,1]).split(sep = ','), int(system_info[2,1])
   
   
#def create_preictal_info_file(des, ): #postponed till futther discussion
    

def fetch_preictal_info(addr):
    '''
    This function extracts the details of seizures from the 'preictal_info.txt' file
    
    Parameter - 
    Addr: Address of the folder which contains 'info_files' folder 
    
    Returns -
    Preictal information - a matrix object
    It contains information about edf files containing seizures, for example name of the file 
    containing seizure, start time of the seizure in that file, minutes of the preictal data that 
    can be extracted safely having atleast 1 hour time gap from any other seizure activity 
    '''
    addr = addr + '//info_files//preictal_info.txt'
    preictal_info = pd.read_csv(addr, sep = '\t').as_matrix()

    return preictal_info
    
    
def fetch_interictal_info(addr):
    '''
    This function extracts the details of files containing normal state from the 'interictal_info.txt' file
    
    Parameter - 
    Addr: Address of the folder which contains 'info_files' folder 
    
    Returns -
    Preictal information - a matrix object
    It contains information about edf files containing seizures, for example name of the file 
    containing seizure, start time of the seizure in that file, minutes of the preictal data that 
    can be extracted safely having atleast 1 hour time gap from any other seizure activity 
    '''
    addr = addr + '//info_files//preictal_info.txt'
    preictal_info = pd.read_csv(addr, sep = '\t').as_matrix()

    return preictal_info



def splitArray(seizure, length):
    '''
    This function splits the EEG data "seizure" into epochs of length "length"
    
    Parameters -
    seizure: It should be a matrix type object.
    length: integer type variable. It is the length in which you want to split the EEG segment
    
    
    '''
    num_of_splits = int(floor(seizure.shape[1]/length))
    print('Number of splits are coming out to be ',num_of_splits)
    splits = np.zeros((num_of_splits, seizure.shape[0],int(length)))
    for i in range(0,num_of_splits):
        splits[i,:,:] = seizure[:,i*int(length):i*int(length) + int(length)]
    
    return splits


def extract_seizure_data(src, des, preictal_info, interictal_info, sampling_freq = 256, time_window = 10, overlap = 0.75):
    '''
    
    
    
    '''
    des = des + '//processed_data'
    # declaring array variable to keep counter of number of files in each folder 
    # index 0 will keep counter for "preictal_5_10"; index 11 will keep counter for "preictal_10_15"
    # and so on till index 10 for "preictal_55_60"
    # Index 11 is for keeping counter for "interictal"
    counter = {'preictal_5_10':0, 'preictal_10_15':0, 'preictal_15_20':0, 'preictal_20_25':0, 'preictal_25_30':0, 'preictal_30_35':0, 
               'preictal_35_40':0, 'preictal_40_45':0, 'preictal_45_50':0, 'preictal_50_55':0, 'preictal_55_60':0}
    
    # ---------------- FIRSTLY EPOCHING PREICTAL DATA ---------------

    start_time = preictal_info[:,2]     #in seconds as provided in patient summary file
    seizure_filename = preictal_info[:,1]
    seizure_prev_filename = preictal_info[:,0]
    prediction_start_time = preictal_info[:,3] #it is in minutes
    #prediction_start_time = np.multiply(prediction_start_time,60) #coverting into seconds
    #taking duration before end to be 5 minutes
    iters = (1/(1-overlap))-1
    
    #extracting, epoching and saving .csv files for each seizure_filename         
    for file_sno in range(len(seizure_filename)):
        
        #checking whether to load previous seizure file 
        if  (start_time[file_sno]-(prediction_start_time[file_sno]*60) ) >= 0:
            # Reading data from .edf file
            f = pyedflib.EdfReader(src +'//' + seizure_filename[file_sno])
            n = f.signals_in_file
            seizure_file = np.zeros((n, f.getNSamples()[0]))
        
            for i in np.arange(n):
                seizure_file[i, :] = f.readSignal(i)  
                
            print('Reading done!!!')
            
                              
        else:
            print('-----Seizure data is in both files----')
            print('Reading data from .edf files')
            # Reading data from .edf files
            f = pyedflib.EdfReader(src +'//' + seizure_filename[file_sno])
            n = f.signals_in_file
            seizure_file = np.zeros((n, f.getNSamples()[0]))
            
            for i in np.arange(n):
                seizure_file[i, :] = f.readSignal(i)  
                
            if (seizure_prev_filename[file_sno] != '-') and ((start_time[file_sno] - prediction_start_time[file_sno]*60) < 0): 
                f = pyedflib.EdfReader(src +'//' + seizure_prev_filename[file_sno])
                n = f.signals_in_file
                seizure_prev_file = np.zeros((n, f.getNSamples()[0]))
                
                for i in np.arange(n):
                    seizure_prev_file[i, :] = f.readSignal(i)
                    
            else:
                seizure_prev_file = None
                    
            print('Reading done!!!')
        
        
        count = 0
        while(prediction_start_time[file_sno]>5):
            
            #conditions to select seizure
            if seizure_prev_file != None:
                #extracting component from seizure_prev part
                start_index = seizure_prev_file.shape[1]- (prediction_start_time[file_sno]*60 - start_time[file_sno])*sampling_freq
                #part1 = seizure_prev_file[:,start_index:]
                end_index = (start_time[file_sno] - prediction_start_time[file_sno]*60 + (prediction_start_time[file_sno]-5)*60)*sampling_freq
                #part2 = seizure_file[:,0:end_index]
                
                #concatening both
                seizure = np.concatenate((seizure_prev_file[:,start_index:],seizure_file[:,0:end_index]), axis = 1)
                print('Seizure data extracted!!!!')
                
            else:
                if ((start_time[file_sno] - prediction_start_time[file_sno]*60) > 0):
                    start_index = seizure_file.shape[1] - (prediction_start_time[file_sno]*60 - start_time[file_sno])*sampling_freq
                    seizure = seizure_file[:,start_index: (start_time[file_sno]-300)*sampling_freq]
                    
                else:
                    if (start_time[file_sno]<600):
                        break
                    else:
                        start_index = (start_time[file_sno] % 300)*sampling_freq
                        seizure = seizure_file[:,start_index:(start_time[file_sno]-300)*sampling_freq]
                        prediction_start_time[file_sno] = int(seizure.shape[1]/sampling_freq)
                        
            seizure_block = seizure[:,count*300:(count+1)*300]
            
            #epoching that seizure block
            # fname_std : it gives the address of the required folder ex preictal_55_60
            fname_std =  'preictal_'+str(prediction_start_time[file_sno]-5)+'_'+str(prediction_start_time[file_sno])
            for i in range(0,int(iters)):                
                split_array = splitArray(seizure_block,sampling_freq*time_window)        
                for sample in split_array:
                    fname = des + '//' + fname_std + '//' + fname_std + '_' + str(counter[fname_std]) + '.csv'
                    np.savetxt(fname, sample, delimiter = ',', fmt = '%f')
                    if counter[fname_std]%10 ==0:
                        print('-------',fname_std,'_',counter[fname_std],'.csv','--------- FORMED!!!\n')
                    
                    counter[fname_std] += 1
                
                seizure_block = seizure_block[:,int(sampling_freq*time_window*(1-overlap)):]
                
                
            
            prediction_start_time[file_sno] -= 5
            count += 1
            
            
            
            
        f._close()
    
    # ---------------- NOW EPOCHING INTERICTAL DATA ---------------
    #Let's first calculate number of interictal epochs required
    num_of_epochs = 0
    for key, value in counter.items():
        num_of_epochs += value
    
    interictal_counter = 0
    fname_std = des + '//interictal//interictal_'
    exit_check = False
    for file_sno in range(len(interictal_info)):
        # Reading data from .edf file
        f = pyedflib.EdfReader(src +'//' + seizure_filename[file_sno])
        n = f.signals_in_file
        seizure_file = np.zeros((n, f.getNSamples()[0]))
    
        for i in np.arange(n):
            seizure_file[i, :] = f.readSignal(i)  
            
        print('Reading done!!!')
        
        num_of_splits = (seizure_file.shape[1])/(256*time_window)
        
        
        if round(num_of_splits) != num_of_splits:
            trim_value = time_window*256*(num_of_splits - floor(num_of_splits))
            seizure_file = seizure_file[:,0:seizure_file.shape[1] - trim_value]
            num_of_splits = floor(num_of_splits)
    
        split_array = np.hsplit(seizure_file,num_of_splits)
        
            
        for sample in split_array:
            fname = fname_std + str(interictal_counter) + '.csv'
            np.savetxt(fname, sample, delimiter = ',', fmt = '%f')
            if interictal_counter%60 ==0:
                print('-------interictal_',interictal_counter,'.csv','--------- FORMED!!!\n')
                    
            interictal_counter += 1
            if interictal_counter >= num_of_epochs:
                exit_check = True
                break
            
        f._close()
        if exit_check == True:
            break



def WPD(signal, level):
    wpd = WaveletPacket(data = signal, wavelet='db4', mode = 'symmetric', maxlevel = level)
    node_names = [node.path for node in wpd.get_level(level)]
    
    wpd_coeffs = []
    for names in node_names:
        wpd_coeffs.append(wpd[names].data)
        
    return wpd_coeffs
    
    
    

def extract_features(coefs):
    
    feature_set = []
    
    for sub_band in coefs:
        sub_band_mean = mean([abs(ele) for ele in sub_band]) 
        avg_power = mean(sub_band**2)
        sd = std(sub_band)
        skewness = skew(sub_band)
        kurt = kurtosis(sub_band)
        feature_set.extend((sub_band_mean,avg_power,sd,skewness,kurt))
    
    
    return feature_set
    



def createDataset(addr, folder_name, num_of_channels = 23, sampling_freq = 256, channels = None):
    '''
    addr: Address of the SPADE folder ex. ..../SPADE
    folder_name: The name of the folder which contains the epochs of which you want to create dataset of 
    '''
    
    addr = addr + '//processed_data'
    folder_addr = addr + '//' + folder_name
    if channels == None:
        channels = range(num_of_channels)
    files = []

    for file_name in os.listdir(folder_addr):
        if file_name.endswith(".csv"):
            files.append(folder_addr + '//' + file_name)
    
    ictal = np.zeros((len(files),num_of_channels,sampling_freq))
    count = 0
    for f in files:
        ictal[count,:,:] = (pd.read_csv(f,header=None)).as_matrix()
        if(count%50==0):    
            print(count)
        count = count + 1
    
    # Performing wavelet Packet decomposition
    level = 4
    number_of_features_extracted = 5
    size_of_feature_vector = len(channels)*(2**level)*number_of_features_extracted
    
    #intializing dataset
    ictal_feature_data = np.zeros((len(ictal_files), size_of_feature_vector))
    
    #creating dataset
    i=0
    for epoch in ictal:
        feature_vector = []
        for ch in channels:
            coefficients = WPD(epoch[ch,:],level)
            features = extract_features( coefficients )
            feature_vector.extend(features)
            
        ictal_feature_data[i,:] = feature_vector
        i = i+1
        if (i%20==0):
            print(i)
    
    np.savetxt(addr + '//' + (folder_name + '_featureData.csv'),
               ictal_feature_data,fmt = '%f', delimiter = ',')
           
          
          

def trainModel(addr,preictal_filename):
    '''
    addr: addr: Address of the SPADE folder ex. ..../SPADE
    
    '''
    
    addr = addr + '//processed_data'
    
    preictal_feature_data = (pd.read_csv(addr + '//' + preictal_filename,
                                    header = None)).as_matrix()

    interictal_feature_data = (pd.read_csv(addr + '//interictal_featureData.csv',
                                        header = None)).as_matrix()
                                    
    random_values = set()
    while True:
      random_values.add(random.randint(0,interictal_feature_data.shape[0]-1))
      if len(random_values) >= preictal_feature_data.shape[0]:
        break
        
    interictal_feature_data = interictal_feature_data[np.array(list(random_values)),:]
    print '\nInterictal dataset also created....'
    zero_col = np.zeros(interictal_feature_data.shape[0])
    one_col = np.ones(preictal_feature_data.shape[0])
    

    Y = np.concatenate((zero_col,one_col), axis = 0)
    X = np.concatenate((interictal_feature_data,preictal_feature_data),axis = 0) #concatenate along rows
    

    #Fitting the SVM to the Training set
    classifier = SVC(kernel = 'linear',random_state=0)
    
    print 'RFE has started'
    selector = RFE(classifier, 110)
    X = selector.fit_transform(X,Y)
    
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
    print 'f1_Score = ',f1Score, '\n\n\n\n'
    
    #Noting down selected features

    selected_features = selector.get_support()
    features = []
    for i in range(1,len(preictal_feature_data[0]) + 1):
        if selected_features[i-1]==True:
            features.append(i)
    print 'features_selected = ',features
    
    associated_information = " Accuracy = " + str(acc*100) + '\nSensitivity = ' + str(sensitivity*100) + '\nSpecificity = ' + \
    str(specificity*100) + '\nPrecision = ' + str(precision) + '\nAuc_Roc_Score = ' + str(roc_score) + \
    '\f1_score = ' + str(f1Score)
    
    #saving model
    joblib.dump(classifier, addr +'//'+ interictal_filename[:(interictal_filename.find('f'))] + 'trainedModel.pkl' )
    #saving results
    np.savetxt( addr +'//'+ interictal_filename[:(interictal_filename.find('f'))] + 'results.txt', associated_information )
    #saving selected features
    np.savetxt( addr +'//'+ interictal_filename[:(interictal_filename.find('f'))] + 'selectedFeatures.csv', features, delimiter = ',')
    
    
