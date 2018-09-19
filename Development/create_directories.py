# -*- coding: utf-8 -*-
"""
Created on Wed August 1 10:40:16 2018

@author: amankumar
"""

from os import mkdir

#path = 'F:\BCI project'

def create_dir(path):
    folders_to_create = ['preictal_5_10','preictal_10_15','preictal_15_20','preictal_20_25','preictal_25_30','preictal_30_35','preictal_35_40',
                         'preictal_40_45','preictal_45_50','preictal_50_55','preictal_55_60', 'interictal']

    main_dir = 'SPADE'
    dataset_dir = 'processed_data'

    #creating main directory
    main_path = path + '/' + main_dir
    mkdir(main_path)

    #creating directory for dataset
    dataset_path = main_path + '/' + dataset_dir
    mkdir(dataset_path)

    for folder in folders_to_create:
        folder_path = dataset_path + '/' + folder
        mkdir(folder_path)


    mkdir(main_path + '/' + 'info_files')