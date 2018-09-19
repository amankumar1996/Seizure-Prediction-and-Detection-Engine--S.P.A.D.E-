# -*- coding: utf-8 -*-
"""
Created on Wed August 1 12:04:16 2018

@author: amankumar
"""

from shutil import copytree
from os import getcwd
#path where files has to be copied
#dest_path = 'E:\Desktop'

def copy_info_files(dest_path):
    dest_path = dest_path + '/' + 'info_files'
    src_path =  getcwd()  + '/' + 'info_files'
    copytree(src_path, dest_path)