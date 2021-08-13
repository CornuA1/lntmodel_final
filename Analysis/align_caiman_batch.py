# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 07:20:50 2020

@author: Keith
"""
import sys, yaml, os
#os.chdir('C:/Users/Keith/Documents/GitHub/LNT')
## with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
##     loc_info = yaml.load(f)
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT'+ os.sep + "Analysis")
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT'+ os.sep + "Imaging")

os.chdir(r'C:\Users\Lou\Documents\repos\LNT')
sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Analysis")
sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Imaging")
sys.path.append(r'C:\Users\Lou\Documents\repos\OASIS-master')

import warnings; warnings.simplefilter('ignore')

from align_caiman import factor_out_baseline, calculate_detrend_and_deconvolve, process_and_align_caiman
    
    
def run_LF191022_1():
    MOUSE= 'LF191022_1'
    sessions = [1114,1115,1121,1125,1204,1207,1209,1211,1213,1215,1217]
    dff_files = [2,4,3,3,2,0,0,0,4,5,0]
    dff_files_ol = [3,6,4,4,3,1,1,1,5,6,1]
    base_path = 'D:/Lukas/data/animals_raw'
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i])
        if dff_files[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
        
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i]) + '_ol'
        if dff_files_ol[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files_ol[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files_ol[i])+'_results.mat'
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        process_and_align_caiman(data_path, sess, dff_file)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)


def run_LF191022_2():
    MOUSE= 'LF191022_2'
    sessions = [1114,1116,1121,1204,1206,1208,1210,1212,1216]
    dff_files = [4,0,5,4,4,4,8,4,0,0]
    dff_files_ol = [5,1,6,7,5,5,9,5,1]
    base_path = 'D:/Lukas/data/animals_raw'
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i])
        if dff_files[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
        
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i]) + '_ol'
        if dff_files_ol[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files_ol[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files_ol[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file, names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
    
    
def run_LF191022_3():
    MOUSE= 'LF191022_3'
    sessions = [1113,1114,1119,1121,1125,1204,1207,1210,1211,1215,1217]
    dff_files = [0,0,0,13,7,8,2,0,2,1,2]
    dff_files_ol = [1,1,1,14,8,9,3,1,3,2,3]
    base_path = 'D:/Lukas/data/animals_raw'
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i])
        if dff_files[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
        
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i]) + '_ol'
        if dff_files_ol[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files_ol[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files_ol[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
    
    
def run_LF191023_blank():
    MOUSE= 'LF191023_blank'
    sessions = [1114,1116,1121,1206,1208,1210,1212,1213,1216,1217]
    dff_files = [6,2,8,0,0,4,0,2,3,4]
    dff_files_ol = [7,3,9,1,1,5,1,3,4,5]
    base_path = 'D:/Lukas/data/animals_raw'
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i])
        if dff_files[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
        
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i]) + '_ol'
        if dff_files_ol[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files_ol[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files_ol[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)


def run_LF191023_blue():
    MOUSE= 'LF191023_blue'
    sessions = [1113,1114,1119,1121,1125,1204,1206,1208,1210,1212,1215,1217]
    dff_files = [3,11,4,10,11,0,2,2,6,2,3,6]
    dff_files_ol = [4,12,5,11,12,1,3,3,7,3,4,7]
    base_path = 'D:/Lukas/data/animals_raw'
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i])
        if dff_files[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
        
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i]) + '_ol'
        if dff_files_ol[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files_ol[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files_ol[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)


def run_LF191024_1():
    MOUSE= 'LF191024_1'
    sessions = [1114,1115,1121,1204,1207,1210]
    dff_files = [0,1,0,11,4,2]
    dff_files_ol = [1,2,1,12,5,3]
    base_path = 'D:/Lukas/data/animals_raw'
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i])
        if dff_files[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)
        
    for i in range(len(sessions)):
        sess = '2019' + str(sessions[i]) + '_ol'
        if dff_files_ol[i] < 10:
            dff_file = 'M01_000_00'+str(dff_files_ol[i])+'_results.mat'
        else:
            dff_file = 'M01_000_0'+str(dff_files_ol[i])+'_results.mat'
        data_path = base_path + os.sep + MOUSE + os.sep + sess
        names = None
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if '.csv' in name:
                   names = name
           for name in dirs:
               if '.csv' in name:
                   names = name
        process_and_align_caiman(data_path, sess, dff_file,names)
        factor_out_baseline(data_path, sess, dff_file)
        calculate_detrend_and_deconvolve(data_path, sess, dff_file)

    
if __name__ == '__main__':
    
    run_LF191022_2()
#    run_LF191022_3()
#    run_LF191023_blank()
#    run_LF191023_blue()
#    run_LF191024_1()
    
