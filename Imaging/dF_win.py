# -*- coding: utf-8 -*-
"""
Calculate dF/F(0) using a moving window to calculate f0.

Created on Tue Sep 09 11:52:04 2016

@author: Lukas Fischer

"""

# append necessary paths (for windows and unix-based machines)
import sys
sys.path.append("../Analysis")
sys.path.append("../../General/Imaging")

import numpy as np
from filter_dF import filter_dF

from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
    
def dF_win( raw_imaging, fs=60 ):  
    # filter requirements.
    order = 6
    cutoff = 2 # desired cutoff frequency of the filter, Hz   
    
    # size of moving window in seconds
    win = 60
    
    # calculate the number of samples in the window (dependent on fs)
    win_samples = win * fs
    
    dF_mat = np.zeros((raw_imaging.shape))    
    
    # loop through every ROI
    for i in range(raw_imaging.shape[1]):    
        print(i)
        # filter signal
        raw_filtered = raw_imaging[:,i] #butter_lowpass_filter(raw_imaging[:,i], cutoff, fs, order)  
        # calculate dF using sliding window
        for j,ft in enumerate(raw_filtered):
            # calculate f0, if-clauses are for boundary conditions at beginning and end of trace
            if j-int(win_samples/2) < 0:
                f0 = np.percentile(raw_filtered[0:j+int(win_samples/2)],30) 
            elif j+int(win_samples/2) > raw_filtered.size:
                f0 = np.percentile(raw_filtered[j-int(win_samples/2):raw_filtered.size-1],30)
            else:
                f0 = np.percentile(raw_filtered[j-int(win_samples/2):j+int(win_samples/2)],30)
            
            # calculate difference between dF and f0(t)
            dF_F = raw_filtered[j] - f0
            dF_mat[j,i] = dF_F/np.absolute(f0)
        
    return dF_mat, f0
