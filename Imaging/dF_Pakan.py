# -*- coding: utf-8 -*-
"""
Calculate dF/F(0) as done in Pakan et.al. 2016

"
502 Pixel intensity within each ROI 502 was averaged to create a raw fluorescence
503 time series F(t). Baseline fluorescence F0 was computed for each neuron by taking the 5th
504 percentile of the smoothed F(t) (1Hz lowpass, zero-phase, 60th-order FIR filter) over each
505 trial (F0(t)), averaged across all trials. As a consequence, the same baseline F0 was used for
506 computing the changes in fluorescence in darkness and during visual stimulation. The
507 change in fluorescence relative to baseline, ΔF/F0 was computed by taking the difference
508 between F and F0(t) and dividing by F0.
"


Created on Tue Aug 30 11:52:04 2016

@author: lukasfischer

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
    
def dF_Pakan( raw_imaging ):  
    # filter requirements.
    order = 6
    fs = 60       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz   
    
    dF_mat = np.zeros((raw_imaging.shape))    
    
    for i in range(raw_imaging.shape[1]):    
    
        # filter signal
        raw_filtered = butter_lowpass_filter(raw_imaging[:,i], cutoff, fs, order)  
    
        # calculate f0
        f0 = np.percentile(raw_filtered,5)
        
        # calculate difference between dF and f0(t)
        dF_F = raw_filtered - f0
        dF_mat[:,i] = np.absolute(dF_F/f0)
        
    return dF_mat