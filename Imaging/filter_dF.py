# -*- coding: utf-8 -*-
"""
Apply filter to dF/F input data and return filtered signal.

Parameters
-------
dF_data : ndarray    
          dF/F input data (single ROI)

Outputs
-------
dF_filtered : ndarray
        filtered dF/F signal

Created on Sun Aug  7 19:09:56 2016

@author: lukasfischer
"""

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

def filter_dF( dF_data, filterprops ):
    # filter requirements.
    order = 6
    fs = filterprops[0]
    cutoff = filterprops[1] # desired cutoff frequency of the filter, Hz    
    
    return butter_lowpass_filter(dF_data, cutoff, fs, order)