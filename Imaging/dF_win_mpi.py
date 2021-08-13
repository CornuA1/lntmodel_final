# -*- coding: utf-8 -*-
"""
Calculate dF/F(0) using a moving window to calculate f0. Run multiple workers
to increase speed. Currently, low pass filtering is NOT implemented.

Created on Tue Sep 09 11:52:04 2016

@author: Lukas Fischer

"""

# append necessary paths (for windows and unix-based machines)
import sys
import multiprocessing
sys.path.append("../Analysis")
sys.path.append("../../General/Imaging")

import numpy as np
#from filter_dF import filter_dF

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

def dF_col_worker(inp):
    # filter requirements.
    order = 6
    fs = 60       # sample rate, Hz
    cutoff = 2 # desired cutoff frequency of the filter, Hz
    dF_mat_col = np.zeros(np.size(inp))
    # size of moving window in seconds
    win = 60
    # calculate the number of samples in the window (dependent on fs)
    win_samples = win * fs
    # filter signal
    raw_filtered = butter_lowpass_filter(inp, cutoff, fs, order)
    raw_filtered = inp
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
        dF_mat_col[j] = dF_F/np.absolute(f0)

    return dF_mat_col, f0

def dF_win( raw_imaging ):


    dF_mat = np.zeros((raw_imaging.shape))
    dF_mat_f0 = np.zeros((raw_imaging.shape))

    # start pool
    p = multiprocessing.Pool()
    # process each column in individual workers
    col_collect = p.map(dF_col_worker,raw_imaging.T)

    # write result into individual columns
    for i,col in enumerate(col_collect):
        dF_mat[:,i] = col[0]
        dF_mat_f0[:,i] = col[1]

    #for i in range(raw_imaging.shape[1]):


    return dF_mat, dF_mat_f0
