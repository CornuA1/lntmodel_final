# -*- coding: utf-8 -*-
"""
Calculate dF/F for a given ROI. Baseline F is calculated as the average
flourescence around a given sample defined by t_win.

Parameters
-------
raw_dF : ndarray
    Input flourescence signal. Can be single column or multi-column

t_win : int tuple
    define the boundaries of the 
    
fs : double
    sampling rate at wich imaging data was acquired
    
Outputs
-------
raw_dF : ndarray
    dF/F values for input. Same shape as raw_dF input.

"""

import numpy as np

def calc_dF( raw_dF, t_win, fs ):
    # make local copy of dataset
    temp_dF = np.copy(raw_dF)
    baseline_F_all = np.zeros((raw_dF.shape))
    # calculate the number of samples surrounding each timepoint to calc
    # baseline F
    t_start = int(t_win[0]*fs)
    t_end = int(t_win[1]*fs)   
    
    for i,c in enumerate(temp_dF.T):
        dF_c = np.zeros((np.size(c)))
        for j,r in enumerate(c):
            # get indeces for start and end of window
            cur_win_start = j-t_start
            cur_win_end = j+t_end
            # make sure window does not overshoot the dataset on either side
            if j-t_start < 0:
                cur_win_start = 0
            if j+t_end > np.size(c)-1:
                cur_win_end = np.size(c)-1                
            # calc the baseline
            baseline_F = np.mean(c[cur_win_start:cur_win_end])
            dF_c[j] = (r-baseline_F)/baseline_F
            baseline_F_all[j,i] = baseline_F
        
        dF_c[dF_c<0]=0
        temp_dF[:,i] = dF_c
#        temp_dF[:,i] = (temp_dF[:,i]-baseline_F)/baseline_F
    return temp_dF,baseline_F_all