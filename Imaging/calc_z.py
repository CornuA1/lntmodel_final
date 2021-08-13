# -*- coding: utf-8 -*-
"""
Calculate the z-score for a given ROI. Baseline F is calculated as the average
flourescence around a given sample defined by t_win.

Parameters
-------
raw_dF : ndarray
    Input flourescence signal. Can be single column or multi-column
    
Outputs
-------
roi_z : ndarray
    z-scores for input. Same shape as raw_dF input.

"""

import numpy as np

def calc_z( raw_dF ):
    # make local copy of dataset
    temp_dF = np.copy(raw_dF)
    
    for i,c in enumerate(temp_dF.T):
        mu = np.mean(c)
        std = np.std(c)
        temp_dF[:,i] = (temp_dF[:,i]-mu)/std
        
    return temp_dF