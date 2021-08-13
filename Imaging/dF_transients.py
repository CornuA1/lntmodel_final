# -*- coding: utf-8 -*-
"""
Return indices at which a transient (significant deflection
from dF/F baseline) has been detected.

Parameters
-------
df_data : ndarray    
          dF data. One column per ROI. 

detect_params : list
        flexible list of input parameters specifying parameters based
        on which transients should be detected

Outputs
-------
transient_indices : list of ndarray
            list of tuples, each containing the onset and offset index of a
            transient
          
Created on Sat Aug  6 09:56:49 2016

@author: lukasfischer
"""

import numpy as np

def calc_baseline_percentile( dF_roi_data, percentile ):
    """ calculate the baseline dF level based on a given percentile """  
    # sort samples in ascending order of their value
    samples_sorted = np.sort(dF_roi_data)
    num_samples = int(samples_sorted.shape[0] * (percentile/100))
    
    # calculate the mean and stdev of the given percentile of samples
    baseline_dF = np.mean(samples_sorted[0:num_samples])    
    baseline_stdev = np.std(samples_sorted[0:num_samples])
    
    return baseline_dF, baseline_stdev

def stdev_threshold( dF_data, stdev_thresh, minlength ):
    """ 
    return indices at which a transient has exceeded a certain standard deviation of the overall signal 

    stdev_thresh : int
            number standard deviations dF needs to exceed to be detected as above threshold
            
    minlength : int
            min number of samples a transient has to be above threshold to
            be returned. This helps to weed out noise spikes.

    """
    
    # if only a single column is passed, shape it into 2d array so that indexing
    # doesn't trip up
    if dF_data.ndim < 2:
        dF_data = np.atleast_2d(dF_data).T
    # create 2d list where each column holds the transients for each ROI
    transient_list = [[] for i in range(np.size(dF_data,1))]
    roi_stdev = np.zeros((np.size(dF_data,1),)) 
    # iterate through every ROI and find indeces that exceed criterion
    for i,r in enumerate(dF_data.T):
        baseline_dF, baseline_stdev = calc_baseline_percentile(r,20)
        roi_stdev = baseline_dF + (baseline_stdev*stdev_thresh)
        exceed_idx = np.where(r > roi_stdev)[0]
        # find consecutive indeces to identify them as contiguous transients
        exceed_diff = np.diff(exceed_idx)   
        exceed_diff = np.insert(exceed_diff,0,0)
        transient_onsets = np.where(exceed_diff > 1)[0]
        # for each transient, find offset point
        for j,t in enumerate(exceed_idx[transient_onsets]):  
            try:
                transient_offset = np.where(r[t:] < roi_stdev)[0][0]
            except IndexError:
                # if there is no offset because the transient reaches the 
                # end of the recording: set to 0
                transient_offset = 0
            # check if transient satisfies minlength criterion
            if ((t+transient_offset) - exceed_idx[transient_onsets[j]]) > minlength:
                transient_list[i].append([exceed_idx[transient_onsets[j]],t+transient_offset])
        #print('done with roi ', str(i) ,' . threshold: ', str(roi_stdev),' Found ', str(len(transient_list[i])), ' transients')    
    return transient_list
            

def dF_transients( dF_data, detect_params=[-1] ):
    if detect_params[0] == 'stdev_threshold':
        transient_indices = stdev_threshold(dF_data, detect_params[1], detect_params[2])
    return transient_indices
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    