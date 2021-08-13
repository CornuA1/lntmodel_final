# -*- coding: utf-8 -*-
"""
Return indeces of transient onsets in imaging data. Different detection 
algorithms are (or will be) available.

Created on Sun Jun  5 17:22:56 2016

@author: lukasfischer
"""

import numpy as np

def thresholded( im_trace, props ):
    """ 
    return indices of transient onset. 

    props[1] - threshold (dF/F): value that has to be exceeded to 
                     be detected as movement
                     
    props[2] gap_tolerance (frames): dips in value below threshold for this 
                    period are ignored
    
    """    
    
    
    transient_threshold = props[1]
    gap_tolerance = props[2]
    # get indeces above speed threshold
    im_high = np.where(im_trace > transient_threshold)[0]

    # use diff to find gaps between episodes of high speed
    idx_diff = np.diff(im_high)
    idx_diff = np.insert(idx_diff,0,0)
    
    # find indeces where speed exceeds threshold 
    onset_idx = im_high[np.where(idx_diff > gap_tolerance)[0]]
    
    return onset_idx

def transient_ind( im_trace, props=[-1] ):
    """ call event filter based on filterprops parameters and return results """
    
    # call desired event filter function
    if props[0] == 'threshold':
        event_indices = thresholded(im_trace, props)
    else:
        raise ValueError('Event type not recognised.')
        
    return event_indices