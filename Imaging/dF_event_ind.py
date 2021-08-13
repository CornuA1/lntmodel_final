# -*- coding: utf-8 -*-
"""
Return list of indices of given events in dF/F traces.

Created on Sat Aug  6 09:51:49 2016

@author: Lukas Fischer
"""

def transient_peak( dF_data, filterprops ):
    pass

def dF_event_ind( dF_data, filterprops=[-1]):
    # call desired event filter function
    if filterprops[0] == 'transient_peak':
        event_indices = transient_peak(dF_data, filterprops)
    else:
        raise ValueError('Event type not recognised.')
    return event_indices
