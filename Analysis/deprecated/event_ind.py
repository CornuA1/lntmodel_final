"""
Return list of indices at which a given behavioural event (e.g. reward
dispensed) occured.

Parameters
-------
raw_behav : ndarray    
            raw behaviour dataset

eventprops : tuple
            provide information on which type of event should be detected and 
            required paramaters. Since different events types may need 
            different parameters, this function argument is intentionally 
            flexible
            
Outputs
-------
event_indices : ndarray
            indices of raw_behav, and trial numbers at indices for the given event in each trial
	
"""


import numpy as np
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

def movement_onset( raw_behav, speed_threshold, gap_tolerance ):
    """ 
    return indices of movement onset. Use lowpass-filtered signal
    to avoid single spikes in runnign speed triggering an event

    speed_threshold (cm/sec): running speed that has to be exceeded to 
                     be detected as movement
                     
    gap_tolerance (sec): dips in speed below speed_threshold for this period are 
                   ignored
    
    """
    
    # filter requirements.
    order = 6
    fs = int(np.size(raw_behav,0)/raw_behav[-1,0])       # sample rate, Hz
    cutoff = 4 # desired cutoff frequency of the filter, Hz    
    
    speed_filtered = butter_lowpass_filter(raw_behav[:,3], cutoff, fs, order)    
    
    # get indeces above speed threshold
    speed_high_idx = np.where(speed_filtered > speed_threshold)[0]
    
    # use diff to find gaps between episodes of high speed
    idx_diff = np.diff(speed_high_idx)
    idx_diff = np.insert(idx_diff,0,0)
    
    # convert gap tolerance from cm to number of frames
    gap_tolerance_frames = int(gap_tolerance/raw_behav[0,2])
    
    # find indeces where speed exceeds threshold 
    onset_idx = speed_high_idx[np.where(idx_diff > gap_tolerance_frames)[0]]
    
    return onset_idx

def trial_transition( raw_behav ):
    """ return indices of trial transitions """
    # find indeces of trial transitions, insert 0 to get first trial as well
    return np.where(np.insert(np.diff(raw_behav[:,6]),0,0) > 0)[0]
   
def event_rewards( raw_behav, track, filterprops ):
    """ return indices when the animal received a reward. Disregard default rewards. """
    # calculate the first order difference in reward-column
    rew_diff = np.diff(raw_behav[:,5])
    # insert 0 at beginning to re-align diff vector with original vector,
    # as that has n-1 number elements
    rew_diff = np.insert(rew_diff,0,0)
    # find indices where valve had just opened (since the column is set to 1 
    # whenever the valve has been opened to dispense a reward in response to 
    # licking)
    e_ind = np.where(rew_diff == 1)[0]
    # filter out reward events that happened not on the desired track
    if track != -1:
        for i,ind in enumerate(e_ind):
            if raw_behav[ind,4] != track:
                e_ind[i] = -1
        e_ind = e_ind[np.where(e_ind != -1)]
    
    return e_ind


def event_ind( raw_behav, track, filterprops=[-1] ):
    """ call event filter based on filterprops parameters and return results """
    
    # call desired event filter function
    if filterprops[0] == 'reward':
        event_indices = event_rewards(raw_behav, track, filterprops)
    elif filterprops[0] == 'trial_transition':
        event_indices = trial_transition(raw_behav)
    elif filterprops[0] == 'movement_onset':
        event_indices = movement_onset(raw_behav, filterprops[1], filterprops[2])
    else:
        raise ValueError('Event type not recognised.')
    return np.transpose(np.vstack((event_indices,raw_behav[event_indices,6])))