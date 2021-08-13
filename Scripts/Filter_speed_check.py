"""
Test new filter to select trials where animal has been running

"""

%load_ext autoreload
%autoreload
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import h5py
import sys
import os
# append necessary paths (for windows and unix-based machines)
sys.path.append("./Analysis")

from filter_trials import filter_trials
from scipy import stats
from scipy import signal

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

d = 'Day201819_dark_1'
m = 'LF171212_1'

h5dat = h5py.File('/Users/lukasfischer/Google Drive/MTH3_data/animals_h5/LF171212_1/LF171212_1.h5', 'r')

# load datasets and close HDF5 file again
#b_ds = np.copy(h5dat[str(d) + '/' + m + '/raw_data'])
b_aligned = np.copy(h5dat[d + '/' + '/behaviour_aligned'])
b_raw = np.copy(h5dat[d + '/' + '/raw_data'])
dF = np.copy(h5dat[str(d) + '/' + '/dF_win'])
h5dat.close()

# filter requirements.
order = 6
fs = int(np.size(b_aligned,0)/b_aligned[-1,0])       # sample rate, Hz
cutoff = 1 # desired cutoff frequency of the filter, Hz

b_aligned[:,8] = butter_lowpass_filter(b_aligned[:,8], cutoff, fs, order)
#trials_passive = filter_trials( b_aligned, [], ['animal_notrunning',1,2])
#print(trials_passive)

trial_list = np.unique(b_aligned[:,6])
trials_pass = []

filterprops = [0,1,2]

# run through every trial and test if running speed was within criteria
for i,t in enumerate(trial_list):
    cur_trial = b_aligned[b_aligned[:,6]==t,:]
    cur_trial_latency = np.mean(cur_trial[:,2])
    trial_speed = speed_filtered[b_aligned[:,6]==t]
    fig = plt.figure(figsize=(16,7))
    plt.plot(cur_trial[:,8],lw=1)
    plt.axhline(1)
    plt.ylim([-5,40])
    # calculate how many samples speed is allowed to be above threshold (calc for each trial as it might fluctuate)
    samples_thresh = filterprops[2] / cur_trial_latency
    # get indices where animal was below threshold
    thresh_idx = np.where(cur_trial[:,8] > filterprops[1])[0]
    # check where multiple indices where not below threshold
    if np.size(thresh_idx) > samples_thresh:
        plt.title('PASS')
        trials_pass.append(t)
    else:
        plt.title('FAIL')
        pass

print(trials_pass)

#fig = plt.figure(figsize=(16,7))
#plt.plot(b_aligned[20000:21000,8],lw=1)
#plt.plot(speed_filtered[20000:21000],lw=1)
#plt.axhline(1)
