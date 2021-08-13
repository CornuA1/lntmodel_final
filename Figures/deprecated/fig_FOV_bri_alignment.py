"""

Plot FOV brightness vs imaging and behavioral data to determine the impact of screen brightness in a given session

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
from MTH3_recalc_df import recalc_df

d = 'Day201852'
m = 'LF171211_2'

h5dat = h5py.File('/Users/lukasfischer/Google Drive/MTH3_data/animals_h5/' + m + '/' + m + '.h5', 'r')

# load datasets and close HDF5 file again
#b_ds = np.copy(h5dat[str(d) + '/' + m + '/raw_data'])
if True:
    b_aligned = np.copy(h5dat[d + '/' + '/behaviour_aligned'])
    b_raw = np.copy(h5dat[d + '/' + '/raw_data'])
    dF = np.copy(h5dat[str(d) + '/' + '/dF_win'])
    fovbri_raw = np.copy(h5dat[str(d) + '/' + '/FOV_bri'])
    fovbri_aligned = np.copy(h5dat[str(d) + '/' + '/FOV_bri_aligned'])
    h5dat.close()

    # calculate mean acquisition framerate for behavior and imaging
    fs_behav = 1/np.mean(b_raw[:,2])
    fs_f = fovbri_raw.shape[0]/b_raw[-1,0]

    # factor by which to downsample from aligned frames to original imaging frames
    b_to_f_fs_factor = fs_behav/fs_f


    # crop behavior and/or imaging data to align
    crop_behav_frames_start = 0
    crop_behav_frames_end = 0

    crop_f_frames_start = 6
    crop_f_frames_end = 0

    b_seq_al = b_aligned[:,4]
    b_seq_al = b_seq_al[crop_behav_frames_start:b_seq_al.shape[0]-crop_behav_frames_end]
    f_trace = dF_aligned = signal.resample(fovbri_aligned.T[crop_f_frames_start:fovbri_aligned.T.shape[0]-crop_f_frames_end], fovbri_aligned.T.shape[0], axis=0)

    # plot data to check alignment
    plot_start = 0
    plot_end = 1000
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(111)
    with sns.axes_style("dark"):
        ax2 = plt.twinx()
    ax1.plot(f_trace,lw=1)
    ax2.plot(b_seq_al,lw=1,c='r')
    ax2.set_xlim([plot_start,plot_end])

    crop_f_frames_start_orig = int(np.round(crop_f_frames_start/b_to_f_fs_factor,0))
    crop_f_frames_end_orig = int(np.round(crop_f_frames_end/b_to_f_fs_factor,0))
    print(crop_f_frames_start_orig, crop_f_frames_end_orig)

# set whether to actually re-align datasets and write to HDF5-file
if False:
    # downsample from number of frames cropped in the aligned dataset to the original dataset

    recalc_df(m, d, [0,1], [crop_f_frames_start_orig,crop_f_frames_end_orig])
