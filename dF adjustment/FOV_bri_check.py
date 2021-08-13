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
import yaml

# append necessary paths (for windows and unix-based machines)
sys.path.append("./Analysis")
with open('./loc_settings.yaml', 'r') as f:
            content = yaml.load(f)

from filter_trials import filter_trials
from scipy import stats
from scipy import signal

d = 'Day201852'
m = 'LF171211_2'

h5path = content['imaging_dir'] + m + '/' + m + '.h5'
h5dat = h5py.File(h5path, 'r')
print(h5path)
#h5dat = h5py.File('/Users/lukasfischer/Google Drive/MTH3_data/animals_h5/LF171211_2/LF171211_2.h5', 'r')

# load datasets and close HDF5 file again
#b_ds = np.copy(h5dat[str(d) + '/' + m + '/raw_data'])
b_aligned = np.copy(h5dat[d + '/' + '/behaviour_aligned'])
b_raw = np.copy(h5dat[d + '/' + '/raw_data'])
dF = np.copy(h5dat[str(d) + '/' + '/dF_win'])
fovbri_raw = np.copy(h5dat[str(d) + '/' + '/FOV_bri'])
fovbri_aligned = np.copy(h5dat[str(d) + '/' + '/FOV_bri_aligned'])
h5dat.close()

# crop first few behavior frames to calibrate alignment of grating presentation and imaging
skip_frames_behavior_start = 0
skip_frames_behavior_end = b_seq_al.shape[0] - 0

skip_frame_imaging_start = 7
skip_frame_imaging_end = fovbri_aligned.shape[0] - 0
b_seq_al = b_aligned[:,4]
b_seq_al = b_seq_al[skip_frames_behavior_start:skip_frames_behavior_end]

fobri_aligned_al = fovbri_aligned[skip_frame_imaging_start:skip_frame_imaging_end]

fig = plt.figure(figsize=(16,7))
ax2 = fig.add_subplot(111)
ax2.plot(fobri_aligned_al,lw=1)
with sns.axes_style("dark"):
    ax2_2 = plt.twinx()
ax2_2.plot(b_seq_al,lw=1,c='r')
ax2_2.set_ylim([3,5])

view_idx = [38000,fovbri_aligned.shape[0]]

ax2_2.set_xlim(view_idx)
ax2.set_xlim(view_idx)
ax2.set_ylim([800,1800])
