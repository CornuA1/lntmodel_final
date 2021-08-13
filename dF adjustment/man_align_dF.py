"""
manually align imaging and behavior data

its currently a little clumsy to use. First you check the aligned brightness
signal against the transitions to the black box by setting dF_start and dF_end. Its important to check the beginning and the end
of each recording to make sure there is no drift within the recording.

Once you are happy with the alignment, set the bottom block to True - it will
run the resampling for all ROIs and write it back to the HDF5-file.

"""

# %load_ext autoreload
# %autoreload
# %matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import h5py
import sys
import os
import yaml
from scipy.interpolate import griddata
import multiprocessing


# append necessary paths (for windows and unix-based machines)
sys.path.append("./Analysis")
with open('./loc_settings.yaml', 'r') as f:
            content = yaml.load(f)

from filter_trials import filter_trials
from scipy import stats
from scipy import signal

def resample_roi(inp):
    #print('processing ROI: ' + str(inp[2]))
    return signal.resample(inp[0], inp[1], axis=0)

def write_h5(h5dat, day, dset, dset_name):
    """ Write dataset to HDF-5 file. Overwrite if it already exists. """
    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset(str(d) + '/' + dset_name,
                             data=dset, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat[str(d) + '/' + dset_name]
        h5dat.create_dataset(str(d) + '/' + dset_name,
                             data=dset, compression='gzip')

d = 'Day20170804'
m = 'LF170613_1'

h5path = content['imaging_dir'] + m + '/' + m + '.h5'
h5dat = h5py.File(h5path, 'r')
print(h5path)

# load datasets and close HDF5 file again
#b_ds = np.copy(h5dat[str(d) + '/' + m + '/raw_data'])
b_aligned = np.copy(h5dat[d + '/' + '/behaviour_aligned'])
b_raw = np.copy(h5dat[d + '/' + '/raw_data'])
dF = np.copy(h5dat[str(d) + '/' + '/dF_win'])
fovbri_raw = np.copy(h5dat[str(d) + '/' + '/FOV_bri'])
fovbri_aligned = np.copy(h5dat[str(d) + '/' + '/FOV_bri_aligned'])
h5dat.close()

num_ts_behaviour = b_aligned.shape[0]

# get rid of extra dimension
fovbri_raw = np.squeeze(fovbri_raw)
fovbri_aligned = np.squeeze(fovbri_aligned)

# initial frames to calibrate alignment of grating presentation and imaging
dF_start = 0
dF_end = 0

skip_frames_behavior_start = 0
skip_frames_behavior_end = 0

skip_frame_imaging_start = dF_start
skip_frame_imaging_end = fovbri_aligned.shape[0] - dF_end

fobri_aligned_al = signal.resample(fovbri_aligned[skip_frame_imaging_start:skip_frame_imaging_end], num_ts_behaviour, axis=0)

b_seq_al = b_aligned[:,4]
b_seq_al = b_seq_al[skip_frames_behavior_start:b_aligned.shape[0]-skip_frames_behavior_end]


view_idx = [0,1200]
view_idx2 = [0,39241]

fig = plt.figure(figsize=(16,7))
ax2 = fig.add_subplot(211)
ax3 = fig.add_subplot(212)
ax2.plot(fobri_aligned_al[view_idx[0]:view_idx[1]],lw=1)
with sns.axes_style("dark"):
    ax2_2 = ax2.twinx()
ax2_2.plot(b_seq_al[view_idx[0]:view_idx[1],],lw=1,c='r')
ax2_2.set_ylim([3,5])

ax3.plot(fobri_aligned_al[view_idx2[0]:view_idx2[1]],lw=1)
with sns.axes_style("dark"):
    ax3_2 = ax3.twinx()
ax3_2.plot(b_seq_al[view_idx2[0]:view_idx2[1],],lw=1,c='r')
ax3_2.set_ylim([3,5])

ax2.set_ylim([0,10000])

ax3.set_xlim([28000,29241])

if False:
    print('Cropping imaging data')
    print('Cropping and resampling ROI data from length ', str(dF.shape[0]), ' to ', str(dF.shape[0]-dF_start-dF_end))
    dF = dF[dF_start:,:]
    dF = dF[:dF.shape[0]-dF_end,:]
    # resample dF/F signal
    print('Resampling imaging data...')
    p = multiprocessing.Pool()
    #dF_aligned = signal.resample(dF_aligned, num_ts_behaviour, axis=0)
    # create tuples of (column of dF_aligned, num_ts_behaviour)
    dF_ts_tuples = []
    for j,col in enumerate(dF.T):
        dF_ts_tuples.append((col,num_ts_behaviour,j))
    resampled_roi = p.map(resample_roi, dF_ts_tuples)

    dF_aligned_resampled = np.zeros((num_ts_behaviour,np.size(dF,1)))
    for i,col in enumerate(resampled_roi):
        dF_aligned_resampled[:,i] = col
    print('done resampling...')
    print('writing results to file...')
    h5dat = h5py.File(h5path, 'r+')
    write_h5(h5dat, d, dF_aligned_resampled, '/dF_win')
    write_h5(h5dat, d, fobri_aligned_al, '/FOV_bri_aligned')
    h5dat.flush()
    h5dat.close()
    print('done')
