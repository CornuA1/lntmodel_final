"""
deconvolve calcium signal and store estimated spike in HDF5 files

"""

from scipy.signal import butter, filtfilt
import seaborn as sns
import numpy as np
import warnings
import sys, os
import yaml
import h5py
from oasis.functions import deconvolve, estimate_parameters, foopsi
from oasis.oasis_methods import oasisAR1

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

def write_h5(h5dat, day, dset, dset_name):
    """ Write dataset to HDF-5 file. Overwrite if it already exists. """
    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset(str(day) + '/' + dset_name,
                             data=dset, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat[str(day) + '/' + dset_name]
        h5dat.create_dataset(str(day) + '/' + dset_name,
                             data=dset, compression='gzip')

def deconvolve_df(roidata,frame_latency,s_min=0.6, spikrate_window=0.5):
    # deconvolve applies oasisAR1 of len(g)==1 and oasisAR2 if len(g)==2
    # calling deconvolve like this will call constrained_oasisAR1() with some default paramters. Here we use it to get a baseline estimate (b)
    fudge_factor = .98
    g,sn = estimate_parameters(roidata, p=1, fudge_factor=fudge_factor)
    # deconvolve (which in turn runs constrained_AR1) just to estimate the baseline (b)
    c_g, s, b, _, lam = deconvolve(roidata,g,sn)
    # now we call oasisAR1 manually so we can set the s_min parameter
    c_AR1,s_AR1 = oasisAR1(roidata, g[0], s_min=0.1)

    # calculate spike train and apply causal sliding boxcar window
    spikes_idx = np.where(s_AR1 > 0)[0]
    spiketrain = np.zeros(len(roidata))
    spiketrain[spikes_idx] = 1
    # make sliding window (sec)
    sliding_window_size = [spikrate_window,0]
    sliding_window_time = sliding_window_size[0] + sliding_window_size[1]
    sliding_window_idx = [int(np.round(sliding_window_size[0]/frame_latency,0)),int(np.round(sliding_window_size[1]/frame_latency,0))]
    inst_spikerate = np.zeros(len(roidata))
    for i in range(len(spiketrain)):
        if i - sliding_window_idx[0] < 0:
            num_spikes = np.sum(spiketrain[0:i])
        elif i + sliding_window_idx[1] > len(spiketrain):
            num_spikes = np.sum(spiketrain[i:-1])
        else:
            num_spikes = np.sum(spiketrain[i-sliding_window_idx[0]:i+sliding_window_idx[1]])

        inst_spikerate[i] = num_spikes/sliding_window_time

    return s_AR1,spiketrain,inst_spikerate

def deconvolve_and_save(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, s_min=0.55):
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # if we want to run through all the rois, just say all
    if NUM_ROIS == 'all':
        h5dat = h5py.File(h5path, 'r')
        dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
        h5dat.close()
        NUM_ROIS = dF_ds.shape[1]
        write_to_dict = True
        print('number of rois: ' + str(NUM_ROIS))

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[SESSION + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    # dF_ds = np.copy(h5dat[sess + '/dF_original'])
    h5dat.close()
    frame_latency = 1/(dF_ds.shape[0]/(behav_ds[-1,0] - behav_ds[0,0]))

    df_deconvolved = np.zeros(dF_ds.shape)
    spiketrain = np.zeros(dF_ds.shape)
    spikerate = np.zeros(dF_ds.shape)
    print('sesssion: ' + SESSION)
    for roi in range(NUM_ROIS):
        print('roi: ' + str(roi))
        df_deconvolved[:,roi],spiketrain[:,roi],spikerate[:,roi] = deconvolve_df(dF_ds[:,roi],frame_latency,s_min)
        h5dat = h5py.File(h5path, 'r+')
        write_h5(h5dat, SESSION, df_deconvolved, '/dF_deconvolved')
        write_h5(h5dat, SESSION, spiketrain, '/spiketrain')
        write_h5(h5dat, SESSION, spikerate, '/spikerate')
        h5dat.flush()
        h5dat.close()

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[SESSION_OPENLOOP + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[SESSION_OPENLOOP + '/dF_win'])
    h5dat.close()
    frame_latency = 1/(dF_ds.shape[0]/(behav_ds[-1,0] - behav_ds[0,0]))

    df_deconvolved = np.zeros(dF_ds.shape)
    spiketrain = np.zeros(dF_ds.shape)
    spikerate = np.zeros(dF_ds.shape)
    print('sesssion: ' + SESSION_OPENLOOP)
    for roi in range(NUM_ROIS):
        print('roi: ' + str(roi))
        df_deconvolved[:,roi],spiketrain[:,roi],spikerate[:,roi] = deconvolve_df(dF_ds[:,roi],frame_latency,s_min)
        h5dat = h5py.File(h5path, 'r+')
        write_h5(h5dat, SESSION_OPENLOOP, df_deconvolved, '/dF_deconvolved')
        write_h5(h5dat, SESSION_OPENLOOP, spiketrain, '/spiketrain')
        write_h5(h5dat, SESSION_OPENLOOP, spikerate, '/spikerate')
        h5dat.flush()
        h5dat.close()


def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    NUM_ROIS = 'all'
    deconvolve_and_save(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS,s_min=0.2)

def run_LF171211_1_Day2018321_2():
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    SESSION_OPENLOOP = 'Day2018321_openloop_2'
    NUM_ROIS = 'all'
    deconvolve_and_save(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS,s_min=0.6)

if __name__ == '__main__':
    # run_LF170613_1_Day20170804()
    run_LF171211_1_Day2018321_2()
    print('done')
