"""
plot section of individual traces of individual, subcellular components

@author: Lukas Fischer

"""

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import seaborn as sns
import numpy as np
import warnings
import h5py
import sys, os
import yaml

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
import seaborn as sns
sns.set_style('white')



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_ind_trace_behavior(mouse, sess_soma, sess_pair, roi, pair_roi, t_start, t_stop):
    h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
    # print(h5path)
    # print(mouse, sess_soma)
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[sess_soma + '/dF_original'])
    dF_ds_pair = np.copy(h5dat[sess_pair + '/dF_original'])
    h5dat.close()


    # set up figure
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax2.spines['left'].set_linewidth(2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    order = 6
    fs = 15.5  # sample rate, Hz
    cutoff = 4 # desired cutoff frequency of the filter, Hz

    t_start_idx = (np.abs(np.arange(0,dF_ds.shape[0]/fs,1/fs) - t_start)).argmin()
    t_stop_idx = (np.abs(np.arange(0,dF_ds.shape[0]/fs,1/fs) - t_stop)).argmin()
    # plot dF trace
    pair_trace = np.zeros((t_stop_idx-t_start_idx, len(pair_roi)))
    filter_trace = 0
    if filter_trace == 1:
        soma_trace = butter_lowpass_filter(dF_ds[t_start_idx:t_stop_idx,roi], cutoff, fs, order)
        for i,pr in enumerate(pair_roi):
            pair_trace[:,i] = butter_lowpass_filter(dF_ds_pair[t_start_idx:t_stop_idx,pr], cutoff, fs, order)
    else:
        soma_trace = dF_ds[t_start_idx:t_stop_idx,roi]
        for i,pr in enumerate(pair_roi):
            pair_trace[:,i] = dF_ds_pair[t_start_idx:t_stop_idx,pr]


    ax1.plot(pair_trace,lw=0.5, c ='#C1272D')
    ax1.plot(soma_trace,lw=1, c='k')

    ax2.plot(pair_trace,lw=0.2)
    ax2.plot(soma_trace,lw=0.2, c='b')
    # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,roi],c='r',lw=1)

    ax1.set_xticks([0,5*fs])
    ax1.set_xticklabels(['0','5'])

    ax1.set_xlim([0,t_stop_idx-t_start_idx])
    ax2.set_xlim([0,t_stop_idx-t_start_idx])

    ax1.set_ylim([-1,6])
    ax2.set_ylim([-1,6])
    # ax2.set_ylim([-5,40])

    fname = 'subcell_trace_orig' + mouse + '_' + sess_soma
    subfolder = []
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    print('done')


if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'svg'


    MOUSE = 'LF180913_1'
    SESSION_SOMA = 'Day20181023_soma'
    SESSION_PAIR = 'Day20181023_obliques'
    # SESSION_PAIR = SESSION_SOMA
    SOMA_ROI = 0
    PAIR_ROI = [0,1,2,3]
    # PAIR_ROI = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    # PAIR_ROI = [41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
    # PAIR_ROI = [61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83]
    plot_ind_trace_behavior(MOUSE, SESSION_SOMA, SESSION_PAIR, SOMA_ROI, PAIR_ROI, 0, 250)

    # SESSION = 'Day201784'
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 461, 570)
