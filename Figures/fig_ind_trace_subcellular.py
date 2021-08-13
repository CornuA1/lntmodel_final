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
    behav_ds = np.copy(h5dat[sess_soma + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess_soma + '/dF_win'])
    dF_ds_pair = np.copy(h5dat[sess_pair + '/dF_win'])
    h5dat.close()

    # sess_soma = 'Day201748_2'
    # h5dat = h5py.File(h5path, 'r')
    # behav_ds2 = np.copy(h5dat[sess_soma + '/behaviour_aligned'])
    # dF_ds2 = np.copy(h5dat[sess_soma + '/dF_win'])
    # h5dat.close()

    # set up figure
    fig = plt.figure(figsize=(32,4))
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

    t_start_idx = (np.abs(behav_ds[:,0] - t_start)).argmin()
    t_stop_idx = (np.abs(behav_ds[:,0] - t_stop)).argmin()

    order = 6
    fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
    cutoff = 2 # desired cutoff frequency of the filter, Hz
    # plot dF trace
    pair_trace = np.zeros((dF_ds_pair.shape[0]-1, len(pair_roi)))
    filter_trace = 1
    if filter_trace == 1:
        soma_trace = butter_lowpass_filter(dF_ds[t_start_idx:t_stop_idx,roi], cutoff, fs, order)
        for i,pr in enumerate(pair_roi):
            pair_trace[:,i] = butter_lowpass_filter(dF_ds_pair[t_start_idx:t_stop_idx,pr], cutoff, fs, order)
    else:
        soma_trace = dF_ds[t_start_idx:t_stop_idx,roi]
        for i,pr in enumerate(pair_roi):
            pair_trace[:,i] = dF_ds_pair[t_start_idx:t_stop_idx,pr]

    ax1.plot(soma_trace,lw=0.2, c='b')
    ax1.plot(pair_trace,lw=0.2)

    ax2.plot(pair_trace,lw=0.2)
    ax2.plot(soma_trace,lw=0.2, c='b')
    # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,roi],c='r',lw=1)

    # filter and plot running speed trace
    if sess_soma.find('openloop') > 0:
        speed_filtered = butter_lowpass_filter(behav_ds[:,8], cutoff, fs, order)
    else:
        speed_filtered = butter_lowpass_filter(behav_ds[:,3], cutoff, fs, order)
    # ax2.plot(speed_filtered[t_start_idx:t_stop_idx],c='g',lw=2)

    # shade areas corresponding to the landmark
    if sess_soma.find('dark') == -1:
        landmark = [200,240]
        lm_temp = behav_ds[:,1]
        lm_start_idx = np.where(lm_temp > landmark[0])[0]
        lm_end_idx = np.where(lm_temp < landmark[1])[0]
        lm_idx = np.intersect1d(lm_start_idx,lm_end_idx)
        lm_diff = np.diff(lm_idx)
        lm_end = np.where(lm_diff>1)[0]
        lm_start = np.insert(lm_end,0,0)+1
        lm_end = np.append(lm_end,lm_idx.size-1)
        if lm_start.size > lm_end.size:
            lm_end.append(np.size(behav_ds),0)

        for i,lm in enumerate(lm_start):
            # print(behav_ds.shape)
            # print(lm_idx)
            if behav_ds[lm_idx[lm],4] != 5:
                if lm_idx[lm_start[i]] > t_start_idx and lm_idx[lm_start[i]] < t_stop_idx:
                    if behav_ds[lm_idx[lm],4] == 3:
                        ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#45EDFF',alpha=0.2)
                        ax2.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#45EDFF',alpha=0.2)
                    else:
                        ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#F900FF',alpha=0.2)
                        ax2.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#F900FF',alpha=0.2)

     # plot time where animal was in blackbox
    bb_temp = behav_ds[:,4]
    bb_temp[np.where(behav_ds[:,4]!=5)[0]] = 0
    bb_diff = np.diff(bb_temp)
    bb_start = np.where(bb_diff>1)[0]
    bb_end = np.where(bb_diff<0)[0]
    if bb_start.size > bb_end.size:
        np.append(bb_end, np.size(behav_ds,0)-1)

    for i,bb in enumerate(bb_start):
        if bb_start[i] > t_start_idx and bb_start[i] < t_stop_idx:
            ax1.axvspan(bb_start[i],bb_end[i],color='0.85')
            ax2.axvspan(bb_start[i],bb_end[i],color='0.85')


    ax1.set_xlim([0,t_stop_idx-t_start_idx])
    ax2.set_xlim([0,t_stop_idx-t_start_idx])

    one_sec = (t_stop_idx-t_start_idx)/(t_stop - t_start)
    ax1.set_xticks([0,5*one_sec])
    ax1.set_xticklabels(['0','5'])

    ax1.set_yticks([0,2,4,6])
    ax1.set_yticklabels(['0','2','4','6'])
    ax1.set_ylabel('dF/F', fontsize=16)

    # ax2.set_yticks([0,10,20,30,40])
    # ax2.set_yticklabels(['0','10','20','30','40'])
    # ax2.set_ylabel('speed (cm/sec)', fontsize=16)

    ax1.set_ylim([-1,6])
    ax2.set_ylim([-1,6])
    # ax2.set_ylim([-5,40])

    fname = 'subcell_trace' + mouse + '_' + sess_soma
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
    fformat = 'png'

    MOUSE = 'LF170613_1'
    SESSION_SOMA = '20170804'
    
    SOMA_ROI = 0
    PAIR_ROI = [1]
    # PAIR_ROI = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    # PAIR_ROI = [41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
    # PAIR_ROI = [61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83]
    plot_ind_trace_behavior(MOUSE, SESSION_SOMA, SESSION_PAIR, SOMA_ROI, PAIR_ROI, 0, 100000)

    # SESSION = 'Day201784'
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 461, 570)
