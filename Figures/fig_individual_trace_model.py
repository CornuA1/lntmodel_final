"""
plot section of individual traces during behaviour

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
import scipy.io as sio
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

from filter_trials import filter_trials
from event_ind import event_ind
from load_filelist_model import load_filelist
import seaborn as sns
sns.set_style('white')

fformat = 'svg'

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_ind_trace_behavior(mouse, sess, roi, t_start, t_stop, ylims =[], plot_task_feature='landmark_shaded', speed_col=3):
    filedict = load_filelist()
    data_path = filedict[mouse][sess]
    print(mouse, sess)
    behav_ds = sio.loadmat(data_path[0])["data"]
    dF_ds = sio.loadmat(data_path[1])["data"]

    # set up figure
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    t_start_idx = (np.abs(behav_ds[:,0] - t_start)).argmin()
    t_stop_idx = (np.abs(behav_ds[:,0] - t_stop)).argmin()
    x_vals = np.linspace(0, t_stop_idx-t_start_idx, t_stop_idx-t_start_idx)

    # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,roi],c='r',lw=1)
    min_dspeed = -30
    max_dspeed = 45
    
    

    # filter and plot running speed trace
    order = 6
    fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
    cutoff = 10 # desired cutoff frequency of the filter, Hz

    speed_mouse = butter_lowpass_filter(behav_ds[:,speed_col], cutoff, fs, order)
    speed_vr = 30
    speed_filtered = speed_mouse - speed_vr
    speed_plot = speed_filtered[t_start_idx:t_stop_idx]
    # ax2.plot(x_vals,speed_plot,c='g',lw=2)

    # plot speed as a function of how different it is from the VR speed
    points = np.array([x_vals, speed_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(min_dspeed, max_dspeed)
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    
    lc.set_array(speed_plot)
    lc.set_linewidth(2)
    line = ax2.add_collection(lc)
    # fig.colorbar(line, ax=ax2)


    # plot dF trace
    cutoff = 1
    dF_ds[:,roi] = butter_lowpass_filter(dF_ds[:,roi], cutoff, fs, order)
    ax1.plot(x_vals,dF_ds[t_start_idx:t_stop_idx,roi],c='k',lw=2,zorder=3)

    if plot_task_feature == 'landmark_shaded':
        # shade areas corresponding to the landmark
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
            if behav_ds[lm_idx[lm],4]!=5:
                if lm_idx[lm_start[i]] > t_start_idx and lm_idx[lm_start[i]] < t_stop_idx:
                    if behav_ds[lm_idx[lm],4] == 3:
                        ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='0.9')
                    else:
                        ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='0.7')

    elif plot_task_feature is 'trialonset':
        trials_short = filter_trials( behav_ds, [], ['tracknumber',3])
        trials_long = filter_trials( behav_ds, [], ['tracknumber',4])
        trials_all = np.union1d(trials_short,trials_long)
        events = event_ind(behav_ds, ['trial_transition'], trials_all)
        for e in events:
            ax1.axvline(e[0]-t_start_idx,c='#39B54A',lw=3,zorder=2)

    elif plot_task_feature is 'lmcenter':
        trials_short = filter_trials( behav_ds, [], ['tracknumber',3])
        trials_long = filter_trials( behav_ds, [], ['tracknumber',4])
        trials_all = np.union1d(trials_short,trials_long)
        events = event_ind(behav_ds, ['at_location', 220], trials_all)
        for e in events:
            ax1.axvline(e[0]-t_start_idx,c='r',lw=3,zorder=2)

    elif plot_task_feature is 'rewards':
        trials_short = filter_trials( behav_ds, [], ['tracknumber',3])
        trials_long = filter_trials( behav_ds, [], ['tracknumber',4])
        trials_all = np.union1d(trials_short,trials_long)
        events = event_ind(behav_ds, ['rewards_all', -1], trials_all)
        for e in events:
            ax1.axvline(e[0]-t_start_idx,c='#29ABE2',lw=3,zorder=2)


    ax1.set_xlim([0,t_stop_idx-t_start_idx])
    ax2.set_xlim([0,t_stop_idx-t_start_idx])

    one_sec = (t_stop_idx-t_start_idx)/(t_stop - t_start)
    ax1.set_xticks([0,one_sec,5*one_sec])
    ax1.set_xticklabels(['0','1','5'])

    # ax1.set_yticks([0,0.1,0.5,2,4,6])
    # ax1.set_yticklabels(['0','0.1','0.5','2','4','6'])
    # ax1.set_ylabel('dF/F', fontsize=16)

    ax2.set_ylim([-32,45])
    ax2.set_yticks([-30,-15,0,15,30])
    ax2.set_yticklabels(['-30','-15','0','15','30'])
    ax2.set_ylabel('speed (cm/sec)', fontsize=16)

    # if ylims is []:
    #     ax1.set_ylim([-0.1,10])
    # else:
    #     ax1.set_ylim(ylims)

    # ax2.set_ylim([-30,45])


    fname = 'ind_trace' + mouse + '_' + sess + '_' + str(roi) + '_' + str(t_start)

    subfolder = 'ind_traces'
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat


    fig.savefig(fname, format=fformat,dpi=150)

    print(fname)
    print('done')


if __name__ == '__main__':
    # MOUSE = 'LF191023_blank'
    # SESSION = '1206 ol'
    # plot_task_feature = 'lmcenter'
    # ROI = 59
    # speed_col = 8
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 450, 590, [-0.5,6], plot_task_feature, speed_col) 
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 0, 140, [-0.5,6], plot_task_feature, speed_col)
    
    MOUSE = 'LF191023_blank'
    SESSION = '1114 ol'
    plot_task_feature = 'lmcenter'
    ROI = 15
    speed_col = 8
    plot_ind_trace_behavior(MOUSE, SESSION, ROI, 450, 590, [-0.5,6], plot_task_feature, speed_col) #1285, 1367
    plot_ind_trace_behavior(MOUSE, SESSION, ROI, 0, 140, [-0.5,6], plot_task_feature, speed_col) #1285, 1367

   
