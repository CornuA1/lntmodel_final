    """
Plot trace of an individual ROI centered around the end end of the landmark

Calculate the standard deviation of peak brightness datapoint

Only calculate

@author: lukasfischer

"""

import numpy as np
import h5py
import sys
import yaml
import os
import json
import warnings; warnings.simplefilter('ignore')
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
from filter_trials import filter_trials
from scipy import stats
from scipy import signal

def fig_landmark_centered(h5path, sess, roi, fname, ylims=[], fformat='png', subfolder=[]):
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    # timewindow (minus and plus time in seconds), and
    MAX_TIMEWINDOW = np.asarray([5,5])
    # how many seconds in the blackbox get added on either side
    BLACKBOX_TIME = np.asarray([2,2])
    # calculate how many indexes that is in the current recording
    max_timewindow_idx = (MAX_TIMEWINDOW/behav_ds[0,2]).astype('int')
    # calculate the added frames for the blackbox
    blackbox_idx = (BLACKBOX_TIME/behav_ds[0,2]).astype('int')
    # determine maximum number of samples per trial
    t_max = (max_timewindow_idx[0]+blackbox_idx[0]) + (max_timewindow_idx[1]+blackbox_idx[1])
    # store center of window in case it is not symmetric
    window_center = max_timewindow_idx[0]+blackbox_idx[0]

    # calcluate standard deviation of ROI traces
    roi_std = np.std(dF_ds[:,roi])
    # threshold the response of a roi in a given trial has to exceed count its response toward the tuning of the cell
    trial_std_threshold = 3
    # on which fraction of trials did the roi exceed 3 standard deviations
    roi_active_fraction_short = 0
    roi_active_fraction_long = 0

    # specify track numbers
    track_short = 3
    track_long = 4

    # ylims
    min_y = -0.3
    max_y = 0.0
    if not ylims == []:
        min_y = ylims[0]
        max_y = ylims[1]

    # create figure and axes to later plot on
    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    # get indices of desired behavioural event
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])

    events_short = event_ind(behav_ds,['at_location', 240], trials_short)
    events_long = event_ind(behav_ds,['at_location', 240], trials_long)

    # get indices of trial start and end (or max_event_timewindow if trial is long)
    trial_dF_short = np.zeros((np.size(events_short[:,0]),2))
    for i,cur_ind in enumerate(events_short):
        cur_trial_idx = [np.where(behav_ds[:,6] == cur_ind[1])[0][0],np.where(behav_ds[:,6] == cur_ind[1])[0][-1]]
        # # determine indices of beginning and end of timewindow
        if cur_ind[0] - max_timewindow_idx[0] > cur_trial_idx[0]:
            trial_dF_short[i,0] = cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0])
        else:
            if cur_trial_idx[0] - blackbox_idx[0] < 0:
                trial_dF_short[i,0] = 0
            else:
                trial_dF_short[i,0] = cur_trial_idx[0] - blackbox_idx[0]

        if cur_ind[0] + max_timewindow_idx[1] < cur_trial_idx[1]:
            trial_dF_short[i,1] = cur_ind[0] + (max_timewindow_idx[1] + blackbox_idx[1])
        else:
            if cur_trial_idx[1] + blackbox_idx[1] > np.size(behav_ds,0):
                trial_dF_short[i,1] = np.size(behav_ds,0)
            else:
                trial_dF_short[i,1] = cur_trial_idx[1] + blackbox_idx[1]

    # grab dF data for each trial
    cur_trial_dF_short = np.full((np.size(events_short[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(events_short[:,0]))
    cur_trial_max_idx_short = np.empty(0)
    for i in range(np.size(trial_dF_short,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF_short[i,0]:trial_dF_short[i,1],roi]
        cur_trial_event_idx[i] = events_short[i,0] - trial_dF_short[i,0]
        trace_start = window_center - cur_trial_event_idx[i]
        cur_trial_dF_short[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        # only consider roi's max dF value in a given trial if it exceeds threshold
        if np.amax(cur_sweep) > trial_std_threshold * roi_std:
            cur_trial_max_idx_short = np.append(cur_trial_max_idx_short,np.argmax(cur_sweep))
        if np.amax(cur_sweep) > max_y:
            max_y = np.amax(cur_sweep)

    # plot individual traces
    for i,ct in enumerate(cur_trial_dF_short):
        ax1.plot(ct,c='0.65',lw=1)
    # ax1_1 = ax1.twinx()
    if len(cur_trial_max_idx_short) > 1:
        roi_active_fraction_short = len(cur_trial_max_idx_short)/np.size(trial_dF_short,0)
        sns.distplot(cur_trial_max_idx_short,hist=False,kde=False,rug=True,ax=ax1)

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(cur_trial_dF_short.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < 0.8:
            mean_valid_indices.append(i)
    ax1.plot(np.nanmean(cur_trial_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
    ax1.axvline(window_center,c='r',lw=2)

    # get indices of trial start and end (or max_event_timewindow if trial is long)
    trial_dF_long = np.zeros((np.size(events_long[:,0]),2))
    for i,cur_ind in enumerate(events_long):
        cur_trial_idx = [np.where(behav_ds[:,6] == cur_ind[1])[0][0],np.where(behav_ds[:,6] == cur_ind[1])[0][-1]]
        # # determine indices of beginning and end of timewindow
        if cur_ind[0] - max_timewindow_idx[0] > cur_trial_idx[0]:
            trial_dF_long[i,0] = cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0])
        else:
            if cur_trial_idx[0] - blackbox_idx[0] < 0:
                trial_dF_long[i,0] = 0
            else:
                trial_dF_long[i,0] = cur_trial_idx[0] - blackbox_idx[0]

        if cur_ind[0] + max_timewindow_idx[1] < cur_trial_idx[1]:
            trial_dF_long[i,1] = cur_ind[0] + (max_timewindow_idx[1] + blackbox_idx[1])
        else:
            if cur_trial_idx[1] + blackbox_idx[1] > np.size(behav_ds,0):
                trial_dF_long[i,1] = np.size(behav_ds,0)
            else:
                trial_dF_long[i,1] = cur_trial_idx[1] + blackbox_idx[1]

    # grab dF data for each trial
    cur_trial_dF_long = np.full((np.size(events_long[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(events_long[:,0]))
    cur_trial_max_idx_long = np.empty(0)
    for i in range(np.size(trial_dF_long,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF_long[i,0]:trial_dF_long[i,1],roi]
        cur_trial_event_idx[i] = events_long[i,0] - trial_dF_long[i,0]
        trace_start = window_center - cur_trial_event_idx[i]
        cur_trial_dF_long[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        if np.amax(cur_sweep) > trial_std_threshold * roi_std:
            cur_trial_max_idx_long = np.append(cur_trial_max_idx_long,np.argmax(cur_sweep))
        if np.amax(cur_sweep) > max_y:
            max_y = np.amax(cur_sweep)

    # plot traces
    for i,ct in enumerate(cur_trial_dF_long):
        ax2.plot(ct,c='0.65',lw=1)

    if len(cur_trial_max_idx_long) > 1:
        roi_active_fraction_long = len(cur_trial_max_idx_long)/np.size(trial_dF_long,0)
        sns.distplot(cur_trial_max_idx_long,hist=False,kde=False,rug=True,ax=ax2)

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(cur_trial_dF_long.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < 0.8:
            mean_valid_indices.append(i)
    ax2.plot(np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
    ax2.axvline(window_center,c='r',lw=2)

    sns.heatmap(cur_trial_dF_short,cmap='viridis',vmin=0,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax3)
    sns.heatmap(cur_trial_dF_long,cmap='viridis',vmin=0,yticklabels=events_long[:,1].astype('int'),xticklabels=False,ax=ax4)
    ax3.axvline(window_center,c='r',lw=2)
    ax4.axvline(window_center,c='r',lw=2)

    ax1.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)
    ax2.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)

    ax1.set_title('dF/F vs time SHORT track. Std: ' + str(np.round(np.std(cur_trial_max_idx_short),2)) + ' active: ' + str(np.round(roi_active_fraction_short,2)))
    ax2.set_title('dF/F vs time LONG track. Std: ' + str(np.round(np.std(cur_trial_max_idx_long),2)) + ' active: ' + str(np.round(roi_active_fraction_long,2)))
    ax3.set_title('dF/F vs time SHORT track - heatmap')
    ax4.set_title('dF/F vs time LONG track - heatmap')

    ax1.set_ylim([min_y,max_y])
    ax2.set_ylim([min_y,max_y])

    fig.suptitle('landmark onset centered' + fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    return np.std(cur_trial_max_idx_short),np.std(cur_trial_max_idx_long), roi_active_fraction_short, roi_active_fraction_long

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'

    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    subfolder = MOUSE+'_'+SESSION+'_lmoff'

    session_rois = {
        'mouse_session' : MOUSE+SESSION,
        'lmoff_std_short' : [],
        'lmoff_std_long' : [],
        'lmoff_active_short' : [],
        'lmoff_active_long' : []
    }

    lmoff_std_short = []
    lmoff_std_long = []
    for r in range(152):
        print(r)
        std_short, std_long, active_short, active_long = fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_'+str(r), [], fformat, subfolder)
        session_rois['lmoff_std_short'].append(std_short)
        session_rois['lmoff_std_long'].append(std_long)
        session_rois['lmoff_active_short'].append(active_short)
        session_rois['lmoff_active_long'].append(active_long)

    # print(fig_landmark_centered(h5path, SESSION, 6, MOUSE+'_'+SESSION+'_'+str(6), [], fformat, subfolder))

    with open(loc_info['figure_output_path'] + subfolder + os.sep + 'std_peaks.json','w+') as f:
        json.dump(session_rois,f)

    SESSION = 'Day201748_2'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    subfolder = MOUSE+'_'+SESSION+'_lmoff'

    session_rois = {
        'mouse_session' : MOUSE+SESSION,
        'lmoff_std_short' : [],
        'lmoff_std_long' : [],
        'lmoff_active_short' : [],
        'lmoff_active_long' : []
    }

    lmoff_std_short = []
    lmoff_std_long = []
    for r in range(171):
        print(r)
        std_short, std_long, active_short, active_long = fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_'+str(r), [], fformat, subfolder)
        session_rois['lmoff_std_short'].append(std_short)
        session_rois['lmoff_std_long'].append(std_long)
        session_rois['lmoff_active_short'].append(active_short)
        session_rois['lmoff_active_long'].append(active_long)

    with open(loc_info['figure_output_path'] + subfolder + os.sep + 'std_peaks.json','w+') as f:
        json.dump(session_rois,f)

    SESSION = 'Day201748_3'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    subfolder = MOUSE+'_'+SESSION+'_lmoff'

    session_rois = {
        'mouse_session' : MOUSE+SESSION,
        'lmoff_std_short' : [],
        'lmoff_std_long' : [],
        'lmoff_active_short' : [],
        'lmoff_active_long' : []
    }

    lmoff_std_short = []
    lmoff_std_long = []
    for r in range(50):
        print(r)
        std_short, std_long, active_short, active_long = fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_'+str(r), [], fformat, subfolder)
        session_rois['lmoff_std_short'].append(std_short)
        session_rois['lmoff_std_long'].append(std_long)
        session_rois['lmoff_active_short'].append(active_short)
        session_rois['lmoff_active_long'].append(active_long)

    with open(loc_info['figure_output_path'] + subfolder + os.sep + 'std_peaks.json','w+') as f:
        json.dump(session_rois,f)

    # fig_landmark_centered(h5path, SESSION, 6, MOUSE+'_'+SESSION+'_'+str(6), [], fformat, MOUSE+'_'+SESSION+'_lmoff')
