"""
plot a single eventaligned figure. Intended to customize plots for figures

@author: lukasfischer

"""

import numpy as np
import h5py
import sys
import yaml
import os
import json
import warnings; warnings.simplefilter('ignore')
from scipy.signal import butter, filtfilt
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

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def fig_landmark_centered(h5path, sess, roi, fname, eventshort, eventlong, max_timewindow, blackbox_time, ylims=[], fformat='png', subfolder=[], peak_times=[]):
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()



    # number of trials a roi as to be active (at least one transient) to be counted
    MIN_ACTIVE_TRIALS = 5
    # timewindow (minus and plus time in seconds), and
    MAX_TIMEWINDOW = np.asarray(max_timewindow)
    # how many seconds in the blackbox get added on either side
    BLACKBOX_TIME = np.asarray(blackbox_time)
    # calculate how many indexes that is in the current recording
    frame_latency = behav_ds[0,2]
    max_timewindow_idx = (MAX_TIMEWINDOW/frame_latency).astype('int')
    # calculate the added frames for the blackbox
    blackbox_idx = (BLACKBOX_TIME/frame_latency).astype('int')
    # determine maximum number of samples per trial
    t_max = (max_timewindow_idx[0]+blackbox_idx[0]) + (max_timewindow_idx[1]+blackbox_idx[1])
    # store center of window in case it is not symmetric
    window_center = max_timewindow_idx[0]+blackbox_idx[0]

    # calcluate standard deviation of ROI traces
    roi_std = np.std(dF_ds[:,roi])

    order = 6
    fs = 15.5      # sample rate, Hz
    cutoff = 2 # desired cutoff frequency of the filter, Hz
    dF_ds[:,roi] = butter_lowpass_filter(dF_ds[:,roi], cutoff, fs, order)

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
    fig = plt.figure(figsize=(12,12))
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(322)
    ax3 = plt.subplot(323)
    ax4 = plt.subplot(324)
    ax5 = plt.subplot(325)
    ax6 = plt.subplot(326)

    # get indices of desired behavioural event
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])

    trials_short = filter_trials( behav_ds, [], ['trialnr_range',60,90], trials_short)
    trials_long = filter_trials( behav_ds, [], ['trialnr_range',60,90],trials_long)


    events_short = event_ind(behav_ds, eventshort, trials_short)
    events_long = event_ind(behav_ds, eventlong, trials_long)

    # get indices of trial start and end (or max_event_timewindow if trial is long)
    trial_dF_short = np.zeros((np.size(events_short[:,0]),2))
    for i,cur_ind in enumerate(events_short):
        cur_trial_idx = [np.where(behav_ds[:,6] == cur_ind[1])[0][0],np.where(behav_ds[:,6] == cur_ind[1])[0][-1]]
        # # determine indices of beginning and end of timewindow
        if cur_ind[0] - max_timewindow_idx[0] > cur_trial_idx[0]:
            if cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0]) < 0:
                trial_dF_short[i,0] = 0
            else:
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
    cur_trial_speed_short = np.full((np.size(events_short[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(events_short[:,0]))
    cur_trial_max_idx_short = np.empty(0)
    for i in range(np.size(trial_dF_short,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF_short[i,0]:trial_dF_short[i,1],roi]
        cur_trial_event_idx[i] = events_short[i,0] - trial_dF_short[i,0]
        trace_start = window_center - cur_trial_event_idx[i]
        cur_trial_dF_short[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        cur_trial_speed_short[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF_short[i,0]:trial_dF_short[i,1],3]
        # only consider roi's max dF value in a given trial if it exceeds threshold
        if np.amax(cur_sweep) > trial_std_threshold * roi_std:
            cur_trial_max_idx_short = np.append(cur_trial_max_idx_short,np.nanargmax(cur_trial_dF_short[i,:]))
        if np.amax(cur_sweep) > max_y:
            max_y = np.amax(cur_sweep)

    # plot individual traces
    for i,ct in enumerate(cur_trial_dF_short):
        ax1.plot(ct,c='0.65',lw=2)
    # ax1_1 = ax1.twinx()
    if len(cur_trial_max_idx_short) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction_short = len(cur_trial_max_idx_short)/np.size(trial_dF_short,0)
        # sns.distplot(cur_trial_max_idx_short,hist=False,kde=False,rug=True,ax=ax1)
        roi_std_short = np.std(cur_trial_max_idx_short)
    else:
        roi_active_fraction_short = -1
        roi_std_short = -1

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(cur_trial_dF_short.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < 0.5:
            mean_valid_indices.append(i)
    ax1.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),np.nanmean(cur_trial_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
    # ax1.axvline(window_center,c='r',lw=2)
    roi_meanpeak_short = np.nanmax(np.nanmean(cur_trial_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanmin_short = np.nanmin(np.nanmean(cur_trial_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_short_idx = np.nanargmax(np.nanmean(cur_trial_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_short_time = ((roi_meanpeak_short_idx+mean_valid_indices[0])-window_center) * frame_latency

    if len(peak_times) > 0:
        window_center_time = window_center * frame_latency
        vr_peak_time_short = peak_times[0] + window_center_time
        vr_peak_time_short_idx = (vr_peak_time_short/frame_latency).astype('int')
        roi_meanpeak_short = np.nanmean(cur_trial_dF_short,0)[vr_peak_time_short_idx]
        # ax1.axvline(vr_peak_time_short_idx)
    else:
        pass
        # ax1.axvline((roi_meanpeak_short_idx+mean_valid_indices[0]))

    # get indices of trial start and end (or max_event_timewindow if trial is long)
    trial_dF_long = np.zeros((np.size(events_long[:,0]),2))
    for i,cur_ind in enumerate(events_long):
        cur_trial_idx = [np.where(behav_ds[:,6] == cur_ind[1])[0][0],np.where(behav_ds[:,6] == cur_ind[1])[0][-1]]
        # # determine indices of beginning and end of timewindow
        if cur_ind[0] - max_timewindow_idx[0] > cur_trial_idx[0]:
            if cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0]) < 0:
                trial_dF_long[i,0] = 0
            else:
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
    cur_trial_speed_long = np.full((np.size(events_long[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(events_long[:,0]))
    cur_trial_max_idx_long = np.empty(0)
    for i in range(np.size(trial_dF_long,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF_long[i,0]:trial_dF_long[i,1],roi]
        cur_trial_event_idx[i] = events_long[i,0] - trial_dF_long[i,0]
        trace_start = window_center - cur_trial_event_idx[i]
        cur_trial_dF_long[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        cur_trial_speed_long[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF_long[i,0]:trial_dF_long[i,1],3]
        if np.amax(cur_sweep) > trial_std_threshold * roi_std:
            cur_trial_max_idx_long = np.append(cur_trial_max_idx_long,np.nanargmax(cur_trial_dF_long[i,:]))
        if np.amax(cur_sweep) > max_y:
            max_y = np.amax(cur_sweep)

    # plot traces
    for i,ct in enumerate(cur_trial_dF_long):
        ax2.plot(ct,c='0.65',lw=2)

    if len(cur_trial_max_idx_long) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction_long = len(cur_trial_max_idx_long)/np.size(trial_dF_long,0)
        # sns.distplot(cur_trial_max_idx_long,hist=False,kde=False,rug=True,ax=ax2)
        roi_std_long = np.std(cur_trial_max_idx_long)
    else:
        roi_active_fraction_long = -1
        roi_std_long = -1

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(cur_trial_dF_long.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < 0.5:
            mean_valid_indices.append(i)
    ax2.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
    # ax2.axvline(window_center,c='r',lw=2)
    roi_meanpeak_long = np.nanmax(np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanmin_long = np.nanmin(np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_long_idx = np.nanargmax(np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_long_time = ((roi_meanpeak_long_idx+mean_valid_indices[0])-window_center) * frame_latency

    if len(peak_times) > 0:
        window_center_time = window_center * frame_latency
        vr_peak_time_long = peak_times[1] + window_center_time
        vr_peak_time_long_idx = (vr_peak_time_long/frame_latency).astype('int')
        roi_meanpeak_long = np.nanmean(cur_trial_dF_long,0)[vr_peak_time_long_idx]
        # ax2.axvline(vr_peak_time_long_idx)
    else:
        pass
        # ax2.axvline((roi_meanpeak_long_idx+mean_valid_indices[0]))

    sns.heatmap(cur_trial_dF_short,cmap='viridis',vmin=0,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax3)
    sns.heatmap(cur_trial_dF_long,cmap='viridis',vmin=0,yticklabels=events_long[:,1].astype('int'),xticklabels=False,ax=ax4)
    sns.heatmap(cur_trial_speed_short,cmap='viridis',vmin=0,vmax=60,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax5)
    sns.heatmap(cur_trial_speed_long,cmap='viridis',vmin=0,vmax=60,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax6)
    ax3.axvline(window_center,c='r',lw=2)
    ax4.axvline(window_center,c='r',lw=2)
    ax5.axvline(window_center,c='r',lw=2)
    ax6.axvline(window_center,c='r',lw=2)

    # ax1.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)
    # ax2.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)

    mean_amplitude_short = roi_meanpeak_short - roi_meanmin_short
    mean_amplitude_long = roi_meanpeak_long - roi_meanmin_long

    # ax1.set_title('dF/F vs time SHORT track. Std: ' + str(np.round(roi_std_short,2)) + ' active: ' + str(np.round(roi_active_fraction_short,2)) + ' peak: ' + str(np.round(roi_meanpeak_short,2)) + ' meanamp:' + str(mean_amplitude_short))
    # ax2.set_title('dF/F vs time LONG track. Std: ' + str(np.round(roi_std_long,2)) + ' active: ' + str(np.round(roi_active_fraction_long,2)) + ' peak: ' + str(np.round(roi_meanpeak_long,2)) + ' meanamp:' + str(mean_amplitude_long))
    # ax3.set_title('dF/F vs time SHORT track - heatmap')
    # ax4.set_title('dF/F vs time LONG track - heatmap')

    ax1.set_ylim([-0.1,0.7])
    ax2.set_ylim([-0.1,0.7])
    ax1.set_xlim([0,t_max])
    ax2.set_xlim([0,t_max])

    ax1.set_xticks([1/frame_latency,2/frame_latency])
    ax1.set_xticklabels(['1','2'])

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

    fig.tight_layout()
    fformat = 'svg'
    fig.suptitle(str(roi), wrap=True)
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

    return roi_std_short,roi_std_long, roi_active_fraction_short, roi_active_fraction_long, roi_meanpeak_short, roi_meanpeak_long, roi_meanpeak_short_time, roi_meanpeak_long_time



def do_single():
    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # NUM_ROIS = [40]
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # filterprop_short = ['at_location', 240]
    # filterprop_long = ['at_location', 240]
    # even_win = [6,6]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmoff'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    #
    # for r in NUM_ROIS:
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [], fformat, subfolder)

    MOUSE = 'LF170214_1'
    SESSION = 'Day2017714'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = [41]
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    filterprop_short = ['at_location', 240]
    filterprop_long = ['at_location', 240]
    even_win = [6,6]
    blackbox_win = [2,2]
    SUBNAME = 'lmoff'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    for r in NUM_ROIS:
        fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [], fformat, subfolder)


if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'

    # run_LF170110_2_Day201748_1()
    # run_LF170110_2_Day201748_2()
    # run_LF170110_2_Day201748_3()
    # run_LF170421_2_Day2017719()
    # run_LF170421_2_Day2017720()
    # run_LF170420_1_Day201783()
    # run_LF170420_1_Day2017719()
    # run_LF170222_1_Day201776()
    # run_LF170110_2_Day2017331()
    # run_LF170613_1_Day201784()
    # run_LF171212_2_Day2018218_1()
    # run_LF171212_2_Day2018218_2()

    # run_LF170214_1_Day201777()
    # run_LF170214_1_Day2017714()
    # run_LF171211_2_Day201852()
    # run_LF18112_2_Day2018322_1()
    # run_LF18112_2_Day2018322_2()

    do_single()
