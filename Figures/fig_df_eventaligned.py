"""
Plot trace of an individual ROI centered around a given location

Calculate the standard deviation of peak brightness datapoint

Only calculate

@author: lukasfischer

"""

import numpy as np
import scipy as sp
import h5py
import sys
import yaml
import os
import json
import ipdb
from multiprocessing import Process
import warnings; warnings.simplefilter('ignore')
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
import seaborn as sns
sns.set_style("white")

plt.rcParams['svg.fonttype'] = 'none'
os.chdir('C:/Users/Lou/Documents/repos/LNT')
with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
from filter_trials import filter_trials
from order_trials import order_trials
from scipy import stats
from scipy import signal
import scipy.io as sio
from write_dict import write_dict
from analysis_parameters import MIN_FRACTION_ACTIVE, MIN_MEAN_AMP, MIN_ZSCORE, MIN_TRIALS_ACTIVE, MIN_DF, MIN_MEAN_AMP_BOUTONS, MEAN_TRACE_FRACTION, PEAK_MATCH_WINDOW

fformat = 'png'

make_reward = False
make_trialonset = False
make_lmcenter = True
make_firstlick = False

plot_sem_shaded = False
plot_ind_traces = True

# specify how often to shuffle each dataset to get null distribution
NUM_SHUFFLES = 100


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def roi_response_validation(fraction_active, roi_peak_zscore):
    """
    separate function to check whether a given response passes criterion for being considered a real roi_response_validation

    """
    #
    # roi_activity = roi_params[el + '_active_' + tl][roi_idx_num]
    # roi_peak_val = roi_params[el + '_peak_' + tl][roi_idx_num]
    # roi_zscore_val = roi_params[el + '_peak_zscore_' + tl][roi_idx_num]
    # mean_trace = roi_params['space_mean_trace_'+tl][roi_idx_num]
    # # roi_activity = el + '_active_' + tl
    # # roi_peak_val = el + '_peak_' + tl
    # # if roi_params[roi_activity][roi_idx_num] > MIN_TRIALS_ACTIVE and roi_params[roi_peak_val][roi_idx_num] > MIN_DF:
    # #     return True
    # # else:
    # #     return False
    # if plot_layer is 'v1':
    #     mean_amp_diff = MIN_MEAN_AMP_BOUTONS
    # else:
    #     mean_amp_diff = MIN_MEAN_AMP


    if fraction_active > MIN_FRACTION_ACTIVE and roi_peak_zscore > MIN_ZSCORE:
        return True
    else:
        return False

def make_blackbox_loc_continuous(behavior_data):
    """ check if location after a reward is reset to 0, or just keeps increasing. If it is reset, append last location on trial so as to make it continuous """

    # detect all transitions into the blackbox and add the last location value on the trial to location values in the blackbox if a location reset was detected
    trial_transitions = event_ind(behavior_data, ['trial_transition'])
    for tt in trial_transitions:
        # loc_offset keeps track of the value that needs to be added to each location value (there can be multiple resets in a blackbox)
        loc_offset = 0
        tt_idx = tt[0].astype(int)
        if behavior_data[tt_idx,4] == 5:
            bb_trial_idx = np.where(behavior_data[:,6]==behavior_data[tt_idx,6])[0]
            # when we detect a reset (anywhere in the blackbox), add last location on trial (include some tolerance to account for small backtracking of the animal when it receives a reward)
            for i in bb_trial_idx:
                if behavior_data[i,1] < (behavior_data[i-1,1] - 50):
                    behavior_data[i:bb_trial_idx[-1]+1,1] += behavior_data[i-1,1]
                    # print('reset ', str(behavior_data[i,6]), str(np.round(behavior_data[i-1,1],2)), str(i), str(bb_trial_idx[-1]))

    return behavior_data

def calc_aligned_activity(h5path, behav_ds, dF_ds, roi, trialtype, align_type, align_events, max_timewindow_idx, blackbox_idx, t_max, window_center, trial_std_threshold, roi_std, max_y, MIN_ACTIVE_TRIALS, frame_latency, peak_times, ax_object, ax_object2, make_figure):
    """ align dF/F trace of a given roi to events and calculate mean trace """
    # get indices of trial start and end (or max_event_timewindow if trial is long)
    trial_dF = np.zeros((np.size(align_events[:,0]),2))
    for i,cur_ind in enumerate(align_events):
        cur_trial_idx = [np.where(behav_ds[:,6] == cur_ind[1])[0][0],np.where(behav_ds[:,6] == cur_ind[1])[0][-1]]
        # # determine indices of beginning and end of timewindow
        if cur_ind[0] - max_timewindow_idx[0] > cur_trial_idx[0]:
            if cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0]) < 0:
                trial_dF[i,0] = 0
            else:
                trial_dF[i,0] = cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0])
        else:
            if cur_trial_idx[0] - blackbox_idx[0] < 0:
                trial_dF[i,0] = 0
            else:
                trial_dF[i,0] = cur_trial_idx[0] - blackbox_idx[0]

        if cur_ind[0] + max_timewindow_idx[1] < cur_trial_idx[1]:
            trial_dF[i,1] = cur_ind[0] + (max_timewindow_idx[1] + blackbox_idx[1])
        else:
            if cur_trial_idx[1] + blackbox_idx[1] > np.size(behav_ds,0):
                trial_dF[i,1] = np.size(behav_ds,0)
            else:
                trial_dF[i,1] = cur_trial_idx[1] + blackbox_idx[1]

        # print(align_events[i,:], max_timewindow_idx, blackbox_idx, cur_trial_idx, cur_ind[1], trial_dF[i,0], trial_dF[i,1])

    trial_dF = trial_dF.astype(int)
    # grab dF data for each trial
    cur_trial_dF = np.full((np.size(align_events[:,0]),int(t_max)),np.nan)
    cur_trial_dF_shuffled = np.full((np.size(align_events[:,0]),int(t_max), NUM_SHUFFLES),np.nan)
    cur_trial_loc = np.full((np.size(align_events[:,0]),int(t_max)),np.nan)
    cur_trial_speed = np.full((np.size(align_events[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(align_events[:,0]))
    cur_trial_max_idx = np.zeros(0)
    transient_max_loc = np.zeros(0)
    for i in range(np.size(trial_dF,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF[i,0]:trial_dF[i,1],roi]
        cur_trial_event_idx[i] = align_events[i,0] - trial_dF[i,0]
        trace_start = int(window_center - cur_trial_event_idx[i])
        cur_trial_dF[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        cur_trial_loc[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF[i,0]:trial_dF[i,1],1]
        cur_trial_speed[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF[i,0]:trial_dF[i,1],3]
        for j in range(NUM_SHUFFLES):
            shuffled_ds = np.roll(dF_ds[:,roi],np.random.randint(500,dF_ds.shape[0]))
            cur_shuffled_sweep = shuffled_ds[trial_dF[i,0]:trial_dF[i,1]]
            cur_trial_dF_shuffled[i,trace_start:trace_start+len(cur_sweep),j] = cur_shuffled_sweep
        # only consider roi's max dF value in a given trial if it exceeds threshold
        if np.amax(cur_sweep) > trial_std_threshold * roi_std:
            cur_trial_max_idx = np.append(cur_trial_max_idx,np.nanargmax(cur_trial_dF[i,:]))

            # get location of transient peak and location of alignment point
            transient_max_loc_cur = behav_ds[trial_dF[i,0]:trial_dF[i,1],1][np.nanargmax(cur_trial_dF[i,trace_start:trace_start+len(cur_sweep)])]
            transient_align_loc = behav_ds[align_events[i,0].astype(int),1]

            # get the trial number of the max location and check if its in the same trial or in the blackbox (in which case we have to deal with the location in the black box)
            max_loc_trial = behav_ds[trial_dF[i,0]:trial_dF[i,1],6][np.nanargmax(cur_trial_dF[i,trace_start:trace_start+len(cur_sweep)])]
            event_trial = align_events[i,1]
            # if transient location is in previous trial (i.e. in blackbox prior to start of current trial), calculate displacement accordingly
            if max_loc_trial < event_trial:
                # we calculate displacement in the blackbox as the total displacement from the point of the transient max to the start of the trial
                # print('correcting pre blackbox location...')
                blackbox_trial_end_idx = np.where(behav_ds[:,6] == max_loc_trial)[0][-1]
                transient_trial_start_idx = np.where(behav_ds[:,6] == event_trial)[0][0]
                blackbox_pre_trial_loc = behav_ds[blackbox_trial_end_idx,1] - transient_max_loc_cur
                transient_max_loc_cur = behav_ds[transient_trial_start_idx,1] - blackbox_pre_trial_loc

            # if transient location is in the blackbox after the trial, deal with distances accordingly
            if max_loc_trial > event_trial:
                # print('correcting post blackbox location...')
                blackbox_trial_start_idx = np.where(behav_ds[:,6] == max_loc_trial)[0][0]
                transient_trial_end_idx = np.where(behav_ds[:,6] == event_trial)[0][-1]
                blackbox_post_trial_loc = transient_max_loc_cur - behav_ds[blackbox_trial_start_idx,1]
                transient_max_loc_cur = behav_ds[transient_trial_end_idx,1] + blackbox_post_trial_loc

            transient_max_loc_cur = transient_max_loc_cur - transient_align_loc
            transient_max_loc = np.append(transient_max_loc,transient_max_loc_cur)

            # if transient_max_loc_cur > 400 or transient_max_loc_cur < -400:
            #     print('--- suspicious transient max location ---')
            #     print(h5path + ' ' + str(roi) + ' ' + align_type)
            #     print(max_loc_trial, event_trial, transient_max_loc_cur)
                # print(behav_ds[trial_dF[i,0]:trial_dF[i,1],0][np.nanargmax(cur_trial_dF[i,trace_start:trace_start+len(cur_sweep)])])
                # print(trial_dF[i,0], trial_dF[i,1])
                # print(behav_ds[trial_dF[i,0]:trial_dF[i,1],1][np.nanargmax(cur_trial_dF[i,trace_start:trace_start+len(cur_sweep)])])
                # print(behav_ds[trial_dF[i,0]:trial_dF[i,1],4][np.nanargmax(cur_trial_dF[i,trace_start:trace_start+len(cur_sweep)])])
                # if trialtype == 'short':
                #     transient_max_loc_cur = 340
                # elif trialtype == 'long':
                #     transient_max_loc_cur = 400



        if np.amax(cur_sweep) > max_y:
            max_y = np.amax(cur_sweep)

    # ax_object_1 = ax_object.twinx()
    if len(cur_trial_max_idx) >= 0:

        try:
            roi_active_fraction = np.float64(len(cur_trial_max_idx)/np.size(trial_dF,0))
        except:
            ipdb.set_trace()
        if len(cur_trial_max_idx) >= MIN_ACTIVE_TRIALS:
            if make_figure:
                sns.distplot(cur_trial_max_idx,hist=False,kde=False,rug=False,ax=ax_object)
        roi_std = np.std(cur_trial_max_idx)
    else:
        roi_active_fraction = np.int64(0)
        roi_std = np.int64(-1)

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(cur_trial_dF.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < MEAN_TRACE_FRACTION:
            mean_valid_indices.append(i)

    # plot individual traces
    if make_figure and plot_ind_traces:
        for i,ct in enumerate(cur_trial_dF):
            ax_object.plot(ct,c='0.65',lw=1,zorder=2)
    roi_meanpeak = np.nanmax(np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanmin = np.nanmin(np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_sem = stats.sem(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0,nan_policy='omit')
    roi_meanpeak_idx = np.nanargmax(np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_time = ((roi_meanpeak_idx+mean_valid_indices[0])-window_center) * frame_latency
    roi_mean_trace = np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    # print('--- MEAN LOC ---')
    # print(np.nanmean(cur_trial_loc[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_loc = np.nanmean(cur_trial_loc[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)[roi_meanpeak_idx]
    # print(roi_meanpeak_time, roi_meanpeak_idx)
    # print('----------------')

    speed_mean_trace = np.nanmean(cur_trial_speed[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    speed_sem = stats.sem(cur_trial_speed[:,mean_valid_indices[0]:mean_valid_indices[-1]],0,nan_policy='omit')
    roi_mean_trace_shuffled = np.full((roi_mean_trace.shape[0],NUM_SHUFFLES),np.nan)

    for j in range(NUM_SHUFFLES):
        roi_mean_trace_shuffled[:,j] = np.nanmean(cur_trial_dF_shuffled[:,mean_valid_indices[0]:mean_valid_indices[-1],j],0)
        # ax_object.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),roi_mean_trace_shuffled[:,j],c='g',lw=1)

    dF_sem = stats.sem(cur_trial_dF,0,nan_policy='omit')
    shuffled_mean = np.nanmean(roi_mean_trace_shuffled,axis=1)
    shuffled_sem = stats.sem(roi_mean_trace_shuffled, axis=1)
    shuffled_std = np.std(roi_mean_trace_shuffled, axis=1)
    # ipdb.set_trace()
    zscore_trace = (roi_mean_trace - shuffled_mean)/shuffled_std
    peak_zscore = (roi_mean_trace[roi_meanpeak_idx]-shuffled_mean[roi_meanpeak_idx])/shuffled_std[roi_meanpeak_idx]
    peak_zscore_individual = (np.amax(cur_trial_dF)-shuffled_mean[roi_meanpeak_idx])/shuffled_std[roi_meanpeak_idx]
    # print(peak_zscore)

    if make_figure:
        ax_object.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),roi_mean_trace,c='k',lw=2,zorder=4)
        if plot_sem_shaded:
            ax_object.fill_between(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),roi_mean_trace-roi_sem,roi_mean_trace+roi_sem,color='0.5',linewidth=0, alpha=0.5,zorder=3)
        ax_object.axvline(window_center,c='r',lw=2)
        # ax_object2.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),speed_mean_trace,c='g',lw=2,zorder=4)
        # ax_object2.fill_between(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),speed_mean_trace-speed_sem,speed_mean_trace+speed_sem,color='g',linewidth=0, alpha=0.5,zorder=3)
        ax_object2.set_ylim([-5,40])
        # ax_object.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),shuffled_mean,c='g',lw=2)
        # ipdb.set_trace()
        ax_object.fill_between(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),(shuffled_mean-shuffled_std),(shuffled_mean+shuffled_std),color='g',alpha=0.5,zorder=3)
        # print(roi_mean_trace[roi_meanpeak_idx], shuffled_mean[roi_meanpeak_idx], shuffled_std[roi_meanpeak_idx])
        # ax_object2 = ax_object.twinx()
        # ax_object2.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),(roi_mean_trace-shuffled_mean)/shuffled_std,c='m', lw=4)

    # peak_times contains time of mean peak (relative to alignment point) in VR (so this is only relevant for openloop)
    if len(peak_times) > 0:
        # map peak time in VR to peak time in openloop
        window_center_time = window_center * frame_latency
        vr_peak_time = peak_times[0] + window_center_time
        vr_peak_time_start = vr_peak_time - (PEAK_MATCH_WINDOW/2)
        vr_peak_time_end = vr_peak_time + (PEAK_MATCH_WINDOW/2)
        # print(vr_peak_time_start,vr_peak_time,vr_peak_time_end, window_center_time)
        # convert times to indices
        vr_peak_time_win_start_idx = np.amax([(vr_peak_time_start/frame_latency).astype('int'),mean_valid_indices[0]])
        vr_peak_time_win_end_idx = np.amin([(vr_peak_time_end/frame_latency).astype('int'),mean_valid_indices[-1]])

        # in case the indeces are nonsensical (this can happen in edge cases where the peak idx in VR is outside the valid_indeces range), just go with the original index
        if vr_peak_time_win_start_idx > vr_peak_time_win_end_idx:
            vr_peak_time_win_start_idx = (vr_peak_time/frame_latency).astype('int')
            vr_peak_time_win_end_idx = (vr_peak_time/frame_latency).astype('int')+1

        # grab peak value and index from mean trace
        # print(vr_peak_time_win_start_idx, vr_peak_time_win_end_idx)
        # print(cur_trial_dF[:,vr_peak_time_win_start_idx:vr_peak_time_win_end_idx])
        try:
            vr_peak_time_idx = np.argmax(np.nanmean(cur_trial_dF[:,vr_peak_time_win_start_idx:vr_peak_time_win_end_idx],0))
            roi_meanpeak = np.amax(np.nanmean(cur_trial_dF[:,vr_peak_time_win_start_idx:vr_peak_time_win_end_idx],0))
        except ValueError:
            # print(roi, vr_peak_time_win_start_idx, vr_peak_time_win_end_idx)
            vr_peak_time_idx = vr_peak_time_win_start_idx
            roi_meanpeak = np.nanmean(cur_trial_dF[:,vr_peak_time_win_start_idx],0)

        if make_figure:
            ax_object.axvline(vr_peak_time_win_start_idx+vr_peak_time_idx)
    else:
        if make_figure:
            ax_object.axvline((roi_meanpeak_idx+mean_valid_indices[0]))

    return cur_trial_dF, transient_max_loc, roi_mean_trace_shuffled, cur_trial_speed, roi_std, roi_mean_trace, roi_active_fraction, roi_meanpeak, roi_meanpeak_time, roi_meanpeak_loc, roi_meanmin, peak_zscore, peak_zscore_individual, zscore_trace, max_y


def fig_landmark_centered(h5path, sess, roi, fname, eventshort, eventlong, order_short, order_long, max_timewindow, blackbox_time, ylims=[], fformat='png', subfolder=[], peak_times=[], filter=False, filter_speed=False, make_figure=True, raw_data=False):
#    if not raw_data:
#        h5dat = h5py.File(h5path, 'r')
#        behav_ds = np.copy(h5dat['behaviour_aligned'])
#        dF_ds = np.copy(h5dat['dF_aligned'])
#        h5dat.close()
#        
#    else:
#        loaded_data = sio.loadmat(h5path)
#        behav_ds = loaded_data['behaviour_aligned']
#        dF_ds = loaded_data['dF_aligned']
    
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat['behaviour_aligned'])
    dF_ds = np.copy(h5dat['dF_aligned'])
    h5dat.close()

    behav_ds = make_blackbox_loc_continuous(behav_ds)

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
    if filter is True:
        order = 6
        fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
        cutoff = 3 # desired cutoff frequency of the filter, Hz
        dF_ds[:,roi] = butter_lowpass_filter(dF_ds[:,roi], cutoff, fs, order)

    # filter out rare instances of artifactualy high speed values (should at max be 1 datapoint per recording)
    behav_ds[behav_ds[:,3]>80,3] = 0
    if filter_speed is True:
        order = 6
        fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
        cutoff = 1 # desired cutoff frequency of the filter, Hz
        behav_ds[:,3] = butter_lowpass_filter(behav_ds[:,3], cutoff, fs, order)

    # threshold the response of a roi in a given trial has to exceed count its response toward the tuning of the cell
    trial_std_threshold = 3
    # on which fraction of trials did the roi exceed 3 standard deviations
    roi_active_fraction_short = np.int64(0)
    roi_active_fraction_long = np.int64(0)

    # specify track numbers
    track_short = 3
    track_long = 4



    # create figure and axes to later plot on
    if make_figure:
        fig = plt.figure(figsize=(24,16))
        ax1 = plt.subplot2grid((8,8),(0,0), rowspan=2, colspan=4)
        ax2 = plt.subplot2grid((8,8),(0,4), rowspan=2, colspan=4)
        ax3 = plt.subplot2grid((8,8),(2,0), rowspan=2, colspan=4)
        ax4 = plt.subplot2grid((8,8),(2,4), rowspan=2, colspan=4)
        ax5 = plt.subplot2grid((8,8),(4,0), rowspan=2, colspan=4)
        ax6 = plt.subplot2grid((8,8),(4,4), rowspan=2, colspan=4)
        ax7 = plt.subplot2grid((8,8),(6,0), rowspan=2, colspan=2)
        ax8 = plt.subplot2grid((8,8),(6,2), rowspan=2, colspan=2)

        one_sec = np.round(1/frame_latency,0).astype(int)
        # print(one_sec)
        ax1.set_xticks([window_center,window_center+(2*one_sec)])
        ax1.set_xticklabels([0,2])
        ax2.set_xticks([window_center,window_center+(2*one_sec)])
        ax2.set_xticklabels([0,2])
        ax5.set_xticks([window_center,window_center+(2*one_sec)])
        ax5.set_xticklabels([0,2])
        ax6.set_xticks([window_center,window_center+(2*one_sec)])
        ax6.set_xticklabels([0,2])

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

        ax5.spines['left'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')

        ax6.spines['left'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        ax6.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')
    else:
        ax1 = None
        ax2 = None

    # get indices of desired behavioural event
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    if order_short is not None:
        ordered_trial_list = order_trials(behav_ds, trials_short, order_short)
    # trials_short = filter_trials( behav_ds, [], ['trialnr_range',2,100],trials_short)
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])
    if order_long is not None:
        ordered_trial_list = order_trials(behav_ds, trials_long, order_long)

    events_short = event_ind(behav_ds, eventshort, trials_short)
    events_long = event_ind(behav_ds, eventlong, trials_long)

    # specify alignment type based on filters provided (assuming that short and long have the same filters)
    if eventshort[0] == 'trial_transition':
        align_type = 'trialonset'
    elif eventshort[0] == 'rewards_all':
        align_type = 'reward'
    elif eventshort[0] == 'at_location':
        align_type = 'lmcenter'
    elif eventshort[0] == 'first_licks':
        align_type = 'first_licks'

    # ylims
    # ipdb.set_trace()
    min_y = -0.3
    max_y = 0.0

    # if peak times are provided, split them into short and long
    if len(peak_times) > 0:
        peak_time_short = [peak_times[0]]
        peak_time_long = [peak_times[1]]
    else:
        peak_time_short = []
        peak_time_long = []

    # # calculate correlation between speed and dF/F
    # moving_idx = np.where(behav_ds[:,3] > 3)[0]
    # speed_slope, speed_intercept, lo_slope, up_slope = sp.stats.theilslopes(dF_ds[moving_idx,roi], behav_ds[moving_idx,3])
    # print('--- DF/F:SPEED PEARSONR ---')
    # print(speed_slope, speed_intercept)
    # print(sp.stats.pearsonr(behav_ds[moving_idx,3],dF_ds[moving_idx,roi]))
    # print('---------------------------')

    if make_figure:
        cur_trial_dF_short, transient_max_loc_short, roi_mean_trace_shuffled_short, cur_trial_speed_short, roi_std_short, roi_mean_trace_short, roi_active_fraction_short, roi_meanpeak_short, roi_meanpeak_short_time, roi_meanpeak_short_loc, roi_meanmin_short, peak_zscore_short, peak_zscore_short_individual, zscore_trace_short, max_y = \
            calc_aligned_activity(h5path, behav_ds, dF_ds, roi, 'short', align_type, events_short, max_timewindow_idx, blackbox_idx, t_max, window_center, trial_std_threshold, roi_std, max_y, MIN_ACTIVE_TRIALS, frame_latency, peak_time_short, ax1, ax5, make_figure)

        cur_trial_dF_long, transient_max_loc_long, roi_mean_trace_shuffled_long, cur_trial_speed_long, roi_std_long, roi_mean_trace_long, roi_active_fraction_long, roi_meanpeak_long, roi_meanpeak_long_time, roi_meanpeak_long_loc, roi_meanmin_long, peak_zscore_long, peak_zscore_long_individual, zscore_trace_long, max_y = \
            calc_aligned_activity(h5path, behav_ds, dF_ds, roi, 'long', align_type, events_long, max_timewindow_idx, blackbox_idx, t_max, window_center, trial_std_threshold, roi_std, max_y, MIN_ACTIVE_TRIALS, frame_latency, peak_time_long, ax2, ax6, make_figure)

        # ax7.scatter(behav_ds[moving_idx,3],dF_ds[moving_idx,roi])
        # ax7.plot(behav_ds[moving_idx,3], speed_intercept+speed_slope * behav_ds[moving_idx,3], lw=2,c='r')
        # ax7.set_xlim([-1,90])
    else:
        cur_trial_dF_short, transient_max_loc_short, roi_mean_trace_shuffled_short, cur_trial_speed_short, roi_std_short, roi_mean_trace_short, roi_active_fraction_short, roi_meanpeak_short, roi_meanpeak_short_time, roi_meanpeak_short_loc, roi_meanmin_short, peak_zscore_short, peak_zscore_short_individual, zscore_trace_short, max_y = \
            calc_aligned_activity(h5path, behav_ds, dF_ds, roi, 'short', align_type, events_short, max_timewindow_idx, blackbox_idx, t_max, window_center, trial_std_threshold, roi_std, max_y, MIN_ACTIVE_TRIALS, frame_latency, peak_time_short, None, None, make_figure)

        cur_trial_dF_long, transient_max_loc_long, roi_mean_trace_shuffled_long, cur_trial_speed_long, roi_std_long, roi_mean_trace_long, roi_active_fraction_long, roi_meanpeak_long, roi_meanpeak_long_time, roi_meanpeak_long_loc, roi_meanmin_long, peak_zscore_long, peak_zscore_long_individual, zscore_trace_long, max_y = \
            calc_aligned_activity(h5path, behav_ds, dF_ds, roi, 'long', align_type, events_long, max_timewindow_idx, blackbox_idx, t_max, window_center, trial_std_threshold, roi_std, max_y, MIN_ACTIVE_TRIALS, frame_latency, peak_time_long, None, None, make_figure)

    if ylims != []:
        min_y = ylims[0]
        max_y = ylims[1]

    if ylims != []:
        hmmin = 0
        hmmax = ylims[1]
    else:
        hmmin = 0
        hmmax = max_y


    if make_figure and fformat is not 'svg':
        sns.heatmap(cur_trial_dF_short,cmap='viridis',vmin=hmmin,vmax=hmmax,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax3)
        sns.heatmap(cur_trial_dF_long,cmap='viridis',vmin=hmmin,vmax=hmmax,yticklabels=events_long[:,1].astype('int'),xticklabels=False,ax=ax4)
        sns.heatmap(cur_trial_speed_short,cmap='viridis',vmin=0,vmax=60,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax5)
        sns.heatmap(cur_trial_speed_long,cmap='viridis',vmin=0,vmax=60,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax6)
        ax3.axvline(window_center,c='r',lw=2)
        ax4.axvline(window_center,c='r',lw=2)
        ax5.axvline(window_center,c='r',lw=2)
        ax6.axvline(window_center,c='r',lw=2)

        ax1.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)
        ax2.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])

    mean_amplitude_short = roi_meanpeak_short - roi_meanmin_short
    mean_amplitude_long = roi_meanpeak_long - roi_meanmin_long

    if make_figure:
        if roi_response_validation(roi_active_fraction_short, peak_zscore_short):
            ax1.set_title('Active: ' + str(np.round(roi_active_fraction_short,2)) + ' peak: ' + str(np.round(roi_meanpeak_short,2)) + ' LMI: ' + str(np.round((roi_meanpeak_long-roi_meanpeak_short)/(roi_meanpeak_short+roi_meanpeak_long),3)) + ' peak Z: ' +  str(np.round(peak_zscore_short,2)), fontsize=24, color='g')
        else:
            ax1.set_title('Active: ' + str(np.round(roi_active_fraction_short,2)) + ' peak: ' + str(np.round(roi_meanpeak_short,2)) + ' LMI: ' + str(np.round((roi_meanpeak_long-roi_meanpeak_short)/(roi_meanpeak_short+roi_meanpeak_long),3)) + ' peak Z: ' +  str(np.round(peak_zscore_short,2)), fontsize=24, color='r')

        if roi_response_validation(roi_active_fraction_long, peak_zscore_long):
            ax2.set_title('Active: ' + str(np.round(roi_active_fraction_long,2)) + ' peak: ' + str(np.round(roi_meanpeak_long,2))+ ' peak Z: ' +  str(np.round(peak_zscore_long,2)), fontsize=24, color='g')
        else:
            ax2.set_title('Active: ' + str(np.round(roi_active_fraction_long,2)) + ' peak: ' + str(np.round(roi_meanpeak_long,2))+ ' peak Z: ' +  str(np.round(peak_zscore_long,2)), fontsize=24, color='r')
        # ax1.set_title('Active: ' + str(np.round(roi_active_fraction_short,2)) + ' peak: ' + str(np.round(roi_meanpeak_short,2)) + ' LMI: ' + str(np.round((roi_meanpeak_long-roi_meanpeak_short)/(roi_meanpeak_short+roi_meanpeak_long),3)) + ' peak Z: ' +  str(np.round(peak_zscore_short,2)), fontsize=24, c='g')

        ax3.set_title('dF/F vs time SHORT track - heatmap')
        ax4.set_title('dF/F vs time LONG track - heatmap')

        ax1.set_ylim([min_y,max_y])
        ax2.set_ylim([min_y,max_y])
        ax1.set_xlim([0,t_max])
        ax2.set_xlim([0,t_max])
        ax5.set_xlim([0,t_max])
        ax6.set_xlim([0,t_max])

        fig.tight_layout()
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

        plt.close(fig)
        print(fname)

    norm_value = np.amax(dF_ds[:,roi])

    # print(type(roi_std_short),type(roi_std_long), type(roi_active_fraction_short), type(roi_active_fraction_long), type(roi_meanpeak_short), type(roi_meanpeak_long), type(roi_meanpeak_short_time), type(roi_meanpeak_long_time), [np.float32(min_y).item(),np.float32(min_y)], type(norm_value.item()))
    # print(roi_active_fraction_short,roi_active_fraction_long)
    return roi_std_short.item(),roi_std_long.item(), roi_mean_trace_short.tolist(), roi_mean_trace_long.tolist(), \
           transient_max_loc_short.tolist(), transient_max_loc_long.tolist(), \
           roi_active_fraction_short.item(), roi_active_fraction_long.item(), roi_meanpeak_short.item(), roi_meanpeak_long.item(), \
           roi_meanpeak_short_time.item(), roi_meanpeak_long_time.item(), roi_meanpeak_short_loc.item(), roi_meanpeak_long_loc.item(), \
           peak_zscore_short.item(), peak_zscore_long.item(), \
           peak_zscore_short_individual.item(), peak_zscore_long_individual.item(), \
           zscore_trace_short.tolist(), zscore_trace_long.tolist(), \
           [np.float32(min_y).item(),np.float32(max_y).item()], norm_value.item()

def run_analysis(mousename, sessionname, sessionname_openloop, roi_selection, h5_filepath, json_path, subname, sess_subfolder, filterprop_short, filterprop_long, even_win, blackbox_win, session_rois, raw_data=False):
    """ set up function call and dictionary to collect results """

    MOUSE = mousename
    SESSION = sessionname
    SESSION_OPENLOOP = sessionname_openloop
    h5path = h5_filepath
    SUBNAME = subname
    subfolder = sess_subfolder
    if not raw_data:
        subfolder_ol = sess_subfolder + '_openloop'
    else:
        subfolder_ol = sess_subfolder + '_ol'
    # write result to dictionary
    write_to_dict = False
    # create figure (WARNING: significantly slows down execution)
    make_figure = True
    # set up dictionary to hold result parameters from roi
    session_rois[SUBNAME+'_roi_number'] = []
    session_rois[SUBNAME+'_roi_number_ol'] = []
    session_rois[SUBNAME+'_std_short'] = []
    session_rois[SUBNAME+'_std_long'] = []
    session_rois[SUBNAME+'_mean_trace_short'] = []
    session_rois[SUBNAME+'_mean_trace_long'] = []
    session_rois[SUBNAME+'_transient_max_loc_short'] = []
    session_rois[SUBNAME+'_transient_max_loc_long'] = []
    session_rois[SUBNAME+'_active_short'] = []
    session_rois[SUBNAME+'_active_long'] = []
    session_rois[SUBNAME+'_peak_short'] = []
    session_rois[SUBNAME+'_peak_long'] = []
    session_rois[SUBNAME+'_peak_zscore_short'] = []
    session_rois[SUBNAME+'_peak_zscore_long'] = []
    session_rois[SUBNAME+'_zscore_trace_short'] = []
    session_rois[SUBNAME+'_zscore_trace_long'] = []
    session_rois[SUBNAME+'_peak_time_short'] = []
    session_rois[SUBNAME+'_peak_time_long'] = []
    session_rois[SUBNAME+'_peak_loc_short'] = []
    session_rois[SUBNAME+'_peak_loc_long'] = []
    session_rois[SUBNAME+'_peak_loc_short_ol'] = []
    session_rois[SUBNAME+'_peak_loc_long_ol'] = []
    session_rois[SUBNAME+'_std_short_ol'] = []
    session_rois[SUBNAME+'_std_long_ol'] = []
    session_rois[SUBNAME+'_mean_trace_short_ol'] = []
    session_rois[SUBNAME+'_mean_trace_long_ol'] = []
    session_rois[SUBNAME+'_transient_max_loc_short_ol'] = []
    session_rois[SUBNAME+'_transient_max_loc_long_ol'] = []
    session_rois[SUBNAME+'_active_short_ol'] = []
    session_rois[SUBNAME+'_active_long_ol'] = []
    session_rois[SUBNAME+'_peak_short_ol'] = []
    session_rois[SUBNAME+'_peak_long_ol'] = []
    session_rois[SUBNAME+'_peak_zscore_short_ol'] = []
    session_rois[SUBNAME+'_peak_zscore_long_ol'] = []
    session_rois[SUBNAME+'_zscore_trace_short_ol'] = []
    session_rois[SUBNAME+'_zscore_trace_long_ol'] = []
    session_rois[SUBNAME+'_peak_time_short_ol'] = []
    session_rois[SUBNAME+'_peak_time_long_ol'] = []
    session_rois['norm_value'] = []
    session_rois['norm_value_ol'] = []
    session_rois['norm_zscore_value_short'] = []
    session_rois['norm_zscore_value_long'] = []
    session_rois['norm_zscore_value_short_ol'] = []
    session_rois['norm_zscore_value_long_ol'] = []

    # if we want to run through all the rois, just say all
    if roi_selection == 'all':
        if not raw_data:
            h5dat = h5py.File(h5path, 'r')
            dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
            h5dat.close()
            num_rois = dF_ds.shape[1]
            
            df_signal_path = h5_filepath + os.sep + sessionname + os.sep + use_data
            dF_signal_bef =  h5py.File(df_signal_path, 'r')
            dF_aligned = np.copy(dF_signal_bef['dF_aligned'])
            behaviour_aligned = np.copy(dF_signal_bef['behaviour_aligned'])
            roilist = range(len(dF_aligned[0]))
#            roilist = [9]
            dF_signal_bef.close()            
        else:
            loaded_data = sio.loadmat(h5path)
            dF_ds = loaded_data['dF_aligned']
            num_rois = dF_ds.shape[1]

        roilist = np.arange(num_rois).tolist()
        print('analysing ' + roi_selection + ' rois: ' + str(roilist))
    elif roi_selection == 'valid':
        # only use valid rois
        with open(json_path, 'r') as f:
            sess_dict = json.load(f)
        roilist = sess_dict['valid_rois']
        print('analysing ' + roi_selection + ' rois: ' + str(roilist))
    else:
        roilist = roi_selection
        print('analysing custom list of rois: ' + str(roilist))

    # run analysis for vr session
    if raw_data:
        data_paths = h5path
        h5path = data_paths[0]
        h5path_ol = data_paths[1]
    else:
        h5path_ol = h5path
    for i,r in enumerate(roilist):
        print(MOUSE + ' ' + SESSION + ' ' + SUBNAME + ': ' + str(r))
        # VR

        std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, transient_max_loc_short, transient_max_loc_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, roi_meanpeak_short_loc, roi_meanpeak_long_loc, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long, ylims, norm_value = \
            fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, None, None, even_win, blackbox_win, [], fformat, subfolder, [], False, False, make_figure, raw_data)
        session_rois[SUBNAME+'_roi_number'].append(r)
        session_rois[SUBNAME+'_std_short'].append(std_short)
        session_rois[SUBNAME+'_std_long'].append(std_long)
        session_rois[SUBNAME+'_mean_trace_short'].append(roi_mean_trace_short)
        session_rois[SUBNAME+'_mean_trace_long'].append(roi_mean_trace_long)
        session_rois[SUBNAME+'_transient_max_loc_short'].append(transient_max_loc_short)
        session_rois[SUBNAME+'_transient_max_loc_long'].append(transient_max_loc_long)
        session_rois[SUBNAME+'_active_short'].append(active_short)
        session_rois[SUBNAME+'_active_long'].append(active_long)
        session_rois[SUBNAME+'_peak_short'].append(peak_short)
        session_rois[SUBNAME+'_peak_long'].append(peak_long)
        session_rois[SUBNAME+'_peak_zscore_short'].append(peak_zscore_short)
        session_rois[SUBNAME+'_peak_zscore_long'].append(peak_zscore_long)
        session_rois[SUBNAME+'_peak_time_short'].append(meanpeak_short_time)
        session_rois[SUBNAME+'_peak_time_long'].append(meanpeak_long_time)
        session_rois[SUBNAME+'_peak_loc_short'].append(roi_meanpeak_short_loc)
        session_rois[SUBNAME+'_peak_loc_long'].append(roi_meanpeak_long_loc)
        session_rois['norm_value'].append(norm_value)
        session_rois['norm_zscore_value_short'].append(peak_zscore_short_individual)
        session_rois['norm_zscore_value_long'].append(peak_zscore_long_individual)
        session_rois[SUBNAME+'_zscore_trace_short'].append(zscore_trace_short)
        session_rois[SUBNAME+'_zscore_trace_long'].append(zscore_trace_long)
        # OPENLOOP
        if sessionname_openloop is not '' and SUBNAME is not 'firstlick':
            std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, transient_max_loc_short, transient_max_loc_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, roi_meanpeak_short_loc, roi_meanpeak_long_loc, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long, _, norm_value = \
                fig_landmark_centered(h5path_ol, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, None, None, even_win, blackbox_win, ylims, fformat, subfolder_ol, [session_rois[SUBNAME+'_peak_time_short'][i], session_rois[SUBNAME+'_peak_time_long'][i]], False, False, make_figure, raw_data)
            session_rois[SUBNAME+'_roi_number_ol'].append(r)
            session_rois[SUBNAME+'_std_short_ol'].append(std_short)
            session_rois[SUBNAME+'_std_long_ol'].append(std_long)
            session_rois[SUBNAME+'_mean_trace_short_ol'].append(roi_mean_trace_short)
            session_rois[SUBNAME+'_mean_trace_long_ol'].append(roi_mean_trace_long)
            session_rois[SUBNAME+'_transient_max_loc_short_ol'].append(transient_max_loc_short)
            session_rois[SUBNAME+'_transient_max_loc_long_ol'].append(transient_max_loc_long)
            session_rois[SUBNAME+'_active_short_ol'].append(active_short)
            session_rois[SUBNAME+'_active_long_ol'].append(active_long)
            session_rois[SUBNAME+'_peak_short_ol'].append(peak_short)
            session_rois[SUBNAME+'_peak_long_ol'].append(peak_long)
            session_rois[SUBNAME+'_peak_zscore_short_ol'].append(peak_zscore_short)
            session_rois[SUBNAME+'_peak_zscore_long_ol'].append(peak_zscore_long)
            session_rois[SUBNAME+'_peak_time_short_ol'].append(meanpeak_short_time)
            session_rois[SUBNAME+'_peak_time_long_ol'].append(meanpeak_long_time)
            session_rois[SUBNAME+'_peak_loc_short_ol'].append(roi_meanpeak_short_loc)
            session_rois[SUBNAME+'_peak_loc_long_ol'].append(roi_meanpeak_long_loc)
            session_rois['norm_value_ol'].append(norm_value)
            session_rois['norm_zscore_value_short_ol'].append(peak_zscore_short_individual)
            session_rois['norm_zscore_value_long_ol'].append(peak_zscore_long_individual)
            session_rois[SUBNAME+'_zscore_trace_short_ol'].append(zscore_trace_short)
            session_rois[SUBNAME+'_zscore_trace_long_ol'].append(zscore_trace_long)


    if write_to_dict:
        # ipdb.set_trace()
        print('writing to dict')
        # print(session_rois)
        write_dict(MOUSE, SESSION, session_rois)


    # return session_rois

def run_slowfast_analysis(mousename, sessionname, sessionname_openloop, roi_selection, h5_filepath, json_path, subname, sess_subfolder, filterprop_short, filterprop_long, even_win, blackbox_win, session_rois):
    """ set up function call and dictionary to collect results """
    MOUSE = mousename
    SESSION = sessionname
    h5path = h5_filepath
    SUBNAME = subname
    subfolder = sess_subfolder
    # write result to dictionary
    write_to_dict = False
    # create figure (WARNING: significantly slows down execution)
    make_figure = False
    orderprop_short = ['time_between_points','lmcenter','reward']
    orderprop_long = ['time_between_points','lmcenter','reward']

    # set up dictionary to hold result parameters from roi
    session_rois[SUBNAME+'_slow_short_mean'] = []
    session_rois[SUBNAME+'_fast_long_mean'] = []

    session_rois[SUBNAME+'_slow_std_short'] = []
    session_rois[SUBNAME+'_slow_std_long'] = []
    session_rois[SUBNAME+'_fast_std_short'] = []
    session_rois[SUBNAME+'_fast_std_long'] = []

    session_rois[SUBNAME+'_slow_active_short'] = []
    session_rois[SUBNAME+'_slow_active_long'] = []
    session_rois[SUBNAME+'_fast_active_short'] = []
    session_rois[SUBNAME+'_fast_active_long'] = []

    session_rois[SUBNAME+'_slow_peak_short'] = []
    session_rois[SUBNAME+'_slow_peak_long'] = []
    session_rois[SUBNAME+'_fast_peak_short'] = []
    session_rois[SUBNAME+'_fast_peak_long'] = []

    session_rois[SUBNAME+'_slow_peak_time_short'] = []
    session_rois[SUBNAME+'_slow_peak_time_long'] = []
    session_rois[SUBNAME+'_fast_peak_time_short'] = []
    session_rois[SUBNAME+'_fast_peak_time_long'] = []

    # if we want to run through all the rois, just say all
    if roi_selection == 'all':
        h5dat = h5py.File(h5path, 'r')
        dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
        h5dat.close()
        num_rois = dF_ds.shape[1]
        roilist = np.arange(num_rois)
        print('analysing ' + roi_selection + ' rois: ' + str(roilist))
    elif roi_selection == 'valid':
        # only use valid rois
        with open(json_path, 'r') as f:
            sess_dict = json.load(f)
        roilist = sess_dict['valid_rois']
        print('analysing ' + roi_selection + ' rois: ' + str(roilist))
    else:
        roilist = roi_selection
        print('analysing custom list of rois: ' + str(roilist))

    # run analysis for vr session
    for i,r in enumerate(roilist):
        print(SUBNAME + ': ' + str(r))
        std_short, std_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, ylims, norm_value = fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, orderprop_short,orderprop_long, even_win, blackbox_win, [], fformat, subfolder, [], False, False, make_figure)
        # session_rois[SUBNAME+'_slow_roi_number'].append(r)
        session_rois[SUBNAME+'_slow_std_short'].append(std_short)
        session_rois[SUBNAME+'_slow_std_long'].append(std_long)
        session_rois[SUBNAME+'_slow_active_short'].append(active_short)
        session_rois[SUBNAME+'_slow_active_long'].append(active_long)
        session_rois[SUBNAME+'_slow_peak_short'].append(peak_short)
        session_rois[SUBNAME+'_slow_peak_long'].append(peak_long)
        session_rois[SUBNAME+'_slow_peak_time_short'].append(meanpeak_short_time)
        session_rois[SUBNAME+'_slow_peak_time_long'].append(meanpeak_long_time)

    if write_to_dict:
        # print('writing to dict')
        write_dict(MOUSE, SESSION, session_rois)

def run_LF161202_1_Day20170209_l23():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l23'
    SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : -1
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF161202_1_Day20170209_l5():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l5'
    SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : -1
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')


def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = [9] #'valid' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')


def run_LF170110_2_Day201748_1():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    SESSION_OPENLOOP = 'Day201748_openloop_1'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #152
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170110_2_Day201748_2():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_2'
    SESSION_OPENLOOP = 'Day201748_openloop_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #171
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170110_2_Day201748_3():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_3'
    SESSION_OPENLOOP = 'Day201748_openloop_3'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #50
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170421_2_Day2017719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = 'Day2017719_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #96
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170421_2_Day20170719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170719'
    SESSION_OPENLOOP = 'Day20170719_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #123
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170421_2_Day2017720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017720'
    SESSION_OPENLOOP = SESSION + '_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # this session is a special case as we have fewer ROIs in openloop
    #roi_selection = 'valid' #63
    with open(json_path, 'r') as f:
        sess_dict = json.load(f)
    roi_selection = np.intersect1d(np.arange(63),sess_dict['valid_rois']).tolist()

    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170420_1_Day201783():
    MOUSE = 'LF170420_1'
    SESSION = 'Day201783'
    SESSION_OPENLOOP = SESSION + '_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #81
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170420_1_Day2017719():
    MOUSE = 'LF170420_1'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = SESSION + '_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #91
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')


def run_LF170222_1_Day201776():
    MOUSE = 'LF170222_1'
    SESSION = 'Day201776'
    SESSION_OPENLOOP = SESSION + '_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #120
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170222_1_Day2017615():
    MOUSE = 'LF170222_1'
    SESSION = 'Day2017615'
    SESSION_OPENLOOP = SESSION + '_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #120
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')


def run_LF170110_2_Day2017331():
    MOUSE = 'LF170110_2'
    SESSION = 'Day2017331'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 184
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    # SUBNAME = 'lmoff'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 240], ['at_location', 240], [10,10], [2,2], roi_result_params)
    #
    # SUBNAME = 'lmon'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 200], ['at_location', 200], [10,10], [2,2], roi_result_params)
    #
    # SUBNAME = 'reward'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
    #
    # SUBNAME = 'trialonset'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)



def run_LF171211_1_Day2018321_2():
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    SESSION_OPENLOOP = 'Day2018321_openloop_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #170
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF171212_2_Day2018218_1():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_1'
    SESSION_OPENLOOP = 'Day2018218_openloop_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #335
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')


def run_LF171212_2_Day2018218_2():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_2'
    SESSION_OPENLOOP = 'Day2018218_openloop_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #335
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF180119_1_Day2018424_2():
    MOUSE = 'LF180119_1'
    SESSION = 'Day2018424_2'
    SESSION_OPENLOOP = 'Day2018424_openloop_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170214_1_Day201777():
    MOUSE = 'LF170214_1'
    SESSION = 'Day201777'
    SESSION_OPENLOOP = 'Day201777_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF170214_1_Day2017714():
    MOUSE = 'LF170214_1'
    SESSION = 'Day2017714'
    SESSION_OPENLOOP = 'Day2017714_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF171211_2_Day201852():
    MOUSE = 'LF171211_2'
    SESSION = 'Day201852'
    SESSION_OPENLOOP ='Day201852_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF180219_1_Day2018424_0025():
    MOUSE = 'LF180219_1'
    SESSION = 'Day2018424_0025'
    SESSION_OPENLOOP ='Day2018424_openloop_0025'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF180112_2_Day2018424_1():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_1'
    SESSION_OPENLOOP = 'Day2018424_openloop_1'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF180112_2_Day2018424_2():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_2'
    SESSION_OPENLOOP = 'Day2018424_openloop_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF18112_2_Day2018322_2():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018322_2'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'valid'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_LF18112_2_Day2018322_1():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018322_1'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 94
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
        print(MOUSE + ' ' + SESSION + ' first licks done.')


def run_LF170613_1_Day201784():
    MOUSE = 'LF170613_1'
    SESSION = 'Day201784'
    SESSION_OPENLOOP = 'Day201784_openloop'
    NUM_ROIS = 77
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    # SUBNAME = 'lmoff'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 240], ['at_location', 240], [10,10], [2,2], roi_result_params)
    #
    # SUBNAME = 'lmon'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 200], ['at_location', 200], [10,10], [2,2], roi_result_params)

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 320], ['at_location', 380], [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params)

    # SUBNAME = 'lmcenter'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params)

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params)
    print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191022_2_20191116():
    MOUSE = 'LF191022_2'
    SESSION = '20191116'
    SESSION_OPENLOOP = '20191116_ol'
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191022_3_20191119():
    MOUSE = 'LF191022_3'
    SESSION = '20191119'
    SESSION_OPENLOOP = ''
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191022_3_20191204():
    MOUSE = 'LF191022_3'
    SESSION = '20191204'
    SESSION_OPENLOOP = ''
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191024_1_20191115():
    MOUSE = 'LF191024_1'
    SESSION = '20191115'
    SESSION_OPENLOOP = ''
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191024_1_20191204():
    MOUSE = 'LF191024_1'
    SESSION = '20191204'
    SESSION_OPENLOOP = ''
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191023_blank_20191116():
    MOUSE = 'LF191023_blank'
    SESSION = '20191116'
    SESSION_OPENLOOP = '20191116_ol'
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191023_blue_20191119():
    MOUSE = 'LF191023_blue'
    SESSION = '20191119'
    SESSION_OPENLOOP = '20191119_ol'
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = [16] #'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191023_blue_20191204():
    MOUSE = 'LF191023_blue'
    SESSION = '20191204'
    SESSION_OPENLOOP = ''
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = [10] #'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'aligned_data.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' first licks done.')

def run_20191022_3_2019113():
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    SESSION_OPENLOOP = ''
    # SESSION_OPENLOOP = ''
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = [9] #'valid' #105
    raw_file = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'M01_000_000_results.mat'
    raw_file_ol = loc_info['raw_dir'] + MOUSE + os.sep + SESSION_OPENLOOP + os.sep + 'M01_000_000.mat'
    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }
    
    make_firstlick = True;

    if make_reward:
        SUBNAME = 'reward'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' reward done.')

    if make_lmcenter:
        SUBNAME = 'lmcenter'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' lmcenter done.')

    if make_trialonset:
        SUBNAME = 'trialonset'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, [raw_file, raw_file_ol], json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2], roi_result_params, True)
        print(MOUSE + ' ' + SESSION + ' trialonset done.')

    if make_firstlick:
        SUBNAME = 'firstlick'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, raw_file, json_path, SUBNAME, subfolder, ['first_licks'], ['first_licks'], [10,10], [2,2], roi_result_params, False)
        print(MOUSE + ' ' + SESSION + ' first licks done.')


def do_single():

    # MOUSE = 'LF170421_2'
    # SESSION = 'Day20170719'
    # SESSION_OPENLOOP = 'Day20170719_openloop'
    # roi_selection = 10
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # subfolder = ''
    # subfolder_ol = ''
    # fformat = 'png'
    #
    # roi_result_params = {
    #     'mouse_session' : MOUSE+'_'+SESSION,
    #     'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    # }
    #
    # fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,2], fformat, subfolder, [], False, True)
    # fig_landmark_centered(h5path, SESSION_OPENLOOP, roi_selection, MOUSE+'_'+SESSION_OPENLOOP+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,2], fformat, subfolder, [], False, True)

    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    # SESSION_OPENLOOP = 'Day20170804_openloop'
    # roi_selection = 17
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # filterprop_short = ['rewards_all', -1]
    # filterprop_long = ['rewards_all', -1]
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [20,0]
    # blackbox_win = [2,2]
    # SUBNAME = 'reward'
    # subfolder = ''
    # subfolder_ol = ''
    # fformat = 'png'
    #
    # roi_result_params = {
    #     'mouse_session' : MOUSE+'_'+SESSION,
    #     'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    # }
    #
    # # run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, filterprop_short, filterprop_long, [10,10], [2,2], roi_result_params)
    # # for r in NUM_ROIS:
    # fig_landmark_centered(h5path, SESSION_OPENLOOP, roi_selection, MOUSE+'_'+SESSION_OPENLOOP+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,5], fformat, subfolder, [], False, True)
    # fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,5], fformat, subfolder, [], False, True)

    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_2'
    # SESSION_OPENLOOP = 'Day201748_openloop_2'
    # roi_selection = 103
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # fformat = 'png'
    # subfolder = 'testplots'
    # std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, ylims, norm_value = \
    #     fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3], fformat, subfolder, [], False, True)
    # _,_,roi_mean_trace_short_ol, roi_mean_trace_long_ol, _,_,_,_,_,_,_,_,_,_,_,_, = fig_landmark_centered(h5path, SESSION_OPENLOOP, roi_selection, MOUSE+'_'+SESSION_OPENLOOP+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3], fformat, subfolder, [meanpeak_short_time, meanpeak_long_time], False, True)

    # print(np.amax(roi_mean_trace_long))
    # print(np.amax(roi_mean_trace_long_ol))

    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    roi_selection = 9
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    filterprop_short = ['at_location', 220]
    filterprop_long =  ['at_location', 220]
    orderprop_short = None
    orderprop_long = None
    even_win = [10,10]
    blackbox_win = [2,2]
    SUBNAME = 'lmcenter'
    subfolder = 'testplots'
    subfolder_ol = ''


    std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, transient_max_loc_short, transient_max_loc_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, roi_meanpeak_short_loc, roi_meanpeak_long_loc, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long, ylims, norm_value = \
        fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3], fformat, subfolder, [], False, False, True)

    roi_selection = 21
    std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, transient_max_loc_short, transient_max_loc_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, roi_meanpeak_short_loc, roi_meanpeak_long_loc, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long, ylims, norm_value = \
        fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3], fformat, subfolder, [], False, False, True)
    #
    # fig_landmark_centered(h5path, SESSION_OPENLOOP, roi_selection, MOUSE+'_'+SESSION_OPENLOOP+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3], fformat, subfolder, [meanpeak_short_time, meanpeak_long_time], False, False, True)

    # roi_selection = 21
    # std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long,  ylims, norm_value = \
    #     fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3], fformat, subfolder, [], False, False, True)

    # MOUSE = 'LF171211_2'
    # SESSION = 'Day201852'
    # SESSION_OPENLOOP = 'Day201852_openloop'
    # roi_selection = 36
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # filterprop_short = ['trial_transition']
    # filterprop_long = ['trial_transition']
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [0,20]
    # blackbox_win = [2,2]
    # SUBNAME = 'trialonset'
    # subfolder = 'testplots'
    # subfolder_ol = ''
    # fformat = 'png'
    #
    #
    # std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long,  ylims, norm_value = \
    #     fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.2,3], fformat, subfolder, [], True, False, True)
    # fig_landmark_centered(h5path, SESSION_OPENLOOP, roi_selection, MOUSE+'_'+SESSION_OPENLOOP+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.2,3], fformat, subfolder, [meanpeak_short_time, meanpeak_long_time], True, False, True)

    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # SESSION_OPENLOOP = 'Day201777_openloop'
    # roi_selection = 66
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # subfolder = 'testplots'
    # subfolder_ol = ''
    # fformat = 'png'
    # #
    # #
    # std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long,  ylims, norm_value = \
    #     fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.05,0.2], fformat, subfolder, [], True, False, True)
    # fig_landmark_centered(h5path, SESSION_OPENLOOP, roi_selection, MOUSE+'_'+SESSION_OPENLOOP+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.05,0.2], fformat, subfolder, [meanpeak_short_time, meanpeak_long_time], True, False, True)

    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_1'
    # SESSION_OPENLOOP = 'Day201748_openloop_1'
    # roi_selection = 74
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # subfolder = 'testplots'
    # std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long, ylims, norm_value = \
    #     fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+SUBNAME+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3.5], fformat, subfolder, [], False, True)
    #
    # filterprop_short = ['trial_transition']
    # filterprop_long = ['trial_transition']
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [0,20]
    # blackbox_win = [2,2]
    # SUBNAME = 'trialonset'
    #
    # std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long, ylims, norm_value = \
    #     fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+SUBNAME+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3.5], fformat, subfolder, [], False, True)
    #
    # filterprop_short = ['rewards_all', -1]
    # filterprop_long = ['rewards_all', -1]
    # orderprop_short = None
    # orderprop_long = None
    # even_win = [20,0]
    # blackbox_win = [2,2]
    # SUBNAME = 'reward'
    #
    # std_short, std_long, roi_mean_trace_short, roi_mean_trace_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, peak_zscore_short, peak_zscore_long, peak_zscore_short_individual, peak_zscore_long_individual, zscore_trace_short, zscore_trace_long, ylims, norm_value = \
    #     fig_landmark_centered(h5path, SESSION, roi_selection, MOUSE+'_'+SESSION+SUBNAME+'_roi_'+str(roi_selection), filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win,[-0.3,3.5], fformat, subfolder, [], False, True)



if __name__ == '__main__':
    
    run_20191022_3_2019113()
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inlin
    #

    # run_20191022_3_20191119()
    # run_20191023_blank_20191116()

    # run_20191022_2_20191116,\
    # run_20191023_blank_20191116

    # run_20191023_blue_20191119()
#    run_20191023_blue_20191204()
#
#    flist = [
#            run_20191022_3_20191119, \
#             run_20191022_3_20191204, \
#             run_20191023_blue_20191119,\
#             run_20191023_blue_20191204, \
#             run_20191024_1_20191115, \
#             run_20191024_1_20191204
#            ]

    # V1 SESSIONS

    # run_LF170214_1_Day201777()
    # run_LF170214_1_Day2017714()
    # run_LF171211_2_Day201852()
    # run_LF180219_1_Day2018424_0025()
    # run_LF18112_2_Day2018322_1()
    # run_LF18112_2_Day2018322_2()

    # run_LF180112_2_Day2018424_1()
    # run_LF180112_2_Day2018424_2()

    # run_LF180119_1_Day2018424_2()

    # run_LF171211_2_Day201852()
    # flist = [run_LF180112_2_Day2018424_1,
    #          run_LF180112_2_Day2018424_2,
    #          run_LF171211_2_Day201852,
    #          run_LF170214_1_Day201777,
    #          run_LF170214_1_Day2017714
    #         ]

    # run_LF170613_1_Day20170804()
    # run_LF170420_1_Day2017719()
    # run_LF170421_2_Day2017720()
    # RSC SESSIONS
    # flist = [run_LF170613_1_Day20170804, \
    #          run_LF170421_2_Day2017720, \
    #          run_LF170421_2_Day20170719, \
    #          run_LF170110_2_Day201748_1, \
    #          run_LF170110_2_Day201748_2, \
    #          run_LF170110_2_Day201748_3, \
    #          run_LF170420_1_Day2017719, \
    #          run_LF170420_1_Day201783, \
    #          run_LF170222_1_Day201776, \
    #          run_LF171212_2_Day2018218_2, \
    #          run_LF161202_1_Day20170209_l23, \
    #          run_LF161202_1_Day20170209_l5, \
    #          run_LF180112_2_Day2018424_1,\
    #          run_LF180112_2_Day2018424_2, \
    #          run_LF171211_2_Day201852, \
    #          run_LF170214_1_Day201777, \
    #          run_LF170214_1_Day2017714
    #          ]
    # run_LF161202_1_Day20170209_l23()
    # flist = [run_LF161202_1_Day20170209_l23, \
    #          run_LF161202_1_Day20170209_l5 ]
    #
    # jobs = []
    # for fl in flist:
    #     p = Process(target=fl)
    #     jobs.append(p)
    #     p.start()
    #     #
    # for j in jobs:
    #     j.join()

    # p = Process(target=flist[0])
    # p2 = Process(target=flist[1])
    # p3 = Process(target=flist[2])
    # p.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # p8.start()
    # run_LF170421_2_Day2017720()
    # run_LF170421_2_Day20170719()
    # run_LF170110_2_Day201748_1()
    # run_LF170110_2_Day201748_2()
    # run_LF170110_2_Day201748_3()
    # run_LF170420_1_Day2017719()
    # run_LF170420_1_Day201783()
    # run_LF170222_1_Day201776()
    # run_LF171211_1_Day2018321_2()
    # run_LF171212_2_Day2018218_1()
    # run_LF171212_2_Day2018218_2()
    # run_LF170222_1_Day2017615()

    # run_LF170421_2_Day2017719()

    # run_LF171211_1_Day2018321_2()

    # run_LF170222_1_Day201776()
    # # run_LF170110_2_Day2017331()
    # # run_LF170613_1_Day201784()

    # run_LF171212_2_Day2018218_1()
    # run_LF171212_2_Day2018218_2()



    # do_single()
