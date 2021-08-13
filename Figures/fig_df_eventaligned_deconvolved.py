"""
Plot trace of an individual ROI centered around a given location

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
from scipy.signal import butter, filtfilt
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
from filter_trials import filter_trials
from scipy import stats
from scipy import signal
from write_dict import write_dict

fformat = 'png'

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def fig_landmark_centered(h5path, sess, roi, fname, eventshort, eventlong, max_timewindow, blackbox_time, ylims=[], fformat='png', subfolder=[], peak_times=[], filter=False, make_figure=True):
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/spiketrain'])
    spikerate = np.copy(h5dat[sess + '/spikerate'])
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
    roi_std = np.std(spikerate[:,roi])
    if filter==True:
        order = 6
        fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
        cutoff = 5 # desired cutoff frequency of the filter, Hz
        spikerate[:,roi] = butter_lowpass_filter(spikerate[:,roi], cutoff, fs, order)
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
    if make_figure:
        fig = plt.figure(figsize=(24,12))
        ax1 = plt.subplot(323)
        ax2 = plt.subplot(324)
        ax3 = plt.subplot(321)
        ax4 = plt.subplot(322)
        ax5 = plt.subplot(325)
        ax6 = plt.subplot(326)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params( \
            reset='on',
            axis='both', \
            direction='in', \
            length=4, \
            bottom='off', \
            right='off', \
            top='off')

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params( \
            reset='on',
            axis='both', \
            direction='in', \
            length=4, \
            bottom='off', \
            right='off', \
            top='off')

        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.tick_params( \
            reset='on',
            axis='both', \
            direction='in', \
            length=4, \
            bottom='off', \
            right='off', \
            top='off')

        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.tick_params( \
            reset='on',
            axis='both', \
            direction='in', \
            length=4, \
            bottom='off', \
            right='off', \
            top='off')

    # get indices of desired behavioural event
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])

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

    trial_dF_short = trial_dF_short.astype(int)
    # grab dF data for each trial
    cur_trial_dF_short = np.full((np.size(events_short[:,0]),int(t_max)),np.nan)
    cur_trial_spikerate_short = np.full((np.size(events_short[:,0]),int(t_max)),np.nan)
    cur_trial_speed_short = np.full((np.size(events_short[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(events_short[:,0]))
    cur_trial_max_idx_short = np.empty(0)
    for i in range(np.size(trial_dF_short,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF_short[i,0]:trial_dF_short[i,1],roi]
        cur_sweep_spikerate = spikerate[trial_dF_short[i,0]:trial_dF_short[i,1],roi]
        cur_trial_event_idx[i] = events_short[i,0] - trial_dF_short[i,0]
        trace_start = int(window_center - cur_trial_event_idx[i])
        cur_trial_dF_short[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        cur_trial_spikerate_short[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep_spikerate
        cur_trial_speed_short[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF_short[i,0]:trial_dF_short[i,1],3]
        # only consider roi's max dF value in a given trial if it exceeds threshold
        if np.amax(cur_sweep_spikerate) > trial_std_threshold * roi_std:
            cur_trial_max_idx_short = np.append(cur_trial_max_idx_short,np.nanargmax(cur_trial_spikerate_short[i,:]))
        if np.amax(cur_sweep_spikerate) > max_y:
            max_y = np.amax(cur_sweep_spikerate)

    # plot individual traces
    if make_figure:
        for i,ct in enumerate(cur_trial_spikerate_short):
            ax1.plot(ct,c='0.65',lw=1)
    # ax1_1 = ax1.twinx()
    if len(cur_trial_max_idx_short) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction_short = len(cur_trial_max_idx_short)/np.size(trial_dF_short,0)
        if make_figure:
            sns.distplot(cur_trial_max_idx_short,hist=False,kde=False,rug=True,ax=ax1)
        roi_std_short = np.std(cur_trial_max_idx_short)
    else:
        roi_active_fraction_short = -1
        roi_std_short = -1

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    roi_meanpeak_short = -1
    roi_meanmin_short = -1
    roi_meanpeak_short_idx = -1
    roi_meanpeak_short_time = -1
    mean_amplitude_short = -1
    for i,trace in enumerate(cur_trial_spikerate_short.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < 0.5:
            mean_valid_indices.append(i)
    if len(mean_valid_indices) > 0:
        if make_figure:
            ax1.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),np.nanmean(cur_trial_spikerate_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
            ax1.axvline(window_center,c='r',lw=2)
        roi_meanpeak_short = np.nanmax(np.nanmean(cur_trial_spikerate_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
        roi_meanmin_short = np.nanmin(np.nanmean(cur_trial_spikerate_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
        roi_meanpeak_short_idx = np.nanargmax(np.nanmean(cur_trial_spikerate_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
        roi_meanpeak_short_time = ((roi_meanpeak_short_idx+mean_valid_indices[0])-window_center) * frame_latency

        if len(peak_times) > 0:
            window_center_time = window_center * frame_latency
            vr_peak_time_short = peak_times[0] + window_center_time
            vr_peak_time_short_idx = (vr_peak_time_short/frame_latency).astype('int')
            roi_meanpeak_short = np.nanmean(cur_trial_spikerate_short,0)[vr_peak_time_short_idx]
            if make_figure:
                ax1.axvline(vr_peak_time_short_idx)
        else:
            if make_figure:
                ax1.axvline((roi_meanpeak_short_idx+mean_valid_indices[0]))
        mean_amplitude_short = roi_meanpeak_short - roi_meanmin_short


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

    trial_dF_long = trial_dF_long.astype(int)
    # grab dF data for each trial
    cur_trial_dF_long = np.full((np.size(events_long[:,0]),int(t_max)),np.nan)
    cur_trial_spikerate_long = np.full((np.size(events_long[:,0]),int(t_max)),np.nan)
    cur_trial_speed_long = np.full((np.size(events_long[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(events_long[:,0]))
    cur_trial_max_idx_long = np.empty(0)
    for i in range(np.size(trial_dF_long,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF_long[i,0]:trial_dF_long[i,1],roi]
        cur_sweep_spikerate = spikerate[trial_dF_long[i,0]:trial_dF_long[i,1],roi]
        cur_trial_event_idx[i] = events_long[i,0] - trial_dF_long[i,0]
        trace_start = int(window_center - cur_trial_event_idx[i])
        cur_trial_dF_long[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        cur_trial_spikerate_long[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep_spikerate
        cur_trial_speed_long[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF_long[i,0]:trial_dF_long[i,1],3]
        # only consider roi's max dF value in a given trial if it exceeds threshold
        if np.amax(cur_sweep_spikerate) > trial_std_threshold * roi_std:
            cur_trial_max_idx_long = np.append(cur_trial_max_idx_long,np.nanargmax(cur_trial_spikerate_long[i,:]))
        if np.amax(cur_sweep_spikerate) > max_y:
            max_y = np.amax(cur_sweep_spikerate)

    # plot individual traces
    if make_figure:
        for i,ct in enumerate(cur_trial_spikerate_long):
            ax2.plot(ct,c='0.65',lw=1)
    # ax2_1 = ax2.twinx()
    if len(cur_trial_max_idx_long) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction_long = len(cur_trial_max_idx_long)/np.size(trial_dF_long,0)
        if make_figure:
            sns.distplot(cur_trial_max_idx_long,hist=False,kde=False,rug=True,ax=ax2)
        roi_std_long = np.std(cur_trial_max_idx_long)
    else:
        roi_active_fraction_long = -1
        roi_std_long = -1

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    roi_meanpeak_long = -1
    roi_meanmin_long = -1
    roi_meanpeak_long_idx = -1
    roi_meanpeak_long_time = -1
    mean_amplitude_long = -1
    for i,trace in enumerate(cur_trial_spikerate_long.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < 0.5:
            mean_valid_indices.append(i)
    if len(mean_valid_indices) > 0:
        if make_figure:
            ax2.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),np.nanmean(cur_trial_spikerate_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
            ax2.axvline(window_center,c='r',lw=2)
        roi_meanpeak_long = np.nanmax(np.nanmean(cur_trial_spikerate_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
        roi_meanmin_long = np.nanmin(np.nanmean(cur_trial_spikerate_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
        roi_meanpeak_long_idx = np.nanargmax(np.nanmean(cur_trial_spikerate_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
        roi_meanpeak_long_time = ((roi_meanpeak_long_idx+mean_valid_indices[0])-window_center) * frame_latency

        if len(peak_times) > 0:
            window_center_time = window_center * frame_latency
            vr_peak_time_long = peak_times[0] + window_center_time
            vr_peak_time_long_idx = (vr_peak_time_long/frame_latency).astype('int')
            roi_meanpeak_long = np.nanmean(cur_trial_spikerate_long,0)[vr_peak_time_long_idx]
            if make_figure:
                ax2.axvline(vr_peak_time_long_idx)
        else:
            if make_figure:
                ax2.axvline((roi_meanpeak_long_idx+mean_valid_indices[0]))
        mean_amplitude_long = roi_meanpeak_long - roi_meanmin_long


    if ylims != []:
        hmmin = 0
        hmmax = ylims[1]
    else:
        hmmin = 0
        hmmax = max_y

    if make_figure:
        for i in range(cur_trial_dF_short.shape[0]):
            plot_spikes_y = np.full(np.where(cur_trial_dF_short[i,:]>0)[0].shape,i)
            ax3.scatter(np.where(cur_trial_dF_short[i,:]>0)[0], plot_spikes_y,marker='|',c='k')
        for i in range(cur_trial_dF_long.shape[0]):
            plot_spikes_y = np.full(np.where(cur_trial_dF_long[i,:]>0)[0].shape,i)
            ax4.scatter(np.where(cur_trial_dF_long[i,:]>0)[0], plot_spikes_y,marker='|',c='k')

        # sns.heatmap(cur_trial_dF_short,cmap='binary',vmin=hmmin,vmax=hmmax,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax3)

        # sns.heatmap(cur_trial_dF_long,cmap='viridis',vmin=hmmin,vmax=hmmax,yticklabels=events_long[:,1].astype('int'),xticklabels=False,ax=ax4)
        sns.heatmap(cur_trial_speed_short,cmap='viridis',vmin=0,vmax=60,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax5,cbar=False)
        sns.heatmap(cur_trial_speed_long,cmap='viridis',vmin=0,vmax=60,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax6,cbar=False)
        ax3.axvline(window_center,c='r',lw=2)
        ax4.axvline(window_center,c='r',lw=2)
        ax5.axvline(window_center,c='r',lw=2)
        ax6.axvline(window_center,c='r',lw=2)

        ax1.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)
        ax2.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)

        ax1.set_title(str(np.round(roi_std_short,2)) + ' active: ' + str(np.round(roi_active_fraction_short,2)) + ' peak: ' + str(np.round(roi_meanpeak_short,2)) + ' meanamp:' + str(np.round(mean_amplitude_short)), fontsize=32)
        ax2.set_title(str(np.round(roi_std_long,2)) + ' active: ' + str(np.round(roi_active_fraction_long,2)) + ' peak: ' + str(np.round(roi_meanpeak_long,2)) + ' meanamp:' + str(np.round(mean_amplitude_long)), fontsize=32)
        ax3.set_title('dF/F vs time SHORT track - spike rasterplot')
        ax4.set_title('dF/F vs time LONG track - spike rasterplot')


        ax1.set_ylim([min_y,max_y])
        ax2.set_ylim([min_y,max_y])
        ax1.set_xlim([0,t_max])
        ax2.set_xlim([0,t_max])
        ax3.set_xlim([0,t_max])
        ax4.set_xlim([0,t_max])

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
    norm_value = np.amax(spikerate[:,roi])
    return roi_std_short.item(),roi_std_long.item(), roi_active_fraction_short.item(), roi_active_fraction_long.item(), roi_meanpeak_short.item(), roi_meanpeak_long.item(), \
           roi_meanpeak_short_time.item(), roi_meanpeak_long_time.item(), [min_y.item(),max_y.item()], norm_value.item()

def run_analysis(mousename, sessionname, sessionname_openloop, number_of_rois, h5_filepath, subname, sess_subfolder, filterprop_short, filterprop_long, even_win, blackbox_win):
    """ set up function call and dictionary to collect results """

    MOUSE = mousename
    SESSION = sessionname
    SESSION_OPENLOOP = sessionname_openloop
    NUM_ROIS = number_of_rois
    h5path = h5_filepath
    SUBNAME = subname
    subfolder = sess_subfolder
    subfolder_ol = sess_subfolder + '_openloop'
    write_to_dict = True
    make_figure = False

    session_rois = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    # set up dictionary to hold result parameters from roi
    session_rois[SUBNAME+'_roi_number'] = []
    session_rois[SUBNAME+'_roi_number_ol'] = []
    session_rois[SUBNAME+'_std_short'] = []
    session_rois[SUBNAME+'_std_long'] = []
    session_rois[SUBNAME+'_active_short'] = []
    session_rois[SUBNAME+'_active_long'] = []
    session_rois[SUBNAME+'_peak_short'] = []
    session_rois[SUBNAME+'_peak_long'] = []
    session_rois[SUBNAME+'_peak_time_short'] = []
    session_rois[SUBNAME+'_peak_time_long'] = []
    session_rois[SUBNAME+'_std_short_ol'] = []
    session_rois[SUBNAME+'_std_long_ol'] = []
    session_rois[SUBNAME+'_active_short_ol'] = []
    session_rois[SUBNAME+'_active_long_ol'] = []
    session_rois[SUBNAME+'_peak_short_ol'] = []
    session_rois[SUBNAME+'_peak_long_ol'] = []
    session_rois[SUBNAME+'_peak_time_short_ol'] = []
    session_rois[SUBNAME+'_peak_time_long_ol'] = []
    session_rois['norm_value'] = []
    session_rois['norm_value_ol'] = []

    # if we want to run through all the rois, just say all
    if NUM_ROIS == 'all':
        h5dat = h5py.File(h5path, 'r')
        dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
        h5dat.close()
        NUM_ROIS = dF_ds.shape[1]
        write_to_dict = True
        print('number of rois: ' + str(NUM_ROIS))

    # run analysis for vr session
    for i,r in enumerate(range(NUM_ROIS)):
        print(SUBNAME + ': ' + str(r))
        std_short, std_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, ylims, norm_value = fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [], fformat, subfolder, [], False, make_figure)
        session_rois[SUBNAME+'_roi_number'].append(r)
        session_rois[SUBNAME+'_std_short'].append(std_short)
        session_rois[SUBNAME+'_std_long'].append(std_long)
        session_rois[SUBNAME+'_active_short'].append(active_short)
        session_rois[SUBNAME+'_active_long'].append(active_long)
        session_rois[SUBNAME+'_peak_short'].append(peak_short)
        session_rois[SUBNAME+'_peak_long'].append(peak_long)
        session_rois[SUBNAME+'_peak_time_short'].append(meanpeak_short_time)
        session_rois[SUBNAME+'_peak_time_long'].append(meanpeak_long_time)
        session_rois['norm_value'].append(norm_value)
    #
    # # run openloop condition
    # for i,r in enumerate(range(NUM_ROIS)):
    #     print(SUBNAME + ': ' + str(r))
        std_short, std_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, _, norm_value = fig_landmark_centered(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, ylims, fformat, subfolder_ol, [session_rois[SUBNAME+'_peak_time_short'][i], session_rois[SUBNAME+'_peak_time_long'][i]], False, make_figure)
        session_rois[SUBNAME+'_roi_number_ol'].append(r)
        session_rois[SUBNAME+'_std_short_ol'].append(std_short)
        session_rois[SUBNAME+'_std_long_ol'].append(std_long)
        session_rois[SUBNAME+'_active_short_ol'].append(active_short)
        session_rois[SUBNAME+'_active_long_ol'].append(active_long)
        session_rois[SUBNAME+'_peak_short_ol'].append(peak_short)
        session_rois[SUBNAME+'_peak_long_ol'].append(peak_long)
        session_rois[SUBNAME+'_peak_time_short_ol'].append(meanpeak_short_time)
        session_rois[SUBNAME+'_peak_time_long_ol'].append(meanpeak_long_time)
        session_rois['norm_value_ol'].append(norm_value)

    if write_to_dict:
        print('writing to dict')
        write_dict(MOUSE, SESSION, session_rois, False, True, '_dc')

    # return session_rois

def run_LF170110_2_Day201748_1():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    SESSION_OPENLOOP = 'Day201748_openloop_1'
    NUM_ROIS = 'all' #152
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

def run_LF170110_2_Day201748_2():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_2'
    SESSION_OPENLOOP = 'Day201748_openloop_2'
    NUM_ROIS = 'all' #171
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])


def run_LF170110_2_Day201748_3():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_3'
    SESSION_OPENLOOP = 'Day201748_openloop_3'
    NUM_ROIS = 'all' #50
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

def run_LF170421_2_Day2017719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = 'Day2017719_openloop'
    NUM_ROIS = 'all' #96
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'


    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    write_dict(MOUSE, SESSION)

def run_LF170421_2_Day20170719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170719'
    SESSION_OPENLOOP = 'Day20170719_openloop'
    NUM_ROIS = 'all' #123
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

def run_LF170421_2_Day20170720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170720'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'all' #45
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

def run_LF170421_2_Day2017720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017720'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'all' #45
    NUM_ROIS = 63
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])


def run_LF170420_1_Day201783():
    MOUSE = 'LF170420_1'
    SESSION = 'Day201783'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'all' #81
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

def run_LF170420_1_Day2017719():
    MOUSE = 'LF170420_1'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'all' #91
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

def run_LF170222_1_Day201776():
    MOUSE = 'LF170222_1'
    SESSION = 'Day201776'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'all' #120
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])


def run_LF170110_2_Day2017331():
    MOUSE = 'LF170110_2'
    SESSION = 'Day2017331'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'all' #184
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    write_dict(MOUSE, SESSION)

def run_LF170613_1_Day201784():
    MOUSE = 'LF170613_1'
    SESSION = 'Day201784'
    SESSION_OPENLOOP = 'Day201784_openloop'
    NUM_ROIS = 'all' #77
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    write_dict(MOUSE, SESSION)

def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    NUM_ROIS = 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder,['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])


def run_LF171212_2_Day2017218_1():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_1'
    SESSION_OPENLOOP = 'Day2018218_openloop_1'
    NUM_ROIS = 'all' #91
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

def run_LF171212_2_Day2018218_1():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_1'
    SESSION_OPENLOOP = 'Day2018218_openloop_1'
    NUM_ROIS = 'all' #335
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])



def run_LF171212_2_Day2018218_2():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_2'
    SESSION_OPENLOOP = 'Day2018218_openloop_2'
    NUM_ROIS = 'all' #335
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])


def run_LF170214_1_Day201777():
    MOUSE = 'LF170214_1'
    SESSION = 'Day201777'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 163
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    # SUBNAME = 'lmoff'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 240], ['at_location', 240], [10,10], [2,2])
    #
    # SUBNAME = 'lmon'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 200], ['at_location', 200], [10,10], [2,2])
    #
    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])
    #
    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)

def run_LF170214_1_Day2017714():
    MOUSE = 'LF170214_1'
    SESSION = 'Day2017714'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 140
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])
    #
    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)

def run_LF171211_2_Day201852():
    MOUSE = 'LF171211_2'
    SESSION = 'Day201852'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 66
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    # SUBNAME = 'reward'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])
    # #
    # SUBNAME = 'trialonset'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)

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

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])
    #
    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)

def run_LF171211_1_Day2018321_2():
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    SESSION_OPENLOOP = 'Day2018321_openloop_2'
    NUM_ROIS = 'all'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])
    #
    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])


def run_LF18112_2_Day2018322_2():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018322_2'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 283
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])
    #
    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)

def run_LF180112_2_Day2018424_1():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_1'
    SESSION_OPENLOOP = 'Day2018424_openloop_1'
    NUM_ROIS = 73
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)

def run_LF180112_2_Day2018424_2():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_2'
    SESSION_OPENLOOP = 'Day2018424_openloop_2'
    NUM_ROIS = 43
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2])

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], [0,20], [2,2])

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME+'_deconvolved'
    run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], [10,10], [2,2])

    # write result parameters to .json file
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','a') as f:
        json.dump(roi_result_params,f)

def do_single():
    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # NUM_ROIS = [40]
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    #
    # for r in NUM_ROIS:
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [], fformat, subfolder)

    # MOUSE = 'LF170613_1'
    # SESSION = 'Day201784'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # NUM_ROIS = [73]
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # filterprop_short = ['at_location', 320]
    # filterprop_long = ['at_location', 380]
    # even_win = [20,0]
    # blackbox_win = [2,2]
    # SUBNAME = 'reward'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # subfolder_ol = subfolder + '_openloop'
    # fformat = 'png'
    #
    # for r in NUM_ROIS:
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [0,3], fformat, subfolder, [], True)
    #     fig_landmark_centered(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [0,3], fformat, subfolder_ol, [], True)

    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_1'
    # SESSION_OPENLOOP = 'Day201748_openloop_1'
    # NUM_ROIS = [74]
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # subfolder_ol = subfolder + '_openloop'
    # fformat = 'png'

    # for r in NUM_ROIS:
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,2], fformat, subfolder, [], True)
    #     fig_landmark_centered(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,2], fformat, subfolder_ol, [], True)

    # NUM_ROIS = [32]
    # even_win = [0,20]
    # SUBNAME = 'trialonset'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # subfolder_ol = subfolder + '_openloop'
    # filterprop_short = ['trial_transition']
    # filterprop_long = ['trial_transition']
    # for r in NUM_ROIS:
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,2], fformat, subfolder, [], True)
    #     fig_landmark_centered(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,2], fformat, subfolder_ol, [], True)

    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # SESSION_OPENLOOP = 'Day201777_openloop'
    # NUM_ROIS = [40]
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # subfolder_ol = subfolder + '_openloop'
    # fformat = 'png'
    #
    # for r in NUM_ROIS:
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,0.4], fformat, subfolder, [], True)
    #     fig_landmark_centered(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,0.4], fformat, subfolder_ol, [], True)
    #
    # MOUSE = 'LF170214_1'
    # SESSION = 'Day2017714'
    # SESSION_OPENLOOP = 'Day2017714_openloop'
    # NUM_ROIS = [40]
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # filterprop_short = ['at_location', 220]
    # filterprop_long = ['at_location', 220]
    # even_win = [10,10]
    # blackbox_win = [2,2]
    # SUBNAME = 'lmcenter'
    # subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    # subfolder_ol = subfolder + '_openloop'
    # fformat = 'png'
    #
    # for r in NUM_ROIS:
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,0.4], fformat, subfolder, [], True)
    #     fig_landmark_centered(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,0.4], fformat, subfolder_ol, [], True)

    MOUSE = 'LF171211_2'
    SESSION = 'Day201852'
    SESSION_OPENLOOP = 'Day201852_openloop'
    NUM_ROIS = [49]
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    filterprop_short = ['at_location', 220]
    filterprop_long = ['at_location', 220]
    even_win = [10,5]
    blackbox_win = [2,2]
    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    subfolder_ol = subfolder + '_openloop'
    fformat = 'png'

    for r in NUM_ROIS:
        fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,2], fformat, subfolder, [], True)
        # fig_landmark_centered(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, even_win, blackbox_win, [-0.1,2.5], fformat, subfolder_ol, [], True)



if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline





    run_LF170613_1_Day20170804()
    # run_LF170420_1_Day2017719()
    # run_LF170420_1_Day201783()
    # run_LF170421_2_Day20170719()
    # run_LF170421_2_Day2017720() # <-- RUN! (only run on 63 cells as openloop has fewer ROIs for now...)
    # run_LF170110_2_Day201748_1()
    # run_LF170110_2_Day201748_2()
    # run_LF170110_2_Day201748_3()
    # run_LF170222_1_Day201776()


    # run_LF170110_2_Day2017331()
    # run_LF170613_1_Day201784()

    # run_LF171212_2_Day2018218_1()
    # run_LF171212_2_Day2018218_2()
    # run_LF171211_1_Day2018321_2()

    # run_LF170214_1_Day201777()
    # run_LF170214_1_Day2017714()
    # run_LF171211_2_Day201852()
    # run_LF18112_2_Day2018322_1()
    # run_LF18112_2_Day2018322_2()

    # run_LF180112_2_Day2018424_1()
    # run_LF180112_2_Day2018424_2()

    #old
    #run_LF170421_2_Day20170720()

    # do_single()
