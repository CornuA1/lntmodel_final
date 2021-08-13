"""
plot landmark neurons during passive condition in animal stationary vs animal running condition

@author: lukasfischer

"""

import numpy as np
import scipy as sp
from scipy.signal import butter, filtfilt
from scipy import stats
import statsmodels.api as sm
import h5py,os,sys,traceback,matplotlib,warnings,json,yaml
from multiprocessing import Process
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
import ipdb
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')
from event_ind import event_ind
from filter_trials import filter_trials
from write_dict import write_dict
from analysis_parameters import MIN_FRACTION_ACTIVE, MIN_MEAN_AMP, MIN_ZSCORE, MIN_TRIALS_ACTIVE, MIN_DF, PEAK_MATCH_WINDOW, MEAN_TRACE_FRACTION

MIN_TRIALS = 0

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def roi_response_validation(roi_params, tl, el, roi_idx_num):
    """
    separate function to check whether a given response passes criterion for being considered a real roi_response_validation

    """

    roi_activity = roi_params[el + '_active_' + tl][roi_idx_num]
    roi_peak_val = roi_params[el + '_peak_' + tl][roi_idx_num]
    roi_zscore_val = roi_params[el + '_peak_zscore_' + tl][roi_idx_num]
    mean_trace = roi_params['space_mean_trace_'+tl][roi_idx_num]

    if roi_activity > MIN_FRACTION_ACTIVE and roi_zscore_val > MIN_ZSCORE and (np.nanmax(mean_trace) - np.nanmin(mean_trace)) > MIN_MEAN_AMP:
        return True
    else:
        return False

def get_eventaligned_rois(roi_param_list, trialtypes, align_event):
    """ return roi number, peak value and peak time of all neurons that have their max response at <align_even> in VR """
    # hold values of mean peak
    event_list = ['trialonset','lmcenter','reward']
    result_max_peak = {}
    # set up empty dicts so we can later append to them
    for rpl in roi_param_list:
        for tl in trialtypes:
            mouse_sess = rpl[1] + '_' + rpl[2]
            result_max_peak[align_event + '_peakval_' + tl + '_' + mouse_sess] = []
            result_max_peak[align_event + '_peak_time_' + tl + '_' + mouse_sess] = []
            result_max_peak[align_event + '_peakval_ol_' + tl + '_' +  mouse_sess] = []
            result_max_peak[align_event + '_peak_time_ol_' + tl + '_' +  mouse_sess] = []
            result_max_peak[align_event + '_roi_number_' + tl +'_' + mouse_sess] = []

    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        # load roi parameters for given session
        with open(rpl[0],'r') as f:
            roi_params = json.load(f)
        # grab a full list of roi numbers
        roi_list_all = roi_params['valid_rois']
        mouse_sess = rpl[1] + '_' + rpl[2]
        # loop through every roi
        for j,r in enumerate(roi_list_all):
            # loop through every trialtype and alignment point to determine largest response
            for tl in trialtypes:
                max_peak = -99
                max_peak_ol = -99
                roi_num = -1
                valid = False
                peak_event = ''
                peak_trialtype = ''
                for el in event_list:
                    value_key = el + '_peak_' + tl
                    value_key_ol = el + '_peak_' + tl + '_ol'
                    value_key_peaktime = el + '_peak_time_' + tl
                    value_key_peaktime_ol = el + '_peak_time_' + tl + '_ol'
                    # check roi max peak for each alignment point and store wich ones has the highest value
                    if roi_params[value_key][j] > max_peak and roi_response_validation(roi_params, tl, el, j):
                        valid = True
                        # ipdb.set_trace()
                        max_peak = roi_params[value_key][j]
                        max_peak_ol = roi_params[value_key_ol][j]
                        max_peak_time = roi_params[value_key_peaktime][j]
                        max_peak_time_ol = roi_params[value_key_peaktime_ol][j]
                        peak_event = el
                        peak_trialtype = tl
                        roi_num = r
                # write results for alignment point with highest value to results dict
                if valid:
                    if peak_event == align_event:
                        result_max_peak[align_event + '_roi_number_' + peak_trialtype + '_' + mouse_sess].append(roi_num)
                        result_max_peak[align_event + '_peakval_' + peak_trialtype + '_' + mouse_sess].append(max_peak)
                        result_max_peak[align_event + '_peakval_ol_' + peak_trialtype + '_' + mouse_sess].append(max_peak_ol)
                        result_max_peak[align_event + '_peak_time_' + tl + '_' + mouse_sess].append(max_peak_time)
                        result_max_peak[align_event + '_peak_time_ol_' + peak_trialtype + '_' + mouse_sess].append(max_peak_time_ol)

    return result_max_peak

def fig_motor_response_diff(h5path, sess, roi, fname, tracktype, eventprops, max_timewindow, blackbox_time, movement_timewindow, ylims=[], fformat='png', subfolder=[], peak_times=None, make_figure=False):
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
    frame_latency = behav_ds[2,2]
    max_timewindow_idx = (MAX_TIMEWINDOW/frame_latency).astype('int')
    # calculate the added frames for the blackbox
    blackbox_idx = (BLACKBOX_TIME/frame_latency).astype('int')
    # determine maximum number of samples per trial
    t_max = (max_timewindow_idx[0]+blackbox_idx[0]) + (max_timewindow_idx[1]+blackbox_idx[1])
    # store center of window in case it is not symmetric
    window_center = max_timewindow_idx[0]+blackbox_idx[0]

    if peak_times is not None:
        window_center_time = window_center * frame_latency
        vr_peak_time = peak_times + window_center_time
        vr_peak_time_idx = (vr_peak_time/frame_latency).astype('int')
        mov_t_start = (vr_peak_time_idx - (movement_timewindow[0]/frame_latency).astype('int'))
        mov_t_end = (vr_peak_time_idx + (movement_timewindow[1]/frame_latency).astype('int'))
    else:
        mov_t_start = (window_center - (movement_timewindow[0]/frame_latency).astype('int'))
        mov_t_end = (window_center + (movement_timewindow[1]/frame_latency).astype('int'))

    # specify the timewindow within which an animal has to exceed a given threshold to be considered a running trial
    # movement_timewindow = [2,2]

    # threshold above which mean speed has to be for a trial to be considered a running trial
    movement_threshold = 5
    # calcluate standard deviation of ROI traces
    roi_std = np.std(dF_ds[:,roi])
    # threshold the response of a roi in a given trial has to exceed count its response toward the tuning of the cell
    trial_std_threshold = 3
    # on which fraction of trials did the roi exceed 3 standard deviations
    roi_active_fraction_short = 0
    roi_active_fraction_long = 0

    # specify track numbers
    if tracktype == 'short':
        tracknr = 3
    elif tracktype == 'long':
        tracknr = 4
    elif tracktype == 'blackbox':
        tracknr = 5

    # filter and plot running speed trace
    order = 6
    fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz
    behav_ds[:,8] = butter_lowpass_filter(behav_ds[:,8], cutoff, fs, order)

    # ylims
    min_y = -0.3
    max_y = np.amax(dF_ds[:,roi])
    if not ylims == []:
        min_y = ylims[0]
        max_y = ylims[1]

    if make_figure:
        # create figure and axes to later plot on
        fig = plt.figure(figsize=(10,12))
        ax1 = plt.subplot(4,2,1)
        ax2 = plt.subplot(4,2,2)
        ax3 = plt.subplot(4,2,3)
        ax4 = plt.subplot(4,2,4)
        ax5 = plt.subplot(4,2,5)
        ax6 = plt.subplot(4,2,6)
        ax7 = plt.subplot(4,2,7)
        ax8 = plt.subplot(4,2,8)

        one_sec = np.round(1/frame_latency,0).astype(int)
        # print(one_sec)
        ax1.set_xticks([0,5*one_sec])
        ax1.set_xticklabels([0,10])
        ax2.set_xticks([0,5*one_sec])
        ax2.set_xticklabels([0,10])
        ax3.set_xticks([0,5*one_sec])
        ax3.set_xticklabels([0,10])
        ax4.set_xticks([0,5*one_sec])
        ax4.set_xticklabels([0,10])

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

        ax3.spines['left'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            left='off', \
            bottom='off', \
            right='off', \
            top='off')

        ax4.spines['left'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            left='off', \
            bottom='off', \
            right='off', \
            top='off')
    else:
        ax1 = None
        ax2 = None

    # # get indices of desired behavioural event
    trials = filter_trials( behav_ds, [], ['tracknumber',tracknr])
    #
    events = event_ind(behav_ds, eventprops, trials)
    # print(events)

    # get indices of trial start and end (or max_event_timewindow if trial is long)
    trial_dF = np.zeros((np.size(events[:,0]),2)).astype(int)
    for i,cur_ind in enumerate(events):
        cur_trial_idx = [np.where(behav_ds[:,6] == cur_ind[1])[0][0],np.where(behav_ds[:,6] == cur_ind[1])[0][-1]]
        # # determine indices of beginning and end of timewindow
        if cur_ind[0] - max_timewindow_idx[0] > cur_trial_idx[0]:
            if cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0]) < 0:
                trial_dF[i,0] = 0
            else:
                trial_dF[i,0] = int(cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0]))
        else:
            if cur_trial_idx[0] - blackbox_idx[0] < 0:
                trial_dF[i,0] = 0
            else:
                trial_dF[i,0] = int(cur_trial_idx[0] - blackbox_idx[0])

        if cur_ind[0] + max_timewindow_idx[1] < cur_trial_idx[1]:
            trial_dF[i,1] = int(cur_ind[0] + (max_timewindow_idx[1] + blackbox_idx[1]))
        else:
            if cur_trial_idx[1] + blackbox_idx[1] > np.size(behav_ds,0):
                trial_dF[i,1] = int(np.size(behav_ds,0))
            else:
                trial_dF[i,1] = int(cur_trial_idx[1] + blackbox_idx[1])

    # grab dF data for each trial
    cur_trial_dF_run = np.full((np.size(events[:,0]),int(t_max)),np.nan)
    cur_trial_dF_sit = np.full((np.size(events[:,0]),int(t_max)),np.nan)
    cur_trial_speed_run = np.full((np.size(events[:,0]),int(t_max)),np.nan)
    cur_trial_speed_sit = np.full((np.size(events[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(events[:,0]))
    cur_trial_max_idx = np.empty(0)

    # keep track of which trials to delete from the matrix
    not_run_trials = []
    not_sit_trials = []

    for i in range(np.size(trial_dF,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF[i,0]:trial_dF[i,1],roi]
        cur_sweep_speed = behav_ds[trial_dF[i,0]:trial_dF[i,1],8]
        # delete datapoints that are artifactually high
        cur_sweep_speed[cur_sweep_speed > 100] = np.nan
        cur_trial_event_idx[i] = events[i,0] - trial_dF[i,0]
        trace_start = int(window_center - cur_trial_event_idx[i])
        mov_t_start_aligned = int(mov_t_start - trace_start)
        mov_t_end_aligned = int(mov_t_end - trace_start)

        # print(mov_t_start_aligned, mov_t_end_aligned)
        if np.nanmean(cur_sweep_speed[mov_t_start_aligned:mov_t_end_aligned]) > movement_threshold:
            cur_trial_speed_run[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep_speed
            cur_trial_dF_run[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
            not_sit_trials.append(i)
        else:
            cur_trial_speed_sit[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep_speed
            cur_trial_dF_sit[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
            not_run_trials.append(i)

        # only consider roi's max dF value in a given trial if it exceeds threshold
        # if np.amax(cur_sweep) > trial_std_threshold * roi_std:
        #     cur_trial_max_idx = np.append(cur_trial_max_idx,np.nanargmax(cur_trial_dF_sit[i,:]))
        # if np.amax(cur_sweep) > max_y:
        #     max_y = np.amax(cur_sweep)

    # delete rows from matrix which weren't filled
    cur_trial_dF_sit = np.delete(cur_trial_dF_sit, not_sit_trials, 0)
    cur_trial_dF_run = np.delete(cur_trial_dF_run, not_run_trials, 0)
    cur_trial_speed_sit = np.delete(cur_trial_speed_sit, not_sit_trials, 0)
    cur_trial_speed_run = np.delete(cur_trial_speed_run, not_run_trials, 0)
    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices_sit = []
    if len(cur_trial_dF_sit) > 0:
        for i,trace in enumerate(cur_trial_dF_sit.T):
            if np.count_nonzero(np.isnan(trace))/len(trace) < MEAN_TRACE_FRACTION:
                mean_valid_indices_sit.append(i)
    mean_valid_indices_run = []
    if len(cur_trial_dF_run) > 0:
        for i,trace in enumerate(cur_trial_dF_run.T):
            if np.count_nonzero(np.isnan(trace))/len(trace) < MEAN_TRACE_FRACTION:
                mean_valid_indices_run.append(i)

    if peak_times is not None:
        vr_peak_time_start = vr_peak_time - (PEAK_MATCH_WINDOW/2)
        vr_peak_time_end = vr_peak_time + (PEAK_MATCH_WINDOW/2)

        if len(cur_trial_dF_sit) > MIN_TRIALS:
            vr_peak_time_win_start_sit_idx = np.amax([(vr_peak_time_start/frame_latency).astype('int'),mean_valid_indices_sit[0]])
            vr_peak_time_win_end_sit_idx = np.amin([(vr_peak_time_end/frame_latency).astype('int'),mean_valid_indices_sit[-1]])

            # in case the indeces are nonsensical (this can happen in edge cases where the peak idx in VR is outside the valid_indeces range), just go with the original index
            if vr_peak_time_win_start_sit_idx > vr_peak_time_win_end_sit_idx:
                vr_peak_time_win_start_sit_idx = vr_peak_time_idx
                vr_peak_time_win_end_sit_idx = vr_peak_time_idx+1

            try:
                vr_peak_time_sit_idx = np.argmax(np.nanmean(cur_trial_dF_sit[:,vr_peak_time_win_start_sit_idx:vr_peak_time_win_end_sit_idx],0))
                roi_meanpeak_sit = np.nanmean(cur_trial_dF_sit,0)[vr_peak_time_win_start_sit_idx+vr_peak_time_sit_idx]
                if make_figure:
                    ax1.axvline(vr_peak_time_win_start_sit_idx+vr_peak_time_sit_idx, ls='--')
            except ValueError:
                roi_meanpeak_sit = np.nan
                print('WARNING: Trial has no valid values. Skipping trial.')
        else:
            roi_meanpeak_sit = np.nan

        if len(cur_trial_dF_run) > MIN_TRIALS:
            vr_peak_time_win_start_run_idx = np.amax([(vr_peak_time_start/frame_latency).astype('int'),mean_valid_indices_run[0]])
            vr_peak_time_win_end_run_idx = np.amin([(vr_peak_time_end/frame_latency).astype('int'),mean_valid_indices_run[-1]])

            if vr_peak_time_win_start_run_idx > vr_peak_time_win_end_run_idx:
                vr_peak_time_win_start_run_idx = vr_peak_time_idx
                vr_peak_time_win_end_run_idx = vr_peak_time_idx+1

            try:
                vr_peak_time_run_idx = np.argmax(np.nanmean(cur_trial_dF_run[:,vr_peak_time_win_start_run_idx:vr_peak_time_win_end_run_idx],0))
                roi_meanpeak_run = np.nanmean(cur_trial_dF_run,0)[vr_peak_time_win_start_run_idx+vr_peak_time_run_idx]
                if make_figure:
                    ax2.axvline(vr_peak_time_win_start_run_idx+vr_peak_time_run_idx, ls='--')
            except ValueError:
                roi_meanpeak_run = np.nan
                print('WARNING: Trial has no valid values. Skipping trial.')
        else:
            roi_meanpeak_run = np.nan

    else:
        roi_meanpeak_run = np.nanmean(cur_trial_dF_run,0)[window_center]
        roi_meanpeak_sit = np.nanmean(cur_trial_dF_sit,0)[window_center]


    # ax1_1 = ax1.twinx()
    if len(cur_trial_max_idx) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction_short = len(cur_trial_max_idx)/np.size(trial_dF,0)
        # sns.distplot(cur_trial_max_idx,hist=False,kde=False,rug=True,ax=ax1)
        roi_std_short = np.std(cur_trial_max_idx)
    else:
        roi_active_fraction_short = np.int64(-1)
        roi_std_short = np.int64(-1)
    #
    # # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    # for i,trace in enumerate(cur_trial_dF_sit.T):
        # if np.count_nonzero(np.isnan(trace))/len(trace) < 0.5:
        #     mean_valid_indices.append(i)
    plot_individual_trials = True
    plot_sem_shade = False
    if make_figure:
        if len(cur_trial_dF_sit) > MIN_TRIALS:
            ax1.plot(np.arange(mean_valid_indices_sit[0], mean_valid_indices_sit[-1],1),np.nanmean(cur_trial_dF_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0),c='k',lw=2,zorder=4)
            ax3.plot(np.arange(mean_valid_indices_sit[0], mean_valid_indices_sit[-1],1),np.nanmean(cur_trial_speed_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0),c='#008000',lw=2,zorder=4)
        if len(cur_trial_dF_run) > MIN_TRIALS:
            ax2.plot(np.arange(mean_valid_indices_run[0], mean_valid_indices_run[-1],1),np.nanmean(cur_trial_dF_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0),c='k',lw=2,zorder=4)
            ax4.plot(np.arange(mean_valid_indices_run[0], mean_valid_indices_run[-1],1),np.nanmean(cur_trial_speed_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0),c='#008000',lw=2,zorder=4)

        if plot_individual_trials is True:
            for i in range(cur_trial_dF_sit.shape[0]):
                ax1.plot(cur_trial_dF_sit[i],c='0.65',lw=1,zorder=3)
                ax3.plot(cur_trial_speed_sit[i],c='0.65',lw=1,zorder=3)
            for i in range(cur_trial_dF_run.shape[0]):
                ax2.plot(cur_trial_dF_run[i],c='0.65',lw=1,zorder=3)
                ax4.plot(cur_trial_speed_run[i],c='0.65',lw=1,zorder=3)

        if plot_sem_shade is True:
            if len(cur_trial_dF_sit) > MIN_TRIALS:
                sem_dF = stats.sem(cur_trial_dF_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0,nan_policy='omit')
                ax1.fill_between(np.arange(mean_valid_indices_sit[0], mean_valid_indices_sit[-1],1), np.nanmean(cur_trial_dF_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0)-sem_dF, np.nanmean(cur_trial_dF_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0)+sem_dF, alpha=0.25, lw=0, color='0.5')
                sem_speed = stats.sem(cur_trial_speed_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0,nan_policy='omit')
                ax3.fill_between(np.arange(mean_valid_indices_sit[0], mean_valid_indices_sit[-1],1), np.nanmean(cur_trial_speed_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0)-sem_speed, np.nanmean(cur_trial_speed_sit[:,mean_valid_indices_sit[0]:mean_valid_indices_sit[-1]],0)+sem_speed, alpha=0.25, lw=0, color='#008000')

            if len(cur_trial_dF_run) > MIN_TRIALS:
                sem_dF = stats.sem(cur_trial_dF_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0,nan_policy='omit')
                ax2.fill_between(np.arange(mean_valid_indices_run[0], mean_valid_indices_run[-1],1), np.nanmean(cur_trial_dF_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0)-sem_dF, np.nanmean(cur_trial_dF_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0)+sem_dF, alpha=0.25, lw=0, color='0.5')
                sem_speed = stats.sem(cur_trial_speed_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0,nan_policy='omit')
                ax4.fill_between(np.arange(mean_valid_indices_run[0], mean_valid_indices_run[-1],1), np.nanmean(cur_trial_speed_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0)-sem_speed, np.nanmean(cur_trial_speed_run[:,mean_valid_indices_run[0]:mean_valid_indices_run[-1]],0)+sem_speed, alpha=0.25, lw=0, color='#008000')

        ax1.axvline(window_center,c='r',lw=2)
        ax2.axvline(window_center,c='r',lw=2)
        ax3.axvline(window_center,c='r',lw=2)
        ax4.axvline(window_center,c='r',lw=2)
        ax3.axvline(mov_t_start,c='b',ls='--',lw=2)
        ax3.axvline(mov_t_end,c='b',ls='--',lw=2)
        ax4.axvline(mov_t_start,c='b',ls='--',lw=2)
        ax4.axvline(mov_t_end,c='b',ls='--',lw=2)


    if make_figure:
        # print(cur_trial_dF_run)
        if len(cur_trial_dF_sit) > 0:
            sns.heatmap(cur_trial_dF_sit,cmap='viridis',vmin=0,yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax5)
            sns.heatmap(cur_trial_speed_sit,cmap='viridis',vmin=0,yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax7)
        if len(cur_trial_dF_run) > 0:
            sns.heatmap(cur_trial_dF_run,cmap='viridis',vmin=0,yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax6)
            sns.heatmap(cur_trial_speed_run,cmap='viridis',vmin=0,yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax8)

        ax1.set_ylim([min_y,max_y])
        ax2.set_ylim([min_y,max_y])
        ax1.set_xlim([0,t_max])
        ax2.set_xlim([0,t_max])
        ax3.set_xlim([0,t_max])
        ax4.set_xlim([0,t_max])
        ax3.set_ylim([-2,35])
        ax4.set_ylim([-2,35])

        fig.tight_layout()
        fig.suptitle(tracktype + str(roi), wrap=True)
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

        print(fname)

    # convert values to json-compatible format
    if not np.isnan(roi_meanpeak_sit):
        roi_meanpeak_sit = roi_meanpeak_sit.item()
    if not np.isnan(roi_meanpeak_run):
        roi_meanpeak_run = roi_meanpeak_run.item()

    return roi_meanpeak_sit, roi_meanpeak_run

def plot_motor_v_sit_pop(roi_param_list, trialtypes, align_event, fformat='png', subfolder=[]):
    write_to_dict = True
    make_figure = False
    run_timewindow = [1,1]
    # get all rois that are aligned to a given event
    max_peaks = get_eventaligned_rois(roi_param_list, trialtypes, 'lmcenter')

    # for ks in max_peaks.keys():
    #     print(ks)
    # print(max_peaks['lmcenter_peakval_short_LF171212_2_Day20182'])
    # return

    # keep list of existing dictionaries so we can append rather than create new ones
    existing_dicts = []

    # print(max_peaks['lmcenter_roi_number_short_LF170613_1_Day20170804_openloop'])
    # print(max_peaks.keys())

    for rpl in roi_param_list:
        mouse = rpl[1]
        sess = rpl[2]
        h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'



        # sessname_dict = rl.split('_')[6]

        # set up dictionary
        roi_result_params = {
            'lmcenter_roi_number_svr_short' : [],
            'lmcenter_peak_sit_short' : [],
            'lmcenter_peak_run_short' : [],
            'lmcenter_roi_number_svr_long' : [],
            'lmcenter_peak_sit_long' : [],
            'lmcenter_peak_run_long' : [],
            'lmcenter_roi_number_blackbox' : [],
            'lmcenter_peak_sit_blackbox' : [],
            'lmcenter_peak_run_blackbox' : []
        }

        peak_val_key = 'lmcenter_peakval_short_' + rpl[1] + '_' + rpl[2]
        peak_time_key_short = 'lmcenter_peak_time_short_' + rpl[1] + '_' + rpl[2]
        print('processing short trials')
        subfolder = mouse+'_'+sess+'_lmcenter_svr_short'
        roilist_short = max_peaks[align_event + '_roi_number_short_' + rpl[1] + '_' + rpl[2]]
        for i in range(len(roilist_short)):
            # continue
            roi = roilist_short[i]
            peak_sit, peak_run = fig_motor_response_diff(h5path, sess, roi, 'passive_svr_short_' + str(roi), 'short', ['at_location', 220],  [10,10], [2,2], run_timewindow, [], fformat, subfolder, max_peaks[peak_time_key_short][i], make_figure)
            roi_result_params['lmcenter_roi_number_svr_short'].append(roi)
            roi_result_params['lmcenter_peak_sit_short'].append(peak_sit)
            roi_result_params['lmcenter_peak_run_short'].append(peak_run)


        peak_val_key = 'lmcenter_peakval_long_' + rpl[1] + '_' + rpl[2]
        peak_time_key_long = 'lmcenter_peak_time_long_' + rpl[1] + '_' + rpl[2]

        print('processing long trials')
        subfolder = mouse+'_'+sess+'_lmcenter_svr_long'
        roilist_long = max_peaks[align_event + '_roi_number_long_' + rpl[1] + '_' + rpl[2]]
        for i in range(len(roilist_long)):
            # continue
            roi = roilist_long[i]
            peak_sit, peak_run = fig_motor_response_diff(h5path, sess, roi, 'passive_svr_long_' + str(roi), 'long', ['at_location', 220],  [10,10], [2,2], run_timewindow, [], fformat, subfolder, max_peaks[peak_time_key_long][i], make_figure)
            roi_result_params['lmcenter_roi_number_svr_long'].append(roi)
            roi_result_params['lmcenter_peak_sit_long'].append(peak_sit)
            roi_result_params['lmcenter_peak_run_long'].append(peak_run)

        # get union of short and long roilist (converting to numpy format is easiest here)
        short_long_roilist = np.union1d(np.array(roilist_short),np.array(roilist_long)).tolist()
        print('processing blackbox sections')
        subfolder = mouse+'_'+sess+'_lmcenter_svr_blackbox'
        for i in range(len(short_long_roilist)):
            # continue
            roi = short_long_roilist[i]
            peak_sit, peak_run = fig_motor_response_diff(h5path, sess, roi, 'passive_svr_blackbox_' + str(roi), 'blackbox', ['at_time', 1.5],  [5,5], [2,2], [1,1], [], fformat, subfolder, None, make_figure)
            roi_result_params['lmcenter_roi_number_blackbox'].append(roi)
            roi_result_params['lmcenter_peak_sit_blackbox'].append(peak_sit)
            roi_result_params['lmcenter_peak_run_blackbox'].append(peak_run)

        # write to dict
        if write_to_dict:
            write_dict(mouse, rpl[3], roi_result_params)

        # ipdb.set_trace()

def do_single(rpl):
    # get all rois that are aligned to a given event
    max_peaks = get_eventaligned_rois(roi_param_list, trialtypes, 'lmcenter')

    mouse = 'LF170613_1'
    sess = 'Day20170804_openloop'
    h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
    roi = 59
    align_event = 'lmcenter'

    subfolder = 'testplots'
    make_figure = True

    print(max_peaks[align_event + '_roi_number_short_' + mouse + '_' + sess])

    peak_val_key = 'lmcenter_peakval_short_' + mouse + '_' + sess
    peak_time_key_short = 'lmcenter_peak_time_short_' + mouse + '_' + sess
    roilist_roi_idx = max_peaks[align_event + '_roi_number_short_' + mouse + '_' + sess].index(roi)
    peak_sit, peak_run = fig_motor_response_diff(h5path, sess, roi, 'passive_svr_short_' + str(roi), 'short', ['at_location', 220],  [10,10], [2,2], [2,2], [], fformat, subfolder, max_peaks[peak_time_key_short][roilist_roi_idx], make_figure)

    peak_val_key = 'lmcenter_peakval_long_' + mouse + '_' + sess
    peak_time_key_long = 'lmcenter_peak_time_long_' + mouse + '_' + sess
    roilist_roi_idx = max_peaks[align_event + '_roi_number_long_' + mouse + '_' + sess].index(roi)
    peak_sit, peak_run = fig_motor_response_diff(h5path, sess, roi, 'passive_svr_long_' + str(roi), 'long', ['at_location', 220],  [10,10], [2,2], [2,2], [], fformat, subfolder, max_peaks[peak_time_key_long][roilist_roi_idx], make_figure)



    # peak_val_key = 'lmcenter_peakval_long_' + rpl[1] + '_' + rpl[2]
    # peak_time_key_long = 'lmcenter_peak_time_long_' + rpl[1] + '_' + rpl[2]
    # print('processing long trials')
    # roilist_long = max_peaks[align_event + '_roi_number_long_' + rpl[1] + '_' + rpl[2]]
    # roilist_long = [15]
    # for i in range(len(roilist_long)):
    #     # continue
    #     print(i, roilist_long[i])
    #     roi = roilist_long[i]
    #     peak_sit, peak_run = fig_motor_response_diff(h5path, sess, roi, 'passive_svr_long_' + str(roi), 'long', ['at_location', 220],  [10,10], [2,2], [2,2], [], fformat, subfolder, max_peaks[peak_time_key_long][i], make_figure)


    # get union of short and long roilist (converting to numpy format is easiest here)
    # short_long_roilist = np.union1d(np.array(roilist_short),np.array(roilist_long)).tolist()
    # print('processing blackbox sections')
    # roilist_short=[15]
    # for i in range(len(short_long_roilist)):
    #     # continue
    #     print(i, short_long_roilist[i])
    #     roi = short_long_roilist[i]
    #     peak_sit, peak_run = fig_motor_response_diff(h5path, sess, roi, 'passive_svr_blackbox_' + str(roi), 'blackbox', ['at_time', 2.5],  [5,5], [2,2], [1,1], [], fformat, subfolder, None, make_figure)
    #     roi_result_params['lmcenter_roi_number_blackbox'].append(roi)
    #     roi_result_params['lmcenter_peak_sit_blackbox'].append(peak_sit)
    #     roi_result_params['lmcenter_peak_run_blackbox'].append(peak_run)



if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline
    fformat = 'png'

    # list of roi parameter files
    # roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017719', # *
    #                    '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day201783', #, # *
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170222_1_Day201776', # *
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170613_1_Day201784'] # *


                       # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017720',
                       # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day2017719',
                      #'/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_1',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_2',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_3',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day2017331',

    # roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF171211_2_Day201852'] #,
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170214_1_Day2017714/roi_params.json',
    #                  '/Users/lukasfischer/Work/exps/MTH3/figures/LF171211_2_Day201852/roi_params.json',]

    roi_param_list = [
                      ['E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2.json','LF171212_2','Day2018218_openloop_2','Day2018218_2'],
                      ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804.json','LF170613_1','Day20170804_openloop','Day20170804'],
                      ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719.json','LF170421_2','Day20170719_openloop','Day20170719' ],
                      # ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720.json','LF170421_2','Day2017720_openloop','Day2017720'],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783.json','LF170420_1','Day201783_openloop','Day201783'],
                      ['E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776.json','LF170222_1','Day201776_openloop','Day201776']
                     ]

    # roi_param_list = [
    #                   ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day201777.json','LF170214_1','Day201777_openloop','Day201777'],
    #                   ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day2017714.json','LF170214_1','Day2017714_openloop','Day2017714'],
    #                   ['E:\\MTH3_figures\\LF171211_2\\LF171211_2_Day201852.json','LF171211_2','Day201852_openloop','Day201852'],
    #                   ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_openloop_1','Day2018424_1'],
    #                   ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_openloop_2','Day2018424_2']
    #                  ]

    fname = 'passive_motor_v_stat'
    event_list = ['trialonset','lmcenter','reward']
    trialtypes = ['short', 'long']
    subfolder = []

    plot_motor_v_sit_pop(roi_param_list, trialtypes, 'lmcenter', fformat, subfolder)
    # plot_motor_v_sit_pop(roi_param_list, trialtypes, 'trialonset', fformat, subfolder)
    # plot_motor_v_sit_pop(roi_param_list, trialtypes, 'reward', fformat, subfolder)
    # subfolder = 'testplots'
    # do_single([['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804.json','LF170613_1','Day20170804_openloop','Day20170804']])
