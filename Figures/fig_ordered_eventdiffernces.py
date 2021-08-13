"""
Plot trace of an individual ROI centered around a given location using ordered
lists of trials and comparing them

@author: Lukas Fischer

"""

import numpy as np
import h5py,sys,yaml,os,json,matplotlib,warnings
warnings.simplefilter('ignore')
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
from filter_trials import filter_trials
from order_trials import order_trials
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

def plot_traces(plot_events, behav_ds, dF_ds, roi, trial_std_threshold, max_y, max_timewindow, blackbox_time, peak_times, make_figure, plot_color, ax_object,ax_object2,ax_object3):
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
    # get indices of trial start and end (or max_event_timewindow if trial is long)
    trial_dF = np.zeros((np.size(plot_events[:,0]),2))
    for i,cur_ind in enumerate(plot_events):
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

    trial_dF = trial_dF.astype(int)
    # grab dF data for each trial
    cur_trial_dF = np.full((np.size(plot_events[:,0]),int(t_max)),np.nan)
    cur_trial_speed = np.full((np.size(plot_events[:,0]),int(t_max)),np.nan)
    cur_trial_event_idx = np.zeros(np.size(plot_events[:,0]))
    cur_trial_max_idx = np.empty(0)
    for i in range(np.size(trial_dF,0)):
        # grab dF trace
        cur_sweep = dF_ds[trial_dF[i,0]:trial_dF[i,1],roi]
        cur_trial_event_idx[i] = plot_events[i,0] - trial_dF[i,0]
        trace_start = int(window_center - cur_trial_event_idx[i])
        cur_trial_dF[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
        cur_trial_speed[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF[i,0]:trial_dF[i,1],3]
        # only consider roi's max dF value in a given trial if it exceeds threshold
        if np.amax(cur_sweep) > trial_std_threshold * roi_std:
            cur_trial_max_idx = np.append(cur_trial_max_idx,np.nanargmax(cur_trial_dF[i,:]))
        if np.amax(cur_sweep) > max_y:
            max_y = np.amax(cur_sweep)

    # plot individual traces
    if make_figure:
        for i,ct in enumerate(cur_trial_dF):
            ax_object.plot(ct,c=plot_color,lw=1)
    # ax_object_1 = ax_object.twinx()
    if len(cur_trial_max_idx) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction = np.float64(len(cur_trial_max_idx)/np.size(trial_dF,0))
        if make_figure:
            sns.distplot(cur_trial_max_idx,hist=False,kde=False,rug=True,color=plot_color,ax=ax_object)
        roi_std = np.std(cur_trial_max_idx)
    else:
        roi_active_fraction = np.int64(-1)
        roi_std = np.int64(-1)

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(cur_trial_dF.T):
        if np.count_nonzero(np.isnan(trace))/len(trace) < 0.5:
            mean_valid_indices.append(i)
    if make_figure:
        ax_object.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
        ax_object.axvline(window_center,c='r',lw=2)
    roi_meanpeak = np.nanmax(np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanmin = np.nanmin(np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_idx = np.nanargmax(np.nanmean(cur_trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_time = ((roi_meanpeak_idx+mean_valid_indices[0])-window_center) * frame_latency

    if len(peak_times) > 0:
        window_center_time = window_center * frame_latency
        vr_peak_time = peak_times[0] + window_center_time
        vr_peak_time_idx = (vr_peak_time/frame_latency).astype('int')
        roi_meanpeak = np.nanmean(cur_trial_dF,0)[vr_peak_time_idx]
        if make_figure:
            ax_object.axvline(vr_peak_time_idx)
    else:
        if make_figure:
            ax_object.axvline((roi_meanpeak_idx+mean_valid_indices[0]))

    if make_figure:
        sns.heatmap(cur_trial_dF,cmap='viridis',vmax=max_y,yticklabels=plot_events[:,1].astype('int'),xticklabels=False,ax=ax_object2,cbar=False)
    #     sns.heatmap(cur_trial_dF_long,cmap='viridis',vmin=hmmin,vmax=hmmax,yticklabels=events_long[:,1].astype('int'),xticklabels=False,ax=ax4)
        sns.heatmap(cur_trial_speed,cmap='viridis',vmin=0,vmax=60,yticklabels=plot_events[:,1].astype('int'),xticklabels=False,ax=ax_object3,cbar=False)
    #     sns.heatmap(cur_trial_speed_long,cmap='viridis',vmin=0,vmax=60,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax6)
    #     ax3.axvline(window_center,c='r',lw=2)
    #     ax4.axvline(window_center,c='r',lw=2)
    #     ax5.axvline(window_center,c='r',lw=2)
    #     ax6.axvline(window_center,c='r',lw=2)

        # ax1.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)
        # ax2.axhline(roi_std*trial_std_threshold,c='0.8',ls='--',lw=1)


    return t_max, max_y, roi_std, roi_active_fraction, roi_meanpeak, roi_meanpeak_time

def fig_ordered_eventdiff(h5path, sess, roi, fname, eventshort, eventlong, order_short, order_long, max_timewindow, blackbox_time, ylims=[], fformat='png', subfolder=[], peak_times=[], filter=False, make_figure=True):
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    if filter==True:
        order = 6
        fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
        cutoff = 5 # desired cutoff frequency of the filter, Hz
        dF_ds[:,roi] = butter_lowpass_filter(dF_ds[:,roi], cutoff, fs, order)
    # threshold the response of a roi in a given trial has to exceed count its response toward the tuning of the cell
    trial_std_threshold = 3
    # on which fraction of trials did the roi exceed 3 standard deviations
    roi_active_fraction_short = np.int64(0)
    roi_active_fraction_long = np.int64(0)

    # specify track numbers
    track_short = 3
    track_long = 4

    # ylims
    min_y = -0.3
    max_y = np.amax(dF_ds[:,roi])
    if not ylims == []:
        min_y = ylims[0]
        max_y = ylims[1]

    # create figure and axes to later plot on
    if make_figure:
        fig = plt.figure(figsize=(24,12))
        ax1 = plt.subplot(321)
        ax2 = plt.subplot(322)
        ax3 = plt.subplot(323)
        ax4 = plt.subplot(324)
        ax5 = plt.subplot(325)
        ax6 = plt.subplot(326)

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
    else:
        ax1 = None
        ax2 = None
        ax3 = None
        ax4 = None
        ax5 = None
        ax6 = None

    # get indices of desired behavioural event
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    if order_short is not None:
        ordered_trial_list = order_trials(behav_ds, trials_short, order_short)
    slow_fraction = 0.5
    fast_fraction = 0.5
    num_short_trials = ordered_trial_list.shape[0]
    slow_trials = ordered_trial_list[0:np.floor(num_short_trials*slow_fraction).astype(int)]
    fast_trials = ordered_trial_list[np.ceil(num_short_trials*fast_fraction).astype(int):]

    unordered_events_slow = event_ind(behav_ds, eventshort, slow_trials[:,1])
    unordered_events_fast = event_ind(behav_ds, eventshort, fast_trials[:,1])

    events_slow = np.zeros((unordered_events_slow.shape))
    for i,row in enumerate(unordered_events_slow):
        events_slow[(np.where(slow_trials[:,1] == row[1])[0]), :] = row

    events_fast = np.zeros((unordered_events_fast.shape))
    for i,row in enumerate(unordered_events_fast):
        events_fast[(np.where(fast_trials[:,1] == row[1])[0]), :] = row
    # SHORT TRIALS
    plot_color = '#7EC4ED'
    t_max_slow, max_y_slow, roi_std_short_slow, roi_active_fraction_short_slow, roi_meanpeak_short_slow, roi_meanpeak_short_time_slow = plot_traces(events_slow, behav_ds, dF_ds, roi, trial_std_threshold, max_y, max_timewindow, blackbox_time, peak_times, False, plot_color, ax1,ax3,ax5)
    plot_color = '#FF73F6'
    t_max_fast, max_y_fast, roi_std_short_fast, roi_active_fraction_short_fast, roi_meanpeak_short_fast, roi_meanpeak_short_time_fast = plot_traces(events_fast, behav_ds, dF_ds, roi, trial_std_threshold, max_y, max_timewindow, blackbox_time, peak_times, False, plot_color, ax2,ax4,ax6)

    # get indices of desired behavioural event
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])
    if order_long is not None:
        ordered_trial_list = order_trials(behav_ds, trials_long, order_long)

    num_long_trials = ordered_trial_list.shape[0]
    slow_trials = ordered_trial_list[0:np.floor(num_long_trials*slow_fraction).astype(int)]
    fast_trials = ordered_trial_list[np.ceil(num_long_trials*fast_fraction).astype(int):]

    unordered_events_slow = event_ind(behav_ds, eventlong, slow_trials[:,1])
    unordered_events_fast = event_ind(behav_ds, eventlong, fast_trials[:,1])

    events_slow = np.zeros((unordered_events_slow.shape))
    for i,row in enumerate(unordered_events_slow):
        events_slow[(np.where(slow_trials[:,1] == row[1])[0]), :] = row

    events_fast = np.zeros((unordered_events_fast.shape))
    for i,row in enumerate(unordered_events_fast):
        events_fast[(np.where(fast_trials[:,1] == row[1])[0]), :] = row


    # LONG TRIALS
    plot_color = '#7EC4ED'
    t_max_slow, max_y_slow, roi_std_long_slow, roi_active_fraction_long_slow, roi_meanpeak_long_slow, roi_meanpeak_long_time_slow = plot_traces(events_slow, behav_ds, dF_ds, roi, trial_std_threshold, max_y, max_timewindow, blackbox_time, peak_times, True, plot_color, ax1,ax3,ax5)
    plot_color = '#FF73F6'
    t_max_fast, max_y_fast, roi_std_long_fast, roi_active_fraction_long_fast, roi_meanpeak_long_fast, roi_meanpeak_long_time_fast = plot_traces(events_fast, behav_ds, dF_ds, roi, trial_std_threshold, max_y, max_timewindow, blackbox_time, peak_times, True, plot_color, ax2,ax4,ax6)
    if ylims != []:
        hmmin = 0
        hmmax = ylims[1]
    else:
        hmmin = 0
        hmmax = max_y

    # mean_amplitude_short = roi_meanpeak_short - roi_meanmin_short
    # mean_amplitude_long = roi_meanpeak_long - roi_meanmin_long

    if make_figure:
        ax1.set_title(str(np.round(roi_std_long_slow,2)) + ' active: ' + str(np.round(roi_active_fraction_long_slow,2)) + ' peak: ' + str(np.round(roi_meanpeak_long_slow,2)), fontsize=32)
        ax2.set_title(str(np.round(roi_std_long_fast,2)) + ' active: ' + str(np.round(roi_active_fraction_long_fast,2)) + ' peak: ' + str(np.round(roi_meanpeak_long_fast,2)), fontsize=32)
        ax3.set_title('dF/F vs time SHORT track - heatmap')
        ax4.set_title('dF/F vs time LONG track - heatmap')

        ax1.set_ylim([min_y,max_y])
        ax2.set_ylim([min_y,max_y])
        ax1.set_xlim([0,t_max_slow])
        ax2.set_xlim([0,t_max_slow])

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
    norm_value = np.amax(dF_ds[:,roi])
    # print(type(roi_std_short),type(roi_std_long), type(roi_active_fraction_short), type(roi_active_fraction_long), type(roi_meanpeak_short), type(roi_meanpeak_long), type(roi_meanpeak_short_time), type(roi_meanpeak_long_time), [np.float32(min_y).item(),np.float32(min_y)], type(norm_value.item()))
    # print(roi_active_fraction_short,roi_active_fraction_long)
    return [roi_std_short_slow.item(),roi_std_short_fast.item()], [roi_std_long_slow.item(), roi_std_long_fast.item()], \
           [roi_active_fraction_short_slow.item(), roi_active_fraction_short_fast.item()], [roi_active_fraction_long_slow.item(), roi_active_fraction_long_fast.item()], \
           [roi_meanpeak_short_slow.item(), roi_meanpeak_short_fast.item()], [roi_meanpeak_long_slow.item(), roi_meanpeak_long_fast.item()], \
           [roi_meanpeak_short_time_slow.item(), roi_meanpeak_short_time_fast.item()], [roi_meanpeak_long_time_slow.item(), roi_meanpeak_long_time_fast.item()], \
           [], norm_value.item()

def run_slowfast_analysis(mousename, sessionname, sessionname_openloop, roi_selection, h5_filepath, json_path, subname, sess_subfolder, filterprop_short, filterprop_long, orderprop_short, orderprop_long, even_win, blackbox_win, session_rois):
    """ set up function call and dictionary to collect results """
    MOUSE = mousename
    SESSION = sessionname
    h5path = h5_filepath
    SUBNAME = subname
    subfolder = sess_subfolder + '_slowvfast'
    # write result to dictionary
    write_to_dict = True
    # create figure (WARNING: significantly slows down execution)
    make_figure = True

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

        std_short, std_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, ylims, norm_value = fig_ordered_eventdiff(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), filterprop_short, filterprop_long, orderprop_short,orderprop_long, even_win, blackbox_win, [], fformat, subfolder, [], False, make_figure)
        # session_rois[SUBNAME+'_slow_roi_number'].append(r)
        session_rois[SUBNAME+'_slow_std_short'].append(std_short[0])
        session_rois[SUBNAME+'_slow_std_long'].append(std_long[0])
        session_rois[SUBNAME+'_slow_active_short'].append(active_short[0])
        session_rois[SUBNAME+'_slow_active_long'].append(active_long[0])
        session_rois[SUBNAME+'_slow_peak_short'].append(peak_short[0])
        session_rois[SUBNAME+'_slow_peak_long'].append(peak_long[0])
        session_rois[SUBNAME+'_slow_peak_time_short'].append(meanpeak_short_time[0])
        session_rois[SUBNAME+'_slow_peak_time_long'].append(meanpeak_long_time[0])

        session_rois[SUBNAME+'_fast_std_short'].append(std_short[1])
        session_rois[SUBNAME+'_fast_std_long'].append(std_long[1])
        session_rois[SUBNAME+'_fast_active_short'].append(active_short[1])
        session_rois[SUBNAME+'_fast_active_long'].append(active_long[1])
        session_rois[SUBNAME+'_fast_peak_short'].append(peak_short[1])
        session_rois[SUBNAME+'_fast_peak_long'].append(peak_long[1])
        session_rois[SUBNAME+'_fast_peak_time_short'].append(meanpeak_short_time[1])
        session_rois[SUBNAME+'_fast_peak_time_long'].append(meanpeak_long_time[1])

    if write_to_dict:
        # print('writing to dict')
        # print(type(session_rois['reward_slow_peak_time_short']))
        # print(session_rois)
        write_dict(MOUSE, SESSION, session_rois)

def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    roi_selection = 'valid' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)


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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)


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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)


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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)

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

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    SUBNAME = 'reward'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], orderprop_short, orderprop_long, [20,0], [2,2], roi_result_params)

    SUBNAME = 'trialonset'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['trial_transition'], ['trial_transition'], orderprop_short, orderprop_long, [0,20], [2,2], roi_result_params)

    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)


def do_single():
    MOUSE = 'LF170222_1'
    SESSION = 'Day201776'
    SESSION_OPENLOOP = 'Day201776_openloop'
    roi_selection = [9]
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    filterprop_short = ['at_location', 220]
    filterprop_long = ['at_location', 220]
    even_win = [10,10]
    blackbox_win = [2,2]
    SUBNAME = 'lmcenter'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    subfolder_ol = subfolder + '_openloop'
    fformat = 'png'

    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    orderprop_short = ['time_between_points', ['at_location', 220],['rewards_all', -1]]
    orderprop_long = ['time_between_points',['at_location', 220],['rewards_all', -1]]

    run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path, SUBNAME, subfolder, ['at_location', 220], ['at_location', 220], orderprop_short, orderprop_long, [10,10], [2,2], roi_result_params)
    # run_slowfast_analysis(MOUSE, SESSION, SESSION_OPENLOOP, roi_selection, h5path, json_path,  SUBNAME, subfolder, ['rewards_all', -1], ['rewards_all', -1], [20,0], [2,2], roi_result_params)


if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inlin
    # run_LF170613_1_Day20170804()
    # run_LF170421_2_Day20170719()

    # run_LF170110_2_Day201748_1()
    # run_LF170110_2_Day201748_2()
    # run_LF170110_2_Day201748_3()
    # run_LF170420_1_Day2017719()
    # run_LF170420_1_Day201783()
    # -->
    run_LF170222_1_Day201776()
    run_LF170421_2_Day2017720()
    # run_LF171211_1_Day2018321_2()
    # run_LF171212_2_Day2018218_1()
    # run_LF171212_2_Day2018218_2()
    # do_single()


### DUMP
    # # get indices of trial start and end (or max_event_timewindow if trial is long)
    # trial_dF_long = np.zeros((np.size(events_long[:,0]),2))
    # for i,cur_ind in enumerate(events_long):
    #     cur_trial_idx = [np.where(behav_ds[:,6] == cur_ind[1])[0][0],np.where(behav_ds[:,6] == cur_ind[1])[0][-1]]
    #     # # determine indices of beginning and end of timewindow
    #     if cur_ind[0] - max_timewindow_idx[0] > cur_trial_idx[0]:
    #         if cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0]) < 0:
    #             trial_dF_long[i,0] = 0
    #         else:
    #             trial_dF_long[i,0] = cur_ind[0] - (max_timewindow_idx[0] + blackbox_idx[0])
    #     else:
    #         if cur_trial_idx[0] - blackbox_idx[0] < 0:
    #             trial_dF_long[i,0] = 0
    #         else:
    #             trial_dF_long[i,0] = cur_trial_idx[0] - blackbox_idx[0]
    #
    #     if cur_ind[0] + max_timewindow_idx[1] < cur_trial_idx[1]:
    #         trial_dF_long[i,1] = cur_ind[0] + (max_timewindow_idx[1] + blackbox_idx[1])
    #     else:
    #         if cur_trial_idx[1] + blackbox_idx[1] > np.size(behav_ds,0):
    #             trial_dF_long[i,1] = np.size(behav_ds,0)
    #         else:
    #             trial_dF_long[i,1] = cur_trial_idx[1] + blackbox_idx[1]
    #
    # trial_dF_long = trial_dF_long.astype(int)
    # # grab dF data for each trial
    # cur_trial_dF_long = np.full((np.size(events_long[:,0]),int(t_max)),np.nan)
    # cur_trial_speed_long = np.full((np.size(events_long[:,0]),int(t_max)),np.nan)
    # cur_trial_event_idx = np.zeros(np.size(events_long[:,0]))
    # cur_trial_max_idx_long = np.empty(0)
    # for i in range(np.size(trial_dF_long,0)):
    #     # grab dF trace
    #     cur_sweep = dF_ds[trial_dF_long[i,0]:trial_dF_long[i,1],roi]
    #     cur_trial_event_idx[i] = events_long[i,0] - trial_dF_long[i,0]
    #     trace_start = int(window_center - cur_trial_event_idx[i])
    #     cur_trial_dF_long[i,trace_start:trace_start+len(cur_sweep)] = cur_sweep
    #     cur_trial_speed_long[i,trace_start:trace_start+len(cur_sweep)] = behav_ds[trial_dF_long[i,0]:trial_dF_long[i,1],3]
    #     if np.amax(cur_sweep) > trial_std_threshold * roi_std:
    #         cur_trial_max_idx_long = np.append(cur_trial_max_idx_long,np.nanargmax(cur_trial_dF_long[i,:]))
    #     if np.amax(cur_sweep) > max_y:
    #         max_y = np.amax(cur_sweep)
    #
    # # plot traces
    # if make_figure:
    #     for i,ct in enumerate(cur_trial_dF_long):
    #         ax2.plot(ct,c='0.65',lw=1)
    #
    # if len(cur_trial_max_idx_long) >= MIN_ACTIVE_TRIALS:
    #     roi_active_fraction_long = np.float64(len(cur_trial_max_idx_long)/np.size(trial_dF_long,0))
    #     if make_figure:
    #         sns.distplot(cur_trial_max_idx_long,hist=False,kde=False,rug=True,ax=ax2)
    #     roi_std_long = np.std(cur_trial_max_idx_long)
    # else:
    #     roi_active_fraction_long = np.int64(-1)
    #     roi_std_long = np.int64(-1)
    #
    # # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    # mean_valid_indices = []
    # for i,trace in enumerate(cur_trial_dF_long.T):
    #     if np.count_nonzero(np.isnan(trace))/len(trace) < 0.5:
    #         mean_valid_indices.append(i)
    # if make_figure:
    #     ax2.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1),np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0),c='k',lw=2)
    #     ax2.axvline(window_center,c='r',lw=2)
    # roi_meanpeak_long = np.nanmax(np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    # roi_meanmin_long = np.nanmin(np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    # roi_meanpeak_long_idx = np.nanargmax(np.nanmean(cur_trial_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    # roi_meanpeak_long_time = ((roi_meanpeak_long_idx+mean_valid_indices[0])-window_center) * frame_latency
    #
    # if len(peak_times) > 0:
    #     window_center_time = window_center * frame_latency
    #     vr_peak_time_long = peak_times[1] + window_center_time
    #     vr_peak_time_long_idx = (vr_peak_time_long/frame_latency).astype('int')
    #     roi_meanpeak_long = np.nanmean(cur_trial_dF_long,0)[vr_peak_time_long_idx]
    #     if make_figure:
    #         ax2.axvline(vr_peak_time_long_idx)
    # else:
    #     if make_figure:
    #         ax2.axvline((roi_meanpeak_long_idx+mean_valid_indices[0]))
