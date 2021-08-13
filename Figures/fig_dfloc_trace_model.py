"""
Plot trace of an individual ROI vs location

@author: lukasfischer

"""

import h5py, os, sys, traceback, yaml,matplotlib, json
from multiprocessing import Process
import warnings; # warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import matplotlib.cbook
import numpy as np
from scipy import stats
import scipy.io as sio
import seaborn as sns
sns.set_style("white")

plt.rcParams['svg.fonttype'] = 'none'

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
#
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT/Analysis')
# os.chdir(r'C:\Users\Lou\Documents\repos\LNT')
# sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Analysis")
# sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Imaging")
# sys.path.append(r'C:\Users\Lou\Documents\repos\OASIS-master')

sys.path.append(loc_info['base_dir'] + 'Analysis')
sys.path.append(loc_info['base_dir'] + 'Imaging')

figure_output_path = 'C:/Users/lfisc/Work\Projects/Lntmodel/analysis_output'

from filter_trials import filter_trials
from write_dict import write_dict
from event_ind import event_ind
from rewards import rewards
from licks import licks

fformat = 'png'
plt.ioff()
SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'

RUN_LM = True
RUN_TO = False
RUN_GOODBADTRIALS = False

# +/- cm tolerance within which we look for the maximum mean trace value relative to the peak location in VR (i.e. this is only relevant for openloop analysis)
OPENLOOP_PEAK_TOLERANCE = 20
# fraction of trials that have to be present for a given location to be included in the mean trace
MEAN_TRACE_FRACTION = 0.5
# amount of blackbox after a reward to be added to each trial
PRE_TRIAL_TIME = 1.5
REWARD_TIME = 1.5

# percentile of first lickes furthest and closest to the start of the reward zone
FL_THRESHOLD_DISTANCE_PERCENTILE = 33

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


def calc_spacevloc(behav_ds, roi_dF, included_trials, align_point, binnr, maxlength, is_ol_sess=False):
    # figure out which speed column to use. If there is no column 8, then use column 3. If there is col 8 but its filled with -1, also use column 8
    # number_of_shuffles = 10
    if is_ol_sess:
        ol_speed_col = 8
    else:
        ol_speed_col = 3
    try:
        x = behav_ds[0,ol_speed_col]
    except IndexError:
        ol_speed_col = 3

    # if behav_ds[5,8] == -1:
    #     ol_speed_col = 3

    if align_point == 'trialonset':
        start_loc = -10
    else:
        start_loc = 0

    # this may be counterintuitive, but when we align by landmark we don't care too much about the very first bin, aligned to trialonset through
    # we want to skip it to avoid artifactual values at the VERY beginning
    bin_edges = np.linspace(start_loc, maxlength+start_loc, binnr+1)

    # intilize matrix to hold data for all trials of this roi
    mean_dF_trials = np.zeros((np.size(included_trials,0),binnr))
    mean_zscore_trials = np.zeros((np.size(included_trials,0),binnr))
    mean_speed_trials = np.zeros((np.size(included_trials,0),binnr))
    # calculate mean dF vs location for each ROI on short trials

    for k,t in enumerate(included_trials):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        # get indeces for the first <REWARD_TIME> sec after a reward
        cur_trial_rew_loc_idx = np.where((behav_ds[:,0] > behav_ds[behav_ds[:,6]==t,0][-1]) & (behav_ds[:,0] < (behav_ds[behav_ds[:,6]==t,0][-1]+REWARD_TIME)))[0]
        cur_trial_rew_loc = behav_ds[cur_trial_rew_loc_idx,1]

        # get indeces for <PRE_TRIAL_TIME> sec prior to trial onset
        cur_trial_pretrial_loc_idx = np.where((behav_ds[:,0] < behav_ds[behav_ds[:,6]==t,0][0]) & (behav_ds[:,0] > (behav_ds[behav_ds[:,6]==t,0][0]-PRE_TRIAL_TIME)))[0]
        cur_trial_pretrial_loc = behav_ds[cur_trial_pretrial_loc_idx,1]

        # subtract starting location from location samples to align all to trial start
        if align_point == 'trialonset':
            cur_trial_rew_loc = cur_trial_rew_loc - cur_trial_loc[0]
            cur_trial_loc = cur_trial_loc - cur_trial_loc[0]

        if np.size(cur_trial_pretrial_loc) > 0:
            if align_point == 'landmark':
                cur_trial_pretrial_loc = cur_trial_pretrial_loc - cur_trial_pretrial_loc[-1] + cur_trial_loc[0]

            elif align_point == 'trialonset':
                cur_trial_pretrial_loc = cur_trial_pretrial_loc - cur_trial_pretrial_loc[-1]

        cur_trial_dF_roi = roi_dF[behav_ds[:,6]==t]
        cur_trial_speed = behav_ds[behav_ds[:,6]==t,ol_speed_col]

        if np.size(cur_trial_rew_loc) > 0:
            cur_trial_loc = np.append(cur_trial_loc, cur_trial_rew_loc)
            cur_trial_dF_roi = np.append(cur_trial_dF_roi, roi_dF[cur_trial_rew_loc_idx])
            cur_trial_speed = np.append(cur_trial_speed, behav_ds[cur_trial_rew_loc_idx,ol_speed_col])

        if np.size(cur_trial_pretrial_loc) > 0:
            cur_trial_loc = np.insert(cur_trial_loc, 0, cur_trial_pretrial_loc)
            cur_trial_dF_roi = np.insert(cur_trial_dF_roi, 0, roi_dF[cur_trial_pretrial_loc_idx])
            cur_trial_speed = np.insert(cur_trial_speed, 0, behav_ds[cur_trial_pretrial_loc_idx,ol_speed_col])

        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges, (start_loc, maxlength+start_loc))[0]
        mean_speed_trial = stats.binned_statistic(cur_trial_loc, cur_trial_speed, 'mean', bin_edges, (start_loc, maxlength+start_loc))[0]
        # mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
        mean_dF_trials[k,:] = mean_dF_trial
        mean_speed_trials[k,:] = mean_speed_trial
        #print(mean_dF_trial)

    # # now shuffle roi_dF data around on a per trial basis
    # shuffled_dF = np.empty((number_of_shuffles, len(included_trials), binnr))
    # shuffled_dF.fill(np.NaN)
    # for n in range(number_of_shuffles):
    #     shuffled_dF[n, :, :] = mean_dF_trials
    #     # data is only shuffled within spatial bins that previously contained data
    #     for t in range(len(included_trials)):
    #         trial_indices = ~np.isnan(mean_dF_trials[t, :])
    #         shuffled_dF[n, t, trial_indices] = np.roll(mean_dF_trials[t, trial_indices], np.random.randint(0, len(mean_dF_trials[t, trial_indices])))
    #
    # for t in range(len(included_trials)):
    #     for b in range(binnr):
    #         #
    #         mean_zscore_trials[t,b] = (mean_dF_trials[t,b]-np.nanmean(shuffled_dF[:,:,b])) / np.nanstd(shuffled_dF[:,:,b])

    return mean_dF_trials, mean_speed_trials

def response_max(behav_ds, roi_data, trials, align_point, binnr, tracklength, mean_valid_indices, peak_locs, mean_trace, bin_size, peak_tolerance, transient_threshold, max_peak_distance, plot_axis, make_figure, plot_col='0.5', is_ol_sess=False):
    """ determine the location and amplitude of peak response in individual  """
    mean_dF, running_speed = calc_spacevloc(behav_ds, roi_data, trials, align_point, binnr, tracklength, is_ol_sess)
    # for t in mean_dF:
    #     if make_figure:
    #         plot_axis.plot(t,c='0.8')

    meanpeak_transient_distance = []
    meanpeak_transient_peak = []
    for tn,t in enumerate(mean_dF):
        if make_figure:
            plot_axis.plot(t,c='0.8',zorder=2)
        # high_bins = np.where(mean_zscore_trials_short[tn,:] > 1)[0]
        high_bins = np.where(mean_dF[tn,:] > (transient_threshold))[0]
        if len(high_bins) > 0:
            # split high_bins if it exceeds threshold multiple times
            trial_transients = []
            trial_peaks = []
            for hb in np.split(high_bins, np.where(np.diff(high_bins)>2)[0]+1):

                if len(hb) > 2:
                    hb = np.insert(hb,0,hb[0]-1)
                    hb = np.insert(hb,len(hb),hb[-1]+1)
                    if make_figure:
                        plot_axis.plot(hb-1, t[hb-1], c=plot_col,zorder=3)
                    trial_transients = np.append(trial_transients,hb[np.nanargmax(t[hb-1])])
                    trial_peaks = np.append(trial_peaks,np.nanmax(t[hb-1]))
            if len(trial_transients) > 0 and len(trial_peaks) > 0:
                meanpeak_transient_distance.append(trial_transients)
                meanpeak_transient_peak.append(trial_peaks)

    filtered_mean_trace = np.nanmean(mean_dF[:,mean_valid_indices[0]:mean_valid_indices[-1]],axis=0)
    if len(peak_locs) == 0:
        filtered_peak = np.nanmax(filtered_mean_trace)
        meanpeak_idx = np.nanargmax(filtered_mean_trace)
    else:
        # print(meanpeak_idx)
        meanpeak_idx_vr = np.int64(peak_locs[0]/bin_size) - mean_valid_indices[0]
        if peak_tolerance > 0:
            try:
                win_start = meanpeak_idx_vr-peak_tolerance
                win_end = meanpeak_idx_vr+peak_tolerance
                win_start_offset = 0
                win_end_offset = 0
                if win_start < 0:
                    win_start = 0
                    win_start_offset = np.abs(meanpeak_idx_vr-peak_tolerance)
                if win_end > mean_trace.shape[0]:
                    win_end = mean_trace.shape[0]
                    win_end_offset = (meanpeak_idx_vr+peak_tolerance)-mean_trace.shape[0]
                meanpeak_idx = meanpeak_idx_vr + (np.nanargmax(filtered_mean_trace[win_start:win_end])-peak_tolerance+win_start_offset-win_end_offset)
                # meanpeak_idx = meanpeak_idx_vr + (np.nanargmax(filtered_mean_trace[meanpeak_idx_vr-peak_tolerance:meanpeak_idx_vr+peak_tolerance])-peak_tolerance)
            except ValueError:
                meanpeak_idx = meanpeak_idx_vr
        else:
            meanpeak_idx = meanpeak_idx_vr
        if (meanpeak_idx+mean_valid_indices[0]) >= filtered_mean_trace.shape[0]:
            filtered_peak = filtered_mean_trace[-1]
        else:
            filtered_peak = filtered_mean_trace[meanpeak_idx]
    if make_figure:
        plot_axis.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), filtered_mean_trace,c='k',ls='-',lw=2,zorder=4) #'#00AAAA'
        plot_axis.axvline((meanpeak_idx+mean_valid_indices[0]),c='b')

    # determine which transient peaks are within the specified window
    RF_transients = []
    RF_transients_residuals = []
    RF_transients_peak = []
    for j,mtd in enumerate(meanpeak_transient_distance):

        # translate bin number of mean peak into actual distances (cm)
        meanpeak_loc = (meanpeak_idx+mean_valid_indices[0]) * bin_size
        mtd = mtd * bin_size
        # pull out peak values associated with transient_trial_start_idx
        mtp = meanpeak_transient_peak[j]
        # determine which transient peaks are within range
        mtp = mtp[np.where((mtd > (meanpeak_loc-max_peak_distance)) & (mtd < (meanpeak_loc+max_peak_distance)))]
        mtd = mtd[np.where((mtd > (meanpeak_loc-max_peak_distance)) & (mtd < (meanpeak_loc+max_peak_distance)))]
        RF_transients.append(mtd)
        RF_transients_residuals.append(mtd-meanpeak_loc)
        RF_transients_peak.append(mtp)

    # flatten list and calculate deviance paraameters
    RF_transients_residuals = [rftr for sublist in RF_transients_residuals for rftr in sublist]
    RF_transients = [rftr for sublist in RF_transients for rftr in sublist]
    RF_transients_peak = [rfpr for sublist in RF_transients_peak for rfpr in sublist]
    sns.rugplot(np.array(RF_transients)/bin_size,ax=plot_axis)
    mean_transient_distance = np.mean(RF_transients_residuals)
    mean_transient_std = np.std(RF_transients_residuals)
    mean_transient_sem = stats.sem(RF_transients_residuals)
    num_RF_transients = len(RF_transients)/len(trials)
    mean_transient_peak = np.mean(RF_transients_peak)

    return filtered_peak, filtered_mean_trace, mean_dF, running_speed, mean_transient_sem, num_RF_transients, mean_transient_peak

def set_empty_axis(ax_object):
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_visible(False)
    ax_object.spines['bottom'].set_visible(False)
    ax_object.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        length=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    return ax_object

def plot_first_licks(ax, licks, rewards, trials, scatter_color, trial_type, plot_to_axis=True):
    # plot location of first trials on short and long trials
    if plot_to_axis:
        ax.set_ylabel('Trial #')
        ax.axvspan(200,240,color='0.9',zorder=0,lw=0)
        if trial_type == 'short':
            ax.axvspan(320,340,color=scatter_color,zorder=0,alpha=0.2,lw=0)
            ax.set_xlim([0,340])
            ax.set_xticks([0,100,200,300])
            ax.set_xticklabels(['0','100','200','300'])
        elif trial_type == 'long':
            ax.axvspan(380,400,color=scatter_color,zorder=0,alpha=0.2,lw=0)
            ax.set_xlim([0,400])
            ax.set_xticks([0,100,200,300,400])
            ax.set_xticklabels(['0','100','200','300','400'])
        ax.set_xlabel('Location (cm)')

    first_lick = np.empty((0,4))
    first_lick_trials = np.empty((0))
    rewards[:,3] = rewards[:,3] - 1
    default_rewards = np.empty((0,4))
    default_rewards_trials = np.empty((0))
    for r in trials:
        if licks.size > 0:
            licks_all = licks[licks[:,2]==r,:]
            licks_all = licks_all[licks_all[:,1]>150,:]
            # if r == 79:
            #     ipdb.set_trace()
            if licks_all.size == 0:

                rew_lick = rewards[rewards[:,3]==r,:]
                if rew_lick.size > 0:
                    if rew_lick[0,5] == 1:
                        licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                        if trial_type == 'short':
                            licks_all[0,1] = licks_all[0,1] - 320
                        elif trial_type == 'long':
                            licks_all[0,1] = licks_all[0,1] - 380
                        first_lick = np.vstack((first_lick, licks_all[0,:].T))
                        first_lick_trials = np.append(first_lick_trials, r)
                    if rew_lick[0,5] == 2:
                        default_rewards = np.vstack((default_rewards, np.asarray([[rew_lick[0,4], 18, rew_lick[0,3], rew_lick[0,2]]])[0,:].T))
                        default_rewards_trials = np.append(default_rewards_trials, r)
            else:
                if licks_all[0,3] == 3:
                    licks_all = licks_all[licks_all[:,1]<360,:]
                    licks_all[0,1] = licks_all[0,1] - 320
                elif licks_all[0,3] == 4:
                    licks_all = licks_all[licks_all[:,1]<440,:]
                    licks_all[0,1] = licks_all[0,1] - 380

                first_lick = np.vstack((first_lick, licks_all[0,:].T))
                first_lick_trials = np.append(first_lick_trials, r)

    if plot_to_axis:
        if first_lick.size > 0:
            if trial_type == 'short':
                ax.scatter(first_lick[:,1]+320,first_lick_trials,c='k',lw=0)
            elif trial_type == 'long':
                ax.scatter(first_lick[:,1]+380,first_lick_trials,c='k',lw=0)
        if default_rewards.size > 0:
            if trial_type == 'short':

                ax.scatter(default_rewards[:,1]+320,default_rewards_trials,c='r',lw=0)
            elif trial_type == 'long':
                ax.scatter(default_rewards[:,1]+380,default_rewards_trials,c='r',lw=0)

        ax.set_yticklabels([''])

    return first_lick, first_lick_trials, default_rewards, default_rewards_trials

def calc_fl_correlation(mean_dF_short, trials_short, fl_short, fl_trials_short, transient_threshold):
    """ calculate transient parameters based on first lick distance from reward zone onset """
    # meanpeak_transient_distance_short = []
    # meanpeak_transient_peak_short = []
    max_peaks = np.full((len(fl_trials_short)), np.nan)
    max_peaks_trials = np.full((len(fl_trials_short)), np.nan)
    num_transients = np.zeros((len(fl_trials_short),))


    for i,tn in enumerate(fl_trials_short.astype('int')):
        mdf_idx = np.where(trials_short == tn)[0][0]

        high_bins = np.where(mean_dF_short[mdf_idx,:] > (transient_threshold))[0]
        if len(high_bins) > 0:
            # print('transients')
            # split high_bins if it exceeds threshold multiple times
            # trial_transients = []
            trial_peaks = []
            for hb in np.split(high_bins, np.where(np.diff(high_bins)>2)[0]+1):
                if len(hb) > 2:
                    hb = np.insert(hb,0,hb[0]-1)
                    hb = np.insert(hb,len(hb),hb[-1]+1)
                    # ax1.plot(hb-1, t[hb-1], c=SHORT_COLOR,lw=1, zorder=3)
                    # trial_transients = np.append(trial_transients,hb[np.nanargmax(mean_dF_short[mdf_idx,:][hb])])
                    # ipdb.set_trace()
                    trial_peaks = np.append(trial_peaks,np.nanmax(mean_dF_short[mdf_idx,:][hb-1]))

            if len(trial_peaks) > 0:
                # determine largest transient peak
                max_peaks[i] = np.amax(trial_peaks)
                max_peaks_trials[i] = tn
                # meanpeak_transient_distance_short.append(trial_transients)
                # meanpeak_transient_peak_short.append(trial_peaks)

            num_transients[i] = np.float(len(trial_peaks))
        else:
            num_transients[i] = np.float(0)

    return max_peaks, num_transients, max_peaks_trials

def run_goodvbadtrials_figure(mousename, sess, number_of_rois, h5path, json_path, subfolder, load_raw=False, use_data='aligned_data.mat', suffix=''):
    print('Running good vs bad trials...')
    # run analysis for vr session
    if type(number_of_rois) is int:
        roilist = range(number_of_rois)
    else:
        roilist = number_of_rois

    # if we want to run through all the rois, just say all
    if number_of_rois == 'all' and load_raw == False:
        h5dat = h5py.File(h5path, 'r')
        dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
        h5dat.close()
        roilist = np.arange(0,dF_ds.shape[1],1).tolist()
        # write_to_dict = True
        print('number of rois: ' + str(number_of_rois))
    elif number_of_rois == 'all' and load_raw == True:
        processed_data_path = h5path + os.sep + SESSION + os.sep + use_data
        loaded_data = sio.loadmat(processed_data_path)
        behaviour_aligned = loaded_data['behaviour_aligned']
        dF_aligned = loaded_data['calcium_dF']
        roilist = np.arange(0,dF_aligned.shape[1],1).tolist()
    elif number_of_rois == 'valid':
        # only use valid rois
        with open(json_path, 'r') as f:
            sess_dict = json.load(f)
        roilist = sess_dict['valid_rois']
        print('analysing ' + number_of_rois + ' rois: ' + str(roilist))
    else:
        print('analysing custom list of rois: ' + str(roilist))

    if load_raw == False:
        h5dat = h5py.File(h5path, 'r')
        behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
        dF_ds = np.copy(h5dat[sess + '/dF_win'])
        licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
        reward_ds = np.copy(h5dat[sess + '/rewards'])
        h5dat.close()

    track_short = 3
    track_long = 4
    track_dark = 5

    bin_size = 5

    tracklength_short = 400
    tracklength_long = 500
    maxdistance_dark = 500

    binnr_short = int(tracklength_short/bin_size)
    binnr_long = int(tracklength_long/bin_size)
    binnr_dark = int(maxdistance_dark/bin_size)

    num_rois_plot = len(roilist)
    fig_size_y = num_rois_plot
    if fig_size_y < 5:
        fig_size_y = 5
    if fig_size_y > 50:
        fig_size_y = 50
    print('Figure size Y: ' + str(fig_size_y))
    fig = plt.figure(figsize=(10,fig_size_y))
    ax_good_short = []
    ax_bad_short = []
    ax_good_long = []
    ax_bad_long = []
    for i,r in enumerate(roilist):
        ax_good_short.append(plt.subplot2grid((num_rois_plot,4),(i,0),colspan=1,rowspan=1))
        ax_bad_short.append(plt.subplot2grid((num_rois_plot,4),(i,1),colspan=1,rowspan=1))
        ax_good_long.append(plt.subplot2grid((num_rois_plot,4),(i,2),colspan=1,rowspan=1))
        ax_bad_long.append(plt.subplot2grid((num_rois_plot,4),(i,3),colspan=1,rowspan=1))

        ax_good_short[-1].set_xticklabels([])
        ax_bad_short[-1].set_xticklabels([])
        ax_good_long[-1].set_xticklabels([])
        ax_bad_long[-1].set_xticklabels([])

        set_empty_axis(ax_good_short[-1])
        set_empty_axis(ax_bad_short[-1])
        set_empty_axis(ax_good_long[-1])
        set_empty_axis(ax_bad_long[-1])

        ax_good_short[-1].axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
        ax_good_short[-1].axvspan(320/bin_size,340/bin_size,color=SHORT_COLOR,alpha=0.2,zorder=0, linewidth=0)
        ax_bad_short[-1].axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
        ax_bad_short[-1].axvspan(320/bin_size,340/bin_size,color=SHORT_COLOR,alpha=0.2,zorder=0, linewidth=0)

        ax_good_long[-1].axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
        ax_good_long[-1].axvspan(380/bin_size,400/bin_size,color=LONG_COLOR,alpha=0.2,zorder=0, linewidth=0)
        ax_bad_long[-1].axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
        ax_bad_long[-1].axvspan(380/bin_size,400/bin_size,color=LONG_COLOR,alpha=0.2,zorder=0, linewidth=0)

    behav_ds = make_blackbox_loc_continuous(behav_ds)

    trial_std_threshold = 6

    # pull out trial numbers of respective sections
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])

    fl_short, fl_trial_numbers_short, df_short, df_trial_numbers_short  = plot_first_licks(None, licks_ds, reward_ds, trials_short, SHORT_COLOR, 'short', False)
    fl_long, fl_trial_numbers_long, df_long, df_trial_numbers_long = plot_first_licks(None, licks_ds, reward_ds, trials_long, LONG_COLOR, 'long', False)

    short_close_AUC_all = []
    short_close_PEAK_all = []
    short_close_ROB_all = []
    short_far_AUC_all = []
    short_far_PEAK_all = []
    short_far_ROB_all = []

    long_close_AUC_all = []
    long_close_PEAK_all = []
    long_close_ROB_all = []
    long_far_AUC_all = []
    long_far_PEAK_all = []
    long_far_ROB_all = []

    for i,roi in enumerate(roilist):
        roi_std = np.std(dF_ds[dF_ds[:,roi]<np.percentile(dF_ds[:,roi],70),roi])
        transient_threshold = trial_std_threshold * roi_std
        print(binnr_short)
        mean_dF_short, mean_speed_short = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_short, 'landmark', binnr_short, tracklength_short, False)
        mean_dF_long, mean_speed_long = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_long, 'landmark', binnr_long, tracklength_long, False)

        fl_amplitude_short, fl_robustness_short, fl_amplitude_short_trials  = calc_fl_correlation(mean_dF_short, trials_short, fl_short, fl_trial_numbers_short, transient_threshold)
        df_amplitude_short, df_robustness_short, dl_amplitude_short_trials = calc_fl_correlation(mean_dF_short, trials_short, df_short, df_trial_numbers_short, transient_threshold)
        fl_amplitude_long, fl_robustness_long, fl_amplitude_long_trials = calc_fl_correlation(mean_dF_long, trials_long, fl_long, fl_trial_numbers_long, transient_threshold)
        df_amplitude_long, df_robustness_long, dl_amplitude_long_trials = calc_fl_correlation(mean_dF_long, trials_long, df_long, df_trial_numbers_long, transient_threshold)

        # get data of trials where the animal licked close and far from the start of the reward zone
        win_size = 15
        if fl_short.size > 0:
            fl_close = fl_short[np.abs(fl_short[:,1]) < np.percentile(np.abs(fl_short[:,1]), FL_THRESHOLD_DISTANCE_PERCENTILE)]
            fl_close_trials = fl_trial_numbers_short[np.abs(fl_short[:,1]) < np.percentile(np.abs(fl_short[:,1]), FL_THRESHOLD_DISTANCE_PERCENTILE)]
            fl_far = fl_short[np.abs(fl_short[:,1]) > np.percentile(np.abs(fl_short[:,1]), 100 - FL_THRESHOLD_DISTANCE_PERCENTILE)]
            fl_far_trials = fl_trial_numbers_short[np.abs(fl_short[:,1]) > np.percentile(np.abs(fl_short[:,1]), 100 - FL_THRESHOLD_DISTANCE_PERCENTILE)]

            mean_dF_short_close = np.full((fl_close.shape[0], mean_dF_short.shape[1]), np.nan)
            mean_dF_short_close_AUC = np.full((fl_close.shape[0]), np.nan)
            mean_dF_short_close_PEAK = np.full((fl_close.shape[0]), np.nan)
            mean_dF_short_close_ROB = np.array([np.nan])

            mean_dF_short_far = np.full((fl_far.shape[0], mean_dF_short.shape[1]), np.nan)
            mean_dF_short_far_AUC = np.full((fl_far.shape[0]), np.nan)
            mean_dF_short_far_PEAK = np.full((fl_far.shape[0]), np.nan)
            mean_dF_short_far_ROB = np.array([np.nan])

            # first we grab all the traces of the good and bad trials. Then we calculate the peak location of the mean, draw a window around it and calulcat AUC and other parameters
            for j,flt in enumerate(fl_close_trials):
                mean_dF_short_close[j,:] = mean_dF_short[trials_short==flt,:]
                ax_good_short[i].plot(mean_dF_short_close[j,:], c='0.8', lw=2)
            ax_good_short[i].plot(np.arange(75,161,1), np.nanmean(mean_dF_short_close[:,75:161],axis=0), c='k', lw=2)

            # define window from which to draw
            mean_peak = 75 + np.nanargmax(np.nanmean(mean_dF_short_close[:,75:161],axis=0))
            mean_win_min = np.amax([75,mean_peak - win_size])
            mean_win_max = np.amin([161,mean_peak + win_size])

            # calculate AUC and other parameters from window
            # ipdb.set_trace()
            for j,flt in enumerate(fl_close_trials):
                nonnan_mean = mean_dF_short_close[j,mean_win_min:mean_win_max+1]
                nonnan_mean = nonnan_mean[~np.isnan(nonnan_mean)]
                mean_dF_short_close_AUC[j] = np.trapz(nonnan_mean)
                mean_dF_short_close_PEAK[j] = np.amax(nonnan_mean)
            mean_dF_short_close_ROB = len(np.where(mean_dF_short_close[:,mean_peak] > transient_threshold)[0])
            short_close_AUC_mean = np.mean(mean_dF_short_close_AUC)

            # plot window
            ax_good_short[i].axvline(mean_peak, c='r')
            ax_good_short[i].axvline(mean_win_min, c='r', ls='--')
            ax_good_short[i].axvline(mean_win_max, c='r', ls='--')
            ax_good_short[i].set_ylabel(str(np.round(short_close_AUC_mean,2)))
            ax_good_short[i].set_ylabel(str(roi))

            # collect data for all rois to be added to dictionary
            short_close_AUC_all.append(mean_dF_short_close_AUC.tolist())
            short_close_PEAK_all.append(mean_dF_short_close_PEAK.tolist())
            short_close_ROB_all.append(mean_dF_short_close_ROB/len(fl_close_trials))


            # first we grab all the traces of the bad and bad trials. Then we calculate the peak location of the mean, draw a window around it and calulcat AUC and other parameters
            for j,flt in enumerate(fl_far_trials):
                mean_dF_short_far[j,:] = mean_dF_short[trials_short==flt,:]
                ax_bad_short[i].plot(mean_dF_short_far[j,:], c='0.8', lw=2)
            ax_bad_short[i].plot(np.arange(75,161,1), np.nanmean(mean_dF_short_far[:,75:161],axis=0), c='k', lw=2)

            # define window from which to draw
            mean_peak = 75 + np.nanargmax(np.nanmean(mean_dF_short_far[:,75:161],axis=0))
            mean_win_min = np.amax([75,mean_peak - win_size])
            mean_win_max = np.amin([161,mean_peak + win_size])

            # calculate AUC and other parameters from window
            for j,flt in enumerate(fl_far_trials):
                nonnan_mean = mean_dF_short_far[j,mean_win_min:mean_win_max+1]
                nonnan_mean = nonnan_mean[~np.isnan(nonnan_mean)]
                mean_dF_short_far_AUC[j] = np.trapz(nonnan_mean)
                mean_dF_short_far_PEAK[j] = np.amax(nonnan_mean)
            mean_dF_short_far_ROB = len(np.where(mean_dF_short_far[:,mean_peak] > transient_threshold)[0])
            short_far_AUC_mean = np.mean(mean_dF_short_far_AUC)

            # plot window
            ax_bad_short[i].axvline(mean_peak, c='r')
            ax_bad_short[i].axvline(mean_win_min, c='r', ls='--')
            ax_bad_short[i].axvline(mean_win_max, c='r', ls='--')
            # ax_bad_short[i].set_ylabel(str(np.round(short_far_AUC_mean,2)))

            # collect data for all rois to be added to dictionary
            short_far_AUC_all.append(mean_dF_short_far_AUC.tolist())
            short_far_PEAK_all.append(mean_dF_short_far_PEAK.tolist())
            short_far_ROB_all.append(mean_dF_short_far_ROB/len(fl_far_trials))

            ylim_good = ax_good_short[i].get_ylim()
            ylim_bad = ax_bad_short[i].get_ylim()
            ylim_min = np.amax([ylim_good[0],ylim_bad[0]])
            ylim_may = np.amax([ylim_good[1],ylim_bad[1]])
            ax_good_short[i].set_ylim([ylim_min,ylim_may])
            ax_bad_short[i].set_ylim([ylim_min,ylim_may])
            ax_good_short[i].set_xlim([0,170])
            ax_bad_short[i].set_xlim([0,170])

            ax_good_short[i].set_yticklabels([])
            ax_good_short[i].set_xticklabels([])
            ax_bad_short[i].set_yticklabels([])
            ax_bad_short[i].set_xticklabels([])
            ax_good_short[i].set_yticks([])
            ax_good_short[i].set_xticks([])
            ax_bad_short[i].set_yticks([])
            ax_bad_short[i].set_xticks([])

        else:
            fl_amplitude_close_short = np.array([])
            fl_amplitude_far_short = np.array([])

        if fl_long.size > 0:
            fl_close = fl_long[np.abs(fl_long[:,1]) < np.percentile(np.abs(fl_long[:,1]), FL_THRESHOLD_DISTANCE_PERCENTILE)]
            fl_close_trials = fl_trial_numbers_long[np.abs(fl_long[:,1]) < np.percentile(np.abs(fl_long[:,1]), FL_THRESHOLD_DISTANCE_PERCENTILE)]
            fl_far = fl_long[np.abs(fl_long[:,1]) > np.percentile(np.abs(fl_long[:,1]), 100 - FL_THRESHOLD_DISTANCE_PERCENTILE)]
            fl_far_trials = fl_trial_numbers_long[np.abs(fl_long[:,1]) > np.percentile(np.abs(fl_long[:,1]), 100 - FL_THRESHOLD_DISTANCE_PERCENTILE)]

            mean_dF_long_close = np.full((fl_close.shape[0], mean_dF_long.shape[1]), np.nan)
            mean_dF_long_close_AUC = np.full((fl_close.shape[0]), np.nan)
            mean_dF_long_close_PEAK = np.full((fl_close.shape[0]), np.nan)
            mean_dF_long_close_ROB = np.array([np.nan])

            mean_dF_long_far = np.full((fl_far.shape[0], mean_dF_long.shape[1]), np.nan)
            mean_dF_long_far_AUC = np.full((fl_far.shape[0]), np.nan)
            mean_dF_long_far_PEAK = np.full((fl_far.shape[0]), np.nan)
            mean_dF_long_far_ROB = np.array([np.nan])

            # first we grab all the traces of the good and bad trials. Then we calculate the peak location of the mean, draw a window around it and calulcat AUC and other parameters
            for j,flt in enumerate(fl_close_trials):
                mean_dF_long_close[j,:] = mean_dF_long[trials_long==flt,:]
                ax_good_long[i].plot(mean_dF_long_close[j,:], c='0.8', lw=2)
            ax_good_long[i].plot(np.arange(75,191,1), np.nanmean(mean_dF_long_close[:,75:191],axis=0), c='k', lw=2)

            # define window from which to draw
            mean_peak = 75 + np.nanargmax(np.nanmean(mean_dF_long_close[:,75:191],axis=0))
            mean_win_min = np.amax([75,mean_peak - win_size])
            mean_win_max = np.amin([191,mean_peak + win_size])

            # calculate AUC and other parameters from window
            for j,flt in enumerate(fl_close_trials):
                nonnan_mean = mean_dF_long_close[j,mean_win_min:mean_win_max+1]
                nonnan_mean = nonnan_mean[~np.isnan(nonnan_mean)]
                mean_dF_long_close_AUC[j] = np.trapz(nonnan_mean)
                mean_dF_long_close_PEAK[j] = np.amax(nonnan_mean)
            mean_dF_long_close_ROB = len(np.where(mean_dF_long_close[:,mean_peak] > transient_threshold)[0])
            long_close_AUC_mean = np.mean(mean_dF_long_close_AUC)

            # plot window
            ax_good_long[i].axvline(mean_peak, c='r')
            ax_good_long[i].axvline(mean_win_min, c='r', ls='--')
            ax_good_long[i].axvline(mean_win_max, c='r', ls='--')
            # ax_good_long[i].set_ylabel(str(np.round(long_close_AUC_mean,2)))

            # collect data for all rois to be added to dictionary
            long_close_AUC_all.append(mean_dF_long_close_AUC.tolist())
            long_close_PEAK_all.append(mean_dF_long_close_PEAK.tolist())
            long_close_ROB_all.append(mean_dF_long_close_ROB/len(fl_close_trials))

            # first we grab all the traces of the bad and bad trials. Then we calculate the peak location of the mean, draw a window around it and calulcat AUC and other parameters
            for j,flt in enumerate(fl_far_trials):
                mean_dF_long_far[j,:] = mean_dF_long[trials_long==flt,:]
                ax_bad_long[i].plot(mean_dF_long_far[j,:], c='0.8', lw=2)
            ax_bad_long[i].plot(np.arange(75,191,1), np.nanmean(mean_dF_long_far[:,75:191],axis=0), c='k', lw=2)

            # define window from which to draw
            mean_peak = 75 + np.nanargmax(np.nanmean(mean_dF_long_far[:,75:191],axis=0))
            mean_win_min = np.amax([75,mean_peak - win_size])
            mean_win_max = np.amin([191,mean_peak + win_size])

            # calculate AUC and other parameters from window
            for j,flt in enumerate(fl_far_trials):
                nonnan_mean = mean_dF_long_far[j,mean_win_min:mean_win_max+1]
                nonnan_mean = nonnan_mean[~np.isnan(nonnan_mean)]
                mean_dF_long_far_AUC[j] = np.trapz(nonnan_mean)
                mean_dF_long_far_PEAK[j] = np.amax(nonnan_mean)
            mean_dF_long_far_ROB = len(np.where(mean_dF_long_far[:,mean_peak] > transient_threshold)[0])
            long_far_AUC_mean = np.mean(mean_dF_long_far_AUC)

            # plot window
            ax_bad_long[i].axvline(mean_peak, c='r')
            ax_bad_long[i].axvline(mean_win_min, c='r', ls='--')
            ax_bad_long[i].axvline(mean_win_max, c='r', ls='--')
            # ax_bad_long[i].set_ylabel(str(np.round(np.mean(mean_dF_long_far_AUC),2)))

            # collect data for all rois to be added to dictionary
            long_far_AUC_all.append(mean_dF_long_far_AUC.tolist())
            long_far_PEAK_all.append(mean_dF_long_far_PEAK.tolist())
            long_far_ROB_all.append(mean_dF_long_far_ROB/len(fl_far_trials))

            ylim_good = ax_good_long[i].get_ylim()
            ylim_bad = ax_bad_long[i].get_ylim()
            ylim_min = np.amax([ylim_good[0],ylim_bad[0]])
            ylim_may = np.amax([ylim_good[1],ylim_bad[1]])
            ax_good_long[i].set_ylim([ylim_min,ylim_may])
            ax_bad_long[i].set_ylim([ylim_min,ylim_may])
            ax_good_long[i].set_xlim([0,200])
            ax_bad_long[i].set_xlim([0,200])

            ax_good_long[i].set_yticklabels([])
            ax_good_long[i].set_xticklabels([])
            ax_bad_long[i].set_yticklabels([])
            ax_bad_long[i].set_xticklabels([])
            ax_good_long[i].set_yticks([])
            ax_good_long[i].set_xticks([])
            ax_bad_long[i].set_yticks([])
            ax_bad_long[i].set_xticks([])

        else:
            fl_amplitude_close_long = np.array([])
            fl_amplitude_far_long = np.array([])

    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + mousename+'_'+sess + '.' + fformat

    fig.savefig(fname, format=fformat, dpi=400)

    write_to_dict = True
    if write_to_dict:
        print('writing to dictionary.')
        session_rois = {}
        session_rois['space_short_close_AUC_mean'] = short_close_AUC_all
        session_rois['space_short_far_AUC_mean'] = short_far_AUC_all
        session_rois['space_long_close_AUC_mean'] = long_close_AUC_all
        session_rois['space_long_far_AUC_mean'] = long_far_AUC_all

        session_rois['space_short_close_PEAK_mean'] = short_close_PEAK_all
        session_rois['space_short_far_PEAK_mean'] = short_far_PEAK_all
        session_rois['space_long_close_PEAK_mean'] = long_close_PEAK_all
        session_rois['space_long_far_PEAK_mean'] = long_far_PEAK_all

        session_rois['space_short_close_ROB_mean'] = short_close_ROB_all
        session_rois['space_short_far_ROB_mean'] = short_far_ROB_all
        session_rois['space_long_close_ROB_mean'] = long_close_ROB_all
        session_rois['space_long_far_ROB_mean'] = long_far_ROB_all
        write_dict(mousename, sess, session_rois, False, True)

    print(fname)
    # plt.show()

def fig_dfloc_trace_roiparams(h5path, sess, roi, fname, align_point, peak_locs, filterprops_short_1, filterprops_short_2, filterprops_long_1, filterprops_long_2, fformat='png', subfolder=[], c_ylim=[], make_figure=True, load_raw=False,use_data='aligned_data.mat', is_ol_sess=False):

    if load_raw == False:
        df_signal_path = h5path  + os.sep + use_data + '_dF_aligned.mat'
        behavior_path = h5path  + os.sep + use_data + '_behavior_aligned.mat'
        dF_ds = sio.loadmat(df_signal_path)['data']
        behav_ds = sio.loadmat(behavior_path)['data']
        no_blackbox_trials = filter_trials(behav_ds, [], ['tracknumber', 3])
        no_blackbox_trials = np.union1d(no_blackbox_trials, filter_trials(behav_ds, [], ['tracknumber', 4]))
        # ipdb.set_trace()
        behav_licks = behav_ds[np.in1d(behav_ds[:, 4], [3, 4]), :]
        reward_ds = rewards(behav_licks)
        licks_ds,_ = licks(behav_licks, reward_ds)
        licks_ds = np.array(licks_ds)
        
#        h5dat = h5py.File(h5path, 'r')
#        behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
#        dF_ds = np.copy(h5dat[sess + '/dF_win'])
#        licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
#        reward_ds = np.copy(h5dat[sess + '/rewards'])
    else:

        processed_data_path = h5path + os.sep + sess + os.sep + use_data
        loaded_data = sio.loadmat(processed_data_path)
        behav_ds = loaded_data['behaviour_aligned']
        dF_ds = loaded_data['calcium_dF']
        # remove times when mouse was in the black box
        no_blackbox_trials = filter_trials(behav_ds, [], ['tracknumber', 3])
        no_blackbox_trials = np.union1d(no_blackbox_trials, filter_trials(behav_ds, [], ['tracknumber', 4]))
        # ipdb.set_trace()
        behav_licks = behav_ds[np.in1d(behav_ds[:, 4], [3, 4]), :]
        reward_ds = rewards(behav_licks)
        licks_ds,_ = licks(behav_licks, reward_ds)
        licks_ds = np.array(licks_ds)
        # licks_ds = loaded_data['/licks_pre_reward']
        # licks_ds = loaded_data['/licks_pre_reward']
        # reward_ds = loaded_data['/rewards']

    behav_ds = make_blackbox_loc_continuous(behav_ds)

    behav_ds[behav_ds[:,3]>60,3] = 60

    bin_size = 5

    plot_binnr_short = 360/bin_size
    plot_binnr_long = 420/bin_size

    tracklength_short = 400
    tracklength_long = 500
    maxdistance_dark = 500

    binnr_short = int(tracklength_short/bin_size)
    binnr_long = int(tracklength_long/bin_size)
    binnr_dark = int(maxdistance_dark/bin_size)

    track_short = 3
    track_long = 4
    track_dark = 5

    # number of standard deviations the roi signal has to exceed to be counted as active
    trial_std_threshold = 6

    # minimum number of trials a roi as to be active in for some stats to be calculated
    MIN_ACTIVE_TRIALS = 5

    # peak tolerance converted to spatial bins
    peak_tolerance = int(OPENLOOP_PEAK_TOLERANCE/bin_size)

    # calcluate standard deviation of ROI traces of the bottom x percentile
    roi_std = np.std(dF_ds[dF_ds[:,roi]<np.percentile(dF_ds[:,roi],70),roi])
    # roi_std = np.std(placehold[placehold<np.percentile(placehold,70)])


    if make_figure:
        # create figure to later plot on
        fig = plt.figure(figsize=(10,14))
        ax1 = plt.subplot2grid((16,4),(0,0),colspan=2,rowspan=3)
        ax2 = plt.subplot2grid((16,4),(0,2),colspan=2,rowspan=3)
        ax3 = plt.subplot2grid((16,4),(3,0),colspan=2,rowspan=3)
        ax4 = plt.subplot2grid((16,4),(3,2),colspan=2,rowspan=3)
        ax5 = plt.subplot2grid((16,4),(12,0),colspan=1,rowspan=2)
        ax6 = plt.subplot2grid((16,4),(12,1),colspan=1,rowspan=2)
        ax7 = plt.subplot2grid((16,4),(12,2),colspan=1,rowspan=2)
        ax8 = plt.subplot2grid((16,4),(12,3),colspan=1,rowspan=2)
        ax9 = plt.subplot2grid((16,4),(10,0),colspan=1,rowspan=2)
        ax10 = plt.subplot2grid((16,4),(10,1),colspan=1,rowspan=2)
        ax11 = plt.subplot2grid((16,4),(10,2),colspan=1,rowspan=2)
        ax12 = plt.subplot2grid((16,4),(10,3),colspan=1,rowspan=2)
        ax13 = plt.subplot2grid((16,4),(14,0),colspan=1,rowspan=2)
        ax14 = plt.subplot2grid((16,4),(14,1),colspan=1,rowspan=2)
        ax15 = plt.subplot2grid((16,4),(14,2),colspan=1,rowspan=2)
        ax16 = plt.subplot2grid((16,4),(14,3),colspan=1,rowspan=2)
        ax17 = plt.subplot2grid((16,4),(6,0),colspan=1,rowspan=2)
        ax18 = plt.subplot2grid((16,4),(6,1),colspan=1,rowspan=2)
        ax19 = plt.subplot2grid((16,4),(6,2),colspan=1,rowspan=2)
        ax20 = plt.subplot2grid((16,4),(6,3),colspan=1,rowspan=2)

        ax21 = plt.subplot2grid((16,4),(8,0),colspan=1,rowspan=2)
        ax22 = plt.subplot2grid((16,4),(8,1),colspan=1,rowspan=2)
        ax23 = plt.subplot2grid((16,4),(8,2),colspan=1,rowspan=2)
        ax24 = plt.subplot2grid((16,4),(8,3),colspan=1,rowspan=2)

        xmin = -2
        xmax_short = 68
        xmax_long = 80

        if align_point == 'landmark':
            # plot landmark and reward zone as shaded areas
            ax1.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
            ax1.axvspan(320/bin_size,340/bin_size,color=SHORT_COLOR,alpha=0.2,zorder=0, linewidth=0)
            ax2.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
            ax2.axvspan(380/bin_size,400/bin_size,color=LONG_COLOR,alpha=0.2,zorder=0, linewidth=0)

            ax9.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
            ax9.axvspan(320/bin_size,340/bin_size,color=SHORT_COLOR,alpha=0.2,zorder=0, linewidth=0)
            ax10.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
            ax10.axvspan(380/bin_size,400/bin_size,color=SHORT_COLOR,alpha=0.2,zorder=0, linewidth=0)

            ax11.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
            ax11.axvspan(320/bin_size,340/bin_size,color=LONG_COLOR,alpha=0.2,zorder=0, linewidth=0)
            ax12.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0, linewidth=0)
            ax12.axvspan(380/bin_size,400/bin_size,color=LONG_COLOR,alpha=0.2,zorder=0, linewidth=0)

        # set axes visibility
        ax1 = set_empty_axis(ax1)
        ax2 = set_empty_axis(ax2)
        ax3 = set_empty_axis(ax3)
        ax4 = set_empty_axis(ax4)
        ax9 = set_empty_axis(ax9)
        ax10 = set_empty_axis(ax10)
        ax11 = set_empty_axis(ax11)
        ax12 = set_empty_axis(ax12)
        ax13 = set_empty_axis(ax13)
        ax14 = set_empty_axis(ax14)
        ax15 = set_empty_axis(ax15)
        ax16 = set_empty_axis(ax16)

        ax3.tick_params(bottom='off')
        ax4.tick_params(bottom='off')

        # plot lines indicating landmark and reward zone in heatmaps
        if align_point == 'landmark':
            ax3.axvline(220/bin_size,c='#FF0000',lw=2)
            ax3.axvline(320/bin_size,c='#29ABE2',lw=2)

            ax4.axvline(220/bin_size,c='#FF0000',lw=2)
            ax4.axvline(380/bin_size,c='#29ABE2',lw=2)

            # ax5.axvline(40,c='#FF0000',lw=2)
            ax5.axvline(220/bin_size,c='#FF0000',lw=2)
            ax5.axvline(320/bin_size,c='#29ABE2',lw=2)

            # ax6.axvline(40,c='0.8',lw=2)
            ax6.axvline(220/bin_size,c='#FF0000',lw=2)
            ax6.axvline(320/bin_size,c='#29ABE2',lw=2)

            # ax7.axvline(40,c='0.8',lw=2)
            ax7.axvline(220/bin_size,c='#FF0000',lw=2)
            ax7.axvline(380/bin_size,c='#29ABE2',lw=2)

            # ax8.axvline(40,c='0.8',lw=2)
            ax8.axvline(220/bin_size,c='#FF0000',lw=2)
            ax8.axvline(380/bin_size,c='#29ABE2',lw=2)

    else:
        ax1 = None
        ax2 = None
        ax3 = None
        ax4 = None
        ax5 = None
        ax6 = None
        ax7 = None
        ax8 = None
        ax9 = None
        ax10 = None
        ax11 = None
        ax12 = None
        ax13 = None
        ax14 = None
        ax15 = None
        ax16 = None
        ax17 = None
        ax18 = None
        ax19 = None
        ax20 = None
        ax21 = None
        ax22 = None
        ax23 = None
        ax24 = None

    # pull out trial numbers of respective sections
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])

    if make_figure:
        fl_short, fl_trial_numbers_short, df_short, df_trial_numbers_short  = plot_first_licks(ax18, licks_ds, reward_ds, trials_short, SHORT_COLOR, 'short', make_figure)
        fl_long, fl_trial_numbers_long, df_long, df_trial_numbers_long = plot_first_licks(ax20, licks_ds, reward_ds, trials_long, LONG_COLOR, 'long', make_figure)
    else:
        fl_short, fl_trial_numbers_short, df_short, df_trial_numbers_short  = plot_first_licks(None, licks_ds, reward_ds, trials_short, SHORT_COLOR, 'short', make_figure)
        fl_long, fl_trial_numbers_long, df_long, df_trial_numbers_long = plot_first_licks(None, licks_ds, reward_ds, trials_long, LONG_COLOR, 'long', make_figure)
    # trials_short = filter_trials( behav_ds, [], ['exclude_earlylick_trials',[100,200]],trials_short)
    # trials_long = filter_trials( behav_ds, [], ['exclude_earlylick_trials',[100,200]],trials_long)
    # mean_dF_short = np.zeros((np.size(trials_short,0),binnr_short))
    # mean_dF_long = np.zeros((np.size(trials_long,0),binnr_long))

    transient_threshold = trial_std_threshold * roi_std
    # specify distance within which transient peaks have to be to be considered within for analysis of their distance to the mean peak
    max_peak_distance = 60
    # keep track of the distance of individual transients to the peak of the mean
    meanpeak_transient_distance_short = []
    meanpeak_transient_peak_short = []
    meanpeak_transient_peak_short_idx = []
    transient_speed_short = []
    cur_trial_max_idx_short = np.empty(0)
    cur_trial_max_idx_long = np.empty(0)
    roi_std_short = np.int64(-1)
    # run through SHORT trials and calculate avg dF/F for each bin and trial
    mean_dF_short, mean_speed_short = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_short, align_point, binnr_short, tracklength_short, is_ol_sess)
    mean_dF_long, mean_speed_long = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_long, align_point, binnr_long, tracklength_long, is_ol_sess)
    # mean_dF_short, mean_speed_short = calc_spacevloc(behav_ds, placehold, trials_short, align_point, binnr_short, tracklength_short, is_ol_sess)
    # mean_dF_long, mean_speed_long = calc_spacevloc(behav_ds, placehold, trials_long, align_point, binnr_long, tracklength_long, is_ol_sess)
    fl_amplitude_short, fl_robustness_short, _ = calc_fl_correlation(mean_dF_short, trials_short, fl_short, fl_trial_numbers_short, transient_threshold)
    df_amplitude_short, df_robustness_short, _ = calc_fl_correlation(mean_dF_short, trials_short, df_short, df_trial_numbers_short, transient_threshold)
    fl_amplitude_long, fl_robustness_long, _ = calc_fl_correlation(mean_dF_long, trials_long, fl_long, fl_trial_numbers_long, transient_threshold)
    df_amplitude_long, df_robustness_long, _ = calc_fl_correlation(mean_dF_long, trials_long, df_long, df_trial_numbers_long, transient_threshold)

    # get data of trials where the animal licked close and far from the start of the reward zone
    if fl_short.size > 0:
        fl_close = fl_short[np.abs(fl_short[:,1]) < np.percentile(np.abs(fl_short[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        fl_amplitude_close_short = fl_amplitude_short[np.abs(fl_short[:,1]) < np.percentile(np.abs(fl_short[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        fl_far = fl_short[np.abs(fl_short[:,1]) > np.percentile(np.abs(fl_short[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        fl_amplitude_far_short = fl_amplitude_short[np.abs(fl_short[:,1]) > np.percentile(np.abs(fl_short[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        if make_figure:
            ax22.bar([1,2], [np.nanmean(fl_amplitude_close_short), np.nanmean(fl_amplitude_far_short)], [0.4,0.4], yerr=[stats.sem(fl_amplitude_close_short,nan_policy='omit'), stats.sem(fl_amplitude_far_short,nan_policy='omit')], ecolor='k' , align='center', color=[SHORT_COLOR, 'w'], edgecolor=[SHORT_COLOR,SHORT_COLOR], linewidth=3)
    else:
        fl_amplitude_close_short = np.array([])
        fl_amplitude_far_short = np.array([])

    if fl_long.size > 0:
        fl_close = fl_long[np.abs(fl_long[:,1]) < np.percentile(np.abs(fl_long[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        fl_amplitude_close_long = fl_amplitude_long[np.abs(fl_long[:,1]) < np.percentile(np.abs(fl_long[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        fl_far = fl_long[np.abs(fl_long[:,1]) > np.percentile(np.abs(fl_long[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        fl_amplitude_far_long = fl_amplitude_long[np.abs(fl_long[:,1]) > np.percentile(np.abs(fl_long[:,1]),FL_THRESHOLD_DISTANCE_PERCENTILE)]
        if make_figure:
            ax24.bar([1,2], [np.nanmean(fl_amplitude_close_short), np.nanmean(fl_amplitude_far_short)], [0.4,0.4], yerr=[stats.sem(fl_amplitude_close_short,nan_policy='omit'), stats.sem(fl_amplitude_far_short,nan_policy='omit')], ecolor='k' , align='center', color=[LONG_COLOR, 'w'], edgecolor=[LONG_COLOR,LONG_COLOR], linewidth=3)
    else:
        fl_amplitude_close_long = np.array([])
        fl_amplitude_far_long = np.array([])

    # get rid of nan values for correlation calculation
    fl_short_pearson = fl_short[:,1][~np.isnan(fl_short[:,1])]
    fl_short_pearson = fl_short[:,1][~np.isnan(fl_amplitude_short)]
    fl_amplitude_short_pearson = fl_amplitude_short[~np.isnan(fl_short[:,1])]
    fl_amplitude_short_pearson = fl_amplitude_short[~np.isnan(fl_amplitude_short)]
    if fl_short_pearson.shape[0] > 1 and fl_amplitude_short_pearson.shape[0] > 1:
        fl_amp_short_r, fl_amp_short_p = stats.pearsonr(fl_short_pearson, fl_amplitude_short_pearson)
        fl_amp_short_r = np.float64(fl_amp_short_r)
        fl_amp_short_p = np.float64(fl_amp_short_p)
    else:
        fl_amp_short_r = np.float64(0)
        fl_amp_short_p = np.float64(0)

    if make_figure:
        ax21.set_title('r = ' + str(np.round(fl_amp_short_r,2)) + ' p = ' + str(np.round(fl_amp_short_p,2)))
        ax21.scatter(fl_short[:,1], fl_amplitude_short, c=SHORT_COLOR, lw=0)
        ax21.scatter(df_short[:,1], df_amplitude_short, c='r', lw=0)
        ax21.set_ylabel('dF/F')
        ax21.set_xlabel('Distance from reward (cm)')

    # ax22.bar([1,2], [np.mean(fl_robustness_short), np.mean(df_robustness_short)], [0.4,0.4], yerr=[stats.sem(fl_robustness_short), stats.sem(df_robustness_short)], ecolor='k' , align='center', color=[SHORT_COLOR, 'w'], edgecolor=[SHORT_COLOR,SHORT_COLOR], linewidth=3)

    # ax22.scatter(np.full_like(fl_amplitude_close,1), fl_amplitude_close)
    # ax22.scatter(np.full_like(fl_amplitude_far,2), fl_amplitude_far)
    # ipdb.set_trace()
    fl_robust_short = np.nanmean(fl_robustness_short)
    df_robust_short = np.nanmean(df_robustness_short)
    # # if we have nan robustness
    # print(df_robust_short)
    # if np.isnan(df_robust_short) or np.isnan(fl_robust_short):
    #     ipdb.set_trace()
    #     pass
    # if len(fl_robustness_short) == 0 and len(fl_short) > 0:
    #     fl_robust_short = 0
    # if len(df_robustness_short) == 0 and len(fl_short) > 0:
    #     df_robust_short = 0

    fl_amp_short = np.nanmean(fl_amplitude_short)
    df_amp_short = np.nanmean(df_amplitude_short)
    if make_figure:
        ax22.set_xlim(0.2,2.8)
        ax22.set_xticks([1,2])
        ax22.set_xticklabels(['close', 'far'])


    for tn,t in enumerate(mean_dF_short):
        if make_figure:
            ax1.plot(t,c='0.8',zorder=2)
        # high_bins = np.where(mean_zscore_trials_short[tn,:] > 1)[0]
        high_bins = np.where(mean_dF_short[tn,:] > (transient_threshold))[0]
        if len(high_bins) > 0:
            # split high_bins if it exceeds threshold multiple times
            trial_transients = []
            trial_peaks = []
            trial_transient_speed = []
            for hb in np.split(high_bins, np.where(np.diff(high_bins)>2)[0]+1):
                if len(hb) > 2:
                    hb = np.insert(hb,0,hb[0]-1)
                    hb = np.insert(hb,len(hb),hb[-1]+1)
                    if make_figure:
                        ax1.plot(hb-1, t[hb-1], c=SHORT_COLOR,lw=1, zorder=3)
                    trial_transients = np.append(trial_transients,hb[np.nanargmax(t[hb-1])])
                    trial_peaks = np.append(trial_peaks,np.nanmax(t[hb-1]))
                    trial_transient_speed = np.append(trial_transient_speed,np.nanmean(mean_speed_short[tn,hb-1]))

            if len(trial_transients) > 0 and len(trial_peaks) > 0:
                meanpeak_transient_distance_short.append(trial_transients)
                meanpeak_transient_peak_short.append(trial_peaks)
                transient_speed_short.append(trial_transient_speed)

    sem_dF_s = stats.sem(mean_dF_short,0,nan_policy='omit')
    if make_figure:
        ax1.axhline(trial_std_threshold * roi_std, ls='--', lw=2, c='0.8')

    # flatten arrays
    all_transient_speed_short = [tss for tss_list in transient_speed_short for tss in tss_list]
    all_meanpeak_transient_peak_short = [tss for tss_list in meanpeak_transient_peak_short for tss in tss_list]
    if make_figure:
        ax17.scatter(all_transient_speed_short,all_meanpeak_transient_peak_short, c=SHORT_COLOR, lw=0)
    if len(all_meanpeak_transient_peak_short) > 10 and True:
        try:
            corr_speed_short, peak_intercept, lo_slope, up_slope = stats.theilslopes(all_meanpeak_transient_peak_short, all_transient_speed_short)
            corr_speed_r_short,corr_speed_p_short = stats.pearsonr(all_meanpeak_transient_peak_short, all_transient_speed_short)
            # this stupid workaround below is required because stats.pearsonr surprisingly returns a python builtin float instead of np.float64
            corr_speed_r_short = np.float64(corr_speed_r_short)
            corr_speed_p_short = np.float64(corr_speed_p_short)
            if make_figure:
                ax17.plot(all_transient_speed_short, peak_intercept+corr_speed_short * np.array(all_transient_speed_short), lw=2,c='r')
                ax17.set_title('r = ' + str(round(corr_speed_r_short,2)) + ' p = ' + str(round(corr_speed_p_short,2)))
        except IndexError:
            corr_speed_short = np.float64(np.nan)
            corr_speed_r_short = np.float64(np.nan)
            corr_speed_p_short = np.float64(np.nan)
            peak_intercept = np.float64(np.nan)
    else:
        corr_speed_short = np.float64(np.nan)
        corr_speed_r_short = np.float64(np.nan)
        corr_speed_p_short = np.float64(np.nan)
        peak_intercept = np.float64(np.nan)

    if len(all_transient_speed_short) > 0:
        trial_speed_max_x_short = np.nanmax(all_transient_speed_short)
    else:
        trial_speed_max_x_short = 0

    if len(all_transient_speed_short) > 1 and len(all_meanpeak_transient_peak_short) > 2:
        pearson_r_short, pearson_p_short = stats.pearsonr(all_transient_speed_short, all_meanpeak_transient_peak_short)
    else:
        pearson_r_short = np.float(0)
        pearson_p_short = np.float(0)
    # pearson_r_short, pearson_p_short = stats.pearsonr(all_transient_speed_short, all_meanpeak_transient_peak_short)
    # print('speed corr short: ' + str(corr_speed_short))

    # calculate the number of trials in which the calcium signal went above the threshold
    if len(cur_trial_max_idx_short) >= 0:
        roi_active_fraction_short = np.float64(len(cur_trial_max_idx_short)/np.size(trials_short,0))
        if make_figure:
            sns.distplot(cur_trial_max_idx_short,hist=False,kde=False,rug=True,ax=ax1)
        roi_std_short = np.std(cur_trial_max_idx_short)

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(mean_dF_short.T):
        if np.count_nonzero(np.isnan(trace))/trace.shape[0] < MEAN_TRACE_FRACTION:
            mean_valid_indices.append(i)

    mean_trace_short = np.nanmean(mean_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    # if a peak location is provided (the case when we want to know the neuron's response in openloop condition at its VR peak)
    if len(peak_locs) == 0:
        roi_meanpeak_short = np.nanmax(mean_trace_short)
        roi_meanpeak_short_idx = np.nanargmax(mean_trace_short)
        roi_meanpeak_short_location = (roi_meanpeak_short_idx+mean_valid_indices[0]) * bin_size
    else:
        roi_meanpeak_short_idx_vr = np.int64(peak_locs[0]/bin_size) - mean_valid_indices[0]
        if peak_tolerance > 0:
            try:
                win_start = roi_meanpeak_short_idx_vr-peak_tolerance
                win_end = roi_meanpeak_short_idx_vr+peak_tolerance
                win_start_offset = 0
                win_end_offset = 0
                if win_start < 0:
                    win_start = 0
                    win_start_offset = np.abs(roi_meanpeak_short_idx_vr-peak_tolerance)
                if win_end > mean_trace_short.shape[0]:
                    win_end = mean_trace_short.shape[0]
                    win_end_offset = (roi_meanpeak_short_idx_vr+peak_tolerance)-mean_trace_short.shape[0]
                roi_meanpeak_short_idx = roi_meanpeak_short_idx_vr + (np.nanargmax(mean_trace_short[win_start:win_end])-peak_tolerance+win_start_offset-win_end_offset)
            except ValueError:
                roi_meanpeak_short_idx = roi_meanpeak_short_idx_vr
        else:
            roi_meanpeak_short_idx = roi_meanpeak_short_idx_vr
        roi_meanpeak_short = mean_trace_short[roi_meanpeak_short_idx]
        roi_meanpeak_short_location = np.int64(peak_locs[0])

    # trial_speed = []
    # trial_df = []
    # for tn,t in enumerate(mean_dF_short):
    #     peak_idx = roi_meanpeak_short_idx+mean_valid_indices[0]
    #     trial_speed.append(np.nanmean(mean_speed_short[tn,peak_idx-speed_correlation_range:peak_idx+speed_correlation_range]))

    if make_figure:
        ax1.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_trace_short,c='k',lw=3,zorder=4)
        ax1.axvline((roi_meanpeak_short_idx+mean_valid_indices[0]),c='b')

    # determine which transient peaks are within the specified window
    RF_transients_short = []
    RF_transients_residuals_short = []
    RF_transients_peak_short = []
    for j,mtd in enumerate(meanpeak_transient_distance_short):
        # translate bin number of mean peak into actual distances (cm)
        meanpeak_loc = (roi_meanpeak_short_idx+mean_valid_indices[0]) * bin_size
        mtd = mtd * bin_size
        # pull out peak values associated with transient_trial_start_idx
        mtp = meanpeak_transient_peak_short[j]
        # determine which transient peaks are within range
        mtp = mtp[np.where((mtd > (meanpeak_loc-max_peak_distance)) & (mtd < (meanpeak_loc+max_peak_distance)))]
        mtd = mtd[np.where((mtd > (meanpeak_loc-max_peak_distance)) & (mtd < (meanpeak_loc+max_peak_distance)))]
        RF_transients_short.append(mtd)
        RF_transients_residuals_short.append(mtd-meanpeak_loc)
        RF_transients_peak_short.append(mtp)

    # flatten list and calculate deviance paraameters
    RF_transients_residuals_short = [rftr for sublist in RF_transients_residuals_short for rftr in sublist]
    RF_transients_short = [rftr for sublist in RF_transients_short for rftr in sublist]
    RF_transients_peak_short = [rfpr for sublist in RF_transients_peak_short for rfpr in sublist]
    if make_figure:
        sns.rugplot(np.array(RF_transients_short)/bin_size,ax=ax1)
    mean_transient_distance_short = np.mean(RF_transients_residuals_short)
    mean_transient_std_short = np.std(RF_transients_residuals_short)
    mean_transient_sem_short = stats.sem(RF_transients_residuals_short)
    num_RF_transients_short = len(RF_transients_short)/len(trials_short)
    mean_transient_peak_short = np.mean(RF_transients_peak_short)

    mean_trace_short_start = np.int64(mean_valid_indices[0])

    # run through FILTER 1 for SHORT trials and calculate avg dF/F for each bin and trial
    trials_short_succ = filter_trials( behav_ds, [], filterprops_short_1, trials_short)
    trials_short_succ_num = len(trials_short_succ)
    short_succ_speed = []

    if len(trials_short_succ) > 0:
        filtered_short_1_vr_peak, filtered_short_1_mean_trace, mean_dF_short_succ, short_succ_speed, filtered_short_1_mean_transient_sem, filtered_short_1_num_RF_transients, filtered_short_1_mean_transient_peak = response_max(behav_ds, dF_ds[:,roi], trials_short_succ, align_point, binnr_short, tracklength_short, mean_valid_indices, peak_locs, mean_trace_short, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax9, make_figure, SHORT_COLOR, is_ol_sess)
        # filtered_short_1_vr_peak, filtered_short_1_mean_trace, mean_dF_short_succ, short_succ_speed, filtered_short_1_mean_transient_sem, filtered_short_1_num_RF_transients, filtered_short_1_mean_transient_peak = response_max(behav_ds, placehold, trials_short_succ, align_point, binnr_short, tracklength_short, mean_valid_indices, peak_locs, mean_trace_short, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax9, make_figure, SHORT_COLOR, is_ol_sess)
    else:
        filtered_short_1_mean_trace = np.empty(0)
        filtered_short_1_vr_peak = np.float64(np.nan)
        filtered_short_1_mean_transient_sem = np.float64(np.nan)
        filtered_short_1_num_RF_transients = 0
        filtered_short_1_mean_transient_peak = np.float64(np.nan)

    # run through FILTER 2 for SHORT trials and calculate avg dF/F for each bin and trial
    trials_short_unsucc = filter_trials( behav_ds, [], filterprops_short_2, trials_short)
    # print(trials_short_unsucc)
    short_unsucc_speed = []
    trials_short_unsucc_num = len(trials_short_unsucc)
    # ipdb.set_trace()
    if len(trials_short_unsucc) > 0:
        filtered_short_2_vr_peak, filtered_short_2_mean_trace, mean_dF_short_unsucc, short_unsucc_speed, filtered_short_2_mean_transient_sem, filtered_short_2_num_RF_transients, filtered_short_2_mean_transient_peak = response_max(behav_ds, dF_ds[:,roi], trials_short_unsucc, align_point, binnr_short, tracklength_short, mean_valid_indices, peak_locs, mean_trace_short, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax10, make_figure, SHORT_COLOR, is_ol_sess)
        # filtered_short_2_vr_peak, filtered_short_2_mean_trace, mean_dF_short_unsucc, short_unsucc_speed, filtered_short_2_mean_transient_sem, filtered_short_2_num_RF_transients, filtered_short_2_mean_transient_peak = response_max(behav_ds, placehold, trials_short_unsucc, align_point, binnr_short, tracklength_short, mean_valid_indices, peak_locs, mean_trace_short, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax10, make_figure, SHORT_COLOR, is_ol_sess)

    else:
        filtered_short_2_mean_trace = np.empty(0)
        filtered_short_2_vr_peak = np.float64(np.nan)
        filtered_short_2_mean_transient_sem = np.float64(np.nan)
        filtered_short_2_num_RF_transients = 0
        filtered_short_2_mean_transient_peak = np.float64(np.nan)

    # run through LONG trials
    # keep track of the distance of individual transients to the peak of the mean
    meanpeak_transient_distance_long = []
    meanpeak_transient_peak_long = []
    transient_speed_long = []
    mean_dF_long,mean_speed_long = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_long, align_point, binnr_long, tracklength_long, is_ol_sess)
    # mean_dF_long,mean_speed_long = calc_spacevloc(behav_ds, placehold, trials_long, align_point, binnr_long, tracklength_long, is_ol_sess)
    fl_amplitude_long, fl_robustness_long, _ = calc_fl_correlation(mean_dF_long, trials_long, fl_long, fl_trial_numbers_long, transient_threshold)
    df_amplitude_long, df_robustness_long, _ = calc_fl_correlation(mean_dF_long, trials_long, df_long, df_trial_numbers_long, transient_threshold)

    # get rid of nan values for correlation calculation
    fl_long_pearson = fl_long[:,1][~np.isnan(fl_long[:,1])]
    fl_long_pearson = fl_long[:,1][~np.isnan(fl_amplitude_long)]
    fl_amplitude_long_pearson = fl_amplitude_long[~np.isnan(fl_long[:,1])]
    fl_amplitude_long_pearson = fl_amplitude_long[~np.isnan(fl_amplitude_long)]
    # fl_amp_long_r, fl_amp_long_p = stats.pearsonr(fl_long_pearson, fl_amplitude_long_pearson)
    # fl_amp_long_r = np.float64(fl_amp_long_r)
    # fl_amp_long_p = np.float64(fl_amp_long_p)
    if fl_long_pearson.shape[0] > 1 and fl_amplitude_long_pearson.shape[0] > 1:
        fl_amp_long_r, fl_amp_long_p = stats.pearsonr(fl_long_pearson, fl_amplitude_long_pearson)
        fl_amp_long_r = np.float64(fl_amp_long_r)
        fl_amp_long_p = np.float64(fl_amp_long_p)
    else:
        fl_amp_long_r = np.float64(0)
        fl_amp_long_p = np.float64(0)

    if make_figure:
        ax23.set_title('r = ' + str(np.round(fl_amp_long_r,2)) + ' p = ' + str(np.round(fl_amp_long_p,2)))

        ax23.scatter(fl_long[:,1], fl_amplitude_long, c=LONG_COLOR, lw=0)
        ax23.scatter(df_long[:,1], df_amplitude_long, c='r', lw=0)
        ax23.set_ylabel('dF/F')
        ax23.set_xlabel('Distance from reward (cm)')

    # ax24.bar([1,2], [np.mean(fl_robustness_long), np.mean(df_robustness_long)], [0.4,0.4], yerr=[stats.sem(fl_robustness_long), stats.sem(df_robustness_long)], ecolor='k' , align='center', color=[LONG_COLOR, 'w'], edgecolor=[LONG_COLOR,LONG_COLOR], linewidth=3)
    # ipdb.set_trace()
    fl_robust_long = np.nanmean(fl_robustness_long)
    df_robust_long = np.nanmean(df_robustness_long)
    fl_amp_long = np.nanmean(fl_amplitude_long)
    df_amp_long = np.nanmean(df_amplitude_long)
    if make_figure:
        ax24.set_xlim(0.2,2.8)

        ax21.set_xlim([np.amin([ax21.get_xlim()[0], ax23.get_xlim()[0]]), np.amax([ax21.get_xlim()[1], ax23.get_xlim()[1]])])
        ax23.set_xlim([np.amin([ax21.get_xlim()[0], ax23.get_xlim()[0]]), np.amax([ax21.get_xlim()[1], ax23.get_xlim()[1]])])

        ax22.set_ylim([np.amin([ax22.get_ylim()[0], ax24.get_ylim()[0]]), np.amax([ax22.get_ylim()[1], ax24.get_ylim()[1]])])
        ax24.set_ylim([np.amin([ax22.get_ylim()[0], ax24.get_ylim()[0]]), np.amax([ax22.get_ylim()[1], ax24.get_ylim()[1]])])

    for tn,t in enumerate(mean_dF_long):
        if make_figure:
            ax2.plot(t,c='0.8',zorder=2)
        # high_bins = np.where(mean_zscore_trials_long[tn,:] > 1)[0]
        high_bins = np.where(mean_dF_long[tn,:] > (trial_std_threshold * roi_std))[0]
        if len(high_bins) > 0:
            # split high_bins if it exceeds threshold multiple times
            trial_transients = []
            trial_peaks = []
            trial_transient_speed = []
            for hb in np.split(high_bins, np.where(np.diff(high_bins)>2)[0]+1):

                if len(hb) > 2:
                    hb = np.insert(hb,0,hb[0]-1)
                    hb = np.insert(hb,len(hb),hb[-1]+1)
                    if make_figure:
                        ax2.plot(hb-1, t[hb-1], c=LONG_COLOR,lw=1,zorder=3)
                    trial_transients = np.append(trial_transients,hb[np.nanargmax(t[hb-1])])
                    trial_peaks = np.append(trial_peaks,np.nanmax(t[hb-1]))
                    trial_transient_speed = np.append(trial_transient_speed,np.nanmean(mean_speed_long[tn,hb-1]))

            if len(trial_transients) > 0 and len(trial_peaks) > 0:
                meanpeak_transient_distance_long.append(trial_transients)
                meanpeak_transient_peak_long.append(trial_peaks)
                transient_speed_long.append(trial_transient_speed)

    sem_dF_l = stats.sem(mean_dF_long,0,nan_policy='omit')
    # avg_mean_dF_long = np.nanmean(mean_dF_long,axis=0)
    if make_figure:
        # ax2.plot(avg_mean_dF_long,c=LONG_COLOR,lw=3) #'#FF00FF'
        ax2.axhline(trial_std_threshold * roi_std, ls='--', lw=2, c='0.8')

    # flatten arrays
    all_transient_speed_long = [tss for tss_list in transient_speed_long for tss in tss_list]
    all_meanpeak_transient_peak_long = [tss for tss_list in meanpeak_transient_peak_long for tss in tss_list]
    if make_figure:
        ax19.scatter(all_transient_speed_long,all_meanpeak_transient_peak_long, c=LONG_COLOR, lw=0)
    if len(all_transient_speed_long) > 10 and True:
        try:
            corr_speed_long, peak_intercept, lo_slope, up_slope = stats.theilslopes(all_meanpeak_transient_peak_long, all_transient_speed_long)
            corr_speed_r_long,corr_speed_p_long = stats.pearsonr(all_meanpeak_transient_peak_long, all_transient_speed_long)
            # this stupid workaround below is required because stats.pearsonr surprisingly returns a python builtin float instead of np.float64
            corr_speed_r_long = np.float64(corr_speed_r_long)
            corr_speed_p_long = np.float64(corr_speed_p_long)
            if make_figure:
                ax19.plot(all_transient_speed_long, peak_intercept+corr_speed_long * np.array(all_transient_speed_long), lw=2,c='r')
                ax19.set_title('r = ' + str(round(corr_speed_r_long,2)) + ' p = ' + str(round(corr_speed_p_long,2)))
        except IndexError:
            corr_speed_long = np.float64(np.nan)
            corr_speed_r_long = np.float64(np.nan)
            corr_speed_p_long = np.float64(np.nan)
            peak_intercept = np.float64(np.nan)
    else:
        corr_speed_long = np.float64(np.nan)
        corr_speed_r_long = np.float64(np.nan)
        corr_speed_p_long = np.float64(np.nan)
        peak_intercept = np.float64(np.nan)

    if len(all_transient_speed_long) > 0:
        speed_corr_x = np.amax([np.nanmax(all_transient_speed_long), trial_speed_max_x_short]) + 5
    else:
        speed_corr_x = trial_speed_max_x_short + 5

    if make_figure:
        ax17.set_xlim([0,speed_corr_x])
        ax19.set_xlim([0,speed_corr_x])
        
    if len(all_transient_speed_long) > 1 and len(all_meanpeak_transient_peak_long) > 2:
        pearson_r_long, pearson_p_long = stats.pearsonr(all_transient_speed_long, all_meanpeak_transient_peak_long)
    else:
        pearson_r_long = np.float(0)
        pearson_p_long = np.float(0)

    # print('speed corr long: ' + str(corr_speed_long))

    # calculate the number of trials in which the calcium signal went above the threshold
    if len(cur_trial_max_idx_long) >= 0:
        roi_active_fraction_long = np.float64(len(cur_trial_max_idx_long)/np.size(trials_long,0))
        if make_figure:
            sns.distplot(cur_trial_max_idx_long,hist=False,kde=False,rug=True,ax=ax2)
        roi_std_long = np.std(cur_trial_max_idx_long)
    # ax2.fill_between(np.arange(len(avg_mean_dF_long)), avg_mean_dF_long - sem_dF_l, avg_mean_dF_long + sem_dF_l, color = LONG_COLOR, alpha = 0.2)
    mean_valid_indices = []
    for i,trace in enumerate(mean_dF_long.T):
        if np.count_nonzero(np.isnan(trace))/trace.shape[0] < MEAN_TRACE_FRACTION:
            mean_valid_indices.append(i)

    mean_trace_long = np.nanmean(mean_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    # if a peak location is provided (the case when we want to know the neuron's response in openloop condition at its VR peak)
    if len(peak_locs) == 0:
        roi_meanpeak_long = np.nanmax(mean_trace_long)
        roi_meanpeak_long_idx = np.nanargmax(mean_trace_long)
        roi_meanpeak_long_location = (roi_meanpeak_long_idx+mean_valid_indices[0]) * bin_size
    else:
        roi_meanpeak_long_idx_vr = np.int64(peak_locs[1]/bin_size) - mean_valid_indices[0]
        if peak_tolerance > 0:
            try:
                win_start = roi_meanpeak_long_idx_vr-peak_tolerance
                win_end = roi_meanpeak_long_idx_vr+peak_tolerance
                win_start_offset = 0
                win_end_offset = 0
                if win_start < 0:
                    win_start = 0
                    win_start_offset = np.abs(roi_meanpeak_long_idx_vr-peak_tolerance)
                if win_end > mean_trace_long.shape[0]:
                    win_end = mean_trace_long.shape[0]
                    win_end_offset = (roi_meanpeak_long_idx_vr+peak_tolerance)-mean_trace_long.shape[0]
                roi_meanpeak_long_idx = roi_meanpeak_long_idx_vr + (np.nanargmax(mean_trace_long[win_start:win_end])-peak_tolerance+win_start_offset-win_end_offset)
                # roi_meanpeak_long_idx = np.nanargmax(mean_trace_long[max(0,roi_meanpeak_long_idx_vr-peak_tolerance):min(mean_trace_long.shape[0],roi_meanpeak_long_idx_vr+peak_tolerance)])
            except ValueError:
                roi_meanpeak_long_idx = roi_meanpeak_long_idx_vr
        else:
            roi_meanpeak_long_idx = roi_meanpeak_long_idx_vr
        roi_meanpeak_long = mean_trace_long[roi_meanpeak_long_idx]
        roi_meanpeak_long_location = np.int64(peak_locs[1])

    if make_figure:
        ax2.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_trace_long,c='k',lw=3,zorder=4)
        ax2.axvline((roi_meanpeak_long_idx+mean_valid_indices[0]),c='b')

    
    # determine which transient peaks are within the specified window
    RF_transients_long = []
    RF_transients_residuals_long = []
    RF_transients_peak_long = []
    for j,mtd in enumerate(meanpeak_transient_distance_long):

        # translate bin number of mean peak into actual distances (cm)
        meanpeak_loc = (roi_meanpeak_long_idx+mean_valid_indices[0]) * bin_size
        mtd = mtd * bin_size
        # pull out peak values associated with transient_trial_start_idx
        mtp = meanpeak_transient_peak_long[j]
        # determine which transient peaks are within range
        mtp = mtp[np.where((mtd > (meanpeak_loc-max_peak_distance)) & (mtd < (meanpeak_loc+max_peak_distance)))]
        mtd = mtd[np.where((mtd > (meanpeak_loc-max_peak_distance)) & (mtd < (meanpeak_loc+max_peak_distance)))]
        RF_transients_long.append(mtd)
        RF_transients_residuals_long.append(mtd-meanpeak_loc)
        RF_transients_peak_long.append(mtp)

    # flatten list and calculate deviance paraameters
    RF_transients_residuals_long = [rftr for sublist in RF_transients_residuals_long for rftr in sublist]
    RF_transients_long = [rftr for sublist in RF_transients_long for rftr in sublist]
    RF_transients_peak_long = [rfpr for sublist in RF_transients_peak_long for rfpr in sublist]
    sns.rugplot(np.array(RF_transients_long)/bin_size,ax=ax2)
    mean_transient_distance_long = np.mean(RF_transients_residuals_long)
    mean_transient_std_long = np.std(RF_transients_residuals_long)
    mean_transient_sem_long = stats.sem(RF_transients_residuals_long)
    num_RF_transients_long = len(RF_transients_long)/len(trials_long)
    mean_transient_peak_long = np.mean(RF_transients_peak_long)

    mean_trace_long_start = np.int64(mean_valid_indices[0])

    # run through FILTER 1 for LONG trials and calculate avg dF/F for each bin and trial
    trials_long_succ = filter_trials( behav_ds, [], filterprops_long_1, trials_long)
    long_succ_speed = []
    trials_long_succ_num = len(trials_long_succ)
    if len(trials_long_succ) > 0:

        filtered_long_1_vr_peak, filtered_long_1_mean_trace, mean_dF_long_succ, long_succ_speed, filtered_long_1_mean_transient_sem, filtered_long_1_num_RF_transients, filtered_long_1_mean_transient_peak = response_max(behav_ds, dF_ds[:,roi], trials_long_succ, align_point, binnr_long, tracklength_long, mean_valid_indices, peak_locs, mean_trace_long, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax11, make_figure, LONG_COLOR, is_ol_sess)
        # filtered_long_1_vr_peak, filtered_long_1_mean_trace, mean_dF_long_succ, long_succ_speed, filtered_long_1_mean_transient_sem, filtered_long_1_num_RF_transients, filtered_long_1_mean_transient_peak = response_max(behav_ds, placehold, trials_long_succ, align_point, binnr_long, tracklength_long, mean_valid_indices, peak_locs, mean_trace_long, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax11, make_figure, LONG_COLOR, is_ol_sess)

    else:
        filtered_long_1_mean_trace = np.empty(0)
        filtered_long_1_vr_peak = np.float64(np.nan)
        filtered_long_1_mean_transient_sem = np.float64(np.nan)
        filtered_long_1_num_RF_transients = 0
        filtered_long_1_mean_transient_peak = np.float64(np.nan)

    # run through FILTER 2 for LONG trials and calculate avg dF/F for each bin and trial
    trials_long_unsucc = filter_trials( behav_ds, [], filterprops_long_2, trials_long)
    long_unsucc_speed = []
    trials_long_unsucc_num = len(trials_long_unsucc)
    if len(trials_long_unsucc) > 0:

        filtered_long_2_vr_peak, filtered_long_2_mean_trace, mean_dF_long_unsucc, long_unsucc_speed, filtered_long_2_mean_transient_sem, filtered_long_2_num_RF_transients, filtered_long_2_mean_transient_peak = response_max(behav_ds, dF_ds[:,roi], trials_long_unsucc, align_point, binnr_long, tracklength_long, mean_valid_indices, peak_locs, mean_trace_long, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax12, make_figure, LONG_COLOR, is_ol_sess)
        # filtered_long_2_vr_peak, filtered_long_2_mean_trace, mean_dF_long_unsucc, long_unsucc_speed, filtered_long_2_mean_transient_sem, filtered_long_2_num_RF_transients, filtered_long_2_mean_transient_peak = response_max(behav_ds, placehold, trials_long_unsucc, align_point, binnr_long, tracklength_long, mean_valid_indices, peak_locs, mean_trace_long, bin_size, peak_tolerance, transient_threshold, max_peak_distance, ax12, make_figure, LONG_COLOR, is_ol_sess)

    else:
        filtered_long_2_mean_trace = np.empty(0)
        filtered_long_2_vr_peak = np.float64(np.nan)
        filtered_long_2_mean_transient_sem = np.float64(np.nan)
        filtered_long_2_num_RF_transients = 0
        filtered_long_2_mean_transient_peak = np.float64(np.nan)
        # ax9.set_title(filterprops_short_1[0])
        # ax10.set_title(filterprops_short_2[0])
        # ax11.set_title(filterprops_long_1[0])
        # ax12.set_title(filterprops_long_2[0])


    # determine scaling of y-axis max values
    max_y_short = np.nanmax(np.nanmax(mean_dF_short))
    max_y_long = np.nanmax(np.nanmax(mean_dF_long))
    max_y = np.amax([max_y_short, max_y_long])
    heatmap_max = np.amax([np.nanmax(np.nanmean(mean_dF_short,axis=0)),np.nanmax(np.nanmean(mean_dF_long,axis=0))]) #+ 1

    if c_ylim != []:
        hmmin = c_ylim[0]
        hmmax = c_ylim[2]
    else:
        hmmin = 0
        hmmax = heatmap_max

    # plot heatmaps
    if make_figure:
        if hmmax >= 0:
            sns.heatmap(mean_dF_short,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_short.astype('int'),xticklabels=True,ax=ax3)
            sns.heatmap(mean_dF_long,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_long.astype('int'),xticklabels=False,ax=ax4)
            if len(short_succ_speed) > 0:
                sns.heatmap(mean_dF_short_succ,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_short_succ.astype('int'),xticklabels=True,ax=ax5)
                sns.heatmap(short_succ_speed,cbar=True,cmap='viridis',yticklabels=trials_short_succ.astype('int'),xticklabels=True,ax=ax13)
            if len(short_unsucc_speed) > 0:
                sns.heatmap(mean_dF_short_unsucc,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_short_unsucc.astype('int'),xticklabels=True,ax=ax6)
                sns.heatmap(short_unsucc_speed,cbar=True,cmap='viridis',yticklabels=trials_short_unsucc.astype('int'),xticklabels=True,ax=ax14)
            if len(long_succ_speed) > 0:
                sns.heatmap(mean_dF_long_succ,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_long_succ.astype('int'),xticklabels=True,ax=ax7)
                sns.heatmap(long_succ_speed,cbar=True,cmap='viridis',yticklabels=trials_long_succ.astype('int'),xticklabels=True,ax=ax15)
            if len(long_unsucc_speed) > 0:
                sns.heatmap(mean_dF_long_unsucc,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_long_unsucc.astype('int'),xticklabels=True,ax=ax8)
                sns.heatmap(long_unsucc_speed,cbar=True,cmap='viridis',yticklabels=trials_long_unsucc.astype('int'),xticklabels=True,ax=ax16)
        else:
            try:
                sns.heatmap(mean_dF_short,cbar=True,cmap='viridis',yticklabels=trials_short.astype('int'),xticklabels=True,ax=ax3)
                sns.heatmap(mean_dF_long,cbar=True,cmap='viridis',yticklabels=trials_long.astype('int'),xticklabels=False,ax=ax4)
                if len(short_succ_speed) > 0:
                    sns.heatmap(mean_dF_short_succ,cbar=True,cmap='viridis',yticklabels=trials_short_succ.astype('int'),xticklabels=True,ax=ax5)
                    sns.heatmap(short_succ_speed,cbar=True,cmap='viridis',yticklabels=trials_long_succ.astype('int'),xticklabels=True,ax=ax13)
                if len(short_unsucc_speed) > 0:
                    sns.heatmap(mean_dF_short_unsucc,cbar=True,cmap='viridis',yticklabels=trials_short_unsucc.astype('int'),xticklabels=True,ax=ax6)
                    sns.heatmap(short_unsucc_speed,cbar=True,cmap='viridis',yticklabels=trials_long_unsucc.astype('int'),xticklabels=True,ax=ax14)
                if len(long_succ_speed) > 0:
                    sns.heatmap(mean_dF_long_succ,cbar=True,cmap='viridis',yticklabels=trials_short_succ.astype('int'),xticklabels=True,ax=ax7)
                    sns.heatmap(long_succ_speed,cbar=True,cmap='viridis',yticklabels=trials_long_succ.astype('int'),xticklabels=True,ax=ax15)
                if len(long_unsucc_speed) > 0:
                    sns.heatmap(mean_dF_long_unsucc,cbar=True,cmap='viridis',yticklabels=trials_short_unsucc.astype('int'),xticklabels=True,ax=ax8)
                    sns.heatmap(long_unsucc_speed,cbar=True,cmap='viridis',yticklabels=trials_long_unsucc.astype('int'),xticklabels=True,ax=ax16)
            except ValueError:
                print('WARNING: ValueError encountered. Most likely min or max values for the heatmap are messed up.')

        if c_ylim == []:
            # ax1.set_ylim([-0.5,max_y])
            # ax2.set_ylim([-0.5,max_y])
            ax9.set_ylim([-0.5,max_y])
            ax10.set_ylim([-0.5,max_y])
            ax11.set_ylim([-0.5,max_y])
            ax12.set_ylim([-0.5,max_y])
            c_ylim = [-0.5,max_y,heatmap_max]
        else:

            ax1.set_ylim(c_ylim[0:2])
            ax2.set_ylim(c_ylim[0:2])
            ax9.set_ylim(c_ylim[0:2])
            ax10.set_ylim(c_ylim[0:2])
            ax11.set_ylim(c_ylim[0:2])
            ax12.set_ylim(c_ylim[0:2])

        ax1.set_title(str(np.round(mean_transient_sem_short,2)) + ' ' + str(np.round(num_RF_transients_short,2)) + ' ' + str(np.round(mean_transient_peak_short,2)), fontsize=8)
        ax2.set_title(str(np.round(mean_transient_sem_long,2)) + ' ' + str(np.round(num_RF_transients_long,2)) + ' ' + str(np.round(mean_transient_peak_long,2)), fontsize=8)

        # set axis labels
        ax1.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size])
        ax1.set_xticklabels(['0','100','200','300','400'])
        ax1.set_xlabel('Location (cm)')
        ax1.set_xlim([xmin,plot_binnr_short])

        ax2.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size,500/bin_size])
        ax2.set_xticklabels(['0','100','200','300','400','500'])
        ax2.set_xlabel('Location (cm)')
        ax2.set_xlim([xmin,plot_binnr_long])

        ax3.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size])
        ax3.set_xticklabels(['0','100','200','300','400'])
        ax3.set_xlim([xmin,plot_binnr_short])

        ax4.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size])
        ax4.set_xticklabels(['0','100','200','300','400'])
        ax4.set_xlim([xmin,plot_binnr_long])

        ax5.set_xlim([xmin,plot_binnr_short])
        ax5.set_xticks([])
        ax5.set_xticklabels([])
        ax6.set_xlim([xmin,plot_binnr_short])
        ax6.set_xticks([])
        ax6.set_xticklabels([])

        ax13.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size])
        ax13.set_xticklabels(['0','100','200','300','400'])
        ax13.set_xlim([xmin,plot_binnr_short])

        ax14.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size])
        ax14.set_xticklabels(['0','100','200','300','400'])
        ax14.set_xlim([xmin,plot_binnr_short])

        ax15.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size,500/bin_size])
        ax15.set_xticklabels(['0','100','200','300','400','500'])
        ax15.set_xlim([xmin,plot_binnr_long])

        ax8.set_xticks([])
        ax8.set_xticklabels([])
        ax8.set_xlim([xmin,plot_binnr_long])

        ax7.set_xticks([])
        ax7.set_xticklabels([])
        ax7.set_xlim([xmin,plot_binnr_long])

        ax16.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size,500/bin_size])
        ax16.set_xticklabels(['0','100','200','300','400','500'])
        ax16.set_xlim([xmin,plot_binnr_long])

        ax9.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size])
        ax9.set_xticklabels(['0','100','200','300','400'])
        ax9.set_xlim([xmin,plot_binnr_short])

        ax10.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size])
        ax10.set_xticklabels(['0','100','200','300','400'])
        ax10.set_xlim([xmin,plot_binnr_short])

        ax11.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size,500/bin_size])
        ax11.set_xticklabels(['0','100','200','300','400','500'])
        ax11.set_xlim([xmin,plot_binnr_long])

        ax12.set_xticks([0,100/bin_size,200/bin_size,300/bin_size,400/bin_size,500/bin_size])
        ax12.set_xticklabels(['0','100','200','300','400','500'])
        ax12.set_xlim([xmin,plot_binnr_long])

        plt.tight_layout()

        # fig.suptitle(fname, wrap=True)
        if not os.path.isdir(figure_output_path +  os.sep + subfolder):
            os.mkdir(figure_output_path + os.sep + subfolder)
        fname = figure_output_path +  os.sep +  subfolder + os.sep + fname + '.' + fformat
        print(fname)
        try:
            fig.savefig(fname, format=fformat, dpi=300)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

    # print(type(roi_std_short),type(roi_active_fraction_short),type(roi_meanpeak_short),type(roi_meanpeak_short_location),type(mean_trace_short_start))
    # if isinstance(corr_speed_p_short, float) or isinstance(corr_speed_p_long, float):
    #     print('############################################################')
    #     print(h5path, sess)
    #     print(corr_speed_p_short,corr_speed_p_long)
    #     print(type(corr_speed_p_short), type(corr_speed_p_long))
        # print('############################################################')
    # ipdb.set_trace()

    return roi_std_short.item(),roi_std_long.item(), \
           roi_active_fraction_short.item(),roi_active_fraction_long.item(), \
           roi_meanpeak_short.item(),roi_meanpeak_long.item(), \
           roi_meanpeak_short_location.item(),roi_meanpeak_long_location.item(), \
           mean_trace_short.tolist(),mean_trace_long.tolist(), \
           mean_trace_short_start.item(), mean_trace_long_start.item(), \
           mean_transient_sem_short.item(), mean_transient_sem_long.item(), \
           num_RF_transients_short, num_RF_transients_long, \
           mean_transient_peak_short.item(), mean_transient_peak_long.item(), \
           corr_speed_short.item(), corr_speed_long.item(), \
           corr_speed_r_short.item(), corr_speed_r_long.item(), \
           corr_speed_p_short.item(), corr_speed_p_long.item(), \
           fl_amp_short_r.item(), fl_amp_short_p.item(), \
           fl_amp_long_r.item(), fl_amp_long_p.item(), \
           fl_robust_short.item(), df_robust_short.item(), \
           fl_robust_long.item(), df_robust_long.item(), \
           fl_amp_short.item(), df_amp_short.item(), \
           fl_amp_long.item(), df_amp_long.item(), \
           fl_amplitude_close_short.tolist(), fl_amplitude_far_short.tolist(), \
           fl_amplitude_close_long.tolist(), fl_amplitude_far_long.tolist(), \
           filtered_short_1_mean_trace.tolist(), \
           filtered_short_2_mean_trace.tolist(), \
           filtered_long_1_mean_trace.tolist(), \
           filtered_long_2_mean_trace.tolist(), \
           filtered_short_1_vr_peak.item(), \
           filtered_short_2_vr_peak.item(), \
           filtered_long_1_vr_peak.item(), \
           filtered_long_2_vr_peak.item(), \
           filtered_short_1_mean_transient_sem.item(), \
           filtered_long_1_mean_transient_sem.item(), \
           filtered_short_1_num_RF_transients, \
           filtered_long_1_num_RF_transients, \
           filtered_short_1_mean_transient_peak.item(), \
           filtered_long_1_mean_transient_peak.item(), \
           filtered_short_2_mean_transient_sem.item(), \
           filtered_long_2_mean_transient_sem.item(), \
           filtered_short_2_num_RF_transients, \
           filtered_long_2_num_RF_transients, \
           filtered_short_2_mean_transient_peak.item(), \
           filtered_long_2_mean_transient_peak.item(), \
           trials_short_succ_num, trials_short_unsucc_num, trials_long_succ_num, trials_long_unsucc_num, \
           c_ylim



def run_analysis(mousename, sessionname, sessionname_openloop, number_of_rois, h5_filepath, json_path, subname, align_point, sess_subfolder, session_rois, ol_sess=False, trial_onset=False, load_raw=False,use_data='aligned_data.mat', suffix=''):
    """ set up function call and dictionary to collect results """

    MOUSE = mousename
    SESSION = sessionname
    SESSION_OPENLOOP = sessionname_openloop
    NUM_ROIS = number_of_rois
    h5path = h5_filepath
    SUBNAME = subname
    subfolder = sess_subfolder
    if ol_sess and suffix == '':
        subfolder_ol = sess_subfolder + '_ol' 
    elif ol_sess and suffix != '':
        subfolder_ol = sess_subfolder + '_' + suffix
    write_to_dict = False
    make_figure = True
    subfolder_ol = sess_subfolder + '_ol' 

    ol_sess=False

    # set up dictionary to hold result parameters from roi
    session_rois[SUBNAME+'_roi_number'] = []
    session_rois[SUBNAME+'_roi_number_ol'] = []
    session_rois[SUBNAME+'_std_short'] = []
    session_rois[SUBNAME+'_std_long'] = []
    session_rois[SUBNAME+'_active_short'] = []
    session_rois[SUBNAME+'_active_long'] = []
    session_rois[SUBNAME+'_peak_short'] = []
    session_rois[SUBNAME+'_peak_long'] = []
    session_rois[SUBNAME+'_peak_loc_short'] = []
    session_rois[SUBNAME+'_peak_loc_long'] = []
    session_rois[SUBNAME+'_mean_trace_short'] = []
    session_rois[SUBNAME+'_mean_trace_long'] = []
    session_rois[SUBNAME+'_mean_trace_start_short'] = []
    session_rois[SUBNAME+'_mean_trace_start_long'] = []
    session_rois[SUBNAME+'_meanpeak_trial_deviance_short'] = []
    session_rois[SUBNAME+'_meanpeak_trial_deviance_long'] = []
    session_rois[SUBNAME+'_meanpeak_num_RF_transients_short'] = []
    session_rois[SUBNAME+'_meanpeak_num_RF_transients_long'] = []
    session_rois[SUBNAME+'_meanpeak_transient_meanpeak_short'] = []
    session_rois[SUBNAME+'_meanpeak_transient_meanpeak_long'] = []
    session_rois[SUBNAME+'_transient_speed_slope_short'] = []
    session_rois[SUBNAME+'_transient_speed_slope_long'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonr_short'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonp_short'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonr_long'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonp_long'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonr_short'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonp_short'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonr_long'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonp_long'] = []
    session_rois[SUBNAME+'_success_robustness_short'] = []
    session_rois[SUBNAME+'_success_robustness_long'] = []
    session_rois[SUBNAME+'_default_robustness_short'] = []
    session_rois[SUBNAME+'_default_robustness_long'] = []
    session_rois[SUBNAME+'_success_amplitude_short'] = []
    session_rois[SUBNAME+'_success_amplitude_long'] = []
    session_rois[SUBNAME+'_default_amplitude_short'] = []
    session_rois[SUBNAME+'_default_amplitude_long'] = []
    session_rois[SUBNAME+'_fl_amplitude_close_short'] = []
    session_rois[SUBNAME+'_fl_amplitude_far_short'] = []
    session_rois[SUBNAME+'_fl_amplitude_close_long'] = []
    session_rois[SUBNAME+'_fl_amplitude_far_long'] = []
    session_rois[SUBNAME+'_fl_amplitude_close_short_ol'] = []
    session_rois[SUBNAME+'_fl_amplitude_far_short_ol'] = []
    session_rois[SUBNAME+'_fl_amplitude_close_long_ol'] = []
    session_rois[SUBNAME+'_fl_amplitude_far_long_ol'] = []

    session_rois[SUBNAME+'_std_short_ol'] = []
    session_rois[SUBNAME+'_std_long_ol'] = []
    session_rois[SUBNAME+'_active_short_ol'] = []
    session_rois[SUBNAME+'_active_long_ol'] = []
    session_rois[SUBNAME+'_peak_short_ol'] = []
    session_rois[SUBNAME+'_peak_long_ol'] = []
    session_rois[SUBNAME+'_peak_loc_short_ol'] = []
    session_rois[SUBNAME+'_peak_loc_long_ol'] = []
    session_rois[SUBNAME+'_mean_trace_short_ol'] = []
    session_rois[SUBNAME+'_mean_trace_long_ol'] = []
    session_rois[SUBNAME+'_mean_trace_start_short_ol'] = []
    session_rois[SUBNAME+'_mean_trace_start_long_ol'] = []
    session_rois[SUBNAME+'_meanpeak_trial_deviance_short_ol'] = []
    session_rois[SUBNAME+'_meanpeak_trial_deviance_long_ol'] = []
    session_rois[SUBNAME+'_meanpeak_num_RF_transients_short_ol'] = []
    session_rois[SUBNAME+'_meanpeak_num_RF_transients_long_ol'] = []
    session_rois[SUBNAME+'_meanpeak_transient_meanpeak_short_ol'] = []
    session_rois[SUBNAME+'_meanpeak_transient_meanpeak_long_ol'] = []
    session_rois[SUBNAME+'_transient_speed_slope_short_ol'] = []
    session_rois[SUBNAME+'_transient_speed_slope_long_ol'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonr_short_ol'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonp_short_ol'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonr_long_ol'] = []
    session_rois[SUBNAME+'_transient_speed_pearsonp_long_ol'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonr_short_ol'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonp_short_ol'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonr_long_ol'] = []
    session_rois[SUBNAME+'_fl_distamp_pearsonp_long_ol'] = []
    session_rois[SUBNAME+'_success_robustness_short_ol'] = []
    session_rois[SUBNAME+'_success_robustness_long_ol'] = []

    session_rois[SUBNAME+'_filter_1_mean_trace_short'] = []
    session_rois[SUBNAME+'_filter_1_mean_trace_long'] = []
    session_rois[SUBNAME+'_filter_2_mean_trace_short'] = []
    session_rois[SUBNAME+'_filter_2_mean_trace_long'] = []
    session_rois[SUBNAME+'_filter_1_peak_short'] = []
    session_rois[SUBNAME+'_filter_1_peak_long'] = []
    session_rois[SUBNAME+'_filter_2_peak_short'] = []
    session_rois[SUBNAME+'_filter_2_peak_long'] = []
    session_rois[SUBNAME+'_filter_1_numtrials_short'] = []
    session_rois[SUBNAME+'_filter_1_numtrials_long'] = []
    session_rois[SUBNAME+'_filter_2_numtrials_short'] = []
    session_rois[SUBNAME+'_filter_2_numtrials_long'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_short'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_long'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_short'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_long'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_short'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_long'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_short'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_long'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_short'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_long'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_short'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_long'] = []
    session_rois[SUBNAME+'_filter_1_mean_trace_short_ol'] = []
    session_rois[SUBNAME+'_filter_1_mean_trace_long_ol'] = []
    session_rois[SUBNAME+'_filter_2_mean_trace_short_ol'] = []
    session_rois[SUBNAME+'_filter_2_mean_trace_long_ol'] = []
    session_rois[SUBNAME+'_filter_1_peak_short_ol'] = []
    session_rois[SUBNAME+'_filter_1_peak_long_ol'] = []
    session_rois[SUBNAME+'_filter_2_peak_short_ol'] = []
    session_rois[SUBNAME+'_filter_2_peak_long_ol'] = []
    session_rois[SUBNAME+'_filter_1_numtrials_short_ol'] = []
    session_rois[SUBNAME+'_filter_1_numtrials_long_ol'] = []
    session_rois[SUBNAME+'_filter_2_numtrials_short_ol'] = []
    session_rois[SUBNAME+'_filter_2_numtrials_long_ol'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_short_ol'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_long_ol'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_short_ol'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_long_ol'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_short_ol'] = []
    session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_long_ol'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_short_ol'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_long_ol'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_short_ol'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_long_ol'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_short_ol'] = []
    session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_long_ol'] = []

    # run analysis for vr session
    if type(NUM_ROIS) is int:
        roilist = range(NUM_ROIS)
    else:
        roilist = NUM_ROIS

    # if we want to run through all the rois, just say all
    if NUM_ROIS == 'all' and load_raw == False:
#        h5dat = h5py.File(h5path, 'r')
#        dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
#        h5dat.close()
#        roilist = np.arange(0,dF_ds.shape[1],1).tolist()
#        # write_to_dict = True
#        print('number of rois: ' + str(NUM_ROIS))
        df_signal_path = h5_filepath  + os.sep + use_data + '_dF_aligned.mat'
        behavior_path = h5_filepath  + os.sep + use_data + '_behavior_aligned.mat'
        dF_aligned = sio.loadmat(df_signal_path)['data']
        behaviour_aligned = sio.loadmat(behavior_path)['data']
        roilist = range(len(dF_aligned[0]))
#        roilist = lotalota
    elif NUM_ROIS == 'all' and load_raw == True:
        processed_data_path = h5path + os.sep + SESSION + os.sep + use_data
        loaded_data = sio.loadmat(processed_data_path)
        behaviour_aligned = loaded_data['behaviour_aligned']
        dF_aligned = loaded_data['calcium_dF']
        
        roilist = np.arange(0,dF_aligned.shape[1],1).tolist()
    elif NUM_ROIS == 'valid':
        # only use valid rois
        with open(json_path, 'r') as f:
            sess_dict = json.load(f)
        roilist = sess_dict['valid_rois']
        print('analysing ' + NUM_ROIS + ' rois: ' + str(roilist))
    else:
        print('analysing custom list of rois: ' + str(roilist))
    #
    # with open(json_path, 'r') as f:
    #     sess_dict = json.load(f)
    # print(sess_dict['valid_rois'])
    for r in roilist:
        print(SUBNAME + ': ' + str(r))

        std_short, std_long, \
        active_short, active_long, \
        peak_short, peak_long, \
        meanpeak_short_loc, meanpeak_long_loc, \
        mean_trace_short, mean_trace_long, \
        mean_trace_short_start, mean_trace_long_start, \
        mean_transient_std_short, mean_transient_std_long, \
        num_RF_transients_short, num_RF_transients_long, \
        mean_transient_peak_short, mean_transient_peak_long, \
        corr_speed_short, corr_speed_long, \
        corr_speed_r_short, corr_speed_r_long, \
        corr_speed_p_short, corr_speed_p_long, \
        fl_amp_short_r, fl_amp_short_p, \
        fl_amp_long_r, fl_amp_long_p, \
        fl_robust_short, df_robust_short, \
        fl_robust_long, df_robust_long, \
        fl_amp_short, df_amp_short, \
        fl_amp_long, df_amp_long, \
        fl_amplitude_close_short, fl_amplitude_far_short, \
        fl_amplitude_close_long, fl_amplitude_far_long, \
        reward_mean_trace_short,missed_mean_trace_short, \
        reward_mean_trace_long,missed_mean_trace_long, \
        reward_peak_short,missed_peak_short, \
        reward_peak_long,missed_peak_long, \
        filter_1_meanpeak_trial_deviance_short, filter_1_meanpeak_trial_deviance_long, \
        filter_1_meanpeak_num_RF_transients_short, filter_1_meanpeak_num_RF_transients_long, \
        filter_1_meanpeak_transient_meanpeak_short, filter_1_meanpeak_transient_meanpeak_long, \
        filter_2_meanpeak_trial_deviance_short, filter_2_meanpeak_trial_deviance_long, \
        filter_2_meanpeak_num_RF_transients_short, filter_2_meanpeak_num_RF_transients_long, \
        filter_2_meanpeak_transient_meanpeak_short, filter_2_meanpeak_transient_meanpeak_long, \
        trials_short_succ_num, trials_short_unsucc_num, trials_long_succ_num, trials_long_unsucc_num, c_ylim = \
            fig_dfloc_trace_roiparams(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), align_point, [], ['trial_successful'], ['trial_unsuccessful'],['trial_successful'],['trial_unsuccessful'], fformat, subfolder, [], make_figure, load_raw, use_data, False)

        # print(fl_amplitude_close_short)
        session_rois[SUBNAME+'_roi_number'].append(r)
        session_rois[SUBNAME+'_std_short'].append(std_short)
        session_rois[SUBNAME+'_std_long'].append(std_long)
        session_rois[SUBNAME+'_active_short'].append(active_short)
        session_rois[SUBNAME+'_active_long'].append(active_long)
        session_rois[SUBNAME+'_peak_short'].append(peak_short)
        session_rois[SUBNAME+'_peak_long'].append(peak_long)
        session_rois[SUBNAME+'_peak_loc_short'].append(meanpeak_short_loc)
        session_rois[SUBNAME+'_peak_loc_long'].append(meanpeak_long_loc)
        session_rois[SUBNAME+'_mean_trace_short'].append(mean_trace_short)
        session_rois[SUBNAME+'_mean_trace_long'].append(mean_trace_long)
        session_rois[SUBNAME+'_mean_trace_start_short'].append(mean_trace_short_start)
        session_rois[SUBNAME+'_mean_trace_start_long'].append(mean_trace_long_start)
        session_rois[SUBNAME+'_meanpeak_trial_deviance_short'].append(mean_transient_std_short)
        session_rois[SUBNAME+'_meanpeak_trial_deviance_long'].append(mean_transient_std_long)
        session_rois[SUBNAME+'_meanpeak_num_RF_transients_short'].append(num_RF_transients_short)
        session_rois[SUBNAME+'_meanpeak_num_RF_transients_long'].append(num_RF_transients_long)
        session_rois[SUBNAME+'_meanpeak_transient_meanpeak_short'].append(mean_transient_peak_short)
        session_rois[SUBNAME+'_meanpeak_transient_meanpeak_long'].append(mean_transient_peak_short)
        session_rois[SUBNAME+'_transient_speed_slope_short'].append(corr_speed_short)
        session_rois[SUBNAME+'_transient_speed_slope_long'].append(corr_speed_long)
        session_rois[SUBNAME+'_transient_speed_pearsonr_short'].append(corr_speed_r_short)
        session_rois[SUBNAME+'_transient_speed_pearsonp_short'].append(corr_speed_p_short)
        session_rois[SUBNAME+'_transient_speed_pearsonr_long'].append(corr_speed_r_long)
        session_rois[SUBNAME+'_transient_speed_pearsonp_long'].append(corr_speed_p_long)
        session_rois[SUBNAME+'_fl_distamp_pearsonr_short'].append(fl_amp_short_r)
        session_rois[SUBNAME+'_fl_distamp_pearsonp_short'].append(fl_amp_short_p)
        session_rois[SUBNAME+'_fl_distamp_pearsonr_long'].append(fl_amp_long_r)
        session_rois[SUBNAME+'_fl_distamp_pearsonp_long'].append(fl_amp_long_p)
        session_rois[SUBNAME+'_fl_amplitude_close_short'].append(fl_amplitude_close_short)
        session_rois[SUBNAME+'_fl_amplitude_far_short'].append(fl_amplitude_far_short)
        session_rois[SUBNAME+'_fl_amplitude_close_long'].append(fl_amplitude_close_long)
        session_rois[SUBNAME+'_fl_amplitude_far_long'].append(fl_amplitude_far_long)
        session_rois[SUBNAME+'_success_robustness_short'].append(fl_robust_short)
        session_rois[SUBNAME+'_success_robustness_long'].append(fl_robust_long)
        session_rois[SUBNAME+'_default_robustness_short'].append(df_robust_short)
        session_rois[SUBNAME+'_default_robustness_long'].append(df_robust_long)
        session_rois[SUBNAME+'_success_amplitude_short'].append(fl_amp_short)
        session_rois[SUBNAME+'_success_amplitude_long'].append(fl_amp_long)
        session_rois[SUBNAME+'_default_amplitude_short'].append(df_amp_short)
        session_rois[SUBNAME+'_default_amplitude_long'].append(df_amp_long)
        session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_short'].append(filter_1_meanpeak_trial_deviance_short)
        session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_long'].append(filter_1_meanpeak_trial_deviance_long)
        session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_short'].append(filter_1_meanpeak_num_RF_transients_short)
        session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_long'].append(filter_1_meanpeak_num_RF_transients_long)
        session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_short'].append(filter_1_meanpeak_transient_meanpeak_short)
        session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_long'].append(filter_1_meanpeak_transient_meanpeak_long)
        session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_short'].append(filter_2_meanpeak_trial_deviance_short)
        session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_long'].append(filter_2_meanpeak_trial_deviance_long)
        session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_short'].append(filter_2_meanpeak_num_RF_transients_short)
        session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_long'].append(filter_2_meanpeak_num_RF_transients_long)
        session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_short'].append(filter_2_meanpeak_transient_meanpeak_short)
        session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_long'].append(filter_2_meanpeak_transient_meanpeak_long)
        session_rois[SUBNAME+'_filter_1_mean_trace_short'].append(reward_mean_trace_short)
        session_rois[SUBNAME+'_filter_1_mean_trace_long'].append(missed_mean_trace_short)
        session_rois[SUBNAME+'_filter_2_mean_trace_short'].append(reward_mean_trace_long)
        session_rois[SUBNAME+'_filter_2_mean_trace_long'].append(missed_mean_trace_long)
        session_rois[SUBNAME+'_filter_1_peak_short'].append(reward_peak_short)
        session_rois[SUBNAME+'_filter_2_peak_short'].append(missed_peak_short)
        session_rois[SUBNAME+'_filter_1_peak_long'].append(reward_peak_long)
        session_rois[SUBNAME+'_filter_2_peak_long'].append(missed_peak_long)
        session_rois[SUBNAME+'_filter_1_numtrials_short'].append(trials_short_succ_num)
        session_rois[SUBNAME+'_filter_2_numtrials_short'].append(trials_short_unsucc_num)
        session_rois[SUBNAME+'_filter_1_numtrials_long'].append(trials_long_succ_num)
        session_rois[SUBNAME+'_filter_2_numtrials_long'].append(trials_long_unsucc_num)

        if ol_sess:
            # print(meanpeak_short_loc)
            # N.B.: filter_1 is not running (=below threshold), filter_2 is running (=above threshold)
            std_short, std_long, \
            active_short, active_long, \
            peak_short, peak_long, \
            meanpeak_short_loc, meanpeak_long_loc, \
            mean_trace_short, mean_trace_long, \
            mean_trace_short_start, mean_trace_long_start, \
            mean_transient_std_short, mean_transient_std_long, \
            num_RF_transients_short, num_RF_transients_long, \
            mean_transient_peak_short, mean_transient_peak_long, \
            corr_speed_short, corr_speed_long, \
            corr_speed_r_short, corr_speed_r_long, \
            corr_speed_p_short, corr_speed_p_long, \
            fl_amp_short_r, fl_amp_short_p, \
            fl_amp_long_r, fl_amp_long_p, \
            fl_robust_short, df_robust_short, \
            fl_robust_long, df_robust_long, \
            fl_amp_short, df_amp_short, \
            fl_amp_long, df_amp_long, \
            fl_amplitude_close, fl_amplitude_far, \
            fl_amplitude_close_long, fl_amplitude_far_long, \
            run_passive_mean_trace_short,norun_passive_mean_trace_short, \
            run_passive_mean_trace_long,norun_passive_mean_trace_long, \
            run_peak_short,norun_peak_short, \
            run_peak_long,norun_peak_long, \
            filter_1_meanpeak_trial_deviance_short_ol, filter_1_meanpeak_trial_deviance_long_ol, \
            filter_1_meanpeak_num_RF_transients_short_ol, filter_1_meanpeak_num_RF_transients_long_ol, \
            filter_1_meanpeak_transient_meanpeak_short_ol, filter_1_meanpeak_transient_meanpeak_long_ol, \
            filter_2_meanpeak_trial_deviance_short_ol, filter_2_meanpeak_trial_deviance_long_ol, \
            filter_2_meanpeak_num_RF_transients_short_ol, filter_2_meanpeak_num_RF_transients_long_ol, \
            filter_2_meanpeak_transient_meanpeak_short_ol, filter_2_meanpeak_transient_meanpeak_long_ol, \
            trials_short_succ_num, trials_short_unsucc_num, trials_long_succ_num, trials_long_unsucc_num,_ = \
            fig_dfloc_trace_roiparams(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), align_point, [meanpeak_short_loc, meanpeak_long_loc], ['animal_running',3,meanpeak_short_loc,[50,50], True, trial_onset],  ['animal_running',3,meanpeak_short_loc,[50,50], False, trial_onset],['animal_running',3,meanpeak_long_loc,[50,50], True, trial_onset],  ['animal_running',3,meanpeak_long_loc,[50,50], False, trial_onset], fformat, subfolder_ol, c_ylim, make_figure, load_raw,use_data,True)

            session_rois[SUBNAME+'_roi_number_ol'].append(r)
            session_rois[SUBNAME+'_std_short_ol'].append(std_short)
            session_rois[SUBNAME+'_std_long_ol'].append(std_long)
            session_rois[SUBNAME+'_active_short_ol'].append(active_short)
            session_rois[SUBNAME+'_active_long_ol'].append(active_long)
            session_rois[SUBNAME+'_peak_short_ol'].append(peak_short)
            session_rois[SUBNAME+'_peak_long_ol'].append(peak_long)
            session_rois[SUBNAME+'_peak_loc_short_ol'].append(meanpeak_short_loc)
            session_rois[SUBNAME+'_peak_loc_long_ol'].append(meanpeak_long_loc)
            session_rois[SUBNAME+'_mean_trace_short_ol'].append(mean_trace_short)
            session_rois[SUBNAME+'_mean_trace_long_ol'].append(mean_trace_long)
            session_rois[SUBNAME+'_mean_trace_start_short_ol'].append(mean_trace_short_start)
            session_rois[SUBNAME+'_mean_trace_start_long_ol'].append(mean_trace_long_start)
            session_rois[SUBNAME+'_meanpeak_trial_deviance_short_ol'].append(mean_transient_std_short)
            session_rois[SUBNAME+'_meanpeak_trial_deviance_long_ol'].append(mean_transient_std_long)
            session_rois[SUBNAME+'_meanpeak_num_RF_transients_short_ol'].append(num_RF_transients_short)
            session_rois[SUBNAME+'_meanpeak_num_RF_transients_long_ol'].append(num_RF_transients_long)
            session_rois[SUBNAME+'_meanpeak_transient_meanpeak_short_ol'].append(mean_transient_peak_short)
            session_rois[SUBNAME+'_meanpeak_transient_meanpeak_long_ol'].append(mean_transient_peak_short)
            session_rois[SUBNAME+'_transient_speed_slope_short_ol'].append(corr_speed_short)
            session_rois[SUBNAME+'_transient_speed_slope_long_ol'].append(corr_speed_long)
            session_rois[SUBNAME+'_transient_speed_pearsonr_short_ol'].append(corr_speed_r_short)
            session_rois[SUBNAME+'_transient_speed_pearsonp_short_ol'].append(corr_speed_p_short)
            session_rois[SUBNAME+'_transient_speed_pearsonr_long_ol'].append(corr_speed_r_long)
            session_rois[SUBNAME+'_transient_speed_pearsonp_long_ol'].append(corr_speed_p_long)
            session_rois[SUBNAME+'_fl_distamp_pearsonr_short_ol'].append(fl_amp_short_r)
            session_rois[SUBNAME+'_fl_distamp_pearsonp_short_ol'].append(fl_amp_short_p)
            session_rois[SUBNAME+'_fl_distamp_pearsonr_long_ol'].append(fl_amp_long_r)
            session_rois[SUBNAME+'_fl_distamp_pearsonp_long_ol'].append(fl_amp_long_p)
            session_rois[SUBNAME+'_fl_amplitude_close_short_ol'].append(fl_amplitude_close_short)
            session_rois[SUBNAME+'_fl_amplitude_far_short_ol'].append(fl_amplitude_far_short)
            session_rois[SUBNAME+'_fl_amplitude_close_long_ol'].append(fl_amplitude_close_long)
            session_rois[SUBNAME+'_fl_amplitude_far_long_ol'].append(fl_amplitude_far_long)
            session_rois[SUBNAME+'_filter_1_mean_trace_short_ol'].append(run_passive_mean_trace_short)
            session_rois[SUBNAME+'_filter_1_mean_trace_long_ol'].append(norun_passive_mean_trace_short)
            session_rois[SUBNAME+'_filter_2_mean_trace_short_ol'].append(run_passive_mean_trace_long)
            session_rois[SUBNAME+'_filter_2_mean_trace_long_ol'].append(norun_passive_mean_trace_long)
            session_rois[SUBNAME+'_filter_1_peak_short_ol'].append(run_peak_short)
            session_rois[SUBNAME+'_filter_2_peak_short_ol'].append(norun_peak_short)
            session_rois[SUBNAME+'_filter_1_peak_long_ol'].append(run_peak_long)
            session_rois[SUBNAME+'_filter_2_peak_long_ol'].append(norun_peak_long)
            session_rois[SUBNAME+'_filter_1_numtrials_short_ol'].append(trials_short_succ_num)
            session_rois[SUBNAME+'_filter_2_numtrials_short_ol'].append(trials_short_unsucc_num)
            session_rois[SUBNAME+'_filter_1_numtrials_long_ol'].append(trials_long_succ_num)
            session_rois[SUBNAME+'_filter_2_numtrials_long_ol'].append(trials_long_unsucc_num)
            session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_short_ol'].append(filter_1_meanpeak_trial_deviance_short_ol)
            session_rois[SUBNAME+'_filter_1_meanpeak_trial_deviance_long_ol'].append(filter_1_meanpeak_trial_deviance_long_ol)
            session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_short_ol'].append(filter_1_meanpeak_num_RF_transients_short_ol)
            session_rois[SUBNAME+'_filter_1_meanpeak_num_RF_transients_long_ol'].append(filter_1_meanpeak_num_RF_transients_long_ol)
            session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_short_ol'].append(filter_1_meanpeak_transient_meanpeak_short_ol)
            session_rois[SUBNAME+'_filter_1_meanpeak_transient_meanpeak_long_ol'].append(filter_1_meanpeak_transient_meanpeak_long_ol)
            session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_short_ol'].append(filter_2_meanpeak_trial_deviance_short_ol)
            session_rois[SUBNAME+'_filter_2_meanpeak_trial_deviance_long_ol'].append(filter_2_meanpeak_trial_deviance_long_ol)
            session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_short_ol'].append(filter_2_meanpeak_num_RF_transients_short_ol)
            session_rois[SUBNAME+'_filter_2_meanpeak_num_RF_transients_long_ol'].append(filter_2_meanpeak_num_RF_transients_long_ol)
            session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_short_ol'].append(filter_2_meanpeak_transient_meanpeak_short_ol)
            session_rois[SUBNAME+'_filter_2_meanpeak_transient_meanpeak_long_ol'].append(filter_2_meanpeak_transient_meanpeak_long_ol)

    # ipdb.set_trace()
    if write_to_dict:
        print('writing to dictionary.')
        write_dict(MOUSE, SESSION, session_rois, False, True, suffix)

    return session_rois


def run_for_mouse(MOUSE, sessions):
    
    data_path = 'C:/Users/lfisc/Work/Projects/Lntmodel/data_2p/dataset' + os.sep
    base_path = 'C:/Users/lfisc/Google Drive/MTH3_data/animals_raw'
    for s in sessions:
        SESSION = '2019'+str(s)
        use_data = MOUSE + '_' + SESSION
        
        SESSION_OPENLOOP = ''
        NUM_ROIS = 'all' # 'all' #52
        json_path = base_path + os.sep + MOUSE + os.sep + SESSION + '.json'
        # dictionary that will hold the results of the analyses
        roi_result_params = {
            'mouse_session' : MOUSE+'_'+SESSION,
            'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
        }
        #
        SUBNAME = 'space'
        align_point = 'landmark'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, data_path, json_path, SUBNAME, align_point, subfolder, roi_result_params, False, False, False, use_data, '')


if __name__ == '__main__':

    # 
    # run_for_mouse('LF191022_1', [1114,1115,1121,1125,1204,1207,1209,1211,1213,1215,1217])
    run_for_mouse('LF191022_1', [1204])
    # run_for_mouse('LF191022_2', [1114,1116,1121,1204,1206,1208,1210,1212,1216])
    # run_for_mouse('LF191022_3', [1113,1114,1119,1121,1125,1204,1207,1210,1211,1215,1217])
    # run_for_mouse('LF191023_blank', [1114,1116,1121,1206,1208,1210,1212,1213,1216,1217])
    # run_for_mouse('LF191023_blue', [1113,1114,1119,1121,1125,1204,1206,1208,1210,1212,1215,1217])
    # run_for_mouse('LF191024_1', [1114,1115,1121,1204,1207,1210])

    print('done')








