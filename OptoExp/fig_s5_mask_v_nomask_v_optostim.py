"""
Plot basic behaviour of a given animal and day.

"""

import numpy as np
import h5py
import warnings
import os
import sys
import yaml
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import statsmodels.api as sm
import scipy as sp
from scipy import stats
from scipy.signal import butter, filtfilt
import ipdb


import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

with open(loc_info['yaml_file'], 'r') as f:
    project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')
sys.path.append(loc_info['base_dir'] + 'Figures')

from filter_trials import filter_trials
# from fig_behavior_stage5 import fig_behavior_stage5 as fig_behavior_stage5_normal

MAKE_IND_FIGURES = True
SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def plot_licks(ax, licks, rewards, raw, trials, scatter_color):
    # scatterplot of licks/rewards in order of trial number
    trial_start = np.empty((0,3))
    trial_start_trials = np.empty((0))
    all_licks = []
    all_rewards_succ = 0
    all_rewards_fail = 0
    for i,r in enumerate(trials):
        plot_licks_x = licks[licks[:,2]==r,1]
        plot_rewards_x = rewards[rewards[:,3]==r,1]
        cur_trial_start = raw[raw[:,6]==r,1][0]
        cur_trial_start_time = raw[raw[:,6]==r,0][0]
        cur_trial_end_time = raw[raw[:,6]==r,0][-1]
        trial_start = np.vstack((trial_start, [cur_trial_start,cur_trial_start_time,cur_trial_end_time]))
        trial_start_trials = np.append(trial_start_trials, r)
        if rewards[rewards[:,3]==r,5] == 1:
            col = '#00C40E'
            all_rewards_succ = all_rewards_succ + 1
        else:
            col = 'r'
            all_rewards_fail = all_rewards_fail + 1

        # if reward location is recorded at beginning of track, set it to end of track
        if plot_rewards_x < 300:
            plot_rewards_x = 340

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],i)
            ax.scatter(plot_licks_x, plot_licks_y,c=scatter_color,lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = i
            ax.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
        if np.size(cur_trial_start) > 0:
            plot_starts_y = i
            ax.scatter(cur_trial_start, plot_starts_y,c=scatter_color,marker='>',lw=0)

        # clause below implemented because we have a single instance where an animal apparently licked >150 times in almost the same location --> extensive grooming?
        if len(plot_licks_x.tolist()) < 150:
            all_licks.append(len(plot_licks_x.tolist()))

    # ipdb.set_trace()
    if all_rewards_succ > 0:
        fraction_successful = all_rewards_fail/all_rewards_succ
    else:
        fraction_successful = np.nan
    return all_licks, fraction_successful

def plot_first_licks(ax, licks, rewards, trials, scatter_color, plot_to_axis=True):
    # plot location of first trials on short and long trials
    first_lick = np.empty((0,4))
    first_lick_trials = np.empty((0))
    for r in trials:
        licks_all = licks[licks[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>101,:]
        if licks_all.size == 0:
             rew_lick = rewards[rewards[:,3]==r,:]
             if rew_lick.size > 0:
                 if rew_lick[0,5] == 1:
                     licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                     first_lick = np.vstack((first_lick, licks_all[0,:].T))
                     first_lick_trials = np.append(first_lick_trials, r)
        else:
            if licks_all[0,3] == 3:
                licks_all = licks_all[licks_all[:,1]<338,:]
            elif licks_all[0,3] == 4:
                licks_all = licks_all[licks_all[:,1]<398,:]
            first_lick = np.vstack((first_lick, licks_all[0,:].T))
            first_lick_trials = np.append(first_lick_trials, r)

    if plot_to_axis:
        if first_lick.size > 0:
            # print(first_lick[:,1],first_lick_trials,scatter_color)
            ax.scatter(first_lick[:,1],first_lick_trials,c=scatter_color,lw=0)
            ax_2 = ax.twinx()
            sns.kdeplot(first_lick[:,1],c=scatter_color,ax=ax_2)
            ax_2.set_xlim([50,400])
            ax_2.set_yticklabels([''])

        ax.axvline(101,lw=2,ls='--',c='0.8')
        ax.axvline(np.median(first_lick[:,1]), ls='--', c=scatter_color)
        ax.set_xlim([50,400])

        ax.set_yticklabels([''])

    return first_lick

def plot_running_speed(ax, raw, trials, binsize, binnr, plot_color):
    # set up speed filters
    order = 6
    fs = int(np.size(raw,0)/raw[-1,0])
    cutoff = 1
    speed_vector = butter_lowpass_filter(raw[:,3], cutoff, fs, order)
    # plot running speed
    all_speed_vals = np.empty(0)
    mean_speed = np.empty((trials.shape[0],int(binnr)))
    mean_speed[:] = np.NAN
    max_y_short = 0

    for i,t in enumerate(trials):
        cur_trial = raw[raw[:,6]==t,:]
        cur_trial_start_bin = np.floor(cur_trial[0,1]/binsize)
        cur_trial_end_bin = np.ceil(cur_trial[-1,1]/binsize)
        cur_trial_bins = cur_trial_end_bin - cur_trial_start_bin
        # make sure the session didn't end right after a trial started causing the script to trip
        if cur_trial_bins > 0:
            cur_trial_speed = speed_vector[raw[:,6]==t]
            cur_trial_speed[cur_trial_speed>80] = np.nan
            all_speed_vals = np.concatenate((all_speed_vals,cur_trial_speed[cur_trial_speed>3]))
            mean_speed_trial = stats.binned_statistic(cur_trial[:,1], cur_trial_speed, 'mean', cur_trial_bins)[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_end_bin)] = mean_speed_trial
            # ax.plot(np.linspace(cur_trial_start_bin,cur_trial_end_bin,cur_trial_bins),mean_speed_trial,c='0.8',alpha=0.5,zorder=2)
        #     max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])
    #
    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax.plot(np.linspace(0,binnr-1,binnr),mean_speed_sess_short,lw=2,c=plot_color,zorder=3)
    ax.fill_between(np.linspace(0,binnr-1,binnr),mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, linewidth=0, color=plot_color,alpha=0.2)
    ax.set_xlim([50/binsize,binnr])
    # ax.set_ylim([0,np.nanmax(mean_speed_sess_short)])

    return all_speed_vals

def shuffled_task_score(mask_on_trials_short, mask_on_trials_long, licks_ds, reward_ds, actual_TS, plot_color, plot_ax):

    # determine number of trials for each type
    num_short_trials = len(mask_on_trials_short)
    num_long_trials = len(mask_on_trials_long)

    num_shuffles = 1000
    shuffled_TS = []

    if num_short_trials > 0 and num_long_trials > 0:

        for n in range(num_shuffles):
            # pool together all trial number so we can randomly draw from them later
            all_trial_nums = np.union1d(mask_on_trials_short, mask_on_trials_long)
            shuffled_short_trials = np.random.choice(all_trial_nums, num_short_trials, replace=False)
            shuffled_long_trials = np.random.choice(all_trial_nums, num_long_trials, replace=False)

            first_lick_shuffled_short = plot_first_licks(None, licks_ds, reward_ds, shuffled_short_trials, SHORT_COLOR, False)
            first_lick_shuffled_long = plot_first_licks(None, licks_ds, reward_ds, shuffled_long_trials, LONG_COLOR, False)

            shuffled_TS.append(np.nanmedian(first_lick_shuffled_long[:,1]) - np.nanmedian(first_lick_shuffled_short[:,1]))

    plot_ax.hist(shuffled_TS,bins=np.arange(-30,60,4), color=plot_color, normed=True)
    plot_ax.axvline(actual_TS,lw=2,c='r')
    z_score = (actual_TS - np.mean(shuffled_TS))/np.std(shuffled_TS)
    plot_ax.set_title(np.round(z_score,2))
    return np.mean(shuffled_TS), z_score

def fig_behavior_stage5(h5path, sess, fname, fformat='png', subfolder=[], trialnrrange=None):
    # load data
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(12,18))
    # fig.suptitle(fname)
    ax1 = plt.subplot2grid((126,12),(0,0), rowspan=20, colspan=6)
    ax2 = plt.subplot2grid((126,12),(20,0), rowspan=18, colspan=6)
    ax3 = plt.subplot2grid((126,12),(38,0), rowspan=18, colspan=6)

    ax4 = plt.subplot2grid((126,12),(0,6), rowspan=20, colspan=6)
    ax5 = plt.subplot2grid((126,12),(20,6), rowspan=18, colspan=6)
    ax6 = plt.subplot2grid((126,12),(38,6), rowspan=18, colspan=6)

    ax7 = plt.subplot2grid((126,12),(56,0), rowspan=14, colspan=6)
    ax8 = plt.subplot2grid((126,12),(56,6), rowspan=14, colspan=6)

    ax9 = plt.subplot2grid((126,12),(70,0), rowspan=14, colspan=6)
    ax10 = plt.subplot2grid((126,12),(70,6), rowspan=14, colspan=6)

    ax11 = plt.subplot2grid((126,12),(84,0), rowspan=14, colspan=3)
    ax12 = plt.subplot2grid((126,12),(84,3), rowspan=14, colspan=3)
    ax13 = plt.subplot2grid((126,12),(84,6), rowspan=14, colspan=3)
    ax14 = plt.subplot2grid((126,12),(84,9), rowspan=14, colspan=3)

    ax15 = plt.subplot2grid((126,12),(98,0), rowspan=14, colspan=3)
    ax16 = plt.subplot2grid((126,12),(98,3), rowspan=14, colspan=3)
    ax17 = plt.subplot2grid((126,12),(98,6), rowspan=14, colspan=3)
    ax18 = plt.subplot2grid((126,12),(98,9), rowspan=14, colspan=3)

    ax19 = plt.subplot2grid((126,12),(112,0), rowspan=14, colspan=4)
    ax20 = plt.subplot2grid((126,12),(112,4), rowspan=14, colspan=4)
    ax21 = plt.subplot2grid((126,12),(112,8), rowspan=14, colspan=4)

    fig.suptitle(subfolder + sess)

    ax1.set_title('Short trials - OFF')
    ax1.set_ylabel('Trial #')
    ax2.set_title('Short trials - MASK ON STIM OFF')
    ax2.set_ylabel('Trial #')
    ax3.set_title('Short trials - MASK ON STIM ON')
    ax3.set_ylabel('Trial #')
    ax3.set_xlabel('Location (cm)')

    ax1.set_xlim([50,340])
    ax2.set_xlim([50,340])
    ax3.set_xlim([50,340])
    ax4.set_xlim([50,400])
    ax5.set_xlim([50,400])
    ax6.set_xlim([50,400])

    ax4.set_title('Short trials - OFF')
    ax5.set_title('Short trials - MASK ON STIM OFF')
    ax6.set_title('Short trials - MASK ON STIM ON')
    ax6.set_xlabel('Location (cm)')

    ax7.set_title('First licks SHORT')
    ax8.set_title('First licks LONG')

    ax9.set_title('Running speed SHORT')
    ax10.set_title('Running speed LONG')

    # divide trials up into mask on and mask off trials.
    short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
    if trialnrrange is not None:
        short_trials = filter_trials( raw_ds, [], ['trialnr_range',trialnrrange[0],trialnrrange[1]],short_trials)
    # short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],short_trials)
    # short_trials = filter_trials( raw_ds, [], ['maxrewardtime',15],short_trials)
    mask_off_trials_short = filter_trials( raw_ds, [], ['opto_mask_light_off'],short_trials)
    mask_on_trials_short = filter_trials( raw_ds, [], ['opto_mask_on_stim_off'],short_trials)
    stim_on_trials_short = filter_trials( raw_ds, [], ['opto_stim_on'],short_trials)

    long_trials = filter_trials( raw_ds, [], ['tracknumber',4])
    if trialnrrange is not None:
        long_trials = filter_trials( raw_ds, [], ['trialnr_range',trialnrrange[0],trialnrrange[1]],long_trials)
    # long_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],long_trials)
    # long_trials = filter_trials( raw_ds, [], ['maxrewardtime',15],long_trials)
    mask_off_trials_long = filter_trials( raw_ds, [], ['opto_mask_light_off'],long_trials)
    mask_on_trials_long = filter_trials( raw_ds, [], ['opto_mask_on_stim_off'],long_trials)
    stim_on_trials_long = filter_trials( raw_ds, [], ['opto_stim_on'],long_trials)

    # ipdb.set_trace()

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color=SHORT_COLOR,alpha=0.3,zorder=9)
    ax2.axvspan(200,240,color='0.9',zorder=0)
    ax2.axvspan(320,340,color=SHORT_COLOR,alpha=0.3,zorder=9)
    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(320,340,color=SHORT_COLOR,alpha=0.3,zorder=9)

    ax4.axvspan(200,240,color='0.9',zorder=0)
    ax4.axvspan(380,400,color=LONG_COLOR,alpha=0.3,zorder=9)
    ax5.axvspan(200,240,color='0.9',zorder=0)
    ax5.axvspan(380,400,color=LONG_COLOR,alpha=0.3,zorder=9)
    ax6.axvspan(200,240,color='0.9',zorder=0)
    ax6.axvspan(380,400,color=LONG_COLOR,alpha=0.3,zorder=9)

    all_licks_mask_off_short,_ = plot_licks(ax1, licks_ds, reward_ds, raw_ds, mask_off_trials_short, '0.5')
    all_licks_mask_on_short, sr_mask_on_short = plot_licks(ax2, licks_ds, reward_ds, raw_ds, mask_on_trials_short, 'k')
    all_licks_stim_on_short, sr_stim_on_short = plot_licks(ax3, licks_ds, reward_ds, raw_ds, stim_on_trials_short, '#128FCF')
    all_licks_mask_off_long, _ = plot_licks(ax4, licks_ds, reward_ds, raw_ds, mask_off_trials_long, '0.5')
    all_licks_mask_on_long, sr_mask_on_long = plot_licks(ax5, licks_ds, reward_ds, raw_ds, mask_on_trials_long, 'k')
    all_licks_stim_on_long, sr_stim_on_long = plot_licks(ax6, licks_ds, reward_ds, raw_ds, stim_on_trials_long, '#128FCF')

    first_lick_mask_off_short = plot_first_licks(ax7, licks_ds, reward_ds, mask_off_trials_short, '0.5')
    first_lick_mask_on_short = plot_first_licks(ax7, licks_ds, reward_ds, mask_on_trials_short, 'k')
    first_lick_stim_on_short = plot_first_licks(ax7, licks_ds, reward_ds, stim_on_trials_short, '#128FCF')
    first_lick_mask_off_long = plot_first_licks(ax8, licks_ds, reward_ds, mask_off_trials_long, '0.5')
    first_lick_mask_on_long = plot_first_licks(ax8, licks_ds, reward_ds, mask_on_trials_long, 'k')
    first_lick_stim_on_long = plot_first_licks(ax8, licks_ds, reward_ds, stim_on_trials_long, '#128FCF')

    first_lick_mask_off_short = plot_first_licks(ax19, licks_ds, reward_ds, mask_off_trials_short, '0.5')
    first_lick_mask_off_long = plot_first_licks(ax19, licks_ds, reward_ds, mask_off_trials_long, 'g')

    first_lick_mask_on_short = plot_first_licks(ax20, licks_ds, reward_ds, mask_on_trials_short, 'k')
    first_lick_mask_on_long = plot_first_licks(ax20, licks_ds, reward_ds, mask_on_trials_long, '#009EFF')

    first_lick_stim_on_short = plot_first_licks(ax21, licks_ds, reward_ds, stim_on_trials_short, 'k')
    first_lick_stim_on_long = plot_first_licks(ax21, licks_ds, reward_ds, stim_on_trials_long, '#128FCF')

    stim_off_TS = np.median(first_lick_mask_on_long[:,1]) - np.median(first_lick_mask_on_short[:,1])
    stim_on_TS = np.median(first_lick_stim_on_long[:,1]) - np.median(first_lick_stim_on_short[:,1])
    ax19.set_title('Task score: ' + str(np.round(np.median(first_lick_mask_off_long[:,1]) - np.median(first_lick_mask_off_short[:,1]),1)))
    ax20.set_title('Task score: ' + str(np.round(stim_off_TS,1)))
    ax21.set_title('Task score: ' + str(np.round(stim_on_TS,1)))

    shuffled_TS_mask_on_stim_off, z_score_stim_off = shuffled_task_score(mask_on_trials_short, mask_on_trials_long, licks_ds, reward_ds, stim_off_TS, 'k', ax15)
    shuffled_TS_mask_on_stim_on, z_score_stim_on = shuffled_task_score(stim_on_trials_short, stim_on_trials_long, licks_ds, reward_ds, stim_on_TS, '#128FCF', ax16)
    print('Shuffled task score STIM OFF: ' + str(shuffled_TS_mask_on_stim_off))
    print('Shuffled task score STIM ON : ' + str(shuffled_TS_mask_on_stim_on))

    binsize = 2
    binnr_short = 340/binsize
    # plot_running_speed(ax9, raw_ds, mask_off_trials_short, binsize, binnr_short, '0.5')
    mask_trials_speed_short = plot_running_speed(ax9, raw_ds, mask_on_trials_short, binsize, binnr_short, 'k')
    stim_trials_speed_short = plot_running_speed(ax9, raw_ds, stim_on_trials_short, binsize, binnr_short, '#128FCF')

    binnr_long = 400/binsize
    # print(mask_on_trials_long,stim_on_trials_long)
    # plot_running_speed(ax10, raw_ds, mask_off_trials_long, binsize, binnr_long, '0.5')
    mask_trials_speed_long = plot_running_speed(ax10, raw_ds, mask_on_trials_long, binsize, binnr_long, 'k')
    stim_trials_speed_long = plot_running_speed(ax10, raw_ds, stim_on_trials_long, binsize, binnr_long, '#128FCF')
    # print(mask_trials_speed_long, stim_trials_speed_long)


    ax9.axvspan(200/binsize,240/binsize,color='0.9',zorder=0)
    ax9.axvspan(320/binsize,340/binsize,color=SHORT_COLOR,alpha=0.3,zorder=9)
    ax10.axvspan(200/binsize,240/binsize,color='0.9',zorder=0)
    ax10.axvspan(380/binsize,400/binsize,color=LONG_COLOR,alpha=0.3,zorder=9)

    # bootstrap differences between pairs of first lick locations
    if np.size(first_lick_mask_off_short) > 0 and np.size(first_lick_mask_on_short) > 0 and np.size(first_lick_stim_on_short) > 0:
        num_shuffles = 10000
        mask_off_bootstrap_short = np.random.choice(first_lick_mask_off_short[:,1], num_shuffles)
        mask_on_bootstrap_short = np.random.choice(first_lick_mask_on_short[:,1], num_shuffles)
        stim_on_bootstrap_short = np.random.choice(first_lick_stim_on_short[:,1], num_shuffles)

        bootstrap_diff_mask_on_mask_off_short = mask_on_bootstrap_short - mask_off_bootstrap_short
        bootstrap_diff_mask_on_stim_on_short = mask_on_bootstrap_short - stim_on_bootstrap_short
        bootstrap_diff_mask_off_stim_on_short = mask_off_bootstrap_short - stim_on_bootstrap_short

        fl_diff_mask_on_mask_off_short = np.mean(bootstrap_diff_mask_on_mask_off_short)/np.std(bootstrap_diff_mask_on_mask_off_short)
        fl_diff_mask_on_stim_on_short = np.mean(bootstrap_diff_mask_on_stim_on_short)/np.std(bootstrap_diff_mask_on_stim_on_short)
        fl_diff_mask_off_stim_on_short = np.mean(bootstrap_diff_mask_off_stim_on_short)/np.std(bootstrap_diff_mask_off_stim_on_short)

        sns.distplot(bootstrap_diff_mask_on_mask_off_short,ax=ax11,color='0.5')
        vl_handle = ax5.axvline(np.mean(bootstrap_diff_mask_on_mask_off_short),c='0.5')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_mask_off_short))
        # ax11.legend()
        ax11.set_title('mask on vs mask off SHORT')
        ax11.set_yticklabels('')
        ax11.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_on_stim_on_short,ax=ax12,color='#128FCF')
        vl_handle = ax12.axvline(np.mean(bootstrap_diff_mask_on_stim_on_short),c='#128FCF')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_stim_on_short))
        # ax12.legend()
        ax12.set_title('mask on vs stim on SHORT')
        ax12.set_yticklabels('')
        ax12.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_off_stim_on_short,ax=ax13,color='k')
        vl_handle = ax13.axvline(np.mean(bootstrap_diff_mask_off_stim_on_short),c='k')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_off_stim_on_short))
        # ax13.legend()
        ax13.set_title('mask off vs stim on SHORT')
        ax13.set_yticklabels('')
        ax13.set_xlim([-100,100])

    # bootstrap differences between pairs of first lick locations
    if np.size(first_lick_mask_off_long) > 0 and np.size(first_lick_mask_on_long) > 0 and np.size(first_lick_stim_on_long) > 0:
        num_shuffles = 10000
        mask_off_bootstrap_long = np.random.choice(first_lick_mask_off_long[:,1], num_shuffles)
        mask_on_bootstrap_long = np.random.choice(first_lick_mask_on_long[:,1], num_shuffles)
        stim_on_bootstrap_long = np.random.choice(first_lick_stim_on_long[:,1], num_shuffles)

        bootstrap_diff_mask_on_mask_off_long = mask_on_bootstrap_long - mask_off_bootstrap_long
        bootstrap_diff_mask_on_stim_on_long = mask_on_bootstrap_long - stim_on_bootstrap_long
        bootstrap_diff_mask_off_stim_on_long = mask_off_bootstrap_long - stim_on_bootstrap_long

        fl_diff_mask_on_mask_off_long = np.mean(bootstrap_diff_mask_on_mask_off_long)/np.std(bootstrap_diff_mask_on_mask_off_long)
        fl_diff_mask_on_stim_on_long = np.mean(bootstrap_diff_mask_on_stim_on_long)/np.std(bootstrap_diff_mask_on_stim_on_long)
        fl_diff_mask_off_stim_on_long = np.mean(bootstrap_diff_mask_off_stim_on_long)/np.std(bootstrap_diff_mask_off_stim_on_long)

        sns.distplot(bootstrap_diff_mask_on_mask_off_long,ax=ax15,color='0.5')
        vl_handle = ax5.axvline(np.mean(bootstrap_diff_mask_on_mask_off_long),c='0.5')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_mask_off_long))
        # ax15.legend()
        ax15.set_title('mask on vs mask off long')
        ax15.set_yticklabels('')
        ax15.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_on_stim_on_long,ax=ax16,color='#128FCF')
        vl_handle = ax16.axvline(np.mean(bootstrap_diff_mask_on_stim_on_long),c='#128FCF')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_stim_on_long))
        # ax16.legend()
        ax16.set_title('mask on vs stim on long')
        ax16.set_yticklabels('')
        ax16.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_off_stim_on_long,ax=ax17,color='k')
        vl_handle = ax17.axvline(np.mean(bootstrap_diff_mask_off_stim_on_long),c='k')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_off_stim_on_long))
        # ax17.legend()
        ax17.set_title('mask off vs stim on long')
        ax17.set_yticklabels('')
        ax17.set_xlim([-100,100])

    ax14.bar([1],[np.var(first_lick_mask_off_short[:,1])],color='0.5')
    ax14.bar([2],[np.var(first_lick_mask_on_short[:,1])],color='k')
    ax14.bar([3],[np.var(first_lick_stim_on_short[:,1])],color='#128FCF')
    ax14.set_xlim([0.5,4.5])
    ax14.set_title('first lick variance')
    ax14.set_xticks([1.4,2.4,3.4])
    ax14.set_xticklabels(['off','mask','stim'])
    ax14.spines['right'].set_visible(False)
    ax14.spines['top'].set_visible(False)

    ax15.spines['bottom'].set_linewidth(2)
    ax15.spines['top'].set_visible(False)
    ax15.spines['right'].set_visible(False)
    ax15.spines['left'].set_linewidth(2)
    ax15.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax16.spines['bottom'].set_linewidth(2)
    ax16.spines['top'].set_visible(False)
    ax16.spines['right'].set_visible(False)
    ax16.spines['left'].set_linewidth(2)
    ax16.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax18.bar([1],[np.var(first_lick_mask_off_long[:,1])],color='0.5')
    ax18.bar([2],[np.var(first_lick_mask_on_long[:,1])],color='k')
    ax18.bar([3],[np.var(first_lick_stim_on_long[:,1])],color='#128FCF')
    ax18.set_xlim([0.5,4.5])
    ax18.set_title('first lick variance')
    ax18.set_xticks([1.4,2.4,3.4])
    ax18.set_xticklabels(['off','mask','stim'])
    ax18.spines['right'].set_visible(False)
    ax18.spines['top'].set_visible(False)

    if MAKE_IND_FIGURES:
        fig.tight_layout()
        if subfolder != []:
            if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
                os.mkdir(loc_info['figure_output_path'] + subfolder)
            fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
        else:
            fname = loc_info['figure_output_path'] + fname + '.' + fformat
        try:
            fig.savefig(fname, format=fformat, dpi=150)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info().print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print(fname)
    else:
        print(fname)

    return np.median(first_lick_mask_off_long[:,1]) - np.median(first_lick_mask_off_short[:,1]), \
           np.median(first_lick_mask_on_long[:,1]) - np.median(first_lick_mask_on_short[:,1]), \
           np.median(first_lick_stim_on_long[:,1]) - np.median(first_lick_stim_on_short[:,1]), \
           [mask_trials_speed_short,stim_trials_speed_short,mask_trials_speed_long,stim_trials_speed_long], \
           [all_licks_mask_on_short,all_licks_stim_on_short,all_licks_mask_on_long,all_licks_stim_on_long], \
           [shuffled_TS_mask_on_stim_off, shuffled_TS_mask_on_stim_on], \
           [sr_mask_on_short, sr_stim_on_short, sr_mask_on_long, sr_stim_on_long]

def make_summary_figure():
    fl_diff_median_mask_off = []
    fl_diff_median_mask_on = []
    fl_diff_median_stim_on = []

    fl_diff_median_mask_off_exp = []
    fl_diff_median_mask_on_exp = []
    fl_diff_median_stim_on_exp = []
    shuffled_fl_diff_median_mask_on_exp = []
    shuffled_fl_diff_median_stim_on_exp = []

    speeds_all = []
    licks_all_opto = []

    all_success_rates = []

    naive_trialnr_range = [0,80]

    MOUSE = 'LF180728_1'
    s = 'Day2018923' #,'Day2018928'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on,_,_,_,_ = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto',[40,80])
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    # print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018928'
    s = 'Day2018105'
    diff_mask_off,diff_mask_on,diff_stim_on,speeds,licks,ts_shuffled, success_rates = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto')
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    shuffled_fl_diff_median_mask_on_exp.append(ts_shuffled[0])
    shuffled_fl_diff_median_stim_on_exp.append(ts_shuffled[1])
    # print(MOUSE + ' ' + s + ' done.')
    speeds_all.append(speeds)
    licks_all_opto.append(licks)
    all_success_rates.append(success_rates)

    MOUSE = 'LF180514_1'
    s = 'Day2018813'#,'Day2018924']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on,_,_,_,_ = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto',naive_trialnr_range)
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    # print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018924'
    s = 'Day2018105'
    diff_mask_off,diff_mask_on,diff_stim_on,speeds,licks,ts_shuffled, success_rates = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto')
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    shuffled_fl_diff_median_mask_on_exp.append(ts_shuffled[0])
    shuffled_fl_diff_median_stim_on_exp.append(ts_shuffled[1])
    # print(MOUSE + ' ' + s + ' done.')
    speeds_all.append(speeds)
    licks_all_opto.append(licks)
    all_success_rates.append(success_rates)

    # # # #
    MOUSE = 'LF180515_1'
    s = 'Day2018910'#,'Day2018924']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on,_,_,_,_ = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto',naive_trialnr_range)
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    # print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018924'
    s = 'Day2018105'
    diff_mask_off,diff_mask_on,diff_stim_on,speeds,licks,ts_shuffled, success_rates = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto')
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    shuffled_fl_diff_median_mask_on_exp.append(ts_shuffled[0])
    shuffled_fl_diff_median_stim_on_exp.append(ts_shuffled[1])
    # print(MOUSE + ' ' + s + ' done.')
    speeds_all.append(speeds)
    licks_all_opto.append(licks)
    all_success_rates.append(success_rates)

    MOUSE = 'LF180920_1'
    s = 'Day20181025'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on,_,_,_,_ = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto',[0,60])
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    # print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018924'
    s = 'Day2018126'
    diff_mask_off,diff_mask_on,diff_stim_on,speeds,licks,ts_shuffled, success_rates = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto')
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    shuffled_fl_diff_median_mask_on_exp.append(ts_shuffled[0])
    shuffled_fl_diff_median_stim_on_exp.append(ts_shuffled[1])
    # print(MOUSE + ' ' + s + ' done.')
    speeds_all.append(speeds)
    licks_all_opto.append(licks)
    all_success_rates.append(success_rates)

    MOUSE = 'LF180919_1'
    s = 'Day20181025'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on,_,_,_,_ = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto',[20,60])
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    # print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018924'
    s = 'Day2018122'
    # ipdb.set_trace()
    diff_mask_off,diff_mask_on,diff_stim_on,speeds,licks,ts_shuffled, success_rates = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, 'Opto')
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    shuffled_fl_diff_median_mask_on_exp.append(ts_shuffled[0])
    shuffled_fl_diff_median_stim_on_exp.append(ts_shuffled[1])
    # print(MOUSE + ' ' + s + ' done.')
    speeds_all.append(speeds)
    licks_all_opto.append(licks)
    all_success_rates.append(success_rates)

    all_sr_mask_short = []
    all_sr_stim_short = []
    all_sr_mask_long = []
    all_sr_stim_long = []
    for alr in all_success_rates:
        all_sr_mask_short.append(alr[0])
        all_sr_stim_short.append(alr[1])
        all_sr_mask_long.append(alr[2])
        all_sr_stim_long.append(alr[3])

    print('--- FRACTION SUCCESSFUL MASK VS STIM ----')
    print(sp.stats.ttest_rel(np.concatenate((all_sr_mask_short,all_sr_mask_long)),np.concatenate((all_sr_stim_short,all_sr_stim_long))))
    print('mean SR mask (short/long combined): ' + str(np.mean(np.concatenate((all_sr_mask_short,all_sr_mask_long)))) + ' SEM: ' + str(sp.stats.sem(np.concatenate((all_sr_mask_short,all_sr_mask_long)))))
    print('mean SR stim (short/long combined): ' + str(np.mean(np.concatenate((all_sr_stim_short,all_sr_stim_long)))) + ' SEM: ' + str(sp.stats.sem(np.concatenate((all_sr_stim_short,all_sr_stim_long)))))
    print('-----------------------------------------')


    # carry out statistical analysis. This is not (yet) the correct test: we are treating each group independently, rather than taking into account within-group and between-group variance
    # print(np.array(fl_diff_median_mask_off_exp),np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp))
    # print(sp.stats.f_oneway(np.array(fl_diff_median_mask_off_exp),np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp)))
    # group_labels = ['fl_diff_median_mask_off_exp'] * np.array(fl_diff_median_mask_off_exp).shape[0] + \
    #                ['fl_diff_median_mask_on_exp'] * np.array(fl_diff_median_mask_on_exp).shape[0] + \
    #                ['fl_diff_median_stim_on_exp'] * np.array(fl_diff_median_stim_on_exp).shape[0]
    #
    # mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(fl_diff_median_mask_off_exp),np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp))),group_labels)
    # posthoc_res_ss = mc_res_ss.tukeyhsd()
    # print(posthoc_res_ss)
    print(np.mean(fl_diff_median_mask_on_exp),sp.stats.sem(fl_diff_median_mask_on_exp),np.mean(fl_diff_median_stim_on_exp),sp.stats.sem(fl_diff_median_stim_on_exp))
    t,p = sp.stats.ttest_rel(np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp))
    print(t,p)

    # create figure to later plot on
    fig = plt.figure(figsize=(14,14))
    ax1 = plt.subplot2grid((3,6),(0,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((3,6),(0,1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((3,6),(0,2), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((3,6),(1,0), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((3,6),(1,2), rowspan=1, colspan=4)
    ax6 = plt.subplot2grid((3,6),(2,0), rowspan=1, colspan=1)
    ax7 = plt.subplot2grid((3,6),(2,2), rowspan=1, colspan=2)
    ax8 = plt.subplot2grid((3,6),(2,4), rowspan=1, colspan=2)
    ax9 = plt.subplot2grid((3,6),(0,3), rowspan=1, colspan=1)
    # ipdb.set_trace()
    parts = ax9.boxplot([shuffled_fl_diff_median_mask_on_exp,shuffled_fl_diff_median_stim_on_exp, fl_diff_median_mask_on_exp,fl_diff_median_stim_on_exp],
        patch_artist=True,showfliers=False,
        whiskerprops=dict(color='w', linestyle='-', linewidth=0, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        capprops=dict(color='w', alpha=0.0),
        widths=(1,1,1,1),positions=(0,1.5,3,4.5))

    colors = ['0.5','0.5','k','#128FCF']
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set_edgecolor(color[1])
        patch.set_alpha(0.3)
        patch.set_linewidth(0)

    colors = ['k','k','k','k']
    for patch, color in zip(parts['medians'], colors):
        # patch.set_facecolor(color)
        patch.set_color(color)
        patch.set_alpha(1.0)
        patch.set_linewidth(2)

    # ipdb.set_trace()
    ax9.scatter(np.full(len(shuffled_fl_diff_median_mask_on_exp),0), shuffled_fl_diff_median_mask_on_exp, s=60, linewidths=2, edgecolor='k', c='w', zorder=5)
    ax9.scatter(np.full(len(shuffled_fl_diff_median_stim_on_exp),1.5), shuffled_fl_diff_median_stim_on_exp, s=60, linewidths=2, edgecolor='#128FCF', c='w', zorder=5)
    ax9.scatter(np.full(len(fl_diff_median_mask_on_exp),3), fl_diff_median_mask_on_exp, s=60, linewidths=2, edgecolor='k', c='k', zorder=5)
    ax9.scatter(np.full(len(fl_diff_median_stim_on_exp),4.5), fl_diff_median_stim_on_exp, s=60, linewidths=2, edgecolor='#128FCF', c='#128FCF', zorder=5)

    print('--- SHUFFLED VS. REAL ANOVA FOR OPTO TASK SCORE ---')
    print(sp.stats.f_oneway(np.array(shuffled_fl_diff_median_mask_on_exp),np.array(shuffled_fl_diff_median_stim_on_exp),np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp)))
    group_labels = ['shuffled_fl_diff_median_mask_on_exp'] * np.array(shuffled_fl_diff_median_mask_on_exp).shape[0] + \
                   ['shuffled_fl_diff_median_stim_on_exp'] * np.array(shuffled_fl_diff_median_stim_on_exp).shape[0] + \
                   ['fl_diff_median_mask_on_exp'] * np.array(fl_diff_median_mask_on_exp).shape[0] + \
                   ['fl_diff_median_stim_on_exp'] * np.array(fl_diff_median_stim_on_exp).shape[0]

    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(shuffled_fl_diff_median_mask_on_exp),np.array(shuffled_fl_diff_median_stim_on_exp),np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp))),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)
    print('----------------------------------------')

    ax9.set_ylim([-3,60])
    ax9.set_xlim([-1,5.5])
    ax9.set_xticks([0.75, 3.75])
    ax9.set_xticklabels(['Shuffled', 'Actual'], rotation=45)

    ax9.spines['bottom'].set_linewidth(2)
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.spines['left'].set_linewidth(2)
    ax9.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')


    mask_all = []
    stim_all = []
    speed_histo_bins = np.arange(0,90,1)
    mask_speeds_short = np.zeros((len(speeds_all),len(speed_histo_bins)-1))
    stim_speeds_short = np.zeros((len(speeds_all),len(speed_histo_bins)-1))
    mask_speeds_long = np.zeros((len(speeds_all),len(speed_histo_bins)-1))
    stim_speeds_long = np.zeros((len(speeds_all),len(speed_histo_bins)-1))

    for i,sa in enumerate(speeds_all):
        ax4.scatter([0],np.mean(np.concatenate((sa[0],sa[2]))),s=120,linewidths=3,facecolor='w',edgecolor='#128FCF', zorder=3)
        ax4.scatter([1],np.mean(np.concatenate((sa[1],sa[3]))),s=120, linewidths=3,facecolor='#128FCF',edgecolor='#128FCF', zorder=3)
        ax4.plot([0,1], [np.mean(np.concatenate((sa[0],sa[2]))), np.mean(np.concatenate((sa[1],sa[3])))], c='0.5', lw=4,zorder=2)
        mask_all.append(np.mean(np.concatenate((sa[0],sa[2]))))
        stim_all.append(np.mean(np.concatenate((sa[1],sa[3]))))

        sns.distplot(sa[0],hist=False,kde_kws={"color": "k","ls":"--","alpha":1.0, "lw":3}, ax=ax5)
        sns.distplot(sa[1],hist=False,kde_kws={"color": "#128FCF","alpha":1.0, "lw":3}, ax=ax5)
        sns.distplot(sa[2],hist=False,kde_kws={"color": "k","ls":"--","alpha":1.0, "lw":3}, ax=ax5)
        sns.distplot(sa[3],hist=False,kde_kws={"color": "#128FCF","alpha":1.0, "lw":3}, ax=ax5)

        ms,_ = np.histogram(sa[0], np.arange(0,90,1))
        ss,_ = np.histogram(sa[1], np.arange(0,90,1))
        ml,_ = np.histogram(sa[2], np.arange(0,90,1))
        sl,_ = np.histogram(sa[3], np.arange(0,90,1))

        # ax7.plot(np.cumsum(ms)/np.nanmax(np.cumsum(ms)), c='0.8')

        mask_speeds_short[i,:] = np.cumsum(ms)/np.nanmax(np.cumsum(ms))
        stim_speeds_short[i,:] = np.cumsum(ss)/np.nanmax(np.cumsum(ss))
        mask_speeds_long[i,:] = np.cumsum(ml)/np.nanmax(np.cumsum(ml))
        stim_speeds_long[i,:] = np.cumsum(sl)/np.nanmax(np.cumsum(sl))

        # sns.distplot(sa[1],hist=False,kde_kws={"color": "#128FCF","alpha":1.0, "lw":3,"cumulative": "True"}, ax=ax7)
        # sns.distplot(sa[2],hist=False,kde_kws={"color": "k","alpha":1.0, "lw":3,"cumulative": "True"}, ax=ax7)
        # sns.distplot(sa[3],hist=False,kde_kws={"color": "#128FCF","alpha":1.0, "lw":3,"cumulative": "True"}, ax=ax7)


    cumsum_speed_mask_short = np.nanmean(mask_speeds_short,0)
    sem_speed_mask_short = stats.sem(mask_speeds_short,0,nan_policy='omit')
    cumsum_speed_stim_short = np.nanmean(stim_speeds_short,0)
    sem_speed_stim_short = stats.sem(stim_speeds_short,0,nan_policy='omit')
    ax7.plot(np.arange(0,89,1),cumsum_speed_stim_short, lw=4, c='#128FCF', solid_capstyle='butt')
    ax7.fill_between(np.arange(0,89,1),cumsum_speed_stim_short-sem_speed_stim_short, cumsum_speed_stim_short+sem_speed_stim_short, linewidth=0, color='#128FCF',alpha=0.2)
    ax7.plot(np.arange(0,89,1),cumsum_speed_mask_short, lw=4, ls='--', c='k', solid_capstyle='butt')
    ax7.fill_between(np.arange(0,89,1),cumsum_speed_mask_short-sem_speed_mask_short, cumsum_speed_mask_short+sem_speed_mask_short, linewidth=0, color='k',alpha=0.2)

    cumsum_speed_mask_long = np.nanmean(mask_speeds_long,0)
    sem_speed_mask_long = stats.sem(mask_speeds_long,0,nan_policy='omit')
    cumsum_speed_stim_long = np.nanmean(stim_speeds_long,0)
    sem_speed_stim_long = stats.sem(stim_speeds_long,0,nan_policy='omit')
    ax8.plot(np.arange(0,89,1),cumsum_speed_stim_long, lw=4, c='#128FCF', solid_capstyle='butt')
    ax8.fill_between(np.arange(0,89,1),cumsum_speed_stim_long-sem_speed_stim_long, cumsum_speed_stim_long+sem_speed_stim_long, linewidth=0, color='#128FCF',alpha=0.2)
    ax8.plot(np.arange(0,89,1),cumsum_speed_mask_long, lw=4, ls='--', c='k', solid_capstyle='butt')
    ax8.fill_between(np.arange(0,89,1),cumsum_speed_mask_long-sem_speed_mask_long, cumsum_speed_mask_long+sem_speed_mask_long, linewidth=0, color='k',alpha=0.2)

    print('--- K-S TEST FOR CUMULATIVE SPEED DISTRIBUTION ---')
    print(sp.stats.ks_2samp(cumsum_speed_mask_long, cumsum_speed_stim_long))
    print(sp.stats.ks_2samp(cumsum_speed_mask_short, cumsum_speed_stim_short))
    print('--------------------------------------------------')

    print('--- RUNNING SPEED WILCOXON ---')
    print(stats.wilcoxon(mask_all,stim_all))

    print('mean mask: ', str(np.mean(mask_all)))
    print('mean mask: ', str(np.mean(stim_all)))

    ax7.set_ylim([0,1.05])
    ax8.set_ylim([0,1.05])

    ax4.set_ylim([0,60])
    ax4.set_xlim([-0.5,1.5])
    ax4.set_xticks([0,1])
    ax4.set_xticklabels([])
    ax4.set_ylabel('speed (cm/sec)', fontsize=16)

    ax4.spines['bottom'].set_linewidth(2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_linewidth(2)
    ax4.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax7.spines['bottom'].set_linewidth(2)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.spines['left'].set_linewidth(2)
    ax7.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax8.spines['bottom'].set_linewidth(2)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['left'].set_linewidth(2)
    ax8.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax5.set_ylim([0,0.08])
    ax5.set_xlim([0,90])
    ax5.spines['bottom'].set_linewidth(2)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['left'].set_linewidth(2)
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


    mask_all_licks = []
    stim_all_licks = []
    for la in licks_all_opto:
        ax6.scatter([0],np.mean(np.concatenate((la[0],la[2]))),s=120,linewidths=3,facecolor='k',edgecolor='k', zorder=3)
        ax6.scatter([1],np.mean(np.concatenate((la[1],la[3]))),s=120, linewidths=3,facecolor='#128FCF',edgecolor='#128FCF', zorder=3)
        ax6.plot([0,1], [np.mean(np.concatenate((la[0],la[2]))), np.mean(np.concatenate((la[1],la[3])))], c='0.5', lw=4,zorder=2)
        mask_all_licks.append(np.mean(np.concatenate((la[0],la[2]))))
        stim_all_licks.append(np.mean(np.concatenate((la[1],la[3]))))


    print('--- LICKING WILCOXON ---')
    print(stats.wilcoxon(mask_all_licks,stim_all_licks))
    print('------------------------')

    ax6.set_ylim([0,10])
    ax6.set_xlim([-0.5,1.5])
    ax6.set_xticks([0,1])
    ax6.set_xticklabels([])
    ax6.set_ylabel('licks/trial', fontsize=16)

    ax6.spines['bottom'].set_linewidth(2)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_linewidth(2)
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

    marker_size = 140

    ax1.scatter([0,0,0,0,0],fl_diff_median_mask_off,s=marker_size, facecolor='0.5', edgecolor='0.5', linewidth=2,zorder=3)
    ax1.scatter([1,1,1,1,1],fl_diff_median_mask_on_exp,s=marker_size, facecolor='k', edgecolor='k', linewidth=2,zorder=3)
    ax1.scatter([2,2,2,2,2],fl_diff_median_stim_on_exp,s=marker_size, facecolor='#128FCF', edgecolor='#128FCF', linewidth=2,zorder=3)
    ax1.plot([0,1,2],[fl_diff_median_mask_off[0],fl_diff_median_mask_on_exp[0],fl_diff_median_stim_on_exp[0]],lw=2,c='0.5')
    ax1.plot([0,1,2],[fl_diff_median_mask_off[1],fl_diff_median_mask_on_exp[1],fl_diff_median_stim_on_exp[1]],lw=2,c='0.5')
    ax1.plot([0,1,2],[fl_diff_median_mask_off[2],fl_diff_median_mask_on_exp[2],fl_diff_median_stim_on_exp[2]],lw=2,c='0.5')
    ax1.plot([0,1,2],[fl_diff_median_mask_off[3],fl_diff_median_mask_on_exp[3],fl_diff_median_stim_on_exp[3]],lw=2,c='0.5')
    ax1.plot([0,1,2],[fl_diff_median_mask_off[4],fl_diff_median_mask_on_exp[4],fl_diff_median_stim_on_exp[4]],lw=2,c='0.5')

    ax2.scatter([1,1,1,1,1],fl_diff_median_mask_on_exp,s=marker_size, facecolor='k', edgecolor='k', linewidth=2,zorder=3)
    ax2.scatter([2,2,2,2,2],fl_diff_median_stim_on_exp,s=marker_size, facecolor='#128FCF', edgecolor='#128FCF', linewidth=2,zorder=3)
    ax2.plot([1,2],[fl_diff_median_mask_on_exp[0],fl_diff_median_stim_on_exp[0]],lw=2,c='0.5')
    ax2.plot([1,2],[fl_diff_median_mask_on_exp[1],fl_diff_median_stim_on_exp[1]],lw=2,c='0.5')
    ax2.plot([1,2],[fl_diff_median_mask_on_exp[2],fl_diff_median_stim_on_exp[2]],lw=2,c='0.5')
    ax2.plot([1,2],[fl_diff_median_mask_on_exp[3],fl_diff_median_stim_on_exp[3]],lw=2,c='0.5')
    ax2.plot([1,2],[fl_diff_median_mask_on_exp[4],fl_diff_median_stim_on_exp[4]],lw=2,c='0.5')

    for i,fld in enumerate(fl_diff_median_mask_on_exp):
        ax3.scatter(1,fl_diff_median_mask_on_exp[i]/fl_diff_median_mask_on_exp[i],s=marker_size, facecolor='k', edgecolor='k', linewidth=2,zorder=3)
        ax3.scatter(2,fl_diff_median_stim_on_exp[i]/fl_diff_median_mask_on_exp[i],s=marker_size, facecolor='#128FCF', edgecolor='#128FCF', linewidth=2,zorder=3)
        ax3.plot([1,2],[fl_diff_median_mask_on_exp[i]/fl_diff_median_mask_on_exp[i],fl_diff_median_stim_on_exp[i]/fl_diff_median_mask_on_exp[i]],c='0.5')
    # ax3.plot([1,2],[fl_diff_median_mask_on_exp[1],fl_diff_median_stim_on_exp[1]/fl_diff_median_mask_on_exp[1]],c='0.5')
    # ax3.plot([1,2],[fl_diff_median_mask_on_exp[2],fl_diff_median_stim_on_exp[2]/fl_diff_median_mask_on_exp[2]],c='0.5')
    # ax3.plot([1,2],[fl_diff_median_mask_on_exp[3],fl_diff_median_stim_on_exp[3]/fl_diff_median_mask_on_exp[3]],c='0.5')
    # ax3.plot([1,2],[fl_diff_median_mask_on_exp[4],fl_diff_median_stim_on_exp[4]/fl_diff_median_mask_on_exp[4]],c='0.5')

    if p < 0.005:
        ax1.text(1.34,59,'**',fontsize=24)
        # ax1.plot([1,2],[60,60],lw=3, c='0.5')

    ax1.set_xticks([0,1,2])
    ax1.set_yticks([0,10,20,30,40,50,60])
    ax1.set_xlim([-0.4,2.5])
    ax1.set_ylim([-10,60])

    # ax1.set_xticklabels(['naive','expert+mask','expert+mask+stim'], rotation=45, fontsize=16)

    ax1.set_ylabel('task score', fontsize=16)
    ax2.set_ylabel('task score', fontsize=16)
    ax3.set_ylabel('normalized task score', fontsize=16)

    ax1.spines['left'].set_linewidth(3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(3)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=24, \
        length=6, \
        width=4, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax2.spines['left'].set_linewidth(2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax3.spines['left'].set_linewidth(2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    subfolder = 'Opto'
    fname = 'Summary2'

    fig.tight_layout()
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info().print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    print(fname)

# def make_summary_plus_normal_animals_figure():
#     fl_diff_median_mask_off = []
#     fl_diff_median_mask_on = []
#     fl_diff_median_stim_on = []
#
#     fl_diff_median_mask_off_exp = []
#     fl_diff_median_mask_on_exp = []
#     fl_diff_median_stim_on_exp = []
#
#     MOUSE = 'LF180728_1'
#     s = 'Day2018923' #,'Day2018928'
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
#     fl_diff_median_mask_off.append(diff_mask_off)
#     fl_diff_median_mask_on.append(diff_mask_on)
#     fl_diff_median_stim_on.append(diff_stim_on)
#     print(MOUSE + ' ' + s + ' done.')
#
#     # s = 'Day2018928'
#     s = 'Day2018105'
#     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
#     fl_diff_median_mask_off_exp.append(diff_mask_off)
#     fl_diff_median_mask_on_exp.append(diff_mask_on)
#     fl_diff_median_stim_on_exp.append(diff_stim_on)
#     print(MOUSE + ' ' + s + ' done.')
#
#     # #
#     MOUSE = 'LF180514_1'
#     s = 'Day2018813'#,'Day2018924']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
#     fl_diff_median_mask_off.append(diff_mask_off)
#     fl_diff_median_mask_on.append(diff_mask_on)
#     fl_diff_median_stim_on.append(diff_stim_on)
#     print(MOUSE + ' ' + s + ' done.')
#
#     # s = 'Day2018924'
#     s = 'Day2018105'
#     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
#     fl_diff_median_mask_off_exp.append(diff_mask_off)
#     fl_diff_median_mask_on_exp.append(diff_mask_on)
#     fl_diff_median_stim_on_exp.append(diff_stim_on)
#     print(MOUSE + ' ' + s + ' done.')
#     # #
#     # # # # #
#     MOUSE = 'LF180515_1'
#     s = 'Day2018910'#,'Day2018924']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
#     fl_diff_median_mask_off.append(diff_mask_off)
#     fl_diff_median_mask_on.append(diff_mask_on)
#     fl_diff_median_stim_on.append(diff_stim_on)
#     print(MOUSE + ' ' + s + ' done.')
#
#     # s = 'Day2018924'
#     s = 'Day2018105'
#     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
#     fl_diff_median_mask_off_exp.append(diff_mask_off)
#     fl_diff_median_mask_on_exp.append(diff_mask_on)
#     fl_diff_median_stim_on_exp.append(diff_stim_on)
#     print(MOUSE + ' ' + s + ' done.')
#
#     ####
#     tscore_fig = plt.figure(figsize=(4,3))
#     tscore_graph = plt.subplot(111)
#
#     datasets = [['LF170222_1','Day20170405','Day20170615'],['LF170420_1','Day20170524','Day20170719'],
#                 ['LF170110_2','Day20170210','Day20170331'],['LF170214_1','Day20170509','Day201777'],
#                 ['LF171211_2','Day2018321','Day201852']]
#
#     fl_nov = []
#     fl_exp = []
#     for ds in datasets:
#         print(ds[0])
#         h5path = loc_info['imaging_dir'] + ds[0] + '/' + ds[0] + '.h5'
#         fl_nov.append(fig_behavior_stage5_normal(h5path, ds[1], ds[0]+ds[1], fformat, ds[0]))
#         fl_exp.append(fig_behavior_stage5_normal(h5path, ds[2], ds[0]+ds[2], fformat, ds[0]))
#
#
#
#     # novice_expert_ax.scatter([1,1,1,1,1,1],np.nanmean(tscore_all[:,3:6],1), s=150, c='0.5',zorder=2)
#     #
#     # fname = loc_info['figure_output_path'] + 'stage5_novice_expert_test' + '.' + fformat
#     # novice_expert_fig.savefig(fname, format=fformat)
#     # fname = loc_info['figure_output_path'] + 'stage5_expert_histo' + '.' + fformat
#     # tscore_fig.savefig(fname, format=fformat, dpi=300)
#     ####
#     t,p = sp.stats.ttest_rel(np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp))
#     print(t,p)
#
#     # create figure to later plot on
#     fig = plt.figure(figsize=(4,6))
#     ax1 = plt.subplot(111)
#     ax1.scatter([0 for i in fl_nov], fl_nov, s=150, facecolor='0.8', edgecolor='0.8',linewidth=2,zorder=2)
#     ax1.scatter([1 for i in fl_exp], fl_exp, s=150, facecolor='0.8', edgecolor='0.8',linewidth=2,zorder=2)
#     for i in range(len(fl_nov)):
#         ax1.plot([0,1],[fl_nov[i],fl_exp[i]],c='0.8',lw=2)
#
#     ax1.scatter([0,0,0],fl_diff_median_mask_off,s=80, facecolor='0.5', edgecolor='0.5', linewidth=2,zorder=3)
#     ax1.scatter([1,1,1],fl_diff_median_mask_on_exp,s=80, facecolor='w', edgecolor='k', linewidth=2,zorder=3)
#     ax1.scatter([2,2,2],fl_diff_median_stim_on_exp,s=80, facecolor='k', edgecolor='k', linewidth=2,zorder=3)
#     ax1.plot([0,1,2],[fl_diff_median_mask_off[0],fl_diff_median_mask_on_exp[0],fl_diff_median_stim_on_exp[0]],c='0.5',lw=2)
#     ax1.plot([0,1,2],[fl_diff_median_mask_off[1],fl_diff_median_mask_on_exp[1],fl_diff_median_stim_on_exp[1]],c='0.5',lw=2)
#     ax1.plot([0,1,2],[fl_diff_median_mask_off[2],fl_diff_median_mask_on_exp[2],fl_diff_median_stim_on_exp[2]],c='0.5',lw=2)
#
#     if p < 0.005:
#         ax1.text(1.34,55,'**',fontsize=24)
#         ax1.plot([1,2],[56,56],lw=3, c='0.5')
#
#     ax1.set_xticks([0,1,2])
#     ax1.set_xticklabels(['naive','expert + mask','expert + mask + opto'], rotation=45, fontsize=16)
#
#     ax1.set_ylabel('Task score', fontsize=16)
#
#     ax1.set_ylim([-10,60])
#
#     ax1.spines['left'].set_linewidth(2)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.tick_params( \
#         axis='both', \
#         direction='in', \
#         labelsize=16, \
#         length=4, \
#         width=2, \
#         bottom='on', \
#         right='off', \
#         top='off')
#
#     subfolder = 'Opto'
#     fname = 'Summary'
#
#     fig.tight_layout()
#     if subfolder != []:
#         if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
#             os.mkdir(loc_info['figure_output_path'] + subfolder)
#         fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
#     else:
#         fname = loc_info['figure_output_path'] + fname + '.' + fformat
#     try:
#         fig.savefig(fname, format=fformat)
#     except:
#         exc_type, exc_value, exc_traceback = sys.exc_info().print_exception(exc_type, exc_value, exc_traceback,
#                               limit=2, file=sys.stdout)

if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    fformat = 'png'

    # with open(loc_info['yaml_archive'], 'r') as f:
    #     project_metainfo = yaml.load(f)

    make_summary_figure()
    # make_summary_plus_normal_animals_figure()

    # MOUSE = 'LF180728_1'
    # SESSION = ['Day20181017']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    # #
    # # # #
    # MOUSE = 'LF180514_1'
    # SESSION = ['Day20181017','Day20181022']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    # #
    # # # # #
    # MOUSE = 'LF180515_1'
    # SESSION = ['Day20181017']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    # MOUSE = 'LF180919_1'
    # SESSION = ['Day20181025','Day20181026']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    # #
    # MOUSE = 'LF180920_1'
    # # SESSION = ['Day20181129','Day2018125','Day2018126','Day20181210']
    # SESSION = ['Day20181025','Day20181026','Day20181028']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    #
    # MOUSE = 'LF180905_1'
    # SESSION = ['Day20181130']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
