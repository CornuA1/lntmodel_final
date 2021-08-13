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
import statsmodels.api as sm
import scipy as sp
from scipy import stats
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

with open(loc_info['yaml_file'], 'r') as f:
    project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')

from filter_trials import filter_trials

def plot_licks(ax, licks, rewards, raw, trials, scatter_color):
    # scatterplot of licks/rewards in order of trial number
    trial_start = np.empty((0,3))
    trial_start_trials = np.empty((0))
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
        else:
            col = 'r'

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

def plot_first_licks(ax, licks, rewards, trials, scatter_color, min_loc=101):
    # plot location of first trials on short and long trials
    first_lick = np.empty((0,4))
    first_lick_trials = np.empty((0))
    for r in trials:
        licks_all = licks[licks[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>min_loc,:]
        # if not lick in this trial was found, check if only one lick at the reward location was detected
        if licks_all.size == 0:
             rew_lick = rewards[rewards[:,3]==r,:]
             if rew_lick.size > 0:
                 if rew_lick[0,5] == 1:
                     licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                     first_lick = np.vstack((first_lick, licks_all[0,:].T))
                     first_lick_trials = np.append(first_lick_trials, r)
        else:
            # if licks_all[0,3] == 3:
            #     licks_all = licks_all[licks_all[:,1]<338,:]
            # elif licks_all[0,3] == 4:
            #     licks_all = licks_all[licks_all[:,1]<398,:]
            first_lick = np.vstack((first_lick, licks_all[0,:].T))
            first_lick_trials = np.append(first_lick_trials, r)

    ax.scatter(first_lick[:,1],first_lick_trials,c=scatter_color,lw=0)
    if first_lick.size > 0:
        ax_2 = ax.twinx()
        sns.kdeplot(first_lick[:,1],c=scatter_color,ax=ax_2)
        ax_2.set_xlim([50,400])
        ax_2.set_yticklabels([''])

    ax.axvline(min_loc,lw=2,ls='--',c='0.8')
    ax.axvline(np.median(first_lick[:,1]), ls='--', c=scatter_color)
    ax.set_xlim([50,400])

    ax.set_yticklabels([''])

    return first_lick

def plot_running_speed(ax, raw, trials, binsize, binnr, plot_color):
    # plot running speed
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
            mean_speed_trial = stats.binned_statistic(cur_trial[:,1], cur_trial[:,3], 'mean', cur_trial_bins)[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_end_bin)] = mean_speed_trial
            # ax.plot(np.linspace(cur_trial_start_bin,cur_trial_end_bin,cur_trial_bins),mean_speed_trial,c='0.8',alpha=0.5,zorder=2)
        #     max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])
    #
    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax.plot(np.linspace(0,binnr-1,binnr),mean_speed_sess_short,c=plot_color,zorder=3)
    ax.fill_between(np.linspace(0,binnr-1,binnr),mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color=plot_color,alpha=0.2)
    ax.set_xlim([50/binsize,binnr])
    ax.set_ylim([0,np.nanmax(mean_speed_sess_short)])

def fig_behavior_stage5(h5path, sess, fname, fformat='png', subfolder=[]):
    # load data
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(12,18))
    # fig.suptitle(fname)
    ax1 = plt.subplot2grid((126,12),(0,0), rowspan=28, colspan=6)
    ax2 = plt.subplot2grid((126,12),(28,0), rowspan=14, colspan=6)
    ax3 = plt.subplot2grid((126,12),(42,0), rowspan=14, colspan=6)

    ax4 = plt.subplot2grid((126,12),(0,6), rowspan=28, colspan=6)
    ax5 = plt.subplot2grid((126,12),(28,6), rowspan=14, colspan=6)
    ax6 = plt.subplot2grid((126,12),(42,6), rowspan=14, colspan=6)

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

    ax1.set_xlim([50,460])
    ax2.set_xlim([50,460])
    ax3.set_xlim([50,460])
    ax4.set_xlim([50,460])
    ax5.set_xlim([50,460])
    ax6.set_xlim([50,460])

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
    # short_trials = filter_trials( raw_ds, [], ['trialnr_range',0,100],short_trials)
    short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],short_trials)
    # short_trials = filter_trials( raw_ds, [], ['maxrewardtime',15],short_trials)
    mask_off_trials_short = filter_trials( raw_ds, [], ['opto_mask_light_off'],short_trials)
    mask_on_trials_short = filter_trials( raw_ds, [], ['opto_mask_on_stim_off'],short_trials)
    stim_on_trials_short = filter_trials( raw_ds, [], ['opto_stim_on'],short_trials)

    long_trials = filter_trials( raw_ds, [], ['tracknumber',4])
    # long_trials = filter_trials( raw_ds, [], ['trialnr_range',0,100],long_trials)
    long_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],long_trials)
    # long_trials = filter_trials( raw_ds, [], ['maxrewardtime',15],long_trials)
    mask_off_trials_long = filter_trials( raw_ds, [], ['opto_mask_light_off'],long_trials)
    mask_on_trials_long = filter_trials( raw_ds, [], ['opto_mask_on_stim_off'],long_trials)
    stim_on_trials_long = filter_trials( raw_ds, [], ['opto_stim_on'],long_trials)

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)
    ax2.axvspan(200,240,color='0.9',zorder=0)
    ax2.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)
    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    ax4.axvspan(200,240,color='0.9',zorder=0)
    ax4.axvspan(380,400,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=9)
    ax5.axvspan(200,240,color='0.9',zorder=0)
    ax5.axvspan(380,400,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=9)
    ax6.axvspan(200,240,color='0.9',zorder=0)
    ax6.axvspan(380,400,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=9)

    plot_licks(ax1, licks_ds, reward_ds, raw_ds, mask_off_trials_short, 'k')
    plot_licks(ax2, licks_ds, reward_ds, raw_ds, mask_on_trials_short, 'b')
    plot_licks(ax3, licks_ds, reward_ds, raw_ds, stim_on_trials_short, 'm')
    plot_licks(ax4, licks_ds, reward_ds, raw_ds, mask_off_trials_long, 'k')
    plot_licks(ax5, licks_ds, reward_ds, raw_ds, mask_on_trials_long, 'b')
    plot_licks(ax6, licks_ds, reward_ds, raw_ds, stim_on_trials_long, 'm')

    first_lick_mask_off_short = plot_first_licks(ax7, licks_ds, reward_ds, mask_off_trials_short, 'k')
    first_lick_mask_on_short = plot_first_licks(ax7, licks_ds, reward_ds, mask_on_trials_short, 'b')
    first_lick_stim_on_short = plot_first_licks(ax7, licks_ds, reward_ds, stim_on_trials_short, 'm')
    first_lick_mask_off_long = plot_first_licks(ax8, licks_ds, reward_ds, mask_off_trials_long, 'k')
    first_lick_mask_on_long = plot_first_licks(ax8, licks_ds, reward_ds, mask_on_trials_long, 'b')
    first_lick_stim_on_long = plot_first_licks(ax8, licks_ds, reward_ds, stim_on_trials_long, 'm')

    first_lick_mask_off_short = plot_first_licks(ax19, licks_ds, reward_ds, mask_off_trials_short, 'k')
    first_lick_mask_off_long = plot_first_licks(ax19, licks_ds, reward_ds, mask_off_trials_long, '0.5')

    first_lick_mask_on_short = plot_first_licks(ax20, licks_ds, reward_ds, mask_on_trials_short, 'b')
    first_lick_mask_on_long = plot_first_licks(ax20, licks_ds, reward_ds, mask_on_trials_long, '#009EFF')

    first_lick_stim_on_short = plot_first_licks(ax21, licks_ds, reward_ds, stim_on_trials_short, 'm')
    first_lick_stim_on_long = plot_first_licks(ax21, licks_ds, reward_ds, stim_on_trials_long, '#E6A2FF')


    ax19.set_title('Task score: ' + str(np.round(np.median(first_lick_mask_off_long[:,1]) - np.median(first_lick_mask_off_short[:,1]),1)))
    ax20.set_title('Task score: ' + str(np.round(np.median(first_lick_mask_on_long[:,1]) - np.median(first_lick_mask_on_short[:,1]),1)))
    ax21.set_title('Task score: ' + str(np.round(np.median(first_lick_stim_on_long[:,1]) - np.median(first_lick_stim_on_short[:,1]),1)))

    binsize = 2
    binnr_short = 460/binsize
    plot_running_speed(ax9, raw_ds, mask_off_trials_short, binsize, binnr_short, 'k')
    plot_running_speed(ax9, raw_ds, mask_on_trials_short, binsize, binnr_short, 'b')
    plot_running_speed(ax9, raw_ds, stim_on_trials_short, binsize, binnr_short, 'm')

    binnr_long = 460/binsize
    plot_running_speed(ax10, raw_ds, mask_off_trials_long, binsize, binnr_long, 'k')
    plot_running_speed(ax10, raw_ds, mask_on_trials_long, binsize, binnr_long, 'b')
    plot_running_speed(ax10, raw_ds, stim_on_trials_long, binsize, binnr_long, 'm')

    ax9.axvspan(200/binsize,240/binsize,color='0.9',zorder=0)
    ax9.axvspan(320/binsize,340/binsize,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)
    ax10.axvspan(200/binsize,240/binsize,color='0.9',zorder=0)
    ax10.axvspan(380/binsize,400/binsize,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=9)

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

        sns.distplot(bootstrap_diff_mask_on_mask_off_short,ax=ax11,color='k')
        vl_handle = ax5.axvline(np.mean(bootstrap_diff_mask_on_mask_off_short),c='k')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_mask_off_short))
        ax11.legend()
        ax11.set_title('mask on vs mask off SHORT')
        ax11.set_yticklabels('')
        ax11.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_on_stim_on_short,ax=ax12,color='m')
        vl_handle = ax12.axvline(np.mean(bootstrap_diff_mask_on_stim_on_short),c='m')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_stim_on_short))
        ax12.legend()
        ax12.set_title('mask on vs stim on SHORT')
        ax12.set_yticklabels('')
        ax12.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_off_stim_on_short,ax=ax13,color='b')
        vl_handle = ax13.axvline(np.mean(bootstrap_diff_mask_off_stim_on_short),c='b')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_off_stim_on_short))
        ax13.legend()
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

        sns.distplot(bootstrap_diff_mask_on_mask_off_long,ax=ax15,color='k')
        vl_handle = ax5.axvline(np.mean(bootstrap_diff_mask_on_mask_off_long),c='k')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_mask_off_long))
        ax15.legend()
        ax15.set_title('mask on vs mask off long')
        ax15.set_yticklabels('')
        ax15.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_on_stim_on_long,ax=ax16,color='m')
        vl_handle = ax16.axvline(np.mean(bootstrap_diff_mask_on_stim_on_long),c='m')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_stim_on_long))
        ax16.legend()
        ax16.set_title('mask on vs stim on long')
        ax16.set_yticklabels('')
        ax16.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_off_stim_on_long,ax=ax17,color='b')
        vl_handle = ax17.axvline(np.mean(bootstrap_diff_mask_off_stim_on_long),c='b')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_off_stim_on_long))
        ax17.legend()
        ax17.set_title('mask off vs stim on long')
        ax17.set_yticklabels('')
        ax17.set_xlim([-100,100])

    ax14.bar([1],[np.var(first_lick_mask_off_short[:,1])],color='k')
    ax14.bar([2],[np.var(first_lick_mask_on_short[:,1])],color='b')
    ax14.bar([3],[np.var(first_lick_stim_on_short[:,1])],color='m')
    ax14.set_xlim([0.5,4.5])
    ax14.set_title('first lick variance')
    ax14.set_xticks([1.4,2.4,3.4])
    ax14.set_xticklabels(['off','mask','stim'])
    ax14.spines['right'].set_visible(False)
    ax14.spines['top'].set_visible(False)

    ax18.bar([1],[np.var(first_lick_mask_off_long[:,1])],color='k')
    ax18.bar([2],[np.var(first_lick_mask_on_long[:,1])],color='b')
    ax18.bar([3],[np.var(first_lick_stim_on_long[:,1])],color='m')
    ax18.set_xlim([0.5,4.5])
    ax18.set_title('first lick variance')
    ax18.set_xticks([1.4,2.4,3.4])
    ax18.set_xticklabels(['off','mask','stim'])
    ax18.spines['right'].set_visible(False)
    ax18.spines['top'].set_visible(False)

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

    return np.median(first_lick_mask_off_long[:,1]) - np.median(first_lick_mask_off_short[:,1]), \
           np.median(first_lick_mask_on_long[:,1]) - np.median(first_lick_mask_on_short[:,1]), \
           np.median(first_lick_stim_on_long[:,1]) - np.median(first_lick_stim_on_short[:,1])

def make_summary_figure():
    fl_diff_median_mask_off = []
    fl_diff_median_mask_on = []
    fl_diff_median_stim_on = []

    fl_diff_median_mask_off_exp = []
    fl_diff_median_mask_on_exp = []
    fl_diff_median_stim_on_exp = []

    MOUSE = 'LF180728_1'
    s = 'Day2018923' #,'Day2018928'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018928'
    s = 'Day2018105'
    diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    print(MOUSE + ' ' + s + ' done.')

    # #
    MOUSE = 'LF180514_1'
    s = 'Day2018813'#,'Day2018924']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018924'
    s = 'Day2018105'
    diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    print(MOUSE + ' ' + s + ' done.')
    #
    # # # #
    MOUSE = 'LF180515_1'
    s = 'Day2018910'#,'Day2018924']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    fl_diff_median_mask_off.append(diff_mask_off)
    fl_diff_median_mask_on.append(diff_mask_on)
    fl_diff_median_stim_on.append(diff_stim_on)
    print(MOUSE + ' ' + s + ' done.')

    # s = 'Day2018924'
    s = 'Day2018105'
    diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    fl_diff_median_mask_off_exp.append(diff_mask_off)
    fl_diff_median_mask_on_exp.append(diff_mask_on)
    fl_diff_median_stim_on_exp.append(diff_stim_on)
    print(MOUSE + ' ' + s + ' done.')

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
    print(sp.stats.ttest_rel(np.array(fl_diff_median_mask_on_exp),np.array(fl_diff_median_stim_on_exp)))

    # create figure to later plot on
    fig = plt.figure(figsize=(4,6))
    ax1 = plt.subplot(111)

    ax1.scatter([0,0,0],fl_diff_median_mask_off,s=80, facecolor='k', edgecolor='k', linewidth=2,zorder=3)
    ax1.scatter([1,1,1],fl_diff_median_mask_on_exp,s=80, facecolor='w', edgecolor='b', linewidth=2,zorder=3)
    ax1.scatter([2,2,2],fl_diff_median_stim_on_exp,s=80, facecolor='b', edgecolor='b', linewidth=2,zorder=3)
    ax1.plot([0,1,2],[fl_diff_median_mask_off[0],fl_diff_median_mask_on_exp[0],fl_diff_median_stim_on_exp[0]],c='k')
    ax1.plot([0,1,2],[fl_diff_median_mask_off[1],fl_diff_median_mask_on_exp[1],fl_diff_median_stim_on_exp[1]],c='k')
    ax1.plot([0,1,2],[fl_diff_median_mask_off[2],fl_diff_median_mask_on_exp[2],fl_diff_median_stim_on_exp[2]],c='k')

    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(['naive','expert + mask','expert + mask + opto'], rotation=45, fontsize=16)

    ax1.set_ylabel('Task score', fontsize=16)

    ax1.spines['left'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    subfolder = 'Opto'
    fname = 'Summary'

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

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'svg'

    with open(loc_info['yaml_archive'], 'r') as f:
        project_metainfo = yaml.load(f)

    # make_summary_figure()

    # MOUSE = 'LF180728_1'
    # SESSION = ['Day20181017']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     diff_mask_off,diff_mask_on,diff_stim_on = fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    #
    # # #
    MOUSE = 'LF180514_1'
    SESSION = ['Day20181017','Day20181018']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    for s in SESSION:
        fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    #
    # # # #
    MOUSE = 'LF180515_1'
    SESSION = ['Day20181018','Day20181020']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    for s in SESSION:
        fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
