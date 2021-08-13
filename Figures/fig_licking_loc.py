"""
Plot licking behavior as a function of location across multiple days.
Separate short and long trials into individual subpanels

"""

import numpy as np
import h5py
import warnings
import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

with open(loc_info['yaml_file'], 'r') as f:
    project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')

from filter_trials import filter_trials

def fig_licking_loc(h5path, mouse, sess, fname, fformat='png', subfolder=[]):
    licks_all = None
    rewards_all = None
    raw_all = None
    for s in sess:
        h5dat = h5py.File(h5path, 'r')
        if licks_all is not None:
            licks_all_ds = np.copy(h5dat[s + '/licks_pre_reward'])
            licks_all_ds[:,2] = licks_all_ds[:,2]+last_trial
            licks_all = np.append(licks_all,licks_all_ds,axis=0)

            rewards_all_ds = np.copy(h5dat[s + '/rewards'])
            rewards_all_ds[:,3] = rewards_all_ds[:,3]+last_trial
            rewards_all = np.append(rewards_all,rewards_all_ds,axis=0)

            raw_all_ds = np.copy(h5dat[s + '/raw_data'])
            raw_all_ds[:,6] = raw_all_ds[:,6]+last_trial
            raw_all = np.append(raw_all,raw_all_ds,axis=0)
            last_trial = raw_all[-1,6]
        else:
            licks_all = np.copy(h5dat[s + '/licks_pre_reward'])
            rewards_all = np.copy(h5dat[s + '/rewards'])
            raw_all = np.copy(h5dat[s + '/raw_data'])
            last_trial = raw_all[-1,6]

        h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(24,6))
    ax1 = plt.subplot2grid((8,2),(0,0), rowspan=6)
    ax2 = plt.subplot2grid((8,2),(0,1), rowspan=6)
    ax3 = plt.subplot2grid((8,2),(6,0), rowspan=6, colspan=2)

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=0)
    ax1.axvspan(320,360,color='0.9',alpha=0.8,zorder=0)

    ax2.axvspan(200,240,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=0)
    ax2.axvspan(380,420,color='0.9',alpha=0.8,zorder=0)

    short_trials = filter_trials( raw_all, [], ['tracknumber',3])
    long_trials = filter_trials( raw_all, [], ['tracknumber',4])

    # get trial numbers to be plotted
    lick_trials = np.unique(licks_all[:,2])
    reward_trials = np.unique(rewards_all[:,3])-1
    scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
    scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
    scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
    scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
    scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

    ax1.set_ylim([-1,len(np.unique(scatter_rowlist_short))])
    ax2.set_ylim([-1,len(np.unique(scatter_rowlist_long))])

    ax1.set_xlim([0,360])
    ax2.set_xlim([0,420])

    ax3.set_xlim([240,380])

    # scatterplot of licks/rewards in order of trial number
    for i,r in enumerate(scatter_rowlist_map_short):
        plot_licks_x = licks_all[licks_all[:,2]==r,1]
        plot_rewards_x = rewards_all[rewards_all[:,3]-1==r,1]
        cur_trial_start = raw_all[raw_all[:,6]==r,1][0]
        if rewards_all[rewards_all[:,3]-1==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        # if reward location is recorded at beginning of track, set it to end of track
        if plot_rewards_x < 300:
            plot_rewards_x = 335

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_short[i])
            ax1.scatter(plot_licks_x, plot_licks_y,c=sns.xkcd_rgb["windows blue"],lw=0,s=12)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = scatter_rowlist_short[i]
            ax1.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0,s=15)
        # if np.size(cur_trial_start) > 0:
        #     plot_starts_y = scatter_rowlist_short[i]
        #     ax1.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)

    # scatterplot of licks/rewards in order of trial number
    for i,r in enumerate(scatter_rowlist_map_long):
        plot_licks_x = licks_all[licks_all[:,2]==r,1]
        plot_rewards_x = rewards_all[rewards_all[:,3]-1==r,1]
        cur_trial_start = raw_all[raw_all[:,6]==r,1][0]
        if rewards_all[rewards_all[:,3]-1==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        # if reward location is recorded at beginning of track, set it to end of track
        if plot_rewards_x < 300:
            plot_rewards_x = 400

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_long[i])
            ax2.scatter(plot_licks_x, plot_licks_y,c=sns.xkcd_rgb["dusty purple"],lw=0,s=12)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = scatter_rowlist_long[i]
            ax2.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0,s=15)
        # if np.size(cur_trial_start) > 0:
        #     plot_starts_y = scatter_rowlist_long[i]
        #     ax2.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)

    # plot location of first trials on short and long trials
    first_lick_short = []
    first_lick_short_trials = []
    first_lick_long = []
    first_lick_long_trials = []
    for r in lick_trials:
        licks_trial = licks_all[licks_all[:,2]==r,:]
        if not licks_trial.size == 0:
            licks_trial = licks_trial[licks_trial[:,1]>240,:]
        else:
            rew_lick = rewards_ds[rewards_ds[:,3]-1==r,:]
            if r%20 <= 10:
                licks_trial = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 3]])
            else:
                licks_trial = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 4]])
        if licks_trial.shape[0]>0:
            lick = licks_trial[0]
            if lick[3] == 3:
                first_lick_short.append(lick[1])
                first_lick_short_trials.append(r)
            elif lick[3] == 4:
                first_lick_long.append(lick[1])
                first_lick_long_trials.append(r)
    # ax3.scatter(first_lick_short,first_lick_short_trials,c=sns.xkcd_rgb["windows blue"],lw=0,zorder=1)
    # ax3.scatter(first_lick_long,first_lick_long_trials,c=sns.xkcd_rgb["dusty purple"],lw=0,zorder=1)
    # ax3.axvline(np.median(first_lick_short), c=sns.xkcd_rgb["windows blue"], lw=2)
    # ax3.axvline(np.median(first_lick_long), c=sns.xkcd_rgb["dusty purple"], lw=2)
    ax3.scatter(np.median(first_lick_long),1, marker='s', s=80, edgecolors=sns.xkcd_rgb["dusty purple"], c='w', lw=2)
    ax3.scatter(np.median(first_lick_short),1, marker='s', s=80, edgecolors=sns.xkcd_rgb["windows blue"], c='w', lw=2)

    ax3.set_ylim([0,2])

    # calculate the confidence intervals for first licks from a bootstrapped distribution
    # number of resamples
    bootstrapdists = 1000
    # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
    fl_short_bootstrap = np.empty((len(first_lick_short),bootstrapdists))
    fl_short_bootstrap[:] = np.nan
    # vector holding bootstrap variance estimate
    bt_mean_diff = np.empty((bootstrapdists,))
    bt_mean_diff[:] = np.nan

    for j in range(bootstrapdists):
        if len(first_lick_short) > 0:
            fl_short_bootstrap[:,j] = np.random.choice(first_lick_short, len(first_lick_short))
            bt_mean_diff[j] = np.nanmedian(fl_short_bootstrap[:,j]) - np.nanmedian(first_lick_short)
        else:
            bt_mean_diff[j] = np.nan
    bt_CI_5_short = np.percentile(bt_mean_diff[:],5)
    bt_CI_95_short = np.percentile(bt_mean_diff[:],95)
    ax3.axvspan(np.nanmedian(first_lick_short)+bt_CI_5_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.3,zorder=0)
    ax3.axvspan(np.nanmedian(first_lick_short)+bt_CI_95_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.3,zorder=0)

    # calculate the confidence intervals for first licks from a bootstrapped distribution
    # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
    fl_long_bootstrap = np.empty((len(first_lick_long),bootstrapdists))
    fl_long_bootstrap[:] = np.nan
    # vector holding bootstrap variance estimate
    bt_mean_diff = np.empty((bootstrapdists,))
    bt_mean_diff[:] = np.nan

    for j in range(bootstrapdists):
        if len(first_lick_long) > 0:
            fl_long_bootstrap[:,j] = np.random.choice(first_lick_long, len(first_lick_long))
            bt_mean_diff[j] = np.nanmedian(fl_long_bootstrap[:,j]) - np.nanmedian(first_lick_long)
        else:
            bt_mean_diff[j] = np.nan
    bt_CI_5_long = np.percentile(bt_mean_diff[:],5)
    bt_CI_95_long = np.percentile(bt_mean_diff[:],95)

    ax3.axvspan(np.nanmedian(first_lick_long)+bt_CI_5_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.3,zorder=0)
    ax3.axvspan(np.nanmedian(first_lick_long)+bt_CI_95_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.3,zorder=0)



    fig.suptitle('Licking behavior vs location: ' + fname + ' Task score: ' + str(np.median(first_lick_long) - np.median(first_lick_short)), wrap=True)
    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
