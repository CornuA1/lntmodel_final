# -*- coding: utf-8 -*-
"""
Plot basic behaviour of a given animal and day

Created on Wed Mar  1 17:22:09 2017

@author: lukasfischer
"""


import numpy as np
import h5py
import warnings
import sys
sys.path.append("D:/Post-Baccalaureate/Research/Harnett/Programming/GitHub/in_vivo/MTH3/Analysis")

from scipy import stats

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from filter_trials import filter_trials

sns.set_style('white')

MOUSE = 'LF161031_4'
DAY   = '20170220'

h5dat = h5py.File('/Users/lukasfischer/Work/exps/MTH3/MTH3.h5', 'r')
print('/Day'+DAY+'/'+MOUSE)
# load datasets and close HDF5 file again
raw_ds = np.copy(h5dat['/Day'+DAY+'/'+MOUSE + '/raw_data'])
licks_ds = np.copy(h5dat['/Day'+DAY+'/'+MOUSE + '/licks_pre_reward'])
reward_ds = np.copy(h5dat['/Day'+DAY+'/'+MOUSE + '/rewards'])
speed_ds = np.copy(h5dat['/Day'+DAY+'/'+MOUSE + '/speedvloc'])
h5dat.close()

# create figure to later plot on
fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot2grid((7,2),(0,0),rowspan=3)
ax3 = plt.subplot2grid((7,2),(0,1),rowspan=3)
ax2 = plt.subplot2grid((7,2),(3,0),rowspan=2)
ax4 = plt.subplot2grid((7,2),(3,1),rowspan=2)
ax5 = plt.subplot2grid((7,2),(5,0),rowspan=2)
ax6 = plt.subplot2grid((7,2),(5,1),rowspan=2)

ax1.set_xlim([0,400])
ax2.set_xlim([0,80])
ax3.set_xlim([0,500])
ax4.set_xlim([0,100])

ax1.set_ylabel('Trial #')
ax1.set_xlabel('Location (cm)')
ax1.set_title('Rasterplot of short trials')
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


ax3.set_ylabel('Trial #')
ax3.set_xlabel('Location (cm)')
ax3.set_title('Rasterplot of long trials')
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

ax2.set_ylabel('Running speed (cm/sec)')
ax2.set_xlabel('Location (cm)')

# plot landmark and rewarded area as shaded zones
ax1.axvspan(200,240,color='0.9',zorder=0)
ax1.axvspan(320,340,color='#D2F2FF',zorder=0)
#ax1.axvline(320,c='0.7',ls='--',lw=2)

ax2.axvspan(40,48,color='0.9',zorder=0)
ax2.axvspan(64,68,color='#D2F2FF',zorder=0)
#ax2.axvline(40,c='0.7',ls='--',lw=2)

ax3.axvspan(200,240,color='0.9',zorder=0)
ax3.axvspan(380,400,color='#D2F2FF',zorder=0)

ax4.axvspan(40,48,color='0.9',zorder=0)
ax4.axvspan(76,80,color='#D2F2FF',zorder=0)
ax4.set_ylabel('Running speed (cm/sec)')
ax4.set_xlabel('Location (cm)')

# make array of y-axis locations for licks. If clause to check for empty arrays
if np.size(licks_ds) > 0 or np.size(reward_ds) > 0:
    # only plot trials where either a lick and/or a reward were detected
    # therefore: pull out trial numbers from licks and rewards dataset and map to
    # a new list of rows used for plotting
    short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
    long_trials = filter_trials( raw_ds, [], ['tracknumber',4])

    # get trial numbers to be plotted
    lick_trials = np.unique(licks_ds[:,2])
    reward_trials = np.unique(reward_ds[:,3])-1
    scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
    scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
    scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
    scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
    scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

    # determine starting location of a given trial and get sorted indices
    cur_trial = []
    for r in scatter_rowlist_map:
        cur_trial.append(raw_ds[raw_ds[:,6]==r,1][0])
    cur_trial_order = np.argsort(cur_trial)

    scatter_rowlist_map_sorted = scatter_rowlist_map[cur_trial_order]

    # scatterplot of licks/rewards in order of trial number
    for i,r in enumerate(scatter_rowlist_map_short):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]-1==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        if reward_ds[reward_ds[:,3]-1==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_short[i])
            ax1.scatter(plot_licks_x, plot_licks_y,c='k',lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = scatter_rowlist_short[i]
            ax1.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
        if np.size(cur_trial_start) > 0:
            plot_starts_y = scatter_rowlist_short[i]
            ax1.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)


    # scatterplot of licks/rewards in order of trial number
    for i,r in enumerate(scatter_rowlist_map_long):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]-1==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        if reward_ds[reward_ds[:,3]-1==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_long[i])
            ax3.scatter(plot_licks_x, plot_licks_y,c='k',lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = scatter_rowlist_long[i]
            ax3.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
        if np.size(cur_trial_start) > 0:
            plot_starts_y = scatter_rowlist_long[i]
            ax3.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)


    # plot running speed
    bin_size = 5
    binnr_short = 460/bin_size
    mean_speed = np.empty((np.size(scatter_rowlist_map_short,0),binnr_short))
    mean_speed[:] = np.NAN
    max_y_short = 0
    for i,t in enumerate(scatter_rowlist_map_short):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)

        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], raw_ds[raw_ds[:,6]==t,
                                   3], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_speed[i,cur_trial_start_bin:cur_trial_bins] = mean_speed_trial
            #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
            max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])

    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax2.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='g',zorder=3)
    ax2.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color='g',alpha=0.2)


    # plot running speed
    bin_size = 5
    binnr_long = 460/bin_size
    mean_speed = np.empty((np.size(scatter_rowlist_long,0),binnr_long))
    mean_speed[:] = np.NAN
    max_y_long = 0
    for i,t in enumerate(scatter_rowlist_map_long):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)
        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], raw_ds[raw_ds[:,6]==t,
                                   3], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_speed[i,cur_trial_start_bin:cur_trial_bins] = mean_speed_trial
            #ax4.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
            max_y_long = np.amax([max_y_long,np.amax(mean_speed_trial)])

    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_long = np.nanmean(mean_speed,0)
    ax4.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c='g',zorder=3)
    ax4.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color='g',alpha=0.2)


    ax2.set_ylim([0,np.amax([max_y_short,max_y_long])])
    ax4.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='0.7',zorder=2)
    ax2.plot(np.linspace(0,binnr_short-1,binnr_long),mean_speed_sess_long,c='0.7',zorder=2)
    ax4.set_ylim([0,np.amax([max_y_short,max_y_long])])
    ax2.set_title('Running speed short track')
    ax4.set_title('Running speed long track')

    # graph of location of first lick vs. starting position
    first_lick = []
    lick_trial_start = []
    for r in lick_trials:
        licks_all = licks_ds[licks_ds[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>240,:]
        if licks_all.shape[0]>0:
            if licks_all[0,3] == 3:
                first_lick.append(320-licks_all[licks_all[:,2]==r,1][0])
                lick_trial_start.append(raw_ds[raw_ds[:,6]==r,1][0])
            if licks_all[0,3] == 4:
                first_lick.append(380-licks_all[licks_all[:,2]==r,1][0])
                lick_trial_start.append(raw_ds[raw_ds[:,6]==r,1][0])

    ax6.scatter(lick_trial_start,first_lick)
    ax6.set_xlim([50,150])
    ax6.set_ylabel('Distance from reward zone (cm)')
    ax6.set_xlabel('Starting position (cm)')
    ax6.set_title('First lick distance from reward vs. starting location')

    # plot location of first trials on short and long trials
    first_lick_short = []
    first_lick_short_trials = []
    first_lick_long = []
    first_lick_long_trials = []
    for r in lick_trials:
        licks_all = licks_ds[licks_ds[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>240,:]
        if licks_all.shape[0]>0:
            lick = licks_all[0]
            if lick[3] == 3:
                first_lick_short.append(lick[1])
                first_lick_short_trials.append(r)
            elif lick[3] == 4:
                first_lick_long.append(lick[1])
                first_lick_long_trials.append(r)

    ax5.scatter(first_lick_short,first_lick_short_trials,c=sns.xkcd_rgb["windows blue"],lw=0)
    ax5.scatter(first_lick_long,first_lick_long_trials,c=sns.xkcd_rgb["dusty purple"],lw=0)
    ax5_1 = ax5.twinx()
    if len(first_lick_short) > 0:
        sns.kdeplot(np.asarray(first_lick_short),c=sns.xkcd_rgb["windows blue"],label='short',shade=True,ax=ax5_1)
    if len(first_lick_long) > 0:
        sns.kdeplot(np.asarray(first_lick_long),c=sns.xkcd_rgb["dusty purple"],label='long',shade=True,ax=ax5_1)

    ax5.set_xlim([250,450])
    ax5.set_ylabel('Trial #')
    ax5_1.set_ylabel('KDE of first licks')
    ax5.set_xlabel('Location (cm)')
    ax5.set_title('Distribution of first licks after the landmark location, short and long trials')
    stat, pval = stats.mannwhitneyu(first_lick_short,first_lick_long)
    if pval < 0.005:
        ax5.annotate('pval: <0.005',xy=(203,5))
    else:
        ax5.annotate('pval: ' + str(np.round(pval,5)),xy=(203,5))

    median_diff = (np.median(first_lick_long)-np.median(first_lick_short))
    ax5.annotate('median diff: ' + str(round(median_diff,1)) + ' cm',xy=(203,15))

    ax1.set_ylim([20,50])
    ax1.set_xlim([0,400])
    ax3.set_ylim([20,50])
    ax3.set_xlim([0,400])

    plt.tight_layout()

fname = 'MTH3_behaviour_'+MOUSE+'_'+DAY+'.svg'
fig.savefig(fname, format="svg")
