"""
Plot basic behaviour of a given animal and day.

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
from event_ind import event_ind

def fig_s2_mask_v_nomask(h5path, sess, fname, fformat='png', subfolder=[]):
    # load data
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(16,12))
    fig.suptitle(fname)
    ax1 = plt.subplot2grid((100,6),(0,0), rowspan=28, colspan=2)
    ax2 = plt.subplot2grid((100,6),(0,2), rowspan=20, colspan=2)
    ax3 = plt.subplot2grid((100,6),(35,0), rowspan=28, colspan=2)
    ax4 = plt.subplot2grid((100,6),(27,2), rowspan=20, colspan=2)
    ax5 = plt.subplot2grid((100,6),(54,2), rowspan=20, colspan=1)
    ax7 = plt.subplot2grid((100,6),(54,3), rowspan=20, colspan=1)
    ax8 = plt.subplot2grid((100,6),(80,2), rowspan=20, colspan=1)
    ax9 = plt.subplot2grid((100,6),(80,3), rowspan=20, colspan=1)
    ax6 = plt.subplot2grid((100,6),(70,0), rowspan=28, colspan=2)
    ax10 = plt.subplot2grid((100,6),(0,4), rowspan=30, colspan=2)
    ax11 = plt.subplot2grid((100,6),(37,4), rowspan=20, colspan=2)
    ax12 = plt.subplot2grid((100,6),(64,4), rowspan=30, colspan=2)

    ax1.set_xlim([50,340])
    ax1.set_ylabel('Trial #')
    ax1.set_title('LIGHTS OFF')

    ax2.set_xlabel('Location (cm)')
    ax2.set_title('Location of first lick')

    ax3.set_xlim([50,340])
    ax3.set_ylabel('Trial #')
    ax3.set_title('MASK LIGHT ONLY')

    ax6.set_xlim([50,340])
    ax6.set_ylabel('Trial #')
    ax6.set_xlabel('Location (cm)')
    ax6.set_title('MASK LIGHT + OPTO STIM')

    ax10.set_title('trial start vs first lick location (cm)')

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    ax6.axvspan(200,240,color='0.9',zorder=0)
    ax6.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    fl_diff = 0
    t_score = 0

    # divide trials up into mask on and mask off trials.
    short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
    # short_trials = filter_trials( raw_ds, [], ['trialnr_range',0,100],short_trials)
    #short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,240]],short_trials)
    # short_trials = filter_trials( raw_ds, [], ['maxrewardtime',10],short_trials)


    mask_off_trials = filter_trials( raw_ds, [], ['opto_mask_light_off'],short_trials)

    mask_on_trials = filter_trials( raw_ds, [], ['opto_mask_on_stim_off'],short_trials)
    stim_on_trials = filter_trials( raw_ds, [], ['opto_stim_on'],short_trials)


    ax1.set_ylim([0,len(mask_off_trials)])
    ax3.set_ylim([0,len(mask_on_trials)])

    # scatterplot of licks/rewards in order of trial number
    trial_start_mask_off = np.empty((0,3))
    trial_start_mask_off_trials = np.empty((0))
    for i,r in enumerate(mask_off_trials):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        cur_trial_start_time = raw_ds[raw_ds[:,6]==r,0][0]
        cur_trial_end_time = raw_ds[raw_ds[:,6]==r,0][-1]
        trial_start_mask_off = np.vstack((trial_start_mask_off, [cur_trial_start,cur_trial_start_time,cur_trial_end_time]))
        trial_start_mask_off_trials = np.append(trial_start_mask_off_trials, r)
        if reward_ds[reward_ds[:,3]==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        # if reward location is recorded at beginning of track, set it to end of track
        if plot_rewards_x < 300:
            plot_rewards_x = 340

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],i)
            ax1.scatter(plot_licks_x, plot_licks_y,c='k',lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = i
            ax1.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
        if np.size(cur_trial_start) > 0:
            plot_starts_y = i
            ax1.scatter(cur_trial_start, plot_starts_y,c='k',marker='>',lw=0)

    trial_start_mask_on = np.empty((0,3))
    trial_start_mask_on_trials = np.empty((0))
    for i,r in enumerate(mask_on_trials):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        cur_trial_start_time = raw_ds[raw_ds[:,6]==r,0][0]
        cur_trial_end_time = raw_ds[raw_ds[:,6]==r,0][-1]
        trial_start_mask_on = np.vstack((trial_start_mask_on, [cur_trial_start,cur_trial_start_time,cur_trial_end_time]))
        trial_start_mask_on_trials = np.append(trial_start_mask_on_trials, r)
        if reward_ds[reward_ds[:,3]==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        # if reward location is recorded at beginning of track, set it to end of track
        if plot_rewards_x < 300:
            plot_rewards_x = 400

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],i)
            ax3.scatter(plot_licks_x, plot_licks_y,c='b',lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = i
            ax3.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
        if np.size(cur_trial_start) > 0:
            plot_starts_y = i
            ax3.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)

    trial_start_stim_on = np.empty((0,3))
    trial_start_stim_on_trials = np.empty((0))
    for i,r in enumerate(stim_on_trials):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        cur_trial_start_time = raw_ds[raw_ds[:,6]==r,0][0]
        cur_trial_end_time = raw_ds[raw_ds[:,6]==r,0][-1]
        trial_start_stim_on = np.vstack((trial_start_stim_on, [cur_trial_start,cur_trial_start_time,cur_trial_end_time]))
        trial_start_stim_on_trials = np.append(trial_start_stim_on_trials, r)
        if reward_ds[reward_ds[:,3]==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        # if reward location is recorded at beginning of track, set it to end of track
        if plot_rewards_x < 300:
            plot_rewards_x = 400

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],i)
            ax6.scatter(plot_licks_x, plot_licks_y,c='m',lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = i
            ax6.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
        if np.size(cur_trial_start) > 0:
            plot_starts_y = i
            ax6.scatter(cur_trial_start, plot_starts_y,c='m',marker='>',lw=0)

    # plot location of first trials on short and long trials
    first_lick_mask_off = np.empty((0,4))
    first_lick_mask_off_trials = np.empty((0))
    for r in mask_off_trials:
        licks_all = licks_ds[licks_ds[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>101,:]
        if licks_all.size == 0:
             rew_lick = reward_ds[reward_ds[:,3]==r,:]
             if rew_lick.size > 0:
                 licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                 first_lick_mask_off = np.vstack((first_lick_mask_off, licks_all[0,:].T))
                 first_lick_mask_off_trials = np.append(first_lick_mask_off_trials, r)
        else:
            first_lick_mask_off = np.vstack((first_lick_mask_off, licks_all[0,:].T))
            first_lick_mask_off_trials = np.append(first_lick_mask_off_trials, r)

    first_lick_mask_on = np.empty((0,4))
    first_lick_mask_on_trials = np.empty((0))
    for r in mask_on_trials:
        licks_all = licks_ds[licks_ds[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>101,:]
        if licks_all.size == 0:
             rew_lick = reward_ds[reward_ds[:,3]==r,:]
             if rew_lick.size > 0:
                 licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                 first_lick_mask_on = np.vstack((first_lick_mask_on, licks_all[0,:].T))
                 first_lick_mask_on_trials = np.append(first_lick_mask_on_trials, r)
        else:
            first_lick_mask_on = np.vstack((first_lick_mask_on, licks_all[0,:].T))
            first_lick_mask_on_trials = np.append(first_lick_mask_on_trials, r)

    first_lick_stim_on = np.empty((0,4))
    first_lick_stim_on_trials = np.empty((0))
    for r in stim_on_trials:
        licks_all = licks_ds[licks_ds[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>101,:]
        if licks_all.size == 0:
             rew_lick = reward_ds[reward_ds[:,3]==r,:]
             if rew_lick.size > 0:
                 licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                 first_lick_stim_on = np.vstack((first_lick_stim_on, licks_all[0,:].T))
                 first_lick_stim_on_trials = np.append(first_lick_stim_on_trials, r)
        else:
            first_lick_stim_on = np.vstack((first_lick_stim_on, licks_all[0,:].T))
            first_lick_stim_on_trials = np.append(first_lick_stim_on_trials, r)

    # ax2.scatter(first_lick_mask_off[:,1],np.ones(first_lick_mask_off.shape[0]),c=sns.xkcd_rgb["windows blue"],lw=0)
    ax2.scatter(first_lick_mask_off[:,1],first_lick_mask_off_trials,c='k',lw=0)
    ax2.scatter(first_lick_mask_on[:,1],first_lick_mask_on_trials,c='b',lw=0)
    ax2.scatter(first_lick_stim_on[:,1],first_lick_stim_on_trials,c='m',lw=0)
    ax2_2 = ax2.twinx()
    sns.kdeplot(first_lick_mask_off[:,1],c='k',ax=ax2_2)
    if np.size(first_lick_mask_on) > 0:
        sns.kdeplot(first_lick_mask_on[:,1],c='b',ax=ax2_2)
    if np.size(first_lick_stim_on) > 0:
        sns.kdeplot(first_lick_stim_on[:,1],c='m',ax=ax2_2)

    ax2.axvline(101,lw=2,ls='--',c='0.8')
    ax2.set_xlim([50,400])
    ax2_2.set_xlim([50,400])
    ax2_2.set_yticklabels([''])
    ax2.set_yticklabels([''])

    # plot running speed
    bin_size = 2
    binnr_short = 340/bin_size
    mean_speed = np.empty((mask_off_trials.shape[0],int(binnr_short)))
    mean_speed[:] = np.NAN
    max_y_short = 0
    for i,t in enumerate(mask_off_trials):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_start_bin = np.floor(cur_trial[0,1]/bin_size)
        cur_trial_end_bin = np.ceil(cur_trial[-1,1]/bin_size)
        cur_trial_bins = cur_trial_end_bin - cur_trial_start_bin

        # make sure the session didn't end right after a trial started causing the script to trip
        if cur_trial_bins > 0:
            mean_speed_trial = stats.binned_statistic(cur_trial[:,1], cur_trial[:,3], 'mean', cur_trial_bins)[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_end_bin)] = mean_speed_trial
            # ax4.plot(np.linspace(cur_trial_start_bin,cur_trial_end_bin,cur_trial_bins),mean_speed_trial,c='0.8',alpha=0.5,zorder=2)
        #     max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])
    #
    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax4.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='k',zorder=3)
    ax4.fill_between(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color='k',alpha=0.2)

    mean_speed = np.empty((mask_on_trials.shape[0],int(binnr_short)))
    mean_speed[:] = np.NAN
    max_y_short = 0
    for i,t in enumerate(mask_on_trials):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_start_bin = np.floor(cur_trial[0,1]/bin_size)
        cur_trial_end_bin = np.ceil(cur_trial[-1,1]/bin_size)
        cur_trial_bins = cur_trial_end_bin - cur_trial_start_bin

        # make sure the session didn't end right after a trial started causing the script to trip
        if cur_trial_bins > 0:
            mean_speed_trial = stats.binned_statistic(cur_trial[:,1], cur_trial[:,3], 'mean', cur_trial_bins)[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_end_bin)] = mean_speed_trial
            # ax4.plot(np.linspace(cur_trial_start_bin,cur_trial_end_bin,cur_trial_bins),mean_speed_trial,c='b',alpha=0.2,zorder=2)
        #     max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])
    #
    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax4.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='b',zorder=3)
    ax4.fill_between(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color='b',alpha=0.2)

    mean_speed = np.empty((stim_on_trials.shape[0],int(binnr_short)))
    mean_speed[:] = np.NAN
    max_y_short = 0
    for i,t in enumerate(stim_on_trials):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_start_bin = np.floor(cur_trial[0,1]/bin_size)
        cur_trial_end_bin = np.ceil(cur_trial[-1,1]/bin_size)
        cur_trial_bins = cur_trial_end_bin - cur_trial_start_bin

        # make sure the session didn't end right after a trial started causing the script to trip
        if cur_trial_bins > 0:
            mean_speed_trial = stats.binned_statistic(cur_trial[:,1], cur_trial[:,3], 'mean', cur_trial_bins)[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_end_bin)] = mean_speed_trial
            # ax4.plot(np.linspace(cur_trial_start_bin,cur_trial_end_bin,cur_trial_bins),mean_speed_trial,c='b',alpha=0.2,zorder=2)
        #     max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])
    #
    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax4.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='m',zorder=3)
    ax4.fill_between(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color='m',alpha=0.2)

    ax4.set_xlim([50/bin_size,binnr_short])
    ax4.set_title('running speed')
    ax4.set_xticks([25,40,60,80,100,120,140,160])
    ax4.set_xticklabels(['50','80','120','160','200','240','280','320'])
    ax4.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0)
    ax4.axvspan(320/bin_size,340/bin_size,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    # bootstrap differences between pairs of first lick locations
    if np.size(first_lick_mask_on) > 0 and np.size(first_lick_stim_on) > 0:

        num_shuffles = 10000
        mask_off_bootstrap = np.random.choice(first_lick_mask_off[:,1],num_shuffles)
        mask_on_bootstrap = np.random.choice(first_lick_mask_on[:,1],num_shuffles)
        stim_on_bootstrap = np.random.choice(first_lick_stim_on[:,1],num_shuffles)

        bootstrap_diff_mask_on_mask_off = mask_on_bootstrap - mask_off_bootstrap
        bootstrap_diff_mask_on_stim_on = mask_on_bootstrap - stim_on_bootstrap
        bootstrap_diff_mask_off_stim_on = mask_off_bootstrap - stim_on_bootstrap

        fl_diff_mask_on_mask_off = np.mean(bootstrap_diff_mask_on_mask_off)/np.std(bootstrap_diff_mask_on_mask_off)
        fl_diff_mask_on_stim_on = np.mean(bootstrap_diff_mask_on_stim_on)/np.std(bootstrap_diff_mask_on_stim_on)
        fl_diff_mask_off_stim_on = np.mean(bootstrap_diff_mask_off_stim_on)/np.std(bootstrap_diff_mask_off_stim_on)

        sns.distplot(bootstrap_diff_mask_on_mask_off,ax=ax5,color='k')
        vl_handle = ax5.axvline(np.mean(bootstrap_diff_mask_on_mask_off),c='k')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_mask_off))
        ax5.legend()
        ax5.set_title('mask on vs mask off')
        ax5.set_yticklabels('')
        ax5.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_on_stim_on,ax=ax7,color='m')
        vl_handle = ax7.axvline(np.mean(bootstrap_diff_mask_on_stim_on),c='m')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_on_stim_on))
        ax7.legend()
        ax7.set_title('mask on vs stim on')
        ax7.set_yticklabels('')
        ax7.set_xlim([-100,100])

        sns.distplot(bootstrap_diff_mask_off_stim_on,ax=ax8,color='b')
        vl_handle = ax8.axvline(np.mean(bootstrap_diff_mask_off_stim_on),c='b')
        vl_handle.set_label('z-score = ' + str(fl_diff_mask_off_stim_on))
        ax8.legend()
        ax8.set_title('mask off vs stim on')
        ax8.set_yticklabels('')
        ax8.set_xlim([-100,100])

        # plot variance of first lick location
        # print(np.mean(first_lick_mask_off[:,1]), np.var(first_lick_mask_off[:,1]))
        # print(np.mean(first_lick_mask_on[:,1]), np.var(first_lick_mask_on[:,1]))
        # print(np.mean(first_lick_stim_on[:,1]), np.var(first_lick_stim_on[:,1]))

    ax9.bar([1],[np.var(first_lick_mask_off[:,1])],color='k')
    ax9.bar([2],[np.var(first_lick_mask_on[:,1])],color='b')
    ax9.bar([3],[np.var(first_lick_stim_on[:,1])],color='m')
    ax9.set_xlim([0.5,4.5])
    ax9.set_title('first lick variance')
    ax9.set_xticks([1.4,2.4,3.4])
    ax9.set_xticklabels(['off','mask','stim'])
    ax9.spines['right'].set_visible(False)
    ax9.spines['top'].set_visible(False)

    #ax10.scatter()

    # match location of trial starts to location of first licks
    mask_off_start_lick_trials = np.intersect1d(trial_start_mask_off_trials, first_lick_mask_off_trials)
    mask_off_loc_pairs = np.empty((0,2))
    for m in mask_off_start_lick_trials:
        trial_start_loc = trial_start_mask_off[trial_start_mask_off_trials == m,0]
        first_lick_loc = first_lick_mask_off[first_lick_mask_off_trials == m,1]
        mask_off_loc_pairs = np.vstack((mask_off_loc_pairs,np.asarray([trial_start_loc,first_lick_loc]).T))

    mask_on_start_lick_trials = np.intersect1d(trial_start_mask_on_trials, first_lick_mask_on_trials)
    mask_on_loc_pairs = np.empty((0,2))
    for m in mask_on_start_lick_trials:
        trial_start_loc = trial_start_mask_on[trial_start_mask_on_trials == m,0]
        first_lick_loc = first_lick_mask_on[first_lick_mask_on_trials == m,1]
        mask_on_loc_pairs = np.vstack((mask_on_loc_pairs,np.asarray([trial_start_loc,first_lick_loc]).T))

    stim_on_start_lick_trials = np.intersect1d(trial_start_stim_on_trials, first_lick_stim_on_trials)
    stim_on_loc_pairs = np.empty((0,2))
    for m in stim_on_start_lick_trials:
        trial_start_loc = trial_start_stim_on[trial_start_stim_on_trials == m,0]
        first_lick_loc = first_lick_stim_on[first_lick_stim_on_trials == m,1]
        stim_on_loc_pairs = np.vstack((stim_on_loc_pairs,np.asarray([trial_start_loc,first_lick_loc]).T))

    ax10.scatter(mask_off_loc_pairs[:,1], mask_off_loc_pairs[:,0],c='k')
    ax10.scatter(mask_on_loc_pairs[:,1], mask_on_loc_pairs[:,0],c='b')
    ax10.scatter(stim_on_loc_pairs[:,1], stim_on_loc_pairs[:,0],c='m')

    mask_off_linregress = stats.linregress(mask_off_loc_pairs[:,1], mask_off_loc_pairs[:,0])
    ax10.plot(np.arange(240,360,1),(np.arange(240,360,1)*mask_off_linregress[0])+mask_off_linregress[1],c='k')
    print(mask_off_linregress)
    if np.size(mask_on_loc_pairs) > 0:
        mask_on_linregress = stats.linregress(mask_on_loc_pairs[:,1], mask_on_loc_pairs[:,0])
        print(mask_on_linregress)
        ax10.plot(np.arange(240,360,1),(np.arange(240,360,1)*mask_on_linregress[0])+mask_on_linregress[1],c='b')
    if np.size(stim_on_loc_pairs) > 0:
        stim_on_linregress = stats.linregress(stim_on_loc_pairs[:,1], stim_on_loc_pairs[:,0])
        print(stim_on_linregress)
        ax10.plot(np.arange(240,360,1),(np.arange(240,360,1)*stim_on_linregress[0])+stim_on_linregress[1],c='m')

    ax11.hist(trial_start_mask_off[:,2] - trial_start_mask_off[:,1],bins=np.arange(0,61,1),histtype='step',fill=False,lw=2,color='k')
    ax11.hist(trial_start_mask_on[:,2] - trial_start_mask_on[:,1],bins=np.arange(0,61,1),histtype='step',fill=False,lw=2,color='b')
    ax11.hist(trial_start_stim_on[:,2] - trial_start_stim_on[:,1],bins=np.arange(0,61,1),histtype='step',fill=False,lw=2,color='m')
    ax11.set_xlim([0,60])

    # match time of trial starts to time of first licks
    # mask_off_start_lick_trials = np.intersect1d(trial_start_mask_off_trials, first_lick_mask_off_trials)
    # mask_off_loc_pairs = np.empty((0,2))
    # for m in mask_off_start_lick_trials:
    #     trial_duration = trial_start_mask_off[trial_start_mask_off_trials == m,2] - trial_start_mask_off[trial_start_mask_off_trials == m,2]
    #     first_lick_time = first_lick_mask_off[first_lick_mask_off_trials == m,0]
    #     mask_off_loc_pairs = np.vstack((mask_off_loc_pairs,np.asarray([trial_start_loc,first_lick_loc]).T))


    # # calculate the confidence intervals for first licks from a bootstrapped distribution
    # # number of resamples
    # bootstrapdists = 1000
    # # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
    # fl_short_bootstrap = np.empty((len(first_lick_short),bootstrapdists))
    # fl_short_bootstrap[:] = np.nan
    # # vector holding bootstrap variance estimate
    # bt_mean_diff = np.empty((bootstrapdists,))
    # bt_mean_diff[:] = np.nan
    #
    # for j in range(bootstrapdists):
    #     if len(first_lick_short) > 0:
    #         fl_short_bootstrap[:,j] = np.random.choice(first_lick_short, len(first_lick_short))
    #         bt_mean_diff[j] = np.nanmedian(fl_short_bootstrap[:,j]) - np.nanmedian(first_lick_short)
    #     else:
    #         bt_mean_diff[j] = np.nan
    # bt_CI_5_short = np.percentile(bt_mean_diff[:],5)
    # bt_CI_95_short = np.percentile(bt_mean_diff[:],95)
    # ax4.axvspan(np.nanmedian(first_lick_short)+bt_CI_5_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.2)
    # ax4.axvspan(np.nanmedian(first_lick_short)+bt_CI_95_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.2)
    #
    # # calculate the confidence intervals for first licks from a bootstrapped distribution
    # # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
    # fl_long_bootstrap = np.empty((len(first_lick_long),bootstrapdists))
    # fl_long_bootstrap[:] = np.nan
    # # vector holding bootstrap variance estimate
    # bt_mean_diff = np.empty((bootstrapdists,))
    # bt_mean_diff[:] = np.nan
    #
    # for j in range(bootstrapdists):
    #     if len(first_lick_long) > 0:
    #         fl_long_bootstrap[:,j] = np.random.choice(first_lick_long, len(first_lick_long))
    #         bt_mean_diff[j] = np.nanmedian(fl_long_bootstrap[:,j]) - np.nanmedian(first_lick_long)
    #     else:
    #         bt_mean_diff[j] = np.nan
    # bt_CI_5_long = np.percentile(bt_mean_diff[:],5)
    # bt_CI_95_long = np.percentile(bt_mean_diff[:],95)
    #
    # ax4.axvspan(np.nanmedian(first_lick_long)+bt_CI_5_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.2)
    # ax4.axvspan(np.nanmedian(first_lick_long)+bt_CI_95_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.2)
    #
    # ax4.set_xlim([240,380])
    #
    # if np.nanmedian(first_lick_long)+bt_CI_5_long > np.nanmedian(first_lick_short) or np.nanmedian(first_lick_short)+bt_CI_95_short < np.nanmedian(first_lick_long):
    #     print('significant!')
    #
    # if np.size(first_lick_short) > 0 and np.size(first_lick_long):
    #     t_score = np.mean(first_lick_long) - np.mean(first_lick_short)
    #     ax4.text(320, 10, 't-score: '+str(t_score))
    #
    #
    #
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

    # return t_score

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'

    # MOUSE = 'LF180514_1'
    # SESSION = ['Day2018812']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_s2_mask_v_nomask(h5path, s, MOUSE+s+'_opto', fformat, MOUSE)

    # MOUSE = 'LF180515_1'
    # SESSION = ['Day201894']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_s2_mask_v_nomask(h5path, s, MOUSE+s+'_opto', fformat, MOUSE)

    MOUSE = 'LF180622_1'
    SESSION = ['Day201894']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    for s in SESSION:
        fig_s2_mask_v_nomask(h5path, s, MOUSE+s+'_opto', fformat, MOUSE)
