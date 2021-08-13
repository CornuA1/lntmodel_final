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

def fig_s2_mask_v_nomask(h5path, sess, fname, fformat='png', subfolder=[]):
    # load data
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(14,6))
    fig.suptitle(fname)
    ax1 = plt.subplot2grid((10,2),(0,0), rowspan=6)
    ax2 = plt.subplot2grid((10,2),(6,0), rowspan=2)
    ax3 = plt.subplot2grid((10,2),(0,1), rowspan=6)
    ax4 = plt.subplot2grid((10,2),(6,1), rowspan=2)
    ax5 = plt.subplot2grid((10,2),(8,0), rowspan=2)

    ax1.set_xlim([50,340])
    ax1.set_ylabel('Trial #')
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('MASK LIGHT OFF')

    ax2_2 = ax2.twinx()
    ax2.set_xlabel('Location (cm)')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2_2.spines['right'].set_visible(False)
    ax2_2.spines['top'].set_visible(False)

    ax3.set_xlim([50,340])
    ax3.set_ylabel('Trial #')
    ax3.set_xlabel('Location (cm)')
    ax3.set_title('MASK LIGHT ON')

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    fl_diff = 0
    t_score = 0

    # divide trials up into mask on and mask off trials.
    short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
    mask_off_trials = filter_trials( raw_ds, [], ['opto_mask_light_off',3],short_trials)
    mask_on_trials = filter_trials( raw_ds, [], ['opto_mask_light_on',4],short_trials)

    ax1.set_ylim([0,len(mask_off_trials)])
    ax3.set_ylim([0,len(mask_on_trials)])

    # scatterplot of licks/rewards in order of trial number
    for i,r in enumerate(mask_off_trials):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
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

    for i,r in enumerate(mask_on_trials):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
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

    # plot location of first trials on short and long trials
    first_lick_mask_off = np.empty((0,4))
    first_lick_mask_off_trials = np.empty((0))
    first_lick_mask_on = np.empty((0,4))
    first_lick_mask_on_trials = np.empty((0))
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

    # ax2.scatter(first_lick_mask_off[:,1],np.ones(first_lick_mask_off.shape[0]),c=sns.xkcd_rgb["windows blue"],lw=0)
    ax2.scatter(first_lick_mask_off[:,1],first_lick_mask_off_trials,c='k',lw=0)
    ax2.scatter(first_lick_mask_on[:,1],first_lick_mask_on_trials,c='b',lw=0)
    ax2_2 = ax2.twinx()
    sns.kdeplot(first_lick_mask_off[:,1],c='k',ax=ax2_2)
    sns.kdeplot(first_lick_mask_on[:,1],c='b',ax=ax2_2)

    ax2.axvline(101,lw=2,ls='--',c='0.8')
    ax2.set_xlim([50,340])
    ax2_2.set_xlim([50,340])

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

    ax4.set_xlim([50/bin_size,binnr_short])


    # bootstrap differences between pairs of first lick locations
    num_shuffles = 10000
    off_bootstrap = np.random.choice(first_lick_mask_off[:,1],num_shuffles)
    on_bootstrap = np.random.choice(first_lick_mask_on[:,1],num_shuffles)
    bootstrap_diff = on_bootstrap - off_bootstrap
    # tval,pval = stats.ttest_1samp(bootstrap_diff,0)
    # pval = np.size(np.where(bootstrap_diff < 0))/num_shuffles
    fl_diff = np.mean(bootstrap_diff)/np.std(bootstrap_diff)

    sns.distplot(bootstrap_diff,ax=ax5)
    vl_handle = ax5.axvline(np.mean(bootstrap_diff),c='b')
    vl_handle.set_label('z-score = ' + str(fl_diff))
    ax5.legend()

    return



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
    ax4.axvspan(np.nanmedian(first_lick_short)+bt_CI_5_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.2)
    ax4.axvspan(np.nanmedian(first_lick_short)+bt_CI_95_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.2)

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

    ax4.axvspan(np.nanmedian(first_lick_long)+bt_CI_5_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.2)
    ax4.axvspan(np.nanmedian(first_lick_long)+bt_CI_95_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.2)

    ax4.set_xlim([240,380])

    if np.nanmedian(first_lick_long)+bt_CI_5_long > np.nanmedian(first_lick_short) or np.nanmedian(first_lick_short)+bt_CI_95_short < np.nanmedian(first_lick_long):
        print('significant!')

    if np.size(first_lick_short) > 0 and np.size(first_lick_long):
        t_score = np.mean(first_lick_long) - np.mean(first_lick_short)
        ax4.text(320, 10, 't-score: '+str(t_score))



    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    return t_score

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'

    # MOUSE = 'LF180514_1'
    # SESSION = ['Day201884','Day201885']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_s2_mask_v_nomask(h5path, s, MOUSE+s, fformat, MOUSE)

    MOUSE = 'LF180622_1'
    SESSION = ['Day2018829']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    for s in SESSION:
        fig_s2_mask_v_nomask(h5path, s, MOUSE+s, fformat, MOUSE)
