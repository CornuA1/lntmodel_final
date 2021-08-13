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


def fig_behavior_stage5(h5path, sess, fname, fformat='png', subfolder=[]):
    # load data

    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(14,6))
    fig.suptitle(fname)
    ax1 = plt.subplot2grid((8,2),(0,0), rowspan=6)
    ax2 = plt.subplot2grid((8,2),(6,0), rowspan=2)
    ax3 = plt.subplot2grid((8,2),(0,1), rowspan=6)
    ax4 = plt.subplot2grid((8,2),(6,1), rowspan=2)

    ax1.set_xlim([50,340])
    ax1.set_ylabel('Trial #')
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Short trials')

    #ax2.set_xlim([10,67])
    #ax2.set_ylabel('Speed (cm/sec)')
    #ax2.set_xlabel('Location (cm)')

    ax3.set_xlim([50,400])
    ax3.set_ylabel('Trial #')
    ax3.set_xlabel('Location (cm)')
    ax3.set_title('Short trials')

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(380,400,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=9)

    fl_diff = 0
    t_score = 0

    # make array of y-axis locations for licks. If clause to check for empty arrays
    if np.size(licks_ds) > 0 and np.size(reward_ds) > 0:
        # only plot trials where either a lick and/or a reward were detected
        # therefore: pull out trial numbers from licks and rewards dataset and map to
        # a new list of rows used for plotting

        short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
        #short_trials = filter_trials( raw_ds, [], ['opto_stim_on'],short_trials)
        long_trials = filter_trials( raw_ds, [], ['tracknumber',4])
        #long_trials = filter_trials( raw_ds, [], ['opto_stim_on'],long_trials)

        # get trial numbers to be plotted
        lick_trials = np.unique(licks_ds[:,2])
        reward_trials = np.unique(reward_ds[:,3])-1
        scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
        scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
        scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
        scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
        scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

        ax1.set_ylim([0,len(np.unique(scatter_rowlist_short))])

        # scatterplot of licks/rewards in order of trial number
        for i,r in enumerate(scatter_rowlist_map_short):
            plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
            plot_rewards_x = reward_ds[reward_ds[:,3]-1==r,1]
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            if reward_ds[reward_ds[:,3]-1==r,5] == 1:
                col = '#00C40E'
            else:
                col = 'r'

            # if reward location is recorded at beginning of track, set it to end of track
            if plot_rewards_x < 300:
                plot_rewards_x = 340

            # plot licks and rewards
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

            # if reward location is recorded at beginning of track, set it to end of track
            if plot_rewards_x < 300:
                plot_rewards_x = 400

            # plot licks and rewards
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
        if False:
            bin_size = 5
            binnr_short = 400/bin_size
            mean_speed = np.empty((np.size(scatter_rowlist_map_short,0),int(binnr_short)))
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
                    mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
                    #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
                    max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])

            sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
            mean_speed_sess_short = np.nanmean(mean_speed,0)
            ax2.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c=sns.xkcd_rgb["windows blue"],zorder=3)
            ax2.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color=sns.xkcd_rgb["windows blue"],alpha=0.2)

            # plot running speed
            binnr_long = 400/bin_size
            mean_speed = np.empty((np.size(scatter_rowlist_map_long,0),int(binnr_long)))
            mean_speed[:] = np.NAN
            max_y_long = 0
            for i,t in enumerate(scatter_rowlist_map_long):
                cur_trial = raw_ds[raw_ds[:,6]==t,:]
                cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
                cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
                cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)

                if cur_trial_bins-cur_trial_start_bin > 0:
                    print(np.size(mean_speed_trial))
                    mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], raw_ds[raw_ds[:,6]==t,
                                                                  3], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
                    mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
                    #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
                    max_y_long = np.amax([max_y_long,np.amax(mean_speed_trial)])

            sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
            mean_speed_sess_long = np.nanmean(mean_speed,0)
            ax2.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c=sns.xkcd_rgb["dusty purple"],zorder=3)
            ax2.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color=sns.xkcd_rgb["dusty purple"],alpha=0.2)

        # plot location of first trials on short and long trials
        first_lick_short = []
        first_lick_short_trials = []
        first_lick_long = []
        first_lick_long_trials = []
        for r in lick_trials:
            licks_all = licks_ds[licks_ds[:,2]==r,:]
            if not licks_all.size == 0:
                licks_all = licks_all[licks_all[:,1]>101,:]
            else:
                rew_lick = rewards_ds[rewards_ds[:,3]-1==r,:]
                if r%20 <= 10:
                    licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 3]])
                else:
                    licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 4]])
            if licks_all.shape[0]>0:
                lick = licks_all[0]
                if lick[3] == 3:
                    first_lick_short.append(lick[1])
                    first_lick_short_trials.append(r)
                elif lick[3] == 4:
                    first_lick_long.append(lick[1])
                    first_lick_long_trials.append(r)

        first_lick_short_pairs = np.vstack((first_lick_short,first_lick_short_trials))
        first_lick_short_pairs = first_lick_short_pairs[:,np.in1d(first_lick_short_pairs[1,:],short_trials)]
        first_lick_long_pairs = np.vstack((first_lick_long,first_lick_long_trials))
        first_lick_long_pairs = first_lick_long_pairs[:,np.in1d(first_lick_long_pairs[1,:],long_trials)]

        ax4.scatter(first_lick_short_pairs[0,:],first_lick_short_pairs[1,:],c=sns.xkcd_rgb["windows blue"],lw=0)
        ax4.scatter(first_lick_long_pairs[0,:],first_lick_long_pairs[1,:],c=sns.xkcd_rgb["dusty purple"],lw=0)
        ax4.axvline(np.median(first_lick_short), c=sns.xkcd_rgb["windows blue"], lw=2)
        ax4.axvline(np.median(first_lick_long), c=sns.xkcd_rgb["dusty purple"], lw=2)

        if np.size(first_lick_short) > 10:
            fl_short_running_avg = np.convolve(first_lick_short,np.ones(10),'valid')/10
            ax4.plot(fl_short_running_avg, first_lick_short_trials[5:len(first_lick_short_trials)-4], c=sns.xkcd_rgb["windows blue"], lw=2)

        if np.size(first_lick_long) > 10:
            fl_long_running_avg = np.convolve(first_lick_long,np.ones(10),'valid')/10
            ax4.plot(fl_long_running_avg, first_lick_long_trials[5:len(first_lick_long_trials)-4], c=sns.xkcd_rgb["dusty purple"], lw=2)

        # bootstrap differences between pairs of first lick locations
        if np.size(first_lick_short) > 5 and np.size(first_lick_long) > 5:
            num_shuffles = 10000
            short_bootstrap = np.random.choice(first_lick_short,num_shuffles)
            long_bootstrap = np.random.choice(first_lick_long,num_shuffles)
            bootstrap_diff = long_bootstrap - short_bootstrap
            # tval,pval = stats.ttest_1samp(bootstrap_diff,0)
            # pval = np.size(np.where(bootstrap_diff < 0))/num_shuffles
            fl_diff = np.mean(bootstrap_diff)/np.std(bootstrap_diff)

            sns.distplot(bootstrap_diff,ax=ax2)
            vl_handle = ax2.axvline(np.mean(bootstrap_diff),c='b')
            vl_handle.set_label('z-score = ' + str(fl_diff))
            ax2.legend()



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
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    fformat = 'png'

    with open(loc_info['yaml_archive'], 'r') as f:
        project_metainfo = yaml.load(f)

    #
    # MOUSE = 'LF180514_1'
    # SESSION = ['Day201894','Day201895','Day201896','Day201898','Day2018910','Day2018911']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    #
    # MOUSE = 'LF180515_1'
    # SESSION = ['Day201894','Day201895','Day201896','Day201898','Day2018910','Day2018911']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
    #
    # MOUSE = 'LF180622_1'
    # SESSION = ['Day201894','Day201895','Day201896','Day201898','Day2018910','Day2018911']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)

    MOUSE = 'LF180728_1'
    SESSION = ['Day2018926']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    for s in SESSION:
        fig_behavior_stage5(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)
