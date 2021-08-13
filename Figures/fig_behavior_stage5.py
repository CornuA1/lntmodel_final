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
# import ipdb
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("white")

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
#
# with open(loc_info['yaml_file'], 'r') as f:
#     project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')

from filter_trials import filter_trials
from load_behavior_data import load_data
from rewards import rewards
from licks import licks_nopost as licks

def load_data_h5(h5path, sess):
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()
    return raw_ds, licks_ds, reward_ds

def load_raw_data(raw_filename, sess):
    raw_data = load_data(raw_filename, 'vr')
    all_licks = licks(raw_data)
    trial_licks = all_licks[np.in1d(all_licks[:, 3], [3, 4]), :]
    reward =  rewards(raw_data)
    return raw_data, trial_licks, reward

def fig_behavior_stage5(h5path, sess, fname, fformat='png', subfolder=[], load_raw=False):
    # load data

    # h5dat = h5py.File(h5path, 'r')
    # raw_ds = np.copy(h5dat[sess + '/raw_data'])
    # licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    # reward_ds = np.copy(h5dat[sess + '/rewards'])
    # h5dat.close()

    if load_raw == False:
        raw_ds, licks_ds, reward_ds = load_data_h5(h5path, sess)
    else:
        # ipdb.set_trace()
        raw_filename = data_path + os.sep + sess[0] + os.sep + sess[1]
        raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sess)


    # create figure to later plot on
    fig = plt.figure(figsize=(14,8))
    fig.suptitle(fname)
    ax1 = plt.subplot2grid((10,2),(0,0), rowspan=6)
    ax2 = plt.subplot2grid((10,2),(6,0), rowspan=2)
    ax3 = plt.subplot2grid((10,2),(0,1), rowspan=6)
    ax4 = plt.subplot2grid((10,2),(6,1), rowspan=2)
    ax5 = plt.subplot2grid((10,2),(8,0), rowspan=2)
    ax6 = plt.subplot2grid((10,2),(8,1), rowspan=2)

    ax1.set_xlim([50,400])
    ax1.set_ylabel('Trial #')
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Short trials')

    #ax2.set_xlim([10,67])
    #ax2.set_ylabel('Speed (cm/sec)')
    #ax2.set_xlabel('Location (cm)')

    ax3.set_xlim([50,480])
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
        long_trials = filter_trials( raw_ds, [], ['tracknumber',4])

        # short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],short_trials)
        # long_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],long_trials)


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
            ax5.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c=sns.xkcd_rgb["windows blue"],zorder=3)
            ax5.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color=sns.xkcd_rgb["windows blue"],alpha=0.2)

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
                    # print(np.size(mean_speed_trial))
                    mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], raw_ds[raw_ds[:,6]==t,
                                                                  3], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
                    mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
                    #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
                    max_y_long = np.amax([max_y_long,np.amax(mean_speed_trial)])

            sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
            mean_speed_sess_long = np.nanmean(mean_speed,0)
            ax6.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c=sns.xkcd_rgb["dusty purple"],zorder=3)
            ax6.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color=sns.xkcd_rgb["dusty purple"],alpha=0.2)

        # plot location of first trials on short and long trials
        first_lick_short = []
        first_lick_short_trials = []
        first_lick_long = []
        first_lick_long_trials = []
        for r in lick_trials:
            licks_all = licks_ds[licks_ds[:,2]==r,:]
            if not licks_all.size == 0:
                licks_all = licks_all[licks_all[:,1]>240,:]
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
        ax4.scatter(first_lick_short,first_lick_short_trials,c=sns.xkcd_rgb["windows blue"],lw=0)
        ax4.scatter(first_lick_long,first_lick_long_trials,c=sns.xkcd_rgb["dusty purple"],lw=0)
        # ax4.axvline(np.median(first_lick_short), c=sns.xkcd_rgb["windows blue"], lw=2)
        # ax4.axvline(np.median(first_lick_long), c=sns.xkcd_rgb["dusty purple"], lw=2)

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
        bootstrapdists = 100
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

        if np.size(first_lick_short) > 0 and np.size(first_lick_long) > 0:
            t_score = np.median(first_lick_long) - np.median(first_lick_short)
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

    print(fname)
    return t_score

if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline
#
    fformat = 'png'
    

    MOUSE = 'LF191022_1'
    SESSION = [['20191126','MTH3_vr1_s5lr_20191126_1743.csv']]
    data_path = loc_info['raw_dir'] + MOUSE
    # for s in SESSION:
        # fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)

    # MOUSE = 'LF191023_blank'
    # SESSION = [['20191116','MTH3_vr1_s5r_20191116_1919.csv']]
    # data_path = loc_info['raw_dir'] + MOUSE
    # for s in SESSION:
    #     fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)

    MOUSE = 'LF191023_blue'
    SESSION = [['20191116','MTH3_vr1_s5r_20191116_1919.csv']]
    data_path = loc_info['raw_dir'] + MOUSE
    for s in SESSION:
        fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)

    # MOUSE = 'LF191024_1'
    # SESSION = [['20191126','MTH3_vr1_s5lr_20191126_1849.csv']]
    # data_path = loc_info['raw_dir'] + MOUSE
    # for s in SESSION:
    #     fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)

#
#     with open(loc_info['yaml_archive'], 'r') as f:
#         project_metainfo = yaml.load(f)
#
#     # sum_fig = plt.figure(figsize=(10,6))
#     # sum_ax = plt.subplot(111)
#     # sum_ax.set_xlabel('Session #')
#     #tn_ax.set_xlabel('Session #')
#     # sum_ax.set_ylabel('Task Score')
#     # tn_ax.set_ylabel('Number of trials')
#     # plt.tight_layout()
#     #
#     # tscore_all = np.empty((6,27))
#     # tscore_all[:] = np.nan
#
#     ### START TASK SCORE HISTOGRAM FIGURE ###
#     # expert_tscore_hist_fig = plt.figure(figsize=(4,4))
#     # expert_tscore_hist_ax = plt.subplot(111)
#
#     # datasets = [['LF170613_1','Day201784'],['LF170222_1','Day20170615'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170420_1','Day20170719'],
#     #             ['LF170110_2','Day20170331'],['LF170214_1','Day201777'],['LF170214_1','Day2017714']]
#     #
#     # fl_all = []
#     # for ds in datasets:
#     #     print(ds[0], ds[1])
#     #     h5path = loc_info['imaging_dir'] + ds[0] + '/' + ds[0] + '.h5'
#     #     fldiff = fig_behavior_stage5(h5path, ds[1], ds[0]+ds[1], fformat, ds[0])
#     #     fl_all.append(fldiff)
#     #
#     # sns.distplot(fl_all, kde=False, ax=expert_tscore_hist_ax, bins=6, color="k", hist_kws={"linewidth": 0})
#     # expert_tscore_hist_ax.set_xlim([0,60])
#     # expert_tscore_hist_ax.set_yticks([0,1,2,3])
#     # expert_tscore_hist_ax.set_yticklabels(['0','1','2','3'])
#     #
#     # expert_tscore_hist_ax.set_xticks([0,20,40,60])
#     # expert_tscore_hist_ax.set_xticklabels(['0','20','40','60'])
#     #
#     # expert_tscore_hist_ax.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=16)
#     # expert_tscore_hist_ax.spines['right'].set_visible(False)
#     # expert_tscore_hist_ax.spines['top'].set_visible(False)
#     #
#     # expert_tscore_hist_ax.spines['left'].set_linewidth(2)
#     # expert_tscore_hist_ax.spines['bottom'].set_linewidth(2)
#     #
#     # fname = loc_info['figure_output_path'] + 'stage5_expert_histo' + '.' + fformat
#     # expert_tscore_hist_fig.savefig(fname, format=fformat, dpi=300)
#
#     ### END TASK SCORE HISTOGRAM FIGURE ###
#
#     ### TASK SCORE SCATTER PLOT ###
#
#     tscore_fig = plt.figure(figsize=(4,4))
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
#         fl_nov.append(fig_behavior_stage5(h5path, ds[1], ds[0]+ds[1], fformat, ds[0]))
#         fl_exp.append(fig_behavior_stage5(h5path, ds[2], ds[0]+ds[2], fformat, ds[0]))
#
#     # sum_ax.plot(np.nanmean(tscore_all,0),lw=2,c='k')
#     #
#     # sum_ax.legend(mousenames, loc=2)
#     # fname = loc_info['figure_output_path'] + 'allmice_stage5_summary' + '.' + fformat
#     # sum_fig.savefig(fname, format=fformat)
#     #
#     # for i,t in enumerate([0,0,0,0,0,0]):
#     #     if not np.isnan(tscore_all[i,0]):
#     #         novice_expert_ax.plot([0,1],[np.nanmean(tscore_all[i,0:3]),np.nanmean(tscore_all[i,3:6])], c='k', lw=2,zorder=1)
#     # #
#     tscore_graph.scatter([0 for i in fl_nov], fl_nov, s=150, c='w',zorder=2)
#     tscore_graph.scatter([1 for i in fl_exp], fl_exp, s=150, c='k',zorder=2)
#     # novice_expert_ax.scatter([1,1,1,1,1,1],np.nanmean(tscore_all[:,3:6],1), s=150, c='k',zorder=2)
#     #
#     # fname = loc_info['figure_output_path'] + 'stage5_novice_expert_test' + '.' + fformat
#     # novice_expert_fig.savefig(fname, format=fformat)
#     fname = loc_info['figure_output_path'] + 'stage5_expert_histo' + '.' + fformat
#     tscore_fig.savefig(fname, format=fformat, dpi=300)
#
#     # novice_expert_fig = plt.figure(figsize=(4,6))
#     # novice_expert_ax = plt.subplot(111)
#
#     ### END TASK SCORE SCATTER PLOT ###
#
#     mousenames = ['LF171211_1','LF180112_2','LF180119_1','LF171212_2','LF170214_1','LF170222_1','LF170613_1']
#
    # MOUSE = 'LF161202_1'
#     # SESSION = ['Day2018315','Day2018316','Day2018317','Day2018319','Day2018320','Day2018321','Day2018322_1','Day2018322_2',
#     # 'Day2018324','Day2018326','Day2018327','Day2018328','Day2018329','Day2018330','Day201842']
    # SESSION = ['Day20170126','Day20170127','Day20170130','Day20170208','Day20170209','Day20170213','Day20170214','Day20170215','Day20170216']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#         #fl_mean_diff.append(fldiff)

    # MOUSE = 'LF161206_1'
    # # SESSION = ['Day2018315','Day2018316','Day2018317','Day2018319','Day2018320','Day2018321','Day2018322_1','Day2018322_2',
    # # 'Day2018324','Day2018326','Day2018327','Day2018328','Day2018329','Day2018330','Day201842']
    # SESSION = ['Day20170123','Day20170126','Day20170127','Day20170208','Day20170209','Day20170210','Day20170213','Day20170214','Day20170215','Day20170216']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
        #fl_mean_diff.append(fldiff)
#     #
    # MOUSE = 'LF171211_1'
    # # SESSION = ['Day2018314_1','Day2018315','Day2018316','Day2018317','Day2018319','Day2018320','Day2018321','Day2018322',
    # #     'Day2018324','Day2018326','Day2018327','Day2018328','Day2018329','Day2018330','Day201842','Day201843']
    # SESSION = ['Day2018312','Day2018314_1','Day2018316','Day2018321']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    #
    # fl_mean_diff = []
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
    #     fl_mean_diff.append(fldiff)
#     #
#     # sum_ax.plot(fl_mean_diff)
#     # tscore_all[0,0:len(SESSION)] = fl_mean_diff
#
#     MOUSE = 'LF180112_2'
#     # SESSION = ['Day2018315','Day2018316','Day2018317','Day2018319','Day2018320','Day2018321','Day2018322_1','Day2018322_2',
#     # 'Day2018324','Day2018326','Day2018327','Day2018328','Day2018329','Day2018330','Day201842']
#     SESSION = ['Day2018424']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     # #
#     # fl_mean_diff = []
#     # for s in SESSION:
#     #     print(MOUSE, s)
#     #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#     #     fl_mean_diff.append(fldiff)
#     # #
#     # sum_ax.plot(fl_mean_diff)
#     # tscore_all[1,0:len(SESSION)] = fl_mean_diff
#     #
    # MOUSE = 'LF180119_1'
    # SESSION = ['Day2018316_1','Day2018317','Day2018319','Day2018320','Day2018321','Day2018322',
    # 'Day2018324','Day2018326','Day2018327','Day2018328','Day2018329','Day2018330','Day201843']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    #
    # fl_mean_diff = []
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
    #     fl_mean_diff.append(fldiff)
#     #
#     # sum_ax.plot(fl_mean_diff)
#     # tscore_all[2,0:len(SESSION)] = fl_mean_diff
#
    MOUSE = 'LF171212_2'
    SESSION = ['Day2018218_1','Day2018317','Day2018321','Day2018324','Day2018328']
#     # # SESSION = ['Day2018219','Day2018220','Day2018317','Day2018320','Day2018321','Day2018322']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # SESSION = ['Day2018321']
#     # #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
#     #
    # fl_mean_diff = []
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
    #     fl_mean_diff.append(fldiff)
#     #
#     # sum_ax.plot(fl_mean_diff)
#     # tscore_all[3,0:len(SESSION)] = fl_mean_diff
#     #
#     #
#     #
#     # MOUSE = 'LF170214_1'
#     # SESSION = ['Day20170410',
#     # 'Day20170414','Day20170417','Day20170509','Day20170510','Day20170511','Day20170515',
#     # 'Day20170516','Day20170517','Day20170518','Day20170522','Day20170523','Day20170524',
#     # 'Day20170526','Day20170529','Day20170530','Day20170531','Day20170601','Day20170602','Day2017613','Day2017626',
#     # 'Day2017627','Day2017629','Day2017630','Day2017714','Day201773','Day201777']
#     # SESSION = ['Day20170410','Day20170414','Day20170417','Day2017714','Day201773','Day201777']
#     # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     # # #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
#     #
#     # fl_mean_diff = []
#     # for s in SESSION:
#     #     print(MOUSE, s)
#     #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#     #     fl_mean_diff.append(fldiff)
#     # sum_ax.plot(fl_mean_diff)
#     # tscore_all[4,0:len(SESSION)] = fl_mean_diff
#     # #
#     # #
#     # #
    MOUSE = 'LF170222_1'
    # SESSION = ['Day20170403','Day20170410','Day20170411','Day20170412','Day20170413','Day20170414','Day20170417',
    # 'Day20170418','Day20170420','Day20170515','Day20170516','Day20170517',
    # 'Day20170518','Day20170519','Day20170522','Day20170523','Day20170524','Day20170526',
    # 'Day20170529','Day20170530','Day20170531','Day20170601','Day20170602','Day20170612','Day20170613','Day20170615']
    # SESSION = ['Day20170403','Day20170410','Day20170411','Day20170612','Day20170613','Day20170615']
#     SESSION = ['Day201776']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
#     fl_mean_diff = []
#     for s in SESSION:
#         print(MOUSE, s)
#         fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#         fl_mean_diff.append(fldiff)
# #     # sum_ax.plot(fl_mean_diff)
#     # tscore_all[5,0:len(SESSION)] = fl_mean_diff
#
#
#
    MOUSE = 'LF170613_1'
    # # SESSION = ['Day20170717','Day20170718','Day2017719','Day20170720','Day20170721','Day20170803','Day20170804']
    SESSION = 'Day20170804'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat, MOUSE)
    #
    # fl_mean_diff = []
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
    #     fl_mean_diff.append(fldiff)
    # sum_ax.plot(fl_mean_diff)
#
#     MOUSE = 'LF170525_1'
#     SESSION = ['Day20170801','Day20170802','Day201783','Day20170804','Day20170723']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# #     # fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
# #     # #
#     fl_mean_diff = []
#     for s in SESSION:
#         print(MOUSE, s)
#         fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#         fl_mean_diff.append(fldiff)
# #     # sum_ax.plot(fl_mean_diff)
#
#     # sum_ax.plot(np.nanmean(tscore_all,0),lw=2,c='k')
#     #
#     # sum_ax.legend(mousenames, loc=2)
#     # fname = loc_info['figure_output_path'] + 'allmice_stage5_summary' + '.' + fformat
#     # sum_fig.savefig(fname, format=fformat)
#     #
#     # for i,t in enumerate([0,0,0,0,0,0]):
#     #     if not np.isnan(tscore_all[i,0]):
#     #         novice_expert_ax.plot([0,1],[np.nanmean(tscore_all[i,0:3]),np.nanmean(tscore_all[i,3:6])], c='k', lw=2,zorder=1)
#     # #
#     # novice_expert_ax.scatter([0,0,0,0,0,0],np.nanmean(tscore_all[:,0:3],1), s=150, c='w',zorder=2)
#     # novice_expert_ax.scatter([1,1,1,1,1,1],np.nanmean(tscore_all[:,3:6],1), s=150, c='k',zorder=2)
#     #
#     # fname = loc_info['figure_output_path'] + 'stage5_novice_expert_test' + '.' + fformat
#     # novice_expert_fig.savefig(fname, format=fformat)
#
#     MOUSE = 'LF171211_2'
# #     # SESSION = ['Day2018314_1','Day2018315','Day2018316','Day2018317','Day2018319','Day2018320','Day2018321','Day2018322',
# #     #     'Day2018324','Day2018326','Day2018327','Day2018328','Day2018329','Day2018330','Day201842','Day201843']
# #     # SESSION = ['Day2018314_1','Day2018315','Day2018316','Day2018330','Day201842','Day201843']
#     SESSION = ['Day2018425']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# #     #
#     fl_mean_diff = []
#     for s in SESSION:
#         print(MOUSE, s)
#         fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#         # fl_mean_diff.append(fldiff)
#
#
#     # MOUSE = project_metainfo['muscimol2_mice']
#     #
#     # for m in MOUSE:
#     #     print(m)
#     #     h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
#     #     #for s in project_metainfo['mus2_mice_stage5'][m]:
#     #     for s in project_metainfo['mus2_mice_MUSCIMOL'][m]:
#     #         # print(h5path, s)
#     #         fig_behavior_stage5(h5path, s, m+s, fformat, 'mus2')
#     #
#     # for m in MOUSE:
#     #     print(m)
#     #     h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
#     #     #for s in project_metainfo['mus2_mice_stage5'][m]:
#     #     for s in project_metainfo['mus2_mice_CONTROL'][m]:
#     #         # print(h5path, s)
#     #         fig_behavior_stage5(h5path, s, m+s, fformat, 'mus2')
#     #
#     # for m in MOUSE:
#     #     print(m)
#     #     h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
#     #     #for s in project_metainfo['mus2_mice_stage5'][m]:
#     #     for s in project_metainfo['mus2_mice_stage5'][m]:
#     #         # print(h5path, s)
#     #         fig_behavior_stage5(h5path, s, m+s, fformat, 'mus2')
#
#     # MOUSE = 'LF171016_3'
#     # SESSION = 'Day2017117'
#     # h5path = loc_info['muscimol_2_datafile'] + MOUSE + '/' + MOUSE + '.h5'
#     # fig_behavior_stage2(h5path, SESSION, MOUSE+SESSION, fformat)
#
#     MOUSE = 'LF170214_1'
#     # 'Day20170407','Day20170408','Day20170410','Day20170411','Day20170412','Day20170413','Day20170414','Day20170417',
#     SESSION = ['Day20170509','Day20170510','Day20170408','Day20170511','Day20170408','Day20170512','Day20170408','Day20170515','Day20170516','Day20170517']
#     #SESSION = ['Day20170518','Day20170519','Day20170522','Day20170523','Day20170524','Day20170525','Day20170526','Day20170529','Day20170530','Day20170531']
#     # SESSION = ['Day20170601','Day20170602','Day2017613','Day201777']
#     # SESSION = ['Day201773','Day2017627','Day2017630','Day2017626','Day2017629']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     # for s in SESSION:
#     #     fig_behavior_stage5(h5path, s, MOUSE+s, fformat)
#
#     MOUSE = 'LF170110_1'
#     SESSION = ['Day20170130','Day20170209','Day20170210','Day20170213','Day20170214','Day20170215','Day20170217','Day20170329',
#     'Day20170330','Day20170331','Day201741','Day20170404','Day20170410','Day20170412','Day20170413','Day20170414','Day20170417',
#     'Day20170418']
# #     #
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# #     # #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
# #     #
#     fl_mean_diff = []
#     for s in SESSION:
#         print(MOUSE, s)
#         fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#         fl_mean_diff.append(fldiff)
#     # sum_ax.plot(fl_mean_diff)

    MOUSE = 'LF180219_1'
    SESSION = 'Day2018424_0000'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # print(fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat))

    MOUSE = 'LF170110_2'
    SESSION = ['Day20170209_l23']
    # SESSION = ['Day20170130','Day20170209','Day20170210','Day20170213','Day20170214','Day20170215','Day20170217','Day20170329',
    # 'Day20170330','Day20170331','Day20170403','Day20170408','Day20170410','Day20170412','Day20170413','Day20170414','Day20170417',
    # 'Day20170418']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     # #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
#     # #
    # fl_mean_diff = []
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
    #     fl_mean_diff.append(fldiff)
    # sum_ax.plot(fl_mean_diff)
#
#     # MOUSE = 'LF170110_2'
#     # SESSION = ['Day20170403','Day20170405','Day20170410','Day20170411','Day20170413','Day20170414','Day20170417','Day20170418','Day20170420',
#     # 'Day20170509','Day20170512','Day20170515','Day20170516','Day20170517','Day20170518','Day20170519','Day20170522','Day20170523',
#     # 'Day20170524','Day20170525','Day20170526','Day20170529','Day20170530','Day20170531','Day20170601','Day20170602','Day20170612',
#     # 'Day20170613','Day20170615']
#     # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     # #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
#     # #
#     # fl_mean_diff = []
#     # for s in SESSION:
#     #     print(MOUSE, s)
#     #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#     #     fl_mean_diff.append(fldiff)
#     # sum_ax.plot(fl_mean_diff)
#
#     MOUSE = 'LF171211_2'
#     SESSION = 'Day201852'
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#     # fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
#
#     #
#     #
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_2'
#     SESSION = ['Day2018317']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat, MOUSE)
#     #
#     # for s in SESSION:
#     #     print(MOUSE, s)
#     #     fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#
    # MOUSE = 'LF170612_1'
    # SESSION = ['Day20170718','Day20170719','Day20170720','Day20170721','Day20170722','Day20170801','Day20170802','Day20170803','Day20170804']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # fl_mean_diff = []
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
    #     fl_mean_diff.append(fldiff)
     # sum_ax.plot(fl_mean_diff)
#
#     MOUSE = 'LF170421_2'
#     # SESSION = ['Day201776','Day20170717','Day20170718','Day20170719','Day20170720','Day20170721','Day20170722','Day20170723']
#     SESSION = ['Day201776']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# #     # #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
# #     #
#     fl_mean_diff = []
#     for s in SESSION:
#         print(MOUSE, s)
#         fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#         fl_mean_diff.append(fldiff)
#     # sum_ax.plot(fl_mean_diff)
#
    MOUSE = 'LF170420_1'
    # SESSION = ['Day20170612','Day20170718','Day20170719','Day20170720','Day20170721','Day20170722','Day20170723',
    #     'Day20170801','Day20170802','Day20170803','Day20170804']
    SESSION = 'Day2017629'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
#     #
    # fl_mean_diff = []
    # for s in SESSION:
    #     print(MOUSE, s)
    #     fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
    #     fl_mean_diff.append(fldiff)
    # sum_ax.plot(fl_mean_diff)
#
#     MOUSE = 'LF170413_1'
#     SESSION = ['Day20170614','Day2017629','Day2017713','Day2017714','Day201777']
#     h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# #     # #fig_behavior_stage5(h5path, SESSION, MOUSE+SESSION, fformat)
# #     #
#     fl_mean_diff = []
#     for s in SESSION:
#         print(MOUSE, s)
#         fldiff = fig_behavior_stage5(h5path, s, MOUSE+s, fformat, MOUSE)
#         fl_mean_diff.append(fldiff)
#     # sum_ax.plot(fl_mean_diff)
