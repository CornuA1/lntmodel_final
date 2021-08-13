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
plt.rcParams['svg.fonttype'] = 'none'
from scipy import stats
import seaborn as sns
# import ipdb
import scipy.io as sio

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
from scipy.signal import butter, filtfilt

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

def load_data_h5(h5path, sess):
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()
    return raw_ds, licks_ds, reward_ds

def load_raw_data(raw_filename, sess):
#    raw_data = load_data(raw_filename, 'vr')
    raw_data = np.genfromtxt(raw_filename, delimiter=';')
    all_licks = licks(raw_data)
    trial_licks = all_licks[np.in1d(all_licks[:, 3], [3, 4]), :]
    reward =  rewards(raw_data)
    return raw_data, trial_licks, reward

def load_eye_data(sess):
    processed_data_path = loc_info['raw_dir'] + sess[2] + os.sep + sess[0] + os.sep + sess[1]
    loaded_data = sio.loadmat(processed_data_path)
    raw_data = loaded_data['behaviour_aligned']

    pupil_area = loaded_data['eye_data_aligned']
    pupil_x = loaded_data['eye_x_aligned']
    pupil_x = pupil_x.T - np.mean(pupil_x.T)
    pupil_y = loaded_data['eye_y_aligned']
    pupil_y = pupil_y.T - np.mean(pupil_y.T)
    all_licks = licks(raw_data)
    if len(all_licks) > 0:
        trial_licks = all_licks[np.in1d(all_licks[:, 3], [3, 4]), :]
    else:
        trial_licks = np.empty((0,6))
    reward =  rewards(raw_data, 6, False)
    return raw_data, trial_licks, reward, pupil_area, pupil_x, pupil_y


def fig_behavior_stage5(h5path, sess, fname, fformat='png', subfolder=[], load_raw=False):
    # load data

    # h5dat = h5py.File(h5path, 'r')
    # raw_ds = np.copy(h5dat[sess + '/raw_data'])
    # licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    # reward_ds = np.copy(h5dat[sess + '/rewards'])
    # h5dat.close()


    pupil_data = False
    if load_raw == False:
        raw_ds, licks_ds, reward_ds = load_data_h5(h5path, sess)
    elif len(sess) == 2:
        # ipdb.set_trace()
        raw_filename = h5path + os.sep + sess[0] + os.sep + sess[1]
        raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sess)
    elif len(sess) == 3:
        raw_ds, licks_ds, reward_ds, pupil_area, pupil_x, pupil_y = load_eye_data(sess)
        pupil_data = True

    # create figure to later plot on
    fig = plt.figure(figsize=(14,8))
    fig.suptitle(fname)
    ax1 = plt.subplot2grid((10,10),(0,0), rowspan=6, colspan=5)
    ax2 = plt.subplot2grid((10,10),(6,0), rowspan=2, colspan=5)
    ax3 = plt.subplot2grid((10,10),(0,5), rowspan=6, colspan=5)
    ax4 = plt.subplot2grid((10,10),(6,5), rowspan=2, colspan=5)
    ax5 = plt.subplot2grid((10,10),(8,0), rowspan=2, colspan=5)
    ax6 = plt.subplot2grid((10,10),(8,5), rowspan=2, colspan=2)

    ax1.set_xlim([50,360])
    ax1.set_ylabel('Trial #')
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Short trials')

    #ax2.set_xlim([10,67])
    #ax2.set_ylabel('Speed (cm/sec)')
    #ax2.set_xlabel('Location (cm)')

    ax3.set_xlim([50,440])
    ax3.set_ylabel('Trial #')
    ax3.set_xlabel('Location (cm)')
    ax3.set_title('Short trials')

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,360,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(380,440,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=9)

    fl_diff = 0
    t_score = 0

    # make array of y-axis locations for licks. If clause to check for empty arrays
    if np.size(licks_ds) > 0 or np.size(reward_ds) > 0:
        # only plot trials where either a lick and/or a reward were detected
        # therefore: pull out trial numbers from licks and rewards dataset and map to
        # a new list of rows used for plotting
        short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
        long_trials = filter_trials( raw_ds, [], ['tracknumber',4])

        # short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],short_trials)
        # long_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],long_trials)


        # get trial numbers to be plotted
        # ipdb.set_trace()
        reward_ds[:,3] = reward_ds[:,3]
        lick_trials = np.unique(licks_ds[:,2])
        reward_trials = np.unique(reward_ds[:,3])
        scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
        scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
        scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
        scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
        scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

        # ax1.set_ylim([0,len(np.unique(scatter_rowlist_short))])

        # scatterplot of licks/rewards in order of trial number
        for i,r in enumerate(scatter_rowlist_map_short):
            plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
            plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
            # print(plot_rewards_x)
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            if reward_ds[reward_ds[:,3]==r,5] == 1:
                col = '#00C40E'
            else:
                col = 'r'

            # if reward location is recorded at beginning of track, set it to end of track
            if plot_rewards_x < 300:
                plot_rewards_x = 360

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
            plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            if reward_ds[reward_ds[:,3]==r,5] == 1:
                col = '#00C40E'
            else:
                col = 'r'

            # if reward location is recorded at beginning of track, set it to end of track
            if plot_rewards_x < 300:
                plot_rewards_x = 440

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


        if pupil_data:
            # plot pupil data
            bin_size = 5
            binnr_short = 360/bin_size
            mean_pupil_A = np.empty((np.size(scatter_rowlist_map_short,0),int(binnr_short)))
            mean_pupil_A[:] = np.NAN
            mean_pupil_x = np.empty((np.size(scatter_rowlist_map_short,0),int(binnr_short)))
            mean_pupil_x[:] = np.NAN
            mean_pupil_y = np.empty((np.size(scatter_rowlist_map_short,0),int(binnr_short)))
            mean_pupil_y[:] = np.NAN
            max_y_short = 0

            # ipdb.set_trace()
            for i,t in enumerate(scatter_rowlist_map_short):
                cur_trial = raw_ds[raw_ds[:,6]==t,:]
                cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
                cur_trial_start = raw_ds[raw_ds[:,6]==t,1][0]
                cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)

                if cur_trial_bins-cur_trial_start_bin > 0:
                    mean_pupil_A_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_area[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
                    mean_pupil_A[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_A_trial

                    mean_pupil_x_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_x[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
                    mean_pupil_x[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_x_trial

                    mean_pupil_y_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_y[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
                    mean_pupil_y[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_y_trial
                    #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_pupil_A_trial,c='0.8')
                    max_y_short = np.amax([max_y_short,np.amax(mean_pupil_A_trial)])


            sem_speed = stats.sem(mean_pupil_A,0,nan_policy='omit')
            mean_pupil_A_sess_short = np.nanmean(mean_pupil_A,0)
            binnr_short = int(binnr_short)
            ax5.plot(np.linspace(0,binnr_short-1,binnr_short),mean_pupil_A_sess_short,c=sns.xkcd_rgb["windows blue"],zorder=3)
            ax5.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_pupil_A_sess_short-sem_speed, mean_pupil_A_sess_short+sem_speed, color=sns.xkcd_rgb["windows blue"],alpha=0.2)

            sem_speed = stats.sem(mean_pupil_x,0,nan_policy='omit')
            mean_pupil_x_sess_short = np.nanmean(mean_pupil_x,0)
            ax6.plot(np.linspace(0,binnr_short-1,binnr_short),mean_pupil_x_sess_short,c='g',zorder=3)
            ax6.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_pupil_x_sess_short-sem_speed, mean_pupil_x_sess_short+sem_speed, color='g',alpha=0.2)

            sem_speed = stats.sem(mean_pupil_y,0,nan_policy='omit')
            mean_pupil_y_sess_short = np.nanmean(mean_pupil_y,0)
            ax6.plot(np.linspace(0,binnr_short-1,binnr_short),mean_pupil_y_sess_short,c='r',zorder=3)
            ax6.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_pupil_y_sess_short-sem_speed, mean_pupil_y_sess_short+sem_speed, color='r',alpha=0.2)

            ax5.axvspan(40,48, color='0.9')
            ax6.axvspan(40,48, color='0.9')
            ax5.set_ylim([900,1400])
            ax6.set_ylim([-2,2])

        # plot running speed
        plot_running_speed = True
        if plot_running_speed:
            bin_size = 5
            binnr_short = 360/bin_size
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
            binnr_short = int(binnr_short)
            ax2.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c=sns.xkcd_rgb["windows blue"],zorder=3)
            ax2.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color=sns.xkcd_rgb["windows blue"],alpha=0.2)

            # plot running speed
            binnr_long = 440/bin_size
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
            binnr_long = int(binnr_long)
            ax2.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c=sns.xkcd_rgb["dusty purple"],zorder=3)
            ax2.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color=sns.xkcd_rgb["dusty purple"],alpha=0.2)

            ax2.set_ylim([0,100])
            # ax6.set_ylim([30,60])

        # plot location of first trials on short and long trials
        first_lick_short = []
        first_lick_short_trials = []
        first_lick_long = []
        first_lick_long_trials = []
        all_licks_short = []
        all_licks_long = []
        for r in scatter_rowlist_map:
            licks_all = licks_ds[licks_ds[:,2]==r,:]
            licks_all = licks_all[licks_all[:,1]>150,:]
            if not licks_all.size == 0:
                licks_all = licks_all[licks_all[:,1]>240,:]
            else:
                rew_lick = reward_ds[reward_ds[:,3]==r,:]
                if rew_lick.size > 0:
                    if rew_lick[0][5] == 1.0:
                        licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
#                        if r%20 <= 10:
#                            licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 3]])
#                        else:
#                            licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 4]])
            if licks_all.shape[0]>0:
                lick = licks_all[0]
                if lick[3] == 3:
                    first_lick_short.append(lick[1])
                    first_lick_short_trials.append(r)
                    all_licks_short.extend(licks_all[:,1])
                elif lick[3] == 4:
                    first_lick_long.append(lick[1])
                    first_lick_long_trials.append(r)
                    all_licks_long.extend(licks_all[:,1])
        ax4.scatter(first_lick_short,first_lick_short_trials,c=sns.xkcd_rgb["windows blue"],lw=0)
        ax4.scatter(first_lick_long,first_lick_long_trials,c=sns.xkcd_rgb["dusty purple"],lw=0)
        # ax4.axvline(np.median(first_lick_short), c=sns.xkcd_rgb["windows blue"], lw=2)
        # ax4.axvline(np.median(first_lick_long), c=sns.xkcd_rgb["dusty purple"], lw=2)
        
        sns.kdeplot(first_lick_short,shade=True,color='#FF8000',ax=ax5)
        sns.kdeplot(first_lick_long,shade=True,color='#0025D0',ax=ax5)
        
        sns.distplot(first_lick_short, kde_kws={'cumulative': True},color='#EC008C',ax=ax6)
        sns.distplot(first_lick_long, kde_kws={'cumulative': True},color='#ED7EC6',ax=ax6)
        
        sns.rugplot(first_lick_short, height=0.1,color='#EC008C',ax=ax6)
        sns.rugplot(first_lick_long, height=0.1,color='#ED7EC6',ax=ax6)
        ax5.set_xlim([240,450])
        ax6.set_ylim([0,1.1])
        ax6.set_xlim([240,400])
        ax6.set_xticks([240,320,380])
        ax6.set_xticklabels(['240','320','380'])
        sns.despine(top=True, right=True, left=False, bottom=False)
        ax6.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=14, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')
        

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

            # sns.distplot(bootstrap_diff,ax=ax2)
            # vl_handle = ax2.axvline(np.mean(bootstrap_diff),c='b')
            # vl_handle.set_label('z-score = ' + str(fl_diff))
            # ax2.legend()



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

#        ax4.set_xlim([240,380])

        if np.nanmedian(first_lick_long)+bt_CI_5_long > np.nanmedian(first_lick_short) or np.nanmedian(first_lick_short)+bt_CI_95_short < np.nanmedian(first_lick_long):
            print('significant!')

        if np.size(first_lick_short) > 0 and np.size(first_lick_long) > 0:
            t_score = np.median(first_lick_long) - np.median(first_lick_short)
            ax4.text(320, 10, 't-score: '+str(t_score))

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
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    print(fname)
    return t_score

def filter_pupil_data(raw_ds, pupil_area, pupil_x, pupil_y):
    # filter data (remove times the animal is stationary and when we see outliers in the pupil area,xy data as these are likely not accurate measurements)
    order = 6
    fs = int(np.size(raw_ds,0)/raw_ds[-1,0])       # sample rate, Hz
    cutoff = 2 # desired cutoff frequency of the filter, Hz
    pupildata_rejection_threshold = 4

    # filter speed
    # speed_filtered = butter_lowpass_filter(raw_ds[:,3], cutoff, fs, order)
    # pupil_area = pupil_area[speed_filtered>1]
    # raw_ds = raw_ds[speed_filtered>1,:]

    # remove frames where pupil data is outside our standard deviation-based threshold
    pupil_area_std = np.std(pupil_area)
    pupil_area_mean = np.mean(pupil_area)

    pupil_x_std = np.std(pupil_x)
    pupil_x_mean = np.mean(pupil_x)

    pupil_y_std = np.std(pupil_y)
    pupil_y_mean = np.mean(pupil_y)


    # ipdb.set_trace()
    # filter instances where pupil data falls outside threshold
    raw_ds = raw_ds[np.where((pupil_area > pupil_area_mean-pupil_area_std*pupildata_rejection_threshold) & (pupil_area < pupil_area_mean+pupil_area_std*pupildata_rejection_threshold))[0],:]
    pupil_x = pupil_x[np.where((pupil_area > pupil_area_mean-pupil_area_std*pupildata_rejection_threshold) & (pupil_area < pupil_area_mean+pupil_area_std*pupildata_rejection_threshold))[0]]
    pupil_y = pupil_y[np.where((pupil_area > pupil_area_mean-pupil_area_std*pupildata_rejection_threshold) & (pupil_area < pupil_area_mean+pupil_area_std*pupildata_rejection_threshold))[0]]
    pupil_area = pupil_area[np.where((pupil_area > pupil_area_mean-pupil_area_std*pupildata_rejection_threshold) & (pupil_area < pupil_area_mean+pupil_area_std*pupildata_rejection_threshold))[0]]

    raw_ds = raw_ds[np.where((pupil_x > pupil_x_mean-pupil_x_std*pupildata_rejection_threshold) & (pupil_x < pupil_x_mean+pupil_x_std*pupildata_rejection_threshold))[0],:]
    pupil_area = pupil_area[np.where((pupil_x > pupil_x_mean-pupil_x_std*pupildata_rejection_threshold) & (pupil_x < pupil_x_mean+pupil_x_std*pupildata_rejection_threshold))[0]]
    pupil_y = pupil_y[np.where((pupil_x > pupil_x_mean-pupil_x_std*pupildata_rejection_threshold) & (pupil_x < pupil_x_mean+pupil_x_std*pupildata_rejection_threshold))[0]]
    pupil_x = pupil_x[np.where((pupil_x > pupil_x_mean-pupil_x_std*pupildata_rejection_threshold) & (pupil_x < pupil_x_mean+pupil_x_std*pupildata_rejection_threshold))[0]]

    raw_ds = raw_ds[np.where((pupil_x > pupil_y_mean-pupil_y_std*pupildata_rejection_threshold) & (pupil_x < pupil_y_mean+pupil_y_std*pupildata_rejection_threshold))[0],:]
    pupil_area = pupil_area[np.where((pupil_x > pupil_y_mean-pupil_y_std*pupildata_rejection_threshold) & (pupil_x < pupil_y_mean+pupil_y_std*pupildata_rejection_threshold))[0]]
    pupil_x = pupil_x[np.where((pupil_x > pupil_y_mean-pupil_y_std*pupildata_rejection_threshold) & (pupil_x < pupil_y_mean+pupil_y_std*pupildata_rejection_threshold))[0]]
    pupil_y = pupil_y[np.where((pupil_x > pupil_y_mean-pupil_y_std*pupildata_rejection_threshold) & (pupil_x < pupil_y_mean+pupil_y_std*pupildata_rejection_threshold))[0]]

    return raw_ds, pupil_area, pupil_x, pupil_y

def plot_pupil_data(raw_ds, pupil_area, pupil_x, pupil_y, axes1, axes2, axes3):
    # axes1.scatter(raw_ds[:,1],pupil_area)

    short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
    long_trials = filter_trials( raw_ds, [], ['tracknumber',4])

    # plot pupil data
    bin_size = 5
    binnr_short = 360/bin_size
    mean_pupil_A_short = np.full((np.size(short_trials,0),int(binnr_short)),np.nan)
    mean_pupil_x_short = np.full((np.size(short_trials,0),int(binnr_short)),np.nan)
    mean_pupil_y_short = np.full((np.size(short_trials,0),int(binnr_short)),np.nan)
    # ipdb.set_trace()
    for i,t in enumerate(short_trials):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_pupil_A = pupil_area[raw_ds[:,6]==t,:]

        cur_trial_bins = np.round(cur_trial[-1,1]/bin_size,0)
        cur_trial_start = cur_trial[0,1]
        cur_trial_start_bin = np.round(cur_trial_start/bin_size,0)

        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_pupil_A_short_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_area[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_pupil_A_short[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_A_short_trial
            # axes2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_pupil_A_short_trial,c='0.8',alpha=0.5)

            mean_pupil_x_short_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_x[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_pupil_x_short[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_x_short_trial
            # axes3.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_pupil_x_short_trial,c='r',alpha=0.5)

            mean_pupil_y_short_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_y[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_pupil_y_short[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_y_short_trial
            # axes3.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_pupil_y_short_trial,c='g',alpha=0.5)


    bin_size = 5
    binnr_long = 440/bin_size
    mean_pupil_A_long = np.full((np.size(long_trials,0),int(binnr_long)),np.nan)
    mean_pupil_x_long = np.full((np.size(long_trials,0),int(binnr_long)),np.nan)
    mean_pupil_y_long = np.full((np.size(long_trials,0),int(binnr_long)),np.nan)
    # ipdb.set_trace()
    for i,t in enumerate(long_trials):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_pupil_A = pupil_area[raw_ds[:,6]==t,:]

        cur_trial_bins = np.round(cur_trial[-1,1]/bin_size,0)
        cur_trial_start = cur_trial[0,1]
        cur_trial_start_bin = np.round(cur_trial_start/bin_size,0)

        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_pupil_A_long_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_area[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_pupil_A_long[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_A_long_trial
            # axes2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_pupil_A_long_trial,c='0.8',alpha=0.5)

            mean_pupil_x_long_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_x[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_pupil_x_long[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_x_long_trial
            # axes3.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_pupil_x_long_trial,c='r',alpha=0.5)

            mean_pupil_y_long_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], pupil_y[raw_ds[:,6]==t,0], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_pupil_y_long[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_pupil_y_long_trial
            # axes3.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_pupil_y_long_trial,c='g',alpha=0.5)

    sem_pupil_A = stats.sem(mean_pupil_A_short,0,nan_policy='omit')
    mean_pupil_A_sess_short = np.nanmean(mean_pupil_A_short,0)
    axes2.plot(np.linspace(0,binnr_short-1,binnr_short),mean_pupil_A_sess_short,SHORT_COLOR,lw=2,zorder=3)
    axes2.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_pupil_A_sess_short-sem_pupil_A, mean_pupil_A_sess_short+sem_pupil_A, color=SHORT_COLOR,alpha=0.2, linewidth=0)
    #
    sem_pupil_x = stats.sem(mean_pupil_x_short,0,nan_policy='omit')
    mean_pupil_x_sess_short = np.nanmean(mean_pupil_x_short,0)
    axes3.plot(np.linspace(0,binnr_short-1,binnr_short),mean_pupil_x_sess_short,c='g',zorder=3)
    axes3.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_pupil_x_sess_short-sem_pupil_x, mean_pupil_x_sess_short+sem_pupil_x, color='g',alpha=0.2, linewidth=0)
    #
    sem_pupil_y = stats.sem(mean_pupil_y_short,0,nan_policy='omit')
    mean_pupil_y_sess_short = np.nanmean(mean_pupil_y_short,0)
    axes3.plot(np.linspace(0,binnr_short-1,binnr_short),mean_pupil_y_sess_short,c='r',zorder=3)
    axes3.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_pupil_y_sess_short-sem_pupil_y, mean_pupil_y_sess_short+sem_pupil_y, color='r',alpha=0.2, linewidth=0)

    sem_pupil_A = stats.sem(mean_pupil_A_long,0,nan_policy='omit')
    mean_pupil_A_sess_long = np.nanmean(mean_pupil_A_long,0)
    axes2.plot(np.linspace(0,binnr_long-1,binnr_long),mean_pupil_A_sess_long,LONG_COLOR,lw=2,zorder=3)
    axes2.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_pupil_A_sess_long-sem_pupil_A, mean_pupil_A_sess_long+sem_pupil_A, color=LONG_COLOR,alpha=0.2, linewidth=0)
    #
    sem_pupil_x = stats.sem(mean_pupil_x_long,0,nan_policy='omit')
    mean_pupil_x_sess_long = np.nanmean(mean_pupil_x_long,0)
    axes3.plot(np.linspace(0,binnr_long-1,binnr_long),mean_pupil_x_sess_long,c='g',zorder=3)
    axes3.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_pupil_x_sess_long-sem_pupil_x, mean_pupil_x_sess_long+sem_pupil_x, color='g',alpha=0.2, linewidth=0)
    #
    sem_pupil_y = stats.sem(mean_pupil_y_long,0,nan_policy='omit')
    mean_pupil_y_sess_long = np.nanmean(mean_pupil_y_long,0)
    axes3.plot(np.linspace(0,binnr_long-1,binnr_long),mean_pupil_y_sess_long,c='r',zorder=3)
    axes3.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_pupil_y_sess_long-sem_pupil_y, mean_pupil_y_sess_long+sem_pupil_y, color='r',alpha=0.2, linewidth=0)

    prelm_A_short = np.nanmean(pupil_area[np.where((raw_ds[:,1] > 100) & (raw_ds[:,1] < 200))[0]])
    lm_A_short = np.nanmean(pupil_area[np.where((raw_ds[:,1] > 200) & (raw_ds[:,1] < 240))[0]])
    postlm_A_short = np.nanmean(pupil_area[np.where((raw_ds[:,1] > 240) & (raw_ds[:,1] < 320))[0]])

    axes2.set_xlim([10,80])
    axes3.set_xlim([10,80])
    axes2.set_ylim([800,1600])

    axes2.spines['bottom'].set_linewidth(2)
    axes2.spines['top'].set_visible(False)
    axes2.spines['right'].set_visible(False)
    axes2.spines['left'].set_linewidth(2)
    axes2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    axes2.set_xticks([20,44,64,76])
    axes2.set_xticklabels(['100', '220', '320', '380'], fontsize=12)
    axes2.axvline(44,lw=3,c='r', ls='--')
    axes2.axvline(64,lw=3,c=SHORT_COLOR,ls='--')
    axes2.axvline(76,lw=3,c=LONG_COLOR,ls='--')


    return axes1, axes2, axes3, [prelm_A_short,lm_A_short,postlm_A_short]

def pupil_data_summary():
    fname = 'pupil_data'
    subfolder = 'pupil_data'

    datasets = [['LF191022_3', '20191204','20191204_ol'], ['LF191023_blue', '20191204', '20191204_ol'], ['LF191024_1', '20191204', '20191204_ol']]
    # datasets = [ ['LF191024_1', '20191204', '20191204_ol']]

    # plot pupil data
    bin_size = 5
    binnr_short = 360/bin_size
    binnr_long = 440/bin_size

    # collect results
    pupil_A_short = []
    pupil_A_short_ol = []

    for ds in datasets:
        # create figure to later plot on
        fig = plt.figure(figsize=(10,8))

        ax1 = plt.subplot2grid((12,20),(0,0), rowspan=5, colspan=5)
        ax2 = plt.subplot2grid((12,20),(0,5), rowspan=5, colspan=5)
        ax3 = plt.subplot2grid((12,20),(5,0), rowspan=3, colspan=10)
        ax4 = plt.subplot2grid((12,20),(8,0), rowspan=3, colspan=10)
        ax5 = plt.subplot2grid((12,20),(0,10), rowspan=5, colspan=5)
        ax6 = plt.subplot2grid((12,20),(0,15), rowspan=5, colspan=5)
        ax7 = plt.subplot2grid((12,20),(5,10), rowspan=3, colspan=10)
        ax8 = plt.subplot2grid((12,20),(8,10), rowspan=3, colspan=10)


        raw_ds, licks_ds, reward_ds, pupil_area, pupil_x, pupil_y = load_eye_data([ds[1],'aligned_eyedata.mat',ds[0]])
        raw_ds_ol, licks_ds_ol, reward_ds_ol, pupil_area_ol, pupil_x_ol, pupil_y_ol = load_eye_data([ds[2],'aligned_eyedata.mat',ds[0]])

        raw_ds, pupil_area, pupil_x, pupil_y = filter_pupil_data(raw_ds, pupil_area, pupil_x, pupil_y)
        raw_ds_ol, pupil_area_ol, pupil_x_ol, pupil_y_ol = filter_pupil_data(raw_ds_ol, pupil_area_ol, pupil_x_ol, pupil_y_ol)

        ax1, ax3, ax4, short_data = plot_pupil_data(raw_ds, pupil_area, pupil_x, pupil_y, ax1, ax3, ax4)
        ax5, ax7, ax8, short_data_ol  = plot_pupil_data(raw_ds_ol, pupil_area_ol, pupil_x_ol, pupil_y_ol, ax5, ax7, ax8)

        pupil_A_short.append(short_data)
        pupil_A_short_ol.append(short_data_ol)

        fig.tight_layout()

        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname_save = loc_info['figure_output_path'] + subfolder + os.sep + fname + '_'  + ds[0] + '.' + fformat

        fig.savefig(fname_save, format=fformat)
        plt.close()
        print(fname_save)

    # plot summar data
    fig = plt.figure(figsize=(6,5))
    ax1 = plt.subplot(111)

    pupil_A_short_prelm = [pAs[0] for pAs in pupil_A_short]
    pupil_A_short_lm = [pAs[1] for pAs in pupil_A_short]
    pupil_A_short_postlm = [pAs[2] for pAs in pupil_A_short]

    pupil_A_short_prelm_ol = [pAs[0] for pAs in pupil_A_short_ol]
    pupil_A_short_lm_ol = [pAs[1] for pAs in pupil_A_short_ol]
    pupil_A_short_postlm_ol = [pAs[2] for pAs in pupil_A_short_ol]

    print(stats.f_oneway(np.array(pupil_A_short_prelm),np.array(pupil_A_short_lm),np.array(pupil_A_short_postlm),np.array(pupil_A_short_prelm_ol),np.array(pupil_A_short_lm_ol),np.array(pupil_A_short_postlm_ol)))

    for pAs in pupil_A_short:
        ax1.scatter([0,1,2], pAs, c='k', linewidths=0, s=80,zorder=2)
        ax1.plot([0,1,2], pAs, c='k', lw=2, zorder=1)

    for pAs in pupil_A_short_ol:
        ax1.scatter([0,1,2], pAs, c='0.8', linewidths=0, s=80,zorder=2)
        ax1.plot([0,1,2], pAs, c='0.8', lw=2, zorder=1)

    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=2, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax1.set_xlim([-0.5,2.5])
    ax1.set_ylabel('Pupil Area (a.u.)', fontsize=16)
    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(['Pre-landmark', 'Landmark', 'Post-landmark'], rotation=45, fontsize=14)

    fig.tight_layout()

    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname_save = loc_info['figure_output_path'] + subfolder + os.sep + 'pupil_data_summary.' + fformat
    fig.savefig(fname_save, format=fformat)
    plt.close()
    print(fname_save)
    pass

def run_LF191022_24(numbers):
    if 0 in numbers:
        MOUSE = 'LF191022_1'
        SESSION = [['20191203','MTH3_vr3_s5r2_2019123_1713.csv']]
        # SESSION = [['20191204','MTH3_vr1_s5r2_2019124_1947.csv']]
        # SESSION = [['20191205','MTH3_vr1_s5r2_2019125_2023.csv']]
        data_path = loc_info['raw_dir'] + MOUSE
        for s in SESSION:
            print(data_path)
            fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)
    if 1 in numbers:
        MOUSE = 'LF191022_2'
        SESSION = [['20191203','MTH3_vr1_s5r2_2019123_1744.csv']]
        SESSION = [['20191204','MTH3_vr1_s5r2_2019124_2150.csv']]
        data_path = loc_info['raw_dir'] + MOUSE
        for s in SESSION:
            fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)
    if 2 in numbers:
        MOUSE = 'LF191022_3'
        SESSION = [['20191203','MTH3_vr1_s5r2_2019123_1849.csv']]
        SESSION = [['20191204','MTH3_vr1_s5r2_2019124_2317.csv']]
        SESSION = [['20191204','aligned_eyedata.mat', MOUSE]]
        SESSION = [['20191204_ol','aligned_eyedata.mat', MOUSE]]
        data_path = loc_info['raw_dir'] + MOUSE
        for s in SESSION:
            fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)
    if 3 in numbers:
        MOUSE = 'LF191023_blank'
        SESSION = [['20191203','MTH3_vr4_s5r2_2019123_1717.csv']]
        SESSION = [['20191205','MTH3_vr1_s5r2_2019125_1727.csv']]
        SESSION = [['20191203','MTH3_vr4_s5r2_2019123_1717.csv'],['20191205','MTH3_vr1_s5r2_2019125_1727.csv'],['20191206','MTH3_vr1_s5r2_2019126_1831.csv']]
        SESSION = [['20191114','MTH3_vr_s5r_20191114_225.csv'],['20191210','MTH3_vr1_s5r2_20191210_214.csv']]
        data_path = loc_info['raw_dir'] + MOUSE
        for s in SESSION:
            fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)
    if 4 in numbers:
        MOUSE = 'LF191023_blue'
        SESSION = [['20191203','MTH3_vr4_s5r2_2019123_1823.csv']]
        SESSION = [['20191204','MTH3_vr1_s5r2_2019124_1834.csv']]
        SESSION = [['20191204','aligned_eyedata.mat', MOUSE]]
        SESSION = [['20191204_ol','aligned_eyedata.mat', MOUSE]]
        data_path = loc_info['raw_dir'] + MOUSE
        for s in SESSION:
            fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)
    if 5 in numbers:
        MOUSE = 'LF191024_1'
        SESSION = [['20191203','MTH3_vr3_s5r2_2019123_1824.csv']]
        SESSION = [['20191115','MTH3_vr1_s5r_20191115_2115.csv']]
        SESSION = [['20191204','MTH3_vr1_s5r2_2019125_016.csv']]
        SESSION = [['20191204','aligned_eyedata.mat', MOUSE]]
        # SESSION = [['20191204_ol','aligned_eyedata.mat', MOUSE]]
        data_path = loc_info['raw_dir'] + MOUSE
        for s in SESSION:
            fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)
            
if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline
#
    fformat = 'svg'

    # pupil_data_summary()

    run_LF191022_24([0])
    # run_LF191022_24([5])
