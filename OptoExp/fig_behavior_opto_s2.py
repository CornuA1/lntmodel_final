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
from select_trials import select_trials
# from smi import smi
from smzscore import smzscore as smi


def fig_behavior_stage2(h5path, sess, fname, fformat='png', subfolder=[]):
    # load data
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # calculate SMI
    raw_short = select_trials(raw_ds, 'raw', 0)
    licks_short = select_trials(raw_ds, 'licks', 0)
    reward_ds[:,3] = reward_ds[:,3]
    reward_ds[:,2] = 3

    sr = np.size(np.unique(reward_ds[reward_ds[:,5]==1,3])) / np.size(np.unique(reward_ds[:,3]))

    smi_d, shuffled_sr = smi( raw_ds, licks_ds, reward_ds, sr, [320,340], 340, 10, 340, 0)

    # create figure to later plot on
    fig = plt.figure(figsize=(12,6))
    fig.suptitle(fname)
    ax1 = plt.subplot2grid((8,2),(0,0), rowspan=6)
    ax2 = plt.subplot2grid((8,2),(6,0), rowspan=2)
    ax3 = plt.subplot2grid((8,2),(0,1), rowspan=6)
    ax4 = plt.subplot2grid((8,2),(6,1), rowspan=2)

    ax1.set_xlim([50,340])
    ax1.set_ylabel('Trial #')
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Short trials')

    ax2.set_xlim([10,67])
    ax2.set_ylabel('Speed (cm/sec)')
    ax2.set_xlabel('Location (cm)')

    ax3.set_xlim([50,400])
    ax3.set_ylabel('Trial #')
    ax3.set_xlabel('Location (cm)')
    ax3.set_title('Short trials')

    ax4.set_xlim([10,79])
    ax4.set_ylabel('Speed (cm/sec)')
    ax4.set_xlabel('Location (cm)')

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color='#D2F2FF',zorder=0)

    # make array of y-axis locations for licks. If clause to check for empty arrays
    if np.size(licks_ds) > 0 or np.size(reward_ds) > 0:
        # only plot trials where either a lick and/or a reward were detected
        # therefore: pull out trial numbers from licks and rewards dataset and map to
        # a new list of rows used for plotting
        short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
        # get trial numbers to be plotted
        lick_trials = np.unique(licks_ds[:,2])
        reward_trials = np.unique(reward_ds[:,3])
        scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
        scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
        scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))

        ax1.set_ylim([0,len(np.unique(scatter_rowlist_short))])


        # scatterplot of licks/rewards in order of trial number
        for i,r in enumerate(scatter_rowlist_map_short):
            plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
            plot_rewards_x = reward_ds[reward_ds[:,3]==r,1]
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            if reward_ds[reward_ds[:,3]==r,5] == 1:
                col = '#00C40E'
            else:
                col = 'r'

            # if reward location is recorded at beginning of track, set it to end of track
            if plot_rewards_x < 300:
                plot_rewards_x= 338

            # plot licks and rewards
            if np.size(plot_licks_x) > 0:
                plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_short[i])
                ax1.scatter(plot_licks_x, plot_licks_y,c=sns.xkcd_rgb["windows blue"],lw=0)
            if np.size(plot_rewards_x) > 0:
                plot_rewards_y = scatter_rowlist_short[i]
                ax1.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
            if np.size(cur_trial_start) > 0:
                plot_starts_y = scatter_rowlist_short[i]
                ax1.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)

        # plot running speed
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
        ax2.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='g',zorder=3)
        ax2.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color='g',alpha=0.2)

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

    return smi_d, len(np.unique(scatter_rowlist_short))

def summary_plot(smi_days, trialnr_days, sessions, fname, fformat='png', subfolder=[]):
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    #print(np.arange(0,len(smi_days),1))
    ax1.set_xticks(np.arange(0,len(smi_days),1))
    ax1.set_xticklabels(sessions, rotation=45)
    ax2.set_xticks(np.arange(0,len(smi_days),1))
    ax2.set_xticklabels(sessions, rotation=45)
    #print(smi_days)
    ax1.plot(smi_days)
    ax2.plot(trialnr_days)
    ax1.set_ylabel('spatial modulation index')
    ax1.set_xlabel('Session')

    fname = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE+s+'summary' + '.' + fformat
    fig.savefig(fname, format=fformat)

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'

    MOUSE = 'LF180515_1'
    SESSION = ['Day201885']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    for s in SESSION:
        fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)

    # MOUSE = 'LF180514_1'
    # SESSION = ['Day2018726','Day2018727','Day2018728','Day2018729','Day2018730','Day201881','Day201883','Day201884']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for s in SESSION:
    #     print(s)
    #     fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
