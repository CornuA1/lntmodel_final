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
    print(h5path)
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()


    # calculate SMI
    raw_short = select_trials(raw_ds, 'raw', 0)
    licks_short = select_trials(raw_ds, 'licks', 0)
    reward_ds[:,3] = reward_ds[:,3] - 1
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

    # MOUSE = 'LF180515_1'
    # SESSION = 'Day2018726'
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # fig_behavior_stage2(h5path, SESSION, MOUSE+SESSION, fformat, MOUSE)

    MOUSE = 'LF180514_1'
    SESSION = 'Day2018726'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    fig_behavior_stage2(h5path, SESSION, MOUSE+SESSION, fformat, MOUSE)

    # MOUSE = 'LF171016_2'
    # SESSION = 'Day20171114'
    # h5path = loc_info['muscimol_2_datafile'] + MOUSE + '/' + MOUSE + '.h5'
    # fig_behavior_stage2(h5path, SESSION, MOUSE+SESSION, fformat)

    # MOUSE = 'LF170214_1'
    # SESSION = ['Day20170406','Day20170407','Day20170408','Day20170410',]
    # h5path = loc_info['muscimol_2_datafile'] + MOUSE + '/' + MOUSE + '.h5'
    # fig_behavior_stage2(h5path, SESSION, MOUSE+SESSION, fformat)



    # sum_fig = plt.figure(figsize=(10,6))
    # sum_ax = plt.subplot(121)
    # tn_ax = plt.subplot(122)
    # sum_ax.set_xlabel('Session #')
    # tn_ax.set_xlabel('Session #')
    # sum_ax.set_ylabel('Spatial modulation index (SMI)')
    # tn_ax.set_ylabel('Number of trials')
    # plt.tight_layout()
    #
    # novice_expert_fig = plt.figure(figsize=(4,6))
    # novice_expert_ax = plt.subplot(111)
    #
    # smi_all = np.empty((7,12))
    # smi_all[:] = np.nan
    #
    # tn_all = np.empty((7,12))
    # tn_all[:] = np.nan
    #
    # mousenames = ['LF171211_1','LF180112_2','LF171212_2','LF171211_2','LF180119_1']
    #
    #
    # # tn_ax.plot(np.arange(0,len(SESSION)),trialnr_days)
    # MOUSE = 'LF180122_2'
    # SESSION = ['Day201835','Day201836','Day201837','Day201838','Day201839','Day2018312','Day2018313','Day2018314','Day2018315']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # smi_days = []
    # trialnr_days = []
    # for s in SESSION:
    #     s_d, t_d = fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
    #     smi_days.append(s_d)
    #     trialnr_days.append(t_d)
    # summary_plot(smi_days, trialnr_days, SESSION, MOUSE+s, fformat, MOUSE)
    #
    # smi_all[1,0:len(SESSION)] = smi_days
    # tn_all[1,0:len(SESSION)] = trialnr_days
    # sum_ax.plot(np.arange(0,len(SESSION)),smi_days)
    #
    #
    MOUSE = 'LF171211_1'
    SESSION = ['Day2018215','Day201835','Day201836','Day201837','Day201838','Day201839','Day201839','Day2018312']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    smi_days = []
    trialnr_days = []
    for s in SESSION:
        s_d, t_d = fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
    #     smi_days.append(s_d)
    #     trialnr_days.append(t_d)
    # summary_plot(smi_days, trialnr_days, SESSION, MOUSE+s, fformat, MOUSE)
    # # sum_ax.plot(np.arange(0,len(SESSION)),smi_days)
    # # tn_ax.plot(np.arange(0,len(SESSION)),trialnr_days)
    # smi_all[2,0:len(SESSION)] = smi_days
    # tn_all[2,0:len(SESSION)] = trialnr_days
    #
    # #
    # MOUSE = 'LF180112_2'
    # SESSION = ['Day201835','Day201836','Day201837','Day201838','Day201839','Day2018313','Day2018314','Day2018315']
    # SESSION = ['Day201835','Day201836','Day201837','Day2018313','Day2018314','Day2018315']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # smi_days = []
    # trialnr_days = []
    # for s in SESSION:
    #     s_d, t_d = fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
    #     smi_days.append(s_d)
    #     trialnr_days.append(t_d)
    # summary_plot(smi_days, trialnr_days, SESSION, MOUSE+s, fformat, MOUSE)
    # # sum_ax.plot(np.arange(0,len(SESSION)),smi_days)
    # # tn_ax.plot(np.arange(0,len(SESSION)),trialnr_days)
    # smi_all[3,0:len(SESSION)] = smi_days
    # tn_all[3,0:len(SESSION)] = trialnr_days
    # #
    # MOUSE = 'LF171212_2'
    # SESSION = ['Day2018123','Day2018124','Day2018125','Day2018126','Day2018128','Day2018129',
    # 'Day2018130','Day2018131','Day201821','Day201826','Day201827','Day201828','Day2018212_1','Day201831',
    # 'Day201832','Day201833','Day201835','Day201836','Day201837','Day201838','Day201839','Day2018312','Day2018313']
    # # SESSION = ['Day201832','Day201833','Day201835','Day201836','Day201837','Day201838','Day201839','Day2018312','Day2018313','Day2018314','Day2018315']
    # # SESSION = ['Day201832','Day201833','Day201835','Day2018313','Day2018314','Day2018315']
    # # SESSION = ['Day2018212_1']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # smi_days = []
    # trialnr_days = []
    # for s in SESSION:
    #     s_d, t_d = fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
    #     smi_days.append(s_d)
    #     trialnr_days.append(t_d)
    # summary_plot(smi_days, trialnr_days, SESSION, MOUSE+s, fformat, MOUSE)
    # # sum_ax.plot(np.arange(0,len(SESSION[0:9])),smi_days[0:9])
    # # tn_ax.plot(np.arange(0,len(SESSION[0:9])),trialnr_days[0:9])
    # smi_all[4,0:len(SESSION[0:9])] = smi_days[0:9]
    # tn_all[4,0:len(SESSION[0:9])] = trialnr_days[0:9]
    #
    #
    # MOUSE = 'LF171211_2'
    # SESSION = ['Day201831','Day201832','Day201833','Day201835','Day201836','Day201837','Day201838','Day201839','Day2018312','Day2018313','Day2018314','Day2018315']
    # # SESSION = ['Day201831','Day201832','Day201833','Day2018313','Day2018314','Day2018315']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # smi_days = []
    # trialnr_days = []
    # for s in SESSION:
    #     s_d, t_d = fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
    #     smi_days.append(s_d)
    #     trialnr_days.append(t_d)
    # summary_plot(smi_days, trialnr_days, SESSION, MOUSE+s, fformat, MOUSE)
    # # sum_ax.plot(np.arange(0,len(SESSION)),smi_days)
    # # tn_ax.plot(np.arange(0,len(SESSION)),trialnr_days)
    # smi_all[5,0:len(SESSION)] = smi_days
    # tn_all[5,0:len(SESSION)] = trialnr_days
    #
    #
    # MOUSE = 'LF180119_1'
    # SESSION = ['Day201835','Day201836','Day201837','Day201838','Day201839','Day2018312','Day2018314','Day2018315']
    # # SESSION = ['Day201835','Day201836','Day201837','Day2018312','Day2018314','Day2018315']
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # smi_days = []
    # trialnr_days = []
    # for s in SESSION:
    #     s_d, t_d = fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
    #     smi_days.append(s_d)
    #     trialnr_days.append(t_d)
    # summary_plot(smi_days, trialnr_days, SESSION, MOUSE+s, fformat, MOUSE)
    # # sum_ax.plot(np.arange(0,len(SESSION)),smi_days)
    # # tn_ax.plot(np.arange(0,len(SESSION)),trialnr_days)
    # smi_all[6,0:len(SESSION)] = smi_days
    # tn_all[6,0:len(SESSION)] = trialnr_days
    #
    # sum_ax.plot(np.transpose(smi_all),c='0.7')
    # tn_ax.plot(np.transpose(tn_all))
    #
    # sum_ax.plot(np.nanmean(smi_all,0),lw=3,c='k')
    # tn_ax.plot(np.nanmean(tn_all,0),lw=3,c='k')
    #
    # sum_ax.set_xlim([0,5])
    # #sum_ax.set_ylim([-2,40])
    #
    # #sum_ax.legend(mousenames, loc=2)
    #
    # fname = loc_info['figure_output_path'] + 'allmice_stage2_summary' + '.' + fformat
    # sum_fig.savefig(fname, format=fformat)
    #
    # print(smi_all)
    #
    #
    # # Make fig: mean first 3 days vs mean last 3 days
    # # for i,t in enumerate([0,0,0,0,0,0,0]):
    # #     if not np.isnan(smi_all[i,0]):
    # #         novice_expert_ax.plot([0,1],[np.nanmean(smi_all[i,0:3]),np.nanmean(smi_all[i,3:6])], c='k', lw=2,zorder=1)
    # #
    # # novice_expert_ax.scatter([0,0,0,0,0,0,0],np.nanmean(smi_all[:,0:3],1), s=150, c='w',zorder=2)
    # # novice_expert_ax.scatter([1,1,1,1,1,1,1],np.nanmean(smi_all[:,3:6],1), s=150, c='k',zorder=2)
    # #
    # # fname = loc_info['figure_output_path'] + 'stage2_novice_expert' + '.' + fformat
    # # novice_expert_fig.savefig(fname, format=fformat)
    #
    # # MOUSE = 'LF180226_1'
    # # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # # SESSION = 'Day2018323_1'
    # # fig_behavior_stage2(h5path, SESSION, MOUSE+SESSION, fformat, MOUSE)
    # #
    # # MOUSE = 'LF171212_2'
    # # SESSION = 'Day201828'
    # # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # # fig_behavior_stage2(h5path, SESSION, MOUSE+SESSION, fformat, MOUSE)
    # # MOUSE = 'LF180122_1'
    # # SESSION = ['Day201832','Day201833','Day201835','Day201836','Day201837','Day201838','Day201839','Day2018312','Day2018313','Day2018314','Day2018315']
    # # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # # smi_days = []
    # # trialnr_days = []
    # # for s in SESSION:
    # #     s_d, t_d = fig_behavior_stage2(h5path, s, MOUSE+s, fformat, MOUSE)
    # #     smi_days.append(s_d)
    # #     trialnr_days.append(t_d)
    # # summary_plot(smi_days, trialnr_days, SESSION, MOUSE+s, fformat, MOUSE)
    # #
    # smi_all[0,0:len(SESSION)] = smi_days
    # tn_all[0,0:len(SESSION)] = trialnr_days
    # # sum_ax.plot(np.arange(0,len(SESSION)),smi_days)
    # # tn_ax.plot(np.arange(0,len(SESSION)),trialnr_days)
    #
    #
