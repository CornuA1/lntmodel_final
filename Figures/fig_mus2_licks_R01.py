"""
Plot licking raster plots for RSC Spring 2018 R01

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

def fig_behavior(h5path, sess, fname, fformat='png', subfolder=[]):
    # load data
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(14,6))
    ax1 = plt.subplot2grid((8,2),(0,0), rowspan=6)
    ax2 = plt.subplot2grid((8,2),(6,0), rowspan=2)
    ax3 = plt.subplot2grid((8,2),(0,1), rowspan=6)
    ax4 = plt.subplot2grid((8,2),(6,1), rowspan=2)

    ax1.set_xlim([50,340])
    ax1.set_ylabel('Trial #')
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Short trials')

    ax2.set_xlim([10,65])
    ax2.set_ylabel('Speed (cm/sec)')
    ax2.set_xlabel('Location (cm)')

    ax3.set_xlim([50,400])
    ax3.set_ylabel('Trial #')
    ax3.set_xlabel('Location (cm)')
    ax3.set_title('Short trials')

    ax4.set_xlim([10,77])
    ax4.set_ylabel('Speed (cm/sec)')
    ax4.set_xlabel('Location (cm)')

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color=sns.xkcd_rgb["windows blue"],alpha=0.3,zorder=9)

    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(380,400,color=sns.xkcd_rgb["dusty purple"],alpha=0.3,zorder=9)

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

    ax1.set_ylim([-1,len(np.unique(scatter_rowlist_short))])
    ax3.set_ylim([-1,len(np.unique(scatter_rowlist_long))])

    ax2.set_ylim([0,100])
    ax4.set_ylim([0,100])

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
            plot_rewards_x = 335

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_short[i])
            ax1.scatter(plot_licks_x, plot_licks_y,c='k',lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = scatter_rowlist_short[i]
            ax1.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)

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
            plot_rewards_x = 395

        # plot licks and rewards
        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_long[i])
            ax3.scatter(plot_licks_x, plot_licks_y,c='k',lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = scatter_rowlist_long[i]
            ax3.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)

    # plot running speed
    bin_size = 5
    binnr_short = 340/bin_size
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
            ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8',lw=0.5)
            max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])

    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax2.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='k',lw=2,zorder=3)
    #ax2.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color=sns.xkcd_rgb["windows blue"],alpha=0.2)

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
            mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], raw_ds[raw_ds[:,6]==t,
                                                          3], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
            ax4.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8',lw=0.5)
            max_y_long = np.amax([max_y_long,np.amax(mean_speed_trial)])

    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_long = np.nanmean(mean_speed,0)
    ax4.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c='k',lw=2,zorder=3)
    #ax4.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color=sns.xkcd_rgb["dusty purple"],alpha=0.2)

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

    mean_fl_short = np.mean(first_lick_short)
    mean_fl_long = np.mean(first_lick_long)



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

    return mean_fl_long - mean_fl_short

if __name__ == '__main__':
    fformat = 'png'

    with open(loc_info['yaml_archive'], 'r') as f:
        project_metainfo = yaml.load(f)

    fld_all = np.zeros((3,3))

    fld = []

    m = 'LF171016_2'
    SESSIONS = ['Day2017121','Day20171214','Day20171215']
    h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
    for s in SESSIONS:
        fld.append(fig_behavior(h5path, s, m+s, fformat, 'mus2_R01'))
    fld_all[:,0] = fld
    fld = []


    m = 'LF171016_5'
    SESSIONS = ['Day2017121','Day20171212','Day20171214']
    h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
    for s in SESSIONS:
        fld.append(fig_behavior(h5path, s, m+s, fformat, 'mus2_R01'))
    fld_all[:,1] = fld
    fld=[]

    m = 'LF171016_6'
    SESSIONS = ['Day2017122','Day2017125','Day20171214']
    h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
    for s in SESSIONS:
        fld.append(fig_behavior(h5path, s, m+s, fformat, 'mus2_R01'))
    fld_all[:,2] = fld

    print(fld_all)

    # create figure to later plot on
    fig = plt.figure(figsize=(2,4))
    ax1 = plt.subplot(111)
    ax1.plot(fld_all,c='0.7',lw=3,zorder=1)
    ax1.scatter([0,0,0],fld_all[0,:], s=100, c='k',zorder=2)
    ax1.scatter([1,1,1],fld_all[1,:], s=100, c='k',zorder=2)
    ax1.scatter([2,2,2],fld_all[2,:], s=100, c='w',zorder=2)

    ax1.set_ylim([0,55])
    ax1.set_xlim([-0.2,2.2])
    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(['NOVICE','EXPERT','MUSCIMOL'],rotation=45)

    fname = loc_info['figure_output_path'] + 'mus2_R01' + os.sep + 'mus2_comp' + '.' + fformat

    fig.savefig(fname, format=fformat)
    plt.show()


    print('done')
