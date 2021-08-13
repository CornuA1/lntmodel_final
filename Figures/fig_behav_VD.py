"""
Plot basic behavior for the visual discrimination task

@author: lukasfischer

"""

import sys, yaml, os
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + "Analysis")
import warnings; warnings.simplefilter('ignore')

# import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from scipy import stats
from load_behavior_data import load_data
from filter_trials import filter_trials
from rewards import rewards
from licks import licks_nopost as licks

TRACK_SHORT = 7
TRACK_LONG = 9

def set_empty_axis(ax_object):
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_visible(False)
    ax_object.spines['bottom'].set_visible(False)
    ax_object.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        length=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    return ax_object


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
    trial_licks = all_licks[np.in1d(all_licks[:, 3], [TRACK_SHORT, TRACK_LONG]), :]
    reward =  rewards(raw_data)
    return raw_data, trial_licks, reward

def fig_behav_VD(data_path, sess, fname, fformat='png', subfolder=[]):
    #raw_ds, licks_ds, reward_ds = load_data_h5(data_path, sess)
    raw_filename = data_path + os.sep + sess[0] + os.sep + sess[1]
    raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sess)


    # if the animal hasn't licked at all in a session, just retun without generating a figure
    if np.size(licks_ds) == 0:
        print('NO LICKS DETECTED')
        return
    # create figure to later plot on
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((4,4),(0,0),rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((4,4),(0,2),rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((4,4),(2,0),rowspan=2, colspan=2)
    ax4 = plt.subplot2grid((4,4),(2,2),rowspan=2, colspan=1)
    ax5 = plt.subplot2grid((4,4),(2,3),rowspan=2, colspan=1)

    # remove frame around panel
    set_empty_axis(ax1)
    set_empty_axis(ax2)

    # set axis limits
    ax1.set_xlim([50,250])
    ax2.set_xlim([50,250])
    ax3.set_xlim([0,49])
    ax4.set_xlim([0,49])
    ax5.set_xlim([0.8,3])

    ax1.set_ylabel('Trial #')
    ax1.set_xlabel('Location (cm)')

    ax2.set_ylabel('Trial #')
    ax2.set_xlabel('Location (cm)')

    ax3.set_ylabel('Licks/cm')
    ax4.set_ylabel('running speed (cm/sec)')
    ax5.set_ylabel('avg. licks/trial')

    # plot landmark landmark zones as shaded area
    ax1.axvspan(205,245,color='0.9',zorder=0)
    ax2.axvspan(205,245,color='0.9',zorder=0)
    #ax3.axvspan(205,245,color='0.9',zorder=0)
    #ax4.axvspan(205,245,color='0.9',zorder=0)

    # only plot trials where either a lick and/or a reward were detected
    # therefore: pull out trial numbers from licks and rewards dataset and map to
    # a new list of rows used for plotting
    short_trials = filter_trials( raw_ds, [], ['tracknumber',7])
    long_trials = filter_trials( raw_ds, [], ['tracknumber',9])

    ax1.set_ylim([0,np.size(np.unique(short_trials))])
    ax2.set_ylim([0,np.size(np.unique(long_trials))])

    # get trial numbers to be plotted
    #lick_trials = np.unique(licks_ds[:,2])
    #reward_trials = np.unique(reward_ds[:,3])-1
    #scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
    #scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
    scatter_rowlist_map_short = short_trials
    scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
#    scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
    scatter_rowlist_map_long = long_trials
    scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

    # determine in which trials animal licked, separated by track type
    # get rows where valve opens
    rew_rows_diff = np.diff(raw_ds[:,5])
    rew_idx = np.where(rew_rows_diff > 0)[0]
    rew_track = raw_ds[rew_idx, 4]
    rew_hit = raw_ds[rew_idx+1, 5]

    hit_rate = 0
    rew_trials_count = 0
    false_alarm = 0
    nrew_trials_counter = 0

    for i,rt in enumerate(rew_track):
        if rt == 7:
            rew_trials_count = rew_trials_count + 1
            if rew_hit[i] == 1:
                hit_rate = hit_rate + 1
        if rt == 9:
            nrew_trials_counter = nrew_trials_counter + 1
            if rew_hit[i] == 1:
                false_alarm = false_alarm + 1

    hit_rate = hit_rate / rew_trials_count
    false_alarm = false_alarm / nrew_trials_counter

    # set min and max values so we don't get -inf or inf with the ICDF
    hit_rate = np.amin([hit_rate, 0.99])
    hit_rate = np.amax([hit_rate, 0.01])

    false_alarm = np.amin([false_alarm, 0.99])
    false_alarm = np.amax([false_alarm, 0.01])

    d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm)

    # adjust trial number for blackbox to correspond to the trial number of the trial just before it
    #rew_rows[rew_rows[:,4]==5,6] = rew_rows[rew_rows[:,4]==5,6] - 1
    #rew_rows = np.where(raw_ds[:,5]==1)[0]
    #rew_rows[rew_rows[:,4]==5,6] = rew_rows[rew_rows[:,4]==5,6] - 1
    #rew_rows_onset = rew_rows[np.diff(rew_rows) > 1]
    #rew_rows_onset = np.insert(rew_rows_onset,0,0)

    #print(rew_rows)
    #print(rew_rows_onset)
    #hit_rate = np.unique(short_trials[:,6]).shape[0] / lick_trials.shape[0]
    #print(hit_rate)

    # scatterplot of licks/rewards in order of trial number
    for i,r in enumerate(scatter_rowlist_map_short):
        plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
        plot_rewards_x = reward_ds[reward_ds[:,3]-1==r,1]
        cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
        if reward_ds[reward_ds[:,3]-1==r,5] == 1:
            col = '#00C40E'
        else:
            col = 'r'

        if plot_rewards_x == 0.0:
            plot_rewards_x = 245

        if np.size(plot_licks_x) > 0:
            plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_short[i])
            ax1.scatter(plot_licks_x, plot_licks_y, c='k', lw=0)
        if np.size(plot_rewards_x) > 0:
            plot_rewards_y = scatter_rowlist_short[i]
            ax1.scatter(plot_rewards_x, plot_rewards_y, c=col, lw=0)
        if np.size(cur_trial_start) > 0:
            plot_starts_y = scatter_rowlist_short[i]
            ax1.scatter(cur_trial_start, plot_starts_y, c='b', marker='>', lw=0)

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
            ax2.scatter(plot_licks_x, plot_licks_y, c='k', lw=0)

        if np.size(cur_trial_start) > 0:
            plot_starts_y = scatter_rowlist_long[i]
            ax2.scatter(cur_trial_start, plot_starts_y, c='b', marker='>', lw=0)

    # plot licks per cm
    bin_size = 5
    # why are both binnr_short and binnr_long using this 460 number?
    binnr_short = 245/bin_size
    lick_histogram = np.array([])
    lick_short_num = []
    for l in scatter_rowlist_map_short:
        lick_histogram = np.append(lick_histogram, licks_ds[licks_ds[:,2]==l,1])
        # safe how many lick were detected in each trial
        lick_short_num.append(np.size(licks_ds[licks_ds[:,2]==l,1]))

    #n = ax3.hist(lick_histogram, int(binnr_short))
    lickhist = np.histogram(lick_histogram, int(binnr_short), [0,245])
    lickspcm = lickhist[0]/np.size(np.unique(short_trials))
    ax3.plot(lickspcm)
    #max_y_short = np.amax(n[0])

    # plot licks per cm
    binnr_long = 245/bin_size
    lick_histogram = np.array([])
    lick_long_num = []
    for l in scatter_rowlist_map_long:
        lick_histogram = np.append(lick_histogram, licks_ds[licks_ds[:,2]==l,1])
        # safe how many lick were detected in each trial
        lick_long_num.append(np.size(licks_ds[licks_ds[:,2]==l,1]))

    lickhist = np.histogram(lick_histogram, int(binnr_short), [0,245])
    lickspcm = lickhist[0]/np.size(np.unique(short_trials))
    ax3.plot(lickspcm,c='r')

    ax5.bar(1, np.sum(lick_short_num) / np.size(scatter_rowlist_map_short))
    ax5.bar(2, np.sum(lick_long_num) / np.size(scatter_rowlist_map_long))

    ax5.set_title(d_prime)

    # plot running speed
    bin_size = 5
    binnr_short = 245/bin_size
    mean_speed = np.empty((np.size(scatter_rowlist_map_short,0),int(binnr_short)))
    mean_speed[:] = np.NAN
    max_y_short = 0
    for i,t in enumerate(scatter_rowlist_map_short):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
        cur_trial_start = raw_ds[raw_ds[:,6]==t,1][0]
        cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)

        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], raw_ds[raw_ds[:,6]==t,
                                                          3], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
            #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
            max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])

    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmean(mean_speed,0)
    ax4.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='g',zorder=3)
    ax4.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color='g',alpha=0.2)


    # plot running speed
    bin_size = 5
    binnr_long = 245/bin_size
    mean_speed = np.empty((np.size(scatter_rowlist_long,0),int(binnr_long)))
    mean_speed[:] = np.NAN
    max_y_long = 0
    for i,t in enumerate(scatter_rowlist_map_long):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
        cur_trial_start = raw_ds[raw_ds[:,6]==t,1][0]
        cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)
        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], raw_ds[raw_ds[:,6]==t,
                                   3], 'mean', cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
            mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
            #ax4.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
            max_y_long = np.amax([max_y_long,np.amax(mean_speed_trial)])

    sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
    mean_speed_sess_long = np.nanmean(mean_speed,0)
    ax4.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c='r',zorder=3)
    ax4.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color='r',alpha=0.2)

    ax4.plot(np.linspace(0,binnr_short-1,binnr_long),mean_speed_sess_long,c='0.7',zorder=2)
    ax4.set_ylim([0,np.nanmax([max_y_short,max_y_long])])

    plt.tight_layout()

    fig.suptitle('visual discrimination' + fname, wrap=True)

    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    fig.savefig(fname, format=fformat)

    print(fname)


def run_LF180515_1():
    MOUSE = 'LF180515_1'
    SESSION = ['Day20181125','Day20181127','Day20181128']
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    for s in SESSION:
        fig_behav_VD(h5path, s, MOUSE+s+'_stim_on', fformat, MOUSE)

def run_Pumba():
    MOUSE = 'Pumba'
    SESSION = [['20190509','MTH3_VD_201959_1620.csv'],
               ['20190510','MTH3_VD_2019510_1636.csv'],
               ['20190516','MTH3_VD_2019516_1857.csv']]

    data_path = loc_info['raw_dir'] + MOUSE
    for s in SESSION:
        fig_behav_VD(data_path, s, MOUSE+s[0]+'_VD', fformat, MOUSE)

def run_Jimmy():
    MOUSE = 'Jimmy'
    SESSION = [['20190509','MTH3_VD_201959_1651.csv'],
               ['20190510','MTH3_VD_2019510_1711.csv'],
               ['20190516','MTH3_VD_2019516_1932.csv']]

    data_path = loc_info['raw_dir'] + MOUSE
    for s in SESSION:
        fig_behav_VD(data_path, s, MOUSE+s[0]+'_VD', fformat, MOUSE)


if __name__ == '__main__':
    fformat = 'png'

    # run_LF180515_1()
    run_Pumba()
    run_Jimmy()
