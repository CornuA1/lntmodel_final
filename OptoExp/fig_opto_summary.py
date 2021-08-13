"""
Plot summary data for opto experiments

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

def first_licks(licks, rewards, trials, scatter_color):
    # plot location of first trials on short and long trials
    first_lick = np.empty((0,4))
    first_lick_trials = np.empty((0))
    for r in trials:
        licks_all = licks[licks[:,2]==r,:]
        licks_all = licks_all[licks_all[:,1]>101,:]
        if licks_all.size == 0:
             rew_lick = rewards[rewards[:,3]==r,:]
             if rew_lick.size > 0:
                 if rew_lick[0,5] == 1:
                     licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                     first_lick = np.vstack((first_lick, licks_all[0,:].T))
                     first_lick_trials = np.append(first_lick_trials, r)
        else:
            if licks_all[0,3] == 3:
                licks_all = licks_all[licks_all[:,1]<338,:]
            elif licks_all[0,3] == 4:
                licks_all = licks_all[licks_all[:,1]<398,:]
            first_lick = np.vstack((first_lick, licks_all[0,:].T))
            first_lick_trials = np.append(first_lick_trials, r)

    return first_lick

def tscore_summary(datasets, fname, fformat='png', subfolder=[]):
    # load datasets
    raw_ds = []
    licks_ds = []
    reward_ds = []
    for ds in datasets:
        h5path = loc_info['imaging_dir'] + ds[0] + '/' + ds[0] + '.h5'
        h5dat = h5py.File(h5path, 'r')
        raw_ds.append(np.copy(h5dat[ds[1] + '/raw_data']))
        licks_ds.append(np.copy(h5dat[ds[1] + '/licks_pre_reward']))
        reward_ds.append(np.copy(h5dat[ds[1] + '/rewards']))
        h5dat.close()

    # loop through all datasets and get location of first licks
    mask_on_first_licks_loc_short = []
    mask_on_first_licks_loc_long = []
    stim_on_first_licks_loc_short = []
    stim_on_first_licks_loc_long = []
    for i in range(len(raw_ds)):
        # divide trials up into mask on and mask off trials.
        short_trials = filter_trials( raw_ds[i], [], ['tracknumber',3])
        mask_on_short_trials = filter_trials( raw_ds[i], [], ['opto_mask_on_stim_off'],short_trials)
        stim_on_short_trials = filter_trials( raw_ds[i], [], ['opto_stim_on'],short_trials)

        long_trials = filter_trials( raw_ds[i], [], ['tracknumber',4])
        mask_on_long_trials = filter_trials( raw_ds[i], [], ['opto_mask_on_stim_off'],long_trials)
        stim_on_long_trials = filter_trials( raw_ds[i], [], ['opto_stim_on'],long_trials)

        mask_on_first_licks_loc_short.append(first_licks(licks_ds[i], reward_ds[i], mask_on_short_trials, 'b'))
        mask_on_first_licks_loc_long.append(first_licks(licks_ds[i], reward_ds[i], mask_on_long_trials, '#009EFF'))

        stim_on_first_licks_loc_short.append(first_licks(licks_ds[i], reward_ds[i], stim_on_short_trials, 'm'))
        stim_on_first_licks_loc_long.append(first_licks(licks_ds[i], reward_ds[i], stim_on_long_trials, '#E6A2FF'))

    # create figure to later plot on
    fig = plt.figure(figsize=(4,4))
    ax1 = plt.subplot(111)

    mask_on_first_lick_median_short = np.empty((0,1))
    mask_on_first_lick_median_long = np.empty((0,1))
    stim_on_first_lick_median_short = np.empty((0,1))
    stim_on_first_lick_median_long = np.empty((0,1))
    for i in range(len(mask_on_first_licks_loc_short)):
        mask_on_first_lick_median_short = np.append(mask_on_first_lick_median_short, np.median(mask_on_first_licks_loc_short[i][:,1]))
        mask_on_first_lick_median_long = np.append(mask_on_first_lick_median_long, np.median(mask_on_first_licks_loc_long[i][:,1]))
        stim_on_first_lick_median_short = np.append(stim_on_first_lick_median_short, np.median(stim_on_first_licks_loc_short[i][:,1]))
        stim_on_first_lick_median_long = np.append(stim_on_first_lick_median_long, np.median(stim_on_first_licks_loc_long[i][:,1]))

    mask_on_diff = mask_on_first_lick_median_long - mask_on_first_lick_median_short
    stim_on_diff = stim_on_first_lick_median_long - stim_on_first_lick_median_short
    ax1.scatter(np.full(len(mask_on_first_lick_median_short),0), mask_on_diff, s=80, facecolor='k', edgecolor='k', linewidth=2,zorder=3)
    ax1.scatter(np.full(len(stim_on_first_lick_median_short),1), stim_on_diff, s=80, facecolor='w', edgecolor='k', linewidth=2,zorder=3)

    for i in range(len(mask_on_first_lick_median_short)):
        ax1.plot([0,1],[mask_on_diff[i], stim_on_diff[i]],lw=2,c='k',zorder=2)

    ax1.set_xlim([-0.2,1.2])
    ax1.set_ylim([0,70])

    print(mask_on_diff, stim_on_diff)
    print(stats.ttest_rel(mask_on_diff, stim_on_diff))

    # create subfolder (if necessary) and store figure
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info().print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

if __name__ == '__main__':
    %load_ext autoreload
    %matplotlib inline
    %autoreload

    fformat = 'png'

    datasets = [['LF180514_1', 'Day2018924'],['LF180515_1', 'Day2018924'],['LF180728_1', 'Day2018928']]
    tscore_summary(datasets, 'Opto_summary_result', fformat, 'Opto')
