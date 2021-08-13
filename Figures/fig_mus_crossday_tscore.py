"""
Plot licking task-score (from shuffled distribution) for STAGE 5 of the
muscimol2 experiment

this ONLY works with blocks of 5 short/long trial structure, NOT randomized

"""

# %load_ext autoreload
# %autoreload
# %matplotlib inline

import numpy as np
import h5py
import warnings
import sys
import os
import yaml
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

with open(loc_info['yaml_archive'], 'r') as f:
    project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')

def fig_tscore(h5path, sess, fname, fformat='png', subfolder=[]):

    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    rewards_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # set location of reward zone
    rz = [380, 400]
    # location of default reward and tracklength
    default = 400
    tracklength = 400

    # create figure to later plot on
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,2),(0,0),colspan=2)
    #ax2 = plt.subplot2grid((1,2),(0,1),rowspan=3)

    # determine the maximum number of training days on stage 5
    max_days = 0
    for mouse in MOUSE:
        max_days = max(max_days,len(project_metainfo['mus_mice_stage5'][mouse]))

    # create array to hold taskscore values
    days_tscore = np.zeros((len(m),max_days))
    days_tscore[:] = np.nan

    # get trial numbers in which animal has licked
    lick_trials = np.unique(licks_ds[:,2])
    reward_trials = rewards_ds[rewards_ds[:,5]==1,3]-1
    #reward_trials = np.unique(reward_ds[:,3])
    lick_trials = np.union1d(lick_trials,reward_trials)

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
            # if rew_lick[1] < 300:
            #     plot_rewards_x = 340
        if licks_all.shape[0] > 0:
            lick = licks_all[0]
            if lick[3] == 3:
                first_lick_short.append(lick[1])
                first_lick_short_trials.append(r)
            elif lick[3] == 4:
                first_lick_long.append(lick[1])
                first_lick_long_trials.append(r)

    ax1.scatter(first_lick_short, np.linspace(0,len(first_lick_short)-1,len(first_lick_short)), c=sns.xkcd_rgb["windows blue"],lw=0,s=40)
    ax1.scatter(first_lick_long, np.linspace(0,len(first_lick_long)-1,len(first_lick_long)),c=sns.xkcd_rgb["dusty purple"],lw=0,s=40)

    ax1.axvline(np.median(first_lick_short), c=sns.xkcd_rgb["windows blue"], lw=2)
    ax1.axvline(np.median(first_lick_long), c=sns.xkcd_rgb["dusty purple"], lw=2)

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
    ax1.axvspan(np.nanmedian(first_lick_short)+bt_CI_5_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.2)
    ax1.axvspan(np.nanmedian(first_lick_short)+bt_CI_95_short,np.nanmedian(first_lick_short), color=sns.xkcd_rgb["windows blue"], ls='--',alpha=0.2)

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

    ax1.axvspan(np.nanmedian(first_lick_long)+bt_CI_5_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.2)
    ax1.axvspan(np.nanmedian(first_lick_long)+bt_CI_95_long,np.nanmedian(first_lick_long), color=sns.xkcd_rgb["dusty purple"], ls='--',alpha=0.2)

    sig = 0
    if np.nanmedian(first_lick_long)+bt_CI_5_long > np.nanmedian(first_lick_short) and np.nanmedian(first_lick_short)+bt_CI_95_short < np.nanmedian(first_lick_long):
        fig.suptitle(str(np.median(first_lick_long) - np.median(first_lick_short)) + '*', wrap=True)
        sig = 1
    else:
        fig.suptitle(str(np.median(first_lick_long) - np.median(first_lick_short)) + 'n.s.', wrap=True)
        sig = -1

    ax1.set_xlim([240,380])

    plt.tight_layout()

    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    #print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    return str(np.median(first_lick_long) - np.median(first_lick_short))#, sig

if __name__ == '__main__':

    MOUSE = project_metainfo['muscimol2_mice']

    # day = 'Day20171123'

    # output figure format
    fformat = 'png'

    # filename and subfolder for output figure
    fname = 'mus2_tscore'
    subfolder = 'mus2_tscore'

    sig = []
    for m in MOUSE:
        h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
        m_sig = []
        for s in project_metainfo['mus_mice_stage5'][m]:
            m_sig.append(fig_tscore(h5path, s, m+s, fformat, subfolder))
        sig.append(m_sig)

    sig = np.asarray(sig).astype('float')

    # create figure to later plot on
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.plot(np.transpose(sig), c='0.8')
    ax1.plot(np.nanmean(np.transpose(sig),1), lw=2, c='k')
    ax1.axhline(0, lw=2, c='0.8', ls='--')

    plt.show()
    #print(project_metainfo['mus_mice_stage5'][MOUSE[0])
    #print(ax1.get_xticks().astype(int))
    #ax1.set_xticklabels(project_metainfo['mus_mice_stage5'][MOUSE[0]][ax1.get_xticks().astype(int)], rotation=45)
