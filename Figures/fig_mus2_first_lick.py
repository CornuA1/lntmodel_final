"""
Plot difference in mean first lick location of animals of Muscimol experiment

@author: lukasfischer

"""

# %matplotlib inline

# load local settings file
import matplotlib
import numpy as np
#matplotlib.use(fformat,force=True)
from matplotlib import pyplot as plt
import warnings; warnings.simplefilter('ignore')
import sys

import yaml
import h5py
from scipy import stats
import seaborn as sns
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
            content = yaml.load(f)

with open(content['yaml_archive'], 'r') as f:
            project_metainfo = yaml.load(f)

# load animal group data from project YAML file
m = project_metainfo['muscimol2_mice']

days_ACSF = project_metainfo['mus2_mice_CONTROL']
days_MUS = project_metainfo['mus2_mice_MUSCIMOL']

def mus2_first_lick():
    mean_fl_ACSF_short = np.zeros((len(m),))
    mean_fl_ACSF_long = np.zeros((len(m),))
    mean_fl_MUS_short = np.zeros((len(m),))
    mean_fl_MUS_long = np.zeros((len(m),))

    licks_ACSF = None
    licks_MUS = None
    # load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous

    for i,mouse in enumerate(m):
        # load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous
        for da in days_ACSF[mouse]:
            h5path = content['muscimol_2_datafile'] + mouse + '/' + mouse + '.h5'
            if licks_ACSF is not None:
                h5dat = h5py.File(h5path, 'r')
                licks_ACSF_ds = np.copy(h5dat[da + '/'+ mouse + '/licks_pre_reward'])
                h5dat.close()
                licks_ACSF_ds[:,2] = licks_ACSF_ds[:,2]+last_trial_ACSF
                licks_ACSF = np.append(licks_ACSF,licks_ACSF_ds,axis=0)
                last_trial_ACSF = licks_ACSF[-1,2]
            else:
                h5dat = h5py.File(h5path, 'r')
                print(h5dat, da)
                licks_ACSF = np.copy(h5dat[da + '/licks_pre_reward'])
                h5dat.close()
                last_trial_ACSF = licks_ACSF[-1,2]

        # load MUSCIMOL datasets - when datasets are appended, re-enumerate trial numbers to be continuous
        for da in days_MUS[mouse]:
            h5path = content['muscimol_2_datafile'] + mouse + '/' + mouse + '.h5'
            if licks_MUS is not None:
                h5dat = h5py.File(h5path, 'r')
                licks_MUS_ds = np.copy(h5dat[da + '/'+ mouse + '/licks_pre_reward'])
                h5dat.close()
                licks_MUS_ds[:,2] = licks_MUS_ds[:,2]+last_trial_MUS
                licks_MUS = np.append(licks_MUS,licks_MUS_ds,axis=0)
                last_trial_MUS = licks_MUS[-1,2]
            else:
                h5dat = h5py.File(h5path, 'r')
                licks_MUS = np.copy(h5dat[da + '/licks_pre_reward'])
                h5dat.close()
                last_trial_MUS = licks_MUS[-1,2]

        first_lick_short_ACSF = []
        first_lick_short_trials_ACSF = []
        first_lick_long_ACSF = []
        first_lick_long_trials_ACSF = []
        for r in np.unique(licks_ACSF[:,2]):
            licks_trial = licks_ACSF[licks_ACSF[:,2]==r,:]
            licks_trial = licks_trial[licks_trial[:,1]>240,:]
            if licks_trial.shape[0]>0:
                if licks_trial[0,3] == 3:
                    first_lick_short_ACSF.append(licks_trial[0,1])
                    first_lick_short_trials_ACSF.append(r)
                elif licks_trial[0,3] == 4:
                    first_lick_long_ACSF.append(licks_trial[0,1])
                    first_lick_long_trials_ACSF.append(r)

        # determine location of first licks - MUSCIMOL
        first_lick_short_MUS = []
        first_lick_short_trials_MUS = []
        first_lick_long_MUS = []
        first_lick_long_trials_MUS = []
        for r in np.unique(licks_MUS[:,2]):
            licks_trial = licks_MUS[licks_MUS[:,2]==r,:]
            licks_trial = licks_trial[licks_trial[:,1]>240,:]
            if licks_trial.shape[0]>0:
                if licks_trial[0,3] == 3:
                    first_lick_short_MUS.append(licks_trial[0,1])
                    first_lick_short_trials_MUS.append(r)
                elif licks_trial[0,3] == 4:
                    first_lick_long_MUS.append(licks_trial[0,1])
                    first_lick_long_trials_MUS.append(r)

        mean_fl_ACSF_short[i] = np.mean(first_lick_short_ACSF)
        mean_fl_ACSF_long[i] = np.mean(first_lick_long_ACSF)
        mean_fl_MUS_short[i] = np.mean(first_lick_short_MUS)
        mean_fl_MUS_long[i] = np.mean(first_lick_long_MUS)

        licks_ACSF = None
        licks_MUS = None

    fig = plt.figure(figsize=(4,4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)


    mm_ACSF_short = np.mean(mean_fl_ACSF_short)
    sem_ACSF_short = stats.sem(mean_fl_ACSF_short)
    mm_ACSF_long = np.mean(mean_fl_ACSF_long)
    sem_ACSF_long = stats.sem(mean_fl_ACSF_long)

    mm_MUS_short = np.mean(mean_fl_MUS_short)
    sem_MUS_short = stats.sem(mean_fl_MUS_short)
    mm_MUS_long = np.mean(mean_fl_MUS_long)
    sem_MUS_long = stats.sem(mean_fl_MUS_long)

    # mm_DIFF_short = mean_fl_ACSF_short - mean_fl_MUS_short
    # mm_DIFF_long = mean_fl_ACSF_long - mean_fl_MUS_long

    mm_DIFF_short = mean_fl_MUS_long - mean_fl_MUS_short
    mm_DIFF_long = mean_fl_ACSF_long - mean_fl_ACSF_short

    print(mm_DIFF_short)
    print(mm_DIFF_long)


    ax2.plot([1,2,3,4])

    plt.suptitle('t-score')
    ax1.set_xlim([-0.2,1.2])
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['ACSF','MUSCIMOL'], rotation=45)

    ax1.scatter(np.ones(mm_DIFF_short.shape[0]), mm_DIFF_short,s=80,c=['w'],edgecolors='k',zorder=1)
    ax1.scatter(np.zeros(mm_DIFF_long.shape[0]), mm_DIFF_long,s=80,c=['k'],edgecolors='k',zorder=1)

    for i in range(mm_DIFF_short.shape[0]):
        ax1.plot([1,0], [mm_DIFF_short[i], mm_DIFF_long[i]],c='k',zorder=0)

    fname = content['figure_output_path'] + 'MUS_DIFF_mice' + '.' + 'png'
    print(fname)
    fig.savefig(fname, format='png')


if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    fformat = 'png'

    with open(content['yaml_archive'], 'r') as f:
        project_metainfo = yaml.load(f)

    MOUSE = project_metainfo['muscimol2_mice']
    mus2_first_lick()
    # for m in MOUSE:
    #     h5path = loc_info['muscimol_2_datafile'] + m + '/' + m + '.h5'
    #     #for s in project_metainfo['mus2_mice_stage5'][m]:
    #     for s in project_metainfo['mus2_mice_MUSCIMOL'][m]:
    #         print(h5path, s)
    #         fig_behavior_stage5(h5path, s, m+s, fformat, 'mus2')
