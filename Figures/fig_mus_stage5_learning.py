"""
Plot learning of stage 5 in first group of muscimol animals

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

def fig_mus_stage5_learning(h5path, mouse, sess, fname, fformat='png', subfolder=[]):
    # find days in which mice an entry for a given mouse exists (not all mice were trained on all days)
    h5dat = h5py.File(h5path, 'r')
    days = [name for name in h5dat]
    days_avail = []
    print(mouse)
    # determine which days contain data from that animal and are stage 5
    for d in days:
        tm = [mouse for mouse in h5dat['/'+d]]
        if mouse in tm:
            behav_ds = np.copy(h5dat['/' + d + '/' + mouse + '/raw_data'])
            if np.unique(behav_ds[:,4]).shape[0] == 3:
                days_avail.append(d)
    h5dat.close()

    fl_diff = []

    for da in days_avail:
        print(da)
        h5dat = h5py.File(h5path, 'r')
        raw_ds = np.copy(h5dat['/' + da + '/' + mouse + '/raw_data'])
        licks_ds = np.copy(h5dat['/' + da + '/' + mouse + '/licks_pre_reward'])
        reward_ds = np.copy(h5dat['/' + da + '/' + mouse + '/rewards'])
        h5dat.close()

        # only plot trials where either a lick and/or a reward were detected
        # therefore: pull out trial numbers from licks and rewards dataset and map to
        # a new list of rows used for plotting
        short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
        long_trials = filter_trials( raw_ds, [], ['tracknumber',4])

        # get trial numbers to be plotted
        if licks_ds.shape[0] > 0:
            lick_trials = np.unique(licks_ds[:,2])
            reward_trials = np.unique(reward_ds[:,3])-1
            scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
            scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
            scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
            scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
            scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

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

            fl_diff.append(mean_fl_long - mean_fl_short)
        else:
            fl_diff.append(np.nan)

    return fl_diff

if __name__ == '__main__':
    fformat = 'png'
    subfolder = 'mus_learning'
    fname = 'mus_learn'

    with open(loc_info['yaml_archive'], 'r') as f:
        project_metainfo = yaml.load(f)

    MICE = ['LF161031_1','LF161031_2','LF161031_3','LF161031_4','LF161031_5','LF161031_6','LF161031_7','LF161031_8']

    #SESSIONS = ['Day20171214']
    h5path = loc_info['muscimol_file']

    fig = plt.figure(figsize=(4,4))
    ax1 = plt.subplot(111)
    for m in MICE:
        fld = fig_mus_stage5_learning(h5path, m, [], 'mus_learning', fformat, 'mus2_R01')
        ax1.plot(fld)

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
    print('done')
