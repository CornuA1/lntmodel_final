"""
Plot licking z-score (from shuffled distribution) for STAGE 2 of the
muscimol2 experiment

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
# import seaborn as sns


with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

# with open(loc_info['yaml_file'], 'r') as f:
#     project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')

from smzscore import smzscore

# output figure format
fformat = 'svg'

m = ['LF171016_1','LF171016_2','LF171016_3','LF171016_4','LF171016_5','LF171016_6']

days_training_stage2 = {
     'LF171016_1': ['Day2017111', 'Day2017112', 'Day2017113', 'Day2017115', 'Day2017116',
        'Day2017117', 'Day2017118', 'Day2017119', 'Day20171112', 'Day20171113', 'Day20171114',
        'Day20171115'],
      'LF171016_2': ['Day2017111', 'Day2017112', 'Day2017113', 'Day2017115', 'Day2017116',
        'Day2017117', 'Day2017118', 'Day2017119', 'Day20171112', 'Day20171113', 'Day20171114',
        'Day20171115'],
      'LF171016_3': ['Day2017111', 'Day2017112', 'Day2017113', 'Day2017115', 'Day2017116',
        'Day2017117', 'Day2017118', 'Day2017119', 'Day20171112', 'Day20171113', 'Day20171114',
        'Day20171115'],
      'LF171016_4': ['Day2017111', 'Day2017112', 'Day2017113', 'Day2017115', 'Day2017116',
        'Day2017117', 'Day2017118', 'Day2017119', 'Day20171112', 'Day20171113', 'Day20171114',
        'Day20171115'],
      'LF171016_5': ['Day2017111', 'Day2017112', 'Day2017113', 'Day2017115', 'Day2017116',
        'Day2017117', 'Day2017118', 'Day2017119', 'Day20171112', 'Day20171113', 'Day20171114',
        'Day20171115'],
      'LF171016_6': ['Day2017111', 'Day2017112', 'Day2017113', 'Day2017115', 'Day2017116',
        'Day2017117', 'Day2017118', 'Day2017119', 'Day20171112', 'Day20171113', 'Day20171114',
        'Day20171115']
    }

# load animal group data from project YAML file
# m = project_metainfo['muscimol2_mice']
# days_training_stage2 = project_metainfo['mus_mice_stage2']

# set location of reward zone
rz = [320, 340]
# location of default reward and tracklength
default = 340
tracklength = 340
# number of shuffles for smz calculation
shuffles = 1000

# determine the maximum number of training days on stage 2
max_days = 0
for mouse in m:
    max_days = max(max_days,len(days_training_stage2[mouse]))

# create array to hold smz values
days_smz = np.zeros((len(m),max_days))
days_smz[:] = np.nan

# h5path = loc_info['muscimol_2_datafile'] + 'LF171016_6' + '/' + 'LF171016_6' + '.h5'
# day = 'Day2017118'
# h5dat = h5py.File(h5path, 'r')
# raw_ds = np.copy(h5dat[day + '/raw_data'])
# licks_ds = np.copy(h5dat[day + '/licks_pre_reward'])
# rewards_ds = np.copy(h5dat[day + '/rewards'])
# h5dat.close()
# # adjust trial number in rewards dataset
# rewards_ds[:,3] = rewards_ds[:,3] - 1
# # calculate success rate
# sr = np.where(rewards_ds[:,5] == 1)[0].shape[0] / rewards_ds.shape[0]
# # calculate smz
# session_smz = smz( raw_ds, licks_ds, rewards_ds, sr, rz, default, shuffles, tracklength )
# print(session_smz)

for i,mouse in enumerate(m):
    for j,day in enumerate(days_training_stage2[mouse]):
        # load data'

        h5path = 'E:\\MTH3_data\\MTH3_data\\animals_mus\\' + mouse + '\\' + mouse + '.h5'
        h5dat = h5py.File(h5path, 'r')
        raw_ds = np.copy(h5dat[day + '/raw_data'])
        licks_ds = np.copy(h5dat[day + '/licks_pre_reward'])
        rewards_ds = np.copy(h5dat[day + '/rewards'])
        h5dat.close()
        # adjust trial number in rewards dataset
        rewards_ds[:,3] = rewards_ds[:,3] - 1
        # calculate success rate
        sr = np.where(rewards_ds[:,5] == 1)[0].shape[0] / rewards_ds.shape[0]
        # calculate smz
        session_smz, sr_shuffled = smzscore( raw_ds, licks_ds, rewards_ds, sr, rz, default, shuffles, tracklength )
        days_smz[i,j] = session_smz

print(days_smz)
# create figure
fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.set_ylabel('Z-score')
ax1.set_xlabel('Session')
ax1.axhline(1,lw=2,c='0.75',ls='--')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params( \
    reset='on',
    axis='both', \
    direction='in', \
    length=4, \
    right='off', \
    top='off')

for smz_m in days_smz:
    ax1.plot(smz_m, c='0.5', lw=1)

ax1.plot(np.nanmean(days_smz,0),c='k',lw=2)


plt.tight_layout()
fname = loc_info['figure_output_path'] + 'mus2_smz_days.' + fformat
plt.show()
fig.savefig(fname, format=fformat)
