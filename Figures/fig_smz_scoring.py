"""
Plot principle of spatial modulation scoring

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


sys.path.append(loc_info['base_dir'] + 'Analysis')

from smzscore import smzscore
from fig_behavior_summary import fig_behavior_stage5

# output figure format
fformat = 'svg'
# set location of reward zone
rz = [320, 340]
# location of default reward and tracklength
default = 340
tracklength = 340
# number of shuffles for SMI calculation
shuffles = 1000

# h5path = 'E:\\MTH3_data\\MTH3_data\\animals_mus\\LF171016_3\\LF171016_3.h5'
h5path = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170222_1\\LF170222_1.h5'
day = 'Day201776'
day = 'Day20170403'
h5dat = h5py.File(h5path, 'r')
raw_ds = np.copy(h5dat[day + '/raw_data'])
licks_ds = np.copy(h5dat[day + '/licks_pre_reward'])
rewards_ds = np.copy(h5dat[day + '/rewards'])
h5dat.close()

fig_behavior_stage5(h5path,day, 'LF170222_1'+day, fformat, 'testplots', False)
# adjust trial number in rewards dataset
rewards_ds[:,3] = rewards_ds[:,3] - 1
# calculate success rate
sr = np.where(rewards_ds[:,5] == 1)[0].shape[0] / rewards_ds.shape[0]
# calculate SMI
session_smi, shuffled_sr = smzscore( raw_ds, licks_ds, rewards_ds, sr, rz, default, shuffles, tracklength )
shuffled_mean = np.mean(shuffled_sr)
shuffled_std = np.std(shuffled_sr)
zscore = (sr - shuffled_mean) / shuffled_std
# create figure
fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.set_ylabel('count')
ax1.set_xlabel('Fraction successful trials')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params( \
    reset='on',
    axis='both', \
    direction='in', \
    length=4, \
    right='off', \
    top='off')

ax1.hist(shuffled_sr, bins=10, range=[0,1], facecolor='0.7', lw=0)
# sns.distplot(shuffled_sr)
ax1.axvline(sr, c='r', lw=2)
ax1.axvline(shuffled_mean, c='k', lw=2)
ax1.axvline(shuffled_mean + (3*shuffled_std), c='0.5', ls='--', lw=2)

fig.suptitle(zscore)

plt.tight_layout()
fname = loc_info['figure_output_path'] + os.sep + 'testplots' + os.sep + 'LF170222_1'+day+'_SMZ_novice.' + fformat
# plt.show()
fig.savefig(fname, format=fformat)
print(fname)
