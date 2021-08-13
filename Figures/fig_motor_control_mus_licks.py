"""
Plot first lick location during control and muscimol condition in motor control experiment


"""
%reset -f -s
%matplotlib inline
%load_ext autoreload
%autoreload

import numpy as np
import h5py
import warnings
import sys
import os
import yaml
sys.path.append("./Analysis")


from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
from filter_trials import filter_trials

sns.set_style('white')

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

# open HDF5-file, and gather initial folder structure information to populate dropdown menu
m = ['FB170208','FB170210','FB170217','NM170216']

#m = ['FB170210','FB170216','FB170217','NM170216']

days_ACSF = {'FB170208':['20170425','20170427'],
             'FB170210':['20170425','20170427'],
             'FB170217':['20170424','20170426'],
             'NM170216':['20170425','20170427'],
            }

days_MUS =  {'FB170208':['20170424','20170426'],
             'FB170210':['20170424','20170426'],
             'FB170217':['20170425','20170427'],
             'NM170216':['20170424','20170426'],
            }

fld_MUS = []
fld_ACSF = []

# load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous
for mouse in m:
    # load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous
    # for each mice, concatenate all licks from all days of ACSF condition
    for da in days_ACSF[mouse]:
        print(da, mouse)
        fl_acsf_ds = []
        h5dat = h5py.File('/Users/lukasfischer/Dropbox (MIT)/MTH3/MTH3control-edited.h5', 'r')
        licks_ACSF = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
        raw_ACSF = np.copy(h5dat['/Day' + da + '/'+ mouse + '/raw_data'])
        h5dat.close()


        licks_ACSF=licks_ACSF[licks_ACSF[:,3]!=5]
        # determine location of first licks - ACSF
        first_lick_short_ACSF = []
        first_lick_short_trials_ACSF = []
        first_lick_long_ACSF = []
        first_lick_long_trials_ACSF = []
        for r in np.unique(licks_ACSF[:,2]):
            start_location = raw_ACSF[raw_ACSF[:,6]==r][0][1]
            licks_trial = licks_ACSF[licks_ACSF[:,2]==r,:]
            licks_trial = licks_trial[licks_trial[:,1]>(start_location+10),:]
            if licks_trial.shape[0]>0:
                if licks_trial[0,3] == 7:
                    first_lick_short_ACSF.append(licks_trial[0,1])
                    first_lick_short_trials_ACSF.append(r)
                elif licks_trial[0,3] == 8:
                    first_lick_long_ACSF.append(licks_trial[0,1])
                    first_lick_long_trials_ACSF.append(r)
        fl_acsf_ds.append(np.median(first_lick_long_ACSF) - np.median(first_lick_short_ACSF))
    fld_ACSF.append(np.mean(fl_acsf_ds))

    # load MUSCIMOL datasets - when datasets are appended, re-enumerate trial numbers to be continuous
     # for each mice, concante all licks from all days of Muscimol condition
    for da in days_MUS[mouse]:
        fl_mus_ds = []
        h5dat = h5py.File('/Users/lukasfischer/Dropbox (MIT)/MTH3/MTH3control-edited.h5', 'r')
        licks_MUS = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
        raw_MUS = np.copy(h5dat['/Day' + da + '/'+ mouse + '/raw_data'])
        h5dat.close()

        licks_MUS=licks_MUS[licks_MUS[:,3]!=5]
        # determine location of first licks - MUSCIMOL
        first_lick_short_MUS = []
        first_lick_short_trials_MUS = []
        first_lick_long_MUS = []
        first_lick_long_trials_MUS = []

        for r in np.unique(licks_MUS[:,2]):
            start_location = raw_MUS[raw_MUS[:,6]==r][0][1]
            licks_trial = licks_MUS[licks_MUS[:,2]==r,:]
            licks_trial = licks_trial[licks_trial[:,1]>(start_location+0),:]
            if licks_trial.shape[0]>0:
                if licks_trial[0,3] == 7:
                    first_lick_short_MUS.append(licks_trial[0,1])
                    first_lick_short_trials_MUS.append(r)
                elif licks_trial[0,3] == 8:
                    first_lick_long_MUS.append(licks_trial[0,1])
                    first_lick_long_trials_MUS.append(r)
        fl_mus_ds.append(np.median(first_lick_long_MUS) - np.median(first_lick_short_MUS))
    fld_MUS.append(np.mean(fl_mus_ds))
print(fld_ACSF, fld_MUS)



fformat = 'svg'
# create figure to later plot on
fig = plt.figure(figsize=(2,4))
ax1 = plt.subplot(111)
ax1.plot([fld_ACSF,fld_MUS],c='0.7',lw=2,zorder=1)
ax1.scatter([0,0,0,0],fld_ACSF, s=100, c='k',zorder=2)
ax1.scatter([1,1,1,1],fld_MUS, s=100, c='w',zorder=2)
#
ax1.set_ylim([0,70])
ax1.set_xlim([-0.2,1.2])
ax1.set_xticks([0,1])
ax1.set_xticklabels(['ACSF','MUSCIMOL'],rotation=45)

fname = loc_info['figure_output_path'] + 'mus_motor' + os.sep + 'mus_motor_comp' + '.' + fformat

fig.savefig(fname, format=fformat)


plt.show()
