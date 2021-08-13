""" Sandbox for testing functions, should be on .gitignore """

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys
sys.path.append("./Analysis")
import seaborn as sns
sns.set_style('white')

# load local settings file
import matplotlib
import numpy as np
import warnings; warnings.simplefilter('ignore')
from filter_trials import filter_trials
from scipy import stats
import yaml
import h5py
import json

import os
with open('./loc_settings.yaml', 'r') as f:
            content = yaml.load(f)

session = 'Day2018424_openloop_2'
mouse = 'LF180119_1'
h5path = content['imaging_dir'] + mouse + '/' + mouse + '.h5'
h5dat = h5py.File(h5path, 'r+')
behav_ds = np.copy(h5dat[session + '/behaviour_aligned'])
raw_behav_ds = np.copy(h5dat[session + '/raw_data'])
dF_ds = np.copy(h5dat[session + '/dF_win'])

#print(raw_behav_ds[0:-1,0]-raw_behav_ds[0,0])

init_offset = (np.sum(raw_behav_ds[:,2]))-(raw_behav_ds[-1,0]-raw_behav_ds[0,0])
print(np.sum(raw_behav_ds[:,2]), raw_behav_ds[-1,0], raw_behav_ds[0,0])
#print(np.cumsum(raw_behav_ds[:,2]))

start_frame = 300
stop_frame = 310
compare_frame = 305


fig = plt.figure()
plt.plot(raw_behav_ds[start_frame:stop_frame,0]-raw_behav_ds[0,0])
plt.plot(np.cumsum(raw_behav_ds[start_frame:stop_frame,2])+np.sum(raw_behav_ds[:start_frame,2])-init_offset)
fname = content['figure_output_path'] + 'time_adjust.png'
fig.savefig(fname, format='png')

print((np.sum(raw_behav_ds[:compare_frame,2])-init_offset) - (raw_behav_ds[compare_frame,0]-raw_behav_ds[0,0]))

if False:
    init_offset = (np.sum(raw_behav_ds[:,2]))-(raw_behav_ds[-1,0]-raw_behav_ds[0,0])

    behaviour_aligned = np.copy(raw_behav_ds)
    if init_offset < 0.05:
        print('adjusting timestamps without init offset')
        behaviour_aligned[:,0] = np.cumsum(raw_behav_ds[:,2])
    else:
        print('adjusting timestamps with init offset')
        behaviour_aligned[:,0] = np.cumsum(raw_behav_ds[:,2]) + raw_behav_ds[0,0] - init_offset

    #print(np.cumsum(raw_behav_ds[:,2]))
    # h5dat = h5py.File(h5path, 'r+')

    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset(session + '/' + 'raw_test',
                             data=behaviour_aligned, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat[session + '/' + 'raw_test']
        h5dat.create_dataset(session + '/' + 'raw_test',
                             data=behaviour_aligned, compression='gzip')

    h5dat.flush()
    h5dat.close()
