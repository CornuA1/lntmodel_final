"""
Script to fix reward datasets (for those datasets that have been imported with
the old version of the rewards script, i.e. before 18/08/04)

"""
%load_ext autoreload
%autoreload
%matplotlib inline

import numpy as np
import csv
import sys
import os
import yaml
import h5py

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

with open(loc_info['yaml_file'], 'r') as f:
    project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')

from rewards import rewards_legacy, rewards

MOUSE = 'LF180514_1'
SESSION = ['Day2018726','Day2018727','Day2018728','Day2018729','Day2018730']
h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
for s in SESSION:
    h5dat = h5py.File(h5path, 'r+')
    raw_ds = np.copy(h5dat[s + '/raw_data'])
    rews2 = rewards(raw_ds)
    del h5dat[s + '/' + 'rewards']
    h5dat.create_dataset(s + '/' + 'rewards',data=rews2, compression='gzip')
    h5dat.close()
