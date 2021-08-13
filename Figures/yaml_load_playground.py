"""
Playground to load stuff from the MTH3_active.yaml file

"""

import numpy as np
import h5py
import warnings
import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

with open(loc_info['yaml_file'], 'r') as f:
    project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')

from load_h5_data import load_h5_data
from load_h5_data import load_h5_dslist
from load_h5_data import load_recinfo

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    mice = []
    # behav_collection, dF_collection, rec_info = load_h5_data(['GCAMP6f_A30_ALL'], [['mouse', 'LF170110_2'], ['rectype', 'REGULAR']])
    rec_info = load_recinfo(['GCAMP6f_A30_ALL'], [['rectype','REGULAR'],['level','SOMA_L5']])

    num_ds = len(rec_info[0])
    for i in range(num_ds):
        mice.append([rec_info[0][i],'Day' + str(rec_info[1][i]),rec_info[4][i]])
    sum_rois = 0
    for m in mice:
        print(m)
        sum_rois += len(m[2])
