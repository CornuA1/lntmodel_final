#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:13:59 2021

@author: lukasfischer
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
plt.rcParams['svg.fonttype'] = 'none'
from scipy import stats
import seaborn as sns
import scipy.io as sio

sns.set_style("white")

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
#
# with open(loc_info['yaml_file'], 'r') as f:
#     project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis\\')
sys.path.append(loc_info['base_dir'] + 'Figures')

from filter_trials import filter_trials
from load_behavior_data import load_data
from rewards import rewards
from licks import licks_nopost as licks
from scipy.signal import butter, filtfilt
import fig_behavior_stage5

SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'

if __name__ == '__main__':
    
    fformat = 'png'
    MOUSE = 'LF191022_1'
    SESSION = [['20191203','MTH3_vr3_s5r2_2019123_1713.csv']]
    data_path = loc_info['raw_dir'] + MOUSE
    
    for s in SESSION:
        print(data_path)
        fig_behavior_stage5.fig_behavior_stage5(data_path, s,  MOUSE+s[0], fformat, MOUSE, True)