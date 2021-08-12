# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:39:50 2021

Plot mouse location, attractor location

@author: lfisc
"""

import os, yaml
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from matplotlib.collections import LineCollection
from scipy.io import loadmat
# from scipy import stats
# plt.rcParams['svg.fonttype'] = 'none'

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.safe_load(f)
    
fformat = '.png'

def make_folder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

if __name__ == '__main__':
    plot_trial = 5
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104262' + os.sep + 'srug_cont' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    trialdata_filepath = loc_info['figure_output_path'] + os.sep + 'EC2 2104262' + os.sep + 'srug_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    