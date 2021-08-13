# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:52:57 2021

@author: lfisc
"""

import os, sys, yaml, matplotlib
# from multiprocessing import Process
import warnings; # warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import matplotlib.cbook
import numpy as np
import scipy.io as sio
import seaborn as sns
sns.set_style("white")

plt.rcParams['svg.fonttype'] = 'none'

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + 'Analysis')
sys.path.append(loc_info['base_dir'] + 'Imaging')
FIGURE_OUTPUT_DIRECTORY = loc_info['figure_output_path']

from filter_trials import filter_trials
from write_dict import write_dict
from event_ind import event_ind
from rewards import rewards
from licks import licks


def run_for_mouse(MOUSE, sessions):
    
    data_path = loc_info['imaging_dir']
    base_path = loc_info['raw_dir']
    for s in sessions:
        SESSION = '2019'+str(s)
        use_data = MOUSE + '_' + SESSION
        
    #     SESSION_OPENLOOP = ''
    #     NUM_ROIS = 'all' # 'all' #52
    #     json_path = base_path + os.sep + MOUSE + os.sep + SESSION + '.json'
    #     # dictionary that will hold the results of the analyses
    #     roi_result_params = {
    #         'mouse_session' : MOUSE+'_'+SESSION,
    #         'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    #     }
    #     #
        SUBNAME = 'space'
        align_point = 'landmark'
        subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
        roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, data_path, json_path, SUBNAME, align_point, subfolder, roi_result_params, False, False, False, use_data, '')


if __name__ == '__main__':
    run_for_mouse('LF191022_1', [1204])
    print('done')