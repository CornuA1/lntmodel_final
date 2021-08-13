#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:03:10 2018

@author: katya
"""
#parse roi classification file for sessions that have behavior aligned data
def parse_record_file(included_recordings):
    import matplotlib
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("./Analysis")
    import pickle
    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal
    import statsmodels.api as sm
    import yaml
    import pandas as pd
    import h5py
    import json
    import seaborn as sns
    sns.set_style('white')
    import os
    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)
    
    good_recordings = []
    for r in included_recordings:
        # load individual dataset
        print(r)
        mouse = r[0]
        # len > 2 indicates that we want to use a classification file in r[2], but plotting the data for r[1]
        
        h5path = content['imaging_dir'] + mouse + '/' + mouse + '.h5'
        h5dat = h5py.File(h5path, 'r')
        good_rec = []
        good_rec.append(r[0])
        for session in h5dat.keys():
            count = 0
            if '_' in session:
                continue
            for name in h5dat[session].items():
               if 'behaviour_aligned' in name[0]:
                   count = count + 1
               if 'dF_win' in name[0]:
                   count = count + 1
            if count == 2:
                good_rec.append(session)
        good_recordings.append(good_rec)         
        h5dat.close()
    
    recordings = {
            'good_recordings': good_recordings
            
            }
    with open(content['figure_output_path'] + os.sep + 'recordings_with_behav_inc420.json','w+') as f:
        json.dump(recordings,f)

if __name__ == "__main__":
    figure_datasets = [['LF170222_1'], ['LF170110_2'], ['LF170420_1'], ['LF170421_2'], ['LF170613_1']]
    parse_record_file(figure_datasets)