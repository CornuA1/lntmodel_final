#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:35:20 2021

@author: lukasfischer
"""

import csv, os, yaml, warnings
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
sns.set_style("white")
warnings.filterwarnings('ignore')

# load yaml file with local filepaths
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
    
fname = "total_analysis"

TRIAL_THRESHOLD = 50

file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + fname + ".mat"
data = sp.io.loadmat(file_path)

all_sessions = [[('LF191022_1','20191114'),
       ('LF191022_1','20191115'),
       ('LF191022_1','20191121'),
       ('LF191022_1','20191125'),
       ('LF191022_1','20191204'),
       ('LF191022_1','20191207'),
       ('LF191022_1','20191209'),
       ('LF191022_1','20191211'),
       ('LF191022_1','20191213'),
       ('LF191022_1','20191215'),
       ('LF191022_1','20191217')],
       [('LF191022_2','20191114'),
       ('LF191022_2','20191116'),
       ('LF191022_2','20191121'),
       ('LF191022_2','20191204'),
       ('LF191022_2','20191206'),
       ('LF191022_2','20191208'),
       ('LF191022_2','20191210'),
       ('LF191022_2','20191212'),
       ('LF191022_2','20191216')],
       [('LF191022_3','20191113'),
       ('LF191022_3','20191114'),
       ('LF191022_3','20191119'),
       ('LF191022_3','20191121'),
       ('LF191022_3','20191204'),
       ('LF191022_3','20191207'),
       ('LF191022_3','20191210'),
       ('LF191022_3','20191211'),
       ('LF191022_3','20191215'),
       ('LF191022_3','20191217')],
       [('LF191023_blank','20191114'),
       ('LF191023_blank','20191116'),
       ('LF191023_blank','20191121'),
       ('LF191023_blank','20191206'),
       ('LF191023_blank','20191208'),
       ('LF191023_blank','20191210'),
       ('LF191023_blank','20191212'),
       ('LF191023_blank','20191213'),
       ('LF191023_blank','20191216'),
       ('LF191023_blank','20191217')],
       [('LF191023_blue','20191113'),
       ('LF191023_blue','20191114'),
       ('LF191023_blue','20191119'),
       ('LF191023_blue','20191121'),
       ('LF191023_blue','20191125'),
       ('LF191023_blue','20191204'),
       ('LF191023_blue','20191206'),
       ('LF191023_blue','20191208'),
       ('LF191023_blue','20191210'),
       ('LF191023_blue','20191212'),
       ('LF191023_blue','20191215'),
       ('LF191023_blue','20191217')],
        [('LF191024_1','20191114'),
        ('LF191024_1','20191115'),
        ('LF191024_1','20191121'),
        ('LF191024_1','20191204'),
        ('LF191024_1','20191207'),
        ('LF191024_1','20191210')]]

tscore_all = []
egoallo_all = []
n_trials = []
for animal in all_sessions:
    animal_tscore = []
    for session in animal:
        print(session[0],session[1],data[session[0] + '_' + session[1]][0][0])
        if data[session[0] + '_' + session[1]][0][1] > TRIAL_THRESHOLD:
            animal_tscore.append(data[session[0] + '_' + session[1]][0][0])
        #     egoallo_all.append(data[animal + '_' + session][0][2])
        #     n_trials.append(data[animal + '_' + session][0][1])
    tscore_all.append(animal_tscore)
    
fig = plt.figure(figsize=(5,5))
ax = fig.subplots(1,1)

max_sess = 0
for tsc in tscore_all:
    ax.plot(tsc, c='0.7')
    if len(tsc) > max_sess:
        max_sess = len(tsc)
        
mean_sess = np.full((6,max_sess), np.nan)
for i,tsc in enumerate(tscore_all):
    mean_sess[i,0:len(tsc)] = tsc
    
mean_trace = np.nanmean(mean_sess, axis=0)
ax.plot(mean_trace, lw=3, c='k')

sns.despine(top=True, right=True, left=False, bottom=False)
ax.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=2, \
        width=1, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    
fig.savefig(file_path + "learning_fig.svg", format='svg')
print("saved" + file_path + "learning_fig.svg")