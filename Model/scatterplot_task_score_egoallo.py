#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:36:14 2021

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

TRIAL_THRESHOLD = 0

file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + fname + ".mat"
data = sp.io.loadmat(file_path)

naive = [('LF191022_1','20191115'),('LF191022_3','20191113'),('LF191023_blue','20191119'),('LF191022_2','20191116'),('LF191023_blank','20191114'),('LF191024_1','20191114')]
# expert = [('LF191022_1','20191209'),('LF191022_3','20191207'),('LF191023_blue','20191208'),('LF191022_2','20191210'),('LF191023_blank','20191210'),('LF191024_1','20191210')]
expert = [('LF191022_1','20191204'),('LF191022_2','20191210'),('LF191022_3','20191207'),('LF191023_blank','20191206'),('LF191023_blue','20191204'),('LF191024_1','20191204')]
all_sessions = [('LF191022_1','20191114'),
       ('LF191022_1','20191115'),
       ('LF191022_1','20191121'),
       ('LF191022_1','20191125'),
       ('LF191022_1','20191204'),
       ('LF191022_1','20191207'),
       ('LF191022_1','20191209'),
       ('LF191022_1','20191211'),
       ('LF191022_1','20191213'),
       ('LF191022_1','20191215'),
       ('LF191022_1','20191217'),
       ('LF191022_2','20191114'),
       ('LF191022_2','20191116'),
       ('LF191022_2','20191121'),
       ('LF191022_2','20191204'),
       ('LF191022_2','20191206'),
       ('LF191022_2','20191208'),
       ('LF191022_2','20191210'),
       ('LF191022_2','20191212'),
       ('LF191022_2','20191216'),
       ('LF191022_3','20191113'),
       ('LF191022_3','20191114'),
       ('LF191022_3','20191119'),
       ('LF191022_3','20191121'),
       ('LF191022_3','20191204'),
       ('LF191022_3','20191207'),
       ('LF191022_3','20191210'),
       ('LF191022_3','20191211'),
       ('LF191022_3','20191215'),
       ('LF191022_3','20191217'),
       ('LF191023_blank','20191114'),
       ('LF191023_blank','20191116'),
       ('LF191023_blank','20191121'),
       ('LF191023_blank','20191206'),
       ('LF191023_blank','20191208'),
       ('LF191023_blank','20191210'),
       ('LF191023_blank','20191212'),
       ('LF191023_blank','20191213'),
       ('LF191023_blank','20191216'),
       ('LF191023_blank','20191217'),
       ('LF191023_blue','20191113'),
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
       ('LF191023_blue','20191217'),
        ('LF191024_1','20191114'),
        ('LF191024_1','20191115'),
        ('LF191024_1','20191121'),
        ('LF191024_1','20191204'),
        ('LF191024_1','20191207'),
        ('LF191024_1','20191210')
       ]

print("------ NAIVE --------")

tscore_naive = []
egoallo_naive = []
ntrials_naive = []
for animal,session in naive:
    print(animal,session,data[animal + '_' + session])
    if data[animal + '_' + session][0][1] > TRIAL_THRESHOLD:
        tscore_naive.append(data[animal + '_' + session][0][0])
        egoallo_naive.append(data[animal + '_' + session][0][2])
        ntrials_naive.append(data[animal + '_' + session][0][1])

print("------ EXPERT --------")

tscore_expert = []
egoallo_expert = []
ntrials_expert = []
for animal,session in expert:
    print(animal,session, data[animal + '_' + session])
    if data[animal + '_' + session][0][1] > TRIAL_THRESHOLD:
        tscore_expert.append(data[animal + '_' + session][0][0])
        egoallo_expert.append(data[animal + '_' + session][0][2])
        ntrials_expert.append(data[animal + '_' + session][0][1])
    
print("------ ALL --------")    

tscore_all = []
egoallo_all = []
n_trials = []
for animal,session in all_sessions:
    print(animal,session, data[animal + '_' + session])
    if data[animal + '_' + session][0][1] > TRIAL_THRESHOLD:
        tscore_all.append(data[animal + '_' + session][0][0])
        egoallo_all.append(data[animal + '_' + session][0][2])
        n_trials.append(data[animal + '_' + session][0][1])
    
fig = plt.figure(figsize=(3,20))
(ax,ax2,ax3,ax4) = fig.subplots(4,1)
n_animals_naive = len(tscore_naive)
n_animals_expert = len(tscore_expert)
ax.scatter(np.zeros((n_animals_naive,1)), egoallo_naive, c='0.5')
ax.scatter(np.ones((n_animals_expert,1)), egoallo_expert, c='0.5')


_,p_ttest = sp.stats.ttest_rel(egoallo_naive,egoallo_expert)

ax.set_xlim([-0.2,2])
ax.set_ylim([0.4,1])

if n_animals_naive == n_animals_expert:
    for i in range(n_animals_naive):
        ax.plot([0,1], [egoallo_naive[i], egoallo_expert[i]], c='0.7')
    # ax.plot([0,1], [np.mean(egoallo_naive), np.mean(egoallo_expert)], c='k')
ax.plot([0,1], [np.mean(egoallo_naive), np.mean(egoallo_expert)], marker='o', c='k', lw=3)
ax.set_xlim([-0.1,2.1])

ax2.scatter(tscore_naive, egoallo_naive, color='0.5')
ax2.scatter(tscore_expert, egoallo_expert, color='0.5')
ax2.set_ylim([0.4,1])
ax2.set_xticks([-30,0,30,60,90])
ax2.set_xticklabels(['-30','0','30','60','90'])
corr_ne,p_ne = sp.stats.spearmanr(np.hstack((tscore_naive,tscore_expert)), np.hstack((egoallo_naive,egoallo_expert)))

ax3.scatter(tscore_all, egoallo_all, c='0.5')
ax3.set_ylim([0.3,1])
corr,p = sp.stats.spearmanr(tscore_all, egoallo_all)
ax3.set_ylim([0.3,1])
ax3.set_xticks([-30,0,30,60,90])
ax3.set_xticklabels(['-30','0','30','60','90'])

ax4.scatter(n_trials, egoallo_all)
corr_nt,p_nt = sp.stats.spearmanr(n_trials, egoallo_all)

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
    
ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=2, \
        width=1, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    
ax3.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=2, \
        width=1, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

print("------ STATS --------")
print("naive vs. expert: " + str(sp.stats.ttest_rel(egoallo_naive, egoallo_expert)))
print("naive vs. expert Spearman: " + str(sp.stats.spearmanr(tscore_all, egoallo_all)))
print("---------------------")
    

fig.savefig("C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\manuscript\\Figure 1\\egoallo_scatterplot.svg", format='svg')
print("saved" + file_path + "_fig.svg")