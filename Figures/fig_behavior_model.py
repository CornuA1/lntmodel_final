#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:13:59 2021

@author: lukasfischer
"""

import numpy as np
# import h5py
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

sys.path.append(loc_info['base_dir'] + 'Analysis')

from filter_trials import filter_trials
from load_behavior_data import load_data
from rewards import rewards
from licks import licks_nopost as licks
from scipy.signal import butter, filtfilt

SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'

REWARDZONE_COLOR = '#1C75BC'
BLACKBOX_COLOR = 'k'

REWARD_ZONE_LENGTH = 40

def make_folder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

def load_raw_data(raw_filename, sess):
#    raw_data = load_data(raw_filename, 'vr')
    raw_data = np.genfromtxt(raw_filename, delimiter=';')
    all_licks = licks(raw_data)
    if len(all_licks) > 0:
        trial_licks = all_licks[np.in1d(all_licks[:, 3], [3, 4]), :]
    else:
        trial_licks = np.zeros((0,3))
    reward =  rewards(raw_data)
    return raw_data, trial_licks, reward

def line_of_trials(fformat, MOUSE, SESSION, data_path, bar_trials):
    # load data
    raw_filename = data_path + os.sep + SESSION[0] + os.sep + SESSION[1]
    save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + MOUSE + os.sep + MOUSE+'_'+SESSION[0]
    raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, SESSION[0])

    # create figure
    fig = plt.figure(figsize=(10,2))
    fig.suptitle(MOUSE+SESSION[0])
    ax = plt.axes()

    # plot_licks = filter_trials( raw_ds, [], ['trialnr_range',142,147])

    # plot_licks_x = licks_ds[licks_ds[:,2]==r,1]

    ax.set_xlim(0,500)
    
    tot_distance = 0
    for i in bar_trials:
        print(i)
        raw_trial = raw_ds[raw_ds[:,6]==i,:]
        plot_licks_x = licks_ds[licks_ds[:,2]==i,1]
        plot_licks_x = plot_licks_x[plot_licks_x > 105]
        # print(np.unique(raw_trial[:,6]), np.unique(raw_trial[:,4]), raw_trial[0,1], raw_trial[-1,1])
        
        if raw_trial[0,4] != 5:
            trial_init_coord = raw_trial[0,1]
            trial_final_coord = raw_trial[-1,1]
            
            trial_start = tot_distance
            trial_end = tot_distance + trial_final_coord - trial_init_coord
            landmark_start = tot_distance + 200 - trial_init_coord
            landmark_end = tot_distance + 240 - trial_init_coord
            rzone_start = trial_end
            rzone_end = trial_end + REWARD_ZONE_LENGTH
            
            ax.axvspan(tot_distance, landmark_start, color='0.8', zorder=5)
            ax.axvspan(landmark_start,landmark_end , color='#EC008C', zorder=5)
            ax.axvspan(landmark_end, trial_end, color='0.8', zorder=5)
            ax.axvspan(rzone_start, rzone_end, color='#1C75BC', zorder=5)
            ax.scatter(plot_licks_x - trial_init_coord + tot_distance + 5, np.zeros((plot_licks_x.shape[0],1)), marker='o', c='k', zorder=6)
            
            tot_distance = tot_distance + trial_final_coord - trial_init_coord + 40
            print(tot_distance, trial_end, raw_trial[0,4])
        else:
            ax.axvspan(tot_distance, tot_distance + 100, color='k', zorder=5)
            print(tot_distance, tot_distance + 100, raw_trial[0,4])
            tot_distance = tot_distance + 100

    ax.set_xlim(0,tot_distance)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(save_filename + "trialsample_fig.svg", format='svg')
    print("saved " + save_filename + "trialsample_fig.svg")

def all_trials_lm(fformat, MOUSE, SESSION, data_path):
    """ Plot behavior aligned to landmark """
    # load data
    raw_filename = data_path + os.sep + SESSION[0] + os.sep + SESSION[1]
    save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + MOUSE + os.sep + MOUSE+'_'+SESSION[0]
    raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, SESSION[0])

    max_trial = raw_ds[-1,6]
    if raw_ds[-1,4] == 5:
        max_trial = 1-max_trial
        
    bar_trials = range(2,int(max_trial))

    # create figure
    fig = plt.figure(figsize=(10,5))
    fig.suptitle(MOUSE+SESSION[0])
    ax = plt.axes()
    
    ax.set_xlim([0,450])
    ax.set_ylim([0,1])
    
    trial_height = 1/len(bar_trials) * 2
    yfloor = 0
    
    for i in bar_trials:
        raw_trial = raw_ds[raw_ds[:,6]==i,:]
        plot_licks_x = licks_ds[licks_ds[:,2]==i,1]
        plot_licks_x = plot_licks_x[plot_licks_x > 105]
        reward_x = reward_ds[reward_ds[:,3]==i,1]
        if reward_ds[reward_ds[:,3]==i,5] == 1:
            reward_color = '#FFFF00'
        else:
            reward_color = '#FF0000'
            
        if raw_trial[0,4] != 5:
            trial_init_coord = raw_trial[0,1]
            trial_final_coord = raw_trial[-1,1]
            
            trial_start = raw_trial[0,1]
            trial_end_short = 320
            trial_end_long = 380
            landmark_start = 200
            landmark_end = 240

            # print(raw_trial[0,4])
            if raw_trial[0,4] == 3:
                ax.axvspan(0, trial_start, yfloor, yfloor+trial_height, color=BLACKBOX_COLOR, zorder=5)
                ax.axvspan(trial_start, landmark_start, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(landmark_start,landmark_end, yfloor, yfloor+trial_height , color='#EC008C', zorder=5)
                ax.axvspan(landmark_end, trial_end_short, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(trial_end_short, trial_end_short+REWARD_ZONE_LENGTH, yfloor, yfloor+trial_height, color=REWARDZONE_COLOR, zorder=5)
                # ax.axvspan(trial_end_short+40, 450, yfloor, yfloor+trial_height, color='#474747', zorder=5)
                
                ax.scatter(plot_licks_x, np.ones((plot_licks_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c='k', zorder=6)
                ax.scatter(reward_x, np.ones((reward_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c=reward_color, zorder=6)
                
            elif raw_trial[0,4] == 4:
                ax.axvspan(0, trial_start, yfloor, yfloor+trial_height, color=BLACKBOX_COLOR, zorder=5)
                ax.axvspan(trial_start, landmark_start, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(landmark_start,landmark_end, yfloor, yfloor+trial_height , color='#ED7EC6', zorder=5)
                ax.axvspan(landmark_end, trial_end_long, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(trial_end_long, trial_end_long+REWARD_ZONE_LENGTH, yfloor, yfloor+trial_height, color=REWARDZONE_COLOR, zorder=5)
                # ax.axvspan(trial_end_long+40, 450, yfloor, yfloor+trial_height, color='#474747', zorder=5)
                
                ax.scatter(plot_licks_x, np.ones((plot_licks_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c='k', zorder=6)
                ax.scatter(reward_x, np.ones((reward_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c=reward_color, zorder=6)
                
            yfloor = yfloor+trial_height
    
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.savefig(save_filename + "alltrials_lm_fig.svg", format='svg')
    print("saved " + save_filename + "alltrials_lm_fig.svg")

def all_trials_ts(fformat, MOUSE, SESSION, data_path):
    """ Plot behavior aligned to trial start """
    # load data
    raw_filename = data_path + os.sep + SESSION[0] + os.sep + SESSION[1]
    save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + MOUSE + os.sep + MOUSE+'_'+SESSION[0]
    raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, SESSION[0])

    max_trial = raw_ds[-1,6]
    if raw_ds[-1,4] == 5:
        max_trial = 1-max_trial
        
    bar_trials = range(2,int(max_trial))

    # create figure
    fig = plt.figure(figsize=(10,5))
    fig.suptitle(MOUSE+SESSION[0])
    ax = plt.axes()
    
    ax.set_xlim([0,450])
    ax.set_ylim([0,1])
    
    trial_height = 1/len(bar_trials) * 2
    yfloor = 0
    
    for i in bar_trials:
        raw_trial = raw_ds[raw_ds[:,6]==i,:]
        plot_licks_x = licks_ds[licks_ds[:,2]==i,1]
        plot_licks_x = plot_licks_x[plot_licks_x > 105]
        reward_x = reward_ds[reward_ds[:,3]==i,1]
        if reward_ds[reward_ds[:,3]==i,5] == 1:
            reward_color = '#FFFF00'
        else:
            reward_color = '#FF0000'
            
        if raw_trial[0,4] != 5:
            trial_init_coord = raw_trial[0,1]
            trial_final_coord = raw_trial[-1,1]
            
            trial_start = 60
            trial_end_short = 320 - trial_init_coord + 75
            trial_end_long = 380 - trial_init_coord + 75
            landmark_start = 200 - trial_init_coord + 75
            landmark_end = 240 - trial_init_coord + 75

            # print(raw_trial[0,4])
            if raw_trial[0,4] == 3:
                ax.axvspan(0, trial_start, yfloor, yfloor+trial_height, color=BLACKBOX_COLOR, zorder=5)
                ax.axvspan(trial_start, landmark_start, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(landmark_start,landmark_end, yfloor, yfloor+trial_height , color='#EC008C', zorder=5)
                ax.axvspan(landmark_end, trial_end_short, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(trial_end_short, trial_end_short+REWARD_ZONE_LENGTH, yfloor, yfloor+trial_height, color=REWARDZONE_COLOR, zorder=5)
                # ax.axvspan(trial_end_short+40, 450, yfloor, yfloor+trial_height, color='#474747', zorder=5)
                
                plot_licks_x = plot_licks_x - trial_init_coord + 75
                reward_x = reward_x - trial_init_coord + 75
                ax.scatter(plot_licks_x, np.ones((plot_licks_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c='k', zorder=6)
                ax.scatter(reward_x, np.ones((reward_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c=reward_color, zorder=6)
                
            elif raw_trial[0,4] == 4:
                ax.axvspan(0, trial_start, yfloor, yfloor+trial_height, color=BLACKBOX_COLOR, zorder=5)
                ax.axvspan(trial_start, landmark_start, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(landmark_start,landmark_end, yfloor, yfloor+trial_height , color='#ED7EC6', zorder=5)
                ax.axvspan(landmark_end, trial_end_long, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(trial_end_long, trial_end_long+REWARD_ZONE_LENGTH, yfloor, yfloor+trial_height, color=REWARDZONE_COLOR, zorder=5)
                # ax.axvspan(trial_end_long+40, 450, yfloor, yfloor+trial_height, color='#474747', zorder=5)
                
                plot_licks_x = plot_licks_x - trial_init_coord + 75
                reward_x = reward_x - trial_init_coord + 75
                ax.scatter(plot_licks_x, np.ones((plot_licks_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c='k', zorder=6)
                ax.scatter(reward_x, np.ones((reward_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c=reward_color, zorder=6)
                
            yfloor = yfloor+trial_height
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.savefig(save_filename + "alltrials_ts_fig.svg", format='svg')
    print("saved " + save_filename + "alltrials_ts_fig.svg")

def all_trials_rw(fformat, MOUSE, SESSION, data_path):
    """ Plot behavior aligned to reward """
    # load data
    raw_filename = data_path + os.sep + SESSION[0] + os.sep + SESSION[1]
    save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + MOUSE + os.sep + MOUSE+'_'+SESSION[0]
    path_name = loc_info['figure_output_path'] + 'task_performance' + os.sep + MOUSE
    raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, SESSION[0])

    max_trial = raw_ds[-1,6]
    if raw_ds[-1,4] == 5:
        max_trial = max_trial-1
        
    bar_trials = range(2,int(max_trial))

    # create figure
    fig = plt.figure(figsize=(10,5))
    fig.suptitle(MOUSE+SESSION[0])
    ax = plt.axes()
    
    ax.set_xlim([0,450])
    ax.set_ylim([0,1])
    
    trial_height = 1/len(bar_trials) * 2
    yfloor = 0
    
    for i in bar_trials:
        raw_trial = raw_ds[raw_ds[:,6]==i,:]
        plot_licks_x = licks_ds[licks_ds[:,2]==i,1]
        plot_licks_x = plot_licks_x[plot_licks_x > 105]
        reward_x = reward_ds[reward_ds[:,3]==i,1]
        if reward_ds[reward_ds[:,3]==i,5] == 1:
            reward_color = '#FFFF00'
        else:
            reward_color = '#FF0000'
            
        if raw_trial[0,4] != 5:
            trial_init_coord = raw_trial[0,1]
            trial_final_coord = raw_trial[-1,1]
            
            orig_trial_end_short = 320
            orig_trial_end_long = 380
            
            short_reward_correction = 330 - orig_trial_end_short
            long_reward_correction = 330 - orig_trial_end_long
            
            trial_start_short = raw_trial[0,1] + short_reward_correction
            trial_start_long = raw_trial[0,1] + long_reward_correction
            
            trial_end_short = 320 + short_reward_correction
            trial_end_long = 380 + long_reward_correction
            
            landmark_start_short = 200 + short_reward_correction
            landmark_end_short = 240 + short_reward_correction
            
            landmark_start_long = 200 + long_reward_correction
            landmark_end_long = 240 + long_reward_correction
            
            

            # print(raw_trial[0,4])
            if raw_trial[0,4] == 3:
                ax.axvspan(0, trial_start_short, yfloor, yfloor+trial_height, color=BLACKBOX_COLOR, zorder=5)
                ax.axvspan(trial_start_short, landmark_start_short, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(landmark_start_short,landmark_end_short, yfloor, yfloor+trial_height , color='#EC008C', zorder=5)
                ax.axvspan(landmark_end_short, trial_end_short, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(trial_end_short, trial_end_short+REWARD_ZONE_LENGTH, yfloor, yfloor+trial_height, color=REWARDZONE_COLOR, zorder=5)
                # ax.axvspan(trial_end_short+40, 450, yfloor, yfloor+trial_height, color='#474747', zorder=5)
                
                plot_licks_x = plot_licks_x + short_reward_correction
                reward_x = reward_x + short_reward_correction
                ax.scatter(plot_licks_x, np.ones((plot_licks_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c='k', zorder=6)
                ax.scatter(reward_x, np.ones((reward_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c=reward_color, zorder=6)
                
            elif raw_trial[0,4] == 4:
                ax.axvspan(0, trial_start_long, yfloor, yfloor+trial_height, color=BLACKBOX_COLOR, zorder=5)
                ax.axvspan(trial_start_long, landmark_start_long, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(landmark_start_long,landmark_end_long, yfloor, yfloor+trial_height , color='#ED7EC6', zorder=5)
                ax.axvspan(landmark_end_long, trial_end_long, yfloor, yfloor+trial_height, color='0.8', zorder=5)
                ax.axvspan(trial_end_long, trial_end_long+REWARD_ZONE_LENGTH, yfloor, yfloor+trial_height, color=REWARDZONE_COLOR, zorder=5)
                # ax.axvspan(trial_end_long+40, 450, yfloor, yfloor+trial_height, color='#474747', zorder=5)
                
                plot_licks_x = plot_licks_x + long_reward_correction
                reward_x = reward_x + long_reward_correction
                ax.scatter(plot_licks_x, np.ones((plot_licks_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c='k', zorder=6)
                ax.scatter(reward_x, np.ones((reward_x.shape[0],1))*yfloor+(trial_height/2), marker='o', c=reward_color, zorder=6)
                
            yfloor = yfloor+trial_height
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if not os.path.isdir(path_name):
            os.mkdir(path_name)
    fig.savefig(save_filename + "alltrials_rw_fig.svg", format='svg')
    print("saved " + save_filename + "alltrials_rw_fig.svg")
    

def licks_per_trial(behavior_data_path, sessions_vr, sessions_ol):
    
    num_trials = 30
    
    licks_per_trial_vr = np.zeros((num_trials,len(sessions_vr)))
    licks_per_trial_ol = np.zeros((num_trials,len(sessions_ol)))
    correct_trial_vr = np.zeros((num_trials,len(sessions_vr)))
    correct_trial_ol = np.zeros((num_trials,len(sessions_ol)))
    for i in range(len(sessions_vr)):
        data_path = behavior_data_path + os.sep + sessions_vr[i][0]
        raw_filename = data_path + os.sep + sessions_vr[i][1] + os.sep + sessions_vr[i][2]
        save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_vr[i][0] + os.sep + sessions_vr[i][0]+'_'+sessions_vr[i][1]
        path_name = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_vr[i][0]
        raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sessions_vr[i][2])
        
        # determine max trial, if the last trial is the black box, remove it
        max_trial = raw_ds[-1,6]
        if raw_ds[-1,4] == 5:
            max_trial = max_trial-1
            
        # trial number range start from where we start calculating the average licks/trial
        trial_start = max_trial-(num_trials*2)
        
        # run through trials within range
        k=0
        for j in np.arange(trial_start,max_trial,1):
            raw_trial = raw_ds[raw_ds[:,6]==j,:]
            plot_licks_x = licks_ds[licks_ds[:,2]==j,1]
            plot_licks_x = plot_licks_x[plot_licks_x > 105]
            reward_x = reward_ds[reward_ds[:,3]==j,1]
            
            # if reward was triggered, add an extra lick
            rw_lick = 0
            if reward_ds[reward_ds[:,3]==j,5] == 1:
                rw_lick = 1
                
            if raw_trial[0,4] != 5:
                # keep track of number of licks in a given trial. Add a lick if reward was triggered
                licks_per_trial_vr[k,i] = plot_licks_x.shape[0] + rw_lick
                # keep track of whether a reward was triggered by the mouse or not
                correct_trial_vr[k,i] = rw_lick
                k = k+1
                
            
                
    for i in range(len(sessions_ol)):
        data_path = behavior_data_path + os.sep + sessions_ol[i][0]
        raw_filename = data_path + os.sep + sessions_ol[i][1] + os.sep + sessions_ol[i][2]
        save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_ol[i][0] + os.sep + sessions_ol[i][0]+'_'+sessions_ol[i][1]
        path_name = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_ol[i][0]
        raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sessions_ol[i][2])
        
        # determine max trial, if the last trial is the black box, remove it
        max_trial = raw_ds[-1,6]
        if raw_ds[-1,4] == 5:
            max_trial = max_trial-1
            
        # trial number range start from where we start calculating the average licks/trial
        trial_start = 1
        
        # run through trials within range
        k=0
        for j in np.arange(trial_start,num_trials*2,1):
            raw_trial = raw_ds[raw_ds[:,6]==j,:]
            plot_licks_x = licks_ds[licks_ds[:,2]==j,1]
            plot_licks_x = plot_licks_x[plot_licks_x > 105]
            reward_x = reward_ds[reward_ds[:,3]==j,1]
            
            # if reward was triggered, add an extra lick
            rw_lick = 0
            if reward_ds[reward_ds[:,3]==j,5] == 1:
                rw_lick = 1
                
            if raw_trial[0,4] != 5:
                # keep track of number of licks in a given trial. Add a lick if reward was triggered
                licks_per_trial_ol[k,i] = plot_licks_x.shape[0] + rw_lick
                # keep track of whether a reward was triggered by the mouse or not
                correct_trial_ol[k,i] = rw_lick 
                k = k+1
                        
    combined_graph = np.vstack((licks_per_trial_vr,licks_per_trial_ol))
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes()  
    
    frac_correct_vr = np.sum(correct_trial_vr,1) / correct_trial_vr.shape[1]
    frac_correct_ol = np.sum(correct_trial_ol,1) / correct_trial_ol.shape[1]
    frac_correct_all = np.hstack((frac_correct_vr,frac_correct_ol))
    
    sem_combined_graph = np.std(combined_graph,1) / np.sqrt(combined_graph.shape[1])
    mean_combined_graph = np.mean(combined_graph,1)
    plt.plot(np.arange(0,num_trials*2,1),frac_correct_all, c='k', lw=2)
    # ax.fill_between(np.arange(combined_graph.shape[0]), mean_combined_graph - sem_combined_graph, mean_combined_graph + sem_combined_graph, color = '0.5', alpha = 0.2)
    ax.set_xticks([0,15,30,45,60])
    ax.set_xticklabels(['-30','-15','0','15','30'])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','0.5','1'])
    sns.despine(top=True, right=True, left=False, bottom=False)
    ax.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=3, \
        width=1, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    
    path_name = loc_info['figure_output_path'] + 'task_performance'
    if not os.path.isdir(path_name):
            os.mkdir(path_name)
    fig.savefig(path_name + os.sep + "context_change_fraction_correct.svg", format='svg')
    print("saved " + path_name + os.sep + "context_change_fraction_correct.svg")

def running_speed(behavior_data_path, sessions_vr, sessions_ol):
    
    speed_bins = np.arange(0,105,5)
    running_speeds_vr = np.zeros((len(sessions_vr),speed_bins.shape[0]-1))
    for i in range(len(sessions_vr)):
        data_path = behavior_data_path + os.sep + sessions_vr[i][0]
        raw_filename = data_path + os.sep + sessions_vr[i][1] + os.sep + sessions_vr[i][2]
        save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_vr[i][0] + os.sep + sessions_vr[i][0]+'_'+sessions_vr[i][1]
        path_name = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_vr[i][0]
        raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sessions_vr[i][2])
        
        session_running_speed = raw_ds[raw_ds[:,4]!=5,3]
        running_speeds_vr[i,:] = np.histogram(session_running_speed, bins=speed_bins)[0]
     
        
    running_speeds_ol = np.zeros((len(sessions_ol),speed_bins.shape[0]-1))
    for i in range(len(sessions_ol)):
        data_path = behavior_data_path + os.sep + sessions_ol[i][0]
        raw_filename = data_path + os.sep + sessions_ol[i][1] + os.sep + sessions_ol[i][2]
        save_filename = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_ol[i][0] + os.sep + sessions_ol[i][0]+'_'+sessions_ol[i][1]
        path_name = loc_info['figure_output_path'] + 'task_performance' + os.sep + sessions_ol[i][0]
        raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sessions_ol[i][2])
        
        session_running_speed = raw_ds[raw_ds[:,4]!=5,8]
        running_speeds_ol[i,:] = np.histogram(session_running_speed, bins=speed_bins)[0]
        

    fig = plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
   
    for i in range(len(sessions_vr)):
        ax1.plot(running_speeds_vr[i,1:], c='0.7')
        ax2.plot(running_speeds_ol[i,1:], c='0.7')
    
    ax1.plot(np.mean(running_speeds_vr[:,1:],axis=0),c='k',lw=2)
    ax2.plot(np.mean(running_speeds_ol[:,1:],axis=0),c='k',lw=2)
    
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    
    ax2.set_xticks([0,5,10,15,19])
    ax2.set_xticklabels([5,25,50,75,100])
    
    ax2.set_xlabel('Running speed (cm/sec)')
    ax2.set_ylabel('Count')
    
    
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    ax1.tick_params(left=True,bottom=True)
    plt.tight_layout()
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'running speed vr ol')
    fname = loc_info['figure_output_path'] + os.sep + 'running speed vr ol' + os.sep + "running speed vr ol"+ '.svg'
    plt.savefig(fname, dpi=300)
    print('saved ' + fname)

if __name__ == '__main__':
    # set plotting parameters
    bar_trials = range(1,8)
    # expert = [('LF191022_1','20191204'),('LF191022_2','20191210'),('LF191022_3','20191207'),('LF191023_blank','20191206'),('LF191023_blue','20191204'),('LF191024_1','20191204')]
    behavior_data_path = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\data_2p"
    
    sessions_vr = [['LF191022_1','20191204','MTH3_vr1_s5r2_2019124_1947.csv'],
                   ['LF191022_2','20191210','MTH3_vr1_s5r2_20191210_2335.csv'],
                   ['LF191022_3','20191207','MTH3_vr1_s5r2_2019128_111.csv'],
                   ['LF191023_blank','20191206','MTH3_vr1_s5r2_2019126_1831.csv'],
                   ['LF191023_blue','20191204','MTH3_vr1_s5r2_2019124_1834.csv'],
                   ['LF191024_1','20191204','MTH3_vr1_s5r2_2019125_016.csv']]
    sessions_ol = [['LF191022_1','20191204_ol','MTH3_vr1_openloop_2019124_2048.csv'],
                   ['LF191022_2','20191210_ol','MTH3_vr1_openloop_20191211_036.csv'],
                   ['LF191022_3','20191207_ol','MTH3_vr1_openloop_2019128_139.csv'],
                   ['LF191023_blank','20191206_ol','MTH3_vr1_openloop_2019126_1939.csv'],
                   ['LF191023_blue','20191204_ol','MTH3_vr1_openloop_2019124_1919.csv'],
                   ['LF191024_1','20191204_ol','MTH3_vr1_openloop_2019125_058.csv']]
    
    # licks_per_trial(behavior_data_path, sessions_vr, sessions_ol)
    running_speed(behavior_data_path, sessions_vr, sessions_ol)
    
    # set filenames
    fformat = 'png'
    # MOUSE = 'LF191022_1'
    # SESSION = ['20191204','MTH3_vr1_s5r2_2019124_1947.csv']
    # SESSION = ['20191204_ol','MTH3_vr1_openloop_2019124_2048.csv']
    
    # MOUSE = 'LF191022_2'
    # SESSION = ['20191210','MTH3_vr1_s5r2_20191210_2335.csv']
    # SESSION = ['20191210_ol','MTH3_vr1_openloop_20191211_036.csv']
    
    MOUSE = 'LF191022_3'
    SESSION = ['20191207','MTH3_vr1_s5r2_2019128_111.csv']
    # SESSION = ['20191207_ol','MTH3_vr1_openloop_2019128_139.csv']
    
    # MOUSE = 'LF191023_blank'
    # SESSION = ['20191206','MTH3_vr1_s5r2_2019126_1831.csv']
    # SESSION = ['20191206_ol','MTH3_vr1_openloop_2019126_1939.csv']
    
    # MOUSE = 'LF191023_blue'
    # SESSION = ['20191204','MTH3_vr1_s5r2_2019124_1834.csv']
    # SESSION = ['20191204_ol','MTH3_vr1_openloop_2019124_1919.csv']
    
    # MOUSE = 'LF191024_1'
    # SESSION = ['20191204','MTH3_vr1_s5r2_2019125_016.csv']
    # SESSION = ['20191204_ol','MTH3_vr1_openloop_2019125_058.csv']
    
    
    
    # MOUSE = 'LF191022_1'
    # SESSION = ['20191203','MTH3_vr3_s5r2_2019123_1713.csv']
    data_path = behavior_data_path + os.sep + MOUSE
    
    # line_of_trials(fformat, MOUSE, SESSION, data_path, bar_trials)
    # all_trials_ts(fformat, MOUSE, SESSION, data_path)
    all_trials_lm(fformat, MOUSE, SESSION, data_path)
    # all_trials_rw(fformat, MOUSE, SESSION, data_path)