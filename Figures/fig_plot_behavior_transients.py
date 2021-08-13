# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:01:12 2019

@author: murra
"""

import numpy as np
import os
import sys
import yaml
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
sns.set_style("white")

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '\\Analysis')

from filter_trials import filter_trials
from load_behavior_data import load_data
from rewards import rewards
from licks import licks_nopost as licks

TRACK_SHORT = 7
TRACK_LONG = 9

def load_raw_data(raw_filename, sess):
    raw_data = load_data(raw_filename, 'vr')
    all_licks = licks(raw_data)
    trial_licks = all_licks[np.in1d(all_licks[:, 3], [TRACK_SHORT, TRACK_LONG]), :]
    reward =  rewards(raw_data)
    return raw_data, trial_licks, reward

def fig_behavior_stage2(data_path, sess, fname, fformat='png', subfolder=[]):

    raw_filename = data_path + os.sep + sess[0] + os.sep + sess[1]
    raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sess)
#    reward_ds[:,3] = reward_ds[:,3] - 1
#    reward_ds[:,2] = 3

    # create figure to later plot on
    fig = plt.figure(figsize=(12,6))
#    fig.suptitle(fname)
    ax1 = plt.subplot2grid((12,6),(0,0), rowspan=12, colspan=3)
    ax2 = plt.subplot2grid((12,6),(0,3), rowspan=12, colspan=3)
    ax1.set_ylabel('# of Licks')
    ax1.set_xlabel('Location (cm)')
    ax2.set_xlabel('Location (cm)')
    ax1.set_title('Track Short: ' + str(TRACK_SHORT))
    ax2.set_title('Track Long: ' + str(TRACK_LONG))
    
    licks_in_trial_short = np.array([])
    licks_in_trial_long = np.array([])
    
    if np.size(licks_ds) > 0 or np.size(reward_ds) > 0:

        short_trials = filter_trials( raw_ds, [], ['tracknumber',7])
        long_trials = filter_trials( raw_ds, [], ['tracknumber',9])        
        # get trial numbers to be plotted
        lick_trials = np.unique(licks_ds[:,2])
        reward_trials = np.unique(reward_ds[:,3])
        scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
        scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
        scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
        
        for i,r in enumerate(scatter_rowlist_map_short):
            plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
            round_licks = np.around(plot_licks_x)
            licks_in_trial_short = np.append(licks_in_trial_short, round_licks)
        
        for i,r in enumerate(scatter_rowlist_map_long):
            plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
            round_licks = np.around(plot_licks_x)
            licks_in_trial_long = np.append(licks_in_trial_long, round_licks)
        
        licks_in_trial_short = np.concatenate(licks_in_trial_short, axis=None)
        ax1_xtick_pre = range(0,300)
        ax1_xtick_pro = []
        ax1_ytick = []
        for tick in ax1_xtick_pre:
            t_o_f = np.in1d(licks_in_trial_short, tick)
            count = 0
            if True in t_o_f:
                ax1_xtick_pro.append(tick)
                for pos in t_o_f:
                    if pos:
                        count += 1
                ax1_ytick.append(count)
        y_tot = 0
        ax1_xfin = []
        ax1_yfin = []
        for tick in range(len(ax1_xtick_pro)):
            y_tot += ax1_ytick[tick]
            if (tick != 0 and tick % 10 == 0) or tick == range(len(ax1_xtick_pro))[-1]:
                ax1_xfin.append(str(ax1_xtick_pro[tick]))
                ax1_yfin.append(y_tot)
                y_tot = 0
        y_pos_short = np.arange(len(ax1_xfin))
        ax1.bar(y_pos_short, ax1_yfin, align='center', alpha=0.5)
        ax1.set_xticks(y_pos_short)
        ax1.set_xticklabels(ax1_xfin)

        licks_in_trial_long = np.concatenate(licks_in_trial_long, axis=None)
        ax2_xtick_pre = range(0,300)
        ax2_xtick_pro = []
        ax2_ytick = []
        for tick in ax2_xtick_pre:
            t_o_f = np.in1d(licks_in_trial_long, tick)
            count = 0
            if True in t_o_f:
                ax2_xtick_pro.append(tick)
                for pos in t_o_f:
                    if pos:
                        count += 1
                ax2_ytick.append(count)
        y_tot = 0
        ax2_xfin = []
        ax2_yfin = []
        for tick in range(len(ax2_xtick_pro)):
            y_tot += ax2_ytick[tick]
            if (tick != 0 and tick % 10 == 0) or tick == range(len(ax2_xtick_pro))[-1]:
                ax2_xfin.append(str(ax2_xtick_pro[tick]))
                ax2_yfin.append(y_tot)
                y_tot = 0
        y_pos_long = np.arange(len(ax2_xfin))
        ax2.bar(y_pos_long, ax2_yfin, align='center', alpha=0.5)
        ax2.set_xticks(y_pos_long)
        ax2.set_xticklabels(ax2_xfin)

    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()

def transients_bin(dF_aligned, behaviour_aligned, speed_align, track_align, fname, fformat='png', subfolder = []):
    bh_range_val = range(int(round(np.amin(behaviour_aligned))), int(round(np.amax(behaviour_aligned)+1)))
    bin_list = [x for x in bh_range_val if (x != bh_range_val[0] and x % 10 == 0) or x == bh_range_val[-1]]
    frame_list_of_lis_short = []
    frame_list_of_lis_long = []
    for val in range(len(bin_list)):
        arg_array = []
        if val == 0:
            for frame in range(len(behaviour_aligned)):
                if behaviour_aligned[frame] <= bin_list[val]:
                    arg_array.append(frame)
        else:
            for frame in range(len(behaviour_aligned)):
                if bin_list[val - 1] < behaviour_aligned[frame] <= bin_list[val]:
                    arg_array.append(frame)                    
        cur_lis_short = []
        cur_lis_long = []
        for arg in arg_array:
            if track_align[arg] == 7:
                cur_lis_short.append(arg)
            elif track_align[arg] == 9:
                cur_lis_long.append(arg)
#            else:
#                print(track_align[arg])
        frame_list_of_lis_short.append(cur_lis_short)
        frame_list_of_lis_long.append(cur_lis_long)
    tran_val_list_short = []
    for bin_frame in frame_list_of_lis_short:
        tot_val = 0
        for val in bin_frame:
            if speed_align[val] >= 0.5:
                tot_val += dF_aligned[:,0][val]
        if len(bin_frame) == 0:
            tran_val_list_short.append(0)
        else:
            tran_val_list_short.append(tot_val/len(bin_frame))
        
    tran_val_list_long = []
    for bin_frame in frame_list_of_lis_long:
        tot_val = 0
        for val in bin_frame:
            if speed_align[val] >= 0.5:
                tot_val += dF_aligned[:,0][val]
        if len(bin_frame) == 0:
            tran_val_list_long.append(0)
        else:
            tran_val_list_long.append(tot_val/len(bin_frame))        
    fig = plt.figure(figsize=(12,6))
#    fig.suptitle(fname)
    ax1 = plt.subplot2grid((12,6),(0,0), rowspan=12, colspan=3)
    ax2 = plt.subplot2grid((12,6),(0,3), rowspan=12, colspan=3)
    ax1.set_ylabel('df/F Value')
    ax1.set_xlabel('Location (cm)')
    ax2.set_xlabel('Location (cm)')
    ax1.set_title('Track Short: ' + str(TRACK_SHORT))
    ax2.set_title('Track Long: ' + str(TRACK_LONG))
    y_pos = np.arange(len(bin_list))
    ax1_xfin = [str(x) for x in bin_list]
    ax1.bar(y_pos, tran_val_list_short, align='center', alpha=0.5)
    ax2.bar(y_pos, tran_val_list_long, align='center', alpha=0.5)
    ax1.set_xticks(y_pos)
    ax1.set_xticklabels(ax1_xfin)
    ax2.set_xticks(y_pos)
    ax2.set_xticklabels(ax1_xfin)

    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
    
def run_Jimmy():
    MOUSE = 'Jimmy'
    session = ['190507_3','MTH3_VD_201957_1636.csv', 'aligned_data.mat']
    data_path = loc_info['raw_dir'] +  '\\' + MOUSE
    processed_data_path = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + session[0] + os.sep + session[2]
    loaded_data = sio.loadmat(processed_data_path)
    dF_aligned = loaded_data['dF_aligned']
    behaviour_aligned = loaded_data['behaviour_aligned'][:,1]
    speed_align = loaded_data['behaviour_aligned'][:,3]
    track_align = loaded_data['behaviour_aligned'][:,4]
    fig_behavior_stage2(data_path, session, MOUSE+session[0]+'_vr', fformat, MOUSE)
    transients_bin(dF_aligned, behaviour_aligned, speed_align, track_align, MOUSE+session[0]+'_tran', fformat, MOUSE)

if __name__ == '__main__':

    fformat = 'png'
    run_Jimmy()