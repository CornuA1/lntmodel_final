# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:26:42 2021

@author: lfisc

Plot various graphs for figure 5

"""

import os, yaml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.io import loadmat
from scipy import stats
plt.rcParams['svg.fonttype'] = 'none'

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.safe_load(f)
    
fformat = '.svg'

def make_folder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

def load_data(mouse, session, ol = False):
    '''
    loads behavior data for a mouse and session

    Parameters
    ----------
    mouse : str
        mouse id.
    session : str
        session.

    Returns
    -------
    behav_data : array
        behavioral data.

    '''
    try:
        if ol:
            filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'old' + os.sep + 'aligned_data.mat'
        else:
            filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'old' + os.sep + 'aligned_data.mat'
        # filename_ol = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data_ol.mat'
        aligned_data = loadmat(filename)
    except:
        try:
            if ol:
                filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data.mat'
            else:
                filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data.mat'
            # filename_ol = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data_ol.mat'
            aligned_data = loadmat(filename)
        except:
            if ol:
                filename = loc_info['raw_dir'] + os.sep + session + os.sep + 'aligned_data_ol.mat'
            else:
                filename = loc_info['raw_dir'] + os.sep + 'aligned_data.mat'
            aligned_data = loadmat(filename)
    behav_data = aligned_data['behaviour_aligned']
    return behav_data

def filter_trials(behav_data, trial_type):
    '''
    get behavioral data for short or long trials

    Parameters
    ----------
    behav_data : array
        behavioral data.
    trial_type : str
        'short' or 'long'.

    Returns
    -------
    trials : array
        data from either short or long trials.

    '''
    
    all_trials = False
    ol_sess = False
    
    if trial_type == 'short':
        trial_type = 3
    elif trial_type == 'long':
        trial_type = 4
    elif trial_type == 'both':
        all_trials = True
        trial_type = -1
    elif trial_type == 'ol_fast':
        ol_sess = True
        
    trials = [] # list of trials; each entry is a trial
    # trial_temp = [] # list of data points for a trial, which are then appended to trials
    # trial_num = 1 # trial number, the aligned data starts at trial 1
    trial_start = np.zeros((0,))
    
    sess_trials = np.unique(behav_data[:,6]) # get all trial numbers in a session
    
    for t in sess_trials:
        cur_trial = behav_data[behav_data[:,6]==t,:]
        if cur_trial[0,4] == trial_type and not ol_sess:
            trials.append(cur_trial)
            if cur_trial[0,1] > 40: # the threshold is just a cutoff to toss out the very first trial of a session that starts at 0
                trial_start = np.hstack((trial_start,cur_trial[0,1]))
        elif all_trials and cur_trial[0,4] != 5 and not ol_sess:
            trials.append(cur_trial)
            if cur_trial[0,1] > 40: # the threshold is just a cutoff to toss out the very first trial of a session that starts at 0
                trial_start = np.hstack((trial_start,cur_trial[0,1]))
        elif ol_sess and trial_type == 'ol_fast' and cur_trial[0,4] != 5:
            if cur_trial[0,3] > 20:
                trials.append(cur_trial)
                if cur_trial[0,1] > 40: # the threshold is just a cutoff to toss out the very first trial of a session that starts at 0
                    trial_start = np.hstack((trial_start,cur_trial[0,1]))
                
            
    
    # for n in range (len(behav_data[:,0])):
    #     if trial_num != behav_data[n,6]:
    #     # if trial_num != current trial number, this means it's the start of a new trial
    #         if behav_data[n-1,4] == trial_type:
    #             # add to trials if it's the right trial type
    #             trials.append(trial_temp)
            
    #         # reset
    #         trial_temp = []
    #         trial_num = behav_data[n,6]

    #     if behav_data[n,4] == trial_type:
    #     # if it's the right trial type, add data points to trial_temp
    #         trial_temp.append(behav_data[n,:])

    return trials, np.mean(trial_start)

def plot_control_traces():
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    plot_trial = 90
    plot_neuron = 15 #6 #4
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104262' + os.sep + 'srug_cont' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(5,7.5))
    gs = fig.add_gridspec(9, 1)
    ax1 = fig.add_subplot(gs[0:3,:])
    ax2 = fig.add_subplot(gs[4:6,:])
    ax3 = fig.add_subplot(gs[7:9,:])
    # ax3 = fig.add_subplot(gs[14:16,:])
    # ax5 = fig.add_subplot(gs[10:13,:])

    x = 1
    ax1.plot(timestamps[:],v[:,plot_neuron], c='k', lw=0.5)
    ax2.plot(neuron_I[:,plot_neuron,0])
    ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    ax3.set_ylabel('running speed (cm/sec)')
    ax3.set_xlabel('Time (sec)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'neuron_trace_CONT_' + str(plot_neuron) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

def plot_vr_traces(plot_trial, plot_neuron):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104262' + os.sep + 'srug_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(5,7.5))
    gs = fig.add_gridspec(9, 1)
    ax1 = fig.add_subplot(gs[0:3,:])
    ax2 = fig.add_subplot(gs[4:6,:])
    ax3 = fig.add_subplot(gs[7:9,:])
    # ax3 = fig.add_subplot(gs[14:16,:])
    # ax5 = fig.add_subplot(gs[10:13,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps[:],v[:,plot_neuron], c='k', lw=0.5)
    ax2.plot(neuron_I[:,plot_neuron,0])
    ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    ax3.set_ylabel('running speed (cm/sec)')
    ax3.set_xlabel('Time (sec)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'neuron_trace_VR_neuron_' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)
    
def plot_ol_traces(plot_trial, plot_neuron):
    
    # MOUSE = 'LF191022_1'
    # SESSION = '20191213_ol'
    
    trial_nr_offset = 0
    # plot_trial = 41 #45#42#21
    
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    # behav_data = load_data(MOUSE, SESSION, ol = True)
    # trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104282' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']
    mouse_vel = res['mouse_vel'] 
    vr_vel = res['vr_vel'] 

    speeddiff = (mouse_vel - vr_vel) * 1000
    
    fig = plt.figure(figsize=(5,7.5))
    gs = fig.add_gridspec(12, 1)
    ax1 = fig.add_subplot(gs[0:3,:])
    ax2 = fig.add_subplot(gs[4:6,:])
    ax3 = fig.add_subplot(gs[7:9,:])
    ax4 = fig.add_subplot(gs[10:12,:])
    # ax5 = fig.add_subplot(gs[10:13,:])

    
    ax1.plot(timestamps[:],v[:,plot_neuron], c='k', lw=0.5)
    ax2.plot(neuron_I[:,plot_neuron,0])
    ax2.plot(neuron_I[:,plot_neuron,1])
    
    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    min_dspeed = -30
    max_dspeed = 45
    x_vals = np.linspace(0, int(timestamps[-1]), speeddiff.shape[0])    
    points = np.array([x_vals, speeddiff]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(min_dspeed, max_dspeed)
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    
    lc.set_array(speeddiff)
    lc.set_linewidth(2)
    line = ax3.add_collection(lc)
    
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial-trial_nr_offset][:,0] - trials[plot_trial-trial_nr_offset][0,0], trials[plot_trial-trial_nr_offset][:,8],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3.plot(trials[plot_trial-trial_nr_offset][:,0] - trials[plot_trial-trial_nr_offset][0,0], trials[plot_trial-trial_nr_offset][:,3],c ='#CA457A')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    ax3.set_ylabel('running speed (cm/sec)')
    ax3.set_xlabel('Time (sec)')
    
    ax4.plot(timestamps, mouse_locs)
    
    ax3.set_ylim([-30,45])
    ax3.set_yticks([-30,0,30])
    ax3.set_yticklabels([-30,0,30])
    ax3.set_xlim([0, int(timestamps[-1])])
    
    ax1.set_xlim([0,timestamps[-1]])
    ax2.set_xlim([0,timestamps[-1]])
    ax3.set_xlim([0,timestamps[-1]])
    ax4.set_xlim([0,timestamps[-1]])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'neuron_trace_OL_neuron_' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)
    # plt.close()

def plot_single_comp_s_vr_traces(plot_trial, plot_neuron):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'single_comp_s' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2' + os.sep + 'single_comp_s_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(5,7.5))
    gs = fig.add_gridspec(9, 1)
    ax1 = fig.add_subplot(gs[0:3,:])
    ax2 = fig.add_subplot(gs[4:6,:])
    ax3 = fig.add_subplot(gs[7:9,:])
    # ax3 = fig.add_subplot(gs[14:16,:])
    # ax5 = fig.add_subplot(gs[10:13,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps[:],v[:,plot_neuron], c='k', lw=0.5)
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    ax3.set_ylabel('running speed (cm/sec)')
    ax3.set_xlabel('Time (sec)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'neuron_trace_single_comp_s_VR_neuron_' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

def plot_single_comp_s_ol_traces(plot_trial, plot_neuron):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2' + os.sep + 'single_comp_s_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(5,7.5))
    gs = fig.add_gridspec(9, 1)
    ax1 = fig.add_subplot(gs[0:3,:])
    ax2 = fig.add_subplot(gs[4:6,:])
    ax3 = fig.add_subplot(gs[7:9,:])
    # ax3 = fig.add_subplot(gs[14:16,:])
    # ax5 = fig.add_subplot(gs[10:13,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps[:],v[:,plot_neuron], c='k', lw=0.5)
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    ax3.set_ylabel('running speed (cm/sec)')
    ax3.set_xlabel('Time (sec)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'neuron_trace_single_comp_s_OL_neuron_' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

def plot_single_comp_l_vr_traces(plot_trial, plot_neuron):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'single_comp_l' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2' + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(5,7.5))
    gs = fig.add_gridspec(9, 1)
    ax1 = fig.add_subplot(gs[0:3,:])
    ax2 = fig.add_subplot(gs[4:6,:])
    ax3 = fig.add_subplot(gs[7:9,:])
    # ax3 = fig.add_subplot(gs[14:16,:])
    # ax5 = fig.add_subplot(gs[10:13,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps[:],v[:,plot_neuron], c='k', lw=0.5)
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    ax3.set_ylabel('running speed (cm/sec)')
    ax3.set_xlabel('Time (sec)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'neuron_trace_single_comp_l_VR_neuron_' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

def plot_single_comp_l_ol_traces(plot_trial, plot_neuron):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2' + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(5,7.5))
    gs = fig.add_gridspec(9, 1)
    ax1 = fig.add_subplot(gs[0:3,:])
    ax2 = fig.add_subplot(gs[4:6,:])
    ax3 = fig.add_subplot(gs[7:9,:])
    # ax3 = fig.add_subplot(gs[14:16,:])
    # ax5 = fig.add_subplot(gs[10:13,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps[:],v[:,plot_neuron], c='k', lw=0.5)
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    ax3.set_ylabel('running speed (cm/sec)')
    ax3.set_xlabel('Time (sec)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'neuron_trace_single_comp_l_OL_neuron_' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

def plot_vr_loc_and_force(plot_trial, plot_neuron):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104262' + os.sep + 'srug_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0:1,:])
    ax2 = fig.add_subplot(gs[1:,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps,mouse_locs, c='k', lw=1)
    ax1.plot(timestamps,can_locs, c='#2E3191', ls='--', lw=1)
    ax2.plot(timestamps, np.abs(mouse_locs-can_locs), c='k', lw=2, zorder=2)
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(timestamps, np.abs(force), c='#EB008B', alpha=0.2, lw=2, zorder=1)
    
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax2_2, right=False, top=True)
    
    ax1.tick_params(left='on',bottom='on',direction='out')
    ax2.tick_params(left='on',bottom='on',direction='out')
    
    ax1.set_ylim([50,350])
    ax2.set_ylim([0,25])
    
    plt.tight_layout()
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force' + os.sep + 'neuron_loc_and_force_VR' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)
   
def plot_cont_loc_and_force(plot_trial, plot_neuron, force_ylim=None):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104262' + os.sep + 'srug_cont' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0:1,:])
    ax2 = fig.add_subplot(gs[1:,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps,mouse_locs, c='k', lw=1)
    ax1.plot(timestamps,can_locs, c='#2E3191', ls='--', lw=1)
    ax2.plot(timestamps, np.abs(mouse_locs-can_locs), c='k', lw=2, zorder=2)
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(timestamps, np.abs(force), c='#EB008B', alpha=0.2, lw=2, zorder=1)
    
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax2_2, right=False, top=True)
    
    ax1.tick_params(left='on',bottom='on',direction='out')
    ax2.tick_params(left='on',bottom='on',direction='out')
    
    ax1.set_ylim([50,350])
    ax2.set_ylim([0,25])
    if force_ylim is not None:
        ax2_2.set_ylim(force_ylim)
    
    plt.tight_layout()
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force' + os.sep + 'neuron_loc_and_force_VR' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname) 
   
def plot_ol_loc_and_force(plot_trial, plot_neuron):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213_ol'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104262' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0:1,:])
    ax2 = fig.add_subplot(gs[1:,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps,mouse_locs, c='k', lw=1)
    ax1.plot(timestamps,can_locs, c='#2E3191', ls='--', lw=1)
    ax2.plot(timestamps, np.abs(mouse_locs-can_locs), c='k', lw=2, zorder=2)
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(timestamps, np.abs(force), c='#EB008B', alpha=0.2, lw=2, zorder=1)
    
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax2_2, right=False, top=True)
    
    ax1.tick_params(left='on',bottom='on',direction='out')
    ax2.tick_params(left='on',bottom='on',direction='out')
    
    ax1.set_ylim([50,350])
    # ax2.set_ylim([0,25])
    
    plt.tight_layout()
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force' + os.sep + 'neuron_loc_and_force_OL' + str(plot_neuron) + '_trial_' + str(plot_trial) + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

def plot_single_s_loc_and_force(plot_trial, plot_neuron, fname_suffix):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210429' + os.sep + 'single_comp_s' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0:1,:])
    ax2 = fig.add_subplot(gs[1:,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps,mouse_locs, c='k', lw=1)
    ax1.plot(timestamps,can_locs, c='#2E3191', ls='--', lw=1)
    ax2.plot(timestamps, np.abs(mouse_locs-can_locs), c='k', lw=2, zorder=2)
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(timestamps, np.abs(force), c='#EB008B', alpha=0.2, lw=2, zorder=1)
    
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax2_2, right=False, top=True)
    
    ax1.tick_params(left='on',bottom='on',direction='out')
    ax2.tick_params(left='on',bottom='on',direction='out')
    
    ax1.set_ylim([50,350])
    ax2.set_ylim([0,25])
    
    plt.tight_layout()
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force' + os.sep + 'neuron_loc_and_force_VR' + str(plot_neuron) + '_trial_' + str(plot_trial) + '_' + fname_suffix + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname) 

def plot_single_s_ol_loc_and_force(plot_trial, plot_neuron, fname_suffix):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210507' + os.sep + 'single_comp_s_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0:1,:])
    ax2 = fig.add_subplot(gs[1:,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps,mouse_locs, c='k', lw=1)
    ax1.plot(timestamps,can_locs, c='#2E3191', ls='--', lw=1)
    ax2.plot(timestamps, np.abs(mouse_locs-can_locs), c='k', lw=2, zorder=2)
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(timestamps, np.abs(force), c='#EB008B', alpha=0.2, lw=2, zorder=1)
    
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax2_2, right=False, top=True)
    
    ax1.tick_params(left='on',bottom='on',direction='out')
    ax2.tick_params(left='on',bottom='on',direction='out')
    
    ax1.set_ylim([50,350])
    ax2.set_ylim([0,25])
    
    plt.tight_layout()
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force' + os.sep + 'neuron_loc_and_force_VR' + str(plot_neuron) + '_trial_' + str(plot_trial) + '_' + fname_suffix + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname) 

def plot_single_l_loc_and_force(plot_trial, plot_neuron, fname_suffix):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210429' + os.sep + 'single_comp_l' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0:1,:])
    ax2 = fig.add_subplot(gs[1:,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps,mouse_locs, c='k', lw=1)
    ax1.plot(timestamps,can_locs, c='#2E3191', ls='--', lw=1)
    ax2.plot(timestamps, np.abs(mouse_locs-can_locs), c='k', lw=2, zorder=2)
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(timestamps, np.abs(force), c='#EB008B', alpha=0.2, lw=2, zorder=1)
    
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax2_2, right=False, top=True)
    
    ax1.tick_params(left='on',bottom='on',direction='out')
    ax2.tick_params(left='on',bottom='on',direction='out')
    
    ax1.set_ylim([50,350])
    ax2.set_ylim([0,25])
    
    plt.tight_layout()
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force' + os.sep + 'neuron_loc_and_force_VR' + str(plot_neuron) + '_trial_' + str(plot_trial) + '_' + fname_suffix + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname) 

def plot_single_l_ol_loc_and_force(plot_trial, plot_neuron, fname_suffix):
    
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    # plot_trial = 21
    # plot_neuron = 17
    f_bin_edges = np.arange(50,360,10)
    
    behav_data = load_data(MOUSE, SESSION, ol = False)
    trials, ave_mouse_start = filter_trials(behav_data, 'short')
    
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210507' + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(plot_trial) + '.npz'
    res = np.load(results_vr_sess)
    timestamps = res['timestamps']
    mouse_locs = res['mouse_locs']
    can_locs = res['can_locs']
    v = res['v']
    spikes = res['spikes']
    force = res['force']
    neuron_I = res['neuron_I']

    fig = plt.figure(figsize=(3,5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0:1,:])
    ax2 = fig.add_subplot(gs[1:,:])

    lm_ts = timestamps[np.where(mouse_locs > 220)[0][0]]
    ax1.axvline(lm_ts, lw=1, c='#EB008B', ls='--')
    
    ax1.plot(timestamps,mouse_locs, c='k', lw=1)
    ax1.plot(timestamps,can_locs, c='#2E3191', ls='--', lw=1)
    ax2.plot(timestamps, np.abs(mouse_locs-can_locs), c='k', lw=2, zorder=2)
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(timestamps, np.abs(force), c='#EB008B', alpha=0.2, lw=2, zorder=1)
    
    # ax2.plot(neuron_I[:,plot_neuron,0])
    # ax2.plot(neuron_I[:,plot_neuron,1])
    # ax3.plot(trials[plot_trial][:,0], trials[plot_trial][:,8])
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3], c='g')
    # ax3.plot(trials[plot_trial][:,0] - trials[plot_trial][0,0], trials[plot_trial][:,3],c ='g')
    # ax3_2 = ax3.twiny()
    # ax3_2.plot(timestamps[:],mouse_locs[:],c='r')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('membrane potential (mV)')
    
    ax2.set_ylabel('injected current (pA)')
    ax2.set_xlabel('Time (ms)')
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax2_2, right=False, top=True)
    
    ax1.tick_params(left='on',bottom='on',direction='out')
    ax2.tick_params(left='on',bottom='on',direction='out')
    
    ax1.set_ylim([50,350])
    ax2.set_ylim([0,25])
    
    plt.tight_layout()
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_loc_and_force' + os.sep + 'neuron_loc_and_force_VR' + str(plot_neuron) + '_trial_' + str(plot_trial) + '_' + fname_suffix + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname) 

def sim_speeddiff_response():
    ''' calculate neuron response as a function of difference between VR and mouse running speed '''
    
    trials = np.arange(0,50,1)
    
    fig = plt.figure(figsize=(7.5,10))
    gs = fig.add_gridspec(1, 10)
    ax1 = fig.add_subplot(gs[0,0:4])
    ax2 = fig.add_subplot(gs[0,6:10])
    
    all_num_spikes = np.zeros((0,))
    all_force = np.zeros((0,))
    all_speed_diff = np.zeros((0,))
    all_neuron_spikes = np.zeros((0,2))
    for t in trials:
        
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_2neurons_(300, 50)noise_thresh_1.75' + os.sep + 'results_trial_' + str(t) + '.npz'
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104282'  + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep + 'results_trial_' + str(t) + '.npz'
        res = np.load(results_vr_sess)
        timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        can_locs = res['can_locs']
        v = res['v']
        spikes = res['spikes']
        force = res['force']
        neuron_I = res['neuron_I'] 
        coh = res['coh'] 
        mouse_vel = res['mouse_vel'] 
        vr_vel = res['vr_vel'] 
            
        # fig = plt.figure(figsize=(5,7.5))
        # gs = fig.add_gridspec(12, 1)
        # ax1 = fig.add_subplot(gs[0:3,:])
        # ax2 = fig.add_subplot(gs[4:6,:])
        # ax3 = fig.add_subplot(gs[7:9,:])
        # ax4 = fig.add_subplot(gs[10:12,:])
        
        # ax1.plot(timestamps, mouse_vel)
        # ax1.plot(timestamps, vr_vel)
        # ax2.plot(timestamps, coh)
        
        # ax3.plot(timestamps,neuron_I[:,plot_neuron,0])
        # ax3_2 = ax3.twinx()
        # ax3_2.plot(timestamps,neuron_I[:,plot_neuron,1],c='r')
        
        trial_duration = (timestamps[-1] - timestamps[0]) / 1000 # determine trial duration, convert from msec to sec
        
        all_speed_diff = np.append(all_speed_diff,np.mean(mouse_vel - vr_vel) * 1000)
        all_force = np.append(all_force, np.sum(force)) # add up total force exerted
        all_num_spikes = np.append(all_num_spikes, len(np.where(spikes > 0)[0]))
        
        trial_spikes = []
        for s in spikes.T:
            trial_spikes.append(len(np.where(s > 0)[0]) / trial_duration)
                                
        # all_neuron_spikes.append(trial_spikes)
        all_neuron_spikes = np.vstack((all_neuron_spikes, np.array([np.ones((len(trial_spikes),)) * all_speed_diff[-1], trial_spikes]).T))
        
    # fit 2nd order polynomial to data
    spike_fit = np.polyfit(all_neuron_spikes[:,0],all_neuron_spikes[:,1], 2)
    force_fit = np.polyfit(all_speed_diff, all_force, 2)
    p_spike = np.poly1d(spike_fit)
    p_force = np.poly1d(force_fit)
    
    plot_xvals = np.linspace(-30, 50, 1000)
    
    
    # for i,ansp in enumerate(all_neuron_spikes):
    #     ax1.scatter(np.ones((len(ansp),)) * all_speed_diff[i], ansp, c=np.ones((len(ansp),)) * all_speed_diff[i], cmap='plasma', norm=plt.Normalize(vmin=-30, vmax=50))
        
    
    ax1.scatter(all_neuron_spikes[:,0],all_neuron_spikes[:,1], c=all_neuron_spikes[:,0], cmap='plasma', alpha=0.5)
    ax2.scatter(all_speed_diff,all_force, c=all_speed_diff, cmap='plasma', alpha=0.5)
    
    ax1.plot(plot_xvals, p_spike(plot_xvals), c='w', ls='-', lw=6, zorder=4)
    ax1.plot(plot_xvals, p_spike(plot_xvals), c='g', ls='-', lw=4, zorder=6)
    ax2.plot(plot_xvals, p_force(plot_xvals), c='g', ls='-', lw=4, zorder=6)
    
    ax1.axvline(plot_xvals[np.argmax(p_spike(plot_xvals))], color='g', ls='--', lw=2, zorder=5)
    ax2.axvline(plot_xvals[np.argmax(p_force(plot_xvals))], color='g', ls='--', lw=2, zorder=5)
    
    # calculate binned averages
    range_min = -30
    range_max = 50
    bin_edges = np.linspace(range_min, range_max, 5)
    avg_bin_short, bins_short, binnumber_short = stats.binned_statistic(all_neuron_spikes[:,0],all_neuron_spikes[:,1], 'mean', bin_edges, (range_min, range_max))

    ax1.plot([-20,0,20,40], avg_bin_short, c='k', lw=4, zorder=7)
    # ax1.set_yscale('log')

    ax1.set_xticks([-30,-15,0,15,30,45])
    ax1.set_xticklabels([-30,-15,0,15,30,45])
    
    ax2.set_xticks([-30,-15,0,15,30,45])
    ax2.set_xticklabels([-30,-15,0,15,30,45])
    
    ax1.set_ylim([0,5])
    ax2.set_ylim([0,20])
    
    ax1.set_ylabel('Mean spikerate (Hz)')
    ax1.set_xlabel('∆Speed (cm/sec)')
    
    ax2.set_ylabel('Total force (a.u.)')
    ax2.set_xlabel('∆Speed (cm/sec)')

    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    
    print("======= RESULTS =======")
    print("apex spike fit" + str(plot_xvals[np.argmax(p_spike(plot_xvals))]))
    print("apex force fit" + str(plot_xvals[np.argmax(p_force(plot_xvals))]))
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'sim_ol_speeddiff')
    fname = loc_info['figure_output_path'] + os.sep + 'sim_ol_speeddiff' + os.sep + 'ol_speeddiff_ol' + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

def plot_thresh_range():
    
    thresholds = np.array([1,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2,2.125,2.25])
    data_list = ["C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1.125\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1.25\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1.375\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1.5\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1.625\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_1.875\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_2\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_2.125\\data.csv",
                 "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_2.25\\data.csv"]#,
                 # "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_2.375\\data.csv",
                 # "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 2105102\\thresh_range\\behav_trials_100neurons_(300, 50)noise_thresh_2.5\\data.csv"]
    
    fig = plt.figure(figsize=(3,7))
    ax1 = fig.add_subplot(1,1,1)
    
    init_error = np.zeros((thresholds.shape[0],))
    final_error = np.zeros((thresholds.shape[0],))
    for i,dl in enumerate(data_list):
        sim_data = np.loadtxt(dl, delimiter=',', skiprows=1)
        init_error[i] = np.mean(np.abs(sim_data[:,1])) 
        final_error[i] = np.mean(np.abs(sim_data[:,2]))

    
    poly_fit = np.polyfit(thresholds, final_error, 2)
    fit_obj = np.poly1d(poly_fit)
    fit_xvals = np.linspace(0.94, 2.25, 100)
    ax1.plot(fit_xvals, fit_obj(fit_xvals), c='k', ls='-', lw=2, zorder=5)

    ax1.scatter(thresholds, final_error, c='r', zorder=6)
    
    ax1.set_ylabel('Mean error (cm)')
    ax1.set_xlabel('Threshold (a.u.)')
    
    sns.despine(ax=ax1, right=True, top=True)
    
    ax1.set_ylim([0,13])
    
    print("======== RESULTS =========")
    print(fit_xvals[np.argmin(fit_obj(fit_xvals))])
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'threshold_range')
    fname = loc_info['figure_output_path'] + os.sep + 'threshold_range' + os.sep + 'threshold_range' + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)

if __name__ == '__main__':
    
    # Plot traces with single neuron Vm trace, injected current and running speed
    # plot_trial = np.arange(0,100,1)
    # plot_neuron = 2
    # for pltr in plot_trial:
    #     plot_ol_traces(pltr, plot_neuron)
    #     plot_vr_traces(pltr, plot_neuron)
    
    
    # plot_ol_traces(4, 2)
    # plot_ol_traces(81, 1)    
    # plot_ol_traces(1, 1)
    # plot_ol_traces(67, 1)
    # plot_ol_traces(90, 2)
    # plot_ol_traces(90, 1)
    # plot_vr_traces(90, 1)
    # plot_control_traces()
    
    # plot_single_comp_l_vr_traces(11, 0)
    # plot_single_comp_l_ol_traces(11, 0)
    # plot_single_comp_s_vr_traces(4, 0)
    # plot_single_comp_s_ol_traces(19, 0)
    # plot_thresh_range()
    
    
    # Plot scatterplot for spikerate and force/trial for openloop simulation run
    # sim_speeddiff_response() # run for fig 5 plasma scatterplot
    
    # Plot mouse location, attractor location and corresponding force as well as delta loc
    plot_trial = np.arange(0,20,1)
    for pltr in plot_trial:
        # plot_vr_loc_and_force(pltr, 1)
        # plot_ol_loc_and_force(pltr + 20, 1)
        plot_cont_loc_and_force(pltr + 40, 1, [0,0.1])
        # plot_single_s_loc_and_force(pltr, 1, 'single_s')
        # plot_single_s_ol_loc_and_force(pltr, 1, 'single_s_ol')
        # plot_single_l_loc_and_force(pltr, 1, 'single_l')
        # plot_single_l_ol_loc_and_force(pltr, 1, 'single_l_ol')