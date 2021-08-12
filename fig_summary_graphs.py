# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:12:33 2021

@author: lfisc
"""

import yaml, os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import ttest_rel
import seaborn as sns
from sklearn import metrics
plt.rcParams['svg.fonttype'] = 'none'

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.safe_load(f)

fformat = '.svg'

def make_folder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

def correction_plot():
    ''' plot how well the CAN was corrected after landmark inputs '''
    
    correction_real = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210419\\srug_real\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    correction_control = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210419\\srug_cont\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    
    res_data_real = np.genfromtxt(correction_real, delimiter=',', skip_header=1)
    res_data_control = np.genfromtxt(correction_control, delimiter=',', skip_header=1)
    
    num_trials = res_data_real.shape[0]
    
    fig = plt.figure(figsize=(3,5))
    # gs = fig.add_gridspec(15, 1)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    # ax1.scatter(np.zeros((num_trials,)), res_data_real[:,1])
    for rdr in res_data_real:
        ax1.plot([0,1],np.abs(rdr[1:]), marker='o', mfc='w', c='0.8', lw='0.5', alpha=0.5)
    ax1.plot([0,1], np.mean(np.abs(res_data_real[:,1:]),0), c='k', marker='o')
    
    for rdc in res_data_control:
        ax2.plot([0,1],np.abs(rdc[1:]), marker='o', mfc='w', c='0.8', lw='0.5', alpha=0.5)
    ax2.plot([0,1], np.mean(np.abs(res_data_control[:,1:]),0), c='k', marker='o')
        
    ax1.set_xlim([-0.1,1.1])
    ax2.set_xlim([-0.1,1.1])
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    print("======= RESULTS =======")
    print("paired t-test sd: " + str(ttest_rel(np.abs(res_data_real[:,1]),np.abs(res_data_real[:,2]))))
    print("mean final error sd: " + str(np.mean(np.abs(res_data_real[:,2]))) + " +/- " + str(sp.stats.sem(np.abs(res_data_real[:,2]))))
    print("paired t-test sd: " + str(ttest_rel(np.abs(res_data_control[:,1]),np.abs(res_data_control[:,2]))))
    print("mean final error sd: " + str(np.mean(np.abs(res_data_control[:,2]))) + " +/- " + str(sp.stats.sem(np.abs(res_data_control[:,2]))))
    fname = loc_info['figure_output_path'] + os.sep + 'real_v_control' + fformat
    plt.savefig(fname, dpi=100)
    print('saved: ' + fname)
    print("======================")

def correction_plot_1comp():
    ''' plot how well the CAN was corrected after landmark inputs '''
    
    correction_real = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210419\\single_comp_s\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    correction_control = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210419\\single_comp_l\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    
    res_data_real = np.genfromtxt(correction_real, delimiter=',', skip_header=1)
    res_data_control = np.genfromtxt(correction_control, delimiter=',', skip_header=1)
    
    num_trials = res_data_real.shape[0]
    
    fig = plt.figure(figsize=(3,5))
    # gs = fig.add_gridspec(15, 1)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    # ax1.scatter(np.zeros((num_trials,)), res_data_real[:,1])
    for rdr in res_data_real:
        ax1.plot([0,1],np.abs(rdr[1:]), marker='o', mfc='w', c='0.8', lw='0.5', alpha=0.5)
    ax1.plot([0,1], np.mean(np.abs(res_data_real[:,1:]),0), c='k', marker='o')
    
    for rdc in res_data_control:
        ax2.plot([0,1],np.abs(rdc[1:]), marker='o', mfc='w', c='0.8', lw='0.5', alpha=0.5)
    ax2.plot([0,1], np.mean(np.abs(res_data_control[:,1:]),0), c='k', marker='o')
    
    print(ttest_rel(np.abs(res_data_real[:,1]),np.abs(res_data_real[:,2])))
    print(ttest_rel(np.abs(res_data_control[:,1]),np.abs(res_data_control[:,2])))
    
    ax1.set_xlim([-0.1,1.1])
    ax2.set_xlim([-0.1,1.1])
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    fname = loc_info['figure_output_path'] + os.sep + '1_comp_lin_v_nonlin' + fformat
    plt.savefig(fname, dpi=100)    
    
def loc_vs_force():
    ''' Plot location vs. force '''
    
    num_trials = 100
    f_bin_edges = np.arange(50,360,10)
    trials = np.arange(0,num_trials,1)

    
    fig = plt.figure(figsize=(4,8))
    # gs = fig.add_gridspec(15, 1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    fvloc = np.zeros((len(f_bin_edges)-1,0))
    for i in range(100):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc = np.hstack((fvloc,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
        
    ax1.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2, lw=0.5)
    ax1.fill_between(f_bin_edges[:-1], np.zeros((fvloc.shape[0],)), np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2)
    
    fvloc_c = np.zeros((len(f_bin_edges)-1,0))
    for i in range(100):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'srug_cont' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc_c = np.hstack((fvloc_c,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
     
    
    # ax1.plot(mouse_locs, force)
    ax2.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2, lw=0.5)
    ax2.fill_between(f_bin_edges[:-1], np.zeros((fvloc_c.shape[0],)), np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2)
    
    ax1.set_ylim([0,2.5])
    ax2.set_ylim([0,2.5])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    
    fvloc_all = np.zeros((fvloc.shape[1]))
    for i,fv in enumerate(fvloc.T):
        fvloc_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
        
    fvloc_c_all = np.zeros((fvloc_c.shape[1]))
    for i,fv in enumerate(fvloc_c.T):
        fvloc_c_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
    print("====== RESULTS ======")
    print("Mean AUC sd: " + str(np.mean(fvloc_all)) + " +/- " + str(sp.stats.sem(fvloc_all)))
    print("Mean AUC c: " + str(np.mean(fvloc_c_all)) + " +/- " + str(sp.stats.sem(fvloc_c_all)))
    
    fig.suptitle('Real vs. Control')
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'force_rc')
    fname = loc_info['figure_output_path'] + os.sep + 'force_rc' + fformat
    plt.savefig(fname, dpi=100)
    print('saved: ' + fname)
    print("====================")

def loc_vs_force_single_comp():
    ''' Plot location vs. force '''
    
    num_trials = 100
    f_bin_edges = np.arange(50,360,10)
    trials = np.arange(0,num_trials,1)

    
    fig = plt.figure(figsize=(4,8))
    # gs = fig.add_gridspec(15, 1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    fvloc = np.zeros((len(f_bin_edges)-1,0))
    for i in range(100):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'single_comp_s' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc = np.hstack((fvloc,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
        
    ax1.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2, lw=0.5)
    ax1.fill_between(f_bin_edges[:-1], np.zeros((fvloc.shape[0],)), np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2)
    
    fvloc = np.zeros((len(f_bin_edges)-1,0))
    for i in range(100):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'single_comp_l' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc = np.hstack((fvloc,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
     
    
    # ax1.plot(mouse_locs, force)
    ax2.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2, lw=4)
    ax2.fill_between(f_bin_edges[:-1], np.zeros((fvloc.shape[0],)), np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2)
    
    ax1.set_ylim([0,8])
    ax2.set_ylim([0,8])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    
    fig.suptitle('Single comp, supralinear vs. linear')
    
    ax1.set_xlabel('Location (cm)')
    ax1.set_ylabel('Force (a.u.)')
    
    ax2.set_xlabel('Location (cm)')
    ax2.set_ylabel('Force (a.u.)')
    
    
    plt.tight_layout()
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'force_single_comp')
    fname = loc_info['figure_output_path'] + os.sep + 'force_single_comp' + os.sep + 'lin_vs_nonlin_force' + fformat
    plt.savefig(fname, dpi=100)
    print('saved: ' + fname)

def loc_vs_force_single_comp_l_ol():
    ''' Plot location vs. force '''
    
    num_trials = 40
    f_bin_edges = np.arange(50,360,10)
    trials = np.arange(0,num_trials,1)

    fig = plt.figure(figsize=(4,5))
    # gs = fig.add_gridspec(15, 1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    fvloc = np.zeros((len(f_bin_edges)-1,0))
    for i in range(num_trials):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210507' + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_5neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc = np.hstack((fvloc,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
        
    ax1.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2, lw=4)
    ax1.fill_between(f_bin_edges[:-1], np.zeros((fvloc.shape[0],)), np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2)
    
    fvloc_c = np.zeros((len(f_bin_edges)-1,0))
    for i in range(num_trials):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'single_comp_l' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_5neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
        fvloc_c_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc_c = np.hstack((fvloc_c,fvloc_c_trial.reshape((fvloc_c_trial.shape[0],1))))
     
    
    # ax1.plot(mouse_locs, force)
    ax2.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2, lw=4)
    ax2.fill_between(f_bin_edges[:-1], np.zeros((fvloc_c.shape[0],)), np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2)
    
    ax1.set_ylim([0,6.5])
    ax2.set_ylim([0,6.5])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    
    fig.suptitle('Linear VR vs. OL')
    
    ax1.set_xlabel('Location (cm)')
    ax1.set_ylabel('Force (a.u.)')
    
    ax2.set_xlabel('Location (cm)')
    ax2.set_ylabel('Force (a.u.)')
    
    
    plt.tight_layout()
    
    fvloc_all = np.zeros((fvloc.shape[1]))
    for i,fv in enumerate(fvloc.T):
        fvloc_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
        
    fvloc_c_all = np.zeros((fvloc_c.shape[1]))
    for i,fv in enumerate(fvloc_c.T):
        fvloc_c_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
    print("====== RESULTS ======")
    print("Mean AUC sd: " + str(np.mean(fvloc_all)) + " +/- " + str(sp.stats.sem(fvloc_all)))
    print("Mean AUC c: " + str(np.mean(fvloc_c_all)) + " +/- " + str(sp.stats.sem(fvloc_c_all)))
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'force_single_comp')
    fname = loc_info['figure_output_path'] + os.sep + 'force_single_comp' + os.sep + 'l_vr_ol' + fformat
    plt.savefig(fname, dpi=100)
    print('saved: ' + fname)

def loc_vs_force_single_comp_s_ol():
    ''' Plot location vs. force '''
    
    num_trials = 40
    f_bin_edges = np.arange(50,360,10)
    trials = np.arange(0,num_trials,1)

    fig = plt.figure(figsize=(4,5))
    # gs = fig.add_gridspec(15, 1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    fvloc = np.zeros((len(f_bin_edges)-1,0))
    for i in range(num_trials):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210507' + os.sep + 'single_comp_s_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_5neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc = np.hstack((fvloc,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
        
    ax1.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2, lw=0.5)
    ax1.fill_between(f_bin_edges[:-1], np.zeros((fvloc.shape[0],)), np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2)
    
    fvloc_c = np.zeros((len(f_bin_edges)-1,0))
    for i in range(num_trials):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 210426' + os.sep + 'single_comp_s' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'single_comp_s_ol' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
        fvloc_c_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc_c = np.hstack((fvloc_c,fvloc_c_trial.reshape((fvloc_c_trial.shape[0],1))))
     
    
    # ax1.plot(mouse_locs, force)
    ax2.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2, lw=0.5)
    ax2.fill_between(f_bin_edges[:-1], np.zeros((fvloc_c.shape[0],)), np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2)
    
    ax1.set_ylim([0,8])
    ax2.set_ylim([0,8])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    
    fig.suptitle('supralinear VR vs ol')
    
    ax1.set_xlabel('Location (cm)')
    ax1.set_ylabel('Force (a.u.)')
    
    ax2.set_xlabel('Location (cm)')
    ax2.set_ylabel('Force (a.u.)')
    
    
    plt.tight_layout()
    
    
    fvloc_all = np.zeros((fvloc.shape[1]))
    for i,fv in enumerate(fvloc.T):
        fvloc_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
        
    fvloc_c_all = np.zeros((fvloc_c.shape[1]))
    for i,fv in enumerate(fvloc_c.T):
        fvloc_c_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
    print("====== RESULTS ======")
    print("Mean AUC sd: " + str(np.mean(fvloc_all)) + " +/- " + str(sp.stats.sem(fvloc_all)))
    print("Mean AUC c: " + str(np.mean(fvloc_c_all)) + " +/- " + str(sp.stats.sem(fvloc_c_all)))
    make_folder(loc_info['figure_output_path'] + os.sep + 'force_single_comp')
    fname = loc_info['figure_output_path'] + os.sep + 'force_single_comp' + os.sep + 's_vr_ol' + fformat
    plt.savefig(fname, dpi=100)
    print('saved: ' + fname)


def loc_vs_force_openloop():
    ''' Plot location vs. force for the openloop condition '''
    
    num_trials = 100
    f_bin_edges = np.arange(50,360,10)
    trials = np.arange(20,20+num_trials,1)

    
    fig = plt.figure(figsize=(4,5))
    # gs = fig.add_gridspec(15, 1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    fvloc = np.zeros((len(f_bin_edges)-1,0))
    for i in range(num_trials):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 21042' + os.sep + 'srug_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'single_comp_l_ol' + os.sep + 'behav_trials_5neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc = np.hstack((fvloc,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
        
    ax1.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2, lw=0.5)
    ax1.fill_between(f_bin_edges[:-1], np.zeros((fvloc.shape[0],)), np.nanmean(np.abs(fvloc),1), color='#EB008B', alpha = .2)
    
    fvloc_c = np.zeros((len(f_bin_edges)-1,0))
    for i in range(num_trials):
        results_vr_sess = loc_info['figure_output_path'] + os.sep + 'EC2 2104282' + os.sep + 'srug_ol_real' + os.sep + 'behav_trials_100neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        # results_vr_sess = loc_info['figure_output_path'] + os.sep + 'srug_real' + os.sep + 'behav_trials_10neurons_(300, 50)noise_thresh_1.75' + os.sep +  'results_trial_' + str(i) + '.npz'
        res = np.load(results_vr_sess)
        # timestamps = res['timestamps']
        mouse_locs = res['mouse_locs']
        # can_locs = res['can_locs']
        # v = res['v']
        # spikes = res['spikes']
        force = res['force']
        # neuron_I = res['neuron_I']
        
          
        fvloc_trial,plot_bins,_ = sp.stats.binned_statistic(mouse_locs,np.abs(force),'sum',bins=f_bin_edges)
        fvloc_c = np.hstack((fvloc_c,fvloc_trial.reshape((fvloc_trial.shape[0],1))))
        
    ax2.plot(f_bin_edges[:-1], np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2, lw=0.5)
    ax2.fill_between(f_bin_edges[:-1], np.zeros((fvloc_c.shape[0],)), np.nanmean(np.abs(fvloc_c),1), color='#EB008B', alpha = .2)
    
    ax1.set_ylim([0,2.5])
    ax2.set_ylim([0,2.5])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    
    fvloc_all = np.zeros((fvloc.shape[1]))
    for i,fv in enumerate(fvloc.T):
        fvloc_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
        
    fvloc_c_all = np.zeros((fvloc_c.shape[1]))
    for i,fv in enumerate(fvloc_c.T):
        fvloc_c_all[i] = metrics.auc(f_bin_edges[:-1], np.abs(fv))
    print("====== RESULTS ======")
    print("Mean AUC sd: " + str(np.mean(fvloc_all)) + " +/- " + str(sp.stats.sem(fvloc_all)))
    print("Mean AUC c: " + str(np.mean(fvloc_c_all)) + " +/- " + str(sp.stats.sem(fvloc_c_all)))
    
    fig.suptitle('VR vs. Openloop')
    make_folder(loc_info['figure_output_path'] + os.sep + 'force_rc')
    fname = loc_info['figure_output_path'] + os.sep + 'force_rc' + os.sep + 'forcediff_ol' + fformat
    plt.savefig(fname, dpi=100)   
    print('saved ' + fname)
    
if __name__ == '__main__':
    # correction_plot()
    # correction_plot_1comp()
    
    loc_vs_force()
    # loc_vs_force_openloop()
    # loc_vs_force_single_comp()
    # loc_vs_force_single_comp_l_ol()
    # loc_vs_force_single_comp_s_ol()