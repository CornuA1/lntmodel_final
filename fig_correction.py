# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:12:33 2021

@author: lfisc
"""

import csv, yaml, os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import ttest_rel
plt.rcParams['svg.fonttype'] = 'none'

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.safe_load(f)

fformat = '.svg'

def correction_plot():
    ''' plot how well the CAN was corrected after landmark inputs '''
    
    correction_real = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210426\\srug_real\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    correction_control = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210426\\srug_cont\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    
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
    
    fname = loc_info['figure_output_path'] + os.sep + 'real_v_control' + fformat
    plt.savefig(fname, dpi=100)

def correction_plot_1comp():
    ''' plot how well the CAN was corrected after landmark inputs '''
    
    correction_real = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210426\\single_comp_s\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    correction_control = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\simulation_output\\EC2 210426\\single_comp_l\\behav_trials_100neurons_(300, 50)noise_thresh_1.75\\data.csv"
    
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

if __name__ == '__main__':
    correction_plot()
    # correction_plot_1comp()