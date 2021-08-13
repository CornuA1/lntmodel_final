# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:28:31 2021

@author: lfisc

plot heatmap of glm coefficients

"""

import csv, os, yaml, warnings
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")
warnings.filterwarnings('ignore')


# load yaml file with local filepaths
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
    
def allo_fraction(coeff_vec):
    # coeff_vec[coeff_vec < 0] = 0
    coeff_vec = np.abs(coeff_vec)
    allo_coeffs = np.sum(coeff_vec[1:37])
    ego_coeffs = np.sum(coeff_vec[37:])
    
    if np.sum(coeff_vec) > 0.0:
        allo_frac = allo_coeffs / (allo_coeffs + ego_coeffs)
    else:
        allo_frac = np.nan

    return allo_frac
    
def make_folder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
def plot_heatmap(sessions, fname_detail):
    
    fig = plt.figure(figsize=(15,4))
    gs = fig.add_gridspec(20, 6)
    ax1 = fig.add_subplot(gs[10:20,0:1])
    ax2 = fig.add_subplot(gs[10:20,1:2])
    ax3 = fig.add_subplot(gs[10:20,2:3])
    ax4 = fig.add_subplot(gs[10:20,3:4])
    ax5 = fig.add_subplot(gs[10:20,4:5])
    ax6 = fig.add_subplot(gs[10:20,5:6])
    
    ax7 = fig.add_subplot(gs[0:7,0:1])
    ax8 = fig.add_subplot(gs[0:7,1:2])
    ax9 = fig.add_subplot(gs[0:7,2:3])
    ax10 = fig.add_subplot(gs[0:7,3:4])
    ax11 = fig.add_subplot(gs[0:7,4:5])
    ax12 = fig.add_subplot(gs[0:7,5:6])
    
    # ax1 = fig.add_subplot(1,6,1)  
    # ax2 = fig.add_subplot(1,6,2)  
    # ax3 = fig.add_subplot(1,6,3)  
    # ax4 = fig.add_subplot(1,6,4)  
    # ax5 = fig.add_subplot(1,6,5)  
    # ax6 = fig.add_subplot(1,6,6)  
    
    
    all_glm_data = []
    for s in sessions:
        glm_data = loadmat(loc_info['imaging_dir'] + s[0] + '_' + s[1] + '_glm_data.mat')
        all_glm_data.append(glm_data['data'])
      
    ax1 = sns.heatmap(all_glm_data[0].T, vmin=0, vmax=0.5, cbar=False, ax=ax1)
    ax2 = sns.heatmap(all_glm_data[1].T, vmin=0, vmax=0.5, cbar=False, ax=ax2)
    ax3 = sns.heatmap(all_glm_data[2].T, vmin=0, vmax=0.5, cbar=False, ax=ax3)
    ax4 = sns.heatmap(all_glm_data[3].T, vmin=0, vmax=0.5, cbar=False, ax=ax4)
    ax5 = sns.heatmap(all_glm_data[4].T, vmin=0, vmax=0.5, cbar=False, ax=ax5)
    ax6 = sns.heatmap(all_glm_data[5].T, vmin=0, vmax=0.5, cbar=False, ax=ax6)
    
    ax1.set_xticks([36])
    ax2.set_xticks([36])
    ax3.set_xticks([36])
    ax4.set_xticks([36])
    ax5.set_xticks([36])
    ax6.set_xticks([36])
    
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])
    
    ax7.bar(np.arange(0,all_glm_data[0][1:36].T.shape[1]), np.mean(all_glm_data[0][1:36].T, axis=0), color='#EC008C', edgecolor=None, linewidth=0)
    ax7.bar(np.arange(36,36+all_glm_data[0][36:].T.shape[1]), np.mean(np.abs(all_glm_data[0][36:].T), axis=0), color='#009444', edgecolor=None, linewidth=0)
    ax7.set_xlim([0,53])
    ax7.set_ylim([0,0.25])
    
    ax8.bar(np.arange(0,all_glm_data[1][1:36].T.shape[1]), np.mean(all_glm_data[1][1:36].T, axis=0), color='#EC008C', edgecolor=None, linewidth=0)
    ax8.bar(np.arange(36,36+all_glm_data[1][36:].T.shape[1]), np.mean(np.abs(all_glm_data[1][36:].T), axis=0), color='#009444', edgecolor=None, linewidth=0)
    ax8.set_xlim([0,53])
    ax8.set_ylim([0,0.25])
    
    ax9.bar(np.arange(0,all_glm_data[2][1:36].T.shape[1]), np.mean(all_glm_data[2][1:36].T, axis=0), color='#EC008C', edgecolor=None, linewidth=0)
    ax9.bar(np.arange(36,36+all_glm_data[2][36:].T.shape[1]), np.mean(np.abs(all_glm_data[2][36:].T), axis=0), color='#009444', edgecolor=None, linewidth=0)
    ax9.set_xlim([0,53])
    ax9.set_ylim([0,0.25])
    
    ax10.bar(np.arange(0,all_glm_data[3][1:36].T.shape[1]), np.mean(all_glm_data[3][1:36].T, axis=0), color='#EC008C', edgecolor=None, linewidth=0)
    ax10.bar(np.arange(36,36+all_glm_data[3][36:].T.shape[1]), np.mean(np.abs(all_glm_data[3][36:].T), axis=0), color='#009444', edgecolor=None, linewidth=0)
    ax10.set_xlim([0,53])
    ax10.set_ylim([0,0.25])
    
    ax11.bar(np.arange(0,all_glm_data[4][1:36].T.shape[1]), np.mean(all_glm_data[4][1:36].T, axis=0), color='#EC008C', edgecolor=None, linewidth=0)
    ax11.bar(np.arange(36,36+all_glm_data[4][36:].T.shape[1]), np.mean(np.abs(all_glm_data[4][36:].T), axis=0), color='#009444', edgecolor=None, linewidth=0)
    ax11.set_xlim([0,53])
    ax11.set_ylim([0,0.25])
    
    ax12.bar(np.arange(0,all_glm_data[5][1:36].T.shape[1]), np.mean(all_glm_data[5][1:36].T, axis=0), color='#EC008C', edgecolor=None, linewidth=0)
    ax12.bar(np.arange(36,36+all_glm_data[5][36:].T.shape[1]), np.mean(np.abs(all_glm_data[5][36:].T), axis=0), color='#009444', edgecolor=None, linewidth=0)
    ax12.set_xlim([0,53])
    ax12.set_ylim([0,0.25])
    
    ax7.set_yticks([0,0.2])
    sns.despine(ax=ax7, right=True, top=True)
    
    ax8.set_yticks([0,0.2])
    sns.despine(ax=ax8, right=True, top=True)
    
    ax9.set_yticks([0,0.2])
    sns.despine(ax=ax9, right=True, top=True)
    
    ax10.set_yticks([0,0.2])
    sns.despine(ax=ax10, right=True, top=True)
    
    ax11.set_yticks([0,0.2])
    sns.despine(ax=ax11, right=True, top=True)
    
    ax12.set_yticks([0,0.2])
    sns.despine(ax=ax12, right=True, top=True)

    make_folder("C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\manuscript\\Figure 1 supp 1\\")
    fname = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\manuscript\\Figure 1 supp 1\\" + fname_detail + ".svg"
    fig.savefig(fname, format='svg')
    print("saved " + fname)
    
    
    fig = plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot(1,1,1)
    
    all_mean_coefs = np.zeros((6,53))
    all_mean_coefs[0,:] = np.mean(np.abs(all_glm_data[0].T), axis=0)
    all_mean_coefs[1,:] = np.mean(np.abs(all_glm_data[1].T), axis=0)
    all_mean_coefs[2,:] = np.mean(np.abs(all_glm_data[2].T), axis=0)
    all_mean_coefs[3,:] = np.mean(np.abs(all_glm_data[3].T), axis=0)
    all_mean_coefs[4,:] = np.mean(np.abs(all_glm_data[4].T), axis=0)
    all_mean_coefs[5,:] = np.mean(np.abs(all_glm_data[5].T), axis=0)
    
    # below we set 3 coefficients to NaN since they code for locations that don't exist on the short track
    all_mean_coefs[:,16:19] = np.nan
    mean_mean_coefs = np.nanmean(all_mean_coefs, axis=0)
    

    ax1.bar(np.arange(1,37), mean_mean_coefs[1:37], color = '#EC008C', edgecolor=None, linewidth=0)
    ax1.bar(np.arange(37,53), mean_mean_coefs[37:], color = '#009444', edgecolor=None, linewidth=0 )
    
    ax1.set_ylim([0,0.2])
    
    ax1.set_yticks([0,0.2])
    sns.despine(ax=ax1, right=True, top=True)
    
    
    fname = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\manuscript\\Figure 1 supp 1\\" + "all_mice_" +  fname_detail + ".svg"
    fig.savefig(fname, format='svg')
    print("saved " + fname)

    fig = plt.figure(figsize=(1,3))
    ax1 = fig.add_subplot(1,1,1)

    bar_allo_mean = np.nanmean(all_mean_coefs[:,1:37],axis=1)
    bar_allo_sem = sp.stats.sem(bar_allo_mean,nan_policy='omit')
    
    bar_ego_mean = np.nanmean(all_mean_coefs[:,37:],axis=1)
    bar_ego_sem = sp.stats.sem(bar_ego_mean,nan_policy='omit')
    
    # allo_frac = allo_fraction(np.mean(all_mean_coefs,axis=0))
    
    # print('allo fraction: ' + str(allo_frac))
    
    # sem_mean_coefs = sp.stats.sem(all_mean_coefs, axis=0)
    ax1.bar([0,1], [np.mean(bar_allo_mean), np.mean(bar_ego_mean)], color=['#EC008C', '#009444'], width=0.8)
    ax1.errorbar([0,1], [np.mean(bar_allo_mean), np.mean(bar_ego_mean)], yerr=[bar_allo_sem,bar_ego_sem], ecolor='k')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
    ax1.set_ylim([0,0.05])
    ax1.set_xlim([-0.7,1.7])
    ax1.set_xticks([])

    print(sp.stats.ttest_ind(bar_allo_mean,bar_ego_mean,nan_policy='omit'))    
    print(sp.stats.mannwhitneyu(bar_allo_mean,bar_ego_mean)) 
    fname = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\manuscript\\Figure 1\\" + "egoallo_weights_" +  fname_detail + ".svg"
    fig.savefig(fname, format='svg')
    print("saved " + fname)

    # "C:\Users\lfisc\Work\Projects\Lntmodel\data_2p\dataset\LF191022_1_20191125_glm_data.mat"

if __name__ == '__main__':
    naive = [('LF191022_1','20191115'),('LF191022_3','20191113'),('LF191023_blue','20191119'),('LF191022_2','20191116'),('LF191023_blank','20191114'),('LF191024_1','20191114')]
    expert = [('LF191022_1','20191209'),('LF191022_2','20191210'),('LF191022_3','20191210'),('LF191023_blank','20191210'),('LF191023_blue','20191210'),('LF191024_1','20191210')]
    # expert = [('LF191022_1','20191209'),('LF191022_3','20191207'),('LF191023_blue','20191208'),('LF191022_2','20191210'),('LF191023_blank','20191210'),('LF191024_1','20191210')]
    # expert = [('LF191022_1','20191204'),('LF191022_2','20191210'),('LF191022_3','20191207'),('LF191023_blank','20191206'),('LF191023_blue','20191204'),('LF191024_1','20191204')]
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
    
    plot_heatmap(naive, 'naive')
    plot_heatmap(expert, 'expert')
    # plot_heatmap(all_sessions, 'all')
    # plot_heatmap([['LF191022_1','20191115']])