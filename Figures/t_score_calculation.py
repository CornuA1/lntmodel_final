# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:56:14 2021

@author: Keith, Lukas
"""
import numpy as np
import h5py
import warnings
import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
from scipy import stats
# import seaborn as sns
import scipy.io as sio

# sns.set_style("white")

#os.chdir('C:/Users/Keith/Documents/GitHub/LNT')
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append("./Analysis")
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT'+ os.sep + "Imaging")
#sys.path.append('C:/Users/Keith/Documents/GitHub/OASIS-master')

# os.chdir(r'C:\Users\Lou\Documents\repos\LNT')
# sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Analysis")
# sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Imaging")
# sys.path.append(r'C:\Users\Lou\Documents\repos\OASIS-master')

from filter_trials import filter_trials
from load_behavior_data import load_data
from rewards import rewards
from licks import licks_nopost as licks
from scipy.signal import butter, filtfilt
from load_filelist_model import load_filelist

SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'

def load_data_h5(h5path, sess):
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()
    return raw_ds, licks_ds, reward_ds

def load_raw_data(raw_filename, sess):
#    raw_data = load_data(raw_filename, 'vr')
    raw_data = np.genfromtxt(raw_filename, delimiter=';')
    all_licks = licks(raw_data)
    trial_licks = all_licks[np.in1d(all_licks[:, 3], [3, 4]), :]
    reward =  rewards(raw_data)
    return raw_data, trial_licks, reward

def t_score_cal(h5path, sess, fname, load_raw=False):
    # load data

    if load_raw == False:
        raw_ds, licks_ds, reward_ds = load_data_h5(h5path, sess)
    elif len(sess) == 2:
        # ipdb.set_trace()
        raw_filename = h5path + os.sep + sess[0] + os.sep + sess[1]
        raw_ds, licks_ds, reward_ds = load_raw_data(raw_filename, sess)
    elif len(sess) == 3:
        print('No.')

    fl_diff = 0
    t_score = 0

    # make array of y-axis locations for licks. If clause to check for empty arrays
    if np.size(licks_ds) > 0 or np.size(reward_ds) > 0:
        # only plot trials where either a lick and/or a reward were detected
        # therefore: pull out trial numbers from licks and rewards dataset and map to
        # a new list of rows used for plotting
        short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
        long_trials = filter_trials( raw_ds, [], ['tracknumber',4])

        # short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],short_trials)
        # long_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],long_trials)


        # get trial numbers to be plotted
        # ipdb.set_trace()
        reward_ds[:,3] = reward_ds[:,3]
        lick_trials = np.unique(licks_ds[:,2])
        reward_trials = np.unique(reward_ds[:,3])
        scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
        scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
        scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
        scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
        scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

        first_lick_short = []
        first_lick_short_trials = []
        first_lick_long = []
        first_lick_long_trials = []
        for r in scatter_rowlist_map:
            licks_all = licks_ds[licks_ds[:,2]==r,:]
            licks_all = licks_all[licks_all[:,1]>150,:]
            if not licks_all.size == 0:
                licks_all = licks_all[licks_all[:,1]>240,:]
            else:
                rew_lick = reward_ds[reward_ds[:,3]==r,:]
                if rew_lick.size > 0:
                    if rew_lick[0][5] == 1.0:
                        licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
#                        if r%20 <= 10:
#                            licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 3]])
#                        else:
#                            licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 4]])
            if licks_all.shape[0]>0:
                lick = licks_all[0]
                if lick[3] == 3:
                    first_lick_short.append(lick[1])
                    first_lick_short_trials.append(r)
                elif lick[3] == 4:
                    first_lick_long.append(lick[1])
                    first_lick_long_trials.append(r)

        if np.size(first_lick_short) > 10:
            fl_short_running_avg = np.convolve(first_lick_short,np.ones(10),'valid')/10

        if np.size(first_lick_long) > 10:
            fl_long_running_avg = np.convolve(first_lick_long,np.ones(10),'valid')/10

        # bootstrap differences between pairs of first lick locations
        if np.size(first_lick_short) > 5 and np.size(first_lick_long) > 5:
            num_shuffles = 10000
            short_bootstrap = np.random.choice(first_lick_short,num_shuffles)
            long_bootstrap = np.random.choice(first_lick_long,num_shuffles)
            bootstrap_diff = long_bootstrap - short_bootstrap
            # tval,pval = stats.ttest_1samp(bootstrap_diff,0)
            # pval = np.size(np.where(bootstrap_diff < 0))/num_shuffles
            fl_diff = np.mean(bootstrap_diff)/np.std(bootstrap_diff)

            # sns.distplot(bootstrap_diff,ax=ax2)
            # vl_handle = ax2.axvline(np.mean(bootstrap_diff),c='b')
            # vl_handle.set_label('z-score = ' + str(fl_diff))
            # ax2.legend()



        # calculate the confidence intervals for first licks from a bootstrapped distribution
        # number of resamples
        bootstrapdists = 100
        # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
        fl_short_bootstrap = np.empty((len(first_lick_short),bootstrapdists))
        fl_short_bootstrap[:] = np.nan
        # vector holding bootstrap variance estimate
        bt_mean_diff = np.empty((bootstrapdists,))
        bt_mean_diff[:] = np.nan

        for j in range(bootstrapdists):
            if len(first_lick_short) > 0:
                fl_short_bootstrap[:,j] = np.random.choice(first_lick_short, len(first_lick_short))
                bt_mean_diff[j] = np.nanmedian(fl_short_bootstrap[:,j]) - np.nanmedian(first_lick_short)
            else:
                bt_mean_diff[j] = np.nan
        bt_CI_5_short = np.percentile(bt_mean_diff[:],5)
        bt_CI_95_short = np.percentile(bt_mean_diff[:],95)

        # calculate the confidence intervals for first licks from a bootstrapped distribution
        # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
        fl_long_bootstrap = np.empty((len(first_lick_long),bootstrapdists))
        fl_long_bootstrap[:] = np.nan
        # vector holding bootstrap variance estimate
        bt_mean_diff = np.empty((bootstrapdists,))
        bt_mean_diff[:] = np.nan

        for j in range(bootstrapdists):
            if len(first_lick_long) > 0:
                fl_long_bootstrap[:,j] = np.random.choice(first_lick_long, len(first_lick_long))
                bt_mean_diff[j] = np.nanmedian(fl_long_bootstrap[:,j]) - np.nanmedian(first_lick_long)
            else:
                bt_mean_diff[j] = np.nan
        bt_CI_5_long = np.percentile(bt_mean_diff[:],5)
        bt_CI_95_long = np.percentile(bt_mean_diff[:],95)

#        if np.nanmedian(first_lick_long)+bt_CI_5_long > np.nanmedian(first_lick_short) or np.nanmedian(first_lick_short)+bt_CI_95_short < np.nanmedian(first_lick_long):
#            print('significant!')

        if np.size(first_lick_short) > 0 and np.size(first_lick_long) > 0:
            t_score = np.median(first_lick_long) - np.median(first_lick_short)

    return t_score


def simple_t_score(h5path):
    # load data
    raw_ds, licks_ds, reward_ds = load_raw_data(h5path, 'okay')

    fl_diff = 0
    t_score = 0

    # make array of y-axis locations for licks. If clause to check for empty arrays
    if np.size(licks_ds) > 0 or np.size(reward_ds) > 0:
        # only plot trials where either a lick and/or a reward were detected
        # therefore: pull out trial numbers from licks and rewards dataset and map to
        # a new list of rows used for plotting
        short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
        long_trials = filter_trials( raw_ds, [], ['tracknumber',4])

        # short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],short_trials)
        # long_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[100,200]],long_trials)


        # get trial numbers to be plotted
        # ipdb.set_trace()
        reward_ds[:,3] = reward_ds[:,3]
        lick_trials = np.unique(licks_ds[:,2])
        reward_trials = np.unique(reward_ds[:,3])
        scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
        scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
        scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
        scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
        scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

        first_lick_short = []
        first_lick_short_trials = []
        first_lick_long = []
        first_lick_long_trials = []
        for r in scatter_rowlist_map:
            licks_all = licks_ds[licks_ds[:,2]==r,:]
            licks_all = licks_all[licks_all[:,1]>150,:]
            if not licks_all.size == 0:
                licks_all = licks_all[licks_all[:,1]>240,:]
            else:
                rew_lick = reward_ds[reward_ds[:,3]==r,:]
                if rew_lick.size > 0:
                    if rew_lick[0][5] == 1.0:
                        licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
#                        if r%20 <= 10:
#                            licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 3]])
#                        else:
#                            licks_all = np.asarray([[0, rew_lick[0,1], rew_lick[0,3], 4]])
            if licks_all.shape[0]>0:
                lick = licks_all[0]
                if lick[3] == 3:
                    first_lick_short.append(lick[1])
                    first_lick_short_trials.append(r)
                elif lick[3] == 4:
                    first_lick_long.append(lick[1])
                    first_lick_long_trials.append(r)

        if np.size(first_lick_short) > 10:
            fl_short_running_avg = np.convolve(first_lick_short,np.ones(10),'valid')/10

        if np.size(first_lick_long) > 10:
            fl_long_running_avg = np.convolve(first_lick_long,np.ones(10),'valid')/10

        # bootstrap differences between pairs of first lick locations
        if np.size(first_lick_short) > 5 and np.size(first_lick_long) > 5:
            num_shuffles = 10000
            short_bootstrap = np.random.choice(first_lick_short,num_shuffles)
            long_bootstrap = np.random.choice(first_lick_long,num_shuffles)
            bootstrap_diff = long_bootstrap - short_bootstrap
            # tval,pval = stats.ttest_1samp(bootstrap_diff,0)
            # pval = np.size(np.where(bootstrap_diff < 0))/num_shuffles
            fl_diff = np.mean(bootstrap_diff)/np.std(bootstrap_diff)

            # sns.distplot(bootstrap_diff,ax=ax2)
            # vl_handle = ax2.axvline(np.mean(bootstrap_diff),c='b')
            # vl_handle.set_label('z-score = ' + str(fl_diff))
            # ax2.legend()



        # calculate the confidence intervals for first licks from a bootstrapped distribution
        # number of resamples
        bootstrapdists = 100
        # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
        fl_short_bootstrap = np.empty((len(first_lick_short),bootstrapdists))
        fl_short_bootstrap[:] = np.nan
        # vector holding bootstrap variance estimate
        bt_mean_diff = np.empty((bootstrapdists,))
        bt_mean_diff[:] = np.nan

        for j in range(bootstrapdists):
            if len(first_lick_short) > 0:
                fl_short_bootstrap[:,j] = np.random.choice(first_lick_short, len(first_lick_short))
                bt_mean_diff[j] = np.nanmedian(fl_short_bootstrap[:,j]) - np.nanmedian(first_lick_short)
            else:
                bt_mean_diff[j] = np.nan
        bt_CI_5_short = np.percentile(bt_mean_diff[:],5)
        bt_CI_95_short = np.percentile(bt_mean_diff[:],95)

        # calculate the confidence intervals for first licks from a bootstrapped distribution
        # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
        fl_long_bootstrap = np.empty((len(first_lick_long),bootstrapdists))
        fl_long_bootstrap[:] = np.nan
        # vector holding bootstrap variance estimate
        bt_mean_diff = np.empty((bootstrapdists,))
        bt_mean_diff[:] = np.nan

        for j in range(bootstrapdists):
            if len(first_lick_long) > 0:
                fl_long_bootstrap[:,j] = np.random.choice(first_lick_long, len(first_lick_long))
                bt_mean_diff[j] = np.nanmedian(fl_long_bootstrap[:,j]) - np.nanmedian(first_lick_long)
            else:
                bt_mean_diff[j] = np.nan
        bt_CI_5_long = np.percentile(bt_mean_diff[:],5)
        bt_CI_95_long = np.percentile(bt_mean_diff[:],95)

#        if np.nanmedian(first_lick_long)+bt_CI_5_long > np.nanmedian(first_lick_short) or np.nanmedian(first_lick_short)+bt_CI_95_short < np.nanmedian(first_lick_long):
#            print('significant!')

        if np.size(first_lick_short) > 0 and np.size(first_lick_long) > 0:
            t_score = np.median(first_lick_long) - np.median(first_lick_short)

    return t_score, licks_ds[-1][2]


def run_tscores(MOUSE,session1,session2):
    SESSION = [session1,session2]
    data_path = 'D:/Lukas/data/animals_raw' + os.sep + MOUSE
    t_scores = []
    
    for s in SESSION:
        t_scores.append(t_score_cal(data_path, s,  MOUSE+s[0], True))
        
    summary_path = data_path + os.sep + MOUSE + 'summaryMatchFile.mat'
    summary_dict =  sio.loadmat(summary_path)
    summary_dict['naive_score'] = t_scores[0]
    summary_dict['expert_score'] = t_scores[1]
    sio.savemat(summary_path, summary_dict)
    print('done...')
    

def run_t_score_mouse(MOUSE):
    base_path = 'D:/Lukas/data/animals_raw'
    data_path = base_path + os.sep + MOUSE
    names = []
    for root, dirs, files in os.walk(data_path, topdown=False):
       for name in files:
           if '.csv' in name and 'openloop' not in name and 'old' not in root and 'ol' not in root:
               if int(root[-8:]) <= 20191217:
                   names.append(os.path.join(root, name))
    t_scores = []
    tiral_nums = []
    for i in names:
        t_s, t_n = simple_t_score(i)
        t_scores.append(t_s)
        tiral_nums.append(t_n)
    return t_scores, tiral_nums

def run_all_mice():
    mice = ['LF191022_1','LF191022_2','LF191022_3','LF191023_blank','LF191023_blue','LF191024_1']
    mice_dict = {}
    for m in mice:
        perfor, trial_num = run_t_score_mouse(m)
        perfor_save = m+'_performance'
        number_save = m+'_trial_numbers'
        mice_dict[perfor_save] = perfor
        mice_dict[number_save] = trial_num
    sio.savemat('D:/Lukas/data/animals_raw/table.mat', mice_dict)
    
def calc_allo_perct(fileDIR):    
    loaded_data = sio.loadmat(fileDIR)
    coefData = loaded_data['aveCoefAll'][:,0]
    
    for i in range(len(coefData)):
        if coefData[i] < 0:
            coefData[i] = 0

    reg = 0
    total = 0
    for i in range(len(coefData)):
        if coefData[i] >= 0:
            if i < 36:
                reg += coefData[i]
            total += coefData[i]
    allo_perct = float(reg/total)
    
    return allo_perct
    
def runAnalysisAllMiceAllSessions():
    data_path_root = "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\data_2p"
    mice = ['LF191022_1', 'LF191022_2', 'LF191022_3', 'LF191023_blank', 'LF191023_blue', 'LF191024_1']
    
    for m in mice:
        data_path = data_path_root + os.sep + m
        names = []
        roots = []
        return_dict = {}
        for root, dirs, files in os.walk(data_path, topdown=False):
           for name in files:
               if 'summaryFile' in name and 'openloop' not in name and 'old' not in root and 'ol' not in root:
                   if int(root[-8:]) <= 20191217:
                       names.append(os.path.join(root, name))
                       roots.append(root)

    csvs = []
    for x in roots:
        for root, dirs, files in os.walk(x, topdown=False):
           for name in files:
               if '.csv' in name and 'old' not in root:
                   csvs.append(os.path.join(root, name))
    allo_scatter = []
    trial_scores = []
    for x in range(len(csvs)):
#        print(names[x])
#        print(csvs[x])
        dict_cur_name = str(roots[x][len(data_path)+1:-9]+'_'+roots[x][-8:])
        allo_perct_cur = calc_allo_perct(names[x])
        trial_score, number_trials = simple_t_score(csvs[x])
        return_dict[dict_cur_name] = (trial_score, number_trials, allo_perct_cur)
        allo_scatter.append(allo_perct_cur)
        trial_scores.append(trial_score)
    sio.savemat("C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\data_2p\\total_analysis.mat", return_dict)
    return allo_scatter, trial_scores

def runTScore_all_mice():
    filedict = load_filelist()
    mice = ['LF191022_1', 'LF191022_2', 'LF191022_3', 'LF191023_blank', 'LF191023_blue', 'LF191024_1']
    tscores = []
    
    for m in mice:
        behavior_files = filedict[m]['training_data']
        mouse_tscore = []
        for bf in behavior_files:
            trial_score, number_trials = simple_t_score(bf)
            mouse_tscore.append(trial_score)
        tscores.append(mouse_tscore)
        
    maxlen = 0
    for ts in tscores:
        maxlen = np.amax([maxlen,len(ts)])
     
    tscore_mat = np.zeros((len(mice), maxlen)) * np.nan
    
    for i,ts in enumerate(tscores):
        tscore_mat[i,0:len(ts)] = ts
        
    ts_sem = stats.sem(tscore_mat, axis=0, nan_policy='omit')
    
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    for ts in tscores:
        ax1.plot(np.arange(0,len(ts),1), ts, c='0.7', zorder=3)
        
    # ax1.plot(np.arange(0,tscore_mat.shape[1],1), np.nanmean(tscore_mat,axis=0), c='k', lw=3, zorder=4)
    
    ax1.errorbar(np.arange(0,tscore_mat.shape[1]),
                 np.nanmean(tscore_mat,axis=0),
                 yerr = ts_sem,c='k',lw=3,zorder=5)
    
    ax1.set_xticks([0,2,4,6,8,10,12,14,16,18])
    ax1.set_xticklabels([0,2,4,6,8,10,12,14,16,18])
    ax1.set_ylabel('Task score (cm)')
    ax1.set_xlabel('Session #')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 1/tscore.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 1/tscore.svg")
    

if __name__ == '__main__':
    
    # allo_scatter, trial_scores = runAnalysisAllMiceAllSessions()    
    runTScore_all_mice()
        
        
                   
                   
                   
                   