#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:40:23 2021

@author: lukasfischer

GLM coefficient mapping:
    Total: 53 coefficients
    0: intercept
    1-18: 20 cm spatial bins, short track (this includes 3 coefficients that are always 0 since they are for locations that don't exist on the track which are removed from analysis)
    19-36: 20 cm spatial bins, long track
    37: low speed/stationary (<0.33 cm/sec)
    38: high speed (>3,5 cm/sec)
    39: running speed (linear)
    40: lick
    41-43: reward short (3 predictors, 1 centered at reward point and 2 preceeding it by 60 and 120 frames (= ~1 and 2 seconds))
    44-46: trial start short (3 predictors, 1 centered at reward point and 2 succeeding it by 60 and 120 frames (= ~1 and 2 seconds))
    47-49: reward long (3 predictors, 1 centered at reward point and 2 preceeding it by 60 and 120 frames (= ~1 and 2 seconds))
    50-52: trial start long (3 predictors, 1 centered at reward point and 2 succeeding it by 60 and 120 frames (= ~1 and 2 seconds))
    
"""

import os, yaml
import numpy as np
import scipy.io as sio
from scipy import stats
from scipy import signal
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from scipy.signal import butter, filtfilt
from scipy.stats import siegelslopes
sns.set_style("white")
plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['xtick.major.size'] = 5
# plt.rcParams['xtick.major.width'] = 1
# plt.rcParams['ytick.major.size'] = 5
# plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
# plt.rcParams['xtick.top'] = False
# plt.rcParams['ytick.right'] = False

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

from load_filelist_model import load_filelist

fformat = 'png'

MAKE_PAIR_FIGURES = False
MAKE_SINGLE_TRACE = False
MAKE_SPEED_TRACE_FIGRE = True

# only one of the options below should be true at a time unless you want data from both
NAIVE_OL = True
EXPERT_S1_OL = False
EXPERT_S2_OL = True

FORCE_CC = False
FORCE_SHUFFLE_CC = False

CORR_MAXLAG = 800
COMP_OFFSET = 0
NUM_SHUFFLE = 2 # 100 # 2 is only if we don't actually do the shuffle comparison

ACORR_THRESH = 0.25

sns.despine()
   

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calculate_speed_df_corr(behavior_data, roi_df, mouse_speed_col=3, fname=''):
    max_trial = np.amax(behavior_data[-1,6]).astype('int')
    
    dF_trace = np.zeros((0,))    # df/f concatenated for all trials
    speed_trace = np.zeros((0,)) # mouse running speed concatenated for all trials
        
    for t in range(1,max_trial):
        # pull out current trial and corresponding dF data and bin it
        cur_trial = behavior_data[behavior_data[:,6]==t,:]
        cur_trial_speed = cur_trial[:,mouse_speed_col]
        cur_trial_dF_roi = roi_df[behavior_data[:,6]==t]
        if cur_trial[0,4] == 3 or cur_trial[0,4] == 4:
            dF_trace = np.hstack((dF_trace, cur_trial_dF_roi))
            speed_trace = np.hstack((speed_trace, cur_trial_speed))
    
    lin_fit = siegelslopes(dF_trace, speed_trace)
    # fig = plt.figure(figsize=(2.3,2.5))
    # ax1 = fig.add_subplot(1,1,1)
    # ax1.scatter(speed_trace, dF_trace, cmap='plasma', c=speed_trace-30, norm=plt.Normalize(vmin=-30, vmax=30), s=5, alpha = 1.0, linewidth=0)
    
    # speed_x = np.linspace(-10,60,500)
    # ax1.plot(speed_x, lin_fit[1] + lin_fit[0] * speed_x, 'r-', lw=0.5)
    # ax1.plot(speed_x, np.zeros((speed_x.shape[0],)), 'r--', lw=0.5)
    # ax1.set_xlabel('Running speed (cm/sec)')
    # ax1.set_ylabel('dF/F')
    # ax1.set_xlim([-10,60])
    
    # sns.despine(ax=ax1, right=True, top=True)
    
    # if not os.path.isdir(loc_info['figure_output_path'] + 'roi_speecorr'):
    #         os.mkdir(loc_info['figure_output_path'] + 'roi_speecorr')
    # plt.savefig(loc_info['figure_output_path'] + 'roi_speecorr' + os.sep + 'roi_' + str(fname) + '.svg', format='svg')
    # print("saved " + loc_info['figure_output_path'] + 'roi_speecorr' + os.sep + 'roi_' + str(fname) + '.svg')
    
    return lin_fit
    

def calculate_locbinned_traces(behavior_data, roi_df, shuffle_dF=False):
    ''' 
    calculate activity as a function of space 
    
    returns all trials concatenated for crosscorrellation analysis as matrix of individual trials
    
    '''
    # behavior_data = sio.loadmat(behavior_file)["data"]
    # dF_data = sio.loadmat(dF_file)["data"]
    
    max_trial = np.amax(behavior_data[-1,6]).astype('int')
    
    # latest location for where a trial can start
    start_loc = 100
    final_loc_short = 320
    final_loc_long = 380
    
    nbins_short = 44
    nbins_long = 55
    
    # fig = plt.figure(figsize=(10,5))
    # (ax1,ax2) = fig.subplots(1,2)
    
    # ax1.set_xlim([0,45])
    # ax2.set_xlim([0,56])
    
    locbinned_trace_short = np.zeros((0,1))
    locbinned_trace_long = np.zeros((0,1))
    
    traces_short = np.zeros((0,nbins_short))
    traces_long = np.zeros((0,nbins_long))
    
    filter_data = False
    if filter_data == True:
        order = 6
        fs = int(np.size(behavior_data,0)/behavior_data[-1,0])       # sample rate, Hz
        cutoff = 3 # desired cutoff frequency of the filter, Hz
        roi_df = butter_lowpass_filter(roi_df, cutoff, fs, order)
    # else:
    #     roi_df = dF_data[:,roi]
        
    if shuffle_dF:
        roi_df = np.roll(roi_df, np.random.randint(5000,roi_df.shape[0]))

    for t in range(1,max_trial):
        # pull out current trial and corresponding dF data and bin it
        cur_trial = behavior_data[behavior_data[:,6]==t,:]
        cur_trial_loc = cur_trial[:,1]
        cur_trial_dF_roi = roi_df[behavior_data[:,6]==t]
        
        if cur_trial[0,4] == 3:
            # make vector of bin edges
            bin_edges = np.linspace(start_loc, final_loc_short, nbins_short+1)
            
            # bin by space and concatenate to binned trace
            mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges, (0, 500))[0]
            mean_dF_trial = np.reshape(mean_dF_trial, (len(bin_edges)-1,1))
            # if shuffle_dF:
            #     mean_dF_trial = np.roll(mean_dF_trial, np.random.randint(0,45))
            locbinned_trace_short = np.vstack((locbinned_trace_short, mean_dF_trial))
            traces_short = np.vstack((traces_short,mean_dF_trial.T))
            # ax1.plot(mean_dF_trial)
            
        elif cur_trial[0,4] == 4:
            # make vector of bin edges
            bin_edges = np.linspace(start_loc, final_loc_long, nbins_long+1)
            
            # bin by space and concatenate to binned trace
            mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges, (0, 500))[0]
            mean_dF_trial = np.reshape(mean_dF_trial, (len(bin_edges)-1,1))
            # if shuffle_dF:
            #     mean_dF_trial = np.roll(mean_dF_trial, np.random.randint(0,56))
            locbinned_trace_long = np.vstack((locbinned_trace_long, mean_dF_trial))
            traces_long = np.vstack((traces_long,mean_dF_trial.T))
            # ax2.plot(mean_dF_trial)
            
    # fig = plt.figure(figsize=(10,5))
    # (ax1,ax2) = fig.subplots(2,1)
    # ax1.plot(locbinned_trace_short)
    # ax2.plot(locbinned_trace_long)
    return locbinned_trace_short, locbinned_trace_long, traces_short, traces_long

def calculate_trials_running_speed(behavior_data, speed_col):
    ''' calculate locbinned speed of individual trials '''
    max_trial = np.amax(behavior_data[-1,6]).astype('int')
    
    # latest location for where a trial can start
    start_loc = 100
    final_loc_short = 320
    final_loc_long = 380
    
    nbins_short = 44
    nbins_long = 55
    
    locbinned_trace_short = np.zeros((0,1))
    locbinned_trace_long = np.zeros((0,1))
    
    speed_traces_short = np.zeros((0,nbins_short))
    speed_traces_long = np.zeros((0,nbins_long))
    
    vr_speed_short = np.zeros((0,))
    vr_speed_long = np.zeros((0,))
    
    filter_data = True
    if filter_data == True:
        order = 6
        fs = int(np.size(behavior_data,0)/behavior_data[-1,0])       # sample rate, Hz
        cutoff = 1 # desired cutoff frequency of the filter, Hz
        sess_speed = butter_lowpass_filter(behavior_data[:,speed_col], cutoff, fs, order)
    else:
        sess_speed = behavior_data[:,speed_col]
        
    for t in range(1,max_trial):
        # pull out current trial and corresponding dF data and bin it
        cur_trial = behavior_data[behavior_data[:,6]==t,:]
        cur_trial_loc = cur_trial[:,1]
        cur_trial_speed = sess_speed[behavior_data[:,6]==t]
        cur_trial_vr = behavior_data[behavior_data[:,6]==t,3]
        
        if cur_trial[0,4] == 3:
            # make vector of bin edges
            bin_edges = np.linspace(start_loc, final_loc_short, nbins_short+1)
            
            # bin by space and concatenate to binned trace
            mean_speed_trial = stats.binned_statistic(cur_trial_loc, cur_trial_speed, 'mean', bin_edges, (0, 500))[0]
            mean_speed_trial = np.reshape(mean_speed_trial, (len(bin_edges)-1,1))
            # if shuffle_dF:
            #     mean_speed_trial = np.roll(mean_speed_trial, np.random.randint(0,45))
            # locbinned_trace_short = np.vstack((locbinned_trace_short, mean_speed_trial))
            speed_traces_short = np.vstack((speed_traces_short,mean_speed_trial.T))
            vr_speed_short = np.hstack((vr_speed_short, np.round(np.mean(cur_trial_vr),0)))
            # ax1.plot(mean_speed_trial)
            
        elif cur_trial[0,4] == 4:
            # make vector of bin edges
            bin_edges = np.linspace(start_loc, final_loc_long, nbins_long+1)
            
            # bin by space and concatenate to binned trace
            mean_speed_trial = stats.binned_statistic(cur_trial_loc, cur_trial_speed, 'mean', bin_edges, (0, 500))[0]
            mean_speed_trial = np.reshape(mean_speed_trial, (len(bin_edges)-1,1))
            # if shuffle_dF:
            #     mean_speed_trial = np.roll(mean_speed_trial, np.random.randint(0,56))
            # locbinned_trace_long = np.vstack((locbinned_trace_long, mean_speed_trial))
            speed_traces_long = np.vstack((speed_traces_long,mean_speed_trial.T))
            vr_speed_long = np.hstack((vr_speed_long, np.round(np.mean(cur_trial_vr),0)))
            # ax2.plot(mean_speed_trial)
    return vr_speed_short, vr_speed_long, speed_traces_short, speed_traces_long

def autocorr(x,lags):
    '''numpy.corrcoef, partial'''

    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)

def crosscorr(x1,x2,lags):
    '''numpy.corrcoef, partial'''

    x1[np.isnan(x1)] = 0
    x1 = np.squeeze(x1)
    
    x2[np.isnan(x2)] = 0
    x2 = np.squeeze(x2)

    max_len = np.amin([x1.shape[0], x2.shape[0]])
    x1 = x1[0:max_len-1]
    x2 = x2[0:max_len-1]

    corr=[1. if l==0 else np.corrcoef(x1[l:],x2[:-l])[0][1] for l in lags]
    return np.array(corr)

# def crosscorr2(x1,x2,lags):
#     '''numpy.corrcoef, partial'''

#     x1[np.isnan(x1)] = 0
#     x1 = np.squeeze(x1)
    
#     x2[np.isnan(x2)] = 0
#     x2 = np.squeeze(x2)

#     max_len = np.amin([x1.shape[0], x2.shape[0]])
#     x1 = x1[0:max_len-1]
#     x2 = x2[0:max_len-1]

#     corr2 = signal.correlate(x1, x2)
#     lags2 = signal.correlation_lags(len(x1), len(x2))

#     corr=[1. if l==0 else np.corrcoef(x1[l:],x2[:-l])[0][1] for l in lags]
#     return np.array(corr)


def n_to_m_crosscorr(x1_files, x2_files):
    behavior_data_x1 = sio.loadmat(x1_files[0])["data"]
    dF_data_x1 = sio.loadmat(x1_files[1])["data"]
    behavior_data_x2 = sio.loadmat(x2_files[0])["data"]
    dF_data_x2 = sio.loadmat(x2_files[1])["data"]
    
    num_rois_x1 = dF_data_x1.shape[1]
    num_rois_x2 = dF_data_x2.shape[1]
    
    # dF_data_x1 = sio.loadmat(x1_files[1])["data"]
    # num_rois_x1 = dF_data_x1.shape[1]
    # dF_data_x2 = sio.loadmat(x2_files[1])["data"]
    # num_rois_x2 = dF_data_x2.shape[1]
    
    locbinned_short_x1 = []
    locbinned_long_x1 = []
    locbinned_short_x2 = []
    locbinned_long_x2 = []
    
    for roi in range(num_rois_x1):
        lbs_x1, lbl_x1,_,_ = calculate_locbinned_traces(behavior_data_x1, dF_data_x1[:,roi])    
        locbinned_short_x1.append(lbs_x1)
        
    for roi in range(num_rois_x2):
        lbs_x2, lbl_x2,_,_ = calculate_locbinned_traces(behavior_data_x2, dF_data_x2[:,roi])
        locbinned_short_x2.append(lbs_x2)
       
    # locbinned_trace_short1[np.isnan(locbinned_trace_short1)] = 0
    # locbinned_trace_short1 = np.squeeze(locbinned_trace_short1)    
       
    # locbinned_trace_short2[np.isnan(locbinned_trace_short2)] = 0
    # locbinned_trace_short2 = np.squeeze(locbinned_trace_short2)
    
    cc_matrix = np.zeros((num_rois_x1,num_rois_x2,int(CORR_MAXLAG/4)))
    # print(num_rois_x1)
    for i, lbs_x1 in enumerate(locbinned_short_x1):
        # print(i)
        for j, lbs_x2 in enumerate(locbinned_short_x2):
            cc_matrix[i,j,:] = crosscorr(locbinned_short_x1[i],locbinned_short_x2[j],np.arange(0,CORR_MAXLAG,4))
    
    # fig = plt.figure(figsize=(5,5))
    # ax1 = fig.subplots(1,1)
    # ax1.plot(y)
    return cc_matrix
    
def cc_distribution(cc_res):
    ''' take the cross correlation matrix and find max value for each pair'''

    cc_max = np.zeros(cc_res.shape[0:2])
    
    for i,cc_row in enumerate(cc_max):
        for j,cc_col in enumerate(cc_row):
            if i is not j:
                cc_max[i,j] = np.amax(cc_res[i,j,COMP_OFFSET:])
    
    return cc_max

def shuffled_cc_distribution(shuffled_cc):
    ''' take the shuffled cross correlations and calculate the max for each shuffled instance '''
    cc_max = np.zeros(shuffled_cc.shape[0])
    for i,cc_row in enumerate(cc_max):
        cc_max[i] = np.amax(shuffled_cc[i,:])
    return cc_max

def get_maxcc_roipair(cc_res, roi1, roi2):
    ''' return the maximum cross correlation value of a given pair of neurons '''
    return np.amax(cc_res[roi1,roi2,COMP_OFFSET:])

def cc_dists(cc_res_file, roi_matches_file, match_cols):  
    ''' determine cross correlation distributions of non-matching and matching neurons. '''
    
    # load cross correlation data (n-to-m cc matrix for all neuron pairs)
    cc_res = np.load(cc_res_file)
    roi_match = sio.loadmat(roi_matches_file)["data"]
    roi_match = roi_match.astype('int')
    
    # convert from 1-indexed to 0-indexed
    roi_match = roi_match - 1
    
    # determine the maximum cc value
    maxcc_all = cc_distribution(cc_res)
    
    # determine the max cc value of all matched pairs
    matched_cc = np.zeros((roi_match.shape[0],))
    for i,rm in enumerate(roi_match):
        matched_cc[i] = get_maxcc_roipair(cc_res, rm[match_cols[0]], rm[match_cols[1]])
    
    maxcc_all = maxcc_all.flatten()
    
    # calculate the zscores of all neuron pairs (matched AND non-matched pairs)
    maxcc_with_matches = np.hstack((maxcc_all, matched_cc))
    cc_zscore = stats.zscore(maxcc_with_matches);
    
    # separate out matched and non-matched neuron pairs
    zscore_null = cc_zscore[0:-roi_match.shape[0]]
    zscore_matched = cc_zscore[maxcc_with_matches.shape[0] - roi_match.shape[0]:]
    
    # list of roi pairs corresponding to rows in zscore and matched_cc
    roi_matched_pairs = np.array([roi_match[:,match_cols[0]], roi_match[:,match_cols[1]]]).T
    
    return zscore_null, zscore_matched, maxcc_all, matched_cc, roi_matched_pairs
  

def calc_cc(s1, s2, roi_matches_file, ccmat_filepath, match_cols , force_cc=False):
    
    if os.path.exists(ccmat_filepath) and not force_cc:
        cc_res = np.load(ccmat_filepath)
    else:
        cc_res = n_to_m_crosscorr(s1, s2)
        np.save(ccmat_filepath,cc_res)
    
    return cc_dists(ccmat_filepath, roi_matches_file, match_cols)  

def calc_speed_df_corr(s, s_vr, roimatch_file, roimatch_col, plot_filename=None):
    print(os.path.split(s[0])[-1])
    
    # load roi match data
    roi_match = sio.loadmat(roimatch_file)["data"]
    roi_match = roi_match.astype('int')
    # roi_match = [[1,1]]
    # rois = roi_match[:,roimatch_col]
    
    # convert from 1-indexed to 0-indexed
    roi_match = roi_match - 1

    behavior_data = sio.loadmat(s[0])["data"]
    dF_data = sio.loadmat(s[1])["data"]
    
    # vr_behavior_data = sio.loadmat(s_vr[0])["data"]
    # vr_dF_data = sio.loadmat(s_vr[1])["data"]

    # for rois in roi_match:
    #     roi = rois[roimatch_col]
    #     calculate_speed_df_corr(behavior_data, dF_data[:,roi], mouse_speed_col=8, fname=roi)   
    
    calculate_speed_df_corr(behavior_data, dF_data[:,27], mouse_speed_col=8, fname=28)   

def calc_speed_v_df(s, s_vr, roimatch_file, roimatch_col, plot_filename=None ):
    ''' Calculate running speed vs. df/f response '''
    
    print(os.path.split(s[0])[-1])
    
    # load roi match data
    roi_match = sio.loadmat(roimatch_file)["data"]
    roi_match = roi_match.astype('int')
    # roi_match = [[1,1]]
    # rois = roi_match[:,roimatch_col]
    
    # convert from 1-indexed to 0-indexed
    roi_match = roi_match - 1
    
    
    
    behavior_data = sio.loadmat(s[0])["data"]
    dF_data = sio.loadmat(s[1])["data"]
    
    vr_behavior_data = sio.loadmat(s_vr[0])["data"]
    vr_dF_data = sio.loadmat(s_vr[1])["data"]
    
    # get running speed for each trial
    vr_speed_s, vr_speed_l, speed_traces_short, speed_traces_long = calculate_trials_running_speed(behavior_data, 8)   
    speed_traces_short = speed_traces_short - vr_speed_s.reshape((vr_speed_s.shape[0],1))
    speed_traces_long = speed_traces_long - vr_speed_l.reshape((vr_speed_l.shape[0],1))
    
    max_speed = np.amax([np.nanmax(np.nanmax(speed_traces_short)), np.nanmax(np.nanmax(speed_traces_long))])
    min_speed = np.amin([np.nanmin(np.nanmin(speed_traces_short)), np.nanmin(np.nanmin(speed_traces_long))])
    
    all_AUC_short = np.zeros((0,))
    all_AUC_long = np.zeros((0,))
    
    all_speed_short = np.zeros((0,))
    all_speed_long = np.zeros((0,))
    
    # for rois in roi_match:
    #     roi = rois[roimatch_col]
    
    zz = 1
    for rois in roi_match:
        roi = rois[roimatch_col]
        
        # determine whether or not to include roi by checking autocorrelation during VR session
        lbs_x1, lbl_x1,traces_short_x1,traces_long_x1 = calculate_locbinned_traces(vr_behavior_data, vr_dF_data[:,rois[0]], False)   
        ac_short = crosscorr(lbs_x1,lbs_x1,np.arange(0,CORR_MAXLAG,4))
        ac_long = crosscorr(lbl_x1,lbl_x1,np.arange(0,CORR_MAXLAG,4))
        
        # run with high cc (like 0.5)
        if np.amax(ac_short[20:]) > 0.25 or np.amax(ac_long[20:]) > 0.25:
            print(zz)
            zz = zz + 1
            # Calculate running speed correlation and regress out of roi df, then get df/f for each trial
            roi_df = dF_data[:,roi]
            lin_corr = calculate_speed_df_corr(behavior_data, roi_df, mouse_speed_col=8)
            speed_adusted_roi_df = roi_df - (lin_corr[0] * behavior_data[:,8]) 
            _,_,df_traces_short,df_traces_long = calculate_locbinned_traces(behavior_data, speed_adusted_roi_df, False)   
            
            # calculate AUC for each trial
            traces_short = np.zeros((0,df_traces_short.shape[1]))
            speeds_short = np.zeros((0,df_traces_short.shape[1]))
            trial_AUC_short = np.zeros((0,))
            trial_speed_short = np.zeros((0,))
            for i,dft in enumerate(df_traces_short):
                trial_AUC = metrics.auc(np.arange(0,len(dft)), dft)
                trial_df_max = np.nanmax(dft)
                speed_AUC = metrics.auc(np.arange(0,speed_traces_short.shape[1]), speed_traces_short[i,:])
                speed_mean = np.mean(speed_traces_short[i,:])
                if vr_speed_s[i] > 15 and trial_df_max > 0.2:
                    trial_AUC_short = np.hstack((trial_AUC_short, trial_AUC))
                    # trial_AUC_short = np.hstack((trial_AUC_short, trial_df_max))
                    # trial_speed_short = np.hstack((trial_speed_short, speed_AUC))
                    trial_speed_short = np.hstack((trial_speed_short, speed_mean))
                    traces_short = np.vstack((traces_short, dft.T))
                    speeds_short = np.vstack((speeds_short, speed_traces_short[i,:]))
                
            # calculate AUC for each trial
            traces_long = np.zeros((0,df_traces_long.shape[1]))
            speeds_long = np.zeros((0,df_traces_long.shape[1]))
            trial_AUC_long = np.zeros((0,))
            trial_speed_long = np.zeros((0,))
            for i,dft in enumerate(df_traces_long):
                trial_AUC = metrics.auc(np.arange(0,len(dft)), dft)
                trial_df_max = np.nanmax(dft)
                speed_AUC = metrics.auc(np.arange(0,speed_traces_long.shape[1]), speed_traces_long[i,:])
                speed_mean = np.mean(speed_traces_long[i,:])
                if vr_speed_l[i] > 15 and trial_df_max > 0.2:
                    trial_AUC_long = np.hstack((trial_AUC_long, trial_AUC))
                    # trial_AUC_long = np.hstack((trial_AUC_long, trial_df_max))
                    # trial_speed_long = np.hstack((trial_speed_long, speed_AUC))
                    trial_speed_long = np.hstack((trial_speed_long, speed_mean))
                    traces_long = np.vstack((traces_long, dft.T))
                    speeds_long = np.vstack((speeds_long, speed_traces_long[i,:]))
        
            # collect results of this roi
            all_AUC_short = np.hstack((all_AUC_short, trial_AUC_short))
            all_AUC_long = np.hstack((all_AUC_long, trial_AUC_long))
        
            all_speed_short = np.hstack((all_speed_short, trial_speed_short))
            all_speed_long = np.hstack((all_speed_long, trial_speed_long))
    
            if MAKE_SPEED_TRACE_FIGRE:
                if speeds_short.shape[0] > 0 and speeds_long.shape[0] > 0:
                    plot_traces_speed(speeds_short,traces_short,speeds_long,traces_long, roi, min_speed, max_speed, plot_filename)
    
    return all_speed_short, all_speed_long, all_AUC_short, all_AUC_long
    

def calc_cc_shuffled(s1, s2, roi_matches_file, shufflemat_filepath, pair_col, force_shuffle=False, plot_filename=None):
    ''' 
    Calculate the cross correlation matrix and z-scores by first creating a shuffled
    cc null distribution where we calculate cc values between the activity of the
    neuron in the first session with its own shuffled activity of the same session.
    
    We then calculate the z-score of the cc between session 1 and session 2 relative to 
    the null distribution.
    
    '''
    # cc_res = np.load(cc_res_file)
    print(s1)
    shuffle_dF_data = True
    roi_match = sio.loadmat(roi_matches_file)["data"]
    roi_match = roi_match.astype('int')
    
    # keep track of which trial type produced the higher cc value for each neuron pair
    short_long_cc = np.zeros((roi_match.shape[0]))
    
    # convert from 1-indexed to 0-indexed
    roi_match = roi_match - 1
    
    
    cc_matches = np.zeros((roi_match.shape[0],))
    ac_x1 = np.zeros((roi_match.shape[0],))
    
    # calculate cross correlation between pairs of neurons
    for i,rm in enumerate(roi_match):
        # load data
        behavior_data_x1 = sio.loadmat(s1[0])["data"]
        dF_data_x1 = sio.loadmat(s1[1])["data"]
        behavior_data_x2 = sio.loadmat(s2[0])["data"]
        dF_data_x2 = sio.loadmat(s2[1])["data"]

        # calculate correlation
        lbs_x1, lbl_x1,traces_short_x1,traces_long_x1 = calculate_locbinned_traces(behavior_data_x1, dF_data_x1[:,rm[pair_col[0]]], False)   
        lbs_x2, lbl_x2,traces_short_x2,traces_long_x2 = calculate_locbinned_traces(behavior_data_x2, dF_data_x2[:,rm[pair_col[1]]], False)
        
        # calc cc and evaluate which cross correlation is bigger (short or long trials)
        cc_pair_short = crosscorr(lbs_x1,lbs_x2,np.arange(0,CORR_MAXLAG,4))
        cc_pair_long = crosscorr(lbl_x1,lbl_x2,np.arange(0,CORR_MAXLAG,4))
        ac_short = crosscorr(lbs_x1,lbs_x1,np.arange(0,CORR_MAXLAG,4))
        ac_long = crosscorr(lbl_x1,lbl_x1,np.arange(0,CORR_MAXLAG,4))
        
        max_cc_pair_short = np.amax(cc_pair_short[1:])
        max_cc_pair_long = np.amax(cc_pair_long[1:])
        if max_cc_pair_short > max_cc_pair_long:
            cc_matches[i] = max_cc_pair_short
            ac_x1[i] = np.nanmax(ac_short[20:]) # offset of 20 to avoid initial peak
            short_long_cc[i] = 0
        else:
            cc_matches[i] = max_cc_pair_long
            ac_x1[i] = np.nanmax(ac_long[20:]) # offset of 20 to avoid initial peak
            short_long_cc[i] = 1
            
        # cc_matches[i] = np.amax(max_cc_pair_short, max_cc_pair_long)
    
    # check if shuffled distribution has been calculated before, if not, do it. Otherwise, just load it.
    if os.path.exists(shufflemat_filepath) and not force_shuffle:
        cc_shuffle = np.load(shufflemat_filepath)
    else:
        cc_shuffle = np.zeros((roi_match.shape[0],NUM_SHUFFLE))
        for i,rm in enumerate(roi_match):
            print(i)
            behavior_data_x1 = sio.loadmat(s1[0])["data"]
            dF_data_x1 = sio.loadmat(s1[1])["data"]
            behavior_data_x2 = sio.loadmat(s2[0])["data"]
            dF_data_x2 = sio.loadmat(s2[1])["data"]
            # create distribution of shuffled crosscorrelations of the first session 
            cc_shuffle_roi_short = np.zeros((NUM_SHUFFLE, int(CORR_MAXLAG/4)))
            cc_shuffle_roi_long = np.zeros((NUM_SHUFFLE, int(CORR_MAXLAG/4)))
            for j in range(NUM_SHUFFLE):
                lbs_x1, lbl_x1,_,_ = calculate_locbinned_traces(behavior_data_x1, dF_data_x1[:,rm[pair_col[0]]], False)   
                lbs_x2, lbl_x2,_,_ = calculate_locbinned_traces(behavior_data_x2, dF_data_x2[:,rm[pair_col[1]]], shuffle_dF_data)   
                cc_shuffle_roi_short[j,:] = crosscorr(lbs_x1,lbs_x2,np.arange(0,CORR_MAXLAG,4))
                cc_shuffle_roi_long[j,:] = crosscorr(lbl_x1,lbl_x2,np.arange(0,CORR_MAXLAG,4))
                
            # calculate cc for trial type which produced higher cc value in non-shuffled cc
            if short_long_cc[i] == 0:
                cc_max_short = shuffled_cc_distribution(cc_shuffle_roi_short[:,1:])
                cc_shuffle[i,:] = cc_max_short
            else:  
                cc_max_long = shuffled_cc_distribution(cc_shuffle_roi_long[:,1:])
                cc_shuffle[i,:] = cc_max_long
                
        np.save(shufflemat_filepath, cc_shuffle)
    
    # reject those rois that don't clear a certain min autocorrelation
    maxcc_with_matches = np.hstack((cc_shuffle, cc_matches.reshape(cc_matches.shape[0],1)))   
    maxcc_with_matches_temp = np.zeros((0,maxcc_with_matches.shape[1]))
    roi_match_temp = np.zeros((0,roi_match.shape[1]))
    for i,ac_val in enumerate(ac_x1):
        if ac_val > ACORR_THRESH:
            maxcc_with_matches_temp = np.vstack((maxcc_with_matches_temp, maxcc_with_matches[i,:]))
            roi_match_temp = np.vstack((roi_match_temp,roi_match[i,:]))
    maxcc_with_matches = maxcc_with_matches_temp
    roi_match = roi_match_temp.astype('int')
    
    zscore_matched_all = np.zeros((roi_match.shape[0],))
    zscore_mean_null_all = np.zeros((roi_match.shape[0],))
    zscore_null_dist = np.zeros((roi_match.shape[0],NUM_SHUFFLE))
    allo_frac = np.zeros((roi_match.shape[0],2))
    for i,rm in enumerate(roi_match): # need to fix indexing here after we remove neurons with low ac
        
        # calculate z-score
        cc_zscore = stats.zscore(maxcc_with_matches[i,:]);

        # separate out matched and non-matched neuron pairs
        zscore_null = cc_zscore[0:-1]
        zscore_matched = cc_zscore[-1]
        zscore_matched_all[i] = zscore_matched
        zscore_mean_null_all[i] = np.mean(zscore_null)
        zscore_null_dist[i,:] = zscore_null  
         
        # calculate the fraction of allo coefficients
        glm_coeffs_x1 = sio.loadmat(s1[2])["data"]
        glm_coeffs_x2 = sio.loadmat(s2[2])["data"]
        allo_frac_x1 = allo_fraction(glm_coeffs_x1[:,rm[pair_col[0]]])
        allo_frac_x2 = allo_fraction(glm_coeffs_x2[:,rm[pair_col[1]]])
                
        if ~np.isnan(allo_frac_x1) and ~np.isnan(allo_frac_x2):
            allo_frac[i,0] = allo_frac_x1
            allo_frac[i,1] = allo_frac_x2
        else:
            allo_frac[i,0] = np.nan
            allo_frac[i,1] = np.nan
        
        if MAKE_PAIR_FIGURES:
            lbs_x1, lbl_x1,traces_short_x1,traces_long_x1 = calculate_locbinned_traces(behavior_data_x1, dF_data_x1[:,rm[pair_col[0]]], False)   
            lbs_x2, lbl_x2,traces_short_x2,traces_long_x2 = calculate_locbinned_traces(behavior_data_x2, dF_data_x2[:,rm[pair_col[1]]], False)
            # glm_coeffs_x1 = sio.loadmat(s1[2])["data"]
            # glm_coeffs_x2 = sio.loadmat(s2[2])["data"]
            plot_roi_pair(traces_short_x1, traces_long_x1, traces_short_x2, traces_long_x2, glm_coeffs_x1, glm_coeffs_x2, [rm[pair_col[0]],rm[pair_col[1]]], maxcc_with_matches[i,-1], zscore_matched, ac_x1[i], plot_filename);
            
        if MAKE_SINGLE_TRACE:
            plot_roi = 4
            if rm[0] == plot_roi:
                behavior_data = sio.loadmat(s1[0])["data"]
                fs = int(np.size(behavior_data,0)/behavior_data[-1,0])       # sample rate, Hz
                lbs_x1, lbl_x1,traces_short_x1,traces_long_x1 = calculate_locbinned_traces(behavior_data_x1, dF_data_x1[:,rm[pair_col[0]]], False)   
                plot_single_trace(traces_short_x1, [23,31,39], fs)
        pass
            
    return zscore_null_dist, zscore_matched_all, cc_matches, cc_shuffle, allo_frac

def plot_single_trace(traces, trials, fs):
    single_trace = np.zeros((0,))
    for t in trials:
        single_trace = np.hstack((single_trace, traces[t,:]))
    
    order = 6
    cutoff = 5 # desired cutoff frequency of the filter, Hz
    single_trace = butter_lowpass_filter(single_trace, cutoff, fs, order)
    
    fig = plt.figure(figsize=(5,2))
    ax1 = fig.subplots(1,1)
    ax1.plot(single_trace)
    # ax2.plot(locbinned_trace_long)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Single_trace.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Single_trace.svg")
    

def plot_traces_speed(speeds_short,traces_short,speeds_long,traces_long, roi, min_speed, max_speed, plot_filename):
    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(12, 12)
    ax1 = fig.add_subplot(gs[0:5,0:4])
    ax2 = fig.add_subplot(gs[0:5,5:9])
    # ax3 = fig.add_subplot(gs[0:5,4:5])
    # ax4 = fig.add_subplot(gs[0:5,9:10])
    
    # try:
    max_speed = max_speed - 20
    # max_speed = np.amax([np.nanmax(np.nanmax(speeds_short)), np.nanmax(np.nanmax(speeds_long))])
    # min_speed = np.amin([np.nanmin(np.nanmin(speeds_short)), np.nanmin(np.nanmin(speeds_long))])
    # except:
        # x = 1    
    
    if traces_short.shape[0] > 1 and traces_long.shape[0] > 1:
        colors = plt.cm.plasma(np.linspace(0,1,1000))
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm, ax=ax1)
        plt.colorbar(sm, ax=ax2)
        
        speed_AUC_short = np.zeros((0,))
        speed_mean_short = np.zeros((0,))
        for ss in speeds_short:
            speed_AUC_short = np.hstack((speed_AUC_short,metrics.auc(np.arange(0,ss.shape[0]), ss)))
            speed_mean_short = np.hstack((speed_mean_short, np.mean(ss)))
                      
        speed_zero_short = speed_AUC_short[(np.abs(speed_AUC_short - 0)).argmin()]
        speed_AUC_norm_short = (speed_AUC_short - np.amin(speed_AUC_short)) / (np.amax(speed_AUC_short) - np.amin(speed_AUC_short))
        speed_mean_norm_short2 = (speed_mean_short - (-30)) / 75 #(max_speed - min_speed)
        speed_mean_norm_short2[speed_mean_norm_short2 > 1] = 1
        speed_mean_norm_short2[speed_mean_norm_short2 < 0] = 0
        zero_speed_short = (0 - min_speed) / (max_speed - min_speed)
        speed_zero_norm_short = (speed_zero_short - np.amin(speed_AUC_short)) / (np.amax(speed_AUC_short) - np.amin(speed_AUC_short))
        speed_color_index_short = np.round(speed_mean_norm_short2 * 999, 0)
        
        for i,ts in enumerate(traces_short):
            ax1.plot(ts, color=colors[int(speed_color_index_short[i])])    

        
        speed_AUC_long = np.zeros((0,))
        speed_mean_long = np.zeros((0,))
        for ss in speeds_long:
            speed_AUC_long = np.hstack((speed_AUC_long,metrics.auc(np.arange(0,ss.shape[0]), ss)))
            speed_mean_long = np.hstack((speed_mean_long, np.mean(ss)))
        
        speed_zero_long = speed_AUC_long[(np.abs(speed_AUC_long - 0)).argmin()]
        speed_AUC_norm_long = (speed_AUC_long - np.amin(speed_AUC_long)) / (np.amax(speed_AUC_long) - np.amin(speed_AUC_long))
        # speed_mean_norm_long2 = (speed_mean_long - min_speed) / 60 #(max_speed - min_speed)
        speed_mean_norm_long2 = (speed_mean_long - (-30)) / 75 #(max_speed - min_speed)
        zero_speed_long = (0 - min_speed) / (max_speed - 30)
        speed_mean_norm_long2[speed_mean_norm_long2 > 1] = 1
        speed_mean_norm_long2[speed_mean_norm_long2 < 0] = 0
        speed_zero_norm_long = (speed_zero_long - np.amin(speed_AUC_long)) / (np.amax(speed_AUC_long) - np.amin(speed_AUC_long))
        speed_color_index_long = np.round(speed_mean_norm_long2 * 999, 0)
        
        for i,ts in enumerate(traces_long):
            ax2.plot(ts, color=colors[int(speed_color_index_long[i])])   
        
        # plt.colorbar(sm, ax=ax3)
        
        fig.suptitle('roi: ' + str(roi) + ' zs: ' + str(np.round(zero_speed_short,2)) + ' zl:' + str(np.round(zero_speed_long,2)))
        
        if not os.path.isdir(loc_info['figure_output_path'] + plot_filename):
            os.mkdir(loc_info['figure_output_path'] + plot_filename)
        fname = loc_info['figure_output_path'] + plot_filename + os.sep + plot_filename + '_' + str(roi) + '.' + fformat
    
        fig.savefig(fname, format=fformat, dpi=400)
        print('saved ' + fname)
        plt.close(fig)

def plot_roi_pair(traces_short_x1, traces_long_x1, traces_short_x2, traces_long_x2, glm_coeffs_x1, glm_coeffs_x2, rm, max_cc_pair, zscore_matched, ac, plot_filename):
    fig = plt.figure(figsize=(15,5))
    gs = fig.add_gridspec(11, 23)
    ax1 = fig.add_subplot(gs[0:5,0:5])
    ax2 = fig.add_subplot(gs[0:5,6:11])
    ax3 = fig.add_subplot(gs[6:9,0:5])
    ax4 = fig.add_subplot(gs[9:,0:5])
    ax5 = fig.add_subplot(gs[6:9,6:11])
    ax6 = fig.add_subplot(gs[9:,6:11])
    
    ax7 = fig.add_subplot(gs[0:5,12:17])
    ax8 = fig.add_subplot(gs[0:5,18:23])
    ax9 = fig.add_subplot(gs[6:9,12:17])
    ax10 = fig.add_subplot(gs[9:,12:17])
    ax11 = fig.add_subplot(gs[6:9,18:23])
    ax12 = fig.add_subplot(gs[9:,18:23])
    
    x_bins_short = np.arange(0,traces_short_x1.shape[1])
    x_bins_long = np.arange(0,traces_long_x1.shape[1])
    for i in range(traces_short_x1.shape[0]):
        ax1.plot(x_bins_short, traces_short_x1[i,:], c='0.8')
    for i in range(traces_long_x1.shape[0]):
        ax2.plot(x_bins_long, traces_long_x1[i,:], c='0.8')
    
    ax1.axvspan(20,24,color='#ED7EC6',zorder=0)
    ax2.axvspan(20,24,color='#EC008C',zorder=0)
    
    sem_dF_s = stats.sem(traces_short_x1,0,nan_policy='omit')
    mean_df_short = np.nanmean(traces_short_x1,0)
    ax1.fill_between(np.arange(len(mean_df_short)), mean_df_short - sem_dF_s, mean_df_short + sem_dF_s, color = '0.5', alpha = 0.3, lw=0,zorder=5)
    ax1.plot(x_bins_short, np.nanmean(traces_short_x1,0), c='k', lw=2,zorder=5)
        
    sem_dF_l = stats.sem(traces_long_x1,0,nan_policy='omit')
    mean_df_long = np.nanmean(traces_long_x1,0)
    ax2.fill_between(np.arange(len(mean_df_long)), mean_df_long - sem_dF_l, mean_df_long + sem_dF_l, color = '0.5', alpha = 0.3, lw=0,zorder=5)
    ax2.plot(x_bins_long, np.nanmean(traces_long_x1,0), c='k', lw=2,zorder=5)
    
    # disregard coefficient values < 0
    glm_coeffs_x1[glm_coeffs_x1[:,rm[0]] < 0,rm[0]] = 0
    allo_coeffs_short = glm_coeffs_x1[1:19,rm[0]]
    allo_coeffs_long = glm_coeffs_x1[19:37,rm[0]]
    ego_coeffs_short = glm_coeffs_x1[41:47,rm[0]]
    ego_coeffs_long = glm_coeffs_x1[47:53,rm[0]]
    coeffs_both = glm_coeffs_x1[37:41,rm[0]]
    
    ax3.barh(np.arange(0,allo_coeffs_short.shape[0]), allo_coeffs_short, color='#ED7EC6')
    ax4.barh(np.arange(0,ego_coeffs_short.shape[0]), ego_coeffs_short, color='#008000')
    ax5.barh(np.arange(0,allo_coeffs_long.shape[0]), allo_coeffs_long, color='#EC008C')
    ax6.barh(np.arange(0,ego_coeffs_long.shape[0]), ego_coeffs_long, color='#008000')
    
    ax7.axvspan(20,24,color='#ED7EC6',zorder=0)
    ax8.axvspan(20,24,color='#EC008C',zorder=0)
    
    x_bins_short = np.arange(0,traces_short_x2.shape[1])
    x_bins_long = np.arange(0,traces_long_x2.shape[1])
    for i in range(traces_short_x2.shape[0]):
        ax7.plot(x_bins_short, traces_short_x2[i,:], c='0.8')
    for i in range(traces_long_x2.shape[0]):
        ax8.plot(x_bins_long, traces_long_x2[i,:], c='0.8')
    
    sem_dF_s = stats.sem(traces_short_x2,0,nan_policy='omit')
    mean_df_short = np.nanmean(traces_short_x2,0)
    ax7.fill_between(np.arange(len(mean_df_short)), mean_df_short - sem_dF_s, mean_df_short + sem_dF_s, color = '0.5', alpha = 0.3, lw=0,zorder=5)
    ax7.plot(x_bins_short, np.nanmean(traces_short_x2,0), c='k', lw=2,zorder=5)
    
    sem_dF_l = stats.sem(traces_long_x2,0,nan_policy='omit')
    mean_df_long = np.nanmean(traces_long_x2,0)
    ax8.fill_between(np.arange(len(mean_df_long)), mean_df_long - sem_dF_l, mean_df_long + sem_dF_l, color = '0.5', alpha = 0.3, lw=0,zorder=5)
    ax8.plot(x_bins_long, np.nanmean(traces_long_x2,0), c='k', lw=2,zorder=5)
    
    glm_coeffs_x2[glm_coeffs_x2[:,rm[1]] < 0,rm[1]] = 0
    allo_coeffs_short = glm_coeffs_x2[1:19,rm[1]]
    allo_coeffs_long = glm_coeffs_x2[19:37,rm[1]]
    ego_coeffs_short = glm_coeffs_x2[41:47,rm[1]]
    ego_coeffs_long = glm_coeffs_x2[47:53,rm[1]]
    coeffs_both = glm_coeffs_x2[37:41,rm[1]]
    
    ax9.barh(np.arange(0,allo_coeffs_short.shape[0]), allo_coeffs_short, color='#ED7EC6')
    ax10.barh(np.arange(0,ego_coeffs_short.shape[0]), ego_coeffs_short, color='#008000')
    ax11.barh(np.arange(0,allo_coeffs_long.shape[0]), allo_coeffs_long, color='#EC008C')
    ax12.barh(np.arange(0,ego_coeffs_long.shape[0]), ego_coeffs_long, color='#008000')
    
    max_y = np.amax((ax1.get_ylim(), ax2.get_ylim(), ax7.get_ylim(), ax8.get_ylim()))
    # min_y = np.amin((ax1.get_ylim(), ax2.get_ylim(), ax7.get_ylim(), ax8.get_ylim()))
    ax1.set_ylim([-0.2,max_y])
    ax2.set_ylim([-0.2,max_y])
    ax7.set_ylim([-0.2,max_y])
    ax8.set_ylim([-0.2,max_y])
    
    max_x_allo = np.amax((ax3.get_xlim(),ax5.get_xlim(),ax9.get_xlim(),ax11.get_xlim()))
    max_x_ego = np.amax((ax4.get_xlim(),ax6.get_xlim(),ax10.get_xlim(),ax12.get_xlim()))

    
    ax3.set_xlim([0,max_x_allo])
    ax4.set_xlim([0,max_x_ego])
    ax5.set_xlim([0,max_x_allo])
    ax6.set_xlim([0,max_x_ego])
    ax9.set_xlim([0,max_x_allo])
    ax10.set_xlim([0,max_x_ego])
    ax11.set_xlim([0,max_x_allo])
    ax12.set_xlim([0,max_x_ego])
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(True)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(True)
    ax4.spines['bottom'].set_visible(True)
    
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(True)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_visible(True)
    ax6.spines['bottom'].set_visible(True)

    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.spines['bottom'].set_visible(False)
    ax9.spines['left'].set_visible(True)
    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.spines['left'].set_visible(True)
    ax10.spines['bottom'].set_visible(True)
    
    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['bottom'].set_visible(False)
    ax11.spines['left'].set_visible(True)
    ax12.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['left'].set_visible(True)
    ax12.spines['bottom'].set_visible(True)
    
    coeff_ratio_x1 = ego_allo_ratio(glm_coeffs_x1[:,rm[0]])
    coeff_ratio_x2 = ego_allo_ratio(glm_coeffs_x2[:,rm[1]])
    
    allo_frac_x1 = allo_fraction(glm_coeffs_x1[:,rm[0]])
    allo_frac_x2 = allo_fraction(glm_coeffs_x2[:,rm[1]])

    avg_allo_coeff_x1 = avg_allo_coeff(glm_coeffs_x1[:,rm[0]])
    avg_allo_coeff_x2 = avg_allo_coeff(glm_coeffs_x2[:,rm[1]])
    avg_coeff = (avg_allo_coeff_x1 + avg_allo_coeff_x2)/2
    
    fig.suptitle('roi pair: ' + str(rm) + ' ac: ' + str(np.round(ac,2)) + ' cc: ' +  str(np.round(max_cc_pair,3)) + 
                 ' zscore: ' + str(np.round(zscore_matched,2))  + ' avg. allo coeff: ' + str(np.round(avg_coeff,2)) + 
                 ' ear x1: ' + str(np.round(coeff_ratio_x1,2)) + ' ear x2: ' +  str(np.round(coeff_ratio_x2,2)) + 
                 ' af x1: ' + str(np.round(allo_frac_x1,2)) + ' af x2: ' + str(np.round(allo_frac_x2,2)))
    
    if not os.path.isdir(loc_info['figure_output_path'] + plot_filename):
        os.mkdir(loc_info['figure_output_path'] + plot_filename)
    fname = loc_info['figure_output_path'] + plot_filename + os.sep + 'roipair_' + str(rm[0]) + '_' + str(rm[1]) + '.' + fformat

    fig.savefig(fname, format=fformat, dpi=400)
    print('saved ' + fname)
    plt.close(fig)

def avg_allo_coeff(coeff_vec):
    allo_coeffs = coeff_vec[1:37]
    allo_coeffs = allo_coeffs[np.nonzero(allo_coeffs)]
    return np.nanmean(allo_coeffs)

def allo_fraction(coeff_vec):
    # coeff_vec[coeff_vec < 0] = 0
    coeff_vec = np.abs(coeff_vec)
    coeff_vec[16:19] = np.nan
    allo_coeffs = np.nansum(coeff_vec[1:37])
    ego_coeffs = np.nansum(coeff_vec[37:])
    
    if np.nansum(coeff_vec) > 0.0:
        allo_frac = allo_coeffs / (allo_coeffs + ego_coeffs)
    else:
        allo_frac = np.nan

    return allo_frac

def coeff_sparseness(coeff_vec):
    ''' Calculate the sparseness-coefficient of allocentric coefficients(max allo coeff value / number of nonzero coeffs) '''
    allo_coeffs = coeff_vec[1:37]
    max_allo = np.amax(allo_coeffs)
    return max_allo / len(np.nonzero(allo_coeffs))

def ego_allo_ratio(coeff_vec):
    allo_coeffs = np.sum(coeff_vec[1:37])
    ego_coeffs = np.sum(coeff_vec[37:])
    return (allo_coeffs - ego_coeffs) / (allo_coeffs + ego_coeffs)

def df_AUC_vr_ol(sess_vr, sess_ol, roi_matches_file, plot_fname):
    ''' calculate activity in VR vs. OL condition '''
    
    roi_match = sio.loadmat(roi_matches_file)["data"]
    roi_match = roi_match.astype('int')
    
    roi_match = roi_match - 1
    
    behavior_data_vr = sio.loadmat(sess_vr[0])["data"]
    dF_data_vr = sio.loadmat(sess_vr[1])["data"]
    
    behavior_data_ol = sio.loadmat(sess_ol[0])["data"]
    dF_data_ol = sio.loadmat(sess_ol[1])["data"]
    
    # calculate AUC and peak df/f per spatial bin of all neurons in VR and OL session
    lbs_auc_vr_all = []
    lbl_auc_vr_all = []
    lbs_peakdf_vr_all = []
    lbl_peakdf_vr_all = []
    
    # for roi in roi_match[:,0]:
    for roi in np.arange(0,dF_data_vr.shape[1]):
        lbs_vr, lbl_vr,traces_short_vr,traces_long_vr = calculate_locbinned_traces(behavior_data_vr, dF_data_vr[:,roi], False)
        speed_vr_short, speed_vr_long, speed_traces_vr_short, speed_traces_vr_long = calculate_trials_running_speed(behavior_data_vr, 3)   
        lbs_vr = lbs_vr[~np.isnan(lbs_vr)]
        lbl_vr = lbl_vr[~np.isnan(lbl_vr)]
        lbs_auc_vr_all.append(metrics.auc(np.arange(0,len(lbs_vr)),lbs_vr) / len(lbs_vr))
        lbl_auc_vr_all.append(metrics.auc(np.arange(0,len(lbl_vr)),lbl_vr) / len(lbl_vr))
        lbs_peakdf_vr_all.append(np.nanmax(np.nanmean(traces_short_vr,0)))
        lbl_peakdf_vr_all.append(np.nanmax(np.nanmean(traces_long_vr,0)))
        
    lbs_auc_ol_all = []
    lbl_auc_ol_all = []
    lbs_peakdf_ol_all = []
    lbl_peakdf_ol_all = []
    # for roi in roi_match[:,1]:
    for roi in np.arange(0,dF_data_ol.shape[1]):
        lbs_ol, lbl_ol,traces_short_ol,traces_long_ol = calculate_locbinned_traces(behavior_data_ol, dF_data_ol[:,roi], False)
        speed_ol_short, speed_ol_long, speed_traces_ol_short, speed_traces_ol_long = calculate_trials_running_speed(behavior_data_ol, 3)   
        lbs_vr = lbs_vr[~np.isnan(lbs_vr)]
        lbl_vr = lbl_vr[~np.isnan(lbl_vr)]
        lbs_auc_ol_all.append(metrics.auc(np.arange(0,len(lbs_ol)),lbs_ol) / len(lbs_ol))
        lbl_auc_ol_all.append(metrics.auc(np.arange(0,len(lbl_ol)),lbl_ol) / len(lbl_ol))
        lbs_peakdf_ol_all.append(np.nanmax(np.nanmean(traces_short_ol,0)))
        lbl_peakdf_ol_all.append(np.nanmax(np.nanmean(traces_long_ol,0)))
        
    # calculate mean AUC for VR and OL sessions. Add auc of short and long trial before dividing by number of rois
    total_auc = 0
    for l in lbl_auc_vr_all:
        total_auc = total_auc + np.sum(l)
    
    for l in lbs_auc_vr_all:
        total_auc = total_auc + np.sum(l)
    mean_vr_AUC = total_auc / dF_data_vr.shape[1]
    
    total_peakdf = 0
    for l in lbl_peakdf_vr_all:
        total_peakdf = total_peakdf + np.sum(l)
    
    for l in lbs_peakdf_vr_all:
        total_peakdf = total_peakdf + np.sum(l)
    mean_vr_peakdf = total_peakdf / dF_data_vr.shape[1]
    

    total_auc = 0
    for l in lbl_auc_ol_all:
        total_auc = total_auc + np.sum(l)
    
    for l in lbs_auc_ol_all:
        total_auc = total_auc + np.sum(l)
    mean_ol_AUC = total_auc / dF_data_ol.shape[1]
    
    total_peakdf = 0
    for l in lbl_peakdf_ol_all:
        total_peakdf = total_peakdf + np.sum(l)
    
    for l in lbs_peakdf_ol_all:
        total_peakdf = total_peakdf + np.sum(l)
    mean_ol_peakdf = total_peakdf / dF_data_ol.shape[1]
    
    return mean_vr_AUC, mean_ol_AUC, mean_vr_peakdf, mean_ol_peakdf

def df_AUC(sess_vr, roi_matches_file, roi_matches_columns):
    ''' calculate activity in a given session '''
    
    roi_match = sio.loadmat(roi_matches_file)["data"]
    roi_match = roi_match.astype('int')
    roi_match = roi_match - 1
    
    AUC_all = np.zeros((len(sess_vr),))
    maxdf_all = np.zeros((len(sess_vr),))
    # run through every session
    for i,s in enumerate(sess_vr):
        behavior_data_vr = sio.loadmat(s[0])["data"]
        dF_data_vr = sio.loadmat(s[1])["data"]
      
        # calculate AUC per spatial bin of all neurons in VR and OL session
        lbs_auc_vr_all = []
        lbl_auc_vr_all = []
        lbs_maxdf_vr_all = []
        lbl_maxdf_vr_all = []
        # for roi in roi_match[:,roi_matches_columns[i]]:
        for roi in np.arange(0,dF_data_vr.shape[1]):
            lbs_vr, lbl_vr,traces_short_vr,traces_long_vr = calculate_locbinned_traces(behavior_data_vr, dF_data_vr[:,roi], False)
            speed_vr_short, speed_vr_long, speed_traces_vr_short, speed_traces_vr_long = calculate_trials_running_speed(behavior_data_vr, 3)   
            lbs_vr = lbs_vr[~np.isnan(lbs_vr)]
            lbl_vr = lbl_vr[~np.isnan(lbl_vr)]
            lbs_auc_vr_all.append(metrics.auc(np.arange(0,len(lbs_vr)),lbs_vr) / len(lbs_vr))
            lbl_auc_vr_all.append(metrics.auc(np.arange(0,len(lbl_vr)),lbl_vr) / len(lbl_vr))
            lbs_maxdf_vr_all.append(np.nanmax(np.nanmean(traces_short_vr,0)))
            lbl_maxdf_vr_all.append(np.nanmax(np.nanmean(traces_long_vr,0)))
                    
        # calculate mean AUC for VR and OL sessions. Add auc of short and long trial before dividing by number of rois
        total_auc = 0
        for l in lbl_auc_vr_all:
            total_auc = total_auc + np.sum(l)
        
        for l in lbs_auc_vr_all:
            total_auc = total_auc + np.sum(l)
        mean_vr_AUC = total_auc / dF_data_vr.shape[1]   
        
        total_maxdf = 0
        for l in lbl_maxdf_vr_all:
            total_maxdf = total_maxdf + np.sum(l)
        
        for l in lbs_maxdf_vr_all:
            total_maxdf = total_maxdf + np.sum(l)
        mean_vr_maxdf = total_maxdf / dF_data_vr.shape[1]   
        
        AUC_all[i] = mean_vr_AUC
        maxdf_all[i] = mean_vr_maxdf

    return AUC_all, maxdf_all


def run_cc_analysis(sess_x1, sess_x2, roimatch, roimatch_columns, cc_file=None, cc_shuffle=None, plot_filename=None):
    # get cc for matched pairs
    zscore_null, zscore_matched, cc_null, cc_matched, roi_matched_pairs = calc_cc(sess_x1, sess_x2, roimatch, cc_file, roimatch_columns, FORCE_CC)
    zscore_null, zscore_matched, cc_matched, cc_shuffled, allo_frac = calc_cc_shuffled(sess_x1, sess_x2, roimatch, cc_shuffle, roimatch_columns, FORCE_SHUFFLE_CC, plot_filename)
    
    
    # load glm coefficients
    # glm_coeffs_x1 = sio.loadmat(sess_x1[2])["data"]
    # glm_coeffs_x2 = sio.loadmat(sess_x2[2])["data"]
    
    # make matrix, one column containing cc values of matched pairs, the other column containing glm coefficient ratio
    pair_ccglm = np.zeros((zscore_matched.shape[0],2))
    # pair_ccglm[:,0] = zscore_matched
    
    # calculate average ego-allo ratio of the neuron in both sessions
    # for i,rmp in enumerate(roi_matched_pairs):
    #     ea_ratio_x1 = ego_allo_ratio(glm_coeffs_x1[:,rmp[0]])
    #     ea_ratio_x2 = ego_allo_ratio(glm_coeffs_x2[:,rmp[1]])
    #     avg_ea_ratio = (ea_ratio_x1 + ea_ratio_x2)/2
    #     pair_ccglm[i,1] = avg_ea_ratio
    
    return zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac

# def cc_22_1_l23_expert(zscore_null_all, zscore_matched_all):
#     ''' evaluate cross-correlation between neuron pairs for respective session '''
#     return run_cc_analysis(filedict["LF191022_1"]["1204"], filedict["LF191022_1"]["1207"], filedict["LF191022_1"]["roimatch"],  [0,1], filedict["LF191022_1"]["cc_12041207"], filedict["LF191022_1"]["cshuffle_12041207"], 'LF191022_1_rm_12041207')


def l23_naive_expert():
    # load session data
    fname = "total_analysis"

    TRIAL_THRESHOLD = 0
    
    file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + fname + ".mat"
    data = sio.loadmat(file_path)
    
    naive = [('LF191022_1','20191115'),('LF191022_2','20191116'),('LF191022_3','20191113'),('LF191023_blank','20191114'),('LF191023_blue','20191119'),('LF191024_1','20191115')]
    expert = [('LF191022_1','20191209'),('LF191022_2','20191210'),('LF191022_3','20191207'),('LF191023_blank','20191210'),('LF191023_blue','20191208'),('LF191024_1','20191210')]
    # expert = [('LF191022_1','20191204'),('LF191022_2','20191210'),('LF191022_3','20191207'),('LF191023_blank','20191206'),('LF191023_blue','20191204'),('LF191024_1','20191204')]
    
    tscore_naive = []
    egoallo_naive = []
    ntrials_naive = []
    for animal,session in naive:
        # print(animal,session,data[animal + '_' + session])
        if data[animal + '_' + session][0][1] > TRIAL_THRESHOLD:
            tscore_naive.append(data[animal + '_' + session][0][0])
            egoallo_naive.append(data[animal + '_' + session][0][2])
            ntrials_naive.append(data[animal + '_' + session][0][1])
    
    tscore_expert = []
    egoallo_expert = []
    ntrials_expert = []
    for animal,session in expert:
        # print(animal,session, data[animal + '_' + session])
        if data[animal + '_' + session][0][1] > TRIAL_THRESHOLD:
            tscore_expert.append(data[animal + '_' + session][0][0])
            egoallo_expert.append(data[animal + '_' + session][0][2])
            ntrials_expert.append(data[animal + '_' + session][0][1])
            
    
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    allo_frac_all = np.zeros((2,))
    allo_frac_all_ind = np.zeros((2,))
    tscore_all_ind = np.zeros((2,))
    
    # zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_1"]["1114"], 
    #                                                           filedict["LF191022_1"]["1209"], 
    #                                                           filedict["LF191022_1"]["roimatch naive"],  
    #                                                           [0,1], 
    #                                                           filedict["LF191022_1"]["cc_11141209"], 
    #                                                           filedict["LF191022_1"]["cshuffle_11141209"], 
    #                                                           'LF191022_1_rm_11141209')
    # zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    # zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    # cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    # cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_1"]["1115"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch naive 2"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_11151209"], 
                                                              filedict["LF191022_1"]["cshuffle_11151209"], 
                                                              'LF191022_1_rm_11151209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_all = np.vstack((allo_frac_all, np.nanmean(allo_frac,axis=0)))
    allo_frac_all_ind = np.vstack((allo_frac_all_ind, allo_frac))
    tscore_all_ind = np.vstack((tscore_all_ind, np.array([np.ones(allo_frac.shape[0])*tscore_naive[0], np.ones(allo_frac.shape[0])*tscore_expert[0]]).T))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_2"]["1116"], 
                                                              filedict["LF191022_2"]["1210"], 
                                                              filedict["LF191022_2"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191022_2"]["cc_11161210"], 
                                                              filedict["LF191022_2"]["cshuffle_11161210"], 
                                                              'LF191022_2_rm_111561210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_all = np.vstack((allo_frac_all, np.nanmean(allo_frac,axis=0)))
    allo_frac_all_ind = np.vstack((allo_frac_all_ind, allo_frac))
    tscore_all_ind = np.vstack((tscore_all_ind, np.array([np.ones(allo_frac.shape[0])*tscore_naive[1], np.ones(allo_frac.shape[0])*tscore_expert[1]]).T))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_3"]["1113"], 
                                                              filedict["LF191022_3"]["1207"], 
                                                              filedict["LF191022_3"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191022_3"]["cc_11131207"], 
                                                              filedict["LF191022_3"]["cshuffle_11131207"], 
                                                              'LF191022_3_rm_11131207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_all = np.vstack((allo_frac_all, np.nanmean(allo_frac,axis=0)))
    allo_frac_all_ind = np.vstack((allo_frac_all_ind, allo_frac))
    tscore_all_ind = np.vstack((tscore_all_ind, np.array([np.ones(allo_frac.shape[0])*tscore_naive[2], np.ones(allo_frac.shape[0])*tscore_expert[2]]).T))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blank"]["1114"], 
                                                              filedict["LF191023_blank"]["1210"], 
                                                              filedict["LF191023_blank"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191023_blank"]["cc_11141210"], 
                                                              filedict["LF191023_blank"]["cshuffle_11141210"], 
                                                              'LF191023_blank_rm_11141210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_all = np.vstack((allo_frac_all, np.nanmean(allo_frac,axis=0)))
    allo_frac_all_ind = np.vstack((allo_frac_all_ind, allo_frac))
    tscore_all_ind = np.vstack((tscore_all_ind, np.array([np.ones(allo_frac.shape[0])*tscore_naive[3], np.ones(allo_frac.shape[0])*tscore_expert[3]]).T))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blue"]["1119"], 
                                                              filedict["LF191023_blue"]["1208"], 
                                                              filedict["LF191023_blue"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191023_blue"]["cc_11191208"], 
                                                              filedict["LF191023_blue"]["cshuffle_11191208"], 
                                                              'LF191023_blue_rm_11191208')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_all = np.vstack((allo_frac_all, np.nanmean(allo_frac,axis=0)))
    allo_frac_all_ind = np.vstack((allo_frac_all_ind, allo_frac))
    tscore_all_ind = np.vstack((tscore_all_ind, np.array([np.ones(allo_frac.shape[0])*tscore_naive[4], np.ones(allo_frac.shape[0])*tscore_expert[4]]).T))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191024_1"]["1115"], 
                                                              filedict["LF191024_1"]["1210"], 
                                                              filedict["LF191024_1"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191024_1"]["cc_11151210"], 
                                                              filedict["LF191024_1"]["cshuffle_11151210"], 
                                                              'LF191024_1_rm_11151210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_all = np.vstack((allo_frac_all, np.nanmean(allo_frac,axis=0)))
    allo_frac_all_ind = np.vstack((allo_frac_all_ind, allo_frac))
    tscore_all_ind = np.vstack((tscore_all_ind, np.array([np.ones(allo_frac.shape[0])*tscore_naive[5], np.ones(allo_frac.shape[0])*tscore_expert[5]]).T))
    
    allo_frac_all = allo_frac_all[1:,:]
    allo_frac_all_ind = allo_frac_all_ind[1:,:]
    tscore_all_ind = tscore_all_ind[1:,:]
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots(3,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,10,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,10,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,10,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,10,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,10,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,10,0.5), color='r', ax=ax_n2)
    ax[2,0].scatter(np.zeros(allo_frac_all.shape[0]), allo_frac_all[:,0])
    ax[2,0].scatter(np.ones(allo_frac_all.shape[0]), allo_frac_all[:,1])
    ax[2,0].set_ylim([-0.1,1.1])
    
    ax[2,1].scatter(tscore_naive, allo_frac_all[:,0])
    ax[2,1].scatter(tscore_expert, allo_frac_all[:,1])
    
    ax[2,2].scatter(tscore_all_ind[:,0], allo_frac_all_ind[:,0])
    ax[2,2].scatter(tscore_all_ind[:,1], allo_frac_all_ind[:,1])
    
    
    
    print('===== L2/3 - NAIVE EXPERT =====')
    print(stats.ttest_ind(zscore_null_all.flatten(),zscore_matched_all))
    print(stats.mannwhitneyu(zscore_null_all.flatten(),zscore_matched_all))
    print('--- ALL FRAC MEAN ---')
    print(stats.ttest_ind(allo_frac_all[:,0],allo_frac_all[:,1], nan_policy='omit'))
    print(stats.spearmanr(np.hstack((tscore_naive,tscore_expert)), np.hstack((allo_frac_all[:,0],allo_frac_all[:,1])), nan_policy='omit'))
    print(np.nanmean(allo_frac_all,axis=0))
    print('--- ALL FRAC INDIVIDUAL ---')
    print(stats.ttest_ind(allo_frac_all_ind[:,0],allo_frac_all_ind[:,1], nan_policy='omit'))
    print(stats.spearmanr(np.hstack((tscore_all_ind[:,0],tscore_all_ind[:,1])), np.hstack((allo_frac_all_ind[:,0],allo_frac_all_ind[:,1])), nan_policy='omit'))
    print(np.nanmean(allo_frac_all_ind,axis=0))
    
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccne_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccne.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccne.svg")

def l23_tscore_corr():
    ''' 
    Calculate the fraction of allocentric coefficients for every sesseion where we have matched ROIs 
    
    Because of the nature of how we matched ROIs, we get 'naive' and 'expert' coefficients for each pair of sessions.
    However, we store coefficients with their respective task score so that the naive and expert classification doesn't really matter
    
    '''
    
     # load session data
    fname = "total_analysis"

    TRIAL_THRESHOLD = 0
    
    file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + fname + ".mat"
    data = sio.loadmat(file_path)
    
    # naive = [('LF191022_1','20191114'),('LF191022_1','20191115'),('LF191022_1','20191121'),('LF191022_1','20191204'),('LF191022_1','20191207'),('LF191022_2','20191116'),('LF191022_3','20191113'),('LF191023_blank','20191114'),('LF191023_blue','20191119'),('LF191024_1','20191115')]
    expert = [('LF191022_1','20191209'),
              ('LF191022_2','20191210'),
              ('LF191022_3','20191207'),
              ('LF191023_blank','20191210'),
              ('LF191023_blue','20191208'),
              ('LF191024_1','20191210')]
    # expert = [('LF191022_1','20191204'),('LF191022_2','20191210'),('LF191022_3','20191207'),('LF191023_blank','20191206'),('LF191023_blue','20191204'),('LF191024_1','20191204')]
    
    naive=[('LF191022_1','20191114'),
           ('LF191022_1','20191115'),
           ('LF191022_1','20191121'),
           ('LF191022_1','20191204'),
           ('LF191022_1','20191207'),
           
           ('LF191022_2','20191116'),
           ('LF191022_2','20191206'),
           ('LF191022_2','20191208'),
           
           ('LF191022_3','20191113'),
           ('LF191022_3','20191204'),
           ('LF191022_3','20191210'),
           
           ('LF191023_blank','20191114'),
           ('LF191023_blank','20191206'),
           ('LF191023_blank','20191208'),        
           
           ('LF191023_blue','20191119'),
           ('LF191023_blue','20191204'),
           ('LF191023_blue','20191210'),
           
           ('LF191024_1','20191115'),
           ('LF191024_1','20191204'),
           ('LF191024_1','20191207'),
          ]
    
    tscore_naive = []
    egoallo_naive = []
    ntrials_naive = []
    for animal,session in naive:
        # print(animal,session,data[animal + '_' + session])
        if data[animal + '_' + session][0][1] > TRIAL_THRESHOLD:
            tscore_naive.append(data[animal + '_' + session][0][0])
            egoallo_naive.append(data[animal + '_' + session][0][2])
            ntrials_naive.append(data[animal + '_' + session][0][1])
    
    tscore_expert = []
    egoallo_expert = []
    ntrials_expert = []
    for animal,session in expert:
        # print(animal,session, data[animal + '_' + session])
        if data[animal + '_' + session][0][1] > TRIAL_THRESHOLD:
            tscore_expert.append(data[animal + '_' + session][0][0])
            egoallo_expert.append(data[animal + '_' + session][0][2])
            ntrials_expert.append(data[animal + '_' + session][0][1])
            
    
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    allo_frac_all = np.zeros((2,))
    allo_frac_naive_ind = np.zeros((1,))
    allo_frac_expert_ind = np.zeros((1,))
    tscore_naive_ind = np.zeros((1,))
    tscore_expert_ind = np.zeros((1,))
    
    all_frac_all_mean = np.zeros((1,))
    tscore_all = np.zeros((1,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_1"]["1114"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_11141209"], 
                                                              filedict["LF191022_1"]["cshuffle_11141209"], 
                                                              'LF191022_1_rm_11141209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[0]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[0]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,1])))
    tscore_all = np.hstack((tscore_all, tscore_naive[0]))
    tscore_all = np.hstack((tscore_all, tscore_expert[0]))
    
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_1"]["1115"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch naive 2"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_11151209"], 
                                                              filedict["LF191022_1"]["cshuffle_11151209"], 
                                                              'LF191022_1_rm_11151209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[1]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[1]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_1"]["1121"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch naive 3"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_11211209"], 
                                                              filedict["LF191022_1"]["cshuffle_11211209"], 
                                                              'LF191022_1_rm_11211209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[2]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[2]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_1"]["1204"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191022_1"]["cc_12041209"], 
                                                              filedict["LF191022_1"]["cshuffle_12041209"], 
                                                              'LF191022_1_rm_12041209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[3]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[3]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_1"]["1207"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191022_1"]["cc_12071209"], 
                                                              filedict["LF191022_1"]["cshuffle_12071209"], 
                                                              'LF191022_1_rm_12071209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[4]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[4]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_2"]["1116"], 
                                                              filedict["LF191022_2"]["1210"], 
                                                              filedict["LF191022_2"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191022_2"]["cc_11161210"], 
                                                              filedict["LF191022_2"]["cshuffle_11161210"], 
                                                              'LF191022_2_rm_11161210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[5]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[1]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,1])))
    tscore_all = np.hstack((tscore_all, tscore_naive[5]))
    tscore_all = np.hstack((tscore_all, tscore_expert[1]))
    
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_2"]["1206"], 
                                                              filedict["LF191022_2"]["1210"], 
                                                              filedict["LF191022_2"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191022_2"]["cc_12061210"], 
                                                              filedict["LF191022_2"]["cshuffle_12061210"], 
                                                              'LF191022_2_rm_12061210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[6]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[6]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_2"]["1208"], 
                                                              filedict["LF191022_2"]["1210"], 
                                                              filedict["LF191022_2"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191022_2"]["cc_12081210"], 
                                                              filedict["LF191022_2"]["cshuffle_12081210"], 
                                                              'LF191022_2_rm_12081210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[7]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[7]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_3"]["1113"], 
                                                              filedict["LF191022_3"]["1207"], 
                                                              filedict["LF191022_3"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191022_3"]["cc_11131207"], 
                                                              filedict["LF191022_3"]["cshuffle_11131207"], 
                                                              'LF191022_3_rm_11131207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[8]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[2]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,1])))
    tscore_all = np.hstack((tscore_all, tscore_naive[8]))
    tscore_all = np.hstack((tscore_all, tscore_expert[2]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_3"]["1204"], 
                                                              filedict["LF191022_3"]["1207"], 
                                                              filedict["LF191022_3"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191022_3"]["cc_12041207"], 
                                                              filedict["LF191022_3"]["cshuffle_12041207"], 
                                                              'LF191022_3_rm_12041207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[9]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[2]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[9]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191022_3"]["1210"], 
                                                              filedict["LF191022_3"]["1207"], 
                                                              filedict["LF191022_3"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191022_3"]["cc_12071210"], 
                                                              filedict["LF191022_3"]["cshuffle_12071210"], 
                                                              'LF191022_3_rm_12071210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[10]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[2]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[10]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blank"]["1114"], 
                                                              filedict["LF191023_blank"]["1210"], 
                                                              filedict["LF191023_blank"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191023_blank"]["cc_11141210"], 
                                                              filedict["LF191023_blank"]["cshuffle_11141210"], 
                                                              'LF191023_blank_rm_11141210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[11]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[3]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,1])))
    tscore_all = np.hstack((tscore_all, tscore_naive[11]))
    tscore_all = np.hstack((tscore_all, tscore_expert[3]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blank"]["1206"], 
                                                              filedict["LF191023_blank"]["1210"], 
                                                              filedict["LF191023_blank"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191023_blank"]["cc_12061210"], 
                                                              filedict["LF191023_blank"]["cshuffle_12061210"], 
                                                              'LF191023_blank_rm_12061210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[12]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[12]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blank"]["1208"], 
                                                              filedict["LF191023_blank"]["1210"], 
                                                              filedict["LF191023_blank"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191023_blank"]["cc_12081210"], 
                                                              filedict["LF191023_blank"]["cshuffle_12081210"], 
                                                              'LF191023_blank_rm_12081210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[13]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[13]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blue"]["1119"], 
                                                              filedict["LF191023_blue"]["1208"], 
                                                              filedict["LF191023_blue"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191023_blue"]["cc_11191208"], 
                                                              filedict["LF191023_blue"]["cshuffle_11191208"], 
                                                              'LF191023_blue_rm_11191208')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[14]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[4]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,1])))
    tscore_all = np.hstack((tscore_all, tscore_naive[14]))
    tscore_all = np.hstack((tscore_all, tscore_expert[4]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blue"]["1204"], 
                                                              filedict["LF191023_blue"]["1208"], 
                                                              filedict["LF191023_blue"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191023_blue"]["cc_12041208"], 
                                                              filedict["LF191023_blue"]["cshuffle_12041208"], 
                                                              'LF191023_blue_rm_12041208')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[15]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[15]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191023_blue"]["1210"], 
                                                              filedict["LF191023_blue"]["1208"], 
                                                              filedict["LF191023_blue"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191023_blue"]["cc_12081210"], 
                                                              filedict["LF191023_blue"]["cshuffle_12081210"], 
                                                              'LF191023_blue_rm_12081210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[16]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[16]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191024_1"]["1115"], 
                                                              filedict["LF191024_1"]["1210"], 
                                                              filedict["LF191024_1"]["roimatch naive"],  
                                                              [0,1], 
                                                              filedict["LF191024_1"]["cc_11151210"], 
                                                              filedict["LF191024_1"]["cshuffle_11151210"], 
                                                              'LF191024_1_rm_11151210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    allo_frac_expert_ind = np.hstack((allo_frac_expert_ind, allo_frac[:,1]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[17]]))))
    tscore_expert_ind = np.hstack((tscore_expert_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_expert[5]]))))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,1])))
    tscore_all = np.hstack((tscore_all, tscore_naive[17]))
    tscore_all = np.hstack((tscore_all, tscore_expert[5]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191024_1"]["1204"], 
                                                              filedict["LF191024_1"]["1210"], 
                                                              filedict["LF191024_1"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191024_1"]["cc_12041210"], 
                                                              filedict["LF191024_1"]["cshuffle_12041210"], 
                                                              'LF191024_1_rm_12041210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[18]]))))  
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[18]))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled, allo_frac = run_cc_analysis(filedict["LF191024_1"]["1207"], 
                                                              filedict["LF191024_1"]["1210"], 
                                                              filedict["LF191024_1"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191024_1"]["cc_12071210"], 
                                                              filedict["LF191024_1"]["cshuffle_12071210"], 
                                                              'LF191024_1_rm_12071210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    # stack allo fraction and tscore values
    allo_frac_naive_ind = np.hstack((allo_frac_naive_ind, allo_frac[:,0]))
    tscore_naive_ind = np.hstack((tscore_naive_ind, np.squeeze(np.array([np.ones(allo_frac.shape[0])*tscore_naive[19]]))))    
    all_frac_all_mean = np.hstack((all_frac_all_mean, np.nanmean(allo_frac[:,0])))
    tscore_all = np.hstack((tscore_all, tscore_naive[19]))
    
    allo_frac_naive_ind = allo_frac_naive_ind[1:]
    allo_frac_expert_ind = allo_frac_expert_ind[1:]
    tscore_naive_ind = tscore_naive_ind[1:]
    tscore_expert_ind = tscore_expert_ind[1:]
    all_frac_all_mean = all_frac_all_mean[1:]
    tscore_all = tscore_all[1:]
    
    allo_frac_all = np.hstack((allo_frac_naive_ind,allo_frac_expert_ind))
    tscores = np.hstack((tscore_naive_ind,tscore_expert_ind))
    
    fig = plt.figure(figsize=(4,3))
    ax = fig.subplots(1,2)
    ax[0].scatter(tscores, allo_frac_all, color='0.7')
    ax[1].scatter(tscore_all, all_frac_all_mean, color='0.7')
    
    ax[0].set_ylim([-0.05,1.05])
    ax[1].set_ylim([0,1])
    
    ax[0].set_xlim([-30,92])
    ax[0].set_xticks([-30,0,30,60,90])
    ax[0].set_xticklabels([-30,0,30,60,90])
    
    ax[1].set_xlim([-30,92])
    ax[1].set_xticks([-30,0,30,60,90])
    ax[1].set_xticklabels([-30,0,30,60,90])
    
    # fit a linear regressions
    res = stats.linregress(x=tscores, y=allo_frac_all)
    ax[0].plot(tscores, res.intercept + res.slope*tscores, 'r', label='fitted line', lw=2)
    
    res = stats.linregress(x=tscore_all, y=all_frac_all_mean)
    ax[1].plot(tscore_all, res.intercept + res.slope*tscore_all, 'r', label='fitted line', lw=2)
    
    # res = stats.linregress(x=tscores, y=allo_frac_all)
    # print(res)
    # ax[0].plot(tscores, res.intercept + res.slope*allo_frac_all, 'r', label='fitted line', lw=2)
    # corr_speed_short, peak_intercept, lo_slope, up_slope = stats.theilslopes(all_frac_all_mean, tscore_all)
    # ax[1].plot(tscore_all, peak_intercept+corr_speed_short * tscore_all, 'r', label='fitted line', lw=2)
    print('--- ALL FRAC INDIVIDUAL ---')
    print(stats.spearmanr(tscores, allo_frac_all, nan_policy='omit'))
    print(stats.spearmanr(tscore_all, all_frac_all_mean, nan_policy='omit'))
    print('---------------------------')
    
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    
    plt.tight_layout()

    fig.savefig("C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\manuscript\\Figure 1\\allofrac_tracked.svg", format='svg')
    print("saved" + "C:\\Users\\lfisc\\Work\\Projects\\Lntmodel\\manuscript\\Figure 1\\allofrac_tracked.svg")
    

def l23_sess_1_2():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1204"], 
                                                              filedict["LF191022_1"]["1207"], 
                                                              filedict["LF191022_1"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_12041207"], 
                                                              filedict["LF191022_1"]["cshuffle_12041207"], 
                                                              'LF191022_1_rm_12041207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_2"]["1206"], 
                                                              filedict["LF191022_2"]["1208"], 
                                                              filedict["LF191022_2"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191022_2"]["cc_12061208"], 
                                                              filedict["LF191022_2"]["cshuffle_12061208"], 
                                                              'LF191022_2_rm_12061208')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_3"]["1204"], 
                                                              filedict["LF191022_3"]["1207"], 
                                                              filedict["LF191022_3"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191022_3"]["cc_12041207"], 
                                                              filedict["LF191022_3"]["cshuffle_12041207"], 
                                                              'LF191022_3_rm_12041207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blank"]["1206"], 
                                                              filedict["LF191023_blank"]["1208"], 
                                                              filedict["LF191023_blank"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191023_blank"]["cc_12061208"], 
                                                              filedict["LF191023_blank"]["cshuffle_12061208"], 
                                                              'LF191023_blank_rm_12061208')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blue"]["1204"], 
                                                              filedict["LF191023_blue"]["1208"], 
                                                              filedict["LF191023_blue"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191023_blue"]["cc_12041208"], 
                                                              filedict["LF191023_blue"]["cshuffle_12041208"], 
                                                              'LF191023_blue_rm_12041208')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191024_1"]["1204"], 
                                                              filedict["LF191024_1"]["1207"], 
                                                              filedict["LF191024_1"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191024_1"]["cc_12041207"], 
                                                              filedict["LF191024_1"]["cshuffle_12041207"], 
                                                              'LF191024_1_rm_12041207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,10,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,10,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,10,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,10,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,10,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,10,0.5), color='r', ax=ax_n2)
    
    ax_n2.spines['top'].set_visible(False)
    ax_n2.spines['right'].set_visible(False)
    ax_n2.spines['left'].set_visible(True)
    ax_n2.spines['bottom'].set_visible(True)

    ax[1,3].spines['top'].set_visible(False)
    ax[1,3].spines['right'].set_visible(False)
    ax[1,3].spines['left'].set_visible(True)
    ax[1,3].spines['bottom'].set_visible(True)
    
    print('===== L2/3 - S1 S2 =====')
    print(stats.ttest_ind(zscore_null_all.flatten(),zscore_matched_all))
    print(stats.mannwhitneyu(zscore_null_all.flatten(),zscore_matched_all))
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc12_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc12.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc12.svg")

def l23_sess_2_3 ():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1207"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191022_1"]["cc_12071209"], 
                                                              filedict["LF191022_1"]["cshuffle_12071209"], 
                                                              'LF191022_1_rm_12071209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_2"]["1208"], 
                                                              filedict["LF191022_2"]["1210"], 
                                                              filedict["LF191022_2"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191022_2"]["cc_12081210"], 
                                                              filedict["LF191022_2"]["cshuffle_12081210"], 
                                                              'LF191022_2_rm_12081210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_3"]["1207"], 
                                                              filedict["LF191022_3"]["1210"], 
                                                              filedict["LF191022_3"]["roimatch"],  
                                                              [2,1], 
                                                              filedict["LF191022_3"]["cc_12071210"], 
                                                              filedict["LF191022_3"]["cshuffle_12071210"], 
                                                              'LF191022_3_rm_12071210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blank"]["1208"], 
                                                              filedict["LF191023_blank"]["1210"], 
                                                              filedict["LF191023_blank"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191023_blank"]["cc_12081210"], 
                                                              filedict["LF191023_blank"]["cshuffle_12081210"], 
                                                              'LF191023_blank_rm_12081210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blue"]["1208"], 
                                                              filedict["LF191023_blue"]["1210"], 
                                                              filedict["LF191023_blue"]["roimatch"],  
                                                              [2,1], 
                                                              filedict["LF191023_blue"]["cc_12081210"], 
                                                              filedict["LF191023_blue"]["cshuffle_12081210"], 
                                                              'LF191023_blue_rm_12081210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191024_1"]["1207"], 
                                                              filedict["LF191024_1"]["1210"], 
                                                              filedict["LF191024_1"]["roimatch"],  
                                                              [1,2], 
                                                              filedict["LF191024_1"]["cc_12071210"], 
                                                              filedict["LF191024_1"]["cshuffle_12071210"], 
                                                              'LF191024_1_rm_12071210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,5,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,5,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,5,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,5,0.5), color='r', ax=ax_n2)
    
    print('===== L2/3 - S2 S3 =====')
    print(stats.ttest_ind(zscore_null_all.flatten(),zscore_matched_all))
    print(stats.mannwhitneyu(zscore_null_all.flatten(),zscore_matched_all))
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc23_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc23.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc23.svg")
    
def l23_sess_1_3 ():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1204"], 
                                                              filedict["LF191022_1"]["1209"], 
                                                              filedict["LF191022_1"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191022_1"]["cc_12041209"], 
                                                              filedict["LF191022_1"]["cshuffle_12041209"], 
                                                              'LF191022_1_rm_12041209')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_2"]["1206"], 
                                                              filedict["LF191022_2"]["1210"], 
                                                              filedict["LF191022_2"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191022_2"]["cc_12061210"], 
                                                              filedict["LF191022_2"]["cshuffle_12061210"], 
                                                              'LF191022_2_rm_12061210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_3"]["1204"], 
                                                              filedict["LF191022_3"]["1210"], 
                                                              filedict["LF191022_3"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191022_3"]["cc_12041210"], 
                                                              filedict["LF191022_3"]["cshuffle_12041210"], 
                                                              'LF191022_3_rm_12041210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blank"]["1206"], 
                                                              filedict["LF191023_blank"]["1210"], 
                                                              filedict["LF191023_blank"]["roimatch"],  
                                                              [0,2], 
                                                              filedict["LF191023_blank"]["cc_12061210"], 
                                                              filedict["LF191023_blank"]["cshuffle_12061210"], 
                                                              'LF191023_blank_rm_12061210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blue"]["1204"], 
                                                              filedict["LF191023_blue"]["1210"], 
                                                              filedict["LF191023_blue"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191023_blue"]["cc_12041210"], 
                                                              filedict["LF191023_blue"]["cshuffle_12041210"], 
                                                              'LF191023_blue_rm_12041210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191024_1"]["1204"], 
                                                              filedict["LF191024_1"]["1210"], 
                                                              filedict["LF191024_1"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191024_1"]["cc_12041210"], 
                                                              filedict["LF191024_1"]["cshuffle_12041210"], 
                                                              'LF191024_1_rm_12041210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,10,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,10,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,10,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,10,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,10,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,10,0.5), color='r', ax=ax_n2)
    
    print('===== L2/3 - S1 S3 =====')
    print(stats.ttest_ind(zscore_null_all.flatten(),zscore_matched_all))
    print(stats.mannwhitneyu(zscore_null_all.flatten(),zscore_matched_all))
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13.svg")
    
def l5_sess_1_2 ():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1213"], 
                                                              filedict["LF191022_1"]["1215"], 
                                                              filedict["LF191022_1"]["roimatch l5"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_12131215"], 
                                                              filedict["LF191022_1"]["cshuffle_12131215"], 
                                                              'LF191022_1_rm_12131215')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_2"]["1212"], 
                                                              filedict["LF191022_2"]["1216"], 
                                                              filedict["LF191022_2"]["roimatch l5"],  
                                                              [0,1], 
                                                              filedict["LF191022_2"]["cc_12121216"], 
                                                              filedict["LF191022_2"]["cshuffle_12121216"], 
                                                              'LF191022_2_rm_12121216')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_3"]["1211"], 
                                                              filedict["LF191022_3"]["1215"], 
                                                              filedict["LF191022_3"]["roimatch l5"],  
                                                              [0,1], 
                                                              filedict["LF191022_3"]["cc_12111215"], 
                                                              filedict["LF191022_3"]["cshuffle_12111215"], 
                                                              'LF191022_3_rm_12111215')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blank"]["1213"], 
                                                              filedict["LF191023_blank"]["1216"], 
                                                              filedict["LF191023_blank"]["roimatch l5"],  
                                                              [0,1],
                                                              filedict["LF191023_blank"]["cc_12131216"], 
                                                              filedict["LF191023_blank"]["cshuffle_12131216"], 
                                                              'LF191023_blank_rm_12131216')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blue"]["1212"], 
                                                              filedict["LF191023_blue"]["1215"], 
                                                              filedict["LF191023_blue"]["roimatch l5"],  
                                                              [0,1], 
                                                              filedict["LF191023_blue"]["cc_12121215"], 
                                                              filedict["LF191023_blue"]["cshuffle_12121215"], 
                                                              'LF191023_blue_rm_12121215')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,5,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,5,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,5,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,5,0.5), color='r', ax=ax_n2)
    
    print('===== L5 - S1 S2 =====')
    print(stats.ttest_ind(zscore_null_all.flatten(),zscore_matched_all))
    print(stats.mannwhitneyu(zscore_null_all.flatten(),zscore_matched_all))
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13.svg")
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl512_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl512.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl512.svg")

def l5_sess_2_3 ():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1215"], 
                                                              filedict["LF191022_1"]["1217"], 
                                                              filedict["LF191022_1"]["roimatch l5"],  
                                                              [1,2], 
                                                              filedict["LF191022_1"]["cc_12151217"], 
                                                              filedict["LF191022_1"]["cshuffle_12151217"], 
                                                              'LF191022_1_rm_12151217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
      
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_3"]["1212"], 
                                                              filedict["LF191022_3"]["1217"], 
                                                              filedict["LF191022_3"]["roimatch l5"],  
                                                              [1,2], 
                                                              filedict["LF191022_3"]["cc_12121217"], 
                                                              filedict["LF191022_3"]["cshuffle_12121217"], 
                                                              'LF191022_3_rm_12121217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blank"]["1216"], 
                                                              filedict["LF191023_blank"]["1217"], 
                                                              filedict["LF191023_blank"]["roimatch l5"],  
                                                              [1,2],
                                                              filedict["LF191023_blank"]["cc_12161217"], 
                                                              filedict["LF191023_blank"]["cshuffle_12161217"], 
                                                              'LF191023_blank_rm_12161217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blue"]["1215"], 
                                                              filedict["LF191023_blue"]["1217"], 
                                                              filedict["LF191023_blue"]["roimatch l5"],  
                                                              [1,2], 
                                                              filedict["LF191023_blue"]["cc_12151217"], 
                                                              filedict["LF191023_blue"]["cshuffle_12151217"], 
                                                              'LF191023_blue_rm_12151217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,5,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,5,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,5,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,5,0.5), color='r', ax=ax_n2)
    
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl513_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl513.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl513.svg")
    
def l5_sess_1_3 ():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1213"], 
                                                              filedict["LF191022_1"]["1217"], 
                                                              filedict["LF191022_1"]["roimatch l5"],  
                                                              [0,2], 
                                                              filedict["LF191022_1"]["cc_12131217"], 
                                                              filedict["LF191022_1"]["cshuffle_12131217"], 
                                                              'LF191022_1_rm_12131217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
      
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_3"]["1211"], 
                                                              filedict["LF191022_3"]["1217"], 
                                                              filedict["LF191022_3"]["roimatch l5"],  
                                                              [0,2], 
                                                              filedict["LF191022_3"]["cc_12111217"], 
                                                              filedict["LF191022_3"]["cshuffle_12111217"], 
                                                              'LF191022_3_rm_12111217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blank"]["1213"], 
                                                              filedict["LF191023_blank"]["1217"], 
                                                              filedict["LF191023_blank"]["roimatch l5"],  
                                                              [0,2],
                                                              filedict["LF191023_blank"]["cc_12131217"], 
                                                              filedict["LF191023_blank"]["cshuffle_12131217"], 
                                                              'LF191023_blank_rm_12131217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blue"]["1212"], 
                                                              filedict["LF191023_blue"]["1217"], 
                                                              filedict["LF191023_blue"]["roimatch l5"],  
                                                              [0,2], 
                                                              filedict["LF191023_blue"]["cc_12121217"], 
                                                              filedict["LF191023_blue"]["cshuffle_12121217"], 
                                                              'LF191023_blue_rm_12121217')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,5,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,5,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,5,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,5,0.5), color='r', ax=ax_n2)
    
    print('===== L5 - S1 S3 =====')
    print(stats.ttest_ind(zscore_null_all.flatten(),zscore_matched_all))
    print(stats.mannwhitneyu(zscore_null_all.flatten(),zscore_matched_all))
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13.svg")
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl513_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl513.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccl513.svg")

def ol_l23_sess_1():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1204"], 
                                                              filedict["LF191022_1"]["1204 ol"], 
                                                              filedict["LF191022_1"]["roimatch ol 1204"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_ol_1204"], 
                                                              filedict["LF191022_1"]["cshuffle_ol_1204"], 
                                                              'LF191022_1_rm_ol_1204')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_2"]["1210"], 
                                                              filedict["LF191022_2"]["1210 ol"], 
                                                              filedict["LF191022_2"]["roimatch ol 1210"],  
                                                              [0,1], 
                                                              filedict["LF191022_2"]["cc_ol_1210"], 
                                                              filedict["LF191022_2"]["cshuffle_ol_1210"], 
                                                              'LF191022_2_rm_ol_1210')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_3"]["1207"], 
                                                              filedict["LF191022_3"]["1207 ol"], 
                                                              filedict["LF191022_3"]["roimatch ol 1207"],  
                                                              [0,1], 
                                                              filedict["LF191022_3"]["cc_ol_1207"], 
                                                              filedict["LF191022_3"]["cshuffle_ol_1207"], 
                                                              'LF191022_3_rm_ol_1207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blank"]["1206"], 
                                                              filedict["LF191023_blank"]["1206 ol"], 
                                                              filedict["LF191023_blank"]["roimatch ol 1206"],  
                                                              [0,1], 
                                                              filedict["LF191023_blank"]["cc_ol_1206"], 
                                                              filedict["LF191023_blank"]["cshuffle_ol_1206"], 
                                                              'LF191023_blank_rm_ol_1206')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191023_blue"]["1204"], 
                                                              filedict["LF191023_blue"]["1204 ol"], 
                                                              filedict["LF191023_blue"]["roimatch ol 1204"],  
                                                              [0,1], 
                                                              filedict["LF191023_blue"]["cc_ol_1204"], 
                                                              filedict["LF191023_blue"]["cshuffle_ol_1204"], 
                                                              'LF191023_blue_rm_ol_1204')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191024_1"]["1204"], 
                                                              filedict["LF191024_1"]["1204 ol"], 
                                                              filedict["LF191024_1"]["roimatch ol 1204"],  
                                                              [0,1], 
                                                              filedict["LF191024_1"]["cc_ol_1204"], 
                                                              filedict["LF191024_1"]["cshuffle_ol_1204"], 
                                                              'LF191024_1_rm_ol_1204')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,5,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,5,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,5,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,5,0.5), color='r', ax=ax_n2)
    
    np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccol1_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccol1.svg", format='svg')
    print("saved " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_ccol1.svg")

def ol_speedcorr():
    print('=== calculating speed vs. df/f correlation ===')
    
    speed_short_group = np.zeros((0,))
    speed_long_group = np.zeros((0,))
    AUC_short_group = np.zeros((0,))
    AUC_long_group = np.zeros((0,))
    
    calc_speed_df_corr(filedict["LF191022_1"]["1204 ol"], filedict["LF191022_1"]["1204"], filedict["LF191022_1"]["roimatch ol 1204"], 1, 'LF191022_1_1204_ol_speedtrace')
    # if NAIVE_OL:
    #     calc_speed_df_corr(filedict["LF191022_1"]["1115 ol"], filedict["LF191022_1"]["1115"], filedict["LF191022_1"]["roimatch ol 1115"], 1, 'LF191022_1_1115_ol_speedtrace')
        # speed_short, speed_long, AUC_short, AUC_long = calc_speed_df_corr(filedict["LF191022_1"]["1115 ol"], filedict["LF191022_1"]["1115"], filedict["LF191022_1"]["roimatch ol 1115"], 1, 'LF191022_1_1115_ol_speedtrace')
        # speed_short_group = np.hstack((speed_short_group, speed_short))
        # speed_long_group = np.hstack((speed_long_group, speed_long))
        # AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        # AUC_long_group = np.hstack((AUC_long_group, AUC_long))

def ol_speeddiff_response():
    
    print('=== calculating speed vs. df/f ===')
    
    speed_short_group = np.zeros((0,))
    speed_long_group = np.zeros((0,))
    AUC_short_group = np.zeros((0,))
    AUC_long_group = np.zeros((0,))
    
    if NAIVE_OL:
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_1"]["1115 ol"], filedict["LF191022_1"]["1115"], filedict["LF191022_1"]["roimatch ol 1115"], 1, 'LF191022_1_1115_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_2"]["1116 ol"], filedict["LF191022_2"]["1116"], filedict["LF191022_2"]["roimatch ol 1116"], 1, 'LF191022_2_1116_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_3"]["1114 ol"], filedict["LF191022_3"]["1114"], filedict["LF191022_3"]["roimatch ol 1114"], 1, 'LF191022_3_1114_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191023_blank"]["1114 ol"], filedict["LF191023_blank"]["1114"], filedict["LF191023_blank"]["roimatch ol 1114"], 1, 'LF191023_blank_1114_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191023_blue"]["1119 ol"], filedict["LF191023_blue"]["1119"], filedict["LF191023_blue"]["roimatch ol 1119"], 1, 'LF191023_blue_1119_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191024_1"]["1115 ol"], filedict["LF191024_1"]["1115"], filedict["LF191024_1"]["roimatch ol 1115"], 1, 'LF191024_1_1115_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        range_min = -30
        range_max = 15
        fit_plot_range = 10
    
    if EXPERT_S1_OL:
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_1"]["1204 ol"], filedict["LF191022_1"]["1204"], filedict["LF191022_1"]["roimatch ol 1204"], 1, 'LF191022_1_1204_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))        
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_2"]["1210 ol"], filedict["LF191022_2"]["1210"], filedict["LF191022_2"]["roimatch ol 1210"], 1, 'LF191022_2_1210_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_3"]["1207 ol"], filedict["LF191022_3"]["1207"], filedict["LF191022_3"]["roimatch ol 1207"], 1, 'LF191022_3_1207_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
            
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191023_blank"]["1206 ol"], filedict["LF191023_blank"]["1206"], filedict["LF191023_blank"]["roimatch ol 1206"], 1, 'LF191023_blank_1206_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191023_blue"]["1204 ol"], filedict["LF191023_blue"]["1204"], filedict["LF191023_blue"]["roimatch ol 1204"], 1, 'LF191023_blue_1204_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191024_1"]["1204 ol"], filedict["LF191024_1"]["1204"], filedict["LF191024_1"]["roimatch ol 1204"], 1, 'LF191024_1_1204_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        range_min = -30
        range_max = 50
        fit_plot_range = 50
    
    if EXPERT_S2_OL:
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_1"]["1209 ol"], filedict["LF191022_1"]["1209"], filedict["LF191022_1"]["roimatch ol 1209"], 1, 'LF191022_1_1209_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_2"]["1210 ol"], filedict["LF191022_2"]["1210"], filedict["LF191022_2"]["roimatch ol 1210"], 1, 'LF191022_2_1210_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191022_3"]["1207 ol"], filedict["LF191022_3"]["1207"], filedict["LF191022_3"]["roimatch ol 1207"], 1, 'LF191022_3_1207_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191023_blank"]["1210 ol"], filedict["LF191023_blank"]["1210"], filedict["LF191023_blank"]["roimatch ol 1210"], 1, 'LF191023_blank_1210_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191023_blue"]["1210 ol"], filedict["LF191023_blue"]["1210"], filedict["LF191023_blue"]["roimatch ol 1210"], 1, 'LF191023_blue_1210_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        speed_short, speed_long, AUC_short, AUC_long = calc_speed_v_df(filedict["LF191024_1"]["1210 ol"], filedict["LF191024_1"]["1210"], filedict["LF191024_1"]["roimatch ol 1210"], 1, 'LF191024_1_1210_ol_speedtrace')
        speed_short_group = np.hstack((speed_short_group, speed_short))
        speed_long_group = np.hstack((speed_long_group, speed_long))
        AUC_short_group = np.hstack((AUC_short_group, AUC_short))
        AUC_long_group = np.hstack((AUC_long_group, AUC_long))
        
        range_min = -30
        range_max = 50
        fit_plot_range = 50
       
   
    # split data into running speed smaller than and larger than VR speed
    short_slow = speed_short_group[speed_short_group <= 0]
    short_fast = speed_short_group[speed_short_group > 0]
    short_AUC_slow = AUC_short_group[speed_short_group <= 0]
    short_AUC_fast = AUC_short_group[speed_short_group > 0]
    
    long_slow = speed_long_group[speed_long_group <= 0]
    long_fast = speed_long_group[speed_long_group > 0]
    long_AUC_slow = AUC_long_group[speed_long_group <= 0]
    long_AUC_fast = AUC_long_group[speed_long_group > 0]
    
    short_slow_fit = np.polyfit(short_slow, short_AUC_slow, 1)
    short_fast_fit = np.polyfit(short_fast, short_AUC_fast, 1)
    p_slow_short = np.poly1d(short_slow_fit)
    p_fast_short = np.poly1d(short_fast_fit)
    
    long_slow_fit = np.polyfit(long_slow, long_AUC_slow, 1)
    long_fast_fit = np.polyfit(long_fast, long_AUC_fast, 1)
    p_slow_long = np.poly1d(long_slow_fit)
    p_fast_long = np.poly1d(long_fast_fit)
    
    
    # fit 2nd order polynomial to data
    short_fit = np.polyfit(speed_short_group, AUC_short_group, 2)
    long_fit = np.polyfit(speed_long_group, AUC_long_group, 2)
    p_short = np.poly1d(short_fit)
    p_long = np.poly1d(long_fit)
    
    all_fit = np.polyfit(np.hstack((speed_short_group,speed_long_group)), np.hstack((AUC_short_group,AUC_long_group)),2)
    p_all = np.poly1d(all_fit)
    
    fig = plt.figure(figsize=(5,10))
    ax1 = fig.subplots(1,1)
    # ax1.scatter(speed_short_group, AUC_short_group, color='0.7', s=100, alpha = 0.2, linewidth=0)
    color_min = np.amin([np.nanmin(speed_short_group),np.nanmin(speed_long_group)])
    color_max = np.amax([np.nanmax(speed_short_group),np.nanmax(speed_long_group)])
    ax1.scatter(speed_short_group, AUC_short_group, c=speed_short_group, cmap='plasma', norm=plt.Normalize(vmin=color_min, vmax=color_max), s=100, alpha = 0.5, linewidth=0)
    ax1.scatter(speed_long_group, AUC_long_group, c=speed_long_group, cmap='plasma', norm=plt.Normalize(vmin=color_min, vmax=color_max), s=100, alpha = 0.5, linewidth=0)

    # range_min = -1500
    # range_max = 3000
    
    # fit_xvals = np.linspace(-30, 60, 1000)
    fit_xvals = np.linspace(range_min, range_max, 1000)
    plot_xvals = np.linspace(range_min, fit_plot_range, 1000)
    
    # ax1.plot(fit_xvals, p_short(plot_xvals), c='0.5', ls='--')
    # ax1.plot(fit_xvals, p_long(plot_xvals), c='k', ls='--')
    ax1.plot(plot_xvals, p_all(plot_xvals), c='r', ls='-', lw=4, zorder=5)
    ax1.plot(plot_xvals, p_all(plot_xvals), c='w', ls='-', lw=8, zorder=4)
    # ax1.plot(fit_xvals, p_all(plot_xvals), c='k', ls='--')
    
    # ax1.axvline(fit_xvals[np.argmax(p_short(fit_xvals))], color='0.5', ls='--')
    # ax1.axvline(fit_xvals[np.argmax(p_long(fit_xvals))], color='k', ls='--')
    ax1.axvline(plot_xvals[np.argmax(p_all(plot_xvals))], color='r', ls='--', lw=4, zorder=5)
    # ax1.axvline(plot_xvals[np.argmax(p_all(plot_xvals))], color='w', ls='--', lw=4, zorder=4)
    
    
    # ax1.plot(fit_xvals, p_slow_short(fit_xvals), c='k')
    # ax1.plot(fit_xvals, p_fast_short(fit_xvals), c='k')
    
    # ax1.plot(fit_xvals, p_slow_long(fit_xvals), c='0.5')
    # ax1.plot(fit_xvals, p_fast_long(fit_xvals), c='0.5')
    
    bin_edges = np.linspace(range_min, range_max, 5)
    avg_bin_short, bins_short, binnumber_short = stats.binned_statistic(speed_short_group, AUC_short_group, 'mean', bin_edges, (range_min, range_max))
    avg_bin_long, bins_long, binnumber_long = stats.binned_statistic(speed_long_group, AUC_long_group, 'mean', bin_edges, (range_min, range_max))
    
    # fig = plt.figure(figsize=(5,5))
    # ax1 = fig.subplots(1,1)
    
    bp_data = []
    bp_sem = np.zeros((0,))
    bp_mean = np.zeros((0,))
    for i,b in enumerate(bins_short):
        bp_data.append(np.hstack((AUC_short_group[binnumber_short==i], AUC_long_group[binnumber_long==i])))
        bp_sem = np.append(bp_sem, stats.sem(bp_data[-1], nan_policy='omit'))
        bp_mean = np.append(bp_mean, np.nanmean(bp_data[-1]))
    
    bp_data_short = []
    bp_sem_short = np.zeros((0,))
    bp_mean_short = np.zeros((0,))
    for i,b in enumerate(bins_short):
        bp_data_short.append(AUC_short_group[binnumber_short==i])
        bp_sem_short = np.append(bp_sem_short, stats.sem(bp_data_short[-1], nan_policy='omit'))
        bp_mean_short = np.append(bp_mean_short, np.nanmean(bp_data_short[-1]))
        
    bp_data_long = []
    bp_sem_long = np.zeros((0,))
    bp_mean_long = np.zeros((0,))
    for i,b in enumerate(bins_long):
        bp_data_long.append(AUC_long_group[binnumber_long==i])
        bp_sem_long = np.append(bp_sem_long, stats.sem(bp_data_long[-1], nan_policy='omit'))
        bp_mean_long = np.append(bp_mean_long, np.nanmean(bp_data_long[-1]))
       
    ax1.plot(np.linspace(range_min, range_max, len(bp_mean)), bp_mean, c='k', lw=4, zorder=5)        
    ax1.plot(np.linspace(range_min, range_max, len(bp_mean)), bp_mean, c='w', lw=8, zorder=4)
    # ax1.fill_between(np.linspace(range_min, range_max, len(bp_mean)), bp_mean - bp_sem, bp_mean + bp_sem, color = 'b', alpha = 0.3, lw=0,zorder=5)   
    
    # ax1.plot(np.linspace(range_min, range_max, len(bp_mean_short)), bp_mean_short, c='b')    
    # ax1.plot(np.linspace(range_min, range_max, len(bp_mean_long)), bp_mean_long, c='r')    
    # ax1.fill_between(np.linspace(range_min, range_max, len(bp_mean_short)), bp_mean_short - bp_sem_short, bp_mean_short + bp_sem_short, color = 'b', alpha = 0.3, lw=0,zorder=5)
    # ax1.fill_between(np.linspace(range_min, range_max, len(bp_mean_long)), bp_mean_long - bp_sem_long, bp_mean_long + bp_sem_long, color = 'r', alpha = 0.3, lw=0,zorder=5)
    
    # bp = ax1.boxplot(bp_data,
    #         vert=True, patch_artist=True, bootstrap=1000, showcaps=False, positions=[-30,-20,-10,0,10,20,30,40,50,60], widths=6, showfliers=False,
    #         whiskerprops=dict(linestyle='-', color='black', linewidth=2, solid_capstyle='butt'),
    #         medianprops=dict(color='black', linewidth=2, solid_capstyle='butt'))
    
    # ax2 = ax1.twiny()
    # ax1.plot(fit_xvals, p_short(fit_xvals), c='0.5', ls='--')
    # ax1.plot(fit_xvals, p_long(fit_xvals), c='k', ls='--')
    
    # ax1.axvline(fit_xvals[np.argmax(p_short(fit_xvals))], color='0.5', ls='--')
    # ax1.axvline(fit_xvals[np.argmax(p_long(fit_xvals))], color='k', ls='--')
    
    # print(stats.pearsonr(speed_short_group, AUC_short_group))
    # print(stats.pearsonr(speed_long_group, AUC_long_group))
    
    # ax1.set_ylim([0,130])
    ax1.set_yscale('log')
    
    if EXPERT_S1_OL:
        ax1.set_ylim([5,180])
        ax1.set_xticks([-30,-15,0,15,30,45])
        fname = 'OL_S1_speed_scatter'
    elif NAIVE_OL:
        ax1.set_ylim([2,120])
        ax1.set_xticks([-30,-15,0,15,30])
        ax1.set_xlim([-32,30])
        fname = 'OL_Naive_speed_scatter'
    elif EXPERT_S2_OL:
        ax1.set_ylim([5,180])
        ax1.set_xticks([-30,-15,0,15,30,45])
        fname = 'OL_S2_speed_scatter'
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_linewidth(3)
    ax1.spines['bottom'].set_linewidth(3)
    ax1.tick_params( \
        direction='out', \
        length=3, \
        width=3, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    
    plt.tight_layout()
    
    print('==================================')
    print(plot_xvals[np.argmax(p_all(plot_xvals))])
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/" + fname + "OL_speed_scatter.png", format='png', dpi=400)
    print("Saved: " + "C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/" + fname + "OL_speed_scatter.png")
    

def l23_expert_VR_vs_OL():
    
    print('=== calculating dF AUC VR vs. OL ===')
    AUC_VR = np.zeros((0,))
    AUC_OL = np.zeros((0,))
    peakdf_VR = np.zeros((0,))
    peakdf_OL = np.zeros((0,))
    
    if NAIVE_OL:
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_1"]["1115"], filedict["LF191022_1"]["1115 ol"], filedict["LF191022_1"]["roimatch ol 1115"], 'LF191022_1_AUC_vr_ol_1115')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_2"]["1116"], filedict["LF191022_2"]["1116 ol"], filedict["LF191022_2"]["roimatch ol 1116"], 'LF191022_2_AUC_vr_ol_1116')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_3"]["1114"], filedict["LF191022_3"]["1114 ol"], filedict["LF191022_3"]["roimatch ol 1114"], 'LF191022_3_AUC_vr_ol_1114')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191023_blank"]["1114"], filedict["LF191023_blank"]["1114 ol"], filedict["LF191023_blank"]["roimatch ol 1114"], 'LF191023_blank_AUC_vr_ol_1114')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191023_blue"]["1119"], filedict["LF191023_blue"]["1119 ol"], filedict["LF191023_blue"]["roimatch ol 1119"], 'LF191023_blue_AUC_vr_ol_1119')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191024_1"]["1115"], filedict["LF191024_1"]["1115 ol"], filedict["LF191024_1"]["roimatch ol 1115"], 'LF191024_1_AUC_vr_ol_1115')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
    
    if EXPERT_S1_OL:
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_1"]["1204"], filedict["LF191022_1"]["1204 ol"], filedict["LF191022_1"]["roimatch ol 1204"], 'LF191022_1_AUC_vr_ol_1204')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_2"]["1210"], filedict["LF191022_2"]["1210 ol"], filedict["LF191022_2"]["roimatch ol 1210"], 'LF191022_2_AUC_vr_ol_1210')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_3"]["1207"], filedict["LF191022_3"]["1207 ol"], filedict["LF191022_3"]["roimatch ol 1207"], 'LF191022_3_AUC_vr_ol_1207')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191023_blank"]["1206"], filedict["LF191023_blank"]["1206 ol"], filedict["LF191023_blank"]["roimatch ol 1206"], 'LF191023_blank_AUC_vr_ol_1206')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191023_blue"]["1204"], filedict["LF191023_blue"]["1204 ol"], filedict["LF191023_blue"]["roimatch ol 1204"], 'LF191023_blue_AUC_vr_ol_1204')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191024_1"]["1204"], filedict["LF191024_1"]["1204 ol"], filedict["LF191024_1"]["roimatch ol 1204"], 'LF191024_1_AUC_vr_ol_1204')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
    
    if EXPERT_S2_OL:
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_1"]["1209"], filedict["LF191022_1"]["1209 ol"], filedict["LF191022_1"]["roimatch ol 1209"], 'LF191022_1_AUC_vr_ol_1209')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_2"]["1210"], filedict["LF191022_2"]["1210 ol"], filedict["LF191022_2"]["roimatch ol 1210"], 'LF191022_2_AUC_vr_ol_1210')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191022_3"]["1207"], filedict["LF191022_3"]["1207 ol"], filedict["LF191022_3"]["roimatch ol 1207"], 'LF191022_3_AUC_vr_ol_1207')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191023_blank"]["1210"], filedict["LF191023_blank"]["1210 ol"], filedict["LF191023_blank"]["roimatch ol 1210"], 'LF191023_blank_AUC_vr_ol_1210')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191023_blue"]["1210"], filedict["LF191023_blue"]["1210 ol"], filedict["LF191023_blue"]["roimatch ol 1210"], 'LF191023_blue_AUC_vr_ol_1210')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session))
        
        AUC_VR_session, AUC_OL_session, peakdf_VR_session, peakdf_OL_session = df_AUC_vr_ol(filedict["LF191024_1"]["1210"], filedict["LF191024_1"]["1210 ol"], filedict["LF191024_1"]["roimatch ol 1210"], 'LF191024_1_AUC_vr_ol_1210')
        AUC_VR = np.hstack((AUC_VR, AUC_VR_session))
        AUC_OL = np.hstack((AUC_OL, AUC_OL_session))
        peakdf_VR = np.hstack((peakdf_VR, peakdf_VR_session))
        peakdf_OL = np.hstack((peakdf_OL, peakdf_OL_session ))
        
    # print(np.round(AUC_VR,2))
    # print(np.round(AUC_OL,2))
    # print(stats.mannwhitneyu(AUC_VR,AUC_OL))
    
    
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    # ax1.scatter(np.zeros((len(AUC_VR),)), AUC_VR, color='0.5', s=60)
    # ax1.scatter(np.ones((len(AUC_OL),)), AUC_OL, color='0.5', s=60)
    # ax1.scatter([0,1],[np.mean(AUC_VR),np.mean(AUC_OL)], color='k', s=60)
    
    for i in range(len(AUC_VR)):
        ax1.plot([0,1], [AUC_VR[i],AUC_OL[i]], marker='o', c='0.5', lw=2)
        
    ax1.plot([0,1], [np.mean(AUC_VR),np.mean(AUC_OL)], marker='o', c='k', lw=2)
    
    for i in range(len(peakdf_VR)):
        ax2.plot([0,1], [peakdf_VR[i],peakdf_OL[i]], marker='o', c='0.5', lw=2)
    ax2.plot([0,1], [np.mean(peakdf_VR),np.mean(peakdf_OL)], marker='o', c='k', lw=2)
    
    ax1.set_ylim([0,0.5])
    ax1.set_xlim([-0.1,1.1])
    ax2.set_xlim([-0.1,1.1])
    
    ax1.set_ylabel('Mean AUC')
    ax2.set_ylabel('Mean Peak')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.tick_params( \
        direction='out', \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
        
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.tick_params( \
        direction='out', \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    
    print("=========== RESULTS ================")
    print('AUC comparison:')
    print(stats.ttest_rel(AUC_VR,AUC_OL))
    print('Mean AUC VR: ' + str(np.nanmean(AUC_VR)) + ' SEM: ' + str(stats.sem(AUC_VR)))
    print('Mean AUC OL: ' + str(np.nanmean(AUC_OL)) + ' SEM: ' + str(stats.sem(AUC_OL)))
    print('Peak VR comparison:')
    print(stats.ttest_rel(peakdf_VR,peakdf_OL))
    print('Mean Peak VR: ' + str(np.nanmean(peakdf_VR)) + ' SEM: ' + str(stats.sem(peakdf_VR)))
    print('Mean Peak OL: ' + str(np.nanmean(peakdf_OL)) + ' SEM: ' + str(stats.sem(peakdf_OL)))
        
    plt.tight_layout()
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/VR_vs_OL.svg", format='svg')
    print("saved C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/VR_vs_OL.svg")
    print('====================================')
    
def l23_activity_expert_VR(): 
    ''' calculate neural activity for expert l2/3 sessions '''
    print('=== calculating dF AUC for VR expert sessions ===')
    
    df_activity = np.zeros((3,0))
    maxdf_all = np.zeros((3,0))

    df_session, maxdf_session = df_AUC([filedict["LF191022_1"]["1204"],filedict["LF191022_1"]["1207"],filedict["LF191022_1"]["1209"]],filedict["LF191022_1"]["roimatch"],[0,1,2])
    df_activity = np.hstack((df_activity, df_session.reshape((df_session.shape[0],1))))
    maxdf_all = np.hstack((maxdf_all, maxdf_session.reshape((maxdf_session.shape[0],1))))
       
    df_session, maxdf_session = df_AUC([filedict["LF191022_2"]["1206"],filedict["LF191022_2"]["1208"],filedict["LF191022_2"]["1210"]],filedict["LF191022_2"]["roimatch"],[0,1,2])
    df_activity = np.hstack((df_activity, df_session.reshape((df_session.shape[0],1))))
    maxdf_all = np.hstack((maxdf_all, maxdf_session.reshape((maxdf_session.shape[0],1))))
    
    df_session, maxdf_session = df_AUC([filedict["LF191022_3"]["1204"],filedict["LF191022_3"]["1207"],filedict["LF191022_3"]["1210"]],filedict["LF191022_3"]["roimatch"],[0,2,1])
    df_activity = np.hstack((df_activity, df_session.reshape((df_session.shape[0],1))))
    maxdf_all = np.hstack((maxdf_all, maxdf_session.reshape((maxdf_session.shape[0],1))))

    df_session, maxdf_session = df_AUC([filedict["LF191023_blank"]["1206"],filedict["LF191023_blank"]["1208"],filedict["LF191023_blank"]["1210"]],filedict["LF191023_blank"]["roimatch"],[0,1,2])
    df_activity = np.hstack((df_activity, df_session.reshape((df_session.shape[0],1))))
    maxdf_all = np.hstack((maxdf_all, maxdf_session.reshape((maxdf_session.shape[0],1))))
    
    df_session, maxdf_session = df_AUC([filedict["LF191023_blue"]["1204"],filedict["LF191023_blue"]["1208"],filedict["LF191023_blue"]["1210"]],filedict["LF191023_blue"]["roimatch"],[0,2,1])
    df_activity = np.hstack((df_activity, df_session.reshape((df_session.shape[0],1))))
    maxdf_all = np.hstack((maxdf_all, maxdf_session.reshape((maxdf_session.shape[0],1))))
    
    df_session, maxdf_session = df_AUC([filedict["LF191024_1"]["1204"],filedict["LF191024_1"]["1207"],filedict["LF191024_1"]["1210"]],filedict["LF191024_1"]["roimatch"],[0,1,2])
    df_activity = np.hstack((df_activity, df_session.reshape((df_session.shape[0],1))))
    maxdf_all = np.hstack((maxdf_all, maxdf_session.reshape((maxdf_session.shape[0],1))))
    
    print(df_activity)
    print(maxdf_session)
    # plt.plot(df_activity)
    plt.plot(maxdf_all)
    print(stats.f_oneway(df_activity[0],df_activity[1],df_activity[2]))
    print(stats.f_oneway(maxdf_all[0],maxdf_all[1],maxdf_all[2]))
    
    fig = plt.figure(figsize=(6,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    ax1.scatter(np.zeros((df_activity.shape[1],)), df_activity[0,:], color='0.5', s=60)
    ax1.scatter(np.ones((df_activity.shape[1],)), df_activity[1,:], color='0.5', s=60)
    ax1.scatter(np.ones((df_activity.shape[1],))*2, df_activity[2,:], color='0.5', s=60)
    ax1.scatter([0,1,2],np.mean(df_activity,1), color='k', s=60)
    
    for i in range(df_activity.shape[1]):
        ax1.plot([0,1,2], df_activity[:,i], c='0.5', lw=2)
        
    ax1.plot([0,1,2], np.mean(df_activity,1), c='k', lw=2)
    
    ax1.set_ylim([0,0.3])
    ax1.set_yticks([0.0,0.1,0.2,0.3])
    ax1.set_yticklabels(['0.0','0.1','0.2','0.3'])
    ax1.set_xlim([-0.1,2.1])
    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(['S1', 'S2', 'S3'])
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.tick_params( \
        direction='out', \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    
    ax2.scatter(np.zeros((maxdf_all.shape[1],)), maxdf_all[0,:], color='0.5', s=60)
    ax2.scatter(np.ones((maxdf_all.shape[1],)), maxdf_all[1,:], color='0.5', s=60)
    ax2.scatter(np.ones((maxdf_all.shape[1],))*2, maxdf_all[2,:], color='0.5', s=60)
    ax2.scatter([0,1,2],np.mean(maxdf_all,1), color='k', s=60)
    
    for i in range(maxdf_all.shape[1]):
        ax2.plot([0,1,2], maxdf_all[:,i], c='0.5', lw=2)
        
    ax2.plot([0,1,2], np.mean(maxdf_all,1), c='k', lw=2)
    
    ax2.set_ylim([0,0.9])
    ax2.set_yticks([0.0,0.3,0.6,0.9])
    ax2.set_yticklabels(['0.0','0.3','0.6','0.9'])
    ax2.set_xlim([-0.1,2.1])
    ax2.set_xticks([0,1,2])
    ax2.set_xticklabels(['S1', 'S2', 'S3'])
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.tick_params( \
        direction='out', \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
        
    plt.tight_layout()
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/AUC_VR_expert.svg", format='svg')
    print("saved C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/AUC_VR_expert.svg")
    
    
    
    print('====================================')

def test_analysis():
    zscore_null_all = np.zeros((NUM_SHUFFLE,))
    zscore_matched_all = np.zeros((0,))
    cc_matched_all = np.zeros((0,))
    cc_shuffled_all= np.zeros((NUM_SHUFFLE,))
    
    zscore_null, zscore_matched, pair_ccglm, cc_matched, cc_shuffled = run_cc_analysis(filedict["LF191022_1"]["1204"], 
                                                              filedict["LF191022_1"]["1207"], 
                                                              filedict["LF191022_1"]["roimatch"],  
                                                              [0,1], 
                                                              filedict["LF191022_1"]["cc_12041207"], 
                                                              filedict["LF191022_1"]["cshuffle_12041207"], 
                                                              'LF191022_1_rm_12041207')
    zscore_null_all = np.vstack((zscore_null_all, zscore_null))
    zscore_matched_all = np.hstack((zscore_matched_all, zscore_matched))
    cc_matched_all = np.hstack((cc_matched_all, cc_matched))
    cc_shuffled_all = np.vstack((cc_shuffled_all, cc_shuffled))

    fig = plt.figure(figsize=(10,5))
    ax = fig.subplots(2,4)
    ax[0,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1))
    ax[0,1].hist(cc_matched_all, np.arange(0,1.5,0.1))   
    ax[1,0].hist(cc_shuffled_all.flatten(), np.arange(0,1.5,0.1),edgecolor='black',facecolor='None')
    ax_n = ax[1,0].twinx()
    ax_n.hist(cc_matched_all, np.arange(0,1.5,0.1),edgecolor='r',facecolor='None')
    # ax[1,1].scatter(pair_ccglm[:,0], pair_ccglm[:,1])
    # ax[1,0].scatter(zscore_null,cc_null)
    sns.distplot(cc_shuffled_all.flatten(),bins=np.arange(0,1.5,0.1), color='k', ax=ax[1,1])
    sns.distplot(cc_matched_all, bins=np.arange(0,1.5,0.1), color='r', ax=ax[1,1])
    
    ax[0,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5))
    ax[0,3].hist(zscore_matched_all, np.arange(-5,5,0.5))   
    ax[1,2].hist(zscore_null_all.flatten(), np.arange(-5,5,0.5),edgecolor='black',facecolor='None')
    ax_n = ax[1,2].twinx()
    ax_n.hist(zscore_matched_all, np.arange(-5,5,0.5),edgecolor='r',facecolor='None')
    ax_n2 = ax[1,3].twinx()
    sns.distplot(zscore_null_all.flatten(),bins=np.arange(-5,5,0.5), color='k', ax=ax_n2)
    sns.distplot(zscore_matched_all, bins=np.arange(-5,5,0.5), color='r', ax=ax_n2)
    
    print(stats.ttest_ind(zscore_null_all.flatten(),zscore_matched_all))
    print(stats.mannwhitneyu(zscore_null_all.flatten(),zscore_matched_all))
    # np.savez("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc12_data",cc_shuffled_all,cc_matched_all,zscore_null_all,zscore_matched_all)


def cc_boxplot():
    load_data = np.load("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc12_data.npz") 
    plot_data = { 's12' : {}, 
                  's13' : {} }
    # plot_data['s12']['cc_shuffled_all']
    (plot_data['s12']['cc_shuffled_all'],plot_data['s12']['cc_matched_all'],plot_data['s12']['zscore_null_all'],plot_data['s12']['zscore_matched_all']) = (load_data['arr_0'],load_data['arr_1'],load_data['arr_2'],load_data['arr_3'])
    
    load_data = np.load("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Histo_cc13_data.npz") 
    (plot_data['s13']['cc_shuffled_all'],plot_data['s13']['cc_matched_all'],plot_data['s13']['zscore_null_all'],plot_data['s13']['zscore_matched_all']) = (load_data['arr_0'],load_data['arr_1'],load_data['arr_2'],load_data['arr_3'])
    
    bp_data = [plot_data['s12']['zscore_matched_all'],plot_data['s12']['zscore_null_all'].flatten(),plot_data['s13']['zscore_matched_all'],plot_data['s13']['zscore_null_all'].flatten()]
    fig = plt.figure(figsize=(5,5))
    ax = fig.subplots(1,1)
    bp = ax.boxplot(bp_data,
                     vert=True, patch_artist=True, bootstrap=1000, showcaps=False, positions=[1,2,4,5], showfliers=False,
                     whiskerprops=dict(linestyle='-', color='black', linewidth=2, solid_capstyle='butt'),
                     medianprops=dict(color='black', linewidth=2, solid_capstyle='butt'))

    colors = [ 'r','0.5','r','0.5']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.set_ylabel('Z-score')
    ax.set_xticks([1.5,4.5])
    ax.set_xticklabels(['Sess. 1-2', 'Sess. 1-3'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params( \
        direction='out', \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    
    print("=== BOXPLOT DATA ===")
    print('median cc s12: ' + str(np.median(bp_data[0])) + ' IQR: ' + str(stats.iqr(bp_data[0])))
    print('median shuffle s12: ' + str(np.median(bp_data[1])) + ' IQR: ' + str(stats.iqr(bp_data[1])))
    print('median cc s13: ' + str(np.median(bp_data[2])) + ' IQR: ' + str(stats.iqr(bp_data[2])))
    print('median shuffle s13: ' + str(np.median(bp_data[2])) + ' IQR: ' + str(stats.iqr(bp_data[3])))
    print('number of neurons s12: ' + str(bp_data[0].shape[0]))
    print('number of neurons s13: ' + str(bp_data[2].shape[0]))
    print("====================")
        
    fig.savefig("C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Crossday_corr_boxplot.svg", format='svg')
    print('Saved: C:/Users/lfisc/Work/Projects/Lntmodel/manuscript/Figure 2/Crossday_corr_boxplot.svg')
    

if __name__ == '__main__':
    
    filedict = load_filelist()
    # 
    # l23_sess_1_2()
    # l23_sess_1_3()
    # l23_sess_2_3()
    # l5_sess_1_2()
    # l5_sess_2_3()
    # l5_sess_1_3()
    # l23_naive_expert()
    # l23_tscore_corr()
    
    # ol_l23_sess_1()
    
    # l23_expert_VR_vs_OL()
    
    l23_activity_expert_VR()
    
    # test_analysis()
    #
    
    # ol_speeddiff_response() ##
    # ol_speedcorr()
    
    
    # cc_boxplot()
    
    