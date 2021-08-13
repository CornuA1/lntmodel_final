# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:49:16 2019

@author: Lou
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:53:16 2019

@author: Lou
"""

import sys, os, yaml
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f, Loader=yaml.FullLoader)
sys.path.append(loc_info['base_dir'] + '/Analysis')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as scistat
import seaborn as sns
sns.set_style('white')

def plot_trans(dF_signal, baseline_percentile, rois_list, on_rois, off_rois, transients, sess, subfolder, fname):
    num_rois = len(rois_list)
    axrows_per_roi = int(100/num_rois)
    roi_ax = []
    std_threshold = 6
    fig = plt.figure(figsize=(100,15))
    for i,r in enumerate(rois_list):
        if i > 0:
            roi_ax.append(plt.subplot2grid((100,50),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50,sharex=roi_ax[0]))
            plt.ylabel(str(r + 1))
        else:
            roi_ax.append(plt.subplot2grid((100,50),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50))
            plt.title(sess)
            plt.ylabel(str(r + 1))
        roi_ax[-1].spines['left'].set_linewidth(2)
        roi_ax[-1].spines['top'].set_visible(False)
        roi_ax[-1].spines['right'].set_visible(False)
        roi_ax[-1].spines['bottom'].set_visible(False)
        roi_ax[-1].tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            bottom='off', \
            right='off', \
            top='off')
        
    for i,roi in enumerate(rois_list):
        rois_unfiltered = dF_signal[:,roi]
        percentile_low_idx = np.where(rois_unfiltered < np.percentile(rois_unfiltered,baseline_percentile))[0]
        rois_std = np.std(rois_unfiltered[percentile_low_idx])
        rois_mean = np.mean(rois_unfiltered[percentile_low_idx])
        roi_ax[i].axhspan(rois_mean, (std_threshold*rois_std)+rois_mean, color='0.9',zorder=0)
        rois_idx = np.arange(0,len(dF_signal[:,roi]),1)
        roi_ax[i].plot(rois_idx, rois_unfiltered,label='dF/F',c='k',lw=1)
        for tran in transients:
            if tran[0] == roi:
                for on_df in range(len(on_rois[roi])):
                    if on_rois[roi][on_df] < tran[1] < off_rois[roi][on_df]:
                        if tran[2] > 0.6:
                            roi_ax[i].plot(np.arange(on_rois[roi][on_df],off_rois[roi][on_df],1),rois_unfiltered[on_rois[roi][on_df]:off_rois[roi][on_df]],c='red',lw=1.5)
                        else:
                            roi_ax[i].plot(np.arange(on_rois[roi][on_df],off_rois[roi][on_df],1),rois_unfiltered[on_rois[roi][on_df]:off_rois[roi][on_df]],c='blue',lw=1.5)
        
    plt.tight_layout()    
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)

def compare_rois(dF_signal, sess, fs, subfolder, fname, baseline_percentile):
    std_threshold = 6
    frame_latency = 1/fs
    rois_list = range(len(dF_signal[0]))
    rois_rejected = []
    rois_approved = []
    transients = []
    ac_on_rois = []
    ac_off_rois = []
    re_on_rois = []
    re_off_rois = []
    for roi in rois_list:
        rois_unfiltered = dF_signal[:,roi]
        min_transient_length_sec = 0.4
        min_transient_length = min_transient_length_sec/frame_latency
        percentile_low_idx = np.where(rois_unfiltered < np.percentile(rois_unfiltered,baseline_percentile))[0]
        rois_std = np.std(rois_unfiltered[percentile_low_idx])
        rois_mean = np.mean(rois_unfiltered[percentile_low_idx])
        transient_high = np.where(rois_unfiltered > (std_threshold*rois_std)+rois_mean)[0]       
        if transient_high.size == 0:
            re_on_rois.append([])
            re_off_rois.append([])
            rois_rejected.append(roi)
            break
        idx_diff = np.diff(transient_high)
        idx_diff = np.insert(idx_diff,0,transient_high[0])
        gap_tolerance_frames = int(0.2/frame_latency)    
        onset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]]-1
        offset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]-1]+1
        offset_idx = np.roll(offset_idx, -1)
        index_adjust = 0
        for j in range(len(onset_idx)):
            temp_length = offset_idx[j-index_adjust] - onset_idx[j-index_adjust]
            if temp_length < min_transient_length:
                onset_idx = np.delete(onset_idx,j-index_adjust)
                offset_idx = np.delete(offset_idx,j-index_adjust)
                index_adjust += 1
        above_1_std = np.where(rois_unfiltered > (2*rois_std)+rois_mean)[0]
        one_std_idx_diff = np.diff(above_1_std)
        one_std_idx_diff = np.insert(one_std_idx_diff,0,one_std_idx_diff[0])
        one_std_onset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]]-1
        one_std_offset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]-1]+1
        onsets_pre = []
        offsets_pre = []
        onsets = []
        offsets = []
        for oi in onset_idx:
            closest_idx = one_std_onset_idx - oi
            closest_idx_neg = np.where(closest_idx < 0)[0]
            if closest_idx_neg.size == 0:
                closest_idx_neg = [-1]
                one_std_idx = 0
            else:
                one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
            onsets_pre.append(oi-one_std_idx)
        for oi in offset_idx:
            closest_idx = one_std_offset_idx - oi
            closest_idx_neg = np.where(closest_idx > 0)[0]
            if closest_idx_neg.size == 0:
                closest_idx_neg = [-1]
                one_std_idx = 0
            else:
                one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
            offsets_pre.append(oi+one_std_idx)
        for j in range(len(onsets_pre)):
            if j == 0 or (onsets_pre[j] != onsets_pre[j-1] and offsets_pre[j] != offsets_pre[j-1]):
                onsets.append(onsets_pre[j])
                offsets.append(offsets_pre[j])

        pearson_value = scistat.pearsonr(dF_signal[:,0],rois_unfiltered)[0]
        if pearson_value >= 0.10 and len(onsets) > 0:
            rois_approved.append(roi)
            ac_on_rois.append(onsets)
            ac_off_rois.append(offsets)
        else:
            rois_rejected.append(roi)
            re_on_rois.append(onsets)
            re_off_rois.append(offsets)
    
    for roi in range(len(rois_approved)):
        rois_unfiltered = dF_signal[:,rois_approved[roi]]
        cur_on_roi = ac_on_rois[roi]
        cur_off_roi = ac_off_rois[roi]
        for tran_0 in range(len(cur_on_roi)):
            count = 0
            ave_tran = (cur_on_roi[tran_0] + cur_off_roi[tran_0]) / 2
            transients.append([rois_approved[roi], ave_tran, np.max(rois_unfiltered[cur_on_roi[tran_0]:cur_off_roi[tran_0]])])
            for tran_1 in range(len(ac_on_rois)):
                if roi != tran_1:
                    comp_on_roi = ac_on_rois[tran_1]
                    comp_off_roi = ac_off_rois[tran_1]
                    for tran_2 in range(len(comp_on_roi)):
                        ave_comp_tran = (comp_on_roi[tran_2] + comp_off_roi[tran_2]) / 2
                        if (cur_on_roi[tran_0] < ave_comp_tran < cur_off_roi[tran_0]) or (comp_on_roi[tran_2] < ave_tran < comp_off_roi[tran_2]):
                            count += 1
            if (count / len(rois_approved)) > 1:
                transients[-1].append(1)
            else:
                transients[-1].append(count / len(rois_approved))
        
#    for roi in rois_rejected:
#        rois_unfiltered = dF_signal[:,roi]
#        cur_on_roi = re_on_rois[roi]
#        cur_off_roi = re_off_rois[roi]
#        for tran_0 in range(len(cur_on_roi)):
#            count = 0
#            ave_tran = (cur_on_roi[tran_0] + cur_off_roi[tran_0]) / 2
#            transients.append([roi, ave_tran, np.max(rois_unfiltered[cur_on_roi[tran_0]:cur_off_roi[tran_0]])])
#            for tran_1 in range(len(re_on_rois)):
#                if roi != tran_1:
#                    comp_on_roi = re_on_rois[tran_1]
#                    comp_off_roi = re_off_rois[tran_1]
#                    for tran_2 in range(len(comp_on_roi)):
#                        ave_comp_tran = (comp_on_roi[tran_2] + comp_off_roi[tran_2]) / 2
#                        if (cur_on_roi[tran_0] < ave_comp_tran < cur_off_roi[tran_0]) or (comp_on_roi[tran_2] < ave_tran < comp_off_roi[tran_2]):
#                            count += 1
#            transients[-1].append(count / len(rois_rejected))
#
#    end = True
#    count = 1
#    while end:
#        ned_stat = len(rois_approved) - (count - 1)*10
#        if ned_stat < 10:
#            plot_trans(dF_signal, baseline_percentile, rois_approved[((count - 1)*10):], ac_on_rois, ac_off_rois, transients, sess, subfolder, fname=fname + str(count))
#            end = False
#        else:
#            plot_trans(dF_signal, baseline_percentile, rois_approved[((count - 1)*10):(count *10)], ac_on_rois, ac_off_rois, transients, sess, subfolder, fname=fname + str(count))
#        count += 1
#    plot_trans(dF_signal, baseline_percentile, rois_rejected, on_rois, off_rois, transients, sess, subfolder, fname=fname + 'reject')
#    plot_trans(dF_signal, baseline_percentile, rois_approved[0,25,30,50,60], ac_on_rois, ac_off_rois, transients, sess, subfolder, fname=fname + str(count))

    
    return transients

def run_Buddha():
    subfolder='Buddha transients'
    fname='compare'
    MOUSE = 'Buddha'
    transient_list = []
    session = ['190802_0','190802_1','190802_2','190802_3','190802_4']
    for sess in session:
        processed_data_path = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
        loaded_data = sio.loadmat(processed_data_path)
        dF_signal = loaded_data['dF_data']
        transient_list += compare_rois(dF_signal, sess, fs=15.5, subfolder='Buddha transients', fname='compare', baseline_percentile=60)
    subfolder='Buddha transients'
    fname='compare'
    fig = plt.figure(figsize=(100,15))
    ax1 = plt.subplot2grid((100,100),(0,0), rowspan=100, colspan=100)
#    ax1.set_title('Peak Transient Amp vs. fraction shared')
#    ax1.set_ylabel('Fraction Shared %')
#    ax1.set_xlabel('Peak Transient dF/F')
    for tran in transient_list:
        ax1.plot(tran[2],tran[3],color='red',marker='o')
    
    plt.tight_layout()    
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)

if __name__ == '__main__':
    fformat = 'png'
    run_Buddha()
    
    
#    print(transient_list)
#    tran_value = []
#    for tran_it in transient_list:
#        count = 0
#        average_df = 0
#        average_per = 0
#        for tran in tran_it:
#            if tran[1] == 'a':
#                average_df += tran[4]
#                average_per += tran[5]
#                count += 1
#        if len(tran_it) != 0 and count != 0:
#            tran_value.append([average_df/count,average_per/count])
##        if len(tran_it) != 0:
##            tran_value.append([average_df/len(tran_it),average_per/len(tran_it)])
#        
#    subfolder='Buddha transients'
#    fname='compare_all'
#    fig = plt.figure(figsize=(100,15))
#    ax1 = plt.subplot2grid((100,100),(0,0), rowspan=100, colspan=100)
#    for data in tran_value:
#        ax1.plot(data[0],data[1],color='red',marker='o')
#    plt.tight_layout()    
#    if subfolder != []:
#        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
#            os.mkdir(loc_info['figure_output_path'] + subfolder)
#        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
#    else:
#        fname = loc_info['figure_output_path'] + fname + '.' + fformat
#    print(fname)
#    fig.savefig(fname, format=fformat,dpi=150)
#        