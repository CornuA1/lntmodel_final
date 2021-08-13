# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:37:20 2019

@author: Lou
"""

import sys, os, yaml
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f, Loader=yaml.FullLoader)
sys.path.append(loc_info['base_dir'] + '/Analysis')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy.io as sio
import scipy.stats as scistat
import seaborn as sns
sns.set_style('white')



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def id_plot_transients(dF_signal_pre, dF_signal_post, rois_pre, rois_post, sess_pre, sess_post, subfolder=[], fname='transients', transients_yn=False):
    
    fs=15.5
    num_rois_pre = len(rois_pre)
    num_rois_post = len(rois_post)
    axrows_per_roi_pre = int(100/num_rois_pre)
    axrows_per_roi_post = int(100/num_rois_post)
    roi_ax_pre = []
    roi_ax_post = []
    fig = plt.figure(figsize=(100,15))

    for i,r in enumerate(rois_pre):
        if i > 0:
            roi_ax_pre.append(plt.subplot2grid((100,100),(i*axrows_per_roi_pre,0), rowspan=axrows_per_roi_pre, colspan=50,sharex=roi_ax_pre[0]))
            plt.ylabel(str(r))
        else:
            roi_ax_pre.append(plt.subplot2grid((100,100),(i*axrows_per_roi_pre,0), rowspan=axrows_per_roi_pre, colspan=50))
            plt.title(sess_pre)
            
        roi_ax_pre[-1].spines['left'].set_linewidth(2)
        roi_ax_pre[-1].spines['top'].set_visible(False)
        roi_ax_pre[-1].spines['right'].set_visible(False)
        roi_ax_pre[-1].spines['bottom'].set_visible(False)
        roi_ax_pre[-1].tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            bottom='off', \
            right='off', \
            top='off')

    for i,r in enumerate(rois_post):
        if i > 0:
            roi_ax_post.append(plt.subplot2grid((100,100),(i*axrows_per_roi_post,50), rowspan=axrows_per_roi_post, colspan=50,sharex=roi_ax_post[0]))
            plt.ylabel(str(r))
        else:
            roi_ax_post.append(plt.subplot2grid((100,100),(i*axrows_per_roi_post,50), rowspan=axrows_per_roi_post, colspan=50))
            plt.title(sess_post)
        
        roi_ax_post[-1].spines['left'].set_linewidth(2)
        roi_ax_post[-1].spines['top'].set_visible(False)
        roi_ax_post[-1].spines['right'].set_visible(False)
        roi_ax_post[-1].spines['bottom'].set_visible(False)
        roi_ax_post[-1].tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            bottom='off', \
            right='off', \
            top='off')
        
    
    # set standard deviation threshold above which traces have to be to be considered a transient
    std_threshold = 6

    # calculate frame to frame latency
    frame_latency = 1/fs
    
    # low pass filter trace
#    order = 6
#    cutoff = 4 # desired cutoff frequency of the filter, Hz
    pre_trunk_filt = np.array([])
    post_trunk_filt = np.array([])
    pre_on_trunk = []
    pre_off_trunk = []
    post_on_trunk = []
    post_off_trunk = []
    return_lis = []


    for i,roi in enumerate(rois_pre):
        if i == 0:
             baseline_percentile = 70
#             pre_trunk_filt = butter_lowpass_filter(dF_signal_pre[:,roi], cutoff, fs, order)
             pre_trunk_filt = dF_signal_pre[:,roi]
        else:
             baseline_percentile = 70
#        rois_filtered = butter_lowpass_filter(dF_signal_pre[:,roi], cutoff, fs, order)
        rois_unfiltered = dF_signal_pre[:,roi]
        rois_filtered = rois_unfiltered

        # set minimum transient length in seconds that has to be above threshold
        min_transient_length_sec = 0.4

        # min transient length in number of frames
        min_transient_length = min_transient_length_sec/frame_latency
        
        # get standard deviation of lower 80% of samples
        percentile_low_idx = np.where(rois_filtered < np.percentile(rois_filtered,baseline_percentile))[0]
    
        rois_std = np.std(rois_filtered[percentile_low_idx])
        rois_mean = np.mean(rois_filtered[percentile_low_idx])
        roi_ax_pre[i].axhspan(rois_mean, (std_threshold*rois_std)+rois_mean, color='0.9',zorder=0)

        # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,rois],c='r',lw=1)
        rois_idx = np.arange(0,len(dF_signal_pre[:,roi]),1)
        roi_ax_pre[i].plot(rois_idx, rois_filtered,label='dF/F',c='k',lw=1)
    
        # get indeces above speed threshold
        transient_high = np.where(rois_filtered > (std_threshold*rois_std)+rois_mean)[0]
        
        if transient_high.size == 0:
            # fig.suptitle(fname, wrap=True)
            if subfolder != []:
                if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
                    os.mkdir(loc_info['figure_output_path'] + subfolder)
                fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
            else:
                fname = loc_info['figure_output_path'] + fname + '.' + fformat
            print(fname)
            fig.savefig(fname, format=fformat,dpi=150)
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # use diff to find gaps between episodes of high speed
        idx_diff = np.diff(transient_high)
        idx_diff = np.insert(idx_diff,0,transient_high[0])
    
        # convert gap tolerance from cm to number of frames
        gap_tolerance_frames = int(0.2/frame_latency)
        # find indeces where speed exceeds threshold. If none are found, return
    
        # find transient onset and offset points. Colapse transients that briefly dip below threshold
        onset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]]-1
        offset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]-1]+1
        # this is necessary as the first index of the offset is actually the end of the last one (this has to do with indexing conventions on numpy)
        offset_idx = np.roll(offset_idx, -1)

        # calculate the length of each transient and reject those too short to be considered
        index_adjust = 0
        for j in range(len(onset_idx)):
            temp_length = offset_idx[j-index_adjust] - onset_idx[j-index_adjust]
            if temp_length < min_transient_length:
                onset_idx = np.delete(onset_idx,j-index_adjust)
                offset_idx = np.delete(offset_idx,j-index_adjust)
                index_adjust += 1

        # find the onset by looking for the point where the transient drops below 1 std
        above_1_std = np.where(rois_filtered > (2*rois_std)+rois_mean)[0]
        one_std_idx_diff = np.diff(above_1_std)
        one_std_idx_diff = np.insert(one_std_idx_diff,0,one_std_idx_diff[0])
    
        one_std_onset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]]-1
        one_std_offset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]-1]+1

        onsets_pre = []
        onsets = []
        offsets_pre = []
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
        
        # find max transient length
        max_transient_length = 0
        for j in range(len(onsets_pre)):
            if j == 0 or (onsets_pre[j] != onsets_pre[j-1] and offsets_pre[j] != offsets_pre[j-1]):
                onsets.append(onsets_pre[j])
                offsets.append(offsets_pre[j])
            if offsets_pre[j]-onsets_pre[j] > max_transient_length:
                max_transient_length = offsets_pre[j]-onsets_pre[j]
        all_transients = np.full((len(onsets),max_transient_length), np.nan)
        all_transients_norm = np.full((len(onsets),max_transient_length), np.nan)
        for j in range(len(onsets)):
            all_transients[j,0:offsets[j]-onsets[j]] = rois_unfiltered[onsets[j]:offsets[j]]
            all_transients_norm[j,0:offsets[j]-onsets[j]] = (rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]])/(np.nanmax(rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]]))
        for j in range(len(onsets)):
            max_plot_val = np.max(rois_filtered[onsets[j]:offsets[j]])
            max_plot_position = np.where(rois_filtered[onsets[j]:offsets[j]] == max_plot_val)[0] + onsets[j]
            if i == 0 and transients_yn:
                pre_on_trunk.append(onsets[j])
                pre_off_trunk.append(offsets[j])
                roi_ax_pre[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
            elif transients_yn:
                for tran in range(len(pre_on_trunk)):
                    if pre_on_trunk[tran] < max_plot_position < pre_off_trunk[tran]:
                        roi_ax_pre[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
                        break
                    elif tran == len(pre_on_trunk) - 1:
# blue
                        roi_ax_pre[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='b',lw=1.5)
                        return_lis.append([fname, 'pre', roi, onsets[j], offsets[j], offsets[j] - onsets[j], round(max_plot_val, 2)])
            else:
                roi_ax_pre[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
        if i == 1 and transients_yn:
            for k in range(len(pre_on_trunk)):
                count = 0
                mean_trunk_pre = (pre_on_trunk[k]+pre_off_trunk[k])/2
                for l in range(len(onsets)):
                    mean_tuft_pre = (onsets[l]+offsets[l])/2
                    if onsets[l] <= mean_trunk_pre <= offsets[l] or pre_on_trunk[k] <= mean_tuft_pre <= pre_off_trunk[k]:
                        count += 1
                if count == 0:
# lime
                    roi_ax_pre[0].plot(np.arange(pre_on_trunk[k],pre_off_trunk[k],1),pre_trunk_filt[pre_on_trunk[k]:pre_off_trunk[k]],c='lime',lw=1.5)
            
    for i,roi in enumerate(rois_post):
        if i == 0:
             baseline_percentile = 70
#             post_trunk_filt = butter_lowpass_filter(dF_signal_post[:,roi], cutoff, fs, order)
             post_trunk_filt = dF_signal_post[:,roi]
        else:
             baseline_percentile = 70
#        rois_filtered = butter_lowpass_filter(dF_signal_post[:,roi], cutoff, fs, order)
        rois_unfiltered = dF_signal_post[:,roi]
        rois_filtered = rois_unfiltered

        # set minimum transient length in seconds that has to be above threshold
        min_transient_length_sec = 0.4

        # min transient length in number of frames
        min_transient_length = min_transient_length_sec/frame_latency
        
        # get standard deviation of lower 80% of samples
        percentile_low_idx = np.where(rois_filtered < np.percentile(rois_filtered,baseline_percentile))[0]
    
        rois_std = np.std(rois_filtered[percentile_low_idx])
        rois_mean = np.mean(rois_filtered[percentile_low_idx])
        roi_ax_post[i].axhspan(rois_mean, (std_threshold*rois_std)+rois_mean, color='0.9',zorder=0)

        # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,rois],c='r',lw=1)
        rois_idx = np.arange(0,len(dF_signal_post[:,roi]),1)
        roi_ax_post[i].plot(rois_idx, rois_filtered,label='dF/F',c='k',lw=1)
    
        # get indeces above speed threshold
        transient_high = np.where(rois_filtered > (std_threshold*rois_std)+rois_mean)[0]
        
        if transient_high.size == 0:
            # fig.suptitle(fname, wrap=True)
            if subfolder != []:
                if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
                    os.mkdir(loc_info['figure_output_path'] + subfolder)
                fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
            else:
                fname = loc_info['figure_output_path'] + fname + '.' + fformat
            print(fname)
            fig.savefig(fname, format=fformat,dpi=150)
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        # ax1.plot(rois_idx[transient_high], rois_filtered[transient_high],c='r',lw=1)
    
    
        # use diff to find gaps between episodes of high speed
        idx_diff = np.diff(transient_high)
        idx_diff = np.insert(idx_diff,0,transient_high[0])
    
        # convert gap tolerance from cm to number of frames
        gap_tolerance_frames = int(0.2/frame_latency)
        # find indeces where speed exceeds threshold. If none are found, return
    
        # find transient onset and offset points. Colapse transients that briefly dip below threshold
        onset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]]-1
        offset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]-1]+1
        # this is necessary as the first index of the offset is actually the end of the last one (this has to do with indexing conventions on numpy)
        offset_idx = np.roll(offset_idx, -1)

        # calculate the length of each transient and reject those too short to be considered
        index_adjust = 0
        for j in range(len(onset_idx)):
            temp_length = offset_idx[j-index_adjust] - onset_idx[j-index_adjust]
            if temp_length < min_transient_length:
                onset_idx = np.delete(onset_idx,j-index_adjust)
                offset_idx = np.delete(offset_idx,j-index_adjust)
                index_adjust += 1

        # find the onset by looking for the point where the transient drops below 1 std
        above_1_std = np.where(rois_filtered > (2*rois_std)+rois_mean)[0]
        one_std_idx_diff = np.diff(above_1_std)
        one_std_idx_diff = np.insert(one_std_idx_diff,0,one_std_idx_diff[0])
    
        one_std_onset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]]-1
        one_std_offset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]-1]+1

        onsets_pre = []
        onsets = []
        offsets_pre = []
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
        
        # find max transient length
        max_transient_length = 0
        for j in range(len(onsets_pre)):
            if j == 0 or (onsets_pre[j] != onsets_pre[j-1] and offsets_pre[j] != offsets_pre[j-1]):
                onsets.append(onsets_pre[j])
                offsets.append(offsets_pre[j])
            if offsets_pre[j]-onsets_pre[j] > max_transient_length:
                max_transient_length = offsets_pre[j]-onsets_pre[j]
        all_transients = np.full((len(onsets),max_transient_length), np.nan)
        all_transients_norm = np.full((len(onsets),max_transient_length), np.nan)
        for j in range(len(onsets)):
            all_transients[j,0:offsets[j]-onsets[j]] = rois_unfiltered[onsets[j]:offsets[j]]
            all_transients_norm[j,0:offsets[j]-onsets[j]] = (rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]])/(np.nanmax(rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]]))
        for j in range(len(onsets)):
            max_plot_val = np.max(rois_filtered[onsets[j]:offsets[j]])
            max_plot_position = np.where(rois_filtered[onsets[j]:offsets[j]] == max_plot_val)[0] + onsets[j]            
            if i == 0 and transients_yn:
                post_on_trunk.append(onsets[j])
                post_off_trunk.append(offsets[j])
                roi_ax_post[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
            elif transients_yn:
                for tran in range(len(post_on_trunk)):
                    if post_on_trunk[tran] < max_plot_position < post_off_trunk[tran]:
                        roi_ax_post[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
                        break
                    elif tran == len(post_on_trunk) - 1:
# blue
                        roi_ax_post[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='b',lw=1.5)
                        return_lis.append([fname, 'post', roi, onsets[j], offsets[j], offsets[j] - onsets[j], round(max_plot_val, 2)])
            else:
                roi_ax_post[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
        if i == 1 and transients_yn:
            for k in range(len(post_on_trunk)):
                count = 0
                mean_trunk_post = (post_on_trunk[k]+post_off_trunk[k])/2
                for l in range(len(onsets)):
                    mean_tuft_post = (onsets[l]+offsets[l])/2
                    if onsets[l] <= mean_trunk_post <= offsets[l] or post_on_trunk[k] <= mean_tuft_post <= post_off_trunk[k]:
                        count += 1
# lime
                if count == 0:
                    roi_ax_post[0].plot(np.arange(post_on_trunk[k],post_off_trunk[k],1),post_trunk_filt[post_on_trunk[k]:post_off_trunk[k]],c='lime',lw=1.5)

    return_trunk_lis = []
    return_trunk_lis.append([pre_on_trunk, pre_off_trunk])
    return_trunk_lis.append([post_on_trunk, post_off_trunk])
    plt.tight_layout()
    
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)
    
    return return_trunk_lis, return_lis

def sort_func(trunk):
    total = 0
    total += int(trunk[0][-1]) * 1000000000 + trunk[2] * 100000 + trunk[3]
    if trunk[1][-1] == 'e':
        total += 10000000
    else:
        total += 20000000
    return total

def ind_list_comp(return_trunk, return_tufts):
    app_tufts = []
    for tuft in return_tufts:
        count = 0
        mean_tuft = (tuft[3] + tuft[4])/2
        for trunks in return_trunk:
            for trunk in range(len(trunks[0])):
                trunk_mean = (trunks[0][trunk] + trunks[1][trunk])/2
                if trunks[0][trunk] <= mean_tuft <= trunks[1][trunk] or tuft[3] <= trunk_mean <= tuft[4]:
                    count += 1
        if count == 0:
            app_tufts.append(tuft)
    app_tufts.sort(key=sort_func)
    for fin_tuft in app_tufts:
        print(fin_tuft)
    
def compare_rois(dF_signal_pre, dF_signal_post, rois_0000, sess_pre, sess_post, subfolder=[], fname='transients'):
#    fs = 15.5
#    order = 6
#    cutoff = 4
    rois_pre = enumerate(range(len(dF_signal_pre[0])))
    rois_post = enumerate(range(len(dF_signal_post[0])))
    filt_rois_pre = []
    filt_rois_post = []
    pre_out = [elem for elem in rois_0000]
    post_out = [elem for elem in rois_0000]
    pre_rej = [elem for elem in rois_0000]
    post_rej = [elem for elem in rois_0000]
    return_lis = []
    return_tufts = []
    
    for i,roi in enumerate(rois_0000):
#        filt_rois_pre.append([[butter_lowpass_filter(dF_signal_pre[:,roi], cutoff, fs, order),roi]])
#        filt_rois_post.append([[butter_lowpass_filter(dF_signal_post[:,roi], cutoff, fs, order),roi]])
        filt_rois_pre.append([[dF_signal_pre[:,roi],roi]])
        filt_rois_post.append([[dF_signal_post[:,roi],roi]])

    for i,roi in rois_pre:
        if roi not in rois_0000:
#            rois_filtered_pre = butter_lowpass_filter(dF_signal_pre[:,roi], cutoff, fs, order)
            rois_filtered_pre = dF_signal_pre[:,roi]
            pre_roi_pearson = []
            for ii in range(len(filt_rois_pre)):
                pre_roi_pearson.append(scistat.pearsonr(rois_filtered_pre,filt_rois_pre[ii][0][0])[0])
            if round(sorted(pre_roi_pearson)[-2], 1) >= .5:
                pre_out.append(roi)
            if round(max(pre_roi_pearson), 1) >= .5:
                in_int = pre_roi_pearson.index(max(pre_roi_pearson))
                filt_rois_pre[in_int].append([rois_filtered_pre,roi])
            else:
                pre_rej.append(roi)
                
    for i,roi in rois_post:
        if roi not in rois_0000:
#            rois_filtered_post = butter_lowpass_filter(dF_signal_post[:,roi], cutoff, fs, order)
            rois_filtered_post = dF_signal_post[:,roi]
            post_roi_pearson = []
            for ii in range(len(filt_rois_post)):
                post_roi_pearson.append(scistat.pearsonr(rois_filtered_post,filt_rois_post[ii][0][0])[0])
            if round(sorted(post_roi_pearson)[-2], 1) >= .5:
                post_out.append(roi)
            elif round(max(post_roi_pearson), 1) >= .5:
                in_int = post_roi_pearson.index(max(post_roi_pearson))
                filt_rois_post[in_int].append([rois_filtered_post,roi])
            else:
                post_rej.append(roi)
                
    for i,roi in enumerate(rois_0000):
        cur_roi_lis_pre = []
        cur_roi_lis_post = []
        for ii in range(len(filt_rois_pre[i])):
            cur_roi_lis_pre.append(filt_rois_pre[i][ii][1])
        for ii in range(len(filt_rois_post[i])):
            cur_roi_lis_post.append(filt_rois_post[i][ii][1])
        fname_n = fname + '_' + str(i + 1)
        trans_trunk, trans_tufts = id_plot_transients(dF_signal_pre, dF_signal_post, cur_roi_lis_pre, cur_roi_lis_post, sess_pre, sess_post, subfolder, fname_n, transients_yn = True)
        return_lis = return_lis + trans_tufts
        return_tufts = return_tufts + trans_trunk
    id_plot_transients(dF_signal_pre, dF_signal_post, pre_out, post_out, sess_pre, sess_post, subfolder, fname='outliers',  transients_yn=False)
    id_plot_transients(dF_signal_pre, dF_signal_post, pre_rej, post_rej, sess_pre, sess_post, subfolder, fname='rejects', transients_yn=False)
    return_lis_ref = [elem for elem in return_lis if elem[-2] >= 1 and elem[-1] >= 1.0]
    return_lis_ref.sort(key=lambda x : x[-1], reverse=True)
    ind_list_comp(trans_trunk, return_lis_ref)

def trunk_analysis():
    sigfile = 'sig_data_0000.mat'
    rois_0000 = [0]
    
    MOUSE_jim = 'Jimmy'
    sess_list_jim = ['190507_3','190508_8','190514_10','190627_0','190709_0','190715_1']
#    sess_list_jim = ['190507_3','190508_8']
    for sess in sess_list_jim:
        processed_data_path_jim = loc_info['raw_dir'] + '\\' + MOUSE_jim + os.sep + sess + os.sep + sigfile
        loaded_data_jim = sio.loadmat(processed_data_path_jim)['dF_data']
        if sess == sess_list_jim[0]:
            dF_signal_jim = loaded_data_jim
        else:
            for data in range(len(loaded_data_jim[0])):
                if len(dF_signal_jim) > len(loaded_data_jim[:,data]):
                    dF_signal_jim = dF_signal_jim[:(len(loaded_data_jim[:,data]))]
                    dF_signal_jim = np.column_stack((dF_signal_jim,loaded_data_jim[:,data]))
                elif len(dF_signal_jim) <= len(loaded_data_jim[:,data]):
                    crop_data = loaded_data_jim[:,data][:(len(dF_signal_jim))]
                    dF_signal_jim = np.column_stack((dF_signal_jim,crop_data))
                else:
                    print('Welp, I didn\'t predict this.')
        
    MOUSE_pum = 'Pumba'
    sess_list_pum = ['190503_2','190508_9','190514_11','190627_3','190709_2','190715_3']
#    sess_list_pum = ['190503_2','190508_9']
    for sess in sess_list_pum:
        processed_data_path_pum = loc_info['raw_dir'] + '\\' + MOUSE_pum + os.sep + sess + os.sep + sigfile
        loaded_data_pum = sio.loadmat(processed_data_path_pum)['dF_data']
        if sess in sess_list_pum[0]:
            dF_signal_pum = loaded_data_pum
        else:
            for data in range(len(loaded_data_pum[0])):
                if len(dF_signal_pum) > len(loaded_data_pum[:,data]):
                    dF_signal_pum = dF_signal_pum[:(len(loaded_data_pum[:,data]))]
                    dF_signal_pum = np.column_stack((dF_signal_pum,loaded_data_pum[:,data]))
                elif len(dF_signal_pum) <= len(loaded_data_pum[:,data]):
                    crop_data = loaded_data_pum[:,data][:(len(dF_signal_pum))]
                    dF_signal_pum = np.column_stack((dF_signal_pum,crop_data))
                else:
                    print('Welp, I didn\'t predict this.')
    
    compare_rois(dF_signal_jim, dF_signal_pum, rois_0000, MOUSE_jim, MOUSE_pum, subfolder='trunk analysis', fname='trunks')
        
def run_Jimmy_analysis():
    MOUSE = 'Jimmy'
    rois_0000 = [0,1,2]
    
    sess_pre = '190507_3'
    processed_data_path_pre = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess_pre + os.sep + 'sig_data.mat'
    loaded_data_pre = sio.loadmat(processed_data_path_pre)
    dF_signal_pre = loaded_data_pre['dF_data']
    
    sess_post = '190717_0'
    processed_data_path_post = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess_post + os.sep + 'sig_data.mat'
    loaded_data_post = sio.loadmat(processed_data_path_post)
    dF_signal_post = loaded_data_post['dF_data']
    
    compare_rois(dF_signal_pre, dF_signal_post, rois_0000, sess_pre, sess_post, subfolder='Jimmy Summary', fname='summary')
    
def run_Pumba_analysis():
    MOUSE = 'Pumba'
    rois_0000 = [0,1,2]
    
    sess_pre = '190503_2'
    processed_data_path_pre = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess_pre + os.sep + 'sig_data.mat'
    loaded_data_pre = sio.loadmat(processed_data_path_pre)
    dF_signal_pre = loaded_data_pre['dF_data']    
    
    sess_post = '190508_9'
    processed_data_path_post = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess_post + os.sep + 'sig_data.mat'
    loaded_data_post = sio.loadmat(processed_data_path_post)
    dF_signal_post = loaded_data_post['dF_data']
    
    compare_rois(dF_signal_pre, dF_signal_post, rois_0000, sess_pre, sess_post, subfolder='Jimmy Summary', fname='summary')
    
if __name__ == '__main__':
    
    fformat = 'png'
    
    run_Jimmy_analysis()
#    run_Pumba_analysis()
#    trunk_analysis()