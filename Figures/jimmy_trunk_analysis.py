# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:27:54 2019

@author: Lou
"""

import sys, os, yaml
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f, Loader=yaml.FullLoader)
sys.path.append(loc_info['base_dir'] + '/Analysis')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
sns.set_style('white')


def id_and_plot_transients(dF_signal, rois, fs, baseline_percentile, subfolder=[], fname='transients'):
    fs=15.5
    num_rois = len(rois)
    axrows_per_roi = int(100/num_rois)
    roi_ax = []
    fig = plt.figure(figsize=(100,15))

    for i,r in enumerate(rois):
        if i > 0:
            roi_ax.append(plt.subplot2grid((100,100),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50,sharex=roi_ax[0]))
            plt.ylabel(str(r))
        else:
            roi_ax.append(plt.subplot2grid((100,100),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50))
            plt.title('Jimmy Trunk Transients')
            
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
    
    ax3 = plt.subplot2grid((100,100),(0,55), rowspan=100, colspan=50)
    # set standard deviation threshold above which traces have to be to be considered a transient
    std_threshold = 6

    # calculate frame to frame latency
    frame_latency = 1/fs
    tran_c_list = []

    for i,roi in enumerate(rois):
        tran_count = 0
        rois_unfiltered = dF_signal[:,roi]
        rois_filtered = rois_unfiltered

        # set minimum transient length in seconds that has to be above threshold
        min_transient_length_sec = 0.4

        # min transient length in number of frames
        min_transient_length = min_transient_length_sec/frame_latency
        
        # get standard deviation of lower 80% of samples
        percentile_low_idx = np.where(rois_filtered < np.percentile(rois_filtered,baseline_percentile))[0]
    
        rois_std = np.std(rois_filtered[percentile_low_idx])
        rois_mean = np.mean(rois_filtered[percentile_low_idx])
        roi_ax[i].axhspan(rois_mean, (std_threshold*rois_std)+rois_mean, color='0.9',zorder=0)

        # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,rois],c='r',lw=1)
        rois_idx = np.arange(0,len(dF_signal[:,roi]),1)
        roi_ax[i].plot(rois_idx, rois_filtered,label='dF/F',c='k',lw=1)
    
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
            if max_plot_val >= 0.1:
                tran_count += 1
            roi_ax[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
        mut_list = rois_unfiltered
        while mut_list[-1] == 0:
            mut_list = mut_list[:-1]
        tran_c_list.append(tran_count / len(mut_list))

    sess_modified = ['507: 1','507: 2','507: 3','508: 1','508: 2','508: 3','514','627: 1','627: 2','627: 3','709: 1','709: 2','715']
    ax3.plot(sess_modified,tran_c_list,c='r',lw=2.0)
    ax3.set_ylabel('Ratio of Transient df/F to Frame Length of Session' )
    ax3.set_xlabel('Trunk Date and Number')
    ax3.set_title('Transient Decrease')
    plt.tight_layout()
    
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)

def Jimmy_trunk_analysis():
    sigfile = 'sig_data_0000.mat'    
    MOUSE = 'Jimmy'
    sess_list = ['190507_3','190508_8','190514_10','190627_0','190709_0','190715_1']
    for sess in sess_list:
        processed_data_path = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess + os.sep + sigfile
        loaded_data = sio.loadmat(processed_data_path)['dF_data']
        if sess == sess_list[0]:
            dF_signal = loaded_data
        else:
            for data in range(len(loaded_data[0])):
                if len(dF_signal) > len(loaded_data[:,data]):
                    extend_data = np.append(loaded_data[:,data], [ 0 for i in range(len(dF_signal) - len(loaded_data[:,data]))])
                    dF_signal = np.column_stack((dF_signal, extend_data))
                elif len(dF_signal) <= len(loaded_data[:,data]):
                    while len(dF_signal) < len(loaded_data[:,data]):
                        zero_lis = [[ 0 for i in range(len(dF_signal[0]))]]
                        dF_signal = np.append(dF_signal, zero_lis, axis=0)
                    dF_signal = np.column_stack((dF_signal, loaded_data[:,data]))
                else:
                    print('Welp, didn\'t expect this.')
    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,7,8,9,10,12,13,14], fs=15.5, baseline_percentile=70, subfolder='Jimmy', fname='trunk_jim')
    
def Pumba_trunk_analysis():
    sigfile = 'sig_data_0000.mat'    
    MOUSE = 'Pumba'
    sess_list = ['190503_2','190508_9','190514_11','190627_3','190709_2','190715_3']
    for sess in sess_list:
        processed_data_path = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess + os.sep + sigfile
        loaded_data = sio.loadmat(processed_data_path)['dF_data']
        if sess == sess_list[0]:
            dF_signal = loaded_data
        else:
            for data in range(len(loaded_data[0])):
                if len(dF_signal) > len(loaded_data[:,data]):
                    extend_data = np.append(loaded_data[:,data], [ 0 for i in range(len(dF_signal) - len(loaded_data[:,data]))])
                    dF_signal = np.column_stack((dF_signal, extend_data))
                elif len(dF_signal) <= len(loaded_data[:,data]):
                    while len(dF_signal) < len(loaded_data[:,data]):
                        zero_lis = [[ 0 for i in range(len(dF_signal[0]))]]
                        dF_signal = np.append(dF_signal, zero_lis, axis=0)
                    dF_signal = np.column_stack((dF_signal, loaded_data[:,data]))
                else:
                    print('Welp, didn\'t expect this.')
    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,12], fs=15.5, baseline_percentile=70, subfolder='Pumba', fname='trunk_pum')
                    
if __name__ == '__main__':
    
    fformat = 'png'
    
    Jimmy_trunk_analysis()
#    Pumba_trunk_analysis()