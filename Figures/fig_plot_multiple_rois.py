"""
plot section of individual traces of individual, subcellular components

@author: Lukas Fischer

"""

import sys, os, yaml
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy.io as sio
from align_sig import calc_dF, process_and_align_sigfile
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

def plot_ind_trace_behavior(behav_ds, dF_ds, dF_ds_pair, soma_roi, pair_rois, t_start, t_stop, speed_col=3):

    # set up figure
    fig = plt.figure(figsize=(32,4))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax2.spines['left'].set_linewidth(2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    t_start_idx = (np.abs(behav_ds[:,0] - t_start)).argmin()
    t_stop_idx = (np.abs(behav_ds[:,0] - t_stop)).argmin()

    order = 6
    fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
    cutoff = 2 # desired cutoff frequency of the filter, Hz
    # plot dF trace
    pair_trace = np.zeros((dF_ds_pair[t_start_idx:t_stop_idx,0].shape[0], len(pair_rois)))
    filter_trace = 1
    if filter_trace == 1:
        soma_trace = butter_lowpass_filter(dF_ds[t_start_idx:t_stop_idx,soma_roi], cutoff, fs, order)
        for i,pr in enumerate(pair_rois):
            pair_trace[:,i] = butter_lowpass_filter(dF_ds_pair[t_start_idx:t_stop_idx,pr], cutoff, fs, order)
    else:
        soma_trace = dF_ds[t_start_idx:t_stop_idx,soma_roi]
        for i,pr in enumerate(pair_rois):
            pair_trace[:,i] = dF_ds_pair[t_start_idx:t_stop_idx,pr]

    ax1.plot(soma_trace,lw=0.2, c='b')
    ax1.plot(pair_trace,lw=0.2)

    ax2.plot(pair_trace,lw=0.2)
    ax2.plot(soma_trace,lw=0.2, c='b')
    # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,roi],c='r',lw=1)

    # filter and plot running speed trace
    speed_filtered = butter_lowpass_filter(behav_ds[:,speed_col], cutoff, fs, order)
    # ax2.plot(speed_filtered[t_start_idx:t_stop_idx],c='g',lw=2)

    # shade areas corresponding to the landmark
    landmark = [200,240]
    lm_temp = behav_ds[:,1]
    lm_start_idx = np.where(lm_temp > landmark[0])[0]
    lm_end_idx = np.where(lm_temp < landmark[1])[0]
    lm_idx = np.intersect1d(lm_start_idx,lm_end_idx)
    lm_diff = np.diff(lm_idx)
    lm_end = np.where(lm_diff>1)[0]
    lm_start = np.insert(lm_end,0,0)+1
    lm_end = np.append(lm_end,lm_idx.size-1)
    if lm_start.size > lm_end.size:
        lm_end.append(np.size(behav_ds),0)

    for i,lm in enumerate(lm_start):
        # print(behav_ds.shape)
        # print(lm_idx)
        if behav_ds[lm_idx[lm],4] != 5:
            if lm_idx[lm_start[i]] > t_start_idx and lm_idx[lm_start[i]] < t_stop_idx:
                if behav_ds[lm_idx[lm],4] == 3:
                    ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#45EDFF',alpha=0.2)
                    ax2.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#45EDFF',alpha=0.2)
                else:
                    ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#F900FF',alpha=0.2)
                    ax2.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='#F900FF',alpha=0.2)

     # plot time where animal was in blackbox
    bb_temp = behav_ds[:,4]
    bb_temp[np.where(behav_ds[:,4]!=5)[0]] = 0
    bb_diff = np.diff(bb_temp)
    bb_start = np.where(bb_diff>1)[0]
    bb_end = np.where(bb_diff<0)[0]
    if bb_start.size > bb_end.size:
        np.append(bb_end, np.size(behav_ds,0)-1)

    for i,bb in enumerate(bb_start):
        if bb_start[i] > t_start_idx and bb_start[i] < t_stop_idx:
            ax1.axvspan(bb_start[i],bb_end[i],color='0.85')
            ax2.axvspan(bb_start[i],bb_end[i],color='0.85')


    ax1.set_xlim([0,t_stop_idx-t_start_idx])
    ax2.set_xlim([0,t_stop_idx-t_start_idx])

    one_sec = (t_stop_idx-t_start_idx)/(t_stop - t_start)
    ax1.set_xticks([0,5*one_sec])
    ax1.set_xticklabels(['0','5'])

    ax1.set_yticks([0,2,4,6])
    ax1.set_yticklabels(['0','2','4','6'])
    ax1.set_ylabel('dF/F', fontsize=16)

    # ax2.set_yticks([0,10,20,30,40])
    # ax2.set_yticklabels(['0','10','20','30','40'])
    # ax2.set_ylabel('speed (cm/sec)', fontsize=16)

    ax1.set_ylim([-1,15])
    ax2.set_ylim([-1,15])
    # ax2.set_ylim([-5,40])

    fname = 'subcell_trace'
    subfolder = []
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat

    fig.savefig(fname, format=fformat,dpi=300)

    print(fname)
    print('done')

def id_and_plot_transients(dF_signal, rois, fs, subfolder=[], fname='transients', baseline_percentile=70):
    
    num_rois = len(rois)
    axrows_per_roi = int(100/num_rois)
    roi_ax = []
    
    fig = plt.figure(figsize=(30,15))
    for i,r in enumerate(rois):
        if i > 0:
            roi_ax.append(plt.subplot2grid((100,100),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50,sharex=roi_ax[0]))
        else:
            roi_ax.append(plt.subplot2grid((100,100),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50))
        
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
        
       
    ax3 = plt.subplot2grid((100,100),(0,50), rowspan=50, colspan=25)
    ax4 = plt.subplot2grid((100,100),(0,75), rowspan=50, colspan=25)
    ax5 = plt.subplot2grid((100,100),(50,50), rowspan=50, colspan=25)
    ax6 = plt.subplot2grid((100,100),(50,75), rowspan=50, colspan=25)

    
    # set standard deviation threshold above which traces have to be to be considered a transient
    std_threshold = 4

    # calculate frame to frame latency
    frame_latency = 1/fs
    
    # low pass filter trace
    order = 6
    cutoff = 10 # desired cutoff frequency of the filter, Hz
    
    for i,roi in enumerate(rois):
#        rois_filtered = butter_lowpass_filter(dF_signal[:,roi], cutoff, fs, order)
        rois_unfiltered = dF_signal[:,roi]
        rois_filtered = rois_unfiltered

        # set minimum transient length in seconds that has to be above threshold
        min_transient_length_sec = 0.2

        # min transient length in number of frames
        min_transient_length = min_transient_length_sec/frame_latency
        
        # get standard deviation of lower 80% of samples
        percentile_low_idx = np.where(rois_filtered < np.percentile(rois_filtered,baseline_percentile))[0]
    
        rois_std = np.std(rois_filtered[percentile_low_idx])
        rois_mean = np.mean(rois_filtered[percentile_low_idx])
#        roi_ax[i].axhline((std_threshold*rois_std)+rois_mean,ls='--',c='0.8',lw=2)
#        roi_ax[i].axhline(rois_mean,ls='--',c='g',lw=2,alpha=0.5)
        roi_ax[i].axhspan(rois_mean, (std_threshold*rois_std)+rois_mean, color='0.9',zorder=0)

        # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,rois],c='r',lw=1)
        rois_idx = np.arange(0,len(dF_signal[:,roi]),1)
        roi_ax[i].plot(rois_idx, rois_filtered,label='dF/F',c='k',lw=1)
    
        # get indeces above speed threshold
        transient_high = np.where(rois_filtered > (std_threshold*rois_std)+rois_mean)[0]
        
        if transient_high.size != 0:
#            # fig.suptitle(fname, wrap=True)
#            if subfolder != []:
#                if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
#                    os.mkdir(loc_info['figure_output_path'] + subfolder)
#                fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
#            else:
#                fname = loc_info['figure_output_path'] + fname + '.' + fformat
#            print(fname)
#            fig.savefig(fname, format=fformat,dpi=150)
#            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        # ax1.plot(rois_idx[transient_high], rois_filtered[transient_high],c='r',lw=1)
    
#        roi_ax[i].plot(rois_idx[percentile_low_idx], rois_filtered[percentile_low_idx],c='g',lw=1)
    
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
            above_1_std = np.where(rois_filtered > (rois_std)+rois_mean)[0]
            one_std_idx_diff = np.diff(above_1_std)
            one_std_idx_diff = np.insert(one_std_idx_diff,0,one_std_idx_diff[0])
        
            one_std_onset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]]-1
            one_std_offset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]-1]+1
    
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
                # one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
                onsets.append(oi-one_std_idx)
                # ax1.axvline(oi-one_std_idx, c='0.6', lw=0.5, ls='--')
        
            for oi in offset_idx:
                closest_idx = one_std_offset_idx - oi
                closest_idx_neg = np.where(closest_idx > 0)[0]
                if closest_idx_neg.size == 0:
                    closest_idx_neg = [-1]
                    one_std_idx = 0
                else:
                    one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
                offsets.append(oi+one_std_idx)
                # ax1.axvline(oi+one_std_idx, c='0.6', lw=0.5, ls='--')
            
            # find max transient length
            max_transient_length = 0
            for j in range(len(onsets)):
                if offsets[j]-onsets[j] > max_transient_length:
                    max_transient_length = offsets[j]-onsets[j]
        
            all_transients = np.full((len(onsets),max_transient_length), np.nan)
            all_transients_norm = np.full((len(onsets),max_transient_length), np.nan)
            for j in range(len(onsets)):
                all_transients[j,0:offsets[j]-onsets[j]] = rois_unfiltered[onsets[j]:offsets[j]]
                all_transients_norm[j,0:offsets[j]-onsets[j]] = (rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]])/(np.nanmax(rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]]))
                # print((rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]])/(np.nanmax(rois_unfiltered[onsets[j]:offsets[j]]-rois_unfiltered[onsets[j]])))
        
            for j in range(len(onsets)):
                roi_ax[i].plot(np.arange(onsets[j],offsets[j],1),rois_filtered[onsets[j]:offsets[j]],c='r',lw=1.5)
        
            for j in range(all_transients.shape[0]):
                ax3.plot(all_transients[j],c='0.8',lw=0.5)
                ax4.plot(all_transients_norm[j],c='0.8',lw=0.5)
        
            ax3.plot(np.nanmean(all_transients,0),c='k', lw=2)
            ax4.plot(np.nanmean(all_transients_norm,0),c='k', lw=1)
    plt.tight_layout()
    
    
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)
    
     # if we want to plot interactive, only take a random subsample of datapoints, otherwise the display gets really slow
     
    
#    ax5.scatter(pil_raw_F,rio_raw_F)
#
#    max_xy_val = np.amax([pil_raw_F,rio_raw_F])
#    ax5.set_xlim([0,max_xy_val])
#    ax5.set_ylim([0,max_xy_val])
    # ax5.set_xlabel('neuropil values')
    # ax5.set_ylabel('rois values')

def run_LF170613_1():
    MOUSE= 'LF170613_1'
    sess = '20170804'
    sigfile = 'M01_000_004_rigid.sig'
    meta_file = 'M01_000_004_rigid.bri'
    behavior_file = 'MTH3_vr1_20170804_1708.csv'
    data_path = loc_info['raw_dir'] + MOUSE
    
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'aligned_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    behaviour_aligned = loaded_data['behaviour_aligned']
    dF_aligned = loaded_data['dF_aligned']
    
    # dF_signal = calc_dF(data_path, sess, sigfile, meta_file, sbx_version=1, session_crop=[0,1], method=2, roiss=[0,1,2,3])
    # dF_aligned, behaviour_aligned, bri_aligned = process_and_align_sigfile(data_path, sess, sigfile, behavior_file, meta_file, sbx_version=1, session_crop=[0,1], method=2, roiss=[0,1,2,3])
        
#    plot_ind_trace_behavior(behaviour_aligned, dF_aligned, dF_aligned, 0, [1,2,3], 0, 1000)
    
    id_and_plot_transients(dF_aligned, 0, fs=15.5, subfolder=[], baseline_percentile=70)
    
#    plt.figure()
#    plt.plot(dF_aligned[:,0])
#    plt.plot(dF_aligned[:,1])
#    plt.plot(dF_aligned[:,2])
#    plt.show()

def run_LF190409_1():
    MOUSE= 'LF190409_1'
    sess = '190514_1'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    data_path = loc_info['raw_dir'] + MOUSE
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    dF_signal = loaded_data['dF_data']

    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10], fs=15.5, subfolder='LF190409_1 transients', fname='transients_0', baseline_percentile=70)
    id_and_plot_transients(dF_signal, [0,11,12,13,14,15,16,17,18,19,20], fs=15.5, subfolder='LF190409_1 transients', fname='transients_1', baseline_percentile=70)
    id_and_plot_transients(dF_signal, [0,21,22,23,24,25,26,27,28,29,30], fs=15.5, subfolder='LF190409_1 transients', fname='transients_2', baseline_percentile=70)
#    id_and_plot_transients(dF_signal, [0,31,32,33,34,35,36,37,38,39,40], fs=15.5, subfolder='LF190409_1 transients', fname='transients_3', baseline_percentile=70)
#    id_and_plot_transients(dF_signal, [0,41,42,43,44,45,46,47,48,49,50], fs=15.5, subfolder='LF190409_1 transients', fname='transients_4', baseline_percentile=70)
#    id_and_plot_transients(dF_signal, [0,51,52,53,54,55,56], fs=15.5, subfolder='LF190409_1 transients', fname='transients_5', baseline_percentile=70)
    
def run_Jimmy():
    MOUSE= 'Jimmy'
    sess = '20190719_004'
#    sigfile = 'M01_000_000.sig'
#    meta_file = 'M01_000_000.extra'
#    data_path = loc_info['raw_dir'] + MOUSE
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    dF_signal = loaded_data['dF_data']

    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10,12], fs=15.5, subfolder='Jimmy spine transients', fname='transients_0', baseline_percentile=70)

def run_LF190716_1():
    MOUSE= 'LF190716_1'
#    sess = '20190722_003'
#    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
#    loaded_data = sio.loadmat(processed_data_path)
#    dF_signal = loaded_data['dF_data']
#    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10,12], fs=31, subfolder='Jimmy spine transients', fname='transients_0', baseline_percentile=70)

    sess = '20190722_002'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    dF_signal = loaded_data['dF_data']

    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10,12,13], fs=31, subfolder='Jimmy spine transients', fname='transients_0', baseline_percentile=50)

def run_Buddha_0802():
    MOUSE= 'Buddha'
    sess = '190802_2'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    dF_signal = loaded_data['dF_data']
    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10], fs=15.5, subfolder='Buddha transients', fname='transients_0', baseline_percentile=50)
#    id_and_plot_transients(dF_signal, [10,11,12,13,14,15,16,17,18,19,20], fs=15.5, subfolder='Buddha transients', fname='transients_1', baseline_percentile=50)
#    id_and_plot_transients(dF_signal, [20,21,22,23,24,25,26,27,28,29,30], fs=15.5, subfolder='Buddha transients', fname='transients_2', baseline_percentile=50)
#    id_and_plot_transients(dF_signal, [30,31,32,33,34,35], fs=15.5, subfolder='Buddha transients', fname='transients_3', baseline_percentile=50)

if __name__ == '__main__':
    
    fformat = 'png'

#    run_LF170613_1()
#    run_LF190409_1()
#    run_Jimmy()
#    run_LF190716_1()
    run_Buddha_0802()

   