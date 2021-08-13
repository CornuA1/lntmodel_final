"""
landmark-aligned neurons

@author: lukasfischer

"""

import numpy as np
import scipy as sp
import statsmodels.api as sm
import h5py
import os
import sys
import traceback
import matplotlib
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import json
import yaml
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

def roi_response_validation(roi_params, tl, el, roi_idx_num):
    """
    separate function to check whether a given response passes criterion for being considered a real roi_response_validation

    """

    MIN_TRIALS_ACTIVE = 0.25
    MIN_DF = 0.1

    roi_activity = el + '_active_' + tl
    roi_peak_val = el + '_peak_' + tl
    if roi_params[roi_activity][roi_idx_num] > MIN_TRIALS_ACTIVE and roi_params[roi_peak_val][roi_idx_num] > MIN_DF:
        return True
    else:
        return False

def get_eventaligned_rois(roi_param_list, trialtypes, align_event):
    # hold values of mean peak
    result_max_peak = {}
    result_max_peak_ol = {}
    # for each landmark neuron, store the max peak of the respective opposite landmark
    result_aligned_roi_num = {}
    # set up empty dicts so we can later append to them
    for rpl in roi_param_list:
        for tl in trialtypes:
            result_max_peak[align_event + '_peakval_' + tl + '_' + rpl.split(os.sep)[7]] = []
            result_max_peak_ol[align_event + '_peakval_' + tl + '_' +  rpl.split(os.sep)[7]] = []
            result_aligned_roi_num[align_event + '_roi_number_' + tl +'_' +  rpl.split(os.sep)[7]] = []

    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        # load roi parameters for given session
        with open(rpl + '/roi_params.json','r') as f:
            roi_params = json.load(f)
        # grab a full list of roi numbers
        roi_list_all = roi_params[align_event + '_roi_number']
        # loop through every roi
        for j,r in enumerate(roi_list_all):
            # loop through every trialtype and alignment point to determine largest response
            for tl in trialtypes:
                max_peak = 0
                max_peak_ol = 0
                roi_num = -1
                valid = 0
                for el in event_list:
                    value_key = el + '_peak_' + tl
                    value_key_ol = el + '_peak_' + tl + '_ol'
                    if roi_params[value_key][j] > max_peak and roi_response_validation(roi_params, tl, el, j):
                        valid = 1
                        max_peak = roi_params[value_key][j]
                        max_peak_ol = roi_params[value_key_ol][j]
                        peak_event = el
                        peak_trialtype = tl
                        roi_num = j
                if valid == 1:
                    if peak_event == align_event and roi_num != -1:
                        if peak_trialtype == 'short':
                            result_aligned_roi_num[align_event + '_roi_number_' + peak_trialtype + '_' + rpl.split(os.sep)[7]].append(roi_num)
                            result_max_peak[align_event + '_peakval_' + peak_trialtype + '_' + rpl.split(os.sep)[7]].append(max_peak)
                            result_max_peak_ol[align_event + '_peakval_' + peak_trialtype + '_' +  rpl.split(os.sep)[7]].append(max_peak_ol)

                        elif peak_trialtype == 'long':
                            result_aligned_roi_num[align_event + '_roi_number_' + peak_trialtype + '_' + rpl.split(os.sep)[7]].append(roi_num)
                            result_max_peak[align_event + '_peakval_' + peak_trialtype + '_' + rpl.split(os.sep)[7]].append(max_peak)
                            result_max_peak_ol[align_event + '_peakval_' + peak_trialtype + '_' +  rpl.split(os.sep)[7]].append(max_peak_ol)

    return result_aligned_roi_num, result_max_peak, result_max_peak_ol

def matching_space_aligned_rois(roi_param_list, roilist, trialtypes, align_event):
    """ retrieve the space-aligned values from neurons in roilist """
    # run through all roi_param files and create empty dictionary lists that we can later append to
    result_aligned_roi_num = {}
    result_max_peak = {}
    result_max_peak_ol = {}
    result_mean_trace = {}
    result_mean_trace_ol = {}
    for i,rpl in enumerate(roi_param_list):
        for tl in trialtypes:
            result_max_peak[align_event + '_peakval_' + tl + '_' + rpl.split('/')[7]] = []
            result_max_peak_ol[align_event + '_peakval_' + tl + '_' + rpl.split('/')[7]] = []
            result_aligned_roi_num[align_event + '_roi_number_' + tl + '_' + rpl.split('/')[7]] = []
            result_mean_trace[align_event + '_mean_trace_' + tl + '_' + rpl.split('/')[7]] = []
            result_mean_trace_ol[align_event + '_mean_trace_' + tl + '_' + rpl.split('/')[7]] = []
            result_mean_trace[align_event + '_mean_trace_start_' + tl + '_' + rpl.split('/')[7]] = []
            result_mean_trace_ol[align_event + '_mean_trace_start_' + tl + '_' + rpl.split('/')[7]] = []

    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        # load roi parameters for given session
        with open(rpl + '/roi_params_space.json','r') as f:
            roi_params = json.load(f)
        # grab a full list of roi numbers
        roi_list_all_space = roi_params['space_roi_number']
        for tl in trialtypes:
            for roi in roilist[align_event + '_roi_number_' + tl + '_' + rpl.split('/')[7]]:
                space_roi_idx = np.argwhere(np.asarray(roi_list_all_space) == roi)[0][0]
                result_aligned_roi_num[align_event + '_roi_number_' + tl + '_' + rpl.split('/')[7]].append(space_roi_idx)
                result_max_peak[align_event + '_peakval_' + tl + '_' + rpl.split('/')[7]].append(roi_params['space_peak_' + tl][space_roi_idx])
                result_max_peak_ol[align_event + '_peakval_' + tl + '_' + rpl.split('/')[7]].append(roi_params['space_peak_' + tl + '_ol'][space_roi_idx])
                result_mean_trace[align_event + '_mean_trace_' + tl + '_' + rpl.split('/')[7]].append(roi_params['space_mean_trace_' + tl][space_roi_idx])
                result_mean_trace_ol[align_event + '_mean_trace_' + tl + '_' + rpl.split('/')[7]].append(roi_params['space_mean_trace_' + tl + '_ol'][space_roi_idx])
                result_mean_trace[align_event + '_mean_trace_start_' + tl + '_' + rpl.split('/')[7]].append(roi_params['space_mean_trace_start_' + tl][space_roi_idx])
                result_mean_trace_ol[align_event + '_mean_trace_start_' + tl + '_' + rpl.split('/')[7]].append(roi_params['space_mean_trace_start_' + tl + '_ol'][space_roi_idx])

    return result_aligned_roi_num, result_max_peak, result_max_peak_ol, result_mean_trace, result_mean_trace_ol

def make_roi_heatmap(roi_param_list, trialtypes, result_mean_trace, result_mean_trace_ol, ax_object3, ax_object4, ax_object5, ax_object6):
    binnr_short = 80
    binnr_long = 100

    # heatmap data
    mean_trace_short_heatmap = np.zeros((0,binnr_short))
    mean_trace_long_heatmap = np.zeros((0,binnr_long))
    mean_trace_short_heatmap_ol = np.zeros((0,binnr_short))
    mean_trace_long_heatmap_ol = np.zeros((0,binnr_long))

    mt_short_norm_val_short = []
    mt_short_norm_val_long = []


    # run through all roi_param files
    for rpl in roi_param_list:
        for tl in trialtypes:
            mean_trace_key = 'lmcenter_mean_trace_' + tl + '_' + rpl.split('/')[7]
            trace_start_idx = 'lmcenter_mean_trace_start_' + tl + '_' + rpl.split('/')[7]
            if tl == 'short':
                # print(len(result_mean_trace[mean_trace_key]))
                for i in range(len(result_mean_trace[mean_trace_key])):
                    # pull out mean trace
                    mt = np.asarray(result_mean_trace[mean_trace_key][i])
                    mt_short_norm_val_short.append(np.amax(mt))
                    # normalize to max value
                    mt = mt / np.amax(mt)
                    # pull out where the mean trace starts
                    mt_start = result_mean_trace[trace_start_idx][i]
                    # add trace to 0-padded matrix for plotting
                    mt_padded = np.zeros((binnr_short,1))
                    mt_padded[mt_start:(len(mt)+mt_start),0] = mt
                    # append to heatmap matrix
                    mean_trace_short_heatmap = np.vstack((mean_trace_short_heatmap, mt_padded.T))

                    mt = np.asarray(result_mean_trace_ol[mean_trace_key][i])
                    # mt = mt / mt_short_norm_val_short[i]
                    mt = mt / np.amax(mt)
                    mt_start = result_mean_trace_ol[trace_start_idx][i]
                    mt_padded = np.zeros((binnr_short,1))
                    mt_padded[mt_start:(len(mt)+mt_start),0] = mt
                    mean_trace_short_heatmap_ol = np.vstack((mean_trace_short_heatmap_ol, mt_padded.T))

            elif tl == 'long':
                for i in range(len(result_mean_trace[mean_trace_key])):
                    mt = np.asarray(result_mean_trace[mean_trace_key][i])
                    mt_short_norm_val_long.append(np.amax(mt))
                    mt = mt / np.amax(mt)
                    mt_start = result_mean_trace[trace_start_idx][i]
                    mt_padded = np.zeros((binnr_long,1))
                    mt_padded[mt_start:(len(mt)+mt_start),0] = mt
                    mean_trace_long_heatmap = np.vstack((mean_trace_long_heatmap, mt_padded.T))

                    # np.asarray(result_mean_trace_ol[mean_trace_key]):
                    mt = np.asarray(result_mean_trace_ol[mean_trace_key][i])
                    # mt = mt / mt_short_norm_val_long[i]
                    mt = mt / np.amax(mt)
                    mt_start = result_mean_trace_ol[trace_start_idx][i]
                    mt_padded = np.zeros((binnr_long,1))
                    mt_padded[mt_start:(len(mt)+mt_start),0] = mt
                    mean_trace_long_heatmap_ol = np.vstack((mean_trace_long_heatmap_ol, mt_padded.T))



    # sort by peak activity
    mean_dF_sort_short = np.zeros(mean_trace_short_heatmap.shape[0])
    for i, row in enumerate(mean_trace_short_heatmap):
        mean_dF_sort_short[i] = np.argmax(row)
    sort_ind_short = np.argsort(mean_dF_sort_short)

    # sort by peak activity
    mean_dF_sort_long = np.zeros(mean_trace_long_heatmap.shape[0])
    for i, row in enumerate(mean_trace_long_heatmap):
        mean_dF_sort_long[i] = np.argmax(row)
    sort_ind_long = np.argsort(mean_dF_sort_long)

    sns.heatmap(mean_trace_short_heatmap[sort_ind_short,:],vmin=0,vmax=1,cbar=False,cmap='viridis',xticklabels=True,ax=ax_object3)
    sns.heatmap(mean_trace_long_heatmap[sort_ind_long,:],vmin=0,vmax=1,cbar=False,cmap='viridis',xticklabels=True,ax=ax_object4)

    sns.heatmap(mean_trace_short_heatmap_ol[sort_ind_short,:],vmin=0,vmax=1,cbar=False,cmap='viridis',xticklabels=True,ax=ax_object5)
    sns.heatmap(mean_trace_long_heatmap_ol[sort_ind_long,:],vmin=0,vmax=1,cbar=False,cmap='viridis',xticklabels=True,ax=ax_object6)

    ax_object3.set_xlim([21,65])
    ax_object4.set_xlim([22,78])
    ax_object5.set_xlim([21,65])
    ax_object6.set_xlim([22,78])

    ax_object3.axvline(40,c='0.8',ls='--',lw=3)
    ax_object3.axvline(48,c='0.8',ls='--',lw=3)
    ax_object3.axvline(64,c='r',ls='--',lw=3)
    ax_object4.axvline(40,c='0.8',ls='--',lw=3)
    ax_object4.axvline(48,c='0.8',ls='--',lw=3)
    ax_object4.axvline(76,c='r',ls='--',lw=3)
    ax_object5.axvline(40,c='0.8',ls='--',lw=3)
    ax_object5.axvline(48,c='0.8',ls='--',lw=3)
    ax_object5.axvline(64,c='r',ls='--',lw=3)
    ax_object6.axvline(40,c='0.8',ls='--',lw=3)
    ax_object6.axvline(48,c='0.8',ls='--',lw=3)
    ax_object6.axvline(76,c='r',ls='--',lw=3)

    ax_object3.set_yticks([0,256])
    ax_object3.set_yticklabels(['0','256'])
    ax_object3.set_ylabel('Cell #', fontsize=24)
    ax_object3.set_xticks([40,44,48,64])
    ax_object3.set_xticklabels(['','landmark','','reward'], rotation = 0, fontsize=24)
    ax_object5.set_yticks([0,256])
    ax_object5.set_yticklabels(['0','256'])
    ax_object5.set_ylabel('Cell #', fontsize=24)
    ax_object5.set_xticks([40,44,48,64])
    ax_object5.set_xticklabels(['','landmark','','reward'], rotation = 0, fontsize=24)

    ax_object4.set_yticks([0,256])
    ax_object4.set_yticklabels(['0','256'])
    ax_object4.set_ylabel('Cell #', fontsize=24)
    ax_object4.set_xticks([40,44,48,76])
    ax_object4.set_xticklabels(['','landmark','','reward'], rotation = 0, fontsize=24)
    ax_object6.set_yticks([0,256])
    ax_object6.set_yticklabels(['0','256'])
    ax_object6.set_ylabel('Cell #', fontsize=24)
    ax_object6.set_xticks([40,44,48,76])
    ax_object6.set_xticklabels(['','landmark','','reward'], rotation = 0, fontsize=24)

    ax_object4.set_yticks([])
    ax_object4.set_yticklabels([])
    ax_object4.set_ylabel('')
    ax_object6.set_yticks([])
    ax_object6.set_yticklabels([])
    ax_object6.set_ylabel('')


def plot_lmcenter_population(roi_param_list, trialtypes, ax_object1, ax_object2, ax_object3, ax_object4, ax_object5, ax_object6):
    """ plot all lmcenter-aligned neurons as a function of time relative to alignment point """

    roilist, max_peak, max_peak_ol = get_eventaligned_rois(roi_param_list, trialtypes, 'lmcenter')
    roilist_space, max_peak_space, max_peak_ol_space, result_mean_trace, result_mean_trace_ol = matching_space_aligned_rois(roi_param_list, roilist, trialtypes, 'lmcenter')
    make_roi_heatmap(roi_param_list, trialtypes, result_mean_trace, result_mean_trace_ol, ax_object3, ax_object4, ax_object5, ax_object6)

    for rpl in roi_param_list:
        cur_sess = rpl.split('/')[7]
        # print(roilist)
        ax_object1.scatter(max_peak_ol['lmcenter_peakval_short_' + cur_sess], max_peak['lmcenter_peakval_short_' + cur_sess], linewidths = 0, c='0.8', zorder=3)
        ax_object1.scatter(max_peak_ol['lmcenter_peakval_long_' + cur_sess], max_peak['lmcenter_peakval_long_' + cur_sess], linewidths = 0, c='0.5', zorder=3)

        ax_object2.scatter(max_peak['lmcenter_peakval_short_' + cur_sess],max_peak_space['lmcenter_peakval_short_' + cur_sess], linewidths = 0, c='0.8', zorder=3)
        ax_object2.scatter(max_peak['lmcenter_peakval_long_' + cur_sess],max_peak_space['lmcenter_peakval_long_' + cur_sess], linewidths = 0, c='0.5', zorder=3)

    ax_object1.plot([0,3],[0,3], c='k', ls='--', zorder=2)
    ax_object1.set_xlim([0,3])
    ax_object1.set_ylim([0,3])

    ax_object1.set_xlabel('peak response PASSIVE (dF/F)', fontsize=22)
    ax_object1.set_ylabel('peak response ACTIVE (dF/F)', fontsize=22)

    ax_object2.set_xlabel('peak response TIME (dF/F)', fontsize=22)
    ax_object2.set_ylabel('peak response SPACE (dF/F)', fontsize=22)
    ax_object2.plot([0,3],[0,3], c='k', ls='--', zorder=2)
    ax_object2.set_xlim([0,3])
    ax_object2.set_ylim([0,3])

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline
    fformat = 'png'

    # list of roi parameter files
    roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_1',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_2',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_3',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day2017331',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017719',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017720',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day201783',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day2017719',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170222_1_Day201776',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170613_1_Day201784'
                     ]

    fname = 'lmcenter figure ol normalized'
    event_list = ['trialonset','lmcenter','reward']
    trialtypes = ['short', 'long']
    subfolder = []

    # create figure and axes to later plot on
    fig = plt.figure(figsize=(12,16))
    ax1 = plt.subplot2grid((3,4),(0,0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid((3,4),(0,2), rowspan=1, colspan=2)
    # ax3 = plt.subplot2grid((2,4),(0,1), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((3,4),(1,0), rowspan=1, colspan=2)
    ax5 = plt.subplot2grid((3,4),(1,2), rowspan=1, colspan=2)
    ax6 = plt.subplot2grid((3,4),(2,0), rowspan=1, colspan=2)
    ax7 = plt.subplot2grid((3,4),(2,2), rowspan=1, colspan=2)

    # event_maxresponse(roi_param_list, event_list, trialtypes, ax1, ax3, ax4, ax5, ax2)
    plot_lmcenter_population(roi_param_list, trialtypes, ax1, ax2, ax4, ax5, ax6, ax7)

    fformat = 'png'
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
