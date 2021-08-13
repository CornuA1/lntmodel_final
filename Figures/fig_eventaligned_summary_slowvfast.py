"""
scatterplot of ROI amplitudes in VR and openloop.

v 2.0 is written to accomodate new .json dictionary structure where all results of a session are in one file

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
plt.rcParams['svg.fonttype'] = 'none'
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

    roi_activity = el + speed + '_active_' + tl
    roi_peak_val = el + speed + '_peak_' + tl
    if roi_params[roi_activity][roi_idx_num] > MIN_TRIALS_ACTIVE and roi_params[roi_peak_val][roi_idx_num] > MIN_DF:
        return True
    else:
        return False


def event_maxresponse(roi_param_list, event_list, trialtypes, ax_object, ax_object2, ax_object5, normalize=False):
    """
    determine which alignment gave the max mean response

    roi_param_list : list of roi param files (fulle file paths)

    event_list : list of alignments available

    """
    # when we normalize, values range from 0-1, when we don't, we want to accommodate slightly below 0 values
    if normalize:
        min_xy = 0
    else:
        min_xy = -0.2

    # dictionary to hold counters of peak response per alignment point
    result_counter = {}
    for i in range(len(roi_param_list)):
        for tl in trialtypes:
            result_counter['roicounter_' + tl + str(i)] = 0
            result_counter['roinumlist_' + tl + str(i)] = []
            for el in event_list:
                result_counter[el + '_peakcounter_' + tl + str(i)] = 0


    # hold values of mean peak
    result_max_peak = {}
    result_max_peak_ol = {}
    # hold values of peak aligned to trial onset vs. actual alignment
    result_trialonset_max_peak = {}
    result_lmcenter_max_peak = {}
    # for each landmark neuron, store the max peak of the respective opposite landmark
    result_matching_landmark_long = {}
    result_matching_landmark_short = {}
    # set up empty dicts so we can later append to them
    for el in event_list:
        for tl in trialtypes:
            result_max_peak[el + '_peakval_' + tl] = []
            result_max_peak_ol[el + '_peakval_' + tl] = []
            result_trialonset_max_peak[el + '_peakval_' + tl] = []
            result_lmcenter_max_peak[el + '_peakval_' + tl] = []
            result_matching_landmark_long[el + '_peakval_' + tl] = []
            result_matching_landmark_short[el + '_peakval_' + tl] = []

    # differences in peak amplitude between landmarks
    result_matching_landmark_normdiff = {
        'lm_diff_short' : [],
        'lm_diff_long' : []
    }

    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        print(rpl)
        # load roi parameters for given session
        with open(rpl,'r') as f:
            roi_params = json.load(f)

        # grab a full list of roi numbers (it doesn't matter which event is chosen here)
        roi_list_all = roi_params['valid_rois']

        # loop through every roi
        for j,r in enumerate(roi_list_all):
            if normalize:
                roi_normalization_value = roi_params['norm_value'][r]
            else:
                roi_normalization_value = 1
            # loop through every trialtype and alignment point to determine largest response
            for tl in trialtypes:
                valid = 0
                max_peak = 0
                max_peak_ol = 0
                peak_trialonset = 0
                peak_trialonset_activity = 0
                peak_lm_long = 0
                peak_lm_short = 0
                peak_lm_time_short = 0
                peak_lm_time_long = 0
                peak_lm_normdiff = 0
                peak_lm_type = ''
                peak_event = ''
                peak_trialtype = ''
                for el in event_list:
                    value_key = el + speed + '_peak_' + tl
                    value_key_ol = el + speed + '_peak_' + tl + '_ol'
                    if (roi_params[value_key][j]/roi_normalization_value) > max_peak and roi_response_validation(roi_params, tl, el, j):
                        valid = 1
                        max_peak = roi_params[value_key][j]/roi_normalization_value
                        # max_peak_ol = roi_params[value_key_ol][j]/roi_normalization_value
                        # grab peak value for trialonset condition
                        if roi_response_validation(roi_params, tl, 'trialonset', j):
                            peak_trialonset = roi_params['trialonset' + speed + '_peak_'+tl][j]/roi_normalization_value
                        else:
                            peak_trialonset = 0
                        # grab peak value for lmcenter condition
                        if roi_response_validation(roi_params, tl, 'lmcenter', j):
                            peak_lmcenter = roi_params['lmcenter' + speed + '_peak_'+tl][j]/roi_normalization_value
                        else:
                            peak_lmcenter = 0
                        # if the peak response is by the landmark, check which trialtype elicits the larger response and store the normalized difference
                        if el == 'lmcenter':
                            if tl == 'short' and max_peak > roi_params[el + speed + '_peak_long'][j]:
                                peak_lm_long = roi_params['lmcenter' + speed + '_peak_long'][j]
                                peak_lm_normdiff = (roi_params[el + speed +'_peak_long'][j]/max_peak)-1
                                peak_lm_type = 'short'
                            elif tl == 'long' and max_peak > roi_params[el + speed + '_peak_short'][j]:
                                peak_lm_short = roi_params['lmcenter' + speed + '_peak_short'][j]
                                peak_lm_normdiff = ((roi_params[el + speed + '_peak_short'][j]/max_peak)-1)*-1
                                peak_lm_type = 'long'
                        peak_event = el
                        peak_trialtype = tl
                if valid == 1:
                    # add 1/total number of rois to get fraction
                    result_counter[peak_event + '_peakcounter_' + peak_trialtype + str(i)] = result_counter[peak_event + '_peakcounter_' + peak_trialtype + str(i)] + (1/len(roi_list_all))
                    result_counter['roicounter_' + peak_trialtype + str(i)] = result_counter['roicounter_' + peak_trialtype + str(i)] + 1
                    result_counter['roinumlist_' + peak_trialtype + str(i)].append(r)
                    result_max_peak[peak_event + '_peakval_' + peak_trialtype].append(max_peak)
                    # result_max_peak_ol[peak_event + '_peakval_' + peak_trialtype].append(max_peak_ol)
                    # print(peak_trialonset, peak_lmcenter)
                    result_trialonset_max_peak[peak_event + '_peakval_' + peak_trialtype].append(peak_trialonset)
                    result_lmcenter_max_peak[peak_event + '_peakval_' + peak_trialtype].append(peak_lmcenter)
                    # print(np.asarray(result_trialonset_max_peak['trialonset_peakval_short']).shape, np.asarray(result_lmcenter_max_peak['trialonset_peakval_short']).shape)
                    # result_matching_landmark_normdiff
                    if peak_event == 'lmcenter':
                        if peak_trialtype == 'short':
                            result_matching_landmark_long['lmcenter_peakval_' + peak_trialtype].append(peak_lm_long)
                        elif peak_trialtype == 'long':
                            result_matching_landmark_short['lmcenter_peakval_' + peak_trialtype].append(peak_lm_short)
                    if peak_event == 'lmcenter':
                        if peak_lm_type == 'short':
                            result_matching_landmark_normdiff['lm_diff_short'].append(peak_lm_normdiff)
                        elif peak_lm_type == 'long':
                            result_matching_landmark_normdiff['lm_diff_long'].append(peak_lm_normdiff)

    # accumulate all the data and plot
    num_trialonset_short = []
    num_trialonset_long = []
    num_reward_short = []
    num_reward_long = []
    num_lmcenter_short = []
    num_lmcenter_long = []

    tot_num_rois_short = []
    tot_num_rois_long = []
    tot_num_rois_all = []

    for i in range(len(roi_param_list)):
        num_trialonset_short.append(result_counter['trialonset_peakcounter_short' + str(i)])
        num_trialonset_long.append(result_counter['trialonset_peakcounter_long' + str(i)])
        num_reward_short.append(result_counter['reward_peakcounter_short' + str(i)])
        num_reward_long.append(result_counter['reward_peakcounter_long' + str(i)])
        num_lmcenter_short.append(result_counter['lmcenter_peakcounter_short' + str(i)])
        num_lmcenter_long.append(result_counter['lmcenter_peakcounter_long' + str(i)])

        tot_num_rois_short.append(result_counter['roicounter_short'+ str(i)])
        tot_num_rois_long.append(result_counter['roicounter_long'+ str(i)])
        tot_num_rois_all.append(np.union1d(np.asarray(result_counter['roinumlist_short'+ str(i)]),np.asarray(result_counter['roinumlist_long'+ str(i)])).shape[0])

    tot_num_short = 0
    for rs in tot_num_rois_short:
        tot_num_short += rs
    tot_num_long = 0
    for rs in tot_num_rois_long:
        tot_num_long += rs
    tot_num_all = 0
    for rs in tot_num_rois_all:
        tot_num_all += rs


    print('---')
    print('total number of rois active on short track: ' + str(tot_num_short))
    print('total number of rois active on long track: ' + str(tot_num_long))
    print('total number of rois active on all: ' + str(tot_num_all))
    print('---')

    # group data of fractions by animal? DEPENDS ON ORDER OF ROI_PARAM_LIST
    group_by_animal = False
    if group_by_animal == True:
        num_trialonset_short = [np.mean(num_trialonset_short[0:2]),np.mean(num_trialonset_short[3:4]),np.mean(num_trialonset_short[5:6]),num_trialonset_short[7]]
        num_trialonset_long = [np.mean(num_trialonset_long[0:2]),np.mean(num_trialonset_long[3:4]),np.mean(num_trialonset_long[5:6]),num_trialonset_long[7]]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0:2]),np.mean(num_lmcenter_short[3:4]),np.mean(num_lmcenter_short[5:6]),num_lmcenter_short[7]]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0:2]),np.mean(num_lmcenter_long[3:4]),np.mean(num_lmcenter_long[5:6]),num_lmcenter_long[7]]
        num_reward_short = [np.mean(num_reward_short[0:2]),np.mean(num_reward_short[3:4]),np.mean(num_reward_short[5:6]),num_reward_short[7]]
        num_reward_long = [np.mean(num_reward_long[0:2]),np.mean(num_reward_long[3:4]),np.mean(num_reward_long[5:6]),num_reward_long[7]]

    ax_object.scatter(np.full_like(num_trialonset_short,0,dtype=np.double),np.array(num_trialonset_short),c='0.8',linewidths=0,s=80,zorder=2)
    ax_object2.scatter(np.full_like(num_trialonset_long,0,dtype=np.double),np.array(num_trialonset_long),c='0.5',linewidths=0,s=80,zorder=2)
    ax_object.scatter(np.full_like(num_lmcenter_short,0.6,dtype=np.double),np.array(num_lmcenter_short),c='0.8',linewidths=0,s=80,zorder=2)
    ax_object2.scatter(np.full_like(num_lmcenter_long,0.6,dtype=np.double),np.array(num_lmcenter_long),c='0.5',linewidths=0,s=80,zorder=2)
    ax_object.scatter(np.full_like(num_reward_short,1.2,dtype=np.double),np.array(num_reward_short),c='0.8',linewidths=0,s=80,zorder=2)
    ax_object2.scatter(np.full_like(num_reward_long,1.2,dtype=np.double),np.array(num_reward_long),c='0.5',linewidths=0,s=80,zorder=2)

    # connect datapoints with lines
    for i in range(len(num_trialonset_short)):
        ax_object.plot([0.0,0.6,1.2],[num_trialonset_short[i],num_lmcenter_short[i],num_reward_short[i]],zorder=0,c='0.8')
    ax_object.plot([0.0,0.6,1.2],[np.mean(  np.array(num_trialonset_short)),np.mean(np.array(num_lmcenter_short)),np.mean(np.array(num_reward_short))],zorder=3,c='k',lw=3,ls='--')
    for i in range(len(num_trialonset_short)):
        ax_object2.plot([0.0,0.6,1.2],[num_trialonset_long[i],num_lmcenter_long[i],num_reward_long[i]],zorder=0,c='0.5')
    ax_object2.plot([0.0,0.6,1.2],[np.mean(np.array(num_trialonset_long)),np.mean(np.array(num_lmcenter_long)),np.mean(np.array(num_reward_long))],zorder=3,c='k',lw=3,ls='--')

    # plot mean datapoint across animals
    ax_object.scatter(0, np.mean(np.array(num_trialonset_short)),c='k',s=120,linewidths=0,zorder=3)
    ax_object2.scatter(0, np.mean(np.array(num_trialonset_long)),c='k',s=120,linewidths=0,zorder=3)
    ax_object.scatter(0.6, np.mean(np.array(num_lmcenter_short)),c='k',s=120,linewidths=0,zorder=3)
    ax_object2.scatter(0.6, np.mean(np.array(num_lmcenter_long)),c='k',s=120,linewidths=0,zorder=3)
    ax_object.scatter(1.2, np.mean(np.array(num_reward_short)),c='k',s=120,linewidths=0,zorder=3)
    ax_object2.scatter(1.2, np.mean(np.array(num_reward_long)),c='k',s=120,linewidths=0,zorder=3)

    # carry out statistical analysis. This is not (yet) the correct test: we are treating each group independently, rather than taking into account within-group and between-group variance
    print(sp.stats.f_oneway(np.array(num_trialonset_short),np.array(num_trialonset_long),np.array(num_lmcenter_short),np.array(num_lmcenter_long),np.array(num_reward_short),np.array(num_reward_long)))
    group_labels = ['trialonset_short'] * np.array(num_trialonset_short).shape[0] + \
                   ['trialonset_long'] * np.array(num_trialonset_long).shape[0] + \
                   ['lmcenter_short'] * np.array(num_lmcenter_short).shape[0] + \
                   ['lmcenter_long'] * np.array(num_lmcenter_long).shape[0] + \
                   ['reward_short'] * np.array(num_reward_short).shape[0] + \
                   ['reward_long'] * np.array(num_reward_long).shape[0]
    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(num_trialonset_short),np.array(num_trialonset_long),np.array(num_lmcenter_short),np.array(num_lmcenter_long),np.array(num_reward_short),np.array(num_reward_long))),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)

    # annotate plots and set axis limits
    ax_object.set_xticks([0,0.6,1.2])
    ax_object.set_xticklabels(['trial onset','landmark','reward'], rotation=45)
    ax_object2.set_xticks([0,0.6,1.2])
    ax_object2.set_xticklabels(['trial onset','landmark','reward'], rotation=45)
    ax_object.set_ylabel('Fraction of neurons')
    ax_object2.set_ylabel('Fraction of neurons')

    ax_object.spines['left'].set_linewidth(2)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['bottom'].set_visible(False)
    ax_object.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax_object2.spines['left'].set_linewidth(2)
    ax_object2.spines['top'].set_visible(False)
    ax_object2.spines['right'].set_visible(False)
    ax_object2.spines['bottom'].set_visible(False)
    ax_object2.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    max_y = np.amax(ax_object.get_ylim())
    if np.amax(ax_object2.get_ylim()) > max_y:
        max_y = np.amax(ax_object2.get_ylim())
    ax_object.set_ylim([0,0.5])
    ax_object2.set_ylim([0,0.5])

    # c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA42E', '#00FF12']

    sns.distplot(result_matching_landmark_normdiff['lm_diff_short'],bins=np.linspace(-1,0,11),kde=False,color='0.8',ax=ax_object5,label='short preferring')
    sns.distplot(result_matching_landmark_normdiff['lm_diff_long'],bins=np.linspace(0,1,11),kde=False,color='0.5',ax=ax_object5,label='long preferring')


def roi_amplitude_scatter(roi_param_list, event_list, trialtypes, ax_object, ax_object2, ax_object3, ax_object4, normalize=False):
    """
    plot the amplitude of ROIs against all three alignment points

    RETURN : ax_object

    """
    if normalize:
        min_xy = 0
    else:
        min_xy = -0.2

    # hold values of mean peak
    result_max_peak = {}
    result_max_peak_ol = {}
    # set up empty dicts so we can later append to them
    for el in event_list:
        for tl in trialtypes:
            result_max_peak[el + '_peak_' + tl] = []

    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        # load roi parameters for given session
        with open(rpl,'r') as f:
            roi_params = json.load(f)

        # grab a full list of roi numbers (it doesn't matter which event is chosen here)
        roi_list_all = roi_params['valid_rois']
        # roi_list_all = np.arange(len(roi_params['trialonset' + speed + '_peak_short']))
        # loop through every roi
        for j,r in enumerate(roi_list_all):
            if normalize:
                roi_normalization_value = roi_params['norm_value'][r]
            else:
                roi_normalization_value = 1

            # loop through every trialtype and alignment point to determine largest response
            for tl in trialtypes:
                valid = 0
                max_peak_trialonset = 0
                max_peak_lmcenter = 0
                max_peak_reward = 0
                # loop through every event and store peak value
                # if roi_response_validation(roi_params, tl, 'trialonset', j):
                max_peak_trialonset = roi_params['trialonset' + speed + '_peak_' + tl][j]/roi_normalization_value
                # if roi_response_validation(roi_params, tl, 'lmcenter', j):
                max_peak_lmcenter = roi_params['lmcenter' + speed + '_peak_' + tl][j]/roi_normalization_value
                # if roi_response_validation(roi_params, tl, 'reward', j):
                max_peak_reward = roi_params['reward' + speed + '_peak_' + tl][j]/roi_normalization_value

                result_max_peak['trialonset_peak_' + tl].append(max_peak_trialonset)
                result_max_peak['lmcenter_peak_' + tl].append(max_peak_lmcenter)
                result_max_peak['reward_peak_' + tl].append(max_peak_reward)

    # separate out neurons by what alignment point they better align to and plot them in different colors
    trialonset_neurons_idx = np.array(result_max_peak['trialonset_peak_short']) > np.array(result_max_peak['lmcenter_peak_short'])
    landmark_neurons_idx = np.array(result_max_peak['trialonset_peak_short']) < np.array(result_max_peak['lmcenter_peak_short'])
    ax_object.scatter(np.array(result_max_peak['trialonset_peak_short'])[trialonset_neurons_idx],np.array(result_max_peak['lmcenter_peak_short'])[trialonset_neurons_idx],c='g',linewidths=1,s=40)
    ax_object.scatter(np.array(result_max_peak['trialonset_peak_short'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter_peak_short'])[landmark_neurons_idx],c='m',linewidths=1,s=40)

    reward_neurons_idx = np.array(result_max_peak['reward_peak_short']) > np.array(result_max_peak['lmcenter_peak_short'])
    landmark_neurons_idx = np.array(result_max_peak['reward_peak_short']) < np.array(result_max_peak['lmcenter_peak_short'])
    ax_object2.scatter(np.array(result_max_peak['reward_peak_short'])[reward_neurons_idx],np.array(result_max_peak['lmcenter_peak_short'])[reward_neurons_idx],c='g',linewidths=1,s=40)
    ax_object2.scatter(np.array(result_max_peak['reward_peak_short'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter_peak_short'])[landmark_neurons_idx],c='m',linewidths=1,s=40)

    trialonset_neurons_idx = np.array(result_max_peak['trialonset_peak_long']) > np.array(result_max_peak['lmcenter_peak_long'])
    landmark_neurons_idx = np.array(result_max_peak['trialonset_peak_long']) < np.array(result_max_peak['lmcenter_peak_long'])
    ax_object3.scatter(np.array(result_max_peak['trialonset_peak_long'])[trialonset_neurons_idx],np.array(result_max_peak['lmcenter_peak_long'])[trialonset_neurons_idx],c='g',linewidths=1,s=40)
    ax_object3.scatter(np.array(result_max_peak['trialonset_peak_long'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter_peak_long'])[landmark_neurons_idx],c='m',linewidths=1,s=40)

    reward_neurons_idx = np.array(result_max_peak['reward_peak_long']) > np.array(result_max_peak['lmcenter_peak_long'])
    landmark_neurons_idx = np.array(result_max_peak['reward_peak_long']) < np.array(result_max_peak['lmcenter_peak_long'])
    ax_object4.scatter(np.array(result_max_peak['reward_peak_long'])[reward_neurons_idx],np.array(result_max_peak['lmcenter_peak_long'])[reward_neurons_idx],c='g',linewidths=1,s=40)
    ax_object4.scatter(np.array(result_max_peak['reward_peak_long'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter_peak_long'])[landmark_neurons_idx],c='m',linewidths=1,s=40)

    # print(result_max_peak['lmcenter_peak_long'],result_max_peak['reward_peak_long'])

    # print(ax_object.get_xticklabels())
    ax_object.set_xlabel('peak response TRIALONSET SHORT', fontsize=18)
    ax_object.set_ylabel('peak response LMCENTER SHORT', fontsize=18)
    ax_object2.set_xlabel('peak response REWARD SHORT', fontsize=18)
    ax_object2.set_ylabel('peak response LMCENTER SHORT', fontsize=18)
    ax_object3.set_xlabel('peak response TRIALONSET LONG', fontsize=18)
    ax_object3.set_ylabel('peak response LMCENTER LONG', fontsize=18)
    ax_object4.set_xlabel('peak response REWARD LONG', fontsize=18)
    ax_object4.set_ylabel('peak response LMCENTER LONG', fontsize=18)

    ax_object.set_xlim([0,1])
    ax_object.set_ylim([0,1])
    ax_object2.set_xlim([0,1])
    ax_object2.set_ylim([0,1])
    ax_object3.set_xlim([0,1])
    ax_object3.set_ylim([0,1])
    ax_object4.set_xlim([0,1])
    ax_object4.set_ylim([0,1])
    ax_object.plot([0,1],[0,1], ls='--', c='k')
    ax_object2.plot([0,1],[0,1], ls='--', c='k')
    ax_object3.plot([0,1],[0,1], ls='--', c='k')
    ax_object4.plot([0,1],[0,1], ls='--', c='k')

    ax_object.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=17, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax_object2.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=17, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax_object3.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=17, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax_object4.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=17, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')


    return

if __name__ == '__main__':
    # list of roi parameter files
    suffix = ''
    speed = '_' + 'slow'

    # all
    roi_param_list = ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day2017719'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_1'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_2'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_3'+suffix+'.json',
                      'E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776'+suffix+'.json',
                      'E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_1'+suffix+'.json',
                      'E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2'+suffix+'.json',
                      'E:\\MTH3_figures\\LF171211_1\\LF171211_1_Day2018321_2'+suffix+'.json',
                      ]

    # L2/3
    # roi_param_list = ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804'+suffix+'.json',
    #                   'E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_1'+suffix+'.json',
    #                   'E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_2'+suffix+'.json',
    #                   'E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_3'+suffix+'.json',
    #                   'E:\\MTH3_figures\\LF171211_2\\LF171211_2_Day201852'+suffix+'.json'
    #                   ]

    # LAYER 5
    # roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017719/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day201783/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170222_1_Day201776/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170613_1_Day201784/roi_params.json']

    # V1
    # roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF170214_1_Day201777/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170214_1_Day2017714/roi_params.json',
    #                  '/Users/lukasfischer/Work/exps/MTH3/figures/LF171211_2_Day201852/roi_params.json',]



    fname = 'summary figure slowvfast'
    event_list = ['trialonset','lmcenter','reward']
    trialtypes = ['short', 'long']
    subfolder = 'summary'
    normalize = True

    # create figure and axes to later plot on
    fig = plt.figure(figsize=(12,24))
    ax1 = plt.subplot2grid((4,4),(0,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((4,4),(0,2), rowspan=1, colspan=2)
    ax3 = plt.subplot2grid((4,4),(0,1), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((4,4),(1,0), rowspan=1, colspan=2)
    ax5 = plt.subplot2grid((4,4),(1,2), rowspan=1, colspan=2)
    ax6 = plt.subplot2grid((4,4),(2,0), rowspan=1, colspan=2)
    ax7 = plt.subplot2grid((4,4),(2,2), rowspan=1, colspan=2)
    ax8 = plt.subplot2grid((4,4),(3,0), rowspan=1, colspan=2)
    ax9 = plt.subplot2grid((4,4),(3,2), rowspan=1, colspan=2)

    event_maxresponse(roi_param_list, event_list, trialtypes, ax1, ax3, ax4, normalize)
    roi_amplitude_scatter(roi_param_list, event_list, trialtypes, ax6, ax7, ax8, ax9, normalize)

    fformat = 'png'
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + suffix + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + suffix + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    plt.close(fig)
