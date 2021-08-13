"""
scatterplot of ROI amplitudes in VR and openloop

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

    roi_activity = el + '_active_' + tl
    roi_peak_val = el + '_peak_' + tl
    if roi_params[roi_activity][roi_idx_num] > MIN_TRIALS_ACTIVE and roi_params[roi_peak_val][roi_idx_num] > MIN_DF:
        return True
    else:
        return False


def event_maxresponse(roi_param_list, event_list, trialtypes, ax_object, ax_object2, ax_object3, ax_object4, ax_object5):
    """
    determine which alignment gave the max mean response

    roi_param_list : list of roi param files (fulle file paths)

    event_list : list of alignments available

    """

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
        roi_list_all = roi_params['lmcenter_roi_number']

        # loop through every roi
        for j,r in enumerate(roi_list_all):
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
                    value_key = el + '_peak_' + tl
                    value_key_ol = el + '_peak_' + tl + '_ol'
                    if roi_params[value_key][j] > max_peak and roi_response_validation(roi_params, tl, el, j):
                        valid = 1
                        max_peak = roi_params[value_key][j]
                        max_peak_ol = roi_params[value_key_ol][j]
                        # grab peak value for trialonset condition
                        if roi_response_validation(roi_params, tl, 'trialonset', j):
                            peak_trialonset = roi_params['trialonset_peak_'+tl][j]
                        else:
                            peak_trialonset = 0
                        # grab peak value for lmcenter condition
                        if roi_response_validation(roi_params, tl, 'lmcenter', j):
                            peak_lmcenter = roi_params['lmcenter_peak_'+tl][j]
                        else:
                            peak_lmcenter = 0
                        # if the peak response is by the landmark, check which trialtype elicits the larger response and store the normalized difference
                        if el == 'lmcenter':
                            if tl == 'short' and max_peak > roi_params[el + '_peak_long'][j]:
                                peak_lm_long = roi_params['lmcenter_peak_long'][j]
                                peak_lm_normdiff = (roi_params[el + '_peak_long'][j]/max_peak)-1
                                peak_lm_type = 'short'
                            elif tl == 'long' and max_peak > roi_params[el + '_peak_short'][j]:
                                peak_lm_short = roi_params['lmcenter_peak_short'][j]
                                peak_lm_normdiff = ((roi_params[el + '_peak_short'][j]/max_peak)-1)*-1
                                peak_lm_type = 'long'
                        peak_event = el
                        peak_trialtype = tl
                if valid == 1:
                    # add 1/total number of rois to get fraction
                    result_counter[peak_event + '_peakcounter_' + peak_trialtype + str(i)] = result_counter[peak_event + '_peakcounter_' + peak_trialtype + str(i)] + (1/len(roi_list_all))
                    result_counter['roicounter_' + peak_trialtype + str(i)] = result_counter['roicounter_' + peak_trialtype + str(i)] + 1
                    result_counter['roinumlist_' + peak_trialtype + str(i)].append(r)
                    result_max_peak[peak_event + '_peakval_' + peak_trialtype].append(max_peak)
                    result_max_peak_ol[peak_event + '_peakval_' + peak_trialtype].append(max_peak_ol)
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
    ax_object.set_ylim([0,max_y])
    ax_object2.set_ylim([0,max_y])

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA42E', '#00FF12']
    ax_object3.scatter(np.asarray(result_max_peak['trialonset_peakval_short']), np.asarray(result_max_peak_ol['trialonset_peakval_short']),s=10,color='0.8', label='short')
    ax_object3.scatter(np.asarray(result_max_peak['trialonset_peakval_long']), np.asarray(result_max_peak_ol['trialonset_peakval_long']),s=10,color='0.5', label='long')
    ax_object3.scatter(np.asarray(result_max_peak['lmcenter_peakval_short']), np.asarray(result_max_peak_ol['lmcenter_peakval_short']),s=10,color='0.8')
    ax_object3.scatter(np.asarray(result_max_peak['lmcenter_peakval_long']), np.asarray(result_max_peak_ol['lmcenter_peakval_long']),s=10,color='0.5')
    ax_object3.scatter(np.asarray(result_max_peak['reward_peakval_short']), np.asarray(result_max_peak_ol['reward_peakval_short']),s=10,color='0.8')
    ax_object3.scatter(np.asarray(result_max_peak['reward_peakval_long']), np.asarray(result_max_peak_ol['reward_peakval_long']),s=10,color='0.5')
    ax_object3.legend()

    max_peak_all = np.concatenate((np.asarray(result_max_peak['reward_peakval_short']),np.asarray(result_max_peak['reward_peakval_long']),np.asarray(result_max_peak['trialonset_peakval_short']),np.asarray(result_max_peak['trialonset_peakval_long']),np.asarray(result_max_peak['lmcenter_peakval_short']),np.asarray(result_max_peak['lmcenter_peakval_long'])))
    max_peak_all_ol = np.concatenate((np.asarray(result_max_peak_ol['reward_peakval_long']),np.asarray(result_max_peak_ol['reward_peakval_short']),np.asarray(result_max_peak_ol['trialonset_peakval_long']),np.asarray(result_max_peak_ol['trialonset_peakval_short']),np.asarray(result_max_peak_ol['lmcenter_peakval_short']),np.asarray(result_max_peak_ol['lmcenter_peakval_long'])))

    # sns.regplot(max_peak_all_ol, max_peak_all,scatter=False,ci=None, ax=ax_object3)

    # print(np.asarray(result_max_peak_ol['lmcenter_peakval_long']))
    # exog_var = sm.add_constant(np.asarray(result_max_peak_ol['lmcenter_peakval_short']))
    # print(np.asarray(result_max_peak['lmcenter_peakval_short']).shape)
    # print(exog_var.shape)
    #
    # rlmresBI = sm.OLS(np.asarray(result_max_peak['lmcenter_peakval_short']),exog_var,M=sm.robust.norms.TukeyBiweight()).fit()
    # rlmresBI = sm.RLM(np.asarray(result_max_peak['lmcenter_peakval_short']),exog_var,M=sm.robust.norms.TukeyBiweight()).fit()
    # rlmresBI = sm.RLM(testvar,exog_var,M=sm.robust.norms.TukeyBiweight()).fit()
    # ax_object3.plot(np.asarray(result_max_peak_ol['lmcenter_peakval_short']),rlmresBI.fittedvalues,label='TukeyBiweight')
    # ax_object3.plot(testvar,rlmresBI.fittedvalues,label='TukeyBiweight')
    # print(rlmresBI.summary(yname='y',xname=['var_%d' % i for i in range(len(rlmresBI.params))]))

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(max_peak_all, max_peak_all_ol)
    # p = np.polyfit(max_peak_all_ol, max_peak_all,1)
    # ax_object3.plot(max_peak_all_ol, p[1]+p[0]*max_peak_all_ol)
    print('slope, intercept, p-value of linear regression: '  + str([slope, intercept, p_value]))
    print('paired t-test (VR vs OL): ' + str(sp.stats.ttest_rel(max_peak_all, max_peak_all_ol)))
    # ax_object3.plot(intercept + slope*max_peak_all, max_peak_all, 'k', label='fitted line')
    ax_object3.plot(max_peak_all, intercept + slope*max_peak_all, 'k', label='fitted line')

    # determine maximum value and scale y-axis
    max_y = np.nanmax(np.concatenate((np.asarray(result_max_peak_ol['trialonset_peakval_short']),np.asarray(result_max_peak['trialonset_peakval_short']),
                                      np.asarray(result_max_peak_ol['trialonset_peakval_long']), np.asarray(result_max_peak['trialonset_peakval_long']),
                                      np.asarray(result_max_peak_ol['lmcenter_peakval_short']), np.asarray(result_max_peak['lmcenter_peakval_short']),
                                      np.asarray(result_max_peak_ol['lmcenter_peakval_long']), np.asarray(result_max_peak['lmcenter_peakval_long']),
                                      np.asarray(result_max_peak_ol['reward_peakval_short']), np.asarray(result_max_peak['reward_peakval_short']),
                                      np.asarray(result_max_peak_ol['reward_peakval_long']), np.asarray(result_max_peak['reward_peakval_long']))))

    ax_object3.plot([0,max_y],[0,max_y],lw=2,c='k',ls='--')
    if max_y > 3:
        max_y=3
    ax_object3.set_xlim([-0.2,max_y])
    ax_object3.set_ylim([-0.2,max_y])
    ax_object3.set_xlabel('peak response PASSIVE')
    ax_object3.set_ylabel('peak response VR')
    ax_object3.legend()

    # plot mean peak aligned to its optimal alignment point vs trialonset (i.e. trial start time)
    ax_object4.scatter(np.asarray(result_trialonset_max_peak['lmcenter_peakval_short']), np.asarray(result_max_peak['lmcenter_peakval_short']),s=10,color='0.8')
    ax_object4.scatter(np.asarray(result_trialonset_max_peak['reward_peakval_short']), np.asarray(result_max_peak['reward_peakval_short']),s=10,color='0.8')
    ax_object4.scatter(np.asarray(result_trialonset_max_peak['lmcenter_peakval_long']), np.asarray(result_max_peak['lmcenter_peakval_long']),s=10,color='0.5')
    ax_object4.scatter(np.asarray(result_trialonset_max_peak['reward_peakval_long']), np.asarray(result_max_peak['reward_peakval_long']),s=10,color='0.5')
    ax_object4.scatter(np.asarray(result_trialonset_max_peak['trialonset_peakval_short']), np.asarray(result_lmcenter_max_peak['trialonset_peakval_short']),s=10,color='0.8')
    ax_object4.scatter(np.asarray(result_trialonset_max_peak['trialonset_peakval_long']), np.asarray(result_lmcenter_max_peak['trialonset_peakval_long']),s=10,color='0.5')
    ax_object4.plot([0,max_y],[0,max_y],lw=2,c='k',ls='--')
    ax_object4.set_xlim([-0.2,max_y])
    ax_object4.set_ylim([-0.2,max_y])
    ax_object4.set_xlabel('peak response aligned to trial onset')
    ax_object4.set_ylabel('peak response aligned to landmark or reward')

    sns.distplot(result_matching_landmark_normdiff['lm_diff_short'],bins=np.linspace(-1,0,11),kde=False,color='0.8',ax=ax_object5,label='short preferring')
    sns.distplot(result_matching_landmark_normdiff['lm_diff_long'],bins=np.linspace(0,1,11),kde=False,color='0.5',ax=ax_object5,label='long preferring')
    ax_object5.legend()
    ax_object5.set_xlim([-1.2,1.2])
    ax_object5.set_xlabel('landmark response difference (normalized)')
    ax_object5.set_ylabel('number of neurons')
    # ax_object5.scatter(np.asarray(result_matching_landmark_long['lmcenter_peakval_short']), np.asarray(result_max_peak['lmcenter_peakval_short']),s=10,marker='.',color='0.8')
    # ax_object5.scatter(np.asarray(result_max_peak['lmcenter_peakval_long']), np.asarray(result_matching_landmark_short['lmcenter_peakval_long']),s=10,marker='.',color='0.5')

    # ax_object5.plot([0,max_y],[0,max_y],lw=2,c='k',ls='--')
    # ax_object5.set_xlim([-0.2,max_y])
    # ax_object5.set_ylim([-0.2,max_y])

    # lm_response_diff_noramlised = []
    # for i in range(len(result_matching_landmark_long['lmcenter_peakval_short'])):
    #     lm_response_diff_noramlised.append((result_max_peak['lmcenter_peakval_short'][i] - result_matching_landmark_long['lmcenter_peakval_short'][i])/result_max_peak['lmcenter_peakval_short'][i])
    # for i in range(len(result_matching_landmark_long['lmcenter_peakval_short'])):
    #     lm_response_diff_noramlised.append((result_max_peak['lmcenter_peakval_long'][i] - result_matching_landmark_short['lmcenter_peakval_long'][i])/result_max_peak['lmcenter_peakval_short'][i])
    # sns.distplot(lm_response_diff_noramlised, ax = ax_object5)
    # # sns.distplot(np.asarray(result_max_peak['lmcenter_peakval_long']) - np.asarray(result_matching_landmark_short['lmcenter_peakval_long']), ax = ax_object5)
    # sns.jointplot(result_matching_landmark_long['lmcenter_peakval_short'], result_max_peak['lmcenter_peakval_short'], kind="kde")



def roi_amplitude_scatter(roi_param_list, roilist, ax_object):
    """
    plot the amplitude of ROIs in VR and openloop condition

    roi_param_list : list of roi param files (fulle file paths)

    roilist : list (one for each element in roi_param_list) containing list of rois to be plotted

    ax_object : axis object to be plotted on

    RETURN : ax_object

    """

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA42E', '#00FF12']
    SUBNAME = 'lmcenter'

    # keep track of max value for axis
    max_xy = 0

    # run through all roi_param files and add datapoints to scatterplot
    for i,rpl in enumerate(roi_param_list):
        # load roi parameters for given session
        with open(rpl,'r') as f:
            roi_params = json.load(f)
        # cycle through all rois, the the roi list of lmoff as reference (they should all be the same or at least contain the same rois even if not in the same order)
        for roi in roi_params['trialonset_roi_number']:
            # need to parse arrays from json file into numpy like this
            lmoff_roi_numbers = np.array(roi_params['trialonset_roi_number'])
        # plot datapoints on scatterplot
        ax_object.scatter(roi_params[SUBNAME+'_peak_short_ol'], roi_params[SUBNAME+'_peak_short'], color=c[np.mod(i,len(c))], marker='o',linewidths=0,zorder=2,label=rpl.split(os.sep)[-2])
        if np.nanmax(roi_params[SUBNAME+'_peak_short_ol']) > max_xy:
            max_xy = np.nanmax(roi_params[SUBNAME+'_peak_short_ol'])
        if np.nanmax(roi_params[SUBNAME+'_peak_short']) > max_xy:
            max_xy = np.nanmax(roi_params[SUBNAME+'_peak_short'])

    ax_object.plot([0,max_xy],[0,max_xy], c='0.5',lw=1, ls='--',zorder=1)
    # if max_xy > 4:
    #     max_xy=4
    ax_object.set_xlim([-0.2,max_xy])
    ax_object.set_ylim([-0.2,max_xy])
    ax_object.legend(fontsize=12)
    ax_object.set_ylabel('peak response VR')
    ax_object.set_xlabel('peak response PASSIVE')

if __name__ == '__main__':
    # list of roi parameter files
    roi_param_list = [# '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_1/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_2/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day201748_3/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017719/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017720/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day201783/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day2017719/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170613_1_Day201784/roi_params.json'
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170613_1_Day20170804/roi_params.json'
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170222_1_Day201776/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170222_1_Day2017615/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF170110_2_Day2017331/roi_params.json',
                      # '/Users/lukasfischer/Work/exps/MTH3/figures/LF171212_2_Day2018218_2/roi_params.json'
                     ]

    # roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF170421_2_Day2017719/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170420_1_Day201783/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170222_1_Day201776/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170613_1_Day201784/roi_params.json']

    # roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF170214_1_Day201777/roi_params.json',
    #                   '/Users/lukasfischer/Work/exps/MTH3/figures/LF170214_1_Day2017714/roi_params.json',
    #                  '/Users/lukasfischer/Work/exps/MTH3/figures/LF171211_2_Day201852/roi_params.json',]

    fname = 'summary figure'
    event_list = ['trialonset','lmcenter','reward']
    trialtypes = ['short', 'long']
    subfolder = []

    # create figure and axes to later plot on
    fig = plt.figure(figsize=(12,12))
    ax1 = plt.subplot2grid((2,4),(0,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((2,4),(0,2), rowspan=1, colspan=2)
    ax3 = plt.subplot2grid((2,4),(0,1), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((2,4),(1,0), rowspan=1, colspan=2)
    ax5 = plt.subplot2grid((2,4),(1,2), rowspan=1, colspan=2)
    # ax6 = plt.subplot2grid((3,4),(2,0), rowspan=1, colspan=2)

    event_maxresponse(roi_param_list, event_list, trialtypes, ax1, ax3, ax4, ax5, ax2)
    # roi_amplitude_scatter(roi_param_list, [], ax6)

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
