"""
Plot transient parameters for a given session for vr vs openloop condition

@author: lukasfischer

"""

import json, yaml, ipdb, os, sys
import warnings; warnings.simplefilter('ignore')
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import statsmodels as sm_all
import seaborn as sns
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
plt.rcParams['svg.fonttype'] = 'none'
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

from analysis_parameters import MIN_FRACTION_ACTIVE, MIN_MEAN_AMP, MIN_ZSCORE

subfolder = 'vrol_transients'
fformat = 'svg'
SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'
trialtypes = ['short', 'long']

def roi_response_validation(roi_params, tl, el, roi_idx_num):
    """
    separate function to check whether a given response passes criterion for being considered a real roi_response_validation

    """

    roi_activity = roi_params[el + '_active_' + tl][roi_idx_num]
    roi_peak_val = roi_params[el + '_peak_' + tl][roi_idx_num]
    roi_zscore_val = roi_params[el + '_peak_zscore_' + tl][roi_idx_num]
    mean_trace = roi_params['space_mean_trace_'+tl][roi_idx_num]

    if roi_activity > MIN_FRACTION_ACTIVE and roi_zscore_val > MIN_ZSCORE and (np.nanmax(mean_trace) - np.nanmin(mean_trace)) > MIN_MEAN_AMP:
        return True
    else:
        return False

def transient_analysis_validation(roi_params, tl, roi_idx_num):
    """
    separate function to check whether a given response passes criterion for being considered a real roi_response_validation

    """

    # ipdb.set_trace()
    numRF_transients = roi_params['space_meanpeak_num_RF_transients_' + tl][roi_idx_num]
    if numRF_transients > 0.2:
        return True
    else:
        return False


def get_eventaligned_rois(roi_params, align_event):
    """ return roi number, peak value and peak time of all neurons that have their max response at <align_even> in VR """
    # hold values of mean peak
    event_list = ['trialonset','lmcenter','reward']
    result_max_peak = {}
    # set up empty dicts so we can later append to them
    for tl in trialtypes:
        result_max_peak[align_event + '_valid_number_' + tl] = []
        result_max_peak[align_event + '_roi_number_' + tl] = []

    # grab a full list of roi numbers
    roi_list_all = roi_params['valid_rois']
    # loop through every roi
    for j,r in enumerate(roi_list_all):
        # loop through every trialtype and alignment point to determine largest response
        for tl in trialtypes:
            max_peak = -99
            roi_num = -1
            valid_num = -1
            valid = False
            peak_event = ''
            peak_trialtype = ''
            for el in event_list:
                value_key = el + '_peak_' + tl
                # check roi max peak for each alignment point and store wich ones has the highest value
                if roi_params[value_key][j] > max_peak and roi_response_validation(roi_params, tl, el, j) and transient_analysis_validation(roi_params, tl, j):
                    valid = True
                    max_peak = roi_params[value_key][j]
                    peak_event = el
                    peak_trialtype = tl
                    roi_num = r
                    valid_num = j
            # write results for alignment point with highest value to results dict
            if valid and peak_event == align_event:
                result_max_peak[align_event + '_valid_number_' + tl].append(valid_num)
                result_max_peak[align_event + '_roi_number_' + peak_trialtype].append(roi_num)

    return result_max_peak

def set_boxplot_appearance_sit_run(parts):
    colors = [['#3953A3','#3953A3'],['#FF0000','#FF0000'],['k','k']]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

def set_boxplot_appearance_short(parts):
    colors = [[SHORT_COLOR,SHORT_COLOR],['w',SHORT_COLOR]]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

def set_boxplot_appearance_long(parts):
    colors = [[LONG_COLOR,LONG_COLOR],['w',LONG_COLOR]]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

def set_empty_axis(ax_object):
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_linewidth(2)
    ax_object.spines['bottom'].set_visible(False)
    ax_object.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        length=4, \
        width=2, \
        left='on', \
        bottom='off', \
        right='off', \
        top='off')
    return ax_object

def plot_goodvbad_activity(roi_param_list_all, fname, sim_data=False):
    # create figure with axes
    fig = plt.figure(figsize=(6,8))
    ax1 = plt.subplot2grid((8,12),(0,0),colspan=4,rowspan=4)
    ax2 = plt.subplot2grid((8,12),(0,4),colspan=4,rowspan=4)
    ax3 = plt.subplot2grid((8,12),(0,8),colspan=4,rowspan=4)
    ax4 = plt.subplot2grid((8,12),(4,0),colspan=4,rowspan=4)
    ax5 = plt.subplot2grid((8,12),(4,4),colspan=4,rowspan=4)
    ax6 = plt.subplot2grid((8,12),(4,8),colspan=4,rowspan=4)

    set_empty_axis(ax1)
    set_empty_axis(ax2)
    set_empty_axis(ax3)
    set_empty_axis(ax4)
    set_empty_axis(ax5)
    set_empty_axis(ax6)

    num_rois_short = []
    num_rois_long = []
    num_sig = 0
    num_trend = 0

    diff_AUC_short_all = []
    diff_AUC_long_all = []
    zscore_AUC_short_all = []
    zscore_AUC_long_all = []

    diff_PEAK_short_all = []
    diff_PEAK_long_all = []
    zscore_PEAK_short_all = []
    zscore_PEAK_long_all = []

    diff_AUC_short_all_sig = []
    diff_AUC_long_all_sig = []
    diff_PEAK_short_all_sig = []
    diff_PEAK_long_all_sig = []

    diff_AUC_short_all_trend = []
    diff_AUC_long_all_trend = []
    diff_PEAK_short_all_trend = []
    diff_PEAK_long_all_trend = []

    diff_ROB_short_all = []
    diff_ROB_long_all = []

    shuffles = 10000

    # loop through all .json files
    for rpl in roi_param_list_all:
        with open(rpl[0],'r') as f:
            roi_params = json.load(f)
        print(rpl)
        # get eventaligned rois
        trialonset_rois = get_eventaligned_rois(roi_params, 'trialonset')
        lmcenter_rois = get_eventaligned_rois(roi_params, 'lmcenter')
        reward_rois = get_eventaligned_rois(roi_params, 'reward')

        # throw all different neuron types together
        included_rois_short = [i for i in lmcenter_rois['lmcenter_valid_number_short']]
        included_rois_short.extend([i for i in trialonset_rois['trialonset_valid_number_short']])
        included_rois_short.extend([i for i in reward_rois['reward_valid_number_short']])

        included_rois_long = [i for i in lmcenter_rois['lmcenter_valid_number_long']]
        included_rois_long.extend([i for i in trialonset_rois['trialonset_valid_number_long']])
        included_rois_long.extend([i for i in reward_rois['reward_valid_number_long']])

        # make list of actual roi numbers
        roi_numbers_short = [i for i in lmcenter_rois['lmcenter_roi_number_short']]
        roi_numbers_short.extend([i for i in trialonset_rois['trialonset_roi_number_short']])
        roi_numbers_short.extend([i for i in reward_rois['reward_roi_number_short']])

        roi_numbers_long = [i for i in lmcenter_rois['lmcenter_roi_number_long']]
        roi_numbers_long.extend([i for i in trialonset_rois['trialonset_roi_number_long']])
        roi_numbers_long.extend([i for i in reward_rois['reward_roi_number_long']])

        # count overall number of neurons
        num_rois_short.append(len(included_rois_short))
        num_rois_long.append(len(included_rois_long))

        # calculate differences between conditions for each roi
        # ipdb.set_trace()
        for i,irs in enumerate(included_rois_short):
            short_close_AUC = roi_params['space_short_close_AUC_mean'][irs]
            short_far_AUC = roi_params['space_short_far_AUC_mean'][irs]

            short_close_PEAK = roi_params['space_short_close_PEAK_mean'][irs]
            short_far_PEAK = roi_params['space_short_far_PEAK_mean'][irs]
            short_close_ROB = roi_params['space_short_close_ROB_mean'][irs]
            short_far_ROB = roi_params['space_short_far_ROB_mean'][irs]

            long_close_AUC = roi_params['space_long_close_AUC_mean'][irs]
            long_far_AUC = roi_params['space_long_far_AUC_mean'][irs]
            long_close_PEAK = roi_params['space_long_close_PEAK_mean'][irs]
            long_far_PEAK = roi_params['space_long_far_PEAK_mean'][irs]
            long_close_ROB = roi_params['space_long_close_ROB_mean'][irs]
            long_far_ROB = roi_params['space_long_far_ROB_mean'][irs]

            diff_AUC_short_shuffled = []
            for j in range(shuffles):
                shuffled_seq = np.concatenate((short_close_AUC,short_far_AUC))
                np.random.shuffle(shuffled_seq)
                shuffle_closed_AUC = shuffled_seq[0:int(shuffled_seq.shape[0]/2)]
                shuffle_far_AUC = shuffled_seq[int(shuffled_seq.shape[0]/2):]
                diff_AUC_short_shuffled.append(np.mean(shuffle_closed_AUC) - np.mean(shuffle_far_AUC))
            zscore_AUC_short_all.append(((np.mean(short_close_AUC) - np.mean(short_far_AUC)) - np.mean(diff_AUC_short_shuffled)) / np.std(diff_AUC_short_shuffled))

            diff_AUC_long_shuffled = []
            for j in range(shuffles):
                shuffled_seq = np.concatenate((long_close_AUC,long_far_AUC))
                np.random.shuffle(shuffled_seq)
                shuffle_closed_AUC = shuffled_seq[0:int(shuffled_seq.shape[0]/2)]
                shuffle_far_AUC = shuffled_seq[int(shuffled_seq.shape[0]/2):]
                diff_AUC_long_shuffled.append(np.mean(shuffle_closed_AUC) - np.mean(shuffle_far_AUC))
            zscore_AUC_long_all.append(((np.mean(long_close_AUC) - np.mean(long_far_AUC)) - np.mean(diff_AUC_long_shuffled)) / np.std(diff_AUC_long_shuffled))

            diff_PEAK_short_shuffled = []
            for j in range(shuffles):
                shuffled_seq = np.concatenate((short_close_PEAK,short_far_PEAK))
                np.random.shuffle(shuffled_seq)
                shuffle_closed_PEAK = shuffled_seq[0:int(shuffled_seq.shape[0]/2)]
                shuffle_far_PEAK = shuffled_seq[int(shuffled_seq.shape[0]/2):]
                diff_PEAK_short_shuffled.append(np.mean(shuffle_closed_PEAK) - np.mean(shuffle_far_PEAK))
            zscore_PEAK_short_all.append(((np.mean(short_close_PEAK) - np.mean(short_far_PEAK)) - np.mean(diff_PEAK_short_shuffled)) / np.std(diff_PEAK_short_shuffled))

            diff_PEAK_long_shuffled = []
            for j in range(shuffles):
                shuffled_seq = np.concatenate((long_close_PEAK,long_far_PEAK))
                np.random.shuffle(shuffled_seq)
                shuffle_closed_PEAK = shuffled_seq[0:int(shuffled_seq.shape[0]/2)]
                shuffle_far_PEAK = shuffled_seq[int(shuffled_seq.shape[0]/2):]
                diff_PEAK_long_shuffled.append(np.mean(shuffle_closed_PEAK) - np.mean(shuffle_far_PEAK))
            zscore_PEAK_long_all.append(((np.mean(long_close_PEAK) - np.mean(long_far_PEAK)) - np.mean(diff_PEAK_long_shuffled)) / np.std(diff_PEAK_long_shuffled))

            # # test for statistically significant difference
            # s, p_short_AUC = stats.ttest_rel(short_close_AUC, short_far_AUC)
            # s, p_short_PEAK = stats.ttest_rel(short_close_PEAK, short_far_PEAK)
            # s, p_long_AUC = stats.ttest_rel(long_close_AUC, long_far_AUC)
            # s, p_long_PEAK = stats.ttest_rel(long_close_PEAK, long_far_PEAK)

            min_difference = 0.4

            short_close_PEAK_diff = np.mean(short_close_PEAK) - np.mean(short_far_PEAK)
            if (zscore_PEAK_short_all[-1] < -1.96 or zscore_PEAK_short_all[-1] > 1.96 or \
                zscore_PEAK_short_all[-1] < -1.96 or zscore_PEAK_short_all[-1] > 1.96) and \
                (short_close_PEAK_diff < -min_difference or short_close_PEAK_diff > min_difference):
                diff_PEAK_short_all_sig.extend([np.mean(short_close_PEAK) - np.mean(short_far_PEAK)])
                num_sig = num_sig + 1
                print('SHORT: ' + str(roi_numbers_short[i]))
            elif short_close_PEAK_diff < -min_difference or short_close_PEAK_diff > min_difference:
                diff_PEAK_short_all_trend.extend([short_close_PEAK_diff])
                num_trend = num_trend + 1
                # print('SHORT TREND: ' + str(roi_numbers_short[irs]))
            else:
                diff_PEAK_short_all.extend([short_close_PEAK_diff])


            long_far_PEAK_diff = np.mean(long_close_PEAK) - np.mean(long_far_PEAK)
            if (zscore_PEAK_long_all[-1] < -1.96 or zscore_PEAK_long_all[-1] > 1.96 or \
                zscore_PEAK_long_all[-1] < -1.96 or zscore_PEAK_long_all[-1] > 1.96) and \
                (long_far_PEAK_diff < -min_difference or long_far_PEAK_diff > min_difference):
                diff_PEAK_long_all_sig.extend([np.mean(long_close_PEAK) - np.mean(long_far_PEAK)])
                num_sig = num_sig + 1
                print('LONG: ' + str(roi_numbers_short[i]))
            elif long_far_PEAK_diff < -min_difference or long_far_PEAK_diff > min_difference:
                diff_PEAK_long_all_trend.extend([long_far_PEAK_diff])
                num_trend = num_trend + 1
                # print('LONG TREND: ' + str(roi_numbers_short[irs]))
            else:
                diff_PEAK_long_all.extend([long_far_PEAK_diff])



            if zscore_AUC_short_all[-1] < -1.96 or zscore_AUC_short_all[-1] > 1.96:
                diff_AUC_short_all_sig.extend([np.mean(short_close_AUC) - np.mean(short_far_AUC)])
            else:
                diff_AUC_short_all.extend([np.mean(short_close_AUC) - np.mean(short_far_AUC)])

            if zscore_AUC_long_all[-1] < -1.96 or zscore_AUC_long_all[-1] > 1.96:
                diff_AUC_long_all_sig.extend([np.mean(long_close_AUC) - np.mean(long_far_AUC)])
            else:
                diff_AUC_long_all.extend([np.mean(long_close_AUC) - np.mean(long_far_AUC)])

            # if zscore_PEAK_short_all[-1] < -1.96 or zscore_PEAK_short_all[-1] > 1.96:
            #     diff_PEAK_short_all_sig.extend([np.mean(short_close_PEAK) - np.mean(short_far_PEAK)])
            # else:
            #     diff_PEAK_short_all.extend([np.mean(short_close_PEAK) - np.mean(short_far_PEAK)])
            #
            # if zscore_PEAK_long_all[-1] < -1.96 or zscore_PEAK_long_all[-1] > 1.96:
            #     diff_PEAK_long_all_sig.extend([np.mean(long_close_PEAK) - np.mean(long_far_PEAK)])
            # else:
            #     diff_PEAK_long_all.extend([np.mean(long_close_PEAK) - np.mean(long_far_PEAK)])



    x_pos_short = np.full_like(diff_AUC_short_all,0) + (np.random.randn(len(diff_AUC_short_all)) * 0.075)
    ax1.scatter(x_pos_short, np.array(diff_AUC_short_all),s=40, edgecolors='0.8', facecolors='none', label='short')
    x_pos_short = np.full_like(diff_AUC_short_all_sig,0) + (np.random.randn(len(diff_AUC_short_all_sig)) * 0.075)
    ax1.scatter(x_pos_short, np.array(diff_AUC_short_all_sig),s=40, edgecolors=SHORT_COLOR, facecolors='none', label='short')

    x_pos_long = np.full_like(diff_AUC_long_all,0) + (np.random.randn(len(diff_AUC_long_all)) * 0.075) + 1
    ax1.scatter(x_pos_long, np.array(diff_AUC_long_all),s=40, edgecolors='0.8', facecolors='none', label='long')
    x_pos_long = np.full_like(diff_AUC_long_all_sig,0) + (np.random.randn(len(diff_AUC_long_all_sig)) * 0.075) + 1
    ax1.scatter(x_pos_long, np.array(diff_AUC_long_all_sig),s=40, edgecolors=LONG_COLOR, facecolors='none', label='long')

    x_pos_short = np.full_like(zscore_AUC_short_all,0) + (np.random.randn(len(zscore_AUC_short_all)) * 0.075)
    ax4.scatter(x_pos_short, np.array(zscore_AUC_short_all),s=40, edgecolors=SHORT_COLOR, facecolors='none', label='long')

    x_pos_long = np.full_like(zscore_AUC_long_all,0) + (np.random.randn(len(zscore_AUC_long_all)) * 0.075) + 1
    ax4.scatter(x_pos_long, np.array(zscore_AUC_long_all),s=40, edgecolors=LONG_COLOR, facecolors='none', label='long')



    x_pos_short = np.full_like(diff_PEAK_short_all,0) + (np.random.randn(len(diff_PEAK_short_all)) * 0.075)
    ax2.scatter(x_pos_short, np.array(diff_PEAK_short_all),s=40, edgecolors='0.8', facecolors='none', label='short')
    x_pos_short = np.full_like(diff_PEAK_short_all_trend,0) + (np.random.randn(len(diff_PEAK_short_all_trend)) * 0.075)
    ax2.scatter(x_pos_short, np.array(diff_PEAK_short_all_trend),s=40, edgecolors=SHORT_COLOR, facecolors='none', label='short')
    x_pos_short = np.full_like(diff_PEAK_short_all_sig,0) + (np.random.randn(len(diff_PEAK_short_all_sig)) * 0.075)
    ax2.scatter(x_pos_short, np.array(diff_PEAK_short_all_sig),s=40, edgecolors=SHORT_COLOR, facecolors=SHORT_COLOR, label='short')

    x_pos_long = np.full_like(diff_PEAK_long_all,0) + (np.random.randn(len(diff_PEAK_long_all)) * 0.075) + 1
    ax2.scatter(x_pos_long, np.array(diff_PEAK_long_all),s=40, edgecolors='0.8', facecolors='none', label='long')
    x_pos_long = np.full_like(diff_PEAK_long_all_trend,0) + (np.random.randn(len(diff_PEAK_long_all_trend)) * 0.075) + 1
    ax2.scatter(x_pos_long, np.array(diff_PEAK_long_all_trend),s=40, edgecolors=LONG_COLOR, facecolors='none', label='long')
    x_pos_long = np.full_like(diff_PEAK_long_all_sig,0) + (np.random.randn(len(diff_PEAK_long_all_sig)) * 0.075) + 1
    ax2.scatter(x_pos_long, np.array(diff_PEAK_long_all_sig),s=40, edgecolors=LONG_COLOR, facecolors=LONG_COLOR, label='long')

    x_pos_short = np.full_like(zscore_PEAK_short_all,0) + (np.random.randn(len(zscore_PEAK_short_all)) * 0.075)
    ax5.scatter(x_pos_short, np.array(zscore_PEAK_short_all),s=40, edgecolors=SHORT_COLOR, facecolors='none', label='long')

    x_pos_long = np.full_like(zscore_PEAK_long_all,0) + (np.random.randn(len(zscore_PEAK_long_all)) * 0.075) + 1
    ax5.scatter(x_pos_long, np.array(zscore_PEAK_long_all),s=40, edgecolors=LONG_COLOR, facecolors='none', label='long')

    # ipdb.set_trace()
    print('NUMBER SIGNIFICANT: ' + str(num_sig))
    print('NUMBER TRENDING: ' + str(num_trend))
    print('NUMBERS TASK ACTIVE TOTAL: ' + str(np.sum(num_rois_short) + np.sum(num_rois_long)))
    print('FRACTION SIGNIFICANT: ' + str(num_sig/(np.sum(num_rois_short) + np.sum(num_rois_long))))
    print('FRACTION TRENDING: ' + str(num_trend/(np.sum(num_rois_short) + np.sum(num_rois_long))))


    # parts = ax1.boxplot([diff_AUC_short_all,diff_AUC_long_all],
    #     whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
    #     medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
    #     showfliers=False,patch_artist=True,widths=(0.7,0.7),positions=(1,4))
    # set_boxplot_appearance_short(parts)

    plt.tight_layout()
    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    fig.savefig(fname, format=fformat, dpi=300)
    print(fname)

def plot_vrol_transient_parameters(roi_param_list_all, fname, sim_data=False):
    """ collect data from .json files and plot """

    # create figure with axes
    fig = plt.figure(figsize=(6,12))
    ax1 = plt.subplot2grid((12,12),(0,0),colspan=4,rowspan=4)
    ax2 = plt.subplot2grid((12,12),(0,4),colspan=4,rowspan=4)
    ax3 = plt.subplot2grid((12,12),(0,8),colspan=4,rowspan=4)
    ax4 = plt.subplot2grid((12,12),(4,0),colspan=4,rowspan=4)
    ax5 = plt.subplot2grid((12,12),(4,4),colspan=4,rowspan=4)
    ax6 = plt.subplot2grid((12,12),(4,8),colspan=4,rowspan=4)
    ax7 = plt.subplot2grid((12,12),(8,0),colspan=4,rowspan=4)
    ax8 = plt.subplot2grid((12,12),(8,4),colspan=4,rowspan=4)
    ax9 = plt.subplot2grid((12,12),(8,8),colspan=4,rowspan=4)
    set_empty_axis(ax1)
    set_empty_axis(ax2)
    set_empty_axis(ax3)
    set_empty_axis(ax4)
    set_empty_axis(ax5)
    set_empty_axis(ax6)
    set_empty_axis(ax7)
    set_empty_axis(ax8)
    set_empty_axis(ax9)


    num_rois_short = []
    num_rois_long = []

    # initialize arrays that will hold data of all sessions
    trial_deviance_all_short = []
    trial_deviance_all_long = []
    trial_deviance_all_short_ol = []
    trial_deviance_all_long_ol = []

    filter_1_trial_deviance_all_short = []
    filter_1_trial_deviance_all_long = []
    filter_1_trial_deviance_all_short_ol = []
    filter_1_trial_deviance_all_long_ol = []

    filter_2_trial_deviance_all_short = []
    filter_2_trial_deviance_all_long = []
    filter_2_trial_deviance_all_short_ol = []
    filter_2_trial_deviance_all_long_ol = []

    num_RF_transients_short = []
    num_RF_transients_long = []
    num_RF_transients_short_ol = []
    num_RF_transients_long_ol = []

    filter_1_num_RF_transients_short = []
    filter_1_num_RF_transients_long = []
    filter_1_num_RF_transients_short_ol = []
    filter_1_num_RF_transients_long_ol = []

    filter_2_num_RF_transients_short = []
    filter_2_num_RF_transients_long = []
    filter_2_num_RF_transients_short_ol = []
    filter_2_num_RF_transients_long_ol = []

    meanpeak_transient_all_short = []
    meanpeak_transient_all_long = []
    meanpeak_transient_all_short_ol = []
    meanpeak_transient_all_long_ol = []

    filter_1_meanpeak_transient_all_short = []
    filter_1_meanpeak_transient_all_long = []
    filter_1_meanpeak_transient_all_short_ol = []
    filter_1_meanpeak_transient_all_long_ol = []

    filter_2_meanpeak_transient_all_short = []
    filter_2_meanpeak_transient_all_long = []
    filter_2_meanpeak_transient_all_short_ol = []
    filter_2_meanpeak_transient_all_long_ol = []


    # loop through all .json files
    for rpl in roi_param_list_all:
        with open(rpl[0],'r') as f:
            roi_params = json.load(f)

        if sim_data == False:
            trialonset_rois = get_eventaligned_rois(roi_params, 'trialonset')
            lmcenter_rois = get_eventaligned_rois(roi_params, 'lmcenter')
            reward_rois = get_eventaligned_rois(roi_params, 'reward')
        else:
            # ipdb.set_trace()
            valid_rois = roi_params['valid_rois']
            lmcenter_rois = {
                'lmcenter_valid_number_short' : valid_rois,
                'lmcenter_roi_number_short' : valid_rois,
                'lmcenter_valid_number_long' : valid_rois,
                'lmcenter_roi_number_long' : valid_rois
            }
            trialonset_rois = {
                'trialonset_valid_number_short' : [],
                'trialonset_roi_number_short' : [],
                'trialonset_valid_number_long' : [],
                'trialonset_roi_number_long' : []
            }
            reward_rois = {
                'reward_valid_number_short' : [],
                'reward_roi_number_short' : [],
                'reward_valid_number_long' : [],
                'reward_roi_number_long' : []
            }

        included_rois_short = [i for i in lmcenter_rois['lmcenter_valid_number_short']]
        included_rois_short.extend([i for i in trialonset_rois['trialonset_valid_number_short']])
        included_rois_short.extend([i for i in reward_rois['reward_valid_number_short']])

        included_rois_long = [i for i in lmcenter_rois['lmcenter_valid_number_long']]
        included_rois_long.extend([i for i in trialonset_rois['trialonset_valid_number_long']])
        included_rois_long.extend([i for i in reward_rois['reward_valid_number_long']])

        num_rois_short.append(len(included_rois_short))
        num_rois_long.append(len(included_rois_long))

        # append data for each individual session
        trial_deviance_all_short.extend(roi_params['space_meanpeak_trial_deviance_short'][i] for i in included_rois_short)
        trial_deviance_all_short_ol.extend(roi_params['space_meanpeak_trial_deviance_short_ol'][i] for i in  included_rois_short)
        trial_deviance_all_long.extend(roi_params['space_meanpeak_trial_deviance_long'][i] for i in included_rois_long)
        trial_deviance_all_long_ol.extend(roi_params['space_meanpeak_trial_deviance_long_ol'][i] for i in  included_rois_long)

        filter_1_trial_deviance_all_short.extend(roi_params['space_filter_1_meanpeak_trial_deviance_short'][i] for i in included_rois_short)
        filter_1_trial_deviance_all_long.extend(roi_params['space_filter_1_meanpeak_trial_deviance_long'][i] for i in included_rois_long)
        filter_2_trial_deviance_all_short.extend(roi_params['space_filter_2_meanpeak_trial_deviance_short'][i] for i in included_rois_short)
        filter_2_trial_deviance_all_long.extend(roi_params['space_filter_2_meanpeak_trial_deviance_long'][i] for i in included_rois_long)

        filter_1_trial_deviance_all_short_ol.extend(roi_params['space_filter_1_meanpeak_trial_deviance_short_ol'][i] for i in  included_rois_short)
        filter_1_trial_deviance_all_long_ol.extend(roi_params['space_filter_1_meanpeak_trial_deviance_long_ol'][i] for i in  included_rois_long)
        filter_2_trial_deviance_all_short_ol.extend(roi_params['space_filter_2_meanpeak_trial_deviance_short_ol'][i] for i in  included_rois_short)
        filter_2_trial_deviance_all_long_ol.extend(roi_params['space_filter_2_meanpeak_trial_deviance_long_ol'][i] for i in  included_rois_long)

        num_RF_transients_short.extend(roi_params['space_meanpeak_num_RF_transients_short'][i] for i in included_rois_short)
        num_RF_transients_short_ol.extend(roi_params['space_meanpeak_num_RF_transients_short_ol'][i] for i in  included_rois_short)
        num_RF_transients_long.extend(roi_params['space_meanpeak_num_RF_transients_long'][i] for i in included_rois_long)
        num_RF_transients_long_ol.extend(roi_params['space_meanpeak_num_RF_transients_long_ol'][i] for i in  included_rois_long)

        filter_1_num_RF_transients_short.extend(roi_params['space_filter_1_meanpeak_num_RF_transients_short'][i] for i in included_rois_short)
        filter_2_num_RF_transients_short.extend(roi_params['space_filter_2_meanpeak_num_RF_transients_short'][i] for i in included_rois_short)
        filter_1_num_RF_transients_long.extend(roi_params['space_filter_1_meanpeak_num_RF_transients_long'][i] for i in included_rois_long)
        filter_2_num_RF_transients_long.extend(roi_params['space_filter_2_meanpeak_num_RF_transients_long'][i] for i in included_rois_long)

        filter_1_num_RF_transients_short_ol.extend(roi_params['space_filter_1_meanpeak_num_RF_transients_short_ol'][i] for i in  included_rois_short)
        filter_1_num_RF_transients_long_ol.extend(roi_params['space_filter_1_meanpeak_num_RF_transients_long_ol'][i] for i in  included_rois_long)
        filter_2_num_RF_transients_short_ol.extend(roi_params['space_filter_2_meanpeak_num_RF_transients_short_ol'][i] for i in  included_rois_short)
        filter_2_num_RF_transients_long_ol.extend(roi_params['space_filter_2_meanpeak_num_RF_transients_long_ol'][i] for i in  included_rois_long)

        meanpeak_transient_all_short.extend(roi_params['space_meanpeak_transient_meanpeak_short'][i] for i in included_rois_short)
        meanpeak_transient_all_short_ol.extend(roi_params['space_meanpeak_transient_meanpeak_short_ol'][i] for i in  included_rois_short)
        meanpeak_transient_all_long.extend(roi_params['space_meanpeak_transient_meanpeak_long'][i] for i in included_rois_long)
        meanpeak_transient_all_long_ol.extend(roi_params['space_meanpeak_transient_meanpeak_long_ol'][i] for i in  included_rois_long)

        filter_1_meanpeak_transient_all_short.extend(roi_params['space_filter_1_meanpeak_transient_meanpeak_short'][i] for i in included_rois_short)
        filter_1_meanpeak_transient_all_long.extend(roi_params['space_filter_1_meanpeak_transient_meanpeak_long'][i] for i in included_rois_long)
        filter_2_meanpeak_transient_all_short.extend(roi_params['space_filter_2_meanpeak_transient_meanpeak_short'][i] for i in included_rois_short)
        filter_2_meanpeak_transient_all_long.extend(roi_params['space_filter_2_meanpeak_transient_meanpeak_long'][i] for i in included_rois_long)

        filter_1_meanpeak_transient_all_short_ol.extend(roi_params['space_filter_1_meanpeak_transient_meanpeak_short_ol'][i] for i in  included_rois_short)
        filter_1_meanpeak_transient_all_long_ol.extend(roi_params['space_filter_1_meanpeak_transient_meanpeak_long_ol'][i] for i in  included_rois_long)
        filter_2_meanpeak_transient_all_short_ol.extend(roi_params['space_filter_2_meanpeak_transient_meanpeak_short_ol'][i] for i in  included_rois_short)
        filter_2_meanpeak_transient_all_long_ol.extend(roi_params['space_filter_2_meanpeak_transient_meanpeak_long_ol'][i] for i in  included_rois_long)

    # convert everything to numpy arrays for convenience
    trial_deviance_all_short = np.array(trial_deviance_all_short)
    trial_deviance_all_short_ol = np.array(trial_deviance_all_short_ol)
    trial_deviance_all_long = np.array(trial_deviance_all_long)
    trial_deviance_all_long_ol = np.array(trial_deviance_all_long_ol)

    filter_1_trial_deviance_all_short = np.array(filter_1_trial_deviance_all_short)
    filter_1_trial_deviance_all_short_ol = np.array(filter_1_trial_deviance_all_short_ol)
    filter_1_trial_deviance_all_long = np.array(filter_1_trial_deviance_all_long)
    filter_1_trial_deviance_all_long_ol = np.array(filter_1_trial_deviance_all_long_ol)

    filter_2_trial_deviance_all_short = np.array(filter_2_trial_deviance_all_short)
    filter_2_trial_deviance_all_short_ol = np.array(filter_2_trial_deviance_all_short_ol)
    filter_2_trial_deviance_all_long = np.array(filter_2_trial_deviance_all_long)
    filter_2_trial_deviance_all_long_ol = np.array(filter_2_trial_deviance_all_long_ol)

    num_RF_transients_short = np.array(num_RF_transients_short)
    num_RF_transients_short_ol = np.array(num_RF_transients_short_ol)
    num_RF_transients_long = np.array(num_RF_transients_long)
    num_RF_transients_long_ol = np.array(num_RF_transients_long_ol)

    filter_1_num_RF_transients_short = np.array(filter_1_num_RF_transients_short)
    filter_1_num_RF_transients_short_ol = np.array(filter_1_num_RF_transients_short_ol)
    filter_1_num_RF_transients_long = np.array(filter_1_num_RF_transients_long)
    filter_1_num_RF_transients_long_ol = np.array(filter_1_num_RF_transients_long_ol)

    filter_2_num_RF_transients_short = np.array(filter_2_num_RF_transients_short)
    filter_2_num_RF_transients_short_ol = np.array(filter_2_num_RF_transients_short_ol)
    filter_2_num_RF_transients_long = np.array(filter_2_num_RF_transients_long)
    filter_2_num_RF_transients_long_ol = np.array(filter_2_num_RF_transients_long_ol)

    meanpeak_transient_all_short = np.array(meanpeak_transient_all_short)
    meanpeak_transient_all_short_ol = np.array(meanpeak_transient_all_short_ol)
    meanpeak_transient_all_long = np.array(meanpeak_transient_all_long)
    meanpeak_transient_all_long_ol = np.array(meanpeak_transient_all_long_ol)

    filter_1_meanpeak_transient_all_short = np.array(filter_1_meanpeak_transient_all_short)
    filter_1_meanpeak_transient_all_short_ol = np.array(filter_1_meanpeak_transient_all_short_ol)
    filter_1_meanpeak_transient_all_long = np.array(filter_1_meanpeak_transient_all_long)
    filter_1_meanpeak_transient_all_long_ol = np.array(filter_1_meanpeak_transient_all_long_ol)

    filter_2_meanpeak_transient_all_short = np.array(filter_2_meanpeak_transient_all_short)
    filter_2_meanpeak_transient_all_short_ol = np.array(filter_2_meanpeak_transient_all_short_ol)
    filter_2_meanpeak_transient_all_long = np.array(filter_2_meanpeak_transient_all_long)
    filter_2_meanpeak_transient_all_long_ol = np.array(filter_2_meanpeak_transient_all_long_ol)

    parts = ax1.boxplot([trial_deviance_all_short[~np.isnan(trial_deviance_all_short)],trial_deviance_all_short_ol[~np.isnan(trial_deviance_all_short_ol)]],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7),positions=(1,4))
    set_boxplot_appearance_short(parts)

    # ipdb.set_trace()

    parts = ax1.boxplot([trial_deviance_all_long[~np.isnan(trial_deviance_all_long)],trial_deviance_all_long_ol[~np.isnan(trial_deviance_all_long_ol)]],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7),positions=(2,5))
    set_boxplot_appearance_long(parts)

    parts = ax4.boxplot([np.concatenate((filter_2_trial_deviance_all_short_ol[~np.isnan(filter_2_trial_deviance_all_short_ol)],filter_2_trial_deviance_all_long_ol[~np.isnan(filter_2_trial_deviance_all_long_ol)])),
                         np.concatenate((filter_1_trial_deviance_all_short_ol[~np.isnan(filter_1_trial_deviance_all_short_ol)],filter_1_trial_deviance_all_long_ol[~np.isnan(filter_1_trial_deviance_all_long_ol)])),
                         np.concatenate((trial_deviance_all_short[~np.isnan(trial_deviance_all_short)],trial_deviance_all_long[~np.isnan(trial_deviance_all_long)]))],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='0.5', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7,0.7),positions=(1,2,3))
    set_boxplot_appearance_sit_run(parts)



    parts = ax2.boxplot([num_RF_transients_short[~np.isnan(num_RF_transients_short)],num_RF_transients_short_ol[~np.isnan(num_RF_transients_short_ol)]],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7),positions=(1,4))
    set_boxplot_appearance_short(parts)

    parts = ax2.boxplot([num_RF_transients_long[~np.isnan(num_RF_transients_long)],num_RF_transients_long_ol[~np.isnan(num_RF_transients_long_ol)]],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7),positions=(2,5))
    set_boxplot_appearance_long(parts)

    parts = ax5.boxplot([np.concatenate((filter_2_num_RF_transients_short_ol[~np.isnan(filter_2_num_RF_transients_short_ol)],filter_2_num_RF_transients_long_ol[~np.isnan(filter_2_num_RF_transients_long_ol)])),
                         np.concatenate((filter_1_num_RF_transients_short_ol[~np.isnan(filter_1_num_RF_transients_short_ol)],filter_1_num_RF_transients_long_ol[~np.isnan(filter_1_num_RF_transients_long_ol)])),
                         np.concatenate((num_RF_transients_short[~np.isnan(num_RF_transients_short)],num_RF_transients_long[~np.isnan(num_RF_transients_long)]))],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='0.5', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7,0.7),positions=(1,2,3))
    set_boxplot_appearance_sit_run(parts)



    parts = ax3.boxplot([meanpeak_transient_all_short[~np.isnan(meanpeak_transient_all_short)],meanpeak_transient_all_short_ol[~np.isnan(meanpeak_transient_all_short_ol)]],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7),positions=(1,4))
    set_boxplot_appearance_short(parts)

    parts = ax3.boxplot([meanpeak_transient_all_long[~np.isnan(meanpeak_transient_all_long)],meanpeak_transient_all_long_ol[~np.isnan(meanpeak_transient_all_long_ol)]],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7),positions=(2,5))
    set_boxplot_appearance_long(parts)

    parts = ax6.boxplot([np.concatenate((filter_2_meanpeak_transient_all_short_ol[~np.isnan(filter_2_meanpeak_transient_all_short_ol)],filter_2_meanpeak_transient_all_long_ol[~np.isnan(filter_2_meanpeak_transient_all_long_ol)])),
                         np.concatenate((filter_1_meanpeak_transient_all_short_ol[~np.isnan(filter_1_meanpeak_transient_all_short_ol)],filter_1_meanpeak_transient_all_long_ol[~np.isnan(filter_1_meanpeak_transient_all_long_ol)])),
                         np.concatenate((meanpeak_transient_all_short[~np.isnan(meanpeak_transient_all_short)],meanpeak_transient_all_long[~np.isnan(meanpeak_transient_all_long)]))],
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='0.5', linewidth=2, solid_capstyle='butt'),
        showfliers=False,patch_artist=True,widths=(0.7,0.7,0.7),positions=(1,2,3))
    set_boxplot_appearance_sit_run(parts)

    # parts = ax6.boxplot(,]],
    #     whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
    #     medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
    #     patch_artist=True,widths=(0.7,0.7),positions=(2,5))
    # set_boxplot_appearance_sit_run(parts)

    print('--- KRUSKAL WALLIS VR vs OL ---')
    print(sp.stats.kruskal(trial_deviance_all_short,trial_deviance_all_short_ol,trial_deviance_all_long,trial_deviance_all_long_ol,
            num_RF_transients_short,num_RF_transients_short_ol,num_RF_transients_long,num_RF_transients_long_ol,
            meanpeak_transient_all_short,meanpeak_transient_all_short_ol,meanpeak_transient_all_long,meanpeak_transient_all_long_ol,nan_policy='omit'))

    _, p_dv_short = sp.stats.mannwhitneyu(trial_deviance_all_short[~np.isnan(trial_deviance_all_short)],trial_deviance_all_short_ol[~np.isnan(trial_deviance_all_short_ol)])
    _, p_dv_long = sp.stats.mannwhitneyu(trial_deviance_all_long[~np.isnan(trial_deviance_all_long)],trial_deviance_all_long_ol[~np.isnan(trial_deviance_all_long_ol)])
    _, p_numRF_short = sp.stats.mannwhitneyu(num_RF_transients_short[~np.isnan(num_RF_transients_short)],num_RF_transients_short_ol[~np.isnan(num_RF_transients_short_ol)])
    _, p_numRF_long = sp.stats.mannwhitneyu(num_RF_transients_long[~np.isnan(num_RF_transients_long)],num_RF_transients_long_ol[~np.isnan(num_RF_transients_long_ol)])
    _, p_meanpeak_short = sp.stats.mannwhitneyu(meanpeak_transient_all_short[~np.isnan(meanpeak_transient_all_short)],meanpeak_transient_all_short_ol[~np.isnan(meanpeak_transient_all_short_ol)])
    _, p_meanpeak_long = sp.stats.mannwhitneyu(meanpeak_transient_all_long[~np.isnan(meanpeak_transient_all_long)],meanpeak_transient_all_long_ol[~np.isnan(meanpeak_transient_all_long_ol)])

    p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_dv_short,p_dv_long,p_numRF_short,p_numRF_long,p_meanpeak_short,p_meanpeak_long],alpha=0.05,method='bonferroni')

    # ipdb.set_trace()

    print('CORRECTED P-VALUES:')
    print('short medians dv: ', str(np.round(np.nanmedian(trial_deviance_all_short),2)), ' ', str(np.round(np.nanmedian(trial_deviance_all_short_ol),2)))
    print('long medians dv: ', str(np.round(np.nanmedian(trial_deviance_all_long),2)), ' ', str(np.round(np.nanmedian(trial_deviance_all_long_ol),2)))
    print('short medians numRF: ', str(np.round(np.nanmedian(num_RF_transients_short),2)), ' ', str(np.round(np.nanmedian(num_RF_transients_short_ol),2)))
    print('long medians numRF: ', str(np.round(np.nanmedian(num_RF_transients_long),2)), ' ', str(np.round(np.nanmedian(num_RF_transients_long_ol),2)))
    print('short medians meanpeak: ', str(np.round(np.nanmedian(meanpeak_transient_all_short),2)), ' ', str(np.round(np.nanmedian(meanpeak_transient_all_short_ol),2)))
    print('long medians meanpeak: ', str(np.round(np.nanmedian(meanpeak_transient_all_long),2)), ' ', str(np.round(np.nanmedian(meanpeak_transient_all_long_ol),2)))
    print('dv vs dv OL short: ' + str(p_corrected[1][0]))
    print('dv vs dv OL long: ' + str(p_corrected[1][1]))
    print('numRF vs numRF short OL: ' + str(p_corrected[1][2]))
    print('numRF vs numRF long OL: ' + str(p_corrected[1][3]))
    print('meanpeak vs meanpeak OL short: ' + str(p_corrected[1][4]))
    print('meanpeak vs meanpeak OL long: ' + str(p_corrected[1][5]))
    print('--------------------')
    print('--------------------')

    print('--- KRUSKAL WALLIS SIT vs RUN ---')
    # print(sp.stats.kruskal(filter_2_trial_deviance_all_short_ol,filter_1_trial_deviance_all_short_ol,filter_2_trial_deviance_all_long_ol,filter_1_trial_deviance_all_long_ol,
    #         filter_2_num_RF_transients_short_ol,filter_1_num_RF_transients_short_ol,filter_2_num_RF_transients_long_ol,filter_1_num_RF_transients_long_ol,
    #         filter_2_meanpeak_transient_all_short_ol,filter_1_meanpeak_transient_all_short_ol,filter_2_meanpeak_transient_all_long_ol,filter_1_meanpeak_transient_all_long_ol,nan_policy='omit'))

    # ipdb.set_trace()
    f2td = np.concatenate((filter_2_trial_deviance_all_short_ol,filter_2_trial_deviance_all_long_ol))
    f1td = np.concatenate((filter_1_trial_deviance_all_short_ol,filter_1_trial_deviance_all_long_ol))
    td = np.concatenate((trial_deviance_all_short, trial_deviance_all_long))
    print(sp.stats.kruskal(f2td,f1td,td,nan_policy='omit'))
    _, p_f2f1 = sp.stats.mannwhitneyu(f2td[~np.isnan(f2td)],f1td[~np.isnan(f1td)])
    _, p_f2td = sp.stats.mannwhitneyu(f2td[~np.isnan(f2td)],td[~np.isnan(td)])
    _, p_f1td = sp.stats.mannwhitneyu(f1td[~np.isnan(f1td)],td[~np.isnan(td)])
    p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_f2f1,p_f2td,p_f1td],alpha=0.05,method='bonferroni')
    print('median jitter sit: ' + str(np.nanmedian(f2td)))
    print('median jitter run: ' + str(np.nanmedian(f1td)))
    print('median jitter vr: ' + str(np.nanmedian(td)))
    print('jitter f2 vs. f1: '  + str(p_corrected[1][0]))
    print('jitter f2 vs. td: '  + str(p_corrected[1][1]))
    print('jitter f1 vs. td: '  + str(p_corrected[1][2]))

    f2nr = np.concatenate((filter_2_num_RF_transients_short_ol,filter_2_num_RF_transients_long_ol))
    f1nr = np.concatenate((filter_1_num_RF_transients_short_ol,filter_1_num_RF_transients_long_ol))
    nr = np.concatenate((num_RF_transients_short, num_RF_transients_long))
    print(sp.stats.kruskal(f2nr,f1nr,nr,nan_policy='omit'))
    _, p_f2f1 = sp.stats.mannwhitneyu(f2nr[~np.isnan(f2nr)],f1nr[~np.isnan(f1nr)])
    _, p_f2nr = sp.stats.mannwhitneyu(f2nr[~np.isnan(f2nr)],td[~np.isnan(nr)])
    _, p_f1nr = sp.stats.mannwhitneyu(f1nr[~np.isnan(f1nr)],td[~np.isnan(nr)])
    p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_f2f1,p_f2nr,p_f1nr],alpha=0.05,method='bonferroni')
    print('median num RF sit: ' + str(np.nanmedian(f2nr)))
    print('median num RF run: ' + str(np.nanmedian(f1nr)))
    print('median num RF vr: ' + str(np.nanmedian(nr)))
    print('num RF f2 vs. f1: '  + str(p_corrected[1][0]))
    print('num RF f2 vs. td: '  + str(p_corrected[1][1]))
    print('num RF f1 vs. td: '  + str(p_corrected[1][2]))


    f2mp = np.concatenate((filter_2_meanpeak_transient_all_short_ol,filter_2_meanpeak_transient_all_long_ol))
    f1mp = np.concatenate((filter_1_meanpeak_transient_all_short_ol,filter_1_meanpeak_transient_all_long_ol))
    mp = np.concatenate((meanpeak_transient_all_short, meanpeak_transient_all_long))
    print(sp.stats.kruskal(f2mp,f1mp,mp,nan_policy='omit'))
    _, p_f2f1 = sp.stats.mannwhitneyu(f2mp[~np.isnan(f2mp)],f1mp[~np.isnan(f1mp)])
    _, p_f2mp = sp.stats.mannwhitneyu(f2mp[~np.isnan(f2mp)],mp[~np.isnan(mp)])
    _, p_f1mp = sp.stats.mannwhitneyu(f1mp[~np.isnan(f1mp)],mp[~np.isnan(mp)])
    p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_f2f1,p_f2mp,p_f1mp],alpha=0.05,method='bonferroni')
    print('median peak amp sit: ' + str(np.nanmedian(f2mp)))
    print('median peak amp run: ' + str(np.nanmedian(f1mp)))
    print('median peak amp vr: ' + str(np.nanmedian(mp)))
    print('peak f2 vs. f1: '  + str(p_corrected[1][0]))
    print('peak f2 vs. td: '  + str(p_corrected[1][1]))
    print('peak f1 vs. td: '  + str(p_corrected[1][2]))


    # p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_dv_short,p_dv_long,p_numRF_short,p_numRF_long,p_meanpeak_short,p_meanpeak_long],alpha=0.05,method='bonferroni')

    # print('CORRECTED P-VALUES:')
    # print('short medians dv: ', str(np.round(np.nanmedian(filter_2_trial_deviance_all_short_ol),2)), ' ', str(np.round(np.nanmedian(filter_1_trial_deviance_all_short_ol),2)))
    # print('long medians dv: ', str(np.round(np.nanmedian(filter_2_trial_deviance_all_long_ol),2)), ' ', str(np.round(np.nanmedian(filter_1_trial_deviance_all_long_ol),2)))
    # print('short medians numRF: ', str(np.round(np.nanmedian(filter_2_num_RF_transients_short_ol),2)), ' ', str(np.round(np.nanmedian(filter_1_num_RF_transients_short_ol),2)))
    # print('long medians numRF: ', str(np.round(np.nanmedian(filter_2_num_RF_transients_long_ol),2)), ' ', str(np.round(np.nanmedian(filter_1_num_RF_transients_long_ol),2)))
    # print('short medians meanpeak: ', str(np.round(np.nanmedian(filter_2_meanpeak_transient_all_short_ol),2)), ' ', str(np.round(np.nanmedian(filter_1_meanpeak_transient_all_long_ol),2)))
    # print('long medians meanpeak: ', str(np.round(np.nanmedian(filter_2_meanpeak_transient_all_long_ol),2)), ' ', str(np.round(np.nanmedian(filter_1_meanpeak_transient_all_long_ol),2)))
    # print('dv vs dv OL short: ' + str(p_corrected[1][0]))
    # print('dv vs dv OL long: ' + str(p_corrected[1][1]))
    # print('numRF vs numRF short OL: ' + str(p_corrected[1][2]))
    # print('numRF vs numRF long OL: ' + str(p_corrected[1][3]))
    # print('meanpeak vs meanpeak OL short: ' + str(p_corrected[1][4]))
    # print('meanpeak vs meanpeak OL long: ' + str(p_corrected[1][5]))
    # print('--------------------')

    print('--- N-NUMBERS ---')
    print('short: ' + str(sum(num_rois_short)))
    print('long: ' + str(sum(num_rois_long)))
    print('-----------------')

    ax1.set_xlim([0,6])
    ax1.set_xticklabels(['vr', 'ol'])
    ax2.set_xlim([0,6])
    ax2.set_xticklabels(['vr', 'ol'])
    ax3.set_xlim([0,6])
    ax3.set_xticklabels(['vr', 'ol'])

    ax4.set_xlim([0,4])
    ax4.set_xticklabels(['sit', 'run'])
    ax5.set_xlim([0,4])
    ax5.set_xticklabels(['sit', 'run'])
    ax6.set_xlim([0,4])
    ax6.set_xticklabels(['sit', 'run'])
    ax7.set_xlim([0,6])
    ax8.set_xlim([0,6])
    ax9.set_xlim([0,6])

    ax1.set_ylim([0,30])
    ax2.set_ylim([0,1.4])
    ax3.set_ylim([0,7])

    ax4.set_ylim([0,40])
    ax5.set_ylim([0,1.4])
    ax6.set_ylim([0,8])

    ax1.set_ylabel('avg. transient jitter')
    ax4.set_ylabel('avg. transient jitter (sit vs. run)')
    ax2.set_ylabel('avg. number of transients/trial')
    ax5.set_ylabel('avg. number of transients/trial (sit vs. run)')
    ax3.set_ylabel('avg. transient amplitude')
    ax6.set_ylabel('avg. transient amplitude (sit vs. run)')

    plt.tight_layout()
    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    fig.savefig(fname, format=fformat, dpi=300)
    print(fname)


if __name__ == '__main__':
    roi_param_list_all = [
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_1.json'],
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_2.json'],
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_3.json'],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day2017719.json'],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783.json'],
                      ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719.json'],
                      # ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720.json'],
                      ['E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776.json'],
                      ['E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2.json'],
                      ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804.json'],
                      # ['E:\\MTH3_figures\\LF161202_1\\LF161202_1_Day20170209_l23.json'],
                      # ['E:\\MTH3_figures\\LF161202_1\\LF161202_1_Day20170209_l5.json']
                     ]

    # plot_vrol_transient_parameters(roi_param_list_all, 'vrol_transients_all')
    # plot_goodvbad_activity(roi_param_list_all, 'goodvbad')

    roi_param_list_all = [
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783.json'],
                      ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719.json'],
                      # ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720.json'],
                      ['E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776.json'],
                      ['E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2.json'],
                      ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804.json'],
                     ]

    plot_vrol_transient_parameters(roi_param_list_all, 'vrol_transients_svr')

    # roi_param_list_sim = [
    #                 ['E:\\MTH3_figures\\SIM_1\\SIM_1_Day1jitter.json']
    #                 ]
    # plot_vrol_transient_parameters(roi_param_list_sim, 'vrol_transients_sim_jitter', True)
    # roi_param_list_sim = [
    #                 ['E:\\MTH3_figures\\SIM_1\\SIM_1_Day1wide_RF.json']
    #                 ]
    # plot_vrol_transient_parameters(roi_param_list_sim, 'vrol_transients_sim_wide_RF', True)
