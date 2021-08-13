"""
scatterplot of ROI amplitudes in VR and openloop.

v 2.0 is written to accomodate new .json dictionary structure where all results of a session are in one file

@author: lukasfischer

"""

import h5py, os, sys, traceback, matplotlib, json, yaml, warnings
import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels as sm_all
import ipdb
from matplotlib import pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_style("white")
with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')
from analysis_parameters import MIN_FRACTION_ACTIVE, MIN_MEAN_AMP, MIN_ZSCORE, MIN_TRIALS_ACTIVE, MIN_DF, MIN_MEAN_AMP_BOUTONS


fformat = 'svg'

DO_OL = True

def roi_response_validation(roi_params, tl, el, roi_idx_num):
    """
    separate function to check whether a given response passes criterion for being considered a real roi_response_validation

    """

    roi_activity = roi_params[el + '_active_' + tl][roi_idx_num]
    roi_peak_val = roi_params[el + '_peak_' + tl][roi_idx_num]
    roi_zscore_val = roi_params[el + '_peak_zscore_' + tl][roi_idx_num]
    mean_trace = roi_params['space_mean_trace_'+tl][roi_idx_num]
    # roi_activity = el + '_active_' + tl
    # roi_peak_val = el + '_peak_' + tl
    # if roi_params[roi_activity][roi_idx_num] > MIN_TRIALS_ACTIVE and roi_params[roi_peak_val][roi_idx_num] > MIN_DF:
    #     return True
    # else:
    #     return False
    if plot_layer is 'v1':
        mean_amp_diff = MIN_MEAN_AMP_BOUTONS
    else:
        mean_amp_diff = MIN_MEAN_AMP


    if roi_activity > MIN_FRACTION_ACTIVE and roi_zscore_val > MIN_ZSCORE and (np.nanmax(mean_trace) - np.nanmin(mean_trace)) > mean_amp_diff:
        return True
    else:
        return False


def event_maxresponse(roi_param_list_all, event_list, trialtypes, peak_metric, ax_object, ax_object2, ax_object3, ax_object4, ax_object5, ax_object6, ax_object7, ax_object8, ax_object9, ax_object10, ax_object11, ax_object12, ax_object13, ax_object14, ax_object15, ax_object16, ax_object17, ax_object18, ax_object19, ax_object20, ax_object21, ax_object22, ax_object23, ax_object24, ax_object25, normalize=False, plot_layer='all'):
    """
    determine which alignment gave the max mean response

    roi_param_list : list of roi param files (fulle file paths)

    event_list : list of alignments available

    """
    # flag any suspicious rois and remove them from further analysis
    roi_error = False

    # separate out the path to the json file from the rest of the list which also contains animal and session names
    roi_param_list = []
    for rpl_all in roi_param_list_all:
        roi_param_list.append(rpl_all[0])
    # when we normalize, values range from 0-1, when we don't, we want to accommodate slightly below 0 values
    if normalize:
        min_xy = 0
    else:
        min_xy = -0.2

    # dictionary to hold counters of peak response per alignment point
    result_counter = {}

    for i in range(len(roi_param_list)):
        result_counter['num_valid_rois' + str(i)] = 0
        for tl in trialtypes:
            result_counter['roicounter_' + tl + str(i)] = 0
            result_counter['roinumlist_' + tl + str(i)] = []
            # keep track of peak values in VR vs OL broken down per animal
            result_counter['roipeak_' + tl + str(i)] = []
            result_counter['roipeak_ol_' + tl + str(i)] = []
            for el in event_list:
                # count the number of neurons with peaks at a given alignment point
                result_counter[el + '_peakcounter_' + tl + str(i)] = 0
                result_counter[el + '_numrois_' + tl + str(i)] = 0

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
            result_max_peak[el + '_peakloc_' + tl] = []
            result_max_peak[el + '_active_' + tl] = []
            result_max_peak[el + '_meanpeak_' + tl] = []
            result_max_peak_ol[el + '_peakval_' + tl] = []
            result_max_peak_ol[el + '_peakloc_' + tl] = []
            result_max_peak_ol[el + '_active_' + tl] = []
            result_max_peak_ol[el + '_meanpeak_' + tl] = []
            result_trialonset_max_peak[el + '_peakval_' + tl] = []
            result_lmcenter_max_peak[el + '_peakval_' + tl] = []
            result_matching_landmark_long[el + '_peakval_' + tl] = []
            result_matching_landmark_short[el + '_peakval_' + tl] = []
            result_max_peak['space_peak_' + tl] = []
            result_max_peak['space_peak_' + tl + '_' + el] = []
            result_max_peak['space_speedcorr_' + tl] = []
            result_max_peak['space_speedcorr_' + tl + '_' + el] = []
            result_max_peak['space_speedcorr_p_' + tl] = []
            result_max_peak['space_speedcorr_p_' + tl + '_' + el] = []
            result_max_peak['space_speedcorr_r_' + tl] = []
            result_max_peak['space_speedcorr_r_' + tl + '_' + el] = []
            result_max_peak['fl_distamp_r_' + tl] = []
            result_max_peak['fl_distamp_r_' + tl + '_' + el] = []
            result_max_peak['fl_distamp_p_' + tl] = []
            result_max_peak['fl_distamp_p_' + tl + '_' + el] = []
            result_max_peak['fl_amplitude_close_' + tl] = []
            result_max_peak['fl_amplitude_far_' + tl] = []
            result_max_peak['success_robustness_' + tl] = []
            result_max_peak['success_robustness_' + tl + '_' + el] = []
            result_max_peak['default_robustness_' + tl] = []
            result_max_peak['default_robustness_' + tl + '_' + el] = []
            result_max_peak['success_amplitude_' + tl] = []
            result_max_peak['success_amplitude_' + tl + '_' + el] = []
            result_max_peak['default_amplitude_' + tl] = []
            result_max_peak['default_amplitude_' + tl + '_' + el] = []
            result_max_peak_ol['space_peak_' + tl] = []
            result_max_peak_ol['space_peak_' + tl + '_' + el] = []
            result_max_peak_ol['space_speedcorr_' + tl] = []
            result_max_peak_ol['space_speedcorr_' + tl + '_' + el] = []
            result_max_peak_ol['space_speedcorr_p_' + tl] = []
            result_max_peak_ol['space_speedcorr_p_' + tl + '_' + el] = []
            result_max_peak_ol['space_speedcorr_r_' + tl] = []
            result_max_peak_ol['space_speedcorr_r_' + tl + '_' + el] = []
            result_max_peak_ol['fl_distamp_r_' + tl] = []
            result_max_peak_ol['fl_distamp_r_' + tl + '_' + el] = []
            result_max_peak_ol['fl_distamp_p_' + tl] = []
            result_max_peak_ol['fl_distamp_p_' + tl + '_' + el] = []
            result_max_peak_ol['fl_amplitude_close_' + tl] = []
            result_max_peak_ol['fl_amplitude_far_' + tl] = []
            result_max_peak_ol['success_robustness_' + tl] = []
            result_max_peak_ol['success_robustness_' + tl + '_' + el] = []

    # differences in peak amplitude between landmarks
    result_matching_landmark_normdiff = {
        'lm_diff_short' : [],
        'lm_diff_long' : [],
        'to_diff_short' : [],
        'to_diff_long' : [],
        'rw_diff_short' : [],
        'rw_diff_long' : []
    }

    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        print(rpl)
        # load roi parameters for given session
        with open(rpl,'r') as f:
            roi_params = json.load(f)

        # grab roi numbers to be included
        if len(roi_param_list_all[i]) > 4:
            custom_roilist = roi_param_list_all[i][4]
        else:
            custom_roilist = None

        if custom_roilist is None:
            roi_list_all = roi_params['valid_rois']
            # norm_list_all = roi_params['norm_value']
            result_counter['num_valid_rois' + str(i)] = len(roi_list_all)
            print(len(roi_list_all))
        else:
            roi_list_all = custom_roilist
            result_counter['num_valid_rois' + str(i)] = len(custom_roilist)
            print(len(roi_list_all))

        # loop through every roi
        for j,r in enumerate(roi_list_all):
            if normalize:
                roi_normalization_value = roi_params['norm_value'][j]
                # print(r, len(roi_params['norm_value']))
            else:
                roi_normalization_value = 1
            # loop through every trialtype and alignment point to determine largest response
            for tl in trialtypes:
                roi_error = False
                valid = False
                max_peak = 0
                mean_peak_loc = 0
                max_peak_ol = 0
                peak_trialonset = 0
                peak_trialonset_activity = 0
                peak_lm_long = 0
                peak_lm_short = 0
                peak_lm_time_short = 0
                peak_lm_time_long = 0
                peak_lm_normdiff = 0
                peak_lm_type = ''
                peak_to_type = ''
                peak_rw_type = ''
                peak_event = ''
                peak_trialtype = ''
                for el in event_list:
                    # create dictionary keys depending on which metric is being used (mean dF/F or zscore)
                    value_key = el + peak_metric + tl
                    value_key_ol = el + peak_metric + tl + '_ol'
                    space_value_key = 'space_peak_' + tl
                    space_value_key_ol = 'space_peak_' + tl + '_ol'
                    speedcorr_value_key = 'space_transient_speed_slope_' + tl
                    speedcorr_value_key_ol = 'space_transient_speed_slope_' + tl + '_ol'
                    speedcorr_r_value_key = 'space_transient_speed_pearsonr_' + tl
                    speedcorr_r_value_key_ol = 'space_transient_speed_pearsonr_' + tl + '_ol'
                    speedcorr_p_value_key = 'space_transient_speed_pearsonp_' + tl
                    speedcorr_p_value_key_ol = 'space_transient_speed_pearsonp_' + tl + '_ol'
                    fl_distamp_r_value_key = 'space_fl_distamp_pearsonr_'  + tl
                    fl_distamp_r_value_key_ol = 'space_fl_distamp_pearsonr_'  + tl + '_ol'
                    fl_distamp_p_value_key = 'space_fl_distamp_pearsonp_'  + tl
                    fl_distamp_p_value_key_ol = 'space_fl_distamp_pearsonp_'  + tl + '_ol'
                    fl_amplitude_close_key = 'space_fl_amplitude_close_' + tl
                    fl_amplitude_close_key_ol = 'space_fl_amplitude_close_' + tl + '_ol'
                    fl_amplitude_far_key = 'space_fl_amplitude_far_' + tl
                    fl_amplitude_far_key_ol = 'space_fl_amplitude_far_' + tl + '_ol'
                    success_robustness_value_key = 'space_success_robustness_'  + tl
                    success_robustness_value_key_ol = 'space_success_robustness_'  + tl + '_ol'
                    default_robustness_value_key = 'space_default_robustness_'  + tl
                    default_robustness_value_key_ol = 'space_default_robustness_'  + tl + '_ol'
                    success_amplitude_value_key = 'space_success_amplitude_'  + tl
                    success_amplitude_value_key_ol = 'space_success_amplitude_'  + tl + '_ol'
                    default_amplitude_value_key = 'space_default_amplitude_'  + tl
                    default_amplitude_value_key_ol = 'space_default_amplitude_'  + tl + '_ol'

                    # success_robustness
                    # space_value_key = 'space_filter_1_peak_' + tl
                    # space_value_key_ol = 'space_filter_1_peak_' + tl + '_ol'
                    if peak_metric is '_peak_':
                        value_key_trace = el + '_mean_trace_' +  tl
                        value_key_trace_ol = el + '_active_' + tl + '_ol'
                    else:
                        value_key_trace = el + '_mean_trace_' + tl
                        value_key_trace_ol = el + '_mean_trace_' + tl + '_ol'

                    if roi_params[value_key][j]/roi_normalization_value > max_peak and roi_response_validation(roi_params, tl, el, j):
                        valid = True
                        max_peak = roi_params[value_key][j]/roi_normalization_value
                        mean_peak = np.amax(np.array(roi_params[value_key_trace][j]))/roi_normalization_value
                        mean_peak_loc = np.nanmedian(np.array(roi_params[el + '_transient_max_loc_' + tl][j]))
                        space_max_peak = roi_params[space_value_key][j]/roi_normalization_value
                        speedcorr_slope = roi_params[speedcorr_value_key][j]
                        speedcorr_r = roi_params[speedcorr_r_value_key][j]
                        speedcorr_p = roi_params[speedcorr_p_value_key][j]
                        # print(roi_params[fl_distamp_r_value_key])
                        fl_distamp_r = roi_params[fl_distamp_r_value_key][j]
                        fl_distamp_p = roi_params[fl_distamp_p_value_key][j]

                        fl_amplitude_close = np.divide(roi_params[fl_amplitude_close_key][j],roi_normalization_value)
                        fl_amplitude_far = np.divide(roi_params[fl_amplitude_far_key][j],roi_normalization_value)
                        success_robustness = roi_params[success_robustness_value_key][j]
                        default_robustness = roi_params[default_robustness_value_key][j]
                        success_amplitude = roi_params[success_amplitude_value_key][j]
                        default_amplitude = roi_params[default_amplitude_value_key][j]
                        if space_max_peak > 8:
                            mean_peak = 0
                            roi_error = True
                        max_active = roi_params[el + '_active_' + tl][j]
                        # check if openloop values exist, if they don't exist (for sessions were no openloop was run), just return nan
                        try:
                            max_peak_ol = roi_params[value_key_ol][j]/roi_normalization_value
                            max_active_ol = roi_params[el + '_active_' + tl + '_ol'][j]
                            space_max_peak_ol = roi_params[space_value_key_ol][j]/roi_normalization_value
                            speedcorr_slope_ol = roi_params[speedcorr_value_key_ol][j]
                            speedcorr_r_ol = roi_params[speedcorr_r_value_key_ol][j]
                            speedcorr_p_ol = roi_params[speedcorr_p_value_key_ol][j]
                            fl_distamp_r_ol = roi_params[fl_distamp_r_value_key_ol][j]/roi_normalization_value
                            fl_distamp_p_ol = roi_params[fl_distamp_p_value_key_ol][j]/roi_normalization_value
                            fl_amplitude_close_ol = np.divide(roi_params[fl_amplitude_close_key_ol][j],roi_normalization_value)
                            fl_amplitude_far_ol = np.divide(roi_params[fl_amplitude_far_key_ol][j],roi_normalization_value)
                            # success_robustness_ol = roi_params[success_robustness_value_key_ol][j]
                            mean_peak_ol = np.amax(np.array(roi_params[el + '_mean_trace_' +  tl + '_ol'][j]))/roi_normalization_value
                        except IndexError:
                            max_peak_ol = np.nan
                            max_active_ol = np.nan
                            mean_peak_ol = np.nan
                            space_max_peak_ol = np.nan
                            speedcorr_slope_ol = np.nan
                            fl_distamp_r_ol = np.nan
                            fl_distamp_p_ol = np.nan
                            fl_amplitude_close_ol = np.nan
                            fl_amplitude_far_ol = np.nan
                            speedcorr_r_ol = np.nan
                            speedcorr_p_ol = np.nan
                            # success_robustness_ol = np.nan

                        # grab peak value for trialonset condition
                        if roi_response_validation(roi_params, tl, 'trialonset', j):
                            peak_trialonset = roi_params['trialonset'+peak_metric+tl][j]/roi_normalization_value
                        else:
                            peak_trialonset = 0
                        # grab peak value for lmcenter condition
                        if roi_response_validation(roi_params, tl, 'lmcenter', j):
                            peak_lmcenter = roi_params['lmcenter'+peak_metric+tl][j]/roi_normalization_value
                        else:
                            peak_lmcenter = 0

                        peak_event = el
                        peak_trialtype = tl
                if valid and not roi_error:
                    # if the peak response is by the landmark, check which trialtype elicits the larger response and store the normalized difference
                    if peak_event == 'lmcenter':
                        # ipdb.set_trace()
                        if tl == 'short' and max_peak > roi_params['lmcenter' + peak_metric + 'long'][j]/roi_normalization_value:
                            # peak_lm_long = roi_params['lmcenter' + peak_metric + 'long'][j]/roi_normalization_value
                            # peak_lm_normdiff = ((roi_params['lmcenter' + peak_metric + 'long'][j]/roi_normalization_value)/max_peak)-1
                            peak_lm_index = (max_peak-(roi_params['lmcenter' + peak_metric + 'long'][j]/roi_normalization_value))/(max_peak+(roi_params['lmcenter' + peak_metric + 'long'][j]/roi_normalization_value))*-1
                            peak_lm_type = 'short'
                        elif tl == 'long' and max_peak > roi_params['lmcenter' + peak_metric + 'short'][j]/roi_normalization_value:
                            # peak_lm_short = roi_params['lmcenter' + peak_metric + 'short'][j]/roi_normalization_value
                            # peak_lm_normdiff = (((roi_params['lmcenter' + peak_metric + 'short'][j]/roi_normalization_value)/max_peak)-1)*-1
                            peak_lm_index = ((max_peak-(roi_params['lmcenter' + peak_metric + 'short'][j]/roi_normalization_value))/(max_peak+(roi_params['lmcenter' + peak_metric + 'short'][j]/roi_normalization_value)))
                            peak_lm_type = 'long'
                    if peak_event == 'trialonset':
                        if tl == 'short' and max_peak > roi_params['trialonset' + peak_metric + 'long'][j]/roi_normalization_value:
                            peak_to_index = (max_peak-(roi_params['trialonset' + peak_metric + 'long'][j]/roi_normalization_value))/(max_peak+(roi_params['trialonset' + peak_metric + 'long'][j]/roi_normalization_value))*-1
                            peak_to_type = 'short'
                        elif tl == 'long' and max_peak > roi_params['trialonset' + peak_metric + 'short'][j]/roi_normalization_value:
                            peak_to_index = ((max_peak-(roi_params['trialonset' + peak_metric + 'short'][j]/roi_normalization_value))/(max_peak+(roi_params['trialonset' + peak_metric + 'short'][j]/roi_normalization_value)))
                            peak_to_type = 'long'
                    if peak_event == 'reward':
                        if tl == 'short' and max_peak > roi_params['reward' + peak_metric + 'long'][j]/roi_normalization_value:
                            peak_rw_index = (max_peak-(roi_params['reward' + peak_metric + 'long'][j]/roi_normalization_value))/(max_peak+(roi_params['reward' + peak_metric + 'long'][j]/roi_normalization_value))*-1
                            peak_rw_type = 'short'
                        elif tl == 'long' and max_peak > roi_params['reward' + peak_metric + 'short'][j]/roi_normalization_value:
                            peak_rw_index = ((max_peak-(roi_params['reward' + peak_metric + 'short'][j]/roi_normalization_value))/(max_peak+(roi_params['reward' + peak_metric + 'short'][j]/roi_normalization_value)))
                            peak_rw_type = 'long'


                    # add 1/total number of rois to get fraction
                    # print('adding to ' + peak_event + ' ' + peak_trialtype)
                    # print(np.round(np.array(roi_params['trialonset_peak_' + tl][j]),2),np.round(np.array(roi_params['lmcenter_peak_' + tl][j]),2),np.round(np.array(roi_params['reward_peak_' + tl][j]),2))
                    result_counter[peak_event + '_peakcounter_' + peak_trialtype + str(i)] = result_counter[peak_event + '_peakcounter_' + peak_trialtype + str(i)] + (1/len(roi_list_all))
                    result_counter[peak_event + '_numrois_' + peak_trialtype + str(i)] = result_counter[peak_event + '_numrois_' + peak_trialtype + str(i)] + 1
                    result_counter['roicounter_' + peak_trialtype + str(i)] = result_counter['roicounter_' + peak_trialtype + str(i)] + 1
                    result_counter['roinumlist_' + peak_trialtype + str(i)].append(r)
                    result_counter['roipeak_' + peak_trialtype + str(i)].append(max_peak)
                    result_counter['roipeak_ol_' + peak_trialtype + str(i)].append(max_peak_ol)
                    result_max_peak[peak_event + '_peakval_' + peak_trialtype].append(max_peak)
                    result_max_peak_ol[peak_event + '_peakval_' + peak_trialtype].append(max_peak_ol)
                    result_max_peak[peak_event + '_meanpeak_' + peak_trialtype].append(mean_peak)
                    result_max_peak_ol[peak_event + '_meanpeak_' + peak_trialtype].append(mean_peak_ol)
                    result_max_peak[peak_event + '_peakloc_' + peak_trialtype].append(mean_peak_loc)
                    result_max_peak['space_peak_' + peak_trialtype].append(space_max_peak)
                    result_max_peak['space_peak_' + peak_trialtype + '_' + peak_event].append(space_max_peak)
                    result_max_peak_ol['space_peak_' + peak_trialtype].append(space_max_peak_ol)
                    result_max_peak_ol['space_peak_' + peak_trialtype + '_' + peak_event].append(space_max_peak_ol)
                    result_max_peak['space_speedcorr_' + peak_trialtype].append(speedcorr_slope)
                    result_max_peak['space_speedcorr_' + peak_trialtype + '_' + peak_event].append(speedcorr_slope)
                    result_max_peak['space_speedcorr_r_' + peak_trialtype].append(speedcorr_r)
                    result_max_peak['space_speedcorr_r_' + peak_trialtype + '_' + peak_event].append(speedcorr_r)
                    result_max_peak['space_speedcorr_p_' + peak_trialtype].append(speedcorr_p)
                    result_max_peak['space_speedcorr_p_' + peak_trialtype + '_' + peak_event].append(speedcorr_p)

                    result_max_peak_ol['space_speedcorr_' + peak_trialtype].append(speedcorr_slope_ol)
                    result_max_peak_ol['space_speedcorr_' + peak_trialtype + '_' + peak_event].append(speedcorr_slope_ol)
                    result_max_peak_ol['space_speedcorr_r_' + peak_trialtype].append(speedcorr_r_ol)
                    result_max_peak_ol['space_speedcorr_r_' + peak_trialtype + '_' + peak_event].append(speedcorr_r_ol)
                    result_max_peak_ol['space_speedcorr_p_' + peak_trialtype].append(speedcorr_p_ol)
                    result_max_peak_ol['space_speedcorr_p_' + peak_trialtype + '_' + peak_event].append(speedcorr_p_ol)

                    result_max_peak['fl_distamp_r_' + peak_trialtype].append(fl_distamp_r)
                    result_max_peak['fl_distamp_r_' + peak_trialtype + '_' + peak_event].append(fl_distamp_r)
                    result_max_peak['fl_distamp_p_' + peak_trialtype].append(fl_distamp_p)
                    result_max_peak['fl_distamp_p_' + peak_trialtype + '_' + peak_event].append(fl_distamp_p)

                    result_max_peak_ol['fl_distamp_r_' + peak_trialtype].append(fl_distamp_r_ol)
                    result_max_peak_ol['fl_distamp_r_' + peak_trialtype + '_' + peak_event].append(fl_distamp_r_ol)
                    result_max_peak_ol['fl_distamp_p_' + peak_trialtype].append(fl_distamp_p_ol)
                    result_max_peak_ol['fl_distamp_p_' + peak_trialtype + '_' + peak_event].append(fl_distamp_p_ol)

                    result_max_peak['fl_amplitude_close_' + tl].append(fl_amplitude_close)
                    result_max_peak['fl_amplitude_far_' + tl].append(fl_amplitude_far)
                    result_max_peak_ol['fl_amplitude_close_' + tl].append(fl_amplitude_close_ol)
                    result_max_peak_ol['fl_amplitude_far_' + tl].append(fl_amplitude_far_ol)

                    result_max_peak['success_robustness_' + tl].append(success_robustness)
                    result_max_peak['success_robustness_' + tl + '_' + el].append(success_robustness)
                    result_max_peak['default_robustness_' + tl].append(default_robustness)
                    result_max_peak['default_robustness_' + tl + '_' + el].append(default_robustness)
                    result_max_peak['success_amplitude_' + tl].append(success_amplitude)
                    result_max_peak['success_amplitude_' + tl + '_' + el].append(success_amplitude)
                    result_max_peak['default_amplitude_' + tl].append(default_amplitude)
                    result_max_peak['default_amplitude_' + tl + '_' + el].append(default_amplitude)



                    if max_active < 0:
                        max_active = 0
                    if max_active_ol < 0:
                        max_active_ol = 0
                    result_max_peak[peak_event + '_active_' + peak_trialtype].append(max_active)
                    result_max_peak_ol[peak_event + '_active_' + peak_trialtype].append(max_active_ol)

                    # print(peak_trialonset, peak_lmcenter)
                    result_trialonset_max_peak[peak_event + '_peakval_' + peak_trialtype].append(peak_trialonset)
                    result_lmcenter_max_peak[peak_event + '_peakval_' + peak_trialtype].append(peak_lmcenter)
                    # print(np.asarray(result_trialonset_max_peak['trialonset_peakval_short']).shape, np.asarray(result_lmcenter_max_peak['trialonset_peakval_short']).shape)
                    # result_matching_landmark_normdiff
                    if peak_event == 'lmcenter':
                        if peak_trialtype == 'short' and peak_lm_type == 'short':
                            # result_matching_landmark_long['lmcenter_peakval_' + peak_trialtype].append(peak_lm_long)
                            result_matching_landmark_normdiff['lm_diff_short'].append(peak_lm_index) #peak_lm_normdiff
                        elif peak_trialtype == 'long' and peak_lm_type == 'long':
                            # result_matching_landmark_short['lmcenter_peakval_' + peak_trialtype].append(peak_lm_short)
                            result_matching_landmark_normdiff['lm_diff_long'].append(peak_lm_index)
                    if peak_event == 'trialonset':
                        if peak_trialtype == 'short' and peak_to_type == 'short':
                            result_matching_landmark_normdiff['to_diff_short'].append(peak_to_index) #peak_lm_normdiff
                        elif peak_trialtype == 'long' and peak_to_type == 'long':
                            result_matching_landmark_normdiff['to_diff_long'].append(peak_to_index)
                    if peak_event == 'reward':
                        if peak_trialtype == 'short' and peak_rw_type == 'short':
                            result_matching_landmark_normdiff['rw_diff_short'].append(peak_rw_index) #peak_lm_normdiff
                        elif peak_trialtype == 'long' and peak_rw_type == 'long':
                            result_matching_landmark_normdiff['rw_diff_long'].append(peak_rw_index)

    # datasets to not be considered because they have no openloop session (have to be at end of rpl_all)
    if plot_layer is 'all':
        crop_ds = 2
    elif plot_layer is 'l23':
        crop_ds = 1
    elif plot_layer is 'l5':
        crop_ds = 1
    elif plot_layer is 'v1':
        crop_ds = 0
    else:
        crop_ds = 0

    # default colors for scatterplots of fraction of neurons
    scatter_color_short = '0.85'
    scatter_color_long = '0.7'

    # roilist_all = np.unique(np.array(result_counter['roinumlist_short0']))
    # roilist_all = np.unique(np.array(result_counter['roinumlist_long0']))
    # roilist_all.append(result_counter['roinumlist_short0'])

    # accumulate all the data and plot
    tot_num_valid_rois = []
    num_trialonset_short = []
    num_trialonset_long = []
    num_reward_short = []
    num_reward_long = []
    num_lmcenter_short = []
    num_lmcenter_long = []

    totrois_trialonset_short = []
    totrois_trialonset_long = []
    totrois_reward_short = []
    totrois_reward_long = []
    totrois_lmcenter_short = []
    totrois_lmcenter_long = []

    tot_num_rois_short = []
    tot_num_rois_long = []
    tot_num_rois_all = []

    if plot_layer is 'all':
        # calculate the mean difference in VR and OL conditions of all rois per animal
        mean_short = []
        mean_short_ol = []
        mean_long = []
        mean_long_ol = []
        mean_diff_short_index = []
        mean_diff_long_index = []
        mean_diff_all_index = []
        task_scores = []
        for i,rpl_all in enumerate(roi_param_list_all[0:-crop_ds]):
            mean_short.append(np.mean(np.array(result_counter['roipeak_short'+ str(i)])))
            mean_short_ol.append(np.mean(np.array(result_counter['roipeak_ol_short'+ str(i)])))
            mean_long.append(np.mean(np.array(result_counter['roipeak_long'+ str(i)])))
            mean_long_ol.append(np.mean(np.array(result_counter['roipeak_ol_long'+ str(i)])))
            # short_index = np.mean((np.array(result_counter['roipeak_short'+ str(i)]) - np.array(result_counter['roipeak_ol_short'+ str(i)]))) /       \
            #               np.mean((np.array(result_counter['roipeak_short'+ str(i)]) + np.array(result_counter['roipeak_ol_short'+ str(i)])))
            # long_index = np.mean((np.array(result_counter['roipeak_long'+ str(i)]) - np.array(result_counter['roipeak_ol_long'+ str(i)]))) /            \
            #              np.mean((np.array(result_counter['roipeak_long'+ str(i)]) + np.array(result_counter['roipeak_ol_long'+ str(i)])))

            short_index = np.mean((np.array(result_counter['roipeak_short'+ str(i)]) - np.array(result_counter['roipeak_ol_short'+ str(i)])))
            long_index = np.mean((np.array(result_counter['roipeak_long'+ str(i)]) - np.array(result_counter['roipeak_ol_long'+ str(i)])))

            mean_diff_short_index.append(short_index)
            mean_diff_long_index.append(long_index)
            mean_diff_all_index.append((short_index+long_index)/2)
            task_scores.append(rpl_all[3])

        # for i,rpl_all in enumerate(roi_param_list_all[0:-crop_ds]):
        #     print(rpl_all[0])
        #     print(np.round(np.array(mean_short)[i] - np.array(mean_short_ol)[i], 3))
        # print()

        mean_diff_short_index[0] = np.mean(mean_diff_short_index[0:3])
        del(mean_diff_short_index[1:3])
        mean_diff_long_index[0] = np.mean(mean_diff_long_index[0:3])
        del(mean_diff_long_index[1:3])
        mean_diff_all_index[0] = np.mean(mean_diff_all_index[0:3])
        del(mean_diff_all_index[1:3])
        task_scores[0] = np.mean(task_scores[0:3])
        del(task_scores[1:3])

        reg_res_short = sp.stats.theilslopes(mean_diff_short_index,task_scores)
        reg_res_long = sp.stats.theilslopes(mean_diff_long_index,task_scores)
        reg_res_all = sp.stats.theilslopes(mean_diff_all_index,task_scores)

        reg_lin_short = sp.stats.linregress(task_scores,mean_diff_short_index)
        reg_lin_long = sp.stats.linregress(task_scores,mean_diff_long_index)
        reg_lin_all = sp.stats.linregress(task_scores,mean_diff_all_index)

        ax_object13.spines['bottom'].set_linewidth(2)
        ax_object13.spines['top'].set_visible(False)
        ax_object13.spines['right'].set_visible(False)
        ax_object13.spines['left'].set_linewidth(2)
        ax_object13.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=18, \
            length=4, \
            width=4, \
            bottom='on', \
            left='on', \
            right='off', \
            top='off')

        # print('linear regression result: ')
        # print(reg_lin_short)
        # print(reg_lin_long)
        # print(reg_lin_all)

        # ax_object12.scatter(mean_short_ol,mean_short,s=60,c=scatter_color_short)
        # ax_object12.scatter(mean_long_ol,mean_long,s=60,c=scatter_color_long)
        # ax_object12.set_xlim([-0.1,2])
        # ax_object12.set_ylim([-0.1,2])
        # ax_object12.plot([0,2],[0,2],c='k',ls='--')
        ax_object13.scatter(task_scores,mean_diff_short_index,s=60,c=scatter_color_short)
        ax_object13.scatter(task_scores,mean_diff_long_index,s=60,c=scatter_color_long)
        # ax_object13.scatter(task_scores,mean_diff_all_index,s=60,c='k')
        ax_object13.plot(task_scores, reg_res_short[1] + reg_res_short[0] * np.array(task_scores), c=scatter_color_short, lw=2)
        ax_object13.plot(task_scores, reg_res_long[1] + reg_res_long[0] * np.array(task_scores), c=scatter_color_long, lw=2)
        # ax_object13.plot(task_scores, reg_res_all[1] + reg_res_all[0] * np.array(task_scores), c='k', lw=2)
        # ax_object13.set_ylim([0.2,0.8])
        ax_object13.set_xlim([0,80])
        ax_object13.set_xticks([0,20,40,60,80])
        ax_object13.set_xticklabels([0,20,40,60,80])
        ax_object13.set_xlabel('task score')
        ax_object13.set_ylabel('VR vs OL response amplitude ratio')




    for i in range(len(roi_param_list)):
        tot_num_valid_rois.append(result_counter['num_valid_rois' + str(i)])
        totrois_trialonset_short.append(result_counter['trialonset_numrois_short' + str(i)])
        totrois_trialonset_long.append(result_counter['trialonset_numrois_long' + str(i)])
        totrois_reward_short.append(result_counter['reward_numrois_short' + str(i)])
        totrois_reward_long.append(result_counter['reward_numrois_long' + str(i)])
        totrois_lmcenter_short.append(result_counter['lmcenter_numrois_short' + str(i)])
        totrois_lmcenter_long.append(result_counter['lmcenter_numrois_long' + str(i)])

        num_trialonset_short.append(result_counter['trialonset_peakcounter_short' + str(i)])
        num_trialonset_long.append(result_counter['trialonset_peakcounter_long' + str(i)])
        num_reward_short.append(result_counter['reward_peakcounter_short' + str(i)])
        num_reward_long.append(result_counter['reward_peakcounter_long' + str(i)])
        num_lmcenter_short.append(result_counter['lmcenter_peakcounter_short' + str(i)])
        num_lmcenter_long.append(result_counter['lmcenter_peakcounter_long' + str(i)])

        tot_num_rois_short.append(result_counter['roicounter_short'+ str(i)])
        tot_num_rois_long.append(result_counter['roicounter_long'+ str(i)])
        tot_num_rois_all.append(np.union1d(np.asarray(result_counter['roinumlist_short'+ str(i)]),np.asarray(result_counter['roinumlist_long'+ str(i)])).shape[0])

    # print(num_trialonset_short, num_lmcenter_short, num_reward_short)

    tot_num_short = 0
    for rs in tot_num_rois_short:
        tot_num_short += rs
    tot_num_long = 0
    for rs in tot_num_rois_long:
        tot_num_long += rs
    tot_num_all = 0
    for rs in tot_num_rois_all:
        tot_num_all += rs


    print('--------------------------------------------------------')
    print('total number of rois active on short track: ' + str(tot_num_short))
    print('total number of rois active on long track: ' + str(tot_num_long))
    print('total number of rois active on all: ' + str(tot_num_all))
    # print(len(tot_num_valid_rois), np.sum(tot_num_valid_rois))
    print('--------------------------------------------------------')


    numbins = 30
    # group data by animals (THIS DEPENDS ON ORDER OF roi_param_list!)
    # group_by_animal  True
    # print(roi_param_list[10],num_trialonset_long[10])

    # if plot_layer is 'all':
    #
    #     # group fraction neurons by layer and do statistical difference testing
    #     num_l23 = [np.mean(num_trialonset_short[0:3]),num_trialonset_short[7],num_trialonset_short[10],np.mean(num_trialonset_long[0:3]),num_trialonset_long[7],num_trialonset_long[10],
    #                np.mean(num_lmcenter_short[0:3]),num_lmcenter_short[7],num_lmcenter_short[10],np.mean(num_lmcenter_long[0:3]),num_lmcenter_long[7],num_lmcenter_long[10],
    #                np.mean(num_reward_short[0:3]),num_reward_short[7],num_reward_short[10],np.mean(num_reward_long[0:3]),num_reward_long[7],num_reward_long[10]]
    #
    #     num_l5 = [np.mean(num_trialonset_short[3:5]),np.mean(num_trialonset_short[5:7]),num_trialonset_short[8],num_trialonset_short[9],num_trialonset_short[11],np.mean(num_trialonset_long[3:5]),np.mean(num_trialonset_long[5:7]),num_trialonset_long[8],num_trialonset_long[9],num_trialonset_long[11],
    #               np.mean(num_lmcenter_short[3:5]),np.mean(num_lmcenter_short[5:7]),num_lmcenter_short[8],num_lmcenter_short[9],num_lmcenter_short[11],np.mean(num_lmcenter_long[3:5]),np.mean(num_lmcenter_long[5:7]),num_lmcenter_long[8],num_lmcenter_long[9],num_lmcenter_long[11],
    #               np.mean(num_reward_short[3:5]),np.mean(num_reward_short[5:7]),num_reward_short[8],num_reward_short[9],num_reward_short[11],np.mean(num_reward_long[3:5]),np.mean(num_reward_long[5:7]),num_reward_long[8],num_reward_long[9],num_reward_long[11]]
    #
    #     num_trialonset_l23 = [np.mean(num_trialonset_short[0:3]),num_trialonset_short[7],num_trialonset_short[10],np.mean(num_trialonset_long[0:3]),num_trialonset_long[7],num_trialonset_long[10]]
    #     num_lmcenter_l23 = [np.mean(num_lmcenter_short[0:3]),num_lmcenter_short[7],num_lmcenter_short[10],np.mean(num_lmcenter_long[0:3]),num_lmcenter_long[7],num_lmcenter_long[10]]
    #     num_reward_l23 = [np.mean(num_reward_short[0:3]),num_reward_short[7],num_reward_short[10],np.mean(num_reward_long[0:3]),num_reward_long[7],num_reward_long[10]]
    #
    #     num_trialonset_l5 = [np.mean(num_trialonset_short[3:5]),np.mean(num_trialonset_short[5:7]),num_trialonset_short[8],num_trialonset_short[9],num_trialonset_short[11],np.mean(num_trialonset_long[3:5]),np.mean(num_trialonset_long[5:7]),num_trialonset_long[8],num_trialonset_long[9],num_trialonset_long[11]]
    #     num_lmcenter_l5 = [np.mean(num_lmcenter_short[3:5]),np.mean(num_lmcenter_short[5:7]),num_lmcenter_short[8],num_lmcenter_short[9],num_lmcenter_short[11],np.mean(num_lmcenter_long[3:5]),np.mean(num_lmcenter_long[5:7]),num_lmcenter_long[8],num_lmcenter_long[9],num_lmcenter_long[11]]
    #     num_reward_l5 = [np.mean(num_reward_short[3:5]),np.mean(num_reward_short[5:7]),num_reward_short[8],num_reward_short[9],num_reward_short[11],np.mean(num_reward_long[3:5]),np.mean(num_reward_long[5:7]),num_reward_long[8],num_reward_long[9],num_reward_long[11]]
    #
    #     print('--- L23 VS. L5 ANOVA FOR ALIGNMENT POINTS ---')
    #     print(sp.stats.f_oneway(np.array(num_trialonset_l23),np.array(num_lmcenter_l23),np.array(num_reward_l23),np.array(num_trialonset_l5),np.array(num_lmcenter_l5),np.array(num_reward_l5)))
    #     group_labels = ['trialonset_l23'] * np.array(num_trialonset_l23).shape[0] + \
    #                    ['lmcenter_l23'] * np.array(num_lmcenter_l23).shape[0] + \
    #                    ['reward_l23'] * np.array(num_reward_l23).shape[0] + \
    #                    ['trialonset_l5'] * np.array(num_trialonset_l5).shape[0] + \
    #                    ['lmcenter_l5'] * np.array(num_lmcenter_l5).shape[0] + \
    #                    ['reward_l5'] * np.array(num_reward_l5).shape[0]
    #
    #     mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(num_trialonset_l23),np.array(num_lmcenter_l23),np.array(num_reward_l23),np.array(num_trialonset_l5),np.array(num_lmcenter_l5),np.array(num_reward_l5))),group_labels)
    #     posthoc_res_ss = mc_res_ss.tukeyhsd()
    #     print(posthoc_res_ss)
    #     print('--- L23 VS. L5 TTEST TOTAL FRACTIONS ---')
    #     print(sp.stats.ttest_ind(num_l23, num_l5))
    #     print('----------------------------------------')
    #
    #     num_trialonset_short = [np.mean(num_trialonset_short[0:3]),np.mean(num_trialonset_short[3:5]),np.mean(num_trialonset_short[5:7]),num_trialonset_short[7],num_trialonset_short[8],num_trialonset_short[9],num_trialonset_short[10],num_trialonset_short[11]]#,num_trialonset_short[12]]
    #     num_trialonset_long = [np.mean(num_trialonset_long[0:3]),np.mean(num_trialonset_long[3:5]),np.mean(num_trialonset_long[5:7]),num_trialonset_long[7],num_trialonset_long[8],num_trialonset_long[9],num_trialonset_long[10],num_trialonset_long[11]]#,num_trialonset_long[12]]
    #     num_lmcenter_short = [np.mean(num_lmcenter_short[0:3]),np.mean(num_lmcenter_short[3:5]),np.mean(num_lmcenter_short[5:7]),num_lmcenter_short[7],num_lmcenter_short[8],num_lmcenter_short[9],num_lmcenter_short[10],num_lmcenter_short[11]]#,num_lmcenter_short[12]]
    #     num_lmcenter_long = [np.mean(num_lmcenter_long[0:3]),np.mean(num_lmcenter_long[3:5]),np.mean(num_lmcenter_long[5:7]),num_lmcenter_long[7],num_lmcenter_long[8],num_lmcenter_long[9],num_lmcenter_long[10],num_lmcenter_long[11]]#,num_lmcenter_long[12]]
    #     num_reward_short = [np.mean(num_reward_short[0:3]),np.mean(num_reward_short[3:5]),np.mean(num_reward_short[5:7]),num_reward_short[7],num_reward_short[8],num_reward_short[9],num_reward_short[10],num_reward_short[11]]#,num_reward_short[12]]
    #     num_reward_long = [np.mean(num_reward_long[0:3]),np.mean(num_reward_long[3:5]),np.mean(num_reward_long[5:7]),num_reward_long[7],num_reward_long[8],num_reward_long[9],num_reward_long[10],num_reward_long[11]]#,num_reward_long[12]]
    #
    #     totrois_trialonset_short = [np.sum(totrois_trialonset_short[0:3]),np.sum(totrois_trialonset_short[3:5]),np.sum(totrois_trialonset_short[5:7]),totrois_trialonset_short[7],totrois_trialonset_short[8],totrois_trialonset_short[9],totrois_trialonset_short[10],totrois_trialonset_short[11]]#,totrois_trialonset_short[12]]
    #     totrois_trialonset_long = [np.sum(totrois_trialonset_long[0:3]),np.sum(totrois_trialonset_long[3:5]),np.sum(totrois_trialonset_long[5:7]),totrois_trialonset_long[7],totrois_trialonset_long[8],totrois_trialonset_long[9],totrois_trialonset_long[10],totrois_trialonset_long[11]]#,totrois_trialonset_long[12]]
    #     totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0:3]),np.sum(totrois_lmcenter_short[3:5]),np.sum(totrois_lmcenter_short[5:7]),totrois_lmcenter_short[7],totrois_lmcenter_short[8],totrois_lmcenter_short[9],totrois_lmcenter_short[10],totrois_lmcenter_short[11]]#,totrois_lmcenter_short[12]]
    #     totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0:3]),np.sum(totrois_lmcenter_long[3:5]),np.sum(totrois_lmcenter_long[5:7]),totrois_lmcenter_long[7],totrois_lmcenter_long[8],totrois_lmcenter_long[9],totrois_lmcenter_long[10],totrois_lmcenter_long[11]]#,totrois_lmcenter_long[12]]
    #     totrois_reward_short = [np.sum(totrois_reward_short[0:3]),np.sum(totrois_reward_short[3:5]),np.sum(totrois_reward_short[5:7]),totrois_reward_short[7],totrois_reward_short[8],totrois_reward_short[9],totrois_reward_short[10],totrois_reward_short[11]]#,totrois_reward_short[12]]
    #     totrois_reward_long = [np.sum(totrois_reward_long[0:3]),np.sum(totrois_reward_long[3:5]),np.sum(totrois_reward_long[5:7]),totrois_reward_long[7],totrois_reward_long[8],totrois_reward_long[9],totrois_reward_long[10],totrois_reward_long[11]]#,totrois_reward_long[12]]
    #
    #     tot_num_valid_rois = [np.sum(tot_num_valid_rois[0:3]),np.sum(tot_num_valid_rois[3:5]),np.sum(tot_num_valid_rois[5:7]),tot_num_valid_rois[7],tot_num_valid_rois[8],tot_num_valid_rois[9],tot_num_valid_rois[10],tot_num_valid_rois[11]]#,tot_num_valid_rois[12]]
    #
    #     scatter_color_short = '#F58020'
    #     scatter_color_long = '#374D9E'
    #     fraction_plot_color_short = '0.80'
    #     fraction_plot_color_long = '0.85'

    if plot_layer is 'all':

        # group fraction neurons by layer and do statistical difference testing
        num_l23 = [np.mean(num_trialonset_short[0:3]),num_trialonset_short[7],num_trialonset_short[10],np.mean(num_trialonset_long[0:3]),num_trialonset_long[7],num_trialonset_long[10],
                   np.mean(num_lmcenter_short[0:3]),num_lmcenter_short[7],num_lmcenter_short[10],np.mean(num_lmcenter_long[0:3]),num_lmcenter_long[7],num_lmcenter_long[10],
                   np.mean(num_reward_short[0:3]),num_reward_short[7],num_reward_short[10],np.mean(num_reward_long[0:3]),num_reward_long[7],num_reward_long[10]]

        num_l5 = [np.mean(num_trialonset_short[3:5]),np.mean(num_trialonset_short[5]),num_trialonset_short[7],num_trialonset_short[8],num_trialonset_short[10],np.mean(num_trialonset_long[3:5]),np.mean(num_trialonset_long[5]),num_trialonset_long[7],num_trialonset_long[8],num_trialonset_long[10],
                  np.mean(num_lmcenter_short[3:5]),np.mean(num_lmcenter_short[5]),num_lmcenter_short[7],num_lmcenter_short[8],num_lmcenter_short[10],np.mean(num_lmcenter_long[3:5]),np.mean(num_lmcenter_long[5]),num_lmcenter_long[7],num_lmcenter_long[8],num_lmcenter_long[10],
                  np.mean(num_reward_short[3:5]),np.mean(num_reward_short[5]),num_reward_short[7],num_reward_short[8],num_reward_short[10],np.mean(num_reward_long[3:5]),np.mean(num_reward_long[5]),num_reward_long[7],num_reward_long[8],num_reward_long[10]]

        num_trialonset_l23 = [np.mean(num_trialonset_short[0:3]),num_trialonset_short[7],num_trialonset_short[10],np.mean(num_trialonset_long[0:3]),num_trialonset_long[7],num_trialonset_long[10]]
        num_lmcenter_l23 = [np.mean(num_lmcenter_short[0:3]),num_lmcenter_short[7],num_lmcenter_short[10],np.mean(num_lmcenter_long[0:3]),num_lmcenter_long[7],num_lmcenter_long[10]]
        num_reward_l23 = [np.mean(num_reward_short[0:3]),num_reward_short[7],num_reward_short[10],np.mean(num_reward_long[0:3]),num_reward_long[7],num_reward_long[10]]

        num_trialonset_l5 = [np.mean(num_trialonset_short[3:5]),np.mean(num_trialonset_short[5]),num_trialonset_short[7],num_trialonset_short[8],num_trialonset_short[10],np.mean(num_trialonset_long[3:5]),np.mean(num_trialonset_long[5]),num_trialonset_long[7],num_trialonset_long[8],num_trialonset_long[10]]
        num_lmcenter_l5 = [np.mean(num_lmcenter_short[3:5]),np.mean(num_lmcenter_short[5]),num_lmcenter_short[7],num_lmcenter_short[8],num_lmcenter_short[10],np.mean(num_lmcenter_long[3:5]),np.mean(num_lmcenter_long[5]),num_lmcenter_long[7],num_lmcenter_long[8],num_lmcenter_long[10]]
        num_reward_l5 = [np.mean(num_reward_short[3:5]),np.mean(num_reward_short[5]),num_reward_short[7],num_reward_short[8],num_reward_short[10],np.mean(num_reward_long[3:5]),np.mean(num_reward_long[5]),num_reward_long[7],num_reward_long[8],num_reward_long[10]]

        print('--- L23 VS. L5 ANOVA FOR ALIGNMENT POINTS ---')
        print(sp.stats.f_oneway(np.array(num_trialonset_l23),np.array(num_lmcenter_l23),np.array(num_reward_l23),np.array(num_trialonset_l5),np.array(num_lmcenter_l5),np.array(num_reward_l5)))
        group_labels = ['trialonset_l23'] * np.array(num_trialonset_l23).shape[0] + \
                       ['lmcenter_l23'] * np.array(num_lmcenter_l23).shape[0] + \
                       ['reward_l23'] * np.array(num_reward_l23).shape[0] + \
                       ['trialonset_l5'] * np.array(num_trialonset_l5).shape[0] + \
                       ['lmcenter_l5'] * np.array(num_lmcenter_l5).shape[0] + \
                       ['reward_l5'] * np.array(num_reward_l5).shape[0]

        mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(num_trialonset_l23),np.array(num_lmcenter_l23),np.array(num_reward_l23),np.array(num_trialonset_l5),np.array(num_lmcenter_l5),np.array(num_reward_l5))),group_labels)
        posthoc_res_ss = mc_res_ss.tukeyhsd()
        print(posthoc_res_ss)
        print('--- L23 VS. L5 TTEST TOTAL FRACTIONS ---')
        print(sp.stats.ttest_ind(num_l23, num_l5))
        print('----------------------------------------')

        num_trialonset_short = [np.mean(num_trialonset_short[0:3]),np.mean(num_trialonset_short[3:5]),np.mean(num_trialonset_short[5]),num_trialonset_short[6],num_trialonset_short[7],num_trialonset_short[8],num_trialonset_short[9],num_trialonset_short[10]]#,num_trialonset_short[12]]
        num_trialonset_long = [np.mean(num_trialonset_long[0:3]),np.mean(num_trialonset_long[3:5]),np.mean(num_trialonset_long[5]),num_trialonset_long[6],num_trialonset_long[7],num_trialonset_long[8],num_trialonset_long[9],num_trialonset_long[10]]#,num_trialonset_long[12]]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0:3]),np.mean(num_lmcenter_short[3:5]),np.mean(num_lmcenter_short[5]),num_lmcenter_short[6],num_lmcenter_short[7],num_lmcenter_short[8],num_lmcenter_short[9],num_lmcenter_short[10]]#,num_lmcenter_short[12]]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0:3]),np.mean(num_lmcenter_long[3:5]),np.mean(num_lmcenter_long[5]),num_lmcenter_long[6],num_lmcenter_long[7],num_lmcenter_long[8],num_lmcenter_long[9],num_lmcenter_long[10]]#,num_lmcenter_long[12]]
        num_reward_short = [np.mean(num_reward_short[0:3]),np.mean(num_reward_short[3:5]),np.mean(num_reward_short[5]),num_reward_short[6],num_reward_short[7],num_reward_short[8],num_reward_short[9],num_reward_short[10]]#,num_reward_short[12]]
        num_reward_long = [np.mean(num_reward_long[0:3]),np.mean(num_reward_long[3:5]),np.mean(num_reward_long[5]),num_reward_long[6],num_reward_long[7],num_reward_long[8],num_reward_long[9],num_reward_long[10]]#,num_reward_long[12]]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0:3]),np.sum(totrois_trialonset_short[3:5]),np.sum(totrois_trialonset_short[5]),totrois_trialonset_short[6],totrois_trialonset_short[7],totrois_trialonset_short[8],totrois_trialonset_short[9],totrois_trialonset_short[10]]#,totrois_trialonset_short[12]]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0:3]),np.sum(totrois_trialonset_long[3:5]),np.sum(totrois_trialonset_long[5]),totrois_trialonset_long[6],totrois_trialonset_long[7],totrois_trialonset_long[8],totrois_trialonset_long[9],totrois_trialonset_long[10]]#,totrois_trialonset_long[12]]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0:3]),np.sum(totrois_lmcenter_short[3:5]),np.sum(totrois_lmcenter_short[5]),totrois_lmcenter_short[6],totrois_lmcenter_short[7],totrois_lmcenter_short[8],totrois_lmcenter_short[9],totrois_lmcenter_short[10]]#,totrois_lmcenter_short[12]]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0:3]),np.sum(totrois_lmcenter_long[3:5]),np.sum(totrois_lmcenter_long[5]),totrois_lmcenter_long[6],totrois_lmcenter_long[7],totrois_lmcenter_long[8],totrois_lmcenter_long[9],totrois_lmcenter_long[10]]#,totrois_lmcenter_long[12]]
        totrois_reward_short = [np.sum(totrois_reward_short[0:3]),np.sum(totrois_reward_short[3:5]),np.sum(totrois_reward_short[5]),totrois_reward_short[6],totrois_reward_short[7],totrois_reward_short[8],totrois_reward_short[9],totrois_reward_short[10]]#,totrois_reward_short[12]]
        totrois_reward_long = [np.sum(totrois_reward_long[0:3]),np.sum(totrois_reward_long[3:5]),np.sum(totrois_reward_long[5]),totrois_reward_long[6],totrois_reward_long[7],totrois_reward_long[8],totrois_reward_long[9],totrois_reward_long[10]]#,totrois_reward_long[12]]

        tot_num_valid_rois = [np.sum(tot_num_valid_rois[0:3]),np.sum(tot_num_valid_rois[3:5]),np.sum(tot_num_valid_rois[5]),tot_num_valid_rois[6],tot_num_valid_rois[7],tot_num_valid_rois[8],tot_num_valid_rois[9],tot_num_valid_rois[10]]#,tot_num_valid_rois[12]]

        scatter_color_short = '#F58020'
        scatter_color_long = '#374D9E'
        fraction_plot_color_short = '0.80'
        fraction_plot_color_long = '0.85'

    # group data of fractions by animal? DEPENDS ON ORDER OF ROI_PARAM_LIST
    # group_by_animal_l23 = False
    # print(roi_param_list[10],num_trialonset_long[10])
    if plot_layer is 'l23':
        num_trialonset_short = [np.mean(num_trialonset_short[0]),np.mean(num_trialonset_short[1:4]),np.mean(num_trialonset_short[4])]#,np.mean(num_trialonset_short[5])]
        num_trialonset_long = [np.mean(num_trialonset_long[0]),np.mean(num_trialonset_long[1:4]),np.mean(num_trialonset_long[4])]#,np.mean(num_trialonset_long[5])]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0]),np.mean(num_lmcenter_short[1:4]),np.mean(num_lmcenter_short[4])]#,np.mean(num_lmcenter_short[5])]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0]),np.mean(num_lmcenter_long[1:4]),np.mean(num_lmcenter_long[4])]#,np.mean(num_lmcenter_long[5])]
        num_reward_short = [np.mean(num_reward_short[0]),np.mean(num_reward_short[1:4]),np.mean(num_reward_short[4])]#,np.mean(num_reward_short[5])]
        num_reward_long = [np.mean(num_reward_long[0]),np.mean(num_reward_long[1:4]),np.mean(num_reward_long[4])]#,np.mean(num_reward_long[5])]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0]),np.sum(totrois_trialonset_short[1:4]),np.sum(totrois_trialonset_short[4])]#,np.sum(totrois_trialonset_short[5])]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0]),np.sum(totrois_trialonset_long[1:4]),np.sum(totrois_trialonset_long[4])]#,np.sum(totrois_trialonset_long[5])]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0]),np.sum(totrois_lmcenter_short[1:4]),np.sum(totrois_lmcenter_short[4])]#,np.sum(totrois_lmcenter_short[5])]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0]),np.sum(totrois_lmcenter_long[1:4]),np.sum(totrois_lmcenter_long[4])]#,np.sum(totrois_lmcenter_long[5])]
        totrois_reward_short = [np.sum(totrois_reward_short[0]),np.sum(totrois_reward_short[1:4]),np.sum(totrois_reward_short[4])]#,np.sum(totrois_reward_short[5])]
        totrois_reward_long = [np.sum(totrois_reward_long[0]),np.sum(totrois_reward_long[1:4]),np.sum(totrois_reward_long[4])]#,np.sum(totrois_reward_long[5])]

        tot_num_valid_rois = [np.sum(tot_num_valid_rois[0]),np.sum(tot_num_valid_rois[1:4]),np.sum(tot_num_valid_rois[4])]#,np.sum(tot_num_valid_rois[5])]
        scatter_color_short = '#F7901E'
        scatter_color_long = '#F15E36'
        fraction_plot_color_short = '#F9B485'
        fraction_plot_color_long = '#F79F63'
        numbins = 30

    # group_by_animal_l5 = False
    # print(roi_param_list[10],num_trialonset_long[10])
    if plot_layer is 'l5':
        num_trialonset_short = [np.mean(num_trialonset_short[0]),np.mean(num_trialonset_short[1:3]),np.mean(num_trialonset_short[3]),np.mean(num_trialonset_short[4]),np.mean(num_trialonset_short[5])]
        num_trialonset_long = [np.mean(num_trialonset_long[0]),np.mean(num_trialonset_long[1:3]),np.mean(num_trialonset_long[3]),np.mean(num_trialonset_long[4]),np.mean(num_trialonset_long[5])]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0]),np.mean(num_lmcenter_short[1:3]),np.mean(num_lmcenter_short[3]),np.mean(num_lmcenter_short[4]),np.mean(num_lmcenter_short[5])]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0]),np.mean(num_lmcenter_long[1:3]),np.mean(num_lmcenter_long[3]),np.mean(num_lmcenter_long[4]),np.mean(num_lmcenter_long[5])]
        num_reward_short = [np.mean(num_reward_short[0]),np.mean(num_reward_short[1:3]),np.mean(num_reward_short[3]),np.mean(num_reward_short[4]),np.mean(num_reward_short[5])]
        num_reward_long = [np.mean(num_reward_long[0]),np.mean(num_reward_long[1:3]),np.mean(num_reward_long[3]),np.mean(num_reward_long[4]),np.mean(num_reward_long[5])]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0]),np.sum(totrois_trialonset_short[1:3]),np.sum(totrois_trialonset_short[3]),np.sum(totrois_trialonset_short[4]),np.sum(totrois_trialonset_short[5])]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0]),np.sum(totrois_trialonset_long[1:3]),np.sum(totrois_trialonset_long[3]),np.sum(totrois_trialonset_long[4]),np.sum(totrois_trialonset_long[5])]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0]),np.sum(totrois_lmcenter_short[1:3]),np.sum(totrois_lmcenter_short[3]),np.sum(totrois_lmcenter_short[4]),np.sum(totrois_lmcenter_short[5])]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0]),np.sum(totrois_lmcenter_long[1:3]),np.sum(totrois_lmcenter_long[3]),np.sum(totrois_lmcenter_long[4]),np.sum(totrois_lmcenter_long[5])]
        totrois_reward_short = [np.sum(totrois_reward_short[0]),np.sum(totrois_reward_short[1:3]),np.sum(totrois_reward_short[3]),np.sum(totrois_reward_short[4]),np.sum(totrois_reward_short[5])]
        totrois_reward_long = [np.sum(totrois_reward_long[0]),np.sum(totrois_reward_long[1:3]),np.sum(totrois_reward_long[3]),np.sum(totrois_reward_long[4]),np.sum(totrois_reward_long[5])]

        tot_num_valid_rois = [np.sum(tot_num_valid_rois[0]),np.sum(tot_num_valid_rois[1:3]),np.sum(tot_num_valid_rois[3]),np.sum(tot_num_valid_rois[4]),np.sum(tot_num_valid_rois[5])]

        scatter_color_short = '#F970F5'
        scatter_color_long = '#FF00FF'
        fraction_plot_color_short = '#FCA3FA'
        fraction_plot_color_long = '#F98FF5'
        numbins = 30

    if plot_layer is 'v1':
        num_trialonset_short = [np.mean(num_trialonset_short[0:2]),np.mean(num_trialonset_short[2]),np.mean(num_trialonset_short[3:5]),np.mean(num_trialonset_short[5])]
        num_trialonset_long = [np.mean(num_trialonset_long[0:2]),np.mean(num_trialonset_long[2]),np.mean(num_trialonset_long[3:5]),np.mean(num_trialonset_long[5])]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0:2]),np.mean(num_lmcenter_short[2]),np.mean(num_lmcenter_short[3:5]),np.mean(num_lmcenter_short[5])]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0:2]),np.mean(num_lmcenter_long[2]),np.mean(num_lmcenter_long[3:5]),np.mean(num_lmcenter_long[5])]
        num_reward_short = [np.mean(num_reward_short[0:2]),np.mean(num_reward_short[2]),np.mean(num_reward_short[3:5]),np.mean(num_reward_short[5])]
        num_reward_long = [np.mean(num_reward_long[0:2]),np.mean(num_reward_long[2]),np.mean(num_reward_long[3:5]),np.mean(num_reward_long[5])]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0:2]),np.sum(totrois_trialonset_short[2]),np.sum(totrois_trialonset_short[3:5]),np.sum(totrois_trialonset_short[5])]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0:2]),np.sum(totrois_trialonset_long[2]),np.sum(totrois_trialonset_long[3:5]),np.sum(totrois_trialonset_long[5])]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0:2]),np.sum(totrois_lmcenter_short[2]),np.sum(totrois_lmcenter_short[3:5]),np.sum(totrois_lmcenter_short[5])]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0:2]),np.sum(totrois_lmcenter_long[2]),np.sum(totrois_lmcenter_long[3:5]),np.sum(totrois_lmcenter_long[5])]
        totrois_reward_short = [np.sum(totrois_reward_short[0:2]),np.sum(totrois_reward_short[2]),np.sum(totrois_reward_short[3:5]),np.sum(totrois_reward_short[5])]
        totrois_reward_long = [np.sum(totrois_reward_long[0:2]),np.sum(totrois_reward_long[2]),np.sum(totrois_reward_long[3:5]),np.sum(totrois_reward_long[5])]

        tot_num_valid_rois = [np.sum(tot_num_valid_rois[0:2]),np.sum(tot_num_valid_rois[2]),np.sum(tot_num_valid_rois[3:5]),np.sum(tot_num_valid_rois[5])]

        scatter_color_short = '#00B0B9'
        scatter_color_long = '#006FB9'
        fraction_plot_color_short = '0.80'
        fraction_plot_color_long = '0.85'
        numbins = 30

    if plot_layer is 'naive':
        num_trialonset_short = [np.mean(num_trialonset_short[0]),np.mean(num_trialonset_short[1]),np.mean(num_trialonset_short[2]),np.mean(num_trialonset_short[3])]#,np.mean(num_trialonset_short[5])]
        num_trialonset_long = [np.mean(num_trialonset_long[0]),np.mean(num_trialonset_long[1]),np.mean(num_trialonset_long[2]),np.mean(num_trialonset_long[3])]#,np.mean(num_trialonset_long[5])]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0]),np.mean(num_lmcenter_short[1]),np.mean(num_lmcenter_short[2]),np.mean(num_lmcenter_short[3])]#,np.mean(num_lmcenter_short[5])]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0]),np.mean(num_lmcenter_long[1]),np.mean(num_lmcenter_long[2]),np.mean(num_lmcenter_long[3])]#,np.mean(num_lmcenter_long[5])]
        num_reward_short = [np.mean(num_reward_short[0]),np.mean(num_reward_short[1]),np.mean(num_reward_short[2]),np.mean(num_reward_short[3])]#,np.mean(num_reward_short[5])]
        num_reward_long = [np.mean(num_reward_long[0]),np.mean(num_reward_long[1]),np.mean(num_reward_long[2]),np.mean(num_reward_long[3])]#,np.mean(num_reward_long[5])]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0]),np.sum(totrois_trialonset_short[1]),np.sum(totrois_trialonset_short[2]),np.sum(totrois_trialonset_short[3])]#,np.sum(totrois_trialonset_short[5])]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0]),np.sum(totrois_trialonset_long[1]),np.sum(totrois_trialonset_long[2]),np.sum(totrois_trialonset_long[3])]#,np.sum(totrois_trialonset_long[5])]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0]),np.sum(totrois_lmcenter_short[1]),np.sum(totrois_lmcenter_short[2]),np.sum(totrois_lmcenter_short[3])]#,np.sum(totrois_lmcenter_short[5])]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0]),np.sum(totrois_lmcenter_long[1]),np.sum(totrois_lmcenter_long[2]),np.sum(totrois_lmcenter_long[3])]#,np.sum(totrois_lmcenter_long[5])]
        totrois_reward_short = [np.sum(totrois_reward_short[0]),np.sum(totrois_reward_short[1]),np.sum(totrois_reward_short[2]),np.sum(totrois_reward_short[3])]#,np.sum(totrois_reward_short[5])]
        totrois_reward_long = [np.sum(totrois_reward_long[0]),np.sum(totrois_reward_long[1]),np.sum(totrois_reward_long[2]),np.sum(totrois_reward_long[3])]#,np.sum(totrois_reward_long[5])]

        scatter_color_short = '#F58020'
        scatter_color_long = '#374D9E'
        fraction_plot_color_short = '0.80'
        fraction_plot_color_long = '0.85'

    if plot_layer is 'naive_matched':
        num_trialonset_short = [np.mean(num_trialonset_short[0]),np.mean(num_trialonset_short[1]),np.mean(num_trialonset_short[2])]#,np.mean(num_trialonset_short[5])]
        num_trialonset_long = [np.mean(num_trialonset_long[0]),np.mean(num_trialonset_long[1]),np.mean(num_trialonset_long[2])]#,np.mean(num_trialonset_long[5])]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0]),np.mean(num_lmcenter_short[1]),np.mean(num_lmcenter_short[2])]#,np.mean(num_lmcenter_short[5])]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0]),np.mean(num_lmcenter_long[1]),np.mean(num_lmcenter_long[2])]#,np.mean(num_lmcenter_long[5])]
        num_reward_short = [np.mean(num_reward_short[0]),np.mean(num_reward_short[1]),np.mean(num_reward_short[2])]#,np.mean(num_reward_short[5])]
        num_reward_long = [np.mean(num_reward_long[0]),np.mean(num_reward_long[1]),np.mean(num_reward_long[2])]#,np.mean(num_reward_long[5])]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0]),np.sum(totrois_trialonset_short[1]),np.sum(totrois_trialonset_short[2])]#,np.sum(totrois_trialonset_short[5])]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0]),np.sum(totrois_trialonset_long[1]),np.sum(totrois_trialonset_long[2])]#,np.sum(totrois_trialonset_long[5])]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0]),np.sum(totrois_lmcenter_short[1]),np.sum(totrois_lmcenter_short[2])]#,np.sum(totrois_lmcenter_short[5])]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0]),np.sum(totrois_lmcenter_long[1]),np.sum(totrois_lmcenter_long[2])]#,np.sum(totrois_lmcenter_long[5])]
        totrois_reward_short = [np.sum(totrois_reward_short[0]),np.sum(totrois_reward_short[1]),np.sum(totrois_reward_short[2])]#,np.sum(totrois_reward_short[5])]
        totrois_reward_long = [np.sum(totrois_reward_long[0]),np.sum(totrois_reward_long[1]),np.sum(totrois_reward_long[2])]#,np.sum(totrois_reward_long[5])]

        scatter_color_short = '#F58020'
        scatter_color_long = '#374D9E'
        fraction_plot_color_short = '0.80'
        fraction_plot_color_long = '0.85'

    if plot_layer is 'expert_matched':
        num_trialonset_short = [np.mean(num_trialonset_short[0]),np.mean(num_trialonset_short[1]),np.mean(num_trialonset_short[2])]#,np.mean(num_trialonset_short[5])]
        num_trialonset_long = [np.mean(num_trialonset_long[0]),np.mean(num_trialonset_long[1]),np.mean(num_trialonset_long[2])]#,np.mean(num_trialonset_long[5])]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0]),np.mean(num_lmcenter_short[1]),np.mean(num_lmcenter_short[2])]#,np.mean(num_lmcenter_short[5])]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0]),np.mean(num_lmcenter_long[1]),np.mean(num_lmcenter_long[2])]#,np.mean(num_lmcenter_long[5])]
        num_reward_short = [np.mean(num_reward_short[0]),np.mean(num_reward_short[1]),np.mean(num_reward_short[2])]#,np.mean(num_reward_short[5])]
        num_reward_long = [np.mean(num_reward_long[0]),np.mean(num_reward_long[1]),np.mean(num_reward_long[2])]#,np.mean(num_reward_long[5])]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0]),np.sum(totrois_trialonset_short[1]),np.sum(totrois_trialonset_short[2])]#,np.sum(totrois_trialonset_short[5])]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0]),np.sum(totrois_trialonset_long[1]),np.sum(totrois_trialonset_long[2])]#,np.sum(totrois_trialonset_long[5])]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0]),np.sum(totrois_lmcenter_short[1]),np.sum(totrois_lmcenter_short[2])]#,np.sum(totrois_lmcenter_short[5])]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0]),np.sum(totrois_lmcenter_long[1]),np.sum(totrois_lmcenter_long[2])]#,np.sum(totrois_lmcenter_long[5])]
        totrois_reward_short = [np.sum(totrois_reward_short[0]),np.sum(totrois_reward_short[1]),np.sum(totrois_reward_short[2])]#,np.sum(totrois_reward_short[5])]
        totrois_reward_long = [np.sum(totrois_reward_long[0]),np.sum(totrois_reward_long[1]),np.sum(totrois_reward_long[2])]#,np.sum(totrois_reward_long[5])]

        scatter_color_short = '#F58020'
        scatter_color_long = '#374D9E'
        fraction_plot_color_short = '0.80'
        fraction_plot_color_long = '0.85'

    if plot_layer is 'expert_matched_NEW':
        num_trialonset_short = [np.mean(num_trialonset_short[0]),np.mean(num_trialonset_short[1]),np.mean(num_trialonset_short[2])]#,np.mean(num_trialonset_short[5])]
        num_trialonset_long = [np.mean(num_trialonset_long[0]),np.mean(num_trialonset_long[1]),np.mean(num_trialonset_long[2])]#,np.mean(num_trialonset_long[5])]
        num_lmcenter_short = [np.mean(num_lmcenter_short[0]),np.mean(num_lmcenter_short[1]),np.mean(num_lmcenter_short[2])]#,np.mean(num_lmcenter_short[5])]
        num_lmcenter_long = [np.mean(num_lmcenter_long[0]),np.mean(num_lmcenter_long[1]),np.mean(num_lmcenter_long[2])]#,np.mean(num_lmcenter_long[5])]
        num_reward_short = [np.mean(num_reward_short[0]),np.mean(num_reward_short[1]),np.mean(num_reward_short[2])]#,np.mean(num_reward_short[5])]
        num_reward_long = [np.mean(num_reward_long[0]),np.mean(num_reward_long[1]),np.mean(num_reward_long[2])]#,np.mean(num_reward_long[5])]

        totrois_trialonset_short = [np.sum(totrois_trialonset_short[0]),np.sum(totrois_trialonset_short[1]),np.sum(totrois_trialonset_short[2])]#,np.sum(totrois_trialonset_short[5])]
        totrois_trialonset_long = [np.sum(totrois_trialonset_long[0]),np.sum(totrois_trialonset_long[1]),np.sum(totrois_trialonset_long[2])]#,np.sum(totrois_trialonset_long[5])]
        totrois_lmcenter_short = [np.sum(totrois_lmcenter_short[0]),np.sum(totrois_lmcenter_short[1]),np.sum(totrois_lmcenter_short[2])]#,np.sum(totrois_lmcenter_short[5])]
        totrois_lmcenter_long = [np.sum(totrois_lmcenter_long[0]),np.sum(totrois_lmcenter_long[1]),np.sum(totrois_lmcenter_long[2])]#,np.sum(totrois_lmcenter_long[5])]
        totrois_reward_short = [np.sum(totrois_reward_short[0]),np.sum(totrois_reward_short[1]),np.sum(totrois_reward_short[2])]#,np.sum(totrois_reward_short[5])]
        totrois_reward_long = [np.sum(totrois_reward_long[0]),np.sum(totrois_reward_long[1]),np.sum(totrois_reward_long[2])]#,np.sum(totrois_reward_long[5])]

        scatter_color_short = '#F58020'
        scatter_color_long = '#374D9E'
        fraction_plot_color_short = '0.80'
        fraction_plot_color_long = '0.85'

    ax_object.scatter(np.full_like(num_trialonset_short,0,dtype=np.double),np.array(num_trialonset_short),c=fraction_plot_color_short,linewidths=0,s=80,zorder=2)
    ax_object2.scatter(np.full_like(num_trialonset_long,0,dtype=np.double),np.array(num_trialonset_long),c=fraction_plot_color_long,linewidths=0,s=80,zorder=2)
    ax_object.scatter(np.full_like(num_lmcenter_short,0.6,dtype=np.double),np.array(num_lmcenter_short),c=fraction_plot_color_short,linewidths=0,s=80,zorder=2)
    ax_object2.scatter(np.full_like(num_lmcenter_long,0.6,dtype=np.double),np.array(num_lmcenter_long),c=fraction_plot_color_long,linewidths=0,s=80,zorder=2)
    ax_object.scatter(np.full_like(num_reward_short,1.2,dtype=np.double),np.array(num_reward_short),c=fraction_plot_color_short,linewidths=0,s=80,zorder=2)
    ax_object2.scatter(np.full_like(num_reward_long,1.2,dtype=np.double),np.array(num_reward_long),c=fraction_plot_color_long,linewidths=0,s=80,zorder=2)
    # print('### fractions:')
    # print(np.array(num_lmcenter_long))


    flacs = []
    for rmp in result_max_peak['fl_amplitude_close_short']:
        if np.count_nonzero(~np.isnan(np.array(rmp))) > 5:
            flacs.append(np.nanmean(rmp))
    flafs = []
    for rmp in result_max_peak['fl_amplitude_far_short']:
        if np.count_nonzero(~np.isnan(np.array(rmp))) > 5:
            flafs.append(np.nanmean(rmp))
    flacl = []
    for rmp in result_max_peak['fl_amplitude_close_long']:
        if np.count_nonzero(~np.isnan(np.array(rmp))) > 5:
            flacl.append(np.nanmean(rmp))
    flafl = []
    for rmp in result_max_peak['fl_amplitude_far_long']:
        if np.count_nonzero(~np.isnan(np.array(rmp))) > 5:
            flafl.append(np.nanmean(rmp))

    print('--- CLOSE VS FAR TRANSIENT AMPLITUDE ---')
    print(sp.stats.ttest_ind(flacs,flafs))
    print(sp.stats.ttest_ind(flacl,flafl))
    print(sp.stats.ttest_ind(np.append(flacs,flacl),np.append(flafs,flafl)))
    print(np.append(flacs,flacl).shape[0],np.append(flafs,flafl).shape[0])
    print(np.mean(np.append(flacs,flacl)), np.mean(np.append(flafs,flafl)))
    print('----------------------------------------')
    # ipdb.set_trace()

    parts = ax_object25.boxplot([np.append(flacs,flacl), np.append(flafs,flafl)],
        patch_artist=True,showfliers=False,
        whiskerprops=dict(color='w', linestyle='-', linewidth=0, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        capprops=dict(color='w', alpha=0.0),
        widths=(0.75,0.75),positions=(0,1))
    #
    colors = ['0.8', '0.5']
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set_edgecolor(color[1])
        patch.set_alpha(0.5)
        patch.set_linewidth(0)

    x_pos_short = np.full_like(np.append(flacs,flacl),0) + (np.random.randn(np.append(flacs,flacl).shape[0]) * 0.075)
    x_pos_long = np.full_like(np.append(flafs,flafl),1) + (np.random.randn(np.append(flafs,flafl).shape[0]) * 0.075)
    ax_object25.scatter(x_pos_short, np.append(flacs,flacl),s=40, edgecolors='0.8', facecolors='none', label='short')
    ax_object25.scatter(x_pos_long, np.append(flafs,flafl),s=40, edgecolors='0.5', facecolors='none', label='long')
    #
    # ax_object25.scatter(np.full_like(flacs,0), flacs)
    # ax_object25.scatter(np.full_like(flafs,1), flafs)
    # ax_object25.scatter(np.full_like(flacl,3), flacl)
    # ax_object25.scatter(np.full_like(flafl,4), flafl)
    #
    # ax_object25.scatter([0], np.nanmean(flacs), color='r', s=100)
    # ax_object25.scatter([1], np.nanmean(flafs), color='r', s=100)
    # ax_object25.scatter([3], np.nanmean(flacl), color='r', s=100)
    # ax_object25.scatter([4], np.nanmean(flafl), color='r', s=100)



    # connect datapoints with lines
    for i in range(len(num_trialonset_short)):
        ax_object.plot([0.0,0.6,1.2],[num_trialonset_short[i],num_lmcenter_short[i],num_reward_short[i]],zorder=0,c=fraction_plot_color_short,lw=2)
    ax_object.plot([0.0,0.6,1.2],[np.mean(  np.array(num_trialonset_short)),np.mean(np.array(num_lmcenter_short)),np.mean(np.array(num_reward_short))],zorder=2,c='k',lw=5,ls='-')

    for i in range(len(num_trialonset_short)):
        ax_object2.plot([0.0,0.6,1.2],[num_trialonset_long[i],num_lmcenter_long[i],num_reward_long[i]],zorder=0,c=fraction_plot_color_long,lw=2)
    ax_object2.plot([0.0,0.6,1.2],[np.mean(np.array(num_trialonset_long)),np.mean(np.array(num_lmcenter_long)),np.mean(np.array(num_reward_long))],zorder=2,c='k',lw=5,ls='-')

    print('--- NEURON CLASSIFIER FRACTIONS ---')
    print('trial start (short and long) +/- SEM: ', str(np.mean([np.mean(np.array(num_trialonset_short)), np.mean(np.array(num_trialonset_long))])), ' +/- ', str(sp.stats.sem([np.mean(np.array(num_trialonset_short)), np.mean(np.array(num_trialonset_long))])))
    print('landmark (short and long) +/- SEM: ', str(np.mean([np.mean(np.array(num_lmcenter_short)), np.mean(np.array(num_lmcenter_long))])), ' +/- ', str(sp.stats.sem([np.mean(np.array(num_lmcenter_short)), np.mean(np.array(num_lmcenter_long))])))
    print('reward (short and long) +/- SEM: ', str(np.mean([np.mean(np.array(num_reward_short)),np.mean(np.array(num_reward_long))])), ' +/- ', str(sp.stats.sem([np.mean(np.array(num_reward_short)),np.mean(np.array(num_reward_long))])))
    print('-----------------------------------')
    # plot mean datapoint across animals
    ax_object.scatter(0, np.mean(np.array(num_trialonset_short)),marker='s',c='#39B54A',s=300,linewidths=0,zorder=3)
    ax_object2.scatter(0, np.mean(np.array(num_trialonset_long)),marker='s',c='#39B54A',s=300,linewidths=0,zorder=3)
    ax_object.scatter(0.6, np.mean(np.array(num_lmcenter_short)),marker='s',c='#FF0000',s=300,linewidths=0,zorder=3)
    ax_object2.scatter(0.6, np.mean(np.array(num_lmcenter_long)),marker='s',c='#FF0000',s=300,linewidths=0,zorder=3)
    ax_object.scatter(1.2, np.mean(np.array(num_reward_short)),marker='s',c='#29ABE2',s=300,linewidths=0,zorder=3)
    ax_object2.scatter(1.2, np.mean(np.array(num_reward_long)),marker='s',c='#29ABE2',s=300,linewidths=0,zorder=3)

    ax_object.errorbar([0.0,0.6,1.2],[np.mean(np.array(num_trialonset_short)),np.mean(np.array(num_lmcenter_short)),np.mean(np.array(num_reward_short))],yerr=[sp.stats.sem(np.array(num_trialonset_short)),sp.stats.sem(np.array(num_lmcenter_short)),sp.stats.sem(np.array(num_reward_short))],elinewidth=3,capsize=6,capthick=3,zorder=2,c='k',lw=5,ls='-')
    ax_object.errorbar([0.0],[np.mean(np.array(num_trialonset_short))],yerr=[sp.stats.sem(np.array(num_trialonset_short))],elinewidth=4,capsize=6,capthick=4,zorder=3,c='#39B54A',lw=5,ls='-')
    ax_object.errorbar([0.6],[np.mean(np.array(num_lmcenter_short))],yerr=[sp.stats.sem(np.array(num_lmcenter_short))],elinewidth=4,capsize=6,capthick=4,zorder=3,c='#FF0000',lw=5,ls='-')
    ax_object.errorbar([1.2],[np.mean(np.array(num_reward_short))],yerr=[sp.stats.sem(np.array(num_reward_short))],elinewidth=4,capsize=6,capthick=4,zorder=3,c='#29ABE2',lw=5,ls='-')

    ax_object2.errorbar([0.0,0.6,1.2],[np.mean(np.array(num_trialonset_long)),np.mean(np.array(num_lmcenter_long)),np.mean(np.array(num_reward_long))],yerr=[sp.stats.sem(np.array(num_trialonset_long)),sp.stats.sem(np.array(num_lmcenter_long)),sp.stats.sem(np.array(num_reward_long))],elinewidth=3,capsize=6,capthick=3,zorder=2,ecolor='g',c='k',lw=5,ls='-')
    ax_object2.errorbar([0.0],[np.mean(np.array(num_trialonset_long))],yerr=[sp.stats.sem(np.array(num_trialonset_long))],elinewidth=4,capsize=6,capthick=4,zorder=3,c='#39B54A',lw=5,ls='-')
    ax_object2.errorbar([0.6],[np.mean(np.array(num_lmcenter_long))],yerr=[sp.stats.sem(np.array(num_lmcenter_long))],elinewidth=4,capsize=6,capthick=4,zorder=3,c='#FF0000',lw=5,ls='-')
    ax_object2.errorbar([1.2],[np.mean(np.array(num_reward_long))],yerr=[sp.stats.sem(np.array(num_reward_long))],elinewidth=4,capsize=6,capthick=4,zorder=3,c='#29ABE2',lw=5,ls='-')

    if plot_layer is 'all':
        # loop through every mouse and calculate individual fracctions
        ts_fractions = []
        for i in range(len(task_scores)):
            ts_fractions.append(np.mean([num_trialonset_short[i],num_trialonset_long[i], num_lmcenter_short[i],num_lmcenter_long[i], num_reward_short[i],num_reward_long[i]]))

        reg_res_short = sp.stats.theilslopes(ts_fractions,task_scores)
        ax_object12.plot(task_scores, reg_res_short[1] + reg_res_short[0] * np.array(task_scores), c='k', lw=4)
        print('----- TASK SCORE FRACTION CORRELATION ANALYSIS -----')
        r,p = sp.stats.pearsonr(ts_fractions,task_scores)
        print(sp.stats.pearsonr(ts_fractions,task_scores))
        # ipdb.set_trace()
        # ax_object12.scatter(task_scores, num_trialonset_short, c='#39B54A',s=300)
        # ax_object12.scatter(task_scores, num_trialonset_long, c='#39B54A',s=300)
        # ax_object12.scatter(task_scores, num_lmcenter_short, c='#FF0000',s=300)
        # ax_object12.scatter(task_scores, num_lmcenter_long, c='#FF0000',s=300)
        # ax_object12.scatter(task_scores, num_reward_short, c='#29ABE2',s=300)
        # ax_object12.scatter(task_scores, num_reward_long, c='#29ABE2',s=300)

        ax_object12.text(60,0.4, 'p = ' + str(np.round(p,3)))
        ax_object12.set_xlabel('Task Score')
        ax_object12.set_ylabel('Fraction of Neurons')

        ax_object12.set_xlim([0,80])
        ax_object12.set_ylim([-0.05,0.5])
        ax_object12.spines['bottom'].set_linewidth(4)
        ax_object12.spines['top'].set_visible(False)
        ax_object12.spines['right'].set_visible(False)
        ax_object12.spines['left'].set_linewidth(4)
        ax_object12.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=18, \
            length=6, \
            width=4, \
            bottom='on', \
            left='on', \
            right='off', \
            top='off')

    donut_labels = ['unassigned','reward','landmark','trial onset']
    all_valid_rois = np.sum(np.array(tot_num_valid_rois))
    totrois_trialonset_short = np.sum(np.array(totrois_trialonset_short))
    totrois_trialonset_long = np.sum(np.array(totrois_trialonset_long))
    totrois_lmcenter_short = np.sum(np.array(totrois_lmcenter_short))
    totrois_lmcenter_long = np.sum(np.array(totrois_lmcenter_long))
    totrois_reward_short = np.sum(np.array(totrois_reward_short))
    totrois_reward_long = np.sum(np.array(totrois_reward_long))

    unassigned_rois_short = all_valid_rois - (totrois_trialonset_short + totrois_lmcenter_short + totrois_reward_short)
    unassigned_rois_long = all_valid_rois - (totrois_trialonset_long + totrois_lmcenter_long + totrois_reward_long)

    # print(totrois_trialonset_short,totrois_lmcenter_short,totrois_reward_short,unassigned_rois_short)

    donut_sections_short = np.array([unassigned_rois_short, totrois_reward_short, totrois_lmcenter_short, totrois_trialonset_short])
    donut_sections_long = np.array([unassigned_rois_long, totrois_reward_long, totrois_lmcenter_long, totrois_trialonset_long])
    donut_colors = ['0.85','#29ABE2','#FF0000','#39B54A']

    print(donut_labels)
    print(donut_sections_short)
    print(donut_sections_long)

    ax_object6.pie(donut_sections_short, labels=donut_labels, colors=donut_colors, startangle=90 , wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
    center_circle = plt.Circle( (0,0), 0.5, color='white')
    ax_object6.add_artist(center_circle)

    ax_object7.pie(donut_sections_long, labels=donut_labels, colors=donut_colors, startangle=90 , wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
    center_circle = plt.Circle( (0,0), 0.5, color='white')
    ax_object7.add_artist(center_circle)

    # normalize_fractions = True
    # if normalize_fractions:
    #     for nto in np.array(num_trialonset_short):


    # carry out statistical analysis. This is not (yet) the correct test: we are treating each group independently, rather than taking into account within-group and between-group variance
    print('--- SHORT TRIALS ANOVA ---')
    print(sp.stats.f_oneway(np.array(num_trialonset_short),np.array(num_lmcenter_short),np.array(num_reward_short)))
    group_labels = ['trialonset_short'] * np.array(num_trialonset_short).shape[0] + \
                   ['lmcenter_short'] * np.array(num_lmcenter_short).shape[0] + \
                   ['reward_short'] * np.array(num_reward_short).shape[0]

    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(num_trialonset_short),np.array(num_lmcenter_short),np.array(num_reward_short))),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)

    print('--- LONG TRIALS ANOVA ---')
    print(sp.stats.f_oneway(np.array(num_trialonset_long),np.array(num_lmcenter_long),np.array(num_reward_long)))
    group_labels = ['trialonset_long'] * np.array(num_trialonset_long).shape[0] + \
                   ['lmcenter_long'] * np.array(num_lmcenter_long).shape[0] + \
                   ['reward_long'] * np.array(num_reward_long).shape[0]
    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(num_trialonset_long),np.array(num_lmcenter_long),np.array(num_reward_long))),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)

    # annotate plots and set axis limits
    ax_object.set_xticks([0,0.6,1.2])
    ax_object.set_xticklabels(['trial onset','landmark','reward'], rotation=45, fontsize=24)
    ax_object2.set_xticks([0,0.6,1.2])
    ax_object2.set_xticklabels(['trial onset','landmark','reward'], rotation=45, fontsize=24)
    ax_object.set_ylabel('fraction of neurons', fontsize=24)
    ax_object2.set_ylabel('')

    ax_object.spines['left'].set_linewidth(2)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['bottom'].set_linewidth(2)
    ax_object.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax_object2.spines['left'].set_linewidth(2)
    ax_object2.spines['top'].set_visible(False)
    ax_object2.spines['right'].set_visible(False)
    ax_object2.spines['bottom'].set_linewidth(2)
    ax_object2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax_object3.spines['bottom'].set_linewidth(2)
    ax_object3.spines['top'].set_visible(False)
    ax_object3.spines['right'].set_visible(False)
    ax_object3.spines['left'].set_linewidth(2)
    ax_object3.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=4, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax_object4.spines['bottom'].set_linewidth(2)
    ax_object4.spines['top'].set_visible(False)
    ax_object4.spines['right'].set_visible(False)
    ax_object4.spines['left'].set_linewidth(2)
    ax_object4.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=4, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    max_y = np.amax(ax_object.get_ylim())
    if np.amax(ax_object2.get_ylim()) > max_y:
        max_y = np.amax(ax_object2.get_ylim())

    ax_object.set_ylim([0,0.5])
    ax_object2.set_ylim([0,0.5])

    # plot peak response in VR vs peak response in OL (at time of peak in VR [taking into account allowed timewindow])
    ax_object15.hist(np.array(result_max_peak['trialonset_peakloc_short']), bins=20, range=[-20,360], histtype='step', color='#39B54A')
    ax_object15.hist(np.array(result_max_peak['lmcenter_peakloc_short'])+220, bins=20, range=[-20,360], histtype='step', color='#FF0000')
    ax_object15.hist(np.array(result_max_peak['reward_peakloc_short'])+320, bins=20, range=[-20,360], histtype='step', color='#29ABE2')
    # ax_object15.axvline(np.median(np.array(result_max_peak['trialonset_peakloc_short'])), color='#39B54A', lw=2, ls='--')
    # ax_object15.axvline(np.median(np.array(result_max_peak['lmcenter_peakloc_short'])+220), color='#FF0000', lw=2, ls='--')
    # ax_object15.axvline(np.median(np.array(result_max_peak['reward_peakloc_short'])+320), color='#29ABE2', lw=2, ls='--')
    ax_object15.axvline(0, color='#39B54A', lw=2, ls='--')
    ax_object15.axvline(220, color='#FF0000', lw=2, ls='--')
    ax_object15.axvline(320, color='#29ABE2', lw=2, ls='--')
    ax_object15.set_xlim([-20,380])

    bp = ax_object17.boxplot([np.array(result_max_peak['reward_peakloc_short'])+320,np.array(result_max_peak['lmcenter_peakloc_short'])+220,np.array(result_max_peak['trialonset_peakloc_short'])],
                             vert=False, patch_artist=True, bootstrap=1000, showcaps=False, whiskerprops=dict(linestyle='-', color='black', linewidth=5, solid_capstyle='butt'),
                             medianprops=dict(color='black', linewidth=5, solid_capstyle='butt'))

    colors = ['#29ABE2', '#FF0000', '#39B54A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax_object17.set_xlim([-20,380])

    # plot peak response in VR vs peak response in OL (at time of peak in VR [taking into account allowed timewindow])
    ax_object16.hist(np.array(result_max_peak['trialonset_peakloc_long']), bins=24, range=[-20,420], histtype='step', color='#39B54A')
    ax_object16.hist(np.array(result_max_peak['lmcenter_peakloc_long'])+220, bins=24, range=[-20,420], histtype='step', color='#FF0000')
    ax_object16.hist(np.array(result_max_peak['reward_peakloc_long'])+380, bins=24, range=[-20,420], histtype='step', color='#29ABE2')
    # ax_object16.axvline(np.median(np.array(result_max_peak['trialonset_peakloc_long'])), color='#39B54A', lw=2, ls='--')
    # ax_object16.axvline(np.median(np.array(result_max_peak['lmcenter_peakloc_long'])+220), color='#FF0000', lw=2, ls='--')
    # ax_object16.axvline(np.median(np.array(result_max_peak['reward_peakloc_long'])+380), color='#29ABE2', lw=2, ls='--')
    ax_object16.axvline(0, color='#39B54A', lw=2, ls='--')
    ax_object16.axvline(220, color='#FF0000', lw=2, ls='--')
    ax_object16.axvline(380, color='#29ABE2', lw=2, ls='--')
    ax_object16.set_xlim([-20,440])

    bp = ax_object18.boxplot([np.array(result_max_peak['reward_peakloc_long'])+380,np.array(result_max_peak['lmcenter_peakloc_long'])+220,np.array(result_max_peak['trialonset_peakloc_long'])],
                             vert=False, patch_artist=True, bootstrap=1000, showcaps=False, whiskerprops=dict(linestyle='-', color='black', linewidth=5, solid_capstyle='butt'),
                             medianprops=dict(color='black', linewidth=5, solid_capstyle='butt'))

    colors = ['#29ABE2', '#FF0000', '#39B54A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0)

    ax_object18.set_xlim([-20,440])


    landmark_transient_distance_short = np.abs(np.array(result_max_peak['lmcenter_peakloc_short']))
    landmark_transient_distance_long = np.abs(np.array(result_max_peak['lmcenter_peakloc_long']))
    print('--- ABSOLUTE MEAN DISTANCE OF TRANSIENTS FROM ANCHOR POINT --- ')
    print('TRIAL ONSET SHORT: ' + str(np.mean(np.abs(np.array(result_max_peak['trialonset_peakloc_short'])))))
    print('TRIAL ONSET LONG: ' + str(np.mean(np.abs(np.array(result_max_peak['trialonset_peakloc_long'])))))
    print('LMCENTER SHORT: ' + str(np.mean(landmark_transient_distance_short)))
    print('LMCENTER LONG: ' + str(np.mean(landmark_transient_distance_long)))
    print('REWARD SHORT: ' + str(np.mean(np.abs(np.array(result_max_peak['reward_peakloc_short'])))))
    print('REWARD LONG: ' + str(np.mean(np.abs(np.array(result_max_peak['reward_peakloc_long'])))))
    print('---------------------------------------------------------------')

    reward_all = np.concatenate((np.array(result_max_peak['reward_peakloc_long'])+380,np.array(result_max_peak['reward_peakloc_short'])+380))
    landmark_all = np.concatenate((np.array(result_max_peak['lmcenter_peakloc_long'])+220,np.array(result_max_peak['lmcenter_peakloc_short'])+220))
    trialonset_all = np.concatenate((np.array(result_max_peak['trialonset_peakloc_long']),np.array(result_max_peak['trialonset_peakloc_short'])))
    # plot peak response in VR vs peak response in OL (at time of peak in VR [taking into account allowed timewindow])
    ax_object19.hist(trialonset_all, bins=np.arange(-20,420,20), histtype='step', color='#39B54A', linewidth=3)
    ax_object19.hist(landmark_all, bins=np.arange(-20,420,20), histtype='step', color='#FF0000', linewidth=3)
    ax_object19.hist(reward_all, bins=np.arange(-20,420,20), histtype='step', color='#29ABE2', linewidth=3)

    bp = ax_object20.boxplot([reward_all,landmark_all,trialonset_all],
                              vert=False, patch_artist=True, bootstrap=1000, showcaps=False, whiskerprops=dict(linestyle='-', color='black', linewidth=5, solid_capstyle='butt'),
                              medianprops=dict(color='black', linewidth=5, solid_capstyle='butt'))

    colors = ['#29ABE2', '#FF0000', '#39B54A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0)

    ax_object19.set_xlim([-40,440])
    ax_object19.axvline(0, color='#39B54A', lw=3, ls='--')
    ax_object19.axvline(220, color='#FF0000', lw=3, ls='--')
    ax_object19.axvline(380, color='#29ABE2', lw=3, ls='--')
    ax_object19.set_xticks([0,220,380])
    ax_object19.set_xticklabels(['trial onset','landmark','reward'], rotation=45, fontsize=22)
    ax_object19.spines['bottom'].set_linewidth(2)
    ax_object19.spines['top'].set_visible(False)
    ax_object19.spines['right'].set_visible(False)
    ax_object19.spines['left'].set_linewidth(2)
    ax_object19.tick_params( \
        axis='both', \
        direction='put', \
        labelsize=17, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    # ax_object20.set_xticks([0,220,380])
    # ax_object20.set_xticklabels(['trial onset','landmark','reward'], rotation=45, fontsize=22)
    ax_object20.set_xlim([-40,440])
    ax_object20.set_yticks([])
    ax_object20.spines['bottom'].set_visible(False)
    ax_object20.spines['top'].set_visible(False)
    ax_object20.spines['right'].set_linewidth(1)
    ax_object20.spines['left'].set_linewidth(1)
    ax_object20.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=17, \
        length=4, \
        width=2, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')


    print('--- PRE VS POST LANDMARK ---')
    # print(sp.stats.ttest_ind(np.abs(np.array(result_max_peak['lmcenter_peakloc_short'])[np.array(result_max_peak['lmcenter_peakloc_short'])<-20]),np.array(result_max_peak['lmcenter_peakloc_short'])[np.array(result_max_peak['lmcenter_peakloc_short'])>20]))
    # print(sp.stats.ttest_ind(np.abs(np.array(result_max_peak['lmcenter_peakloc_long'])[np.array(result_max_peak['lmcenter_peakloc_long'])<-20]),np.array(result_max_peak['lmcenter_peakloc_long'])[np.array(result_max_peak['lmcenter_peakloc_long'])>20]))
    lm_dist_prelm = np.concatenate((np.abs(np.array(result_max_peak['lmcenter_peakloc_short'])[np.array(result_max_peak['lmcenter_peakloc_short'])<0]),np.abs(np.array(result_max_peak['lmcenter_peakloc_long'])[np.array(result_max_peak['lmcenter_peakloc_long'])<0])))
    lm_dist_postlm = np.concatenate((np.array(result_max_peak['lmcenter_peakloc_short'])[np.array(result_max_peak['lmcenter_peakloc_short'])>0],np.array(result_max_peak['lmcenter_peakloc_long'])[np.array(result_max_peak['lmcenter_peakloc_long'])>0]))
    # lm_dist_prelm = np.abs(np.array(result_max_peak['lmcenter_peakloc_long'])[np.array(result_max_peak['lmcenter_peakloc_long'])<0])
    # lm_dist_postlm = np.array(result_max_peak['lmcenter_peakloc_long'])[np.array(result_max_peak['lmcenter_peakloc_long'])>0]

    print(sp.stats.ttest_ind(lm_dist_prelm,lm_dist_postlm))
    print('----------------------------')

    print('--- PEAK LOCATION ---')
    print('median TRIALONSET SHORT: ' + str(np.median(np.array(result_max_peak['trialonset_peakloc_short']))))
    print('median LMCENTER SHORT: ' + str(np.median(np.array(result_max_peak['lmcenter_peakloc_short'])+220)))
    print('median REWARD SHORT: ' + str(np.median(np.array(result_max_peak['reward_peakloc_short'])+220)))
    print('median TRIALONSET LONG: ' + str(np.median(np.array(result_max_peak['trialonset_peakloc_long']))))
    print('median LMCENTER LONG: ' + str(np.median(np.array(result_max_peak['lmcenter_peakloc_long'])+220)))
    print('median REWARD LONG: ' + str(np.median(np.array(result_max_peak['reward_peakloc_long'])+220)))
    print('--------------------')

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA42E', '#00FF12']
    ax_object3.scatter(np.asarray(result_max_peak['trialonset_peakval_short']),np.asarray(result_max_peak_ol['trialonset_peakval_short']), s=10,color=scatter_color_short, label='short')
    ax_object3.scatter(np.asarray(result_max_peak['trialonset_peakval_long']),np.asarray(result_max_peak_ol['trialonset_peakval_long']) ,s=10,color=scatter_color_long, label='long')
    ax_object3.scatter(np.asarray(result_max_peak['lmcenter_peakval_short']),np.asarray(result_max_peak_ol['lmcenter_peakval_short']) ,s=10,color=scatter_color_short)
    ax_object3.scatter(np.asarray(result_max_peak['lmcenter_peakval_long']),np.asarray(result_max_peak_ol['lmcenter_peakval_long']) ,s=10,color=scatter_color_long)
    ax_object3.scatter(np.asarray(result_max_peak['reward_peakval_short']),np.asarray(result_max_peak_ol['reward_peakval_short']) ,s=10,color=scatter_color_short)
    ax_object3.scatter(np.asarray(result_max_peak['reward_peakval_long']),np.asarray(result_max_peak_ol['reward_peakval_long']) ,s=10,color=scatter_color_long)


    # plot peak response ni VR vs peak response in OL (anywhere)
    # ax_object8.scatter(np.asarray(result_max_peak['trialonset_meanpeak_short']), np.asarray(result_max_peak_ol['trialonset_meanpeak_short']),s=10,color='#39B54A', label='short')
    # ax_object8.scatter(np.asarray(result_max_peak['trialonset_meanpeak_long']), np.asarray(result_max_peak_ol['trialonset_meanpeak_long']),s=10,color='#39B54A', label='long')
    # ax_object8.scatter(np.asarray(result_max_peak['lmcenter_meanpeak_short']), np.asarray(result_max_peak_ol['lmcenter_meanpeak_short']),s=10,color='#FF0000')
    # ax_object8.scatter(np.asarray(result_max_peak['lmcenter_meanpeak_long']), np.asarray(result_max_peak_ol['lmcenter_meanpeak_long']),s=10,color='#FF0000')
    # ax_object8.scatter(np.asarray(result_max_peak['reward_meanpeak_short']), np.asarray(result_max_peak_ol['reward_meanpeak_short']),s=10,color='#29ABE2')
    # ax_object8.scatter(np.asarray(result_max_peak['reward_meanpeak_long']), np.asarray(result_max_peak_ol['reward_meanpeak_long']),s=10,color='#29ABE2')


    print('--- ROBUSTNESS AND AMPLITUDE SUCCESSFUL VS DEFAULT TRIALS ---')
    # ipdb.set_trace()
    succ_short = np.concatenate((result_max_peak['success_robustness_short'],result_max_peak['success_robustness_long']))
    def_short = np.concatenate((result_max_peak['default_robustness_short'],result_max_peak['default_robustness_long']))
    def_short = def_short[~np.isnan(succ_short)]
    succ_short = succ_short[~np.isnan(succ_short)]
    succ_short = succ_short[~np.isnan(def_short)]
    def_short = def_short[~np.isnan(def_short)]
    print(sp.stats.ttest_rel(succ_short, def_short))
    print(np.mean(succ_short), sp.stats.sem(succ_short))
    print(np.mean(def_short), sp.stats.sem(def_short))
    print(len(succ_short),len(def_short))
    succ_short = np.concatenate((result_max_peak['success_amplitude_short'],result_max_peak['success_amplitude_long']))
    def_short = np.concatenate((result_max_peak['default_amplitude_short'],result_max_peak['default_amplitude_long']))
    def_short = def_short[~np.isnan(succ_short)]
    succ_short = succ_short[~np.isnan(succ_short)]
    succ_short = succ_short[~np.isnan(def_short)]
    def_short = def_short[~np.isnan(def_short)]
    print(sp.stats.ttest_rel(succ_short, def_short))
    print(np.mean(succ_short), sp.stats.sem(succ_short))
    print(np.mean(def_short), sp.stats.sem(def_short))
    print(len(succ_short),len(def_short))
    print('--------------------------------------------------------------')
    # ipdb.set_trace()

    ax_object14.scatter(np.asarray(result_max_peak['space_peak_short']), np.asarray(result_max_peak_ol['space_peak_short']), s=7, color=scatter_color_short, label='short')
    ax_object14.scatter(np.asarray(result_max_peak['space_peak_long']), np.asarray(result_max_peak_ol['space_peak_long']), s=7, color=scatter_color_long, label='long')

    non_nan_short = np.asarray(result_max_peak['space_speedcorr_short'])[~np.isnan(result_max_peak['space_speedcorr_short'])]
    non_nan_long = np.asarray(result_max_peak['space_speedcorr_long'])[~np.isnan(result_max_peak['space_speedcorr_long'])]
    non_nan_short_ol = np.asarray(result_max_peak_ol['space_speedcorr_short'])[~np.isnan(result_max_peak_ol['space_speedcorr_short'])]
    non_nan_long_ol =np.asarray(result_max_peak_ol['space_speedcorr_long'])[~np.isnan(result_max_peak_ol['space_speedcorr_long'])]
    # parts = ax_object21.boxplot([non_nan_short, non_nan_long,non_nan_short_ol,non_nan_long_ol],
    #     patch_artist=True,showfliers=False,
    #     whiskerprops=dict(color='w', linestyle='-', linewidth=0, solid_capstyle='butt'),
    #     medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
    #     capprops=dict(color='w', alpha=0.0),
    #     widths=(0.75,0.75, 0.75,0.75),positions=(0,1,3,4))
    #
    # colors = [scatter_color_short, scatter_color_long, scatter_color_short, scatter_color_long]
    # for patch, color in zip(parts['boxes'], colors):
    #     patch.set_facecolor(color)
    #     # patch.set_edgecolor(color[1])
    #     patch.set_alpha(0.5)
    #     patch.set_linewidth(0)


    ax_object21.scatter(np.full(len(non_nan_short),0), non_nan_short,s=80, facecolors='none', edgecolors=scatter_color_short, label='short')
    ax_object21.scatter(np.full(len(non_nan_long),1), non_nan_long,s=80, facecolors='none', edgecolors=scatter_color_long, label='long')
    ax_object21.scatter(np.full(len(non_nan_short_ol),3), non_nan_short_ol,s=80, facecolors='none', edgecolors=scatter_color_short, label='short')
    ax_object21.scatter(np.full(len(non_nan_long_ol),4), non_nan_long_ol,s=80, facecolors='none', edgecolors=scatter_color_long, label='long')

    sns.distplot(non_nan_short, bins=np.arange(-1,1.02,0.02), hist=False, vertical=True,color=scatter_color_short, kde_kws={'shade':True, 'gridsize':300, 'lw':2, 'cut':6 } ,ax=ax_object22)
    sns.distplot(non_nan_long, bins=np.arange(-1,1.02,0.02), hist=False, vertical=True,color=scatter_color_long, kde_kws={'shade':True, 'gridsize':300, 'lw':2, 'cut':6 } , ax=ax_object22)
    ax_object22.invert_xaxis()
    sns.distplot(non_nan_short_ol, bins=np.arange(-1,1.02,0.02), hist=False, vertical=True,color=scatter_color_short, kde_kws={'shade':True, 'gridsize':300, 'lw':2, 'cut':6 } , ax=ax_object23)
    sns.distplot(non_nan_long_ol, bins=np.arange(-1,1.02,0.02), hist=False, vertical=True,color=scatter_color_long, kde_kws={'shade':True, 'gridsize':300, 'lw':2, 'cut':6 } , ax=ax_object23)


    non_nan_short = np.asarray(result_max_peak['space_speedcorr_p_short'])[~np.isnan(result_max_peak['space_speedcorr_p_short'])]
    non_nan_long = np.asarray(result_max_peak['space_speedcorr_p_long'])[~np.isnan(result_max_peak['space_speedcorr_p_long'])]
    speedcorr_r_short = np.asarray(result_max_peak['space_speedcorr_r_short'])[~np.isnan(result_max_peak['space_speedcorr_p_short'])]
    speedcorr_r_long = np.asarray(result_max_peak['space_speedcorr_r_long'])[~np.isnan(result_max_peak['space_speedcorr_p_long'])]
    speedcorr_slope_short = np.asarray(result_max_peak['space_speedcorr_short'])[~np.isnan(result_max_peak['space_speedcorr_p_short'])]
    speedcorr_slope_long = np.asarray(result_max_peak['space_speedcorr_long'])[~np.isnan(result_max_peak['space_speedcorr_p_long'])]

    x_pos_short = np.full(len(speedcorr_r_short),0) + (np.random.randn(len(speedcorr_r_short)) * 0.1)
    x_pos_long = np.full(len(speedcorr_r_long),1) + (np.random.randn(len(speedcorr_r_long)) * 0.1)
    ax_object24.scatter(x_pos_short, speedcorr_r_short,s=40, edgecolors=scatter_color_short, facecolors='none', label='short')
    ax_object24.scatter(x_pos_long, speedcorr_r_long,s=40, edgecolors=scatter_color_long, facecolors='none', label='long')

    parts = ax_object24.boxplot([speedcorr_r_short, speedcorr_r_long],
        patch_artist=True,showfliers=False,
        whiskerprops=dict(color='w', linestyle='-', linewidth=0, solid_capstyle='butt'),
        medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
        capprops=dict(color='w', alpha=0.0),
        widths=(0.75,0.75),positions=(0,1))

    colors = [scatter_color_short, scatter_color_long]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set_edgecolor(color[1])
        patch.set_alpha(0.5)
        patch.set_linewidth(0)


    speedcorr_r_short = speedcorr_r_short[non_nan_short<0.05]
    speedcorr_r_long = speedcorr_r_long[non_nan_long<0.05]
    x_pos_short = np.full(len(speedcorr_r_short),0) + (np.random.randn(len(speedcorr_r_short)) * 0.1)
    x_pos_long = np.full(len(speedcorr_r_long),1) + (np.random.randn(len(speedcorr_r_long)) * 0.1)
    ax_object24.scatter(x_pos_short, speedcorr_r_short,s=40, edgecolors='k', facecolors=scatter_color_short, label='short')
    ax_object24.scatter(x_pos_long, speedcorr_r_long,s=40, edgecolors='k', facecolors=scatter_color_long, label='long')

    # pull out r and slope values for significant neurons

    num_pos_r = len(speedcorr_r_short[speedcorr_r_short>0]) + len(speedcorr_r_long[speedcorr_r_long>0])
    num_neg_r = len(speedcorr_r_short[speedcorr_r_short<0]) + len(speedcorr_r_long[speedcorr_r_long<0])
    print('------- SPEED CORRELATION VR -------')
    print('total number short: ' + str(len(non_nan_short)) + ' of which p < 0.05: ' + str(len(non_nan_short[non_nan_short<0.05])) + ' thats: ' + str(len(non_nan_short[non_nan_short<0.05])/len(non_nan_short)))
    print('total number long: ' + str(len(non_nan_long)) + ' of which p < 0.05: ' + str(len(non_nan_long[non_nan_long<0.05])) + ' thats: ' + str(len(non_nan_long[non_nan_long<0.05])/len(non_nan_long)))
    print('num r > 0 correlated: ' + str(num_pos_r))
    print('num r < 0 correlated: ' + str(num_neg_r))
    print('---------------------------------')
    # ipdb.set_trace()

    # non_nan_short = np.asarray(result_max_peak_ol['space_speedcorr_r_short'])[~np.isnan(result_max_peak_ol['space_speedcorr_r_short'])]
    # non_nan_long = np.asarray(result_max_peak_ol['space_speedcorr_r_long'])[~np.isnan(result_max_peak_ol['space_speedcorr_r_long'])]
    # parts = ax_object24.boxplot([non_nan_short, non_nan_long],
    #     patch_artist=True,showfliers=False,
    #     whiskerprops=dict(color='w', linestyle='-', linewidth=0, solid_capstyle='butt'),
    #     medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
    #     capprops=dict(color='w', alpha=0.0),
    #     widths=(0.75,0.75),positions=(3,4))
    #
    # colors = [scatter_color_short, scatter_color_long]
    # for patch, color in zip(parts['boxes'], colors):
    #     patch.set_facecolor(color)
    #     # patch.set_edgecolor(color[1])
    #     patch.set_alpha(0.5)
    #     patch.set_linewidth(0)

    print('------- SPEED CORRELATION OL -------')
    print('total number short: ' + str(len(non_nan_short)) + ' of which p < 0.05: ' + str(len(non_nan_short[non_nan_short<0.05])))
    print('total number long: ' + str(len(non_nan_long)) + ' of which p < 0.05: ' + str(len(non_nan_long[non_nan_long<0.05])))
    print('---------------------------------')

    # x_pos_short = np.full(len(non_nan_short),3) + (np.random.randn(len(non_nan_short)) * 0.075)
    # x_pos_long = np.full(len(non_nan_long),4) + (np.random.randn(len(non_nan_long)) * 0.075)
    # ax_object24.scatter(x_pos_short, non_nan_short,s=40, edgecolors=scatter_color_short, facecolors='none', label='short')
    # ax_object24.scatter(x_pos_long, non_nan_long,s=40, edgecolors=scatter_color_long, facecolors='none', label='long')
    # ax_object24.set_xlim([-1,5])
    # ax_object24.set_ylabel('r-value')
    # ax_object24.set_xticks([0.5,3.5])
    # ax_object24.set_xticklabels(['VR','DC'])

    ax_object24.set_xlim([-0.7,1.7])
    ax_object24.set_xticks([0, 1])
    ax_object24.set_xticklabels(['short', 'long'], rotation=45)
    ax_object24.set_ylim([-1,1])
    ax_object24.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_object24.set_yticklabels(['-1', '-0.5', '0','0.5', '1'])
    ax_object24.spines['bottom'].set_linewidth(2)
    ax_object24.spines['top'].set_visible(False)
    ax_object24.spines['right'].set_visible(False)
    ax_object24.spines['left'].set_linewidth(2)
    ax_object24.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=17, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')
    # sns.distplot(non_nan_short,color=scatter_color_short,ax=ax_object24)
    # sns.distplot(non_nan_long,color=scatter_color_long,ax=ax_object24)


    ax_object21.set_xlim([-0.8,4.8])
    ax_object21.set_ylim([-1,1])
    ax_object21.set_yticks([-1,0,1])
    ax_object21.set_yticklabels([])
    ax_object21.spines['bottom'].set_linewidth(2)
    ax_object21.spines['top'].set_visible(False)
    ax_object21.spines['right'].set_linewidth(2)
    ax_object21.spines['left'].set_linewidth(2)
    ax_object21.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=17, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='on', \
        top='off')

    xlim = ax_object22.get_xlim()
    ax_object22.set_xticks([xlim[1],xlim[0]])
    ax_object22.set_xticklabels([str(xlim[1]),str(xlim[0])])
    ax_object22.set_ylim([-1,1])
    ax_object22.set_yticks([-1,0,1])
    ax_object22.yaxis.tick_right()
    ax_object22.set_yticklabels(['-1','0','1'])
    # ax_object22.set_xticks([])
    ax_object22.spines['bottom'].set_linewidth(2)
    ax_object22.spines['top'].set_visible(False)
    ax_object22.spines['right'].set_linewidth(2)
    ax_object22.spines['left'].set_visible(False)
    ax_object22.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=17, \
        length=4, \
        width=2, \
        left='off', \
        bottom='on', \
        right='on', \
        top='off')

    xlim = ax_object23.get_xlim()
    ax_object23.set_xticks([xlim[1],xlim[0]])
    ax_object23.set_xticklabels([str(xlim[1]),str(xlim[0])])
    ax_object23.set_ylim([-1,1])
    ax_object23.set_yticks([-1,0,1])
    ax_object23.set_yticklabels(['-1','0','1'])
    # ax_object23.set_xticks([])
    ax_object23.spines['bottom'].set_linewidth(2)
    ax_object23.spines['top'].set_visible(False)
    ax_object23.spines['right'].set_visible(False)
    ax_object23.spines['left'].set_linewidth(2)
    ax_object23.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=17, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')


    # ax_object21.scatter(np.asarray(result_max_peak['space_peak_long']), np.asarray(result_max_peak_ol['space_peak_long']),s=7,color=scatter_color_long, label='long')
    # ipdb.set_trace()
    ax_object8.scatter(np.asarray(result_max_peak['space_peak_short_trialonset']), np.asarray(result_max_peak_ol['space_peak_short_trialonset']),s=7,color='#39B54A', label='short')
    ax_object8.scatter(np.asarray(result_max_peak['space_peak_long_trialonset']), np.asarray(result_max_peak_ol['space_peak_long_trialonset']),s=7,color='#39B54A', label='long')
    ax_object8.scatter(np.asarray(result_max_peak['space_peak_short_lmcenter']), np.asarray(result_max_peak_ol['space_peak_short_lmcenter']),s=7,color='#FF0000', label='short')
    ax_object8.scatter(np.asarray(result_max_peak['space_peak_long_lmcenter']), np.asarray(result_max_peak_ol['space_peak_long_lmcenter']),s=7,color='#FF0000', label='long')
    ax_object8.scatter(np.asarray(result_max_peak['space_peak_short_reward']), np.asarray(result_max_peak_ol['space_peak_short_reward']),s=7,color='#29ABE2', label='short')
    ax_object8.scatter(np.asarray(result_max_peak['space_peak_long_reward']), np.asarray(result_max_peak_ol['space_peak_long_reward']),s=7,color='#29ABE2', label='long')

    max_peak_all = np.concatenate((np.asarray(result_max_peak['reward_peakval_short']),np.asarray(result_max_peak['trialonset_peakval_short']),np.asarray(result_max_peak['lmcenter_peakval_short']),np.asarray(result_max_peak['reward_peakval_long']),np.asarray(result_max_peak['trialonset_peakval_long']),np.asarray(result_max_peak['lmcenter_peakval_long'])))
    max_peak_all_ol = np.concatenate((np.asarray(result_max_peak_ol['reward_peakval_short']),np.asarray(result_max_peak_ol['trialonset_peakval_short']),np.asarray(result_max_peak_ol['lmcenter_peakval_short']),np.asarray(result_max_peak_ol['reward_peakval_long']),np.asarray(result_max_peak_ol['trialonset_peakval_long']),np.asarray(result_max_peak_ol['lmcenter_peakval_long'])))
    max_peak_all_short = np.concatenate((np.asarray(result_max_peak['reward_peakval_short']),np.asarray(result_max_peak['trialonset_peakval_short']),np.asarray(result_max_peak['lmcenter_peakval_short'])))
    max_peak_all_long = np.concatenate((np.asarray(result_max_peak['reward_peakval_long']),np.asarray(result_max_peak['trialonset_peakval_long']),np.asarray(result_max_peak['lmcenter_peakval_long'])))
    max_peak_all_short_ol = np.concatenate((np.asarray(result_max_peak_ol['reward_peakval_short']),np.asarray(result_max_peak_ol['trialonset_peakval_short']),np.asarray(result_max_peak_ol['lmcenter_peakval_short'])))
    max_peak_all_long_ol = np.concatenate((np.asarray(result_max_peak_ol['reward_peakval_long']),np.asarray(result_max_peak_ol['trialonset_peakval_long']),np.asarray(result_max_peak_ol['lmcenter_peakval_long'])))

    mean_peak_all = np.concatenate((np.asarray(result_max_peak['reward_meanpeak_short']),np.asarray(result_max_peak['trialonset_meanpeak_short']),np.asarray(result_max_peak['lmcenter_meanpeak_short']),np.asarray(result_max_peak['reward_meanpeak_long']),np.asarray(result_max_peak['trialonset_meanpeak_long']),np.asarray(result_max_peak['lmcenter_meanpeak_long'])))
    mean_peak_all_ol = np.concatenate((np.asarray(result_max_peak_ol['reward_meanpeak_short']),np.asarray(result_max_peak_ol['trialonset_meanpeak_short']),np.asarray(result_max_peak_ol['lmcenter_meanpeak_short']),np.asarray(result_max_peak_ol['reward_meanpeak_long']),np.asarray(result_max_peak_ol['trialonset_meanpeak_long']),np.asarray(result_max_peak_ol['lmcenter_meanpeak_long'])))

    space_peak_all = np.concatenate((np.asarray(result_max_peak['space_peak_short']),np.asarray(result_max_peak['space_peak_long'])))
    space_peak_all_ol = np.concatenate((np.asarray(result_max_peak_ol['space_peak_short']),np.asarray(result_max_peak_ol['space_peak_long'])))

    if DO_OL:
        print('--- VR vs OL slope: ---')
        # print(np.where(np.isnan(max_peak_all_ol))[0])
        max_peak_all = np.delete(max_peak_all, np.where(np.isnan(max_peak_all))[0])
        max_peak_all = np.delete(max_peak_all, np.where(np.isnan(max_peak_all_ol))[0])
        max_peak_all_ol = np.delete(max_peak_all_ol, np.where(np.isnan(max_peak_all))[0])
        max_peak_all_ol = np.delete(max_peak_all_ol, np.where(np.isnan(max_peak_all_ol))[0])

        mean_peak_all = np.delete(mean_peak_all,np.where(np.isnan(mean_peak_all))[0])
        mean_peak_all = np.delete(mean_peak_all,np.where(np.isnan(mean_peak_all_ol))[0])
        mean_peak_all_ol = np.delete(mean_peak_all_ol,np.where(np.isnan(mean_peak_all))[0])
        mean_peak_all_ol = np.delete(mean_peak_all_ol,np.where(np.isnan(mean_peak_all_ol))[0])

        space_peak_all = np.delete(space_peak_all,np.where(np.isnan(space_peak_all))[0])
        space_peak_all = np.delete(space_peak_all,np.where(np.isnan(space_peak_all_ol))[0])
        space_peak_all_ol = np.delete(space_peak_all_ol,np.where(np.isnan(space_peak_all))[0])
        space_peak_all_ol = np.delete(space_peak_all_ol,np.where(np.isnan(space_peak_all_ol))[0])

        # vrol_correlation_coefficient = sp.stats.pearsonr(max_peak_all_ol, max_peak_all)
        # print(vrol_correlation_coefficient)
        # ipdb.set_trace()
        # print(np.asarray(result_max_peak_ol['lmcenter_peakval_long']))
        exog_var = sm.add_constant(np.asarray(space_peak_all))
        # exog_var = np.asarray(space_peak_all)
        rlmresBI = sm.RLM(space_peak_all_ol,exog_var,M=sm.robust.norms.TukeyBiweight()).fit()

        ax_object14.plot(exog_var,rlmresBI.fittedvalues,label='TukeyBiweight')
        print(rlmresBI.summary())

        # slope, intercept, r_value, p_value, std_err = sp.stats.linregress(max_peak_all,max_peak_all_ol)
        slope, intercept, lo_slope, up_slope = sp.stats.mstats.theilslopes(max_peak_all_ol, max_peak_all)
        slope_mean, intercept_mean, lo_slope_mean, up_slope_mean = sp.stats.mstats.theilslopes(mean_peak_all_ol, mean_peak_all)
        # slope_space, intercept_space, lo_slope_space, up_slope_space = sp.stats.mstats.theilslopes(space_peak_all_ol, space_peak_all)
        slope_space, intercept_space, rval, pval, stderr = sp.stats.linregress(space_peak_all_ol, space_peak_all)

        print('peak to peak slope: ' + str(np.round(slope,4)))
        print('MEAN peak to peak slope: ' + str(np.round(slope_mean,4)))
        print('SPACE peak to peak slope: ' + str(np.round(slope_space,4)))
        print('-----------------------')
            # p = np.polyfit(max_peak_all_ol, max_peak_all,1)
        ax_object3.plot(max_peak_all,intercept+slope*max_peak_all)
    # ax_object14.plot(space_peak_all_ol,intercept+slope_space*space_peak_all_ol)

    # plot summary barcharts with statistical test
    if peak_metric is '_peak_':
        ax_object3.scatter(np.nanmedian(max_peak_all),np.nanmedian(max_peak_all_ol),s=30,color='k')
        ax_object3.errorbar(np.nanmedian(max_peak_all),np.nanmedian(max_peak_all_ol), yerr=np.nanvar(max_peak_all_ol), xerr=np.nanvar(max_peak_all), ecolor='k')
        # ax_object9.bar([0, 1], [np.nanmean(max_peak_all), np.nanmean(max_peak_all_ol)], [0.5,0.5], [0,0], edgecolor=scatter_color_long, color=[scatter_color_long,'none'], linewidth=4, align='edge')
        # ax_object9.errorbar(0.25, np.nanmean(max_peak_all), yerr=sp.stats.sem(max_peak_all,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)
        # ax_object9.errorbar(1.25, np.nanmean(max_peak_all_ol), yerr=sp.stats.sem(max_peak_all_ol,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)

        all_space_trialonset = np.concatenate((result_max_peak['space_peak_short_trialonset'],result_max_peak['space_peak_long_trialonset']))
        all_space_lmcenter = np.concatenate((result_max_peak['space_peak_short_lmcenter'],result_max_peak['space_peak_long_lmcenter']))
        all_space_reward = np.concatenate((result_max_peak['space_peak_short_reward'],result_max_peak['space_peak_long_reward']))

        all_space_trialonset_ol = np.concatenate((result_max_peak_ol['space_peak_short_trialonset'],result_max_peak_ol['space_peak_long_trialonset']))
        all_space_lmcenter_ol = np.concatenate((result_max_peak_ol['space_peak_short_lmcenter'],result_max_peak_ol['space_peak_long_lmcenter']))
        all_space_reward_ol = np.concatenate((result_max_peak_ol['space_peak_short_reward'],result_max_peak_ol['space_peak_long_reward']))

        # ax_object9.bar([0,1,2], [np.nanmean(all_space_trialonset),np.nanmean(all_space_lmcenter),np.nanmean(all_space_reward)],
        #                         [0.5,0.5,0.5], edgecolor=['#39B54A','#FF0000','#29ABE2'], color=['#39B54A','#FF0000','#29ABE2'], linewidth=4, align='edge')
        #
        # ax_object9.bar([4,5,6], [np.nanmean(all_space_trialonset_ol),np.nanmean(all_space_lmcenter_ol),np.nanmean(all_space_reward_ol)],
        #                          [0.5,0.5,0.5], edgecolor=['#39B54A','#FF0000','#29ABE2'], color=['w','w','w'], linewidth=4, align='edge')

        # ipdb.set_trace()
        parts = ax_object9.boxplot([all_space_trialonset,all_space_trialonset_ol[~np.isnan(all_space_trialonset_ol)]],patch_artist=True,showfliers=False,
            whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
            medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
            widths=(0.8,0.8),positions=(0,4))

        colors = [['#39B54A','#39B54A'],['w','#39B54A']]
        for patch, color in zip(parts['boxes'], colors):
            patch.set_facecolor(color[0])
            patch.set_edgecolor(color[1])
            patch.set_linewidth(2)

        parts = ax_object9.boxplot([all_space_lmcenter,all_space_lmcenter_ol[~np.isnan(all_space_lmcenter_ol)]],patch_artist=True,showfliers=False,
            whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
            medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
            widths=(0.8,0.8),positions=(1,5))

        colors = [['#FF0000','#FF0000'],['w','#FF0000']]
        for patch, color in zip(parts['boxes'], colors):
            patch.set_facecolor(color[0])
            patch.set_edgecolor(color[1])
            patch.set_linewidth(2)

        parts = ax_object9.boxplot([all_space_reward,all_space_reward_ol[~np.isnan(all_space_reward_ol)]],patch_artist=True,showfliers=False,
            whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
            medianprops=dict(color='k', linewidth=2, solid_capstyle='butt'),
            widths=(0.8,0.8),positions=(2,6))

        colors = [['#29ABE2','#29ABE2'],['w','#29ABE2']]
        for patch, color in zip(parts['boxes'], colors):
            patch.set_facecolor(color[0])
            patch.set_edgecolor(color[1])
            patch.set_linewidth(2)

        print('--- KRUSKAL WALLIS ---')
        print(sp.stats.kruskal(all_space_trialonset,all_space_lmcenter,all_space_reward,all_space_trialonset_ol,all_space_lmcenter_ol,all_space_reward_ol, nan_policy='omit'))
        _, p_to_tool = sp.stats.wilcoxon(all_space_trialonset,all_space_trialonset_ol)
        _, p_lm_lmol = sp.stats.wilcoxon(all_space_lmcenter,all_space_lmcenter_ol)
        _, p_rw_rwol = sp.stats.wilcoxon(all_space_reward,all_space_reward_ol)

        p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_to_tool,p_lm_lmol,p_rw_rwol],alpha=0.05,method='bonferroni')

        print('CORRECTED P-VALUES:')
        print('to medians: ', str(np.nanmedian(all_space_trialonset)), ' ', str(np.nanmedian(all_space_trialonset_ol)))
        print('lm medians: ', str(np.nanmedian(all_space_lmcenter)), ' ', str(np.nanmedian(all_space_lmcenter_ol)))
        print('rw medians: ', str(np.nanmedian(all_space_reward)), ' ', str(np.nanmedian(all_space_reward_ol)))
        print('to vs to OL: ' + str(p_corrected[1][0]))
        print('lm vs lm OL: ' + str(p_corrected[1][1]))
        print('rw vs rw OL: ' + str(p_corrected[1][2]))
        print('--------------------')


        print('median peak ttest (dF/F):')
        print(sp.stats.ttest_rel(max_peak_all,max_peak_all_ol,nan_policy='omit'))
        ax_object9.set_ylabel('dF/F', fontsize=24)
        # if normalize:
        #     if plot_layer == 'v1':
        #         ax_object9.set_ylim([0,0.35])
        #     else:
        #         ax_object9.set_ylim([0,0.2])
        # else:
        #     ax_object9.set_ylim([0,1])
    elif peak_metric is '_peak_zscore_':
        ax_object3.scatter(np.nanmean(max_peak_all),np.nanmean(max_peak_all_ol),s=30,color='k')
        ax_object3.errorbar(np.nanmean(max_peak_all),np.nanmean(max_peak_all_ol), yerr=np.nanstd(max_peak_all_ol), xerr=np.nanstd(max_peak_all), ecolor='k')
        ax_object9.bar([0, 1], [np.nanmean(max_peak_all), np.nanmean(max_peak_all_ol)], [0.5,0.5], [0,0], edgecolor=scatter_color_long, color=[scatter_color_long,'none'], linewidth=4, align='edge')
        ax_object9.errorbar(0.25, np.nanmean(max_peak_all), yerr=sp.stats.sem(max_peak_all,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)
        ax_object9.errorbar(1.25, np.nanmean(max_peak_all_ol), yerr=sp.stats.sem(max_peak_all_ol,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)
        print('mean peak ttest (zscore):')
        print(sp.stats.ttest_rel(max_peak_all,max_peak_all_ol,nan_policy='omit'))
        ax_object9.set_ylabel('z-score', fontsize=24)
        ax_object9.set_ylim([0,11])
        # ax_object9.scatter(np.full((max_peak_all_short.shape[0]),0.25), max_peak_all_short, c='none', s=40, linewidths=2, edgecolor=scatter_color_short)
        # ax_object9.scatter(np.full((max_peak_all_long.shape[0]), 0.25), max_peak_all_long, c='none', s=40, linewidths=2, edgecolor=scatter_color_long)
        # ax_object9.scatter(np.full((max_peak_all_short_ol.shape[0]),1.25), max_peak_all_short_ol, c='none', s=40, linewidths=2, edgecolor=scatter_color_short)
        # ax_object9.scatter(np.full((max_peak_all_long_ol.shape[0]), 1.25), max_peak_all_long_ol, c='none', s=40, linewidths=2, edgecolor=scatter_color_long)

    ax_object8.scatter(np.nanmedian(mean_peak_all), np.nanmedian(mean_peak_all_ol),s=30,color='k')
    ax_object8.errorbar(np.nanmedian(mean_peak_all), np.nanmedian(mean_peak_all_ol),xerr=np.nanvar(mean_peak_all), yerr=np.nanvar(mean_peak_all_ol), ecolor='k')

    ax_object14.scatter(np.nanmedian(space_peak_all), np.nanmedian(space_peak_all_ol),s=30,color='k')
    ax_object14.errorbar(np.nanmedian(space_peak_all), np.nanmedian(space_peak_all_ol),xerr=np.nanvar(space_peak_all), yerr=np.nanvar(space_peak_all_ol), ecolor='k')

    # print peak response in SPACE in VR vs OL
    normalized_mean_ol = []
    for j,mpa in enumerate(space_peak_all):
        # ax_object10.plot([0.25,1.25],[mean_peak_all[j], mean_peak_all_ol[j]], c='0.8')
        # ax_object10.plot([0.25,1.25],[1, mean_peak_all_ol[j]/mean_peak_all[j]], c='0.8')
        normalized_mean_ol.append(space_peak_all_ol[j]/space_peak_all[j])
    #
    ax_object10.bar([0, 1], [np.nanmean(space_peak_all), np.nanmean(space_peak_all_ol)], [0.5,0.5], [0,0], edgecolor='0.6', color=['0.6','none'], linewidth=4, align='edge')
    ax_object10.errorbar([0.25,1.25], [np.nanmean(space_peak_all), np.nanmean(np.array(space_peak_all_ol))], yerr=[sp.stats.sem(np.array(space_peak_all),nan_policy='omit'),sp.stats.sem(np.array(space_peak_all_ol),nan_policy='omit')], ecolor='k', linewidth=0, elinewidth=2, capsize=5, capthick=2)
    # ax_object10.errorbar(0.25, np.nanmean(space_peak_all), yerr=sp.stats.sem(space_peak_all,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)
    # ax_object10.errorbar(1.25, np.nanmean(space_peak_all_ol), yerr=sp.stats.sem(space_peak_all_ol,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)
    ax_object10.set_ylabel('dF/F peak (space)', fontsize=24)

    print('--- peak ttest (space): ---')
    # print(sp.stats.mannwhitneyu(space_peak_all,space_peak_all_ol))
    print('mean VR: ', str(np.round(np.mean(space_peak_all),4)), ' SEM: ', str(np.round(sp.stats.sem(space_peak_all),4)))
    print('mean OL: ', str(np.round(np.mean(space_peak_all_ol),4)), ' SEM: ', str(np.round(sp.stats.sem(space_peak_all_ol),4)))
    print(sp.stats.ttest_rel(space_peak_all,space_peak_all_ol,nan_policy='omit'))
    print('---------------------------')

    if max_y > 3:
        max_y=3

    # just some special plotting rules for the v1 recordings
    if plot_layer is 'v1':
        if normalize:
            max_y = 1
        else:
            max_y = 10
        if peak_metric is '_peak_':
            ax_object3.set_xlim([-0.05,max_y])
            ax_object3.set_ylim([-0.05,max_y])
            ax_object3.plot([0,max_y],[0,max_y],lw=2,c='k',ls='--')
            ax_object3.set_xlabel('peak response VR (dF/F)', fontsize=24)
            ax_object3.set_ylabel('peak response PASSIVE (dF/F)', fontsize=24)
        elif peak_metric is '_peak_zscore_':
            ax_object3.set_xlim([-1,50])
            ax_object3.set_ylim([-1,50])
            ax_object3.plot([0,50],[0,50],lw=2,c='k',ls='--')
            ax_object3.set_xlabel('peak response VR (z-score)', fontsize=24)
            ax_object3.set_ylabel('peak response PASSIVE (z-score)', fontsize=24)

        ax_object8.set_xlim([-0.05,1])
        ax_object8.set_ylim([-0.05,1])
        ax_object8.plot([0,1],[0,1],lw=2,c='k',ls='--')
        ax_object8.set_xlabel('dF/F mean trace peak VR', fontsize=22)
        ax_object8.set_ylabel('dF/F mean trace peak PASSIVE', fontsize=22)

        ax_object14.set_xlim([-0.05,1])
        ax_object14.set_ylim([-0.05,1])
        ax_object14.plot([0,1],[0,1],lw=2,c='k',ls='--')
        ax_object14.set_xlabel('dF/F peak space', fontsize=22)
        ax_object14.set_ylabel('dF/F peak space PASSIVE', fontsize=22)

        VR_passiv_barcharts_ymax = 2

    else:
        if normalize:
            max_y = 1
        else:
            max_y = 5
        if peak_metric is '_peak_':
            ax_object3.set_xlim([-0.05,max_y])
            ax_object3.set_ylim([-0.05,max_y])
            ax_object3.plot([0,max_y],[0,max_y],lw=2,c='k',ls='--')
            ax_object3.set_xlabel('peak response VR (dF/F)', fontsize=24)
            ax_object3.set_ylabel('peak response PASSIVE (dF/F)', fontsize=24)
        elif peak_metric is '_peak_zscore_':
            ax_object3.set_xlim([-1,50])
            ax_object3.set_ylim([-1,50])
            ax_object3.plot([0,50],[0,50],lw=2,c='k',ls='--')
            ax_object3.set_xlabel('peak response VR (z-score)', fontsize=24)
            ax_object3.set_ylabel('peak response PASSIVE (z-score)', fontsize=24)

        ax_object8.set_xlim([-0.05,0.7])
        ax_object8.set_ylim([-0.05,0.7])
        ax_object8.plot([0,1],[0,1],lw=2,c='k',ls='--')
        ax_object8.set_xlabel('dF/F space peak VR', fontsize=22)
        ax_object8.set_ylabel('dF/F space peak PASSIVE', fontsize=22)

        ax_object14.set_xlim([-0.05,0.7])
        ax_object14.set_ylim([-0.05,0.7])
        ax_object14.plot([0,1],[0,1],lw=2,c='k',ls='--')
        ax_object14.set_xlabel('dF/F space peak VR', fontsize=22)
        ax_object14.set_ylabel('dF/F space peak PASSIVE', fontsize=22)
        VR_passiv_barcharts_ymax = 1

    ax_object9.set_xlim([-1,7])
    ax_object9.set_xticks([1,5])
    ax_object9.set_xticklabels(['VR','PASSIVE'], rotation=45, fontsize=20)
    ax_object9.spines['bottom'].set_linewidth(2)
    ax_object9.spines['top'].set_visible(False)
    ax_object9.spines['right'].set_visible(False)
    ax_object9.spines['left'].set_linewidth(2)
    ax_object9.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=18, \
        length=4, \
        width=4, \
        bottom='on', \
        left='on', \
        right='off', \
        top='off')

    ax_object10.set_xlim([-0.5,2])
    # ax_object10.set_ylim([0,1])
    ax_object10.set_xticks([0.25,1.25])
    ax_object10.set_xticklabels(['VR','PASSIVE'], rotation=45, fontsize=20)
    ax_object10.spines['bottom'].set_linewidth(2)
    ax_object10.spines['top'].set_visible(False)
    ax_object10.spines['right'].set_visible(False)
    ax_object10.spines['left'].set_linewidth(2)
    ax_object10.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=18, \
        length=4, \
        width=4, \
        bottom='on', \
        left='on', \
        right='off', \
        top='off')

    ax_object8.spines['bottom'].set_linewidth(2)
    ax_object8.spines['top'].set_visible(False)
    ax_object8.spines['right'].set_visible(False)
    ax_object8.spines['left'].set_linewidth(2)
    ax_object8.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=18, \
        length=4, \
        width=4, \
        bottom='on', \
        left='on', \
        right='off', \
        top='off')

    ax_object14.spines['bottom'].set_linewidth(2)
    ax_object14.spines['top'].set_visible(False)
    ax_object14.spines['right'].set_visible(False)
    ax_object14.spines['left'].set_linewidth(2)
    ax_object14.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=18, \
        length=4, \
        width=4, \
        bottom='on', \
        left='on', \
        right='off', \
        top='off')


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

    # slope, intercept, r_value, p_value, std_err = sp.stats.linregress(max_peak_all, max_peak_all_ol)
        # p = np.polyfit(max_peak_all_ol, max_peak_all,1)
        # ax_object3.plot(max_peak_all_ol, p[1]+p[0]*max_peak_all_ol)
    # print('slope, intercept, p-value of linear regression: '  + str([slope, intercept, p_value]))
    # print('paired t-test (VR vs OL): ' + str(sp.stats.ttest_rel(max_peak_all, max_peak_all_ol)))
        # ax_object3.plot(intercept + slope*max_peak_all, max_peak_all, 'k', label='fitted line')
    # ax_object3.plot(max_peak_all, intercept + slope*max_peak_all, 'k', label='fitted line')

    # for i in range(len(result_max_peak['trialonset_active_short'])):
    #         ax_object4.scatter([0,1],[np.asarray(result_max_peak['trialonset_active_short'])[i]/np.asarray(result_max_peak['trialonset_active_short'])[i],np.asarray(result_max_peak_ol['trialonset_active_short'])[i]/np.asarray(result_max_peak['trialonset_active_short'])[i]],s=40,color='0.8', label='short')
    #         ax_object4.plot([0,1],[np.asarray(result_max_peak['trialonset_active_short'])[i]/np.asarray(result_max_peak['trialonset_active_short'])[i],np.asarray(result_max_peak_ol['trialonset_active_short'])[i]/np.asarray(result_max_peak['trialonset_active_short'])[i]],color='0.8')
    # for i in range(len(result_max_peak['trialonset_active_long'])):
    #         ax_object4.scatter([0,1],[np.asarray(result_max_peak['trialonset_active_long'])[i]/np.asarray(result_max_peak['trialonset_active_long'])[i],np.asarray(result_max_peak_ol['trialonset_active_long'])[i]/np.asarray(result_max_peak['trialonset_active_long'])[i]],s=40,color='0.6', label='long')
    #         ax_object4.plot([0,1],[np.asarray(result_max_peak['trialonset_active_long'])[i]/np.asarray(result_max_peak['trialonset_active_long'])[i],np.asarray(result_max_peak_ol['trialonset_active_long'])[i]/np.asarray(result_max_peak['trialonset_active_long'])[i]],color='0.6')
    #
    # for i in range(len(result_max_peak['lmcenter_active_short'])):
    #         ax_object4.scatter([0,1],[np.asarray(result_max_peak['lmcenter_active_short'])[i]/np.asarray(result_max_peak['lmcenter_active_short'])[i],np.asarray(result_max_peak_ol['lmcenter_active_short'])[i]/np.asarray(result_max_peak['lmcenter_active_short'])[i]],s=40,color='0.8', label='short')
    #         ax_object4.plot([0,1],[np.asarray(result_max_peak['lmcenter_active_short'])[i]/np.asarray(result_max_peak['lmcenter_active_short'])[i],np.asarray(result_max_peak_ol['lmcenter_active_short'])[i]/np.asarray(result_max_peak['lmcenter_active_short'])[i]],color='0.8')
    # for i in range(len(result_max_peak['lmcenter_active_long'])):
    #         ax_object4.scatter([0,1],[np.asarray(result_max_peak['lmcenter_active_long'])[i]/np.asarray(result_max_peak['lmcenter_active_long'])[i],np.asarray(result_max_peak_ol['lmcenter_active_long'])[i]/np.asarray(result_max_peak['lmcenter_active_long'])[i]],s=40,color='0.6', label='long')
    #         ax_object4.plot([0,1],[np.asarray(result_max_peak['lmcenter_active_long'])[i]/np.asarray(result_max_peak['lmcenter_active_long'])[i],np.asarray(result_max_peak_ol['lmcenter_active_long'])[i]/np.asarray(result_max_peak['lmcenter_active_long'])[i]],color='0.6')
    #
    # for i in range(len(result_max_peak['reward_active_short'])):
    #         ax_object4.scatter([0,1],[np.asarray(result_max_peak['reward_active_short'])[i]/np.asarray(result_max_peak['reward_active_short'])[i],np.asarray(result_max_peak_ol['reward_active_short'])[i]/np.asarray(result_max_peak['reward_active_short'])[i]],s=40,color='0.8', label='short')
    #         ax_object4.plot([0,1],[np.asarray(result_max_peak['reward_active_short'])[i]/np.asarray(result_max_peak['reward_active_short'])[i],np.asarray(result_max_peak_ol['reward_active_short'])[i]/np.asarray(result_max_peak['reward_active_short'])[i]],color='0.8')
    # for i in range(len(result_max_peak['reward_active_long'])):
    #         ax_object4.scatter([0,1],[np.asarray(result_max_peak['reward_active_long'])[i]/np.asarray(result_max_peak['reward_active_long'])[i],np.asarray(result_max_peak_ol['reward_active_long'])[i]/np.asarray(result_max_peak['reward_active_long'])[i]],s=40,color='0.6', label='long')
    #         ax_object4.plot([0,1],[np.asarray(result_max_peak['reward_active_long'])[i]/np.asarray(result_max_peak['reward_active_long'])[i],np.asarray(result_max_peak_ol['reward_active_long'])[i]/np.asarray(result_max_peak['reward_active_long'])[i]],color='0.6')

    # ax_object4.boxplot([np.asarray(result_max_peak['lmcenter_active_short']),np.asarray(result_max_peak_ol['lmcenter_active_short'])])

    # ax_object4.scatter(np.zeros((np.asarray(result_max_peak_ol['trialonset_active_short']).shape[0],)), np.asarray(result_max_peak_ol['trialonset_active_short']))
    # np.asarray(result_max_peak['trialonset_active_short']),s=10,color='0.8', label='short')
    # ax_object4.scatter(np.asarray(result_max_peak_ol['trialonset_active_short']), np.asarray(result_max_peak['trialonset_active_short']),s=10,color=scatter_color_short, label='short')
    # ax_object4.scatter(np.asarray(result_max_peak_ol['trialonset_active_long']), np.asarray(result_max_peak['trialonset_active_long']),s=10,color=scatter_color_long, label='long')
    # ax_object4.scatter(np.asarray(result_max_peak_ol['lmcenter_active_short']), np.asarray(result_max_peak['lmcenter_active_short']),s=10,color=scatter_color_short)
    # ax_object4.scatter(np.asarray(result_max_peak_ol['lmcenter_active_long']), np.asarray(result_max_peak['lmcenter_active_long']),s=10,color=scatter_color_long)
    # ax_object4.scatter(np.asarray(result_max_peak_ol['reward_active_short']), np.asarray(result_max_peak['reward_active_short']),s=10,color=scatter_color_short)
    # ax_object4.scatter(np.asarray(result_max_peak_ol['reward_active_long']), np.asarray(result_max_peak['reward_active_long']),s=10,color=scatter_color_long)
    #
    # ax_object4.set_xlim([-0.1,1.1])
    # ax_object4.set_ylim([-0.1,1.1])
    # ax_object4.set_ylabel('trials active VR', fontsize=24)
    # ax_object4.set_xlabel('trials active PASSIVE', fontsize=24)
    # ax_object4.plot([0,1],[0,1],lw=2,c='k',ls='--')

    mean_active_all = np.concatenate((np.asarray(result_max_peak['reward_active_short']),np.asarray(result_max_peak['trialonset_active_short']),np.asarray(result_max_peak['lmcenter_active_short']),np.asarray(result_max_peak['reward_active_long']),np.asarray(result_max_peak['trialonset_active_long']),np.asarray(result_max_peak['lmcenter_active_long'])))
    mean_active_all_ol = np.concatenate((np.asarray(result_max_peak_ol['reward_active_short']),np.asarray(result_max_peak_ol['trialonset_active_short']),np.asarray(result_max_peak_ol['lmcenter_active_short']),np.asarray(result_max_peak_ol['reward_active_long']),np.asarray(result_max_peak_ol['trialonset_active_long']),np.asarray(result_max_peak_ol['lmcenter_active_long'])))

    mean_active_all = mean_active_all[mean_active_all>0]
    mean_active_all_ol = mean_active_all_ol[mean_active_all>0]
    ax_object11.bar([0, 1], [np.nanmean(mean_peak_all), np.nanmean(mean_peak_all_ol)], [0.5,0.5], [0,0], edgecolor=scatter_color_long, color=[scatter_color_long,'none'], linewidth=4, align='edge')
    ax_object11.errorbar(0.25, np.nanmean(mean_peak_all), yerr=sp.stats.sem(mean_peak_all,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)
    ax_object11.errorbar(1.25, np.nanmean(mean_peak_all_ol), yerr=sp.stats.sem(mean_peak_all_ol,nan_policy='omit'), ecolor='k', elinewidth=2, capsize=5, capthick=2)
    ax_object11.set_ylabel('fraction trials active', fontsize=24)
    print('mean active ttest:')
    print(sp.stats.ttest_rel(mean_active_all,mean_active_all_ol,nan_policy='omit'))

    ax_object11.set_xlim([-0.5,2])
    ax_object11.set_xticks([0.25,1.25])
    ax_object11.set_xticklabels(['VR','PASSIVE'], rotation=45, fontsize=20)
    ax_object11.spines['bottom'].set_linewidth(2)
    ax_object11.spines['top'].set_visible(False)
    ax_object11.spines['right'].set_visible(False)
    ax_object11.spines['left'].set_linewidth(2)
    ax_object11.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=18, \
        length=4, \
        width=4, \
        bottom='on', \
        left='on', \
        right='off', \
        top='off')

    # determine maximum value and scale y-axis
    if not normalize:
        max_y = np.nanmax(np.concatenate((np.asarray(result_max_peak_ol['trialonset_peakval_short']),np.asarray(result_max_peak['trialonset_peakval_short']),
                                          np.asarray(result_max_peak_ol['trialonset_peakval_long']), np.asarray(result_max_peak['trialonset_peakval_long']),
                                          np.asarray(result_max_peak_ol['lmcenter_peakval_short']), np.asarray(result_max_peak['lmcenter_peakval_short']),
                                          np.asarray(result_max_peak_ol['lmcenter_peakval_long']), np.asarray(result_max_peak['lmcenter_peakval_long']),
                                          np.asarray(result_max_peak_ol['reward_peakval_short']), np.asarray(result_max_peak['reward_peakval_short']),
                                          np.asarray(result_max_peak_ol['reward_peakval_long']), np.asarray(result_max_peak['reward_peakval_long']))))
    else:
        max_y = 1


    # ax_object3.legend()

    # plot mean peak aligned to its optimal alignment point vs trialonset (i.e. trial start time)
    # ax_object4.scatter(np.asarray(result_trialonset_max_peak['lmcenter_peakval_short']), np.asarray(result_max_peak['lmcenter_peakval_short']),s=10,color='0.8')
    # ax_object4.scatter(np.asarray(result_trialonset_max_peak['reward_peakval_short']), np.asarray(result_max_peak['reward_peakval_short']),s=10,color='0.8')
    # ax_object4.scatter(np.asarray(result_trialonset_max_peak['lmcenter_peakval_long']), np.asarray(result_max_peak['lmcenter_peakval_long']),s=10,color='0.5')
    # ax_object4.scatter(np.asarray(result_trialonset_max_peak['reward_peakval_long']), np.asarray(result_max_peak['reward_peakval_long']),s=10,color='0.5')
    # ax_object4.scatter(np.asarray(result_trialonset_max_peak['trialonset_peakval_short']), np.asarray(result_lmcenter_max_peak['trialonset_peakval_short']),s=10,color='0.8')
    # ax_object4.scatter(np.asarray(result_trialonset_max_peak['trialonset_peakval_long']), np.asarray(result_lmcenter_max_peak['trialonset_peakval_long']),s=10,color='0.5')
    # ax_object4.plot([min_xy,max_y],[min_xy,max_y],lw=2,c='k',ls='--')
    # ax_object4.set_xlim([min_xy,max_y])
    # ax_object4.set_ylim([min_xy,max_y])
    # ax_object4.set_xlabel('peak response aligned to trial onset')
    # ax_object4.set_ylabel('peak response aligned to landmark or reward')

    # donut_colors = ['0.85','#29ABE2','#FF0000','#39B54A']
    sns.distplot(result_matching_landmark_normdiff['lm_diff_short'],bins=np.linspace(-5,0,numbins),kde=False,color=scatter_color_short,ax=ax_object5,label='short preferring',hist_kws={'alpha':1})
    sns.distplot(result_matching_landmark_normdiff['lm_diff_long'],bins=np.linspace(0,5,numbins),kde=False,color=scatter_color_long,ax=ax_object5,label='long preferring',hist_kws={'alpha':1})
    result_matching_landmark_normdiff_all = np.concatenate((result_matching_landmark_normdiff['lm_diff_short'],result_matching_landmark_normdiff['lm_diff_long']))

    # sns.distplot(np.concatenate((result_matching_landmark_normdiff['lm_diff_short'],result_matching_landmark_normdiff['lm_diff_long'])),bins=np.linspace(-5,5,numbins*2+1),kde=False,hist=True,norm_hist=False, color='#FF0000',ax=ax_object4,hist_kws={"alpha":1, "histtype":"barstacked", "linewidth":"2"})
    # sns.distplot(np.concatenate((result_matching_landmark_normdiff['rw_diff_short'],result_matching_landmark_normdiff['rw_diff_long'])),bins=np.linspace(-5,5,numbins*2+1),kde=False,hist=True,norm_hist=False ,color='#29ABE2',ax=ax_object4,hist_kws={"alpha":1, "histtype":"barstacked", "linewidth":"2"})
    # sns.distplot(np.concatenate((result_matching_landmark_normdiff['to_diff_short'],result_matching_landmark_normdiff['to_diff_long'])),bins=np.linspace(-5,5,numbins*2+1),kde=False,hist=True,norm_hist=False,color='#39B54A',ax=ax_object4,hist_kws={"alpha":1, "histtype":"barstacked", "linewidth":"2"})

    # ax_object15.hist(np.array(result_max_peak['trialonset_peakloc_short']), bins=20, range=[-20,360], histtype='step', color='#39B54A')
    ax_object4.hist([np.concatenate((result_matching_landmark_normdiff['to_diff_short'],result_matching_landmark_normdiff['to_diff_long'])),
                     np.concatenate((result_matching_landmark_normdiff['lm_diff_short'],result_matching_landmark_normdiff['lm_diff_long'])),
                     np.concatenate((result_matching_landmark_normdiff['rw_diff_short'],result_matching_landmark_normdiff['rw_diff_long']))],
                     bins=np.linspace(-5,5,numbins*2+1),range=[-5,5], rwidth=1, histtype='barstacked', color=['#39B54A','#FF0000','#29ABE2'])
    # sns.distplot(result_matching_landmark_normdiff['lm_diff_long'],bins=np.linspace(0,5,numbins),kde=True,color='#FF0000',ax=ax_object4,label='long preferring',hist_kws={"alpha":1})
    # sns.distplot(result_matching_landmark_normdiff['to_diff_long'],bins=np.linspace(0,5,numbins),kde=True,color='#39B54A',ax=ax_object4,label='long preferring',hist_kws={"alpha":1})
    # sns.distplot(result_matching_landmark_normdiff['rw_diff_long'],bins=np.linspace(0,5,numbins),kde=True,color='#29ABE2',ax=ax_object4,label='long preferring',hist_kws={"alpha":1})
    ax_object4.set_xlim([-1,1])
    # sns.distplot(result_matching_landmark_normdiff_all,bins=np.linspace(-5,5,numbins*2),kde=False,color=scatter_color_short,ax=ax_object5,label='short preferring',hist_kws={"alpha":1})
    print('--- LM DIFF K-S TEST ---')
    print(sp.stats.kstest(result_matching_landmark_normdiff_all,'norm'))
    print('------------------------')
    # ax_object5.legend()
    ax_object5.spines['bottom'].set_linewidth(2)
    ax_object5.spines['top'].set_visible(False)
    ax_object5.spines['right'].set_visible(False)
    ax_object5.spines['left'].set_linewidth(2)
    ax_object5.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=18, \
        length=4, \
        width=4, \
        bottom='on', \
        left='on', \
        right='off', \
        top='off')
    ax_object5.set_xlim([-1,1])
    ax_object5.set_xlabel('landmark selectivity index', fontsize=24)
    ax_object5.set_ylabel('number of neurons', fontsize=24)



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
    if plot_layer is 'l23' or plot_layer is 'l5':
        return space_peak_all,space_peak_all_ol,np.array(num_lmcenter_short),np.array(num_lmcenter_long)
    else:
        return space_peak_all,space_peak_all_ol, [result_matching_landmark_normdiff['lm_diff_short'],result_matching_landmark_normdiff['lm_diff_long']], [landmark_transient_distance_short,landmark_transient_distance_long]

def roi_amplitude_scatter(roi_param_list_all, event_list, trialtypes, peak_metric, ax_object, ax_object2, ax_object3, ax_object4, normalize=False):
    """
    plot the amplitude of ROIs against all three alignment points

    RETURN : ax_object

    """

    # separate out the path to the json file from the rest of the list which also contains animal and session names
    roi_param_list = []
    for rpl_all in roi_param_list_all:
        roi_param_list.append(rpl_all[0])

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
            result_max_peak[el + peak_metric + tl] = []

    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        # load roi parameters for given session
        with open(rpl,'r') as f:
            roi_params = json.load(f)

        # grab a full list of roi numbers (it doesn't matter which event is chosen here)
        roi_list_all = roi_params['valid_rois']
        # loop through every roi
        for j,r in enumerate(roi_list_all):
            if normalize:
                roi_normalization_value = roi_params['norm_value'][j]
            else:
                roi_normalization_value = 1

            # loop through every trialtype and alignment point to determine largest response
            for tl in trialtypes:
                valid = False
                max_peak_trialonset = 0
                max_peak_lmcenter = 0
                max_peak_reward = 0
                # loop through every event and store peak value
                if roi_response_validation(roi_params, tl, 'trialonset', j):
                    max_peak_trialonset = roi_params['trialonset' + peak_metric + tl][j]/roi_normalization_value
                if roi_response_validation(roi_params, tl, 'lmcenter', j):
                    max_peak_lmcenter = roi_params['lmcenter' + peak_metric + tl][j]/roi_normalization_value
                if roi_response_validation(roi_params, tl, 'reward', j):
                    max_peak_reward = roi_params['reward' + peak_metric + tl][j]/roi_normalization_value

                result_max_peak['trialonset' + peak_metric + tl].append(max_peak_trialonset)
                result_max_peak['lmcenter' + peak_metric + tl].append(max_peak_lmcenter)
                result_max_peak['reward' + peak_metric + tl].append(max_peak_reward)

    # print(result_max_peak['lmcenter_peak_short'])
    # print(result_max_peak['trialonset_peak_short'])
    # slope, intercept, r_value, p_value, std_err = sp.stats.linregress(np.asarray(result_max_peak['lmcenter_peak_short']), np.asarray(result_max_peak['trialonset_peak_short']))
        # p = np.polyfit(max_peak_all_ol, max_peak_all,1)
    # print('slope, intercept, p-value of linear regression: '  + str([slope, intercept, p_value]))
    # print('paired t-test (VR vs OL): ' + str(sp.stats.ttest_rel(max_peak_all, max_peak_all_ol)))
        # ax_object3.plot(intercept + slope*max_peak_all, max_peak_all, 'k', label='fitted line')
    # ax_object.plot(np.asarray(result_max_peak['trialonset_peak_short']), intercept + slope*np.asarray(result_max_peak['lmcenter_peak_short']), 'k', label='fitted line')

    # ax_object.scatter(result_max_peak['trialonset_peak_short'],result_max_peak['lmcenter_peak_short'],c='k',linewidths=1,s=40)
    # ax_object2.scatter(result_max_peak['reward_peak_short'],result_max_peak['lmcenter_peak_short'],c='k',linewidths=1,s=40)
    # ax_object3.scatter(result_max_peak['trialonset_peak_long'],result_max_peak['lmcenter_peak_long'],c='k',linewidths=1,s=40)
    # ax_object4.scatter(result_max_peak['reward_peak_long'],result_max_peak['lmcenter_peak_long'],c='k',linewidths=1,s=40)

    # separate out neurons by what alignment point they better align to and plot them in different colors
    trialonset_neurons_idx = np.array(result_max_peak['trialonset' + peak_metric + 'short']) > np.array(result_max_peak['lmcenter' + peak_metric + 'short'])
    landmark_neurons_idx = np.array(result_max_peak['trialonset' + peak_metric + 'short']) < np.array(result_max_peak['lmcenter' + peak_metric + 'short'])
    ax_object.scatter(np.array(result_max_peak['trialonset' + peak_metric + 'short'])[trialonset_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'short'])[trialonset_neurons_idx],c='g',linewidths=0,s=40)
    ax_object.scatter(np.array(result_max_peak['trialonset' + peak_metric + 'short'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'short'])[landmark_neurons_idx],c='m',linewidths=0,s=40)

    reward_neurons_idx = np.array(result_max_peak['reward' + peak_metric + 'short']) > np.array(result_max_peak['lmcenter' + peak_metric + 'short'])
    landmark_neurons_idx = np.array(result_max_peak['reward' + peak_metric + 'short']) < np.array(result_max_peak['lmcenter' + peak_metric + 'short'])
    ax_object2.scatter(np.array(result_max_peak['reward' + peak_metric + 'short'])[reward_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'short'])[reward_neurons_idx],c='g',linewidths=0,s=40)
    ax_object2.scatter(np.array(result_max_peak['reward' + peak_metric + 'short'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'short'])[landmark_neurons_idx],c='m',linewidths=0,s=40)

    trialonset_neurons_idx = np.array(result_max_peak['trialonset' + peak_metric + 'long']) > np.array(result_max_peak['lmcenter' + peak_metric + 'long'])
    landmark_neurons_idx = np.array(result_max_peak['trialonset' + peak_metric + 'long']) < np.array(result_max_peak['lmcenter' + peak_metric + 'long'])
    ax_object3.scatter(np.array(result_max_peak['trialonset' + peak_metric + 'long'])[trialonset_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'long'])[trialonset_neurons_idx],c='g',linewidths=0,s=40)
    ax_object3.scatter(np.array(result_max_peak['trialonset' + peak_metric + 'long'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'long'])[landmark_neurons_idx],c='m',linewidths=0,s=40)

    reward_neurons_idx = np.array(result_max_peak['reward' + peak_metric + 'long']) > np.array(result_max_peak['lmcenter' + peak_metric + 'long'])
    landmark_neurons_idx = np.array(result_max_peak['reward' + peak_metric + 'long']) < np.array(result_max_peak['lmcenter' + peak_metric + 'long'])
    ax_object4.scatter(np.array(result_max_peak['reward' + peak_metric + 'long'])[reward_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'long'])[reward_neurons_idx],c='g',linewidths=0,s=40)
    ax_object4.scatter(np.array(result_max_peak['reward' + peak_metric + 'long'])[landmark_neurons_idx],np.array(result_max_peak['lmcenter' + peak_metric + 'long'])[landmark_neurons_idx],c='m',linewidths=0,s=40)


    # print(ax_object.get_xticklabels())
    ax_object.set_xlabel('peak response TRIALONSET SHORT', fontsize=18)
    ax_object.set_ylabel('peak response LMCENTER SHORT', fontsize=18)
    ax_object2.set_xlabel('peak response REWARD SHORT', fontsize=18)
    ax_object2.set_ylabel('peak response LMCENTER SHORT', fontsize=18)
    ax_object3.set_xlabel('peak response TRIALONSET LONG', fontsize=18)
    ax_object3.set_ylabel('peak response LMCENTER LONG', fontsize=18)
    ax_object4.set_xlabel('peak response REWARD LONG', fontsize=18)
    ax_object4.set_ylabel('peak response LMCENTER LONG', fontsize=18)

    if normalize:
        min_y = -0.05
        max_y = 1
    else:
        if peak_metric is '_peak_zscore_':
            min_y = -2
            max_y = 50
        elif peak_metric is '_peak_':
            min_y = -0.2
            max_y = 5
    ax_object.set_xlim([min_y,max_y])
    ax_object.set_ylim([min_y,max_y])
    ax_object2.set_xlim([min_y,max_y])
    ax_object2.set_ylim([min_y,max_y])
    ax_object3.set_xlim([min_y,max_y])
    ax_object3.set_ylim([min_y,max_y])
    ax_object4.set_xlim([min_y,max_y])
    ax_object4.set_ylim([min_y,max_y])
    # ax_object.plot([0,1],[0,1], ls='--', c='k')
    # ax_object2.plot([0,1],[0,1], ls='--', c='k')
    # ax_object3.plot([0,1],[0,1], ls='--', c='k')
    # ax_object4.plot([0,1],[0,1], ls='--', c='k')

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

    ax_object3.spines['bottom'].set_linewidth(2)
    ax_object3.spines['top'].set_visible(False)
    ax_object3.spines['right'].set_visible(False)
    ax_object3.spines['left'].set_linewidth(2)
    ax_object3.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=20, \
        length=4, \
        width=4, \
        left='on', \
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

def vr_ol_difference(max_peak_all,max_peak_all_ol,max_peak_v1,max_peak_v1_ol, subfolder, fname):

    norm_method = 2

    fig = plt.figure(figsize=(3,4))
    ax1 = plt.subplot(111)

    # ax1.scatter(np.full_like(max_peak_all,1), max_peak_all-max_peak_all_ol, color='k',zorder=3)
    # ax1.scatter(np.full_like(max_peak_v1,2), max_peak_v1-max_peak_v1_ol, color='k',zorder=3)

    # sns.stripplot(np.full_like(max_peak_all,0), max_peak_all-max_peak_all_ol,ax=ax1)
    # sns.swarmplot(np.full_like(max_peak_v1,5), max_peak_v1-max_peak_v1_ol,ax=ax1)
    # sns.stripplot(x=np.full_like(max_peak_all,1),y=[max_peak_all-max_peak_all_ol],ax=ax1)

    # print peak response in SPACE in VR vs OL
    normalized_mean_ol_rsc = []
    for j,mpa in enumerate(max_peak_all):
        # set minimum to 0.05 for normalizing to avoid extreme values from near zero values
        if max_peak_all[j] < 0.001:
            max_peak_all[j] = 0.001
        if max_peak_all_ol[j] < 0.001:
            max_peak_all_ol[j] = 0.001

        if norm_method == 1:
            normalized_mean_ol_rsc.append((max_peak_all_ol[j] - max_peak_all[j])/(max_peak_all_ol[j] + max_peak_all[j]))
        elif norm_method == 2:
            normalized_mean_ol_rsc.append(1 - ((max_peak_all[j] - max_peak_all_ol[j])/max_peak_all[j]))
        # normalized_mean_ol_rsc.append((max_peak_all_ol[j] - max_peak_all[j])/max_peak_all[j])

    normalized_mean_ol_rsc = np.array(normalized_mean_ol_rsc)

    normalized_mean_ol_v1 = []
    for j,mpa in enumerate(max_peak_v1):
        # set minimum to 0.05 for normalizing to avoid extreme values from near zero values
        if max_peak_v1[j] < 0.001:
            max_peak_v1[j] = 0.001
        if max_peak_v1_ol[j] < 0.001:
            max_peak_v1_ol[j] = 0.001

        if norm_method == 1:
            normalized_mean_ol_v1.append((max_peak_v1_ol[j] - max_peak_v1[j])/(max_peak_v1_ol[j] + max_peak_v1[j]))
        elif norm_method == 2:
            normalized_mean_ol_v1.append(1 - ((max_peak_v1[j] - max_peak_v1_ol[j])/max_peak_v1[j]))
        # normalized_mean_ol_v1.append((max_peak_v1_ol[j] - max_peak_v1[j])/max_peak_v1[j])
    normalized_mean_ol_v1 = np.array(normalized_mean_ol_v1)

    # parts = plt.violinplot([normalized_mean_ol_rsc*-1,normalized_mean_ol_v1*-1],showmeans=True, showextrema=False)
    # ipdb.set_trace()
    parts = plt.boxplot([normalized_mean_ol_rsc[~np.isnan(normalized_mean_ol_rsc)],normalized_mean_ol_v1],patch_artist=True,
        whiskerprops=dict(linestyle='-', color='black', linewidth=2, solid_capstyle='butt'),
        medianprops=dict(color='black', linewidth=2, solid_capstyle='butt'),
        widths=(0.1,0.1),positions=(1,1.3))
    # ax1.scatter()

    # bp = ax_object17.boxplot([np.array(result_max_peak['reward_peakloc_short'])+320,np.array(result_max_peak['lmcenter_peakloc_short'])+220,np.array(result_max_peak['trialonset_peakloc_short'])],
    #                          vert=False, patch_artist=True, bootstrap=1000, showcaps=False, whiskerprops=dict(linestyle='-', color='black', linewidth=5, solid_capstyle='butt'),
    #                          medianprops=dict(color='black', linewidth=5, solid_capstyle='butt'))

    colors = ['#CCCCCC', '#0970B8']
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_linewidth(1)


    # parts['bodies'][0].set_facecolor('#CCCCCC')
    # parts['bodies'][1].set_facecolor('#0970B8')
    # parts['bodies'][0].set_alpha(1)
    # parts['bodies'][1].set_alpha(1)
    # parts['bodies'][0].set_edgecolor('black')
    # parts['bodies'][1].set_edgecolor('black')

    print('--- RSC, V1 differences ---')
    print('mean normalized OL attenuation RSC: ', str(np.round(np.nanmean(normalized_mean_ol_rsc),4)), ' ', str(np.round(sp.stats.sem(normalized_mean_ol_rsc[~np.isnan(normalized_mean_ol_rsc)]),4)) )
    print('mean normalized OL attenuation V1: ', str(np.round(np.nanmean(normalized_mean_ol_v1),4)), ' ', str(np.round(sp.stats.sem(normalized_mean_ol_v1[~np.isnan(normalized_mean_ol_v1)]),4)))
    # print(sp.stats.ttest_ind(max_peak_all-max_peak_all_ol,  max_peak_v1-max_peak_v1_ol))
    # print(sp.stats.mannwhitneyu(max_peak_all-max_peak_all_ol,  max_peak_v1-max_peak_v1_ol))
    # print(sp.stats.ttest_ind(normalized_mean_ol_rsc[~np.isnan(normalized_mean_ol_rsc)],  normalized_mean_ol_v1[~np.isnan(normalized_mean_ol_v1)]))
    print(sp.stats.mannwhitneyu(normalized_mean_ol_rsc[~np.isnan(normalized_mean_ol_rsc)],  normalized_mean_ol_v1[~np.isnan(normalized_mean_ol_v1)]))
    # print(sp.stats.wilcoxon(normalized_mean_ol_rsc,  normalized_mean_ol_v1))
    print('---------------------------')

    ax1.set_ylim([-0.1,1.6])
    ax1.set_xlim([0.8,1.5])
    ax1.set_xticks([1,1.3])
    ax1.set_xticklabels(['RSC','V1'])
    ax1.set_yticks([0,0.5,1.0,1.5])
    ax1.set_ylabel('Response amplitude decoupled',fontsize=14)

    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

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
    print(fname)

def vr_ol_difference_3(max_peak_all,max_peak_all_ol,max_peak_naive,max_peak_naive_ol,max_peak_v1,max_peak_v1_ol, subfolder, fname):

    norm_method = 2

    fig = plt.figure(figsize=(4,4))
    ax1 = plt.subplot(111)


    # print peak response in SPACE in VR vs OL
    normalized_mean_ol_rsc = []
    for j,mpa in enumerate(max_peak_all):
        # set minimum to 0.05 for normalizing to avoid extreme values from near zero values
        if max_peak_all[j] < 0.001:
            max_peak_all[j] = 0.001
        if max_peak_all_ol[j] < 0.001:
            max_peak_all_ol[j] = 0.001

        if norm_method == 1:
            normalized_mean_ol_rsc.append((max_peak_all_ol[j] - max_peak_all[j])/(max_peak_all_ol[j] + max_peak_all[j]))
        elif norm_method == 2:
            normalized_mean_ol_rsc.append(1 - ((max_peak_all[j] - max_peak_all_ol[j])/max_peak_all[j]))
        # normalized_mean_ol_rsc.append((max_peak_all_ol[j] - max_peak_all[j])/max_peak_all[j])

    normalized_mean_ol_rsc = np.array(normalized_mean_ol_rsc)

    normalized_mean_ol_naive = []
    for j,mpa in enumerate(max_peak_naive):
        # set minimum to 0.05 for normalizing to avoid extreme values from near zero values
        if max_peak_naive[j] < 0.001:
            max_peak_naive[j] = 0.001
        if max_peak_naive_ol[j] < 0.001:
            max_peak_naive_ol[j] = 0.001

        if norm_method == 1:
            normalized_mean_ol_naive.append((max_peak_naive_ol[j] - max_peak_naive[j])/(max_peak_naive_ol[j] + max_peak_naive[j]))
        elif norm_method == 2:
            normalized_mean_ol_naive.append(1 - ((max_peak_naive[j] - max_peak_naive_ol[j])/max_peak_naive[j]))
        # normalized_mean_ol_naive.append((max_peak_all_ol[j] - max_peak_all[j])/max_peak_all[j])

    normalized_mean_ol_naive = np.array(normalized_mean_ol_naive)


    normalized_mean_ol_v1 = []
    for j,mpa in enumerate(max_peak_v1):
        # set minimum to 0.05 for normalizing to avoid extreme values from near zero values
        if max_peak_v1[j] < 0.001:
            max_peak_v1[j] = 0.001
        if max_peak_v1_ol[j] < 0.001:
            max_peak_v1_ol[j] = 0.001

        if norm_method == 1:
            normalized_mean_ol_v1.append((max_peak_v1_ol[j] - max_peak_v1[j])/(max_peak_v1_ol[j] + max_peak_v1[j]))
        elif norm_method == 2:
            normalized_mean_ol_v1.append(1 - ((max_peak_v1[j] - max_peak_v1_ol[j])/max_peak_v1[j]))
        # normalized_mean_ol_v1.append((max_peak_v1_ol[j] - max_peak_v1[j])/max_peak_v1[j])
    normalized_mean_ol_v1 = np.array(normalized_mean_ol_v1)

    # parts = plt.violinplot([normalized_mean_ol_rsc*-1,normalized_mean_ol_v1*-1],showmeans=True, showextrema=False)
    parts = plt.boxplot([normalized_mean_ol_rsc,normalized_mean_ol_naive,normalized_mean_ol_v1],patch_artist=True,
        whiskerprops=dict(linestyle='-', color='black', linewidth=2, solid_capstyle='butt'),
        medianprops=dict(color='black', linewidth=2, solid_capstyle='butt'),
        widths=(0.1,0.1,0.1),positions=(1,1.3,1.6))
    # ax1.scatter()

    # bp = ax_object17.boxplot([np.array(result_max_peak['reward_peakloc_short'])+320,np.array(result_max_peak['lmcenter_peakloc_short'])+220,np.array(result_max_peak['trialonset_peakloc_short'])],
    #                          vert=False, patch_artist=True, bootstrap=1000, showcaps=False, whiskerprops=dict(linestyle='-', color='black', linewidth=5, solid_capstyle='butt'),
    #                          medianprops=dict(color='black', linewidth=5, solid_capstyle='butt'))

    colors = ['#CCCCCC', '#CCCCCC', '#0970B8']
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)


    # parts['bodies'][0].set_facecolor('#CCCCCC')
    # parts['bodies'][1].set_facecolor('#0970B8')
    # parts['bodies'][0].set_alpha(1)
    # parts['bodies'][1].set_alpha(1)
    # parts['bodies'][0].set_edgecolor('black')
    # parts['bodies'][1].set_edgecolor('black')

    print('--- EXPERT, NAIVE, V1 differences ---')
    print('mean normalized OL attenuation EXPERT: ', str(np.round(np.nanmean(normalized_mean_ol_rsc),4)), ' ', str(np.round(sp.stats.sem(normalized_mean_ol_rsc),4)) )
    print('mean normalized OL attenuation NAIVE: ', str(np.round(np.nanmean(normalized_mean_ol_naive),4)), ' ', str(np.round(sp.stats.sem(normalized_mean_ol_naive),4)))
    print('mean normalized OL attenuation V1: ', str(np.round(np.nanmean(normalized_mean_ol_v1),4)), ' ', str(np.round(sp.stats.sem(normalized_mean_ol_v1),4)))
    # print(sp.stats.ttest_ind(max_peak_all-max_peak_all_ol,  max_peak_v1-max_peak_v1_ol))
    # print(sp.stats.mannwhitneyu(max_peak_all-max_peak_all_ol,  max_peak_v1-max_peak_v1_ol))
    print(sp.stats.ttest_ind(normalized_mean_ol_rsc,  normalized_mean_ol_v1))
    print(sp.stats.mannwhitneyu(normalized_mean_ol_rsc,  normalized_mean_ol_v1))
    # print(sp.stats.wilcoxon(normalized_mean_ol_rsc,  normalized_mean_ol_v1))
    print('---------------------------')

    print('--- EXPERT VS. NAIVE VS. V1 ANOVA FOR OL ATTENUATION ---')
    print(sp.stats.f_oneway(normalized_mean_ol_rsc,normalized_mean_ol_naive,normalized_mean_ol_v1))
    group_labels = ['normalized_mean_ol_rsc'] * np.array(normalized_mean_ol_rsc).shape[0] + \
                   ['normalized_mean_ol_naive'] * np.array(normalized_mean_ol_naive).shape[0] + \
                   ['normalized_mean_ol_v1'] * np.array(normalized_mean_ol_v1).shape[0]


    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((normalized_mean_ol_rsc,normalized_mean_ol_naive,normalized_mean_ol_v1)),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)

    ax1.set_ylim([-0.1,1.6])
    ax1.set_xlim([0.8,1.8])
    ax1.set_xticks([1,1.3,1.6])
    ax1.set_xticklabels(['EXPERT','V1','NAIVE'])
    ax1.set_yticks([0,0.5,1.0,1.5])
    ax1.set_ylabel('Response amplitude decoupled',fontsize=14)

    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

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
    print(fname)

def naive_vs_expert_lmi(lmi_hist_data_naive,lmi_hist_data_expert, lm_transient_distance_naive, lm_transient_distance_all, ax_object, subfolder, fname):
    ax_object.spines['bottom'].set_linewidth(2)
    ax_object.spines['top'].set_visible(False)
    ax_object.spines['right'].set_visible(False)
    ax_object.spines['left'].set_linewidth(2)
    ax_object.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=14, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    numbins = 400
    scatter_color_short = '#F58020'
    scatter_color_long = '#374D9E'

    lmi_hist_data_naive[0] = np.abs(lmi_hist_data_naive[0])
    lmi_hist_data_expert[0] = np.abs(lmi_hist_data_expert[0])

    sns.distplot(lmi_hist_data_naive[0],bins=np.linspace(0,1,numbins),norm_hist=True ,kde=False,color=scatter_color_short,ax=ax_object,label='short preferring',hist_kws={'alpha':1,'cumulative':True, 'histtype':'step', 'linewidth':3})
    sns.distplot(lmi_hist_data_naive[1],bins=np.linspace(0,1,numbins),norm_hist=True, kde=False,color=scatter_color_long,ax=ax_object,label='long preferring',hist_kws={'alpha':1,'cumulative':True, 'histtype':'step', 'linewidth':3})

    sns.distplot(lmi_hist_data_expert[0],bins=np.linspace(0,1,numbins),norm_hist=True,kde=False,color=scatter_color_short,ax=ax_object,label='short preferring',hist_kws={'alpha':1,'cumulative':True, 'histtype':'step', 'linewidth':3})
    sns.distplot(lmi_hist_data_expert[1],bins=np.linspace(0,1,numbins),norm_hist=True,kde=False,color=scatter_color_long,ax=ax_object,label='long preferring',hist_kws={'alpha':1,'cumulative':True, 'histtype':'step', 'linewidth':3})

    ax_object.set_xlim([0,1.0])
    ax_object.set_ylim([0,1.0])

    print('--- LMI K-S TEST ---')
    print(sp.stats.ks_2samp(lmi_hist_data_naive[0],lmi_hist_data_expert[0]))
    print(sp.stats.ks_2samp(lmi_hist_data_naive[1],lmi_hist_data_expert[1]))
    print('--------------------')

    print('--- MEAN TRANSIENT DISTANCE FORM LANDMARK TTEST ---')
    # ipdb.set_trace()
    print('mean distance from landmark NAIVE: ' + str(np.mean(np.concatenate((lm_transient_distance_naive[0],lm_transient_distance_naive[1])))))
    print('mean distance from landmark EXPERT: ' + str(np.mean(np.concatenate((lm_transient_distance_all[0],lm_transient_distance_all[1])))))
    print(sp.stats.ttest_ind(lm_transient_distance_naive[0], lm_transient_distance_all[0]))
    print(sp.stats.ttest_ind(lm_transient_distance_naive[1], lm_transient_distance_all[1]))
    print(sp.stats.ttest_ind(np.concatenate((lm_transient_distance_naive[0], lm_transient_distance_naive[1])), np.concatenate((lm_transient_distance_all[0], lm_transient_distance_all[1]))))
    print(sp.stats.ks_2samp(np.concatenate((lm_transient_distance_naive[0], lm_transient_distance_naive[1])), np.concatenate((lm_transient_distance_all[0], lm_transient_distance_all[1]))))
    print('---------------------------------------------------')


if __name__ == '__main__':
    # list of roi parameter files
    suffix = ''

    # all
    roi_param_list_all = [
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_1'+suffix+'.json','LF170110_2','Day201748_1',41.06],
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_2'+suffix+'.json','LF170110_2','Day201748_2',41.06],
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_3'+suffix+'.json','LF170110_3','Day201748_3',41.06],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day2017719'+suffix+'.json','LF170420_1','Day2017719',35.06],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783'+suffix+'.json','LF170420_1','Day201783',52.97],
                      ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719'+suffix+'.json','LF170421_2','Day20170719',38.34],
                      # ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720'+suffix+'.json','LF170421_2','Day2017720',53.99],
                      ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804'+suffix+'.json','LF170613_1','Day20170804',37.02],
                      ['E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776'+suffix+'.json','LF170222_1','Day201776',71.26],
                      ['E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2'+suffix+'.json','LF171212_2','Day2018218_2',16.01],
                      ['E:\\MTH3_figures\\LF161202_1\\LF161202_1_Day20170209_l23' + suffix + '.json','LF161202_1','Day20170209_l23',21.08],
                      ['E:\\MTH3_figures\\LF161202_1\\LF161202_1_Day20170209_l5' + suffix + '.json','LF161202_1','Day20170209_l5',21.08]
                     ]
                     # ['E:\\MTH3_figures\\LF171211_1\\LF171211_1_Day2018321_2'+suffix+'.json','LF171211_1','Day2018321_2',9.98],

    # L2/3
    roi_param_list_l23 = [
                      ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804'+suffix+'.json','LF170613_1','Day20170804',37.02],
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_1'+suffix+'.json','LF170110_2','Day201748_1',41.06],
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_2'+suffix+'.json','LF170110_2','Day201748_2',41.06],
                      ['E:\\MTH3_figures\\LF170110_2\\LF170110_2_Day201748_3'+suffix+'.json','LF170110_3','Day201748_3',41.06],
                      # ['E:\\MTH3_figures\\LF171211_1\\LF171211_1_Day2018321_2'+suffix+'.json','LF171211_1','Day2018321_2',9.98],
                      ['E:\\MTH3_figures\\LF161202_1\\LF161202_1_Day20170209_l23' + suffix + '.json','LF161202_1','Day20170209_l23',21.08]
                     ]

    # LAYER 5
    roi_param_list_l5 = [
                      ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719'+suffix+'.json','LF170421_2','Day20170719',38.34],
                      # ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720'+suffix+'.json','LF170421_2','Day2017720',53.99],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day2017719'+suffix+'.json','LF170420_1','Day2017719',35.06],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783'+suffix+'.json','LF170420_1','Day201783',52.97],
                      ['E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776'+suffix+'.json','LF170222_1','Day201776',71.26],
                      ['E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2'+suffix+'.json','LF171212_2','Day2018218_2',16.01],
                      ['E:\\MTH3_figures\\LF161202_1\\LF161202_1_Day20170209_l5' + suffix + '.json','LF161202_1','Day20170209_l5',21.08]
                     ]
    # V1
    roi_param_list_v1 = [
                      ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day201777.json','LF170214_1','Day201777',38.49],
                      ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day2017714.json','LF170214_1','Day2017714',38.92],
                      ['E:\\MTH3_figures\\LF171211_2\\LF171211_2_Day201852.json','LF171211_2','Day201852',31.07],
                      ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_1',22.63],
                      ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_2',22.63],
                      ['E:\\MTH3_figures\\LF180219_1\\LF180219_1_Day2018424_0025.json','LF180219_1','Day2018424_0025',30.41]
                     ]

    # naive separate
    roi_param_list_naive = [
                      ['E:\\MTH3_figures\\LF191022_3\\LF191022_3_20191119.json','LF191022_3','20191119',0.0],
                      ['E:\\MTH3_figures\\LF191023_blue\\LF191023_blue_20191119.json','LF191023_blue','20191119',0.0],
                      ['E:\\MTH3_figures\\LF191023_blank\\LF191023_blank_20191116.json','LF191023_blank','20191116',0.0],
                      ['E:\\MTH3_figures\\LF191022_2\\LF191022_2_20191116.json','LF191022_2','20191116',0.0],
                     ]

    # naive matched
    roi_param_list_naivematched = [
                      ['E:\\MTH3_figures\\LF191022_3\\LF191022_3_20191119.json','LF191022_3','20191119',0.0, np.arange(0,136)],
                      ['E:\\MTH3_figures\\LF191023_blue\\LF191023_blue_20191119.json','LF191023_blue','20191119',0.0, np.arange(0,134)],
                      ['E:\\MTH3_figures\\LF191024_1\\LF191024_1_20191115.json','LF191024_1','20191115',0.0, np.arange(0,81)],
                     ]

    # expert matched
    # roi_param_list_expertmatched = [
    #                   ['E:\\MTH3_figures\\LF191022_3\\LF191022_3_20191204.json','LF191022_3','20191204',0.0, np.arange(0,136)],
    #                   ['E:\\MTH3_figures\\LF191023_blue\\LF191023_blue_20191204.json','LF191023_blue','20191204',0.0,np.arange(0,134)],
    #                   ['E:\\MTH3_figures\\LF191024_1\\LF191024_1_20191204.json','LF191024_1','20191204',0.0, np.arange(0,81)],
    #                  ]

    roi_param_list_expertmatched = [
                      ['E:\\MTH3_figures\\LF191022_3\\LF191022_3_20191204.json','LF191022_3','20191204',0.0, np.arange(0,154)],
                      ['E:\\MTH3_figures\\LF191023_blue\\LF191023_blue_20191204.json','LF191023_blue','20191204',0.0,np.arange(0,162)],
                      ['E:\\MTH3_figures\\LF191024_1\\LF191024_1_20191204.json','LF191024_1','20191204',0.0, np.arange(0,122)],
                     ]

    roi_param_list_expertmatched_NEW = [
                      ['E:\\MTH3_figures\\LF191022_3\\LF191022_3_20191204.json','LF191022_3','20191204',0.0, np.arange(137,154)],
                      ['E:\\MTH3_figures\\LF191023_blue\\LF191023_blue_20191204.json','LF191023_blue','20191204',0.0,np.arange(135,162)],
                      ['E:\\MTH3_figures\\LF191024_1\\LF191024_1_20191204.json','LF191024_1','20191204',0.0,np.arange(82,122)],
                     ]




    event_list = ['trialonset','lmcenter','reward']
    trialtypes = ['short', 'long']
    peak_metric = '_peak_'
    subfolder = 'summary'
    normalize = True

    # create figure and axes to later plot on
    fig = plt.figure(figsize=(30,22))
    ax1 = plt.subplot2grid((8,120),(0,00), rowspan=2, colspan=10)
    ax2 = plt.subplot2grid((8,120),(0,20), rowspan=2, colspan=20)
    ax3 = plt.subplot2grid((8,120),(0,10), rowspan=2, colspan=10)
    ax4 = plt.subplot2grid((8,120),(2,00), rowspan=2, colspan=20)
    ax5 = plt.subplot2grid((8,120),(2,20), rowspan=2, colspan=20)
    ax6 = plt.subplot2grid((8,120),(4,00), rowspan=2, colspan=20)
    ax7 = plt.subplot2grid((8,120),(4,20), rowspan=2, colspan=20)
    ax8 = plt.subplot2grid((8,120),(6,00), rowspan=2, colspan=20)
    ax9 = plt.subplot2grid((8,120),(6,20), rowspan=2, colspan=20)
    ax10 = plt.subplot2grid((8,120),(0,40), rowspan=2, colspan=20)
    ax11 = plt.subplot2grid((8,120),(2,40), rowspan=2, colspan=20)
    ax12 = plt.subplot2grid((8,120),(4,40), rowspan=2, colspan=20)
    ax13 = plt.subplot2grid((8,120),(6,40), rowspan=2, colspan=10)
    ax14 = plt.subplot2grid((8,120),(6,50), rowspan=2, colspan=10)
    ax15 = plt.subplot2grid((8,120),(2,70), rowspan=2, colspan=10)
    ax16 = plt.subplot2grid((8,120),(2,100), rowspan=2, colspan=20)
    ax17 = plt.subplot2grid((8,120),(2,60), rowspan=2, colspan=10)
    ax18 = plt.subplot2grid((8,120),(4,60), rowspan=2, colspan=20)
    ax19 = plt.subplot2grid((8,120),(0,100), rowspan=2, colspan=20)
    ax20 = plt.subplot2grid((8,120),(3,80), rowspan=2, colspan=20)
    ax21 = plt.subplot2grid((8,120),(2,80), rowspan=1, colspan=20)
    ax22 = plt.subplot2grid((8,120),(5,80), rowspan=1, colspan=20)
    ax23 = plt.subplot2grid((8,120),(6,60), rowspan=2, colspan=20)
    ax24 = plt.subplot2grid((8,120),(6,80), rowspan=2, colspan=20)
    ax25 = plt.subplot2grid((8,120),(0,60), rowspan=2, colspan=10)
    ax26 = plt.subplot2grid((8,120),(0,70), rowspan=2, colspan=20)
    ax27 = plt.subplot2grid((8,120),(0,90), rowspan=2, colspan=10)
    ax28 = plt.subplot2grid((8,120),(4,100), rowspan=2, colspan=14)
    ax29 = plt.subplot2grid((8,120),(6,100), rowspan=2, colspan=20)

    ax1.set_title('ax1')
    ax2.set_title('ax2')
    ax3.set_title('ax3')
    ax4.set_title('ax4')
    ax5.set_title('ax5')
    ax6.set_title('ax6')
    ax7.set_title('ax7')
    ax8.set_title('ax8')
    ax9.set_title('ax9')
    ax10.set_title('ax10')
    ax11.set_title('ax11')
    ax12.set_title('ax12')
    ax13.set_title('ax13')
    ax14.set_title('ax14')
    ax15.set_title('ax15')
    ax16.set_title('ax16')
    ax17.set_title('ax17')
    ax18.set_title('ax18')
    ax19.set_title('ax19')
    ax20.set_title('ax20')
    ax21.set_title('ax21')
    ax22.set_title('ax22')
    ax23.set_title('ax23')
    ax24.set_title('ax24')
    ax25.set_title('ax25')
    ax26.set_title('ax26')
    ax27.set_title('ax27')
    ax28.set_title('ax28')
    ax29.set_title('ax29')

    plot_layer = 'all'

    if plot_layer is 'all':
        max_peak_all,max_peak_all_ol,lmi_hist_data_all, lm_transient_distance_all = event_maxresponse(roi_param_list_all, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        roi_amplitude_scatter(roi_param_list_all, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)


    if plot_layer is 'l23':
        max_peak_l23,max_peak_l23_ol,frac_lmcenter_short,frac_lmcenter_long = event_maxresponse(roi_param_list_l23, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        # ipdb.set_trace()
        roi_amplitude_scatter(roi_param_list_l23, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)

    # plot_layer = 'l5'
    if plot_layer is 'l5':
        max_peak_l5,max_peak_l5_ol,frac_lmcenter_short,frac_lmcenter_long = event_maxresponse(roi_param_list_l5, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        roi_amplitude_scatter(roi_param_list_l5, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)

    # plot_layer = 'v1'
    if plot_layer is 'v1':
        max_peak_v1,max_peak_v1_ol,lmi_hist_data_v1, lm_transient_distance_v1 = event_maxresponse(roi_param_list_v1, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        roi_amplitude_scatter(roi_param_list_v1, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)

    # plot_layer = 'naive'

    if plot_layer is 'naive':
        max_peak_naive,max_peak_naive_ol,lmi_hist_data_naive, lm_transient_distance_naive = event_maxresponse(roi_param_list_naive, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        roi_amplitude_scatter(roi_param_list_all, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)

    # plot_layer = 'naive_matched'

    if plot_layer is 'naive_matched':
        max_peak_naive,max_peak_naive_ol,lmi_hist_data_naive, lm_transient_distance_naive = event_maxresponse(roi_param_list_naivematched, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        roi_amplitude_scatter(roi_param_list_all, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)

    # plot_layer = 'expert_matched'

    if plot_layer is 'expert_matched':
        max_peak_expert,max_peak_expert_ol,lmi_hist_data_expert, lm_transient_distance_expert = event_maxresponse(roi_param_list_expertmatched, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        roi_amplitude_scatter(roi_param_list_all, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)

    # plot_layer = 'expert_matched_NEW'

    if plot_layer is 'expert_matched_NEW':
        max_peak_expertnew,max_peak_expertnew_ol,lmi_hist_data_expertnew, lm_transient_distance_expertnew = event_maxresponse(roi_param_list_expertmatched_NEW, event_list, trialtypes, peak_metric, ax1, ax3, ax4, ax5, ax2, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax26, ax25, ax27, ax28, ax29, normalize, plot_layer)
        roi_amplitude_scatter(roi_param_list_all, event_list, trialtypes, peak_metric, ax6, ax7, ax8, ax9, normalize)

    # print("--- NAIVE vs EXPERT LMI CUMULATIVE HISTOGRAM ---")
    # fname =  'naive vs expert cumulative hist'
    # naive_vs_expert_lmi(lmi_hist_data_naive,lmi_hist_data_all, lm_transient_distance_naive, lm_transient_distance_all, ax29, subfolder, fname)
    #
    # print("--- NAIVE vs V1 LMI CUMULATIVE HISTOGRAM ---")
    # fname =  'naive vs V1 cumulative hist'
    # naive_vs_expert_lmi(lmi_hist_data_naive,lmi_hist_data_v1, lm_transient_distance_naive, lm_transient_distance_v1, ax29, subfolder, fname)
    #
    # print("--- EXPERT vs EXPERT NEW LMI CUMULATIVE HISTOGRAM ---")
    # fname =  'naive vs expertnew cumulative hist'
    # naive_vs_expert_lmi(lmi_hist_data_expert,lmi_hist_data_expertnew, lm_transient_distance_expert, lm_transient_distance_expertnew, ax29, subfolder, fname)

    # print("--- NAIVE vs EXPERT MATCHED LMI CUMULATIVE HISTOGRAM ---")
    # fname =  'naive vs expertnew cumulative hist'
    # naive_vs_expert_lmi(lmi_hist_data_naive,lmi_hist_data_expert, lm_transient_distance_naive, lm_transient_distance_expert, ax29, subfolder, fname)

    # print("--- NAIVE vs EXPERT NEW LMI CUMULATIVE HISTOGRAM ---")
    # fname =  'naive vs expertnew cumulative hist'
    # naive_vs_expert_lmi(lmi_hist_data_naive,lmi_hist_data_expertnew, lm_transient_distance_naive, lm_transient_distance_expertnew, ax29, subfolder, fname)

    fname = 'summary figure ' + plot_layer
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
    print(fname)



    # print('--- VR vs. OL DIFFERENCES ---')
    # fname = 'RSC vs V1 differences'
    # vr_ol_difference(max_peak_all,max_peak_all_ol,max_peak_v1,max_peak_v1_ol, subfolder, fname)
    #
    # print('--- L23 vs. L5 DIFFERENCES ---')
    # fname = 'L23 vs L5 differences'
    # vr_ol_difference(max_peak_l23,max_peak_l23_ol,max_peak_l5,max_peak_l5_ol, subfolder, fname)
    #
    # print('--- L23 vs. V1 DIFFERENCES ---')
    # fname = 'L23 vs V1 differences'
    # vr_ol_difference(max_peak_l23,max_peak_l23_ol,max_peak_v1,max_peak_v1_ol, subfolder, fname)

    # print('--- EXPERT vs. NAIVE DIFFERENCES ---')
    # fname = 'EXPERT vs NAIVE differences'
    # vr_ol_difference(max_peak_all,max_peak_all_ol,max_peak_naive,max_peak_naive_ol, subfolder, fname)

    # print('--- naive vs. v1 DIFFERENCES ---')
    # fname = 'NAIVE vs V1 differences'
    # vr_ol_difference(max_peak_v1,max_peak_v1_ol,max_peak_naive,max_peak_naive_ol, subfolder, fname)

    # print('--- EXPERT vs. NAIVE vs. V1 DIFFERENCES ---')
    # fname = 'EXPERT vs. NAIVE vs V1 differences'
    # vr_ol_difference_3(max_peak_all,max_peak_all_ol,max_peak_naive,max_peak_naive_ol,max_peak_v1,max_peak_v1_ol, subfolder, fname)
