"""
plot summary data for activity of neurons in the passive condition when they are running vs when they are sitting


"""
import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels as sm_all
from scipy import stats
import h5py, os, sys, traceback, matplotlib, warnings, json, yaml
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
warnings.filterwarnings('ignore')
# import seaborn as sns
import ipdb, traceback
# sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

fformat = 'png'

# normalization method. 1...OL/VR. 2...(OL-VR)/(OL+VR). 3...1-(roi_peak_val-x)/roi_peak_val
NORM_METHOD = 3
CLOSE_TO_ZERO_CUTOFF = 0.001
MIN_TRIALS = 3
MIN_DFVAL = -0.01

sys.path.append(loc_info['base_dir'] + '/Analysis')
from analysis_parameters import MIN_FRACTION_ACTIVE, MIN_MEAN_AMP, MIN_ZSCORE, MIN_TRIALS_ACTIVE, MIN_DF, PEAK_MATCH_WINDOW, MEAN_TRACE_FRACTION

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


def get_eventaligned_rois(rpl, trialtypes, align_event):
    """ return roi number, peak value and peak time of all neurons that have their max response at <align_even> in VR """
    # hold values of mean peak
    event_list = ['trialonset','lmcenter','reward']
    result_max_peak = {}
    # set up empty dicts so we can later append to them
    for tl in trialtypes:
        mouse_sess = rpl[1] + '_' + rpl[3]
        result_max_peak[align_event + '_roi_number_' + tl] = []
        result_max_peak[align_event + '_peakval_' + tl] = []

    # load roi parameters for given session
    with open(rpl[0],'r') as f:
        roi_params = json.load(f)
    # grab a full list of roi numbers
    roi_list_all = roi_params['valid_rois']
    mouse_sess = rpl[1] + '_' + rpl[3]
    # loop through every roi
    for j,r in enumerate(roi_list_all):
        # loop through every trialtype and alignment point to determine largest response
        for tl in trialtypes:
            max_peak = -99
            roi_num = -1
            valid = False
            peak_event = ''
            peak_trialtype = ''
            for el in event_list:
                value_key = el + '_peak_' + tl
                # check roi max peak for each alignment point and store wich ones has the highest value
                if roi_params[value_key][j] > max_peak and roi_response_validation(roi_params, tl, el, j):
                    valid = True
                    max_peak = roi_params[value_key][j]
                    peak_event = el
                    peak_trialtype = tl
                    roi_num = r
            # write results for alignment point with highest value to results dict
            if valid and peak_event == align_event:
                result_max_peak[align_event + '_roi_number_' + peak_trialtype].append(roi_num)
                result_max_peak[align_event + '_peakval_' + peak_trialtype].append(max_peak)

    return result_max_peak

def matching_vr_rois(roi_params, trialtype, align_event, bin_type, event_aligned_rois, roivals):
    """ retrieve peak responses of animals in VR """

    # run through all roi_param files and create empty dictionary lists that we can later append to
    roivals[bin_type + '_roi_number'] = []
    roivals[bin_type + '_peakval_' + trialtype] = []
    roivals[bin_type + '_peakval_ol_' + trialtype] = []
    roivals[bin_type + '_peakval_ol_sit_' + trialtype] = []
    roivals[bin_type + '_peakval_ol_run_' + trialtype] = []
    roivals[bin_type + '_numtrials_ol_sit_' + trialtype] = []
    roivals[bin_type + '_numtrials_ol_run_' + trialtype] = []
    roivals['lmcenter_roinumber_blackbox'] = []
    roivals['lmcenter_peak_sit_blackbox'] = []
    roivals['lmcenter_peak_run_blackbox'] = []
    # grab a full list of roi numbers to later match it with rois provided in roilist

    roi_list_vr = roi_params['valid_rois']
    roi_list_vr_ea = event_aligned_rois['lmcenter_roi_number_' + trialtype]
    roi_list_idx_blackbox = roi_params['lmcenter_roi_number_blackbox']
    # print(roi_list_vr_ea)
    # roi_list_vr_svr = roi_params[align_event + '_roi_number_svr_' + trialtype]

    for roi in roi_list_vr_ea:
        roi_idx = np.argwhere(np.asarray(roi_list_vr) == roi)[0][0]
        if roi_params[bin_type + '_peak_' + trialtype][roi_idx] > 0.1 and \
           roi_params[bin_type + '_filter_2_peak_' + trialtype + '_ol'][roi_idx] > MIN_DFVAL and \
           roi_params[bin_type + '_filter_1_peak_' + trialtype + '_ol'][roi_idx] > MIN_DFVAL and \
           roi_params[bin_type + '_filter_2_numtrials_' + trialtype + '_ol'][roi_idx] > MIN_TRIALS and \
           roi_params[bin_type + '_filter_1_numtrials_' + trialtype + '_ol'][roi_idx] > MIN_TRIALS:
            roivals[bin_type + '_roi_number'].append(roi)
            roivals[bin_type + '_peakval_' + trialtype].append(roi_params[bin_type + '_peak_' + trialtype][roi_idx])
            roivals[bin_type + '_peakval_ol_' + trialtype].append(roi_params[bin_type + '_peak_' + trialtype + '_ol'][roi_idx])
            roivals[bin_type + '_peakval_ol_sit_' + trialtype].append(roi_params[bin_type + '_filter_2_peak_' + trialtype + '_ol'][roi_idx])
            roivals[bin_type + '_peakval_ol_run_' + trialtype].append(roi_params[bin_type + '_filter_1_peak_' + trialtype + '_ol'][roi_idx])
            roivals[bin_type + '_numtrials_ol_sit_' + trialtype].append(roi_params[bin_type + '_filter_2_numtrials_' + trialtype + '_ol'][roi_idx])
            roivals[bin_type + '_numtrials_ol_run_' + trialtype].append(roi_params[bin_type + '_filter_1_numtrials_' + trialtype + '_ol'][roi_idx])

        err = False
        try:
            roi_blackbox_idx = np.argwhere(np.asarray(roi_list_idx_blackbox) == roi)[0][0]
        except IndexError as e:
            err = True
            print('WARNING: blackbox value not found')

        if not err:
            # if roi_params['lmcenter_peak_sit_blackbox'][roi_blackbox_idx] < -0.1:
            #     roi_params['lmcenter_peak_sit_blackbox'][roi_blackbox_idx] = 0.1
            # if roi_params['lmcenter_peak_run_blackbox'][roi_blackbox_idx] < -0.1:
            #     roi_params['lmcenter_peak_run_blackbox'][roi_blackbox_idx] = 0.1
            roivals['lmcenter_roinumber_blackbox'].append(roi)
            roivals['lmcenter_peak_sit_blackbox'].append(roi_params['lmcenter_peak_sit_blackbox'][roi_blackbox_idx])
            roivals['lmcenter_peak_run_blackbox'].append(roi_params['lmcenter_peak_run_blackbox'][roi_blackbox_idx])
        else:
            roivals['lmcenter_roinumber_blackbox'].append(-99)
            roivals['lmcenter_peak_sit_blackbox'].append(-99)
            roivals['lmcenter_peak_run_blackbox'].append(-99)

        # roivals['lmcenter_peak_sit_blackbox' + trialtype] = roi_params[align_event + '_peak_sit_blackbox']


    return roivals, roi_list_vr_ea

def scatterplot_svr(roi_param_list, align_event, trialtypes, bin_type, ax_object1, ax_object2, ax_object4, ax_object5, ax_object3, ax_object6, ax_object7, ax_object8, ax_object9, ax_object10, ax_object11, ax_object12, ax_object13):
    """ plot peak activity of neurons during openloop (passive) condition when they are stationary vs when they are running """
    # collect all roi values plotted for statistical getExistingDirectory

    # all roi values of animals either sitting or running in OL
    all_rois_num = []
    all_roi_vals_sit = []
    all_roi_vals_run = []
    all_roi_trialnums_sit = []
    all_roi_trialnums_run = []
    all_roi_vals_sit_norm = []
    all_roi_vals_run_norm = []

    # hold blackbox running values for matched
    all_roi_vals_blackbox_run_norm_olcomp = []

    # roi values of animals sitting or running in blackbox
    all_roi_vals_blackbox_sit = []
    all_roi_vals_blackbox_run = []
    all_roi_vals_blackbox_sit_norm = []
    all_roi_vals_blackbox_run_norm = []

    # roi values during VR navigation
    all_roi_vals_vr = []
    all_roi_vals_vr_norm = []
    # vr_sit, vr_run : like _sit and _run but matched to the same ROI in VR

    # line thickness of individual neurons
    ind_lw = 0.5

    roivals = {}
    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        # print(rpl)
        # load roi parameters for given session
        with open(rpl[0],'r') as f:
            roi_params = json.load(f)
        # run through alignment points and trialtypes
        print(rpl[0])
        all_rois_sess = []
        event_aligned_rois = []
        for tl in trialtypes:
            event_aligned_rois = get_eventaligned_rois(rpl, trialtypes, align_event)
            vr_roisvals, roi_nums = matching_vr_rois(roi_params, tl, align_event, bin_type, event_aligned_rois, roivals)
            # for j,ks in enumerate(vr_roisvals[bin_type + '_roi_number']):
                # print(vr_roisvals[bin_type + '_roi_number'][j], np.round(vr_roisvals[bin_type + '_peakval_ol_run_' + tl][j],2), np.round(vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][j],2))
            # continue
            # print(roi_nums)
            if tl == 'short':
                scatter_color = '0.8'
            elif tl == 'long':
                scatter_color = '0.5'
            # plot amplitudes for sit vs run
            # ax_object1.scatter(roi_params[align_event + '_peak_sit_' + tl], roi_params[align_event + '_peak_run_' + tl], s=5,color=scatter_color)
            # plot VR vs sit and VR vs run

            for i in range(len(vr_roisvals[bin_type + '_peakval_' + tl])):
                # if vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i] > -0.1 and vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i] > -0.1 and vr_roisvals[bin_type + '_peakval_' + tl][i] > 0.1:
                # if not np.isnan(vr_roisvals[bin_type + '_peakval_' + tl][i]) and not np.isnan(roi_params[bin_type + '_peak_sit_' + tl][i]) and not np.isnan(roi_params[bin_type + '_peak_run_' + tl][i]):
                ax_object2.plot([0,1], [vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i],vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i]],c=scatter_color,lw=ind_lw,zorder=0)
                ax_object4.plot([0,1], [vr_roisvals[bin_type + '_peakval_' + tl][i],vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i]],c=scatter_color,lw=ind_lw,zorder=0)
                ax_object5.plot([0,1], [vr_roisvals[bin_type + '_peakval_' + tl][i],vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i]],c=scatter_color,lw=ind_lw,zorder=0)

                all_roi_vals_sit.append(vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i])
                # if vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i] < -0.1:
                #     ipdb.set_trace()
                all_roi_vals_run.append(vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i])
                all_roi_vals_vr.append(vr_roisvals[bin_type + '_peakval_' + tl][i])

                norm_val = vr_roisvals[bin_type + '_peakval_' + tl][i]
                # set a lower bound for the normalization value
                if norm_val < CLOSE_TO_ZERO_CUTOFF:
                    norm_val = CLOSE_TO_ZERO_CUTOFF
                if vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i] < CLOSE_TO_ZERO_CUTOFF:
                    vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i] = CLOSE_TO_ZERO_CUTOFF
                if vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i] < CLOSE_TO_ZERO_CUTOFF:
                    vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i] = CLOSE_TO_ZERO_CUTOFF
                if vr_roisvals['lmcenter_peak_run_blackbox'][i] < CLOSE_TO_ZERO_CUTOFF:
                    vr_roisvals['lmcenter_peak_run_blackbox'][i] = CLOSE_TO_ZERO_CUTOFF

                if NORM_METHOD == 1:
                    ol_sit_norm = vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i]/norm_val
                    ol_run_norm = vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i]/norm_val
                    bb_run_norm = vr_roisvals['lmcenter_peak_run_blackbox'][i]/norm_val
                    vr_val = 0
                elif NORM_METHOD == 2:
                    ol_sit_norm = (vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i] - norm_val)/(norm_val + vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i])
                    ol_run_norm = (vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i] - norm_val)/(norm_val + vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i])
                    bb_run_norm = (vr_roisvals['lmcenter_peak_run_blackbox'][i] - norm_val)/(vr_roisvals['lmcenter_peak_run_blackbox'][i] + norm_val)
                    vr_val = ((norm_val + vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i])+(norm_val + vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i]))/2
                elif NORM_METHOD == 3:
                    ol_sit_norm = 1-((norm_val - vr_roisvals[bin_type + '_peakval_ol_sit_' + tl][i])/norm_val)
                    ol_run_norm = 1-((norm_val - vr_roisvals[bin_type + '_peakval_ol_run_' + tl][i])/norm_val)
                    bb_run_norm = 1-((norm_val - vr_roisvals['lmcenter_peak_run_blackbox'][i])/norm_val)
                    vr_val = 1

                ax_object8.plot([0,1],  [ol_sit_norm,ol_run_norm],lw=ind_lw,c=scatter_color,zorder=2)
                ax_object10.plot([0,1], [vr_val,ol_sit_norm],lw=ind_lw,c=scatter_color,zorder=2)
                ax_object11.plot([0,1], [vr_val,ol_run_norm],lw=ind_lw,c=scatter_color,zorder=2)

                all_roi_trialnums_sit.append(vr_roisvals[bin_type + '_numtrials_ol_sit_' + tl][i])
                all_roi_trialnums_run.append(vr_roisvals[bin_type + '_numtrials_ol_run_' + tl][i])

                if vr_roisvals['lmcenter_roinumber_blackbox'][i] >= 0:
                    all_roi_vals_blackbox_run_norm_olcomp.append(bb_run_norm)
                else:
                    all_roi_vals_blackbox_run_norm_olcomp.append(np.nan)
                #
                all_roi_vals_sit_norm.append(ol_sit_norm)
                all_roi_vals_run_norm.append(ol_run_norm)
                all_roi_vals_vr_norm.append(vr_val)
                # all_rois_sess.append(roi_params[bin_type + '_roi_number_svr_' + tl][i])

            ax_object1.scatter(all_roi_vals_sit, all_roi_vals_run, s=5,color='0.65')
            ax_object7.scatter(all_roi_vals_sit_norm, all_roi_vals_run_norm, s=5,color='0.65')
        # plot blackbox activity
        # valid blackbox idx is just used to filter out any potential
        valid_bb_idx = []
        # find matching roi and determine peak value (=normalizing value) in either short or long
        # roi_list_vr = roi_params['valid_rois']
        # roi_list_blackbox = vr_roisvals['lmcenter_roinumber_blackbox'][i]
        for i in range(len(vr_roisvals['lmcenter_peak_sit_blackbox'])):
            if vr_roisvals[align_event + '_peak_sit_blackbox'][i] > -99 and vr_roisvals[align_event + '_peak_run_blackbox'][i] > -99:
                valid_bb_idx.append(i)
                all_roi_vals_blackbox_sit.append(vr_roisvals['lmcenter_peak_sit_blackbox'][i])
                all_roi_vals_blackbox_run.append(vr_roisvals['lmcenter_peak_run_blackbox'][i])
                ax_object3.plot([0,1], [vr_roisvals['lmcenter_peak_sit_blackbox'][i],vr_roisvals['lmcenter_peak_run_blackbox'][i]],c='0.65',zorder=0)
                # get peak value as normalizing factor.
                if len(vr_roisvals[bin_type + '_peakval_short']) > i and len(vr_roisvals[bin_type + '_peakval_long']) > i:
                    norm_val_short = vr_roisvals[bin_type + '_peakval_short'][i]
                    norm_val_long = vr_roisvals[bin_type + '_peakval_long'][i]
                    norm_val = np.amax([norm_val_short,norm_val_long])
                elif len(vr_roisvals[bin_type + '_peakval_short']) > i:
                    norm_val = vr_roisvals[bin_type + '_peakval_short'][i]
                elif len(vr_roisvals[bin_type + '_peakval_long']) > i:
                    norm_val = vr_roisvals[bin_type + '_peakval_long'][i]

                # a catch just in case the norm value is negative or really small, dispropotionally affecting the normalization
                if norm_val < CLOSE_TO_ZERO_CUTOFF:
                    norm_val = CLOSE_TO_ZERO_CUTOFF
                if vr_roisvals['lmcenter_peak_sit_blackbox'][i] < CLOSE_TO_ZERO_CUTOFF:
                    vr_roisvals['lmcenter_peak_sit_blackbox'][i] = CLOSE_TO_ZERO_CUTOFF
                if vr_roisvals['lmcenter_peak_run_blackbox'][i] < CLOSE_TO_ZERO_CUTOFF:
                    vr_roisvals['lmcenter_peak_run_blackbox'][i] = CLOSE_TO_ZERO_CUTOFF

                if NORM_METHOD == 1:
                    ol_sit_norm = vr_roisvals['lmcenter_peak_sit_blackbox'][i]/norm_val
                    ol_run_norm = vr_roisvals['lmcenter_peak_run_blackbox'][i]/norm_val
                    vr_val = 1 # vr_roisvals[bin_type + '_peakval_' + tl][i]/norm_val
                elif NORM_METHOD == 2:
                    ol_sit_norm = (vr_roisvals['lmcenter_peak_sit_blackbox'][i] - norm_val)/(norm_val + vr_roisvals['lmcenter_peak_sit_blackbox'][i])
                    ol_run_norm = (vr_roisvals['lmcenter_peak_run_blackbox'][i] - norm_val)/(norm_val + vr_roisvals['lmcenter_peak_run_blackbox'][i])
                    vr_val = ((norm_val + vr_roisvals['lmcenter_peak_sit_blackbox'][i])+(norm_val + vr_roisvals['lmcenter_peak_run_blackbox'][i]))/2
                elif NORM_METHOD == 3:
                    ol_sit_norm = 1 - ((norm_val - vr_roisvals['lmcenter_peak_sit_blackbox'][i])/norm_val)
                    ol_run_norm = 1 - ((norm_val - vr_roisvals['lmcenter_peak_run_blackbox'][i])/norm_val)
                    vr_val = 1

                ax_object9.plot([0,1], [ol_sit_norm,ol_run_norm],c='0.65',zorder=2)
                all_roi_vals_blackbox_sit_norm.append(ol_sit_norm)
                all_roi_vals_blackbox_run_norm.append(ol_run_norm)

        # count how many rois from each animal are plotted (only count unique roi numbers so short and long responding rois are not counted double)
        all_rois_num.append(np.unique(np.array(all_rois_sess)).shape[0])

    # BLACKBOX responses
    # ax_object3.scatter([0],[np.nanmean(np.array(all_roi_vals_blackbox_sit))],s=25,marker='s',facecolor='w',linewidths=2,color='k',zorder=4)
    # ax_object3.scatter([1],[np.nanmean(np.array(all_roi_vals_blackbox_run))],s=25,marker='s',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    # ax_object3.plot([0,1],[np.nanmean(np.array(all_roi_vals_blackbox_sit)), np.nanmean(np.array(all_roi_vals_blackbox_run))], lw=3, c='k', zorder=0)
    # ax_object3.errorbar([0,1],[np.nanmean(np.array(all_roi_vals_blackbox_sit)), np.nanmean(np.array(all_roi_vals_blackbox_run))],yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_blackbox_run),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    parts = ax_object3.boxplot([np.array(all_roi_vals_blackbox_sit),np.array(all_roi_vals_blackbox_run)],patch_artist=True,showfliers=False,
    whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
    medianprops=dict(color='black', linewidth=1, solid_capstyle='butt'),
    widths=(0.3,0.3),positions=(0,1))

    colors = [['w','k'],['#FF0000','k']]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

    # BLACKBOX responses NORMALIZED
    ax_object9.scatter([0],[np.mean(np.array(all_roi_vals_blackbox_sit_norm))],s=25,marker='s',facecolor='w',linewidths=2,color='k',zorder=4)
    ax_object9.scatter([1],[np.mean(np.array(all_roi_vals_blackbox_run_norm))],s=25,marker='s',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    ax_object9.plot([0,1],[np.mean(np.array(all_roi_vals_blackbox_sit_norm)), np.mean(np.array(all_roi_vals_blackbox_run_norm))], lw=3, c='k', zorder=3)
    ax_object9.errorbar([0,1],[np.mean(np.array(all_roi_vals_blackbox_sit_norm)), np.mean(np.array(all_roi_vals_blackbox_run_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit_norm),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_blackbox_run_norm),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # OL - motor vs OL + motor
    # ax_object2.scatter([0],[np.nanmean(np.array(all_roi_vals_sit))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    # ax_object2.scatter([1],[np.nanmean(np.array(all_roi_vals_run))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    # ax_object2.plot([0,1],[np.nanmean(np.array(all_roi_vals_sit)), np.nanmean(np.array(all_roi_vals_run))], lw=3, c='k', zorder=3)
    # print(np.nanmean(np.array(all_roi_vals_sit)), np.nanmean(np.array(all_roi_vals_run)))
    # ax_object2.errorbar([0,1],[np.nanmean(np.array(all_roi_vals_sit)), np.nanmean(np.array(all_roi_vals_run))],yerr=[sp.stats.sem(np.array(all_roi_vals_sit),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_run),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    parts = ax_object2.boxplot([np.array(all_roi_vals_sit),np.array(all_roi_vals_run)],patch_artist=True,showfliers=False,
    whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
    medianprops=dict(color='black', linewidth=1, solid_capstyle='butt'),
    widths=(0.3,0.3),positions=(0,1))

    colors = [['#3953A3','#3953A3'],['#FF0000','#FF0000']]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

    # OL - motor vs OL + motor NORMALIZED
    ax_object8.scatter([0],[np.nanmean(np.array(all_roi_vals_sit_norm))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    ax_object8.scatter([1],[np.nanmean(np.array(all_roi_vals_run_norm))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    ax_object8.plot([0,1],[np.nanmean(np.array(all_roi_vals_sit_norm)), np.nanmean(np.array(all_roi_vals_run_norm))], lw=3, c='k', zorder=3)
    ax_object8.errorbar([0,1],[np.nanmean(np.array(all_roi_vals_sit_norm)), np.nanmean(np.array(all_roi_vals_run_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_sit_norm),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_run_norm),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # VR vs OL - motor
    # ax_object4.scatter([0],[np.nanmean(np.array(all_roi_vals_vr))],s=40,marker='s',linewidths=0,c='k',zorder=4)
    # ax_object4.scatter([1],[np.nanmean(np.array(all_roi_vals_sit))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    # ax_object4.plot([0,1],[np.nanmean(np.array(all_roi_vals_vr)), np.nanmean(np.array(all_roi_vals_sit))], lw=3, c='k', zorder=3)
    # ax_object4.errorbar([0,1],[np.nanmean(np.array(all_roi_vals_vr)), np.nanmean(np.array(all_roi_vals_sit))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_sit),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    parts = ax_object4.boxplot([np.array(all_roi_vals_vr),np.array(all_roi_vals_sit)],patch_artist=True,showfliers=False,
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='0.5', linewidth=1, solid_capstyle='butt'),
        widths=(0.3,0.3),positions=(0,1))

    colors = [['k','k'],['#3953A3','#3953A3']]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

    # VR vs OL - motor NORMALIZED
    ax_object10.scatter([0],[np.nanmean(np.array(all_roi_vals_vr_norm))],s=40,marker='s',linewidths=0,color='k',zorder=4)
    ax_object10.scatter([1],[np.nanmean(np.array(all_roi_vals_sit_norm))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    ax_object10.plot([0,1],[np.nanmean(np.array(all_roi_vals_vr_norm)), np.nanmean(np.array(all_roi_vals_sit_norm))], lw=3, c='k', zorder=3)
    ax_object10.errorbar([0,1],[np.nanmean(np.array(all_roi_vals_vr_norm)), np.nanmean(np.array(all_roi_vals_sit_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr_norm),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_sit_norm),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # VR vs OL + motor
    # ax_object5.scatter([0],[np.nanmean(np.array(all_roi_vals_vr))],s=40,marker='s',linewidths=0,color='k',zorder=4)
    # ax_object5.scatter([1],[np.nanmean(np.array(all_roi_vals_run))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    # ax_object5.plot([0,1],[np.nanmean(np.array(all_roi_vals_vr)), np.nanmean(np.array(all_roi_vals_run))], lw=3, c='k', zorder=3)
    # ax_object5.errorbar([0,1],[np.nanmean(np.array(all_roi_vals_vr)), np.nanmean(np.array(all_roi_vals_run))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_run),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    parts = ax_object5.boxplot([np.array(all_roi_vals_vr),np.array(all_roi_vals_run)],patch_artist=True,showfliers=False,
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='0.5', linewidth=1, solid_capstyle='butt'),
        widths=(0.3,0.3),positions=(0,1))

    colors = [['k','k'],['#FF0000','#FF0000']]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

    # VR vs OL + motor NORMALIZED
    ax_object11.scatter([0],[np.nanmean(np.array(all_roi_vals_vr_norm))],s=40,marker='s',linewidths=0,color='k',zorder=4)
    ax_object11.scatter([1],[np.nanmean(np.array(all_roi_vals_run_norm))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    ax_object11.plot([0,1],[np.nanmean(np.array(all_roi_vals_vr_norm)), np.nanmean(np.array(all_roi_vals_run_norm))], lw=3, c='k', zorder=3)
    ax_object11.errorbar([0,1],[np.nanmean(np.array(all_roi_vals_vr_norm)), np.nanmean(np.array(all_roi_vals_run_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr_norm),nan_policy='omit'),sp.stats.sem(np.array(all_roi_vals_run_norm),nan_policy='omit')],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')


    parts = ax_object6.boxplot([np.array(all_roi_vals_blackbox_sit),np.array(all_roi_vals_sit),np.array(all_roi_vals_blackbox_run),np.array(all_roi_vals_run),np.array(all_roi_vals_vr)],patch_artist=True,
        whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
        medianprops=dict(color='0.5', linewidth=1, solid_capstyle='butt'),
        widths=(0.3,0.3,0.3,0.3,0.3),positions=(0,1,2,3,4))

    colors = [['w','k'],['#3953A3','#3953A3'],['#EC2024','k'], ['#EC2024','#EC2024'],['k','k']]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

    # ax_object6.plot([0,1,2,3], [np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_blackbox_run)), np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_vr))])
    # ax_object6.plot([0,1,2,3,4], [np.mean(np.array(all_roi_vals_blackbox_sit)), np.mean(np.array(all_roi_vals_blackbox_run)), np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_vr))],lw=1,c='k',zorder=3)
    # for i in range(np.array(all_roi_vals_blackbox_run).shape[0]):
    #         ax_object6.plot([1,2,3,4], [np.array(all_roi_vals_blackbox_run)[i], np.array(all_roi_vals_sit)[i], np.array(all_roi_vals_run)[i], np.array(all_roi_vals_vr)[i]],lw=1,c='0.8',zorder=3)
    ax_object6.plot([0,1,2,3,4], [np.nanmedian(np.array(all_roi_vals_blackbox_sit)), np.nanmedian(np.array(all_roi_vals_sit)),np.nanmedian(np.array(all_roi_vals_blackbox_run)), np.nanmedian(np.array(all_roi_vals_run)), np.nanmedian(np.array(all_roi_vals_vr))],lw=1,c='k',zorder=3)
    # ax_object6.scatter([0], [np.nanmean(np.array(all_roi_vals_blackbox_sit))],s=25,marker='s',facecolor='w',linewidths=2,color='k',zorder=4)
    # ax_object6.scatter([1], [np.nanmean(np.array(all_roi_vals_blackbox_run))],s=25,marker='s',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    # ax_object6.scatter([2], [np.nanmean(np.array(all_roi_vals_sit))],s=50,marker='s',linewidths=0,c=['#3953A3'],zorder=4)
    # ax_object6.scatter([3], [np.nanmean(np.array(all_roi_vals_run))],s=50,marker='s',linewidths=0,c=['#EC2024'],zorder=4)
    # ax_object6.scatter([4], [np.nanmean(np.array(all_roi_vals_vr))],s=50,marker='s',linewidths=0,c=['k'],zorder=4)
    #
    # ax_object6.errorbar([0,1,2,3,4], [np.nanmean(np.array(all_roi_vals_blackbox_sit)), np.nanmean(np.array(all_roi_vals_blackbox_run)), np.nanmean(np.array(all_roi_vals_sit)), np.nanmean(np.array(all_roi_vals_run)), np.nanmean(np.array(all_roi_vals_vr))], \
    #                     yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_blackbox_run),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_sit),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_run),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_vr),nan_policy='omit')], \
    #                     fmt='', c='k')


    # for i in range(np.array(all_roi_vals_blackbox_run_norm).shape[0]):
    #         ax_object12.plot([1,2,3,4], [np.array(all_roi_vals_blackbox_run_norm)[i], np.array(all_roi_vals_sit_norm)[i], np.array(all_roi_vals_run_norm)[i], np.array(all_roi_vals_vr_norm)[i]],lw=1,c='0.8',zorder=3)

    # parts = ax_object12.boxplot([np.array(all_roi_vals_blackbox_sit_norm),np.array(all_roi_vals_sit_norm),np.array(all_roi_vals_blackbox_run_norm),np.array(all_roi_vals_run_norm),np.array(all_roi_vals_vr_norm)],patch_artist=True,
    #     whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
    #     medianprops=dict(color='black', linewidth=1, solid_capstyle='butt'),
    #     widths=(0.3,0.3,0.3,0.3,0.3),positions=(0,1,2,3,4))
    #
    # colors = [['w','k'], ['#3953A3','#3953A3'],['#EC2024','k'],['#EC2024','#EC2024'],['k','k']]
    # for patch, color in zip(parts['boxes'], colors):
    #     patch.set_facecolor(color[0])
    #     patch.set_edgecolor(color[1])
    #     patch.set_linewidth(2)

    ax_object12.plot([0,1,2,3,4], [np.nanmedian(np.array(all_roi_vals_blackbox_sit_norm)), np.nanmedian(np.array(all_roi_vals_blackbox_run_norm)), np.nanmedian(np.array(all_roi_vals_sit_norm)), np.nanmedian(np.array(all_roi_vals_run_norm)), np.nanmedian(np.array(all_roi_vals_vr_norm))],lw=1,c='k',zorder=3)
    ax_object12.scatter([0], [np.nanmedian(np.array(all_roi_vals_blackbox_sit_norm))],s=25,marker='o',facecolor='w',linewidths=2,color='k',zorder=4)
    ax_object12.scatter([1], [np.nanmedian(np.array(all_roi_vals_blackbox_run_norm))],s=25,marker='o',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    ax_object12.scatter([2], [np.nanmedian(np.array(all_roi_vals_sit_norm))],s=50,marker='o',linewidths=0,c=['#3953A3'],zorder=4)
    ax_object12.scatter([3], [np.nanmedian(np.array(all_roi_vals_run_norm))],s=50,marker='o',linewidths=0,c=['#EC2024'],zorder=4)
    ax_object12.scatter([4], [np.nanmedian(np.array(all_roi_vals_vr_norm))],s=50,marker='o',linewidths=0,c=['k'],zorder=4)


    if NORM_METHOD == 1 or NORM_METHOD == 2:
        ax_object12.scatter([3], [np.nanmedian(np.array(all_roi_vals_blackbox_run_norm)) + (1+np.nanmedian(np.array(all_roi_vals_sit_norm)))],s=50,marker='o',linewidths=0,c=['#9E00A2'],zorder=4)
    elif NORM_METHOD == 3:
        ax_object12.scatter([3], [np.nanmedian(np.array(all_roi_vals_blackbox_run_norm_olcomp) + (np.array(all_roi_vals_sit_norm)))],s=50,marker='o',linewidths=0,c=['#9E00A2'],zorder=4)

    # ax_object12.errorbar([0,1,2,3,4], [np.nanmean(np.array(all_roi_vals_blackbox_sit_norm)), np.nanmean(np.array(all_roi_vals_blackbox_run_norm)), np.nanmean(np.array(all_roi_vals_sit_norm)), np.nanmean(np.array(all_roi_vals_run_norm)), np.nanmean(np.array(all_roi_vals_vr_norm))], \
    #                     yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit_norm),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_blackbox_run_norm),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_sit_norm),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_run_norm),nan_policy='omit'), sp.stats.sem(np.array(all_roi_vals_vr_norm),nan_policy='omit')], \
    #                     fmt='o', c='k')

    #all_roi_vals_blackbox_run_norm_olcomp

    if NORM_METHOD == 1 or NORM_METHOD == 2:
        visual_motor_sum = np.array(all_roi_vals_blackbox_run_norm_olcomp) + (1+np.array(all_roi_vals_sit_norm))
    if NORM_METHOD == 3:
        visual_motor_sum = np.array(all_roi_vals_blackbox_run_norm_olcomp) + (np.array(all_roi_vals_sit_norm))

    # ipdb.set_trace()

    parts = ax_object13.boxplot([np.array(all_roi_vals_run_norm),np.array(visual_motor_sum)[~np.isnan(np.array(visual_motor_sum))]],patch_artist=True,showfliers=False,
    whiskerprops=dict(linestyle='-', color='black', linewidth=1, solid_capstyle='butt'),
    medianprops=dict(color='0.5', linewidth=1, solid_capstyle='butt'),
    widths=(0.3,0.3),positions=(0,1))

    colors = [['#FF0000','#FF0000'],['#9E00A2','#9E00A2']]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color[0])
        patch.set_edgecolor(color[1])
        patch.set_linewidth(2)

    # ax_object13.scatter(np.zeros((len(visual_motor_sum),)), visual_motor_sum, zorder=3)
    # ax_object13.scatter(np.ones((len(visual_motor_sum),)), np.array(all_roi_vals_run_norm), zorder=3)
    for i in range(len(visual_motor_sum)):
        ax_object13.plot([0,1],[np.array(all_roi_vals_run_norm)[i], visual_motor_sum[i]], c='0.8',lw=0.5, zorder=0)

    print('--- LINEAR SUM TEST ---')
    print(sp.stats.ttest_rel(visual_motor_sum,np.array(all_roi_vals_run_norm),nan_policy='omit'))
    print(sp.stats.mannwhitneyu(visual_motor_sum,np.array(all_roi_vals_run_norm)))
    print(sp.stats.wilcoxon(visual_motor_sum,np.array(all_roi_vals_run_norm)))
    print('-----------------------')


    print('--- NUMBER OF TRIALS ---')
    print('sit trials: ', str(np.nanmean(np.array(all_roi_trialnums_sit))), ' sem: ', str(sp.stats.sem(np.array(all_roi_trialnums_sit))))
    print('run trials: ', str(np.nanmean(np.array(all_roi_trialnums_run))), ' sem: ', str(sp.stats.sem(np.array(all_roi_trialnums_run))))
    print('------------------------')

    print('--- MEAN VALUES ---')
    print('VR: ', str(np.nanmean(np.array(all_roi_vals_vr))), ' +/- SEM: ', str(sp.stats.sem(np.array(all_roi_vals_vr))))
    print('OL+MOTOR: ', str(np.nanmean(np.array(all_roi_vals_run))), ' +/- SEM: ', str(sp.stats.sem(np.array(all_roi_vals_run))))
    print('OL-MOTOR: ', str(np.nanmean(np.array(all_roi_vals_sit))), ' +/- SEM: ', str(sp.stats.sem(np.array(all_roi_vals_sit))))
    print('BB SIT: ', str(np.nanmean(np.array(all_roi_vals_blackbox_sit))), ' +/- SEM: ', str(sp.stats.sem(np.array(all_roi_vals_blackbox_sit))))
    print('BB RUN: ', str(np.nanmean(np.array(all_roi_vals_blackbox_run))), ' +/- SEM: ', str(sp.stats.sem(np.array(all_roi_vals_blackbox_run))))
    print('-------------------')

    # print(np.nanmean(np.array(all_roi_vals_run)), np.nanmean(np.array(all_roi_vals_run_norm)))
    #
    # print([np.nanmean(np.array(all_roi_vals_sit)), np.nanmean(np.array(all_roi_vals_blackbox_run)), np.nanmean(np.array(all_roi_vals_run)), np.nanmean(np.array(all_roi_vals_vr))])
    # print(np.sum(all_rois_num))
    # slope, intercept, r_value, p_value, std_err = sp.stats.linregress([0,1,2], [np.nanmean(np.array(all_roi_vals_blackbox_sit)),np.nanmean(np.array(all_roi_vals_sit)), np.nanmean(np.array(all_roi_vals_blackbox_run))])
    # ax_object6.plot(np.array([0,1,2,3,4]), intercept + slope*np.array([0,1,2,3,4]), '0.5', label='fitted line', ls='--', lw=2, zorder=2)
    # ax_object6.scatter([2], [np.nanmean(np.array(all_roi_vals_sit_norm))+np.nanmean(np.array(all_roi_vals_blackbox_run_norm))],s=50,marker='s',linewidths=0,c=['k'],zorder=4)

    # slope, intercept, r_value, p_value, std_err = sp.stats.linregress([0,1,2], [np.nanmean(np.array(all_roi_vals_blackbox_sit_norm)), np.nanmean(np.array(all_roi_vals_blackbox_run_norm)),np.nanmean(np.array(all_roi_vals_sit_norm))])
    # ax_object12.plot(np.array([0,1,2,3,4]), intercept + slope*np.array([0,1,2,3,4]), '0.5', label='fitted line', ls='--', lw=2, zorder=2)

    ax_object1.plot([0,5],[0,5],ls='--',c='k')
    ax_object7.plot([0,5],[0,5],ls='--',c='k')

    # carry out statistical analysis. This is not (yet) the correct test: we are treating each group independently, rather than taking into account within-group and between-group variance
    # print(sp.stats.f_oneway(np.array(all_roi_vals_sit),np.array(all_roi_vals_run),np.array(all_roi_vals_vr),np.array(all_roi_vals_blackbox_sit),np.array(all_roi_vals_blackbox_run)))
    # group_labels = ['all_roi_vals_sit'] * np.array(all_roi_vals_sit).shape[0] + \
    #                ['all_roi_vals_run'] * np.array(all_roi_vals_run).shape[0] + \
    #                ['all_roi_vals_vr'] * np.array(all_roi_vals_vr).shape[0] + \
    #                ['all_roi_vals_blackbox_sit'] * np.array(all_roi_vals_blackbox_sit).shape[0] + \
    #                ['all_roi_vals_blackbox_run'] * np.array(all_roi_vals_blackbox_run).shape[0]
    # mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(all_roi_vals_sit),np.array(all_roi_vals_run),np.array(all_roi_vals_vr),np.array(all_roi_vals_blackbox_sit),np.array(all_roi_vals_blackbox_run))),group_labels)
    # posthoc_res_ss = mc_res_ss.tukeyhsd()

    # carry out statistic testing - first do a kruskal wallis, followed by mannwhitneyu pairwaise comparisons with bonferroni correction at the end
    print('--- KRUSKAL WALLIS ---')
    print('number of neurons: ', str(len(all_roi_vals_sit)))
    print(sp.stats.kruskal(np.array(all_roi_vals_sit),np.array(all_roi_vals_run),np.array(all_roi_vals_vr),np.array(all_roi_vals_blackbox_sit),np.array(all_roi_vals_blackbox_run)))
    _, p_sit_run = sp.stats.mannwhitneyu(all_roi_vals_sit,all_roi_vals_run)
    _, p_sit_vr = sp.stats.mannwhitneyu(all_roi_vals_sit,all_roi_vals_vr)
    _, p_run_vr = sp.stats.mannwhitneyu(all_roi_vals_run,all_roi_vals_vr)
    _, p_sit_bbsit = sp.stats.mannwhitneyu(all_roi_vals_sit,all_roi_vals_blackbox_sit)
    _, p_sit_bbrun = sp.stats.mannwhitneyu(all_roi_vals_sit,all_roi_vals_blackbox_run)
    _, p_run_bbsit = sp.stats.mannwhitneyu(all_roi_vals_run,all_roi_vals_blackbox_sit)
    _, p_run_bbrun = sp.stats.mannwhitneyu(all_roi_vals_run,all_roi_vals_blackbox_run)
    _, p_vr_bbsit = sp.stats.mannwhitneyu(all_roi_vals_vr,all_roi_vals_blackbox_sit)
    _, p_vr_bbrun = sp.stats.mannwhitneyu(all_roi_vals_vr,all_roi_vals_blackbox_run)
    _, p_bbsit_bbrun = sp.stats.mannwhitneyu(all_roi_vals_blackbox_sit,all_roi_vals_blackbox_run)

    p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_sit_run,p_sit_vr,p_run_vr,p_sit_bbsit,p_sit_bbrun,p_run_bbsit,p_run_bbrun,p_vr_bbsit,p_vr_bbrun,p_bbsit_bbrun],alpha=0.05,method='bonferroni')
    # print(p_sit_run,p_sit_vr,p_run_vr,p_sit_bbsit,p_sit_bbrun,p_run_bbsit,p_run_bbrun,p_vr_bbsit,p_vr_bbrun,p_bbsit_bbrun)
    # p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_sit_run,p_sit_vr,p_run_vr],alpha=0.05,method='bonferroni')
    # print(p_sit_run,p_sit_vr,p_run_vr)
    print('CORRECTED P-VALUES:')
    print('passive sit vs. passive run:   ' + str(p_corrected[1][0]))
    print('passive sit vs. vr:            ' + str(p_corrected[1][1]))
    print('passive run vs. vr:            ' + str(p_corrected[1][2]))
    print('passive sit vs. blackbox sit:  ' + str(p_corrected[1][3]))
    print('passive sit vs. blackbox run:  ' + str(p_corrected[1][4]))
    print('passive run vs. blackbox sit:  ' + str(p_corrected[1][5]))
    print('passive run vs. blackbox run:  ' + str(p_corrected[1][6]))
    print('vr vs. blackbox sit:           ' + str(p_corrected[1][7]))
    print('vr vs. blackbox run:           ' + str(p_corrected[1][8]))
    print('blackbox sit vs. blackbox run: ' + str(p_corrected[1][9]))

    # if p_corrected[1][0] < 0.05:
    #     ax_object3.text(0.34,3.6,np.round(p_corrected[1][0],4),fontsize=24)
    #     ax_object3.plot([0,1],[3.7,3.7],lw=3, c='k')

    # t,p = (sp.stats.ttest_rel(np.array(all_roi_vals_vr),np.array(all_roi_vals_sit)))
    # if p < 0.0001:
    #     ax_object4.text(0.34,5.6,'***',fontsize=24)
    #     ax_object4.plot([0,1],[5.7,5.7],lw=3, c='k')
    #
    # t,p = sp.stats.ttest_rel(np.array(all_roi_vals_vr),np.array(all_roi_vals_vr_run))
    # if p < 0.0001:
    #     ax_object5.text(0.34,5.6,'***',fontsize=24)
    #     ax_object5.plot([0,1],[5.7,5.7],lw=3, c='k')
    # t,p = sp.stats.ttest_rel(np.array(all_roi_vals_sit),np.array(all_roi_vals_run))
    # if p < 0.0001:
    #     ax_object2.text(0.34,5.6,'***',fontsize=24)
    #     ax_object2.plot([0,1],[5.7,5.7],lw=3, c='k')
    # ax_object1.scatter(all_roi_vals_blackbox_sit, all_roi_vals_blackbox_run, s=5,color='k')
    # t,p = sp.stats.ttest_rel(np.array(all_roi_vals_blackbox_sit),np.array(all_roi_vals_blackbox_run))
    # if p < 0.001:
    #     ax_object3.text(0.34,5.6,'**',fontsize=24)
    #     ax_object3.plot([0,1],[5.7,5.7],lw=3, c='k')

    # print(posthoc_res_ss)
    # print(p_svr)


if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    # list of roi parameter files ALL RSC
    roi_param_list = [
                      ['E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804.json','LF170613_1','Day20170804_openloop','Day20170804'],
                      ['E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2.json','LF171212_2','Day2018218_openloop_2','Day2018218_2'],
                      ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719.json','LF170421_2','Day20170719_openloop','Day20170719' ],
                      # ['E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720.json','LF170421_2','Day2017720_openloop','Day2017720'],
                      ['E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783.json','LF170420_1','Day201783_openloop','Day201783'],
                      ['E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776.json','LF170222_1','Day201776_openloop','Day201776']
                     ]

    # list of roi parameter files ALL V1
    # roi_param_list = [
    #                   ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day201777.json','LF170214_1','Day201777_openloop','Day201777'],
    #                   ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day2017714.json','LF170214_1','Day2017714_openloop','Day2017714'],
    #                   ['E:\\MTH3_figures\\LF171211_2\\LF171211_2_Day201852.json','LF171211_2','Day201852_openloop','Day201852'],
    #                   ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_openloop_1','Day2018424_1'],
    #                   ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_openloop_2','Day2018424_2']
    #                  ]

    fname      = 'summary figure SVR space'
    align_event = 'lmcenter'
    bin_type = 'space'
    trialtypes = ['short', 'long']
    subfolder  = []

    # create figure and axes to later plot on
    fig  = plt.figure(figsize=(16,8))
    ax2  = plt.subplot2grid((2,80),(0,00), rowspan=1, colspan=10)
    ax3  = plt.subplot2grid((2,80),(0,30), rowspan=1, colspan=10)
    ax4  = plt.subplot2grid((2,80),(0,10), rowspan=1, colspan=10)
    ax5  = plt.subplot2grid((2,80),(0,20), rowspan=1, colspan=10)
    ax8  = plt.subplot2grid((2,80),(0,40), rowspan=1, colspan=10)
    ax9  = plt.subplot2grid((2,80),(0,70), rowspan=1, colspan=10)
    ax10 = plt.subplot2grid((2,80),(0,50), rowspan=1, colspan=10)
    ax11 = plt.subplot2grid((2,80),(0,60), rowspan=1, colspan=10)
    ax1  = plt.subplot2grid((2,80),(1,00), rowspan=1, colspan=15)
    ax6  = plt.subplot2grid((2,80),(1,15), rowspan=1, colspan=15)
    ax7  = plt.subplot2grid((2,80),(1,30), rowspan=1, colspan=15)
    ax12 = plt.subplot2grid((2,80),(1,45), rowspan=1, colspan=15)
    ax13 = plt.subplot2grid((2,80),(1,60), rowspan=1, colspan=10)

    scatterplot_svr(roi_param_list, align_event, trialtypes, bin_type, ax1, ax2, ax4, ax5, ax3, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13)

    ax1.set_xlim([-0.2,4])
    ax1.set_ylim([-0.2,4])
    ax1.set_xlabel('response STATIONARY')
    ax1.set_ylabel('response RUNNING')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    # ax7.set_xlim([-1.1,1.1])
    # ax7.set_ylim([-1.1,1.1])
    ax7.set_xlabel('response STATIONARY (norm)')
    ax7.set_ylabel('response RUNNING (norm)')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.spines['bottom'].set_linewidth(1)
    ax7.spines['left'].set_linewidth(1)
    ax7.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax6.set_xlim([0,4.1])
    # ax6.set_ylim([-0.0,1])
    ax6.set_xticks([0,1,2,3,4])
    ax6.set_xticklabels(['no inputs','only visual','only motor', 'visual + motor','vr navigation'], rotation=45)
    # ax6.set_xlabel('response STATIONARY')
    ax6.set_ylabel('mean response (dF/F)')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['bottom'].set_linewidth(1)
    ax6.spines['left'].set_linewidth(1)
    ax6.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax12.set_xlim([-0.2,4.1])
    ax12.set_xticklabels(['no inputs','only motor','only visual', 'visual + motor','vr navigation'], rotation=45)
    ax12.set_xticks([0,1,2,3,4])
    if NORM_METHOD == 3:
        ax12.set_ylim([0.0,1.05])
        ax12.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax12.set_yticklabels(['0','20','40','60','80','100'], fontsize=12)
        ax12.set_ylabel('response amplitude (% of VR response)', fontsize=12)
    elif NORM_METHOD == 2:
        ax12.set_ylim([-1.1,1.1])
        ax12.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax12.set_yticklabels(['0','20','40','60','80','100'], fontsize=12)
        ax12.set_ylabel('response amplitude (% of VR response)', fontsize=12)


    # ax12.set_xlabel('response STATIONARY')

    ax12.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['bottom'].set_linewidth(1)
    ax12.spines['left'].set_linewidth(1)
    ax12.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax13.set_xlim([-0.5,1.5])
    ax13.set_ylim([-0.1,2.1] )
    # ax12.set_xticks([0,1,2,3,4])
    # ax12.set_xticklabels(['no inputs','only visual','only motor', 'visual + motor','vr navigation'], rotation=45)
    # ax13.set_xticks([0,1,2,3,4])
    # ax13.set_xticklabels(['no inputs','only motor','only visual', 'visual + motor','vr navigation'], rotation=45)
    # ax12.set_xlabel('response STATIONARY')
    ax13.set_yticks([0,0.5,1.0,1.5,2.0])
    ax13.set_yticklabels(['0','50','100','150','200'], fontsize=12)
    ax13.set_ylabel('response amplitude (% of VR response)', fontsize=12)
    ax13.spines['top'].set_visible(False)
    ax13.spines['right'].set_visible(False)
    ax13.spines['bottom'].set_visible(False)
    ax13.spines['left'].set_linewidth(1)
    ax13.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    min_y = -0.2
    max_y = 4
    y_ticks = [0,1,2,3,4]
    y_ticklabels = ['0','1','2','3','4']

    ax2.set_ylim([min_y,max_y])
    ax2.set_ylabel('response (dF/F)', fontsize=16)
    ax2.set_xticks([0,1])
    ax2.set_xticklabels(['passive - motor','passive + motor'], rotation=45)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_ticklabels)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_linewidth(1)

    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax3.set_ylim([min_y,max_y])
    ax3.set_ylabel('response (dF/F)', fontsize=16)
    ax3.set_xticks([0,1])
    ax3.set_xticklabels(['blackbox - motor','blackbox + motor'], rotation=45)
    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(y_ticklabels)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_linewidth(1)

    ax3.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax4.set_ylim([min_y,max_y])
    ax4.set_ylabel('response (dF/F)', fontsize=16)
    ax4.set_xticks([0,1])
    ax4.set_xticklabels(['VR','passive - motor'], rotation=45)
    ax4.set_yticks(y_ticks)
    ax4.set_yticklabels(y_ticklabels)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_linewidth(1)

    ax4.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax5.set_ylim([min_y,max_y])
    ax5.set_ylabel('response (dF/F)', fontsize=16)
    ax5.set_xticks([0,1])
    ax5.set_xticklabels(['VR','passive + motor'], rotation=45)
    ax5.set_yticks(y_ticks)
    ax5.set_yticklabels(y_ticklabels)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_linewidth(1)

    ax5.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    if NORM_METHOD == 1:
        min_y_norm = -0.5
        max_y_norm = 3
    elif NORM_METHOD == 2:
        min_y_norm = -1
        max_y_norm = 2
    elif NORM_METHOD == 3:
        min_y_norm = -3
        max_y_norm = 3


    ax9.set_ylim([min_y,max_y_norm])
    ax9.set_ylabel('response (norm)', fontsize=16)
    ax9.set_xticks([0,1])
    ax9.set_xticklabels(['blackbox - motor','blackbox + motor'], rotation=45)
    ax9.set_yticks([-1,0,1])
    ax9.set_yticklabels(['-1','0','1'])
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.spines['bottom'].set_linewidth(1)
    ax9.spines['left'].set_linewidth(1)

    ax9.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax8.set_ylim([min_y_norm,max_y_norm])
    ax8.set_ylabel('response (norm)', fontsize=16)
    ax8.set_xticks([0,1])
    ax8.set_xticklabels(['passive - motor','passive + motor'], rotation=45)
    ax8.set_yticks([-1,0,1])
    ax8.set_yticklabels(['-1','0','1'])
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['bottom'].set_linewidth(1)
    ax8.spines['left'].set_linewidth(1)

    ax8.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax10.set_ylim([min_y_norm,max_y_norm])
    ax10.set_ylabel('response (norm)', fontsize=16)
    ax10.set_xticks([0,1])
    ax10.set_xticklabels(['VR','passive - motor'], rotation=45)
    ax10.set_yticks([-1,0,1])
    ax10.set_yticklabels(['-1','0','1'])
    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.spines['bottom'].set_linewidth(1)
    ax10.spines['left'].set_linewidth(1)

    ax10.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax11.set_ylim([min_y_norm,max_y_norm])
    ax11.set_ylabel('response (norm)', fontsize=16)
    ax11.set_xticks([0,1])
    ax11.set_xticklabels(['VR','passive + motor'], rotation=45)
    ax11.set_yticks([-1,0,1])
    ax11.set_yticklabels(['-1','0','1'])
    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['bottom'].set_linewidth(1)
    ax11.spines['left'].set_linewidth(1)

    ax11.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=12, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')


    subfolder = 'svr'
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

    print(fname)
