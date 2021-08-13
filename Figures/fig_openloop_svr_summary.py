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
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

fformat = 'png'

def matching_vr_rois(roi_params, trialtype, align_event):
    """ retrieve peak responses of animals in VR """
    # run through all roi_param files and create empty dictionary lists that we can later append to
    result_max_peak = {}
    result_max_peak[align_event + '_peakval_' + trialtype] = []
    result_max_peak[align_event + '_peakval_ol_' + trialtype] = []
    result_max_peak[align_event + '_roi_number'] = []

    # grab a full list of roi numbers to later match it with rois provided in roilist
    roi_list_vr = roi_params['valid_rois']
    roi_list_vr_svr = roi_params[align_event + '_roi_number_svr_' + trialtype]
    # print(roi_list_vr)
    # print(roi_list_vr_svr)
    for roi in roi_list_vr_svr:
        result_max_peak[align_event + '_roi_number'].append(roi)
        roi_idx = np.argwhere(np.asarray(roi_list_vr) == roi)[0][0]
        result_max_peak[align_event + '_peakval_' + trialtype].append(roi_params[align_event + '_peak_' + trialtype][roi_idx])
        result_max_peak[align_event + '_peakval_ol_' + trialtype].append(roi_params[align_event + '_peak_' + trialtype + '_ol'][roi_idx])

    return result_max_peak

def scatterplot_svr(roi_param_list, align_event, trialtypes, ax_object1, ax_object2, ax_object4, ax_object5, ax_object3, ax_object6, ax_object7, ax_object8, ax_object9, ax_object10, ax_object11, ax_object12):
    """ plot peak activity of neurons during openloop (passive) condition when they are stationary vs when they are running """
    # collect all roi values plotted for statistical getExistingDirectory

    # all roi values of animals either sitting or running in OL
    all_rois_num = []
    all_roi_vals_sit = []
    all_roi_vals_run = []
    all_roi_vals_sit_norm = []
    all_roi_vals_run_norm = []
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
    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list):
        # print(rpl)
        # load roi parameters for given session
        with open(rpl,'r') as f:
            roi_params = json.load(f)
        # run through alignment points and trialtypes
        print(rpl)
        all_rois_sess = []
        for tl in trialtypes:
            vr_roisvals = matching_vr_rois(roi_params, tl, align_event)
            if tl == 'short':
                scatter_color = '0.8'
            elif tl == 'long':
                scatter_color = '0.5'
            # plot amplitudes for sit vs run
            # ax_object1.scatter(roi_params[align_event + '_peak_sit_' + tl], roi_params[align_event + '_peak_run_' + tl], s=5,color=scatter_color)
            # plot VR vs sit and VR vs run
            for i in range(len(vr_roisvals[align_event + '_peakval_' + tl])):
                if not np.isnan(vr_roisvals[align_event + '_peakval_' + tl][i]) and not np.isnan(roi_params[align_event + '_peak_sit_' + tl][i]) and not np.isnan(roi_params[align_event + '_peak_run_' + tl][i]):
                    ax_object2.plot([0,1], [roi_params[align_event + '_peak_sit_' + tl][i],roi_params[align_event + '_peak_run_' + tl][i]],c=scatter_color,lw=ind_lw,zorder=2)
                    ax_object4.plot([0,1], [vr_roisvals[align_event + '_peakval_' + tl][i],roi_params[align_event + '_peak_sit_' + tl][i]],c=scatter_color,lw=ind_lw,zorder=2)
                    ax_object5.plot([0,1], [vr_roisvals[align_event + '_peakval_' + tl][i],roi_params[align_event + '_peak_run_' + tl][i]],c=scatter_color,lw=ind_lw,zorder=2)

                    norm_val = vr_roisvals[align_event + '_peakval_' + tl][i]
                    ax_object8.plot([0,1], [roi_params[align_event + '_peak_sit_' + tl][i]/norm_val,roi_params[align_event + '_peak_run_' + tl][i]/norm_val],lw=ind_lw,c=scatter_color,zorder=2)
                    ax_object10.plot([0,1], [vr_roisvals[align_event + '_peakval_' + tl][i]/norm_val,roi_params[align_event + '_peak_sit_' + tl][i]/norm_val],lw=ind_lw,c=scatter_color,zorder=2)
                    ax_object11.plot([0,1], [vr_roisvals[align_event + '_peakval_' + tl][i]/norm_val,roi_params[align_event + '_peak_run_' + tl][i]/norm_val],lw=ind_lw,c=scatter_color,zorder=2)

                    all_roi_vals_sit.append(roi_params[align_event + '_peak_sit_' + tl][i])
                    all_roi_vals_run.append(roi_params[align_event + '_peak_run_' + tl][i])
                    all_roi_vals_vr.append(vr_roisvals[align_event + '_peakval_' + tl][i])

                    all_roi_vals_sit_norm.append(roi_params[align_event + '_peak_sit_' + tl][i]/norm_val)
                    all_roi_vals_run_norm.append(roi_params[align_event + '_peak_run_' + tl][i]/norm_val)
                    all_roi_vals_vr_norm.append(vr_roisvals[align_event + '_peakval_' + tl][i]/norm_val)
                    all_rois_sess.append(roi_params[align_event + '_roi_number_svr_' + tl][i])

            ax_object1.scatter(all_roi_vals_sit, all_roi_vals_run, s=5,color='0.65')
            ax_object7.scatter(all_roi_vals_sit_norm, all_roi_vals_run_norm, s=5,color='0.65')
        # plot blackbox activity
        for i in range(len(roi_params[align_event + '_peak_sit_blackbox'])):
            if not np.isnan(roi_params[align_event + '_peak_sit_blackbox'][i]) and not np.isnan(roi_params[align_event + '_peak_run_blackbox'][i]):
                # find matching roi and determine peak value (=normalizing value) in either short or long
                roi_list_vr = roi_params['valid_rois']
                roi_list_blackbox = roi_params[align_event + '_roi_number_blackbox']
                roi_idx = np.argwhere(np.asarray(roi_list_vr) == np.asarray(roi_list_blackbox)[i])[0][0]
                norm_val_short = roi_params[align_event + '_peak_short'][roi_idx]
                norm_val_long = roi_params[align_event + '_peak_long'][roi_idx]
                norm_val = np.amax([norm_val_short,norm_val_long])
                ax_object3.plot([0,1], [roi_params[align_event + '_peak_sit_blackbox'][i],roi_params[align_event + '_peak_run_blackbox'][i]],c='0.65',zorder=2)
                ax_object9.plot([0,1], [roi_params[align_event + '_peak_sit_blackbox'][i]/norm_val,roi_params[align_event + '_peak_run_blackbox'][i]/norm_val],c='0.65',zorder=2)
                all_roi_vals_blackbox_sit.append(roi_params[align_event + '_peak_sit_blackbox'][i])
                all_roi_vals_blackbox_run.append(roi_params[align_event + '_peak_run_blackbox'][i])
                all_roi_vals_blackbox_sit_norm.append(roi_params[align_event + '_peak_sit_blackbox'][i]/norm_val)
                all_roi_vals_blackbox_run_norm.append(roi_params[align_event + '_peak_run_blackbox'][i]/norm_val)

        # count how many rois from each animal are plotted (only count unique roi numbers so short and long responding rois are not counted double)
        all_rois_num.append(np.unique(np.array(all_rois_sess)).shape[0])

    # BLACKBOX responses
    ax_object3.scatter([0],[np.mean(np.array(all_roi_vals_blackbox_sit))],s=25,marker='s',facecolor='w',linewidths=2,color='k',zorder=4)
    ax_object3.scatter([1],[np.mean(np.array(all_roi_vals_blackbox_run))],s=25,marker='s',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    ax_object3.plot([0,1],[np.mean(np.array(all_roi_vals_blackbox_sit)), np.mean(np.array(all_roi_vals_blackbox_run))], lw=3, c='k', zorder=3)
    ax_object3.errorbar([0,1],[np.mean(np.array(all_roi_vals_blackbox_sit)), np.mean(np.array(all_roi_vals_blackbox_run))],yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit)),sp.stats.sem(np.array(all_roi_vals_blackbox_run))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # BLACKBOX responses NORMALIZED
    ax_object9.scatter([0],[np.mean(np.array(all_roi_vals_blackbox_sit_norm))],s=25,marker='s',facecolor='w',linewidths=2,color='k',zorder=4)
    ax_object9.scatter([1],[np.mean(np.array(all_roi_vals_blackbox_run_norm))],s=25,marker='s',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    ax_object9.plot([0,1],[np.mean(np.array(all_roi_vals_blackbox_sit_norm)), np.mean(np.array(all_roi_vals_blackbox_run_norm))], lw=3, c='k', zorder=3)
    ax_object9.errorbar([0,1],[np.mean(np.array(all_roi_vals_blackbox_sit_norm)), np.mean(np.array(all_roi_vals_blackbox_run_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit_norm)),sp.stats.sem(np.array(all_roi_vals_blackbox_run_norm))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # OL - motor vs OL + motor
    ax_object2.scatter([0],[np.mean(np.array(all_roi_vals_sit))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    ax_object2.scatter([1],[np.mean(np.array(all_roi_vals_run))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    ax_object2.plot([0,1],[np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_run))], lw=3, c='k', zorder=3)
    ax_object2.errorbar([0,1],[np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_run))],yerr=[sp.stats.sem(np.array(all_roi_vals_sit)),sp.stats.sem(np.array(all_roi_vals_run))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # OL - motor vs OL + motor NORMALIZED
    ax_object8.scatter([0],[np.mean(np.array(all_roi_vals_sit_norm))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    ax_object8.scatter([1],[np.mean(np.array(all_roi_vals_run_norm))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    ax_object8.plot([0,1],[np.mean(np.array(all_roi_vals_sit_norm)), np.mean(np.array(all_roi_vals_run_norm))], lw=3, c='k', zorder=3)
    ax_object8.errorbar([0,1],[np.mean(np.array(all_roi_vals_sit_norm)), np.mean(np.array(all_roi_vals_run_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_sit_norm)),sp.stats.sem(np.array(all_roi_vals_run_norm))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # VR vs OL - motor
    ax_object4.scatter([0],[np.mean(np.array(all_roi_vals_vr))],s=40,marker='s',linewidths=0,c='k',zorder=4)
    ax_object4.scatter([1],[np.mean(np.array(all_roi_vals_sit))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    ax_object4.plot([0,1],[np.mean(np.array(all_roi_vals_vr)), np.mean(np.array(all_roi_vals_sit))], lw=3, c='k', zorder=3)
    ax_object4.errorbar([0,1],[np.mean(np.array(all_roi_vals_vr)), np.mean(np.array(all_roi_vals_sit))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr)),sp.stats.sem(np.array(all_roi_vals_sit))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # VR vs OL - motor NORMALIZED
    ax_object10.scatter([0],[np.mean(np.array(all_roi_vals_vr_norm))],s=40,marker='s',linewidths=0,color='k',zorder=4)
    ax_object10.scatter([1],[np.mean(np.array(all_roi_vals_sit_norm))],s=40,marker='s',linewidths=0,c='#3953A3',zorder=4)
    ax_object10.plot([0,1],[np.mean(np.array(all_roi_vals_vr_norm)), np.mean(np.array(all_roi_vals_sit_norm))], lw=3, c='k', zorder=3)
    ax_object10.errorbar([0,1],[np.mean(np.array(all_roi_vals_vr_norm)), np.mean(np.array(all_roi_vals_sit_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr_norm)),sp.stats.sem(np.array(all_roi_vals_sit_norm))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # VR vs OL + motor
    ax_object5.scatter([0],[np.mean(np.array(all_roi_vals_vr))],s=40,marker='s',linewidths=0,color='k',zorder=4)
    ax_object5.scatter([1],[np.mean(np.array(all_roi_vals_run))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    ax_object5.plot([0,1],[np.mean(np.array(all_roi_vals_vr)), np.mean(np.array(all_roi_vals_run))], lw=3, c='k', zorder=3)
    ax_object5.errorbar([0,1],[np.mean(np.array(all_roi_vals_vr)), np.mean(np.array(all_roi_vals_run))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr)),sp.stats.sem(np.array(all_roi_vals_run))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # VR vs OL + motor NORMALIZED
    ax_object11.scatter([0],[np.mean(np.array(all_roi_vals_vr_norm))],s=40,marker='s',linewidths=0,color='k',zorder=4)
    ax_object11.scatter([1],[np.mean(np.array(all_roi_vals_run_norm))],s=40,marker='s',linewidths=0,c='#EC2024',zorder=4)
    ax_object11.plot([0,1],[np.mean(np.array(all_roi_vals_vr_norm)), np.mean(np.array(all_roi_vals_run_norm))], lw=3, c='k', zorder=3)
    ax_object11.errorbar([0,1],[np.mean(np.array(all_roi_vals_vr_norm)), np.mean(np.array(all_roi_vals_run_norm))],yerr=[sp.stats.sem(np.array(all_roi_vals_vr_norm)),sp.stats.sem(np.array(all_roi_vals_run_norm))],elinewidth=3,capsize=0,capthick=0,zorder=2,linewidth=0,c='k',lw=5,ls='--')

    # ax_object6.plot([0,1,2,3], [np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_blackbox_run)), np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_vr))])
    # ax_object6.plot([0,1,2,3,4], [np.mean(np.array(all_roi_vals_blackbox_sit)), np.mean(np.array(all_roi_vals_blackbox_run)), np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_vr))],lw=1,c='k',zorder=3)
    ax_object6.plot([0,1,2,3,4], [np.mean(np.array(all_roi_vals_blackbox_sit)),np.mean(np.array(all_roi_vals_blackbox_run)), np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_vr))],lw=1,c='k',zorder=3)
    ax_object6.scatter([0], [np.mean(np.array(all_roi_vals_blackbox_sit))],s=25,marker='s',facecolor='w',linewidths=2,color='k',zorder=4)
    ax_object6.scatter([1], [np.mean(np.array(all_roi_vals_blackbox_run))],s=25,marker='s',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    ax_object6.scatter([2], [np.mean(np.array(all_roi_vals_sit))],s=50,marker='s',linewidths=0,c=['#3953A3'],zorder=4)
    ax_object6.scatter([3], [np.mean(np.array(all_roi_vals_run))],s=50,marker='s',linewidths=0,c=['#EC2024'],zorder=4)
    ax_object6.scatter([4], [np.mean(np.array(all_roi_vals_vr))],s=50,marker='s',linewidths=0,c=['k'],zorder=4)

    ax_object6.errorbar([0,1,2,3,4], [np.mean(np.array(all_roi_vals_blackbox_sit)), np.mean(np.array(all_roi_vals_blackbox_run)), np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_vr))], \
                        yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit)), sp.stats.sem(np.array(all_roi_vals_blackbox_run)), sp.stats.sem(np.array(all_roi_vals_sit)), sp.stats.sem(np.array(all_roi_vals_run)), sp.stats.sem(np.array(all_roi_vals_vr))], \
                        fmt='', c='k')

    ax_object12.plot([0,1,2,3,4], [np.mean(np.array(all_roi_vals_blackbox_sit_norm)), np.mean(np.array(all_roi_vals_blackbox_run_norm)), np.mean(np.array(all_roi_vals_sit_norm)), np.mean(np.array(all_roi_vals_run_norm)), np.mean(np.array(all_roi_vals_vr_norm))],lw=1,c='k',zorder=3)
    # ax_object12.scatter([0], [np.mean(np.array(all_roi_vals_blackbox_sit_norm))],s=25,marker='s',facecolor='w',linewidths=2,color='k',zorder=4)
    ax_object12.scatter([1], [np.mean(np.array(all_roi_vals_blackbox_run_norm))],s=25,marker='s',facecolor='#EC2024',linewidths=2,color='k',zorder=4)
    ax_object12.scatter([2], [np.mean(np.array(all_roi_vals_sit_norm))],s=50,marker='s',linewidths=0,c=['#3953A3'],zorder=4)
    ax_object12.scatter([3], [np.mean(np.array(all_roi_vals_run_norm))],s=50,marker='s',linewidths=0,c=['#EC2024'],zorder=4)
    ax_object12.scatter([4], [np.mean(np.array(all_roi_vals_vr_norm))],s=50,marker='s',linewidths=0,c=['k'],zorder=4)

    ax_object12.errorbar([0,1,2,3,4], [np.mean(np.array(all_roi_vals_blackbox_sit_norm)), np.mean(np.array(all_roi_vals_blackbox_run_norm)), np.mean(np.array(all_roi_vals_sit_norm)), np.mean(np.array(all_roi_vals_run_norm)), np.mean(np.array(all_roi_vals_vr_norm))], \
                        yerr=[sp.stats.sem(np.array(all_roi_vals_blackbox_sit_norm)), sp.stats.sem(np.array(all_roi_vals_blackbox_run_norm)), sp.stats.sem(np.array(all_roi_vals_sit_norm)), sp.stats.sem(np.array(all_roi_vals_run_norm)), sp.stats.sem(np.array(all_roi_vals_vr_norm))], \
                        fmt='o', c='k')

    # print(np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_run_norm)))
    #
    # print([np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_blackbox_run)), np.mean(np.array(all_roi_vals_run)), np.mean(np.array(all_roi_vals_vr))])
    print(np.sum(all_rois_num))
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress([0,1,2], [np.mean(np.array(all_roi_vals_blackbox_sit)),np.mean(np.array(all_roi_vals_sit)), np.mean(np.array(all_roi_vals_blackbox_run))])
    # ax_object6.plot(np.array([0,1,2,3,4]), intercept + slope*np.array([0,1,2,3,4]), '0.5', label='fitted line', ls='--', lw=2, zorder=2)
    # ax_object6.scatter([2], [np.mean(np.array(all_roi_vals_sit_norm))+np.mean(np.array(all_roi_vals_blackbox_run_norm))],s=50,marker='s',linewidths=0,c=['k'],zorder=4)

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress([0,1,2], [np.mean(np.array(all_roi_vals_blackbox_sit_norm)), np.mean(np.array(all_roi_vals_blackbox_run_norm)),np.mean(np.array(all_roi_vals_sit_norm))])
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
    print(p_sit_run,p_sit_vr,p_run_vr,p_sit_bbsit,p_sit_bbrun,p_run_bbsit,p_run_bbrun,p_vr_bbsit,p_vr_bbrun,p_bbsit_bbrun)
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
                      'E:\\MTH3_figures\\LF170613_1\\LF170613_1_Day20170804.json',
                      'E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day20170719.json',
                      'E:\\MTH3_figures\\LF170421_2\\LF170421_2_Day2017720.json',
                      'E:\\MTH3_figures\\LF170420_1\\LF170420_1_Day201783.json',
                      'E:\\MTH3_figures\\LF170222_1\\LF170222_1_Day201776.json',
                      'E:\\MTH3_figures\\LF171212_2\\LF171212_2_Day2018218_2.json'
                     ]

    # list of roi parameter files ALL V1
    # roi_param_list = [
    #                   'E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day201777.json',
    #                   'E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day2017714.json',
    #                   'E:\\MTH3_figures\\LF171211_2\\LF171211_2_Day201852.json',
    #                   'E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_1.json',
    #                   'E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_2.json'
    #                  ]

    fname      = 'summary figure SVR'
    align_event = 'lmcenter'
    trialtypes = ['short', 'long']
    subfolder  = []

    # create figure and axes to later plot on
    fig = plt.figure(figsize=(16,8))
    ax1 = plt.subplot2grid((2,80),(1,00), rowspan=1, colspan=20)
    ax2 = plt.subplot2grid((2,80),(0,00), rowspan=1, colspan=10)
    ax3 = plt.subplot2grid((2,80),(0,30), rowspan=1, colspan=10)
    ax4 = plt.subplot2grid((2,80),(0,10), rowspan=1, colspan=10)
    ax5 = plt.subplot2grid((2,80),(0,20), rowspan=1, colspan=10)
    ax6 = plt.subplot2grid((2,80),(1,20), rowspan=1, colspan=20)
    ax7 = plt.subplot2grid((2,80),(1,40), rowspan=1, colspan=20)
    ax8 = plt.subplot2grid((2,80),(0,40), rowspan=1, colspan=10)
    ax9 = plt.subplot2grid((2,80),(0,70), rowspan=1, colspan=10)
    ax10 = plt.subplot2grid((2,80),(0,50), rowspan=1, colspan=10)
    ax11 = plt.subplot2grid((2,80),(0,60), rowspan=1, colspan=10)
    ax12 = plt.subplot2grid((2,80),(1,60), rowspan=1, colspan=20)

    scatterplot_svr(roi_param_list, align_event, trialtypes, ax1, ax2, ax4, ax5, ax3, ax6, ax7, ax8, ax9, ax10, ax11, ax12)

    max_y = 6

    ax1.set_xlim([-0.2,max_y])
    ax1.set_ylim([-0.2,max_y])
    ax1.set_xlabel('response STATIONARY')
    ax1.set_ylabel('response RUNNING')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax7.set_xlim([-0.2,max_y])
    ax7.set_ylim([-0.2,max_y])
    ax7.set_xlabel('response STATIONARY (norm)')
    ax7.set_ylabel('response RUNNING (norm)')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.spines['bottom'].set_linewidth(2)
    ax7.spines['left'].set_linewidth(2)
    ax7.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax6.set_xlim([-0.2,4.1])
    ax6.set_ylim([-0.12,1])
    ax6.set_xticks([0,1,2,3,4])
    ax6.set_xticklabels(['no inputs','only visual','only motor', 'visual + motor','vr navigation'], rotation=45)
    # ax6.set_xlabel('response STATIONARY')
    ax6.set_ylabel('mean response (dF/F)')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['bottom'].set_linewidth(2)
    ax6.spines['left'].set_linewidth(2)
    ax6.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax12.set_xlim([-0.2,4.1])
    ax12.set_ylim([-0.15,1.1] )
    ax12.set_xticks([0,1,2,3,4])
    ax12.set_xticklabels(['no inputs','only visual','only motor', 'visual + motor','vr navigation'], rotation=45)
    # ax12.set_xlabel('response STATIONARY')
    ax12.set_ylabel('mean response (norm)')
    ax12.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['bottom'].set_linewidth(2)
    ax12.spines['left'].set_linewidth(2)
    ax12.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax2.set_ylim([-0.5,max_y])
    ax2.set_ylabel('response (dF/F)', fontsize=16)
    ax2.set_xticks([0,1])
    ax2.set_xticklabels(['passive - motor','passive + motor'], rotation=45)
    ax2.set_yticks([0,1,2,3,4])
    ax2.set_yticklabels(['0','1','2','3','4'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)

    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax3.set_ylim([-0.5,max_y])
    ax3.set_ylabel('response (dF/F)', fontsize=16)
    ax3.set_xticks([0,1])
    ax3.set_xticklabels(['blackbox - motor','blackbox + motor'], rotation=45)
    ax3.set_yticks([0,1,2,3,4])
    ax3.set_yticklabels(['0','1','2','3','4'])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_linewidth(2)
    ax3.spines['left'].set_linewidth(2)

    ax3.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax4.set_ylim([-0.5,max_y])
    ax4.set_ylabel('response (dF/F)', fontsize=16)
    ax4.set_xticks([0,1])
    ax4.set_xticklabels(['VR','passive - motor'], rotation=45)
    ax4.set_yticks([0,1,2,3,4])
    ax4.set_yticklabels(['0','1','2','3','4'])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_linewidth(2)
    ax4.spines['left'].set_linewidth(2)

    ax4.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax5.set_ylim([-0.5,max_y])
    ax5.set_ylabel('response (dF/F)', fontsize=16)
    ax5.set_xticks([0,1])
    ax5.set_xticklabels(['VR','passive + motor'], rotation=45)
    ax5.set_yticks([0,1,2,3,4])
    ax5.set_yticklabels(['0','1','2','3','4'])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_linewidth(2)
    ax5.spines['left'].set_linewidth(2)

    ax5.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    max_y_norm = 3

    ax9.set_ylim([-0.5,max_y_norm])
    ax9.set_ylabel('response (norm)', fontsize=16)
    ax9.set_xticks([0,1])
    ax9.set_xticklabels(['blackbox - motor','blackbox + motor'], rotation=45)
    ax9.set_yticks([0,1,2,3])
    ax9.set_yticklabels(['0','1','2','3'])
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.spines['bottom'].set_linewidth(2)
    ax9.spines['left'].set_linewidth(2)

    ax9.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax8.set_ylim([-0.5,max_y_norm])
    ax8.set_ylabel('response (norm)', fontsize=16)
    ax8.set_xticks([0,1])
    ax8.set_xticklabels(['passive - motor','passive + motor'], rotation=45)
    ax8.set_yticks([0,1,2,3])
    ax8.set_yticklabels(['0','1','2','3'])
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['bottom'].set_linewidth(2)
    ax8.spines['left'].set_linewidth(2)

    ax8.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax10.set_ylim([-0.5,max_y_norm])
    ax10.set_ylabel('response (norm)', fontsize=16)
    ax10.set_xticks([0,1])
    ax10.set_xticklabels(['VR','passive - motor'], rotation=45)
    ax10.set_yticks([0,1,2,3])
    ax10.set_yticklabels(['0','1','2','3'])
    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.spines['bottom'].set_linewidth(2)
    ax10.spines['left'].set_linewidth(2)

    ax10.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax11.set_ylim([-0.5,max_y_norm])
    ax11.set_ylabel('response (norm)', fontsize=16)
    ax11.set_xticks([0,1])
    ax11.set_xticklabels(['VR','passive + motor'], rotation=45)
    ax11.set_yticks([0,1,2,3])
    ax11.set_yticklabels(['0','1','2','3'])
    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['bottom'].set_linewidth(2)
    ax11.spines['left'].set_linewidth(2)

    ax11.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
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
