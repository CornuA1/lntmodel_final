"""
Plot differences between layer 2/3 and layer 5 neurons

@author: Lukas Fischer

"""

import h5py, os, sys, traceback, matplotlib, warnings, json, yaml
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

plt.rcParams['svg.fonttype'] = 'none'
warnings.filterwarnings('ignore')
sns.set_style("white")

fformat = 'svg'

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

def FHWM_properties(roi_param_list_l23, roi_param_list_l5, fname):
    """ plot transient paramaters of neuron population """
    # create figure and axes to later plot on
    fig = plt.figure(figsize=(7.8,4))
    ax1 = plt.subplot(251)
    ax2 = plt.subplot(252)
    ax3 = plt.subplot(253)
    ax4 = plt.subplot(254)
    ax5 = plt.subplot(256)
    ax6 = plt.subplot(257)
    ax7 = plt.subplot(258)
    ax8 = plt.subplot(259)
    ax9 = plt.subplot(255)
    ax10 = plt.subplot(2,5,10)

    plot_color_l23 = '#0096B9'
    plot_color_l5 = '#FF00FF'
    hist_binsize = 0.4
    cum_histo_linewidth = 2
    # L23
    # load all fwhm mean values

    fwhm_all_l23 = []
    peak_all_l23 = []
    transient_rate_all_l23 = []
    transient_peaks_all_l23 = []
    auc_all_l23 = []
    roi_n = {}
    for rpl in roi_param_list_l23:
        with open(rpl,'r') as f:
            roi_params = json.load(f)
        valid_roi_idx = np.array(roi_params['valid_rois'])
        print(rpl)
        # print(np.array(roi_params['FWHM_mean'])[valid_roi_idx].shape)
        fwhm_all_l23.append(np.array(roi_params['FWHM_mean'])[valid_roi_idx])
        peak_all_l23.append(np.array(roi_params['norm_value_all']))
        transient_rate_all_l23.append(np.array(roi_params['transient_rate'])[valid_roi_idx])
        auc_all_l23.append(np.array(roi_params['transient_AUC_mean'])[valid_roi_idx])
        print('l23 ' + str(len(valid_roi_idx)))
        roi_n['l23 '+rpl.split(os.sep)[-2]] = str(len(valid_roi_idx))

    # flatten lists
    fwhm_all_l23 = [item for sublist in fwhm_all_l23 for item in sublist]
    fwhm_all_l23 = np.array(fwhm_all_l23)
    fwhm_all_l23 = fwhm_all_l23[~np.isnan(fwhm_all_l23)]
    sns.distplot(fwhm_all_l23, bins=np.arange(0,10.2,hist_binsize), kde=False, norm_hist=True, color=plot_color_l23, ax=ax1, hist_kws={"alpha":0.6, "linewidth": 0,"zorder":3}, label='L2/3')
    # ax1.axvline(np.median(fwhm_all_l23), c=plot_color_l23,ls='--',lw=1)
    sns.distplot(fwhm_all_l23, bins=np.arange(0,10,0.001), kde=False, norm_hist=True, color=plot_color_l23, ax=ax5, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    peak_all_l23 = [item for sublist in peak_all_l23 for item in sublist]
    peak_all_l23 = np.array(peak_all_l23)
    peak_all_l23 = peak_all_l23[~np.isnan(peak_all_l23)]
    sns.distplot(peak_all_l23, bins=np.arange(0,10.5,hist_binsize), kde=False, norm_hist=True, color=plot_color_l23, ax=ax2, hist_kws={"alpha":0.6, "linewidth": 0,"zorder":3})
    # ax2.axvline(np.median(peak_all_l23), c=plot_color_l23,ls='--',lw=1)
    sns.distplot(peak_all_l23, bins=np.arange(0,10,0.001), kde=False, norm_hist=True, color=plot_color_l23, ax=ax6, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    transient_rate_all_l23 = [item for sublist in transient_rate_all_l23 for item in sublist]
    transient_rate_all_l23 = np.array(transient_rate_all_l23)
    transient_rate_all_l23 = transient_rate_all_l23[~np.isnan(transient_rate_all_l23)]
    sns.distplot(transient_rate_all_l23, bins=np.arange(0,7.2,hist_binsize), kde=False, norm_hist=True, color=plot_color_l23, ax=ax3, hist_kws={"alpha":0.6, "linewidth": 0,"zorder":3})
    # ax3.axvline(np.median(transient_rate_all_l23), c=plot_color_l23,ls='--',lw=1)
    sns.distplot(transient_rate_all_l23, bins=np.arange(0,7,0.001), kde=False, norm_hist=True, color=plot_color_l23, ax=ax7, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    auc_all_l23 = [item for sublist in auc_all_l23 for item in sublist]
    auc_all_l23 = np.array(auc_all_l23)
    auc_all_l23 = auc_all_l23[~np.isnan(auc_all_l23)]
    sns.distplot(auc_all_l23, bins=np.arange(0,7.2,hist_binsize), kde=False, norm_hist=True, color=plot_color_l23, ax=ax9, hist_kws={"alpha":0.6, "linewidth": 0,"zorder":3})
    # ax9.axvline(np.median(auc_all_l23), c=plot_color_l23,ls='--',lw=1)
    sns.distplot(auc_all_l23, bins=np.arange(0,7,0.001), kde=False, norm_hist=True, color=plot_color_l23, ax=ax10, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    # load all fwhm mean values
    fwhm_all_l5 = []
    peak_all_l5 = []
    transient_rate_all_l5 = []
    transient_peaks_all_l5 = []
    auc_all_l5 = []
    for rpl in roi_param_list_l5:
        with open(rpl,'r') as f:
            roi_params = json.load(f)
        valid_roi_idx = np.array(roi_params['valid_rois'])
        # print(np.array(roi_params['FWHM_mean'])[valid_roi_idx].shape)
        fwhm_all_l5.append(np.array(roi_params['FWHM_mean'])[valid_roi_idx])
        peak_all_l5.append(np.array(roi_params['norm_value_all'])[valid_roi_idx])
        transient_rate_all_l5.append(np.array(roi_params['transient_rate'])[valid_roi_idx])
        auc_all_l5.append(np.array(roi_params['transient_AUC_mean'])[valid_roi_idx])
        print('l5 ' + str(len(valid_roi_idx)))
        roi_n['l5 '+rpl.split(os.sep)[-2]] = str(len(valid_roi_idx))

    # flatten lists
    fwhm_all_l5 = [item for sublist in fwhm_all_l5 for item in sublist]
    fwhm_all_l5 = np.array(fwhm_all_l5)
    fwhm_all_l5 = fwhm_all_l5[~np.isnan(fwhm_all_l5)]
    sns.distplot(fwhm_all_l5, bins=np.arange(0,10.2,hist_binsize), kde=False, norm_hist=True, color=plot_color_l5, ax=ax1, hist_kws={"alpha":1, "linewidth": 0,"zorder":2}, label='L5')
    # ax1.axvline(np.median(fwhm_all_l5), c=plot_color_l5,ls='--',lw=1)
    sns.distplot(fwhm_all_l5, bins=np.arange(0,10,0.001), kde=False, norm_hist=True, color=plot_color_l5, ax=ax5, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})
    ax1.legend(prop={'size': 6})

    peak_all_l5 = [item for sublist in peak_all_l5 for item in sublist]
    peak_all_l5 = np.array(peak_all_l5)
    peak_all_l5 = peak_all_l5[~np.isnan(peak_all_l5)]
    sns.distplot(peak_all_l5, bins=np.arange(0,10.5,hist_binsize), kde=False, norm_hist=True, color=plot_color_l5, ax=ax2, hist_kws={"alpha":1, "linewidth": 0,"zorder":2})
    # ax2.axvline(np.median(peak_all_l5), c=plot_color_l5,ls='--',lw=1)
    sns.distplot(peak_all_l5, bins=np.arange(0,10,0.001), kde=False, norm_hist=True, color=plot_color_l5, ax=ax6, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    transient_rate_all_l5 = [item for sublist in transient_rate_all_l5 for item in sublist]
    transient_rate_all_l5 = np.array(transient_rate_all_l5)
    transient_rate_all_l5 = transient_rate_all_l5[~np.isnan(transient_rate_all_l5)]
    sns.distplot(transient_rate_all_l5, bins=np.arange(0,7.2,hist_binsize), kde=False, norm_hist=True, color=plot_color_l5, ax=ax3, hist_kws={"alpha":1, "linewidth": 0,"zorder":2})
    # ax3.axvline(np.median(transient_rate_all_l5), c=plot_color_l5,ls='--',lw=1)
    sns.distplot(transient_rate_all_l5, bins=np.arange(0,7,0.001), norm_hist=True, kde=False, color=plot_color_l5, ax=ax7, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    auc_all_l5 = [item for sublist in auc_all_l5 for item in sublist]
    auc_all_l5 = np.array(auc_all_l5)
    auc_all_l5 = auc_all_l5[~np.isnan(auc_all_l5)]
    sns.distplot(auc_all_l5, bins=np.arange(0,7.2,hist_binsize), kde=False, norm_hist=True, color=plot_color_l5, ax=ax9, hist_kws={"alpha":1, "linewidth": 0,"zorder":2})
    # ax9.axvline(np.median(auc_all_l5), c=plot_color_l5,ls='--',lw=1)
    sns.distplot(auc_all_l5, bins=np.arange(0,7,0.001), kde=False, norm_hist=True, color=plot_color_l5, ax=ax10, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    l23_QC = []
    l5_QC = []

    for l23 in roi_param_list_l23:
        print(l23.split(os.sep)[-2])
        l23_QC.append(plot_peak_dist(l23, l23.split(os.sep)[-2]))

    for l5 in roi_param_list_l5:
        print(l5.split(os.sep)[-2])
        l5_QC.append(plot_peak_dist(l5, l5.split(os.sep)[-2]))

    l23_QC = [item for sublist in l23_QC for item in sublist]
    l5_QC = [item for sublist in l5_QC for item in sublist]
    l23_QC = np.array(l23_QC)
    l5_QC = np.array(l5_QC)

    sns.distplot(l23_QC, bins=np.arange(0,1.1,0.08), kde=False, norm_hist=True, color=plot_color_l23, ax=ax4, hist_kws={"alpha":0.6, "linewidth": 0,"zorder":3})
    sns.distplot(l5_QC, bins=np.arange(0,1.1,0.08), kde=False, norm_hist=True, color=plot_color_l5, ax=ax4, hist_kws={"alpha":1, "linewidth": 0,"zorder":2})

    sns.distplot(l23_QC, bins=np.arange(0,1,0.001), kde=False, norm_hist=True, color=plot_color_l23, ax=ax8, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})
    sns.distplot(l5_QC, bins=np.arange(0,1,0.001), kde=False, norm_hist=True, color=plot_color_l5, ax=ax8, hist_kws={"histtype":"step","linewidth":cum_histo_linewidth,"cumulative":True,"alpha":1})

    # ax4.axvline(np.mean(l23_QC), c=plot_color_l23,ls='--',lw=1)
    # ax4.axvline(np.mean(l5_QC), c=plot_color_l5,ls='--', lw=1)
    # ax4.set_title('transient size variability')

    print(roi_n)

    print('number L2/3 neurons: ' + str(len(fwhm_all_l23)))
    print('number L5 neurons: ' + str(len(fwhm_all_l5)))

    print(sp.stats.ttest_ind(fwhm_all_l23, fwhm_all_l5))
    print(sp.stats.ttest_ind(peak_all_l23, peak_all_l5))
    print(sp.stats.ttest_ind(transient_rate_all_l23, transient_rate_all_l5))
    print(sp.stats.ttest_ind(l23_QC, l5_QC))
    print(sp.stats.ttest_ind(auc_all_l23, auc_all_l5))

    # carry out statistical analysis. This is not (yet) the correct test: we are treating each group independently, rather than taking into account within-group and between-group variance
    print(sp.stats.f_oneway(np.array(fwhm_all_l23),np.array(fwhm_all_l5),np.array(peak_all_l23),np.array(peak_all_l5),np.array(transient_rate_all_l23),np.array(transient_rate_all_l5),np.array(l23_QC),np.array(l5_QC),np.array(auc_all_l23),np.array(auc_all_l5)))
    group_labels = ['fwhm_l23'] * np.array(fwhm_all_l23).shape[0] + \
                   ['fwhm_all_l5'] * np.array(fwhm_all_l5).shape[0] + \
                   ['peak_all_l23'] * np.array(peak_all_l23).shape[0] + \
                   ['peak_all_l5'] * np.array(peak_all_l5).shape[0] + \
                   ['transient_rate_all_l23'] * np.array(transient_rate_all_l23).shape[0] + \
                   ['transient_rate_all_l5'] * np.array(transient_rate_all_l5).shape[0] + \
                   ['l23_QC'] * np.array(l23_QC).shape[0] + \
                   ['l5_QC'] * np.array(l5_QC).shape[0] + \
                   ['auc_all_l23'] * np.array(auc_all_l23).shape[0] + \
                   ['auc_all_l5'] * np.array(auc_all_l5).shape[0]
    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.array(fwhm_all_l23),np.array(fwhm_all_l5),np.array(peak_all_l23),np.array(peak_all_l5),np.array(transient_rate_all_l23),np.array(transient_rate_all_l5),np.array(l23_QC),np.array(l5_QC),np.array(auc_all_l23),np.array(auc_all_l5))),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)

    #
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        bottom='on', \
        right='off', \
        top='off')

    ax1.set_xlim([0,10])
    ax1.set_xticks([0,2,4,6,8,10])
    ax1.set_xticklabels(['0','2','4','6','8','10'], fontsize=6)
    ax1.set_xlabel('FWHM width (sec)', fontsize=6)
    # ax1.set_ylabel('fraction of neurons', fontsize=40)
    ax1.set_yticks([0,0.35, 0.7])
    ax1.set_yticklabels(['0','0.35','0.7'], fontsize=6)

    ax2.spines['left'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        bottom='on', \
        right='off', \
        top='off')

    ax2.set_xlim([0,10])
    ax2.set_xticks([0,2,4,6,8,10])
    ax2.set_xticklabels(['0','2','4','6','8','10'], fontsize=6)
    ax2.set_xlabel('transients/min', fontsize=6)
    # ax2.set_ylabel('fraction of neurons', fontsize=48)
    ax2.set_yticks([0,0.35,0.7])
    ax2.set_yticklabels(['0','0.35','0.7'], fontsize=6)

    ax3.spines['left'].set_linewidth(1)
    ax3.spines['bottom'].set_linewidth(1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        bottom='on', \
        right='off', \
        top='off')

    ax3.set_xlim([0,7])
    ax3.set_xticks([0,2,4,6])
    ax3.set_xticklabels(['0','2','4','6'], fontsize=6)
    ax3.set_xlabel('peak dF/F', fontsize=6)
    # ax3.set_ylabel('fraction of neurons', fontsize=48)
    ax3.set_yticks([0,0.4,0.8])
    ax3.set_yticklabels(['0','0.4','0.8'], fontsize=6)

    ax4.spines['left'].set_linewidth(1)
    ax4.spines['bottom'].set_linewidth(1)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        bottom='on', \
        right='off', \
        top='off')

    ax4.set_xlim([0,1])
    ax4.set_xticks([0,0.5,1])
    ax4.set_xticklabels(['0','0.5','1'], fontsize=6)
    ax4.set_xlabel('Q1-Q3 quartile coef.', fontsize=6)
    # ax4.set_ylabel('fraction of neurons', fontsize=48)
    ax4.set_yticks([0,2,4])
    ax4.set_yticklabels(['0','2','4'], fontsize=6)

    ax9.spines['left'].set_linewidth(1)
    ax9.spines['bottom'].set_linewidth(1)
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        bottom='on', \
        right='off', \
        top='off')

    ax9.set_xlim([0,7])
    ax9.set_xticks([0,2,4,6])
    ax9.set_xticklabels(['0','2','4','6'], fontsize=6)
    ax9.set_xlabel('transient AUC', fontsize=6)
    # ax9.set_ylabel('fraction of neurons', fontsize=48)
    ax9.set_yticks([0,0.35])
    ax9.set_yticklabels(['0','0.35'], fontsize=6)

    # ax1.set_title('FWHM (mean for each ROI)')
    # ax2.set_title('transients/min')
    # ax3.set_title('peak dF/F value')
    # ax4.set_title('transient amplitude variability')

    ax5.spines['left'].set_linewidth(1)
    ax5.spines['bottom'].set_linewidth(1)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.set_xlabel('')
    ax5.set_ylabel('')
    ax5.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax6.spines['left'].set_linewidth(1)
    ax6.spines['bottom'].set_linewidth(1)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.set_xlabel('')
    ax6.set_ylabel('')
    ax6.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax7.spines['left'].set_linewidth(1)
    ax7.spines['bottom'].set_linewidth(1)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.set_xlabel('')
    ax7.set_ylabel('')
    ax7.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax8.spines['left'].set_linewidth(1)
    ax8.spines['bottom'].set_linewidth(1)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.set_xlabel('')
    ax8.set_ylabel('')
    ax8.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax10.spines['left'].set_linewidth(1)
    ax10.spines['bottom'].set_linewidth(1)
    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.set_xlabel('')
    ax10.set_ylabel('')
    ax10.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax5.set_xlim([0,9.85])
    ax6.set_xlim([0,9.85])
    ax7.set_xlim([0,6.85])
    ax8.set_xlim([0,0.98])
    ax10.set_xlim([0,6.85])

    ax5.set_ylim([0,1.05])
    ax6.set_ylim([0,1.05])
    ax7.set_ylim([0,1.05])
    ax8.set_ylim([0,1.05])
    ax10.set_ylim([0,1.05])

    ax5.set_xticklabels([''])
    ax6.set_xticklabels([''])
    ax7.set_xticklabels([''])
    ax8.set_xticklabels([''])
    ax10.set_xticklabels([''])

    ax5.set_yticklabels([''])
    ax6.set_yticklabels([''])
    ax7.set_yticklabels([''])
    ax8.set_yticklabels([''])
    ax10.set_yticklabels([''])

    subfolder = 'fwhm'
    fname = fname + '_fwhm'


    fig.tight_layout()

    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat,dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

# def combo_plot(l23_data, l5_data, fname):
def plot_peak_dist(roi_param_list, fname):
    fig = plt.figure(figsize=(10,3))
    # ax_all = []
    ax1 = plt.subplot2grid((1,20), (0,1), colspan=12)
    ax2 = plt.subplot2grid((1,20), (0,14), colspan=8)

    with open(roi_param_list,'r') as f:
        roi_params = json.load(f)

    valid_roi_idx = np.array(roi_params['valid_rois'])
    all_roi_peaks = []
    all_roi_medians = []
    all_quartile_coefficients = []
    for i,vri in enumerate(valid_roi_idx):
        roi_peaks = np.array(roi_params['transient_peaks'][valid_roi_idx[i]]) / roi_params['norm_value_all'][valid_roi_idx[i]]
        # print(roi_peaks)
        if not np.isnan(np.median(roi_peaks)):
            quartiles = np.quantile(roi_peaks,[0.25,0.75])
            quartile_coefficient = (quartiles[1] - quartiles[0]) / (quartiles[1] + quartiles[0])
            all_roi_peaks.append(roi_peaks)
            all_roi_medians.append(np.median(roi_peaks))
            all_quartile_coefficients.append(quartile_coefficient)

        # sns.distplot(roi_peaks, bins=np.arange(0,1.1,0.1), kde=False, color='blue', ax=ax_all[i])
        # ax_all[i].set_title('roi: ' + str(valid_roi_idx[i]) + ' disp: ' + str(np.round(quartile_coefficient,2)))

    # print(all_roi_medians)
    ordered_roi_peaks = [all_roi_peaks for _,all_roi_peaks in sorted(zip(all_roi_medians,all_roi_peaks))]
    ax1.boxplot(ordered_roi_peaks)
    sns.distplot(all_quartile_coefficients, bins=np.arange(0,1.1,0.1), kde=False, color='blue', ax=ax2)

    ax1.set_title('roi transients peak amplitudes (normalized)')
    ax2.set_title('quartile coefficients')

    subfolder = 'fwhm'
    fname = fname + '_peaks'
    # fig.tight_layout()

    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat,dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    return all_quartile_coefficients

    # transient_peaks_all = []
    # for rpl in roi_param_list:
    #     with open(rpl,'r') as f:
    #         roi_params = json.load(f)
    #     valid_roi_idx = np.array(roi_params['valid_rois'])
    #     # print(np.array(roi_params['FWHM_mean'])[valid_roi_idx].shape)
    #     fwhm_all.append(np.array(roi_params['FWHM_mean'])[valid_roi_idx])
    #     peak_all.append(np.array(roi_params['norm_value_all'])[valid_roi_idx])
    #     transient_rate_all.append(np.array(roi_params['transient_rate'])[valid_roi_idx])

def qc_summary_fig(roi_param_list_l23, roi_param_list_l5):

    l23_QC = []
    l5_QC = []

    for l23 in roi_param_list_l23:
        print(l23.split(os.sep)[-2])
        l23_QC.append(plot_peak_dist(l23, l23.split(os.sep)[-2]))

    for l5 in roi_param_list_l5:
        print(l5.split(os.sep)[-2])
        l5_QC.append(plot_peak_dist(l5, l5.split(os.sep)[-2]))

    l23_QC = [item for sublist in l23_QC for item in sublist]
    l5_QC = [item for sublist in l5_QC for item in sublist]
    l23_QC = np.array(l23_QC)
    l5_QC = np.array(l5_QC)

    fig = plt.figure(figsize=(5,5))
    # ax_all = []
    ax1 = plt.subplot(111)
    sns.distplot(l23_QC, bins=np.arange(0,1.1,0.04), kde=False, color='blue', ax=ax1)
    sns.distplot(l5_QC, bins=np.arange(0,1.1,0.04), kde=False, color='purple', ax=ax1)

    ax1.axvline(np.mean(l23_QC), c='blue',ls='--',lw=2)
    ax1.axvline(np.mean(l5_QC), c='purple',ls='--', lw=2)

    print(sp.stats.ttest_ind(l23_QC, l5_QC))

    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params( \
        reset='on',
        axis='both', \
        direction='out', \
        labelsize=6, \
        length=2, \
        width=1, \
        bottom='on', \
        right='off', \
        top='off')

    ax1.set_xticks([0,0.5,1])
    ax1.set_xticklabels(['0','0.5','1'])
    ax1.set_xlabel('Q1-Q3 quart. coef.', fontsize=24)
    ax1.set_ylabel('# neurons', fontsize=24)

    ax1.set_xlim([0,1])

    subfolder = 'fwhm'
    fname = 'layer_QCs_LF161202_1'

    fig.tight_layout()

    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

if __name__ == '__main__':
    suffix = ''

    # LAYER 2/3
    roi_param_list_l23 = [
                      loc_info['figure_output_path'] + 'LF170613_1' + os.sep + 'LF170613_1_Day20170804' + suffix + '.json',
                      loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_1' + suffix + '.json',
                      loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_2' + suffix + '.json',
                      loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_3' + suffix + '.json',
                      # loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l23' + suffix + '.json'
                      # loc_info['figure_output_path'] + 'LF171211_1' + os.sep + 'LF171211_1_Day2018321_2' + suffix + '.json'
                     ]

    # LAYER 5
    roi_param_list_l5 = [
                      loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day20170719' + suffix +  '.json',
                      loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day2017720' + suffix +  '.json',
                      loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day201783' + suffix + '.json',
                      loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day2017719' + suffix + '.json',
                      loc_info['figure_output_path'] + 'LF170222_1' + os.sep + 'LF170222_1_Day201776' + suffix + '.json'
                      # loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l5' + suffix + '.json'
                     ]

    FHWM_properties(roi_param_list_l23, roi_param_list_l5, 'layers_async')

    roi_param_list_l23 = [
                     loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l23' + suffix + '.json',
                     loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day20170209_l23' + suffix + '.json',
                     loc_info['figure_output_path'] + 'LF170110_1' + os.sep + 'LF170110_1_Day20170215_l23' + suffix + '.json',
                     ]

    roi_param_list_l5 = [
                     loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l5' + suffix + '.json',
                     loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day20170209_l5' + suffix + '.json',
                     loc_info['figure_output_path'] + 'LF170110_1' + os.sep + 'LF170110_1_Day20170215_l5' + suffix + '.json',
                     ]

    FHWM_properties(roi_param_list_l23, roi_param_list_l5, 'layers_sync')
    # plot quartile coefficients
    # qc_summary_fig(roi_param_list_l23, roi_param_list_l5)

    # FHWM_properties(roi_param_list_l5, 'l5', 'purple')
