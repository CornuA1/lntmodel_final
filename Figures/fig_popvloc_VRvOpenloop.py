"""
read .json file from population vector analysis for VR and openloop and
calculate differences in location reconstruction errors

max_bin is by default set to 110, which corresponds to 2 cm bins for short length

"""

%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings; warnings.simplefilter('ignore')
import yaml
import codecs, json
import seaborn as sns
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

def fig_VRvOL(fig_datasets, fname, fformat='png', subfolder=[]):

    bin_size = 5
    max_bin = 44

    tracklength_short = 320
    tracklength_long = 380
    track_start = 100
    bin_size = 5
    lm_start = (200-track_start)/bin_size
    lm_end = (240-track_start)/bin_size
    end_bin_short = (tracklength_short-track_start) / bin_size
    end_bin_long = (tracklength_long-track_start) / bin_size

    track_start_bin = 0

    color_short = sns.xkcd_rgb["windows blue"]
    color_long = sns.xkcd_rgb["dusty purple"]

    color_short = '#4BAF0B'
    color_long = '#D3059E'

    std_prelm_ss = []
    std_lm_ss = []
    std_pi_ss = []

    std_prelm_ll = []
    std_lm_ll = []
    std_pi_ll = []
    std_pi_ll_long = []

    std_prelm_sl = []
    std_lm_sl = []
    std_pi_sl = []

    for j in json_files:
        with open(content['figure_output_path'] + subfolder + os.sep + j[0] + '.json') as f:
            popvloc_vr = json.load(f)

        with open(content['figure_output_path'] + subfolder + os.sep + j[1] + '.json') as f:
            popvloc_ol = json.load(f)

        popvec_cc_reconstruction_ss_vr = np.array(popvloc_vr['popvec_cc_reconstruction_ss'])
        ss_reconstruction_bybin_vr = np.zeros((max_bin,np.size(popvec_cc_reconstruction_ss_vr,2)))
        for i in range(popvec_cc_reconstruction_ss_vr.shape[2]):
            bin_diff = popvec_cc_reconstruction_ss_vr[1,:max_bin,i] - popvec_cc_reconstruction_ss_vr[0,:max_bin,i]
            bin_diff = bin_diff * bin_diff
            ss_reconstruction_bybin_vr[:,i] = np.sqrt(bin_diff) * bin_size

            std_prelm_ss = np.append(std_prelm_ss, (np.sum(ss_reconstruction_bybin_vr[track_start_bin:lm_start,i])/lm_start))
            std_lm_ss = np.append(std_lm_ss, (np.sum(ss_reconstruction_bybin_vr[lm_start:lm_end,i])/(lm_end-lm_start)))
            std_pi_ss = np.append(std_pi_ss, (np.sum(ss_reconstruction_bybin_vr[lm_end:end_bin_short,i])/(end_bin_short-lm_end)))

        popvec_cc_reconstruction_ss_ol = np.array(popvloc_ol['popvec_cc_reconstruction_ss'])
        ss_reconstruction_bybin_ol = np.zeros((max_bin,np.size(popvec_cc_reconstruction_ss_ol,2)))
        for i in range(popvec_cc_reconstruction_ss_ol.shape[2]):
            bin_diff = popvec_cc_reconstruction_ss_ol[1,:max_bin,i] - popvec_cc_reconstruction_ss_ol[0,:max_bin,i]
            bin_diff = bin_diff * bin_diff
            ss_reconstruction_bybin_ol[:,i] = np.sqrt(bin_diff) * bin_size



        popvec_cc_reconstruction_ll_vr = np.array(popvloc_vr['popvec_cc_reconstruction_ll'])
        ll_reconstruction_bybin_vr = np.zeros((max_bin,np.size(popvec_cc_reconstruction_ll_vr,2)))
        for i in range(popvec_cc_reconstruction_ll_vr.shape[2]):
            bin_diff = popvec_cc_reconstruction_ll_vr[1,:max_bin,i] - popvec_cc_reconstruction_ll_vr[0,:max_bin,i]
            bin_diff = bin_diff * bin_diff
            ll_reconstruction_bybin_vr[:,i] = np.sqrt(bin_diff) * bin_size

        std_prelm_ll = np.append(std_prelm_ll, (np.sum(ll_reconstruction_bybin_vr[track_start_bin:lm_start,i])/lm_start))
        std_lm_ll = np.append(std_lm_ll, (np.sum(ll_reconstruction_bybin_vr[lm_start:lm_end,i])/(lm_end-lm_start)))
        std_pi_ll = np.append(std_pi_ll, (np.sum(ll_reconstruction_bybin_vr[lm_end:end_bin_short,i])/(end_bin_short-lm_end)))
        std_pi_ll_long = np.append(std_pi_ll_long,np.sum(ll_reconstruction_bybin_vr[end_bin_short:end_bin_long,i])/(end_bin_long-end_bin_short))

        popvec_cc_reconstruction_ll_ol = np.array(popvloc_ol['popvec_cc_reconstruction_ll'])
        ll_reconstruction_bybin_ol = np.zeros((max_bin,np.size(popvec_cc_reconstruction_ll_ol,2)))
        for i in range(popvec_cc_reconstruction_ll_ol.shape[2]):
            bin_diff = popvec_cc_reconstruction_ll_ol[1,:max_bin,i] - popvec_cc_reconstruction_ll_ol[0,:max_bin,i]
            bin_diff = bin_diff * bin_diff
            ll_reconstruction_bybin_ol[:,i] = np.sqrt(bin_diff) * bin_size

    ss_reconstruction_bybin_vr_mean = np.mean(np.mean(ss_reconstruction_bybin_vr,axis=1))
    ss_reconstruction_bybin_vr_stderr = stats.sem(np.mean(ss_reconstruction_bybin_vr,axis=1))
    ss_reconstruction_bybin_ol_mean = np.mean(np.mean(ss_reconstruction_bybin_ol,axis=1))
    ss_reconstruction_bybin_ol_stderr = stats.sem(np.mean(ss_reconstruction_bybin_ol,axis=1))
    ll_reconstruction_bybin_vr_mean = np.mean(np.mean(ll_reconstruction_bybin_vr,axis=1))
    ll_reconstruction_bybin_vr_stderr = stats.sem(np.mean(ll_reconstruction_bybin_vr,axis=1))
    ll_reconstruction_bybin_ol_mean = np.mean(np.mean(ll_reconstruction_bybin_ol,axis=1))
    ll_reconstruction_bybin_ol_stderr = stats.sem(np.mean(ll_reconstruction_bybin_ol,axis=1))

    f_value_ss, p_value_ss = stats.f_oneway(np.mean(ss_reconstruction_bybin_vr,axis=1),np.mean(ss_reconstruction_bybin_ol,axis=1),np.mean(ll_reconstruction_bybin_vr,axis=1),np.mean(ll_reconstruction_bybin_ol,axis=1))
    group_labels = ['ssvr'] * np.mean(ss_reconstruction_bybin_vr,axis=1).shape[0] + ['ssol'] * np.mean(ss_reconstruction_bybin_ol,axis=1).shape[0] + ['llvr'] * np.mean(ll_reconstruction_bybin_vr,axis=1).shape[0] + ['llol'] * np.mean(ll_reconstruction_bybin_ol,axis=1).shape[0]
    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((np.mean(ss_reconstruction_bybin_vr,axis=1),np.mean(ss_reconstruction_bybin_ol,axis=1),np.mean(ll_reconstruction_bybin_vr,axis=1),np.mean(ll_reconstruction_bybin_ol,axis=1))),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)

    # print(ss_reconstruction_bybin_vr_mean.shape,ss_reconstruction_bybin_ol_mean.shape,ll_reconstruction_bybin_vr_mean.shape,ll_reconstruction_bybin_ol_mean.shape)
    print(stats.ttest_ind(np.concatenate((np.mean(ss_reconstruction_bybin_vr,axis=1),np.mean(ss_reconstruction_bybin_ol,axis=1))),np.concatenate((np.mean(ll_reconstruction_bybin_vr,axis=1),np.mean(ll_reconstruction_bybin_ol,axis=1)))))

    fig = plt.figure(figsize=(3,8))
    ax1 = plt.subplot(111)

    ### BLOCK BELOW: plot for individual trial types ###
    # ax1.scatter([0.25,0.75],[ss_reconstruction_bybin_vr_mean,ss_reconstruction_bybin_ol_mean], s=80,color=sns.xkcd_rgb["windows blue"], linewidths=0, zorder=2)
    # ax1.scatter([0.25,0.75],[ll_reconstruction_bybin_vr_mean,ll_reconstruction_bybin_ol_mean], s=80,color=sns.xkcd_rgb["dusty purple"], linewidths=0, zorder=2)
    #
    # ax1.errorbar([0.25,0.75], [ss_reconstruction_bybin_vr_mean,ss_reconstruction_bybin_ol_mean], yerr=[ss_reconstruction_bybin_vr_stderr,ss_reconstruction_bybin_ol_stderr],fmt='none',ecolor='k', zorder=1)
    # ax1.errorbar([0.25,0.75], [ll_reconstruction_bybin_vr_mean,ll_reconstruction_bybin_ol_mean], yerr=[ll_reconstruction_bybin_vr_stderr,ll_reconstruction_bybin_ol_stderr],fmt='none',ecolor='k', zorder=1)
    #
    # ax1.plot([0.25,0.75], [ss_reconstruction_bybin_vr_mean,ss_reconstruction_bybin_ol_mean], lw=3, c=sns.xkcd_rgb["windows blue"])
    # ax1.plot([0.25,0.75], [ll_reconstruction_bybin_vr_mean,ll_reconstruction_bybin_ol_mean], lw=3, c=sns.xkcd_rgb["dusty purple"])
    ### END plot of individual trial types

    ax1.scatter([0.25,0.75],[np.mean([ss_reconstruction_bybin_vr_mean,ll_reconstruction_bybin_vr_mean]),np.mean([ss_reconstruction_bybin_ol_mean,ll_reconstruction_bybin_ol_mean])], s=120,color='0.5', linewidths=0, zorder=2)
    avg_sem_vr = stats.sem(np.concatenate((np.mean(ss_reconstruction_bybin_vr,axis=1),np.mean(ll_reconstruction_bybin_vr,axis=1))))
    avg_sem_ol = stats.sem(np.concatenate((np.mean(ss_reconstruction_bybin_ol,axis=1),np.mean(ll_reconstruction_bybin_ol,axis=1))))
    ax1.errorbar([0.25,0.75], [np.mean([ss_reconstruction_bybin_vr_mean,ll_reconstruction_bybin_vr_mean]),np.mean([ss_reconstruction_bybin_ol_mean,ll_reconstruction_bybin_ol_mean])], yerr=[avg_sem_vr,avg_sem_ol],fmt='none',ecolor='k', zorder=1)
    # ax1.errorbar([0.25,0.75], [ll_reconstruction_bybin_vr_mean,ll_reconstruction_bybin_ol_mean], yerr=[ll_reconstruction_bybin_vr_stderr,ll_reconstruction_bybin_ol_stderr],fmt='none',ecolor='k', zorder=1)

    ax1.plot([0.25,0.75], [np.mean([ss_reconstruction_bybin_vr_mean,ll_reconstruction_bybin_vr_mean]),np.mean([ss_reconstruction_bybin_ol_mean,ll_reconstruction_bybin_ol_mean])], lw=3, c='0.5')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=20)
    ax1.set_xlim([0.2,0.9])
    ax1.set_ylim([0,35])
    ax1.set_xticks([0.25,0.75])
    ax1.set_xticklabels(['VR','Passive'])
    ax1.set_ylabel('prediction error (cm)',fontsize=24)
    # ax1.set_xlabel('location (cm)',fontsize=24)

    plt.tight_layout()

    fname = content['figure_output_path'] + subfolder + os.sep + fname + '_' + str(popvec_cc_reconstruction_ss_vr.shape[2]) + '.' + fformat
    try:
        fig.savefig(fname, format=fformat, dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

if __name__ == "__main__":

    # filenames
    json_files = [['task_engaged_all_popvloc_results_100','task_engaged_all_openloop_popvloc_results_100']]#,'task_engaged_V1_popvloc_results_100','task_engaged_all_openloop_popvloc_results_100','task_engaged_V1_openloop_popvloc_results_100']
    json_files = [['task_engaged_V1_popvloc_results_100','task_engaged_V1_openloop_popvloc_results_100']]

    # figure output parameters
    subfolder = 'popvloc'
    fname = 'popvec_cc_vrol_V1'
    fformat = 'png'

    fig_VRvOL(json_files, fname, fformat, subfolder)
