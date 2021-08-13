"""
read .json file from population vector analysis for VR and openloop and
calculate differences between RSC vr vs openloop and V1 vr vs openloop
differences

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

def fig_RSCvsV1(json_files, fname, fformat='png', subfolder=[]):
    bin_size = 5

    with open(content['figure_output_path'] + subfolder + os.sep + json_files[0][0] + '.json') as f:
        popvloc_RSCvr = json.load(f)
    with open(content['figure_output_path'] + subfolder + os.sep + json_files[0][1] + '.json') as f:
        popvloc_RSCol = json.load(f)
    with open(content['figure_output_path'] + subfolder + os.sep + json_files[1][0] + '.json') as f:
        popvloc_V1vr = json.load(f)
    with open(content['figure_output_path'] + subfolder + os.sep + json_files[1][1] + '.json') as f:
        popvloc_V1ol = json.load(f)

    # --- RSC DATASETS ---
    popvec_cc_reconstruction_ss_vr = np.array(popvloc_RSCvr['popvec_cc_reconstruction_ss'])
    ss_reconstruction_bybin_RSCvr = np.zeros((np.size(popvec_cc_reconstruction_ss_vr,1),np.size(popvec_cc_reconstruction_ss_vr,2)))
    for i in range(popvec_cc_reconstruction_ss_vr.shape[2]):
        bin_diff = popvec_cc_reconstruction_ss_vr[1,:,i] - popvec_cc_reconstruction_ss_vr[0,:,i]
        bin_diff = bin_diff * bin_diff
        ss_reconstruction_bybin_RSCvr[:,i] = np.sqrt(bin_diff) * bin_size

    popvec_cc_reconstruction_ss_ol = np.array(popvloc_RSCol['popvec_cc_reconstruction_ss'])
    ss_reconstruction_bybin_RSCol = np.zeros((np.size(popvec_cc_reconstruction_ss_ol,1),np.size(popvec_cc_reconstruction_ss_ol,2)))
    for i in range(popvec_cc_reconstruction_ss_ol.shape[2]):
        bin_diff = popvec_cc_reconstruction_ss_ol[1,:,i] - popvec_cc_reconstruction_ss_ol[0,:,i]
        bin_diff = bin_diff * bin_diff
        ss_reconstruction_bybin_RSCol[:,i] = np.sqrt(bin_diff) * bin_size

    popvec_cc_reconstruction_ll_vr = np.array(popvloc_RSCvr['popvec_cc_reconstruction_ll'])
    ll_reconstruction_bybin_RSCvr = np.zeros((np.size(popvec_cc_reconstruction_ll_vr,1),np.size(popvec_cc_reconstruction_ll_vr,2)))
    for i in range(popvec_cc_reconstruction_ll_vr.shape[2]):
        bin_diff = popvec_cc_reconstruction_ll_vr[1,:,i] - popvec_cc_reconstruction_ll_vr[0,:,i]
        bin_diff = bin_diff * bin_diff
        ll_reconstruction_bybin_RSCvr[:,i] = np.sqrt(bin_diff) * bin_size

    popvec_cc_reconstruction_ll_ol = np.array(popvloc_RSCol['popvec_cc_reconstruction_ll'])
    ll_reconstruction_bybin_RSCol = np.zeros((np.size(popvec_cc_reconstruction_ll_ol,1),np.size(popvec_cc_reconstruction_ll_ol,2)))
    for i in range(popvec_cc_reconstruction_ll_ol.shape[2]):
        bin_diff = popvec_cc_reconstruction_ll_ol[1,:,i] - popvec_cc_reconstruction_ll_ol[0,:,i]
        bin_diff = bin_diff * bin_diff
        ll_reconstruction_bybin_RSCol[:,i] = np.sqrt(bin_diff) * bin_size

    # --- V1 DATASETS ---
    popvec_cc_reconstruction_ss_vr = np.array(popvloc_V1vr['popvec_cc_reconstruction_ss'])
    ss_reconstruction_bybin_V1vr = np.zeros((np.size(popvec_cc_reconstruction_ss_vr,1),np.size(popvec_cc_reconstruction_ss_vr,2)))
    for i in range(popvec_cc_reconstruction_ss_vr.shape[2]):
        bin_diff = popvec_cc_reconstruction_ss_vr[1,:,i] - popvec_cc_reconstruction_ss_vr[0,:,i]
        bin_diff = bin_diff * bin_diff
        ss_reconstruction_bybin_V1vr[:,i] = np.sqrt(bin_diff) * bin_size

    popvec_cc_reconstruction_ss_ol = np.array(popvloc_V1ol['popvec_cc_reconstruction_ss'])
    ss_reconstruction_bybin_V1ol = np.zeros((np.size(popvec_cc_reconstruction_ss_ol,1),np.size(popvec_cc_reconstruction_ss_ol,2)))
    for i in range(popvec_cc_reconstruction_ss_ol.shape[2]):
        bin_diff = popvec_cc_reconstruction_ss_ol[1,:,i] - popvec_cc_reconstruction_ss_ol[0,:,i]
        bin_diff = bin_diff * bin_diff
        ss_reconstruction_bybin_V1ol[:,i] = np.sqrt(bin_diff) * bin_size

    popvec_cc_reconstruction_ll_vr = np.array(popvloc_V1vr['popvec_cc_reconstruction_ll'])
    ll_reconstruction_bybin_V1vr = np.zeros((np.size(popvec_cc_reconstruction_ll_vr,1),np.size(popvec_cc_reconstruction_ll_vr,2)))
    for i in range(popvec_cc_reconstruction_ll_vr.shape[2]):
        bin_diff = popvec_cc_reconstruction_ll_vr[1,:,i] - popvec_cc_reconstruction_ll_vr[0,:,i]
        bin_diff = bin_diff * bin_diff
        ll_reconstruction_bybin_V1vr[:,i] = np.sqrt(bin_diff) * bin_size

    popvec_cc_reconstruction_ll_ol = np.array(popvloc_V1ol['popvec_cc_reconstruction_ll'])
    ll_reconstruction_bybin_V1ol = np.zeros((np.size(popvec_cc_reconstruction_ll_ol,1),np.size(popvec_cc_reconstruction_ll_ol,2)))
    for i in range(popvec_cc_reconstruction_ll_ol.shape[2]):
        bin_diff = popvec_cc_reconstruction_ll_ol[1,:,i] - popvec_cc_reconstruction_ll_ol[0,:,i]
        bin_diff = bin_diff * bin_diff
        ll_reconstruction_bybin_V1ol[:,i] = np.sqrt(bin_diff) * bin_size

    print(ss_reconstruction_bybin_V1ol.shape,ss_reconstruction_bybin_V1vr.shape)
    ss_reconstruction_RSC_diffs = np.mean(ss_reconstruction_bybin_RSCol,axis=1) - np.mean(ss_reconstruction_bybin_RSCvr,axis=1)
    ll_reconstruction_RSC_diffs = np.mean(ll_reconstruction_bybin_RSCol,axis=1) - np.mean(ll_reconstruction_bybin_RSCvr,axis=1)
    ss_reconstruction_V1_diffs = np.mean(ss_reconstruction_bybin_V1ol,axis=1) - np.mean(ss_reconstruction_bybin_V1vr,axis=1)
    ll_reconstruction_V1_diffs = np.mean(ll_reconstruction_bybin_V1ol,axis=1) - np.mean(ll_reconstruction_bybin_V1vr,axis=1)

    ss_reconstruction_RSC_diffs_sem = stats.sem(np.mean(ss_reconstruction_bybin_RSCol,axis=1) - np.mean(ss_reconstruction_bybin_RSCvr,axis=1))
    ll_reconstruction_RSC_diffs_sem = stats.sem(np.mean(ll_reconstruction_bybin_RSCol,axis=1) - np.mean(ll_reconstruction_bybin_RSCvr,axis=1))
    ss_reconstruction_V1_diffs_sem = stats.sem(np.mean(ss_reconstruction_bybin_V1ol,axis=1) - np.mean(ss_reconstruction_bybin_V1vr,axis=1))
    ll_reconstruction_V1_diffs_sem = stats.sem(np.mean(ll_reconstruction_bybin_V1ol,axis=1) - np.mean(ll_reconstruction_bybin_V1vr,axis=1))

    f_value_ss, p_value_ss = stats.f_oneway(ss_reconstruction_RSC_diffs,ll_reconstruction_RSC_diffs,ss_reconstruction_V1_diffs,ll_reconstruction_V1_diffs)
    group_labels = ['ssRSC'] * ss_reconstruction_RSC_diffs.shape[0] + ['llRSC'] * ll_reconstruction_RSC_diffs.shape[0] + ['ssV1'] * ss_reconstruction_V1_diffs.shape[0] + ['llV1'] * ll_reconstruction_V1_diffs.shape[0]
    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((ss_reconstruction_RSC_diffs,ll_reconstruction_RSC_diffs,ss_reconstruction_V1_diffs,ll_reconstruction_V1_diffs)),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()
    print(posthoc_res_ss)

    RSCvrol_ss_diff = np.mean(ss_reconstruction_RSC_diffs)
    V1vrol_ss_diff = np.mean(ss_reconstruction_V1_diffs)
    RSCvrol_ll_diff = np.mean(ll_reconstruction_RSC_diffs)
    V1vrol_ll_diff = np.mean(ll_reconstruction_V1_diffs)

    fig = plt.figure(figsize=(4,8))
    ax1 = plt.subplot(111)

    ax1.scatter([0.25,0.75],[RSCvrol_ss_diff,V1vrol_ss_diff], s=80,color=sns.xkcd_rgb["windows blue"], linewidths=0, zorder=2)
    ax1.scatter([0.25,0.75],[RSCvrol_ll_diff,V1vrol_ll_diff], s=80,color=sns.xkcd_rgb["dusty purple"], linewidths=0, zorder=2)

    ax1.errorbar([0.25,0.75], [RSCvrol_ss_diff,V1vrol_ss_diff], yerr=[ss_reconstruction_RSC_diffs_sem,ss_reconstruction_V1_diffs_sem],fmt='none',ecolor='k', zorder=1)
    ax1.errorbar([0.25,0.75], [RSCvrol_ll_diff,V1vrol_ll_diff], yerr=[ll_reconstruction_RSC_diffs_sem,ll_reconstruction_V1_diffs_sem],fmt='none',ecolor='k', zorder=1)

    ax1.plot([0.25,0.75], [RSCvrol_ss_diff,V1vrol_ss_diff], lw=3, c=sns.xkcd_rgb["windows blue"])
    ax1.plot([0.25,0.75], [RSCvrol_ll_diff,V1vrol_ll_diff], lw=3, c=sns.xkcd_rgb["dusty purple"])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=20)
    ax1.set_xlim([0.2,0.9])
    #ax1.set_ylim([0,20])
    ax1.set_xticks([0.25,0.75])
    ax1.set_xticklabels(['RSC','V1'])
    ax1.set_ylabel('prediction error (cm)',fontsize=24)
    ax1.set_xlabel('location (cm)',fontsize=24)

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
    json_files = [['task_engaged_all_popvloc_results_100','task_engaged_all_openloop_popvloc_results_100'],['task_engaged_V1_popvloc_results_100','task_engaged_V1_openloop_popvloc_results_100']]
    # figure output parameters
    subfolder = 'popvloc'
    fname = 'popvec_cc_RSCvsV1'
    fformat = 'png'

    fig_RSCvsV1(json_files, fname, fformat, subfolder)
