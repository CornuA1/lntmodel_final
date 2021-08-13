"""
read .json file from population vector analysis for further analysis

"""

# %matplotlib inline

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

def sl_trialtype_diff(fig_datasets, fname, fformat='png', subfolder=[]):

    tracklength_short = 320
    tracklength_long = 380
    track_start = 100
    bin_size = 5
    lm_start = int((200-track_start)/bin_size)
    lm_end = int((240-track_start)/bin_size)
    end_bin_short = int((tracklength_short-track_start) / bin_size)
    end_bin_long = int((tracklength_long-track_start) / bin_size)

    track_start_bin = 0

    color_short = sns.xkcd_rgb["windows blue"]
    color_long = sns.xkcd_rgb["dusty purple"]

    # color_short = '#4BAF0B'
    # color_long = '#D3059E'

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

    for j in fig_datasets:
        with open(content['figure_output_path'] + subfolder + os.sep + j + '.json') as f:
            popvloc_result = json.load(f)

        # print(track_start_bin,lm_start)
        # calculate deviation (step by step so even an idiot like myself doesn't get confused) of reconstruction vs actual location
        popvec_cc_reconstruction_ss = np.array(popvloc_result['popvec_cc_reconstruction_ss'])
        ss_reconstruction_bybin = np.zeros((np.size(popvec_cc_reconstruction_ss,1),np.size(popvec_cc_reconstruction_ss,2)))
        ss_reconstruction_bybin_stderr = np.zeros((np.size(popvec_cc_reconstruction_ss,1)))
        for i in range(popvec_cc_reconstruction_ss.shape[2]):
            bin_diff = popvec_cc_reconstruction_ss[1,:,i] - popvec_cc_reconstruction_ss[0,:,i]
            bin_diff = bin_diff * bin_diff
            ss_reconstruction_bybin[:,i] = np.sqrt(bin_diff) * bin_size
            ss_reconstruction_bybin_stderr = np.sqrt(bin_diff) / np.sqrt(popvec_cc_reconstruction_ss.shape[2])

            #std_prelm_ss = np.append(std_prelm_ss, np.sqrt(np.sum(bin_diff[track_start_bin:lm_start])/(lm_start)))
            # std_lm_ss = np.append(std_lm_ss,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
            print(track_start_bin,lm_start,i)
            std_prelm_ss = np.append(std_prelm_ss, (np.sum(ss_reconstruction_bybin[track_start_bin:lm_start,i])/lm_start))
            std_lm_ss = np.append(std_lm_ss, (np.sum(ss_reconstruction_bybin[lm_start:lm_end,i])/(lm_end-lm_start)))
            std_pi_ss = np.append(std_pi_ss, (np.sum(ss_reconstruction_bybin[lm_end:end_bin_short,i])/(end_bin_short-lm_end)))
            # std_pi_ss = np.append(std_pi_ss,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))
            # ss_reconstruction_bybin[:,i] = np.abs(popvec_cc_reconstruction_ss[1,:,i] - popvec_cc_reconstruction_ss[0,:,i]) * bin_size

        popvec_cc_reconstruction_ll = np.array(popvloc_result['popvec_cc_reconstruction_ll'])
        ll_reconstruction_bybin = np.zeros((np.size(popvec_cc_reconstruction_ll,1),np.size(popvec_cc_reconstruction_ll,2)))
        ll_reconstruction_bybin_stderr = np.zeros((np.size(popvec_cc_reconstruction_ll,1)))
        for i in range(popvec_cc_reconstruction_ll.shape[2]):
            bin_diff = popvec_cc_reconstruction_ll[1,:,i] - popvec_cc_reconstruction_ll[0,:,i]
            bin_diff = bin_diff * bin_diff
            ll_reconstruction_bybin[:,i] = np.sqrt(bin_diff) * bin_size
            ll_reconstruction_bybin_stderr = np.sqrt(bin_diff) / np.sqrt(popvec_cc_reconstruction_ll.shape[2])

            std_prelm_ll = np.append(std_prelm_ll, (np.sum(ll_reconstruction_bybin[track_start_bin:lm_start,i])/lm_start))
            std_lm_ll = np.append(std_lm_ll, (np.sum(ll_reconstruction_bybin[lm_start:lm_end,i])/(lm_end-lm_start)))
            std_pi_ll = np.append(std_pi_ll, (np.sum(ll_reconstruction_bybin[lm_end:end_bin_short,i])/(end_bin_short-lm_end)))
            std_pi_ll_long = np.append(std_pi_ll_long,np.sum(ll_reconstruction_bybin[end_bin_short:end_bin_long,i])/(end_bin_long-end_bin_short))
            # std_pi_ll_long = np.append(std_pi_ll_long,np.sqrt(np.sum(bin_diff[end_bin_short:end_bin_long])/(end_bin_long-end_bin_short)))
            # ll_reconstruction_bybin[:,i] = np.abs(popvec_cc_reconstruction_ll[1,:,i] - popvec_cc_reconstruction_ll[0,:,i]) * bin_size

        # popvec_cc_reconstruction_sl = np.array(popvloc_result['popvec_cc_reconstruction_sl'])
        # sl_reconstruction_bybin = np.zeros((np.size(popvec_cc_reconstruction_sl,1),np.size(popvec_cc_reconstruction_sl,2)))
        # sl_reconstruction_bybin_stderr = np.zeros((np.size(popvec_cc_reconstruction_sl,1)))
        # for i in range(popvec_cc_reconstruction_sl.shape[2]):
        #     bin_diff = popvec_cc_reconstruction_sl[1,:,i] - popvec_cc_reconstruction_sl[0,:,i]
        #     bin_diff = bin_diff * bin_diff
        #     sl_reconstruction_bybin[:,i] = np.sqrt(bin_diff) * bin_size
        #     sl_reconstruction_bybin_stderr = np.sqrt(bin_diff) / np.sqrt(popvec_cc_reconstruction_sl.shape[2])
        #
        #     std_prelm_sl = np.append(std_prelm_sl, np.sqrt(np.sum(bin_diff[track_start_bin:lm_start])/(lm_start)))
        #     std_lm_sl = np.append(std_lm_sl,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        #     std_pi_sl = np.append(std_pi_sl,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))
            # sl_reconstruction_bybin[:,i] = np.abs(popvec_cc_reconstruction_sl[1,:,i] - popvec_cc_reconstruction_sl[0,:,i]) * bin_size

        ss_reconstruction_bybin_graph = np.mean(ss_reconstruction_bybin, axis=1)
        ll_reconstruction_bybin_graph = np.mean(ll_reconstruction_bybin, axis=1)
        # sl_reconstruction_bybin_graph = np.mean(sl_reconstruction_bybin, axis=1)

        # perform statistical tests (one-way anova followed by pairwise tests).
        # f_value_ss, p_value_ss = stats.f_oneway(std_prelm_ss,std_lm_ss,std_pi_ss)
        # group_labels = ['prelmss'] * len(std_prelm_ss) + ['lmss'] * len(std_lm_ss) + ['piss'] * len(std_pi_ss) + ['prelmll'] * len(std_prelm_sl) + ['lmll'] * len(std_lm_sl) + ['pill'] * len(std_pi_sl) + ['prelmsl'] * len(std_prelm_ll) + ['lmsl'] * len(std_lm_ll) + ['pisl'] * len(std_pi_ll)
        # mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ss,std_lm_ss,std_pi_ss,std_prelm_ll,std_lm_ll,std_pi_ll,std_prelm_sl,std_lm_sl,std_pi_sl)),group_labels)
        # posthoc_res_ss = mc_res_ss.tukeyhsd()
        # print(posthoc_res_ss)

        group_labels = ['prelmss'] * len(std_prelm_ss) + ['lmss'] * len(std_lm_ss) + ['piss'] * len(std_pi_ss) + ['prelmll'] * len(std_prelm_ll) + ['lmll'] * len(std_lm_ll) + ['pill'] * len(std_pi_ll)
        mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ss,std_lm_ss,std_pi_ss,std_prelm_ll,std_lm_ll,std_pi_ll)),group_labels)
        posthoc_res_ss = mc_res_ss.tukeyhsd()
        print(posthoc_res_ss)


        # f_value_sl, p_value_sl = stats.f_oneway(std_prelm_sl,std_lm_sl,std_pi_sl)
        # group_labels = ['prelm'] * len(std_prelm_sl) + ['lm'] * len(std_lm_sl) + ['pi'] * len(std_pi_sl)
        # mc_res_sl = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_sl,std_lm_sl,std_pi_sl)),group_labels)
        # posthoc_res_sl = mc_res_sl.tukeyhsd()
        #
        # f_value_ll, p_value_ll = stats.f_oneway(std_prelm_ll,std_lm_ll,std_pi_ll)
        # group_labels = ['prelm'] * len(std_prelm_ll) + ['lm'] * len(std_lm_ll) + ['pi'] * len(std_pi_ll)
        # mc_res_ll = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ll,std_lm_ll,std_pi_ll)),group_labels)
        # posthoc_res_ll = mc_res_ll.tukeyhsd()

        fig = plt.figure(figsize=(8,8))
        ax1 = plt.subplot(111)
        # ax2 = plt.subplot(212)
        boxcar_len = 5
        boxcar = np.ones(boxcar_len)/boxcar_len
        ss_smooth = np.convolve(ss_reconstruction_bybin_graph, boxcar, mode='same')
        ax1.plot(ss_smooth,c=color_short,lw=4)
        # ax1.plot(ss_reconstruction_bybin_graph,c=color_short,lw=4)
        # ax1.fill_between(np.arange(ss_reconstruction_bybin_stderr.shape[0]), ss_reconstruction_bybin_graph,ss_reconstruction_bybin_graph-ss_reconstruction_bybin_stderr,alpha=0.3,color=color_short)
        # ax1.fill_between(np.arange(ss_reconstruction_bybin_stderr.shape[0]), ss_reconstruction_bybin_graph,ss_reconstruction_bybin_graph+ss_reconstruction_bybin_stderr,alpha=0.3,color=color_short)

        ll_smooth = np.convolve(ll_reconstruction_bybin_graph, boxcar, mode='same')
        ax1.plot(ll_smooth,c=color_long,lw=4)
        # ax1.plot(ll_reconstruction_bybin_graph,c=color_long,lw=4)
        # ax1.fill_between(np.arange(ll_reconstruction_bybin_stderr.shape[0]), ll_reconstruction_bybin_graph,ll_reconstruction_bybin_graph-ll_reconstruction_bybin_stderr,alpha=0.3,color=color_long)
        # ax1.fill_between(np.arange(ll_reconstruction_bybin_stderr.shape[0]), ll_reconstruction_bybin_graph,ll_reconstruction_bybin_graph+ll_reconstruction_bybin_stderr,alpha=0.3,color=color_long)

        # ax1.plot(sl_reconstruction_bybin_graph)
        # ax1.fill_between(np.arange(sl_reconstruction_bybin_stderr.shape[0]), sl_reconstruction_bybin_graph,sl_reconstruction_bybin_graph-sl_reconstruction_bybin_stderr,alpha=0.5)
        # ax1.fill_between(np.arange(sl_reconstruction_bybin_stderr.shape[0]), sl_reconstruction_bybin_graph,sl_reconstruction_bybin_graph+sl_reconstruction_bybin_stderr,alpha=0.5)
        # ax1.plot(np.mean(sl_reconstruction_bybin, axis=1))

        # ax1.axvline(50/bin_size,c='0.5',ls='--',lw=2)
        ax1.axvline(lm_start,c='0.5',ls='--',lw=2)
        ax1.axvline(lm_end,c='0.5',ls='--',lw=2)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)

        ax1.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=20)
        # ax1.set_xlim([-10/bin_size,280/bin_size])
        #ax1.set_ylim([0,30])
        # ax1.set_xticks([100/bin_size,140/bin_size,220/bin_size,280/bin_size])
        ax1.set_xticks([20,30,44,56])
        ax1.set_xticklabels(['200','240','320','360'])
        ax1.set_ylabel('prediction error (cm)',fontsize=24)
        ax1.set_xlabel('location (cm)',fontsize=24)

        # ax1.grid(True,axis='y',ls='--')


        # ax2.scatter([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], s=120,color=color_short, linewidths=0, zorder=2)
        # ax2.scatter([0.26,0.51,0.76], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], s=120,color=color_long, linewidths=0, zorder=2)
        # # ax2.scatter([0.26,0.51,0.76,1], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll),np.mean(std_pi_ll_long)], s=120,color=color_long, linewidths=0, zorder=2)
        # # ax2.scatter([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], s=120,color='0.5', linewidths=0, zorder=2)
        #
        # ax2.errorbar([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], yerr=[stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],fmt='none',ecolor='k', zorder=1)
        # ax2.errorbar([0.26,0.51,0.76], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], yerr=[stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],fmt='none',ecolor='k', zorder=1)
        # # ax2.errorbar([0.26,0.51,0.76,1], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll),np.mean(std_pi_ll_long)], yerr=[stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll),stats.sem(std_pi_ll_long)],fmt='none',ecolor='k', zorder=1)
        # # ax2.errorbar([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], yerr=[stats.sem(std_prelm_sl),stats.sem(std_lm_sl),stats.sem(std_pi_sl)],fmt='none',ecolor='k', zorder=1)
        #
        # ax2.plot([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], lw=4, c=color_short)
        # ax2.plot([0.26,0.51,0.76], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], lw=4, c=color_long)
        # # ax2.plot([0.76,1], [np.mean(std_pi_ll),np.mean(std_pi_ll_long)], lw=2, ls='--', c=color_long)
        # # ax2.plot([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], lw=2, c='0.5')
        #
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        #
        # ax2.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=20)
        # ax2.set_xlim([0.2,1.1])
        # ax2.set_xticks([0.26,0.51,0.76])
        # # ax2.set_ylim([0,10])
        # ax2.set_xticklabels(['pre-LM','LM','PI'])
        # ax2.set_ylabel('prediction error (cm)',fontsize=24)
        # ax2.set_xlabel('location (cm)',fontsize=24)

        fname = content['figure_output_path'] + subfolder + os.sep + j + '_' + str(popvec_cc_reconstruction_ss.shape[2]) + '.' + fformat
        try:
            fig.savefig(fname, format=fformat, dpi=300)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

        print(fname)


if __name__ == "__main__":

    # filenames
    # json_files = ['task_engaged_all_popvloc_results_100']#,'task_engaged_all_openloop_popvloc_results_100','task_engaged_V1_popvloc_results_100']#,'task_engaged_all_openloop_popvloc_results_100','task_engaged_V1_openloop_popvloc_results_100']
    # json_files = ['task_engaged_V1_popvloc_results_100']
    # json_files = ['task_engaged_all_openloop_popvloc_results_100']
    json_files = ['all_landmark_popvloc_results_3'] #, 'all_onset_popvloc_results_3']
    #
    # figure output parameters
    subfolder = 'popvloc'
    fname = 'popvec_cc_error'
    fformat = 'png'

    # print('SL:')
    sl_trialtype_diff(json_files, fname, fformat, subfolder)

    # json_files = ['active_all_openloop_popvloc_results_10']
    # sl_trialtype_diff(json_files, fname, fformat, subfolder)

    # json_files = ['task_engaged_all_vrol_popvloc_results_10','task_engaged_V1_vrol_popvloc_results_10']
    # subfolder = 'popvloc'
    # fname = 'popvec_cc_vrol_error'
    # fformat = 'png'
    #
    # print('SL:')
    # sl_trialtype_diff(json_files, fname, fformat, subfolder)

    # figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87],['LF170613_1','Day201784','Day201784_openloop',77]]
    #
    # #figure output parameters
    # subfolder = 'roi_fractions'
    # fname = 'roi_fractions_l23'
    # fformat = 'svg'
    #
    # print('TASK ENGAGED L2/3')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)
    #
    # figure_datasets = [['LF170222_1','Day20170615','Day20170615_openloop',96],['LF170420_1','Day20170719','Day20170719_openloop',95],['LF170421_2','Day20170719','Day20170719_openloop',68],['LF170421_2','Day20170720','Day20170720_openloop',45]]
    #
    # subfolder = 'roi_fractions'
    # fname = 'roi_fractions_l5'
    # fformat = 'svg'
    #
    # print('TASK ENGAGED L5:')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)

    # figure_datasets = [['LF170214_1','Day201777','Day201777_openloop',112],['LF170214_1','Day2017714','Day2017714_openloop',165],['LF171211_2','Day201852_openloop','Day201852',245]]
    # #
    # subfolder = 'roi_fractions'
    # fname = 'roi_fractions_V1'
    # fformat = 'png'
    # print('TASK ENGAGED V1:')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)
