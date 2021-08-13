"""
read .json file from population vector analysis for further analysis

"""

# %matplotlib inline

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings; warnings.simplefilter('ignore')
import yaml
import codecs, json, ipdb
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

    # track start bin is 0 because the input matrix we load is already cropped at track_start
    track_start_bin = 0

    # mean reconstruction error
    dev_all_ss = []
    dev_all_ll = []

    # values of the mean reconstruction error by bin
    reconstruction_graph_short = []
    reconstruction_graph_long = []

    # store values of cross correlation at real location (i.e. along the diagonal of the cc matrix)
    cc_at_loc_short = []
    cc_at_loc_long = []

    color_short = '#F58020'
    color_long = '#374D9E'

    # print(track_start_bin,lm_start, lm_end, end_bin_short, end_bin_long)

    for j in fig_datasets:
        print(j)
        fig = plt.figure(figsize=(12,18))
        fig_x = (tracklength_long - track_start) * 2 + 20
        ax1 = plt.subplot2grid((3,fig_x),(0,0),rowspan=1, colspan=260)
        ax2 = plt.subplot2grid((3,fig_x),(1,0),rowspan=1, colspan=260)
        ax3 = plt.subplot2grid((3,fig_x),(2,0),rowspan=1, colspan=260)
        ax4 = plt.subplot2grid((3,fig_x),(1,300),rowspan=1, colspan=260)
        ax5 = plt.subplot2grid((3,fig_x),(2,300),rowspan=1, colspan=260)
        ax6 = plt.subplot2grid((3,fig_x),(0,300),rowspan=1, colspan=130)
        ax7 = plt.subplot2grid((3,fig_x),(0,430),rowspan=1, colspan=130)

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

        with open(content['figure_output_path'] + subfolder + os.sep + j + '.json') as f:
            popvloc_result = json.load(f)

        # calculate deviation (step by step so even an idiot like myself doesn't get confused) of reconstruction vs actual location
        # ipdb.set_trace()
        popvec_cc_reconstruction_ss = np.array(popvloc_result['popvec_cc_reconstruction_ss'])
        ss_reconstruction_bybin = np.zeros((np.size(popvec_cc_reconstruction_ss,1),np.size(popvec_cc_reconstruction_ss,2)))
        ss_reconstruction_bybin_stderr = np.zeros((np.size(popvec_cc_reconstruction_ss,1)))
        for i in range(popvec_cc_reconstruction_ss.shape[2]):
            # calculate standard deviation by bin
            bin_diff = np.abs((popvec_cc_reconstruction_ss[1,:,i] - popvec_cc_reconstruction_ss[0,:,i]) * bin_size)
            # bin_diff = bin_diff * bin_diff
            # take square root to complete std calculation and multiply by bin size to get to centimeter
            # ss_reconstruction_bybin[:,i] = np.sqrt(bin_diff)
            ss_reconstruction_bybin[:,i] = bin_diff
            ss_reconstruction_bybin_stderr = np.sqrt(bin_diff) / np.sqrt(popvec_cc_reconstruction_ss.shape[2])

            std_prelm_ss = np.append(std_prelm_ss, (np.sum(ss_reconstruction_bybin[track_start_bin:lm_start,i])/lm_start))
            std_lm_ss = np.append(std_lm_ss, (np.sum(ss_reconstruction_bybin[lm_start:lm_end,i])/(lm_end-lm_start)))
            std_pi_ss = np.append(std_pi_ss, (np.sum(ss_reconstruction_bybin[lm_end:end_bin_short,i])/(end_bin_short-lm_end)))

        dev_all_ss.append(ss_reconstruction_bybin)

        popvec_cc_reconstruction_ll = np.array(popvloc_result['popvec_cc_reconstruction_ll'])
        std_all_ll = np.zeros((popvec_cc_reconstruction_ll.shape[0]))
        ll_reconstruction_bybin = np.zeros((np.size(popvec_cc_reconstruction_ll,1),np.size(popvec_cc_reconstruction_ll,2)))
        ll_reconstruction_bybin_stderr = np.zeros((np.size(popvec_cc_reconstruction_ll,1)))
        for i in range(popvec_cc_reconstruction_ll.shape[2]):
            bin_diff = np.abs((popvec_cc_reconstruction_ll[1,:,i] - popvec_cc_reconstruction_ll[0,:,i]) * bin_size)
            # bin_diff = bin_diff * bin_diff
            # ll_reconstruction_bybin[:,i] = np.sqrt(bin_diff) * bin_size
            ll_reconstruction_bybin[:,i] = bin_diff
            ll_reconstruction_bybin_stderr = np.sqrt(bin_diff) / np.sqrt(popvec_cc_reconstruction_ll.shape[2])

            std_prelm_ll = np.append(std_prelm_ll, (np.sum(ll_reconstruction_bybin[track_start_bin:lm_start,i])/lm_start))
            std_lm_ll = np.append(std_lm_ll, (np.sum(ll_reconstruction_bybin[lm_start:lm_end,i])/(lm_end-lm_start)))
            std_pi_ll = np.append(std_pi_ll, (np.sum(ll_reconstruction_bybin[lm_end:end_bin_long,i])/(end_bin_long-lm_end)))

            std_pi_ll_long = np.append(std_pi_ll_long,np.sum(ll_reconstruction_bybin[end_bin_short:end_bin_long,i])/(end_bin_long-end_bin_short))
            # std_pi_ll_long = np.append(std_pi_ll_long,np.sqrt(np.sum(bin_diff[end_bin_short:end_bin_long])/(end_bin_long-end_bin_short)))
            # ll_reconstruction_bybin[:,i] = np.abs(popvec_cc_reconstruction_ll[1,:,i] - popvec_cc_reconstruction_ll[0,:,i]) * bin_size

        dev_all_ll.append(ll_reconstruction_bybin)

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

        # print(ss_reconstruction_bybin.shape)
        ss_reconstruction_bybin_graph = np.nanmedian(ss_reconstruction_bybin, axis=1)
        ll_reconstruction_bybin_graph = np.nanmedian(ll_reconstruction_bybin, axis=1)
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
        # print(posthoc_res_ss)


        # f_value_sl, p_value_sl = stats.f_oneway(std_prelm_sl,std_lm_sl,std_pi_sl)
        # group_labels = ['prelm'] * len(std_prelm_sl) + ['lm'] * len(std_lm_sl) + ['pi'] * len(std_pi_sl)
        # mc_res_sl = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_sl,std_lm_sl,std_pi_sl)),group_labels)
        # posthoc_res_sl = mc_res_sl.tukeyhsd()
        #
        # f_value_ll, p_value_ll = stats.f_oneway(std_prelm_ll,std_lm_ll,std_pi_ll)
        # group_labels = ['prelm'] * len(std_prelm_ll) + ['lm'] * len(std_lm_ll) + ['pi'] * len(std_pi_ll)
        # mc_res_ll = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ll,std_lm_ll,std_pi_ll)),group_labels)
        # posthoc_res_ll = mc_res_ll.tukeyhsd()

        # calculate the standard deviation of each row
        popvec_cc_matrix_ss_mean = np.array(popvloc_result['popvec_cc_matrix_ss_mean'])
        popvec_cc_matrix_ll_mean = np.array(popvloc_result['popvec_cc_matrix_ll_mean'])

        mat_size_short = popvec_cc_matrix_ss_mean.shape[0]
        mat_size_long = popvec_cc_matrix_ll_mean.shape[0]
        cc_at_loc_short.append(popvec_cc_matrix_ss_mean[np.arange(mat_size_short), np.arange(mat_size_short)])
        cc_at_loc_long.append(popvec_cc_matrix_ll_mean[np.arange(mat_size_long), np.arange(mat_size_long)])


        # print(popvec_cc_matrix_ss_mean.shape)
        ax2_img = ax2.pcolor(np.flipud(popvec_cc_matrix_ss_mean),cmap='viridis', vmax=1.0)
        fig.colorbar(ax2_img,ax=ax6)

        ax2.set_xlim([0,popvec_cc_matrix_ss_mean.shape[1]])
        ax2.set_ylim([0,popvec_cc_matrix_ss_mean.shape[0]])
        # ax2.axvline(5, lw=4, ls='--', c='#39B54A')
        # ax2.axvline(24, lw=4, ls='--', c='#FF0000')
        # ax2.axvline(40, lw=4, ls='--', c='#2EA7DF')

        ax2.set_yticks([2,24,44])
        ax2.set_xticks([2,24,44])
        ax2.set_xticklabels(['10','120','220'])
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=17, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='on')


        # ipdb.set_trace()
        ax3.plot(popvec_cc_matrix_ss_mean[:,2], lw=4, c='#39B54A')
        ax3.plot(popvec_cc_matrix_ss_mean[:,24], lw=4, c='#FF0000')
        ax3.plot(popvec_cc_matrix_ss_mean[:,44], lw=4, c='#2EA7DF')

        # print('--- SLICE STDEVs SHORT ---')
        # print(np.std(popvec_cc_matrix_ss_mean[:,5][5:]))
        # print(np.std(popvec_cc_matrix_ss_mean[:,24]))
        # print(np.std(popvec_cc_matrix_ss_mean[:,41][:41]))
        # print('--------------------------')

        ax3.set_xlim([0,45])
        ax3.set_ylim([-0.2,1])

        ax3.set_xticks([2,24,44])
        ax3.set_xticklabels(['10','120','220'])

        ax3.spines['bottom'].set_linewidth(2)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_linewidth(1)
        ax3.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=17, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')

        ax4_img = ax4.pcolor(np.flipud(popvec_cc_matrix_ll_mean),cmap='viridis', vmax=1.0)
        fig.colorbar(ax4_img,ax=ax7)
        ax4.set_xlim([0,popvec_cc_matrix_ll_mean.shape[1]])
        ax4.set_ylim([0,popvec_cc_matrix_ll_mean.shape[0]])
        # ax4.axvline(5, lw=4, ls='--', c='#39B54A')
        # ax4.axvline(24, lw=4, ls='--', c='#FF0000')
        # ax4.axvline(53, lw=4, ls='--', c='#2EA7DF')

        ax4.set_yticks([2,24,56])
        ax4.set_xticks([2,24,56])
        ax4.set_xticklabels(['10','120','280'])
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=17, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='on')

        ax5.plot(popvec_cc_matrix_ll_mean[:,2], lw=4, c='#39B54A')
        ax5.plot(popvec_cc_matrix_ll_mean[:,24], lw=4, c='#FF0000')
        ax5.plot(popvec_cc_matrix_ll_mean[:,56], lw=4, c='#2EA7DF')
        ax5.set_xlim([0,57])
        ax5.set_ylim([-0.2,1])
        ax5.set_xticks([2,24,56])
        ax5.set_xticklabels(['10','120','280'])

        # print('--- SLICE STDEVs LONG ---')
        # print(np.std(popvec_cc_matrix_ll_mean[:,5][5:]))
        # print(np.std(popvec_cc_matrix_ll_mean[:,22]))
        # print(np.std(popvec_cc_matrix_ll_mean[:,53][:53]))
        # print('-------------------------')

        ax5.spines['bottom'].set_linewidth(2)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['left'].set_linewidth(1)
        ax5.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=17, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')

        # ax3_img = ax3.pcolor(popvec_cc_matrix_ll_mean.T,cmap='viridis')
        # ax3.set_xlim([0,popvec_cc_matrix_ll_mean.shape[1]])
        # ax3.set_ylim([0,popvec_cc_matrix_ll_mean.shape[0]])

        # ax2 = plt.subplot(212)
        boxcar_len = 3
        boxcar = np.ones(boxcar_len)/boxcar_len
        ss_smooth = np.convolve(ss_reconstruction_bybin_graph, boxcar, mode='same')
        ax1.plot(ss_smooth,c=color_short,lw=4)
        reconstruction_graph_short.append(ss_smooth)
        # ax1.plot(ss_reconstruction_bybin_graph,c=color_short,lw=4)
        # ax1.fill_between(np.arange(ss_reconstruction_bybin_stderr.shape[0]), ss_reconstruction_bybin_graph,ss_reconstruction_bybin_graph-ss_reconstruction_bybin_stderr,alpha=0.3,color=color_short)
        # ax1.fill_between(np.arange(ss_reconstruction_bybin_stderr.shape[0]), ss_reconstruction_bybin_graph,ss_reconstruction_bybin_graph+ss_reconstruction_bybin_stderr,alpha=0.3,color=color_short)

        ll_smooth = np.convolve(ll_reconstruction_bybin_graph, boxcar, mode='same')
        ax1.plot(ll_smooth,c=color_long,lw=4)
        reconstruction_graph_long.append(ll_smooth)

        # ax1.axvline(50/bin_size,c='0.5',ls='--',lw=2)
        ax1.axvline(lm_start,c='0.5',ls='--',lw=2)
        ax1.axvline(lm_end,c='0.5',ls='--',lw=2)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)

        ax1.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=16)

        ax1.set_xticks([20,30,44,56])
        ax1.set_xticklabels(['200','240','320','360'])
        ax1.set_ylabel('prediction error (cm)',fontsize=18)
        ax1.set_xlabel('location (cm)',fontsize=18)

        fname = content['figure_output_path'] + subfolder + os.sep + j + '_' + str(popvec_cc_reconstruction_ss.shape[2]) + '.' + fformat
        try:
            fig.savefig(fname, format=fformat, dpi=300)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

        print(fname)
        plt.close(fig)

    # compare landmark aligned vs trial onset aligned
    if np.array(dev_all_ss).shape[0] > 1:
        # print('whoop')
        dev_all_ss_lm = np.median(np.array(dev_all_ss)[0,:,:],axis=1)
        dev_all_ss_on = np.median(np.array(dev_all_ss)[1,:,:],axis=1)

        dev_all_ll_lm = np.median(np.array(dev_all_ll)[0,:,:],axis=1)
        dev_all_ll_on = np.median(np.array(dev_all_ll)[1,:,:],axis=1)

        fig = plt.figure(figsize=(7,4))
        ax1 = plt.subplot(122)
        ax2 = plt.subplot(121)
        # ax3 = plt.subplot(133)

        # ax1.scatter(dev_all_ss_lm[0:lm_start], dev_all_ss_on[0:lm_start], c='#39B54A')
        # ax1.scatter(dev_all_ss_lm[lm_start:lm_end], dev_all_ss_on[lm_start:lm_end], c='#FF0000')
        # ax1.scatter(dev_all_ss_lm[lm_end:end_bin_short], dev_all_ss_on[lm_end:end_bin_short], c='#2EA7DF')
        #
        # ax1.plot([0,16],[0,16], lw=2, c='k', ls='--')

        ax1.bar([1,1.8], [np.mean(dev_all_ss_lm),np.mean(dev_all_ss_on)], width=0.5, yerr=[stats.sem(dev_all_ss_lm),stats.sem(dev_all_ss_on)], edgecolor=[color_short], linewidth=4, color=[color_short, 'white'], ecolor='k')
        ax1.bar([3.5,4.3], [np.mean(dev_all_ll_lm),np.mean(dev_all_ll_on)], width=0.5, yerr=[stats.sem(dev_all_ll_lm),stats.sem(dev_all_ll_on)], edgecolor=[color_long], linewidth=4, color=[color_long, 'white'], ecolor='k')
        ax1.set_ylabel('Mean reconstruction error (cm)', fontsize=16)

        print('--- RECONSTRUCTION ERROR ---')
        print('mean reconstruction error short LM: ', str(np.mean(dev_all_ss_lm)), '+/- ', str(stats.sem(dev_all_ss_lm)))
        print('mean reconstruction error short ON: ', str(np.mean(dev_all_ss_on)), '+/- ', str(stats.sem(dev_all_ss_on)))
        print('mean reconstruction error long LM: ', str(np.mean(dev_all_ll_lm)), '+/- ', str(stats.sem(dev_all_ll_lm)))
        print('mean reconstruction error long ON: ', str(np.mean(dev_all_ll_on)), '+/- ', str(stats.sem(dev_all_ll_on)))
        print('----------------------------')

        # print(np.array(cc_at_loc_short).shape)
        ax2.plot(np.array(cc_at_loc_short)[0,:-2], lw=2, c=color_short)
        ax2.plot(np.array(cc_at_loc_long)[0,:-2], lw=2, c=color_long)
        ax2.plot(np.array(cc_at_loc_short)[1,:-2], lw=2, ls=':', c=color_short)
        ax2.plot(np.array(cc_at_loc_long)[1,:-2], lw=2, ls=':', c=color_long)
        # ax2.set_ylim([0.5,0.9])
        ax2.set_ylim([0.2,1.0])
        ax2.set_xlabel('Location (cm)', fontsize=18)
        ax2.set_ylabel('CC at location', fontsize=18)

        # ax3.bar([1,1.8],[np.nanmean(np.array(cc_at_loc_short)[0,:-2]),np.nanmean(np.array(cc_at_loc_short)[1,:-2])],edgecolor=[color_short], linewidth=4, color=[color_short, 'white'], ecolor='k')

        # ax2.plot(np.array(reconstruction_graph_short)[0,:], lw=2, c=color_short)
        # ax2.plot(np.array(reconstruction_graph_long)[0,:], lw=2, c=color_long)
        # ax2.plot(np.array(reconstruction_graph_short)[1,:], lw=2, ls=':', c=color_short)
        # ax2.plot(np.array(reconstruction_graph_long)[1,:], lw=2, ls=':', c=color_long)

        print('--- STATS ---')
        # print(stats.f_oneway(dev_all_ss_lm,dev_all_ss_on,dev_all_ll_lm,dev_all_ll_on))
        #
        # group_labels = ['group_short_1'] * dev_all_ss_lm.shape[0] + \
        #                ['group_short_2'] * dev_all_ss_on.shape[0] + \
        #                ['group_long_1'] * dev_all_ll_lm.shape[0] + \
        #                ['group_long_2'] * dev_all_ll_on.shape[0]
        #
        # mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((dev_all_ss_lm,dev_all_ss_on,dev_all_ll_lm,dev_all_ll_on)),group_labels)
        # posthoc_res_ss = mc_res_ss.tukeyhsd()
        # print(posthoc_res_ss)

        # print(stats.ttest_ind(dev_all_ss_lm,dev_all_ss_on))
        # print(stats.ttest_ind(dev_all_ll_lm,dev_all_ll_on))
        print(stats.mannwhitneyu(dev_all_ss_lm,dev_all_ss_on))
        print(stats.mannwhitneyu(dev_all_ll_lm,dev_all_ll_on))
        print('-------------')

        # print(stats.ttest_ind([dev_all_ss_lm,dev_all_ll_lm],[dev_all_ss_on, dev_all_ll_on]))

        # ax2.set_xlim([0,45])

        ax1.set_xlim([0.5,5.5])
        ax1.set_xticks([1.675,4.35])
        ax1.set_xticklabels(['short trials','long trials'], rotation=45)

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

        ax2.spines['bottom'].set_linewidth(2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(2)
        ax2.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=14, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')

        ax2.set_xticks([2,24,44,56])
        ax2.set_xticklabels(['10','120','220','280'])
        # ax2.set_ylim([0.25,0.9])
        # ax2.set_yticks([0.3,0.5,0.7,0.9])
        # ax2.set_yticklabels(['0.30','0.50','0.70','0.90'])
        ax2.set_yticks([0.5,0.6,0.7,0.8,0.9])
        ax2.set_yticklabels(['0.5','0.6','0.7','0.8','0.9'])
        ax2.set_ylim([0.5,0.9])


        plt.tight_layout()

        fname = content['figure_output_path'] + subfolder + os.sep + j + '_' + str(popvec_cc_reconstruction_ss.shape[2]) + 'lmvr.' + fformat
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
    # json_files = ['all_landmark_popvloc_results_10', 'all_onset_popvloc_results_10']
    # json_files = ['active_v1_popvloc_ results_100','active_v1_openloop_popvloc_results_100']
    # json_files = ['all_landmark_popvloc_results_100','active_all_openloop_popvloc_results_100']
    json_files = ['all_landmark_short_popvloc_results_100','all_landmark_long_popvloc_results_100']
    # json_files = ['all_landmark_naiveexpert_popvloc_results_naiveexpert_10']
    # json_files = ['all_landmark_expertmatched_popvloc_results_5','all_landmark_naivematched_popvloc_results_5']
    # json_files = ['all_landmark_naiveexpert_union_expexp_popvloc_results_naiveexpert_10', 'all_landmark_naiveexpert_union_nainai_popvloc_results_naiveexpert_10']
    # json_files = ['all_landmark_matchedexpert_popvloc_results_5','all_landmark_matchednaive_popvloc_results_5']
    # json_files = ['all_landmark_expert_popvloc_results_100',  'active_all_openloop_popvloc_results_100']
    # json_files = ['all_landmark_expert_popvloc_results_10',  'all_onset_expert_popvloc_results_10']
    # json_files = ['all_landmark_expert_popvloc_results_10',  'active_all_openloop_popvloc_results_10']

    # figure output parameters
    subfolder = 'popvloc'
    fname = 'popvec_cc_error'
    fformat = 'png'

    # print('SL:')
    sl_trialtype_diff(json_files, fname, fformat, subfolder)
    #
    # json_files = ['all_onset_popvloc_results_5']

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
