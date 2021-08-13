"""
Plot trace of an individual ROI vs location. Include plots with
average firing rate of deconvolved trace

@author: lukasfischer

"""

import numpy as np
import h5py
import os
import sys
import traceback
import yaml
import warnings; warnings.simplefilter('ignore')
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from scipy import stats
import json
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')
from filter_trials import filter_trials
from write_dict import write_dict

def fig_dfloc_trace_roiparams(h5path, sess, roi, fname, filterprops_short_1, filterprops_short_2, filterprops_long_1, filterprops_long_2, fformat='png', subfolder=[], c_ylim=[]):
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    spikes_ds = np.copy(h5dat[sess + '/spiketrain'])
    h5dat.close()



    binnr_short = 80
    binnr_long = 100
    binnr_short_spikes = 80
    binnr_long_spikes = 100
    binnr_dark = 100
    tracklength_short = 400
    tracklength_long = 500
    maxdistance_dark = 500
    track_short = 3
    track_long = 4
    track_dark = 5

    frame_latency = 1/(dF_ds.shape[0]/(behav_ds[-1,0] - behav_ds[0,0]))

    # number of standard deviations the roi signal has to exceed to be counted as active
    trial_std_threshold = 3

    # minimum number of trials a roi as to be active in for some stats to be calculated
    MIN_ACTIVE_TRIALS = 5

    # calcluate standard deviation of ROI traces
    roi_std = np.std(dF_ds[:,roi])

    # create figure to later plot on
    fig = plt.figure(figsize=(10,14))
    ax1 = plt.subplot2grid((8,4),(0,0),colspan=2,rowspan=2)
    ax2 = plt.subplot2grid((8,4),(0,2),colspan=2,rowspan=2)
    ax3 = plt.subplot2grid((8,4),(2,0),colspan=2,rowspan=2)
    ax4 = plt.subplot2grid((8,4),(2,2),colspan=2,rowspan=2)
    ax5 = plt.subplot2grid((8,4),(7,0),colspan=1,rowspan=1)
    ax6 = plt.subplot2grid((8,4),(7,1),colspan=1,rowspan=1)
    ax7 = plt.subplot2grid((8,4),(7,2),colspan=1,rowspan=1)
    ax8 = plt.subplot2grid((8,4),(7,3),colspan=1,rowspan=1)
    ax9 = plt.subplot2grid((8,4),(6,0),colspan=1,rowspan=1)
    ax10 = plt.subplot2grid((8,4),(6,1),colspan=1,rowspan=1)
    ax11 = plt.subplot2grid((8,4),(6,2),colspan=1,rowspan=1)
    ax12 = plt.subplot2grid((8,4),(6,3),colspan=1,rowspan=1)
    ax13 = plt.subplot2grid((8,4),(4,0),colspan=2,rowspan=2)
    ax14 = plt.subplot2grid((8,4),(4,2),colspan=2,rowspan=2)

    # plot landmark and reward zone as shaded areas
    ax1.axvspan(40,48,color='0.9',zorder=0)
    ax1.axvspan(64,68,color=sns.xkcd_rgb["windows blue"],alpha=0.2,zorder=0)
    ax2.axvspan(40,48,color='0.9',zorder=0)
    ax2.axvspan(76,80,color=sns.xkcd_rgb["dusty purple"],alpha=0.2,zorder=0)
    ax13.axvspan(40,48,color='0.9',zorder=0)
    ax13.axvspan(64,68,color=sns.xkcd_rgb["windows blue"],alpha=0.2,zorder=0)
    ax14.axvspan(40,48,color='0.9',zorder=0)
    ax14.axvspan(76,80,color=sns.xkcd_rgb["dusty purple"],alpha=0.2,zorder=0)

    ax9.axvspan(40,48,color='0.9',zorder=0)
    ax9.axvspan(64,68,color=sns.xkcd_rgb["windows blue"],alpha=0.2,zorder=0)
    ax10.axvspan(40,48,color='0.9',zorder=0)
    ax10.axvspan(64,68,color=sns.xkcd_rgb["windows blue"],alpha=0.2,zorder=0)

    ax11.axvspan(40,48,color='0.9',zorder=0)
    ax11.axvspan(76,80,color=sns.xkcd_rgb["dusty purple"],alpha=0.2,zorder=0)
    ax12.axvspan(40,48,color='0.9',zorder=0)
    ax12.axvspan(76,80,color=sns.xkcd_rgb["dusty purple"],alpha=0.2,zorder=0)

    # set axis labels
    ax1.set_xticks([0,20,40,60,80])
    ax1.set_xticklabels(['0','100','200','300','400'])
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Roi response on SHORT trials')
    ax2.set_xticks([0,20,40,60,80,100])
    ax2.set_xticklabels(['0','100','200','300','400','500'])
    ax2.set_xlabel('Location (cm)')
    ax2.set_title('Roi response on LONG trials')
    ax1.set_xlim([10,68])
    ax2.set_xlim([10,80])

    ax13.set_xticks([0,20,40,60,80])
    ax13.set_xticklabels(['0','100','200','300','400'])
    ax13.set_xlabel('Location (cm)')
    ax13.set_title('Roi response on SHORT trials')
    ax14.set_xticks([0,20,40,60,80,100])
    ax14.set_xticklabels(['0','100','200','300','400','500'])
    ax14.set_xlabel('Location (cm)')
    ax14.set_title('Roi response on LONG trials')
    ax13.set_xlim([10,68])
    ax14.set_xlim([10,80])

    # ax9.set_title('short, successful')
    # ax10.set_title('short, unsuccessful')
    # ax11.set_title('long, successful')
    # ax12.set_title('long, unsuccessful')

    # set axes visibility
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.spines['left'].set_visible(False)
    ax9.spines['bottom'].set_visible(False)
    ax9.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.spines['left'].set_visible(False)
    ax10.spines['bottom'].set_visible(False)
    ax10.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['left'].set_visible(False)
    ax11.spines['bottom'].set_visible(False)
    ax11.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax12.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['left'].set_visible(False)
    ax12.spines['bottom'].set_visible(False)
    ax12.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax13.spines['top'].set_visible(False)
    ax13.spines['right'].set_visible(False)
    ax13.spines['left'].set_visible(False)
    ax13.spines['bottom'].set_visible(False)
    ax13.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax14.spines['top'].set_visible(False)
    ax14.spines['right'].set_visible(False)
    ax14.spines['left'].set_visible(False)
    ax14.spines['bottom'].set_visible(False)
    ax14.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    # plot lines indicating landmark and reward zone in heatmaps
    ax3.axvline(40,c='0.8',lw=2)
    ax3.axvline(48,c='0.8',lw=2)
    ax3.axvline(64,c='r',lw=2)

    ax4.axvline(40,c='0.8',lw=2)
    ax4.axvline(48,c='0.8',lw=2)
    ax4.axvline(76,c='r',lw=2)

    ax5.axvline(40,c='0.8',lw=2)
    ax5.axvline(48,c='0.8',lw=2)
    ax5.axvline(64,c='r',lw=2)

    ax6.axvline(40,c='0.8',lw=2)
    ax6.axvline(48,c='0.8',lw=2)
    ax6.axvline(64,c='r',lw=2)

    ax7.axvline(40,c='0.8',lw=2)
    ax7.axvline(48,c='0.8',lw=2)
    ax7.axvline(76,c='r',lw=2)

    ax8.axvline(40,c='0.8',lw=2)
    ax8.axvline(48,c='0.8',lw=2)
    ax8.axvline(76,c='r',lw=2)

    # pull out trial numbers of respective sections
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])
    mean_dF_short = np.zeros((np.size(trials_short,0),binnr_short))
    mean_dF_long = np.zeros((np.size(trials_long,0),binnr_long))
    spikerate_short = np.zeros((np.size(trials_short,0),binnr_short_spikes))
    spikerate_long = np.zeros((np.size(trials_long,0),binnr_long_spikes))
    cur_trial_max_idx_short = np.empty(0)
    cur_trial_max_idx_long = np.empty(0)
    # run through SHORT trials and calculate avg dF/F for each bin and trial
    for i,t in enumerate(trials_short):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                               (0.0, tracklength_short))[0]
        mean_dF_short[i,:] = mean_dF_trial
        if np.nanmax(mean_dF_trial) > trial_std_threshold * roi_std:
            cur_trial_max_idx_short = np.append(cur_trial_max_idx_short,np.nanargmax(mean_dF_trial))
        ax1.plot(mean_dF_trial,c='0.8')

    sem_dF_s = stats.sem(mean_dF_short,0,nan_policy='omit')
    # avg_mean_dF_short = np.nanmean(mean_dF_short,axis=0)
    # ax1.plot(avg_mean_dF_short,c=sns.xkcd_rgb["windows blue"],lw=3) #'#00AAAA'
    ax1.axhline(trial_std_threshold * roi_std, ls='--', lw=2, c='0.8')

    if len(cur_trial_max_idx_short) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction_short = len(cur_trial_max_idx_short)/np.size(trials_short,0)
        sns.distplot(cur_trial_max_idx_short,hist=False,kde=False,rug=True,ax=ax1)
        roi_std_short = np.std(cur_trial_max_idx_short)
    else:
        roi_active_fraction_short = -1
        roi_std_short = -1


    roi_std_short = np.std(cur_trial_max_idx_short)
    # ax1.fill_between(np.arange(len(avg_mean_dF_short)), np.nanmean(mean_dF_short,axis=0) - sem_dF_s, np.nanmean(mean_dF_short,axis=0) + sem_dF_s, color = sns.xkcd_rgb["windows blue"], alpha = 0.2)

    # run through SHORT trials and calculate avg SPIKERATE for each bin and trial
    for i,t in enumerate(trials_short):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = spikes_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'sum', binnr_short_spikes,
                                               (0.0, tracklength_short))[0]
        mean_dF_trial_count = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'count', binnr_short_spikes,
                                               (0.0, tracklength_short))[0]
        spikerate_short[i,:] = (mean_dF_trial/mean_dF_trial_count)/frame_latency
        if np.nanmax(mean_dF_trial) > trial_std_threshold * roi_std:
            cur_trial_max_idx_short = np.append(cur_trial_max_idx_short,np.nanargmax(mean_dF_trial))
        # ax13.plot(spikerate_short[i,:],c='0.8',lw=0.5)

    # sem_dF_s = stats.sem(spikerate_short,0,nan_policy='omit')
    # avg_spikerate_short = np.nanmean(spikerate_short,axis=0)
    # ax13.plot(avg_spikerate_short,c=sns.xkcd_rgb["windows blue"],lw=3) #'#00AAAA'
    # ax13.axhline(trial_std_threshold * roi_std, ls='--', lw=2, c='0.8')

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(mean_dF_short.T):
        if np.count_nonzero(np.isnan(trace))/trace.shape[0] < 0.5:
            mean_valid_indices.append(i)
    roi_meanpeak_short = np.nanmax(np.nanmean(mean_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_short_idx = np.nanargmax(np.nanmean(mean_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_short_location = (roi_meanpeak_short_idx+mean_valid_indices[0]) * (tracklength_short/binnr_short)
    mean_trace_short = np.nanmean(mean_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    mean_spikerate_trace_short = np.nanmean(spikerate_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)

    ax1.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_trace_short,c=sns.xkcd_rgb["windows blue"],lw=3)
    ax1.axvline((roi_meanpeak_short_idx+mean_valid_indices[0]),c='b')

    sem_dF_s = stats.sem(spikerate_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0,nan_policy='omit')
    ax13.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_spikerate_trace_short,c=sns.xkcd_rgb["windows blue"],lw=3)
    ax13.fill_between(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_spikerate_trace_short - sem_dF_s, mean_spikerate_trace_short + sem_dF_s, color = '0.8', alpha = 0.2)

    mean_trace_short_start = mean_valid_indices[0]
    max_y_deconv = np.amax(mean_spikerate_trace_short+sem_dF_s)


    # run through LONG trials and calculate avg dF/F for each bin and trial
    for i,t in enumerate(trials_long):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                               (0.0, tracklength_long))[0]
        mean_dF_long[i,:] = mean_dF_trial
        if np.nanmax(mean_dF_trial) > trial_std_threshold * roi_std:
            cur_trial_max_idx_long = np.append(cur_trial_max_idx_long,np.nanargmax(mean_dF_trial))
        ax2.plot(mean_dF_trial,c='0.8')

    sem_dF_l = stats.sem(mean_dF_long,0,nan_policy='omit')
    # avg_mean_dF_long = np.nanmean(mean_dF_long,axis=0)
    # ax2.plot(avg_mean_dF_long,c=sns.xkcd_rgb["dusty purple"],lw=3) #'#FF00FF'
    ax2.axhline(trial_std_threshold * roi_std, ls='--', lw=2, c='0.8')

    if len(cur_trial_max_idx_long) >= MIN_ACTIVE_TRIALS:
        roi_active_fraction_long = len(cur_trial_max_idx_long)/np.size(trials_long,0)
        sns.distplot(cur_trial_max_idx_long,hist=False,kde=False,rug=True,ax=ax2)
        roi_std_long = np.std(cur_trial_max_idx_long)
    else:
        roi_active_fraction_long = -1
        roi_std_long = -1

    # run through SHORT trials and calculate avg SPIKERATE for each bin and trial
    for i,t in enumerate(trials_long):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = spikes_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'sum', binnr_long_spikes,
                                               (0.0, tracklength_long))[0]
        mean_dF_trial_count = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'count', binnr_long_spikes,
                                               (0.0, tracklength_long))[0]
        spikerate_long[i,:] = (mean_dF_trial/mean_dF_trial_count)/frame_latency
        if np.nanmax(mean_dF_trial) > trial_std_threshold * roi_std:
            cur_trial_max_idx_long = np.append(cur_trial_max_idx_long,np.nanargmax(mean_dF_trial))
        # ax14.plot(spikerate_short[i,:],c='0.8')

    # ax2.fill_between(np.arange(len(avg_mean_dF_long)), avg_mean_dF_long - sem_dF_l, avg_mean_dF_long + sem_dF_l, color = sns.xkcd_rgb["dusty purple"], alpha = 0.2)
    mean_valid_indices = []
    for i,trace in enumerate(mean_dF_long.T):
        if np.count_nonzero(np.isnan(trace))/trace.shape[0] < 0.5:
            mean_valid_indices.append(i)
    roi_meanpeak_long = np.nanmax(np.nanmean(mean_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_long_idx = np.nanargmax(np.nanmean(mean_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_long_location = (roi_meanpeak_long_idx+mean_valid_indices[0]) * (tracklength_long/binnr_long)
    mean_trace_long = np.nanmean(mean_dF_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    mean_spikerate_trace_long = np.nanmean(spikerate_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    ax2.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_trace_long,c=sns.xkcd_rgb["dusty purple"],lw=3)
    ax2.axvline((roi_meanpeak_long_idx+mean_valid_indices[0]),c='b')

    sem_dF_s = stats.sem(spikerate_long[:,mean_valid_indices[0]:mean_valid_indices[-1]],0,nan_policy='omit')
    ax14.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_spikerate_trace_long,c=sns.xkcd_rgb["dusty purple"],lw=3)
    ax14.fill_between(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_spikerate_trace_long - sem_dF_s, mean_spikerate_trace_long + sem_dF_s, color = '0.8', alpha = 0.2)

    # ax14.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_spikerate_trace_long,c=sns.xkcd_rgb["dusty purple"],lw=3)
    mean_trace_long_start = mean_valid_indices[0]
    max_y_deconv = np.amax([max_y_deconv,np.amax(mean_spikerate_trace_long+sem_dF_s)])


    # run through SHORT SUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_short_succ = filter_trials( behav_ds, [], filterprops_short_1, trials_short)
    mean_dF_short_succ = np.zeros((np.size(trials_short_succ,0),binnr_short))
    for i,t in enumerate(trials_short_succ):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                               (0.0, tracklength_short))[0]
        mean_dF_short_succ[i,:] = mean_dF_trial
        ax9.plot(mean_dF_trial,c='0.8')

    filtered_short_1_mean_trace = np.nanmean(mean_dF_short_succ,axis=0)
    ax9.plot(filtered_short_1_mean_trace,c=sns.xkcd_rgb["windows blue"],ls='-',lw=2) #'#00AAAA'

    # run through SHORT UNSUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_short_unsucc = filter_trials( behav_ds, [], filterprops_short_2, trials_short)
    mean_dF_short_unsucc = np.zeros((np.size(trials_short_unsucc,0),binnr_short))
    for i,t in enumerate(trials_short_unsucc):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                               (0.0, tracklength_short))[0]
        mean_dF_short_unsucc[i,:] = mean_dF_trial
        ax10.plot(mean_dF_trial,c='0.8')

    filtered_short_2_mean_trace = np.nanmean(mean_dF_short_unsucc,axis=0)
    ax10.plot(filtered_short_2_mean_trace,c=sns.xkcd_rgb["windows blue"],ls='--',lw=2)

        # run through LONG SUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_long_succ = filter_trials( behav_ds, [], filterprops_long_1, trials_long)
    mean_dF_long_succ = np.zeros((np.size(trials_long_succ,0),binnr_long))
    for i,t in enumerate(trials_long_succ):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                               (0.0, tracklength_long))[0]
        mean_dF_long_succ[i,:] = mean_dF_trial
        ax11.plot(mean_dF_trial,c='0.8')
    filtered_long_1_mean_trace = np.nanmean(mean_dF_long_succ,axis=0)
    ax11.plot(filtered_long_1_mean_trace,c=sns.xkcd_rgb["dusty purple"],ls='-',lw=2) #'#00AAAA'

    # run through long UNSUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_long_unsucc = filter_trials( behav_ds, [], filterprops_long_2, trials_long)
    mean_dF_long_unsucc = np.zeros((np.size(trials_long_unsucc,0),binnr_long))
    for i,t in enumerate(trials_long_unsucc):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                               (0.0, tracklength_long))[0]
        mean_dF_long_unsucc[i,:] = mean_dF_trial
        ax12.plot(mean_dF_trial,c='0.8')
    filtered_long_2_mean_trace = np.nanmean(mean_dF_long_unsucc,axis=0)
    ax12.plot(filtered_long_2_mean_trace,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=2)

    ax9.set_title(filterprops_short_1)
    ax10.set_title(filterprops_short_2)
    ax11.set_title(filterprops_long_1)
    ax12.set_title(filterprops_long_2)

    # determine scaling of y-axis max values
    max_y_short = np.nanmax(np.nanmax(mean_dF_short))
    max_y_long = np.nanmax(np.nanmax(mean_dF_long))
    max_y = np.amax([max_y_short, max_y_long])
    heatmap_max = np.amax([np.nanmax(np.nanmean(mean_dF_short,axis=0)),np.nanmax(np.nanmean(mean_dF_long,axis=0))]) #+ 1

    if c_ylim != []:
        hmmin = c_ylim[0]
        hmmax = c_ylim[2]
    else:
        hmmin = 0
        hmmax = heatmap_max


    # plot heatmaps
    if hmmax >= 0:
        sns.heatmap(mean_dF_short,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_short.astype('int'),xticklabels=True,ax=ax3)
        sns.heatmap(mean_dF_long,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=False,xticklabels=False,ax=ax4)

        sns.heatmap(mean_dF_short_succ,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_short_succ.astype('int'),xticklabels=True,ax=ax5)
        sns.heatmap(mean_dF_short_unsucc,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_short_unsucc.astype('int'),xticklabels=True,ax=ax6)
        sns.heatmap(mean_dF_long_succ,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_long_succ.astype('int'),xticklabels=True,ax=ax7)
        sns.heatmap(mean_dF_long_unsucc,cbar=True,vmin=0,vmax=hmmax,cmap='viridis',yticklabels=trials_long_unsucc.astype('int'),xticklabels=True,ax=ax8)
    else:
        sns.heatmap(mean_dF_short,cbar=True,cmap='viridis',yticklabels=trials_short.astype('int'),xticklabels=True,ax=ax3)
        sns.heatmap(mean_dF_long,cbar=True,cmap='viridis',yticklabels=False,xticklabels=False,ax=ax4)

        sns.heatmap(mean_dF_short_succ,cbar=True,cmap='viridis',yticklabels=trials_short_succ.astype('int'),xticklabels=True,ax=ax5)
        sns.heatmap(mean_dF_short_unsucc,cbar=True,cmap='viridis',yticklabels=trials_short_unsucc.astype('int'),xticklabels=True,ax=ax6)
        sns.heatmap(mean_dF_long_succ,cbar=True,cmap='viridis',yticklabels=trials_long_succ.astype('int'),xticklabels=True,ax=ax7)
        sns.heatmap(mean_dF_long_unsucc,cbar=True,cmap='viridis',yticklabels=trials_long_unsucc.astype('int'),xticklabels=True,ax=ax8)

    if c_ylim == []:
        ax1.set_ylim([-0.5,max_y])
        ax2.set_ylim([-0.5,max_y])
        ax9.set_ylim([-0.5,max_y])
        ax10.set_ylim([-0.5,max_y])
        ax11.set_ylim([-0.5,max_y])
        ax12.set_ylim([-0.5,max_y])
        ax13.set_ylim([-0.5,max_y_deconv])
        ax14.set_ylim([-0.5,max_y_deconv])
        c_ylim = [-0.5,max_y,heatmap_max,-0.5,max_y_deconv]
    else:
        ax1.set_ylim(c_ylim[0:2])
        ax2.set_ylim(c_ylim[0:2])
        ax9.set_ylim(c_ylim[0:2])
        ax10.set_ylim(c_ylim[0:2])
        ax11.set_ylim(c_ylim[0:2])
        ax12.set_ylim(c_ylim[0:2])
        ax13.set_ylim(c_ylim[3:5])
        ax14.set_ylim(c_ylim[3:5])

    ax3.set_xticks([0,20,40,60,80])
    ax3.set_xticklabels(['0','100','200','300','400'])
    ax3.set_xlim([10,68])
    ax4.set_xticks([0,20,40,60,80,100])
    ax4.set_xticklabels(['0','100','200','300','400','500'])
    ax4.set_xlim([10,80])

    ax5.set_xticks([0,20,40,60,80])
    ax5.set_xticklabels(['0','100','200','300','400'])
    ax5.set_xlim([10,68])

    ax6.set_xticks([0,20,40,60,80])
    ax6.set_xticklabels(['0','100','200','300','400'])
    ax6.set_xlim([10,68])

    ax7.set_xticks([0,20,40,60,80,100])
    ax7.set_xticklabels(['0','100','200','300','400','500'])
    ax7.set_xlim([10,80])

    ax8.set_xticks([0,20,40,60,80,100])
    ax8.set_xticklabels(['0','100','200','300','400','500'])
    ax8.set_xlim([10,80])

    ax9.set_xticks([0,20,40,60,80])
    ax9.set_xticklabels(['0','100','200','300','400'])
    ax9.set_xlim([10,68])

    ax10.set_xticks([0,20,40,60,80])
    ax10.set_xticklabels(['0','100','200','300','400'])
    ax10.set_xlim([10,68])

    ax11.set_xticks([0,20,40,60,80,100])
    ax11.set_xticklabels(['0','100','200','300','400','500'])
    ax11.set_xlim([10,80])

    ax12.set_xticks([0,20,40,60,80,100])
    ax12.set_xticklabels(['0','100','200','300','400','500'])
    ax12.set_xlim([10,80])

    plt.tight_layout()

    fig.suptitle(fname, wrap=True)
    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    return roi_std_short,roi_std_long,roi_active_fraction_short,roi_active_fraction_long,roi_meanpeak_short,roi_meanpeak_long,roi_meanpeak_short_location,roi_meanpeak_long_location, \
        mean_trace_short.tolist(),mean_trace_long.tolist(),mean_trace_short_start,mean_trace_long_start,filtered_short_1_mean_trace.tolist(),filtered_short_2_mean_trace.tolist(),filtered_long_1_mean_trace.tolist(),filtered_long_2_mean_trace.tolist(),c_ylim

def run_analysis(mousename, sessionname, sessionname_openloop, number_of_rois, h5_filepath, subname, sess_subfolder, session_rois):
    """ set up function call and dictionary to collect results """

    MOUSE = mousename
    SESSION = sessionname
    SESSION_OPENLOOP = sessionname_openloop
    NUM_ROIS = number_of_rois
    h5path = h5_filepath
    SUBNAME = subname
    subfolder = sess_subfolder
    subfolder_ol = sess_subfolder + '_openloop'
    write_to_dict = False

    # set up dictionary to hold result parameters from roi
    session_rois[SUBNAME+'_roi_number'] = []
    session_rois[SUBNAME+'_roi_number_ol'] = []
    session_rois[SUBNAME+'_std_short'] = []
    session_rois[SUBNAME+'_std_long'] = []
    session_rois[SUBNAME+'_active_short'] = []
    session_rois[SUBNAME+'_active_long'] = []
    session_rois[SUBNAME+'_peak_short'] = []
    session_rois[SUBNAME+'_peak_long'] = []
    session_rois[SUBNAME+'_peak_loc_short'] = []
    session_rois[SUBNAME+'_peak_loc_long'] = []
    session_rois[SUBNAME+'_mean_trace_short'] = []
    session_rois[SUBNAME+'_mean_trace_long'] = []
    session_rois[SUBNAME+'_mean_trace_start_short'] = []
    session_rois[SUBNAME+'_mean_trace_start_long'] = []
    session_rois[SUBNAME+'_std_short_ol'] = []
    session_rois[SUBNAME+'_std_long_ol'] = []
    session_rois[SUBNAME+'_active_short_ol'] = []
    session_rois[SUBNAME+'_active_long_ol'] = []
    session_rois[SUBNAME+'_peak_short_ol'] = []
    session_rois[SUBNAME+'_peak_long_ol'] = []
    session_rois[SUBNAME+'_peak_loc_short_ol'] = []
    session_rois[SUBNAME+'_peak_loc_long_ol'] = []
    session_rois[SUBNAME+'_mean_trace_short_ol'] = []
    session_rois[SUBNAME+'_mean_trace_long_ol'] = []
    session_rois[SUBNAME+'_mean_trace_start_short_ol'] = []
    session_rois[SUBNAME+'_mean_trace_start_long_ol'] = []

    session_rois[SUBNAME+'_filter_1_mean_trace_start_short_ol'] = []
    session_rois[SUBNAME+'_filter_1_mean_trace_start_long_ol'] = []
    session_rois[SUBNAME+'_filter_2_mean_trace_start_short_ol'] = []
    session_rois[SUBNAME+'_filter_2_mean_trace_start_long_ol'] = []

    # run analysis for vr session
    if type(NUM_ROIS) is int:
        roilist = range(NUM_ROIS)
    else:
        roilist = NUM_ROIS

    # if we want to run through all the rois, just say all
    if NUM_ROIS == 'all':
        h5dat = h5py.File(h5path, 'r')
        dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
        h5dat.close()
        roilist = np.arange(0,dF_ds.shape[1],1)
        write_to_dict = True
        print('number of rois: ' + str(NUM_ROIS))

    for r in roilist:
        print(SUBNAME + ': ' + str(r))
        std_short, std_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, mean_trace_short, mean_trace_long, mean_trace_short_start, mean_trace_long_start,_,_,_,_, c_ylim = fig_dfloc_trace_roiparams(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), ['trial_successful'], ['trial_unsuccessful'],['trial_successful'],['trial_unsuccessful'], fformat, subfolder, [])
        session_rois[SUBNAME+'_roi_number'].append(r)
        session_rois[SUBNAME+'_std_short'].append(std_short)
        session_rois[SUBNAME+'_std_long'].append(std_long)
        session_rois[SUBNAME+'_active_short'].append(active_short)
        session_rois[SUBNAME+'_active_long'].append(active_long)
        session_rois[SUBNAME+'_peak_short'].append(peak_short)
        session_rois[SUBNAME+'_peak_long'].append(peak_long)
        session_rois[SUBNAME+'_peak_loc_short'].append(meanpeak_short_time)
        session_rois[SUBNAME+'_peak_loc_long'].append(meanpeak_long_time)
        session_rois[SUBNAME+'_mean_trace_short'].append(mean_trace_short)
        session_rois[SUBNAME+'_mean_trace_long'].append(mean_trace_long)
        session_rois[SUBNAME+'_mean_trace_start_short'].append(mean_trace_short_start)
        session_rois[SUBNAME+'_mean_trace_start_long'].append(mean_trace_long_start)

        std_short, std_long, active_short, active_long, peak_short, peak_long, meanpeak_short_time, meanpeak_long_time, mean_trace_short, mean_trace_long, mean_trace_short_start, mean_trace_long_start,run_passive_mean_trace_short,norun_passive_mean_trace_short,run_passive_mean_trace_long,norun_passive_mean_trace_long,_ = fig_dfloc_trace_roiparams(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r),['animal_running',1,2], ['animal_notrunning',1,2],['animal_running',1,2],['animal_notrunning',1,2], fformat, subfolder_ol, c_ylim)
        session_rois[SUBNAME+'_roi_number_ol'].append(r)
        session_rois[SUBNAME+'_std_short_ol'].append(std_short)
        session_rois[SUBNAME+'_std_long_ol'].append(std_long)
        session_rois[SUBNAME+'_active_short_ol'].append(active_short)
        session_rois[SUBNAME+'_active_long_ol'].append(active_long)
        session_rois[SUBNAME+'_peak_short_ol'].append(peak_short)
        session_rois[SUBNAME+'_peak_long_ol'].append(peak_long)
        session_rois[SUBNAME+'_peak_loc_short_ol'].append(meanpeak_short_time)
        session_rois[SUBNAME+'_peak_loc_long_ol'].append(meanpeak_long_time)
        session_rois[SUBNAME+'_mean_trace_short_ol'].append(mean_trace_short)
        session_rois[SUBNAME+'_mean_trace_long_ol'].append(mean_trace_long)
        session_rois[SUBNAME+'_mean_trace_start_short_ol'].append(mean_trace_short_start)
        session_rois[SUBNAME+'_mean_trace_start_long_ol'].append(mean_trace_long_start)
        session_rois[SUBNAME+'_filter_1_mean_trace_start_short_ol'].append(run_passive_mean_trace_short)
        session_rois[SUBNAME+'_filter_1_mean_trace_start_long_ol'].append(norun_passive_mean_trace_short)
        session_rois[SUBNAME+'_filter_2_mean_trace_start_short_ol'].append(run_passive_mean_trace_long)
        session_rois[SUBNAME+'_filter_2_mean_trace_start_long_ol'].append(norun_passive_mean_trace_long)

    if write_to_dict:
        print('writing to dictionary.')
        write_dict(MOUSE, SESSION, roi_result_params)

    # return session_rois

def update_dict(MOUSE, SESSION, roi_result_params):
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)
    if os.path.isfile(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json'):
        with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json', 'r') as f:
            existing_dict = json.load(f)
        existing_dict.update(roi_result_params)
        with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json', 'w') as f:
            json.dump(existing_dict,f)
            print('updating dict')
    else:
        with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json','w') as f:
            print('new dict')
            json.dump(roi_result_params,f)

def run_LF170110_2_Day201748_1():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    SESSION_OPENLOOP = 'Day201748_openloop_1'
    NUM_ROIS = 152
    NUM_ROIS = [74]
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    # update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day201748_2():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_2'
    SESSION_OPENLOOP = 'Day201748_openloop_2'
    NUM_ROIS = 171
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day201748_3():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_3'
    SESSION_OPENLOOP = 'Day201748_openloop_3'
    NUM_ROIS = 50
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170421_2_Day2017719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = 'Day2017719_openloop'
    NUM_ROIS = 96
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170421_2_Day20170719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170719'
    SESSION_OPENLOOP = 'Day20170719_openloop'
    NUM_ROIS = 123
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170421_2_Day2017720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017720'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 45
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170420_1_Day201783():
    MOUSE = 'LF170420_1'
    SESSION = 'Day201783'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 81
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170420_1_Day2017719():
    MOUSE = 'LF170420_1'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 91
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170222_1_Day201776():
    MOUSE = 'LF170222_1'
    SESSION = 'Day201776'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 120
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day2017331():
    MOUSE = 'LF170110_2'
    SESSION = 'Day2017331'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 184
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170613_1_Day201784():
    MOUSE = 'LF170613_1'
    SESSION = 'Day201784'
    SESSION_OPENLOOP = 'Day201784_openloop'
    NUM_ROIS = 77
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    NUM_ROIS = 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = { }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    # update_dict(MOUSE, SESSION, roi_result_params)
    # write_dict(MOUSE, SESSION, roi_result_params)

def run_LF171212_2_Day2018218_2():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_2'
    SESSION_OPENLOOP = 'Day2018218_openloop_2'
    NUM_ROIS = 335
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF171212_2_Day2018321():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018321'
    SESSION_OPENLOOP = 'Day2018321_openloop'
    NUM_ROIS = 171
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF171211_1_Day2018321_2():
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    SESSION_OPENLOOP = 'Day2018321_openloop_2'
    NUM_ROIS = 10#170
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170214_1_Day201777():
    MOUSE = 'LF170214_1'
    SESSION = 'Day201777'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 163
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF170214_1_Day2017714():
    MOUSE = 'LF170214_1'
    SESSION = 'Day2017714'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 140
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF180112_2_Day2018424_1():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_1'
    SESSION_OPENLOOP = 'Day2018424_openloop_1'
    NUM_ROIS = 73
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF180112_2_Day2018424_2():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_2'
    SESSION_OPENLOOP = 'Day2018424_openloop_2'
    NUM_ROIS = 43
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF180119_1_Day2018316_1():
    MOUSE = 'LF180119_1'
    SESSION = 'Day2018316_1'
    SESSION_OPENLOOP = 'Day2018316_openloop_1'
    NUM_ROIS = 271
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF180119_1_Day2018316_2():
    MOUSE = 'LF180119_1'
    SESSION = 'Day2018316_2'
    SESSION_OPENLOOP = 'Day2018316_openloop_2'
    NUM_ROIS = 305
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, SUBNAME, subfolder, roi_result_params)

    update_dict(MOUSE, SESSION, roi_result_params)

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'

    # run_LF170110_2_Day201748_1() #*
    # run_LF170110_2_Day201748_2() *
    # run_LF170110_2_Day201748_3()
    # run_LF170421_2_Day2017719()
    # run_LF170421_2_Day20170719()
    # run_LF170421_2_Day2017720()
    # run_LF170420_1_Day201783()
    # run_LF170420_1_Day2017719()
    # run_LF170222_1_Day201776()
    # run_LF170110_2_Day2017331()
    # run_LF170613_1_Day201784()
    # run_LF170613_1_Day20170804()
    # run_LF171212_2_Day2018218_2()
    # run_LF171212_2_Day2018321()
    run_LF171211_1_Day2018321_2()
    # run_LF180119_1_Day2018316_1()
    # run_LF180119_1_Day2018316_2()

    # run_LF170214_1_Day201777()
    # run_LF170214_1_Day2017714()

    # run_LF180112_2_Day2018424_1()
    # run_LF180112_2_Day2018424_2()

    print('done')
