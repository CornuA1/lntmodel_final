#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:29:29 2018

@author: katya
"""

"""
Plot population activity aligned to location. Select half the trials randomly
to calculate mean response and the other half to calculate the order by
which to plot the sequence

@author: lukasfischer

NOTE: this is an updated version of fig_popvlocHalfsort_stage5. This function can load multiple datasets and concatenate them (as opposed to using behav_collection and dF_collection)


"""


def fig_mean_dF_s5_test(included_recordings, rois='task_engaged_all', trials='both', sortby='both', fformat='png'):
    # load local settings file
    import matplotlib
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("../Analysis")
    import pickle
    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal
    import statsmodels.api as sm
    import yaml
    import pandas as pd
    import h5py
    import json
    import seaborn as sns
    sns.set_style('white')
    import os
    with open('../loc_settings.yaml', 'r') as f:
                content = yaml.load(f)

    # number of times calculations carried out
    NUM_ITER = 1

    # track paramaters (cm)
    reward_distance = 0
    tracklength_short = 320 + reward_distance
    tracklength_long = 380 + reward_distance
    track_start = 100
    bin_size = 5

    # track numbers used in this analysis
    track_short = 3
    track_long = 4
    track_black = 5

    # define spatial bins for analysis
    lm_start = (200-track_start)/bin_size
    lm_end = (240-track_start)/bin_size
    binnr_short = (tracklength_short-track_start)/bin_size
    binnr_long = (tracklength_long-track_start)/bin_size
    reward_bins = reward_distance / bin_size
    start_bin = track_start / bin_size
    end_bin_short = (tracklength_short / bin_size) + reward_bins
    end_bin_long = (tracklength_long / bin_size) + reward_bins


    # define bin-edges manually for better control
    bin_edges_short = np.linspace(track_start, tracklength_short, binnr_short+1)
    bin_edges_long = np.linspace(track_start, tracklength_long, binnr_long+1)

  

    for current_iter in range(NUM_ITER):
        tot_rois = 0
        mean_dF_short_coll_sig = []
        mean_dF_long_coll_sig = []
        print(current_iter)
        ind_dFs = [] #index of each ROI in mean_dF matrix 
        # load data
        for r in included_recordings:
            # load individual dataset
            print(r)
            mouse = r[0]
            # len > 2 indicates that we want to use a classification file in r[2], but plotting the data for r[1]
            
            h5path = content['imaging_dir'] + mouse + '/' + mouse + '.h5'
            h5dat = h5py.File(h5path, 'r')
            for session in r[1:]:
                with open(content['imaging_dir'] + mouse + os.sep + 'roi_classification.json') as f:
                    roi_classification = json.load(f)
                if session not in roi_classification['mouse_session']:
                    continue
                if '_' in session:
                    continue
               
                if (r == 'LF170420_1'):
                    continue
                behav_ds = np.copy(h5dat[session + '/behaviour_aligned'])
                dF_ds = np.copy(h5dat[session + '/dF_win'])
                h5dat.close()
                
                   
                
                # pull out trial numbers of respective sections
                trials_short_sig = filter_trials(behav_ds, [], ['tracknumber',track_short])
                trials_long_sig = filter_trials(behav_ds, [], ['tracknumber',track_long])
                trials_black_sig = filter_trials(behav_ds,[],['tracknumber', 5])
                
                # further filter if only correct or incorrect trials are plotted
                if trials == 'c':
                    trials_short_sig = filter_trials(behav_ds, [], ['trial_successful'],trials_short_sig)
                    trials_long = filter_trials(behav_ds, [], ['trial_successful'],trials_long)
                if trials == 'ic':
                    trials_short_sig = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short_sig)
                    trials_long = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long)
                
                # randomly draw 50% of indices to calculate signal and the other half to calculate the order by which to sort
                   
                # select rois
                if rois == 'all':
                    roi_selection = np.arange(0,np.size(dF_ds,1))
                elif rois == 'task_engaged_all':
                    roi_selection = np.union1d(roi_classification['task_engaged_short'],roi_classification['task_engaged_long'])
                elif rois == 'task_engaged_short':
                    roi_selection = roi_classification['task_engaged_short']
                elif rois == 'task_engaged_long':
                    roi_selection = roi_classification['task_engaged_long']
                elif rois == 'pre_landmark_all':
                    roi_selection = np.union1d(roi_classification['pre_landmark_short'],roi_classification['pre_landmark_long'])
                elif rois == 'pre_landmark_short':
                    roi_selection = roi_classification['pre_landmark_short']
                elif rois == 'pre_landmark_long':
                    roi_selection = roi_classification['pre_landmark_long']
                elif rois == 'landmark_all':
                    roi_selection = np.union1d(roi_classification['landmark_short'],roi_classification['landmark_long'])
                elif rois == 'landmark_short':
                    roi_selection = roi_classification['landmark_short']
                elif rois == 'landmark_long':
                    roi_selection = roi_classification['landmark_long']
                elif rois == 'path_integration_all':
                    roi_selection = np.union1d(roi_classification['path_integration_short'],roi_classification['path_integration_long'])
                elif rois == 'path_integration_short':
                    roi_selection = roi_classification['path_integration_short']
                elif rois == 'path_integration_long':
                    roi_selection = roi_classification['path_integration_long']
                elif rois == 'reward_all':
                    roi_selection = np.union1d(roi_classification['reward_short'],roi_classification['reward_long'])
                elif rois == 'reward_short':
                    roi_selection = roi_classification['reward_short']
                elif rois == 'reward_long':
                    roi_selection = roi_classification['reward_long']
    
                tot_rois += np.size(roi_selection)
    
                # storage for mean ROI data
                mean_dF_short_sig = np.zeros((int(binnr_short),np.size(roi_selection)))
                mean_dF_long_sig = np.zeros((int(binnr_long),np.size(roi_selection)))
                #mean_df_black_sig = np.zeros((int()))
               
                ind_dFs.append(roi_selection.shape[0])
                for j,roi in enumerate(roi_selection):
                    # intilize matrix to hold data for all trials of this roi
                    mean_dF_trials_short_sig = np.zeros((np.size(trials_short_sig,0),int(binnr_short)))
                    mean_dF_trials_long_sig = np.zeros((np.size(trials_long_sig,0),int(binnr_long)))
                    # calculate mean dF vs location for each ROI on short trials
                   
                    for k,t in enumerate(trials_short_sig):
                        # pull out current trial and corresponding dF data and bin it
                        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                        cur_trial_rew_loc = behav_ds[behav_ds[:,6]==t+1,1]
                        # if the location was reset to 0 as the animal got a reward (which is the case when it gets a default reward: add the last location on the track to blackbox location)
                        if np.size(cur_trial_rew_loc) > 0:
                            if cur_trial_rew_loc[0] < 100:
                                cur_trial_rew_loc += cur_trial_loc[-1]
                            cur_trial_loc = np.append(cur_trial_loc, cur_trial_rew_loc)
                        if roi == 61:
                            print('f')
                        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
                        if np.size(cur_trial_rew_loc) > 0:
                            cur_trial_dF_roi = np.append(cur_trial_dF_roi, dF_ds[behav_ds[:,6]==t+1,1])
                        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges_short, (track_start, tracklength_short))[0]
                        #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
                        mean_dF_trials_short_sig[k,:] = mean_dF_trial
                        #print(mean_dF_trial)
    
                    #mean_dF_short_sig[:,j] = np.nanmean(mean_dF_trials_short_sig,0)
                    mean_dF_short = np.nanmean(mean_dF_trials_short_sig,0)
                    #mean_dF_short_sig[:,j] = (mean_dF - np.nanmin(mean_dF))/(np.nanmax(mean_dF) - np.nanmin(mean_dF))
                    #mean_dF_short_sig[:,j] = mean_dF
                    # print(np.nanmin(mean_dF), np.nanmax(mean_dF))
    
                    #mean_dF_short_sig[:,j] /= np.nanmax(np.abs(mean_dF_short_sig[start_bin:end_bin_short,j]))
    
                    # calculate mean dF vs location for each ROI on short trials
                   
    
                    #mean_dF_short_sort[:,j] = np.nanmean(mean_dF_trials_short_sort,0)
                    #mean_dF_short_sort[:,j] /= np.nanmax(np.abs(mean_dF_short_sort[start_bin:end_bin_long,j]))
                   
                    # calculate mean dF vs location for each ROI on long trials
                    for k,t in enumerate(trials_long_sig):
                        # pull out current trial and corresponding dF data and bin it
                        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                        cur_trial_rew_loc = behav_ds[behav_ds[:,6]==t+1,1]
                        # if the location was reset to 0 as the animal got a reward (which is the case when it gets a default reward: add the last location on the track to blackbox location)
                        if np.size(cur_trial_rew_loc) > 0:
                            if cur_trial_rew_loc[0] < 100:
                                cur_trial_rew_loc += cur_trial_loc[-1]
                            cur_trial_loc = np.append(cur_trial_loc, cur_trial_rew_loc)
                        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
                        if np.size(cur_trial_rew_loc) > 0:
                            cur_trial_dF_roi = np.append(cur_trial_dF_roi, dF_ds[behav_ds[:,6]==t+1,1])
    
                        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges_long, (track_start, tracklength_long))[0]
                        #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_long]))
                        mean_dF_trials_long_sig[k,:] = mean_dF_trial
    
                    # mean_dF_long_sig[:,j] = np.nanmean(mean_dF_trials_long_sig,0)
                    # mean_dF_long_sig[:,j] /= np.nanmax(np.abs(mean_dF_long_sig[start_bin:end_bin_long,j]))
                    mean_dF_long = np.nanmean(mean_dF_trials_long_sig,0)
                    #mean_dF_long_sig[:,j] = (mean_dF - np.nanmin(mean_dF))/(np.nanmax(mean_dF) - np.nanmin(mean_dF))
                    #mean_dF_long_sig[:,j] = mean_dF
                
                    #calculate mean for each ROI on black box trials
                    all_trial_loc = []
                    for k,t in enumerate(trials_black_sig):
                        cur_trials = dF_ds[behav_ds[:,6]==t,1]
                        all_trial_loc = np.concatenate((all_trial_loc, cur_trials[:]))
                    mean_dF_black = [np.nanmean(all_trial_loc,0)]
                    all_mean_dF = np.concatenate((mean_dF_black, mean_dF_long, mean_dF_short))
#                    mean_dF_short_sig[:,j] = (mean_dF_short - np.nanmin(all_mean_dF))/(np.nanmax(all_mean_dF) - np.nanmin(all_mean_dF))
#                    mean_dF_long_sig[:,j] = (mean_dF_long - np.nanmin(all_mean_dF))/(np.nanmax(all_mean_dF) - np.nanmin(all_mean_dF))
#                    
                    mean_dF_short_sig[:,j] = (mean_dF_short - np.nanmin(dF_ds[:,roi]))/(np.nanmax(dF_ds[:,roi]) - np.nanmin(dF_ds[:,roi]))
                    mean_dF_long_sig[:,j] = (mean_dF_long - np.nanmin(dF_ds[:,roi]))/(np.nanmax(dF_ds[:,roi]) - np.nanmin(dF_ds[:,roi]))
                    
                if mean_dF_short_coll_sig == []:
                    mean_dF_short_coll_sig = mean_dF_short_sig
                else:
                    mean_dF_short_coll_sig = np.append(mean_dF_short_coll_sig, mean_dF_short_sig, axis=1)
                      
                if mean_dF_long_coll_sig == []:
                    mean_dF_long_coll_sig = mean_dF_long_sig
                else:
                    mean_dF_long_coll_sig = np.append(mean_dF_long_coll_sig, mean_dF_long_sig, axis=1)
    
        # create figure and axes
        fig = plt.figure(figsize=(25,12))
        ax1 = plt.subplot2grid((5,200),(0,0),rowspan=4, colspan=45)
        ax2 = plt.subplot2grid((5,200),(0,50),rowspan=4, colspan=55)
        # ax3 = plt.subplot2grid((5,200),(0,0),rowspan=1, colspan=40)
        # ax4 = plt.subplot2grid((5,200),(0,50),rowspan=1, colspan=55)
        
        # ax6 = plt.subplot2grid((5,2),(4,1),rowspan=1)
    
        
    
       
       
    # sort by peak activity (naming of variables confusing because the script grew organically...)
    mean_dF_sort_short = np.zeros(mean_dF_short_coll_sig.shape[1])
    for i, row in enumerate(mean_dF_short_coll_sig.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_short[i] = np.nanargmax(row)
        else:
            print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
            pass
    sort_ind_short = np.argsort(mean_dF_sort_short)

    plt.tight_layout()
    #start_bin=10
    #ax3.set_ylim([0,30])
    #ax3.set_xlim([start_bin,end_bin_short])
    #ax3.set_ylim([0,40])
#    sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short, sort_ind_short]), cmap='jet', vmin=0.0, vmax=1.0, ax=ax1, cbar=False)

    mean_dF_sort_long = np.zeros(mean_dF_long_coll_sig.shape[1])
    for i, row in enumerate(mean_dF_long_coll_sig.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_long[i] = np.nanargmax(row)
        else:
            # print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
            pass
    sort_ind_long = np.argsort(mean_dF_sort_long)

    # sns.distplot(mean_dF_sort_short,color=sns.xkcd_rgb["windows blue"],bins=22,hist=True, kde=False,ax=ax3)
    # sns.distplot(mean_dF_sort_long,color=sns.xkcd_rgb["dusty purple"],bins=28,hist=True, kde=False,ax=ax4)
    #
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['left'].set_visible(False)
    # ax4.spines['top'].set_visible(False)
    # ax4.spines['right'].set_visible(False)
    # ax4.spines['left'].set_visible(False)

    if sortby == 'none':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'short':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'long':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'both':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)

    ax1.axvline((200/bin_size)-start_bin, lw=3, c='0.8')
    ax1.axvline((240/bin_size)-start_bin, lw=3, c='0.8')
    ax1.axvline((320/bin_size)-start_bin, lw=3, c='0.8')
    ax2.axvline((200/bin_size)-start_bin, lw=3, c='0.8')
    ax2.axvline((240/bin_size)-start_bin, lw=3, c='0.8')
    ax2.axvline((380/bin_size)-start_bin, lw=3, c='0.8')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    #ax3.set_xlim([0,23])
    #ax4.set_xlim([0,35])
    #ax4.set_xlim([0, end_bin_long-start_bin+1])

    #fig.suptitle(fname + '_' + trials + '_' + str([''.join(str(r) for r in ri) for ri in rec_info]),wrap=True)

#    plt.tight_layout()
    # list_test = popvec_cc_reconstruction_ss[0,:,]
    # print(len(popvec_cc_reconstruction_ss.tolist()[0]),len(popvec_cc_reconstruction_ss.tolist()[1]))
    mean_dF_results = {
             'mean_dF_short': mean_dF_short_coll_sig.tolist(),
             'mean_dF_long': mean_dF_long_coll_sig.tolist(),
             'sort_short': sort_ind_short.tolist(),
             'sort_long': sort_ind_long.tolist(),
             'ind_dF': ind_dFs
            }

    with open(content['figure_output_path'] + 'mean_dF_results' + os.sep + '_mean_dF_results_allfiles_norm_entire_sess_inc420rat.json','w+') as f:
        json.dump(mean_dF_results,f)


if __name__ == "__main__":
    # ALL LAYERS, ALL TASK ENGAGED
    # figure_datasets = [['LF170110_2','Day20170331'], ['LF170222_1','Day20170615'],
    # ['LF170420_1','Day20170719'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170613_1','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_all', 'popvloc')
    # # #
    # # # # # #
    # figure_datasets = [['LF170110_2','Day20170331_openloop','Day20170331'],['LF170222_1','Day20170615_openloop','Day20170615'],
    # ['LF170420_1','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170720_openloop','Day20170720'],['LF170613_1','Day201784_openloop','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_all_openloop', 'popvloc')
    # # #
    # #
    # #
    # figure_datasets = [['LF170110_2','Day20170331'],['LF170222_1','Day20170615'],
    # ['LF170420_1','Day20170719'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170613_1','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'short', 'png', 'task_engaged_all_sortby_short', 'popvloc')
    # #
    # figure_datasets = [['LF170110_2','Day20170331_openloop','Day20170331'],['LF170222_1','Day20170615_openloop','Day20170615'],
    # ['LF170420_1','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170720_openloop','Day20170720'],['LF170613_1','Day201784_openloop','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'short', 'png', 'task_engaged_all_openloop_sortby_short', 'popvloc')
    #
    # figure_datasets = [['LF170110_2','Day20170331'],['LF170222_1','Day20170615'],
    # ['LF170420_1','Day20170719'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170613_1','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'long', 'png', 'task_engaged_all_sortby_long', 'popvloc')
    # #
    # figure_datasets = [['LF170110_2','Day20170331_openloop','Day20170331'],['LF170222_1','Day20170615_openloop','Day20170615'],
    # ['LF170420_1','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170720_openloop','Day20170720'],['LF170613_1','Day201784_openloop','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'long', 'png', 'task_engaged_all_openloop_sortby_long', 'popvloc')


    # figure_datasets = [['LF170612_1','Day20170719']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    # figure_datasets = [['LF170612_1','Day20170719_openloop','Day20170719']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc_openloop')
    # figure_datasets = [['LF170613_1','Day2017719']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    # figure_datasets = [['LF170613_1','Day2017719_openloop','Day2017719']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc_openloop')
    import json
    import os
    import yaml
    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)
    with open(content['figure_output_path']  + 'recordings_with_behav_inc420.json','r') as f:
        recordings = json.load(f)
    good_recordings = recordings['good_recordings']
    fig_mean_dF_s5_test(good_recordings, 'task_engaged_all', 'both', 'short', 'png')

    # figure_datasets = [['LF170222_1','Day20170615']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    # figure_datasets = [['LF170420_1','Day20170719']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    # figure_datasets = [['LF170421_2','Day20170719']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    # figure_datasets = [['LF170421_2','Day20170720']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    # figure_datasets = [['LF170110_2','Day20170331']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')

    # l2/3 task engaged all
    # figure_datasets = [['LF170110_2','Day20170331'],['LF170613_1','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_all_L23', 'popvloc')
    # figure_datasets = [['LF170110_2','Day20170331_openloop','Day20170331'],['LF170613_1','Day201784_openloop','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_all_L23_openloop', 'popvloc')
    # # # l5 task engaged all
    # figure_datasets = [['LF170222_1','Day20170615_openloop','Day20170615'],['LF170420_1','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170719_openloop','Day20170719'],['LF170421_2','Day20170720_openloop','Day20170720']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_all_L5_openloop', 'popvloc')
    # figure_datasets = [['LF170222_1','Day20170615'],['LF170420_1','Day20170719'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_all_L5', 'popvloc')
    # individual
    #
    # figure_datasets = [['LF170110_2','Day20170331_openloop'],['LF170222_1','Day20170615_openloop'],
    # ['LF170420_1','Day20170719_openloop'],['LF170421_2','Day20170719_openloop'],['LF170421_2','Day20170720_openloop'],['LF170613_1','Day201784_openloop']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'openloop_task_engaged_all', 'popvloc')
    #
    # figure_datasets = [['LF170110_2','Day20170331_openloop'],['LF170613_1','Day201784_openloop']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'openloop_task_engaged_L23', 'popvloc')
    #
    # figure_datasets = [['LF170222_1','Day20170615_openloop'],['LF170420_1','Day20170719_openloop'],['LF170421_2','Day20170719_openloop'],['LF170421_2','Day20170720_openloop']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'openloop_task_engaged_L5', 'popvloc')


    # figure_datasets = [['LF170214_1','Day201777'],['LF170214_1','Day2017714'],['LF171211_2','Day201852']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_V1', 'popvloc')
    #
#    figure_datasets = [['LF170214_1','Day201777_openloop','Day201777'],['LF170214_1','Day2017714_openloop','Day2017714'],['LF171211_2','Day201852_openloop','Day201852']]
#    fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_V1_openloop', 'popvloc')

    # figure_datasets = [['LF171211_2','Day201852']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    # figure_datasets = [['LF171211_2','Day201852_openloop','Day201852']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    #
    # figure_datasets = [['LF170214_1','Day201777']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')
    #
    # figure_datasets = [['LF170214_1','Day201777_openloop','Day201777']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')


#
