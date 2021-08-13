#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:41:13 2018

@author: katya
"""

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

#finds the mean dF for 
def fig_all_trials_dF_s5_test(included_recordings, rois='task_engaged_all', trials='both', sortby='both', fformat='png'):
    # load local settings file
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("../Analysis")
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

  
    all_trials_dF_results = {}
    for current_iter in range(NUM_ITER):
        tot_rois = 0
        
        print(current_iter)
        # load data
        for r in included_recordings:
            # load individual dataset
            print(r)
            mouse = r[0]
          
            h5path = content['imaging_dir'] + mouse + '/' + mouse + '.h5'
            h5dat = h5py.File(h5path, 'r')
            for session in r[1:]:
                with open(content['imaging_dir'] + mouse + os.sep + 'roi_classification.json') as f:
                    roi_classification = json.load(f)
                if session not in roi_classification['mouse_session']:
                    continue                
                print(session)
                behav_ds = np.copy(h5dat[session + '/behaviour_aligned'])
                dF_ds = np.copy(h5dat[session + '/dF_win'])
                h5dat.close()
                  
                
                # pull out trial numbers of respective sections
                trials_short_sig = filter_trials(behav_ds, [], ['tracknumber',track_short])
                trials_long_sig = filter_trials(behav_ds, [], ['tracknumber',track_long])
              
                # further filter if only correct or incorrect trials are plotted
                if trials == 'c':
                    trials_short_sig = filter_trials(behav_ds, [], ['trial_successful'],trials_short_sig)
                    trials_long_sig = filter_trials(behav_ds, [], ['trial_successful'],trials_long_sig)
                if trials == 'ic':
                    trials_short_sig = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short_sig)
                    trials_long_sig = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long_sig)
                
                
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
                trial_size_long = trials_long_sig.shape
                trial_size_short = trials_short_sig.shape
                
                #holds mean dF signal for all rois on all trials 
                mean_trials_dF_short_sig = np.zeros((int(binnr_short),np.size(roi_selection)))
                mean_trials_dF_long_sig = np.zeros((int(binnr_long),np.size(roi_selection)))
                
                #holds dF signal for all rois on all trials (for trial-by-trial analysis)
                all_trials_dF_short_sig = np.zeros((int(binnr_short)*trial_size_short[0],np.size(roi_selection)))
                all_trials_dF_long_sig = np.zeros((int(binnr_long)*trial_size_long[0],np.size(roi_selection)))
            
               
                for j,roi in enumerate(roi_selection):
                    # intilize matrix to hold data for all trials of this roi
                    all_trials_dF_trials_short_sig = np.zeros((np.size(trials_short_sig,0),int(binnr_short)))
                    all_trials_dF_trials_long_sig = np.zeros((np.size(trials_long_sig,0),int(binnr_long)))
                    
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
                        
                        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
                        if np.size(cur_trial_rew_loc) > 0:
                            cur_trial_dF_roi = np.append(cur_trial_dF_roi, dF_ds[behav_ds[:,6]==t+1,1])
                        all_trials_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges_short, (track_start, tracklength_short))[0]
                        all_trials_dF_trials_short_sig[k,:] = all_trials_dF_trial
                       
    
                    mean_trials_dF_short = np.nanmean(all_trials_dF_trials_short_sig,0)
                    trial_size = all_trials_dF_trials_short_sig.shape
                    
                    #reshapes matrix to hold all short trials for all binned locations(2D shape)
                    all_trials_dF_trial_short_reshape = all_trials_dF_trials_short_sig.reshape(trial_size[0]*trial_size[1])
                   
                   
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
    
                        all_trials_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges_long, (track_start, tracklength_long))[0]
                        all_trials_dF_trials_long_sig[k,:] = all_trials_dF_trial
    
                   
                    trial_size = all_trials_dF_trials_long_sig.shape
                    #reshapes matrix to hold all long trials for all binned locations(2D shape)
                    all_trials_dF_trial_long_reshape = all_trials_dF_trials_long_sig.reshape(trial_size[0]*trial_size[1])
                    mean_trials_dF_long = np.nanmean(all_trials_dF_trials_long_sig,0)
                   
                    #normalize trials by max and mean of all trials for each roi 
                    all_trials_dF_short_sig[:,j] = (all_trials_dF_trial_short_reshape - np.nanmin(dF_ds[:,roi]))/(np.nanmax(dF_ds[:,roi]) - np.nanmin(dF_ds[:,roi]))
                    all_trials_dF_long_sig[:,j] = (all_trials_dF_trial_long_reshape - np.nanmin(dF_ds[:,roi]))/(np.nanmax(dF_ds[:,roi]) - np.nanmin(dF_ds[:,roi]))
                    mean_trials_dF_short_sig[:,j] = (mean_trials_dF_short - np.nanmin(dF_ds[:,roi]))/(np.nanmax(dF_ds[:,roi]) - np.nanmin(dF_ds[:,roi]))
                    mean_trials_dF_long_sig[:,j] = (mean_trials_dF_long - np.nanmin(dF_ds[:,roi]))/(np.nanmax(dF_ds[:,roi]) - np.nanmin(dF_ds[:,roi]))
                
            if all_trials_dF_results == {}:
                all_trials_dF_results = {
                        '%s_mean_dF_long' % mouse: mean_trials_dF_long_sig.tolist(),
                        '%s_mean_dF_short' % mouse: mean_trials_dF_short_sig.tolist(),
                        '%s_all_trial_dF_long' % mouse: all_trials_dF_long_sig.tolist(),
                        '%s_all_trial_dF_short' % mouse: all_trials_dF_short_sig.tolist(),
                        '%s_num_trials_short' % mouse: trial_size_short,
                        '%s_num_trials_long' % mouse: trial_size_long,
                        'track_short': int(binnr_short),
                        'track_long': int(binnr_long)
                        }
            else:
                all_trials_dF_results.update({
                        '%s_mean_dF_long' % mouse: mean_trials_dF_long_sig.tolist(),
                        '%s_mean_dF_short' % mouse: mean_trials_dF_short_sig.tolist(),
                        '%s_all_trial_dF_long' % mouse: all_trials_dF_long_sig.tolist(),
                        '%s_all_trial_dF_short' % mouse: all_trials_dF_short_sig.tolist(),
                        '%s_num_trials_short' % mouse: trial_size_short,
                        '%s_num_trials_long' % mouse: trial_size_long,
                        })
                      
      

    with open(content['figure_output_path'] + 'all_trials_dF_results' + os.sep + '_alltrials_andmean_dF_results_allfiles_norm_entire_sess.json','w+') as f:
        json.dump(all_trials_dF_results,f)


if __name__ == "__main__":
    # ALL LAYERS, ALL TASK ENGAGED
    
    import json
    import os
    import yaml
    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)
    with open(content['figure_output_path']  + 'recordings_with_behav_inc420.json','r') as f:
        recordings = json.load(f)
    good_recordings = recordings['good_recordings']
    fig_all_trials_dF_s5_test(good_recordings, 'task_engaged_all', 'both', 'short', 'png')

    