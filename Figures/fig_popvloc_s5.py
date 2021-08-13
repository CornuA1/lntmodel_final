"""
Plot population activity aligned to location. Select half the trials randomly
to calculate mean response and the other half to calculate the order by
which to plot the sequence

@author: lukasfischer

NOTE: this is an updated version of fig_popvlocHalfsort_stage5. This function can load multiple datasets and concatenate them (as opposed to using behav_collection and dF_collection)


"""


def fig_popvloc_s5(included_recordings, rois='task_engaged_all', trials='both', sortby='both', fformat='png', fname='', subfolder=''):
    # load local settings file
    import matplotlib
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("./Analysis")

    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal
    import statsmodels.api as sm
    import yaml
    import h5py
    import json
    import seaborn as sns
    sns.set_style('white')
    import os
    with open('./loc_settings.yaml', 'r') as f:
                content = yaml.load(f)

    # number of times calculations carried out
    NUM_ITER = 2

    # track paramaters (cm)
    reward_distance = 0
    tracklength_short = 320 + reward_distance
    tracklength_long = 380 + reward_distance
    track_start = 100
    bin_size = 5

    # track numbers used in this analysis
    track_short = 3
    track_long = 4

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


    popvec_cc_matrix_short_size = int(binnr_short)
    popvec_cc_matrix_long_size = int(binnr_long)

    # pull out location bins that exist in both track types
    popvec_cc_matrix_ss = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_ll = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))
    popvec_cc_matrix_sl = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_sl_stretched = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))

    popvec_cc_matrix_ss_pearsonr = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_ll_pearsonr = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))
    popvec_cc_matrix_sl_pearsonr = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))

    popvec_cc_reconstruction_ss = np.zeros((2,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_reconstruction_sl = np.zeros((2,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_reconstruction_ll = np.zeros((2,popvec_cc_matrix_long_size,NUM_ITER))

    std_prelm_ss = []
    std_lm_ss = []
    std_pi_ss = []

    std_prelm_sl = []
    std_lm_sl = []
    std_pi_sl = []

    std_prelm_ll = []
    std_lm_ll = []
    std_pi_ll = []

    for current_iter in range(NUM_ITER):
        tot_rois = 0
        mean_dF_short_coll_sig = []
        mean_dF_short_coll_sort = []
        mean_dF_long_coll_sig = []
        mean_dF_long_coll_sort = []
        mean_dF_short_coll_all = []
        mean_dF_long_coll_all = []

        print(current_iter)
        # load data
        for r in included_recordings:
            # load individual dataset
            print(r)
            mouse = r[0]
            session = r[1]
            # len > 2 indicates that we want to use a classification file in r[2], but plotting the data for r[1]
            if len(r) > 2:
                roi_classification_file = r[2]
            else:
                roi_classification_file = r[1]

            h5path = content['imaging_dir'] + mouse + '/' + mouse + '.h5'
            h5dat = h5py.File(h5path, 'r')
            behav_ds = np.copy(h5dat[session + '/behaviour_aligned'])
            dF_ds = np.copy(h5dat[session + '/dF_win'])
            h5dat.close()

            with open(content['figure_output_path']  + os.sep + 'Figures 20180810' + os.sep + mouse+roi_classification_file + os.sep + 'roi_classification.json') as f:
                roi_classification = json.load(f)

            # pull out trial numbers of respective sections
            trials_short_sig = filter_trials(behav_ds, [], ['tracknumber',track_short])
            trials_long_sig = filter_trials(behav_ds, [], ['tracknumber',track_long])

            # further filter if only correct or incorrect trials are plotted
            if trials == 'c':
                trials_short_sig = filter_trials(behav_ds, [], ['trial_successful'],trials_short_sig)
                trials_long = filter_trials(behav_ds, [], ['trial_successful'],trials_long)
            if trials == 'ic':
                trials_short_sig = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short_sig)
                trials_long = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long)

            # randomly draw 50% of indices to calculate signal and the other half to calculate the order by which to sort
            trials_short_rand = np.random.choice(trials_short_sig, np.size(trials_short_sig), replace=False)
            trials_short_sort = trials_short_rand[np.arange(0,np.floor(np.size(trials_short_sig)/2)).astype(int)]
            trials_short_sig = trials_short_rand[np.arange(np.ceil(np.size(trials_short_sig)/2),np.size(trials_short_sig)).astype(int)]

            trials_long_rand = np.random.choice(trials_long_sig, np.size(trials_long_sig), replace=False)
            trials_long_sort = trials_long_rand[np.arange(0,np.floor(np.size(trials_long_sig)/2)).astype(int)]
            trials_long_sig = trials_long_rand[np.arange(np.ceil(np.size(trials_long_sig)/2),np.size(trials_long_sig)).astype(int)]

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
            mean_dF_short_sig = np.zeros((binnr_short,np.size(roi_selection)))
            mean_dF_short_sort = np.zeros((binnr_short,np.size(roi_selection)))

            mean_dF_long_sig = np.zeros((binnr_long,np.size(roi_selection)))
            mean_dF_long_sort = np.zeros((binnr_long,np.size(roi_selection)))

            for j,roi in enumerate(roi_selection):
                # intilize matrix to hold data for all trials of this roi
                mean_dF_trials_short_sig = np.zeros((np.size(trials_short_sig,0),binnr_short))
                mean_dF_trials_short_sort = np.zeros((np.size(trials_short_sort,0),binnr_short))

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
                    mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges_short, (track_start, tracklength_short))[0]
                    # mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
                    mean_dF_trials_short_sig[k,:] = mean_dF_trial
                    #print(mean_dF_trial)

                #mean_dF_short_sig[:,j] = np.nanmean(mean_dF_trials_short_sig,0)
                mean_dF = np.nanmean(mean_dF_trials_short_sig,0)
                mean_dF_short_sig[:,j] = (mean_dF - np.nanmin(mean_dF))/(np.nanmax(mean_dF) - np.nanmin(mean_dF))
                mean_dF_short_sig[:,j] = mean_dF
                # print(np.nanmin(mean_dF), np.nanmax(mean_dF))

                mean_dF_short_sig[:,j] /= np.nanmax(np.abs(mean_dF_short_sig[start_bin:end_bin_short,j]))

                # calculate mean dF vs location for each ROI on short trials
                for k,t in enumerate(trials_short_sort):
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
                    mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges_short, (track_start, tracklength_short))[0]
                    # mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
                    mean_dF_trials_short_sort[k,:] = mean_dF_trial

                #mean_dF_short_sort[:,j] = np.nanmean(mean_dF_trials_short_sort,0)
                mean_dF_short_sort[:,j] /= np.nanmax(np.abs(mean_dF_short_sort[start_bin:end_bin_long,j]))
                mean_dF = np.nanmean(mean_dF_trials_short_sort,0)
                mean_dF_short_sort[:,j] = (mean_dF - np.nanmin(mean_dF))/(np.nanmax(mean_dF) - np.nanmin(mean_dF))
                mean_dF_short_sort[:,j] = mean_dF

                # calculate mean dF vs location for each ROI on long trials
                mean_dF_trials_long_sig = np.zeros((np.size(trials_long_sig,0),binnr_long))
                mean_dF_trials_long_sort = np.zeros((np.size(trials_long_sort,0),binnr_long))

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
                    # mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_long]))
                    mean_dF_trials_long_sig[k,:] = mean_dF_trial

                # mean_dF_long_sig[:,j] = np.nanmean(mean_dF_trials_long_sig,0)
                mean_dF_long_sig[:,j] /= np.nanmax(np.abs(mean_dF_long_sig[start_bin:end_bin_long,j]))
                mean_dF = np.nanmean(mean_dF_trials_long_sig,0)
                mean_dF_long_sig[:,j] = (mean_dF - np.nanmin(mean_dF))/(np.nanmax(mean_dF) - np.nanmin(mean_dF))
                mean_dF_long_sig[:,j] = mean_dF

                # calculate mean dF vs location for each ROI on long trials
                for k,t in enumerate(trials_long_sort):
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
                    mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_long]))
                    mean_dF_trials_long_sort[k,:] = mean_dF_trial

                # mean_dF_long_sort[:,j] = np.nanmean(mean_dF_trials_long_sort,0)
                # mean_dF_long_sort[:,j] /= np.nanmax(np.abs(mean_dF_long_sort[start_bin:end_bin_long,j]))
                mean_dF = np.nanmean(mean_dF_trials_long_sort,0)
                mean_dF_long_sort[:,j] = (mean_dF - np.nanmin(mean_dF))/(np.nanmax(mean_dF) - np.nanmin(mean_dF))
                mean_dF_long_sort[:,j] = mean_dF

            if mean_dF_short_coll_sig == []:
                mean_dF_short_coll_sig = mean_dF_short_sig
            else:
                mean_dF_short_coll_sig = np.append(mean_dF_short_coll_sig, mean_dF_short_sig, axis=1)

            if mean_dF_short_coll_sort == []:
                mean_dF_short_coll_sort = mean_dF_short_sort
            else:
                mean_dF_short_coll_sort = np.append(mean_dF_short_coll_sort, mean_dF_short_sort, axis=1)

            if mean_dF_long_coll_sig == []:
                mean_dF_long_coll_sig = mean_dF_long_sig
            else:
                mean_dF_long_coll_sig = np.append(mean_dF_long_coll_sig, mean_dF_long_sig, axis=1)

            if mean_dF_long_coll_sort == []:
                mean_dF_long_coll_sort = mean_dF_long_sort
            else:
                mean_dF_long_coll_sort = np.append(mean_dF_long_coll_sort, mean_dF_long_sort, axis=1)

            # calculate population vector from all trials (i.e. calculate the mean of sig and sort vectors)
            if mean_dF_short_coll_all == []:
                mean_dF_short_coll_all = (mean_dF_short_sig + mean_dF_short_sort) / 2
            else:
                mean_dF_short_coll_all = np.append(mean_dF_short_coll_all, (mean_dF_short_sig + mean_dF_short_sort) / 2, axis=1)
            if mean_dF_long_coll_all == []:
                mean_dF_long_coll_all = (mean_dF_long_sig + mean_dF_long_sort) / 2
            else:
                mean_dF_long_coll_all = np.append(mean_dF_long_coll_all, (mean_dF_long_sig + mean_dF_long_sort) / 2, axis=1)

        # run through every element
        for row in range(popvec_cc_matrix_short_size):
            for col in range(popvec_cc_matrix_short_size):
                # popvec_cc_matrix_sl[row,col,current_iter] = np.dot(mean_dF_short_coll_sig[col,:]/np.linalg.norm(mean_dF_short_coll_sig[col,:]), mean_dF_long_coll_sort[row,:]/np.linalg.norm(mean_dF_long_coll_sort[row,:]))
                # popvec_cc_matrix_ss[row,col,current_iter] = np.dot(mean_dF_short_coll_all[col,:]/np.linalg.norm(mean_dF_short_coll_all[col,:]), mean_dF_short_coll_all[row,:]/np.linalg.norm(mean_dF_short_coll_all[row,:]))

                popvec_cc_matrix_ss_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_short_coll_sig[row,:],mean_dF_short_coll_sort[col,:])[0]
                popvec_cc_matrix_sl_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_short_coll_sort[row,:],mean_dF_long_coll_sort[col,:])[0]

        for row in range(popvec_cc_matrix_long_size):
            for col in range(popvec_cc_matrix_long_size):
                # popvec_cc_matrix_ll[row,col,current_iter] = np.dot(mean_dF_long_coll_all[col,:]/np.linalg.norm(mean_dF_long_coll_all[col,:]), mean_dF_long_coll_all[row,:]/np.linalg.norm(mean_dF_long_coll_all[row,:]))
                popvec_cc_matrix_ll_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_long_coll_sig[row,:],mean_dF_long_coll_sort[col,:])[0]

        # run through every row and find element with largest correlation coefficient
        for i,row in enumerate(range(popvec_cc_matrix_short_size)):
            popvec_cc_reconstruction_ss[0,i,current_iter] = i
            popvec_cc_reconstruction_ss[1,i,current_iter] = np.argmax(popvec_cc_matrix_ss_pearsonr[i,:,current_iter])

            popvec_cc_reconstruction_sl[0,i,current_iter] = i
            popvec_cc_reconstruction_sl[1,i,current_iter] = np.argmax(popvec_cc_matrix_sl_pearsonr[i,:,current_iter])

        for i,row in enumerate(range(popvec_cc_matrix_long_size)):
            popvec_cc_reconstruction_ll[0,i,current_iter] = i
            popvec_cc_reconstruction_ll[1,i,current_iter] = np.argmax(popvec_cc_matrix_ll_pearsonr[i,:,current_iter])

        # calculate standard deviation (step by step so even an idiot like myself doesn't get confused) of reconstruction vs actual location
        bin_diff = popvec_cc_reconstruction_ss[1,:,current_iter] - popvec_cc_reconstruction_ss[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_ss = np.append(std_prelm_ss, np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_ss = np.append(std_lm_ss,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_ss = np.append(std_pi_ss,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))

        bin_diff = popvec_cc_reconstruction_sl[1,:,current_iter] - popvec_cc_reconstruction_sl[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_sl = np.append(std_prelm_sl,np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_sl = np.append(std_lm_sl,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_sl = np.append(std_pi_sl,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))

        bin_diff = popvec_cc_reconstruction_ll[1,:,current_iter] - popvec_cc_reconstruction_ll[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_ll = np.append(std_prelm_ll,np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_ll = np.append(std_lm_ll,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_ll = np.append(std_pi_ll,np.sqrt(np.sum(bin_diff[lm_end:end_bin_long])/(end_bin_long-lm_end)))

    #print(np.mean(std_prelm_ss), np.mean(std_lm_ss), np.mean(std_pi_ss))

    # perform statistical tests (one-way anova followed by pairwise tests).
    f_value_ss, p_value_ss = stats.f_oneway(std_prelm_ss,std_lm_ss,std_pi_ss)
    group_labels = ['prelm'] * len(std_prelm_ss) + ['lm'] * len(std_lm_ss) + ['pi'] * len(std_pi_ss)
    mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ss,std_lm_ss,std_pi_ss)),group_labels)
    posthoc_res_ss = mc_res_ss.tukeyhsd()

    f_value_sl, p_value_sl = stats.f_oneway(std_prelm_sl,std_lm_sl,std_pi_sl)
    group_labels = ['prelm'] * len(std_prelm_sl) + ['lm'] * len(std_lm_sl) + ['pi'] * len(std_pi_sl)
    mc_res_sl = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_sl,std_lm_sl,std_pi_sl)),group_labels)
    posthoc_res_sl = mc_res_sl.tukeyhsd()

    f_value_ll, p_value_ll = stats.f_oneway(std_prelm_ll,std_lm_ll,std_pi_ll)
    group_labels = ['prelm'] * len(std_prelm_ll) + ['lm'] * len(std_lm_ll) + ['pi'] * len(std_pi_ll)
    mc_res_ll = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ll,std_lm_ll,std_pi_ll)),group_labels)
    posthoc_res_ll = mc_res_ll.tukeyhsd()

    # calculate mean cc maps
    popvec_cc_matrix_ss_mean = np.mean(popvec_cc_matrix_ss_pearsonr,axis=2)
    popvec_cc_matrix_sl_mean = np.mean(popvec_cc_matrix_sl_pearsonr,axis=2)
    popvec_cc_matrix_ll_mean = np.mean(popvec_cc_matrix_ll_pearsonr,axis=2)

    # calculate reconstructed location estimate from mean cc map
    popvec_cc_reconstruction_ss_mean = np.zeros((2,popvec_cc_matrix_short_size))
    popvec_cc_reconstruction_sl_mean = np.zeros((2,popvec_cc_matrix_short_size))
    popvec_cc_reconstruction_ll_mean = np.zeros((2,popvec_cc_matrix_long_size))

    for i in range(popvec_cc_matrix_short_size):
        popvec_cc_reconstruction_ss_mean[0,i] = i
        popvec_cc_reconstruction_ss_mean[1,i] = np.argmax(popvec_cc_matrix_ss_mean[i,:])

        popvec_cc_reconstruction_sl_mean[0,i] = i
        popvec_cc_reconstruction_sl_mean[1,i] = np.argmax(popvec_cc_matrix_sl_mean[i,:])

    for i,row in enumerate(range(popvec_cc_matrix_long_size)):
        popvec_cc_reconstruction_ll_mean[0,i] = i
        popvec_cc_reconstruction_ll_mean[1,i] = np.argmax(popvec_cc_matrix_ll_mean[i,:])

    # # lm_start = 200/bin_size
    # # lm_end = 240/bin_size
    # # track_short_end = 320/bin_size
    # # track_long_end = 380/bin_size
    #
    # print(lm_end,end_bin_short)
    # print(popvec_cc_reconstruction_ss_mean)
    # bin_diff = popvec_cc_reconstruction_ss_mean[1,:] - popvec_cc_reconstruction_ss_mean[0,:]
    # bin_diff = bin_diff * bin_diff
    # print('SS:')
    # print(bin_diff[lm_end:end_bin_short])
    # print(sum(bin_diff[lm_end:end_bin_short]))
    # # print(np.sqrt(np.sum(bin_diff[0:20])/20))
    # # print(np.sqrt(np.sum(bin_diff[20:28])/8))
    # print(np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))
    #
    # bin_diff = popvec_cc_reconstruction_sl_mean[1,:] - popvec_cc_reconstruction_sl_mean[0,:]
    # bin_diff = bin_diff * bin_diff
    # print('SL:')
    # print(bin_diff[lm_end:end_bin_short])
    # print(sum(bin_diff[lm_end:end_bin_short]))
    # # print(np.sqrt(np.sum(bin_diff[0:20])/20))
    # # print(np.sqrt(np.sum(bin_diff[20:28])/8))
    # print(np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))

    # create figure and axes
    fig = plt.figure(figsize=(25,12))
    ax1 = plt.subplot2grid((5,200),(0,0),rowspan=4, colspan=45)
    ax2 = plt.subplot2grid((5,200),(0,50),rowspan=4, colspan=55)
    # ax3 = plt.subplot2grid((5,200),(0,0),rowspan=1, colspan=40)
    # ax4 = plt.subplot2grid((5,200),(0,50),rowspan=1, colspan=55)
    ax5 = plt.subplot2grid((5,200),(0,110),rowspan=2, colspan=40)
    ax6 = plt.subplot2grid((5,200),(2,110),rowspan=2, colspan=40)
    ax7 = plt.subplot2grid((5,200),(0,155),rowspan=2, colspan=40)
    ax8 = plt.subplot2grid((5,200),(2,155),rowspan=2, colspan=40)
    # ax6 = plt.subplot2grid((5,2),(4,1),rowspan=1)

    #ax7.pcolormesh(popvec_cc_matrix_sl)
    ax7_img = ax7.pcolormesh(popvec_cc_matrix_sl_mean.T,cmap='viridis')
    plt.colorbar(ax7_img, ax=ax7)
    # ax7.plot(popvec_cc_reconstruction_sl_mean[0,:],popvec_cc_reconstruction_sl_mean[1,:],c='r')
    ax7.plot(popvec_cc_reconstruction_ll_mean[0,:],popvec_cc_reconstruction_ll_mean[0,:],c='k',ls='--')
    ax7.axvline((200/bin_size)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    ax7.axvline((240/bin_size)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    ax7.axhline((200/bin_size)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    ax7.axhline((240/bin_size)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    ax7.set_xlabel('long')
    ax7.set_ylabel('short')
    ax7.set_xlim([0,popvec_cc_matrix_short_size])
    ax7.set_ylim([0,popvec_cc_matrix_short_size])
    ax7.set_xticklabels([])
    ax7.set_yticklabels([])

    #ax5.pcolormesh(popvec_cc_matrix_ss)
    ax5_img = ax5.pcolor(popvec_cc_matrix_ss_mean.T,cmap='viridis')
    plt.colorbar(ax5_img, ax=ax5)
    # ax5.plot(popvec_cc_reconstruction_ss_mean[0,:],popvec_cc_reconstruction_ss_mean[1,:],c='r')
    ax5.plot(popvec_cc_reconstruction_ss_mean[0,:],popvec_cc_reconstruction_ss_mean[0,:],c='k',ls='--')
    ax5.axvline((200/bin_size)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    ax5.axvline((240/bin_size)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    ax5.axhline((200/bin_size)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    ax5.axhline((240/bin_size)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    ax5.set_xlabel('short')
    ax5.set_ylabel('short')
    ax5.set_xlim([0,popvec_cc_matrix_short_size])
    ax5.set_ylim([0,popvec_cc_matrix_short_size])
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])

    #ax6.pcolormesh(popvec_cc_matrix_ll)
    ax6_img = ax6.pcolormesh(popvec_cc_matrix_ll_mean.T,cmap='viridis')
    plt.colorbar(ax6_img, ax=ax6)
    # ax6.plot(popvec_cc_reconstruction_ll_mean[0,:],popvec_cc_reconstruction_ll_mean[1,:],c='r')
    ax6.plot(popvec_cc_reconstruction_ll_mean[0,:],popvec_cc_reconstruction_ll_mean[0,:],c='k',ls='--')
    ax6.axvline((200/bin_size)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    ax6.axvline((240/bin_size)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    ax6.axhline((200/bin_size)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    ax6.axhline((240/bin_size)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    ax6.set_xlabel('long')
    ax6.set_ylabel('long')
    ax6.set_xlim([0,popvec_cc_matrix_long_size])
    ax6.set_ylim([0,popvec_cc_matrix_long_size])
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])

    #ax8.set_xlim([0,1])
    # ax8.bar([0.25,0.5,0.75],[np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)],-0.05, color=sns.xkcd_rgb["windows blue"],lw=0,yerr=[stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],ecolor='k')
    # ax8.bar([0.25,0.5,0.75],[np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], 0.05, color=sns.xkcd_rgb["dusty purple"],lw=0,yerr=[stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],ecolor='k')
    # ax8.bar([0.3,0.55,0.8],[np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], 0.05, color='0.5',lw=0,yerr=[stats.sem(std_prelm_sl),stats.sem(std_lm_sl),stats.sem(std_pi_sl)],ecolor='k')

    ax8.scatter([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], s=120,color=sns.xkcd_rgb["windows blue"], linewidths=0, zorder=2)
    ax8.scatter([0.27,0.52,0.77], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], s=120,color=sns.xkcd_rgb["dusty purple"], linewidths=0, zorder=2)
    ax8.scatter([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], s=120,color='0.5', linewidths=0, zorder=2)

    ax8.errorbar([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], yerr=[stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],fmt='none',ecolor='k', zorder=1)
    ax8.errorbar([0.27,0.52,0.77], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], yerr=[stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],fmt='none',ecolor='k', zorder=1)
    ax8.errorbar([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], yerr=[stats.sem(std_prelm_sl),stats.sem(std_lm_sl),stats.sem(std_pi_sl)],fmt='none',ecolor='k', zorder=1)

    ax8.plot([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], lw=2, c=sns.xkcd_rgb["windows blue"])
    ax8.plot([0.27,0.52,0.77], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], lw=2, c=sns.xkcd_rgb["dusty purple"])
    ax8.plot([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], lw=2, c='0.5')

    ax8.tick_params(length=5,width=2,bottom=False,left=True,top=False,right=False,labelsize=14)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['bottom'].set_visible(False)

    ax8.set_xticks([0.27,0.52,0.77])
    ax8.set_xticklabels([], rotation=45, fontsize=20)

    # add roi stats info
    fig.text(0.1,0.1,'total number of rois: ' + str(tot_rois))

    # sort by peak activity (naming of variables confusing because the script grew organically...)
    mean_dF_sort_short = np.zeros(mean_dF_short_coll_sort.shape[1])
    for i, row in enumerate(mean_dF_short_coll_sort.T):
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

    mean_dF_sort_long = np.zeros(mean_dF_long_coll_sort.shape[1])
    for i, row in enumerate(mean_dF_long_coll_sort.T):
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
    popvloc_results = {
        'included_datasets' : included_recordings,
        'short_short' : [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)],
        'short_long' : [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)],
        'long_long' : [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)],
        'short_short_SEM' : [stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],
        'short_long_SEM' : [stats.sem(std_prelm_sl),stats.sem(std_lm_sl),stats.sem(std_pi_sl)],
        'long_long_SEM' : [stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],
        'short_short_f_value' : f_value_ss,
        'short_short_p_value' : p_value_ss,
        'short_short_multicomparison' : [posthoc_res_ss.reject[0].item(),posthoc_res_ss.reject[1].item(),posthoc_res_ss.reject[2].item()],
        'short_long_f_value' : f_value_sl,
        'short_long_p_value' : p_value_sl,
        'short_long_multicomparison' : [posthoc_res_sl.reject[0].item(),posthoc_res_sl.reject[1].item(),posthoc_res_sl.reject[2].item()],
        'long_long_f_value' : f_value_ll,
        'long_long_p_value' : p_value_ll,
        'long_long_multicomparison' : [posthoc_res_ll.reject[0].item(),posthoc_res_ll.reject[1].item(),posthoc_res_ll.reject[2].item()],
        'iterations:' : [NUM_ITER],
        'popvec_cc_reconstruction_ss' : popvec_cc_reconstruction_ss.tolist(),
        'popvec_cc_reconstruction_sl' : popvec_cc_reconstruction_sl.tolist(),
        'popvec_cc_reconstruction_ll' : popvec_cc_reconstruction_ll.tolist()
    }

    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)

    with open(content['figure_output_path'] + 'popvloc' + os.sep + fname + '_popvloc_results_' + str(NUM_ITER) + '.json','w+') as f:
        json.dump(popvloc_results,f)

    fname = content['figure_output_path'] + subfolder + os.sep + fname + '_' + str(NUM_ITER) + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)


if __name__ == "__main__":
    # ALL LAYERS, ALL TASK ENGAGED
    figure_datasets = [['LF170110_2','Day20170331'], ['LF170222_1','Day20170615'],
    ['LF170420_1','Day20170719'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170613_1','Day201784']]
    fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_all', 'popvloc')
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

    # figure_datasets = [['LF170613_1','Day201784']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', figure_datasets[0][0]+figure_datasets[0][1], 'popvloc')

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
    # figure_datasets = [['LF170214_1','Day201777_openloop','Day201777'],['LF170214_1','Day2017714_openloop','Day2017714'],['LF171211_2','Day201852_openloop','Day201852']]
    # fig_popvloc_s5(figure_datasets, 'task_engaged_all', 'both', 'both', 'png', 'task_engaged_V1_openloop', 'popvloc')

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
