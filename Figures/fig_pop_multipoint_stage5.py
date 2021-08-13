"""
Plot population activity aligned by onset, landmark offset and reward

@author: lukasfischer


"""


def fig_pop_multipoint_stage5(behav_collection, dF_collection, rec_info, fname=[], trials='both', sortby='both', fformat='png'):
    # load local settings file
    import matplotlib
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("../Analysis")
    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    import yaml
    import seaborn as sns
    sns.set_style('white')
    import os
    from scipy import signal

    from event_ind import event_ind
    with open('../loc_settings.yaml', 'r') as f:
                content = yaml.load(f)

    sys.path.append(content['base_dir'] + 'Analysis')

    # timewindow (minus and plus time in seconds)
    EVENT_TIMEWINDOW_ONSET = [2,2]
    EVENT_TIMEWINDOW_LANDMARK = [1,3]
    EVENT_TIMEWINDOW_REWARD = [1,3]

    # sampling frequency used for plotting (this is necessary as the imaging 
    # data is reampled and aligned to the behavior data which can fluctuate
    # if the PC couldnt run the VR at exactly 60 Hz)
    fs_plot = 60

    # specify track numbers
    track_short = 3
    track_long = 4

    # Y scale max for heatmap
    YMAX = 3

    ymax_trace = 0

    # create figure to later plot on
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((2,3),(0,0),colspan=1,rowspan=2)
    ax2 = plt.subplot2grid((2,3),(0,1),colspan=1,rowspan=2)
    ax3 = plt.subplot2grid((2,3),(0,2),colspan=1,rowspan=2)

    roi_all_onset = []
    roi_all_landmark = []
    roi_all_reward = []

    for i in range(len(behav_collection)):
        behav_ds = behav_collection[i]
        dF_ds = dF_collection[i]

        # Get event indices and make sure only trials where all 3 events happen are considered
        filt_trials = filter_trials(behav_ds,[],['cut tracknumber', 5])
        if trials=='short':
            filt_trials = filter_trials(behav_ds,[],['tracknumber',track_short], filt_trials)
        elif trials == 'long':
            filt_trials = filter_trials(behav_ds,[],['tracknumber',track_long], filt_trials)

        events_onset = event_ind(behav_ds,['trial_transition',4])
        events_landmark = event_ind(behav_ds,['at_location', 240])
        events_reward = event_ind(behav_ds,['rewards_all',-1])

        pass_t_onset = np.intersect1d(events_onset[:,1],filt_trials)
        pass_t_landmark = np.intersect1d(events_landmark[:,1],filt_trials)
        pass_t_reward = np.intersect1d(events_reward[:,1],filt_trials)

        pass_t_onset = np.intersect1d(pass_t_onset,pass_t_landmark)
        pass_t_onset = np.intersect1d(pass_t_onset,pass_t_reward)
        pass_t_landmark = np.intersect1d(pass_t_landmark,pass_t_onset)
        pass_t_landmark = np.intersect1d(pass_t_landmark,pass_t_reward)
        pass_t_reward = np.intersect1d(pass_t_reward,pass_t_onset)
        pass_t_reward = np.intersect1d(pass_t_reward,pass_t_landmark)

        events_onset = events_onset[np.in1d(events_onset[:,1],pass_t_onset),:]
        events_landmark = events_landmark[np.in1d(events_landmark[:,1],pass_t_landmark),:]
        events_reward = events_reward[np.in1d(events_reward[:,1],pass_t_reward),:]

        # ONSET INDICES
        trial_dF_onset = np.zeros((np.size(events_onset[:,0]),2))
        for j,cur_ind in enumerate(events_onset):
            # determine indices of beginning and end of timewindow
            if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW_ONSET[0] > behav_ds[0,0]:
                trial_dF_onset[j,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW_ONSET[0])[0][-1]
            else:
                trial_dF_onset[j,0] = 0
            if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW_ONSET[1] < behav_ds[-1,0]:
                trial_dF_onset[j,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW_ONSET[1])[0][0]
            else:
                trial_dF_onset[j,1] = np.size(behav_ds,0)

        # LANDMARK INDICES
        trial_dF_landmark = np.zeros((np.size(events_landmark[:,0]),2))
        for j,cur_ind in enumerate(events_landmark):
            # determine indices of beginning and end of timewindow
            if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW_LANDMARK[0] > behav_ds[0,0]:
                trial_dF_landmark[j,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW_LANDMARK[0])[0][-1]
            else:
                trial_dF_landmark[j,0] = 0
            if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW_LANDMARK[1] < behav_ds[-1,0]:
                trial_dF_landmark[j,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW_LANDMARK[1])[0][0]
            else:
                trial_dF_landmark[j,1] = np.size(behav_ds,0)

        # REWARD INDICES
        trial_dF_reward = np.zeros((np.size(events_reward[:,0]),2))
        for j,cur_ind in enumerate(events_reward):
            # determine indices of beginning and end of timewindow
            if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW_REWARD[0] > behav_ds[0,0]:
                trial_dF_reward[j,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW_REWARD[0])[0][-1]
            else:
                trial_dF_reward[j,0] = 0
            if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW_REWARD[1] < behav_ds[-1,0]:
                trial_dF_reward[j,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW_REWARD[1])[0][0]
            else:
                trial_dF_reward[j,1] = np.size(behav_ds,0)

        # determine longest peri-event sweep (necessary due to sometimes varying framerates)
#        t_max = np.amax(trial_dF_onset[:,1] - trial_dF_onset[:,0])
        t_max_onset = np.sum(EVENT_TIMEWINDOW_ONSET)*fs_plot
        t_max_landmark = np.sum(EVENT_TIMEWINDOW_LANDMARK)*fs_plot
        t_max_reward = np.sum(EVENT_TIMEWINDOW_REWARD)*fs_plot

        cur_sweep_resampled_onset = np.zeros((np.size(events_onset[:,0]),int(t_max_onset)))
        cur_sweep_resampled_landmark = np.zeros((np.size(events_landmark[:,0]),int(t_max_landmark)))
        cur_sweep_resampled_reward = np.zeros((np.size(events_reward[:,0]),int(t_max_reward)))
        roi_avg_onset = np.zeros((dF_ds.shape[1],int(t_max_onset)))
        roi_avg_landmark = np.zeros((dF_ds.shape[1],int(t_max_landmark)))
        roi_avg_reward = np.zeros((dF_ds.shape[1],int(t_max_reward)))

        # loop through each ROI, pull out and resample every sweep to match the longest sweep
        for j,r in enumerate(np.transpose(dF_ds)):
            for k in range(np.size(trial_dF_onset,0)):
                cur_sweep = dF_ds[int(trial_dF_onset[k,0]):int(trial_dF_onset[k,1]),j]
                cur_sweep_resampled_onset[k,:] = signal.resample(cur_sweep, int(t_max_onset), axis=0)

            for k in range(np.size(trial_dF_landmark,0)):
                cur_sweep = dF_ds[int(trial_dF_landmark[k,0]):int(trial_dF_landmark[k,1]),j]
                cur_sweep_resampled_landmark[k,:] = signal.resample(cur_sweep, int(t_max_landmark), axis=0)

            for k in range(np.size(trial_dF_reward,0)):
                cur_sweep = dF_ds[int(trial_dF_reward[k,0]):int(trial_dF_reward[k,1]),j]
                cur_sweep_resampled_reward[k,:] = signal.resample(cur_sweep, int(t_max_reward), axis=0)

            # get avg roi values across trials
            roi_avg_onset[j,:] = np.nanmean(cur_sweep_resampled_onset,0)
            #roi_avg_onset[j,:] /= np.nanmax(roi_avg_onset[j,:])
            roi_avg_landmark[j,:] = np.nanmean(cur_sweep_resampled_landmark,0)
            #roi_avg_landmark[j,:] /= np.nanmax(roi_avg_landmark[j,:])
            roi_avg_reward[j,:] = np.nanmean(cur_sweep_resampled_reward,0)
            #roi_avg_reward[j,:] /= np.nanmax(roi_avg_reward[j,:])

        if roi_all_onset == []:
            roi_all_onset = roi_avg_onset
        else:
            roi_all_onset = np.append(roi_all_onset,roi_avg_onset,axis=0)

        if roi_all_landmark == []:
            roi_all_landmark = roi_avg_landmark
        else:
            roi_all_landmark = np.append(roi_all_landmark,roi_avg_landmark,axis=0)

        if roi_all_reward == []:
            roi_all_reward = roi_avg_reward
        else:
            roi_all_reward = np.append(roi_all_reward,roi_avg_reward,axis=0)

    # sort by peak activity across all 3 alignment points
    roi_peak_dF = np.zeros(roi_all_onset.shape[0])
    for i in range(roi_all_onset.shape[0]):
        roi_dF_all = np.hstack([roi_all_onset[i,:],roi_all_landmark[i,:],roi_all_reward[i,:]])
        roi_dF_all_max = np.nanmax(roi_dF_all)
        roi_all_onset[i,:] /= roi_dF_all_max
        roi_all_landmark[i,:] /= roi_dF_all_max
        roi_all_reward[i,:] /= roi_dF_all_max
        roi_peak_dF[i] = np.nanargmax(roi_dF_all)

    peak_dF_sort = np.argsort(roi_peak_dF)

    #print(roi_peak_dF.shape)

#    peak_dF_onset = np.zeros(roi_all_onset.shape[0])
#    for i in range(roi_all_onset.shape[0]):
#        peak_dF_onset[i] = np.nanargmax(roi_all_onset[i,:])
#    peak_dF_onset_sort = np.argsort(peak_dF_onset)

#    peak_dF_landmark = np.zeros(roi_all_landmark.shape[0])
#    for i in range(roi_all_landmark.shape[0]):
#        peak_dF_landmark[i] = np.nanargmax(roi_all_landmark[i,:])
#    peak_dF_landmark_sort = np.argsort(peak_dF_landmark)

#    peak_dF_reward = np.zeros(roi_all_reward.shape[0])
#    for i in range(roi_all_reward.shape[0]):
#        peak_dF_reward[i] = np.nanargmax(roi_all_reward[i,:])
#    peak_dF_reward_sort = np.argsort(peak_dF_reward)

    # calc where to draw the line for the event
    event_loc_onset = (t_max_onset/(EVENT_TIMEWINDOW_ONSET[0] + EVENT_TIMEWINDOW_ONSET[1]))*EVENT_TIMEWINDOW_ONSET[0]
    event_loc_landmark = (t_max_landmark/(EVENT_TIMEWINDOW_LANDMARK[0] + EVENT_TIMEWINDOW_LANDMARK[1]))*EVENT_TIMEWINDOW_LANDMARK[0]
    event_loc_reward = (t_max_reward/(EVENT_TIMEWINDOW_REWARD[0] + EVENT_TIMEWINDOW_REWARD[1]))*EVENT_TIMEWINDOW_REWARD[0]

    sns.heatmap(roi_all_onset[peak_dF_sort,:],cmap='jet',vmin=0,yticklabels=False,xticklabels=False,ax=ax1,cbar=False)
    sns.heatmap(roi_all_landmark[peak_dF_sort,:],cmap='jet',vmin=0,yticklabels=False,xticklabels=False,ax=ax2,cbar=False)
    sns.heatmap(roi_all_reward[peak_dF_sort,:],cmap='jet',vmin=0,yticklabels=False,xticklabels=False,ax=ax3,cbar=False)
    ax1.axvline(event_loc_onset, c='0.8',lw=5)
    ax2.axvline(event_loc_landmark, c='0.8',lw=5)
    ax3.axvline(event_loc_reward, c='0.8',lw=5)

    plt.tight_layout()

    if not os.path.isdir(content['figure_output_path']):
        os.mkdir(content['figure_output_path'])
    fname = content['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat)




















