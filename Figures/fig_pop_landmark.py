"""
Plot population activity aligned to landmark offset

@author: lukasfischer

Parameters
------
behav_collection : list of ndarrays
		behavior data provided, list of datasets

dF_collection : list of ndarrays
		imaging data provided, list of datasets

rec_info : list
		information corresponding to behav_collection and dF_collection

fname : string
		filename for output plot

trials : string
		which trials to include {'c':correct, 'ic':incorrect, 'both':all trials}

sortby : string
		which trial type to sort by {'none':no sorting, 'short':sort by peak dF/F of short trials
									 'long':sort by dF/F peak of long trials, 'both': sort each trial type individually}


"""


def fig_pop_landmark(behav_collection, dF_collection, rec_info, fname=[], trials='both', sortby='both', fformat='agg'):
    # load local settings file
    import matplotlib
    matplotlib.use(fformat)
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("../Analysis")
    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from event_ind import event_ind
    from scipy import stats
    import yaml
    from scipy import signal
    import seaborn as sns
    sns.set_style('white')
    import os
    with open('../loc_settings.yaml', 'r') as f:
                content = yaml.load(f)

    sys.path.append(content['base_dir'] + 'Analysis')

    # timewindow in seconds from the beginning of the trial
    EVENT_TIMEWINDOW = [5,5]

    # track numbers
    track_short = 3
    track_long = 4

    # ROI
    roi_select = 0

    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    all_roi_avg_short = []
    all_roi_avg_long = []
    all_speed_avg_short = []
    all_speed_avg_long = []

    for i in range(len(behav_collection)):
        behav_ds = behav_collection[i]
        dF_ds = dF_collection[i]
        # pull out trial numbers of respective sections
        trials_short = filter_trials(behav_ds, [], ['tracknumber',track_short])
        trials_long = filter_trials(behav_ds, [], ['tracknumber',track_long])

        if trials == 'c':
            trials_short = filter_trials(behav_ds, [], ['trial_successful'],trials_short)
            trials_long = filter_trials(behav_ds, [], ['trial_successful'],trials_long)
        if trials == 'ic':
            trials_short = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short)
            trials_long = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long)

        # pull out indices of trial onsets for short and long trials
        events = event_ind(behav_ds, ['at_location',240])
        events = np.insert(events, 0, [1,1], axis=0)
        trials_short = np.intersect1d(events[:,1],trials_short)
        events_short = events[np.in1d(events[:,1],trials_short),:]
        trials_long = np.intersect1d(events[:,1],trials_long)
        events_long = events[np.in1d(events[:,1],trials_long),:]

        # grab peri-event dF trace for SHORT TRIALS
        trial_dF = np.zeros((np.size(events_short[:,0]),2))
        for i,cur_ind in enumerate(events_short):
            # determine indices of beginning and end of timewindow
            if behav_ds[cur_ind[0],0] - EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
                trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[cur_ind[0],0] - EVENT_TIMEWINDOW[0])[0][-1]
            else:
                trial_dF[i,0] = 0
            if behav_ds[cur_ind[0],0] + EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
                trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[cur_ind[0],0] + EVENT_TIMEWINDOW[1])[0][0]
            else:
                trial_dF[i,1] = np.size(behav_ds,0)

        # determine longest peri-event sweep (necessary due to sometimes varying framerates)
        t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
        cur_sweep_resampled_short = np.zeros((events_short.shape[0],int(t_max)))
        cur_sweep_resampled_short_speed = np.zeros((events_short.shape[0],int(t_max)))
        # resample every sweep to match the longest sweep
        roi_avg_short = np.zeros((t_max,dF_ds.shape[1]))
        for roi in range(dF_ds.shape[1]):
            for i in range(np.size(trial_dF,0)):
                cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
                cur_sweep_resampled_short[i,:] = signal.resample(cur_sweep, t_max, axis=0)
                cur_sweep_speed = behav_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),3]
                cur_sweep_resampled_short_speed[i,:] = signal.resample(cur_sweep_speed, t_max, axis=0)
                ax3.plot(cur_sweep_resampled_short_speed[i,:],c='g',lw=1)
            roi_avg_short[:,roi] = np.mean(cur_sweep_resampled_short, axis=0)/np.max(dF_ds[:,roi])#np.max(np.mean(cur_sweep_resampled_short, axis=0))
            speed_avg_short = np.mean(cur_sweep_resampled_short_speed,axis=0)


        # grab peri-event dF trace for LONG TRIALS

        trial_dF = np.zeros((np.size(events_long[:,0]),2))
        for i,cur_ind in enumerate(events_long):
            # determine indices of beginning and end of timewindow
            if behav_ds[cur_ind[0],0] - EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
                trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[cur_ind[0],0] - EVENT_TIMEWINDOW[0])[0][-1]
            else:
                trial_dF[i,0] = 0
            if behav_ds[cur_ind[0],0] + EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
                trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[cur_ind[0],0] + EVENT_TIMEWINDOW[1])[0][0]
            else:
                trial_dF[i,1] = np.size(behav_ds,0)

        # determine longest peri-event sweep (necessary due to sometimes varying framerates)
        t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
        cur_sweep_resampled_long = np.zeros((events_long.shape[0],int(t_max)))
        cur_sweep_resampled_long_speed = np.zeros((events_long.shape[0],int(t_max)))
        # resample every sweep to match the longest sweep
        roi_avg_long = np.zeros((t_max,dF_ds.shape[1]))
        for roi in range(dF_ds.shape[1]):
            for i in range(np.size(trial_dF,0)):
                cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
                cur_sweep_resampled_long[i,:] = signal.resample(cur_sweep, t_max, axis=0)
                cur_sweep_speed = behav_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),3]
                cur_sweep_resampled_long_speed[i,:] = signal.resample(cur_sweep_speed, t_max, axis=0)
                ax4.plot(cur_sweep_resampled_long_speed[i,:],c='g',lw=1)
            roi_avg_long[:,roi] = np.mean(cur_sweep_resampled_long, axis=0)/np.max(dF_ds[:,roi])
            speed_avg_long = np.mean(cur_sweep_resampled_long_speed,axis=0)

        # sort by peak activity
        mean_dF_sort_short = np.zeros(roi_avg_short.shape[1])
        for i, row in enumerate(np.transpose(roi_avg_short)):
            if not np.all(np.isnan(row)):
                mean_dF_sort_short[i] = np.nanargmax(row)
        sort_ind_short = np.argsort(mean_dF_sort_short)

        mean_dF_sort_long = np.zeros(roi_avg_long.shape[1])
        for i, row in enumerate(np.transpose(roi_avg_long)):
            if not np.all(np.isnan(row)):
                mean_dF_sort_long[i] = np.nanargmax(row)
        sort_ind_long = np.argsort(mean_dF_sort_long)


        if sortby == 'none':
            sns.heatmap(np.transpose(roi_avg_short), cmap='jet', vmin=0.0, vmax=1.0, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(roi_avg_long), cmap='jet', vmin=0.0, vmax=1.0, ax=ax2, cbar=False)
        elif sortby == 'short':
            sns.heatmap(np.transpose(roi_avg_short[:,sort_ind_short]), cmap='jet', vmin=0.0, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(roi_avg_long[:,sort_ind_short]), cmap='jet', vmin=0.0, ax=ax2, cbar=False)
        elif sortby == 'long':
            sns.heatmap(np.transpose(roi_avg_short[:,sort_ind_long]), cmap='jet', vmin=0.0, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(roi_avg_long[:,sort_ind_long]), cmap='jet', vmin=0.0, ax=ax2, cbar=False)
        elif sortby == 'both':
            sns.heatmap(np.transpose(roi_avg_short[:,sort_ind_short]), cmap='jet', vmin=0.0, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(roi_avg_long[:,sort_ind_long]), cmap='jet', vmin=0.0, ax=ax2, cbar=False)

        #sns.heatmap(np.transpose(roi_avg_short), cmap='jet', vmin=0, vmax=1, ax=ax1)


        event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
        ax1.axvline(event_loc,c='r')
        ax2.axvline(event_loc,c='r')
        plt.tight_layout()
        fig.suptitle(fname + '_' + trials + '_' + str([''.join(str(r) for r in ri) for ri in rec_info]),wrap=True)

        fullpath = content['figure_output_path'] + fname
        fig.savefig(fullpath)
