"""
Plot trace of an individual ROI centered around at movement onset

@author: lukasfischer

"""

def fig_moveOnset_centered(h5path, sess, roi, fname, ylims=[-0.5,3], fformat='png', subfolder=[], trialpattern=[]):
    import numpy as np
    import sys
    import os
    import yaml
    from yaml_mouselist import yaml_mouselist
    import warnings; warnings.simplefilter('ignore')

    import matplotlib
    from matplotlib import pyplot as plt
    from event_ind import event_ind
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal

    import h5py

    import seaborn as sns
    sns.set_style("white")

    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    # timewindow (minus and plus time in seconds)
    EVENT_TIMEWINDOW = [3,3]

    if ylims == []:
        ylims = [-0.5,3]

    # specify track numbers
    track_short = 5
    track_long = 5

    # create figure to later plot on
    fig = plt.figure(figsize=(8,10))
    ax1 = plt.subplot(421)
    ax2 = plt.subplot(422)
    ax3 = plt.subplot(423)
    ax4 = plt.subplot(424)
    ax5 = plt.subplot(425)
    ax6 = plt.subplot(426)
    ax7 = plt.subplot(427)
    ax8 = plt.subplot(428)

    ax1.set_title('dF/F vs time')
    ax2.set_title('dF/F vs time')
    ax3.set_title('running speed (cm/s) vs time')
    ax4.set_title('running speed (cm/s) vs time')

    # get indices of desired behavioural event
    events = event_ind(behav_ds,['movement_onset',1,3])

    temp = events.astype('int')
    events = temp

    valid_trials = np.unique(behav_ds[:,6])

    trials_short = filter_trials(behav_ds, [], ['tracknumber',track_short], valid_trials)
    trials_long = filter_trials(behav_ds, [], ['tracknumber',track_long], valid_trials)

    # if a bool pattern was provided, apply it to trial numbers
    if trialpattern != []:
        trials_short = filter_trials(behav_ds, [], ['bool_pattern', trialpattern], trials_short)
        trials_long = filter_trials(behav_ds, [], ['bool_pattern', trialpattern], trials_long)

    # intersect trial numbers of short and long trials with events
    pass_t_short = np.intersect1d(events[:,1],trials_short)
    events_short = events[np.in1d(events[:,1],pass_t_short),:]

    pass_t_long = np.intersect1d(events[:,1],trials_long)
    events_long = events[np.in1d(events[:,1],pass_t_long),:]

    # grab peri-event dF trace for each event
    trial_dF = np.zeros((np.size(events[:,0]),2))
    for i,cur_ind in enumerate(events):
        # determine indices of beginning and end of timewindow
        if behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i,0] = 0
        if behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i,1] = np.size(behav_ds,0)
    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = int(np.amax(trial_dF[:,1] - trial_dF[:,0]))
    cur_sweep_resampled = np.zeros((np.size(events[:,0]),int(t_max)))
    cur_sweep_resampled_speed = np.zeros((np.size(events[:,0]),int(t_max)))

    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF,0)):
        cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
        cur_sweep_resampled[i,:] = signal.resample(cur_sweep, t_max, axis=0)
        ax1.plot(cur_sweep_resampled[i,:],c='0.65',lw=1)

        cur_sweep_speed = behav_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),3]
        cur_sweep_resampled_speed[i,:] = signal.resample(cur_sweep_speed, t_max, axis=0)
        #ax3.plot(cur_sweep_resampled_speed[i,:],lw=1,c='0.5')
    event_avg_short = np.mean(cur_sweep_resampled_speed,axis=0)
    sem_speed_short = stats.sem(cur_sweep_resampled_speed,0,nan_policy='omit')
    ax3.fill_between(np.arange(len(event_avg_short)), event_avg_short-sem_speed_short, event_avg_short, alpha=0.25, lw=0, color=sns.xkcd_rgb["windows blue"])
    ax3.fill_between(np.arange(len(event_avg_short)), event_avg_short+sem_speed_short, event_avg_short, alpha=0.25, lw=0, color=sns.xkcd_rgb["windows blue"])

    # grab peri-event dF trace for each event SHORT TRIALS
    trial_dF_short = np.zeros((np.size(events_short[:,0]),2))
    for i,cur_ind in enumerate(events_short):
        # determine indices of beginning and end of timewindow
        if behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF_short[i,0] = np.where(behav_ds[:,0] < behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF_short[i,0] = 0
        if behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF_short[i,1] = np.where(behav_ds[:,0] > behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF_short[i,1] = np.size(behav_ds,0)
    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = int(np.amax(trial_dF_short[:,1] - trial_dF_short[:,0]))
    cur_sweep_resampled_short = np.zeros((np.size(events_short[:,0]),int(t_max)))

    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF_short,0)):
        cur_sweep = dF_ds[int(trial_dF_short[i,0]):int(trial_dF_short[i,1]),roi]
        cur_sweep_resampled_short[i,:] = signal.resample(cur_sweep, t_max, axis=0)
        #ax7.plot(cur_sweep_resampled_short[i,:],c='0.65',lw=1)
    event_avg_short = np.mean(cur_sweep_resampled_short,axis=0)
    sem_dF_short = stats.sem(cur_sweep_resampled_short,0,nan_policy='omit')
    ax7.fill_between(np.arange(len(event_avg_short)), event_avg_short-sem_dF_short, event_avg_short, alpha=0.25, lw=0, color=sns.xkcd_rgb["windows blue"])
    ax7.fill_between(np.arange(len(event_avg_short)), event_avg_short+sem_dF_short, event_avg_short, alpha=0.25, lw=0, color=sns.xkcd_rgb["windows blue"])

        # grab peri-event dF trace for each event LONG TRIALS
    trial_dF_long = np.zeros((np.size(events_long[:,0]),2))
    for i,cur_ind in enumerate(events_long):
        # determine indices of beginning and end of timewindow
        if behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF_long[i,0] = np.where(behav_ds[:,0] < behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF_long[i,0] = 0
        if behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF_long[i,1] = np.where(behav_ds[:,0] > behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF_long[i,1] = np.size(behav_ds,0)
    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = int(np.amax(trial_dF_long[:,1] - trial_dF_long[:,0]))
    cur_sweep_resampled_long = np.zeros((np.size(events_long[:,0]),int(t_max)))

    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF_long,0)):
        cur_sweep = dF_ds[int(trial_dF_long[i,0]):int(trial_dF_long[i,1]),roi]
        cur_sweep_resampled_long[i,:] = signal.resample(cur_sweep, t_max, axis=0)
        #ax8.plot(cur_sweep_resampled_long[i,:],c='0.65',lw=1)
    event_avg_long = np.mean(cur_sweep_resampled_long,axis=0)
    sem_dF_long = stats.sem(cur_sweep_resampled_long,0,nan_policy='omit')
    ax8.fill_between(np.arange(len(event_avg_long)), event_avg_long-sem_dF_long, event_avg_long, alpha=0.25, lw=0, color=sns.xkcd_rgb["dusty purple"])
    ax8.fill_between(np.arange(len(event_avg_long)), event_avg_long+sem_dF_long, event_avg_long, alpha=0.25, lw=0, color=sns.xkcd_rgb["dusty purple"])

    event_avg = np.mean(cur_sweep_resampled,axis=0)
    speed_avg = np.mean(cur_sweep_resampled_speed,axis=0)
    ax1.plot(event_avg,c='k',lw=2)
    onsetActiveIndex = (np.mean(event_avg[int(np.ceil(event_avg.shape[0]/2)):]) - np.mean(event_avg[0:int(np.floor(event_avg.shape[0]/2))]))# / np.mean(event_avg[0:np.floor(event_avg.shape[0]/2)])
    #onsetActiveIndex_norm = (np.mean(event_avg[int(np.ceil(event_avg.shape[0]/2)):]) - np.mean(event_avg[0:int(np.floor(event_avg.shape[0]/2))])) / np.absolute((np.mean(event_avg[int(np.ceil(event_avg.shape[0]/2)):]) + np.mean(event_avg[0:int(np.floor(event_avg.shape[0]/2))])))
    ax1.set_title(onsetActiveIndex)
    ax7.plot(event_avg_short, c=sns.xkcd_rgb["windows blue"], lw=4)
    ax8.plot(event_avg_long, c=sns.xkcd_rgb["dusty purple"], lw=4)

    ax7.set_ylim(ylims)
    ax8.set_ylim(ylims)

    ax3.plot(speed_avg,c='k',lw=2)
    ax1.set_xlim([0,t_max])
    ax1.set_ylim(ylims)

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]

    if fformat is 'png':
        sns.heatmap(cur_sweep_resampled,cmap='jet',vmin=0,vmax=ylims[1],yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax2)
        sns.heatmap(cur_sweep_resampled_speed,cmap='jet',vmin=0,yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax4)
        sns.heatmap(cur_sweep_resampled_short,cmap='jet',vmin=0,vmax=ylims[1],yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax5)
        sns.heatmap(cur_sweep_resampled_long,cmap='jet',vmin=0,vmax=ylims[1],yticklabels=events_long[:,1].astype('int'),xticklabels=False,ax=ax6)

    ax1.axvline(event_loc,c='r')
    ax2.axvline(event_loc,c='r')
    ax3.axvline(event_loc,c='r')
    ax4.axvline(event_loc,c='r')
    ax5.axvline(event_loc,c='r')
    ax6.axvline(event_loc,c='r')
    ax7.axvline(event_loc,c='r')
    ax8.axvline(event_loc,c='r')

    fig.suptitle('MoveOnset' + fname, wrap=True)
    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    # dirty way to control whether this script also spits out an onset active index for quantification
    if True:

        number_of_shuffles = 100
        shuffledOnsetActiveIndices = np.zeros(number_of_shuffles)

        for shuffle in range(number_of_shuffles):

            # get indices of desired behavioural event
            events = event_ind(behav_ds,['movement_onset',1,1])

            temp = events.astype('int')
            events = temp

            # shuffle event indices by a random integer between 0 and session length
            events += np.random.randint(0, np.size(behav_ds[:, 0]))

            # make sure the events are within the dataset by wrapping around
            events = events % np.size(behav_ds[:, 0])

            # grab peri-event dF trace for each event
            trial_dF = np.zeros((np.size(events[:,0]),2))
            for i,cur_ind in enumerate(events):
                # determine indices of beginning and end of timewindow
                if behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
                    trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0])[0][-1]
                else:
                    trial_dF[i,0] = 0
                if behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
                    trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1])[0][0]
                else:
                    trial_dF[i,1] = np.size(behav_ds,0)

            # determine longest peri-event sweep (necessary due to sometimes varying framerates)
            t_max = int(np.amax(trial_dF[:,1] - trial_dF[:,0]))
            cur_sweep_resampled = np.zeros((np.size(events[:,0]),int(t_max)))

            # resample every sweep to match the longest sweep
            for i in range(np.size(trial_dF,0)):
                cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
                cur_sweep_resampled[i,:] = signal.resample(cur_sweep, t_max, axis=0)

            event_avg = np.mean(cur_sweep_resampled,axis=0)

            shuffledOnsetActiveIndices[shuffle] = (np.mean(event_avg[int(np.ceil(event_avg.shape[0]/2)):]) - np.mean(event_avg[0:int(np.floor(event_avg.shape[0]/2))]))# / np.mean(event_avg[0:np.floor(event_avg.shape[0]/2)])

        z_score = (onsetActiveIndex - np.mean(shuffledOnsetActiveIndices))/np.std(shuffledOnsetActiveIndices)

        return onsetActiveIndex, shuffledOnsetActiveIndices, z_score

def fig_onset_detection_test(h5path, sess, roi, fname, fformat='png', subfolder=[]):
    """ plot figure with running speed trace (filtered) and instances where a movement onset was detected to allow manual inspection of detection performance """
    import numpy as np
    import sys
    import os
    import yaml
    from yaml_mouselist import yaml_mouselist
    import warnings; warnings.simplefilter('ignore')

    import matplotlib
    from matplotlib import pyplot as plt
    from event_ind import event_ind
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal

    import h5py

    import seaborn as sns
    sns.set_style("white")

    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    # timewindow (minus and plus time in seconds)
    EVENT_TIMEWINDOW = [3,3]

    # specify track numbers
    track_short = 5
    track_long = 5
    # get indices of desired behavioural event
    events = event_ind(behav_ds,['movement_onset',1,3])

    # create figure to later plot on
    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(111)
    ax1.plot(behav_ds[:,3],lw=0.3)

    for e in events:
        ax1.axvline(e[0].astype('int'),c='r',lw=0.3)

    fig.suptitle('Movement onset detection (full trace)' + fname, wrap=True)
    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
