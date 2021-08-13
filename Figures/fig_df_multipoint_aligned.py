"""
Plot ROI dF traces centered to trial onset, landmark offset, and reward

@author: lukasfischer

"""


def fig_df_multipoint_aligned(h5path, sess, roi, fname, ylims=[-0.5, 3], fformat='png', subfolder=[]):
    import numpy as np
    import h5py
    import sys
    import os
    import yaml
    import warnings; warnings.simplefilter('ignore')
    import matplotlib
    from matplotlib import pyplot as plt
    from event_ind import event_ind
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal

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
    track_short = 3
    track_long = 4

    # Y scale max for heatmap
    YMAX = 3

    ymax_trace = 0

    # create figure to later plot on
    fig = plt.figure(figsize=(12,12))
    ax1 = plt.subplot2grid((4,3),(0,0),colspan=1,rowspan=1)
    ax2 = plt.subplot2grid((4,3),(0,1),colspan=1,rowspan=1)
    ax3 = plt.subplot2grid((4,3),(0,2),colspan=1,rowspan=1)
    ax4 = plt.subplot2grid((4,3),(1,0),colspan=1,rowspan=1)
    ax5 = plt.subplot2grid((4,3),(1,1),colspan=1,rowspan=1)
    ax6 = plt.subplot2grid((4,3),(1,2),colspan=1,rowspan=1)

    ax7 = plt.subplot2grid((4,3),(2,0),colspan=1,rowspan=1)
    ax8 = plt.subplot2grid((4,3),(2,1),colspan=1,rowspan=1)
    ax9 = plt.subplot2grid((4,3),(2,2),colspan=1,rowspan=1)
    ax10 = plt.subplot2grid((4,3),(3,0),colspan=1,rowspan=1)
    ax11 = plt.subplot2grid((4,3),(3,1),colspan=1,rowspan=1)
    ax12 = plt.subplot2grid((4,3),(3,2),colspan=1,rowspan=1)

    ax1.set_title('trial onset, short')
    ax2.set_title('landmark offset, short')
    ax3.set_title('reward, short')
    
    ax7.set_title('trial onset, long')
    ax8.set_title('landmark offset, long')
    ax9.set_title('reward, long')

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
        labelbottom='off', \
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
        left='off', \
        labelleft='off', \
        labelbottom='off', \
        right='off', \
        top='off')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.tick_params( \
        reset='on',    
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        left='off', \
        labelleft='off', \
        labelbottom='off', \
        right='off', \
        top='off')

    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.spines['left'].set_visible(False)
    ax7.spines['bottom'].set_visible(False)
    ax7.tick_params( \
        reset='on',    
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        labelbottom='off', \
        right='off', \
        top='off')
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['left'].set_visible(False)
    ax8.spines['bottom'].set_visible(False)
    ax8.tick_params( \
        reset='on',    
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        left='off', \
        labelleft='off', \
        labelbottom='off', \
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
        left='off', \
        labelleft='off', \
        labelbottom='off', \
        right='off', \
        top='off')

    # SHORT TRIALS
    filt_trials = filter_trials(behav_ds,[],['cut tracknumber', 5])
    filt_trials = filter_trials(behav_ds,[],['tracknumber',track_short], filt_trials)
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

    # PLOT ONSET
    # grab peri-event dF trace for each event
    trial_dF = np.zeros((np.size(events_onset[:,0]),2))
    for i,cur_ind in enumerate(events_onset):
        # determine indices of beginning and end of timewindow
        if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i,0] = 0
        if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i,1] = np.size(behav_ds,0)

    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_onset_short = np.zeros((np.size(events_onset[:,0]),int(t_max)))
    
    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF,0)):
        cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
        cur_sweep_resampled_onset_short[i,:] = signal.resample(cur_sweep, int(t_max), axis=0)
        ax1.plot(cur_sweep_resampled_onset_short[i,:],c='0.65',lw=1)

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
    ax1.plot(np.nanmean(cur_sweep_resampled_onset_short,axis=0),c='k',lw=3) #'#00AAAA'
    ax1.set_xlim([0,t_max])
    ax4.axvline(event_loc,c='0.8', lw=3)
    ax1.axvline(event_loc,c='r')
    
    ymax_trace = np.amax([ymax_trace,np.amax(cur_sweep_resampled_onset_short)])

    # PLOT LANDMARK OFFSET
    # grab peri-event dF trace for each event
    trial_dF = np.zeros((np.size(events_landmark[:,0]),2))
    for i,cur_ind in enumerate(events_landmark):
        # determine indices of beginning and end of timewindow
        if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i,0] = 0
        if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i,1] = np.size(behav_ds,0)

    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_landmark_short = np.zeros((np.size(events_landmark[:,0]),int(t_max)))
    
    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF,0)):
        cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
        cur_sweep_resampled_landmark_short[i,:] = signal.resample(cur_sweep, int(t_max), axis=0)
        ax2.plot(cur_sweep_resampled_landmark_short[i,:],c='0.65',lw=1)

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
    ax2.plot(np.nanmean(cur_sweep_resampled_landmark_short,axis=0),c='k',lw=3) #'#00AAAA'
    ax2.set_xlim([0,t_max])
    ax5.axvline(event_loc,c='0.8', lw=3)
    ax2.axvline(event_loc,c='r')
    
    ymax_trace = np.amax([ymax_trace,np.amax(cur_sweep_resampled_landmark_short)])

    # PLOT REWARD    
    # grab peri-event dF trace for each event
    trial_dF = np.zeros((np.size(events_reward[:,0]),2))
    for i,cur_ind in enumerate(events_reward):
        # determine indices of beginning and end of timewindow
        if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i,0] = 0
        if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i,1] = np.size(behav_ds,0)

    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_reward_short = np.zeros((np.size(events_reward[:,0]),int(t_max)))
    
    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF,0)):
        cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
        cur_sweep_resampled_reward_short[i,:] = signal.resample(cur_sweep, int(t_max), axis=0)
        ax3.plot(cur_sweep_resampled_reward_short[i,:],c='0.65',lw=1)

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
    ax3.plot(np.nanmean(cur_sweep_resampled_reward_short,axis=0),c='k',lw=3) #'#00AAAA'
    ax3.set_xlim([0,t_max])
    ax6.axvline(event_loc,c='0.8', lw=3)
    ax3.axvline(event_loc,c='r')
    
    ymax_trace = np.amax([ymax_trace,np.amax(cur_sweep_resampled_reward_short)])

    ax1.set_ylim([-0.5,ymax_trace])
    ax2.set_ylim([-0.5,ymax_trace])
    ax3.set_ylim([-0.5,ymax_trace])

    
    # TRIAL ONSET LONG TRIALS
    filt_trials = filter_trials(behav_ds,[],['cut tracknumber', 5])
    filt_trials = filter_trials(behav_ds,[],['tracknumber',track_long], filt_trials)
    events_onset = event_ind(behav_ds,['trial_transition',4])
    events_landmark = event_ind(behav_ds,['at_location', 240])
    events_reward = event_ind(behav_ds,['rewards_all',-1])

    pass_t_onset = np.intersect1d(events_onset[:,1],filt_trials)
    pass_t_landmark = np.intersect1d(events_landmark[:,1],filt_trials)
    pass_t_reward = np.intersect1d(events_reward[:,1],filt_trials)

    events_onset = events_onset[np.in1d(events_onset[:,1],pass_t_onset),:]
    events_landmark = events_landmark[np.in1d(events_landmark[:,1],pass_t_landmark),:]
    events_reward = events_reward[np.in1d(events_reward[:,1],pass_t_reward),:]

    # PLOT ONSET
    # grab peri-event dF trace for each event
    trial_dF = np.zeros((np.size(events_onset[:,0]),2))
    for i,cur_ind in enumerate(events_onset):
        # determine indices of beginning and end of timewindow
        if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i,0] = 0
        if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i,1] = np.size(behav_ds,0)

    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_onset_long = np.zeros((np.size(events_onset[:,0]),int(t_max)))
    
    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF,0)):
        cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
        cur_sweep_resampled_onset_long[i,:] = signal.resample(cur_sweep, int(t_max), axis=0)
        ax7.plot(cur_sweep_resampled_onset_long[i,:],c='0.65',lw=1)

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
    ax7.plot(np.nanmean(cur_sweep_resampled_onset_long,axis=0),c='k',lw=3) #'#00AAAA'
    ax7.set_xlim([0,t_max])
    ax10.axvline(event_loc,c='0.8', lw=3)
    ax7.axvline(event_loc,c='r')
    
    ymax_trace = np.amax([ymax_trace,np.amax(cur_sweep_resampled_onset_long)])

    # PLOT LANDMARK OFFSET
    # grab peri-event dF trace for each event
    trial_dF = np.zeros((np.size(events_landmark[:,0]),2))
    for i,cur_ind in enumerate(events_landmark):
        # determine indices of beginning and end of timewindow
        if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i,0] = 0
        if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i,1] = np.size(behav_ds,0)
    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_landmark_long = np.zeros((np.size(events_landmark[:,0]),int(t_max)))
    
    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF,0)):
        cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
        cur_sweep_resampled_landmark_long[i,:] = signal.resample(cur_sweep, int(t_max), axis=0)
        ax8.plot(cur_sweep_resampled_landmark_long[i,:],c='0.65',lw=1)

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
    ax8.plot(np.nanmean(cur_sweep_resampled_landmark_long,axis=0),c='k',lw=3) #'#00AAAA'
    ax8.set_xlim([0,t_max])
    ax11.axvline(event_loc,c='0.8', lw=3)
    ax8.axvline(event_loc,c='r')
    
    ymax_trace = np.amax([ymax_trace,np.amax(cur_sweep_resampled_landmark_long)])

    # PLOT REWARD    
    # grab peri-event dF trace for each event
    trial_dF = np.zeros((np.size(events_reward[:,0]),2))
    for i,cur_ind in enumerate(events_reward):
        # determine indices of beginning and end of timewindow
        if behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
            trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[int(cur_ind[0]),0]-EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i,0] = 0
        if behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
            trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[int(cur_ind[0]),0]+EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i,1] = np.size(behav_ds,0)

    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = np.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_reward_long = np.zeros((np.size(events_reward[:,0]),int(t_max)))
    
    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF,0)):
        cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),roi]
        cur_sweep_resampled_reward_long[i,:] = signal.resample(cur_sweep, int(t_max), axis=0)
        ax9.plot(cur_sweep_resampled_reward_long[i,:],c='0.65',lw=1)

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
    ax9.plot(np.nanmean(cur_sweep_resampled_reward_long,axis=0),c='k',lw=3) #'#00AAAA'
    ax9.set_xlim([0,t_max])
    ax12.axvline(event_loc,c='0.8', lw=3)
    ax9.axvline(event_loc,c='r')
    ymax_trace = np.amax([ymax_trace,np.amax(cur_sweep_resampled_reward_long)])

    sns.heatmap(cur_sweep_resampled_onset_short,cmap='jet',vmin=0,vmax=ymax_trace,yticklabels=events_onset[:,1],xticklabels=False,ax=ax4,cbar=False)
    sns.heatmap(cur_sweep_resampled_landmark_short,cmap='jet',vmin=0,vmax=ymax_trace,yticklabels=False,xticklabels=False,ax=ax5,cbar=False)
    sns.heatmap(cur_sweep_resampled_reward_short,cmap='jet',vmin=0,vmax=ymax_trace,yticklabels=False,xticklabels=False,ax=ax6,cbar=False)
    sns.heatmap(cur_sweep_resampled_onset_long,cmap='jet',vmin=0,vmax=ymax_trace,yticklabels=events_onset[:,1],xticklabels=False,ax=ax10,cbar=False)
    sns.heatmap(cur_sweep_resampled_landmark_long,cmap='jet',vmin=0,vmax=ymax_trace,yticklabels=False,xticklabels=False,ax=ax11,cbar=False)
    sns.heatmap(cur_sweep_resampled_reward_long,cmap='jet',vmin=0,vmax=ymax_trace,yticklabels=False,xticklabels=False,ax=ax12,cbar=False)

    ax1.set_ylim([-0.5,ymax_trace])
    ax2.set_ylim([-0.5,ymax_trace])
    ax3.set_ylim([-0.5,ymax_trace])

    ax7.set_ylim([-0.5,ymax_trace])
    ax8.set_ylim([-0.5,ymax_trace])
    ax9.set_ylim([-0.5,ymax_trace])

    fig.suptitle(fname, wrap=True)

    plt.tight_layout()

    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat)








