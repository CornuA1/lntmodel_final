    """
Plot trace of an individual ROI centered around the the end of the landmark

@author: lukasfischer

"""

import numpy as np
import h5py
import sys
import yaml
import os
import warnings; warnings.simplefilter('ignore')
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
from filter_trials import filter_trials
from scipy import stats
from scipy import signal

def fig_landmark_centered(h5path, sess, roi, fname, ylims=[], fformat='png', subfolder=[]):
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    # timewindow (minus and plus time in seconds)
    EVENT_TIMEWINDOW = [3,3]

    # specify track numbers
    track_short = 3
    track_long = 4

    # ylims
    min_y = -0.3
    max_y = 0.0
    if not ylims == []:
        min_y = ylims[0]
        max_y = ylims[1]

    # create figure to later plot on
    fig = plt.figure(figsize=(12,12))
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(322)
    ax3 = plt.subplot(323)
    ax4 = plt.subplot(324)
    ax5 = plt.subplot(325)
    ax6 = plt.subplot(326)

    ax1.set_title('dF/F vs time SHORT track')
    ax2.set_title('dF/F vs time LONG track')
    ax3.set_title('dF/F vs time SHORT track - heatmap')
    ax4.set_title('dF/F vs time LONG track - heatmap')
    ax5.set_title('running speed (cm/s) vs time SHORT track')
    ax6.set_title('running speed (cm/s) vs time LONG track')

    # get indices of desired behavioural event
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])

    events_short = event_ind(behav_ds,['at_location', 200], trials_short)
    events_long = event_ind(behav_ds,['at_location', 200], trials_long)

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
    t_max = np.amax(trial_dF_short[:,1] - trial_dF_short[:,0])
    cur_sweep_resampled_short = np.zeros((np.size(events_short[:,0]),int(t_max)))
    cur_sweep_speed_resampled_short = np.zeros((np.size(events_short[:,0]),int(t_max)))

    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF_short,0)):
        # grab dF trace
        cur_sweep = dF_ds[int(trial_dF_short[i,0]):int(trial_dF_short[i,1]),roi]
        cur_sweep_resampled_short[i,:] = signal.resample(cur_sweep, t_max, axis=0)
        if np.amax(cur_sweep_resampled_short[i,:]) > max_y:
            max_y = np.amax(cur_sweep_resampled_short[i,:])
        # grab running speed trace
        cur_sweep_speed = behav_ds[int(trial_dF_short[i,0]):int(trial_dF_short[i,1]),3]
        cur_sweep_speed_resampled_short[i,:] = signal.resample(cur_sweep_speed, t_max, axis=0)
        # plot each trace respectively
        ax1.plot(cur_sweep_resampled_short[i,:],c='0.65',lw=1)
        ax5.plot(cur_sweep_speed_resampled_short[i,:],c='0.65',lw=1)
    event_avg_short = np.mean(cur_sweep_resampled_short,axis=0)
    speed_avg_short = np.mean(cur_sweep_speed_resampled_short,axis=0)
    sem_dF_short = stats.sem(cur_sweep_resampled_short,0,nan_policy='omit')
    ax1.plot(event_avg_short, c = sns.xkcd_rgb["windows blue"], lw=4)
    ax5.plot(speed_avg_short, c = 'g', lw=4)

    # grab peri-event dF trace for each event long TRIALS
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
    t_max = np.amax(trial_dF_long[:,1] - trial_dF_long[:,0])
    cur_sweep_resampled_long = np.zeros((np.size(events_long[:,0]),int(t_max)))
    cur_sweep_speed_resampled_long = np.zeros((np.size(events_short[:,0]),int(t_max)))

    # resample every sweep to match the longest sweep
    for i in range(np.size(trial_dF_long,0)):
        # grab dF trace
        cur_sweep = dF_ds[int(trial_dF_long[i,0]):int(trial_dF_long[i,1]),roi]
        cur_sweep_resampled_long[i,:] = signal.resample(cur_sweep, t_max, axis=0)
        if np.amax(cur_sweep_resampled_long[i,:]) > max_y:
            max_y = np.amax(cur_sweep_resampled_long[i,:])
        # grab running speed trace
        cur_sweep_speed = behav_ds[int(trial_dF_long[i,0]):int(trial_dF_long[i,1]),3]
        cur_sweep_speed_resampled_long[i,:] = signal.resample(cur_sweep_speed, t_max, axis=0)
        # plot traces
        ax2.plot(cur_sweep_resampled_long[i,:],c='0.65',lw=1)
        ax6.plot(cur_sweep_speed_resampled_long[i,:],c='0.65',lw=1)
    event_avg_long = np.mean(cur_sweep_resampled_long,axis=0)
    speed_avg_long = np.mean(cur_sweep_speed_resampled_long,axis=0)
    sem_dF_long = stats.sem(cur_sweep_resampled_long,0,nan_policy='omit')
    ax2.plot(event_avg_long, c = sns.xkcd_rgb["dusty purple"], lw=4)
    ax6.plot(speed_avg_long, c = 'g', lw=4)

    ax1.set_ylim([min_y,max_y])
    ax2.set_ylim([min_y,max_y])
    ax1.set_xlim([0,t_max])
    ax2.set_xlim([0,t_max])

    # calc where to draw the line for the event
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]

    if fformat is 'png':
        # sns.heatmap(cur_sweep_resampled,cmap='jet',vmin=ylims[0],vmax=ylims[1],yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax2)
        # sns.heatmap(cur_sweep_resampled_speed,cmap='jet',vmin=0,yticklabels=events[:,1].astype('int'),xticklabels=False,ax=ax4)
        sns.heatmap(cur_sweep_resampled_short,cmap='viridis',vmin=0,vmax=max_y,yticklabels=events_short[:,1].astype('int'),xticklabels=False,ax=ax3)
        sns.heatmap(cur_sweep_resampled_long,cmap='viridis',vmin=0,vmax=max_y,yticklabels=events_long[:,1].astype('int'),xticklabels=False,ax=ax4)

    ax1.axvline(event_loc,c='r')
    ax2.axvline(event_loc,c='r')
    ax3.axvline(event_loc,c='r')
    ax4.axvline(event_loc,c='r')
    ax5.axvline(event_loc,c='r')
    ax6.axvline(event_loc,c='r')

    fig.suptitle('landmark onset centered' + fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'

    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # for r in range(10):
    #     print(r)
    #     fig_landmark_centered(h5path, SESSION, r, MOUSE+'_'+SESSION+'_'+str(r), [0.0,3], fformat, MOUSE+'_'+SESSION+'_lmoff')
    fig_landmark_centered(h5path, SESSION, 6, MOUSE+'_'+SESSION+'_'+str(6), [], fformat, MOUSE+'_'+SESSION+'_lmon')
