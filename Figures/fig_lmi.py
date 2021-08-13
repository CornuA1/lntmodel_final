"""
calculate locomotion index for ROIs and make histogram figure of lmi distribution

for now we are assuming a dark session (i.e. no filtering by tracktype)

@author: lukasfischer

"""

def fig_LMI(h5path, sess, roi, fname, ylims=[-0.5,3], fformat='png', subfolder=[], trialpattern=[]):
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

    number_of_shuffles = 100
    shuffledOnsetActiveIndices = np.zeros(number_of_shuffles)

    for shuffle in range(number_of_shuffles):

        # get indices of desired behavioural event
        events = event_ind(behav_ds,['movement_onset',1,3])

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
