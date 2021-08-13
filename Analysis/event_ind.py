"""
Return list of indices at which a given behavioural event (e.g. reward
dispensed) occured.

Parameters
-------
raw_behav : ndarray
            raw behaviour dataset

filterprops : tuple
            provide information on which type of event should be detected and
            required paramaters. Since different events types may need
            different parameters, this function argument is intentionally
            flexible

pre_select : list
            list of trials for which events are returned

Outputs
-------
event_indices : ndarray
            indices of raw_behav, and trial numbers at indices for the given event in each trial

"""


import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def movement_onset( raw_behav, speed_threshold, gap_tolerance ):
    """
    return indices of movement onset. Use lowpass-filtered signal
    to avoid single spikes in runnign speed triggering an event

    speed_threshold (cm/sec): running speed that has to be exceeded to
                     be detected as movement

    gap_tolerance (sec): dips in speed below speed_threshold for this period are
                   ignored

    """

    # filter requirements.
    order = 6
    fs = int(np.size(raw_behav,0)/raw_behav[-1,0])       # sample rate, Hz
    cutoff = 4 # desired cutoff frequency of the filter, Hz

    speed_filtered = butter_lowpass_filter(raw_behav[:,3], cutoff, fs, order)

    # get indeces above speed threshold
    speed_high_idx = np.where(speed_filtered > speed_threshold)[0]

    # use diff to find gaps between episodes of high speed
    idx_diff = np.diff(speed_high_idx)
    idx_diff = np.insert(idx_diff,0,0)

    # convert gap tolerance from cm to number of frames
    gap_tolerance_frames = int(gap_tolerance/raw_behav[0,2])

    # find indeces where speed exceeds threshold
    onset_idx = speed_high_idx[np.where(idx_diff > gap_tolerance_frames)[0]]

    return onset_idx

def first_lick( raw_behav ):
    """ return the index of the first licks in each trial """
    # get indeces of every lick and a list of trial numbers
    triallist = np.unique(raw_behav[:,6])
    # get indices of all licks. We starts with index 0 because the very first datapoint often indicates there was a lick but thats an artifcat from the recording system
    all_licks = np.where(raw_behav[:,7]>0)[0]
    all_licks = all_licks[1:]

    firstlick_idx = np.full((np.size(triallist),), np.nan)
    # loop through each trial and get their respective indeces, store index
    # of the first lick in every trial
    for i,t in enumerate(triallist):
        cur_trial_idx = np.where(raw_behav[:,6]==t)[0]
        try:
            firstlick_idx[i] = np.intersect1d(all_licks,cur_trial_idx)[0]
        except IndexError:
            pass

    firstlick_idx = firstlick_idx[~np.isnan(firstlick_idx)]
    return firstlick_idx.astype(int)

def track_transition( raw_behav, tracktype_p, tracktype_s ):
    """
    return indices of track transitions between specific type (which are often
    equivalent to certain trialtypes)

    trialtype_p : int
                  tracktype of preceeding trial

    trialtype_s : int
                  tracktype of succeeding trial

    """

    # get indeces of trial transition points
    trial_diff = np.diff(raw_behav[:,6])
    # insert 0 at beginning to re-align diff vector with original vector,
    # as that has n-1 number elements
    trial_diff = np.insert(trial_diff,0,0)
    # trial transition (tt) indeces
    tt_idx = np.where(trial_diff>0)[0]
    tt_tracktypes = np.zeros((np.size(tt_idx),3))
    # get tracktypes at trial transition points
    tt_tracktypes[:,0] = raw_behav[tt_idx-1,4]
    tt_tracktypes[:,1] = raw_behav[tt_idx,4]
    tt_tracktypes[:,2] = tt_idx
    # find trial transitions that match user input
    tt_return_p = np.where(tt_tracktypes[:,0]==tracktype_p)[0]
    tt_return_s = np.where(tt_tracktypes[:,1]==tracktype_s)[0]
    tt_return = np.intersect1d(tt_return_p,tt_return_s)

    return tt_tracktypes[tt_return,2].astype(int)

def trial_transition( raw_behav ):
    """ return indices of trial transitions """
    # find indeces of trial transitions, insert 0 to get first trial as well
    return np.where(np.diff(np.insert(raw_behav[:,6],0,0)) > 0)[0]

def event_rewards( raw_behav, filterprops ):
    """ return indices when the animal received a reward. Disregard default rewards. """
    # calculate the first order difference in reward-column
    rew_diff = np.diff(raw_behav[:,5])
    # insert 0 at beginning to re-align diff vector with original vector,
    # as that has n-1 number elements
    rew_diff = np.insert(rew_diff,0,0)
    # find indices where valve had just opened (since the column is set to 1
    # whenever the valve has been opened to dispense a reward in response to
    # licking)
    e_ind = np.where(rew_diff == 1)[0]
    # filter out reward events that happened not on the desired track
    if filterprops[1] != -1:
        for i,ind in enumerate(e_ind):
            if raw_behav[ind,4] != filterprops[1]:
                e_ind[i] = -1
        e_ind = e_ind[np.where(e_ind != -1)]

    return e_ind

def rewards_unsuccessful( raw_behav, filterprops ):
    """ return indices when the animal received a default reward. """
    # calculate the first order difference in reward-column
    rew_diff = np.diff(raw_behav[:,5])
    # insert 0 at beginning to re-align diff vector with original vector,
    # as that has n-1 number elements
    rew_diff = np.insert(rew_diff,0,0)
    # find indices where valve had just opened (since the column is set to 1
    # whenever the valve has been opened to dispense a reward in response to
    # licking)
    e_ind = np.where(rew_diff == 2)[0]
    # filter out reward events that happened not on the desired track
    if filterprops[1] != -1:
        for i,ind in enumerate(e_ind):
            if raw_behav[ind,4] != filterprops[1]:
                e_ind[i] = -1
        e_ind = e_ind[np.where(e_ind != -1)]

    return e_ind

def rewards_all( raw_behav, filterprops ):
    """ return indices when the animal received any reward. """
    # calculate the first order difference in reward-column
    rew_diff = np.diff(raw_behav[:,5])
    # insert 0 at beginning to re-align diff vector with original vector,
    # as that has n-1 number elements
    rew_diff = np.insert(rew_diff,0,0)
    # find indices where valve had just opened (since the column is set to 1
    # whenever the valve has been opened to dispense a reward in response to
    # licking)
    e_ind = np.where(rew_diff > 0)[0]
    # filter out reward events that happened not on the desired track
    if filterprops[1] != -1:
        for i,ind in enumerate(e_ind):
            if raw_behav[ind,4] != filterprops[1]:
                e_ind[i] = -1
        e_ind = e_ind[np.where(e_ind != -1)]

    return e_ind

def at_location( raw_behav, filterprops ):
    """ provide first index of each trial when animal reaches a given location """
    triallist = np.unique(raw_behav[:,6])
    location_reached = np.where(raw_behav[:,1] > filterprops[1])[0]
    loc_idx = np.zeros((np.size(triallist),))

    for i,t in enumerate(triallist):
        cur_trial_idx = np.where(raw_behav[:,6]==t)[0]
        try:
            loc_idx[i] = np.intersect1d(location_reached,cur_trial_idx)[0]
            # if raw_behav[int(loc_idx[i]),1] > 320:
            #     loc_idx[i] = -1
        except IndexError:
            loc_idx[i] = -1

    # delete trials where the location wasn't passed
    loc_idx = np.delete(loc_idx, np.where(loc_idx==-1)[0])

    return loc_idx.astype(int)

def at_time( raw_behav, filterprops ):
    """ provide first index of each trial when a certain time has elapsed since the start of the trial """
    triallist = np.unique(raw_behav[:,6])
    # time_reached = np.where(raw_behav[:,0] > filterprops[1])[0]
    trial_start_idx = np.argwhere(np.diff(raw_behav[:,6])>0) + 1
    trial_start_idx = np.insert(trial_start_idx,0,0)
    loc_idx = np.zeros((np.size(triallist),))

    for i,t in enumerate(triallist):
        cur_trial_idx = np.where(raw_behav[:,6]==t)[0]
        cur_triaL_start_t = raw_behav[trial_start_idx[i],0]
        time_reached_idx = np.where(raw_behav[cur_trial_idx,0]-cur_triaL_start_t > filterprops[1])[0]
        try:
            loc_idx[i] = np.intersect1d(time_reached_idx+trial_start_idx[i],cur_trial_idx)[0]
            loc_idx[i] = loc_idx[i]
            # if raw_behav[int(loc_idx[i]),1] > 320:
            #     loc_idx[i] = -1
        except IndexError:
            loc_idx[i] = -1

    # delete trials where the location wasn't passed
    loc_idx = np.delete(loc_idx, np.where(loc_idx==-1)[0])
    return loc_idx.astype(int)

def filter_pre_select(events_and_trials, pre_select):
    """ return only event indices that occur on trials listed in pre-select """
    return events_and_trials[np.in1d(events_and_trials[:,1],pre_select),:]

def event_ind( raw_behav, filterprops=[-1], pre_select=[]):
    """ call event filter based on filterprops parameters and return results """
    # call desired event filter function
    if filterprops[0] == 'reward_successful':
        event_indices = event_rewards(raw_behav, filterprops)
    elif filterprops[0] == 'reward_unsuccessful':
        event_indices = rewards_unsuccessful(raw_behav, filterprops)
    elif filterprops[0] == 'rewards_all':
        event_indices = rewards_all(raw_behav, filterprops)
    elif filterprops[0] == 'trial_transition':
        event_indices = trial_transition(raw_behav)
    elif filterprops[0] == 'movement_onset':
        event_indices = movement_onset(raw_behav, filterprops[1], filterprops[2])
    elif filterprops[0] == 'track_transition':
        event_indices = track_transition(raw_behav, filterprops[1], filterprops[2])
    elif filterprops[0] == 'first_licks':
        event_indices = first_lick( raw_behav )
    elif filterprops[0] == 'at_location':
        event_indices = at_location( raw_behav, filterprops )
    elif filterprops[0] == 'at_time':
        event_indices = at_time( raw_behav, filterprops )
    else:
        raise ValueError('Event type not recognised.')

    events_and_trials =  np.transpose(np.vstack((event_indices,raw_behav[event_indices,6])))

    if pre_select != []:
        return filter_pre_select(events_and_trials, pre_select)
    else:
        return events_and_trials
