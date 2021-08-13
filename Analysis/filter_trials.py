"""
Remove trials that don't match the provided criteria

Parameters
-------
raw_behav : ndarray
            raw behaviour dataset

filterprops : tuple
            provide information on which type of filter should be used and
            according paramaters. Since different filter types may need
            different parameters, this function argument is intentionally
            flexible

pre_select : ndarray
            tiralnumbers that have been pre-filtered. Output will be a
            subset of this input.

Outputs
-------
filt_trials : ndarray
            trialnumbers that pass the filtering criteria



"""

import numpy as np
from rewards import rewards
from licks import licks
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

def exclude_earlylick_trials( raw_behav, filterprops ):
    """
    return only trialnumbers of trials where the animal didn't lick before a certain pointself.
    If filterprops[1] contains a single value, only trials where there are no licks before that point
    are returned.

    If filterprops[2] is a tuple, it defines a window within which no licks are allowed


    """

    # get all trials of a session (including black box)
    trial_list = np.unique(raw_behav[:,6])
    trials_pass = []

    if len(filterprops[1]) > 1:
        window_start = filterprops[1][0]
        window_end = filterprops[1][1]
    else:
        window_start = 0
        window_end = filterprops[1][0]

    # go through every trial, extract lines where animal has licked check if any of the licks happened outside the defined window
    for i,t in enumerate(trial_list):
        cur_trial = raw_behav[raw_behav[:,6]==t,:]
        cur_trial_licks = cur_trial[cur_trial[:,7]>0,:]

        if np.size(cur_trial_licks) > 0:
            cur_trial_licks = cur_trial_licks[cur_trial_licks[:,1] > window_start,:]
            cur_trial_licks = cur_trial_licks[cur_trial_licks[:,1] < window_end,:]

        if np.size(cur_trial_licks) == 0:
            trials_pass.append(t)
    return trials_pass

def opto_mask_on_stim_off( raw_behav ):
    """
    return trial numbers in which the mask light was on but the sitm light was off.

    """

    # get all trials of a session (including black box)
    trial_list = np.unique(raw_behav[:,6])
    trials_pass = []

    # grab every trial and check if opto mask light was on
    for i,t in enumerate(trial_list):
        cur_trial = raw_behav[raw_behav[:,6]==t,:]
        # is opto mask light set to 1 on first line of trial
        if cur_trial[0,9] > 0 and cur_trial[0,10] == 0:
            trials_pass.append(t)

    return trials_pass


def opto_stim_on( raw_behav, filterprops ):
    """
    return trial numbers in which the opto stim light was on. Optionally, if filterprops[1] is set, it will define the intensity threshold

    """

    # get all trials of a session (including black box)
    trial_list = np.unique(raw_behav[:,6])
    trials_pass = []

    if len(filterprops) > 1:
        stim_threshold = filterprops[1]
    else:
        stim_threshold = 0

    # grab every trial and check if opto mask light was on
    for i,t in enumerate(trial_list):
        cur_trial = raw_behav[raw_behav[:,6]==t,:]
        # is opto mask light set to 1 on first line of trial
        if cur_trial[0,10] > stim_threshold:
            trials_pass.append(t)

    return trials_pass

def filter_opto_mask_light_on( raw_behav ):
    """
    return trial numbers in which the opto mask light was ON

    """

    # get all trials of a session (including black box)
    trial_list = np.unique(raw_behav[:,6])
    trials_pass = []

    # grab every trial and check if opto mask light was on
    for i,t in enumerate(trial_list):
        cur_trial = raw_behav[raw_behav[:,6]==t,:]
        # is opto mask light set to 1 on first line of trial
        if cur_trial[0,9] == 1:
            trials_pass.append(t)

    return trials_pass

def filter_opto_mask_light_off( raw_behav ):
    """
    return trial numbers in which the opto mask light was OFF

    """

    # get all trials of a session (including black box)
    trial_list = np.unique(raw_behav[:,6])
    trials_pass = []

    # grab every trial and check if opto mask light was on
    for i,t in enumerate(trial_list):
        cur_trial = raw_behav[raw_behav[:,6]==t,:]
        # is opto mask light set to 1 on first line of trial
        if cur_trial[0,9] == 0:
            trials_pass.append(t)

    return trials_pass

def animal_notrunning( raw_behav, filterprops ):
    """
    return trial numbers for trials in which the animal exceeded a given
    running speed (provided in filterprops)

    filterprops[1] : speed threshold (cm/sec)

    filterprops[2] : max. time above threshold (sec) allowed to be considered
                     not running

    """
    # get all trials of a session (including black box)
    trial_list = np.unique(raw_behav[:,6])
    trials_pass = []

    # filter requirements.
    order = 6
    fs = int(np.size(raw_behav,0)/raw_behav[-1,0])       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz

    speed_filtered = butter_lowpass_filter(raw_behav[:,8], cutoff, fs, order)

    # run through every trial and test if running speed was within criteria
    for i,t in enumerate(trial_list):
        cur_trial = raw_behav[raw_behav[:,6]==t,:]
        trial_speed = speed_filtered[raw_behav[:,6]==t]
        cur_trial_latency = np.mean(cur_trial[:,2])
        # calculate how many samples speed is allowed to be above threshold (calc for each trial as it might fluctuate)
        samples_thresh = filterprops[2] / cur_trial_latency
        # get indices where animal was below threshold
        thresh_idx = np.where(trial_speed > filterprops[1])[0]
        # check where multiple indices where not below threshold
        if np.size(thresh_idx) < samples_thresh:
            trials_pass.append(t)
    return trials_pass

def animal_running( raw_behav, filterprops ):
    """
    return trial numbers for trials in which the animal exceeded a given
    running speed (provided in filterprops)

    filterprops[1] : speed threshold (cm/sec)

    filterprops[2] : location around which the animal has to exceed a certain average speed

    filterprops[3] : spatial window (cm) used to calculate average speed around receptive field center

    filterprops[4] : bool - indicate whether animal has to be above speed threshold (True) or below (False)

    filterprops[5] : bool - indicate whether location is to be trial onset-aligned (True) landmark-aligned (False)

    """

    # extract values for readability
    space_win_start = filterprops[2] - filterprops[3][0]
    space_win_post = filterprops[2] + filterprops[3][1]
    trial_onset_aligned = filterprops[5]

    # get all trials of a session (including black box)
    trial_list = np.unique(raw_behav[:,6])
    trials_pass = []

    # filter requirements.
    order = 6
    fs = int(np.size(raw_behav,0)/raw_behav[-1,0])       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz

    speed_filtered = butter_lowpass_filter(raw_behav[:,8], cutoff, fs, order)

    # run through every trial and test if running speed was within criteria
    if trial_onset_aligned:
        for i,t in enumerate(trial_list):
            # get indeces of current trial
            cur_trial_idx = np.where(raw_behav[:,6]==t)[0]
            # subtract start location from each trial
            cur_trial_loc = raw_behav[cur_trial_idx,1] - raw_behav[cur_trial_idx,1][0]
            # get indeces of location within window
            cur_trial_win_idx = np.where((cur_trial_loc > space_win_start) &
                                         (cur_trial_loc < space_win_post))[0]

            # get the indices of the datapoints within the window in the full dataset
            win_idx = cur_trial_idx[cur_trial_win_idx]
            # calc mean speed within window and append trial number of list if it matches criteria
            if (np.nanmean(speed_filtered[win_idx]) > filterprops[1]) == filterprops[4]:
                trials_pass.append(t)
    else:
        for i,t in enumerate(trial_list):
            # get indeces of elements within the specified window
            cur_trial_win_idx = np.where((raw_behav[:,1] > space_win_start) &
                                         (raw_behav[:,1] < space_win_post) &
                                         (raw_behav[:,6] == t))[0]
            # calc mean speed within window and append trial number of list if it matches criteria
            if (np.nanmean(speed_filtered[cur_trial_win_idx]) > filterprops[1]) == filterprops[4]:
                trials_pass.append(t)


    return trials_pass

    #
    # cur_trial = raw_behav[raw_behav[:,6]==t,:]
    # cur_trial_loc_idx = np.where(raw_behav[:,6]==t)[0]
    # cur_trial_loc = raw_behav[raw_behav[:,6]==t,1]
    #
    # # get indeces for the first <REWARD_TIME> sec after a reward
    # cur_trial_rew_loc_idx = np.where((raw_behav[:,0] > raw_behav[raw_behav[:,6]==t,0][-1]) & (raw_behav[:,0] < (raw_behav[raw_behav[:,6]==t,0][-1]+post_t)))[0]
    # cur_trial_rew_loc = raw_behav[cur_trial_rew_loc_idx,1]
    # cur_trial_loc_idx = np.append(cur_trial_loc_idx,cur_trial_rew_loc_idx)
    #
    # # get indeces for <PRE_TRIAL_TIME> sec prior to trial onset
    # cur_trial_pretrial_loc_idx = np.where((raw_behav[:,0] < raw_behav[raw_behav[:,6]==t,0][0]) & (raw_behav[:,0] > (raw_behav[raw_behav[:,6]==t,0][0]-post_t)))[0]
    # cur_trial_pretrial_loc = raw_behav[cur_trial_pretrial_loc_idx,1]
    # cur_trial_loc_idx = np.insert(cur_trial_loc_idx,0,cur_trial_pretrial_loc_idx)
    #
    # # subtract starting location from location samples to align all to trial start
    # if align_point == 'trialonset':
    #     cur_trial_rew_loc = cur_trial_rew_loc - cur_trial_loc[0]
    #     cur_trial_loc = cur_trial_loc - cur_trial_loc[0]
    #
    # if np.size(cur_trial_pretrial_loc) > 0:
    #     if align_point == 'landmark':
    #         cur_trial_pretrial_loc = cur_trial_pretrial_loc - cur_trial_pretrial_loc[-1] + cur_trial_loc[0]
    #     elif align_point == 'trialonset':
    #         cur_trial_pretrial_loc = cur_trial_pretrial_loc - cur_trial_pretrial_loc[-1]

    # trial_speed = speed_filtered[cur_trial_loc_idx]
    #
    # cur_trial_latency = np.mean(cur_trial[:,2])

    # calculate how many samples speed is allowed to be above threshold (calc for each trial as it might fluctuate)
    # samples_thresh = filterprops[2] / cur_trial_latency
    # get indices where animal was below threshold
    # thresh_idx = np.where(trial_speed > filterprops[1])[0]
    # check where multiple indices where not below threshold
    # if np.size(thresh_idx) > samples_thresh:
    #     trials_pass.append(t)

def bool_pattern( raw_behav, filterprops ):
    """
    return trial numbers for a pattern described by a boolean
    vector. E.g. [False, False, True] will return every third trial. The
    pattern is applied repeatedly for all trials in the provided dataset.

    """
    pattern = filterprops[1]
    trial_list = np.unique(raw_behav[:,6])

    # repeat pattern to fit input array
    pat_repeat = int(len(trial_list)/len(pattern))
    pat_repeat_frac = len(trial_list)%len(pattern)
    pat_apply = np.tile(pattern, pat_repeat)
    pat_apply = np.append(pat_apply,pattern[0:pat_repeat_frac])

    # return trial numbers that match pattern
    return trial_list[pat_apply]

def first_lick_loc( raw_behav, filterprops ):
    """
    return trials in which the first lick has happened within a certain location
    defined in filterprops

    filterprops[1] : int
            lower boundary (cm) for first lick location

    filterprops[2] : int
            upper boundary (cm) for first lick location

    """
    pass

def trialnr_range( raw_behav, filterprops ):
    """
    return trial-numbers withing a given range

    filterprops[1] : min trial nr (including provided number)

    filterprops[2] : max trial nr (including provided number).
                     If no max trial nr is provided, min trial nr to end of session is applied

    """
    pass_trials = np.unique(raw_behav[:,6])
    if len(filterprops) < 3:
        filterprops.append(np.amax(pass_trials))

    pass_trials = pass_trials[pass_trials >= filterprops[1]]
    pass_trials = pass_trials[pass_trials <= filterprops[2]]

    return pass_trials

def filter_tracknumber( raw_behav, trialtype ):
    """ return trials on a specific track (often equivalent with trialtype) """
    return np.unique(raw_behav[raw_behav[:,4]==trialtype,6])

def cut_tracknumber( raw_behav, trialtype ):
    """ return trials on a specific track (often equivalent with trialtype) """
    return np.unique(raw_behav[raw_behav[:,4] != trialtype,6])

def filter_trial_successful(raw_behav):
    """ return trials where animals triggered a reward themselves """
    rew_rows = raw_behav[raw_behav[:,5]==1,:]
    return np.unique(rew_rows[:,6])

def filter_trial_unsuccessful(raw_behav):
    """ return trials where animals triggered a reward themselves """
    rew_rows = raw_behav[raw_behav[:,5]==2,:]
    return np.unique(rew_rows[:,6])

def filter_maxtotaltime(raw_behav, filterprops):
    """ filter trials based on their total time (from reset point to reset point) """
    pass_trials = np.unique(raw_behav[:,6])
    # loop through every trial
    for cur_trial_nr in np.unique(raw_behav[:,6]):
        # pull out all rows corresponding to the current trial number
        cur_trial = raw_behav[np.where( raw_behav[:,6] == cur_trial_nr )]
        # flag trialnumbers which are longer than desired
        if cur_trial[-1,0] - cur_trial[0,0] > filterprops[1]:
            pass_trials[np.where(pass_trials == cur_trial_nr)] = -1

    # remove trial numbers that didn't pass criterion and return list
    return pass_trials[np.where(pass_trials != -1)]

def filter_maxrewardtime(raw_behav, filterprops):
    """ filter trials based on the time between start of trial and reward """
    rew = rewards( raw_behav )
    return rew[rew[:,0] < filterprops[1],3]

def filter_roi_active(raw_behav, dF_resampled, df_stdev, filterprops ):
    """ return trials in which the dF trace exceeded a certain number of STDEV (defined in filterprops) """
    pass_trials = np.unique(raw_behav[:,6])
    for i in np.unique(raw_behav[:,6]):
        raw_trial_dF = dF_resampled[raw_behav[:,6]==i,:]
        # detect trials in which max dF/F is
        if np.amax(raw_trial_dF[:,filterprops[1]]) < df_stdev*filterprops[2]:
            pass_trials[np.where(pass_trials == i)] = -1
    # remove trial numbers that didn't pass criterion and return list
    return pass_trials[np.where(pass_trials != -1)]

def filter_roi_active_abs(raw_behav, dF_resampled, df_stdev, filterprops ):
    """ return trials in which dF exceeds and absolute threshold """
    pass_trials = np.unique(raw_behav[:,6])
    for i in np.unique(raw_behav[:,6]):
        raw_trial_dF = dF_resampled[raw_behav[:,6]==i,:]
        # detect trials in which max dF/F is
        if np.amax(raw_trial_dF[:,filterprops[1]]) < filterprops[2]:
            pass_trials[np.where(pass_trials == i)] = -1
    # remove trial numbers that didn't pass criterion and return list
    return pass_trials[np.where(pass_trials != -1)]

def filter_running_speed(raw_behav, filterprops):
    """ return trials where animal is at a specific speed. The intended use for this function is to filter trials in the openloop condition at particular speeds """
    return np.unique(raw_behav[raw_behav[:,3] == filterprops[1],6])

def first_lick_distance(raw_behav, filterprops):
    """
    return trials that are within closest/furthest x percentile in terms of distance to the reward zone onset

    filterpros[1] : percentile threshold

    filterprops[2] : sign - '<' for first lick has been closer than percentile, '>' for opposite

    filterprops[3] : track number (i.e. trial type) to carry out analysis

    """

    behav_licks = raw_behav[np.in1d(raw_behav[:, 4], [filterprops[3]]), :]
    reward_ds = rewards(behav_licks)
    licks_ds,_ = licks(behav_licks, reward_ds)
    licks_ds = np.array(licks_ds)

    trials = np.unique(behav_licks[:,6])

    pass_percentile = filterprops[1]

    first_lick = np.empty((0,4))
    first_lick_trials = np.empty((0))
    reward_ds[:,3] = reward_ds[:,3] - 1
    default_rewards = np.empty((0,4))
    default_rewards_trials = np.empty((0))
    for r in trials:
        if licks_ds.size > 0:
            licks_all = licks_ds[licks_ds[:,2]==r,:]
            licks_all = licks_all[licks_all[:,1]>150,:]
            # if r == 79:
            #     ipdb.set_trace()
            if licks_all.size == 0:
                rew_lick = reward_ds[reward_ds[:,3]==r,:]
                if rew_lick.size > 0:
                    if rew_lick[0,5] == 1:
                        licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                        if trial_type == 'short':
                            licks_all[0,1] = licks_all[0,1] - 320
                        elif trial_type == 'long':
                            licks_all[0,1] = licks_all[0,1] - 380
                        first_lick = np.vstack((first_lick, licks_all[0,:].T))
                        first_lick_trials = np.append(first_lick_trials, r)
                    if rew_lick[0,5] == 2:
                        default_rewards = np.vstack((default_rewards, np.asarray([[rew_lick[0,4], 18, rew_lick[0,3], rew_lick[0,2]]])[0,:].T))
                        default_rewards_trials = np.append(default_rewards_trials, r)
            else:
                if licks_all[0,3] == 3:
                    try:
                        licks_all = licks_all[licks_all[:,1]<380,:]
                        licks_all[0,1] = licks_all[0,1] - 320
                    except:
                        ipdb.set_trace()
                        pass
                elif licks_all[0,3] == 4:
                    licks_all = licks_all[licks_all[:,1]<440,:]
                    licks_all[0,1] = licks_all[0,1] - 380
                first_lick = np.vstack((first_lick, licks_all[0,:].T))
                first_lick_trials = np.append(first_lick_trials, r)


    if filterprops[2] is '<':
        fl_crit = first_lick[np.abs(first_lick[:,1]) < np.percentile(np.abs(first_lick[:,1]),pass_percentile)]
    elif filterprops[2] is '>':
        fl_crit = first_lick[np.abs(first_lick[:,1]) > np.percentile(np.abs(first_lick[:,1]),pass_percentile)]

    return fl_crit[:,2]

def filter_trials( raw_behav, dF_resampled=[], filterprops=[], pre_select=-1 ):
    """ call filterfunction as specified in filterprops """

    if dF_resampled != []:
        dF_stdev = np.std(dF_resampled[2:,filterprops[1]])
        dF_resampled = select_trials_dF( raw_behav, dF_resampled, mode )

    # call desired filter function
    if filterprops[0] == 'maxtotaltime':
        filt_trials = filter_maxtotaltime(raw_behav, filterprops)
    elif filterprops[0] == 'maxrewardtime':
        filt_trials = filter_maxrewardtime(raw_behav, filterprops)
    elif filterprops[0] == 'roi_active':
        filt_trials = filter_roi_active(raw_behav, dF_resampled, dF_stdev, filterprops)
    elif filterprops[0] == 'roi_active_abs':
        filt_trials = filter_roi_active_abs(raw_behav, dF_resampled, dF_stdev, filterprops)
    elif filterprops[0] == 'trial_successful':
        filt_trials = filter_trial_successful(raw_behav)
    elif filterprops[0] == 'trial_unsuccessful':
        filt_trials = filter_trial_unsuccessful(raw_behav)
    elif filterprops[0] == 'tracknumber':
        filt_trials = filter_tracknumber(raw_behav,filterprops[1])
    elif filterprops[0] == 'cut tracknumber':
        filt_trials = cut_tracknumber(raw_behav,filterprops[1])
    elif filterprops[0] == 'trialnr_range':
        filt_trials = trialnr_range( raw_behav, filterprops )
    elif filterprops[0] == 'bool_pattern':
        filt_trials = bool_pattern( raw_behav, filterprops )
    elif filterprops[0] == 'animal_running':
        filt_trials = animal_running( raw_behav, filterprops )
    elif filterprops[0] == 'animal_notrunning':
        filt_trials = animal_notrunning( raw_behav, filterprops )
    elif filterprops[0] == 'pass':
        filt_trials = np.unique(raw_behav[:,6])
    elif filterprops[0] == 'running_speed':
        filt_trials = filter_running_speed(raw_behav, filterprops)
    elif filterprops[0] == 'opto_mask_light_on':
        filt_trials = filter_opto_mask_light_on(raw_behav)
    elif filterprops[0] == 'opto_mask_light_off':
        filt_trials = filter_opto_mask_light_off(raw_behav)
    elif filterprops[0] == 'opto_stim_on':
        filt_trials = opto_stim_on(raw_behav, filterprops)
    elif filterprops[0] == 'opto_mask_on_stim_off':
        filt_trials = opto_mask_on_stim_off(raw_behav)
    elif filterprops[0] == 'exclude_earlylick_trials':
        filt_trials = exclude_earlylick_trials(raw_behav, filterprops)
    elif filterprops[0] == 'first_lick_distance':
        filt_trials = first_lick_distance(raw_behav, filterprops)
    else:
        raise ValueError('Filter type not recognised.')

    # return trial numbers that have passed the filter and exist in pre_select
    if not isinstance(pre_select, int):
        return np.intersect1d(filt_trials, pre_select)
    else:
        return filt_trials
