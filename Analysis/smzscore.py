"""
Calculate the z-score for the spatial modulation of licking behavior for the
provided session. The spatial zscore is evaluated by calculating the mean
success rate (= fraction of successful trials) for shuffled data, subtracting
it from the actual success rate and dividing it by the standard deviation
of the shuffled distribution. The shuffled trial is considered successful if a
lick happened inside the reward zone after randomising their location.

Only successful trials are used to create the shuffled distribution.

Parameters
-------
licks :     ndarray
            licks dataset
rewards:    ndarray
            reward dataset
success_rate:
            float
            success rate of animal
rz :        tuple [float, float]
            start and end location of reward zone
default :   float
            default reward location
shuffles :  int
            number of shuffled carried out to calculate mean shuffled succes
            rate
tracklength :
            int
            total tracklength
mode :      int
            trial type to carry analysis out on

Outputs
-------
smi :       float
            spatial modulation index

"""

import numpy as np
import ipdb

from shuffle_dset import shuffle_dset
from select_trials import select_trials


def smzscore( raw, licks, rewards, success_rate, rz, default, shuffles, tracklength, mode=-1 ):
    """ Calculate the SMZ (Spatial z-score) for the provided session """

    # select trial type
    rewards = select_trials(rewards, 'reward', mode)
    licks = select_trials(licks, 'licks', mode)

    if licks.shape[0] > 0:

        # determine which trials were successful and only analyse those
        rewards_success = rewards[np.where(rewards[:,5] == 1),:][0]
        # create the 2-d vector that will hold the shuffled datasets
        shuffle_dist = np.zeros((np.shape(licks)[0], np.shape(licks)[1], shuffles))
        shuffled_sr = np.zeros((shuffles,))

        # create shuffled datset
        for i in range(shuffles):
            shuffle_dist[:,:,i] = shuffle_dset( raw, licks, tracklength )

        # loop through all shuffled lick-datasets
        shuffled_sr_all = 0
        for s in range(shuffles):
            # loop through each trial of the current shuffled lick dataset and check if there was at least one lick within the rewarded zone
            shuffled_sr_current = 0
            for t in np.unique(rewards_success[:,3]):
                cur_trial_ind = np.where(shuffle_dist[:,2,s]==t)[0]
                cur_trial = shuffle_dist[cur_trial_ind,:,s]
                if np.size(np.where( (cur_trial[:,1] > rz[0]) & (cur_trial[:,1] < default ) )[0]):
                    shuffled_sr_current += 1
            # calculate success rate for this shuffled licks dataset and add to overall success rate
            try:
                shuffled_sr[s] = shuffled_sr_current/np.shape(np.unique(rewards_success[:,3]))[0]
                shuffled_sr_all += shuffled_sr_current/np.shape(np.unique(rewards_success[:,3]))[0]
            except ZeroDivisionError:
                # if the mouse didn't have a successful trial at all, set SMI to 1
                shuffled_sr_all = 1
                shuffled_sr[s] = 0

        shuffled_sr_all	= shuffled_sr_all/shuffles

        if shuffled_sr_all > 0:
            if success_rate == 0:
                return 0, shuffled_sr
            else:
                return (success_rate - np.mean(shuffled_sr)) / np.std(shuffled_sr), shuffled_sr
        else:
            return (success_rate - 0.1) / 1, shuffled_sr
    else:
        return 0, shuffled_sr
