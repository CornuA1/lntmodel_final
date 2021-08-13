"""
Create a shuffled distribution of a stop of lick dataset. For each trial the
location of the stop or lick is rotated by a random value between 0 and
tracklength. All stops/licks  of a given trial are rotated by the same amount,
but that amount is randomly recalculated for each trial.

Parameters
-------
raw :   ndarray
        raw dataset

licks : ndarray
        licks dataset
tracklength :
            ndarray
            total length of the track
trialnr_column :
            int
            column that contains the trial number

Outputs
-------
shuffled_dset :
            ndarray
            shuffled dataset. Has same format as licks

"""

import ipdb
import numpy as np
from scipy.stats import uniform

if __name__ == "__main__":
	""" This script can not be executed stand-alone """
	print("This script can not be executed stand-alone")

def shuffle_dset( raw, licks, tracklength, trialnr_column=2 ):
    # this is required as otherwise the original dataset would be altered
    shuffled_dset = np.copy(licks)
    # loop through every trial
    for cur_trial_nr in np.unique(shuffled_dset[:,trialnr_column]):
        # generate random value by which location value will be rotated
        cur_trial = raw[raw[:,6]==cur_trial_nr,:]
        rand_rotation = uniform.rvs(loc=cur_trial[0,1], scale=tracklength)
        # retrieve licks from curent trial and add random value
        shuffled_dset[shuffled_dset[:,trialnr_column]==cur_trial_nr, 1] += rand_rotation
        # licks that would now extend beyond the end of the track are wrapped around
        shuffled_dset[np.logical_and(shuffled_dset[:,trialnr_column]==cur_trial_nr, shuffled_dset[:,1] > tracklength),1] -= (tracklength - cur_trial[0,1])
    return shuffled_dset
