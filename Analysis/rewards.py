"""
Find locations where a reward was dispensed as well as time elapsed from start
of the track until receiving a reward

Parameters
-------
raw :   ndarray
        raw dataset
trialnr_column :
        int
        defines the column in which trial number is stored. For a combined-days
        dataset this is in position 13, in single days its in 9

Outputs
-------
reward_locs :   ndarray
                reward dataset

"""

import numpy as np
# import ipdb

if __name__ == "__main__":
    """ This script can not be executed stand-alone """
    print("This script can not be executed stand-alone")

def rewards_legacy(raw, trialnr_column=6):
    """ legacy version of function retained for compatibility and documentation purposes. LF 2018/08/04 """
    # pull out the lines of the raw dataset where a reward was dispensed
    rewards = np.delete(raw, np.where(raw[:,5] < 1), 0)

    # array that stores rewards
    reward_locs = np.zeros((1,6))

    # current trial number
    cur_trial_nr = 1

    while cur_trial_nr <= np.amax(raw[:,trialnr_column]):
        # retrieve individual trial
        cur_trial = np.delete(raw, np.where(raw[:,trialnr_column] != cur_trial_nr-1), 0)

        # pull out reward-line of current trial and append to results-dataset
        cur_rewards = rewards[rewards[:,trialnr_column]==cur_trial_nr-1, :]
        if cur_rewards.shape[0] > 0:
            # clause required as in rare circumstances there is no light-dark transition in Task 8
            reward_locs = np.append(reward_locs, [[cur_rewards[0,0]-cur_trial[0,0], cur_rewards[0,1], cur_rewards[0,4], cur_rewards[0,6], cur_rewards[0,0], cur_rewards[0,5]]], axis=0)

        # increment trial number
        cur_trial_nr += 1

    # cut away first element which is just zeros since we just used that to initialise it
    reward_locs = reward_locs[1:][:]

    return reward_locs

def rewards(raw, trialnr_column=6, corr_trialnr=True):
    if corr_trialnr:
        corr_trialnr = 1
    else:
        corr_trialnr = 0
    # pull out all lines of the raw dataset where a reward was dispensed
    rewards = raw[raw[:,5] > 0,:]

    # array that stores rewards
    reward_locs = np.zeros((1,6))

    # current trial number
    cur_trial_nr = 1

    
    for cur_trial_nr in np.unique(raw[:,trialnr_column]):
        # retrieve individual trial and the reward associated with it
        cur_trial = raw[raw[:,trialnr_column] == cur_trial_nr,:]
        cur_trial_reward = rewards[rewards[:,trialnr_column] == cur_trial_nr+corr_trialnr,:]

        # check if reward exists (not the case for black box trials)
        if np.size(cur_trial_reward) > 0:
            reward_locs = np.append(reward_locs,[[cur_trial_reward[0,0]-cur_trial[0,0], cur_trial[-1,1], cur_trial[-1,4], cur_trial_nr, cur_trial_reward[0,0], cur_trial_reward[0,5]]], axis=0)

    # cut away first element which is just zeros since we just used that to initialise it
    reward_locs = reward_locs[1:][:]

    return reward_locs
