"""
Find locations at which animal as licked the reward spout. Licks that happen
within 20 cm after a reward having been dispensed are stored in a separate
dataset.

Parameters
-------
raw :       ndarray
            raw data
rewards :   ndarray
            reward dataset
trialnr_column :
            int
            column in 'raw' containing the trial number

Outputs
-------
lick_res :  ndarray
            dataset containing lick-data
lick_post_res :
            ndarray
            dataset containing licks that happened after a reward was dispensed
            according to above mentioned criteria.

"""

import numpy as np

if __name__ == "__main__":
	""" This script can not be executed stand-alone """
	print("This script can not be executed stand-alone")

def licks( raw, rewards, trialnr_column=6 ):
	# we will store the indeces of the stops that happen after the reward was dispensed
	post_lick_ind = np.zeros(1,dtype=int)
	# find all samples where a lick was detected
	licks = np.where(raw[:,7] > 0)[0]
	tot_licks = licks.shape[0]
	i = 0
	# remove licks that happened after a reward was dispensed. This loop could be vectorised but I don' have the time right now and the optimisation would be minimal
	while i < tot_licks:
		# pull out current reward. This slightly weird construct with the extra if is because h5py trips up upon an empyt index array
		cur_rew_ind = np.where(rewards[:,3] == raw[licks[i],trialnr_column])[0]
		# if there is a reward: reject licks that happen between tone and reward or are within 10 cm after the reward
		if cur_rew_ind.shape[0]>0:
			cur_rew = rewards[cur_rew_ind,:][0]
			# test if current stops is within a given distance reward being dispensed and some other conditions
			if (((raw[licks[i],0] > cur_rew[4]) and
				(raw[licks[i],1] <= cur_rew[1]+20.0)) and
				(raw[licks[i],trialnr_column] == cur_rew[3])
				):
				post_lick_ind = np.append(post_lick_ind, licks[i])
				licks = np.delete(licks, i)
				tot_licks -= 1
				i-=1
		i+=1

	if licks.shape[0] > 1:
		lick_res = np.zeros((licks.shape[0],4))
		lick_res[:,0] = raw[licks,0]
		lick_res[:,1] = raw[licks,1]
		lick_res[:,2] = raw[licks,6]
		lick_res[:,3] = raw[licks,4]
	else:
		lick_res = []

	if post_lick_ind.shape[0] > 1:
		post_lick_ind = post_lick_ind[1:][:]
		lick_post_res = np.zeros((post_lick_ind.shape[0],4))
		lick_post_res[:,0] = raw[post_lick_ind,0]
		lick_post_res[:,1] = raw[post_lick_ind,1]
		lick_post_res[:,2] = raw[post_lick_ind,6]
		lick_post_res[:,3] = raw[post_lick_ind,4]
	else:
		lick_post_res = []
	return lick_res, lick_post_res

def licks_nopost( raw, trialnr_column=6 ):
	# find all samples where a lick was detected
	licks = np.where(raw[:,7] == 1)[0]

	if licks.shape[0] > 1:
		lick_res = np.zeros((licks.shape[0],4))
		lick_res[:,0] = raw[licks,0]
		lick_res[:,1] = raw[licks,1]
		lick_res[:,2] = raw[licks,6]
		lick_res[:,3] = raw[licks,4]
	else:
		lick_res = []

	return lick_res
