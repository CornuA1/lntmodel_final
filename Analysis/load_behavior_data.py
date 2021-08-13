"""
Load behavior data from raw VR datafile and return dataset in such a manner
that it can be dropped in place of where data is usually loaded from HDF5

@author: lukasfischer

Parameters
--------
fname : string
		file path + name to the desired raw datafile

type : string
	   type of VR session
	   'vr' = regular, closed loop session
	   'openloop' = vr openloop session
	   'dark' = running in the dark
	   'vd' = visual discrimination task session


Outputs
--------
behav_ds : behav_ds
		   behavior dataset as loaded from the raw file


"""

# import csv
from numpy import genfromtxt
from scipy.interpolate import griddata
import numpy as np

def load_data(fname, type='vr', interp=True, tt_reject=True):

	raw_behaviour = genfromtxt(fname, delimiter=';')
	behaviour_aligned = np.copy(raw_behaviour)

	# fix original timestamps - due to a bug in recording system time (not enough precision), we need to reconstruct the time from the latency timestamps
	# first, check if the sum of the latency timestamps is close to the recorded system time. if yes, just replace by cumsum
	# if not: it is almost alway due to an offset in the first 1-2 frames, so we just subtract the difference and set the first couple frames to 0
	init_offset = (np.sum(raw_behaviour[:,2]))-(raw_behaviour[-1,0]-raw_behaviour[0,0])
	if init_offset < 0.05:
		print('adjusting timestamps without init offset')
		behaviour_aligned[:,0] = np.cumsum(raw_behaviour[:,2])
	else:
		print('adjusting timestamps with init offset')
		behaviour_aligned[:,0] = np.cumsum(raw_behaviour[:,2]) + raw_behaviour[0,0] - init_offset

	# calculate average sampling rate for behaviour
	num_ts_behaviour = np.size(behaviour_aligned[:,0])
	raw_behaviour = np.copy(behaviour_aligned)
	if interp==True:
		#behaviour_aligned = np.zeros((np.size(raw_behaviour,0),np.size(raw_behaviour,1)))
		#behaviour_aligned = np.copy(raw_behaviour)
		behaviour_aligned[:,5] = 0
		behaviour_aligned[:,7] = 0
		# to avoid poor alignment of imaging data due fluctuations in frame latency of the VR,
		# create evenly spaced timepoints and interpolate behaviour data to match them
		print('Resampling behavior data...')
		even_ts = np.linspace(behaviour_aligned[0,0], behaviour_aligned[-1,0], np.size(behaviour_aligned,0))
		# behaviour_aligned[:,1] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,1], even_ts, 'linear')
		# behaviour_aligned[:,3] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,3], even_ts, 'linear')
		behaviour_aligned[:,1] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,1], even_ts, 'linear')
		behaviour_aligned[:,3] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,3], even_ts, 'linear')
		if np.size(behaviour_aligned,1) > 8:
			behaviour_aligned[:,8] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,8], even_ts, 'linear')
		behaviour_aligned[:,2] = np.insert(np.diff(even_ts),int(np.mean(even_ts)),0)
		behaviour_aligned[:,0] = even_ts

		# find trial transition points and store which track each trial was
		# carried out on. Further down we will re-assign trial number and tracks
		# as just re-assigning by nearest timepoint (see below) is problematic
		# if it offset is large and fluctuates
		trial_idx = np.where(np.insert(np.diff(behaviour_aligned[:,6]),0,0) > 0)
		trial_idx = np.insert(trial_idx,0,0)
		trial_nrs = behaviour_aligned[trial_idx,6]
		trial_tracks = behaviour_aligned[trial_idx,4]
		# set every reward to just a single row flag (rather than being >0 for as long as the valve is open)
		# as this information is hard to retain after interpolating the dataset
		rew_col = np.diff(raw_behaviour[:,5])
		rew_col = np.insert(rew_col,0,0)
		# find indices where valve was opened and set values accordingly to 1 or 2
		raw_behaviour[:,5] = 0
		valve_open = np.where(rew_col>0)[0]
		if np.size(valve_open) > 0:
			raw_behaviour[valve_open,5] = 1
		valve_open = np.where(rew_col>1)[0]
		if np.size(valve_open) > 0:
			raw_behaviour[valve_open,5] = 2

		# loop through each row of the raw data and find the index of the nearest adjusted timestamp
		# and move the rest of the raw data that hasn't been interpolated to its new location
		print('Finding closest timepoint in resampled data for binary events...')
		new_idx = np.zeros((np.size(raw_behaviour[:,0],0)))
		for i,ats in enumerate(raw_behaviour[:,0]):
			new_idx[i] = (np.abs(behaviour_aligned[:,0]-ats)).argmin()
			# shift licks-column. If a row in the new dataset contains a 1 already,
			# don't shift as we don't want the 1 to be overwritten by a 0 that
			# may fall on the same row
			if behaviour_aligned[int(new_idx[i]),7] == 0:
				behaviour_aligned[int(new_idx[i]),7] = raw_behaviour[i,7]

			behaviour_aligned[int(new_idx[i]),4] = raw_behaviour[i,4]
			if behaviour_aligned[int(new_idx[i]),5] == 0:
				behaviour_aligned[int(new_idx[i]),5] = raw_behaviour[i,5]
				#if raw_behaviour[i,5] == 1:
				#    print(i)
			behaviour_aligned[int(new_idx[i]),6] = raw_behaviour[i,6]
		# pull out adjusted trial transition indices
		new_trial_idx = new_idx[trial_idx]
		new_trial_idx = np.append(new_trial_idx, new_idx[-1])
		# overwrite the trial and track numbers to avoid fluctuation at
		# transition points
		for i in range(1,np.size(new_trial_idx,0)):
			behaviour_aligned[int(new_trial_idx[i-1]+1):int(new_trial_idx[i]+1),4] = trial_tracks[i-1]
			behaviour_aligned[int(new_trial_idx[i-1]+1):int(new_trial_idx[i]+1),6] = trial_nrs[i-1]

	# update number of timepoints for behavior
	num_ts_behaviour = np.size(behaviour_aligned[:,0])

	print('Cleaning up trial transition points..')
	if interp==True and tt_reject==True:
		# delete 3 samples around each trial transition as the interpolation can cause the
		# location to be funky at trial transition. The -1 indexing has to do with
		# the way indeces shift as they are being deleted.
		shifted_trial_idx = np.where(np.insert(np.diff(behaviour_aligned[:,6]),0,0) > 0)[0] - 1
		# keep track by how much we've shifted indeces through deleting rows
		index_adjust = 0
		# loop through each trial transition point
		for i in shifted_trial_idx:
			# detect interpolation artifacts and delete appropriate rows. Allow for a maximum of 7 rows to be deleted
			# first we delete rows
			for k in range(10):
				if behaviour_aligned[i-index_adjust,5] > 0:
					behaviour_aligned[i-index_adjust-1,5] = behaviour_aligned[i-index_adjust,5]
				if behaviour_aligned[i-index_adjust,7] > 0:
					behaviour_aligned[i-index_adjust-1,7] = behaviour_aligned[i-index_adjust,7]
					# print(behaviour_aligned[shifted_trial_idx,1])
				#print('deleting location: ', str(behaviour_aligned[i-index_adjust,1]), ' trial: ', str(behaviour_aligned[i-index_adjust,6]), ' index: ', str(i-index_adjust))
				# dF_aligned_resampled = np.delete(dF_aligned_resampled, i-index_adjust, axis=0)
				behaviour_aligned = np.delete(behaviour_aligned, i-index_adjust, axis=0)
				# bri_aligned = np.delete(bri_aligned, i-index_adjust, axis=0)
				index_adjust += 1
				# if the last datapoint of the corrected trial is larger than the previous one (with some tolerance), quit loop
				if behaviour_aligned[i-index_adjust,1]+1 >= behaviour_aligned[i-index_adjust-1,1]:
					#print('breaking at ', str(k))
					break


	return behaviour_aligned
