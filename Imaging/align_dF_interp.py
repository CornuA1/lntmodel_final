"""
align_dF
align dF traces to match behavioural data using the scipy.signal.resample
function and providng stop/end timepoints. It is assumed that dF/F traces are
collected at a constant sampling rate.

Since behavioural data has been found to fluctuate in a way that can lean to
significant cumulative lag builup, it is recommended to interpolate location and speed data
to evenly spaced timestamps (set interp=True).

Parameters
-------
raw_behaviour : ndarray
        raw behavioural data

raw_dF : ndarray
        raw dF/F data

skip_init : int
        disregards initial period defined by a specific track number provided
        in this argument. If -1, nothing will skipped.

timewindow : [float,float]
        define start and end timepoints of the raw behaviour dataset to which the imaging dataset is
        to be aligned. The entire dF dataset passed into the function will be algined to this time window.
        -1 means from beginning or until end respectively. If a start time is provided, skip_init will be
        ignored.

interp : boolean
        if True: location and speed data will be interpolated to match evenly
        distributed timestamps

tt_reject : boolean
        reject frames around trial transitions as the interpolation of location
        and speed can lead to unclean transition from end to beginning of track.
        Downside: frames are dropped and therefore information (if little) is
        lost and alignment is original tiff-stack is difficult

gcamp_idx_range : [int, int]
        first and last index used from provided raw gcamp trace. -1 skips nothing

"""

import warnings

import numpy as np
from scipy import signal
from scipy.interpolate import griddata

def align_dF( raw_behaviour, raw_dF, frame_bri=[], skip_init=-1, behav_timewindow=[-1,-1], interp=True, tt_reject=True, gcamp_idx_range=[-1,-1] ): #for current frame latency issue end point should be 1790.15s (based on estimate of 40.22Hz sampling)
    # create copy of raw_behaviour to be cropped
    behaviour_aligned = np.copy(raw_behaviour)
    dF_aligned = raw_dF
    print('Cropping behavior data...')
    # remove lines specified in skip_init if behav_timewindow[0] is not specified
    if behav_timewindow[0] == -1:
        behaviour_aligned = raw_behaviour[raw_behaviour[:,4]!=skip_init,:]
    # crop beginning and end
    if behav_timewindow[0] != -1:
        behaviour_aligned = behaviour_aligned[behaviour_aligned[:,0]>behav_timewindow[0],:]
    if behav_timewindow[1] != -1:
        behaviour_aligned = behaviour_aligned[behaviour_aligned[:,0]<behav_timewindow[1],:]

    # remove gcamp indices outside the specified window
    print('Cropping imaging data...')
    if gcamp_idx_range[0] != -1:
        dF_aligned = dF_aligned[gcamp_idx_range[0]:,:]
    if gcamp_idx_range[1] != -1:
        dF_aligned = dF_aligned[:gcamp_idx_range[1],:]


    # calculate average sampling rate for behaviour
    num_ts_behaviour = np.size(behaviour_aligned[:,0])
    # check that the latency between frames of the virtual reality has been
    # fairly constant. If it is not, then the VR may have been jumpy during
    # the session and matching dF samples to behaviour is less accurate as
    # a constant frame latency is assumed
    if np.std(raw_behaviour[:,2]) > 0.005 and interp==False:
        warnings.warn("High fluctuations in frame latency of behavioural data detected. Matching imaging to behavioural data may be negatively affected.")

    raw_behaviour = np.copy(behaviour_aligned)

    if interp==True:
#        behaviour_aligned = np.zeros((np.size(raw_behaviour,0),np.size(raw_behaviour,1)))
        #behaviour_aligned = np.copy(raw_behaviour)
        behaviour_aligned[:,5] = 0
        behaviour_aligned[:,7] = 0
        # to avoid poor alignment of imaging data due fluctuations in frame latency of the VR,
        # create evenly spaced timepoints and interpolate behaviour data to match them
        print('Resampling behavior data...')
        even_ts = np.linspace(behaviour_aligned[0,0], behaviour_aligned[-1,0], np.size(behaviour_aligned,0))
        behaviour_aligned[:,1] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,1], even_ts, 'linear')
        behaviour_aligned[:,3] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,3], even_ts, 'linear')
        if np.size(behaviour_aligned,1) > 8:
            behaviour_aligned[:,8] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,8], even_ts, 'linear')
        behaviour_aligned[:,2] = np.insert(np.diff(even_ts),np.mean(even_ts),0)
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
            if behaviour_aligned[new_idx[i],7] == 0:
                behaviour_aligned[new_idx[i],7] = raw_behaviour[i,7]

            behaviour_aligned[new_idx[i],4] = raw_behaviour[i,4]
            if behaviour_aligned[new_idx[i],5] == 0:
                behaviour_aligned[new_idx[i],5] = raw_behaviour[i,5]
                #if raw_behaviour[i,5] == 1:
                #    print(i)
            behaviour_aligned[new_idx[i],6] = raw_behaviour[i,6]
        # pull out adjusted trial transition indices
        new_trial_idx = new_idx[trial_idx]
        new_trial_idx = np.append(new_trial_idx, new_idx[-1])
        # overwrite the trial and track numbers to avoid fluctuation at
        # transition points
        for i in range(1,np.size(new_trial_idx,0)):
            behaviour_aligned[new_trial_idx[i-1]+1:new_trial_idx[i]+1,4] = trial_tracks[i-1]
            behaviour_aligned[new_trial_idx[i-1]+1:new_trial_idx[i]+1,6] = trial_nrs[i-1]

    # resample dF/F signal
    print('Resampling imaging data...')
    dF_aligned = signal.resample(dF_aligned, num_ts_behaviour, axis=0)
    if frame_bri != []:
        print('Resampling brightness data...')
        bri_aligned = signal.resample(frame_bri, num_ts_behaviour, axis=0)
    else:
        bri_aligned = []
    print('Cleaning up trial transition points..')
    if interp==True and tt_reject==True:
        # delete 3 samples around each trial transition as the interpolation can cause the
        # location to be funky at trial transition. The -1 indexing has to do with
        # the way indeces shift as they are being deleted.
        shifted_trial_idx = np.where(np.insert(np.diff(behaviour_aligned[:,6]),0,0) > 0)[0]
        dF_aligned = np.delete(dF_aligned, shifted_trial_idx, axis=0)
        behaviour_aligned = np.delete(behaviour_aligned, shifted_trial_idx, axis=0)

        shifted_trial_idx = np.where(np.insert(np.diff(behaviour_aligned[:,6]),0,0) > 0)[0]
        dF_aligned = np.delete(dF_aligned, shifted_trial_idx-1, axis=0)
        behaviour_aligned = np.delete(behaviour_aligned, shifted_trial_idx-1, axis=0)

        #dF_aligned = np.delete(dF_aligned, shifted_trial_idx-1, axis=0)
        #behaviour_aligned = np.delete(behaviour_aligned, shifted_trial_idx-1, axis=0)
    return dF_aligned, behaviour_aligned, bri_aligned
