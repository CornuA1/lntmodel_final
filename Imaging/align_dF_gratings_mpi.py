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

behav_idx_range : [float,float]
        define start and end frames of the raw behaviour dataset to which the imaging dataset is
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
import multiprocessing

def resample_roi(inp):
    #print('processing ROI: ' + str(inp[2]))
    return signal.resample(inp[0], inp[1], axis=0)

def align_dF_gratings( raw_behaviour, raw_dF, frame_bri=[], behav_idx_range=[-1,-1], gcamp_idx_range=[-1,-1], interp=True, tt_reject=True, session_crop=[0,1] ): #for current frame latency issue end point should be 1790.15s (based on estimate of 40.22Hz sampling)
    # create copy of raw_behaviour to be cropped

    print('Trimming behavior data...')
    # crop beginning and end
    if behav_idx_range[0] != -1:
        raw_behaviour = raw_behaviour[behav_idx_range[0]:,:]
    if behav_idx_range[1] != -1:
        raw_behaviour = raw_behaviour[:(np.size(raw_behaviour,0)-behav_idx_range[1]),:]
    # remove gcamp indices outside the specified window
    print('Trimming imaging data...')
    if gcamp_idx_range[0] != -1:
        raw_dF = raw_dF[gcamp_idx_range[0]:,:]
    if gcamp_idx_range[1] != -1:
        raw_dF = raw_dF[:(np.size(raw_dF,0)-gcamp_idx_range[1]),:]

    behaviour_aligned = np.copy(raw_behaviour)
    dF_aligned = np.copy(raw_dF)

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
    # check that the latency between frames of the virtual reality has been
    # fairly constant. If it is not, then the VR may have been jumpy during
    # the session and matching dF samples to behaviour is less accurate as
    # a constant frame latency is assumed
    if np.std(raw_behaviour[:,2]) > 0.005 and interp==False:
        warnings.warn("High fluctuations in frame latency of behavioural data detected. Matching imaging to behavioural data may be negatively affected.")

    raw_behaviour = np.copy(behaviour_aligned)
    if interp==True:
#        behaviour_aligned = np.zeros((np.size(raw_behaviour,0),np.size(raw_behaviour,1)))
        behaviour_aligned = np.copy(raw_behaviour)
        # to avoid poor alignment of imaging data due fluctuations in frame latency of the VR,
        # create evenly spaced timepoints and interpolate behaviour data to match them
        even_ts = np.linspace(behaviour_aligned[0,0], behaviour_aligned[-1,0], np.size(behaviour_aligned,0))
        behaviour_aligned[:,2] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,2], even_ts, 'linear')
        behaviour_aligned[:,3] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,3], even_ts, 'linear')
        behaviour_aligned[:,4] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,4], even_ts, 'linear')
        behaviour_aligned[:,7] = griddata(behaviour_aligned[:,0], behaviour_aligned[:,7], even_ts, 'linear')
        behaviour_aligned[:,5] = np.insert(np.diff(even_ts),int(np.mean(even_ts)),0)
        behaviour_aligned[:,0] = even_ts

        # find trial transition points and store which track each trial was
        # carried out on. Further down we will re-assign trial number and tracks
        # as just re-assigning by nearest timepoint (see below) is problematic
        # if it offset is large and fluctuates
        trial_idx = np.where(np.insert(np.diff(behaviour_aligned[:,1]),0,0) != 0)
        trial_idx = np.insert(trial_idx,0,0)
        trial_tracks = behaviour_aligned[trial_idx,1]

        # loop through each row of the raw data and find the index of the nearest adjusted timestamp
        # and move the rest of the raw data that hasn't been interpolated to its new location
        new_idx = np.zeros((np.size(raw_behaviour[:,0],0)))
        for i,ats in enumerate(raw_behaviour[:,0]):
            # shift licks-column. If a row in the new dataset contains a 1 already,
            # don't shift as we don't want the 1 to be overwritten by a 0 that
            # may fall on the same row
            if behaviour_aligned[int(new_idx[i]),8] == 0:
                behaviour_aligned[int(new_idx[i]),8] = raw_behaviour[i,8]

            new_idx[i] = (np.abs(behaviour_aligned[:,0]-ats)).argmin()
            behaviour_aligned[int(new_idx[i]),1] = raw_behaviour[i,1]
        # pull out adjusted trial transition indices
        new_trial_idx = new_idx[trial_idx]
        new_trial_idx = np.append(new_trial_idx, new_idx[-1])
        # overwrite the trial and track numbers to avoid fluctuation at
        # transition points
        for i in range(1,np.size(new_trial_idx,0)):
            behaviour_aligned[int(new_trial_idx[i-1])+1:int(new_trial_idx[i])+1,1] = trial_tracks[i-1]

        # resample dF/F signal
        print('Resampling imaging data...')
        p = multiprocessing.Pool()
        #dF_aligned = signal.resample(dF_aligned, num_ts_behaviour, axis=0)
        # create tuples of (column of dF_aligned, num_ts_behaviour)
        dF_ts_tuples = []
        for j,col in enumerate(dF_aligned.T):
            dF_ts_tuples.append((col,num_ts_behaviour,j))
        resampled_roi = p.map(resample_roi, dF_ts_tuples)

        dF_aligned_resampled = np.zeros((num_ts_behaviour,np.size(dF_aligned,1)))
        for i,col in enumerate(resampled_roi):
            dF_aligned_resampled[:,i] = col

        if np.size(frame_bri) > 0:
            print('Resampling brightness data...')
            bri_aligned = signal.resample(frame_bri, num_ts_behaviour, axis=0)
        else:
            bri_aligned = []

    return dF_aligned, behaviour_aligned, bri_aligned
