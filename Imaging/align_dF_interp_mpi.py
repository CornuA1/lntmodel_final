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

behav_idx_skip : [int,int]
        define frames of the behavior dataset to be skipped at start and end in number of frames.
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

gcamp_idx_skip : [int, int]
        first and last index used from provided raw gcamp trace. -1 skips nothing

session_crop : [float, float]
        fraction of session to be aligned and returned (e.g. [0,0.5] only aligns only the first half. [0.25,0.75] the middle half, etc).
        The behavior data is split based on the timestamp (column 0), imaging data is split based on index numberself.

"""

import warnings

# import ipdb
import numpy as np
from scipy import signal
from scipy.interpolate import griddata
import multiprocessing

def resample_roi(inp):
    #print('processing ROI: ' + str(inp[2]))
    return signal.resample(inp[0], inp[1], axis=0)

def align_dF( raw_behaviour, raw_dF, frame_bri=[], behav_idx_range=[-1,-1], gcamp_idx_range=[-1,-1], interp=True, tt_reject=True, session_crop=[0,1], eye_data=None ): #for current frame latency issue end point should be 1790.15s (based on estimate of 40.22Hz sampling)

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
        if eye_data is not None:
            eye_data['pupil_area_timecourse'] = eye_data['pupil_area_timecourse'][gcamp_idx_range[0]:]
            eye_data['pupil_center_timecourse'][:,0] = eye_data['pupil_center_timecourse'][gcamp_idx_range[0]:,0]
            eye_data['pupil_center_timecourse'][:,1] = eye_data['pupil_center_timecourse'][gcamp_idx_range[0]:,1]
    if gcamp_idx_range[1] != -1:
        print('Cropping end of gcamp signal (' + str(gcamp_idx_range[1]) + ') frames...')
        raw_dF = raw_dF[:(np.size(raw_dF,0)-gcamp_idx_range[1]),:]
        if eye_data is not None:
            eye_data['pupil_area_timecourse'] = eye_data['pupil_area_timecourse'][:(np.size(raw_dF,0)-gcamp_idx_range[1])]
            eye_data['pupil_center_timecourse'][:,0] = eye_data['pupil_center_timecourse'][:(np.size(raw_dF,0)-gcamp_idx_range[1]),0]
            eye_data['pupil_center_timecourse'][:,1] = eye_data['pupil_center_timecourse'][:(np.size(raw_dF,0)-gcamp_idx_range[1]),1]

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

    # if only part of session data is to be used, crop
    raw_behavior_cropped = np.copy(raw_behaviour)
    if session_crop[0] > 0:
        print('Cropping start of data...')
        behavior_start = behaviour_aligned[-1,0] * session_crop[0]
        behavior_start_idx = (np.abs(behaviour_aligned[:,0]-behavior_start)).argmin()
        behaviour_aligned = behaviour_aligned[behavior_start_idx:,:]
        dF_start_idx = int(np.size(dF_aligned,0) * session_crop[0])
        dF_aligned = dF_aligned[dF_start_idx:,:]


    if session_crop[1] < 1:
        print('Cropping end of data...')
        behavior_end = behaviour_aligned[-1,0] * session_crop[1]
        print(behavior_end)
        behavior_end_idx = (np.abs(behaviour_aligned[:,0]-behavior_end)).argmin()
        behaviour_aligned = behaviour_aligned[:behavior_end_idx,:]
        dF_end_idx = int(np.size(dF_aligned,0) * session_crop[1])
        dF_aligned = dF_aligned[dF_end_idx:,:]
    # update number of timepoints for behavior
    num_ts_behaviour = np.size(behaviour_aligned[:,0])

    # resample dF/F signal
    print('Resampling imaging data...')
    #p = multiprocessing.Pool()
    dF_aligned_resampled = signal.resample(dF_aligned, num_ts_behaviour, axis=0)
    # create tuples of (column of dF_aligned, num_ts_behaviour)
    #dF_ts_tuples = []
    #for j,col in enumerate(dF_aligned.T):
    #    dF_ts_tuples.append((col,num_ts_behaviour,j))
    #resampled_roi = p.map(resample_roi, dF_ts_tuples)
#
    #dF_aligned_resampled = np.zeros((num_ts_behaviour,np.size(dF_aligned,1)))
    #for i,col in enumerate(resampled_roi):
     #   dF_aligned_resampled[:,i] = col

    if np.size(frame_bri) > 0:
        print('Resampling brightness data...')
        # check if frame brightness has to be transposed
        if frame_bri.shape[0] < 2:
            print('transposing brigthness data before resampling...')
            bri_aligned = signal.resample(frame_bri.T, num_ts_behaviour, axis=0)
            if eye_data is not None:
                eye_data_aligned = signal.resample(eye_data['pupil_area_timecourse'], num_ts_behaviour, axis=0)
                eye_x_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,0], num_ts_behaviour, axis=0)
                eye_y_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,1], num_ts_behaviour, axis=0)
            else:
                eye_data_aligned = np.nan
                eye_x_aligned = np.nan
                eye_y_aligned = np.nan
        else:
            bri_aligned = signal.resample(frame_bri, num_ts_behaviour, axis=0)
            if eye_data is not None:
                eye_data_aligned = signal.resample(eye_data['pupil_area_timecourse'], num_ts_behaviour, axis=0)
                eye_x_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,0], num_ts_behaviour, axis=0)
                eye_y_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,1], num_ts_behaviour, axis=0)
            else:
                eye_data_aligned = np.nan
                eye_x_aligned = np.nan
                eye_y_aligned = np.nan
    else:
        bri_aligned = []

    if interp==True and tt_reject==True:
        print('Cleaning up trial transition points..')
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
                dF_aligned_resampled = np.delete(dF_aligned_resampled, i-index_adjust, axis=0)
                behaviour_aligned = np.delete(behaviour_aligned, i-index_adjust, axis=0)
                bri_aligned = np.delete(bri_aligned, i-index_adjust, axis=0)
                if eye_data is not None:
                    eye_data_aligned = np.delete(eye_data_aligned, i-index_adjust, axis=0)
                    eye_x_aligned = np.delete(eye_x_aligned, i-index_adjust, axis=0)
                    eye_y_aligned = np.delete(eye_y_aligned, i-index_adjust, axis=0)
                index_adjust += 1
                # if the last datapoint of the corrected trial is larger than the previous one (with some tolerance), quit loop
                if behaviour_aligned[i-index_adjust,1]+1 >= behaviour_aligned[i-index_adjust-1,1]:
                    #print('breaking at ', str(k))
                    break

            # for k in range(10):
            #     print('deleting location: ', str(behaviour_aligned[i-index_adjust+1,1]), ' trial: ', str(behaviour_aligned[i-index_adjust+1,6]), ' index: ', str(i-index_adjust))
            #     dF_aligned_resampled = np.delete(dF_aligned_resampled, i-index_adjust+1, axis=0)
            #     behaviour_aligned = np.delete(behaviour_aligned, i-index_adjust+1, axis=0)
            #     index_adjust += 1
            #     if behaviour_aligned[i-index_adjust+1,1]-1 <= behaviour_aligned[i-index_adjust+2,1]:
            #         break

        # shifted_trial_idx = np.where(np.insert(np.diff(behaviour_aligned[:,6]),0,0) > 0)[0]
        # # print(behaviour_aligned[shifted_trial_idx,1])
        # dF_aligned_resampled = np.delete(dF_aligned_resampled, shifted_trial_idx-1, axis=0)
        # behaviour_aligned = np.delete(behaviour_aligned, shifted_trial_idx-1, axis=0)
        #
        # shifted_trial_idx = np.where(np.insert(np.diff(behaviour_aligned[:,6]),0,0) > 0)[0]
        # # print(behaviour_aligned[shifted_trial_idx,1])
        # dF_aligned_resampled = np.delete(dF_aligned_resampled, shifted_trial_idx-1, axis=0)
        # behaviour_aligned = np.delete(behaviour_aligned, shifted_trial_idx-1, axis=0)

        #dF_aligned = np.delete(dF_aligned, shifted_trial_idx-1, axis=0)
        #behaviour_aligned = np.delete(behaviour_aligned, shifted_trial_idx-1, axis=0)

    if eye_data is None:
        return dF_aligned_resampled, behaviour_aligned, bri_aligned
    else:
        return dF_aligned_resampled, behaviour_aligned, bri_aligned, eye_data_aligned, eye_x_aligned, eye_y_aligned
