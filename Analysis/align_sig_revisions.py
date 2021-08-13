# -*- coding: utf-8 -*-
"""
Read .sig file, calculate dF/F and align to behavior data. Store result in
same file

@author: Lukas Fischer


"""

import sys, yaml, os
os.chdir('C:/Users/Lou/Documents/repos/LNT')
with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + "Analysis")
sys.path.append(loc_info['base_dir'] + "Imaging")
import warnings; warnings.simplefilter('ignore')

import ipdb
import numpy as np
import scipy.io as sio
from dF_win import dF_win
from align_dF_interp_mpi import align_dF
from load_behavior_data import load_data
from scipy.interpolate import griddata
from scipy import signal

def align_eyedata( data_path, sess, behavior_file, eye_file, tt_reject=True ):
    interp = True
    print('Loading eye data...')
    eye_data = sio.loadmat(data_path + os.sep + sess + os.sep + eye_file, appendmat=False )
    print('Loading behavior data for alignment...')
    behavior_file = data_path + os.sep + sess + os.sep + behavior_file
    raw_behaviour = np.genfromtxt(behavior_file, delimiter=';')

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

    print('Resampling eye data...')
    # check if frame brightness has to be transposed
    eye_data_aligned = signal.resample(eye_data['pupil_area_timecourse'], num_ts_behaviour, axis=0)
    eye_x_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,0], num_ts_behaviour, axis=0)
    eye_y_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,1], num_ts_behaviour, axis=0)


    # ipdb.set_trace()
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

                behaviour_aligned = np.delete(behaviour_aligned, i-index_adjust, axis=0)
                eye_data_aligned = np.delete(eye_data_aligned, i-index_adjust,axis=0)
                eye_x_aligned = np.delete(eye_x_aligned, i-index_adjust,axis=0)
                eye_y_aligned = np.delete(eye_y_aligned, i-index_adjust,axis=0)
                index_adjust += 1

                # if the last datapoint of the corrected trial is larger than the previous one (with some tolerance), quit loop
                if behaviour_aligned[i-index_adjust,1]+1 >= behaviour_aligned[i-index_adjust-1,1]:
                    #print('breaking at ', str(k))
                    break

    sio.savemat(data_path + os.sep + sess + os.sep + 'aligned_eyedata.mat', mdict= \
                                        {'behaviour_aligned' : behaviour_aligned, \
                                        'eye_data_aligned' : eye_data_aligned, \
                                        'eye_x_aligned' : eye_x_aligned.T, \
                                        'eye_y_aligned' : eye_y_aligned.T })
    print('saved ' + data_path + os.sep + sess + os.sep + 'aligned_eyedata.mat')

def calc_dF(data_path, sess, sigfile ,meta_file=None, sbx_version=2, session_crop=[0,1], method=2, rois=[], sig_suffix=''):
    if sbx_version == 1:
        frame_brightness = np.genfromtxt( data_path + os.sep + sess + os.sep + meta_file )
        print('Brigthness data found...')
    elif sbx_version == 2:
        # read additional ROI metadata
        print('Brightness + metadata data found...')
        rec_info = sio.loadmat( data_path + os.sep + sess + os.sep + meta_file, appendmat=False )
        frame_brightness = rec_info['meanBrightness']

    sig_filename = data_path + os.sep + sess + os.sep + sigfile
    print('Loading raw ROI data...')
    raw_sig_mat = np.genfromtxt( sig_filename, delimiter=',' )

    print('Calculating dF/F...')
    if sbx_version == 1:
        print('sbx version: 1')
        ROI_gcamp = raw_sig_mat[:, int((np.size(raw_sig_mat, 1) / 3) * 2):int(np.size(raw_sig_mat, 1))]
        PIL_gcamp = raw_sig_mat[:, int(np.size(raw_sig_mat, 1) / 3):int((np.size(raw_sig_mat, 1) / 3) * 2)]

    if sbx_version == 2:
        print('sbx version: 2')
        PIL_gcamp = raw_sig_mat[:, int(np.size(raw_sig_mat, 1) / 2):int(np.size(raw_sig_mat, 1))]
        ROI_gcamp = raw_sig_mat[:, (int(np.size(raw_sig_mat, 1) / np.size(raw_sig_mat, 1))-1):int(np.size(raw_sig_mat, 1) / 2)]

    # apply selected subtraction method
    if method == 2:
        mean_frame_brightness = np.mean(frame_brightness[0])
        dF_signal, f0_sig = dF_win((ROI_gcamp-PIL_gcamp)+mean_frame_brightness)
    else:
        dF_sig_dF, f0_sig = dF_win(ROI_gcamp)
        dF_pil_dF, f0_pil = dF_win(PIL_gcamp)
        dF_signal = dF_sig_dF - dF_pil_dF

    # get a list of all rois
    if len(rois) > 0:
        import_rois = np.array(rois, dtype='int16')
    else:
        import_rois = np.arange(0,dF_signal.shape[1],1)

    # extract only the desired rois
    dF_signal = np.copy(dF_signal[:,import_rois])

    sio.savemat(data_path + os.sep + sess + os.sep + 'sig_data' + sig_suffix + '.mat', mdict={'dF_data' : dF_signal})
    print('done...')

    return dF_signal

def process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eye_file=None, meta_file=None, sbx_version=2, session_crop=[0,1], method=2, rois=[]):

    if eye_file is not None:
        eye_data = sio.loadmat(data_path + os.sep + sess + os.sep + eye_file, appendmat=False )
    else:
        eye_data = None

    if sbx_version == 1:
        frame_brightness = np.genfromtxt( data_path + os.sep + sess + os.sep + meta_file )
        print('Brigthness data found...')
    elif sbx_version == 2:
        # read additional ROI metadata
        print('Brightness + metadata data found...')
        rec_info = sio.loadmat( data_path + os.sep + sess + os.sep + meta_file, appendmat=False)
        frame_brightness = rec_info['meanBrightness']

    # ipdb.set_trace()

    sig_filename = data_path + os.sep + sess + os.sep + sigfile
    print('Loading raw ROI data...')
    raw_sig_mat = np.genfromtxt( sig_filename, delimiter=',' )

    print('Calculating dF/F...')
    if sbx_version == 1:
        print('sbx version: 1')
        ROI_gcamp = raw_sig_mat[:, int((np.size(raw_sig_mat, 1) / 3) * 2):int(np.size(raw_sig_mat, 1))]
        PIL_gcamp = raw_sig_mat[:, int(np.size(raw_sig_mat, 1) / 3):int((np.size(raw_sig_mat, 1) / 3) * 2)]

    if sbx_version == 2:
        print('sbx version: 2')
        PIL_gcamp = raw_sig_mat[:, int(np.size(raw_sig_mat, 1) / 2):int(np.size(raw_sig_mat, 1))]
        ROI_gcamp = raw_sig_mat[:, (int(np.size(raw_sig_mat, 1) / np.size(raw_sig_mat, 1))-1):int(np.size(raw_sig_mat, 1) / 2)]

    # apply selected subtraction method
    if method == 2:
        mean_frame_brightness = np.mean(frame_brightness[0])
        dF_signal, f0_sig = dF_win((ROI_gcamp-PIL_gcamp)+mean_frame_brightness)
    else:
        dF_sig_dF, f0_sig = dF_win(ROI_gcamp)
        dF_pil_dF, f0_pil = dF_win(PIL_gcamp)
        dF_signal = dF_sig_dF - dF_pil_dF

    # get a list of all rois
    if len(rois) > 0:
        import_rois = np.array(rois, dtype='int16')
    else:
        import_rois = np.arange(0,dF_signal.shape[1],1)

    # extract only the desired rois
    dF_signal = np.copy(dF_signal[:,import_rois])
    ROI_gcamp = np.copy(ROI_gcamp[:,import_rois])
    PIL_gcamp = np.copy(PIL_gcamp[:,import_rois])

    print('Loading behavior data for alignment...')
    # behavior_file = data_path + os.sep + sess + os.sep + behavior_file
    # raw_data = load_data(behavior_file, 'vr')
    fname = data_path + os.sep + sess + os.sep + behavior_file
    raw_data = np.genfromtxt(fname, delimiter=';')

    if eye_data is not None:
        dF_aligned, behaviour_aligned, bri_aligned, eye_data_aligned, eye_x_aligned, eye_y_aligned = align_dF(raw_data, dF_signal, frame_brightness,[-1, -1], [-1, -1], True, True, session_crop, eye_data)
    else:
        dF_aligned, behaviour_aligned, bri_aligned = align_dF(raw_data, dF_signal, frame_brightness,[-1, -1], [6, -1], True, True, session_crop, eye_data)
        eye_data_aligned = np.empty(0)
        eye_x_aligned = np.empty(0)
        eye_y_aligned = np.empty(0)

    sio.savemat(data_path + os.sep + sess + os.sep + 'aligned_data.mat', mdict= \
                                        {'dF_aligned' : dF_aligned, \
                                        'behaviour_aligned' : behaviour_aligned, \
                                        'bri_aligned' : bri_aligned, \
                                        'eye_data_aligned' : eye_data_aligned, \
                                        'eye_x_aligned' : eye_x_aligned.T, \
                                        'eye_y_aligned' : eye_y_aligned.T, \
                                        'roiIDs' : rec_info['roiIDs'], \
                                        'roiCoordinates' : rec_info['roiCoordinates']})
    print('done...')

    return dF_aligned, behaviour_aligned, bri_aligned

def run_LF191022_3_20191119():
    MOUSE= 'LF191022_3'
    sess = '20191119'
    sigfile = 'M01_000_000.sig'
    meta_file = 'M01_000_000.extra'
    behavior_file = 'MTH3_vr1_s5r_20191119_1756.csv'
    eyefile = 'M01_000_000_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)


def run_LF191022_3_20191119_ol():
    MOUSE= 'LF191022_3'
    sess = '20191119_ol'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    behavior_file = 'MTH3_vr1_openloop_20191119_1828.csv'
    eyefile = 'M01_000_001_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_3_20191204():
    MOUSE= 'LF191022_3'
    sess = '20191204'
    sigfile = 'M01_000_008.sig'
    meta_file = 'M01_000_008.extra'
    behavior_file = 'MTH3_vr1_s5r2_2019124_2317.csv'
    eyefile = 'M01_000_008_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)
    # align_eyedata( data_path, sess, behavior_file, eyefile )

def run_LF191022_3_20191204_ol():
    MOUSE= 'LF191022_3'
    sess = '20191204_ol'
    sigfile = ''
    meta_file = ''
    behavior_file = 'MTH3_vr1_openloop_2019124_2351.csv'
    eyefile = 'M01_000_009_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    # process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)
    align_eyedata( data_path, sess, behavior_file, eyefile )

def run_LF191023_blue_20191119():
    MOUSE= 'LF191023_blue'
    sess = '20191119'
    sigfile = 'M01_000_004.sig'
    meta_file = 'M01_000_004.extra'
    behavior_file = 'MTH3_vr1_s5r_20191119_1857.csv'
    eyefile = 'M01_000_004_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191023_blue_20191204():
    MOUSE= 'LF191023_blue'
    sess = '20191204'
    sigfile = 'M01_000_000.sig'
    meta_file = 'M01_000_000.extra'
    behavior_file = 'MTH3_vr1_s5r2_2019124_1834.csv'
    eyefile = 'M01_000_000_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)
    # align_eyedata( data_path, sess, behavior_file, eyefile )

def run_LF191023_blue_20191204_ol():
    MOUSE= 'LF191023_blue'
    sess = '20191204_ol'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    behavior_file = 'MTH3_vr1_openloop_2019124_1919.csv'
    eyefile = 'M01_000_001_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    # process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)
    align_eyedata( data_path, sess, behavior_file, eyefile )

def run_LF191024_1_20191115():
    MOUSE= 'LF191024_1'
    sess = '20191115'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    behavior_file = 'MTH3_vr1_s5r_20191115_2115.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191024_1_20191204():
    MOUSE= 'LF191024_1'
    sess = '20191204'
    sigfile = 'M01_000_011.sig'
    meta_file = 'M01_000_011.extra'
    behavior_file = 'MTH3_vr1_s5r2_2019125_016.csv'
    eyefile = 'M01_000_011_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    # process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)
    align_eyedata( data_path, sess, behavior_file, eyefile )

def run_LF191024_1_20191204_ol():
    MOUSE= 'LF191024_1'
    sess = '20191204_ol'
    sigfile = 'M01_000_012.sig'
    meta_file = 'M01_000_012.extra'
    behavior_file = 'MTH3_vr1_openloop_2019125_058.csv'
    eyefile = 'M01_000_012_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    # process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)
    align_eyedata( data_path, sess, behavior_file, eyefile )

def run_LF191023_blue_20191119_ol():
    MOUSE= 'LF191023_blue'
    sess = '20191119_ol'
    sigfile = 'M01_000_005.sig'
    meta_file = 'M01_000_005.extra'
    behavior_file = 'MTH3_vr1_openloop_20191119_1929.csv'
    eyefile = 'M01_000_005_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191023_blank_20191116():
    MOUSE= 'LF191023_blank'
    sess = '20191116'
    sigfile = 'M01_000_002.sig'
    meta_file = 'M01_000_002.extra'
    behavior_file = 'MTH3_vr1_s5r_20191116_1919.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191023_blank_20191116_ol():
    MOUSE= 'LF191023_blank'
    sess = '20191116_ol'
    sigfile = 'M01_000_003.sig'
    meta_file = 'M01_000_003.extra'
    behavior_file = 'MTH3_vr1_openloop_20191116_204.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_1_20191115():
    MOUSE= 'LF191022_1'
    sess = '20191115'
    sigfile = 'M01_000_004.sig'
    meta_file = 'M01_000_004.extra'
    behavior_file = 'MTH3_vr1_s5r_20191115_2225.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_2_20191116():
    MOUSE= 'LF191022_2'
    sess = '20191116'
    sigfile = 'M01_000_000.sig'
    meta_file = 'M01_000_000.extra'
    behavior_file = 'MTH3_vr1_s5r_20191116_1815.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_2_20191116_ol():
    MOUSE= 'LF191022_2'
    sess = '20191116_ol'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    behavior_file = 'MTH3_vr1_openloop_20191116_1849.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_1_20191213():
    MOUSE= 'LF191022_1'
    sess = '20191213'
    sigfile = 'M01_000_004.sig'
    meta_file = 'M01_000_004.extra'
    behavior_file = 'MTH3_vr1_s5r2_20191213_2138.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_1_20191213_ol():
    MOUSE= 'LF191022_1'
    sess = '20191213_ol'
    sigfile = 'M01_000_005.sig'
    meta_file = 'M01_000_005.extra'
    behavior_file = 'MTH3_vr1_openloop_20191213_229.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_1_20191211():
    MOUSE= 'LF191022_1'
    sess = '20191211'
    sigfile = 'M01_000_000.sig'
    meta_file = 'M01_000_000.extra'
    behavior_file = 'MTH3_vr1_s5r2_20191211_170.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_1_20191211_ol():
    MOUSE= 'LF191022_1'
    sess = '20191211_ol'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    behavior_file = 'MTH3_vr1_openloop_20191211_1747.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_1_20191217():
    MOUSE= 'LF191022_1'
    sess = '20191217'
    sigfile = 'M01_000_000.sig'
    meta_file = 'M01_000_000.extra'
    behavior_file = 'MTH3_vr1_s5r2_20191217_1829.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191022_1_20191217_ol():
    MOUSE= 'LF191022_1'
    sess = '20191217_ol'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    behavior_file = 'MTH3_vr1_openloop_20191217_1931.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)


if __name__ == '__main__':

    # run_LF191022_1_20191115()


    # run_LF191022_3_20191119_ol()
#    run_LF191022_3_20191119()
    # run_LF191022_3_20191204()
    # run_LF191022_3_20191204_ol()
    # run_LF191023_blue_20191119()
    # run_LF191023_blue_20191204()
    # run_LF191023_blue_20191204_ol()

    # run_LF191024_1_20191115()
    # run_LF191024_1_20191204()
    # run_LF191024_1_20191204_ol()


    # run_LF191023_blue_20191119_ol()

    # run_LF191022_2_20191116()
    # run_LF191022_2_20191116_ol()
    # run_LF191023_blank_20191116()
    # run_LF191023_blank_20191116_ol()
    run_LF191022_1_20191217()
    run_LF191022_1_20191217_ol()
