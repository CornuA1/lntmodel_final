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
    behavior_file = data_path + os.sep + sess + os.sep + behavior_file
    raw_data = load_data(behavior_file, 'vr')
#    raw_data = np.genfromtxt(fname, delimiter=';')
	
    if eye_data is not None:
        dF_aligned, behaviour_aligned, bri_aligned, eye_data_aligned, eye_x_aligned, eye_y_aligned = align_dF(raw_data, dF_signal, frame_brightness,[-1, -1], [-1, -1], True, True, session_crop, eye_data)
    else:
        dF_aligned, behaviour_aligned, bri_aligned = align_dF(raw_data, dF_signal, frame_brightness,[-1, -1], [-1, -1], True, True, session_crop, eye_data)
        eye_data_aligned = np.empty(0)
        eye_x_aligned = np.empty(0)
        eye_y_aligned = np.empty(0)

    sio.savemat(data_path + os.sep + sess + os.sep + 'aligned_data.mat', mdict= \
                                        {'dF_aligned' : dF_aligned, \
                                        'behaviour_aligned' : behaviour_aligned, \
                                        'bri_aligned' : bri_aligned, \
                                        'eye_data_aligned' : eye_data_aligned, \
                                        'eye_x_aligned' : eye_x_aligned.T, \
                                        'eye_y_aligned' : eye_y_aligned.T })
    print('done...')

    return dF_aligned, behaviour_aligned, bri_aligned

def run_LF170613_1():
    MOUSE= 'LF170613_1'
    sess = '20170804'
    sigfile = 'M01_000_004_rigid.sig'
    meta_file = 'M01_000_004_rigid.bri'
    behavior_file = 'MTH3_vr1_20170804_1708.csv'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, meta_file, sbx_version=1, session_crop=[0,1], method=2)

def run_LF190409_1():
    MOUSE= 'LF190409_1'
    sess = '190514_1'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, sbx_version=2, session_crop=[0,1], method=2, rois=[])

def run_Jimmy():
    MOUSE= 'Jimmy'

#    sess = '20190719_000'
#    sigfile = 'M01_000_000.sig'
#    meta_file = 'M01_000_000.extra'
#    data_path = loc_info['raw_dir'] + MOUSE
#    calc_dF(data_path, sess, sigfile , meta_file, sbx_version=2, session_crop=[0,1], method=2, rois=[])

    sess = '20190719_004'
    sigfile = 'M01_000_004.sig'
    meta_file = 'M01_000_004.extra'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, sbx_version=2, session_crop=[0,1], method=2, rois=[])

def run_LF190716_1():
    MOUSE= 'LF190716_1'
    sess = '20190722_002'
    sigfile = 'M01_000_002.sig'
    meta_file = 'M01_000_002.extra'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, sbx_version=2, session_crop=[0,1], method=2, rois=[])

def run_Buddha_190816_41():
    MOUSE= 'Buddha'
    sess = '190816_41'
    sigfile = 'Buddha_000_041_rigid.sig'
    meta_file = 'Buddha_000_041_rigid.extra'
    sig_suffix = '01'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_042_rigid.sig'
    meta_file = 'Buddha_000_042_rigid.extra'
    sig_suffix = '02'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_043_rigid.sig'
    meta_file = 'Buddha_000_043_rigid.extra'
    sig_suffix = '03'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_044_rigid.sig'
    meta_file = 'Buddha_000_044_rigid.extra'
    sig_suffix = '04'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_045_rigid.sig'
    meta_file = 'Buddha_000_045_rigid.extra'
    sig_suffix = '05'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_046_rigid.sig'
    meta_file = 'Buddha_000_046_rigid.extra'
    sig_suffix = '06'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_047_rigid.sig'
    meta_file = 'Buddha_000_047_rigid.extra'
    sig_suffix = '07'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_048_rigid.sig'
    meta_file = 'Buddha_000_048_rigid.extra'
    sig_suffix = '08'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_049_rigid.sig'
    meta_file = 'Buddha_000_049_rigid.extra'
    sig_suffix = '09'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_050_rigid.sig'
    meta_file = 'Buddha_000_050_rigid.extra'
    sig_suffix = '10'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

def run_Buddha_190816_51():
    MOUSE= 'Buddha'
    sess = '190816_51'
    sigfile = 'Buddha_000_051_rigid.sig'
    meta_file = 'Buddha_000_051_rigid.extra'
    sig_suffix = '01'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_052_rigid.sig'
    meta_file = 'Buddha_000_052_rigid.extra'
    sig_suffix = '02'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_053_rigid.sig'
    meta_file = 'Buddha_000_053_rigid.extra'
    sig_suffix = '03'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_054_rigid.sig'
    meta_file = 'Buddha_000_054_rigid.extra'
    sig_suffix = '04'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_055_rigid.sig'
    meta_file = 'Buddha_000_055_rigid.extra'
    sig_suffix = '05'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_056_rigid.sig'
    meta_file = 'Buddha_000_056_rigid.extra'
    sig_suffix = '06'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_057_rigid.sig'
    meta_file = 'Buddha_000_057_rigid.extra'
    sig_suffix = '07'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_058_rigid.sig'
    meta_file = 'Buddha_000_058_rigid.extra'
    sig_suffix = '08'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_059_rigid.sig'
    meta_file = 'Buddha_000_059_rigid.extra'
    sig_suffix = '09'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_060_rigid.sig'
    meta_file = 'Buddha_000_060_rigid.extra'
    sig_suffix = '10'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

def run_Buddha_190816_61():
    MOUSE= 'Buddha'
    sess = '190816_61'
    sigfile = 'Buddha_000_061_rigid.sig'
    meta_file = 'Buddha_000_061_rigid.extra'
    sig_suffix = '01'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_062_rigid.sig'
    meta_file = 'Buddha_000_062_rigid.extra'
    sig_suffix = '02'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_063_rigid.sig'
    meta_file = 'Buddha_000_063_rigid.extra'
    sig_suffix = '03'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_064_rigid.sig'
    meta_file = 'Buddha_000_064_rigid.extra'
    sig_suffix = '04'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_065_rigid.sig'
    meta_file = 'Buddha_000_065_rigid.extra'
    sig_suffix = '05'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_066_rigid.sig'
    meta_file = 'Buddha_000_066_rigid.extra'
    sig_suffix = '06'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_067_rigid.sig'
    meta_file = 'Buddha_000_067_rigid.extra'
    sig_suffix = '07'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_068_rigid.sig'
    meta_file = 'Buddha_000_068_rigid.extra'
    sig_suffix = '08'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_069_rigid.sig'
    meta_file = 'Buddha_000_069_rigid.extra'
    sig_suffix = '09'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

    sigfile = 'Buddha_000_070_rigid.sig'
    meta_file = 'Buddha_000_070_rigid.extra'
    sig_suffix = '10'
    data_path = loc_info['raw_dir'] + MOUSE
    calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)

def run_LF191022_3_20191119():
    MOUSE= 'LF191022_3'
    sess = '20191119'
    sigfile = 'M01_000_000.sig'
    meta_file = 'M01_000_000.extra'
    behavior_file = 'MTH3_vr1_s5r_20191119_1756.csv'
    eyefile = 'M01_000_000_eye_analyzed.mat'
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191023_blue_20191119():
    MOUSE= 'LF191023_blue'
    sess = '20191119'
    sigfile = 'M01_000_004.sig'
    meta_file = 'M01_000_004.extra'
    behavior_file = 'MTH3_vr1_s5r_20191119_1857.csv'
    eyefile = 'M01_000_004_eye_analyzed.mat'
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
    
    
def run_LF191023_blue_20191210():
    MOUSE= 'LF191023_blue'
    sess = '20191210'
    sigfile = 'M01_000_006.sig'
    meta_file = 'M01_000_006.extra'
    behavior_file = 'MTH3_vr1_s5r2_20191210_2224.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)

def run_LF191023_blue_20191210_ol():
    MOUSE= 'LF191023_blue'
    sess = '20191210_ol'
    sigfile = 'M01_000_007.sig'
    meta_file = 'M01_000_007.extra'
    behavior_file = 'MTH3_vr1_openloop_20191210_232.csv'
    eyefile = None
    data_path = loc_info['raw_dir'] + MOUSE
    process_and_align_sigfile(data_path, sess, sigfile, behavior_file, eyefile, meta_file, sbx_version=2, session_crop=[0,1], method=2)


if __name__ == '__main__':
#    run_LF170613_1()
#    run_LF190409_1()
#    run_Jimmy()
#    run_LF190716_1()
#    run_Buddha_190816_41()
#    run_Buddha_190816_51()
    # run_Buddha_190816_61()
    # run_LF191022_3_20191119()
    # run_LF191023_blue_20191119()
#    run_LF191022_1_20191115()
#    run_LF191022_2_20191116()
    run_LF191023_blue_20191210()
    run_LF191023_blue_20191210_ol()
