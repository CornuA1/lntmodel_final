"""
Align eye and behavior data


@author: Lukas Fischer


"""


import sys, yaml, os
with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + "Analysis")
sys.path.append(loc_info['base_dir'] + "Imaging")
import warnings; warnings.simplefilter('ignore')

import numpy as np
import scipy.io as sio
from scipy import signal
import ipdb
from dF_win_mpi import dF_win
from align_dF_interp_mpi import align_dF
from load_behavior_data import load_data

def align_eye_behavior_data(eye_file, behavior_file, path):
    eye_data = sio.loadmat(path + eye_file )
    behavior_data = load_data(path + behavior_file, 'vr')

    num_ts_behaviour = np.size(behavior_data[:,0])
    eye_data_aligned = signal.resample(eye_data['pupil_area_timecourse'], num_ts_behaviour, axis=0)
    eye_x_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,0], num_ts_behaviour, axis=0)
    eye_y_aligned = signal.resample(eye_data['pupil_center_timecourse'][:,1], num_ts_behaviour, axis=0)

    sio.savemat(path + 'Pupil_aligned.mat', mdict={'pupil_timecourse' : eye_data_aligned, 'eye_x_aligned' : eye_x_aligned, 'eye_y_aligned' : eye_y_aligned})
    sio.savemat(path + 'Behavior.mat', mdict={'behaviora_aligned' : behavior_data})


def run_LF191023_blue_20191119():
    MOUSE= 'LF191023_blue'
    sess = '20191119'
    path = 'E:\\MTH3_data\\MTH3_data\\animals_raw\\LF191023_blue\\20191119\\'
    eyefile = 'M01_000_004_eye_analyzed.mat'
    behavior_file = 'MTH3_vr1_s5r_20191119_1857.csv'

    data_path = loc_info['raw_dir'] + MOUSE
    #calc_dF(data_path, sess, sigfile , meta_file, 2, [0,1], 2, [], sig_suffix)
    align_eye_behavior_data(eyefile, behavior_file, path)
    # sio.savemat('E:\\MTH3_data\\MTH3_data\\animals_raw\\LF191023_blue\\20191119\\Pupil_aligned.mat', mdict={'pupil_timecourse' : eye_data_aligned, 'eye_x_aligned' : eye_x_aligned, 'eye_y_aligned' : eye_y_aligned})
    # sio.savemat('E:\\MTH3_data\\MTH3_data\\animals_raw\\LF191023_blue\\20191119\\Behavior.mat', mdict={'behaviora_aligned' : behavior_data})

def run_LF191022_3_20191119():
    MOUSE= 'LF191022_3'
    sess = '20191119'
    path = 'E:\\MTH3_data\\MTH3_data\\animals_raw\\LF191022_3\\20191119\\'
    eyefile = 'M01_000_000_eye_analyzed.mat'
    behavior_file = 'MTH3_vr1_s5r_20191119_1756.csv'

    data_path = loc_info['raw_dir'] + MOUSE
    align_eye_behavior_data(eyefile, behavior_file, path)



if __name__ == '__main__':

    # run_LF191023_blue_20191119()
    run_LF191022_3_20191119()
