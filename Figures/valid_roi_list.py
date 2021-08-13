"""
list of valid and invalid rois based on manual curation.

"""

import os,yaml,sys,h5py,json
import numpy as np
import scipy.io as sio


with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

from write_dict import write_dict

def roi_validation_raw(dF_ds, json_path):
    """ run through dictionary and reject ROIs based on certain paramters """
    # load dataset
    # h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
    # h5dat = h5py.File(h5path, 'r')
    # behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    # dF_ds = np.copy(h5dat[sess + '/dF_win'])
    # h5dat.close()
    #
    # processed_data_path = loc_info['raw_dir'] + mouse + os.sep + sess + os.sep + 'aligned_data'
    # loaded_data = sio.loadmat(processed_data_path)
    # dF_ds = loaded_data['dF_aligned']

    inv_rois = []

    # load dictionary
    with open(json_path, 'r') as f:
        sess_dict = json.load(f)

    rois = np.arange(0,dF_ds.shape[1],1)
    for r in rois:
        if sess_dict['transient_rate'][r] < 0.5:
            inv_rois.append(r)

    return inv_rois

def roi_validation(mouse, sess, json_path):
    """ run through dictionary and reject ROIs based on certain paramters """
    # load dataset
    h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    inv_rois = []

    # load dictionary
    with open(json_path, 'r') as f:
        sess_dict = json.load(f)

    rois = np.arange(0,dF_ds.shape[1],1)
    for r in rois:
        if sess_dict['transient_rate'][r] < 0.5:
            inv_rois.append(r)

    return inv_rois

def run_LF180119_1_Day2018424_2():
    MOUSE = 'LF180119_1'
    SESSION = 'Day2018424_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = []
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_1_Day20170215_l23():
    MOUSE = 'LF170110_1'
    SESSION = 'Day20170215_l23'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [0,1,2,3,8,10,13,14,15,16,17,18,19,20,21,22,24,25,27,28,31,32,36]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_1_Day20170215_l5():
    MOUSE = 'LF170110_1'
    SESSION = 'Day20170215_l5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [0,2,3,4,5,7,8,9,15,20,23,43,44,62,68,69,71,76]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day20170209_l23():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170209_l23'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [4,6,7,8,10,13,14,20,21,22,23,24,26,29,32,34,35,36,38,39,40,41,43,45,49,50,53,58]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day20170209_l5():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170209_l5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [0,3,5,7,24,26,27,31,35,42,56,70,72,73,78]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF161202_1_Day20170209_l23():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l23'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [2,13,16,19,21,22,28,35,40,77,85,86,87,88,90,91]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF161202_1_Day20170209_l5():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [10,36,38,39,45,52,85,86,87,98,99,109,122,129,130,131,132]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [28,40,46,47,50,63,64,66,67,71,72,73,74,76,81,86,91,95,96,99,103]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170421_2_Day20170719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170719'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [5,13,33,36,38,48,69,70,76,78,86,101,106,111,119]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170421_2_Day2017720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017720'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [58,82,84,85,86,88,89,90,91,96,99,106]
    invalid_rois = np.union1d(invalid_rois,np.arange(64,tot_num_rois,1)).tolist()

    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])


    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day201748_1():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [1,2,3,4,5,15,16,22,23,24,25,31,34,36,39,40,44,48,50,52,54,55,56,57,62,65,66,68,69,70,71,72,73,75,76,79,81,82,83,86,87,88,90,91,93,94,96,87,100,104,107,108,110,114,116,118,119,120,124,125,126,127,131,133,135,136,137,139,140,141,142,143,144,145,146,147,148,149,150,151]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day201748_2():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [6,7,9,10,14,15,17,18,23,32,33,35,42,46,47,51,58,60,65,67,68,75,83,85,94,95,98,110,117,118,119,121,124,131,132,133,134,135,136,142,148,150,153,159,165,167,168,170]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170110_2_Day201748_3():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_3'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [0,1,2,3,4,22,29,30,31,32,35,36,37,45,46,47]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170420_1_Day2017719():
    MOUSE = 'LF170420_1'
    SESSION = 'Day2017719'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [61,65]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170420_1_Day201783():
    MOUSE = 'LF170420_1'
    SESSION = 'Day201783'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [0,10,13,14,71]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170222_1_Day201776():
    MOUSE = 'LF170222_1'
    SESSION = 'Day201776'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [48,80]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170222_1_Day2017615():
    MOUSE = 'LF170222_1'
    SESSION = 'Day2017615'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [27,58,60,69,71,72,73,88,93,101,102,103,108,116,118,119,124,130]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF171211_1_Day2018321_2():
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [1,2,9,10,11,19]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF171212_2_Day2018218_1():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_1'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [38,40,41,42,45,46,47,49,51,53,54,55,56,60,62,63,65,68,69,70,74,75,76,81,82,83,84,85,86,87,88,89,90]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF171212_2_Day2018218_2():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [12,14,17,29,32,35,36,38,44,46,47,48,49,57,79,98,99,100,101,107,116,125,127,129,130,132,133,134,136,137,139,140,143,147,\
                    153,154,155,160,161,164,165,166,168,169,172,173,179,181,185,186,197,204,206,210,211,212,213,214,215,216,217,218,219,223,224,225,\
                    226,227,228,229,230,231,236,237,238,243,244,246,247,248,249,250,252,254,256,258,259,260,261,262,263,264,266,267,269,270,271,\
                    274,276,279,280,281,282,283,284,286,288,289,292,293,294,295,298,303,304,305,306,307,308,310,311,318,319,320,321,322,323,325,\
                    326,329,330,332,334]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    print(valid_rois)

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170214_1_Day201777():
    MOUSE = 'LF170214_1'
    SESSION = 'Day201777'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # list of manually selected boutons which are representative of whole axons
    valid_rois = np.array([3,4,40,84,26,147,152,43,19,94,23,25,26,88,54,52,101,34,35,153,39,40,45,51,52,97,59,86,62,63,65,66,68,69,72,73,88,80,87,90,98,99,111,120,115,116,117,118,121,122,123,127,129,130,160,132,151,155])
    # print(np.sort(valid_rois))
    valid_rois = np.sort(np.unique(valid_rois))

    # manual roi rejection
    # invalid_rois = [0,1,2,3,4,6,8,10,15,17,18,19,22,24,25,26,30,32,33,34,36,37,38,39,41,42,44,48,49,56,57,58,59,63,69,74,79,80,95,96,97,98,100,106,110,117,118,119,126,130,133,134,137,139,146,147,148,150,155,157,158,159,160,161,162]
    invalid_rois = []
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF170214_1_Day2017714():
    MOUSE = 'LF170214_1'
    SESSION = 'Day2017714'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # list of manually selected boutons which are representative of whole axons
    valid_rois = np.array([92,109,98,32,39,124,40,25,94,35,125,115,126,48,50,51,54,63,64,83,95,102,104,130,134,118,121,127,131,132])
    # print(np.sort(valid_rois))
    valid_rois = np.sort(np.unique(valid_rois))
    # manual roi rejection
    invalid_rois = [3,4,5,6,7,8,9,10,12,13,14,15,20,22,24,25,31,34,35,36,37,38,43,44,45,50,54,56,58,59,60,66,67,68,75,76,77,78,79,80,81,82,84,89,90,91,97,100,101,103,107,108,109,110,111,112,113,114,116,117,118,127,129,130,131,132,133]
    invalid_rois = []
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF180112_2_Day2018424_1():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_1'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)


    # list of manually selected boutons which are representative of whole axons
    valid_rois = np.array([0,1,5,9,56,11,18,23,29,31,35,36,47,50,55,60,64,69,72])
    # print(np.sort(valid_rois))
    valid_rois = np.sort(np.unique(valid_rois))

    # manual roi rejection
    invalid_rois = [2,3,4,14,15,17,19,20,24,26,39,40,41,44,45,48,53,54,59,60,63,65,66,67,70,71,72]
    invalid_rois = []
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF180112_2_Day2018424_2():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_2'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # list of manually selected boutons which are representative of whole axons
    valid_rois = np.array([16,5,6,11,17,19,41])
    # print(np.sort(valid_rois))
    valid_rois = np.sort(np.unique(valid_rois))

    # manual roi rejection
    invalid_rois = [29,33,34,35,36,37,38,41,42]
    invalid_rois=[]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF180219_1_Day2018424_0025():
    MOUSE = 'LF180219_1'
    SESSION = 'Day2018424_0025'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    # valid_rois = np.arange(tot_num_rois)

    # list of manually selected boutons which are representative of whole axons
    valid_rois = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    # print(np.sort(valid_rois))
    # valid_rois = np.sort(np.unique(valid_rois))

    # manual roi rejection - because these are BOUTONs, this happens when selecting from grouped boutons
    # invalid_rois = [3,4,5,6,7,9,10,11,13,18,24,28,31,38,40,45,51,53,54,55,56,58,59,63,64,65]
    invalid_rois = []
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF171211_2_Day201852():
    MOUSE = 'LF171211_2'
    SESSION = 'Day201852'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    # valid_rois = np.arange(tot_num_rois)

    # list of manually selected boutons which are representative of whole axons
    valid_rois = np.array([24,44,10,12,39,15,47,17,18,34,36,49])
    # print(np.sort(valid_rois))
    valid_rois = np.sort(np.unique(valid_rois))

    # manual roi rejection - because these are BOUTONs, this happens when selecting from grouped boutons
    # invalid_rois = [3,4,5,6,7,9,10,11,13,18,24,28,31,38,40,45,51,53,54,55,56,58,59,63,64,65]
    invalid_rois = []
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF171211_2_Day201852_matched():
    MOUSE = 'LF171211_2'
    SESSION = 'Day201852_matched'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    # load imaging dataset here just to count total number of rois
    h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    h5dat = h5py.File(h5path, 'r')
    dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
    h5dat.close()
    tot_num_rois = dF_ds.shape[1]
    # valid_rois = np.arange(tot_num_rois)

    # list of manually selected boutons which are representative of whole axons
    valid_rois = np.array([0,1,2,3,4,5,6,7,8,9,10,11])


    # manual roi rejection - because these are BOUTONs, this happens when selecting from grouped boutons
    # invalid_rois = [3,4,5,6,7,9,10,11,13,18,24,28,31,38,40,45,51,53,54,55,56,58,59,63,64,65]
    invalid_rois = []
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation(MOUSE, SESSION, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_3_Day20191119():
    MOUSE = 'LF191022_3'
    SESSION = '20191119'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    # invalid_rois = [4,33,47,48,58]
    invalid_rois = []
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_3_Day20191204():
    MOUSE = 'LF191022_3'
    SESSION = '20191204'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    # invalid_rois = [4,33,47,48,58]
    invalid_rois = []
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191023_blue_Day20191119():
    MOUSE = 'LF191023_blue'
    SESSION = '20191119'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    # invalid_rois = [37,38,51,52,66,87]
    invalid_rois = []
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191023_blue_Day20191204():
    MOUSE = 'LF191023_blue'
    SESSION = '20191204'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = []
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191024_1_Day20191115():
    MOUSE = 'LF191024_1'
    SESSION = '20191115'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = []
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191024_1_Day20191204():
    MOUSE = 'LF191024_1'
    SESSION = '20191204'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = []
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    # auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    # for ir in auto_detected_invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_1_Day20191115():
    MOUSE = 'LF191022_1'
    SESSION = '20191207_ol'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [0,11,12,18,39,45]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_2_Day20191116():
    MOUSE = 'LF191022_2'
    SESSION = '20191116'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [1,18,21,47,76,98,116]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_1_Day20191211():
    MOUSE = 'LF191022_1'
    SESSION = '20191211'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [53,71]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)
    
def run_LF191022_1_Day20191209():
    MOUSE = 'LF191022_1'
    SESSION = '20191209'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [21,34,38,56,57,66,74,79,82]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_1_Day20191213():
    MOUSE = 'LF191022_1'
    SESSION = '20191213'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [1,13,22,25,31,49,87,97]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_1_Day20191215():
    MOUSE = 'LF191022_1'
    SESSION = '20191215'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [34,35,51,53,68,76]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191022_1_Day20191217():
    MOUSE = 'LF191022_1'
    SESSION = '20191217'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [27,57,58,60,65,66,67,68,74,76,84,88,92,94,95,104,107,114,115,116,119,122]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)


def run_LF191023_blank_Day20191116():
    MOUSE = 'LF191023_blank'
    SESSION = '20191116'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [0,1,6,7,10,14,15,18,34,39,40,41,49,50,51,76,79,81,88,89,95,108,121]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191023_blue_Day20191208():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [2,10,11,12,14,15,16,18,19,21,23,27,28,30,32,33,34,36,38,44,57,58,60,61,62,63,64,67,71,74,75,76,77,78]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)

def run_LF191023_blue_Day20191210():
    MOUSE = 'LF191023_blue'
    SESSION = '20191210'

    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    dF_ds = loaded_data['dF_aligned']

    tot_num_rois = dF_ds.shape[1]
    valid_rois = np.arange(tot_num_rois)

    # manual roi rejection
    invalid_rois = [1,2,18,24,37,38,39,43,47,49,55]
    for ir in invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    auto_detected_invalid_rois = roi_validation_raw(dF_ds, json_path)
    for ir in auto_detected_invalid_rois:
        valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])

    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }

    write_dict(MOUSE, SESSION, roi_result_params)


if __name__ == '__main__':

#    run_LF191022_3_Day20191119()
#    run_LF191022_3_Day20191204()
#    run_LF191023_blue_Day20191119()
#    run_LF191023_blue_Day20191204()
#    run_LF191024_1_Day20191115()
#    run_LF191024_1_Day20191204()
     # run_LF191022_1_Day20191115()
#     run_LF191022_1_Day20191217()
      run_LF191023_blue_Day20191210()
    # run_LF191022_2_Day20191116()
    # run_LF191023_blank_Day20191116()

    # TO RUN (delete current .json, re-run entire pipeline)
    # run_LF170613_1_Day20170804()
    # run_LF170421_2_Day20170719()
    # run_LF170421_2_Day2017720()
    # run_LF170110_2_Day201748_1()
    # run_LF170110_2_Day201748_2()
    # run_LF170110_2_Day201748_3()
    # run_LF170420_1_Day2017719()
    # run_LF170420_1_Day201783()
    # run_LF170222_1_Day201776()
    # run_LF171211_1_Day2018321_2()
    # run_LF171212_2_Day2018218_1()
    # run_LF171212_2_Day2018218_2()
    # run_LF170222_1_Day2017615()
    # run_LF161202_1_Day20170209_l23()
    # run_LF161202_1_Day20170209_l5()

    # run_LF180119_1_Day2018424_2()

    # run_LF170214_1_Day201777()
    # run_LF170214_1_Day2017714()
    # run_LF171211_2_Day201852()
    # run_LF180219_1_Day2018424_0025()
    # run_LF180112_2_Day2018424_1()
    # run_LF180112_2_Day2018424_2()

    # run_LF171211_2_Day201852_matched()

    # run_LF170110_2_Day20170209_l23()
    # run_LF170110_2_Day20170209_l5()
    #
    # run_LF170110_1_Day20170215_l23()
    # run_LF170110_1_Day20170215_l5()

    # tot_num_rois = 123
    # invalid_rois = [5,13,33,36,38,48,69,70,76,78,86,101,106,111,119]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    # SESSION = 'Day2017720'
    # tot_num_rois = 107
    # invalid_rois = [58,82,84,85,86,88,89,90,91,96,99,106]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_1'
    # tot_num_rois = 152
    # invalid_rois = [1,2,3,4,5,15,16,22,23,24,25,31,34,36,39,40,44,48,50,52,54,55,56,57,62,65,66,68,69,70,71,72,73,75,76,79,81,82,83,86,87,88,90,91,93,94,96,87,100,104,107,108,110,114,116,118,119,120,124,125,126,127,131,133,135,136,137,139,140,141,142,143,144,145,146,147,148,149,150,151]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # SESSION = 'Day201748_2'
    # tot_num_rois = 171
    # invalid_rois = [6,7,9,10,14,15,,17,18,23,32,33,35,42,46,47,51,58,60,65,67,68,75,83,85,94,95,98,110,117,118,119,121,124,131,132,133,134,135,136,142,148,150,153,159,165,167,168,170]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # SESSION = 'Day201748_3'
    # tot_num_rois = 50
    # invalid_rois = [0,1,2,3,4,22,29,30,31,32,35,36,37,45,46,47]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #


    # MOUSE = 'LF170420_1'
    # SESSION = 'Day2017719'
    # tot_num_rois = 91
    # invalid_rois = [61,65]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # SESSION = 'Day2017719'
    # tot_num_rois = 81
    # invalid_rois = [0,10,13,14,71]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # MOUSE = 'LF171211_1'
    # SESSION = 'Day2018321_2'
    # tot_num_rois = 91
    # invalid_rois = [1,2,9,10,11,19]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # MOUSE = 'LF170222_1'
    # SESSION = 'Day201876'
    # tot_num_rois = 120
    # invalid_rois = [48,80]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # MOUSE = 'LF171212_2'
    # SESSION = 'Day2018182_1'
    # tot_num_rois = 91
    # invalid_rois = [38,40,41,42,45,46,47,49,51,53,54,55,56,60,62,63,65,68,69,70,74,75,76,81,82,83,84,85,86,87,88,89,90]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # SESSION = 'Day2018182_2'
    # tot_num_rois = 336
    # invalid_rois = [12,14,17,29,32,35,36,38,44,46,47,48,49,57,79,98,99,100,101,107,116,125,127,129,130,132,133,134,136,137,139,140,143,147,\
    #                 153,154,155,160,161,164,165,166,168,169,172,173,179,181,185,186,197,204,206,210,211,212,213,214,215,216,217,218,219,223,224,225,\
    #                 226,227,228,229,230,231,236,237,238,243,244,246,247,248,249,250,252,254,256,258,259,260,261,262,263,264,266,267,269,270,271,\
    #                 274,276,279,280,281,282,283,284,286,288,289,292,293,294,295,298,303,304,305,306,307,308,310,311,318,319,320,321,322,323,325,\
    #                 326,329,330,332,334]
    # valid_rois = np.arange(tot_num_rois)
    # for ir in invalid_rois:
    #     valid_rois = np.delete(valid_rois, np.where(valid_rois == ir)[0])
    #
    # roi_result_params = {
    #     'valid_rois' : valid_rois.item(),
    #     'invalid_rois' : invalid_rois.item()
    # }
    # write_dict(MOUSE, SESSION, roi_result_params)
