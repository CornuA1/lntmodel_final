"""
Use raw data stored in hdf5 files to re-calculate dF.

Plan is to include different options to determine how calculate it but for not it will only
include options to just subtract frame brightness and the 'traditional' way of subtracting dF PIL from dF

"""

# Load native libraries
import sys
sys.path.append("./../General/Imaging")
sys.path.append("./Analysis")

# Load data-analysis relevant libraries
import numpy as np
import h5py
import tkinter
from tkinter import filedialog

# Load data-analysis function (these should all be on the Rochefort github repo
from import_dF import import_dF
#from align_dF_gratings import align_dF_gratings as align_dF
from align_dF_interp_mpi import align_dF
from dF_win_mpi import dF_win
import argparse
import scipy.io as sio
import os
import yaml

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

root = tkinter.Tk()
h5dat = None  # filehandle


def open_h5():
    """ open HDF-5 file. First look under 'DATAFILE', if nothing can be found there, open a File-open dialog"""
    try:
        # Use path defined at top of script. If file doesn't exist, open file
        # dialog ot ask for location
        h5dat = h5py.File(DATAFILE, 'r')
    except (IOError, NameError):  # Index error: if no argument was provided, just ask for hdf5 file
        h5file = filedialog.askopenfilenames(title='Please pick HDF5-FILE', filetypes = (("HDF5 files","*.h5"),("all files","*.*")))[0]
        root.update()
        if not h5file:
            raise NameError('Please select HDF5 file.')
        h5dat = h5py.File(h5file)
    finally:
        return h5dat


def write_h5(h5dat, day, dset, dset_name):
    """ Write dataset to HDF-5 file. Overwrite if it already exists. """
    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset(str(day) + '/' + dset_name,
                             data=dset, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat[str(day) + '/' + dset_name]
        h5dat.create_dataset(str(day) + '/' + dset_name,
                             data=dset, compression='gzip')


def open_sigfile():
    sigfile = filedialog.askopenfilenames(initialdir=r'F:', title='Please pick .sig file', filetypes = (("SIG files","*.sig"),("all files","*.*")))[0]
    root.update()
    if not sigfile:
        raise NameError('Please select .sig file.')
    return sigfile


def recalc_df(mouse, analyse_day, session_crop=[0,1], gcamp_idx_range=[-1,-1]):
    """ add .sig file by typing the day and selecting the HDF5 and .sig file """
    # reject sample points around trial transition
    reject_trial_transition = True

    # how to subtract brightness/neuropil
    subtraction_method = 2

    # print('Please select HDF5-File')
    # h5dat = open_h5()
    track = 3
    track_long = 4

    # flag to indicate which metadata for ROIs was provided. 1...'.bri'-file exists. 2...'.extra'file exists
    roi_metadata = -1

    h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
    h5dat = h5py.File(h5path, 'r')
    raw_behav_ds = np.copy(h5dat[analyse_day + '/raw_data'])
    raw_roi_f_ds = np.copy(h5dat[analyse_day + '/roi_raw'])
    raw_pil_f_ds = np.copy(h5dat[analyse_day + '/pil_raw'])
    brigthness_raw_f_ds = np.copy(h5dat[analyse_day + '/FOV_bri'])
    h5dat.close()



    # apply selected subtraction method
    if subtraction_method == 2:
        mean_frame_brightness = np.mean(brigthness_raw_f_ds)
        dF_signal, f0_sig = dF_win((raw_roi_f_ds-raw_pil_f_ds)+mean_frame_brightness)
    else:
        dF_sig_dF, f0_sig = dF_win(ROI_gcamp)
        dF_pil_dF, f0_pil = dF_win(PIL_gcamp)
        dF_signal = dF_sig_dF - dF_pil_dF
    #
    # # dF_signal = dF_sig_dF
    #
    # print('Loading behavior data for alignment...')
    # raw_behaviour = np.copy(h5dat[str(d) + '/raw_data'])
    # print('Aligning behaviour and imaging data...')
    #
    print(gcamp_idx_range)
    dF_aligned, behaviour_aligned, bri_aligned = align_dF(raw_behav_ds, dF_signal, brigthness_raw_f_ds,[-1, -1], gcamp_idx_range, True, reject_trial_transition, session_crop)
    print('Writing data to HDF5-file...')
    h5dat = h5py.File(h5path, 'a')
    write_h5(h5dat, analyse_day, dF_aligned, '/dF_win')
    write_h5(h5dat, analyse_day, dF_signal, '/dF_original')
    write_h5(h5dat, analyse_day, behaviour_aligned, '/behaviour_aligned')
    print('done')
    h5dat.flush()
    h5dat.close()
    # if np.size(frame_brightness) > 0:
    #     print('Writing brightness data...')
    #     write_h5(h5dat, d, frame_brightness.T, '/FOV_bri')
    #     write_h5(h5dat, d, bri_aligned.T, '/FOV_bri_aligned')
    #

if __name__ == "__main__":
    # mouse and session
    mouse = 'LF171211_2'
    analyse_day = 'Day201852_openloop'
    recalc_df(mouse, analyse_day)
    print('done')
