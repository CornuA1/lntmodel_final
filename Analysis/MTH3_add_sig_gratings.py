"""
Add .sig file to gratings dataset.

F0 values are collected but not written

"""

# Load native libraries
import sys
sys.path.append("../../General/Imaging")

# Load data-analysis relevant libraries
import numpy as np
import h5py
import tkinter
from tkinter import filedialog

# Load data-analysis function (these should all be on the Rochefort github repo
from import_dF import import_dF
from align_dF_gratings_mpi import align_dF_gratings as align_dF
#from align_dF_interp import align_dF
from dF_win_mpi import dF_win
import argparse

import scipy.io as sio

root = tkinter.Tk()
h5dat = None  # filehandle

def open_h5():
    """ open HDF-5 file. First look under 'DATAFILE', if nothing can be found there, open a File-open dialog"""
    try:
        # Use path defined at top of script. If file doesn't exist, open file
        # dialog ot ask for location
        h5dat = h5py.File(DATAFILE, 'r')
    except (IOError, NameError):  # Index error: if no argument was provided, just ask for hdf5 file
        h5file = filedialog.askopenfilenames(initialdir=r'C:\Users\The_mothership\Google Drive\MTH3_data\animals_h5', title='Please pick HDF5-FILE', filetypes = (("HDF5 files","*.h5"),("all files","*.*")))[0]
        root.update()
        if not h5file:
            raise NameError('Please select HDF5 file.')
        h5dat = h5py.File(h5file)
    finally:
        return h5dat


def write_h5(h5dat, day, dset, dset_name):
    """ Write dataset to HDF-5 file. Overwrite if it already exists. """
    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset(str(d) + '/' + dset_name,
                             data=dset, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat[str(d) + '/' + dset_name]
        h5dat.create_dataset(str(d) + '/' + dset_name,
                             data=dset, compression='gzip')


def open_sigfile():
    sigfile = filedialog.askopenfilenames(initialdir=r'F:', title='Please pick .sig file', filetypes = (("SIG files","*.sig"),("all files","*.*")))[0]
    root.update()
    if not sigfile:
        raise NameError('Please select .sig file.')
    return sigfile


if __name__ == "__main__":
    """ add .sig file by typing the day and selecting the HDF5 and .sig file """

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--day", default="NULL", metavar="MODE", dest="day",
                      help="day for which to add the imaging data. Has to be same as corresponding ")
    parser.add_argument("-os", "--offset", type=int, default="10", metavar="MODE", dest="offset",
                      help="frames of the behavior file skipped in beginning to align starts of imaging and behavior recording ")
    args = parser.parse_args()
    d = [args.day][0]
    recording_offset = [args.offset][0]

    print('Please select HDF5-File')
    h5dat = open_h5()
    track = 3
    track_long = 4

    # if no days are selected, go through all datasets in HDF5 file
    # if args.day == 'all':
    #     analyse_days = [day for day in h5dat]

    print('days: ' + str(d))
    print('offset: ' + str(args.offset))

    # for d in analyse_days:
    print('Please select .sig file')
    sigfilename = open_sigfile()
    print('Opening ROI metadata file...')
    fname = sigfilename.split('.')[0]
    try:
        frame_brightness = np.genfromtxt( fname+'.bri')
        print('Brigthness data found...')
    except OSError:
        try:
            # read additional ROI metadata
            print('Brigthness + metadata data found...')
            rec_info = sio.loadmat( fname+'.extra')
            frame_brightness = rec_info['meanBrightness']
            roi_size = rec_info['roiSizes']
            roi_coords = rec_info['roiCoordinates']
            if 'version' in rec_info:
                sbx_version = rec_info['version']
            else:
                sbx_version = 1
        except OSError:
            frame_brightness = []
            print('WARNING: no brightness or .extra file found for '+ sigfilename.split('/')[-1])
    gcamp_raw = import_dF(sigfilename)
    print('Calculating dF/F...')

    if sbx_version == 1:
        ROI_gcamp = gcamp_raw[:, int(np.size(gcamp_raw, 1) / 3 * 2):int(np.size(gcamp_raw, 1))]
        PIL_gcamp = gcamp_raw[:, int(np.size(gcamp_raw, 1) / 3):int((np.size(gcamp_raw, 1) / 3) * 2)]

    if sbx_version == 2:
        ROI_gcamp = gcamp_raw[:, int(np.size(gcamp_raw, 1) / 2):int(np.size(gcamp_raw, 1))]
        PIL_gcamp = gcamp_raw[:, (int(np.size(gcamp_raw, 1) / np.size(gcamp_raw, 1))-1):int(np.size(gcamp_raw, 1) / 2)]

    dF_sig_dF,f0_sig = dF_win(ROI_gcamp)
    dF_pil_dF,f0_pil = dF_win(PIL_gcamp)
    dF_signal = dF_sig_dF - dF_pil_dF

    # dF_signal = gcamp_raw[:, 0 : np.size(gcamp_raw, 1) / 3]
    # dF_signal,f0_sig = dF_win(dF_signal)

    reject_trial_transition = True
    session_crop = [0,1]

    print('Aligning behaviour and imaging data...')
    print(d)
    raw_behaviour = np.copy(h5dat[str(d) + '/raw_data'])
    dF_aligned, behaviour_aligned, bri_aligned = align_dF(raw_behaviour, dF_signal, frame_brightness,[recording_offset, -1], [-1, -1], True, reject_trial_transition, session_crop)
    print('Writing data to HDF5-File...')
    write_h5(h5dat, d, ROI_gcamp, '/roi_raw')
    write_h5(h5dat, d, PIL_gcamp, '/pil_raw')
    write_h5(h5dat, d, dF_signal, '/dF_win_unaligned')
    write_h5(h5dat, d, dF_sig_dF, '/dF_win_sig_unaligned')
    write_h5(h5dat, d, dF_pil_dF, '/dF_win_pil_unaligned')
    write_h5(h5dat, d, dF_aligned, '/dF_win')
    write_h5(h5dat, d, behaviour_aligned, '/behaviour_aligned')
    if np.size(frame_brightness) > 0:
        print('Writing brightness data...')
        write_h5(h5dat, d, frame_brightness, '/FOV_bri')
        write_h5(h5dat, d, bri_aligned, '/FOV_bri_aligned')

    print('done')
    h5dat.flush()
    h5dat.close()
