"""
Add .sig file to regular session, openloop or dark running

"""

# Load native libraries
import sys
sys.path.append("./../General/Imaging")

# Load data-analysis relevant libraries
import numpy as np
import h5py
import tkinter
from tkinter import filedialog
import warnings; warnings.simplefilter('ignore')

# Load data-analysis function (these should all be on the Rochefort github repo
from import_dF import import_dF
#from align_dF_gratings import align_dF_gratings as align_dF
from align_dF_interp_mpi import align_dF
from dF_win_mpi import dF_win
import argparse
import scipy.io as sio


root = tkinter.Tk()
h5dat = None  # filehandle


def select_h5():
    """ prompt user to select file. Return filename and path """
    try:
        h5file = filedialog.askopenfilenames(initialdir=r'C:\Users\The_mothership\Google Drive\MTH3_data\animals_h5', title='Please pick HDF5-FILE', filetypes = (("HDF5 files","*.h5"),("all files","*.*")))[0]
        root.update()
        return h5file
    except IndexError:
        print('No HDF5 file selected. Terminating.')
        return None

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
    """ Prompt user to select .sig file """
    sigfile = filedialog.askopenfilenames(initialdir=r'F:', title='Please pick .sig file', filetypes = (("SIG files","*.sig"),("all files","*.*")))[0]
    root.update()
    if not sigfile:
        raise NameError('Please select .sig file.')
    return sigfile

def add_sigfile(args):
    """ add sigfile to session. Align to behavior data if it exists. """
    # read list of rois (if provided)
    import_rois = args.rois
    if import_rois is not -1:
        import_rois = list(map(int, import_rois))
        import_rois = np.array(import_rois)

    analyse_day = args.day
    if analyse_day is None:
        print('ERROR: please provide session identifier')
        return
    else:
        analyse_day = analyse_day[0]

    session_crop = args.crop
    subtraction_method = args.method
    if session_crop is None:
        session_crop = [0,1]
    else:
        session_crop = [float(i) for i in session_crop]

    print('subtraction method: ' + str(subtraction_method))

    if analyse_day.find('dark') != -1:
        print('dark session detected')
        reject_trial_transition = False
    else:
        reject_trial_transition = True

    print('Please select HDF5-File')
    h5file = select_h5()
    if not h5file:
        return

    # flag to indicate which metadata for ROIs was provided. 1...'.bri'-file exists. 2...'.extra'file exists
    roi_metadata = -1
    # for d in analyse_day:
    print('Please select .sig file')
    sigfilename = open_sigfile()
    print('Opening ROI metadata file...')
    fname = sigfilename.split('.')[0]
    try:
        frame_brightness = np.genfromtxt( fname+'.bri')
        sbx_version = 1
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
    print('Loading raw imaging data...')
    gcamp_raw = import_dF(sigfilename)

    print('Calculating dF/F...')
    if sbx_version == 1:
        print('sbx version: 1')
        ROI_gcamp = gcamp_raw[:, int((np.size(gcamp_raw, 1) / 3) * 2):int(np.size(gcamp_raw, 1))]
        PIL_gcamp = gcamp_raw[:, int(np.size(gcamp_raw, 1) / 3):int((np.size(gcamp_raw, 1) / 3) * 2)]

    if sbx_version == 2:
        print('sbx version: 2')
        PIL_gcamp = gcamp_raw[:, int(np.size(gcamp_raw, 1) / 2):int(np.size(gcamp_raw, 1))]
        ROI_gcamp = gcamp_raw[:, (int(np.size(gcamp_raw, 1) / np.size(gcamp_raw, 1))-1):int(np.size(gcamp_raw, 1) / 2)]

    # apply selected subtraction method
    if subtraction_method == 2:
        mean_frame_brightness = np.mean(frame_brightness[0])
        dF_signal, f0_sig = dF_win((ROI_gcamp-PIL_gcamp)+mean_frame_brightness)
    else:
        dF_sig_dF, f0_sig = dF_win(ROI_gcamp)
        dF_pil_dF, f0_pil = dF_win(PIL_gcamp)
        dF_signal = dF_sig_dF - dF_pil_dF

    # get a list of all rois
    all_rois = np.arange(0,dF_signal.shape[1],1)
    if import_rois is -1:
        import_rois = all_rois

    # only select desired ROIs
    print('selecting rois...')
    dF_signal_all = np.copy(dF_signal)
    ROI_gcamp_all = np.copy(ROI_gcamp)
    PIL_gcamp_all = np.copy(PIL_gcamp)

    dF_signal = dF_signal[:,import_rois]
    ROI_gcamp = ROI_gcamp[:,import_rois]
    PIL_gcamp = PIL_gcamp[:,import_rois]

    print(h5file)
    print(analyse_day)
    h5dat = h5py.File(h5file, 'r')
    # if behavior dataset exists, align. Otherwise just write dF data
    if analyse_day in h5dat:
        print('Loading behavior data for alignment...')
        print(str(analyse_day))
        raw_behaviour = np.copy(h5dat[analyse_day + '/raw_data'])
        h5dat.close()
        print('Aligning behaviour and imaging data...')
        dF_aligned, behaviour_aligned, bri_aligned = align_dF(raw_behaviour, dF_signal, frame_brightness,[-1, -1], [-1, -1], True, reject_trial_transition, session_crop)
        print('Writing data to HDF5-file...')
        h5dat = h5py.File(h5file, 'r+')
        write_h5(h5dat, analyse_day, ROI_gcamp, '/roi_raw')
        write_h5(h5dat, analyse_day, PIL_gcamp, '/pil_raw')
        write_h5(h5dat, analyse_day, dF_aligned, '/dF_win')
        write_h5(h5dat, analyse_day, dF_signal, '/dF_original')
        write_h5(h5dat, analyse_day, behaviour_aligned, '/behaviour_aligned')
        if np.size(frame_brightness) > 0:
            print('Writing brightness data...')
            write_h5(h5dat, analyse_day, frame_brightness.T, '/FOV_bri')
            write_h5(h5dat, analyse_day, bri_aligned.T, '/FOV_bri_aligned')
        if import_rois is not -1:
            print('Aligning behaviour and imaging data for ALL rois...')
            dF_aligned_all, behaviour_aligned_all, bri_aligned_all = align_dF(raw_behaviour, dF_signal_all, frame_brightness,[-1, -1], [-1, -1], True, reject_trial_transition, session_crop)
            write_h5(h5dat, analyse_day, ROI_gcamp_all, '/roi_raw_all')
            write_h5(h5dat, analyse_day, PIL_gcamp_all, '/pil_raw_all')
            write_h5(h5dat, analyse_day, dF_aligned_all, '/dF_win_all')
            write_h5(h5dat, analyse_day, dF_signal_all, '/dF_original_all')



    else:
        h5dat.close()
        h5dat = h5py.File(h5file, 'r+')
        print('No behavior data found, saving dF data...')
        write_h5(h5dat, analyse_day, ROI_gcamp, '/roi_raw')
        write_h5(h5dat, analyse_day, PIL_gcamp, '/pil_raw')
        write_h5(h5dat, analyse_day, dF_signal, '/dF_original')
        if np.size(frame_brightness) > 0:
            print('Writing brightness data...')
            write_h5(h5dat, analyse_day, frame_brightness.T, '/FOV_bri')

    h5dat.flush()
    h5dat.close()
    print('done')


if __name__ == "__main__":
    """ add .sig file by typing the day and selecting the HDF5 and .sig file """

    # reject sample points around trial transition
    reject_trial_transition = True

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--day",
                      help="day for which to add the imaging data. Has to be same as corresponding behavior data in HDF5-file", nargs='*')
    parser.add_argument("-c", "--crop",
                      help="crop session as a fraction 1. (e.g. '-c 0 0.5' will only use first half of session)", nargs='*')
    parser.add_argument("-m", "--method", default=2,
                      help="select how neuropil and/or brightness is to be subtracted.", nargs='*')
    parser.add_argument("-r", "--rois", default=-1,
                      help="provide roi number to be imported (indexing starts at 0). Default is all.", nargs='*')
    args = parser.parse_args()

    add_sigfile(args)
