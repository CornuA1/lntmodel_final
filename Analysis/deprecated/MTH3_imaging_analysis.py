"""
Call separate analysis scripts. Load required datasets from HDF5-File,
provide data to analysis script and write the return dataset back to the
HDF5 file

"""

# Load native libraries
from tkinter import filedialog
import sys
sys.path.append("/Users/lukasfischer/github/in_vivo/MTH1/Analysis")
sys.path.append("/Users/lukasfischer/github/in_vivo/General/Imaging")

# Load data-analysis relevant libraries
import numpy as np
import h5py
from align_dF_interp import align_dF
from dF_win_mpi import dF_win



# just replace the line below with the path to your hdf-5 file, but use double-backslashes (look above to line 11 and 12 to see how it should look like)
# DATAFILE = 'D:\Experiments\Data\TNR2\Data\TNR2_batch3.h5' # this should
# work cross-platform but the folder structure of the experiment has to be
# constant
DATAFILE = '/Users/lukasfischer/Work/exps/MTH3/imaging/LF160901_1/LF160901_1.h5'

h5dat = None  # filehandle


def open_h5():
    """ open HDF-5 file. First look under 'DATAFILE', if nothing can be found there, open a File-open dialog"""
    try:
        # Use path defined at top of script. If file doesn't exist, open file
        # dialog ot ask for location
        h5dat = h5py.File(DATAFILE, 'a')
    except IOError:  # Index error: if no argument was provided, just ask for hdf5 file
        h5file = filedialog.askopenfilename(title='Please select HDF5 file')
        h5dat = h5py.File(h5file)
    finally:
        return h5dat


def write_h5(h5dat, day, mouse, dset, dset_name):
    """ Write dataset to HDF-5 file. Overwrite if it already exists. """
    h5dat = open_h5()
    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset(str(d) + '/' + m + '/' +
                             dset_name, data=dset, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat[str(d) + '/' + m + '/' + dset_name]
        h5dat.create_dataset(str(d) + '/' + m + '/' +
                             dset_name, data=dset, compression='gzip')
    h5dat.close()

if __name__ == "__main__":
    """ Start imaging analysis functions from here """
    analyse_days = [[['Day20161214'], ['LF160901_1']]]

    # store SMI values
    BINNR = 24
    BINNR_LONG = 32

    for i in analyse_days:
        days = i[0]
        mice = i[1]
        # give user feedback how much is done
        progress_report = 100 / (len(mice) * len(days))
        j = 1
        print(days, mice)
        for k, d in enumerate(days):
            for l, m in enumerate(mice):
                print('processing: Mouse ', str(m), ' Day ', str(d))

                h5dat = open_h5()
                raw_behaviour = np.copy(h5dat[str(d) + '/' + m + '/raw_data'])
                gcamp_raw = np.copy(h5dat[str(d) + '/' + m + '/gcamp_raw'])

                print('Calculating dF/F...')
                ROI_gcamp = gcamp_raw[
                    :, (np.size(gcamp_raw, 1) / 3) * 2:np.size(gcamp_raw, 1)]
                PIL_gcamp = gcamp_raw[:, np.size(
                    gcamp_raw, 1) / 3:(np.size(gcamp_raw, 1) / 3) * 2]
                dF_sig_dF = dF_win(ROI_gcamp)
                dF_pil_dF = dF_win(PIL_gcamp)

                dF_signal = dF_sig_dF - dF_pil_dF

                # dF_aligned, behaviour_aligned = align_dF( raw_behaviour, dF_signal, -1, [-1,-1], True, False, [-1,14330]) # LF160901_1 2016121
                # dF_aligned, behaviour_aligned = align_dF( raw_behaviour,
                # dF_signal, -1, [-1,-1], True, False, [-1,27818]) # LF160628_2
                # 2016121

                dF_aligned, behaviour_aligned = align_dF(
                    raw_behaviour, dF_signal, -1, [-1, -1], True, True, [-1, -1])
                write_h5(h5dat, d, m, dF_aligned, '/dF_win')
                write_h5(h5dat, d, m, behaviour_aligned, '/behaviour_aligned')

                # gcamp = np.copy(h5dat[str(d) + '/' + m + '/gcamp_aligned'])
                # write_h5( h5dat, d, m, dF_win( gcamp ),'/dF_win')

                print('done')
                h5dat.flush()
                h5dat.close()
#                write_h5( h5dat, d, m, dF_location_avg( raw_behaviour, resampled_dF, BINNR, 120, 0 ),'/dFlocavg_short')
#                write_h5( h5dat, d, m, dF_location_avg( raw_behaviour, resampled_dF, BINNR, 120, 1 ),'/dFlocavg_short_uc')
#                write_h5( h5dat, d, m, dF_location_avg( raw_behaviour, resampled_dF, BINNR_LONG, 160, 2 ),'/dFlocavg_long')
#                write_h5( h5dat, d, m, dF_location_avg( raw_behaviour, resampled_dF, BINNR_LONG, 160, 3 ),'/dFlocavg_long_uc')
#
#                write_h5( h5dat, d, m, dF_trial_locavg( raw_behaviour, resampled_dF, BINNR, 120, 0 ),'/dFtrial_locavg_s')
#                write_h5( h5dat, d, m, dF_trial_locavg( raw_behaviour, resampled_dF, BINNR, 120, 1 ),'/dFtrial_locavg_s_uc')
#                write_h5( h5dat, d, m, dF_trial_locavg( raw_behaviour, resampled_dF, BINNR_LONG, 160, 2 ),'/dFtrial_locavg_l')
#                write_h5( h5dat, d, m, dF_trial_locavg( raw_behaviour, resampled_dF, BINNR_LONG, 160, 3 ),'/dFtrial_locavg_l_uc')
#
#                write_h5( h5dat, d, m, dF_trial_locavg( raw_behaviour, resampled_dF, BINNR, 120, 1 ),'/dFlocavg_short_uc')
#                write_h5( h5dat, d, m, dF_trial_locavg( raw_behaviour, resampled_dF, BINNR_LONG, 160, 2 ),'/dFlocavg_long')
#                write_h5( h5dat, d, m, dF_trial_locavg( raw_behaviour, resampled_dF, BINNR_LONG, 160, 3 ),'/dFlocavg_long_uc')
