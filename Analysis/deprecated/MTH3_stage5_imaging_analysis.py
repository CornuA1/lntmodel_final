"""
Call separate analysis scripts. Load required datasets from HDF5-File,
provide data to analysis script and write the return dataset back to the
HDF5 file

"""

# Load native libraries
import sys
sys.path.append("/Users/lukasfischer/github/in_vivo/MTH3/Analysis")
# Load data-analysis relevant libraries
import numpy as np
import h5py
import tkinter
from tkinter import filedialog

# Load data-analysis function (these should all be on the Rochefort github repo
from rewards import rewards
from licks import licks
from filter_trials import filter_trials




root = tkinter.Tk()
h5dat = None  # filehandle


def open_h5():
    """ open HDF-5 file. First look under 'DATAFILE', if nothing can be found there, open a File-open dialog"""
    try:
        # Use path defined at top of script. If file doesn't exist, open file
        # dialog ot ask for location
        h5dat = h5py.File(DATAFILE, 'r')
    except (IOError, NameError):  # Index error: if no argument was provided, just ask for hdf5 file
        h5file = filedialog.askopenfilenames(title='Please pick HDF5-FILE')[0]
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


if __name__ == "__main__":
    """Start all analysis function from here, selecting mice and days to analyse."""

    # analyse_days = ['20170315', '20170316', '20170317', '20170318']
    analyse_days = []

    h5dat = open_h5()
    track = 3
    track_long = 4

    # if no days are selected, go through all datasets in HDF5 file
    if analyse_days == []:
        analyse_days = [day for day in h5dat]

    print(analyse_days)

    # run through all days but skip grating sessions
    for d in analyse_days:
        if d.find('gratings') == -1:
            print(str(d) + '/raw_data')
            raw = np.copy(h5dat[str(d) + '/raw_data'])
            # remove times when mouse was in the black box
            no_blackbox_trials = filter_trials(raw, [], ['tracknumber', track])
            no_blackbox_trials = np.union1d(
                no_blackbox_trials, filter_trials(raw, [], ['tracknumber', track_long]))
            raw_filt = raw[np.in1d(raw[:, 4], [track, track_long]), :]

            write_h5(h5dat, d, rewards(raw), 'rewards')
            rewards_ds = np.copy(h5dat[str(d) + '/rewards'])

            lick_pre, licks_post = licks(raw_filt, rewards_ds)
            write_h5(h5dat, d, lick_pre, 'licks_pre_reward')
            write_h5(h5dat, d, licks_post, 'licks_post_rewar')

    h5dat.flush()
    h5dat.close()
