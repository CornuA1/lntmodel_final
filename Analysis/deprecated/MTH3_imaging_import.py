# -*- coding: utf-8 -*-
"""
Write imaging data to HDF5 file with accompanying behaviour file (e.g. when the recording was carried out without a behavioural session)

Created on Thu Sep 29 12:42:50 2016

@author: lukasfischer


"""


# Load native libraries
from tkinter import filedialog
import sys
sys.path.append("/Users/lukasfischer/github/in_vivo/MTH1/Analysis")
sys.path.append("/Users/lukasfischer/github/in_vivo/General/Imaging")

# Load data-analysis relevant libraries
import h5py
from import_dF import import_dF



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
    h5dat.flush()
    h5dat.close()

if __name__ == "__main__":
    analyse_days = [[['Day20161214'], ['LF160901_1']]]

    for i in analyse_days:
        days = i[0]
        mice = i[1]
        j = 1
        print(days, mice)
        for k, d in enumerate(days):
            for l, m in enumerate(mice):
                print('processing: Mouse ', m, ' Day ', str(d))
                print('Loading data...')
                write_h5(h5dat, d, m, import_dF(
                    '/Users/lukasfischer/Work/exps/MTH3/imaging/LF160901_1/20161214/M01_003_002_rigid.sig'), 'gcamp_raw')
                #print('Calculating dF/F...')
                #h5dat = open_h5()
                #gcamp = np.copy(h5dat[str(d) + '/' + m + '/gcamp_raw'])
                #write_h5( h5dat, d, m, dF_win( gcamp, 15.5 ),'/dF_win')
