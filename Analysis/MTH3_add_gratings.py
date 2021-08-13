"""
Add behaviour datafiles by prompting user to select raw files. Determine date
day by extracting it from the filename. User can select multiple datafiles
but only one config file that will be applied to all imported datasets.

IMPORTANT: this script is designed to be used on hdf5 files holding data of a single
    mouse only. For group data HDF5_parser.py should be used.

author: Lukas Fischer

"""


import numpy as np
import h5py
import tkinter
from tkinter import filedialog
import csv
import argparse

# Automatically overwrite existing datasets
OVERWRITE = True

# number of rows that contain script IDs at the beginning of each datafile
SCRID_ROWS = 0

# store script ID information
scrid = []

# length of script ID
len_scrid = -1

# amount of time that is skipped at the beginning of every datafile.
STARTSKIP = 0.0

# datadir, this is for convenience
datdir = '/Users/lukasfischer/Work/data/MTH3/'

# holds the trials-list from the config file
trial_list = []
# store the maximum number of characters used to specify trials. Required to store in HDF5 later
trial_def_len = -1

root = tkinter.Tk()

print('ADD DATAFILE FOR GRATINGS: 1) Select raw datafile(s), 2) select HDF5 file')


# ask user to select datafiles to be added
datafilenames = filedialog.askopenfilenames(initialdir=datdir, title = 'Please pick DATAFILES')
root.update()
if not datafilenames:
    raise NameError('Please select datafiles.')

h5filename = filedialog.askopenfilenames(initialdir=datdir, title = 'Please pick HDF5-FILE')
root.update()
if not h5filename:
    raise NameError('Please select HDF5 file.')

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ml", "--multilayer",
                    help="suffix for multilayer recordings. Intended to be used to identify multiple layers of imaging. Input datafiles will be added once for each provided multilayer suffix.", nargs='*')
parser.add_argument("-s", "--suffix",
                    help="generic suffix. Intended to be used if multiple training sessions or recordings have been carried out in a day. Provided suffix will be added to the end of the folder name.", nargs='*')
args = parser.parse_args()

multilayer_suffix = args.multilayer
optional_suffix = args.suffix
print(multilayer_suffix)

# open datafile. Store the filepointer separately as we need it
for row in datafilenames:
    # extract date information from filename
    path = row.split('/')
    day = path[-1]
    folder_suffix = '_gratings'
    day = day[9:17]
    day = day.split('_')[0]
    datfile = open(row, 'r')
    raw_data = csv.reader(open(row), delimiter=';', quoting=csv.QUOTE_NONE)
    # count number of rows in raw-datafile. Unfortunately there is no better way to do this.
    row_count = sum(1 for row in raw_data) - SCRID_ROWS
    # set filepointer to the beginning of the file
    datfile.close()
    datfile = open(row, 'r')
    raw_data = csv.reader(open(row), delimiter=';', quoting=csv.QUOTE_NONE)
    # make ndarray and then write from the list into this array. This is necessary to store it in HDF5 format
    datcont = np.ndarray(shape=(row_count,10), dtype=float)
    # offsets added if a raw datafile had been concatenated as well as how many lines have been skipped
    time_offset = 0
    trial_offset = 0
    skiplines = 0
    # running index to loop through raw datafile
    i = 0
    for line in raw_data:
        datcont[i-skiplines][0] = float(line[0]) + time_offset
        datcont[i-skiplines][1] = line[1]
        datcont[i-skiplines][2] = line[2]
        datcont[i-skiplines][3] = line[3]
        datcont[i-skiplines][4] = line[4]
        datcont[i-skiplines][5] = line[5]
        datcont[i-skiplines][6] = line[6]
        datcont[i-skiplines][7] = line[7]
        datcont[i-skiplines][8] = line[8]
        if len(line) > 9:
            datcont[i-skiplines][9] = line[9]
        i = i+1

    # this copy is necessary so that the new array owns the data (lengthy numpy issue), otherwise we can't resize it
    tarr = np.copy(datcont)
    # remove skipped lines at the end (which are just zeros)
    tarr.resize((row_count - skiplines, 10))

    # open HDF-5 file
    f = h5py.File(h5filename[0], 'a')
    # check if optional suffixes were provided, if yes, create folder and Store behavior data once for each suffix provided
    if multilayer_suffix is not None:
        for osuff in multilayer_suffix:
            # retrieve group identifier, if group doesn't exist, create it
            if optional_suffix is not None:
                folder_suffix =  folder_suffix + '_' + optional_suffix[0]
            #retrieve group identifier, if group doesn't exist, create it
            daygrp = f.require_group('Day' + day + folder_suffix + '_' + osuff)
            try:
                # retrive dataset (just check for first one)
                test_dset = daygrp['raw_data' ]
                # if we want to overwrite: delete old dataset and then re-create with new data
                del daygrp['raw_data']
                dset = daygrp.create_dataset('raw_data', data=tarr,compression='gzip')
            except KeyError:
                dset = daygrp.create_dataset('raw_data', data=tarr,compression='gzip')
    else:
        # retrieve group identifier, if group doesn't exist, create it
        if optional_suffix is not None:
            folder_suffix = folder_suffix + '_' + optional_suffix[0]
        #retrieve group identifier, if group doesn't exist, create it
        daygrp = f.require_group('Day' + day + folder_suffix)
        try:
            # retrive dataset (just check for first one)
            test_dset = daygrp['raw_data' ]
            # if we want to overwrite: delete old dataset and then re-create with new data
            del daygrp['raw_data']
            dset = daygrp.create_dataset('raw_data', data=tarr,compression='gzip')
        except KeyError:
            dset = daygrp.create_dataset('raw_data', data=tarr,compression='gzip')


    print("day:", day, " processed.")
    trial_list = []
    scrid = []
    trial_def_len = -1
    len_scrid = -1
    f.close()
