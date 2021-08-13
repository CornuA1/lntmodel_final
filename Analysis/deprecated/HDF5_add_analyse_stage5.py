"""
Add behaviour datafiles by prompting user to select raw files. Determine date
day by extracting it from the filename. User can select multiple datafiles
but only one config file that will be applied to all imported datasets.

IMPORTANT: this script is designed to be used on hdf5 files holding data of a single
    mouse only. For group data HDF5_parser.py should be used.
    
author: Lukas Fischer

"""
import sys
sys.path.append("/Users/lukasfischer/github/in_vivo/MTH3/Analysis")

import numpy as np
import h5py
import tkinter
from tkinter import filedialog
import csv
import warnings
from itertools import islice

from rewards import rewards
from licks import licks
from filter_trials import filter_trials

def write_h5(h5dat, day, dset, dset_name):
    """ Write dataset to HDF-5 file. Overwrite if it already exists. """
    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset('Day' + str(day) + '/' + dset_name,
                             data=dset, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat['Day' + str(day) + '/' + dset_name]
        h5dat.create_dataset('Day' + str(day) + '/' + dset_name,
                             data=dset, compression='gzip')

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
# store the maximum number of characters used to specify trials. Required
# to store in HDF5 later
trial_def_len = -1

root = tkinter.Tk()

print('ADD BEHAVIOR DATAFILE: 1) Select raw datafile(s), 2) Select config file, 3) select HDF5 file')

# ask user to select datafiles to be added
datafilenames = filedialog.askopenfilenames(
    initialdir=datdir, title='Please pick DATAFILES')
root.update()
if not datafilenames:
    raise NameError('Please select datafiles.')

configfilename = filedialog.askopenfilename(
    initialdir=datdir, title='Please pick CONFIG FILE')
root.update()
if not configfilename:
    warnings.warn("No config file selected", UserWarning)

h5filename = filedialog.askopenfilenames(
    initialdir=datdir, title='Please pick HDF5-FILE')
root.update()
if not h5filename:
    raise NameError('Please select HDF5 file.')


# open datafile. Store the filepointer separately as we need it
for row in datafilenames:
    # extract date infromation from filename
    path = row.split('/')
    day = path[-1]
    folder_suffix = ''
    # try to detect if this is a 'regular' session or an openloop or dark
    # session
    if day.find('openloop') != -1:
        folder_suffix = '_openloop'
        day = day.replace('_openloop', '')
    if day.find('dark') != -1:
        folder_suffix = '_dark'
        day = day.replace('_dark', '')

    day = day[9:17]
    day = day.split('_')[0]
    datfile = open(row, 'r')
    raw_data = csv.reader(open(row), delimiter=';', quoting=csv.QUOTE_NONE)
    # count number of rows in raw-datafile. Unfortunately there is no better
    # way to do this.
    row_count = sum(1 for row in raw_data) - SCRID_ROWS
    # set filepointer to the beginning of the file
    datfile.close()
    datfile = open(row, 'r')
    raw_data = csv.reader(open(row), delimiter=';', quoting=csv.QUOTE_NONE)
    # make ndarray and then write from the list into this array. This is
    # necessary to store it in HDF5 format
    datcont = np.ndarray(shape=(row_count, 8), dtype=float)
    # offsets added if a raw datafile had been concatenated as well as how
    # many lines have been skipped
    time_offset = 0
    trial_offset = 0
    skiplines = 0
    # running index to loop through raw datafile
    i = 0
    for line in raw_data:
        datcont[i - skiplines][0] = float(line[0]) + time_offset
        datcont[i - skiplines][1] = line[1]
        datcont[i - skiplines][2] = line[2]
        datcont[i - skiplines][3] = line[3]
        datcont[i - skiplines][4] = line[4]
        datcont[i - skiplines][5] = line[5]
        datcont[i - skiplines][6] = float(line[6]) + trial_offset
        datcont[i - skiplines][7] = line[7]
        # a datafile reset is being detected by checking if a timestamp is
        # smaller than the preceeding one. If yes, add the last timestamp to
        # the offset
        if datcont[i - skiplines][0] < datcont[i - skiplines - 1][0] and i > 0:
            print(datcont[i - skiplines][0], datcont[i - skiplines - 1][0])
            time_offset += datcont[i - skiplines - 1][0]  # take the last time
            trial_offset += datcont[i - skiplines - 1][6]
            datcont[i - skiplines][0] = float(line[i][0]) + time_offset
            datcont[i - skiplines][6] = float(line[i][6]) + trial_offset
        i += 1
    # this copy is necessary so that the new array owns the data (lengthy
    # numpy issue), otherwise we can't resize it
    tarr = np.copy(datcont)
    # remove skipped lines at the end (which are just zeros)
    tarr.resize((row_count - skiplines, 8))
    # read config file
    conf_file = csv.reader(open(configfilename, 'rt'),
                           delimiter=';', quoting=csv.QUOTE_NONE)
    conf_info = next(islice(conf_file, 0, 1))
    exp = conf_info[1]
    explen = conf_info[3]
    group = conf_info[7]
    dob = conf_info[11]
    strain = conf_info[13]
    stop_thresh = conf_info[15]
    valve_open = conf_info[17]
    comments = conf_info[19]
    # read the list of trial specifications and convert to strings
    for trials in conf_file:
        trial_str = ' '.join(trials)
        trial_list.append(np.string_(trial_str))
        # store length of longest trial specification line
        if(len(trial_str) > trial_def_len):
            trial_def_len = len(trial_str)

    # open HDF-5 file
    f = h5py.File(h5filename[0], 'a')
    # retrieve group identifier, if group doesn't exist, create it
    daygrp = f.require_group('Day' + day + folder_suffix)
    # write general attributes
    daygrp.attrs['Session_length'] = explen.encode('utf8')
    daygrp.attrs['Day'] = day.encode('utf8')
    daygrp.attrs['Group'] = group.encode('utf8')
    daygrp.attrs['DOB'] = dob.encode('utf8')
    daygrp.attrs['Strain'] = strain.encode('utf8')
    daygrp.attrs['Stop_Threshold'] = stop_thresh.encode('utf8')
    daygrp.attrs['Valve_Open_Time'] = valve_open.encode('utf8')
    daygrp.attrs['Comments'] = comments.encode('utf8')

    # check if dataset exists, if yes: ask if user wants to overwrite. If no,
    # create it
    dt_t = 'S' + str(trial_def_len)
    dt_s = 'S' + str(len_scrid)

    try:
        # retrive dataset (just check for first one)
        test_dset = daygrp['raw_data']
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del daygrp['raw_data']
        dset = daygrp.create_dataset('raw_data', data=tarr, compression='gzip')
        del daygrp['trial_list']
        trials_list = daygrp.create_dataset(
            "trial_list", data=trial_list, dtype=dt_t)
    except KeyError:
        dset = daygrp.create_dataset('raw_data', data=tarr, compression='gzip')
        trials_list = daygrp.create_dataset(
            "trial_list", data=trial_list, dtype=dt_t)

    day = day + folder_suffix
    print("day:", day, " processed.")

    trial_list = []
    scrid = []
    trial_def_len = -1
    len_scrid = -1

    track = 3
    track_long = 4

    print('Day' + str(day))
    raw = np.copy(f['Day' + str(day) + '/raw_data'])
    # remove times when mouse was in the black box
    no_blackbox_trials = filter_trials(raw, [], ['tracknumber', track])
    no_blackbox_trials = np.union1d(
        no_blackbox_trials, filter_trials(raw, [], ['tracknumber', track_long]))
    raw_filt = raw[np.in1d(raw[:, 4], [track, track_long]), :]

    write_h5(f, day, rewards(raw), 'rewards')
    rewards_ds = np.copy(f['Day' + str(day) + '/rewards'])

    lick_pre, licks_post = licks(raw_filt, rewards_ds)
    write_h5(f, day, lick_pre, 'licks_pre_reward')
    write_h5(f, day, licks_post, 'licks_post_rewar')

    f.flush()
    f.close()
