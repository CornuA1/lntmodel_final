"""
Return only the selected trial type. Takes multiple different dataset types.

Parameters
-------
dset :      ndarray
            input dataset. Can have different formats such as raw or licks
dset_type : int
            type dataset provided in 'dset'
mode :      int
            desired trialtype {-1: all, 0: cued, 1: uncued, 2: long track cued,
            3: long track uncued}

Outputs
-------
dset :      ndarray
            return dataset, same format as input dataset

"""

import numpy as np

def select_trials( dset, dset_type, mode ):

    # different modes mean different columns are used for identifying the desired trials
    if dset_type == 'raw':
        select_column = 4
    elif dset_type == 'licks' or dset_type == 'stops':
        select_column = 3
    elif dset_type == 'reward':
        select_column = 2
    else:
        raise Exception("Unrecognised dataset type. Please use either 'raw', 'stops', 'licks', or 'reward'. ")

    # remove lines depending on which mode and selection column
    if mode == -1 or mode == 'all':
        pass
    elif mode == 0 or mode == 'short_cued':
        dset = dset[np.where(dset[:,select_column] == 3.0)]
    elif mode == 2 or mode == 'long_cued':
        dset = dset[np.where(dset[:,select_column] == 4.0)]
    else:
        print(mode)
        raise Exception("Unrecognised mode " + str(mode))

    return dset



"""
Select trials for dF/F dataset. The difference here is that we need the raw
behavioural dataset which contains trialtype information to determine which
rows of the dF/F dataset to return

Parameters
-------
dset_behav : ndarray
            raw behaviour dataset.
dset_dF : ndarray
            raw dF/F dataset.
mode :      int
            desired trialtype {-1: all, 0: cued, 1: uncued, 2: long track cued,
            3: long track uncued}

Outputs
-------
dset :      ndarray
            return dataset, same format as input dataset

"""
def select_trials_dF( dset_behav, dset_dF, mode ):
    # remove lines depending on which mode
    if mode == -1 or mode == 'all':
        pass
    elif mode == 0 or mode == 'short_cued':
        dset_dF = dset_dF[np.where(dset_behav[:,4] == 3.0)]
    elif mode == 2 or mode == 'long_cued':
        dset_dF = dset_dF[np.where(dset_behav[:,4] == 4.0)]
    else:
        raise Exception("Unrecognised mode " + mode)

    return dset_dF
