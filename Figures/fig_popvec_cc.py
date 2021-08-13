"""
Calculate population vectors for individual sessions, separated by track
type and make pop-vector cross-correlation plot

@author: Enrique Toloza, 2017, MIT; adapted: Lukas Fischer, 2018, MIT

"""

def fig_popvec_cc(included_recordings, rois='task_engaged_all', trials='both', fformat='png', fname='', subfolder=''):
    # load local settings file
    import matplotlib
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("./Analysis")

    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    import yaml
    import h5py
    import json
    import seaborn as sns
    sns.set_style('white')
    import os
    with open('./loc_settings.yaml', 'r') as f:
                content = yaml.load(f)

    # basic analysis and track parameters
    tracklength_short = 320
    tracklength_long = 380
    track_start = 100
    track_short = 3
    track_long = 4
    # bin from which to start analysing and plotting dF data
    start_bin = 20
    end_bin_short = 64
    end_bin_long = 76

    # size of bin in cm
    bin_size = 5
    binnr_short = tracklength_short/bin_size
    binnr_long = tracklength_long/bin_size

    mean_dF_short_coll_sig = []
    mean_dF_short_coll_sort = []

    mean_dF_long_coll_sig = []
    mean_dF_long_coll_sort = []

    # load data
    for r in included_recordings:
        # load individual dataset
        print(r)
        mouse = r[0]
        session = r[1]
        # len > 2 indicates that we want to use a classification file in r[2], but plotting the data for r[1]
        if len(r) > 2:
            roi_classification_file = r[2]
        else:
            roi_classification_file = r[1]

        h5path = content['imaging_dir'] + mouse + '/' + mouse + '.h5'
        h5dat = h5py.File(h5path, 'r')
        behav_ds = np.copy(h5dat[session + '/behaviour_aligned'])
        dF_ds = np.copy(h5dat[session + '/dF_win'])
        h5dat.close()

        with open(content['figure_output_path'] + mouse+roi_classification_file + os.sep + 'roi_classification.json') as f:
            roi_classification = json.load(f)

        # pull out trial numbers of respective sections
        trials_short_sig = filter_trials(behav_ds, [], ['tracknumber',track_info['short']['track_number']])
        trials_long_sig = filter_trials(behav_ds, [], ['tracknumber',track_info['long']['track_number']])

        # further filter if only correct or incorrect trials are plotted
        if trials == 'c':
            trials_short_sig = filter_trials(behav_ds, [], ['trial_successful'],trials_short_sig)
            trials_long = filter_trials(behav_ds, [], ['trial_successful'],trials_long)
        if trials == 'ic':
            trials_short_sig = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short_sig)
            trials_long = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long)

        for t_t, track_type in enumerate([track_short, track_long]):
            
