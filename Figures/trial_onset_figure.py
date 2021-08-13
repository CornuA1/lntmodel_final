"""Plot population activity aligned to trial onset.

Parameters
------
behav_collection : list of ndarrays
		behavior data provided, list of datasets

dF_collection : list of ndarrays
		imaging data provided, list of datasets

rec_info : list
		information corresponding to behav_collection and dF_collection

fname : string
		filename for output plot

trials : string
		which trials to include {'c':correct, 'ic':incorrect, 'both':all trials}

sortby : string
		which trial type to sort by {'none':no sorting, 'short':sort by peak dF/F of short trials
									 'long':sort by dF/F peak of long trials, 'both': sort each trial type individually}


"""

from filter_trials import filter_trials
from event_ind import event_ind
from matplotlib import pyplot
from scipy import signal
import warnings
import seaborn
import numpy
import h5py
import yaml
import os

def trial_onset_figure(HDF5_path, day, ROI, file_name, trials = 'both', sortby = 'both', file_format = 'png'):

    warnings.filterwarnings('ignore')
    seaborn.set_style('white')
    
    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)
    
    try:
        HDF5_data = h5py.File(HDF5_path, 'r')
    except:
        print('No HDF5 file.')
        return
    
    # load datasets and close HDF5 file again
    try:
        behavioral_data = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
    except:
        print(day + ': No behaviour_aligned.')
        return
    try:
        dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
    except:
        print(day + ': No dF_win.')
        return
    
    if numpy.size(dF_data, 1) == 1:
        print(day + ': Only one ROI detected.')
        return
    
    HDF5_data.close()
    
    try:
        short_trials = filter_trials(behavioral_data, [], ['tracknumber', 3])
        long_trials = filter_trials(behavioral_data, [], ['tracknumber', 4])
    except:
        print(day + ': Unexpected data format.')
        return
    
    if numpy.size(short_trials) == 0:
        print('No short trials detected.')
        return
    if numpy.size(long_trials) == 0:
        print('No long trials detected.')
        return

    # timewindow in seconds from the beginning of the trial
    EVENT_TIMEWINDOW = [3,5]

    figure = pyplot.figure(figsize=(12, 12))
    short = pyplot.subplot(2, 2, 1)
    long = pyplot.subplot(2, 2, 2)
    short_speed = pyplot.subplot(2, 2, 3)
    long_speed = pyplot.subplot(2, 2, 4)

    if trials == 'c':
        short_trials = filter_trials(behavioral_data, [], ['trial_successful'], 3)
        long_trials = filter_trials(behavioral_data, [], ['trial_successful'], 4)
    if trials == 'ic':
        short_trials = filter_trials(behavioral_data, [], ['trial_unsuccessful'], 3)
        long_trials = filter_trials(behavioral_data, [], ['trial_unsuccessful'], 4)

    # pull out indices of trial onsets for short and long trials
    events = event_ind(long_trials, ['trial_transition'])
    events = numpy.insert(events, 0, [1, 1], axis = 0)
    short_trials = numpy.intersect1d(events[:, 1], short_trials)
    short_events = events[numpy.in1d(events[:, 1], short_trials), :]
    long_trials = numpy.intersect1d(events[:, 1], long_trials)
    long_events = events[numpy.in1d(events[:, 1], long_trials),:]

    # grab peri-event dF trace for short trials
    trial_dF = numpy.zeros((numpy.size(short_events[:, 0]), 2))
    
    for i, index in enumerate(short_events):
        
        # determine indices of beginning and end of timewindow
        if behavioral_data[index[0], 0] - EVENT_TIMEWINDOW[0] > behavioral_data[0, 0]:
            trial_dF[i, 0] = numpy.where(behavioral_data[:, 0] < behavioral_data[index[0], 0] - EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i, 0] = 0
        if behavioral_data[index[0], 0] + EVENT_TIMEWINDOW[1] < behavioral_data[-1, 0]:
            trial_dF[i, 1] = numpy.where(behavioral_data[:, 0] > behavioral_data[index[0], 0] + EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i, 1] = numpy.size(behavioral_data, 0)
            
    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = numpy.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_short = numpy.zeros((short_events.shape[0], int(t_max)))
    cur_sweep_resampled_short_speed = numpy.zeros((short_events.shape[0], int(t_max)))

    # resample every sweep to match the longest sweep
    roi_avg_short = numpy.zeros((t_max, dF_data.shape[1]))
    for i in range(numpy.size(trial_dF, 0)):
        cur_sweep = dF_data[int(trial_dF[i, 0]):int(trial_dF[i, 1]), ROI]
        cur_sweep_resampled_short[i, :] = signal.resample(cur_sweep, t_max, axis = 0)
        cur_sweep_speed = behavioral_data[int(trial_dF[i, 0]):int(trial_dF[i, 1]), 3]
        cur_sweep_resampled_short_speed[i, :] = signal.resample(cur_sweep_speed, t_max, axis = 0)
        short_speed.plot(cur_sweep_resampled_short_speed[i, :], c = 'g', lw = 1)
        
    # first normalise mean roi sweep against entire roi signal and then normalise against the max dF or the roi within the sweep window
    roi_avg_short = numpy.mean(cur_sweep_resampled_short, axis = 0)/numpy.max(dF_data[:, ROI])
  
    # grab peri-event dF trace for long trials
    trial_dF = numpy.zeros((numpy.size(long_events[:, 0]), 2))
    
    for i, index in enumerate(long_events):
        
        # determine indices of beginning and end of timewindow
        if behavioral_data[index[0], 0] - EVENT_TIMEWINDOW[0] > behavioral_data[0, 0]:
            trial_dF[i, 0] = numpy.where(behavioral_data[:, 0] < behavioral_data[index[0], 0] - EVENT_TIMEWINDOW[0])[0][-1]
        else:
            trial_dF[i, 0] = 0
        if behavioral_data[index[0], 0] + EVENT_TIMEWINDOW[1] < behavioral_data[-1, 0]:
            trial_dF[i, 1] = numpy.where(behavioral_data[:, 0] > behavioral_data[index[0], 0] + EVENT_TIMEWINDOW[1])[0][0]
        else:
            trial_dF[i, 1] = numpy.size(behavioral_data, 0)
            
    # determine longest peri-event sweep (necessary due to sometimes varying framerates)
    t_max = numpy.amax(trial_dF[:,1] - trial_dF[:,0])
    cur_sweep_resampled_long = numpy.zeros((long_events.shape[0], int(t_max)))
    cur_sweep_resampled_long_speed = numpy.zeros((long_events.shape[0], int(t_max)))

    # resample every sweep to match the longest sweep
    roi_avg_long = numpy.zeros((t_max, dF_data.shape[1]))
    for i in range(numpy.size(trial_dF, 0)):
        cur_sweep = dF_data[int(trial_dF[i, 0]):int(trial_dF[i, 1]), ROI]
        cur_sweep_resampled_long[i, :] = signal.resample(cur_sweep, t_max, axis = 0)
        cur_sweep_speed = behavioral_data[int(trial_dF[i, 0]):int(trial_dF[i, 1]), 3]
        cur_sweep_resampled_long_speed[i, :] = signal.resample(cur_sweep_speed, t_max, axis = 0)
        long_speed.plot(cur_sweep_resampled_long_speed[i, :], c = 'g', lw = 1)
        
    # first normalise mean roi sweep against entire roi signal and then normalise against the max dF or the roi within the sweep window
    roi_avg_long = numpy.mean(cur_sweep_resampled_long, axis = 0)/numpy.max(dF_data[:, ROI])

    # sort by peak activity
    mean_dF_sort_short = numpy.zeros(roi_avg_short.shape[1])
    for i, row in enumerate(numpy.transpose(roi_avg_short)):
        if not numpy.all(numpy.isnan(row)):
            mean_dF_sort_short[i] = numpy.nanargmax(row)
    sort_ind_short = numpy.argsort(mean_dF_sort_short)

    mean_dF_sort_long = numpy.zeros(roi_avg_long.shape[1])
    for i, row in enumerate(numpy.transpose(roi_avg_long)):
        if not numpy.all(numpy.isnan(row)):
            mean_dF_sort_long[i] = numpy.nanargmax(row)
    sort_ind_long = numpy.argsort(mean_dF_sort_long)

    if sortby == 'none':
        seaborn.heatmap(numpy.transpose(roi_avg_short), cmap = 'jet', vmin = 0.0, vmax = 1.0, ax = short, cbar = False)
        seaborn.heatmap(numpy.transpose(roi_avg_long), cmap = 'jet', vmin = 0.0, vmax = 1.0, ax = long, cbar = False)
    elif sortby == 'short':
        seaborn.heatmap(numpy.transpose(roi_avg_short[:, sort_ind_short]), cmap = 'jet', vmin = 0.0, vmax = 1.0, ax = short, cbar = False)
        seaborn.heatmap(numpy.transpose(roi_avg_long[:, sort_ind_short]), cmap = 'jet', vmin = 0.0, vmax = 1.0, ax = long, cbar = False)
    elif sortby == 'long':
        seaborn.heatmap(numpy.transpose(roi_avg_short[:, sort_ind_long]), cmap = 'jet', vmin = 0.0, vmax = 1.0, ax = short, cbar = False)
        seaborn.heatmap(numpy.transpose(roi_avg_long[:, sort_ind_long]), cmap = 'jet', vmin = 0.0, vmax = 1.0, ax = long, cbar = False)
    elif sortby == 'both':
        seaborn.heatmap(numpy.transpose(roi_avg_short[:, sort_ind_short]), cmap = 'jet', vmin = 0.0, ax = short, cbar = False)
        seaborn.heatmap(numpy.transpose(roi_avg_long[:, sort_ind_long]), cmap = 'jet', vmin = 0.0, ax = long, cbar = False)
    
    event_loc = (t_max/(EVENT_TIMEWINDOW[0] + EVENT_TIMEWINDOW[1]))*EVENT_TIMEWINDOW[0]
    short.axvline(event_loc, c = 'r')
    long.axvline(event_loc, c = 'r')
    
    axes = figure.get_axes()

    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.tick_params(reset = 'on', axis= 'both', direction = 'in', length = 4, right = 'off', top='off')
    
    figure.suptitle(file_name, wrap = True, fontsize = 25)
    
    if not os.path.isdir(content['figure_output_path']):
        os.mkdir(content['figure_output_path'])
        
    file_name = content['figure_output_path'] + file_name + '.' + file_format
    
    figure.savefig(file_name, format = file_format)