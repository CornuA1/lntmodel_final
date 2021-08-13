"""

Perform a (spatial) shuffle test on ROI traces to categorize ROIs according to trial location landmarks.

Slightly updated to run as a standalone function/fig that plots and returns a list of ROIs that have been deemed task engaged

Authors: Quique Toloza

"""

import os, ipdb
import numpy
import seaborn
import yaml
from scipy import stats
from matplotlib import pyplot
import matplotlib
import json

import h5py
import numpy as np
import sys
import traceback
import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from filter_trials import filter_trials
from write_dict import write_dict

def fig_shuffled_ROI(h5path, sess, ROI, fname, figure_format = 'png', subfolder=[]):

    h5dat = h5py.File(h5path, 'r')
    behavioral_data = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_data = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    bin_size = 5.0
    number_of_shuffles = 100
    min_robustness = 0.25

    warning_5 = False
    warning_20 = False

    track_types = ['short', 'long']

    track_info = {track_type: {} for track_type in track_types}
    track_trials = {track_type: [] for track_type in track_types}
    categories = {'task_engaged_short': False, \
                  'robustness_short': 0, \
                  'task_engaged_long': False, \
                  'robustness_long': 0
                  }

    track_info['short']['track_number'] = 3
    track_info['short']['landmark_zone'] = [200.0, 240.0]
    track_info['short']['reward_zone'] = [320.0, 340.0]
    track_info['short']['track_length'] = 340.0 + bin_size

    track_info['long']['track_number'] = 4
    track_info['long']['landmark_zone'] = [200.0, 240.0]
    track_info['long']['reward_zone'] = [380.0, 400.0]
    track_info['long']['track_length'] = 400.0 + bin_size

    figure_name = fname + ', ' + 'ROI ' + str(ROI)

    figure = pyplot.figure(figsize = (16, 10))

    # minimize empty figure space
    figure.subplots_adjust(bottom = 0.075, left = 0.05, right = 0.975, wspace = 0.05, hspace = 0.15)

    seaborn.set_style('white')

    # this file contains machine-specific info
    local_settings = loc_info

    # check to make sure we're only using the dF data we need
    if len(dF_data.shape) > 1:
        dF_data = dF_data[:, ROI]

    temp = 0

    for track_type in track_types:
        try:
            track_trials[track_type] = filter_trials(behavioral_data, [], ['tracknumber', track_info[track_type]['track_number']])
        except IndexError:
            print('        Unexpected track number format.')
            return

        temp += len(track_trials[track_type])

    if temp == 0:
        print('        No trials of specified types detected.')
        return

    task_engaged = False
    y_min = 0.0
    y_max = 0.0

    comparisons = {track_type: [] for track_type in track_types}
    robustness = {track_type: [] for track_type in track_types}
    average_robustness = {track_type: [] for track_type in track_types}

    for t_t, track_type in enumerate(track_types):
        landmark_zone = track_info[track_type]['landmark_zone']
        reward_zone = track_info[track_type]['reward_zone']
        track_length = track_info[track_type]['track_length']

        trials = track_trials[track_type]

        normal_plot = pyplot.subplot(2, len(track_types), t_t + 1)
        shuffled_plot = pyplot.subplot(2, len(track_types), t_t + 3)

        # mark key zones on the tracks
        normal_plot.axvspan(landmark_zone[0], landmark_zone[1], color = '0.9', zorder = 1)
        normal_plot.axvspan(reward_zone[0], reward_zone[1], color = '#D2F2FF', zorder = 1)

        shuffled_plot.axvspan(landmark_zone[0], landmark_zone[1], color = '0.9', zorder = 1)
        shuffled_plot.axvspan(reward_zone[0], reward_zone[1], color = '#D2F2FF', zorder = 1)

        N = int(track_length/bin_size)

        distances = numpy.linspace(bin_size/2.0, track_length - bin_size/2.0, N)
        mean_dF = numpy.empty(N)
        mean_dF.fill(numpy.NaN)

        dF = numpy.empty((len(trials), N))
        dF.fill(numpy.NaN)

        # pull out current trial and corresponding dF data and bin it
        for t, trial in enumerate(trials):
            current_trial_location = behavioral_data[behavioral_data[:, 6] == trial, 1]
            current_trial_dF = dF_data[behavioral_data[:, 6] == trial]
            current_trial_start = int(current_trial_location[0]/bin_size)
            current_trial_end = int(current_trial_location[-1]/bin_size)

            # ensure bins make logical sense
            while current_trial_end < current_trial_start:
                current_trial_location = current_trial_location[:-1]
                current_trial_dF = current_trial_dF[:-1]
                current_trial_end = int(current_trial_location[-1]/bin_size)

            # skip trials that are too short, where data ends before the reward zone, or where data extends beyond the track length
            if current_trial_location[-1] < reward_zone[0] or current_trial_location[-1] - current_trial_location[0] < 100.0 or current_trial_end > N:
                # print('rejecting trial '  + str(trial))
                # if current_trial_location[-1] < reward_zone[0]:
                #     print('end < RZ')
                # if current_trial_location[-1] - current_trial_location[0] < 100.0:
                #     print('trial length < 100')
                # if current_trial_end > N:
                #     print('trial end > trial total length')
                continue

            # map the data onto a matrix containing the spatial bins and trials
            dF_trial = stats.binned_statistic(current_trial_location, current_trial_dF, 'mean', (current_trial_end - current_trial_start) + 1, (current_trial_start*bin_size, (current_trial_end + 1)*bin_size))[0]
            dF[t, current_trial_start:current_trial_end + 1] = dF_trial
            normal_plot.plot(distances[current_trial_start:current_trial_end + 1], dF_trial, c = '0.8')
            y_min = numpy.nanmin([y_min, numpy.amin(dF_trial)])
            y_max = numpy.nanmax([y_max, numpy.amax(dF_trial)])

        # exclude trials with no dF data
        trials = trials[~numpy.isnan(dF).all(1)]
        dF = dF[~numpy.isnan(dF).all(1), :]

        for spatial_bin in range(N):
            number_of_nans = numpy.sum(numpy.isnan(dF[:, spatial_bin]))

            if number_of_nans/len(trials) < 0.2:
                if number_of_nans/len(trials) > 0.05:
                    warning_5 = True
                    # print('Warning: More than 5% of trials have NaNs in spatial bin ' + str(spatial_bin))

                mean_dF[spatial_bin] = numpy.nanmean(dF[:, spatial_bin])
            else:
                warning_20 = True
                #print('Warning: More than 20 per cent of trials have NaNs in spatial bin ' + str(spatial_bin))

#        # plot means, excluding spatial bins with NaNs
#        indices = ~numpy.isnan(dF).any(0)
#
#        distances[~indices] = numpy.nan
#
#        mean_dF[indices] = dF[:, indices].mean(axis = 0)
#        mean_dF[~indices] = numpy.nan
        normal_plot.plot(distances, mean_dF, c = seaborn.xkcd_rgb['windows blue'], lw = 3)

        shuffled_dF = numpy.empty((number_of_shuffles, dF.shape[0], N))
        shuffled_dF.fill(numpy.NaN)

        # now shuffle dF data around on a per trial basis
        ipdb.set_trace()
        for n in range(number_of_shuffles):
            shuffled_dF[n, :, :] = dF

            # data is only shuffled within spatial bins that previously contained data
            for t in range(len(trials)):
                trial_indices = ~numpy.isnan(dF[t, :])
                shuffled_dF[n, t, trial_indices] = numpy.roll(dF[t, trial_indices], numpy.random.randint(0, len(dF[t, trial_indices])))

            # same as for the regular data, ignore spatial bins with fewer than 10 non-NaN values
            for spatial_bin in range(N):
                if numpy.sum(~numpy.isnan(shuffled_dF[n, :,  spatial_bin])) < 10:
                    shuffled_dF[n, :, spatial_bin] = numpy.NaN

            # for spatial_bin in range(N):
            #     number_of_nans = numpy.sum(numpy.isnan(shuffled_dF[n, :,  spatial_bin]))
            #
            #     if number_of_nans/len(trials) < 0.2:
            #         if number_of_nans/len(trials) > 0.05:
            #             warning_5 = True
            #             # print('Warning: More than 5% of trials have NaNs in spatial bin ' + str(spatial_bin))
            #         shuffled_dF[n, :, spatial_bin] = numpy.NaN
            #     else:
            #         warning_20 = True
            #         # print('Warning: More than 20 per cent of trials have NaNs in spatial bin ' + str(spatial_bin))

#            shuffled_plot.plot(distances[indices], shuffled_dF[n, :, indices].mean(axis = 1), c = '0.8', lw = 3)
            shuffled_plot.plot(distances, numpy.nanmean(shuffled_dF[n, :, :], axis = 0), c = '0.8', lw = 3)

        # average across trials
#        mean_shuffled_dF = numpy.empty((number_of_shuffles, N))
#        mean_shuffled_dF[:, indices] = shuffled_dF[:, :, indices].mean(axis = 1)
#        mean_shuffled_dF[:, ~indices] = numpy.nan
        mean_shuffled_dF = numpy.nanmean(shuffled_dF, 1)


        # find the standard deviation at each bin
        std_shuffled_dF = mean_shuffled_dF.std(axis = 0)

        # average across shuffles for each spatial bin
        mean_mean_shuffled_dF = mean_shuffled_dF.mean(axis = 0)

        threshold = mean_mean_shuffled_dF + 3.0*std_shuffled_dF

        # compare to threshold at each spatial bin
        comparisons[track_type] = mean_dF - threshold

        consecutive = 0

        robustness[track_type] = dF > threshold
        # print(robustness[track_type])
        robustness[track_type] = numpy.sum(robustness[track_type], axis = 0)/len(trials)
        # print(robustness[track_type])

        #average_robustness[track_type] = numpy.mean(robustness[track_type])
        average_robustness[track_type] = []
        categories['robustness_'+track_type] = []
        #print(average_robustness[track_type])
        roi_classification = np.zeros((2,5))

        # check to see if there are at least 3 consecutive spatial bins over threshold
        for c, comparison in enumerate(comparisons[track_type]):
            if comparison != numpy.nan:
                if comparison > 0.0 and robustness[track_type][c] > min_robustness:
                    if consecutive == 0:
                        first = c
                    average_robustness[track_type].append(robustness[track_type][c])
                    categories['robustness_'+track_type].append(robustness[track_type][c])
                    consecutive += 1
                else:
                    if consecutive >= 3:
                        temp = distances[first:c].mean()
                        categories['task_engaged_'+track_type] = True
                    consecutive = 0
            else:
                average_robustness[track_type] = []
                categories['robustness_'+track_type] = []
                consecutive = 0

            # check for ROI activity at the end of the tracks
            if consecutive >= 3:
                temp = distances[first:c].mean()
                categories['task_engaged_'+track_type] = True

        average_robustness[track_type] = np.mean(average_robustness[track_type])
        categories['robustness_'+track_type] = np.mean(categories['robustness_'+track_type])

        if categories['task_engaged_'+track_type]:
            normal_plot.set_title(track_type + '\n*******' + ' robustness: ' + str(categories['robustness_'+track_type]), fontsize = 20)
            task_engaged = True
        else:
            normal_plot.set_title(track_type + ' robustness: ' + str(categories['robustness_'+track_type]), fontsize = 20)

        normal_plot.plot(distances, threshold, color = 'k', linestyle = 'dashed')
        normal_plot.set_xlim([0, track_length])

        shuffled_plot.plot(distances, mean_dF, c = seaborn.xkcd_rgb['windows blue'], lw = 3)
        shuffled_plot.plot(distances, threshold, color = 'k', linestyle = 'dashed')
        shuffled_plot.set_xlim([0, track_length])
        shuffled_plot.set_xlabel('location (cm)', fontsize = 20)

        if t_t == 0:
            normal_plot.set_ylabel('dF/F', fontsize = 20)
            shuffled_plot.set_ylabel('dF/F', fontsize = 20)

    if task_engaged:
        figure.suptitle(figure_name + '\ntask engaged ROI', wrap = True, fontsize = 20)
    else:
        figure.suptitle(figure_name + '\nnot task engaged ROI', wrap = True, fontsize = 20)

    axes = figure.get_axes()

    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.tick_params(reset = 'on', axis= 'both', direction = 'in', length = 4, right = 'off', top='off')

        # set the y-axis limits according to the maximum dF value in either track type, for comparison
        axis.set_ylim([y_min, y_max])

    if not os.path.isdir(local_settings['figure_output_path'] + subfolder):
        os.mkdir(local_settings['figure_output_path'] + subfolder)
    figure_path = local_settings['figure_output_path'] + subfolder + os.sep + fname + '.' + figure_format
    print(figure_path)
    try:
        figure.savefig(figure_path, format=figure_format)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    # close the figure to save memory
    pyplot.close(figure)

    if warning_5 == True:
        if warning_20 == True:
            print('WARNING: spatial bin with > 20% nan detected. Affected bin removed from analysis')
        else:
            print('WARNING: spatial bin with NaN (<5%) detected')
    return categories

def run_analysis(mousename, sessionname, sessionname_openloop, number_of_rois, h5_filepath, json_path, subname, sess_subfolder, session_rois, ol_sess=True):
    MOUSE = mousename
    SESSION = sessionname
    SESSION_OPENLOOP = sessionname_openloop
    NUM_ROIS = number_of_rois
    h5path = h5_filepath
    SUBNAME = subname
    subfolder = sess_subfolder
    write_to_dict = False
    make_figure = True

    if ol_sess:
        subfolder_ol = sess_subfolder + '_openloop'

    # if we want to run through all the rois, just say all
    if NUM_ROIS == 'all':
        h5dat = h5py.File(h5path, 'r')
        dF_ds = np.copy(h5dat[SESSION + '/dF_win'])
        h5dat.close()
        roilist = np.arange(0,dF_ds.shape[1],1).tolist()
        write_to_dict = True
        print('number of rois: ' + str(NUM_ROIS))
    elif NUM_ROIS == 'valid':
        # only use valid rois
        with open(json_path, 'r') as f:
            sess_dict = json.load(f)
        roilist = sess_dict['valid_rois']
        print('analysing ' + NUM_ROIS + ' rois: ' + str(roilist))
    elif type(NUM_ROIS) is int:
        roilist = range(NUM_ROIS)
        print('analysing ' + str(NUM_ROIS) + ' rois: ' + str(roilist))
    else:
        roilist = NUM_ROIS
        print('analysing custom list of rois: ' + str(roilist))

    session_rois = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'task_engaged_short' : [],
        'robustness_short' : [],
        'task_engaged_long' : [],
        'robustness_long' : []
    }

    # run through all rois and sort data based on classifcation
    for i,r in enumerate(roilist):
        print(SUBNAME + ': ' + str(r))
        categories = fig_shuffled_ROI(h5path, SESSION, r, 'shuffled_roi_' + str(r), 'png', subfolder)
        if categories['task_engaged_short'] == True:
            session_rois['task_engaged_short'].append(1)
            session_rois['robustness_short'].append(categories['robustness_short'])
        else:
            session_rois['task_engaged_short'].append(0)
            session_rois['robustness_short'].append(categories['robustness_short'])
        if categories['task_engaged_long'] == True:
            session_rois['task_engaged_long'].append(1)
            session_rois['robustness_long'].append(categories['robustness_long'])
        else:
            session_rois['task_engaged_long'].append(0)
            session_rois['robustness_long'].append(categories['robustness_long'])

    if write_to_dict:
        print('writing to dictionary.')
        write_dict(MOUSE, SESSION, session_rois)

def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    NUM_ROIS = [1] #'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = { }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170420_1_Day201783():
    MOUSE = 'LF170420_1'
    SESSION = 'Day201783'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170110_2_Day201748_1():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    SESSION_OPENLOOP = 'Day201748_openloop_1'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170110_2_Day201748_2():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_2'
    SESSION_OPENLOOP = 'Day201748_openloop_2'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170110_2_Day201748_3():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_3'
    SESSION_OPENLOOP = 'Day201748_openloop_3'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170421_2_Day20170719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170719'
    SESSION_OPENLOOP = 'Day20170719_openloop'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170421_2_Day2017720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017720'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170420_1_Day2017719():
    MOUSE = 'LF170420_1'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170222_1_Day201776():
    MOUSE = 'LF170222_1'
    SESSION = 'Day201776'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF170222_1_Day2017615():
    MOUSE = 'LF170222_1'
    SESSION = 'Day2017615'
    SESSION_OPENLOOP = SESSION + '_openloop'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF171212_2_Day2018218_2():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_2'
    SESSION_OPENLOOP = 'Day2018218_openloop_2'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF171211_1_Day2018321_2():
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    SESSION_OPENLOOP = 'Day2018321_openloop_2'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'mouse_session_openloop' : MOUSE+'_'+SESSION_OPENLOOP
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, SESSION_OPENLOOP, NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params)

def run_LF161202_1_Day201729_l23():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l23'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, '', NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params, False)

def run_LF161202_1_Day201729_l5():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l5'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, '', NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params, False)

def run_LF170110_2_Day20170209_l23():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170209_l23'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, '', NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params, False)

def run_LF170110_2_Day20170209_l5():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170209_l5'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, '', NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params, False)

def run_LF170110_1_Day20170215_l23():
    MOUSE = 'LF170110_1'
    SESSION = 'Day20170215_l23'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, '', NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params, False)

def run_LF170110_1_Day20170215_l5():
    MOUSE = 'LF170110_1'
    SESSION = 'Day20170215_l5'
    NUM_ROIS = 'valid' # 'all' #105
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    json_path = loc_info['figure_output_path'] + MOUSE + os.sep + MOUSE + '_' + SESSION + '.json'

    # dictionary that will hold the results of the analyses
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
    }

    SUBNAME = 'space_shuffled'
    subfolder = MOUSE+'_'+SESSION+'_'+SUBNAME
    roi_result_params = run_analysis(MOUSE, SESSION, '', NUM_ROIS, h5path, json_path, SUBNAME, subfolder, roi_result_params, False)

if __name__ == '__main__':

    run_LF170613_1_Day20170804()
    # run_LF170420_1_Day201783()
    # run_LF170110_2_Day201748_1() #*
    # run_LF170110_2_Day201748_2() #*
    # run_LF170110_2_Day201748_3()
    # run_LF170421_2_Day20170719()
    # run_LF170421_2_Day2017720()
    # run_LF170420_1_Day2017719()
    # run_LF170222_1_Day201776()
    # run_LF170222_1_Day2017615()
    # run_LF171212_2_Day2018218_2()
    # run_LF171211_1_Day2018321_2()

    # run_LF161202_1_Day201729_l23()
    # run_LF170110_2_Day20170209_l23()
    # run_LF170110_1_Day20170215_l23()
    # run_LF161202_1_Day201729_l5()
    # run_LF170110_2_Day20170209_l5()
    # run_LF170110_1_Day20170215_l5()
