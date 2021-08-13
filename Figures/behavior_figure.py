"""Analyze basic behavior of a given animal on a given day."""

import os
import numpy
import seaborn
from ruamel import yaml
from scipy import stats
from matplotlib import pyplot
from filter_trials import filter_trials
# test adsfasdf
# git test change
# 1
def behavior_figure(raw_data, lick_data, reward_data, group, mouse, day, figure_format = 'png'):

    lick_bin_size = 10.0
    speed_bin_size = 5.0

    track_types = ['short', 'long']

    track_info = {track_type: {} for track_type in track_types}
    track_trials = {track_type: [] for track_type in track_types}
    analysis = {track_type: {'first_lick': [], 'first_lick_trials': [], 'lick_distances': [], 'lick_distribution': [], 'speed_distances': [], 'mean_speed': [], 'speed_standard_error': []} for track_type in track_types}
    plots = {track_type: {'raster': [], 'lick_histogram': [], 'speed_histogram': []} for track_type in track_types}

    track_info['short']['track_number'] = 3
    track_info['short']['landmark_zone'] = [200.0, 240.0]
    track_info['short']['reward_zone'] = [320.0, 340.0]
    track_info['short']['track_length'] = 340.0 + numpy.amax([lick_bin_size, speed_bin_size])

    track_info['long']['track_number'] = 4
    track_info['long']['landmark_zone'] = [200.0, 240.0]
    track_info['long']['reward_zone'] = [380.0, 400.0]
    track_info['long']['track_length'] = 400.0 + numpy.amax([lick_bin_size, speed_bin_size])

    figure_name = 'behavior - ' + mouse + ', ' + day

    figure = pyplot.figure(figsize = (16, 10))

    # minimize empty figure space
    figure.subplots_adjust(bottom = 0.05, left = 0.05, right = 0.975, wspace = 0.2, hspace = 0.25)

    seaborn.set_style('white')

    # this file contains machine-specific info
    try:
        with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
            local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
    except OSError:
        print('        Could not read project .yaml file.')
        return

    if len(lick_data) < 1:
        print('        No licks detected.')
        return

    if len(reward_data) < 1:
        print('        No rewards detected.')
        return

    # find all trials containing either licks or rewards
    analysis['scatter_rowlist_map'] = numpy.union1d(numpy.unique(lick_data[:, 2]), numpy.unique(reward_data[:, 3]) - 1)

    first_lick = []
    lick_trial_start = []
    max_licks = 0.0
    max_speed = 0.0

    for t_t, track_type in enumerate(track_types):
        track_number = track_info[track_type]['track_number']
        landmark_zone = track_info[track_type]['landmark_zone']
        reward_zone = track_info[track_type]['reward_zone']
        track_length = track_info[track_type]['track_length']

        try:
            track_trials[track_type] = trials = filter_trials(raw_data, [], ['tracknumber', track_number])
        except IndexError:
            print('        Unexpected track number format.')
            return

        if len(trials) == 0:
            print('        No ' + track_type + ' trials detected.')
            return

        plots[track_type]['raster'] = raster = pyplot.subplot(len(track_types) + 1, len(track_types), t_t + 1)
        plots[track_type]['lick_histogram'] = lick_histogram = pyplot.subplot(len(track_types) + 1, len(track_types), t_t + len(track_types) + 1)
        plots[track_type]['speed_histogram'] = speed_histogram = pyplot.subplot(len(track_types) + 1, len(track_types), t_t + 2*len(track_types) + 1)

        # plot landmark and rewarded area as shaded zones
        raster.axvspan(landmark_zone[0], landmark_zone[1], color = '0.9', zorder = 1)
        raster.axvspan(reward_zone[0], reward_zone[1], color = '#D2F2FF', zorder = 1)

        lick_histogram.axvspan(landmark_zone[0], landmark_zone[1], color = '0.9', zorder = 1)
        lick_histogram.axvspan(reward_zone[0], reward_zone[1], color = '#D2F2FF', zorder = 1)

        speed_histogram.axvspan(landmark_zone[0], landmark_zone[1], color = '0.9', zorder = 1)
        speed_histogram.axvspan(reward_zone[0], reward_zone[1], color = '#D2F2FF', zorder = 1)

        scatter_rowlist_map = numpy.intersect1d(analysis['scatter_rowlist_map'], trials)
        scatter_rowlist = numpy.arange(len(scatter_rowlist_map))

        licks_N = int(track_length/lick_bin_size)
        speed_N = int(track_length/speed_bin_size)

        speed_distances = numpy.linspace(speed_bin_size/2.0, track_length - speed_bin_size/2.0, speed_N)

        lick_distribution = numpy.array([])
        speed = numpy.empty((len(scatter_rowlist_map), speed_N))
        speed.fill(numpy.NaN)

        # scatterplot of licks and rewards in order of trial number
        for t, trial in enumerate(scatter_rowlist_map):
            current_trial_location = raw_data[raw_data[:, 6] == trial, 1]
            current_trial_speed = raw_data[raw_data[:, 6] == trial, 3]
            current_trial_licks = lick_data[lick_data[:, 2] == trial, 1]
            current_trial_rewards = reward_data[reward_data[:, 3] - 1 == trial, 1]
            current_trial_start = int(current_trial_location[0]/speed_bin_size)
            current_trial_end = int(current_trial_location[-1]/speed_bin_size)

            if reward_data[reward_data[:, 3] - 1 == trial, 5] == 1:
                color = '#00C40E'
            else:
                color = 'r'

            if len(current_trial_licks) > 0:
                lick_trials = numpy.full(len(current_trial_licks), scatter_rowlist[t])
                raster.scatter(current_trial_licks, lick_trials, c = 'k', lw = 0)
            if len(current_trial_rewards) > 0:
                reward_trials = scatter_rowlist[t]
                raster.scatter(current_trial_rewards, reward_trials, c = color, lw = 0)

            start_trials = scatter_rowlist[t]
            raster.scatter(current_trial_start, start_trials, c = 'b', marker = '>', lw = 0)

            # make a histogram of the licks
            lick_distribution = numpy.append(lick_distribution, current_trial_licks)

            # isolate licks occurring after the landmark
            current_trial_licks = current_trial_licks[current_trial_licks > landmark_zone[1]]

            if len(current_trial_licks) > 0:
                first_lick.append(reward_zone[0] - current_trial_licks[0])
                lick_trial_start.append(current_trial_location[0])

                analysis[track_type]['first_lick'].append(current_trial_licks[0])
                analysis[track_type]['first_lick_trials'].append(trial)

            # plot running speed
            while current_trial_end < current_trial_start:
                current_trial_location = current_trial_location[:-1]
                current_trial_speed = current_trial_speed[:-1]
                current_trial_end = int(current_trial_location[-1]/speed_bin_size)

            # skip trials that are too short, where data ends before the reward zone, or where data extends beyond the track length
            if current_trial_location[-1] < reward_zone[0] or current_trial_location[-1] - current_trial_location[0] < 100.0 or current_trial_end > speed_N:
                continue

            speed_trial = stats.binned_statistic(current_trial_location, current_trial_speed, 'mean', (current_trial_end - current_trial_start) + 1, (current_trial_start*speed_bin_size, current_trial_end*speed_bin_size))[0]
            speed[t, current_trial_start:current_trial_end + 1] = speed_trial
            max_speed = numpy.amax([max_speed, numpy.amax(speed_trial)])

        if len(lick_distribution) > 0:
            lick_distribution.sort()

            analysis[track_type]['lick_distances'] = numpy.linspace(0, track_length, licks_N)
            analysis[track_type]['lick_distribution'] = lick_distribution = stats.binned_statistic(lick_distribution, numpy.linspace(0, track_length, len(lick_distribution)), 'count', licks_N)[0]

            max_licks = numpy.amax([max_licks, numpy.amax(lick_distribution)])
        else:
            print('        No ' + track_type + ' licks detected.')
            return

        # remove "bad" trials
        speed = speed[~numpy.isnan(speed).all(1), :]

        # exclude spatial bins with NaNs
        indices = ~numpy.isnan(speed).any(0)
        analysis[track_type]['speed_distances'] = speed_distances[indices]
        analysis[track_type]['mean_speed'] = speed[:, indices].mean(axis = 0)
        analysis[track_type]['speed_standard_error'] = stats.sem(speed[:, indices], 0)

    distribution = pyplot.subplot(len(track_types) + 1, len(track_types), len(track_types)**2 - 2)
    firsts = pyplot.subplot(len(track_types) + 1, len(track_types), len(track_types)**2 - 1)

    axes = figure.get_axes()

    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.tick_params(reset = 'on', axis= 'both', direction = 'in', length = 4, right = 'off', top='off')

    firsts.scatter(lick_trial_start, first_lick)
    firsts.set_xlim([50, 150])
    firsts.set_xlabel('starting position (cm)', fontsize = 15)
    firsts.set_ylabel('first lick distance from reward zone (cm)', fontsize = 10)

    for t_t_1, track_type_1 in enumerate(track_types):
        landmark_zone = track_info[track_type_1]['landmark_zone']
        reward_zone = track_info[track_type_1]['reward_zone']
        track_length = track_info[track_type_1]['track_length']

        lick_distances = analysis[track_type_1]['lick_distances']
        lick_distribution = analysis[track_type_1]['lick_distribution']
        speed_distances = analysis[track_type_1]['speed_distances']
        mean_speed = analysis[track_type_1]['mean_speed']
        speed_standard_error = analysis[track_type_1]['speed_standard_error']

        raster_1 = plots[track_type_1]['raster']
        lick_histogram_1 = plots[track_type_1]['lick_histogram']
        speed_histogram_1 = plots[track_type_1]['speed_histogram']

        lick_histogram_1.plot(lick_distances, lick_distribution)
        speed_histogram_1.plot(speed_distances, mean_speed, c = 'g')
        speed_histogram_1.fill_between(speed_distances, mean_speed - speed_standard_error, mean_speed + speed_standard_error, color = 'g', alpha = 0.2)

        for t_t_2, track_type_2 in enumerate(track_types):
            if t_t_2 != t_t_1:
                plots[track_type_2]['lick_histogram'].plot(lick_distances, lick_distribution, c = '0.7')
                plots[track_type_2]['speed_histogram'].plot(speed_distances, mean_speed, c = '0.7')

        raster_1.set_xlim([0.0, track_length])
        lick_histogram_1.set_xlim([0.0, track_length])
        lick_histogram_1.set_ylim([0.0, 1.2*max_licks])
        speed_histogram_1.set_xlim([0.0, track_length])
        speed_histogram_1.set_ylim([0.0, max_speed])

        raster_1.set_title(track_type_1, fontsize = 20)

        speed_histogram_1.set_xlabel('location (cm)', fontsize = 15)
        speed_histogram_1.set_xticklabels(str(list(range(0, int(track_length), 50)))[1:-1].split(', '))

        if t_t_1 == 0:
            raster_1.set_ylabel('trial', fontsize = 15)
            lick_histogram_1.set_ylabel('licks', fontsize = 15)
            speed_histogram_1.set_ylabel('running speed (cm/sec)', fontsize = 15)

    distribution.scatter(analysis['short']['first_lick'], analysis['short']['first_lick_trials'], c = seaborn.xkcd_rgb['windows blue'], lw = 0)
    distribution.scatter(analysis['long']['first_lick'], analysis['long']['first_lick_trials'], c = seaborn.xkcd_rgb['dusty purple'], lw = 0)
    distribution_twin = distribution.twinx()

    if len(analysis['short']['first_lick']) > 0:
        seaborn.kdeplot(numpy.asarray(analysis['short']['first_lick']), c = seaborn.xkcd_rgb['windows blue'], label = 'first lick (short)', shade = True, ax = distribution_twin)
    if len(analysis['long']['first_lick']) > 0:
        seaborn.kdeplot(numpy.asarray(analysis['long']['first_lick']), c = seaborn.xkcd_rgb['dusty purple'], label = 'first lick (long)', shade = True, ax = distribution_twin)

    distribution.set_xlim([track_info['short']['landmark_zone'][0], track_info['long']['track_length']])
    distribution.set_ylim(ymin = 0)
    distribution_twin.set_ylim(ymin = 0.0)
    distribution.set_xlabel('location (cm)', fontsize = 15)
    distribution.set_ylabel('trial', fontsize = 15)
    distribution_twin.set_ylabel('KDE of first licks', fontsize = 15)

    temp, p_value = stats.mannwhitneyu(analysis['short']['first_lick'], analysis['long']['first_lick'])

    if p_value < 0.005:
        distribution.annotate('p-value: <0.005', xy = (203, 5), fontsize = 10)
    else:
        distribution.annotate('p-value: ' + str(numpy.round(p_value, 5)), xy = (203, 5), fontsize = 10)

    median_difference = numpy.abs(numpy.median(analysis['long']['first_lick']) - numpy.median(analysis['short']['first_lick']))
    distribution.annotate('median difference: ' + str(round(median_difference, 1)) + ' cm', xy = (203, 15), fontsize = 10)

    # do this separately and after all of the other axes have been formatted
    distribution_twin.spines['top'].set_visible(False)
    distribution_twin.spines['left'].set_visible(False)
    distribution_twin.tick_params(axis = 'both', direction = 'in', length = 4, left = 'off', top = 'off')

    task_score = round(float(numpy.abs(numpy.mean(analysis['short']['first_lick']) - numpy.mean(analysis['long']['first_lick']))), 3)

    figure_path = local_settings['figure_output_path']

    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)

    figure_path += os.sep + mouse

    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)

    figure_path += os.sep + 'behavior'

    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)

    figure.suptitle(figure_name + '\nshort trials: ' + str(len(track_trials['short'])) + ' | ' + 'long trials: ' + str(len(track_trials['long'])) + ' | ' + 'task_score: ' + str(task_score), wrap = True, fontsize = 25)

    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)

    # close the figure to save memory
    pyplot.close(figure)

    if not numpy.isnan(task_score):
        try:
            with open(local_settings['yaml_file'], 'r') as yaml_file:
                project_yaml = yaml.round_trip_load(yaml_file, preserve_quotes = True)
        except OSError:
            print('        Could not read project .yaml file.')
            return

        # the yaml file nests data as dictionaries within lists
        modified_yaml = project_yaml.copy()

        modified_yaml = modified_yaml[group]

        for m in range(len(modified_yaml)):
            if mouse in modified_yaml[m]:
                modified_yaml = modified_yaml[m]
                modified_yaml = modified_yaml[mouse]
                break

        try:
            modified_yaml[0]['date']
        except KeyError:
            print('        No session information in project .yaml file.')
            return
        except TypeError:
            print('        No session information in project .yaml file.')
            return

        temp = int(day.replace('Day', '').replace('_openloop', '').replace('_gratings', ''))

        for d in range(len(modified_yaml)):
            if temp == modified_yaml[d]['date']:
                if 'openloop' in day:
                    if modified_yaml[d]['rectype'] == 'OPENLOOP':
                        break
                elif 'gratings' in day:
                    if modified_yaml[d]['rectype'] == 'GRATINGS':
                        break
                else:
                    break
            elif d == len(modified_yaml) - 1:
                print('        Day not found in project .yaml file.')
                return

        modified_yaml = modified_yaml[d]

        modified_yaml['tscore'] = task_score

        project_yaml[group][m][mouse][d] = modified_yaml

        try:
            with open(local_settings['yaml_file'], 'w') as yaml_file:
                yaml.round_trip_dump(project_yaml, yaml_file, default_flow_style = False)
        except OSError:
            print('        Could not modify project .yaml file.')
            return
