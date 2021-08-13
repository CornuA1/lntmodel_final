"""Compare basic behavior of a given animal across multiple days."""

import os
import yaml
import h5py
import numpy
import seaborn
from matplotlib import pyplot

def crossday_behavior_figure(HDF5_path, mouse, figure_format = 'png'):
    
    landmark_zone = 200.0
            
    figure_name = 'crossday_behavior - ' + mouse
    
    figure = pyplot.figure(figsize = (16, 10))

    seaborn.set_style('white')
    
    # this file contains machine-specific info
    try:
        with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
            local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
    except OSError:
        print('        Could not read local settings .yaml file.')
        return
    
    try:
        HDF5_data = h5py.File(HDF5_path, 'r')
    except:
        print('        No HDF5 file.')
        return
    
    days = [day for day in HDF5_data]
    
    to_excise = []
    
    # check for weird days
    for day in days:
        print('    ' + day)
        
        try:
            raw_data = numpy.copy(HDF5_data['/' + day + '/raw_data'])
            try:
                raw_data[:, 6]
            except:
                print('        Unexpected data format.')
                to_excise.append(day)
                continue
        except:
            print('        No raw_data.')
            to_excise.append(day)
            continue
        try:
            lick_data = numpy.copy(HDF5_data['/' + day + '/licks_pre_reward'])
        except:
            print('        No licks_pre_reward.')
            to_excise.append(day)
            continue
        
    for day in to_excise:
        days.remove(day)
        
    # only analyze if there is more than one day
    if len(days) < 1:
        return
    
    time = pyplot.subplot(2, 3, 1)
    trials = pyplot.subplot(2, 3, 2)
    distance = pyplot.subplot(2, 3, 4)
    rewards = pyplot.subplot(2, 3, 5)
    difference = pyplot.subplot(2, 3, 6)
    
    blackbox = numpy.empty((len(days)))
    track = numpy.empty((len(days)))
    number_of_trials = numpy.empty((len(days)))
    total_distance = numpy.empty((len(days)))
    successful_rewards = numpy.empty((len(days)))
    unsuccessful_rewards = numpy.empty((len(days)))
    mean_lick_difference = numpy.empty((len(days)))
    
    for d, day in enumerate(days):
        raw_data = numpy.copy(HDF5_data['/' + day + '/raw_data'])
        lick_data = numpy.copy(HDF5_data['/' + day + '/licks_pre_reward'])
        
        # time in blackbox and on track
        blackbox[d] = numpy.sum(raw_data[raw_data[:, 4] == 5, 2])
        track[d] = numpy.sum(raw_data[raw_data[:, 4] != 5, 2])
        
        # number of trials ran
        number_of_trials[d] = len(numpy.unique(raw_data[raw_data[:, 4] != 5, 6]))
        
        # total distance
        temp_distance = numpy.diff(raw_data[:, 1])
        total_distance[d] = numpy.sum(temp_distance[temp_distance > 0.0])/100.0
        
        # number of rewards (successful and unsuccessful)
        temp_rewards = numpy.diff(raw_data[:, 5])
        successful_rewards[d] = len(temp_rewards[temp_rewards == 1])
        unsuccessful_rewards[d] = len(temp_rewards[temp_rewards == 2])
        
        if numpy.size(lick_data) > 0:
            
            # only plot trials where a lick was detected
            lick_trials = numpy.unique(lick_data[:, 2])
        
            # plot location of first trials on short and long trials
            first_lick_short = []
            first_lick_short_trials = []
            first_lick_long = []
            first_lick_long_trials = []
            
            for trial in lick_trials:
                licks_all = lick_data[lick_data[:, 2] == trial, :]
                
                # isolate licks occurring after the landmark
                licks_all = licks_all[licks_all[:, 1] > landmark_zone + 40.0, :]
                if licks_all.shape[0] > 0:
                    if licks_all[0, 3] == 3:
                        first_lick_short.append(licks_all[0, 1])
                        first_lick_short_trials.append(trial)
                    elif licks_all[0, 3] == 4:
                        first_lick_long.append(licks_all[0, 1])
                        first_lick_long_trials.append(trial)

            mean_lick_difference[d] = numpy.abs(numpy.mean(first_lick_short) - numpy.mean(first_lick_long))
        else:
            mean_lick_difference[d] = numpy.NaN
    
    HDF5_data.close()
    
    # modify the day strings for plotting
    for d, day in enumerate(days):
        days[d] = days[d].replace('Day', '')
        temp = days[d][4:6] + '/' + days[d][6:8] + '/' + days[d][0:4]
        days[d] = days[d].replace(days[d][0:8], temp)
    
    time.plot(blackbox)   
    time.plot(track)
    if len(blackbox) > 9:
        time.set_xticks(numpy.linspace(0, len(blackbox), 10))
    else:
        time.set_xticks(numpy.linspace(0, len(blackbox), len(blackbox)))
    time.set_xticklabels(days)
    time.set_ylabel('time (s)', fontsize = 20)
    time.legend(['blackbox', 'track'], fontsize = 15)
    
    trials.bar(numpy.arange(len(number_of_trials)), number_of_trials, align = 'center')
    if len(number_of_trials) > 9:
        trials.set_xticks(numpy.linspace(0, len(number_of_trials),10))
    else:
        trials.set_xticks(numpy.linspace(0, len(number_of_trials), len(number_of_trials)))
    trials.set_xticklabels(days)
    trials.set_ylabel('number of trials', fontsize = 20)
    
    distance.plot(total_distance)
    if len(total_distance) > 9:
        distance.set_xticks(numpy.linspace(0, len(total_distance), 10) + 0.5)
    else:
        distance.set_xticks(numpy.linspace(0, len(total_distance), len(total_distance)) + 0.5)
    distance.set_xticklabels(days)
    distance.set_ylabel('total distance covered (m)', fontsize = 20)
    
    rewards.plot(successful_rewards)
    rewards.plot(unsuccessful_rewards)
    if len(unsuccessful_rewards) > 9:
        rewards.set_xticks(numpy.linspace(0, len(unsuccessful_rewards), 10))
    else:
        rewards.set_xticks(numpy.linspace(0, len(unsuccessful_rewards), len(unsuccessful_rewards)))
    rewards.set_xticklabels(days)
    rewards.set_ylabel('number of rewards', fontsize = 20)
    rewards.legend(['successful', 'unsuccessful'], fontsize = 15)
    
    difference.plot(mean_lick_difference)
    if len(mean_lick_difference) > 9:
        difference.set_xticks(numpy.linspace(0, len(mean_lick_difference), 10))
    else:
        difference.set_xticks(numpy.linspace(0, len(mean_lick_difference), len(mean_lick_difference)))
    difference.set_xticklabels(days)
    difference.set_ylabel('difference of first lick location (cm)', fontsize = 20)
    
    axes = figure.get_axes()
    
    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    
    # autoformat the figure for dates
    figure.autofmt_xdate()
    
    # minimize empty figure space
    figure.subplots_adjust(bottom = 0.125, left = 0.1, right = 0.95, wspace = 0.25, hspace = 0.075)
        
    figure.suptitle(figure_name, wrap = True, fontsize = 20)
    
    figure_path = local_settings['figure_output_path']
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + mouse
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + 'population vector'
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    # close the figure to save memory
    pyplot.close(figure)