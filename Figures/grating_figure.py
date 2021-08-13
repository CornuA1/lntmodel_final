"""Group ROI traces by corresponding grating stimuli parameters and plot."""

import os
import numpy
import seaborn
from ruamel import yaml
from matplotlib import pyplot

def grating_figure(behavioral_data, dF_data, mouse, day, ROI, figure_format = 'png'):
    
    # set the threshold for classifying a ROI as visually active as a fraction of max dF/F
    active_threshold = 0.1
      
    orientations = numpy.linspace(0.0, 315.0, 8) + 1.0
    spatial_frequencies = numpy.array([0.01, 0.005])
    grating_time = 3.0
    black_box_time = 3.0
    
    figure_name = 'gratings - ' + mouse + ', ' + day + ', ' + 'ROI ' + str(ROI)
    
    figure = pyplot.figure(figsize = (16, 6))
    
    # minimize empty figure space
    figure.subplots_adjust(bottom = 0.1, left = 0.05, right = 0.975, wspace = 0.2, hspace = 0.15)

    seaborn.set_style('white')
    
    # this file contains machine-specific info
    try:
        with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
            local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
    except OSError:
        print('        Could not read local settings .yaml file.')
        return
    
    # check to make sure we're only using the dF data we need
    if len(dF_data.shape) > 1:
        dF_data = dF_data[:, ROI]    
                        
    times = behavioral_data[:, 0]
    
    N = len(times)
    
    # make a list of the grating types
    grating_types = numpy.zeros([len(orientations)*len(spatial_frequencies), 2])
     
    for s_f, spatial_frequency in enumerate(spatial_frequencies):
        for o, orientation in enumerate(orientations):
            grating_types[o + len(orientations)*s_f, :] = [orientation, spatial_frequency] 
            
    # create a counter that increments after each episode of either a grating or black box
    i = -1
    repetition = 0
    black_box = True
    grating = False
    first = True
    
    grating_order = numpy.zeros([N, 3])
    grating_onsets = [False]*N
    
    for t in range(1, N):
        
        # if the trial type changed...
        if behavioral_data[t, 1] != behavioral_data[t - 1, 1]:
            
            # then check if the transition was from black box to grating
            if black_box:
                black_box = False
                grating = True      
                first = True
                
                # indicate that the grating type has changed
                i += 1
                
                if i % grating_types.shape[0] == 0:
                    repetition += 1
                
            # or from grating to black box
            elif grating:
                grating = False
                black_box = True     
           
        # assign the correct grating type and the repetition
        if grating:
            grating_order[t, 0:2] = grating_types[i % grating_types.shape[0], :]
            grating_order[t, 2] = repetition    
    
            # indicate if this is the first index for a given grating type
            if first:
                first = False
                
                grating_onsets[t] = True    
    
    y_min = 0.0
    y_max = 0.0
    
    mean_black_box_responses = numpy.zeros((len(orientations), len(spatial_frequencies)))
    mean_grating_responses = numpy.zeros((len(orientations), len(spatial_frequencies)))
    
    # go through each orientation and spatial frequency and use logical indices to isolate the corresponding dF data
    for o, orientation in enumerate(orientations):
        orientation_mask = grating_order[:, 0] == orientation
        
        for s_f, spatial_frequency in enumerate(spatial_frequencies):
            grating_plot = pyplot.subplot(len(spatial_frequencies), len(orientations) + 1, o + (len(orientations) + 1)*s_f + 1)
            
            spatial_frequency_mask = grating_order[:, 1] == spatial_frequency
            
            # combine logical vectors in list format via list comprehensions
            grating_mask = [orientation_mask[t] and spatial_frequency_mask[t] for t in range(N)]
            
            grating_onset_mask = [grating_mask[t] and grating_onsets[t] for t in range(N)]

            repetitions = len(times[grating_onset_mask])

            trial_times = [[] for repetition in range(repetitions)]
            trial_dF = [[] for repetition in range(repetitions)]
            
            for repetition in range(1, repetitions + 1):
                trial_onset = times[[grating_onset_mask[t] and grating_order[t, 2] == repetition for t in range(N)]]
                
                # now, since we're dealing with numpy arrays, we can just multiply them to combine logical vectors
                indices = [(times >= trial_onset - black_box_time)*(times <= trial_onset + grating_time)]
                
                temp_trial_times = times[indices] - trial_onset
                temp_trial_dF = dF_data[indices]

                # make sure each trial's vectors are the same length so we can average them later. not the most elegant solution, but it works
                if repetition > 1:
                    while len(temp_trial_dF) > len(trial_dF[repetition - 2]):
                        temp_trial_times = numpy.delete(temp_trial_times, -1)
                        temp_trial_dF =  numpy.delete(temp_trial_dF, -1)
                    
                    for other_repetitions in range(repetitions):
                        while len(temp_trial_dF) < len(trial_dF[other_repetitions]):
                            trial_times[other_repetitions] = numpy.delete(trial_times[other_repetitions], -1)
                            trial_dF[other_repetitions] = numpy.delete(trial_dF[other_repetitions], -1)
            
                trial_times[repetition - 1] = temp_trial_times
                trial_dF[repetition - 1] = temp_trial_dF
                
                y_min = numpy.amin([y_min, numpy.amin(temp_trial_dF)])
                y_max = numpy.amax([y_max, numpy.amax(temp_trial_dF)])
                
                grating_plot.plot(temp_trial_times, temp_trial_dF, color = '0.5')
                
            trial_times = numpy.array(trial_times)
            trial_dF = numpy.array(trial_dF)
            
            mean_times = trial_times.mean(0)
            mean_dF = trial_dF.mean(0)
            
            black_box_indices = [(mean_times >= -black_box_time)*(mean_times < 0.0)]
            presentation_indices = [(mean_times >= 0.0)*(mean_times <= grating_time)]
            
            mean_black_box_responses[o, s_f] = mean_dF[black_box_indices].mean()
            mean_grating_responses[o, s_f] = mean_dF[presentation_indices].mean()
            
            grating_plot.plot(mean_times, mean_dF, color = 'k')
            
            grating_plot.axvline(0.0, color = 'crimson', linestyle = 'dashed')
                
            if o == 0:
                grating_plot.set_ylabel(str(spatial_frequency) + ' cycles/deg', fontsize = 20)
                
    # do this before making the polar plot
    axes = figure.get_axes()

    for axis in axes:
        axis.set_xlim([-black_box_time, grating_time])
        axis.set_ylim([y_min, y_max])
        axis.tick_params(reset = 'on', axis = 'both', direction = 'in', length = 4, right = 'off', top = 'off')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
                
    orientations -= 1.0
    orientations *= numpy.pi/180.0
    
    # account for Lukas's angle scheme
    orientations = numpy.mod(-orientations + 2.0*numpy.pi, 2.0*numpy.pi)
                
    gOSIs = numpy.zeros(len(spatial_frequencies), dtype = complex)
    
    active = False
    
    for s_f in range(len(spatial_frequencies)):
        denominator = 0.0
        
        count = 0
        
        for o in range(len(orientations)):
            if numpy.abs(mean_grating_responses[o, s_f] - mean_black_box_responses[o, s_f]) >= active_threshold*numpy.max(dF_data):
                active = True
            
            if mean_grating_responses[o, s_f] >= 0.0:
                gOSIs[s_f] += mean_grating_responses[o, s_f]*numpy.exp(2j*orientations[o])
                denominator += mean_grating_responses[o, s_f]
                
                count += 1
        
        if denominator != 0 and count >= 4:
            gOSIs[s_f] = gOSIs[s_f]/denominator
        elif denominator == 0:
            print('        ROI ' + str(ROI) + ': No positive responses measured.')
            gOSIs[s_f] = numpy.nan
        elif count < 4:
            print('        ROI ' + str(ROI) + ': <50% of responses positive.')
            gOSIs[s_f] = numpy.nan
            
    if not active:
        print('        ROI ' + str(ROI) + ': No response to gratings detected.')
        gOSIs[:] = numpy.nan
    
    for s_f in range(len(spatial_frequencies)):
        polar_plot = pyplot.subplot(len(spatial_frequencies), len(orientations) + 1, (len(orientations) + 1)*(s_f + 1), projection = 'polar')
    
        polar_plot.set_rlim([0.0, 1.0])
        polar_plot.set_rticks([])
        
        for o in range(len(orientations)):
            if mean_grating_responses[o, s_f] < 0.0:
                mean_grating_responses[o, s_f] = 0.0
            
            if mean_grating_responses[:, s_f].max() > 0.0:
                mean_grating_responses[:, s_f] /= mean_grating_responses[:, s_f].max()
        
        # plot it so that the entire line connects
        polar_plot.plot(numpy.append(orientations, orientations[0]), numpy.append(mean_grating_responses[:, s_f], mean_grating_responses[0, s_f]), color = 'k')

    figure.suptitle(figure_name + '\ngOSI: ' + str(round(numpy.linalg.norm(gOSIs[0]), 3)) + ' (' + str(spatial_frequencies[0]) + '), ' + str(round(numpy.linalg.norm(gOSIs[1]), 3)) + ' (' + str(spatial_frequencies[1]) + ')\n', wrap = True, fontsize = 20)

    figure_path = local_settings['figure_output_path']
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + mouse
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + 'gratings'
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    # close the figure to save memory
    pyplot.close(figure)
    
    return gOSIs