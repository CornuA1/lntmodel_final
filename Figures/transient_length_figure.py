"""Group ROI traces by corresponding grating stimuli parameters and plot."""

import os
import h5py
import numpy
import seaborn
from ruamel import yaml
from matplotlib import pyplot

def transient_length_figure(mice, figure_name, figure_format = 'png', hist_color = '#FFFFFF'):
                                
    # set the minimum separation required to distinguish two transients
    minimum_transient_separation = 0.0
    
    # set the maximum transient length (in seconds) above which lengths are binned together 
    maximum_transient_length = 4.0
    
    dF_thresholds = [1.0, 50.0]
    
    figure = pyplot.figure(figsize = (12, 10))
    
    # minimize empty figure space
    figure.subplots_adjust(bottom = 0.1, left = 0.1, right = 0.975, wspace = 0.2, hspace = 0.15)

    seaborn.set_style('white')
    
    # this file contains machine-specific info
    try:
        with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
            local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
    except OSError:
        print('        Could not read local settings .yaml file.')
        return
    
    transient_lengths = []

    for m, mouse in enumerate(mice):
        print(mouse)
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
    
        for day in days:
            if 'gratings' not in day and 'openloop' not in day:
                print('    ' + day)
            
                try:
                    behavioral_data = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
                except KeyError:
                    print('        No behaviour_aligned.')
                    continue
                except OSError:
                    print('        Corrupted behaviour_aligned.')
                    continue
                        
                time = behavioral_data[:, 0]
                
                try:
                    dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
                except KeyError:
                    print('        No dF_win.')
                    continue
                
                ROIs = list(range(dF_data.shape[1]))
                
                for ROI in ROIs:
                    transient_length = 0.0
                    
                    # check if dF data is full of NaNs or contains artefacts
                    if any(numpy.isnan(dF_data[:, ROI])):
                        continue
                    elif any(dF_data[:, ROI] > dF_thresholds[1]):
                        print('        ROI ' + str(ROI) + ': dF/F values exceed 50.')
                        continue
                    
                    # average across shuffles for each spatial bin                        
                    threshold = dF_data[:, ROI].mean(axis = 0) + 3.0*dF_data[:, ROI].std(axis = 0)
                        
                    comparisons = dF_data[:, ROI] - threshold
                    
                    count = 0
                    first = 0
                    last = 0
                    
                    for c, comparison in enumerate(comparisons):
                        if comparison != numpy.nan:
                            if comparison > 0.0:
                                if last - first == 0:
                                    count += 1
                                    
                                    first = c
                                    last = c + 1
                                
                                last = c + 1
                                
                                if last > len(time) - 1:
                                    last = len(time) - 1
                            else:
                                if time[c] - time[last] < minimum_transient_separation:
                                    continue
                                else:
                                    transient_length += time[last] - time[first]
                                    
                                    first = 0
                                    last = 0
                        else:
                            if time[c] - time[last] < minimum_transient_separation:
                                continue
                            else:
                                transient_length += time[last] - time[first]
                                
                                first = 0
                                last = 0
                            
                    # this one catches ROI activity that continues to the end of the track
                    transient_length += time[last] - time[first]
                    
                    if count > 0:
                        transient_length /= count
                        
                    if transient_length > maximum_transient_length:
                        transient_length = maximum_transient_length
                        
                    if transient_length > 0.0:
                        transient_lengths.append(transient_length)
        
    pyplot.xlabel('transient length (s)', fontsize = 20)
    pyplot.xlim([0.0, maximum_transient_length])
    pyplot.tick_params(reset = 'on', axis = 'both', direction = 'in', length = 4, right = 'off', top = 'off')
    pyplot.gca().spines['top'].set_visible(False)
    pyplot.gca().spines['right'].set_visible(False)
    
    max_height = 0.0
    
    x, bins, histogram = pyplot.hist(transient_lengths, numpy.linspace(0.0, maximum_transient_length, 40), facecolor = hist_color, rwidth = 0.85, normed = True)
    
    for item in histogram:
        item.set_height(item.get_height()/sum(x))
            
    for item in histogram:
        max_height = numpy.amax([max_height, item.get_height()])
        
    pyplot.ylim([0.0, 1.2*max_height])
        
    pyplot.axvline(numpy.mean(transient_lengths), color = hist_color, linestyle = 'dashed')    
        
    figure_path = local_settings['figure_output_path']
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    # close the figure to save memory
    pyplot.close(figure)