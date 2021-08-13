"""Group ROI traces by corresponding grating stimuli parameters and plot."""

import os
import csv
import h5py
import numpy
import seaborn
from ruamel import yaml
from matplotlib import pyplot

def receptive_field_histogram_figure(mice, figure_name, figure_format = 'png', hist_color = '#FFFFFF'):
                                
    # set the minimum number of bin separation required to distinguish two transients
    minimum_bin_separation = 2
    
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
    
    receptive_fields = []

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
                csv_filename = local_settings['figure_output_path'] + os.sep + mouse + os.sep + 'shuffled dF' + os.sep + mouse + ' ' + day + ' shuffle_tests.csv'
    
                if os.path.isfile(csv_filename):
                    print('    ' + day)
            
                    with open(csv_filename, 'r') as csv_file:
                        reader = csv.reader(csv_file, delimiter = ';')
                        
                        ROI = -1
                        
                        for r, row in enumerate(reader):
                            if len(row) > 0:
                                
                                # skip the header
                                if r == 0:
                                    continue
                                
                                if ROI == -1:
                                    ROI = float(row[0])
                
                                    receptive_field = 0.0
                                else:
                                    if float(row[0]) != ROI:
                                        ROI = float(row[0])
                                        
                                        receptive_field = 0.0
                                
                                comparisons = numpy.zeros(len(row[1:]))
                                
                                for c in range(len(row[1:])):
                                    comparisons[c] = float(row[c + 1])
                                
                                count = 0
                                first = 0
                                last = 0
                                bin_size = 5.0
                                
                                for c, comparison in enumerate(comparisons):
                                    if comparison != numpy.nan:
                                        if comparison > 0.0:
                                            if last - first == 0:
                                                count += 1
                                                
                                                first = c
                                                last = c + 1
                                            
                                            last = c + 1
                                        else:
                                            if c - last < minimum_bin_separation:
                                                continue
                                            else:
                                                receptive_field += (last - first)*bin_size
                                                
                                                first = 0
                                                last = 0
                                    else:
                                        if c - last < minimum_bin_separation:
                                            continue
                                        else:
                                            receptive_field += (last - first)*bin_size
                                            
                                            first = 0
                                            last = 0
                                        
                                # this one catches ROI activity that continues to the end of the track
                                receptive_field += (last - first)*bin_size
                                
                                if count > 0:
                                    receptive_field /= count
                                    
                                if receptive_field > 0.0:
                                    receptive_fields.append(receptive_field)
                else:
                    print('        Shuffle tests not found.')
        
    pyplot.xlabel('mean receptive field (cm)', fontsize = 20)
    pyplot.xlim([0.0, numpy.max(receptive_fields)])
    pyplot.tick_params(reset = 'on', axis = 'both', direction = 'in', length = 4, right = 'off', top = 'off')
    pyplot.gca().spines['top'].set_visible(False)
    pyplot.gca().spines['right'].set_visible(False)
    
    max_height = 0.0
    
    x, bins, histogram = pyplot.hist(receptive_fields, numpy.linspace(0.0, numpy.max(receptive_fields), 20), facecolor = hist_color, rwidth = 0.85, normed = True)
    
    for item in histogram:
        item.set_height(item.get_height()/sum(x))
            
    for item in histogram:
        max_height = numpy.amax([max_height, item.get_height()])
        
    pyplot.ylim([0.0, 1.2*max_height])
        
    figure_path = local_settings['figure_output_path']
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    # close the figure to save memory
    pyplot.close(figure)