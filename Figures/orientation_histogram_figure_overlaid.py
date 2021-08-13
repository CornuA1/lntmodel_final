"""Group ROI traces by corresponding grating stimuli parameters and plot."""

import os
import csv
import h5py
import numpy
import scipy
import seaborn
from ruamel import yaml
from matplotlib import pyplot

def orientation_histogram_figure_overlaid(first_mice, second_mice, figure_name, figure_format = 'png', histcolor_1 = 'k', histcolor_2 = 'crimson', legend = None):
    
    orientations = numpy.linspace(0.0, 2.0*numpy.pi, 9)
    spatial_frequencies = numpy.array([0.01, 0.005])
    
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
    
    first_gOSIs = numpy.zeros((1, 2))
    first_orientation_counts = numpy.zeros((1, 2))
    
    first = True

    for m, mouse in enumerate(first_mice):
        print(mouse)
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
    
        for day in days:
            if 'gratings' in day:
                csv_filename = local_settings['figure_output_path'] + os.sep + mouse + os.sep + 'gratings' + os.sep + mouse + ' ' + day + ' gOSIs.csv'
                
                if os.path.isfile(csv_filename):
                    print('    ' + day)
    
                    with open(csv_filename, 'r') as csv_file:
                        reader = csv.reader(csv_file, delimiter = ';')
                        
                        for r, row in enumerate(reader):
                            if len(row) > 0:
                                
                                # skip the header
                                if r == 0:
                                    continue
                                
                                if not numpy.isnan(float(row[1])) and not numpy.isnan(float(row[3])):
                                    if first:
                                        first_gOSIs[0, 0] = float(row[1])
                                        first_gOSIs[0, 1] = float(row[3])
                                        first_orientation_counts[0, 0] = numpy.mod(-float(row[2])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        first_orientation_counts[0, 1] = numpy.mod(-float(row[4])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        
                                        first = False
                                    else:
                                        temp = numpy.zeros((1, 2))
                                        temp[0, 0] = float(row[1])
                                        temp[0, 1] = float(row[3])
                                        first_gOSIs = numpy.append(first_gOSIs, temp, 0)
                                        
                                        temp = numpy.zeros((1, 2))
                                        temp[0, 0] = numpy.mod(-float(row[2])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        temp[0, 1] = numpy.mod(-float(row[4])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        first_orientation_counts = numpy.append(first_orientation_counts, temp, 0)
    
    second_gOSIs = numpy.zeros((1, 2))
    second_orientation_counts = numpy.zeros((1, 2))
    
    first = True

    for m, mouse in enumerate(second_mice):
        print(mouse)
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
    
        for day in days:
            if 'gratings' in day:
                csv_filename = local_settings['figure_output_path'] + os.sep + mouse + os.sep + 'gratings' + os.sep + mouse + ' ' + day + ' gOSIs.csv'
                
                if os.path.isfile(csv_filename):
                    print('    ' + day)
    
                    with open(csv_filename, 'r') as csv_file:
                        reader = csv.reader(csv_file, delimiter = ';')
                    
                        for r, row in enumerate(reader):
                            if len(row) > 0:
                                
                                # skip the header
                                if r == 0:
                                    continue
                                
                                if not numpy.isnan(float(row[1])) and not numpy.isnan(float(row[3])):
                                    if first:
                                        second_gOSIs[0, 0] = float(row[1])
                                        second_gOSIs[0, 1] = float(row[3])
                                        second_orientation_counts[0, 0] = numpy.mod(-float(row[2])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        second_orientation_counts[0, 1] = numpy.mod(-float(row[4])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        
                                        first = False
                                    else:
                                        temp = numpy.zeros((1, 2))
                                        temp[0, 0] = float(row[1])
                                        temp[0, 1] = float(row[3])
                                        second_gOSIs = numpy.append(second_gOSIs, temp, 0)
                                        
                                        temp = numpy.zeros((1, 2))
                                        temp[0, 0] = numpy.mod(-float(row[2])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        temp[0, 1] = numpy.mod(-float(row[4])*numpy.pi/180.0 + 2.0*numpy.pi, 2.0*numpy.pi)
                                        second_orientation_counts = numpy.append(second_orientation_counts, temp, 0)

    mann_whitney_p_values = numpy.zeros(len(spatial_frequencies))
        
    for s_f, spatial_frequency in enumerate(spatial_frequencies):
        temp, mann_whitney_p_values[s_f] = scipy.stats.mannwhitneyu(first_gOSIs[:, s_f], second_gOSIs[:, s_f])
        
        pyplot.subplot(len(spatial_frequencies), 2, len(spatial_frequencies)*s_f + 1)
        
        if s_f == len(spatial_frequencies) - 1:
            pyplot.xlabel('gOSI', fontsize = 20)
            
        if s_f == 0:
            pyplot.title('# of ROIs by gOSI', fontsize = 20)
            
        pyplot.xlim([0.0, 1.0])
        pyplot.ylabel(str(spatial_frequency) + ' cycles/deg\np-value: ' + str(round(mann_whitney_p_values[s_f], 5)), fontsize = 20)
        pyplot.tick_params(reset = 'on', axis = 'both', direction = 'in', length = 4, right = 'off', top = 'off')
        pyplot.gca().spines['top'].set_visible(False)
        pyplot.gca().spines['right'].set_visible(False)
        
        max_height = 0.0
        
        x, bins, histogram = pyplot.hist(first_gOSIs[:, s_f], numpy.linspace(0.0, 1.0, 20), facecolor = histcolor_1, rwidth = 0.85)
        
        for item in histogram:
            item.set_height(item.get_height()/sum(x))
            
        for item in histogram:
            max_height = numpy.amax([max_height, item.get_height()])
            
        x, bins, histogram = pyplot.hist(second_gOSIs[:, s_f], numpy.linspace(0.0, 1.0, 20), facecolor = histcolor_2, rwidth = 0.85, alpha = 0.5)
        
        for item in histogram:
            item.set_height(item.get_height()/sum(x))
            
        for item in histogram:
            max_height = numpy.amax([max_height, item.get_height()])
        
        if legend is not None:
            if s_f == 0:
                pyplot.legend(legend, fontsize = 15)
            
        pyplot.ylim([0.0, 1.2*max_height])
            
        pyplot.axvline(numpy.mean(first_gOSIs[:, s_f]), color = histcolor_1, linestyle = 'dashed')    
        pyplot.axvline(numpy.mean(second_gOSIs[:, s_f]), color = histcolor_2, linestyle = 'dashed')    
        
        pyplot.subplot(len(spatial_frequencies), 2, len(spatial_frequencies)*s_f + 2, projection = 'polar')
            
        if s_f == 0:
            pyplot.title('# of ROIs by preferred orientation', fontsize = 20)
                         
        pyplot.gca().set_rlabel_position(15.0)
        
        pyplot.hist(first_orientation_counts[:, s_f], orientations - (orientations[1] - orientations[0])/2.0, facecolor = histcolor_1, rwidth = 0.5)
        pyplot.hist(second_orientation_counts[:, s_f], orientations - (orientations[1] - orientations[0])/2.0, facecolor = histcolor_2, rwidth = 0.5, alpha = 0.5)
            
        figure_path = local_settings['figure_output_path']
        
    pyplot.suptitle(figure_name, fontsize = 20)
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    # close the figure to save memory
    pyplot.close(figure)
    
    return mann_whitney_p_values