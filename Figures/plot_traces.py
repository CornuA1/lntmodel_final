"""Generate figures of shuffled dF tests."""

import os
import csv
import h5py
import numpy
from ruamel import yaml
from matplotlib import pyplot

# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read local settings .yaml file.')

dF_threshold = 1.0
    
#groups = ['L2-3', 'L5', 'V1']
groups = ['V1']

for group in groups:
    if group == 'L2-3':
        mice = ['LF171204_1', 'LF171204_2']
    if group == 'L5':
        mice = ['LF171212_1', 'LF171212_2']
    if group == 'V1':
        mice = ['LF171207_1', 'LF171211_1', 'LF171211_2']
        mice = ['LF171211_2']
    
    for m, mouse in enumerate(mice):
        print(mouse)
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
        days = ['Day2018122_dark']
    
        for day in days:
            if 'dark' in day:
                print('    ' + day)
                
                try:
                    time = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
                except KeyError:
                    print('        No behaviour_aligned.')
                    continue
                
                time = time[:, 0]
            
                try:
                    dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
                except KeyError:
                    print('        No dF_win.')
                    continue
                
                ROIs = list(range(dF_data.shape[1]))
                ROIs = [4, 7, 8]                 
                
                for ROI in ROIs:
                    pyplot.figure(figsize = (8, 8), facecolor = 'white')
                    
                    pyplot.xlabel('time (s)', fontsize = 20)
                    pyplot.ylabel('dF/F', fontsize = 20)
                    pyplot.xlim([time[7000], time[9000]])
                    pyplot.tick_params(axis = 'x', which = 'both', top = 'off', labeltop = 'off')
                    pyplot.tick_params(axis = 'y', which = 'both', right = 'off', labelright = 'off')
                    pyplot.tick_params(axis = 'both', which = 'major', labelsize = 15)
                    pyplot.gca().spines['top'].set_visible(False)
                    pyplot.gca().spines['right'].set_visible(False)
                    
                    pyplot.plot(time[7000:9000], dF_data[7000:9000, ROI], color = 'k', linewidth = 3.0)
                                    
                    figure_path = local_settings['figure_output_path']
                    
                    if not os.path.isdir(figure_path):
                        os.mkdir(figure_path)
                        
                    figure_path += os.sep + mouse
                    
                    if not os.path.isdir(figure_path):
                        os.mkdir(figure_path)
                        
                    figure_name = 'dF trace - ' + mouse + ', ' + day + ', ' + 'ROI ' + str(ROI)
                    
                    pyplot.gcf().savefig(figure_path + os.sep + figure_name + '.' + 'svg', format = 'svg')
                    
                    pyplot.gcf().canvas.draw()
                    pyplot.gcf().canvas.flush_events()
                    
                    # close the figure to save memory
                    pyplot.close(pyplot.gcf())
            
        HDF5_data.close()