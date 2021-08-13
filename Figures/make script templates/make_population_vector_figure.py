"""Generate figures of population vectors and their correlations."""

import os
import sys

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + '..' + os.sep + 'Analysis')

import h5py
import numpy
from ruamel import yaml
from filter_trials import filter_trials
from yaml_mouselist import yaml_mouselist
from population_vector_figure import population_vector_figure

# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read local settings .yaml file.')
    
groups = ['GCAMP6f_A30_ALL', 'GCAMP6f_A30_RBP4', 'GCAMP6f_A30_V1']

for group in groups:
    mice = yaml_mouselist([group])

    for m, mouse in enumerate(mice):
        print(mouse)
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
    
        for day in days:
            print('    ' + day)
        
            try:
                behavioral_data = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
            except KeyError:
                print('        No behaviour_aligned.')
                continue
            
            try:
                dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
            except KeyError:
                print('        No dF_win.')
                continue
    
            if dF_data.shape[1] == 1:
                print('        Only one ROI detected.')
                continue
        
            try:
                short_trials = filter_trials(behavioral_data, [], ['tracknumber', 3])
                long_trials = filter_trials(behavioral_data, [], ['tracknumber', 4])
            except IndexError:
                print('        Unexpected track number format.')
                continue
            
            if len(short_trials) == 0 and len(long_trials) == 0:
                print('        No short or long trials detected.')
                continue
                
            population_vector_figure(behavioral_data, dF_data, group, mouse, day, 'png')
    
        HDF5_data.close()