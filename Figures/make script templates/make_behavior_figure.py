"""Generate figures of basic behavior."""

import os
import sys

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + 'Analysis')

import h5py
import numpy
from ruamel import yaml
from yaml_mouselist import yaml_mouselist
from behavior_figure import behavior_figure
    
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
                raw_data = numpy.copy(HDF5_data[day + '/raw_data'])
            except KeyError:
                print('        No raw_data.')
                continue
            
            try:
                lick_data = numpy.copy(HDF5_data[day + '/licks_pre_reward'])
            except KeyError:
                print('        No licks_pre_reward.')
                continue
            
            if len(lick_data) == 0:
                print('        No licks detected.')
                continue
            
            try:
                reward_data = numpy.copy(HDF5_data[day + '/rewards'])
            except KeyError:
                print('        No rewards.')
                continue
            
            behavior_figure(raw_data, lick_data, reward_data, group, mouse, day, 'png')
    
        HDF5_data.close()