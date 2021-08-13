"""Create and save figures of basic behavior of all animals for all days."""

import sys
import os

# load local settings file
sys.path.append('../Analysis')
sys.path.append('../Figures')

os.chdir('../Analysis')

from trial_onset_figure import trial_onset_figure
from yaml_mouselist import yaml_mouselist
import numpy
import h5py
import yaml

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    content = yaml.load(f)
    
mice = yaml_mouselist(['GCAMP6f_A30_ALL', 'GCAMP6f_A30_RBP4', 'GCAMP6f_A30_V1'])

HDF_list = []

for m, mouse in enumerate(mice):
    print(mouse)
    
    HDF_list.append(content['imaging_dir'] + mouse + os.sep + mouse + '.h5')
    
    try:
        HDF5_data = h5py.File(HDF_list[m], 'r')
    except:
        print('No HDF5 file.')
        continue
    
    days = [day for day in HDF5_data]
    
    unique_days = []
    
    for day in days:
        cropped_days = day.split('_')[0]
        if cropped_days not in unique_days:
            unique_days.append(cropped_days)

    for day in days:
        print('    ' + day)
        
        try:
            dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
        except:
            print(day + ': No dF_win.')
            continue
        
        HDF5_data.close()
        
        ROIs = list(range((numpy.size(dF_data, 1))))
        
        for ROI in ROIs:        
            trial_onset_figure(content['imaging_dir'] + mouse + os.sep + mouse + '.h5', day, ROI, mouse + ', ' + day + ', ' + str(ROI), 'png')