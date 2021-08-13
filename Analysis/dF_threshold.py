"""Generate figures of shuffled dF tests."""

import os
import csv
import h5py
import numpy
from ruamel import yaml

# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read local settings .yaml file.')

dF_threshold = 1.0
    
groups = ['L2-3', 'L5', 'V1']

for group in groups:
    if group == 'L2-3':
        mice = ['LF171204_1', 'LF171204_2']
    if group == 'L5':
        mice = ['LF171212_1', 'LF171212_2']
    if group == 'V1':
        mice = ['LF171207_1', 'LF171211_1', 'LF171211_2']
    
    for m, mouse in enumerate(mice):
        print(mouse)
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
    
        for day in days:
            if 'gratings' in day:
                print('    ' + day)
            
                try:
                    dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
                except KeyError:
                    print('        No dF_win.')
                    continue
                
                ROIs = list(range(dF_data.shape[1]))
        
                with open(local_settings['figure_output_path'] + os.sep + mouse + ' ' + day + ' dF_threshold.csv', 'w') as csv_file:
                    writer = csv.writer(csv_file, delimiter = ';')
                    
                    writer.writerow(['# ROI', 'above threshold'])                    
                
                    for ROI in ROIs:
                        if any(dF_data[:, ROI] >= dF_threshold):
                            writer.writerow([ROI, True])
                        else:
                            print('        ROI ' + str(ROI) + ': No dF/F values above threshold.')
                            
                            writer.writerow([ROI, False])
            
        HDF5_data.close()