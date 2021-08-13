"""Generate figures of grating sessions."""

import os
import sys

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + '..' + os.sep + 'Analysis')

import csv
import h5py
import numpy
from ruamel import yaml
from grating_figure import grating_figure
from yaml_mouselist import yaml_mouselist
    
# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read local settings .yaml file.')
    
dF_thresholds = [1.0, 50.0]
    
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
            if 'gratings' in day:
                print('    ' + day)
                
                csv_path = local_settings['figure_output_path']
                
                if not os.path.isdir(csv_path):
                    os.mkdir(csv_path)
                    
                csv_path += os.sep + mouse
                
                if not os.path.isdir(csv_path):
                    os.mkdir(csv_path)
                    
                csv_path += os.sep + 'gratings'
                
                if not os.path.isdir(csv_path):
                    os.mkdir(csv_path)
                    
                csv_filename = csv_path + os.sep + mouse + ' ' + day + ' gOSIs.csv'
                    
                if not os.path.isfile(csv_filename):
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
                    
                    ROIs = list(range(dF_data.shape[1]))
            
                    gOSI_matrix = numpy.zeros((len(ROIs), 2), dtype = complex)
                    
                    for ROI in ROIs:
                        
                        # check if dF data is full of NaNs or contains artefacts
                        if any(numpy.isnan(dF_data[:, ROI])):
                            continue
                        elif any(dF_data[:, ROI] > dF_thresholds[1]):
                            print('        ROI ' + str(ROI) + ': dF/F values exceed 50.')
                            
                            gOSI_matrix[ROI, :] = numpy.nan
                            continue
    #                    elif not any(dF_data[:, ROI] >= dF_thresholds[0]):
    #                        print('        ROI ' + str(ROI) + ': No dF/F values above threshold.')
    #                        
    #                        gOSI_matrix[ROI, :] = numpy.nan
    #                        continue
                        
                        gOSIs = grating_figure(behavioral_data, dF_data[:, ROI], mouse, day, ROI, 'png')
                
                        gOSI_matrix[ROI, :] = gOSIs
            
                    with open(csv_filename, 'w') as csv_file:
                        writer = csv.writer(csv_file, delimiter = ';')
                        
                        writer.writerow(['# ROI', 'gOSI (0.01 cycles/deg)', 'preferred orientation (0.01 cycles/deg)', 'gOSI (0.005 cycles/deg)', 'preferred orientation (0.005 cycles/deg)'])
                                    
                        for ROI in ROIs:
                            writer.writerow([ROI, numpy.linalg.norm(gOSI_matrix[ROI, 0]), numpy.mod(numpy.angle(gOSI_matrix[ROI, 0])*180.0/numpy.pi + 360.0, 360.0), numpy.linalg.norm(gOSI_matrix[ROI, 1]), numpy.mod(numpy.angle(gOSI_matrix[ROI, 1])*180.0/numpy.pi + 360.0, 360.0)])
            
        HDF5_data.close()