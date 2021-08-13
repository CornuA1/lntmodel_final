"""Generate figures of shuffled dF tests."""

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
from filter_trials import filter_trials
from yaml_mouselist import yaml_mouselist
from shuffled_dF_figure import shuffled_dF_figure

# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read local settings .yaml file.')
    
groups = ['GCAMP6f_A30_ALL', 'GCAMP6f_A30_RBP4', 'GCAMP6f_A30_V1']
                
# these are the centers of the spatial bins
distances = numpy.linspace(5.0/2.0, 405.0 - 5.0/2.0, int(405.0/5.0))
                
header = ['# ROI']

for distance in distances:
    header.append(str(distance) + ' cm')

for group in groups:
    mice = yaml_mouselist([group])
    
    for m, mouse in enumerate(mice):
        mouse = 'LF171212_2'
        
        print(mouse)
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
    
        for day in days:
            day = 'Day2018218_2'
            
            if 'gratings' not in day:
                print('    ' + day)
                    
                csv_path = local_settings['figure_output_path']
                
                if not os.path.isdir(csv_path):
                    os.mkdir(csv_path)
                    
                csv_path += os.sep + mouse
                
                if not os.path.isdir(csv_path):
                    os.mkdir(csv_path)
                    
                csv_path += os.sep + 'shuffled dF'
                
                if not os.path.isdir(csv_path):
                    os.mkdir(csv_path)
                    
                csv_filename = csv_path + os.sep + mouse + ' ' + day + ' shuffle_tests.csv'
                    
                if not os.path.isfile(csv_filename):
                    try:
                        behavioral_data = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
                    except KeyError:
                        print('        No behaviour_aligned.')
                        continue
                    except OSError:
                        print('        Corrupted behaviour_aligned.')
                        continue
                    
                    try:
                        dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
                    except KeyError:
                        print('        No dF_win.')
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
                    
                    ROIs = list(range(dF_data.shape[1]))
                    
                    with open(csv_filename, 'w') as csv_file:
                        writer = csv.writer(csv_file, delimiter = ';')
                        
                        writer.writerow(header)
                        
                        pre_landmark_ROIs = {'short': [], 'long': []}
                        landmark_ROIs = {'short': [], 'long': []}
                        path_integration_ROIs = {'short': [], 'long': []}
                        reward_ROIs = {'short': [], 'long': []}
                        
                        for ROI in ROIs:
                            comparisons = shuffled_dF_figure(behavioral_data, dF_data[:, ROI], group, mouse, day, ROI, 'png')
                    
                            if comparisons is not None:
                                for track_type in ['short', 'long']:
                                    row = [ROI]
                                    
                                    for comparison in comparisons[track_type]:
                                        row.append(comparison)
                                        
                                    writer.writerow(row)
                                    
                                    if track_type == 'short':
                                        landmark_zone = [200.0, 240.0]
                                        reward_zone = [320.0, 340.0]
                                    elif track_type == 'long':
                                        landmark_zone = [200.0, 240.0]
                                        reward_zone = [380.0, 400.0]
    
                                    consecutive = 0
                                        
                                    # check to see if there are at least 3 consecutive spatial bins over threshold
                                    for c, comparison in enumerate(comparisons[track_type]):
                                        if comparison != numpy.nan:
                                            if comparison > 0.0:
                                                if consecutive == 0:
                                                    first = c
                                                    
                                                consecutive += 1
                                            else:
                                                if consecutive >= 3:
                                                    temp = distances[first:c].mean()
                                            
                                                    if 0.0 < temp <= landmark_zone[0]:
                                                        pre_landmark_ROIs[track_type].append(ROI)
                                                    if landmark_zone[0] < temp <= landmark_zone[1]:
                                                        landmark_ROIs[track_type].append(ROI)
                                                    if landmark_zone[1] < temp <= reward_zone[0]:
                                                        path_integration_ROIs[track_type].append(ROI)
                                                    if reward_zone[0] < temp:
                                                        reward_ROIs[track_type].append(ROI)
                                                        
                                                consecutive = 0
                                    
                                    # check for ROI activity at the end of the tracks
                                    if consecutive >= 3:
                                        temp = distances[first:c].mean()
                                        
                                        if 0.0 < temp <= landmark_zone[0]:
                                            pre_landmark_ROIs[track_type].append(ROI)
                                        if landmark_zone[0] < temp <= landmark_zone[1]:
                                            landmark_ROIs[track_type].append(ROI)
                                        if landmark_zone[1] < temp <= reward_zone[0]:
                                            path_integration_ROIs[track_type].append(ROI)
                                        if reward_zone[0] < temp:
                                            reward_ROIs[track_type].append(ROI)
                else:
                    print('        Already shuffle tested. Skipping.')
                
        HDF5_data.close()