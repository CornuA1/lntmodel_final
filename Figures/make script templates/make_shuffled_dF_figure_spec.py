"""Create and save figures of shuffled dF tests."""

import sys
import os

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + 'Analysis')

from shuffled_dF_figure import shuffled_dF_figure
from yaml_mouselist import yaml_mouselist
from filter_trials import filter_trials
from ruamel import yaml
import warnings
import numpy
import h5py

warnings.filterwarnings('ignore')
    
# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file)
except OSError:
    print('        No .yaml file found.')
    
groups= ['GCAMP6f_A30_ALL']

for group in groups:

	mice = yaml_mouselist([group])
	mouse = 'LF170613_1'

	try:
		HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
	except OSError:
		print('    No HDF5 file.')

	days = [day for day in HDF5_data]

	for day in days:
		print('    ' + day)
		
		if int(day) > int('20170802')
		
			try:
				dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
			except KeyError:
				print('        No dF_win.')
				continue

			if dF_data.shape[1] == 1:
				print('        Only one ROI detected.')
				continue

			try:
				behavioral_data = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
			except KeyError:
				print('        No behaviour_aligned.')
				continue

			try:
				short_trials = filter_trials(behavioral_data, [], ['tracknumber', 3])
				long_trials = filter_trials(behavioral_data, [], ['tracknumber', 4])
			except IndexError:
				print('        Unexpected data format.')
				continue
			
			if len(short_trials) == 0 and len(long_trials) == 0:
				print('        No short or long trials detected.')
				continue
			
			ROIs = list(range(dF_data.shape[1]))
			
			for ROI in ROIs:
				shuffled_dF_figure(behavioral_data, dF_data, group, mouse, day, ROI, 'png')
		
	HDF5_data.close()