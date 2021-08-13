"""Create and save figures of shuffled dF tests."""

import sys
import os

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + '..' + os.sep + 'Analysis')

from shuffled_dF_figure import shuffled_dF_figure
from yaml_mouselist import yaml_mouselist
from filter_trials import filter_trials
from ruamel import yaml
import warnings
import numpy as np
import h5py

warnings.filterwarnings('ignore')

# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file)
except OSError:
    print('        Could not read local settings .yaml file.')

groups = ['GCAMP6f_A30_ALL', 'GCAMP6f_A30_RBP4', 'GCAMP6f_A30_V1']

MOUSE = 'LF170613_1'
SESSION = 'Day201784'
h5path = local_settings['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
print(h5path)
h5dat = h5py.File(h5path, 'r')
behavioral_data = np.copy(h5dat[SESSION + '/behaviour_aligned'])
dF_data = np.copy(h5dat[SESSION + '/dF_win'])
h5dat.close()

ROIs = list(range(dF_data.shape[1]))
#ROIs = list(np.arange(0,np.size(dF_data)-2))

#ROIs = [5]
for ROI in ROIs:
    print(ROI)
    shuffled_dF_figure(behavioral_data, dF_data[:, ROI], 'GCAMP6f_A30_ALL', MOUSE, SESSION, ROI, 'png')

# for group in groups:
#     mice = yaml_mouselist([group])
#
#     for m, mouse in enumerate(mice):
#         print(mouse)
#
#         try:
#             HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', 'r')
#         except OSError:
#             print('    No HDF5 file.')
#             continue
#
#         days = [day for day in HDF5_data]
#
#         for day in days:
#             print('    ' + day)
#
#             try:
#                 behavioral_data = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
#             except KeyError:
#                 print('        No behaviour_aligned.')
#                 continue
#
#             try:
#                 dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
#             except KeyError:
#                 print('        No dF_win.')
#                 continue
#
#             try:
#                 short_trials = filter_trials(behavioral_data, [], ['tracknumber', 3])
#                 long_trials = filter_trials(behavioral_data, [], ['tracknumber', 4])
#             except IndexError:
#                 print('        Unexpected track number format.')
#                 continue
#
#             if len(short_trials) == 0 and len(long_trials) == 0:
#                 print('        No short or long trials detected.')
#                 continue
#
#             ROIs = list(range(dF_data.shape[1]))
#
#             for ROI in ROIs:
#                 shuffled_dF_figure(behavioral_data, dF_data[:, ROI], group, mouse, day, ROI, 'png')
