"""Generate figures of grating sessions."""

import os
import sys

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + '..' + os.sep + 'Analysis')

from ruamel import yaml
from yaml_mouselist import yaml_mouselist
from transient_length_figure import transient_length_figure

# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read local settings .yaml file.')

try:
    with open(local_settings['active_yaml_file'], 'r') as yaml_file:
        project_yaml = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read project .yaml file.')
    
groups = ['GCAMP6f_A30_ALL', 'GCAMP6f_A30_RBP4']

L23_mice = []
L5_mice = []

for group in groups:
    mice = yaml_mouselist([group])
    
    for m, mouse in enumerate(mice):    
        if project_yaml[group][m][mouse][0]['level'] == 'SOMA_L23':
            L23_mice.append(mouse)
        elif project_yaml[group][m][mouse][0]['level'] == 'SOMA_L5':
            L5_mice.append(mouse)

figure_name = 'transient length histogram - L23'
transient_length_figure(L23_mice, figure_name, 'svg', '#666666')
                                 
figure_name = 'transient length histogram - L5'
transient_length_figure(L5_mice, figure_name, 'svg', '#B000BE')