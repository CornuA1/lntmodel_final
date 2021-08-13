"""Generate figures of grating sessions."""

import os
import sys

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + '..' + os.sep + 'Analysis')

from ruamel import yaml
from yaml_mouselist import yaml_mouselist
from orientation_histogram_figure import orientation_histogram_figure
from orientation_histogram_figure_overlaid import orientation_histogram_figure_overlaid

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

everyone_else_L23_mice = []
everyone_else_L5_mice = []
everyone_else = []

naive_L23_mice = ['LF171204_1', 'LF171204_2']
naive_L5_mice = ['LF171212_1', 'LF171212_2']
naive_mice = ['LF171204_1', 'LF171204_2', 'LF171212_1', 'LF171212_2']

for group in groups:
    mice = yaml_mouselist([group])
    
    for m, mouse in enumerate(mice):
        if project_yaml[group][m][mouse][0]['level'] == 'SOMA_L23':
            L23_mice.append(mouse)
            
            if mouse in naive_mice:
                naive_L23_mice.append(mouse)
            else:
                everyone_else_L23_mice.append(mouse)
        elif project_yaml[group][m][mouse][0]['level'] == 'SOMA_L5':
            L5_mice.append(mouse)
            
            if mouse in naive_mice:
                naive_L5_mice.append(mouse)
            else:
                everyone_else_L5_mice.append(mouse)
        
        if mouse not in naive_mice:
            everyone_else.append(mouse)

figure_name = 'orientation histogram - L23'
orientation_histogram_figure(L23_mice, figure_name, 'svg', '#666666')
                                 
figure_name = 'orientation histogram - L5'
orientation_histogram_figure(L5_mice, figure_name, 'svg', '#B000BE')
    
figure_name = 'orientation histogram - combined'
orientation_histogram_figure_overlaid(L23_mice, L5_mice, figure_name, 'svg', '#666666', '#B000BE')

figure_name = 'orientation histogram - naive mice'
orientation_histogram_figure(naive_mice, figure_name, 'svg', 'crimson')

figure_name = 'orientation histogram - everyone else'
orientation_histogram_figure(everyone_else, figure_name, 'svg', 'k')

figure_name = 'orientation histogram - naive L23'
orientation_histogram_figure(naive_L23_mice, figure_name, 'svg', '#666666')

figure_name = 'orientation histogram - naive L5'
orientation_histogram_figure(naive_L5_mice, figure_name, 'svg', '#B000BE')
    
figure_name = 'orientation histogram - naive combined'
orientation_histogram_figure_overlaid(naive_L23_mice, naive_L5_mice, figure_name, 'svg', '#666666', '#B000BE')

figure_name = 'orientation histogram - everyone else L23'
orientation_histogram_figure(everyone_else_L23_mice, figure_name, 'svg', '#666666')

figure_name = 'orientation histogram - everyone else L5'
orientation_histogram_figure(everyone_else_L5_mice, figure_name, 'svg', '#B000BE')
    
figure_name = 'orientation histogram - everyone else combined'
orientation_histogram_figure_overlaid(everyone_else_L23_mice, everyone_else_L5_mice, figure_name, 'svg', '#666666', '#B000BE')
    
figure_name = 'orientation histogram - L23 naive vs. trained'
orientation_histogram_figure_overlaid(naive_L23_mice, everyone_else_L23_mice, figure_name, 'svg', 'crimson', 'k', legend = ['naive', 'trained'])
    
figure_name = 'orientation histogram - L5 naive vs. trained'
orientation_histogram_figure_overlaid(naive_L5_mice, everyone_else_L5_mice, figure_name, 'svg', 'crimson', 'k', legend = ['naive', 'trained'])