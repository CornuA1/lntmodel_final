"""Generate figures of grating sessions."""

import os
import sys

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + '..' + os.sep + 'Analysis')

from ruamel import yaml
from yaml_mouselist import yaml_mouselist
from population_vector_all_figure import population_vector_all_figure

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

naive_L23_mice = [{'name': 'LF171204_1', 'group': 'GCAMP6f_A30_ALL'}, {'name': 'LF171204_2', 'group': 'GCAMP6f_A30_ALL'}]
naive_L5_mice = [{'name': 'LF171212_1', 'group': 'GCAMP6f_A30_ALL'}, {'name': 'LF171212_2', 'group': 'GCAMP6f_A30_ALL'}]
naive_mice = [{'name': 'LF171204_1', 'group': 'GCAMP6f_A30_ALL'}, {'name': 'LF171204_2', 'group': 'GCAMP6f_A30_ALL'}, {'name': 'LF171212_1', 'group': 'GCAMP6f_A30_ALL'}, {'name': 'LF171212_2', 'group': 'GCAMP6f_A30_ALL'}]

for group in groups:
    mice = yaml_mouselist([group])
    
    for m, mouse in enumerate(mice):
        if project_yaml[group][m][mouse][0]['level'] == 'SOMA_L23':
            L23_mice.append({'name': mouse, 'group': group})
            
            if mouse in naive_mice:
                naive_L23_mice.append({'name': mouse, 'group': group})
            else:
                everyone_else_L23_mice.append({'name': mouse, 'group': group})
        elif project_yaml[group][m][mouse][0]['level'] == 'SOMA_L5':
            L5_mice.append({'name': mouse, 'group': group})
            
            if mouse in naive_mice:
                naive_L5_mice.append({'name': mouse, 'group': group})
            else:
                everyone_else_L5_mice.append({'name': mouse, 'group': group})
        
        if mouse not in naive_mice:
            everyone_else.append({'name': mouse, 'group': group})

figure_name = 'population vector - L23'
population_vector_all_figure(L23_mice, figure_name, figure_format = 'png')
                                 
figure_name = 'population vector - L5'
population_vector_all_figure(L5_mice, figure_name, figure_format = 'png')

figure_name = 'population vector - naive mice'
population_vector_all_figure(naive_mice, figure_name, figure_format = 'png')

figure_name = 'population vector - everyone else'
population_vector_all_figure(everyone_else, figure_name, figure_format = 'png')

figure_name = 'population vector - naive L23'
population_vector_all_figure(naive_L23_mice, figure_name, figure_format = 'png')

figure_name = 'population vector - naive L5'
population_vector_all_figure(naive_L5_mice, figure_name, figure_format = 'png')

figure_name = 'population vector - everyone else L23'
population_vector_all_figure(everyone_else_L23_mice, figure_name, figure_format = 'png')

figure_name = 'population vector - everyone else L5'
population_vector_all_figure(everyone_else_L5_mice, figure_name, figure_format = 'png')