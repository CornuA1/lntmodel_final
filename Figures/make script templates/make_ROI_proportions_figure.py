"""Generate figures of ROI proportions vs. task score."""

import os
import sys

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + 'Analysis')

from ruamel import yaml
from yaml_mouselist import yaml_mouselist
from ROI_proportions_figure import ROI_proportions_figure
    
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
            
        ROI_proportions_figure(group, mouse, 'png')