"""Generate figures of crossday behavior."""

import os
import sys

# load local settings file
sys.path.append('../Analysis')
sys.path.append('../Figures')

os.chdir('../Analysis')

import yaml
from yaml_mouselist import yaml_mouselist
from crossday_behavior_figure import crossday_behavior_figure
    
# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
except OSError:
    print('        Could not read local settings .yaml file.')
    
mice = yaml_mouselist(['GCAMP6f_A30_ALL', 'GCAMP6f_A30_RBP4', 'GCAMP6f_A30_V1'])

for mouse in mice:
    print(mouse)
        
    crossday_behavior_figure(local_settings['imaging_dir'] + mouse + os.sep + mouse + '.h5', mouse, 'png')