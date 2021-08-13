"""Retrieve lists of mice from groups in project .yaml file."""

import os
from ruamel import yaml

def yaml_mouselist(groups):

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
        return

    mice = []

    # Iterate through groups to collect mice
    for group in groups:
        for mouse in project_yaml[group]:
            mice.append(list(mouse)[0])

    return mice
