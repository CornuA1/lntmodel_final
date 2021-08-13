"""
write data to session dictionary.

Update behavior is as follows: if a key already exists, the existing content will be deleted and replaced with the new data. This avoids redundancy
in the data stored in a key. If a key doesn't exist, it will be created

"""

import numpy as np
import h5py
import os
import yaml

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

def update_dict(MOUSE, SESSION, json_dict):
    # check if directory exists, if not, create it
    if not os.path.isdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION):
        os.mkdir(loc_info['figure_output_path'] + MOUSE+'_'+SESSION)

    # check if .json file exists, if yes: update, else: create new one
    if os.path.isfile(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json'):
        with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json', 'r') as f:
            existing_dict = json.load(f)

        existing_dict.update(roi_result_params)

        with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json', 'w') as f:
            json.dump(existing_dict,f)
            print('updating dict')
    else:
        with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params_space.json','w') as f:
            print('new dict')
            json.dump(roi_result_params,f)
