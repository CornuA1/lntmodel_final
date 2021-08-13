"""
create/update session dictionary.

dictionary structure: a given ROI is either valid or invalid based on basic
parameters such as transients/minute or manual curation. Therefore each roi is
listed under the key 'valid' or 'invalid'. During further analysis, additional
parameters are stored such that for each metric there exists another list of
rois where the index of the roi ID matches the index of the metric. For aligned
analyses (i.e. activity aligned to either a point on the track or by space),
there are separate roi lists for each alignment type and each condition (vr vs
openloop).

e.g.:
    rois_valid : [1,2,5,8,9,10]  # these are the rois deemed valid
    rois_invalid : [3,4,6,7]

    space_roi_number : [1,2,3] # there are all rois for which the 'space' metrics have been calculated
    space_peak_short : [2.4,3.2,1.7] # value indeces correspond to roi indices in 'space_roi_number'

Getting all 'valid' rois for 'space_peak_short' requires getting the intersect of 'rois_valid' and 'space_roi_number'.

When a non-existing key is dumped, its simply added to the dictionary.
When an existing key is dumped, the current content is replaced by the new content.

Parameters:
    keyname : string
              key name

    keyvalues : list
                list of values

"""

import json
import os
import yaml
import copy

# os.chdir('C:/Users/Lou/Documents/repos/LNT')
# with open('../' + os.sep + 'loc_settings.yaml', 'r') as f:
    # loc_info = yaml.load(f)
loc_info = 'Q:\Documents\Harnett UROP\figures'

def write_dict(mouse, session, sess_dict, force_new=False, verbose=True, fname_suffix = ''):
    error_dumping = False
    # check if directory exists, if not: create it
    if verbose:
        print('writing/updating dictionary...')
        print('force_new = ' + str(force_new))
        print('file path = ' + loc_info['figure_output_path'] + mouse + os.sep + mouse+'_'+session+fname_suffix+'.json'  )
    if not os.path.isdir(loc_info['figure_output_path'] + mouse):
        if verbose:
            print('Default path for mouse and session does not exist, creating new directory...')
        os.mkdir(loc_info['figure_output_path'] + mouse)

    # check if dictionary exists, if yes: update. if not: create new one
    if os.path.isfile(loc_info['figure_output_path'] + mouse + os.sep + mouse+'_'+session+fname_suffix+'.json') and not force_new:
        if verbose:
            print('updating existing dictionary...')

        with open(loc_info['figure_output_path'] + mouse + os.sep + mouse+'_'+session+fname_suffix+'.json', 'r') as f:
            print('reading current dict...')
            existing_dict = json.load(f)
        original_dict = copy.deepcopy(existing_dict)
        existing_dict.update(sess_dict)

        with open(loc_info['figure_output_path'] + mouse + os.sep + mouse+'_'+session+fname_suffix+'.json', 'w') as f:
            print('Writing to: ' + loc_info['figure_output_path'] + mouse + os.sep + mouse+'_'+session+fname_suffix+'.json')
            try:
                json.dump(existing_dict,f)
            except TypeError:
                # json.dump(original_dict,f)
                print('ERROR: error occured while writing dictionary to .json file. Original dictionary restored.')
                error_dumping = True
        if error_dumping:
            # the reason why we have to go the round-about route of closing and opening the file after an error has been thrown is that
            # json.dump will physically dumped into the file until the error occured and we want to delete the faulty dumping
            with open(loc_info['figure_output_path'] + mouse + os.sep + mouse+'_'+session+fname_suffix+'.json', 'w') as f:
                json.dump(original_dict,f)

    else:
        if verbose:
            print('Dictionary does not exist, creating new one...')
        with open(loc_info['figure_output_path'] + mouse + os.sep + mouse+'_'+session+fname_suffix+'.json','w') as f:
            json.dump(sess_dict,f)

# def print_dict(dict_to_print)
