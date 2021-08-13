"""
scatterplot of ROI amplitudes in VR and openloop

@author: lukasfischer

"""

import numpy as np
import h5py
import os
import sys
import traceback
import matplotlib
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import json
import yaml
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

if __name__ == '__main__':
    # set parameters of session to plot
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    SUBNAME = 'lmoff'
    subfolder = MOUSE+'_'+SESSION
    fname = MOUSE+'_'+SESSION

    # load roi parameters for given session
    with open(loc_info['figure_output_path'] + MOUSE+'_'+SESSION + os.sep + 'roi_params.json','r') as f:
        roi_params = json.load(f)

    # check that the list of rois are equal
    if not np.array_equal(roi_params['lmoff_roi_number'], roi_params['lmoff_roi_number_ol']) \
        and np.array_equal(roi_params['lmon_roi_number'], roi_params['lmon_roi_number_ol'])  \
        and np.array_equal(roi_params['reward_roi_number'], roi_params['reward_roi_number_ol']) \
        and np.array_equal(roi_params['trialonset_roi_number'], roi_params['trialonset_roi_number_ol']):
        print('WARNING: roi numbers in VR and OL condition not equal.')
    else:
        # create figure and axes to later plot on
        fig = plt.figure(figsize=(4,4))
        ax1 = plt.subplot(111)

        # cycle through all rois, the the roi list of lmoff as reference (they should all be the same or at least contain the same rois even if not in the same order)
        for i,roi in enumerate(roi_params['lmoff_roi_number']):
            # need to parse arrays from json file into numpy like this
            lmoff_roi_numbers = np.array(roi_params['lmoff_roi_number'])

        ax1.scatter(roi_params[SUBNAME+'_peak_short_ol'], roi_params[SUBNAME+'_peak_short'], c='k', linewidths=0,zorder=2)
        ax1.plot([0,1.6],[0,1.6], c='0.5',zorder=1)
        ax1.set_xlim([-0.1,1.6])
        ax1.set_ylim([-0.1,1.6])
        ax1.set_ylabel('peak response VR')
        ax1.set_xlabel('peak response PASSIVE')

        fformat = 'png'
        fig.tight_layout()
        fig.suptitle(fname, wrap=True)
        if subfolder != []:
            if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
                os.mkdir(loc_info['figure_output_path'] + subfolder)
            fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
        else:
            fname = loc_info['figure_output_path'] + fname + '.' + fformat
        try:
            fig.savefig(fname, format=fformat,dpi=150)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
