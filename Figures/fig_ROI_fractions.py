"""
Calculate fractions of ROIs based on shuffle test classification. File contains multiple stand-alone functions (take no parameters) to create different fraction figures. To use, call
desired function from main function

@author: lukasfischer


"""

import matplotlib.pyplot as plt
import numpy as np
import warnings; warnings.simplefilter('ignore')
import yaml
import json
import seaborn as sns
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
    content = yaml.load(f)



def task_engaged_fraction():
    """ Plot the number of task engaged ROIs. Function is stand-alone (takes no parameters) """
    # define which datasets to inclued
    figure_datasets = [['LF170110_2','Day20170331'],['LF170222_1','Day20170615'],
    ['LF170420_1','Day20170719'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170613_1','Day201784']]

    # figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop'],['LF170222_1','Day20170613','Day20170613_openloop'],['LF170222_1','Day20170615','Day20170615_openloop'],
    # ['LF170420_1','Day20170719','Day20170719_openloop'],['LF170421_2','Day20170719','Day20170719_openloop'],['LF170421_2','Day20170720','Day20170720_openloop'],['LF170613_1','Day201784','Day201784_openloop']]

    #figure output parameters
    subfolder = 'roi_fractions'
    fname = 'roi_fractions_all'
    fformat = 'png'

    pre_landmark_all = []
    landmark_all = []
    pi_short = []
    pi_long = []

    # run through every dataset and append ROIs to matrices
    for r in figure_datasets:
        print(r)
        mouse = r[0]
        session = r[1]
        if len(r) > 2:
            roi_classification_file = r[2]
        else:
            roi_classification_file = r[1]

        with open(content['figure_output_path'] + mouse+roi_classification_file + os.sep + 'roi_classification.json') as f:
            roi_classification = json.load(f)

        # read number of rois for individual sections
        pre_landmark_all = np.append(pre_landmark_all, np.union1d(roi_classification['pre_landmark_short'],roi_classification['pre_landmark_long']))
        landmark_all = np.append(landmark_all,np.union1d(roi_classification['landmark_short'],roi_classification['landmark_long']))
        pi_short = np.union1d(pi_short, roi_classification['path_integration_short'])
        pi_short = np.union1d(pi_short, roi_classification['reward_short'])
        pi_long = np.union1d(pi_long, roi_classification['path_integration_long'])
        pi_long = np.union1d(pi_long, roi_classification['reward_long'])

        #print(np.union1d(roi_classification['landmark_short'],roi_classification['landmark_long']))

    # calculate number of ROIs/10 cm
    pre_landmark_all = len(pre_landmark_all) / 10
    landmark_all = len(landmark_all) / 4
    pi_short = len(pi_short) / 8
    pi_long = len(pi_long) / 14
    pi_all = (pi_short + pi_long)/2

    fig = plt.figure(figsize=(4,4))
    ax1 = plt.subplot(111)
    ax1.bar([1,2,3],[pre_landmark_all,landmark_all,pi_all])

    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)


if __name__ == "__main__":
    task_engaged_fraction()
