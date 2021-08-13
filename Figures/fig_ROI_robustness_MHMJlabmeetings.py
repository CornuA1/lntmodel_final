"""
Read robustness.json

@author: lukasfischer


"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings; warnings.simplefilter('ignore')
import yaml
import json
import seaborn as sns
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

def roi_robustness_calculation(fig_datasets, fname, fformat='png', subfolder=[]):

    avg_robustness = []
    avg_robustness_openloop = []

    # run through every dataset and append ROIs to matrices
    for r in figure_datasets:
        print(r)
        mouse = r[0]
        session = r[1]
        ol_session = r[2]

        robustness_short = []
        robustness_short_openloop = []
        robustness_long = []
        robustness_long_openloop = []

        with open(content['figure_output_path'] + mouse+session + os.sep + 'roi_classification.json') as f:
            roi_classification = json.load(f)
        with open(content['figure_output_path'] + mouse+ol_session + os.sep + 'roi_classification.json') as f:
            roi_classification_openloop = json.load(f)

        with open(content['figure_output_path'] + mouse+session + os.sep + 'roi_robustness.json') as f:
            roi_robustness = json.load(f)
        with open(content['figure_output_path'] + mouse+ol_session + os.sep + 'roi_robustness.json') as f:
            roi_robustness_openloop = json.load(f)

        for i in roi_classification['task_engaged_short']:
            robustness_short.append(roi_robustness['robustness_short'][i])
            robustness_short_openloop.append(roi_robustness_openloop['robustness_short'][i])

        for i in roi_classification['task_engaged_long']:
            robustness_long.append(roi_robustness['robustness_long'][i])
            robustness_long_openloop.append(roi_robustness_openloop['robustness_long'][i])

        # robustness_long = roi_robustness['robustness_long']
        # robustness_long_openloop = roi_robustness_openloop['robustness_long']
        avg_robustness.append(np.nanmean(robustness_short))
        avg_robustness.append(np.nanmean(robustness_long))
        avg_robustness_openloop.append(np.nanmean(robustness_short_openloop))
        avg_robustness_openloop.append(np.nanmean(robustness_long_openloop))


    avg_robustness_mean = np.nanmean(avg_robustness)
    avg_robustness_sem = np.nanstd(avg_robustness)/np.sqrt(len(avg_robustness))

    avg_robustness_openloop_mean = np.nanmean(robustness_short_openloop)
    avg_robustness_openloop_sem = np.nanstd(robustness_short_openloop)/np.sqrt(len(robustness_short_openloop))

    print(avg_robustness_mean,avg_robustness_openloop_mean)
    print(stats.ttest_ind(avg_robustness,avg_robustness_openloop))

    fig = plt.figure(figsize=(4,6))
    ax1 = plt.subplot(111)
    ax1.set_xlim([0,3.5])
    ax1.set_ylim([0.2,0.5])
    bar_width = 0.5

    ax1.bar([1-bar_width,2.5-bar_width],[avg_robustness_mean,avg_robustness_openloop_mean], bar_width, color='k', yerr=[avg_robustness_sem,avg_robustness_openloop_sem], ecolor='0.5')

    ax1.set_xticks([0.5,1.7])
    ax1.set_xticklabels(['navigation', 'openloop movie'], rotation=45, fontsize=20)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.tight_layout()
    #
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

    figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87],['LF170222_1','Day20170615','Day20170615_openloop',96],
    ['LF170420_1','Day20170719','Day20170719_openloop',95],['LF170421_2','Day20170719','Day20170719_openloop',68],['LF170421_2','Day20170720','Day20170720_openloop',45],['LF170613_1','Day201784','Day201784_openloop',77]]

    #figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87]]

    #figure output parameters
    subfolder = 'roi_fractions'
    fname = 'roi_robustness_all'
    fformat = 'svg'

    print('ALL:')
    roi_robustness_calculation(figure_datasets, fname, fformat, subfolder)


    figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87],['LF170613_1','Day201784','Day201784_openloop',77]]

    #figure output parameters
    subfolder = 'roi_fractions'
    fname = 'roi_robustness_l23'
    fformat = 'svg'

    print('L2/3:')
    roi_robustness_calculation(figure_datasets, fname, fformat, subfolder)

    figure_datasets = [['LF170222_1','Day20170615','Day20170615_openloop',96],['LF170420_1','Day20170719','Day20170719_openloop',95],['LF170421_2','Day20170719','Day20170719_openloop',68],['LF170421_2','Day20170720','Day20170720_openloop',45]]

    #figure output parameters
    subfolder = 'roi_fractions'
    fname = 'roi_robustness_l5'
    fformat = 'svg'

    print('5:')
    roi_robustness_calculation(figure_datasets, fname, fformat, subfolder)

    figure_datasets = [['LF170214_1','Day201777','Day201777_openloop',112],['LF170214_1','Day2017714','Day2017714_openloop',165]]

    subfolder = 'roi_fractions'
    fname = 'roi_robustness_V1'
    fformat = 'svg'

    roi_robustness_calculation(figure_datasets, fname, fformat, subfolder)
