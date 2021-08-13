"""
Plot number of independent events in soma and obliques or dendrites

"""
import numpy as np
import scipy as sp
import os
import sys
import warnings; warnings.simplefilter('ignore')
import matplotlib
import yaml
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fig = plt.figure(figsize=(3,6))
    ax1 = plt.subplot(111)

    ax1.set_ylabel('independent events', fontsize=24)
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['apical oblique','tuft'],rotation=45)
    ax1.set_yticks([0,0.05,0.1])
    ax1.set_yticklabels(['0 %','5 %','10 %'])
    ax1.set_xlim([-0.2,1.2])
    # ax1.set_ylim([-0.05,1])

    ax1.spines['left'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        axis='both', \
        direction='in', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax1.scatter([0],[0], s=150, c='r', linewidths=0)
    ax1.scatter([1,1,1],[0.05,0.055,0.075], s=150, c='g', linewidths=0)

    ax1.set_ylim([-0.005,0.1])

    subfolder = []
    fname = 'independent_events'
    fformat = 'svg'
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
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


    print('done')
