"""
Calculate fractions of ROIs based on shuffle test classification

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

def roi_fraction_calculation(fig_datasets, fname, fformat='png', subfolder=[]):
    # store lists of ROIs from each animal
    pre_landmark_all = []
    landmark_all = []
    pi_short_all = []
    pi_long_all = []
    pi_all = []

    pre_landmark_all_openloop = []
    landmark_all_openloop = []
    pi_short_all_openloop = []
    pi_long_all_openloop = []
    pi_all_openloop = []

    # calculate overlap of roi types
    pre_landmark_all_intersect = []
    landmark_all_intersect = []
    pi_intersect = []
    task_engaged_all_intersect = []

    # calculate fraction of roi types per animal
    total_rois = 0

    total_rois_numperanimal = []
    pre_landmark_numperanimal = []
    landmark_numperanimal = []
    pi_numperanimal = []

    total_rois_numperanimal_openloop = []
    pre_landmark_numperanimal_openloop = []
    landmark_numperanimal_openloop = []
    pi_numperanimal_openloop = []

    # run through every dataset and append ROIs to matrices
    for r in figure_datasets:
        #print(r)
        mouse = r[0]
        session = r[1]
        ol_session = r[2]
        tot_rois = r[3]

        with open(content['figure_output_path'] + mouse+session + os.sep + 'roi_classification.json') as f:
            roi_classification = json.load(f)

        with open(content['figure_output_path'] + mouse+ol_session + os.sep + 'roi_classification.json') as f:
            roi_classification_openloop = json.load(f)

        # calculate total number of all rois used during roi drawing
        total_rois += tot_rois

        # read from roi classification file and store roi numbers in corresponding vectors
        pre_landmark = np.union1d(roi_classification['pre_landmark_short'],roi_classification['pre_landmark_long'])
        landmark = np.union1d(roi_classification['landmark_short'],roi_classification['landmark_long'])
        landmark = np.union1d(pre_landmark, landmark)
        pi_short = np.union1d(roi_classification['path_integration_short'], roi_classification['reward_short'])
        pi_long = np.union1d(roi_classification['path_integration_long'], roi_classification['reward_long'])
        pi_combined = np.union1d(pi_short,pi_long)

        pre_landmark_openloop = np.union1d(roi_classification_openloop['pre_landmark_short'],roi_classification_openloop['pre_landmark_long'])
        landmark_openloop = np.union1d(roi_classification_openloop['landmark_short'],roi_classification_openloop['landmark_long'])
        landmark_openloop = np.union1d(pre_landmark_openloop, landmark_openloop)
        pi_short_openloop = np.union1d(roi_classification_openloop['path_integration_short'], roi_classification_openloop['reward_short'])
        pi_long_openloop = np.union1d(roi_classification_openloop['path_integration_long'], roi_classification_openloop['reward_long'])
        pi_combined_openloop = np.union1d(pi_short_openloop,pi_long_openloop)

        # calculate number of each roi type per animal
        pre_landmark_numperanimal = np.append(pre_landmark_numperanimal, len(pre_landmark)/tot_rois)
        landmark_numperanimal = np.append(landmark_numperanimal, len(landmark)/tot_rois)
        pi_numperanimal = np.append(pi_numperanimal, (len(pi_short)+len(pi_long))/tot_rois)
        total_rois_numperanimal = np.append(total_rois_numperanimal, len(np.unique(np.concatenate((pre_landmark,landmark,pi_combined))))/tot_rois)

        pre_landmark_numperanimal_openloop = np.append(pre_landmark_numperanimal_openloop, len(pre_landmark_openloop)/tot_rois)
        landmark_numperanimal_openloop = np.append(landmark_numperanimal_openloop, len(landmark_openloop)/tot_rois)
        pi_numperanimal_openloop = np.append(pi_numperanimal_openloop, (len(pi_short_openloop)+len(pi_long_openloop))/tot_rois)
        total_rois_numperanimal_openloop = np.append(total_rois_numperanimal_openloop, len(np.unique(np.concatenate((pre_landmark_openloop,landmark_openloop,pi_combined_openloop))))/tot_rois)

        # read number of rois for individual sections
        pre_landmark_all = np.append(pre_landmark_all, pre_landmark)
        landmark_all = np.append(landmark_all, landmark)
        pi_short_all = np.append(pi_short_all, pi_short)
        pi_long_all = np.append(pi_long_all, pi_long)
        pi_all = np.append(pi_all, pi_combined)

        # read number of rois for individual sections
        pre_landmark_all_openloop = np.append(pre_landmark_all_openloop, pre_landmark_openloop)
        landmark_all_openloop = np.append(landmark_all_openloop, landmark_openloop)
        pi_short_all_openloop = np.append(pi_short_all_openloop, pi_short_openloop)
        pi_long_all_openloop = np.append(pi_long_all_openloop, pi_long_openloop)
        pi_all_openloop = np.append(pi_all_openloop, pi_combined_openloop)

        # add rois that intersect
        pre_landmark_all_intersect = np.append(pre_landmark_all_intersect, np.intersect1d(pre_landmark,pre_landmark_openloop))
        landmark_all_intersect = np.append(landmark_all_intersect, np.intersect1d(landmark,landmark_openloop))
        pi_intersect = np.append(pi_intersect, np.intersect1d(pi_combined,pi_combined_openloop))
        #task_engaged_all_intersect = np.append(task_engaged_all_intersect, np.intersect1d([pre_landmark,landmark,pi_combined],[pre_landmark_openloop,landmark_openloop,pi_combined_openloop]))

    # calculate number of ROIs/10 cm
    # pre_landmark_all = len(pre_landmark_all) / 10
    # landmark_all = len(landmark_all) / 4
    # pi_short_all = len(pi_short_all) / 8
    # pi_long_all = len(pi_long_all) / 14
    # pi_all = (pi_short_all + pi_long_all)/2

    active_roi_fraction = (len(pre_landmark_all)+len(landmark_all)+len(pi_short_all)+len(pi_long_all)) / total_rois
    pre_landmark_fraction = len(pre_landmark_all) / total_rois
    landmark_fraction = len(landmark_all) / total_rois
    pi_all_fraction = (len(pi_all)) / total_rois

    pre_landmark_fraction_sem = np.std(pre_landmark_numperanimal)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem = np.std(landmark_numperanimal)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem = np.std(pi_numperanimal)/np.sqrt(len(figure_datasets))

    active_roi_fraction_openloop = (len(pre_landmark_all_openloop)+len(landmark_all_openloop)+len(pi_short_all_openloop)+len(pi_long_all_openloop)) / total_rois
    pre_landmark_fraction_openloop = len(pre_landmark_all_openloop) / total_rois
    landmark_fraction_openloop = len(landmark_all_openloop) / total_rois
    pi_all_fraction_openloop = (len(pi_all_openloop)) / total_rois

    pre_landmark_fraction_sem_openloop = np.std(pre_landmark_numperanimal_openloop)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem_openloop = np.std(landmark_numperanimal_openloop)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem_openloop = np.std(pi_numperanimal_openloop)/np.sqrt(len(figure_datasets))

    pre_landmark_intersect_fraction = len(pre_landmark_all_intersect)/total_rois
    landmark_intersect_fraction = len(landmark_all_intersect)/total_rois
    pi_intersect_fraction = len(pi_intersect)/total_rois

    #active_roi_fraction_openloop = (len(pre_landmark_all_openloop)+len(landmark_all_openloop)+len(pi_short_all_openloop)+len(pi_long_all_openloop)) / total_rois

    #print('fraction pre-landmark: ', str(pre_landmark_fraction), ' +/- ', str(pre_landmark_fraction_sem), ' openloop: ', str(pre_landmark_fraction_openloop), ' +/- ', str(pre_landmark_fraction_sem_openloop))
    print('fraction landmark: ' + str(landmark_fraction), ' +/- ', str(landmark_fraction_sem), ' openloop: ', str(landmark_fraction_openloop), ' +/- ', str(landmark_fraction_openloop))
    print('total path integration: ' + str(pi_all_fraction), ' +/- ', str(pi_all_fraction_sem), ' openloop: ', str(pi_all_fraction_openloop), ' +/- ', str(pi_all_fraction_sem_openloop))
    #print(stats.ttest_ind(pre_landmark_numperanimal,pre_landmark_numperanimal_openloop))
    print('landmark test: ' + str(stats.ttest_ind(landmark_numperanimal,landmark_numperanimal_openloop)))
    print('path integration test: ' + str(stats.ttest_ind(pi_numperanimal,pi_numperanimal_openloop)))
    #print(stats.ttest_ind(total_rois_numperanimal,total_rois_numperanimal_openloop))

    #print('pre-landmark intersect fraction: ', str(pre_landmark_intersect_fraction))
    print('landmark intersect fraction: ', str(landmark_intersect_fraction))
    print('pi intersect fraction: ', str(pi_intersect_fraction))

    # print(pre_landmark_numperanimal, landmark_numperanimal, pi_numperanimal, total_rois_numperanimal)
    # print(total_rois_numperanimal)
    # print(total_rois_numperanimal_openloop)
    # print(np.std(total_rois_numperanimal)/np.sqrt(len(figure_datasets)),np.std(total_rois_numperanimal_openloop)/np.sqrt(len(figure_datasets)))

    fig = plt.figure(figsize=(4,6))
    ax1 = plt.subplot(111)
    ax1.set_xlim([0,3.5])
    bar_width = 0.5
    # ax1.bar([1-bar_width,2.5-bar_width,4-bar_width],[pre_landmark_fraction,landmark_fraction,pi_all_fraction], bar_width, color='k', yerr=[pre_landmark_fraction_sem,landmark_fraction_sem,pi_all_fraction_sem])
    # ax1.bar([1,2.5,4],[pre_landmark_fraction_openloop,landmark_fraction_openloop,pi_all_fraction_openloop], bar_width, color='w',yerr=[pre_landmark_fraction_sem_openloop,landmark_fraction_sem_openloop,pi_all_fraction_sem_openloop])

    ax1.bar([1-bar_width,2.5-bar_width],[landmark_fraction,pi_all_fraction], bar_width, color='k', yerr=[landmark_fraction_sem,pi_all_fraction_sem], ecolor='0.5')
    ax1.bar([1,2.5],[landmark_fraction_openloop,pi_all_fraction_openloop], bar_width, color='w',yerr=[landmark_fraction_sem_openloop,pi_all_fraction_sem_openloop], ecolor='0.5')

    # ax1.bar([1-bar_width],[pre_landmark_intersect_fraction],bar_width*2,color='m',alpha=0.7)
    # ax1.bar([2.5-bar_width],[landmark_intersect_fraction],bar_width*2,color='m',alpha=0.7)
    # ax1.bar([4-bar_width],[pi_intersect_fraction],bar_width*2,color='m',alpha=0.7)

    # ax1.axhline(pre_landmark_intersect_fraction, xmin=(1-bar_width)/5, xmax=(1+bar_width)/5, color='r', lw=1)
    ax1.axhline(landmark_intersect_fraction, xmin=(1-bar_width)/3.5, xmax=(1+bar_width)/3.5, color='r', lw=1)
    ax1.axhline(pi_intersect_fraction, xmin=(2.5-bar_width)/3.5, xmax=(2.5+bar_width)/3.5, color='r', lw=1)
    ax1.set_xticks([0.5,1.7])
    ax1.set_xticklabels(['landmark', 'path integration'], rotation=45, fontsize=20)
    ax1.set_ylim([0,0.5])

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

    #figure output parameters
    subfolder = 'roi_fractions'
    fname = 'roi_fractions_all'
    fformat = 'png'

    print('TASK ENGAGED ALL:')
    roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)
    #
    # figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87],['LF170613_1','Day201784','Day201784_openloop',77]]
    #
    # #figure output parameters
    # subfolder = 'roi_fractions'
    # fname = 'roi_fractions_l23'
    # fformat = 'svg'
    #
    # print('TASK ENGAGED L2/3')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)
    #
    # figure_datasets = [['LF170222_1','Day20170615','Day20170615_openloop',96],['LF170420_1','Day20170719','Day20170719_openloop',95],['LF170421_2','Day20170719','Day20170719_openloop',68],['LF170421_2','Day20170720','Day20170720_openloop',45]]
    #
    # subfolder = 'roi_fractions'
    # fname = 'roi_fractions_l5'
    # fformat = 'svg'
    #
    # print('TASK ENGAGED L5:')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)

    # figure_datasets = [['LF170214_1','Day201777','Day201777_openloop',112],['LF170214_1','Day2017714','Day2017714_openloop',165]]
    #
    # subfolder = 'roi_fractions'
    # fname = 'roi_fractions_V1'
    # fformat = 'svg'
    # print('TASK ENGAGED V1:')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)
