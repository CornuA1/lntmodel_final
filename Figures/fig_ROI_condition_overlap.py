"""
Calculate fractions of ROIs based on shuffle test classification

@author: lukasfischer


"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
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

    total_rois_fracperanimal = []
    pre_landmark_fracperanimal = []
    landmark_fracperanimal = []
    pi_fracperanimal = []

    total_rois_fracperanimal_openloop = []
    pre_landmark_fracperanimal_openloop = []
    landmark_fracperanimal_openloop = []
    pi_fracperanimal_openloop = []

    pre_landmark_intersect = []
    landmark_intersect = []
    pi_combined_intersect = []

    pre_landmark_intersect_fracperanimal = []
    landmark_intersect_fracperanimal = []
    pi_combined_intersect_fracperanimal = []

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
        # remove ROIs that had their receptive field onset before the PI section
        landmark = landmark[~np.in1d(landmark,pre_landmark)]
        # combine pi and reward classified ROIs (reward really just means they are later in the reward zone)
        pi_short = np.union1d(roi_classification['path_integration_short'], roi_classification['reward_short'])
        pi_long = np.union1d(roi_classification['path_integration_long'], roi_classification['reward_long'])
        pi_combined = np.union1d(pi_short,pi_long)
        # remove ROIs that had their receptive field onset before the PI section
        pi_combined = pi_combined[~np.in1d(pi_combined,np.union1d(pre_landmark, landmark))]

        pre_landmark_openloop = np.union1d(roi_classification_openloop['pre_landmark_short'],roi_classification_openloop['pre_landmark_long'])
        landmark_openloop = np.union1d(roi_classification_openloop['landmark_short'],roi_classification_openloop['landmark_long'])
        # remove ROIs that had their receptive field onset before the PI section
        landmark_openloop = landmark_openloop[~np.in1d(landmark_openloop,pre_landmark_openloop)]
        # combine pi and reward classified ROIs (reward really just means they are later in the reward zone)
        pi_short_openloop = np.union1d(roi_classification_openloop['path_integration_short'], roi_classification_openloop['reward_short'])
        pi_long_openloop = np.union1d(roi_classification_openloop['path_integration_long'], roi_classification_openloop['reward_long'])
        pi_combined_openloop = np.union1d(pi_short_openloop,pi_long_openloop)
        # remove ROIs that had their receptive field onset before the PI section
        pi_combined_openloop = pi_combined_openloop[~np.in1d(pi_combined_openloop,np.union1d(pre_landmark_openloop, landmark_openloop))]

        # calculate number of each roi type per animal
        pre_landmark_fracperanimal = np.append(pre_landmark_fracperanimal, len(pre_landmark)/tot_rois)
        landmark_fracperanimal = np.append(landmark_fracperanimal, len(landmark)/tot_rois)
        pi_fracperanimal = np.append(pi_fracperanimal, (len(pi_combined))/tot_rois)
        total_rois_fracperanimal = np.append(total_rois_fracperanimal, len(np.unique(np.concatenate((pre_landmark,landmark,pi_combined))))/tot_rois)

        pre_landmark_fracperanimal_openloop = np.append(pre_landmark_fracperanimal_openloop, len(pre_landmark_openloop)/tot_rois)
        landmark_fracperanimal_openloop = np.append(landmark_fracperanimal_openloop, len(landmark_openloop)/tot_rois)
        pi_fracperanimal_openloop = np.append(pi_fracperanimal_openloop, (len(pi_combined_openloop))/tot_rois)
        total_rois_fracperanimal_openloop = np.append(total_rois_fracperanimal_openloop, len(np.unique(np.concatenate((pre_landmark_openloop,landmark_openloop,pi_combined_openloop))))/tot_rois)

        # calculate the fraction of intersecting ROIs for each animal
        pre_landmark_intersect = np.append(pre_landmark_intersect, np.intersect1d(pre_landmark,pre_landmark_openloop))
        landmark_intersect = np.append(landmark_intersect, np.intersect1d(landmark,landmark_openloop))
        pi_combined_intersect = np.append(pi_combined_intersect, np.intersect1d(pi_combined,pi_combined_openloop))

        pre_landmark_intersect_fracperanimal = np.append(pre_landmark_intersect_fracperanimal, len(np.intersect1d(pre_landmark,pre_landmark_openloop))/tot_rois)
        landmark_intersect_fracperanimal = np.append(landmark_intersect_fracperanimal, len(np.intersect1d(landmark,landmark_openloop))/tot_rois)
        pi_combined_intersect_fracperanimal = np.append(pi_combined_intersect_fracperanimal, len(np.intersect1d(pi_combined,pi_combined_openloop))/tot_rois)

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

    pre_landmark_fraction_sem = np.std(pre_landmark_fracperanimal)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem = np.std(landmark_fracperanimal)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem = np.std(pi_fracperanimal)/np.sqrt(len(figure_datasets))

    active_roi_fraction_openloop = (len(pre_landmark_all_openloop)+len(landmark_all_openloop)+len(pi_short_all_openloop)+len(pi_long_all_openloop)) / total_rois
    pre_landmark_fraction_openloop = len(pre_landmark_all_openloop) / total_rois
    landmark_fraction_openloop = len(landmark_all_openloop) / total_rois
    pi_all_fraction_openloop = (len(pi_all_openloop)) / total_rois

    pre_landmark_fraction_sem_openloop = np.std(pre_landmark_fracperanimal_openloop)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem_openloop = np.std(landmark_fracperanimal_openloop)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem_openloop = np.std(pi_fracperanimal_openloop)/np.sqrt(len(figure_datasets))

    pre_landmark_intersect_fraction = len(pre_landmark_all_intersect)/total_rois
    landmark_intersect_fraction = len(landmark_all_intersect)/total_rois
    pi_intersect_fraction = len(pi_intersect)/total_rois

    #active_roi_fraction_openloop = (len(pre_landmark_all_openloop)+len(landmark_all_openloop)+len(pi_short_all_openloop)+len(pi_long_all_openloop)) / total_rois

    #print('fraction pre-landmark: ', str(pre_landmark_fraction), ' +/- ', str(pre_landmark_fraction_sem), ' openloop: ', str(pre_landmark_fraction_openloop), ' +/- ', str(pre_landmark_fraction_sem_openloop))
    # print('fraction landmark: ' + str(landmark_fraction), ' +/- ', str(landmark_fraction_sem), ' openloop: ', str(landmark_fraction_openloop), ' +/- ', str(landmark_fraction_openloop))
    # print('total path integration: ' + str(pi_all_fraction), ' +/- ', str(pi_all_fraction_sem), ' openloop: ', str(pi_all_fraction_openloop), ' +/- ', str(pi_all_fraction_sem_openloop))
    #print(stats.ttest_ind(pre_landmark_fracperanimal,pre_landmark_fracperanimal_openloop))
    # print('landmark test: ' + str(stats.ttest_ind(landmark_fracperanimal,landmark_fracperanimal_openloop)))
    # print('path integration test: ' + str(stats.ttest_ind(pi_fracperanimal,pi_fracperanimal_openloop)))
    #print(stats.ttest_ind(total_rois_fracperanimal,total_rois_fracperanimal_openloop))

    #print('pre-landmark intersect fraction: ', str(pre_landmark_intersect_fraction))
    # print('landmark intersect fraction: ', str(landmark_intersect_fraction))
    # print('pi intersect fraction: ', str(pi_intersect_fraction))

    # print(pre_landmark_fracperanimal, landmark_fracperanimal, pi_fracperanimal, total_rois_fracperanimal)
    # print(total_rois_fracperanimal)
    # print(total_rois_fracperanimal_openloop)
    # print(np.std(total_rois_fracperanimal)/np.sqrt(len(figure_datasets)),np.std(total_rois_fracperanimal_openloop)/np.sqrt(len(figure_datasets)))

    # carry out kruskal-wallis test on fractions to determine if there is a difference between samples
    # stat_res, p_val = stats.kruskal(pre_landmark_fracperanimal,pre_landmark_intersect_fracperanimal,landmark_fracperanimal,landmark_intersect_fracperanimal,pi_fracperanimal,pi_combined_intersect_fracperanimal)
    # mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ss,std_lm_ss,std_pi_ss)),group_labels)
    # posthoc_res_ss = mc_res_ss.tukeyhsd()

    fig = plt.figure(figsize=(9,7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    print("--- pre landmark: ---")
    ax1.scatter([1 for x in pre_landmark_fracperanimal], pre_landmark_fracperanimal, s=80, c='k', linewidths=0)
    ax1.scatter([2 for x in pre_landmark_intersect_fracperanimal], pre_landmark_intersect_fracperanimal, s=80, c='w', linewidths=2,zorder=1)
    for i in range(len(pre_landmark_fracperanimal)):
        ax1.plot([1,2],[pre_landmark_fracperanimal[i],pre_landmark_intersect_fracperanimal[i]], c='k',zorder=0)



    ax1.set_xticks([])
    #ax1.set_xticklabels(['pre-LM VR', 'pre-LM passive'], rotation=45, fontsize=20)
    ax1.set_title('pre-landmark', fontsize=20)
    ax1.set_ylim([0,0.5])
    ax1.set_xlim([0.8,2.2])

    ax1.tick_params(length=5,width=2,bottom=False,left=True,top=False,right=False,labelsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel('fraction of task engaged neurons', fontsize=20)

    print("--- landmark: ---")
    ax2.scatter([1 for x in landmark_fracperanimal], landmark_fracperanimal, s=80, c='k', linewidths=0)
    ax2.scatter([2 for x in landmark_intersect_fracperanimal], landmark_intersect_fracperanimal, s=80, c='w', linewidths=2,zorder=1)

    for i in range(len(landmark_fracperanimal)):
        ax2.plot([1,2],[landmark_fracperanimal[i],landmark_intersect_fracperanimal[i]], c='k',zorder=0)

    ax2.set_xticks([])
    #ax2.set_xticklabels(['LM VR', 'LM passive'], rotation=45, fontsize=20)
    ax2.set_title('landmark', fontsize=20)
    ax2.set_ylim([0,0.5])
    ax2.set_xlim([0.8,2.2])

    ax2.tick_params(length=5,width=2,bottom=False,left=True,top=False,right=False,labelsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    #ax2.set_ylabel('fraction of task engaged neurons', fontsize=16)

    print("--- PI: ---")
    ax3.scatter([1 for x in pi_fracperanimal], pi_fracperanimal, s=80, c='k', linewidths=0)
    ax3.scatter([2 for x in pi_combined_intersect_fracperanimal], pi_combined_intersect_fracperanimal, s=80, c='w', linewidths=2,zorder=1)

    for i in range(len(pi_fracperanimal)):
        ax3.plot([1,2],[pi_fracperanimal[i],pi_combined_intersect_fracperanimal[i]], c='k',zorder=0)

    ax3.set_xticks([])
    #ax3.set_xticklabels(['PI VR', 'PI passive'], rotation=45, fontsize=20)
    ax3.set_title('path integration', fontsize=20)
    ax3.set_ylim([0,0.5])
    ax3.set_xlim([0.8,2.2])

    ax3.tick_params(length=5,width=2,bottom=False,left=True,top=False,right=False,labelsize=14)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    #ax3.set_ylabel('fraction of task engaged neurons', fontsize=16)

    fig.tight_layout()
    #
    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat, dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

def roi_fractions_types(fig_datasets, fname, fformat='png', subfolder=[]):
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

    total_rois_fracperanimal = []
    pre_landmark_fracperanimal = []
    landmark_fracperanimal = []
    pi_fracperanimal = []

    total_rois_fracperanimal_openloop = []
    pre_landmark_fracperanimal_openloop = []
    landmark_fracperanimal_openloop = []
    pi_fracperanimal_openloop = []

    pre_landmark_intersect = []
    landmark_intersect = []
    pi_combined_intersect = []

    pre_landmark_intersect_fracperanimal = []
    landmark_intersect_fracperanimal = []
    pi_combined_intersect_fracperanimal = []

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
        # remove ROIs that had their receptive field onset before the PI section
        landmark = landmark[~np.in1d(landmark,pre_landmark)]
        # combine pi and reward classified ROIs (reward really just means they are later in the reward zone)
        pi_short = np.union1d(roi_classification['path_integration_short'], roi_classification['reward_short'])
        pi_long = np.union1d(roi_classification['path_integration_long'], roi_classification['reward_long'])
        pi_combined = np.union1d(pi_short,pi_long)
        # remove ROIs that had their receptive field onset before the PI section
        pi_combined = pi_combined[~np.in1d(pi_combined,np.union1d(pre_landmark, landmark))]

        pre_landmark_openloop = np.union1d(roi_classification_openloop['pre_landmark_short'],roi_classification_openloop['pre_landmark_long'])
        landmark_openloop = np.union1d(roi_classification_openloop['landmark_short'],roi_classification_openloop['landmark_long'])
        # remove ROIs that had their receptive field onset before the PI section
        landmark_openloop = landmark_openloop[~np.in1d(landmark_openloop,pre_landmark_openloop)]
        # combine pi and reward classified ROIs (reward really just means they are later in the reward zone)
        pi_short_openloop = np.union1d(roi_classification_openloop['path_integration_short'], roi_classification_openloop['reward_short'])
        pi_long_openloop = np.union1d(roi_classification_openloop['path_integration_long'], roi_classification_openloop['reward_long'])
        pi_combined_openloop = np.union1d(pi_short_openloop,pi_long_openloop)
        # remove ROIs that had their receptive field onset before the PI section
        pi_combined_openloop = pi_combined_openloop[~np.in1d(pi_combined_openloop,np.union1d(pre_landmark_openloop, landmark_openloop))]

        # calculate number of each roi type per animal
        pre_landmark_fracperanimal = np.append(pre_landmark_fracperanimal, len(pre_landmark)/tot_rois)
        landmark_fracperanimal = np.append(landmark_fracperanimal, len(landmark)/tot_rois)
        pi_fracperanimal = np.append(pi_fracperanimal, (len(pi_combined))/tot_rois)
        total_rois_fracperanimal = np.append(total_rois_fracperanimal, len(np.unique(np.concatenate((pre_landmark,landmark,pi_combined))))/tot_rois)

        pre_landmark_fracperanimal_openloop = np.append(pre_landmark_fracperanimal_openloop, len(pre_landmark_openloop)/tot_rois)
        landmark_fracperanimal_openloop = np.append(landmark_fracperanimal_openloop, len(landmark_openloop)/tot_rois)
        pi_fracperanimal_openloop = np.append(pi_fracperanimal_openloop, (len(pi_combined_openloop))/tot_rois)
        total_rois_fracperanimal_openloop = np.append(total_rois_fracperanimal_openloop, len(np.unique(np.concatenate((pre_landmark_openloop,landmark_openloop,pi_combined_openloop))))/tot_rois)

        # calculate the fraction of intersecting ROIs for each animal
        pre_landmark_intersect = np.append(pre_landmark_intersect, np.intersect1d(pre_landmark,pre_landmark_openloop))
        landmark_intersect = np.append(landmark_intersect, np.intersect1d(landmark,landmark_openloop))
        pi_combined_intersect = np.append(pi_combined_intersect, np.intersect1d(pi_combined,pi_combined_openloop))

        pre_landmark_intersect_fracperanimal = np.append(pre_landmark_intersect_fracperanimal, len(np.intersect1d(pre_landmark,pre_landmark_openloop))/tot_rois)
        landmark_intersect_fracperanimal = np.append(landmark_intersect_fracperanimal, len(np.intersect1d(landmark,landmark_openloop))/tot_rois)
        pi_combined_intersect_fracperanimal = np.append(pi_combined_intersect_fracperanimal, len(np.intersect1d(pi_combined,pi_combined_openloop))/tot_rois)

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

    pre_landmark_fraction_sem = np.std(pre_landmark_fracperanimal)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem = np.std(landmark_fracperanimal)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem = np.std(pi_fracperanimal)/np.sqrt(len(figure_datasets))

    active_roi_fraction_openloop = (len(pre_landmark_all_openloop)+len(landmark_all_openloop)+len(pi_short_all_openloop)+len(pi_long_all_openloop)) / total_rois
    pre_landmark_fraction_openloop = len(pre_landmark_all_openloop) / total_rois
    landmark_fraction_openloop = len(landmark_all_openloop) / total_rois
    pi_all_fraction_openloop = (len(pi_all_openloop)) / total_rois

    pre_landmark_fraction_sem_openloop = np.std(pre_landmark_fracperanimal_openloop)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem_openloop = np.std(landmark_fracperanimal_openloop)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem_openloop = np.std(pi_fracperanimal_openloop)/np.sqrt(len(figure_datasets))

    pre_landmark_intersect_fraction = len(pre_landmark_all_intersect)/total_rois
    landmark_intersect_fraction = len(landmark_all_intersect)/total_rois
    pi_intersect_fraction = len(pi_intersect)/total_rois

    fig = plt.figure(figsize=(4,7))
    ax1 = plt.subplot(111)

    # print("--- pre landmark: ---")
    ax1.scatter([1 for x in pre_landmark_fracperanimal], pre_landmark_fracperanimal, s=100, c='k', linewidths=0)
    ax1.scatter([2 for x in landmark_fracperanimal], landmark_fracperanimal, s=100, c='k', linewidths=0)
    ax1.scatter([3 for x in pi_fracperanimal], pi_fracperanimal, s=100, c='k', linewidths=0)
    # ax1.scatter([2 for x in pre_landmark_intersect_fracperanimal], pre_landmark_intersect_fracperanimal, s=80, c='w', linewidths=2,zorder=1)
    for i in range(len(pre_landmark_fracperanimal)):
        ax1.plot([1,2,3],[pre_landmark_fracperanimal[i],landmark_fracperanimal[i],pi_fracperanimal[i]], c='k',zorder=0)

    ax1.set_xticks([1,2,3])
    ax1.set_xticklabels(['pre-landmark', 'landmark', 'path integration'], rotation=45, ha="right",fontsize=20)
    # ax1.set_title('pre-landmark', fontsize=20)
    ax1.set_ylim([0,0.5])
    ax1.set_xlim([0.8,3.2])

    ax1.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=20)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel('fraction of neurons', fontsize=20)


    fig.tight_layout()
    #
    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat, dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

def roi_fractions_piechart(fig_datasets, fname, fformat='png', subfolder=[]):
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

    total_rois_fracperanimal = []
    pre_landmark_fracperanimal = []
    landmark_fracperanimal = []
    pi_fracperanimal = []

    total_rois_fracperanimal_openloop = []
    pre_landmark_fracperanimal_openloop = []
    landmark_fracperanimal_openloop = []
    pi_fracperanimal_openloop = []

    pre_landmark_intersect = []
    landmark_intersect = []
    pi_combined_intersect = []

    pre_landmark_intersect_fracperanimal = []
    landmark_intersect_fracperanimal = []
    pi_combined_intersect_fracperanimal = []

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
        # remove ROIs that had their receptive field onset before the PI section
        landmark = landmark[~np.in1d(landmark,pre_landmark)]
        # combine pi and reward classified ROIs (reward really just means they are later in the reward zone)
        pi_short = np.union1d(roi_classification['path_integration_short'], roi_classification['reward_short'])
        pi_long = np.union1d(roi_classification['path_integration_long'], roi_classification['reward_long'])
        pi_combined = np.union1d(pi_short,pi_long)
        # remove ROIs that had their receptive field onset before the PI section
        pi_combined = pi_combined[~np.in1d(pi_combined,np.union1d(pre_landmark, landmark))]

        pre_landmark_openloop = np.union1d(roi_classification_openloop['pre_landmark_short'],roi_classification_openloop['pre_landmark_long'])
        landmark_openloop = np.union1d(roi_classification_openloop['landmark_short'],roi_classification_openloop['landmark_long'])
        # remove ROIs that had their receptive field onset before the PI section
        landmark_openloop = landmark_openloop[~np.in1d(landmark_openloop,pre_landmark_openloop)]
        # combine pi and reward classified ROIs (reward really just means they are later in the reward zone)
        pi_short_openloop = np.union1d(roi_classification_openloop['path_integration_short'], roi_classification_openloop['reward_short'])
        pi_long_openloop = np.union1d(roi_classification_openloop['path_integration_long'], roi_classification_openloop['reward_long'])
        pi_combined_openloop = np.union1d(pi_short_openloop,pi_long_openloop)
        # remove ROIs that had their receptive field onset before the PI section
        pi_combined_openloop = pi_combined_openloop[~np.in1d(pi_combined_openloop,np.union1d(pre_landmark_openloop, landmark_openloop))]

        # calculate number of each roi type per animal
        pre_landmark_fracperanimal = np.append(pre_landmark_fracperanimal, len(pre_landmark)/tot_rois)
        landmark_fracperanimal = np.append(landmark_fracperanimal, len(landmark)/tot_rois)
        pi_fracperanimal = np.append(pi_fracperanimal, (len(pi_combined))/tot_rois)
        total_rois_fracperanimal = np.append(total_rois_fracperanimal, len(np.unique(np.concatenate((pre_landmark,landmark,pi_combined))))/tot_rois)

        pre_landmark_fracperanimal_openloop = np.append(pre_landmark_fracperanimal_openloop, len(pre_landmark_openloop)/tot_rois)
        landmark_fracperanimal_openloop = np.append(landmark_fracperanimal_openloop, len(landmark_openloop)/tot_rois)
        pi_fracperanimal_openloop = np.append(pi_fracperanimal_openloop, (len(pi_combined_openloop))/tot_rois)
        total_rois_fracperanimal_openloop = np.append(total_rois_fracperanimal_openloop, len(np.unique(np.concatenate((pre_landmark_openloop,landmark_openloop,pi_combined_openloop))))/tot_rois)

        # calculate the fraction of intersecting ROIs for each animal
        pre_landmark_intersect = np.append(pre_landmark_intersect, np.intersect1d(pre_landmark,pre_landmark_openloop))
        landmark_intersect = np.append(landmark_intersect, np.intersect1d(landmark,landmark_openloop))
        pi_combined_intersect = np.append(pi_combined_intersect, np.intersect1d(pi_combined,pi_combined_openloop))

        pre_landmark_intersect_fracperanimal = np.append(pre_landmark_intersect_fracperanimal, len(np.intersect1d(pre_landmark,pre_landmark_openloop))/tot_rois)
        landmark_intersect_fracperanimal = np.append(landmark_intersect_fracperanimal, len(np.intersect1d(landmark,landmark_openloop))/tot_rois)
        pi_combined_intersect_fracperanimal = np.append(pi_combined_intersect_fracperanimal, len(np.intersect1d(pi_combined,pi_combined_openloop))/tot_rois)

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

    pre_landmark_fraction_sem = np.std(pre_landmark_fracperanimal)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem = np.std(landmark_fracperanimal)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem = np.std(pi_fracperanimal)/np.sqrt(len(figure_datasets))

    active_roi_fraction_openloop = (len(pre_landmark_all_openloop)+len(landmark_all_openloop)+len(pi_short_all_openloop)+len(pi_long_all_openloop)) / total_rois
    pre_landmark_fraction_openloop = len(pre_landmark_all_openloop) / total_rois
    landmark_fraction_openloop = len(landmark_all_openloop) / total_rois
    pi_all_fraction_openloop = (len(pi_all_openloop)) / total_rois

    pre_landmark_fraction_sem_openloop = np.std(pre_landmark_fracperanimal_openloop)/np.sqrt(len(figure_datasets))
    landmark_fraction_sem_openloop = np.std(landmark_fracperanimal_openloop)/np.sqrt(len(figure_datasets))
    pi_all_fraction_sem_openloop = np.std(pi_fracperanimal_openloop)/np.sqrt(len(figure_datasets))

    pre_landmark_intersect_fraction = len(pre_landmark_all_intersect)/total_rois
    landmark_intersect_fraction = len(landmark_all_intersect)/total_rois
    pi_intersect_fraction = len(pi_intersect)/total_rois

    fig = plt.figure(figsize=(4,4))
    ax1 = plt.subplot(111)

    # print("--- pre landmark: ---")

    not_engaged = total_rois-(len(pre_landmark_all)+len(landmark_all)+len(pi_all))
    print(len(pre_landmark_all),len(landmark_all),len(pi_all), total_rois)
    ax1.pie([not_engaged,len(pi_all),len(landmark_all),len(pre_landmark_all),],explode = [0.0,0.1,0.1,0.1],startangle=90, autopct="%2.0f%%",
            labels=['not engaged', 'path integration', 'landmark', 'pre-landmark'],radius=0.7,colors=['w','0.5','0.7','0.9'],wedgeprops = {'linewidth': 1})
    # ax1.axis('equal')
    # print(np.mean(pre_landmark_fracperanimal),np.mean(landmark_fracperanimal),np.mean(pi_fracperanimal))

    plt.tight_layout()
    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat, dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

if __name__ == "__main__":

    figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87],['LF170222_1','Day20170615','Day20170615_openloop',96],
    ['LF170420_1','Day20170719','Day20170719_openloop',95],['LF170421_2','Day20170719','Day20170719_openloop',68],['LF170421_2','Day20170720','Day20170720_openloop',45],['LF170613_1','Day201784','Day201784_openloop',77]]

    # figure output parameters
    subfolder = 'roi_fractions'
    fname = 'roi_fractions_all_pie'
    fformat = 'png'

    # print('TASK ENGAGED ALL:')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)
    roi_fractions_piechart(figure_datasets, fname, fformat, subfolder)
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

    # figure_datasets = [['LF170214_1','Day201777','Day201777_openloop',112],['LF170214_1','Day2017714','Day2017714_openloop',165],['LF171211_2','Day201852_openloop','Day201852',245]]
    # #
    # subfolder = 'roi_fractions'
    # fname = 'roi_fractions_V1'
    # fformat = 'png'
    # print('TASK ENGAGED V1:')
    # roi_fraction_calculation(figure_datasets, fname, fformat, subfolder)
