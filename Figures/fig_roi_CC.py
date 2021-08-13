"""
Calculate cross correlation between ROIs. This is intended for testing the cross correlation between bouton recordings.


"""

%matplotlib inline

# load local settings file
import matplotlib
import numpy as np
import warnings; warnings.simplefilter('ignore')
import sys
sys.path.append("./Analysis")
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import statsmodels.api as sm
import yaml
import h5py
import json
import seaborn as sns
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

def roi_CC(dset, rois='task_engaged_all', fformat='png', fname='', subfolder=''):

    h5path = content['imaging_dir'] + dset[0] + '/' + dset[0] + '.h5'
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[dset[1] + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[dset[1] + '/dF_win'])
    h5dat.close()

    try:
        with open(content['figure_output_path'] + dset[0]+dset[1] + os.sep + 'roi_classification.json') as f:
            roi_classification = json.load(f)
            # select rois
            if rois == 'all':
                roi_selection = np.arange(0,np.size(dF_ds,1))
            elif rois == 'task_engaged_all':
                roi_selection = np.union1d(roi_classification['task_engaged_short'],roi_classification['task_engaged_long'])
            elif rois == 'task_engaged_short':
                roi_selection = roi_classification['task_engaged_short']
            elif rois == 'task_engaged_long':
                roi_selection = roi_classification['task_engaged_long']
            elif rois == 'pre_landmark_all':
                roi_selection = np.union1d(roi_classification['pre_landmark_short'],roi_classification['pre_landmark_long'])
            elif rois == 'pre_landmark_short':
                roi_selection = roi_classification['pre_landmark_short']
            elif rois == 'pre_landmark_long':
                roi_selection = roi_classification['pre_landmark_long']
            elif rois == 'landmark_all':
                roi_selection = np.union1d(roi_classification['landmark_short'],roi_classification['landmark_long'])
            elif rois == 'landmark_short':
                roi_selection = roi_classification['landmark_short']
            elif rois == 'landmark_long':
                roi_selection = roi_classification['landmark_long']
            elif rois == 'path_integration_all':
                roi_selection = np.union1d(roi_classification['path_integration_short'],roi_classification['path_integration_long'])
            elif rois == 'path_integration_short':
                roi_selection = roi_classification['path_integration_short']
            elif rois == 'path_integration_long':
                roi_selection = roi_classification['path_integration_long']
            elif rois == 'reward_all':
                roi_selection = np.union1d(roi_classification['reward_short'],roi_classification['reward_long'])
            elif rois == 'reward_short':
                roi_selection = roi_classification['reward_short']
            elif rois == 'reward_long':
                roi_selection = roi_classification['reward_long']

    except(FileNotFoundError):
        print('WARNING: roi classification file not found, using all ROIs')
        roi_selection = np.arange(0,np.size(dF_ds,1))

    roi_CC_matrix = np.zeros((len(roi_selection),len(roi_selection)))

    for i in range(len(roi_selection)):
        for j in range(len(roi_selection)):
            # roi_CC_matrix[i,j] = signal.correlate(dF_ds[:,i], dF_ds[:,j], mode='valid')
            roi_CC_matrix[i,j] = stats.pearsonr(dF_ds[:,i], dF_ds[:,j])[0]

    fig = plt.figure(figsize=(5,4), dpi=300)
    ax1 = plt.subplot(111)
    ax1_cmap = ax1.pcolormesh(roi_CC_matrix.T,vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(ax1_cmap, ax=ax1)
    ax1.set_xlim([0,len(roi_selection)])
    ax1.set_ylim([0,len(roi_selection)])
    ax1.set_xlabel('ROI')
    ax1.set_ylabel('ROI')
    plt.suptitle(str(dset)+' '+rois)
    plt.show()

    roi_CC_res = {
        'included_datasets' : dset,
        'rois' : rois,
        'roi_CC_matrix' : roi_CC_matrix.tolist()
    }

    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)

    with open(content['figure_output_path'] + subfolder + os.sep + dset[0]+dset[1] + '.json','w+') as f:
        json.dump(roi_CC_res,f)

    fname = content['figure_output_path'] + subfolder + os.sep + dset[0]+dset[1] + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)



if __name__ == "__main__":
    dset = ['LF170214_1','Day201777']
    # dset = ['LF170214_1','Day2017714']
    # dset = ['LF171211_2','Day201852']
    roi_CC(dset, 'task_engaged_all', 'png', dset[0]+dset[1], 'roi_cc')
