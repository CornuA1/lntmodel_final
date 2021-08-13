"""

Testing deconvolution one ROI at a time.

Vive la déconvolution!


@author: rmojica@mit.edu
modified: Lukas Fischer, 11/27/


"""

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

def gcamp6f_spikes(h5path, mouse, sess, roi, fformat='png',subfolder=[]):
    import math
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy import stats
    import seaborn as sns
    sns.set_style("white")
    from skimage.filters import threshold_otsu

    import h5py
    import os
    import sys
    import yaml
    import warnings; warnings.filterwarnings('ignore')

    with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    sampling_rate = 1/behav_ds[0,2]
    print('\nSampling rate \t'+str(round(sampling_rate,2))+' Hz')
    desired_window = 5          # desired time window for peri-event indices calculation (in seconds)

    idx_var = [int((sampling_rate*desired_window)),int((sampling_rate*desired_window))]

    time_window = round(behav_ds[0,0] + behav_ds[idx_var[0],0],4)
    print('Time window \t\t'+str(time_window)+' s')

    m = np.mean(dF_ds[:,roi],axis=0)
    m02 = np.mean(np.power(dF_ds[:,roi],2),axis=0)
    m12 = np.mean(np.multiply(dF_ds[1:-2,roi],dF_ds[0:-3,roi]),axis=0)
    a = ((m**2)-m12)/((m**2)-m02)                   # alpha
    uhat = [dF_ds[1:-1,roi] - a*dF_ds[0:-2,roi]]
    uhat = np.insert(uhat,0,0)                      # spike train
    shat = np.where(uhat > threshold_otsu(uhat))    # estimated spike train

    plt.figure()
    plt.plot(shat)
    plt.figure()
    plt.plot(dF_ds[:,roi])

    # start_idx = 23880

    # plt.plot(dF_ds[:,roi])
    # plt.xlim([start_idx,start_idx+idx_var[0]])
    # plt.ylim([-1,1])

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    # ----
    import yaml
    import h5py
    import os

    with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
       content = yaml.load(f)

    # LF170613_1
    MOUSE = 'LF170613_1'
    h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    SESSION = 'Day20170804'
    gcamp6f_spikes(h5path, MOUSE, SESSION, roi=0,subfolder=MOUSE+'_'+SESSION)
