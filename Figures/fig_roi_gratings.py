"""
Plot response of individual ROIs to grating presentation. Each subplot is one direction and spatial frequency

@author: lukasfischer

"""


def fig_roi_gratings(h5path, sess, roi, fname, ylims=[-0.5,3], fformat='png'):
    import numpy as np
    import h5py
    import sys
    import yaml
    from yaml_mouselist import yaml_mouselist
    import warnings; warnings.simplefilter('ignore')

    import matplotlib
    from matplotlib import pyplot as plt
    from event_ind import event_ind
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal

    import seaborn as sns
    sns.set_style("white")

    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    try:
        h5dat = h5py.File(h5path, 'r')
        behav_ds = np.copy(h5dat['/' + sess + '/behaviour_aligned'])
        dF_ds = np.copy(h5dat['/' + sess + '/dF_win'])
        data_grating = True;
        h5dat.close()
    except KeyError:
        print('NO GRATING DATA FOUND.')
        h5dat.close()
        return

    # timewindow (minus and plus time in seconds)
    EVENT_TIMEWINDOW = [5,5]

    # create figure to later plot on
    fig = plt.figure(figsize=(16,6))
    ax = [
          plt.subplot2grid((2,8),(0,0)),
          plt.subplot2grid((2,8),(0,1)),
          plt.subplot2grid((2,8),(0,2)),
          plt.subplot2grid((2,8),(0,3)),
          plt.subplot2grid((2,8),(0,4)),
          plt.subplot2grid((2,8),(0,5)),
          plt.subplot2grid((2,8),(0,6)),
          plt.subplot2grid((2,8),(0,7)),
          plt.subplot2grid((2,8),(1,0)),
          plt.subplot2grid((2,8),(1,1)),
          plt.subplot2grid((2,8),(1,2)),
          plt.subplot2grid((2,8),(1,3)),
          plt.subplot2grid((2,8),(1,4)),
          plt.subplot2grid((2,8),(1,5)),
          plt.subplot2grid((2,8),(1,6)),
          plt.subplot2grid((2,8),(1,7))
         ]

    # specify track numbers
    track_short = 3
    track_long = 4

    # create an counter that increments after each 3 sec exposure to black or grating
    i = 1
    seq_counter = 0
    seq_nr = np.zeros((np.size(behav_ds,0),))
    while i < np.size(behav_ds,0):
        if behav_ds[i,1] != behav_ds[i-1,1]:
            seq_counter += 1
        seq_nr[i] = seq_counter
        i += 1   
    # get the size of a sequence (should be the same for each as they are all 3 seconds)
    seq_size = np.size(np.where(seq_nr == 1)[0])
    # squence numbers, each row represents one particular orientation

    orient_seq_nrs = [
                      [1, 33, 65, 97, 129, 161, 193, 225],
                      [3, 35, 67, 99, 131, 163, 195, 227],
                      [5, 37, 69, 101, 133, 165, 197, 229],
                      [7, 39, 71, 103, 135, 167, 199, 231],
                      [9, 41, 73, 105, 137, 169, 201, 233],
                      [11, 43, 75, 107, 139, 171, 203, 235],
                      [13, 45, 77, 109, 141, 173, 205, 237],
                      [15, 47, 79, 111, 143, 175, 207, 239],
                      [17, 49, 81, 113, 145, 177, 209, 241],
                      [19, 51, 83, 115, 147, 179, 211, 243],
                      [21, 53, 85, 117, 149, 181, 213, 245],
                      [23, 55, 87, 119, 151, 183, 215, 247],
                      [25, 57, 89, 121, 153, 185, 217, 249],
                      [27, 59, 91, 123, 155, 187, 219, 251],
                      [29, 61, 93, 125, 157, 189, 221, 253],
                      [31, 63, 95, 127, 159, 191, 223, 255]
                     ]

    # calculate baseline as mean across the whole session
    baseline_dF = np.mean(dF_ds[:,roi])
    # number of samples previous to sequence onset (2 seconds)
    pre_seq_samples = int(2/(behav_ds[1,0] - behav_ds[0,0]))
    # loop through each line of the sequences
    for j,oseq in enumerate(orient_seq_nrs):
        # matrix holding dF traces of individual sequences
        orient_dF = np.zeros((seq_size+pre_seq_samples,np.size(oseq))) 
        for i,o in enumerate(oseq):
            orient_idx = np.where(seq_nr==o)[0]
            pre_orient = np.arange(orient_idx[0]-pre_seq_samples-1,orient_idx[0]-1,1)
            orient_dF[0:pre_seq_samples,i] = dF_ds[pre_orient,roi]
            # calculate baseline dF
            #baseline_dF = np.mean(dF_ds[pre_orient,roi])
            if np.size(orient_idx) < seq_size:
                orient_dF[pre_seq_samples:np.size(orient_idx)+pre_seq_samples,i] = dF_ds[orient_idx,roi]
            else:
                orient_dF[pre_seq_samples:,i] = dF_ds[orient_idx[0:seq_size],roi]
            orient_dF[:,i] -= baseline_dF
            ax[j].plot(orient_dF[:,i],c='0.8')
        mean_dF = np.mean(orient_dF,1)
        ax[j].plot(mean_dF,c='k',lw=3)
        ax[j].set_xlim([0,seq_size+30])
        ax[j].set_ylim([-1,2])
        ax[j].axvline(pre_seq_samples,c='k',ls='--')
        
    plt.tight_layout()

    fig.suptitle('Grating' + fname, wrap=True)
    fname = content['figure_output_path'] + 'grating_' + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat)
