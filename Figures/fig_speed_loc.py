"""
Plot average speed across short and long tracks.

Args:
    binwidth: size of bins to average speed across (2cm by default)
    filtering: lowpass filter application (True by default)

Returns:
    Creates a figure with six subplots and populates each with the corresponding
    track's data.
    Figures contain speed and location data.
    Saved in specified subfolder.

author: rmojica@mit.edu
"""

def speed_loc(h5path, mouse, sess, binwidth=2, filtering=True, fname=[], fformat='png',subfolder=[]):
    import math
    import matplotlib
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy import stats
    import seaborn as sns
    sns.set_style("white")

    import h5py
    import os
    import sys
    import yaml
    import warnings; warnings.filterwarnings('ignore')

    sys.path.append('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/Analysis')
    from filter_trials import filter_trials
    from event_ind import butter_lowpass_filter

    with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    h5dat.close()

    fig = plt.figure(figsize=(8,12))
    plt.suptitle(mouse+' '+sess)
    ax1 = plt.subplot(4,1,1)
    ax2 = plt.subplot(4,1,2)
    ax3 = plt.subplot(4,2,5)
    ax4 = plt.subplot(4,2,6)
    ax5 = plt.subplot(4,2,7)
    ax6 = plt.subplot(4,2,8)
    fig.subplots_adjust(hspace=.4)

    # For reference:
        # location = behav_ds[:,1]
        # speed = behav_ds[:,3]
        # track = behav_ds[:,4]
        # trial = behav_ds[:,6]

    track_short = 3
    track_long = 4
    blackbox = 5

    tracklength_short = 320
    tracklength_long = 380

    binwidth = int(binwidth)

    bins_short = tracklength_short//binwidth            # calculate number of bins
    bins_long = tracklength_long//binwidth

    # high_thresh = (np.median(behav_ds[behav_ds[:,4]!=5,3]) + (3 * np.std(behav_ds[behav_ds[:,4]!=5,3])))
    high_thresh = 100 + np.median(behav_ds[behav_ds[:,4] !=5,3])
    low_thresh = 0.7

    if filtering:

        fname = 'speedvloc_' + str(sess)

        # filter requirements.
        order = 6
        fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
        cutoff = 4 # desired cutoff frequency of the filter, Hz

        # print(np.size(behav_ds,0))
        # print(behav_ds[-1,0])
        # print(int(np.size(behav_ds,0)/behav_ds[-1,0]))

        speed_filtered = butter_lowpass_filter(behav_ds[:,3], cutoff, fs, order)
        speed_filtered = np.delete(speed_filtered,np.where(behav_ds[:,1] < 50),0)
        behav_ds = np.delete(behav_ds,np.where(behav_ds[:,1] < 50),0)

        trial_number,trial_ind = np.unique(behav_ds[:,6],return_index=True)
        total_trials = max(trial_number)

        trials_short = filter_trials(behav_ds, [], ['tracknumber', track_short])
        trials_long = filter_trials(behav_ds, [], ['tracknumber', track_long])

        plt.sca(ax2)
        plt.title('Filtered ('+str(cutoff) +'Hz)')

        plt.sca(ax1)
        plt.title('Unfiltered')

        speed_means_short = []
        for i,trial in enumerate(trials_short):
            curr_trial_ds = behav_ds[behav_ds[:,6] == trial,:]
            curr_trial_filt = speed_filtered[behav_ds[:,6] == trial]
            speed_means,speed_edges_short,_ = stats.binned_statistic(curr_trial_ds[:,1],curr_trial_filt,statistic='mean',bins=bins_short,range=(0,tracklength_short))
            speed_means_short.append(speed_means)

        bin_mean_short = np.nanmean(speed_means_short,axis=0)
        bin_sem_short = stats.sem(speed_means_short,axis=0,nan_policy='omit')

        speed_means_long = []
        for i,trial in enumerate(trials_long):
            curr_trial_ds = behav_ds[behav_ds[:,6] == trial,:]
            curr_trial_filt = speed_filtered[behav_ds[:,6] == trial]
            speed_means,speed_edges_long,_ = stats.binned_statistic(curr_trial_ds[:,1],curr_trial_filt,statistic='mean',bins=bins_long,range=(0,tracklength_long))
            speed_means_long.append(speed_means)

        bin_mean_long = np.nanmean(speed_means_long,axis=0)
        bin_sem_long = stats.sem(speed_means_long,axis=0, nan_policy='omit')

        speed_means_long = []
        for i,trial in enumerate(trials_long):
            curr_trial_ds = behav_ds[behav_ds[:,6] == trial,:]
            curr_trial_filt = speed_filtered[behav_ds[:,6] == trial]
            speed_means,speed_edges_long,_ = stats.binned_statistic(curr_trial_ds[:,1],curr_trial_filt,statistic='mean',bins=bins_long,range=(0,tracklength_long))
            speed_means_long.append(speed_means)

        bin_mean_long = np.nanmean(speed_means_long,axis=0)
        bin_sem_long = stats.sem(speed_means_long,axis=0, nan_policy='omit')

    if not filtering:

        fname = 'speedvloc_unfilt' + str(sess)

        behav_ds = np.delete(behav_ds,np.where(behav_ds[:,3] < low_thresh),0)
        behav_ds = np.delete(behav_ds,np.where(behav_ds[:,3] > high_thresh),0)
        behav_ds = np.delete(behav_ds,np.where(behav_ds[:,1] < 50),0)

        trial_number,trial_ind = np.unique(behav_ds[:,6],return_index=True)
        total_trials = max(trial_number)

        trials_short = filter_trials(behav_ds, [], ['tracknumber', track_short])
        trials_long = filter_trials(behav_ds, [], ['tracknumber', track_long])

        speed_means_short = []
        for i,trial in enumerate(trials_short):
            curr_trial = behav_ds[behav_ds[:,6] == trial,:]
            speed_means,speed_edges_short,_ = stats.binned_statistic(curr_trial[:,1],curr_trial[:,3],statistic='mean',bins=bins_short,range=(0,tracklength_short))
            speed_means_short.append(speed_means)

        bin_mean_short = np.nanmean(speed_means_short,axis=0)
        bin_sem_short = stats.sem(speed_means_short,axis=0,nan_policy='omit')

        speed_means_long = []
        for i,trial in enumerate(trials_long):
            curr_trial = behav_ds[behav_ds[:,6] == trial,:]
            speed_means,speed_edges_long,_ = stats.binned_statistic(curr_trial[:,1],curr_trial[:,3],statistic='mean',bins=bins_long,range=(0,tracklength_long))
            speed_means_long.append(speed_means)

        bin_mean_long = np.nanmean(speed_means_long,axis=0)
        bin_sem_long = stats.sem(speed_means_long,axis=0, nan_policy='omit')

    # short track plot
    plt.sca(ax1)
    plt.plot(range(bins_short),bin_mean_short, c='#5AA5E8')
    plt.fill_between(range(bins_short),bin_mean_short-bin_sem_short,bin_mean_short+bin_sem_short,facecolor='#A3B6C6',alpha=0.3)
    loc,_ = plt.xticks()
    plt.xticks(loc[:],[int(x*binwidth) for x in loc[:]])
    plt.title('Short track')
    plt.xlabel('Location (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.axvline(210//binwidth, lw=40, c='0.8',alpha=0.2)
    plt.axvline(330//binwidth, lw=40, c='b',alpha=0.1)
    plt.xlim(40//binwidth,400//binwidth)

    # long track plot
    plt.sca(ax2)
    plt.plot(range(bins_long),bin_mean_long, c='#5AA5E8')
    plt.fill_between(range(bins_long),bin_mean_long-bin_sem_long,bin_mean_long+bin_sem_long,facecolor='#A3B6C6',alpha=0.3)
    loc,_ = plt.xticks()
    plt.xticks(loc[:],[int(x*binwidth) for x in loc[:]])
    plt.title('Long track')
    plt.xlabel('Location (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.axvline(210//binwidth, lw=40, c='0.8',alpha=0.2)
    plt.axvline(390//binwidth, lw=40, c='m',alpha=0.1)
    plt.xlim(40//binwidth,400//binwidth)

    # landmark-centered: Short
    plt.sca(ax3)
    plt.plot(range(bins_short),bin_mean_short,c='#5AA5E8')
    plt.xlim(160//binwidth,260//binwidth)
    plt.fill_between(range(bins_short), bin_mean_short-bin_sem_short,bin_mean_short+bin_sem_short,facecolor='#A3B6C6',alpha=0.3)
    loc,_ = plt.xticks()
    plt.xticks(loc[:],[int(x*binwidth) for x in loc[:]])
    plt.title('Landmark-centered: Short')
    plt.xlabel('Location (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.axvline(210//binwidth,lw=40,c='0.8',alpha=0.2)

    # reward-centered: Short
    plt.sca(ax4)
    plt.plot(range(bins_short),bin_mean_short,c='#5AA5E8')
    plt.xlim(270//binwidth,340//binwidth)
    plt.fill_between(range(bins_short), bin_mean_short-bin_sem_short,bin_mean_short+bin_sem_short,facecolor='#A3B6C6',alpha=0.3)
    loc,_ = plt.xticks()
    plt.xticks(loc[:],[int(x*binwidth) for x in loc[:]])
    plt.title('Reward-centered: Short')
    plt.xlabel('Location (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.axvline(330//binwidth, lw=90, c='b',alpha=0.1)

    # landmark-centered: Long
    plt.sca(ax5)
    plt.plot(range(bins_long),bin_mean_long,c='#5AA5E8')
    plt.xlim(160//binwidth,260//binwidth)
    plt.fill_between(range(bins_long), bin_mean_long-bin_sem_long,bin_mean_long+bin_sem_long,facecolor='#A3B6C6',alpha=0.3)
    loc,_ = plt.xticks()
    plt.xticks(loc[:],[int(x*binwidth) for x in loc[:]])
    plt.title('Landmark-centered: Long')
    plt.xlabel('Location (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.axvline(210//binwidth,lw=40,c='0.8',alpha=0.2)

    # reward-centered: Long
    plt.sca(ax6)
    plt.plot(range(bins_long),bin_mean_long,c='#5AA5E8')
    plt.xlim(330//binwidth,400//binwidth)
    plt.fill_between(range(bins_long), bin_mean_long-bin_sem_long,bin_mean_long+bin_sem_long,facecolor='#A3B6C6',alpha=0.3)
    loc,_ = plt.xticks()
    plt.xticks(loc[:],[int(x*binwidth) for x in loc[:]])
    plt.title('Reward-centered: Long')
    plt.xlabel('Location (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.axvline(390//binwidth, lw=90, c='m',alpha=0.1)

    # if __name__ == "__main__":
    #     if not os.path.isdir(content['figure_output_path'] + subfolder):
    #         os.mkdir(content['figure_output_path'] + subfolder)
    #     fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    #     print(fname)
    #     try:
    #         fig.savefig(fname, format=fformat)
    #     except:
    #         exc_type, exc_value, exc_traceback = sys.exc_info()
    #         traceback.print_exception(exc_type, exc_value, exc_traceback,
    #                               limit=2, file=sys.stdout)

# -----

import yaml
import h5py
import os

with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
    content = yaml.load(f)


# LF180119_1
# MOUSE = 'LF180119_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018316_1'
# speed_loc(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)


# LF171212_2
# MOUSE = 'LF171212_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018218_1'
# speed_loc(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018212_1'
# speed_loc((h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)


# LF180112_2
# MOUSE = 'LF180112_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018315_1'
# speed_loc((h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# LF170613_1
MOUSE = 'LF170613_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day201784'
speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# LF170222_1
MOUSE = 'LF170222_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day20170615'
speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# SESSION = 'Day201776'
# speed_loc(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# LF170421_2
MOUSE = 'LF170421_2'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day20170719'
speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

SESSION = 'Day20170720'
speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# LF170420_1
MOUSE = 'LF170420_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

# SESSION = 'Day20170803'
# speed_loc(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

SESSION = 'Day20170719'
speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# LF170110_2
MOUSE = 'LF170110_2'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

# SESSION = 'Day20170408'
# speed_loc(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

SESSION = 'Day20170331'
speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# # LF170214_1
# MOUSE = 'LF170214_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2017613'
# speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day201777'
# speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2017714'
# speed_loc(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# LF170112_2
MOUSE = 'LF170112_2'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

# SESSION = 'Day20170322'
# speed_loc(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
