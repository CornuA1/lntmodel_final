"""
Written to test out some parameters to try to determine the best
    movement onset detection algorithm.

@author: rmojica@mit.edu
"""

def mvmnt_onset_tests(h5path, mouse, sess, binwidth=1, fname=['dF_mvmt_onset_board_'], fformat='svg',subfolder=[]):
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
    from event_ind import movement_onset
    from event_ind import butter_lowpass_filter

    sys.path.append('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/Figures')
    from fig_speed_loc import speed_loc

    with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    speed_ds = np.copy(behav_ds)
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    # For reference:
        # time = behav_ds[:,0]
        # location = behav_ds[:,1]
        # framerate = behav_ds[:,2]

        # speed = behav_ds[:,3]
        # track = behav_ds[:,4]
        # trial = behav_ds[:,6]
        #
        # track_short = 3
        # track_long = 4
        # blackbox = 5
        #
        # tracklength_short = 320
        # tracklength_long = 380

    high_thresh = 100 + np.median(behav_ds[behav_ds[:,4] != 5, 3])
    low_thresh = 0.0

    behav_ds = np.delete(behav_ds,np.where(behav_ds[:,3] > high_thresh),0)

# ----- Speed v time at movement onset

    if True:

        saving = False

        speed_thr = 1
        gap_tol = 1

        # onset_event = movement_onset(behav_ds, 5, 1)
        onset_event = movement_onset(behav_ds, speed_thr, gap_tol)

        sampling_rate = 1/behav_ds[0,2]
        print('\nSampling rate \t'+str(round(sampling_rate,4))+' Hz')

        idx_var = int((sampling_rate*3)/2)
        elapsed_time = round(behav_ds[onset_event[0]+idx_var,0] - behav_ds[onset_event[0]-idx_var,0],4)
        print('Time \t\t\t'+str(elapsed_time)+' seconds')

        trial = 27

        fig = plt.figure(figsize=(20,10))
        plt.plot(behav_ds[:,0], behav_ds[:,3])
        ymin = min(behav_ds[:,3])
        ymax = max(behav_ds[:,3])
        plt.vlines(behav_ds[onset_event,0],ymin,ymax)
        idx1 = min(behav_ds[behav_ds[:,6]==trial,0])-1
        idx2 = max(behav_ds[behav_ds[:,6]==trial,0])+1
        plt.xlim(idx1,idx2)

        if saving:
            fname = 'test_mvmt_onset_speedvtime' + str(speed_thr) + '_gap_' + str(gap_tol) + str(sess)

            if not os.path.isdir(content['figure_output_path'] + subfolder):
                os.mkdir(content['figure_output_path'] + subfolder)
            fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
            print(fname)
            try:
                fig.savefig(fname, format=fformat, bbox_inches = 'tight', pad_inches=0.3)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)

# ----- Speed v loc with scatterplot at mvmt onset

    if False:

        saving = True

        fig = plt.figure(figsize=(10,9))
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        fig.subplots_adjust(hspace=.4, top=0.93)

        track_short = 3
        track_long = 4
        blackbox = 5

        tracklength_short = 320
        tracklength_long = 380

        binwidth = int(binwidth)

        bins_short = tracklength_short//binwidth            # calculate number of bins
        bins_long = tracklength_long//binwidth

        # high_thresh = (np.median(behav_ds[behav_ds[:,4]!=5,3]) + (3 * np.std(behav_ds[behav_ds[:,4]!=5,3])))
        high_thresh = 100 + np.median(behav_ds[behav_ds[:,4] !=5,3])    # high outliers
        low_thresh = 0.7                                                # low speeds

        # filter requirements.
        order = 6
        fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
        cutoff = 4 # desired cutoff frequency of the filter, Hz

        speed_filtered = butter_lowpass_filter(behav_ds[:,3], cutoff, fs, order)
        speed_filtered = np.delete(speed_filtered,np.where(behav_ds[:,1] < 50),0) # cut all data pionts before 50 cm
        behav_ds = np.delete(behav_ds,np.where(behav_ds[:,1] < 50),0)

        speed_thr = 1
        gap_tol = 1

        onset_event = movement_onset(behav_ds, speed_thr, gap_tol)
        print(onset_event.shape)

        trial_number,trial_ind = np.unique(behav_ds[:,6],return_index=True)
        total_trials = max(trial_number)

        trials_short = filter_trials(behav_ds, [], ['tracknumber', track_short])
        trials_long = filter_trials(behav_ds, [], ['tracknumber', track_long])

        plt.suptitle(mouse+' '+sess+': lowpass filter @ '+str(cutoff)+'Hz')

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

        onset_short_track = [i for i in onset_event if behav_ds[i,4] == 3]
        onset_long_track = [i for i in onset_event if behav_ds[i,4] == 4]

        # short track plot
        plt.sca(ax1)
        plt.plot(range(bins_short),bin_mean_short, c='#5AA5E8')
        plt.vlines(behav_ds[onset_short_track,1], 1, 4, colors='#CD0000', linewidths=0.3)
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
        plt.vlines(behav_ds[onset_long_track,1], 1, 4, colors='#CD0000', linewidths=0.3)
        plt.fill_between(range(bins_long),bin_mean_long-bin_sem_long,bin_mean_long+bin_sem_long,facecolor='#A3B6C6',alpha=0.3)
        loc,_ = plt.xticks()
        plt.xticks(loc[:],[int(x*binwidth) for x in loc[:]])
        plt.title('Long track')
        plt.xlabel('Location (cm)')
        plt.ylabel('Speed (cm/s)')
        plt.axvline(210//binwidth, lw=40, c='0.8',alpha=0.2)
        plt.axvline(390//binwidth, lw=40, c='m',alpha=0.1)
        plt.xlim(40//binwidth,400//binwidth)

        if saving:
            fname = 'test_mvmt_onset_scatman_sp_' + str(speed_thr) + '_gap_' + str(gap_tol) + str(sess)

            if not os.path.isdir(content['figure_output_path'] + subfolder):
                os.mkdir(content['figure_output_path'] + subfolder)
            fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
            print(fname)
            try:
                fig.savefig(fname, format=fformat, bbox_inches = 'tight', pad_inches=0.3)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)

# ----- Speed v loc raw + speed_filtered

    if False:

        saving = True

        track_short = 3
        track_long = 4
        blackbox = 5

        tracklength_short = 320
        tracklength_long = 380

        binwidth = int(binwidth)

        bins_short = tracklength_short//binwidth            # calculate number of bins
        bins_long = tracklength_long//binwidth

        high_thresh = 100 + np.median(behav_ds[behav_ds[:,4] !=5,3])    # high outliers

        cutoffs = [1,2,3,4,5,10]
        for i in range(len(cutoffs)):

            fig = plt.figure(figsize=(8,8))
            ax1 = plt.subplot(2,1,1)
            ax2 = plt.subplot(2,1,2)
            fig.subplots_adjust(hspace=.2, top=0.93)

            # filter requirements
            order = 6
            fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
            cutoff = cutoffs[i] # desired cutoff frequency of the filter, Hz

            speed_filtered = butter_lowpass_filter(behav_ds[:,3], cutoff, fs, order)
            speed_filtered = np.delete(speed_filtered,np.where(behav_ds[:,1] < 50),0) # cut all data pionts before 50 cm
            behav_ds = np.delete(behav_ds,np.where(behav_ds[:,1] < 50),0)

            trial_number,_ = np.unique(behav_ds[:,6],return_index=True)
            total_trials = max(trial_number)

            trials_short = filter_trials(behav_ds, [], ['tracknumber', track_short])
            trials_long = filter_trials(behav_ds, [], ['tracknumber', track_long])

            plt.suptitle(mouse+' '+sess+': lowpass filter @ '+str(cutoff)+'Hz')

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

            # short track plot
            plt.sca(ax1)
            for i,trial in enumerate(trials_short):
                plt.plot(behav_ds[behav_ds[:,6]==trial,1], behav_ds[behav_ds[:,6]==trial,3],c='#CDCDCD',alpha=0.4)
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
            for i,trial in enumerate(trials_long):
                plt.plot(behav_ds[behav_ds[:,6]==trial,1], behav_ds[behav_ds[:,6]==trial,3],c='#CDCDCD', alpha=0.4)
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

            if saving:
                fname = 'test_lowpass_filter_at_' + str(cutoff) + 'Hz_' + str(sess)

                if not os.path.isdir(content['figure_output_path'] + subfolder):
                    os.mkdir(content['figure_output_path'] + subfolder)
                fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
                print(fname)
                try:
                    fig.savefig(fname, format=fformat, bbox_inches = 'tight', pad_inches=0.3)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)

# ----- speed v time at variable mvmt onset paramaters

    if False:

        saving = True

        high_thresh = 100 + np.median(behav_ds[behav_ds[:,4] != 5, 3])
        low_thresh = 0.0

        behav_ds = np.delete(behav_ds,np.where(behav_ds[:,3] > high_thresh),0)

        fig = plt.figure(figsize=(10,10))
        ax1 = plt.subplot(3,3,1)
        ax2 = plt.subplot(3,3,2)
        ax3 = plt.subplot(3,3,3)
        ax4 = plt.subplot(3,3,4)
        ax5 = plt.subplot(3,3,5)
        ax6 = plt.subplot(3,3,6)
        ax7 = plt.subplot(3,3,7)
        ax8 = plt.subplot(3,3,8)
        ax9 = plt.subplot(3,3,9)

        ax_handles = ['ax'+str(x) for x in range(1,10) if True]

        fig.subplots_adjust(hspace=0.4,top=0.975)

        speed_thresh = [0.7,4,7]
        time_thresh = [1,2,3]

        onset_event1 = movement_onset(behav_ds, speed_thresh[0], time_thresh[0])
        onset_event2 = movement_onset(behav_ds, speed_thresh[0], time_thresh[1])
        onset_event3 = movement_onset(behav_ds, speed_thresh[0], time_thresh[2])
        onset_event4 = movement_onset(behav_ds, speed_thresh[1], time_thresh[0])
        onset_event5 = movement_onset(behav_ds, speed_thresh[1], time_thresh[1])
        onset_event6 = movement_onset(behav_ds, speed_thresh[1], time_thresh[2])
        onset_event7 = movement_onset(behav_ds, speed_thresh[2], time_thresh[0])
        onset_event8 = movement_onset(behav_ds, speed_thresh[2], time_thresh[1])
        onset_event9 = movement_onset(behav_ds, speed_thresh[2], time_thresh[2])


        sampling_rate = 1/behav_ds[0,2]
        print('\nSampling rate \t'+str(round(sampling_rate,4))+' Hz')

        idx_var = int((sampling_rate*3)/2)
        elapsed_time = round(behav_ds[onset_event1[0]+idx_var,0] - behav_ds[onset_event1[0]-idx_var,0],4)
        print('Time \t\t\t'+str(elapsed_time)+' seconds')

        onset = 50

        plt.sca(ax1)
        plt.plot(behav_ds[onset_event1[onset]-idx_var:onset_event1[onset]+idx_var,0], behav_ds[onset_event1[onset]-idx_var:onset_event1[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 0.7; time_thresh: 1')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event1[onset]-idx_var,0],behav_ds[onset_event1[onset],0],behav_ds[onset_event1[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax2)
        plt.plot(behav_ds[onset_event2[onset]-idx_var:onset_event2[onset]+idx_var,0], behav_ds[onset_event2[onset]-idx_var:onset_event2[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 0.7; time_thresh: 2')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event2[onset]-idx_var,0],behav_ds[onset_event2[onset],0],behav_ds[onset_event2[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax3)
        plt.plot(behav_ds[onset_event3[onset]-idx_var:onset_event3[onset]+idx_var,0], behav_ds[onset_event3[onset]-idx_var:onset_event3[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 0.7; time_thresh: 3')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event3[onset]-idx_var,0],behav_ds[onset_event3[onset],0],behav_ds[onset_event3[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax4)
        plt.plot(behav_ds[onset_event4[onset]-idx_var:onset_event4[onset]+idx_var,0], behav_ds[onset_event4[onset]-idx_var:onset_event4[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 4; time_thresh: 1')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event4[onset]-idx_var,0],behav_ds[onset_event4[onset],0],behav_ds[onset_event4[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax5)
        plt.plot(behav_ds[onset_event5[onset]-idx_var:onset_event5[onset]+idx_var,0], behav_ds[onset_event5[onset]-idx_var:onset_event5[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 4; time_thresh: 2')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event5[onset]-idx_var,0],behav_ds[onset_event5[onset],0],behav_ds[onset_event5[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax6)
        plt.plot(behav_ds[onset_event6[onset]-idx_var:onset_event6[onset]+idx_var,0], behav_ds[onset_event6[onset]-idx_var:onset_event6[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 4; time_thresh: 3')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event6[onset]-idx_var,0],behav_ds[onset_event6[onset],0],behav_ds[onset_event6[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax7)
        plt.plot(behav_ds[onset_event7[onset]-idx_var:onset_event7[onset]+idx_var,0], behav_ds[onset_event7[onset]-idx_var:onset_event7[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 7; time_thresh: 1')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event7[onset]-idx_var,0],behav_ds[onset_event7[onset],0],behav_ds[onset_event7[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax8)
        plt.plot(behav_ds[onset_event8[onset]-idx_var:onset_event8[onset]+idx_var,0], behav_ds[onset_event8[onset]-idx_var:onset_event8[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 7; time_thresh: 2')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event8[onset]-idx_var,0],behav_ds[onset_event8[onset],0],behav_ds[onset_event8[onset]+idx_var,0]],['-1.5','0','1.5'])

        plt.sca(ax9)
        plt.plot(behav_ds[onset_event9[onset]-idx_var:onset_event9[onset]+idx_var,0], behav_ds[onset_event9[onset]-idx_var:onset_event9[onset]+idx_var,3], color='#BB2E07')
        plt.title('speed_thresh: 7; time_thresh: 3')
        plt.ylabel('Speed (cm/s')
        plt.xlabel('Time (s)')
        plt.xticks([behav_ds[onset_event9[onset]-idx_var,0],behav_ds[onset_event9[onset],0],behav_ds[onset_event9[onset]+idx_var,0]],['-1.5','0','1.5'])

        if saving:
            fname = 'dF_mvmt_onset_event_' + str(onset) + str(sess)

            if not os.path.isdir(content['figure_output_path'] + subfolder):
                os.mkdir(content['figure_output_path'] + subfolder)
            fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
            print(fname)
            try:
                fig.savefig(fname, format=fformat, bbox_inches = 'tight', pad_inches=0.3)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)

# -----

import yaml
import h5py
import os

with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

# LF170613_1
MOUSE = 'LF170613_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day201784'
mvmnt_onset_tests(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
