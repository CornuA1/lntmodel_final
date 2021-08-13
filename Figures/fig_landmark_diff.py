'''
Plot neuronal responses to landmarks as a function of time.
This code can also be adapted to plot responses at any point in the track.

Args:
    sess: regular session, of course
    sess_ol: suffix OL will always refer to Open Loop

Returns:
    Creates a figure with four subplots and populates each with the corresponding
    calcium traces.
    Saved in specified subfolder, with format fformat.

@author: rmojica@mit.edu
'''

def fig_landmark_diff(h5path,mouse,sess,sess_ol,plotting=True,fname=[],fformat='png',subfolder=[]):
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
    from event_ind import event_ind

    with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    behav_ds_ol = np.copy(h5dat[sess_ol + '/behaviour_aligned'])    # openloop
    dF_ds_ol = np.copy(h5dat[sess_ol + '/dF_win'])
    h5dat.close()

    print('\n'+str(mouse)+str(sess))

    # determine sampling rate for session to have accurate time readouts
    sampling_rate = 1/behav_ds[0,2]
    print('CL sampling rate \t\t'+str(round(sampling_rate,4))+' Hz')

    # both sessions should have the same sampling rate but there may be exceptions
    sampling_rate_ol = 1/behav_ds_ol[0,2]
    print('OL sampling rate \t\t'+str(round(sampling_rate,4))+' Hz')

    # desired time window
    TIME_WINDOW = 6

    # peri-event index calculation based on desired time window
    idx_var = [int((sampling_rate*TIME_WINDOW)/2),int((sampling_rate*TIME_WINDOW)/2)]
    idx_var_ol = [int((sampling_rate_ol*TIME_WINDOW)/2),int((sampling_rate_ol*TIME_WINDOW)/2)]

    # determine movement onset for accurate calculations of time spent in trial
    # NOT YET IMPLEMENTED
    # speed_thresh = 1
    # time_thresh = 1
    # onset_event = event_ind(behav_ds,['movement_onset',speed_thresh,time_thresh])
    #
    # _,first_onset_ind = np.unique(onset_event[:,1],return_index=True)
    # first_onset = onset_event[first_onset_ind,0].astype(int)

    # add extra column to dataset to track original index after data filtering
    tmp = int(behav_ds.shape[0])
    behav_ds_idx = []
    for i in range(tmp):
        behav_ds_idx.append([int(i)])
    behav_ds = np.hstack((behav_ds,behav_ds_idx))

    # calculate average time spent in each trial type
    short_trial_time = []
    long_trial_time = []
    for curr_trial_num in np.unique(behav_ds[:,6]):
        curr_trial = behav_ds[np.where(behav_ds[:,6] == curr_trial_num)]
        if all(curr_trial[2:-3,4] == 3):                                # sometimes trial number does not change until a couple frames later
            short_trial_time.append(curr_trial[-1,0] - curr_trial[1,0]) # appends elapsed time for curr_trial
        if all(curr_trial[2:-3,4] == 4):
            long_trial_time.append(curr_trial[-1,0] - curr_trial[1,0])

    # fetch some delicious general stats for time spent in trials
    trial_time_stats = [np.mean(short_trial_time), np.std(short_trial_time), np.mean(long_trial_time), np.std(long_trial_time)]
    print('Avg time:  Short track\t\t'+str(round(trial_time_stats[0],2))+'s')
    print('Avg time:  Long track\t\t'+str(round(trial_time_stats[2],2))+'s')

    short_trials = filter_trials(behav_ds, [], ['tracknumber',3])
    # purge that dataset! (eliminate all those trials where Mr. Dr. Prof. Mouse just stood around)
    filt_short_trials = filter_trials(behav_ds, [], ['maxtotaltime',trial_time_stats[0]+trial_time_stats[1]*3],short_trials)

    long_trials = filter_trials(behav_ds, [], ['tracknumber',4])
    filt_long_trials = filter_trials(behav_ds, [], ['maxtotaltime',trial_time_stats[2]+trial_time_stats[3]*3],long_trials)

    short_trials_ol = filter_trials(behav_ds_ol, [], ['tracknumber',3])
    long_trials_ol = filter_trials(behav_ds_ol, [], ['tracknumber',4])

    LM = [200, 240]    # landmark locations ([onset, offset])
    onset_LM = LM[0]    # desired landmark
    offset_LM = LM[1]

    at_landmark = event_ind(behav_ds,['at_location',onset_LM)      # first index at location 200
    at_landmark = at_landmark.astype(int)

    at_landmark_ol = event_ind(behav_ds_ol,['at_location',onset_LM)
    at_landmark_ol = at_landmark_ol.astype(int)

    # very stupid way of splitting at_landmark into the different trials
    at_landmark_short = []
    lm_short_idx = []
    for i,lm in enumerate(at_landmark[:,1]):
        for k,t in enumerate(filt_short_trials):
            if lm == t:
                at_landmark_short.append(at_landmark[i,0])
                lm_short_idx.append(list(range(at_landmark[i,0]-idx_var[0],at_landmark[i,0]+idx_var[0])))
            else:
                pass
    at_landmark_short = np.array(at_landmark_short)
    lm_short_idx = np.array(lm_short_idx)

    # dividing at_landmark into long trials
    at_landmark_long = []
    lm_long_idx = []
    for i,lm in enumerate(at_landmark[:,1]):
        for k,t in enumerate(filt_long_trials):
            if lm == t:
                at_landmark_long.append(at_landmark[i,0])
                lm_long_idx.append(list(range(at_landmark[i,0]-idx_var[0],at_landmark[i,0]+idx_var[0])))
            else:
                pass
    at_landmark_long = np.array(at_landmark_long)
    lm_long_idx = np.array(lm_long_idx)

    # yep, you guessed it; more splitting of trials
    at_landmark_short_ol = []
    lm_short_ol_idx = []
    for i,lm in enumerate(at_landmark_ol[:,1]):
        for k,t in enumerate(short_trials_ol):
            if lm == t:
                try:
                    at_landmark_short_ol.append(at_landmark_ol[i,0])
                    lm_short_ol_idx.append(list(range(at_landmark_ol[i,0]-idx_var[0],at_landmark_ol[i,0]+idx_var[0])))
                except:
                    pass
            else:
                pass
    at_landmark_short_ol = np.array(at_landmark_short_ol)
    lm_short_ol_idx = np.array(lm_short_ol_idx)

    # ah, familiar grounds
    at_landmark_long_ol = []
    lm_long_ol_idx = []
    for i,lm in enumerate(at_landmark_ol[:,1]):
        for k,t in enumerate(long_trials_ol):
            if lm == t:
                try:
                    at_landmark_long_ol.append(at_landmark_ol[i,0])
                    lm_long_ol_idx.append(list(range(at_landmark_ol[i,0]-idx_var[0],at_landmark_ol[i,0]+idx_var[0])))
                except:
                    pass
            else:
                pass
    at_landmark_long_ol = np.array(at_landmark_long_ol)
    lm_long_ol_idx = np.array(lm_long_ol_idx)

    count_lm_short = 0
    count_lm_long = 0
    count_lm_long_ol = 0
    count_lm_short_ol = 0

    ### get every ROI's avg dF for every trial
    # Regular session
    # Create a figure for each identified ROI in the session
    for roi in range(dF_ds.shape[1]):

        fig = plt.figure(figsize=(10,12))
        ax1 = plt.subplot(421)
        ax2 = ax1                   # create second x-axis to plot binned stats with raw data
        ax3 = plt.subplot(423)
        ax4 = ax3
        ax5 = plt.subplot(422)
        ax6 = ax5
        ax7 = plt.subplot(424)
        ax8 = ax7

        trial_number,trial_ind = np.unique(behav_ds[:,6],return_index=True)
        total_trials = max(trial_number)

        # calculate average dFs
        try:
            dF_mean_short = np.mean(dF_ds[lm_short_idx,roi],axis=0)
            dF_std_short = np.std(dF_ds[lm_short_idx,roi],axis=0)
        # If error, exclude some datapoints
        except IndexError:
            if at_landmark_short[0] - idx_var[0] < 0:
                dF_mean_short = np.mean(dF_ds[lm_short_idx[1:],roi],axis=0)
                dF_std_short = np.std(dF_ds[lm_short_idx[1:],roi],axis=0)
            else:
                dF_mean_short = np.mean(dF_ds[lm_short_idx[:-2],roi],axis=0)
                dF_std_short = np.mean(dF_ds[lm_short_idx[:-2],roi],axis=0)

        # same, but different (this time for long trials)
        try:
            dF_mean_long = np.mean(dF_ds[lm_long_idx,roi],axis=0)
            dF_std_long = np.std(dF_ds[lm_long_idx,roi],axis=0)
        except IndexError:
            if at_landmark_long[0] - idx_var[0] < 0:
                dF_mean_long = np.mean(dF_ds[lm_long_idx[1:],roi],axis=0)
                dF_std_long = np.std(dF_ds[lm_long_idx[1:],roi],axis=0)
            else:
                dF_mean_long = np.mean(dF_ds[lm_long_idx[:-2],roi],axis=0)
                dF_std_long = np.std(dF_ds[lm_long_idx[:-2],roi],axis=0)

        # the following was added last minute, near the end of summer 2018; please excuse the atrocity
        # it counts task-engaged ROIs only if the traces:
        # # exceed two standard deviations at landmarks
        # # this threshold is exceeded in more than 25% of the trials
        if True:
            count = 0
            for i,stat in enumerate(lm_short_idx):
                try:
                    if (any(dF_ds[stat,roi] > dF_std_short[i]*2)):
                        count += 1
                except:
                    pass

            if (count/lm_short_idx.shape[0]) > 0.25:
                if (max(dF_mean_short) > np.mean(dF_std_short)*2):
                    count_lm_short += 1

            count = None                # resetting variable for good measure

            count = 0
            for i,stat in enumerate(lm_long_idx):
                try:
                    if (any(dF_ds[stat,roi] > dF_std_long[i]*2)):
                        count += 1
                except IndexError:
                    pass

            if (count/lm_long_idx.shape[0]) > 0.25:
                if (max(dF_mean_long) > np.mean(dF_std_long)*2):
                    count_lm_long += 1

            count = None

        if plotting:
            for i,l in enumerate(at_landmark_short):
                ax1.axvline((sampling_rate*TIME_WINDOW)/2, lw=2, c='0')
                ax1.plot(dF_ds[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
                plt.sca(ax1)
                plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
                ax2.plot(dF_mean_short,c='#2864AF')
                ax1.set_title('Short track CL',fontdict={'fontweight':'bold'})
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('∆F/F')
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax2.set_ylim(-0.5, 3)

            for i,l in enumerate(at_landmark_long):
                ax3.axvline((sampling_rate*TIME_WINDOW)/2, lw=2, c='0')
                ax3.plot(dF_ds[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
                plt.sca(ax3)
                plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
                ax4.plot(dF_mean_long,c='#6F4B76')
                ax3.set_title('Long track CL',fontdict={'fontweight':'bold'})
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('∆F/F')
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                ax3.spines['left'].set_visible(False)
                ax4.spines['top'].set_visible(False)
                ax4.spines['right'].set_visible(False)
                ax4.spines['left'].set_visible(False)
                ax4.set_ylim(-0.5, 3)

        # Samesies, but for openloop
        trial_number,trial_ind = np.unique(behav_ds_ol[:,6],return_index=True)
        total_trials = max(trial_number)

        try:
            dF_mean_short_ol = np.mean(dF_ds_ol[lm_short_ol_idx,roi],axis=0)
            dF_std_short_ol = np.std(dF_ds_ol[lm_short_ol_idx,roi],axis=0)
        except IndexError:
            if at_landmark_short_ol[0] - idx_var[0] < 0:
                dF_mean_short_ol = np.mean(dF_ds_ol[lm_short_ol_idx[1:],roi],axis=0)
                dF_std_short_ol = np.std(dF_ds_ol[lm_short_ol_idx[1:],roi],axis=0)
            else:
                dF_mean_short_ol = np.mean(dF_ds_ol[lm_short_ol_idx[:-2],roi],axis=0)
                dF_std_short_ol = np.std(dF_ds_ol[lm_short_ol_idx[:-2],roi],axis=0)
        try:
            dF_mean_long_ol = np.mean(dF_ds_ol[lm_long_ol_idx,roi],axis=0)
            dF_std_long_ol = np.std(dF_ds_ol[lm_long_ol_idx,roi],axis=0)
        except IndexError:
            if at_landmark_long_ol[0] - idx_var[0] < 0:
                dF_mean_long_ol = np.mean(dF_ds_ol[lm_long_ol_idx[1:],roi],axis=0)
                dF_std_long_ol = np.std(dF_ds_ol[lm_long_ol_idx[1:],roi],axis=0)
            else:
                dF_mean_long_ol = np.mean(dF_ds_ol[lm_long_ol_idx[:-2],roi],axis=0)
                dF_std_long_ol = np.std(dF_ds_ol[lm_long_ol_idx[:-2],roi],axis=0)

        if True:

            count = 0
            for i,stat in enumerate(lm_short_ol_idx):
                try:
                    if (any(dF_ds_ol[stat,roi] > dF_std_short[i]*2)):
                        count += 1
                except IndexError:
                    pass

            if (count/lm_short_ol_idx.shape[0]) > 0.25:
                if (max(dF_mean_short_ol) > np.mean(dF_std_short_ol)*2):
                    count_lm_short_ol += 1

            count = None

            count = 0
            for i,stat in enumerate(lm_long_ol_idx):
                try:
                    if (any(dF_ds_ol[stat,roi] > dF_std_short[i]*2)):
                        count += 1
                except IndexError:
                    pass

            if (count/lm_long_ol_idx.shape[0]) > 0.25:
                if (max(dF_mean_long_ol) > np.mean(dF_std_long_ol)*2):
                    count_lm_long_ol += 1

            count = None



        ###
        if plotting:
            for i,l in enumerate(at_landmark_short_ol):
                ax5.axvline((sampling_rate_ol*TIME_WINDOW)/2, lw=2, c='0')
                ax5.plot(dF_ds_ol[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
                plt.sca(ax5)
                plt.xticks([(sampling_rate/sampling_rate_ol)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
                ax6.plot(dF_mean_short_ol,c='#4d8ad6')
                ax5.set_title('Short track OL',fontdict={'fontweight':'bold'})
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('∆F/F')
                ax5.spines['top'].set_visible(False)
                ax5.spines['right'].set_visible(False)
                ax5.spines['left'].set_visible(False)
                ax6.spines['top'].set_visible(False)
                ax6.spines['right'].set_visible(False)
                ax6.spines['left'].set_visible(False)
                ax6.set_ylim(-0.5, 3)


            for i,l in enumerate(at_landmark_long_ol):
                ax7.axvline((sampling_rate_ol*TIME_WINDOW)/2, lw=2, c='0')
                ax7.plot(dF_ds_ol[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
                plt.sca(ax7)
                plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
                ax8.plot(dF_mean_long_ol,c='#8c5f95')
                ax7.set_title('Long track OL',fontdict={'fontweight':'bold'})
                ax7.set_xlabel('Time (s)')
                ax7.set_ylabel('∆F/F')
                ax7.spines['top'].set_visible(False)
                ax7.spines['right'].set_visible(False)
                ax7.spines['left'].set_visible(False)
                ax8.spines['top'].set_visible(False)
                ax8.spines['right'].set_visible(False)
                ax8.spines['left'].set_visible(False)
                ax8.set_ylim(-0.5, 3)

            fig.suptitle('Landmark: '+ str(mouse)+' '+str(sess)+'\nROI '+str(roi+1))
            plt.subplots_adjust(top=0.94,hspace=0.4)

            plt.show()

            if onset_LM == 200:
                __fname = 'landmark_onset_dF_ROI_'+str(roi+1)
            elif onset_LM == 240:
                __fname = 'landmark_offset_dF_ROI_'+str(roi+1)

            if __name__ == "__main__":
                if not os.path.isdir(content['figure_output_path'] + subfolder):
                    os.mkdir(content['figure_output_path'] + subfolder)
                fname = content['figure_output_path'] + subfolder + os.sep + __fname + '.' + fformat
                print(fname)
                try:
                    fig.savefig(fname, format=fformat)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)

    # Print how many ROIs qualified as landmark-associated
    print('CL '+str(dF_ds.shape[1]))
    print('\t'+str(sess)+' short: '+ str(count_lm_short))
    print('\t'+str(sess)+' long: '+str(count_lm_long))
    print('\nOL '+str(dF_ds_ol.shape[1]))
    print('\t'+str(sess_ol)+' short: '+str(count_lm_short_ol))
    print('\t'+str(sess_ol)+' long: '+str(count_lm_long_ol))

# ----
import yaml
import h5py
import os

with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

# # RSC

# LF180119_1
MOUSE = 'LF180119_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2018316_1'
SESSION_OL = 'Day2018316_openloop_1'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

SESSION = 'Day2018316_2'
SESSION_OL = 'Day2018316_openloop_2'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# LF171212_2
MOUSE = 'LF171212_2'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2018218_1'
SESSION_OL = 'Day2018218_openloop_1'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

SESSION = 'Day2018218_2'
SESSION_OL = 'Day2018218_openloop_2'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# LF170613_1
MOUSE = 'LF170613_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day201784'
SESSION_OL = 'Day201784_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
SESSION = 'Day2017719'
SESSION_OL = 'Day2017719_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# LF170222_1
MOUSE = 'LF170222_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2017615'
SESSION_OL = 'Day2017615_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

SESSION = 'Day201776'
SESSION_OL = 'Day201776_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# LF170421_2
MOUSE = 'LF170421_2'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2017719'
SESSION_OL = 'Day2017719_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# SESSION = 'Day2017720' # there's a problem with this ds
# SESSION_OL = 'Day2017720_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# LF170420_1
MOUSE = 'LF170420_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2017612'
SESSION_OL = 'Day2017612_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

SESSION = 'Day201783'
SESSION_OL = 'Day201783_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

SESSION = 'Day2017719'
SESSION_OL = 'Day2017719_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# LF170110_2
MOUSE = 'LF170110_2'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day201748_1'
SESSION_OL = 'Day201748_openloop_1'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

SESSION = 'Day2017331'
SESSION_OL = 'Day2017331_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# LF170612_1
MOUSE = 'LF170612_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2017719'
SESSION_OL = 'Day2017719_openloop'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# LF171211_1
MOUSE = 'LF171211_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2018314_1'
SESSION_OL = 'Day2018314_openloop_1'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

SESSION = 'Day2018314_2'
SESSION_OL = 'Day2018314_openloop_2'
fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)


# # V1

# # LF180112_2
# MOUSE = 'LF180112_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018322_1'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# # LF170214_1
# MOUSE = 'LF170214_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day201777'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2017714'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
