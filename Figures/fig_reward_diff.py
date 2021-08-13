'''
Plot neuronal responses to rewards as a function of time.
This code can also be adapted to plot responses at any point in the track.

Args:
    sess: regular session, of course
    sess_ol: suffix OL will always refer to Open Loop

Returns:
    Creates a figure with four subplots and populates each with the corresponding
    calcium traces.
    Saved in specified subfolder, with format fformat.

Twin function of fig_landmark_diff

@author: rmojica@mit.edu
'''

def fig_reward_diff(h5path,mouse,sess,sess_ol,fname='reward_dF_ROI',fformat='png',subfolder=[]):
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

    # determine sampling rate for session to have accurate time readouts
    sampling_rate = 1/behav_ds[0,2]
    print('\nSampling rate \t\t'+str(round(sampling_rate,4))+' Hz')

    # both sessions should have the same sampling rate but there may be exceptions
    sampling_rate_ol = 1/behav_ds_ol[0,2]
    print('OL sampling rate \t\t'+str(round(sampling_rate,4))+' Hz')

    # desired time window for peri-event indices calculation (in seconds)
    TIME_WINDOW = 6

    # peri-event index counts
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

    tracklength_short = 320
    tracklength_long = 380

    # set up bin width to average dF/F across
    binwidth = int(10)

    bins_short = TIME_WINDOW*binwidth            # calculate number of bins
    bins_long = TIME_WINDOW*binwidth

    # add extra column to ds to retain original index after data filtering
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
#        for i in range(len(first_onset_ind)):
#            if curr_trial_num == first_onset_ind[i]:
#                curr_trial_onset = behav_ds[np.where(first_onset[i] == curr_trial[:,9])]
#                print(curr_trial_onset)
#            else:
#                pass
        if all(curr_trial[2:-3,4] == 3):                                # sometimes trial number does not change until a couple frames later
            short_trial_time.append(curr_trial[-1,0] - curr_trial[1,0])
        if all(curr_trial[2:-3,4] == 4):
            long_trial_time.append(curr_trial[-1,0] - curr_trial[1,0])

    # Mmmm, stats (for time spent in trials)
    trial_time_stats = [np.mean(short_trial_time), np.std(short_trial_time), np.mean(long_trial_time), np.std(long_trial_time)]
    print('Avg time:  Short track\t\t'+str(round(trial_time_stats[0],2))+'s')
    print('Avg time:  Long track\t\t'+str(round(trial_time_stats[2],2))+'s')

    short_trials = filter_trials(behav_ds, [], ['tracknumber',3])

    # purge dataset; eliminate all those trials where the mouse just stood around
    filt_short_trials = filter_trials(behav_ds, [], ['maxtotaltime',trial_time_stats[0]+trial_time_stats[1]*3],short_trials)
    print(behav_ds[filt_short_trials,1])
    long_trials = filter_trials(behav_ds, [], ['tracknumber',4])
    filt_long_trials = filter_trials(behav_ds, [], ['maxtotaltime',trial_time_stats[2]+trial_time_stats[3]*3],long_trials)

    short_trials_ol = filter_trials(behav_ds_ol, [], ['tracknumber',3])
    long_trials_ol = filter_trials(behav_ds_ol, [], ['tracknumber',4])

    at_reward = [320, 380]
    at_rew_short = at_reward[0]
    at_rew_long = at_reward[1]

    at_reward = event_ind(behav_ds,['at_location',at_rew_short])         # first index at location 200
    at_reward = at_reward.astype(int)

    at_reward_ol = event_ind(behav_ds_ol,['at_location',at_rew_short])
    at_reward_ol = at_reward_ol.astype(int)

    # very stupid way of dividing at_reward into the different trials
    at_reward_short = []
    lm_short_idx = []
    for i,lm in enumerate(at_reward[:,1]):
        for k,t in enumerate(filt_short_trials):
            if lm == t:
                at_reward_short.append(at_reward[i,0])
                lm_short_idx.append(list(range(at_reward[i,0]-idx_var[0],at_reward[i,0]+idx_var[0])))
            else:
                pass
    at_reward_short = np.array(at_reward_short)
    lm_short_idx = np.array(lm_short_idx)

    at_reward = event_ind(behav_ds,['at_location',at_rew_long])         # first index at location 200
    at_reward = at_reward.astype(int)

    at_reward_long = []
    lm_long_idx = []
    for i,lm in enumerate(at_reward[:,1]):
        for k,t in enumerate(filt_long_trials):
            if lm == t:
                at_reward_long.append(at_reward[i,0])
                lm_long_idx.append(list(range(at_reward[i,0]-idx_var[0],at_reward[i,0]+idx_var[0])))
            else:
                pass
    at_reward_long = np.array(at_reward_long)
    lm_long_idx = np.array(lm_long_idx)

    at_reward_short_ol = []
    lm_short_ol_idx = []
    for i,lm in enumerate(at_reward_ol[:,1]):
        for k,t in enumerate(short_trials_ol):
            if lm == t:
                try:
                    at_reward_short_ol.append(at_reward_ol[i,0])
                    lm_short_ol_idx.append(list(range(at_reward_ol[i,0]-idx_var[0],at_reward_ol[i,0]+idx_var[0])))
                except:
                    pass
            else:
                pass
    at_reward_short_ol = np.array(at_reward_short_ol)
    lm_short_ol_idx = np.array(lm_short_ol_idx)

    at_reward_ol = event_ind(behav_ds_ol,['at_location',at_rew_long])
    at_reward_ol = at_reward_ol.astype(int)

    at_reward_long_ol = []
    lm_long_ol_idx = []
    for i,lm in enumerate(at_reward_ol[:,1]):
        for k,t in enumerate(long_trials_ol):
            if lm == t:
                try:
                    at_reward_long_ol.append(at_reward_ol[i,0])
                    lm_long_ol_idx.append(list(range(at_reward_ol[i,0]-idx_var[0],at_reward_ol[i,0]+idx_var[0])))
                except:
                    pass
            else:
                pass
    at_reward_long_ol = np.array(at_reward_long_ol)
    lm_long_ol_idx = np.array(lm_long_ol_idx)

    ### get every ROI's avg dF for every trial
    # Regular session
    for roi in range(dF_ds.shape[1]):

        fig = plt.figure(figsize=(10,12))
        ax1 = plt.subplot(421)
        ax2 = ax1#.twiny()           # create second x-axis to plot binned stats
        ax3 = plt.subplot(423)
        ax4 = ax3#.twiny()
        ax5 = plt.subplot(422)
        ax6 = ax5#.twiny()
        ax7 = plt.subplot(424)
        ax8 = ax7#.twiny()

        trial_number,trial_ind = np.unique(behav_ds[:,6],return_index=True)
        total_trials = max(trial_number)

        try:
            dF_mean_short = np.mean(dF_ds[lm_short_idx,roi],axis=0)
        except IndexError:
            if at_reward_short[0] - idx_var[0] < 0:
                dF_mean_short = np.mean(dF_ds[lm_short_idx[1:],roi],axis=0)
            else:
                dF_mean_short = np.mean(dF_ds[lm_short_idx[:-2],roi],axis=0)
        try:
            dF_mean_long = np.mean(dF_ds[lm_long_idx,roi],axis=0)
        except IndexError:
            if at_reward_long[0] - idx_var[0] < 0:
                dF_mean_long = np.mean(dF_ds[lm_long_idx[1:],roi],axis=0)
            else:
                dF_mean_long = np.mean(dF_ds[lm_long_idx[:-2],roi],axis=0)

        for i,l in enumerate(at_reward_short):
            ax1.axvline((sampling_rate*TIME_WINDOW)/2, lw=2, c='0')
            ax1.plot(dF_ds[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
            plt.sca(ax1)
            plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
            ax2.plot(dF_mean_short,c='#2864AF')
            ax2.set_xticks([],[])
            ax1.set_title('Short track',fontdict={'fontweight':'bold'})
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('∆F/F')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)

        for i,l in enumerate(at_reward_long):
            ax3.axvline((sampling_rate*TIME_WINDOW)/2, lw=2, c='0')
            ax3.plot(dF_ds[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
            plt.sca(ax3)
            plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
            ax4.plot(dF_mean_long,c='#6F4B76')
            ax4.set_xticks([],[])
            ax3.set_title('Long track',fontdict={'fontweight':'bold'})
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('∆F/F')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['left'].set_visible(False)

        # Openloop
        trial_number,trial_ind = np.unique(behav_ds_ol[:,6],return_index=True)
        total_trials = max(trial_number)

        try:
            dF_mean_short_ol = np.mean(dF_ds_ol[lm_short_ol_idx,roi],axis=0)
        except IndexError:
            if at_reward_short_ol[0] - idx_var[0] < 0:
                dF_mean_short_ol = np.mean(dF_ds_ol[lm_short_ol_idx[1:],roi],axis=0)
            else:
                dF_mean_short_ol = np.mean(dF_ds_ol[lm_short_ol_idx[:-2],roi],axis=0)
        try:
            dF_mean_long_ol = np.mean(dF_ds_ol[lm_long_ol_idx,roi],axis=0)
        except IndexError:
            if at_reward_long_ol[0] - idx_var[0] < 0:
                dF_mean_long_ol = np.mean(dF_ds_ol[lm_long_ol_idx[1:],roi],axis=0)
            else:
                dF_mean_long_ol = np.mean(dF_ds_ol[lm_long_ol_idx[:-2],roi],axis=0)
        ###

        for i,l in enumerate(at_reward_short_ol):
            ax5.axvline((sampling_rate*TIME_WINDOW)/2, lw=2, c='0')
            ax5.plot(dF_ds_ol[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
            plt.sca(ax5)
            plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
            ax6.plot(dF_mean_short_ol,c='#4d8ad6')
            ax6.set_xticks([],[])
            ax5.set_title('Short track OL',fontdict={'fontweight':'bold'})
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('∆F/F')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['left'].set_visible(False)
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            ax6.spines['left'].set_visible(False)

        for i,l in enumerate(at_reward_long_ol):
            ax7.axvline((sampling_rate*TIME_WINDOW)/2, lw=2, c='0')
            ax7.plot(dF_ds_ol[int(l)-idx_var[0]:int(l)+idx_var[0],roi],alpha=0.4,c='#BFC0BF')
            plt.sca(ax7)
            plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*TIME_WINDOW)/2, sampling_rate*TIME_WINDOW], ['-3','0','3'])
            ax8.plot(dF_mean_long_ol,c='#8c5f95')
            ax8.set_xticks([],[])
            ax7.set_title('Long track OL',fontdict={'fontweight':'bold'})
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('∆F/F')
            ax7.spines['top'].set_visible(False)
            ax7.spines['right'].set_visible(False)
            ax7.spines['left'].set_visible(False)
            ax8.spines['top'].set_visible(False)
            ax8.spines['right'].set_visible(False)
            ax8.spines['left'].set_visible(False)

        fig.suptitle(str(mouse)+' '+str(sess)+'\nROI '+str(roi+1))
        plt.subplots_adjust(top=0.94,hspace=0.4)

        plt.show()

        __fname = 'reward_dF_ROI_'+str(roi+1)

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

# ----
import yaml
import h5py
import os

with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

# # RSC

# # LF180119_1
# MOUSE = 'LF180119_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018316_1'
# SESSION_OL = 'Day2018316_openloop_1'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018316_2'
# SESSION_OL = 'Day2018316_openloop_2'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF171212_2
# MOUSE = 'LF171212_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018218_1'
# SESSION_OL = 'Day2018218_openloop_1'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018218_2'
# SESSION_OL = 'Day2018218_openloop_2'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF170613_1
# MOUSE = 'LF170613_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day201784'
# SESSION_OL = 'Day201784_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2017719'
# SESSION_OL = 'Day2017719_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF170222_1
# MOUSE = 'LF170222_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2017615'
# SESSION_OL = 'Day2017615_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day201776'
# SESSION_OL = 'Day201776_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF170421_2
# MOUSE = 'LF170421_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2017719'
# SESSION_OL = 'Day2017719_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2017720' # there's a problem with this ds
# SESSION_OL = 'Day2017720_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF170420_1
# MOUSE = 'LF170420_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2017612'
# SESSION_OL = 'Day2017612_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day201783'
# SESSION_OL = 'Day201783_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2017719'
# SESSION_OL = 'Day2017719_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF170110_2
# MOUSE = 'LF170110_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day201748_1'
# SESSION_OL = 'Day201748_openloop_1'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2017331'
# SESSION_OL = 'Day2017331_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF170612_1
# MOUSE = 'LF170612_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2017719'
# SESSION_OL = 'Day2017719_openloop'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)

# # LF171211_1
# MOUSE = 'LF171211_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018314_1'
# SESSION_OL = 'Day2018314_openloop_1'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018314_2'
# SESSION_OL = 'Day2018314_openloop_2'
# fig_reward_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+SESSION)


# # V1

# # LF180112_2
# MOUSE = 'LF180112_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018322_1'
# fig_reward_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# # LF170214_1
# MOUSE = 'LF170214_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day201777'
# fig_reward_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2017714'
# fig_reward_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
