'''
Plot dF at movement onsetself.
    board: plot all ROIs' dFs for each movement onset event in a single figure

    not board:
        all_rois: plots dFs (raw and averaged) of each ROI at all movement onset events
        shuffling:


@author: rmojica@mit.edu
'''
def dF_movement_onset_board(h5path, mouse, sess, fformat='png',subfolder=[]):
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
    from event_ind import movement_onset, trial_transition, track_transition

    sys.path.append('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/Figures')
    from fig_speed_loc import speed_loc
    from fig_lmi import fig_LMI

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
        # blackbox = 5
        #
        # tracklength_short = 320
        # tracklength_long = 380

    high_thresh = 100 + np.median(behav_ds[behav_ds[:,4] != 5, 3])
    low_thresh = 0.0

    behav_ds = np.delete(behav_ds,np.where(behav_ds[:,3] > high_thresh),0)
    # behav_ds = np.delete(behav_ds,np.where(behav_ds[:,1] < 50),0)

    speed_thresh = 1
    time_thresh = 1
    onset_event = movement_onset(behav_ds, speed_thresh, time_thresh)

    if __name__ == "__main__":
        print('\nOnset events identified with')
        print('\tSpeed threshold:\t'+str(speed_thresh) +' cm/s')
        print('\tTime threshold:\t'+str(time_thresh) + ' s')

    sampling_rate = 1/behav_ds[0,2]
    print('\nSampling rate \t'+str(round(sampling_rate,4))+' Hz')

    # desired_window = 3
    desired_window = 5          # desired time window for peri-event indices calculation (in seconds)

    idx_var = [int((sampling_rate*desired_window)/2),int((sampling_rate*desired_window)/2)]

    time_window = round(behav_ds[onset_event[0]+idx_var[0],0] - behav_ds[onset_event[0]-idx_var[0],0],4)
    print('Time window \t\t'+str(time_window)+' s')

    saving = True

    # create board with all ROIs' dFs for each mvmnt onset event
    if False:
        number_of_plots = dF_ds.shape[1]

        ax_names_gen = lambda x,y: str(x) + str(y)
        ax_names = [ax_names_gen('ax', x) for x in range(dF_ds.shape[1]) if True]

        __fname = 'dF_mvmt_onset_board' + str(sess)

        for idx in range(len(behav_ds[onset_event])):
            fig = plt.figure(figsize=(12,(1.82 * number_of_plots)))
            fig.subplots_adjust(hspace=0.6, top=0.975)

            for i,u in enumerate(range(dF_ds.shape[1])):
                u += 1
                ax_names[i] = plt.subplot(number_of_plots,5,u)

            for roi in range(dF_ds.shape[1]):

                plt.sca(ax_names[roi])
                plt.plot(behav_ds[onset_event[idx]-idx_var[0]:onset_event[idx]+idx_var[0],0], dF_ds[onset_event[idx]-idx_var[0]:onset_event[idx]+idx_var[0],roi], color='#BB2E07')
                plt.xticks([behav_ds[onset_event[idx]-idx_var[0],0],behav_ds[onset_event[idx],0],behav_ds[onset_event[idx]+idx_var[0],0]],['-1.5','0','1.5'])
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                ymin, ymax = ax_names[roi].get_ylim()
                plt.vlines(behav_ds[onset_event[idx],0],ymin,ymax)
                plt.xlabel('Time (s)')

                if roi % 5 == 0:
                    plt.ylabel('∆F/F',fontsize=8)

                first_rois = [0,1,2,3,4]
                if roi in first_rois :
                    plt.title('Onset')
                else:
                    plt.title('ROI '+str(roi),fontsize=10)

                plt.suptitle(mouse+' '+sess+': Movement onset @ '+str(round(behav_ds[onset_event[idx],1],2)) + ' cm')

            if saving:

                if not os.path.isdir(content['figure_output_path'] + subfolder):
                    os.mkdir(content['figure_output_path'] + subfolder)
                fname = content['figure_output_path'] + subfolder + os.sep + __fname + '_at_' + str(int(round(behav_ds[onset_event[idx],1],2))) + '.' + fformat
                print(fname)
                try:
                    fig.savefig(fname, format=fformat, bbox_inches = 'tight', pad_inches=0.3)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)

    if False:

        # Plots dFs of all ROIs on every movement onset event
        if True:

            __fname = 'dF_movement_onset_at_'

            for idx in range(len(behav_ds[onset_event])):
                fig = plt.figure(figsize=(8,8))

                for roi in range(dF_ds.shape[1]):
                    plt.plot(behav_ds[onset_event[idx]-idx_var[0]:onset_event[idx]+idx_var[0],0], dF_ds[onset_event[idx]-idx_var[0]:onset_event[idx]+idx_var[0],roi], color='#BB2E07')
                    plt.xticks([behav_ds[onset_event[idx]-idx_var[0],0],behav_ds[onset_event[idx],0],behav_ds[onset_event[idx]+idx_var[0],0]],['-1.5','0','1.5'])
                    plt.vlines(behav_ds[onset_event[idx],0],-1,15)
                    plt.xlabel('Time (s)')
                    plt.ylabel('∆F/F')

                    plt.suptitle(mouse+' '+sess+': Movement onset @ '+str(round(behav_ds[onset_event[idx],1],2)) + ' cm')

                # if saving:
                #     if not os.path.isdir(content['figure_output_path'] + subfolder):
                #         os.mkdir(content['figure_output_path'] + subfolder)
                #     fname = content['figure_output_path'] + subfolder + os.sep + __fname + str(round(behav_ds[onset_event[idx],1],2)) + '.' + fformat
                #     print(fname)
                #     try:
                #         fig.savefig(fname, format=fformat, bbox_inches = 'tight', pad_inches=0.3)
                #     except:
                #         exc_type, exc_value, exc_traceback = sys.exc_info()
                #         traceback.print_exception(exc_type, exc_value, exc_traceback,
                #                               limit=2, file=sys.stdout)

    # Plots dFs (raw and averaged) of each ROI at all movement onset events
    if False:

        if True:
            secs = 5
            mvmt_onset_ds = np.zeros((len(onset_event),int(sampling_rate*secs)),dtype=int)

            dF_bins = int(sampling_rate*secs)
            print('Bins \t\t\t' +str(dF_bins))

            for roi in range(dF_ds.shape[1]):
                fig = plt.figure(figsize=(6,6))
                ax1 = plt.subplot(111)
                plt.subplots_adjust(top=0.95)

                dF_means = []

                for i,oe in enumerate(onset_event):
                    mvmt_onset_ds[i,:] = np.arange(oe-idx_var[0],oe+idx_var[0],step=1)
                    ax1.plot(dF_ds[mvmt_onset_ds[i,:],roi], alpha=0.4,c='#ff7400')
                    plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*secs)/2, sampling_rate*secs], ['-2.5','0','2.5'])
                    plt.xlabel('Time (s)')
                    plt.ylabel('∆F/F')

                    dF_mean,dF_edges,_ = stats.binned_statistic(mvmt_onset_ds[i,:],dF_ds[mvmt_onset_ds[i,:],roi],statistic='mean',bins=dF_bins)
                    dF_means.append(dF_mean)

                dF_bin_mean = np.nanmean(dF_means,axis=0)
                dF_bin_sem = stats.sem(dF_bin_mean,axis=0,nan_policy='omit')

                ax1.plot(range(dF_bins),dF_bin_mean, c='#000000')
                plt.fill_between(range(dF_bins),dF_bin_mean-dF_bin_sem,dF_bin_mean+dF_bin_sem,facecolor='#5c5c5c',alpha=0.3)
                plt.suptitle(mouse+' '+sess+': Movement onset ROI '+str(roi+1))

        # shuffle dFs to see null distribution of ROI activity
        if False:

            __fname = 'shuffled_dF_ROI_'

            for roi in range(dF_ds.shape[1]):

                fig = plt.figure(figsize=(6,6))
                fig.subplots_adjust(top=0.95)

                dF_shuff = fig_LMI(h5path,sess,roi,'test')
                sns.distplot(dF_shuff, 1000)
                plt.suptitle(mouse+' '+sess+': Shuffled ∆F/F, ROI '+str(roi+1))

                if saving:
                    if not os.path.isdir(content['figure_output_path'] + subfolder):
                        os.mkdir(content['figure_output_path'] + subfolder)
                    fname = content['figure_output_path'] + subfolder + os.sep + __fname + str(roi+1) + '.' + fformat
                    print(fname)
                    try:
                        fig.savefig(fname, format=fformat, bbox_inches = 'tight', pad_inches=0.3)
                    except:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)

    if True:

        tmp = trial_transition(behav_ds)
        tmp = np.insert(tmp,0,0)
        track_onset = [x for x in tmp if behav_ds[x,4] != 5]
        track_onset.remove(0)   # this is bad programming. make case for handling first onset

        mvmt_onset_ds = np.zeros((len(onset_event),int(sampling_rate*5)),dtype=int)
        track_onset_ds = np.zeros((len(track_onset),int(sampling_rate*5)),dtype=int)

        idx_var_one_s = [int((sampling_rate*1)/2),int((sampling_rate*1)/2)]
        time_window = math.ceil(behav_ds[onset_event[0]+idx_var_one_s[0],0] - behav_ds[onset_event[0]-idx_var_one_s[0],0])

        # for roi in range(dF_ds.shape[1]):
        for i,oe in enumerate(onset_event):
            mvmt_onset_ds[i,:] = np.arange(oe-idx_var[0],oe+idx_var[0],step=1)

        for i,oe in enumerate(track_onset):
            try:
                track_onset_ds[i,:] = np.arange(oe-idx_var[0],oe+idx_var[0],step=1)
            except:
                x = np.arange(oe,oe+idx_var[0],step=1,dtype=float)
                f = 0
                while f < len(track_onset):
                    x = np.insert(x,0,np.nan)
                    f += 1
                track_onset_ds[i,:] = x

        mvmt_track_onset = np.zeros((len(onset_event),int(sampling_rate*5)),dtype=int)
        l = 0
        for i,m_oe in enumerate(onset_event):
            ii = m_oe
            for k,t_oe in enumerate(track_onset):
                kk = t_oe
                if int(abs(behav_ds[ii,0] - behav_ds[kk,0])) in np.arange(0,time_window+0.1,step=0.1):
                    mvmt_track_onset[l,:] = np.arange(m_oe-idx_var[0],m_oe+idx_var[0],step=1)
                    l += 1
        mvmt_track_onset =  np.delete(mvmt_track_onset,np.where(mvmt_track_onset[:,0] == 0),0)

        for i,t_oe in enumerate(track_onset_ds):
            for k,m_oe in enumerate(mvmt_onset_ds):
                if any(t_oe == m_oe):
                    np.delete(track_onset_ds, mvmt_onset_ds[k])

        for k,m_oe in enumerate(mvmt_onset_ds):
            for i,t_oe in enumerate(track_onset_ds):
                if any(m_oe == t_oe):
                    np.delete(mvmt_onset_ds, track_onset_ds[i])

        if True:
            for roi in range(dF_ds.shape[1]):
                fig = plt.figure(figsize=(6,6))
                ax1 = plt.subplot(221)
                ax2 = plt.subplot(222)
                ax3 = plt.subplot(223)
                plt.subplots_adjust(hspace=0.3,top=0.95)
                plt.suptitle('ROI '+str(roi+1))

                for i,oe in enumerate(mvmt_track_onset):
                    plt.sca(ax1)
                    ax1.plot(dF_ds[mvmt_track_onset[i],roi],c='#ff7400',alpha=0.4)
                    plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*5)/2, sampling_rate*5], ['-2.5','0','2.5'])
                    plt.xlabel('Time (s)')
                    plt.ylabel('∆F/F')
                    plt.title('Track & movement onset')

                for i,oe in enumerate(track_onset_ds):
                    plt.sca(ax2)
                    ax2.plot(dF_ds[track_onset_ds[i,:],roi],c='#ff7400',alpha=0.4)
                    plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*5)/2, sampling_rate*5], ['-2.5','0','2.5'])
                    plt.xlabel('Time (s)')
                    plt.ylabel('∆F/F')
                    plt.title('Track onset')

                for i,oe in enumerate(mvmt_onset_ds):
                    plt.sca(ax3)
                    # ax3.plot(dF_ds[mvmt_onset_ds[i,:],roi],c='#ff7400',alpha=0.4)
                    plt.xticks([(sampling_rate/sampling_rate)-1, (sampling_rate*5)/2, sampling_rate*5], ['-2.5','0','2.5'])
                    plt.xlabel('Time (s)')
                    plt.ylabel('∆F/F')
                    plt.title('Movement onset')

                plt.sca(ax1)
                ymin,ymax = ax1.get_ylim()
                plt.vlines((sampling_rate*5)/2,ymin,ymax)

                plt.sca(ax2)
                ymin,ymax = ax2.get_ylim()
                plt.vlines((sampling_rate*5)/2,ymin,ymax)

                plt.sca(ax3)
                ymin,ymax = ax3.get_ylim()
                plt.vlines((sampling_rate*5)/2,ymin,ymax)

                plt.show()

# ----

import yaml
import h5py
import os

with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

# # Stage 5 Expert

# RSC

# # LF180119_1
# MOUSE = 'LF180119_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018316_1'
# dF_movement_onset_board(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2018316_2'
# SESSION_OL = 'Day2018316_2_openloop_2'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+'_'+SESSION)

# # LF171212_2
# MOUSE = 'LF171212_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018218_1'
# dF_movement_onset_board(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# SESSION = 'Day2018218_2'
# SESSION_OL = 'Day2018218_openloop_2'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# # LF170613_1
# MOUSE = 'LF170613_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day201784'
# SESSION_OL = 'Day201784_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'landmark_dF'+str(SESSION),subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2017719'
# SESSION_OL = 'Day2017719_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'landmark_dF'+str(SESSION),subfolder=MOUSE+'_'+SESSION)

# # LF170222_1
# MOUSE = 'LF170222_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day20170615'
# SESSION_OL = 'Day20170615_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION),subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day201776'
# SESSION_OL = 'Day201776_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# LF170421_2
MOUSE = 'LF170421_2'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day20170719'
dF_movement_onset_board(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170720'
# SESSION_OL = 'Day20170720_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, subfolder=MOUSE+'_'+SESSION)

# # LF170420_1
# MOUSE = 'LF170420_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day20170612'
# SESSION_OL = 'Day20170612_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day201783'
# SESSION_OL = 'Day201783_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL,'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170719'
# SESSION_OL = 'Day20170719_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+'_'+SESSION)

# # LF170110_2
# MOUSE = 'LF170110_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day201748'
# SESSION_OL = 'Day201748_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170331'
# SESSION_OL = 'Day20170331_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+'_'+SESSION)
#
# # LF170612_1
# SESSION = 'Day2017719'
# SESSION_OL = 'Day2017719_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# # LF171211_1
# SESSION = 'Day2018314_1'
# SESSION_OL = 'Day2018314_openloop_1'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018314_2'
# SESSION_OL = 'Day2018314_openloop_2'
# fig_landmark_diff(h5path, MOUSE, SESSION, SESSION_OL, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)


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
