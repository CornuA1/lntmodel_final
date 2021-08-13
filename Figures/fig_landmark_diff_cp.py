'''
Plot neuron responses to each landmark, both in regular and openloop sessions.

@author: rmojica@mit.edu
'''

def fig_landmark_diff(h5path,mouse,sess,fname='landmark_dF_ROI',fformat='png',subfolder=[]):
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
    h5dat.close()

    tracklength_short = 320
    tracklength_long = 380

    binwidth = 2
    binwidth = int(binwidth)

    bins_short = tracklength_short//binwidth            # calculate number of bins
    bins_long = tracklength_long//binwidth

    # behav_ds = np.delete(behav_ds,np.where(behav_ds[:,4]==5),0)

    # dF_ds = np.delete(dF_ds,np.where(behav_ds[behav_ds[:,4]==3,1] >= tracklength_short-20),0)
    # behav_ds = np.delete(behav_ds,np.where(behav_ds[behav_ds[:,4]==3,1] >= tracklength_short-20),0)

    # dF_ds = np.delete(dF_ds,np.where(behav_ds[behav_ds[:,4]==4,1] >= tracklength_long-20),0)
    # behav_ds = np.delete(behav_ds,np.where(behav_ds[behav_ds[:,4]==4,1] >= tracklength_long-20),0)

    # dF_ds = np.delete(dF_ds,np.where(behav_ds[:,1] < 100),0)
    # behav_ds= np.delete(behav_ds,np.where(behav_ds[:,1] < 100),0)
    # landmark = event_ind(behav_ds,['at_location',200])
    # landmark = landmark.astype(int)

    trials_short = filter_trials(behav_ds, [], ['tracknumber', 3])
    print(trials_short)
    trials_long = filter_trials(behav_ds, [], ['tracknumber', 4])
    print(trials_long)

    ### get every ROI's avg dF for every trial
    # for roi in range(dF_ds.shape[1]):
    roi = 4

    trial_number,trial_ind = np.unique(behav_ds[:,6],return_index=True)
    total_trials = max(trial_number)

    dF_means_short = []
    for i,trial in enumerate(trials_short):
        curr_trial_behav = behav_ds[behav_ds[:,6] == trial,:]
        curr_trial_dF = dF_ds[behav_ds[:,6] == trial,:]
        dF_means,dF_edges_short,_ = stats.binned_statistic(curr_trial_behav[:,1],curr_trial_dF[:,roi],statistic='mean',bins=bins_short,range=(0,tracklength_short))
        dF_means_short.append(dF_means)

    bin_mean_short = np.nanmean(dF_means_short,axis=0)
    bin_sem_short = stats.sem(dF_means_short,axis=0,nan_policy='omit')

    dF_means_long = []
    for i,trial in enumerate(trials_long):
        curr_trial_behav = behav_ds[behav_ds[:,6] == trial,:]
        curr_trial_dF = dF_ds[behav_ds[:,6] == trial,:]
        dF_means,speed_edges_long,_ = stats.binned_statistic(curr_trial_behav[:,1],curr_trial_dF[:,roi],statistic='mean',bins=bins_long,range=(0,tracklength_long))
        dF_means_long.append(dF_means)

    bin_mean_long = np.nanmean(dF_means_long,axis=0)
    bin_sem_long = stats.sem(dF_means_long,axis=0, nan_policy='omit')
    ###


    # fig = plt.figure(figsize=(8,10))
    # ax1 = plt.subplot(211)
    # ax2 = ax1.twiny()           # create second x-axis
    # ax3 = plt.subplot(212)
    # ax4 = ax3.twiny()
    # for i,t in enumerate(trials_short):

    t = 231

    ###
    fig = plt.figure(figsize=(8,10))
    ax1 = plt.subplot(111)
    ax2 = ax1.twiny()
    ###

    ax1.plot(behav_ds[behav_ds[:,6]==t,1],dF_ds[behav_ds[:,6]==t,roi],alpha=0.4,c='#ff7400')
    ax2.plot(range(bins_short),bin_mean_short, c='#000000')
    ax2.set_xticks([],[])
    # ax1.set_xlim(160,260)
    # ax2.set_xlim(160//binwidth,260//binwidth)
    # ax1.axvline(210, lw=90, c='0.9',alpha=0.02)
    ax1.set_title(str(mouse)+' on '+str(sess)+': Short track',fontdict={'fontweight':'bold'})
    ax1.set_xlabel('Location (cm)')
    ax1.set_ylabel('∆F/F')

    ###
    plt.suptitle('Trial '+str(t))
    ###

    # t = 25
    #
    # ###
    # fig = plt.figure(figsize=(8,10))
    # ax1 = plt.subplot(111)
    # # ax2 = ax1.twiny()
    # ###
    #
    # ax1.plot(behav_ds[behav_ds[:,6]==t,1],dF_ds[behav_ds[:,6]==t,roi],alpha=0.4,c='#ff7400')
    # # ax2.plot(range(bins_short),bin_mean_short, c='#000000')
    # # ax2.set_xticks([],[])
    # # ax1.set_xlim(160, 260)
    # # ax2.set_xlim(160//binwidth,260//binwidth)
    # # ax1.axvline(210, lw=90, c='0.9',alpha=0.02)
    # ax1.set_title(str(mouse)+' on '+str(sess)+': Short track',fontdict={'fontweight':'bold'})
    # ax1.set_xlabel('Location (cm)')
    # ax1.set_ylabel('∆F/F')
    #
    # ###
    # plt.suptitle('Trial '+str(t))
    # ###
    #
    # t = 23
    #
    # ###
    # fig = plt.figure(figsize=(8,10))
    # ax1 = plt.subplot(111)
    # # ax2 = ax1.twiny()
    # ###
    #
    # ax1.plot(behav_ds[behav_ds[:,6]==t,1],dF_ds[behav_ds[:,6]==t,roi],alpha=0.4,c='#ff7400')
    # # ax2.plot(range(bins_short),bin_mean_short, c='#000000')
    # # ax2.set_xticks([],[])
    # # ax1.set_xlim(160,260)
    # # ax2.set_xlim(160//binwidth,260//binwidth)
    # # ax1.axvline(210, lw=90, c='0.9',alpha=0.02)
    # ax1.set_title(str(mouse)+' on '+str(sess)+': Short track',fontdict={'fontweight':'bold'})
    # ax1.set_xlabel('Location (cm)')
    # ax1.set_ylabel('∆F/F')
    #
    # ###
    # plt.suptitle('Trial '+str(t))
    # ###

        # for i,t in enumerate(trials_long):
        #     ax3.plot(behav_ds[behav_ds[:,6]==t,1],dF_ds[behav_ds[:,6]==t,roi],alpha=0.4,c='#ff1100')
        #     ax4.plot(range(bins_long),bin_mean_long, c='#000000')
        #     ax4.set_xticks([],[])
        #     ax3.set_xlim(160,260)
        #     ax4.set_xlim(160//binwidth,260//binwidth)
        #     # ax3.axvline(210, lw=90, c='0.9',alpha=0.02)
        #     ax3.set_title(str(mouse)+' on '+str(sess)+': Long track',fontdict={'fontweight':'bold'})
        #     ax3.set_xlabel('Lcoation (cm)')
        #     ax3.set_ylabel('∆F/F')
        #
        # fig.suptitle('ROI '+str(roi+1))
        # plt.subplots_adjust(top=0.94,hspace=0.2)

        # __fname = 'landmark_dF_ROI_'+str(roi+1)

        # if __name__ == "__main__":
        #     if not os.path.isdir(content['figure_output_path'] + subfolder):
        #         os.mkdir(content['figure_output_path'] + subfolder)
        #     fname = content['figure_output_path'] + subfolder + os.sep + __fname + '.' + fformat
        #     print(fname)
        #     try:
        #         fig.savefig(fname, format=fformat)
        #     except:
        #         exc_type, exc_value, exc_traceback = sys.exc_info()
        #         traceback.print_exception(exc_type, exc_value, exc_traceback,
        #                               limit=2, file=sys.stdout)

# ----
import yaml
import h5py
import os

with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

# LF180119_1
MOUSE = 'LF180119_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

SESSION = 'Day2018316_1'
fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2018316_1_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2018316_2'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day2018316_2_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# # LF171212_2
# MOUSE = 'LF171212_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day2018218_1'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018218_1_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018218_2'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day2018218_2_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# # LF170613_1
# MOUSE = 'LF170613_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day201784'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'landmark_dF'+str(SESSION),subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day201784_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'landmark_dF'+str(SESSION),subfolder=MOUSE+'_'+SESSION)

# LF170222_1
# MOUSE = 'LF170222_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day20170615'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION),subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170615_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION),subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day201776'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# SESSION = 'Day201776_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)

# # LF170421_2
# MOUSE = 'LF170421_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day20170719'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170719_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170720'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170720_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, subfolder=MOUSE+'_'+SESSION)

# # LF170420_1
# MOUSE = 'LF170420_1'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day20170612'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170612_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170803' (GOTTA DRAW ROIS)
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170803_openloop' (GOTTA DRAW ROIS)
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170719'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170719_openloop'
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+'_'+SESSION)

# LF170110_2
# MOUSE = 'LF170110_2'
# h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#
# SESSION = 'Day20170408' (NEED TO ADD BEHAVIOR AND ALIGN)
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170408_openloop' (NEED TO ADD BEHAVIOR AND ALIGN)
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+SESSION)
#
# SESSION = 'Day20170331' (NEED TO ADD BEHAVIOR AND ALIGN)
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170331_openloop' (NEED TO ADD BEHAVIOR AND ALIGN)
# fig_landmark_diff(h5path, MOUSE, SESSION, 'speedvloc_'+str(SESSION), subfolder=MOUSE+'_'+SESSION)


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
