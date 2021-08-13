0"""
Plot trace of an individual ROI vs location

@author: lukasfischer

"""


def fig_dfloc_trace(h5path, sess, roi, fname, fformat='png', subfolder=[], c_ylim=[]):
    import numpy as np
    import h5py
    import os
    import sys
    import traceback
    import yaml
    from yaml_mouselist import yaml_mouselist
    import warnings; warnings.simplefilter('ignore')

    import matplotlib
    from matplotlib import pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats

    import seaborn as sns
    sns.set_style("white")

    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    binnr_short = 80
    binnr_long = 100
    binnr_dark = 100
    tracklength_short = 400
    tracklength_long = 500
    maxdistance_dark = 500
    track_short = 3
    track_long = 4
    track_dark = 5

    # create figure to later plot on
    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot2grid((6,4),(0,0),colspan=2,rowspan=2)
    ax2 = plt.subplot2grid((6,4),(0,2),colspan=2,rowspan=2)
    ax3 = plt.subplot2grid((6,4),(2,0),colspan=2,rowspan=2)
    ax4 = plt.subplot2grid((6,4),(2,2),colspan=2,rowspan=2)
    ax5 = plt.subplot2grid((6,4),(5,0),colspan=1,rowspan=1)
    ax6 = plt.subplot2grid((6,4),(5,1),colspan=1,rowspan=1)
    ax7 = plt.subplot2grid((6,4),(5,2),colspan=1,rowspan=1)
    ax8 = plt.subplot2grid((6,4),(5,3),colspan=1,rowspan=1)
    ax9 = plt.subplot2grid((6,4),(4,0),colspan=1,rowspan=1)
    ax10 = plt.subplot2grid((6,4),(4,1),colspan=1,rowspan=1)
    ax11 = plt.subplot2grid((6,4),(4,2),colspan=1,rowspan=1)
    ax12 = plt.subplot2grid((6,4),(4,3),colspan=1,rowspan=1)

    # plot landmark and reward zone as shaded areas
    ax1.axvspan(40,48,color='0.9',zorder=0)
    ax1.axvspan(64,68,color=sns.xkcd_rgb["windows blue"],alpha=0.2,zorder=0)
    ax2.axvspan(40,48,color='0.9',zorder=0)
    ax2.axvspan(76,80,color=sns.xkcd_rgb["dusty purple"],alpha=0.2,zorder=0)

    ax9.axvspan(40,48,color='0.9',zorder=0)
    ax9.axvspan(64,68,color=sns.xkcd_rgb["windows blue"],alpha=0.2,zorder=0)
    ax10.axvspan(40,48,color='0.9',zorder=0)
    ax10.axvspan(64,68,color=sns.xkcd_rgb["windows blue"],alpha=0.2,zorder=0)

    ax11.axvspan(40,48,color='0.9',zorder=0)
    ax11.axvspan(76,80,color=sns.xkcd_rgb["dusty purple"],alpha=0.2,zorder=0)
    ax12.axvspan(40,48,color='0.9',zorder=0)
    ax12.axvspan(76,80,color=sns.xkcd_rgb["dusty purple"],alpha=0.2,zorder=0)

    # set axis labels
    ax1.set_xticks([0,20,40,60,80])
    ax1.set_xticklabels(['0','100','200','300','400'])
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Roi response on SHORT trials')
    ax2.set_xticks([0,20,40,60,80,100])
    ax2.set_xticklabels(['0','100','200','300','400','500'])
    ax2.set_xlabel('Location (cm)')
    ax2.set_title('Roi response on LONG trials')
    ax1.set_xlim([10,68])
    ax2.set_xlim([10,80])

    ax9.set_title('short, successful')
    ax10.set_title('short, unsuccessful')
    ax11.set_title('long, successful')
    ax12.set_title('long, unsuccessful')

    # set axes visibility
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.spines['left'].set_visible(False)
    ax9.spines['bottom'].set_visible(False)
    ax9.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.spines['left'].set_visible(False)
    ax10.spines['bottom'].set_visible(False)
    ax10.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['left'].set_visible(False)
    ax11.spines['bottom'].set_visible(False)
    ax11.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax12.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['left'].set_visible(False)
    ax12.spines['bottom'].set_visible(False)
    ax12.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        length=4, \
        bottom='off', \
        right='off', \
        top='off')

    # plot lines indicating landmark and reward zone in heatmaps
    ax3.axvline(40,c='0.8',lw=2)
    ax3.axvline(48,c='0.8',lw=2)
    ax3.axvline(64,c='r',lw=2)

    ax4.axvline(40,c='0.8',lw=2)
    ax4.axvline(48,c='0.8',lw=2)
    ax4.axvline(76,c='r',lw=2)

    ax5.axvline(40,c='0.8',lw=2)
    ax5.axvline(48,c='0.8',lw=2)
    ax5.axvline(64,c='r',lw=2)

    ax6.axvline(40,c='0.8',lw=2)
    ax6.axvline(48,c='0.8',lw=2)
    ax6.axvline(64,c='r',lw=2)

    ax7.axvline(40,c='0.8',lw=2)
    ax7.axvline(48,c='0.8',lw=2)
    ax7.axvline(76,c='r',lw=2)

    ax8.axvline(40,c='0.8',lw=2)
    ax8.axvline(48,c='0.8',lw=2)
    ax8.axvline(76,c='r',lw=2)

    # pull out trial numbers of respective sections
    trials_short = filter_trials( behav_ds, [], ['tracknumber',track_short])
    trials_long = filter_trials( behav_ds, [], ['tracknumber',track_long])
    mean_dF_short = np.zeros((np.size(trials_short,0),binnr_short))
    mean_dF_long = np.zeros((np.size(trials_long,0),binnr_long))

    # run through SHORT trials and calculate avg dF/F for each bin and trial
    for i,t in enumerate(trials_short):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                               (0.0, tracklength_short))[0]
        mean_dF_short[i,:] = mean_dF_trial
        ax1.plot(mean_dF_trial,c='0.8')

    sem_dF_s = stats.sem(mean_dF_short,0,nan_policy='omit')
    avg_mean_dF_short = np.nanmean(mean_dF_short,axis=0)
    ax1.plot(avg_mean_dF_short,c=sns.xkcd_rgb["windows blue"],lw=3) #'#00AAAA'
    # ax1.fill_between(np.arange(len(avg_mean_dF_short)), np.nanmean(mean_dF_short,axis=0) - sem_dF_s, np.nanmean(mean_dF_short,axis=0) + sem_dF_s, color = sns.xkcd_rgb["windows blue"], alpha = 0.2)

    # run through LONG trials and calculate avg dF/F for each bin and trial
    for i,t in enumerate(trials_long):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                               (0.0, tracklength_long))[0]
        mean_dF_long[i,:] = mean_dF_trial
        ax2.plot(mean_dF_trial,c='0.8')

    sem_dF_l = stats.sem(mean_dF_long,0,nan_policy='omit')
    avg_mean_dF_long = np.nanmean(mean_dF_long,axis=0)
    ax2.plot(avg_mean_dF_long,c=sns.xkcd_rgb["dusty purple"],lw=3) #'#FF00FF'
    # ax2.fill_between(np.arange(len(avg_mean_dF_long)), avg_mean_dF_long - sem_dF_l, avg_mean_dF_long + sem_dF_l, color = sns.xkcd_rgb["dusty purple"], alpha = 0.2)

    # run through SHORT SUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_short_succ = filter_trials( behav_ds, [], ['trial_successful'], trials_short)
    mean_dF_short_succ = np.zeros((np.size(trials_short_succ,0),binnr_short))
    for i,t in enumerate(trials_short_succ):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                               (0.0, tracklength_short))[0]
        mean_dF_short_succ[i,:] = mean_dF_trial
        ax9.plot(mean_dF_trial,c='0.8')
    ax9.plot(np.nanmean(mean_dF_short_succ,axis=0),c=sns.xkcd_rgb["windows blue"],ls='-',lw=2) #'#00AAAA'

    # run through SHORT UNSUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_short_unsucc = filter_trials( behav_ds, [], ['trial_unsuccessful'], trials_short)
    mean_dF_short_unsucc = np.zeros((np.size(trials_short_unsucc,0),binnr_short))
    for i,t in enumerate(trials_short_unsucc):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                               (0.0, tracklength_short))[0]
        mean_dF_short_unsucc[i,:] = mean_dF_trial
        ax10.plot(mean_dF_trial,c='0.8')
    ax10.plot(np.nanmean(mean_dF_short_unsucc,axis=0),c=sns.xkcd_rgb["windows blue"],ls='--',lw=2)

        # run through LONG SUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_long_succ = filter_trials( behav_ds, [], ['trial_successful'], trials_long)
    mean_dF_long_succ = np.zeros((np.size(trials_long_succ,0),binnr_long))
    for i,t in enumerate(trials_long_succ):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                               (0.0, tracklength_long))[0]
        mean_dF_long_succ[i,:] = mean_dF_trial
        ax11.plot(mean_dF_trial,c='0.8')
    ax11.plot(np.nanmean(mean_dF_long_succ,axis=0),c=sns.xkcd_rgb["dusty purple"],ls='-',lw=2) #'#00AAAA'


    # run through long UNSUCCESSFUL trials and calculate avg dF/F for each bin and trial
    trials_long_unsucc = filter_trials( behav_ds, [], ['trial_unsuccessful'], trials_long)
    mean_dF_long_unsucc = np.zeros((np.size(trials_long_unsucc,0),binnr_long))
    for i,t in enumerate(trials_long_unsucc):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                               (0.0, tracklength_long))[0]
        mean_dF_long_unsucc[i,:] = mean_dF_trial
        ax12.plot(mean_dF_trial,c='0.8')
    ax12.plot(np.nanmean(mean_dF_long_unsucc,axis=0),c=sns.xkcd_rgb["dusty purple"],ls='--',lw=2)

    # determine scaling of y-axis max values
    max_y_short = np.nanmax(np.nanmax(mean_dF_short))
    max_y_long = np.nanmax(np.nanmax(mean_dF_long))
    max_y = np.amax([max_y_short, max_y_long])
    heatmap_max = np.amax([np.nanmax(np.nanmean(mean_dF_short,axis=0)),np.nanmax(np.nanmean(mean_dF_long,axis=0))]) #+ 1

    if c_ylim != []:
        hmmin = c_ylim[0]
        hmmax = c_ylim[1]
    else:
        hmmin = 0
        hmmax = heatmap_max


    # plot heatmaps
    sns.heatmap(mean_dF_short,cbar=True,vmin=hmmin,vmax=hmmax,cmap='viridis',yticklabels=trials_short.astype('int'),xticklabels=True,ax=ax3)
    sns.heatmap(mean_dF_long,cbar=True,vmin=hmmin,vmax=hmmax,cmap='viridis',yticklabels=False,xticklabels=False,ax=ax4)

    sns.heatmap(mean_dF_short_succ,cbar=True,vmin=hmmin,vmax=hmmax,cmap='viridis',yticklabels=trials_short_succ.astype('int'),xticklabels=True,ax=ax5)
    sns.heatmap(mean_dF_short_unsucc,cbar=True,vmin=hmmin,vmax=hmmax,cmap='viridis',yticklabels=trials_short_unsucc.astype('int'),xticklabels=True,ax=ax6)
    sns.heatmap(mean_dF_long_succ,cbar=True,vmin=hmmin,vmax=hmmax,cmap='viridis',yticklabels=trials_long_succ.astype('int'),xticklabels=True,ax=ax7)
    sns.heatmap(mean_dF_long_unsucc,cbar=True,vmin=hmmin,vmax=hmmax,cmap='viridis',yticklabels=trials_long_unsucc.astype('int'),xticklabels=True,ax=ax8)

    if c_ylim == []:
        ax1.set_ylim([-0.5,max_y])
        ax2.set_ylim([-0.5,max_y])
        ax9.set_ylim([-0.5,max_y])
        ax10.set_ylim([-0.5,max_y])
        ax11.set_ylim([-0.5,max_y])
        ax12.set_ylim([-0.5,max_y])
        c_ylim = [-0.5,max_y]
    else:
        ax1.set_ylim(c_ylim)
        ax2.set_ylim(c_ylim)
        ax9.set_ylim(c_ylim)
        ax10.set_ylim(c_ylim)
        ax11.set_ylim(c_ylim)
        ax12.set_ylim(c_ylim)

    ax3.set_xticks([0,20,40,60,80])
    ax3.set_xticklabels(['0','100','200','300','400'])
    ax3.set_xlim([10,68])
    ax4.set_xticks([0,20,40,60,80,100])
    ax4.set_xticklabels(['0','100','200','300','400','500'])
    ax4.set_xlim([10,80])

    ax5.set_xticks([0,20,40,60,80])
    ax5.set_xticklabels(['0','100','200','300','400'])
    ax5.set_xlim([10,68])

    ax6.set_xticks([0,20,40,60,80])
    ax6.set_xticklabels(['0','100','200','300','400'])
    ax6.set_xlim([10,68])

    ax7.set_xticks([0,20,40,60,80,100])
    ax7.set_xticklabels(['0','100','200','300','400','500'])
    ax7.set_xlim([10,80])

    ax8.set_xticks([0,20,40,60,80,100])
    ax8.set_xticklabels(['0','100','200','300','400','500'])
    ax8.set_xlim([10,80])

    ax9.set_xticks([0,20,40,60,80])
    ax9.set_xticklabels(['0','100','200','300','400'])
    ax9.set_xlim([10,68])

    ax10.set_xticks([0,20,40,60,80])
    ax10.set_xticklabels(['0','100','200','300','400'])
    ax10.set_xlim([10,68])

    ax11.set_xticks([0,20,40,60,80,100])
    ax11.set_xticklabels(['0','100','200','300','400','500'])
    ax11.set_xlim([10,80])

    ax12.set_xticks([0,20,40,60,80,100])
    ax12.set_xticklabels(['0','100','200','300','400','500'])
    ax12.set_xlim([10,80])

    plt.tight_layout()

    fig.suptitle(fname, wrap=True)
    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    return c_ylim

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'png'
