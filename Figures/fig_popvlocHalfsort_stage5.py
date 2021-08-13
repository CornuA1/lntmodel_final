"""
Plot population activity aligned to location. Select half the trials randomly
to calculate mean response and the other half to calculate the order by
which to plot the sequence

@author: lukasfischer


"""


def fig_popvlocHalfsort_stage5(behav_collection, dF_collection, rec_info, fname=[], trials='both', sortby='both', fformat='png', subfolder=[]):
    # load local settings file
    import matplotlib
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("../Analysis")
    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    import yaml
    import seaborn as sns
    sns.set_style('white')
    import os
    with open('../loc_settings.yaml', 'r') as f:
                content = yaml.load(f)

    sys.path.append(content['base_dir'] + 'Analysis')

    # basic analysis and track parameters
    tracklength_short = 360
    tracklength_long = 420
    track_start = 100
    track_short = 3
    track_long = 4
    # bin from which to start analysing and plotting dF data
    start_bin = 20
    end_bin_short = 68
    end_bin_long = 80

    # size of bin in cm
    bin_size = 5
    binnr_short = tracklength_short/bin_size
    binnr_long = tracklength_long/bin_size

    mean_dF_short_coll_sig = []
    mean_dF_short_coll_sort = []

    mean_dF_long_coll_sig = []
    mean_dF_long_coll_sort = []

    for i in range(len(behav_collection)):
        behav_ds = behav_collection[i]
        dF_ds = dF_collection[i]
        # pull out trial numbers of respective sections
        trials_short_sig = filter_trials(behav_ds, [], ['tracknumber',track_short])
        trials_long_sig = filter_trials(behav_ds, [], ['tracknumber',track_long])

        # further filter if only correct or incorrect trials are plotted
        if trials == 'c':
            trials_short_sig = filter_trials(behav_ds, [], ['trial_successful'],trials_short_sig)
            trials_long = filter_trials(behav_ds, [], ['trial_successful'],trials_long)
        if trials == 'ic':
            trials_short_sig = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short_sig)
            trials_long = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long)

        # randomly draw 50% of indices to calculate signal and the other half to calculate the order by which to sort
        trials_short_rand = np.random.choice(trials_short_sig, np.size(trials_short_sig), replace=False)
        trials_short_sort = trials_short_rand[np.arange(0,np.floor(np.size(trials_short_sig)/2)).astype(int)]
        trials_short_sig = trials_short_rand[np.arange(np.ceil(np.size(trials_short_sig)/2),np.size(trials_short_sig)).astype(int)]

        trials_long_rand = np.random.choice(trials_long_sig, np.size(trials_long_sig), replace=False)
        trials_long_sort = trials_long_rand[np.arange(0,np.floor(np.size(trials_long_sig)/2)).astype(int)]
        trials_long_sig = trials_long_rand[np.arange(np.ceil(np.size(trials_long_sig)/2),np.size(trials_long_sig)).astype(int)]

        # storage for mean ROI data
        mean_dF_short_sig = np.zeros((binnr_short,np.size(dF_ds,1)))
        mean_dF_short_sort = np.zeros((binnr_short,np.size(dF_ds,1)))

        mean_dF_long_sig = np.zeros((binnr_long,np.size(dF_ds,1)))
        mean_dF_long_sort = np.zeros((binnr_long,np.size(dF_ds,1)))

        # loop through all ROIs
        for j in range(np.size(dF_ds,1)):
            mean_dF_trials_short_sig = np.zeros((np.size(trials_short_sig,0),binnr_short))
            mean_dF_trials_short_sort = np.zeros((np.size(trials_short_sort,0),binnr_short))

            # calculate mean dF vs location for each ROI on short trials
            for k,t in enumerate(trials_short_sig):
                # pull out current trial and corresponding dF data and bin it
                cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,j]
                mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                                       (0.0, tracklength_short))[0]
                #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
                mean_dF_trials_short_sig[k,:] = mean_dF_trial

            mean_dF_short_sig[:,j] = np.nanmean(mean_dF_trials_short_sig,0)
            mean_dF_short_sig[:,j] /= np.nanmax(np.abs(mean_dF_short_sig[start_bin:end_bin_short,j]))

            # calculate mean dF vs location for each ROI on short trials
            for k,t in enumerate(trials_short_sort):
                # pull out current trial and corresponding dF data and bin it
                cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,j]
                mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                                       (0.0, tracklength_short))[0]
                #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
                mean_dF_trials_short_sort[k,:] = mean_dF_trial

            mean_dF_short_sort[:,j] = np.nanmean(mean_dF_trials_short_sort,0)
            mean_dF_short_sort[:,j] /= np.nanmax(np.abs(mean_dF_short_sort[start_bin:end_bin_long,j]))

            # calculate mean dF vs location for each ROI on long trials
            mean_dF_trials_long_sig = np.zeros((np.size(trials_long_sig,0),binnr_long))
            mean_dF_trials_long_sort = np.zeros((np.size(trials_long_sort,0),binnr_long))

            # calculate mean dF vs location for each ROI on long trials
            for k,t in enumerate(trials_long_sig):
                # pull out current trial and corresponding dF data and bin it
                cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,j]
                mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                                       (0.0, tracklength_long))[0]
                #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_long]))
                mean_dF_trials_long_sig[k,:] = mean_dF_trial

            mean_dF_long_sig[:,j] = np.nanmean(mean_dF_trials_long_sig,0)
            mean_dF_long_sig[:,j] /= np.nanmax(np.abs(mean_dF_long_sig[start_bin:end_bin_long,j]))

            # calculate mean dF vs location for each ROI on long trials
            for k,t in enumerate(trials_long_sort):
                # pull out current trial and corresponding dF data and bin it
                cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,j]
                mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                                       (0.0, tracklength_long))[0]
                #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_long]))
                mean_dF_trials_long_sort[k,:] = mean_dF_trial

            mean_dF_long_sort[:,j] = np.nanmean(mean_dF_trials_long_sort,0)
            mean_dF_long_sort[:,j] /= np.nanmax(np.abs(mean_dF_long_sort[start_bin:end_bin_long,j]))

        if mean_dF_short_coll_sig == []:
            mean_dF_short_coll_sig = mean_dF_short_sig
        else:
            mean_dF_short_coll_sig = np.append(mean_dF_short_coll_sig, mean_dF_short_sig, axis=1)
        if mean_dF_short_coll_sort == []:
            mean_dF_short_coll_sort = mean_dF_short_sort
        else:
            mean_dF_short_coll_sort = np.append(mean_dF_short_coll_sort, mean_dF_short_sort, axis=1)

        if mean_dF_long_coll_sig == []:
            mean_dF_long_coll_sig = mean_dF_long_sig
        else:
            mean_dF_long_coll_sig = np.append(mean_dF_long_coll_sig, mean_dF_long_sig, axis=1)
        if mean_dF_long_coll_sort == []:
            mean_dF_long_coll_sort = mean_dF_long_sort
        else:
            mean_dF_long_coll_sort = np.append(mean_dF_long_coll_sort, mean_dF_long_sort, axis=1)


    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((5,2),(1,0),rowspan=3)
    ax2 = plt.subplot2grid((5,2),(1,1),rowspan=3)
    ax3 = plt.subplot2grid((5,2),(0,0),rowspan=1)
    ax4 = plt.subplot2grid((5,2),(0,1),rowspan=1)
    # ax5 = plt.subplot2grid((5,2),(4,0),rowspan=1)
    # ax6 = plt.subplot2grid((5,2),(4,1),rowspan=1)

    # sort by peak activity (naming of variables confusing because the script grew organically...)
    mean_dF_sort_short = np.zeros(mean_dF_short_coll_sort.shape[1])
    for i, row in enumerate(np.transpose(mean_dF_short_coll_sort)):
        if not np.all(np.isnan(row)):
            mean_dF_sort_short[i] = np.nanargmax(row[start_bin:end_bin_short])
    sort_ind_short = np.argsort(mean_dF_sort_short)

    print(mean_dF_sort_short)

    sns.distplot(mean_dF_sort_short,color=sns.xkcd_rgb["windows blue"],bins=16,hist=True, kde=False,ax=ax3)
    #start_bin=10
    #ax3.set_ylim([0,30])
    #ax3.set_xlim([start_bin,end_bin_short])
    #ax3.set_ylim([0,40])
#    sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short, sort_ind_short]), cmap='jet', vmin=0.0, vmax=1.0, ax=ax1, cbar=False)

    mean_dF_sort_long = np.zeros(mean_dF_long_coll_sort.shape[1])
    for i, row in enumerate(np.transpose(mean_dF_long_coll_sort)):
        if not np.all(np.isnan(row)):
            mean_dF_sort_long[i] = np.nanargmax(row[start_bin:end_bin_long])
    sort_ind_long = np.argsort(mean_dF_sort_long)
    sns.distplot(mean_dF_sort_long,color=sns.xkcd_rgb["dusty purple"],bins=16,hist=True, kde=False,ax=ax4)
    #ax4.set_ylim([0,30])
    #ax4.set_xlim([start_bin,end_bin_long])
    #ax4.set_ylim([0,40])
#    sns.heatmap(np.transpose(mean_dF_long_coll[start_bin:end_bin_long, sort_ind_long]), cmap='jet', vmin=0.0, vmax=1.0, ax=ax2, cbar=False)

    if sortby == 'none':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[start_bin:end_bin_short,:]), cmap='jet', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[start_bin:end_bin_long,:]), cmap='jet', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'short':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[start_bin:end_bin_short,sort_ind_short]), cmap='jet', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[start_bin:end_bin_long,sort_ind_short]), cmap='jet', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'long':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[start_bin:end_bin_short,sort_ind_long]), cmap='jet', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[start_bin:end_bin_long,sort_ind_long]), cmap='jet', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'both':
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[start_bin:end_bin_short,sort_ind_short]), cmap='jet', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[start_bin:end_bin_long,sort_ind_long]), cmap='jet', vmin=0.0, vmax=1, ax=ax2, cbar=False)

    ax1.axvline((200/bin_size)-start_bin, lw=3, c='0.8')
    ax1.axvline((240/bin_size)-start_bin, lw=3, c='0.8')
    ax1.axvline((320/bin_size)-start_bin, lw=3, c='0.8')
    ax2.axvline((200/bin_size)-start_bin, lw=3, c='0.8')
    ax2.axvline((240/bin_size)-start_bin, lw=3, c='0.8')
    ax2.axvline((380/bin_size)-start_bin, lw=3, c='0.8')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])

    #fig.suptitle(fname + '_' + trials + '_' + str([''.join(str(r) for r in ri) for ri in rec_info]),wrap=True)

#    plt.tight_layout()


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


if __name__ == "__main__":
    print('ERROR: stand-alone version of script not implemented. Please use make-script.')
    #from optparse import OptionParser
    # parse command line arguments
    #parser = OptionParser()
    #parser.add_option("-f", "--file", default="pop_v_loc", dest="filename",
    #                  help="filename for figure", metavar="FILE")
    #parser.add_option("-m", "--mode", default="both", metavar="MODE", dest="mode",
    #                  help="c: correct trials, ic: incorrect trails, both. [default: %default]")
    #(options, args) = parser.parse_args()

    #fig_popvloc_stage5(options.filename, options.mode)
