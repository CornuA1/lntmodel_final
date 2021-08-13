"""
Plot population activity aligned to location

@author: lukasfischer


"""


def fig_popvloc_stage5(behav_collection, dF_collection, rec_info, fname=[], trials='both', sortby='both', fformat='png'):
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
    binnr_short = 80
    binnr_long = 100
    tracklength_short = 400
    tracklength_long = 500
    track_short = 3
    track_long = 4
    # bin from which to start analysing and plotting dF data
    start_bin = 20
    end_bin_short = 65
    end_bin_long = 78

    mean_dF_short_coll = []
    mean_dF_long_coll = []

    for i in range(len(behav_collection)):
        behav_ds = behav_collection[i]
        dF_ds = dF_collection[i]
        # pull out trial numbers of respective sections
        trials_short = filter_trials(behav_ds, [], ['tracknumber',track_short])
        trials_long = filter_trials(behav_ds, [], ['tracknumber',track_long])

        if trials == 'c':
            trials_short = filter_trials(behav_ds, [], ['trial_successful'],trials_short)
            trials_long = filter_trials(behav_ds, [], ['trial_successful'],trials_long)
        if trials == 'ic':
            trials_short = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short)
            trials_long = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long)

        # storage for mean ROI data
        mean_dF_short = np.zeros((binnr_short,np.size(dF_ds,1)))
        mean_dF_long = np.zeros((binnr_long,np.size(dF_ds,1)))

        # loop through all ROIs
        for j in range(np.size(dF_ds,1)):
            mean_dF_trials_short = np.zeros((np.size(trials_short,0),binnr_short))
            mean_dF_trials_long = np.zeros((np.size(trials_long,0),binnr_long))
            # calculate mean dF vs location for each ROI on short trials
            for k,t in enumerate(trials_short):
                # pull out current trial and corresponding dF data and bin it
                cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,j]
                mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_short,
                                                       (0.0, tracklength_short))[0]
                #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
                mean_dF_trials_short[k,:] = mean_dF_trial
            mean_dF_short[:,j] = np.nanmean(mean_dF_trials_short,0)
            mean_dF_short[:,j] /= np.nanmax(np.abs(mean_dF_short[start_bin:end_bin_short,j]))
            # calculate mean dF vs location for each ROI on long trials
            for k,t in enumerate(trials_long):
                # pull out current trial and corresponding dF data and bin it
                cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
                cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,j]
                mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', binnr_long,
                                                       (0.0, tracklength_long))[0]
                #mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_long]))
                mean_dF_trials_long[k,:] = mean_dF_trial
                mean_dF_long[:,j] = np.nanmean(mean_dF_trials_long,0)
            mean_dF_long[:,j] = np.nanmean(mean_dF_trials_long,0)
            mean_dF_long[:,j] /= np.nanmax(mean_dF_long[:,j])

        if mean_dF_short_coll == []:
            mean_dF_short_coll = mean_dF_short
        else:
            mean_dF_short_coll = np.append(mean_dF_short_coll, mean_dF_short, axis=1)

        if mean_dF_long_coll == []:
            mean_dF_long_coll = mean_dF_long
        else:
            mean_dF_long_coll = np.append(mean_dF_long_coll, mean_dF_long, axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((4,2),(1,0),rowspan=3)
    ax2 = plt.subplot2grid((4,2),(1,1),rowspan=3)
    ax3 = plt.subplot2grid((4,2),(0,0),rowspan=1)
    ax4 = plt.subplot2grid((4,2),(0,1),rowspan=1)

    # sort by peak activity
    mean_dF_sort_short = np.zeros(mean_dF_short_coll.shape[1])
    for i, row in enumerate(np.transpose(mean_dF_short_coll)):
        if not np.all(np.isnan(row)):
            mean_dF_sort_short[i] = np.nanargmax(row[start_bin:end_bin_short])
    sort_ind_short = np.argsort(mean_dF_sort_short)

    sns.distplot(mean_dF_sort_short,color=sns.xkcd_rgb["windows blue"],bins=16,kde=False,ax=ax3)
    #ax3.set_ylim([0,30])
#    sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short, sort_ind_short]), cmap='jet', vmin=0.0, vmax=1.0, ax=ax1, cbar=False)

    mean_dF_sort_long = np.zeros(mean_dF_long_coll.shape[1])
    for i, row in enumerate(np.transpose(mean_dF_long_coll)):
        if not np.all(np.isnan(row)):
            mean_dF_sort_long[i] = np.nanargmax(row[start_bin:end_bin_long])
    sort_ind_long = np.argsort(mean_dF_sort_long)
    sns.distplot(mean_dF_sort_long,color=sns.xkcd_rgb["dusty purple"],bins=16,kde=False,ax=ax4)
    #ax4.set_ylim([0,30])
#    sns.heatmap(np.transpose(mean_dF_long_coll[start_bin:end_bin_long, sort_ind_long]), cmap='jet', vmin=0.0, vmax=1.0, ax=ax2, cbar=False)
    #start_bin=0
    if sortby == 'none':
        sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short,:]), cmap='jet', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll[start_bin:end_bin_long,:]), cmap='jet', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'short':
        sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short,sort_ind_short]), cmap='jet', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll[start_bin:end_bin_long,sort_ind_short]), cmap='jet', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'long':
        sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short,sort_ind_long]), cmap='jet', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll[start_bin:end_bin_long,sort_ind_long]), cmap='jet', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    elif sortby == 'both':
        sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short,sort_ind_short]), cmap='jet', vmin=0.0, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll[start_bin:end_bin_long,sort_ind_long]), cmap='jet', vmin=0.0, ax=ax2, cbar=False)

    ax1.axvline(20, lw=3, c='0.8')
    ax1.axvline(28, lw=3, c='0.8')
    ax1.axvline(44, lw=3, c='0.8')
    ax2.axvline(20, lw=3, c='0.8')
    ax2.axvline(28, lw=3, c='0.8')
    ax2.axvline(56, lw=3, c='0.8')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])

    #fig.suptitle(fname + '_' + trials + '_' + str([''.join(str(r) for r in ri) for ri in rec_info]),wrap=True)

#    plt.tight_layout()


    fname = content['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, dpi=400, format=fformat)


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
