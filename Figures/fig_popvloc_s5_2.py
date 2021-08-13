"""
Plot population activity aligned to location. Select half the trials randomly
to calculate mean response and the other half to calculate the order by
which to plot the sequence

@author: lukasfischer

NOTE: this is an updated version of fig_popvlocHalfsort_stage5. This function can load multiple datasets and concatenate them (as opposed to using behavection and dFection)


"""

# load local settings file
import matplotlib
import numpy as np
import warnings; warnings.simplefilter('ignore')
import sys
sys.path.append("./Analysis")
import ipdb

from analysis_parameters import MIN_FRACTION_ACTIVE, MIN_ZSCORE, MIN_MEAN_AMP
from rewards import rewards
from licks import licks

# from analysis_parameters import MIN_MEAN_AMP_BOUTONS as MIN_MEAN_AMP
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib_venn import venn2_circles
from filter_trials import filter_trials
from scipy import stats, signal, io
import scipy.io as sio
import statsmodels.api as sm
import yaml
import h5py
import json
import seaborn as sns
from write_dict import write_dict
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
            loc_info = yaml.load(f)



fformat = 'png'

# number of times calculations carried out
NUM_ITER = 2

# export data to matlab matrices
EXPORT_MATLAB = False

# load raw or hdf5 dataset
LOAD_RAW = False

# run trial-by-trial analysis
TRIAL_BY_TRIAL = False

# bin size (cm)
BIN_SIZE = 5

SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'

# track numbers used in this analysis
TRACK_SHORT = 3
TRACKLENGTH_SHORT = 320
TRACK_LONG = 4
TRACKLENGTH_LONG = 380

# defines the distance cropped at the beginning (landmark aligned) or end (trialonset aligned) to account for the randomized starting position
TRACK_CROP = 100

def export_data_matlab(mouse, session, behaviora_data, dF_data, roi_numbers, trialtype, fname_suffix):
    """ export data of neurons that are included in analysis to matlab format along with behavioral data """
    print(mouse, session)
    fname = loc_info['figure_output_path'] + 'popvloc' + os.sep + 'matlab_'+mouse+'_'+session+'_'+fname_suffix+ '.mat'
    print(fname)
    savemat_data = {}
    savemat_data['behavior'] = behaviora_data[:,[0,1,3,4]]
    savemat_data['df'] = dF_data[:,roi_numbers]
    io.savemat(fname,savemat_data)


def get_eventaligned_rois(roi_params, trialtypes, align_event):
    """ return roi number, peak value and peak time of all neurons that have their max response at <align_even> in VR """
    # hold values of mean peak
    event_list = ['trialonset','lmcenter','reward']
    result_max_peak = {}
    # set up empty dicts so we can later append to them
    for tl in trialtypes:
        result_max_peak[align_event + '_roi_number_' + tl] = []

    # grab a full list of roi numbers
    roi_list_all = roi_params['valid_rois']
    # loop through every roi
    for j,r in enumerate(roi_list_all):
        # loop through every trialtype and alignment point to determine largest response
        for tl in trialtypes:
            max_peak = -99
            roi_num = -1
            valid = False
            peak_event = ''
            peak_trialtype = ''
            for el in event_list:
                value_key = el + '_peak_' + tl
                value_key_ol = el + '_peak_' + tl + '_ol'
                value_key_peaktime = el + '_peak_time_' + tl
                value_key_peaktime_ol = el + '_peak_time_' + tl + '_ol'
                # check roi max peak for each alignment point and store wich ones has the highest value
                if roi_params[value_key][j] > max_peak:
                    valid = True
                    max_peak = roi_params[value_key][j]
                    peak_event = el
                    peak_trialtype = tl
                    roi_num = r
            # write results for alignment point with highest value to results dict
            if valid:
                if peak_event == align_event:
                    result_max_peak[align_event + '_roi_number_' + peak_trialtype].append(roi_num)

    return result_max_peak

def calc_spacevloc(behav_ds, roi_dF, included_trials, trialtype, align_point, binnr, maxlength):
    if align_point == 'trialonset':
        start_loc = 5
    else:
        start_loc = TRACK_CROP

    bin_edges = np.linspace(start_loc, maxlength+start_loc, binnr+1)

    # intilize matrix to hold data for all trials of this roi
    mean_dF_trials = np.zeros((np.size(included_trials,0),binnr))
    # calculate mean dF vs location for each ROI on short trials
    for k,t in enumerate(included_trials):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        if align_point == 'trialonset':
            cur_trial_loc = cur_trial_loc - cur_trial_loc[0]
        # print(start_loc)
        cur_trial_rew_loc = behav_ds[behav_ds[:,6]==t+1,1]
        # if the location was reset to 0 as the animal got a reward (which is the case when it gets a default reward): add the last location on the track to blackbox location)
        if np.size(cur_trial_rew_loc) > 0:
            if cur_trial_rew_loc[0] < 100:
                cur_trial_rew_loc[0] == cur_trial_loc[-1]

            cur_trial_loc = np.append(cur_trial_loc, cur_trial_rew_loc)
        cur_trial_dF_roi = roi_dF[behav_ds[:,6]==t]
        if np.size(cur_trial_rew_loc) > 0:
            cur_trial_dF_roi = np.append(cur_trial_dF_roi, roi_dF[behav_ds[:,6]==t+1])
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges, (start_loc, maxlength+start_loc))[0]
        # mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
        mean_dF_trials[k,:] = mean_dF_trial
        #print(mean_dF_trial)

    #mean_dF_short_sig[:,j] = np.nanmean(mean_dF_trials,0)
    mean_dF = np.nanmean(mean_dF_trials,0)
    mean_dF[mean_dF<0] = 0
    if np.all(np.isnan(mean_dF/np.nanmax(mean_dF))):
        mean_dF_normalized = np.zeros((mean_dF.shape[0],))
    else:
        mean_dF_normalized = mean_dF/np.nanmax(mean_dF)
    return mean_dF_normalized
    # mean_dF_short_sig[:,j] = mean_dF
    # mean_dF_short_sig[:,j] /= np.nanmax(np.abs(mean_dF_short_sig[start_bin:end_bin_short,j]))

def plot_first_licks(licks, rewards, trials, trial_type):
    # plot location of first trials on short and long trials

    first_lick = np.empty((0,4))
    first_lick_trials = np.empty((0))
    rewards[:,3] = rewards[:,3] - 1
    default_rewards = np.empty((0,4))
    default_rewards_trials = np.empty((0))
    for r in trials:
        if licks.size > 0:
            licks_all = licks[licks[:,2]==r,:]
            licks_all = licks_all[licks_all[:,1]>150,:]
            if licks_all.size == 0:
                rew_lick = rewards[rewards[:,3]==r,:]
                if rew_lick.size > 0:
                    if rew_lick[0,5] == 1:
                        licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                        if trial_type == 'short':
                            licks_all[0,1] = licks_all[0,1] - 320
                        elif trial_type == 'long':
                            licks_all[0,1] = licks_all[0,1] - 380
                        first_lick = np.vstack((first_lick, licks_all[0,:].T))
                        first_lick_trials = np.append(first_lick_trials, r)
                    if rew_lick[0,5] == 2:
                        default_rewards = np.vstack((default_rewards, np.asarray([[rew_lick[0,4], 18, rew_lick[0,3], rew_lick[0,2]]])[0,:].T))
                        default_rewards_trials = np.append(default_rewards_trials, r)
            else:
                if licks_all[0,3] == 3:
                    licks_all = licks_all[licks_all[:,1]<338,:]
                    licks_all[0,1] = licks_all[0,1] - 320
                elif licks_all[0,3] == 4:
                    licks_all = licks_all[licks_all[:,1]<398,:]
                    licks_all[0,1] = licks_all[0,1] - 380
                first_lick = np.vstack((first_lick, licks_all[0,:].T))
                first_lick_trials = np.append(first_lick_trials, r)


    return first_lick, first_lick_trials, default_rewards, default_rewards_trials

def population_trial_activity(behav_ds, licks_ds, reward_ds, dF_ds, roi_params, binnr, maxlength, trial_numbers, trial_type, roi_list_all, mouse, session):
    fl_short, fl_trials_short, df_short, df_trials_short = plot_first_licks(licks_ds, reward_ds, trial_numbers, trial_type)

    bin_edges = np.linspace(TRACK_CROP, maxlength+TRACK_CROP, binnr+1)
    num_trials = len(trial_numbers)
    rows_cols = int(np.sqrt(len(trial_numbers))) + 1
    fig = plt.figure(figsize=(10,10))
    plot_axes = []
    all_ax = 1

    # roi_list_all = roi_params['valid_rois']
    roi_data = np.zeros((binnr,len(roi_list_all),len(trial_numbers)))
    trial_fl_data = np.zeros((fl_short.shape[1]+1,len(trial_numbers)))

    for j in range(rows_cols):
        for k in range(rows_cols):
            plot_axes.append(plt.subplot(rows_cols,rows_cols,all_ax))
            plot_axes[all_ax-1].set_xticklabels([])
            plot_axes[all_ax-1].set_yticklabels([])
            plot_axes[all_ax-1].tick_params( \
                axis='both', \
                direction='out', \
                labelsize=20, \
                length=2, \
                width=1, \
                left='off', \
                bottom='on', \
                right='off', \
                top='off')
            all_ax = all_ax + 1


    for k,trn in enumerate(trial_numbers):
        # print('trial: ' + str(k))
        # get trial specific data
        cur_trial_loc = behav_ds[behav_ds[:,6]==trial_numbers[k],1]
        cur_trial_rew_loc = behav_ds[behav_ds[:,6]==trial_numbers[k]+1,1]
        # if the location was reset to 0 as the animal got a reward (which is the case when it gets a default reward): add the last location on the track to blackbox location)
        if np.size(cur_trial_rew_loc) > 0:
            if cur_trial_rew_loc[0] < 100:
                cur_trial_rew_loc[0] == cur_trial_loc[-1]

            cur_trial_loc = np.append(cur_trial_loc, cur_trial_rew_loc)

        # get ROI data for given trial
        all_roi_dF = np.zeros((binnr,len(roi_list_all)))
        # ipdb.set_trace()
        for j, cur_roi in enumerate(roi_list_all):
            cur_trial_dF_roi = dF_ds[behav_ds[:,6]==trial_numbers[k],cur_roi]
            if np.size(cur_trial_rew_loc) > 0:
                cur_trial_dF_roi = np.append(cur_trial_dF_roi, dF_ds[behav_ds[:,6]==trial_numbers[k]+1,cur_roi])

            mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges, (TRACK_CROP, maxlength+TRACK_CROP))[0]

            # mean_dF_trial = mean_dF_trial / np.amax([np.nanmax(mean_dF_trial),0.3])
            # mean_dF_norm = calc_spacevloc(behav_ds, dF_ds[:,j], trial_numbers, '', 'lmcenter', binnr, maxlength)
            all_roi_dF[:,j] = mean_dF_trial


        # ipdb.set_trace()
        mean_dF_sort_short = np.zeros(all_roi_dF.shape[1])
        for i, row in enumerate(all_roi_dF.T):
            if not np.all(np.isnan(row)):
                mean_dF_sort_short[i] = np.nanargmax(row)
            else:
                print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
        sort_ind_short = np.argsort(mean_dF_sort_short)

        # try:
        # print(sort_ind_short)
        sns.heatmap(np.transpose(all_roi_dF[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=plot_axes[k], cbar=False)
        plot_axes[k].set_yticklabels([])
        plot_axes[k].set_xticklabels([])

        # plot_axes[k].set_title(str(np.where(trial_numbers == k)))
        # ipdb.set_trace()
        if len(np.where(fl_trials_short == trn)[0]) > 0:
            trial_fl_data[0:4,k] = fl_short[np.where(fl_trials_short == trn)[0], :]
            trial_fl_data[4,k] = 1
            plot_axes[k].set_title(str(trn) + ' ' + str(np.round(trial_fl_data[1,k],2)), color='g', fontsize=10)

        elif len(np.where(df_trials_short == trn)[0]) > 0:
            trial_fl_data[0:4,k] = df_short[np.where(df_trials_short == trn)[0], :]
            trial_fl_data[4,k] = 2
            plot_axes[k].set_title(str(trn), color='r', fontsize=10 )

        roi_data[:,:,k] = all_roi_dF


    fname = loc_info['figure_output_path'] + 'popvloc' + os.sep + mouse + '_' + session + '_trials_' + trial_type + '.' + fformat
    fig.savefig(fname, format=fformat)
    print(fname)
    # #
    # plt.show()
    # ipdb.set_trace()
    return roi_data, trial_fl_data



def calc_trial_cc(mean_dF_sig, roi_trial_data, trial_fl_data, binnr, trial_type, mouse, session):
    """ calculate cc maps for each trial and the mean session activity """
    # set up figure
    fl_error = np.zeros((3,np.where(trial_fl_data[4,:]==1)[0].shape[0]))
    rows_cols = int(np.sqrt(roi_trial_data.shape[2])) + 1
    fig = plt.figure(figsize=(10,10))
    plot_axes = []
    all_ax = 1
    for j in range(rows_cols):
        for k in range(rows_cols):
            plot_axes.append(plt.subplot(rows_cols,rows_cols,all_ax))
            plot_axes[all_ax-1].set_xticklabels([])
            plot_axes[all_ax-1].set_yticklabels([])
            plot_axes[all_ax-1].tick_params( \
                axis='both', \
                direction='out', \
                labelsize=20, \
                length=2, \
                width=1, \
                left='off', \
                bottom='on', \
                right='off', \
                top='off')
            all_ax = all_ax + 1


    popvec_cc_matrix_pearsonr_trials = np.zeros((binnr,binnr,roi_trial_data.shape[2]))
    popvec_cc_reconstruction = np.zeros((2,binnr,roi_trial_data.shape[2]))
    popvec_cc_mean_trial_error = np.zeros((roi_trial_data.shape[2],1))

    # run through every trial and calculate chec
    k=0
    for tr in range(roi_trial_data.shape[2]):
        for row in range(binnr):
            for col in range(binnr):
                # ipdb.set_trace()
                try:
                    popvec_cc_matrix_pearsonr_trials[row,col,tr] = stats.pearsonr(mean_dF_sig[row,:],roi_trial_data[col,:,tr])[0]
                except:
                    popvec_cc_matrix_pearsonr_trials[row,col,tr] = np.nan
                    # pass


        sns.heatmap(np.transpose(popvec_cc_matrix_pearsonr_trials[:,:,tr]), cmap='viridis', vmin=0.0, vmax=1, ax=plot_axes[tr], cbar=False)
        plot_axes[tr].set_yticklabels([])
        plot_axes[tr].set_xticklabels([])

        for i,row in enumerate(range(binnr)):
            popvec_cc_reconstruction[0,i,tr] = i
            try:
                popvec_cc_reconstruction[1,i,tr] = np.nanargmax(popvec_cc_matrix_pearsonr_trials[i,:,tr])
            except ValueError:
                popvec_cc_reconstruction[1,i,tr] = 0

        popvec_cc_mean_trial_error[tr] = np.mean(np.abs(popvec_cc_reconstruction[1,:,tr] - popvec_cc_reconstruction[0,:,tr]) * BIN_SIZE)
        # print(np.mean(np.abs(popvec_cc_reconstruction[1,:,tr] - popvec_cc_reconstruction[0,:,tr]) * BIN_SIZE))
        # plot_axes[tr].set_title(str(np.round(popvec_cc_mean_trial_error[tr],2)))
        if trial_fl_data[4,tr] == 1:
            plot_axes[tr].set_title(str(np.round(popvec_cc_mean_trial_error[tr],2)[0]) + ' ' + str(np.round(trial_fl_data[1,tr],2)), color='g', fontsize=10)
            fl_error[0:2,k] = np.array([popvec_cc_mean_trial_error[tr],trial_fl_data[1,tr]])
            fl_error[2,k] = np.nanmean(popvec_cc_matrix_pearsonr_trials[:,:,tr].diagonal())
            # print(popvec_cc_mean_trial_error[tr])
            k = k+1
        elif trial_fl_data[4,tr] == 2:
            plot_axes[tr].set_title(str(np.round(popvec_cc_mean_trial_error[tr],2)[0]), color='r', fontsize=10)


    fname = loc_info['figure_output_path'] + 'popvloc' + os.sep + mouse + '_' + session + '_trial_cc_' + trial_type + '.' + fformat
    fig.savefig(fname, format=fformat)
    fig = plt.figure(figsize=(6,3))
    ax1 = plt.subplot(121)
    ax1.scatter(fl_error[1,:], fl_error[0,:])
    error_slope, peak_intercept, lo_slope, up_slope = stats.theilslopes( fl_error[0,:], fl_error[1,:] )
    error_r, error_p = stats.pearsonr( fl_error[0,:], fl_error[1,:] )
    ax1.plot(fl_error[1,:], peak_intercept+error_slope * np.array(fl_error[1,:]), lw=2,c='r')
    ax1.set_xlabel('reward prediction error (cm)')
    ax1.set_ylabel('location reconstruction error (cm)')
    ax1.set_ylim([0,100])
    ax2 = plt.subplot(122)
    corr_slope, peak_intercept, lo_slope, up_slope = stats.theilslopes( fl_error[2,:], fl_error[1,:] )
    corr_r, corr_p = stats.pearsonr( fl_error[2,:], fl_error[1,:] )
    ax2.plot(fl_error[1,:], peak_intercept+corr_slope * np.array(fl_error[1,:]), lw=2,c='r')
    ax2.scatter(fl_error[1,:], fl_error[2,:])
    ax2.set_xlabel('reward prediction error (cm)')
    ax2.set_ylabel('cc at location')
    ax2.set_ylim([-1,1])
    plt.tight_layout()
    fname = loc_info['figure_output_path'] + 'popvloc' + os.sep + mouse + '_' + session + '_trial_cc_results_' + trial_type + '.' + fformat
    fig.savefig(fname, format=fformat)
    print(fname)
    # plt.show()
    # ipdb.set_trace()

    return popvec_cc_matrix_pearsonr_trials, error_slope, corr_slope, error_r, error_p, corr_r, corr_p

def popvloc_individual(r, trials, celltypes, split_data, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname, tot_rois_short, tot_rois_long, trial_by_trial = False, make_individual_figure=True, subfolder='', firstlick_split = None, custom_rois_short=None, custom_rois_long=None, roi_union=False):
    """ make popvloc for individual session """
    # load individual dataset
    mouse = r[1]
    session = r[2]
    custom_roilist = None
    validate_rois = True
    # ipdb.set_trace()
    if len(r) > 3 or roi_union == True:
        custom_roilist = r[3]

    # if len(r[0].split(os.sep)[-1].split('_')) > 3:
    #     session = r[0].split(os.sep)[-1].split('_')[-2] + '_' + r[0].split(os.sep)[-1].split('_')[-1]
    #     session = session.split('.')[-2]
    # else:
    #     session = r[0].split(os.sep)[-1].split('_')[-1].split('.')[-2]
    # print(mouse, session)

    if not LOAD_RAW:
        h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
        h5dat = h5py.File(h5path, 'r')
        behav_ds = np.copy(h5dat[session + '/behaviour_aligned'])
        dF_ds = np.copy(h5dat[session + '/dF_win'])
        licks_ds = np.copy(h5dat[session + '/licks_pre_reward'])
        reward_ds = np.copy(h5dat[session + '/rewards'])
        h5dat.close()
    else:
        processed_data_path = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data.mat'
        loaded_data = sio.loadmat(processed_data_path)
        behav_ds = loaded_data['behaviour_aligned']
        dF_ds = loaded_data['dF_aligned']
        # roiIDs = loaded_data['roiIDs']
        if custom_roilist is not None:
            dF_ds = dF_ds[:,np.in1d(np.arange(dF_ds.shape[1]),custom_roilist)]
        behav_licks = behav_ds[np.in1d(behav_ds[:, 4], [3, 4]), :]
        reward_ds = rewards(behav_licks)
        licks_ds,_ = licks(behav_licks, reward_ds)
        licks_ds = np.array(licks_ds)

    with open(r[0]) as f:
        roi_params = json.load(f)

    # pull out trial numbers of respective sections
    trials_short_sig = filter_trials(behav_ds, [], ['tracknumber',TRACK_SHORT])
    trials_long_sig = filter_trials(behav_ds, [], ['tracknumber',TRACK_LONG])

    if firstlick_split is not None:
        # we have to get trials based on first lick distance based on trial type which is being substituted in below into the respective filterprops
        firstlick_split.append(3)
        trials_short_sig = filter_trials(behav_ds, [], firstlick_split)
        firstlick_split[3] = 4
        trials_long_sig = filter_trials(behav_ds, [], firstlick_split)
        print('num short trials: ' + str(len(trials_short_sig)))
        print('num long trials: ' + str(len(trials_long_sig)))

    # further filter if only correct or incorrect trials are plotted
    if trials == 'c':
        trials_short_sig = filter_trials(behav_ds, [], ['trial_successful'],trials_short_sig)
        trials_long = filter_trials(behav_ds, [], ['trial_successful'],trials_long)
    if trials == 'ic':
        trials_short_sig = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_short_sig)
        trials_long = filter_trials(behav_ds, [], ['trial_unsuccessful'],trials_long)

    if split_data:
        # randomly draw 50% of indices to calculate signal and the other half to calculate the order by which to sort
        trials_short_rand = np.random.choice(trials_short_sig, np.size(trials_short_sig), replace=False)
        trials_short_sort = trials_short_rand[np.arange(0,np.floor(np.size(trials_short_sig)/2)).astype(int)]
        trials_short_sig = trials_short_rand[np.arange(np.ceil(np.size(trials_short_sig)/2),np.size(trials_short_sig)).astype(int)]

        trials_long_rand = np.random.choice(trials_long_sig, np.size(trials_long_sig), replace=False)
        trials_long_sort = trials_long_rand[np.arange(0,np.floor(np.size(trials_long_sig)/2)).astype(int)]
        trials_long_sig = trials_long_rand[np.arange(np.ceil(np.size(trials_long_sig)/2),np.size(trials_long_sig)).astype(int)]
    else:
        trials_short_sort = trials_short_sig
        trials_long_sort = trials_long_sig

    # select rois, delete the ones that don't pass criteria
    # ipdb.set_trace()
    if custom_roilist is None:
        roi_selection_short = np.array(roi_params['valid_rois'])
    elif custom_rois_short is not None and not roi_union:
        roi_selection_short = custom_rois_short
        validate_rois = False
    else:
        roi_selection_short = custom_roilist

    # use only selected rois
    if validate_rois:
        selected_alignment_short = np.array([])
        if 'trialonset' in celltypes:
            max_peaks = get_eventaligned_rois(roi_params, ['short'], 'trialonset')
            selected_alignment_short = np.union1d(selected_alignment_short, np.array(max_peaks['trialonset_roi_number_short']))

        if 'lmcenter' in celltypes:
            max_peaks = get_eventaligned_rois(roi_params, ['short'], 'lmcenter')
            selected_alignment_short = np.union1d(selected_alignment_short, np.array(max_peaks['lmcenter_roi_number_short']))

        if 'reward' in celltypes:
            max_peaks = get_eventaligned_rois(roi_params, ['short'], 'reward')
            selected_alignment_short = np.union1d(selected_alignment_short, np.array(max_peaks['reward_roi_number_short']))
        selected_alignment_short = selected_alignment_short.astype(int)

        # determine which rois are passing the criteria for any of the alignment points
        roi_mean_trace = np.array(roi_params['space_mean_trace_short'])
        roi_active_short_trialonset = np.array(roi_params['trialonset_active_short'])
        roi_zscore_short_trialonset = np.array(roi_params['trialonset_peak_zscore_short'])

        roi_active_short_lmcenter = np.array(roi_params['lmcenter_active_short'])
        roi_zscore_short_lmcenter = np.array(roi_params['lmcenter_peak_zscore_short'])

        roi_active_short_reward = np.array(roi_params['reward_active_short'])
        roi_zscore_short_reward = np.array(roi_params['reward_peak_zscore_short'])

        zscore_roi = np.logical_or(roi_zscore_short_trialonset >= MIN_ZSCORE, roi_zscore_short_reward >= MIN_ZSCORE)
        zscore_roi = np.logical_or(zscore_roi, roi_zscore_short_lmcenter >= MIN_ZSCORE)

        active_roi = np.logical_or(roi_active_short_lmcenter >= MIN_FRACTION_ACTIVE, roi_active_short_trialonset >= MIN_FRACTION_ACTIVE)
        active_roi = np.logical_or(active_roi, roi_active_short_reward >= MIN_FRACTION_ACTIVE)

        # get final list of rois that pass all activity and zscore criteria
        active_roi = np.logical_and(zscore_roi, active_roi)
        # active_roi = np.logical_and(active_roi, roi_task_engaged_short > 0)
        # active_roi = roi_task_engaged_short > 0

        # check which rois pass mean amp criterion
        for i in range(len(roi_selection_short)):
            if (np.nanmax(roi_mean_trace[i]) - np.nanmin(roi_mean_trace[i])) < MIN_MEAN_AMP:
                active_roi[i] = False

        # in a somewhat roundabout way this is how we intersect valid rois with our rois that are algined to a certain event
        active_roi = np.where(active_roi==False)[0]
        roi_selection_short = np.delete(roi_selection_short, active_roi)
        roi_selection_short = np.intersect1d(roi_selection_short, selected_alignment_short)

    # if we want to do a union on
    # ipdb.set_trace()
    if roi_union is True and custom_rois_short is not None:
        roi_selection_short = np.union1d(custom_rois_short, roi_selection_short)

    # select rois, delete the ones that don't pass criteria
    if custom_roilist is None:
        roi_selection_long = np.array(roi_params['valid_rois'])
    elif custom_rois_long is not None and not roi_union:
        roi_selection_long = custom_rois_long
        validate_rois = False
    else:
        roi_selection_long = custom_roilist


    if validate_rois:
        # use only selected rois
        selected_alignment_long = np.array([])
        if 'trialonset' in celltypes:
            max_peaks = get_eventaligned_rois(roi_params, ['long'], 'trialonset')
            selected_alignment_long = np.union1d(selected_alignment_long, np.array(max_peaks['trialonset_roi_number_long']))

        if 'lmcenter' in celltypes:
            max_peaks = get_eventaligned_rois(roi_params, ['long'], 'lmcenter')
            selected_alignment_long = np.union1d(selected_alignment_long, np.array(max_peaks['lmcenter_roi_number_long']))

        if 'reward' in celltypes:
            max_peaks = get_eventaligned_rois(roi_params, ['long'], 'reward')
            selected_alignment_long = np.union1d(selected_alignment_long, np.array(max_peaks['reward_roi_number_long']))
        selected_alignment_long = selected_alignment_long.astype(int)

        # roi_task_engaged_long = np.array(roi_params['task_engaged_long'])
        roi_mean_trace = np.array(roi_params['space_mean_trace_long'])
        roi_active_long_trialonset = np.array(roi_params['trialonset_active_long'])
        roi_zscore_long_trialonset = np.array(roi_params['trialonset_peak_zscore_long'])

        roi_active_long_lmcenter = np.array(roi_params['lmcenter_active_long'])
        roi_zscore_long_lmcenter = np.array(roi_params['lmcenter_peak_zscore_long'])

        roi_active_long_reward = np.array(roi_params['reward_active_long'])
        roi_zscore_long_reward = np.array(roi_params['reward_peak_zscore_long'])

        zscore_roi = np.logical_or(roi_zscore_long_trialonset >= MIN_ZSCORE, roi_zscore_long_reward >= MIN_ZSCORE)
        zscore_roi = np.logical_or(zscore_roi, roi_zscore_long_lmcenter >= MIN_ZSCORE)

        active_roi = np.logical_or(roi_active_long_lmcenter >= MIN_FRACTION_ACTIVE, roi_active_long_trialonset >= MIN_FRACTION_ACTIVE)
        active_roi = np.logical_or(active_roi, roi_active_long_reward >= MIN_FRACTION_ACTIVE)

        active_roi = np.logical_and(zscore_roi, active_roi)
        # active_roi = np.logical_and(active_roi, roi_task_engaged_long > 0)
        # active_roi = roi_task_engaged_long > 0

        for i in range(len(roi_selection_long)):
            if (np.nanmax(roi_mean_trace[i]) - np.nanmin(roi_mean_trace[i])) < MIN_MEAN_AMP:
                active_roi[i] = False

        active_roi = np.where(active_roi==False)[0]
        roi_selection_long = np.delete(roi_selection_long, active_roi)
        roi_selection_long = np.intersect1d(roi_selection_long, selected_alignment_long)

    # if we want to do a union on
    if roi_union is True and custom_rois_long is not None:
        roi_selection_long = np.union1d(custom_rois_long, roi_selection_long)

    if trial_by_trial:
        roi_trial_data_short, trial_fl_data_short = population_trial_activity(behav_ds, licks_ds, reward_ds, dF_ds, roi_params, binnr_short, maxlength_short, trials_short_sig, 'short', roi_selection_short, mouse, session)
        roi_trial_data_long, trial_fl_data_long = population_trial_activity(behav_ds, licks_ds, reward_ds, dF_ds, roi_params, binnr_long, maxlength_long, trials_long_sig, 'long', roi_selection_long, mouse, session)
    else:
        roi_trial_data_short = []
        roi_trial_data_long = []
        trial_fl_data_short = []
        trial_fl_data_long = []

    if EXPORT_MATLAB:
        export_data_matlab(mouse, session, behav_ds, dF_ds, roi_selection_short, 'short', 'short_'+fname)
        export_data_matlab(mouse, session, behav_ds, dF_ds, roi_selection_long, 'long', 'long_'+fname)
        export_data_matlab(mouse, session, behav_ds, dF_ds, np.union1d(roi_selection_short,roi_selection_long), 'long', 'all_'+fname)

    # print('---')
    # print(roi_params['valid_rois'][16], np.round(roi_active_long_reward[16],2), np.round(roi_zscore_long_reward[16],2), np.round(np.nanmax(roi_mean_trace[16]) - np.nanmin(roi_mean_trace[16]),2))
    # print(roi_selection_long.shape, roi_selection_long)
    # print(roi_selection_short.shape, roi_selection_long.shape)

    # Figure out which neurons overlap
    if sortby == 'both':
        neurons_overlap = np.intersect1d(roi_selection_short,roi_selection_long)
        only_short_rois = roi_selection_short
        only_long_rois = roi_selection_long
        for no in neurons_overlap:
            only_short_rois = np.delete(only_short_rois, np.where(only_short_rois==no)[0])
        for no in neurons_overlap:
            only_long_rois = np.delete(only_long_rois, np.where(only_long_rois==no)[0])
    else:
        neurons_overlap = []
        only_short_rois = []
        only_long_rois = []

    # determine which trial type is used to select and order neurons by
    if sortby == 'short':
        roi_selection_long = roi_selection_short
    elif sortby == 'long':
        roi_selection_short = roi_selection_long

    tot_rois_short += np.size(roi_selection_short)
    tot_rois_long += np.size(roi_selection_long)

    # storage for mean ROI data
    mean_dF_short_sig = np.zeros((binnr_short,np.size(roi_selection_short)))
    mean_dF_short_sort = np.zeros((binnr_short,np.size(roi_selection_short)))

    mean_dF_long_sig = np.zeros((binnr_long,np.size(roi_selection_long)))
    mean_dF_long_sort = np.zeros((binnr_long,np.size(roi_selection_long)))

    for j,roi in enumerate(roi_selection_short):
        # intilize matrix to hold data for all trials of this roi
        mean_dF_trials_short_sig = np.zeros((np.size(trials_short_sig,0),binnr_short))
        mean_dF_trials_short_sort = np.zeros((np.size(trials_short_sort,0),binnr_short))
        # calculate mean dF vs location for each ROI on trials
        try:
            mean_dF_short_sig[:,j] = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_short_sig, 'short', align_point, binnr_short, maxlength_short)
            mean_dF_short_sort[:,j] = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_short_sort, 'short', align_point, binnr_short, maxlength_short)
        except:
            ipdb.set_trace()
        # if not split_data:
        #     mean_dF_short_sig[np.argwhere(mean_dF_short_sig[:,j]==1),j] = 0

    for j,roi in enumerate(roi_selection_long):
        # intilize matrix to hold data for all trials of this roi
        mean_dF_trials_long_sig = np.zeros((np.size(trials_long_sig,0),binnr_long))
        mean_dF_trials_long_sort = np.zeros((np.size(trials_long_sort,0),binnr_long))
        # calculate mean dF vs location for each ROI on trials
        mean_dF_long_sig[:,j] = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_long_sig, 'long', align_point, binnr_long, maxlength_long)
        mean_dF_long_sort[:,j] = calc_spacevloc(behav_ds, dF_ds[:,roi], trials_long_sort, 'long', align_point, binnr_long, maxlength_long)
        # if not split_data:
        #     mean_dF_long_sig[np.argwhere(mean_dF_long_sig[:,j]==1),j] = 0

    if make_individual_figure:
        # create figure and axes
        fig = plt.figure(figsize=(20,15))
        ax1 = plt.subplot2grid((1,2),(0,0))
        ax2 = plt.subplot2grid((1,2),(0,1))

        # sort by peak activity (naming of variables confusing because the script grew organically...)
        mean_dF_sort_short = np.zeros(mean_dF_short_sort.shape[1])
        for i, row in enumerate(mean_dF_short_sort.T):
            if not np.all(np.isnan(row)):
                mean_dF_sort_short[i] = np.nanargmax(row)
            else:
                print('WARNING: sort signal with all NaN encountered. ROI not plotted.')

        sort_ind_short = np.argsort(mean_dF_sort_short)

        mean_dF_sort_long = np.zeros(mean_dF_long_sort.shape[1])
        for i, row in enumerate(mean_dF_long_sort.T):
            if not np.all(np.isnan(row)):
                mean_dF_sort_long[i] = np.nanargmax(row)
            else:
                print('WARNING: sort signal with all NaN encountered. ROI not plotted.')

        sort_ind_long = np.argsort(mean_dF_sort_long)

        if sortby == 'none':
            sns.heatmap(np.transpose(mean_dF_short_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'short':
            sns.heatmap(np.transpose(mean_dF_short_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'long':
            sns.heatmap(np.transpose(mean_dF_short_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'both':
            if np.size(mean_dF_short_sig > 0):
                sns.heatmap(np.transpose(mean_dF_short_sig[:,sort_ind_short]), cmap='viridis', yticklabels=np.array(roi_selection_short)[sort_ind_short], vmin=0.0, vmax=1, ax=ax1, cbar=False)
            if np.size(mean_dF_long_sig > 0):
                sns.heatmap(np.transpose(mean_dF_long_sig[:,sort_ind_long]), cmap='viridis', yticklabels=np.array(roi_selection_long)[sort_ind_long], vmin=0.0, vmax=1, ax=ax2, cbar=False)

        # ax1.axvline((200/BIN_SIZE)-start_bin, lw=3, c='0.8')
        # ax1.axvline((240/BIN_SIZE)-start_bin, lw=3, c='0.8')
        # ax1.axvline((320/BIN_SIZE)-start_bin, lw=3, c='0.8')
        # ax2.axvline((200/BIN_SIZE)-start_bin, lw=3, c='0.8')
        # ax2.axvline((240/BIN_SIZE)-start_bin, lw=3, c='0.8')
        # ax2.axvline((380/BIN_SIZE)-start_bin, lw=3, c='0.8')

        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)

        fname = loc_info['figure_output_path'] + subfolder + os.sep + mouse+'_'+session + '_' + fname + '.' + fformat
        print(fname)
        # try:
        fig.savefig(fname, format=fformat)
        # except:
        #     exc_type, exc_value, exc_traceback = sys.exc_info()
        #     traceback.print_exception(exc_type, exc_value, exc_traceback,
        #                           limit=2, file=sys.stdout)

    # ipdb.set_trace()

    return mean_dF_short_sig, mean_dF_short_sort, \
            mean_dF_long_sig, mean_dF_long_sort, \
            tot_rois_short, tot_rois_long, \
            [neurons_overlap, only_short_rois, only_long_rois], \
            roi_trial_data_short, roi_trial_data_long, \
            trial_fl_data_short, trial_fl_data_long, \
            roi_selection_short, roi_selection_long

def fig_popvloc_naiveexpert(roi_param_list, celltypes=['trialonset','lmcenter','reward'], reward_distance=0, trials='both', sortby='both', split_data=True, align_point = 'landmark', fname='', subfolder='', write_to_dict=False, sort_VR=None, which_cc='expert_naive'):

    maxlength_short = TRACKLENGTH_SHORT - TRACK_CROP + reward_distance
    maxlength_long = TRACKLENGTH_LONG - TRACK_CROP + reward_distance
    # define spatial bins for analysis
    binnr_short = int(maxlength_short/BIN_SIZE)
    binnr_long = int(maxlength_long/BIN_SIZE)
    reward_bins = int(reward_distance/BIN_SIZE)

    # set up track parameters based on alignment
    if align_point == 'trialonset':
        track_start = 5
    elif align_point == 'landmark':
        track_start = TRACK_CROP
    else:
        print('ERROR: alignment point does not exist, Terminating.')
        return

    start_bin = int(track_start / BIN_SIZE)
    lm_start = int((100)/BIN_SIZE)
    lm_end = int((140)/BIN_SIZE)
    end_bin_short = int((maxlength_short / BIN_SIZE) + reward_bins)
    end_bin_long = int((maxlength_long / BIN_SIZE) + reward_bins)

    popvec_cc_matrix_short_size = int(binnr_short)
    popvec_cc_matrix_long_size = int(binnr_long)

    histo_short = np.zeros((popvec_cc_matrix_short_size, NUM_ITER))
    histo_long = np.zeros((popvec_cc_matrix_long_size, NUM_ITER))

    # pull out location bins that exist in both track types
    popvec_cc_matrix_ss = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_ll = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))
    popvec_cc_matrix_sl = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_sl_stretched = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))

    popvec_cc_matrix_ss_pearsonr = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_ll_pearsonr = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))
    popvec_cc_matrix_sl_pearsonr = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))

    popvec_cc_reconstruction_ss = np.zeros((2,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_reconstruction_sl = np.zeros((2,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_reconstruction_ll = np.zeros((2,popvec_cc_matrix_long_size,NUM_ITER))

    std_prelm_ss = []
    std_lm_ss = []
    std_pi_ss = []

    std_prelm_sl = []
    std_lm_sl = []
    std_pi_sl = []

    std_prelm_ll = []
    std_lm_ll = []
    std_pi_ll = []

    for current_iter in range(NUM_ITER):
        print('CURRENT ITERATION: ' + str(current_iter))

        tot_rois_short = 0
        tot_rois_long = 0
        mean_dF_short_coll_sig = np.empty((binnr_short,0))
        mean_dF_short_coll_sort = np.empty((binnr_short,0))
        mean_dF_long_coll_sig = np.empty((binnr_long,0))
        mean_dF_long_coll_sort = np.empty((binnr_long,0))
        mean_dF_short_coll_all = np.empty((binnr_short,0))
        mean_dF_long_coll_all = np.empty((binnr_long,0))

        overlapping_rois_all = []
        only_short_rois_all = []
        only_long_rois_all = []

        trial_error_slopes_short = []
        trial_corr_slopes_short = []
        trial_error_pearsonr_pval_short = []
        trial_error_pearsonp_pval_short = []
        trial_corr_pearsonr_pval_short = []
        trial_corr_pearsonp_pval_short = []

        trial_error_slopes_long = []
        trial_corr_slopes_long = []
        trial_error_pearsonr_pval_long = []
        trial_error_pearsonp_pval_long = []
        trial_corr_pearsonr_pval_long = []
        trial_corr_pearsonp_pval_long = []

        # load data
        # ipdb.set_trace()
        for r in roi_param_list:

            # calculate regular population activity maps (split trials 50:50)
            mean_dF_short_expert, mean_dF_short_expert_sort, \
            mean_dF_long_expert, mean_dF_long_expert_sort, \
            tot_rois_short_expert, tot_rois_long_expert, rois_overlap_expert, \
            _,_,_,_, \
            roi_selection_short_expert, roi_selection_long_expert = \
                popvloc_individual(r[1], trials, celltypes, True, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname, tot_rois_short, tot_rois_long, TRIAL_BY_TRIAL, True, subfolder)

            # ipdb.set_trace()

            mean_dF_short_naive, mean_dF_short_naive_sort, \
            mean_dF_long_naive, mean_dF_long_naive_sort, \
            tot_rois_short_naive, tot_rois_long_naive, rois_overlap_naive, \
            _,_,_,_, \
            roi_selection_short_expert, roi_selection_long_expert = \
                popvloc_individual(r[0], trials, celltypes, True, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname, tot_rois_short, tot_rois_long, TRIAL_BY_TRIAL, True, subfolder, None, roi_selection_short_expert, roi_selection_long_expert, F)

            # ipdb.set_trace()

            mean_dF_short_expert, mean_dF_short_expert_sort, \
            mean_dF_long_expert, mean_dF_long_expert_sort, \
            tot_rois_short_expert, tot_rois_long_expert, rois_overlap_expert, \
            _,_,_,_, \
            roi_selection_short_expert, roi_selection_long_expert = \
                popvloc_individual(r[1], trials, celltypes, True, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname, tot_rois_short, tot_rois_long, TRIAL_BY_TRIAL, True, subfolder, None, roi_selection_short_expert, roi_selection_long_expert, True)


            # ipdb.set_trace()

            if which_cc is 'expert_naive':
                mean_dF_short_coll_sig = np.append(mean_dF_short_coll_sig, mean_dF_short_expert, axis=1)
                mean_dF_short_coll_sort = np.append(mean_dF_short_coll_sort, mean_dF_short_naive, axis=1)
                mean_dF_long_coll_sig = np.append(mean_dF_long_coll_sig, mean_dF_long_expert, axis=1)
                mean_dF_long_coll_sort = np.append(mean_dF_long_coll_sort, mean_dF_long_naive, axis=1)
                mean_dF_short_coll_all = np.append(mean_dF_short_coll_all, (mean_dF_short_expert + mean_dF_short_naive) / 2, axis=1)
                mean_dF_long_coll_all = np.append(mean_dF_long_coll_all, (mean_dF_long_expert + mean_dF_long_naive) / 2, axis=1)
            elif  which_cc is 'expert_expert':
                mean_dF_short_coll_sig = np.append(mean_dF_short_coll_sig, mean_dF_short_expert, axis=1)
                mean_dF_short_coll_sort = np.append(mean_dF_short_coll_sort, mean_dF_short_expert_sort, axis=1)
                mean_dF_long_coll_sig = np.append(mean_dF_long_coll_sig, mean_dF_long_expert, axis=1)
                mean_dF_long_coll_sort = np.append(mean_dF_long_coll_sort, mean_dF_long_expert_sort, axis=1)
                mean_dF_short_coll_all = np.append(mean_dF_short_coll_all, mean_dF_short_expert, axis=1)
                mean_dF_long_coll_all = np.append(mean_dF_long_coll_all, mean_dF_long_expert, axis=1)
            elif  which_cc is 'naive_naive':
                mean_dF_short_coll_sig = np.append(mean_dF_short_coll_sig, mean_dF_short_naive, axis=1)
                mean_dF_short_coll_sort = np.append(mean_dF_short_coll_sort, mean_dF_short_naive_sort, axis=1)
                mean_dF_long_coll_sig = np.append(mean_dF_long_coll_sig, mean_dF_long_naive, axis=1)
                mean_dF_long_coll_sort = np.append(mean_dF_long_coll_sort, mean_dF_long_naive_sort, axis=1)
                mean_dF_short_coll_all = np.append(mean_dF_short_coll_all, mean_dF_short_naive, axis=1)
                mean_dF_long_coll_all = np.append(mean_dF_long_coll_all, mean_dF_long_naive, axis=1)

        #if there are fewer neuron for short trials: just take the first n=number of neurons on short trials of the long trials for comparison.
        num_short_neurons = mean_dF_short_coll_sort.shape[1]
        num_long_neurons = mean_dF_long_coll_sort.shape[1]
        if num_short_neurons < num_long_neurons:
            short_vec_rois = np.arange(num_short_neurons)
            # long_vec_rois = np.sort(np.random.choice(num_long_neurons, num_short_neurons, replace=False))
            long_vec_rois = np.arange(num_short_neurons)
        elif num_short_neurons > num_long_neurons:
            short_vec_rois = np.sort(np.random.choice(num_short_neurons, num_long_neurons, replace=False))
            long_vec_rois = np.arange(num_long_neurons)


        for row in range(popvec_cc_matrix_short_size):
            for col in range(popvec_cc_matrix_short_size):
                popvec_cc_matrix_ss_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_short_coll_sig[row,:],mean_dF_short_coll_sort[col,:])[0]

        for row in range(popvec_cc_matrix_long_size):
            for col in range(popvec_cc_matrix_long_size):
                popvec_cc_matrix_ll_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_long_coll_sig[row,:],mean_dF_long_coll_sort[col,:])[0]

        # run through every row and find element with largest correlation coefficient
        for i,row in enumerate(range(popvec_cc_matrix_short_size)):
            popvec_cc_reconstruction_ss[0,i,current_iter] = i
            try:
                popvec_cc_reconstruction_ss[1,i,current_iter] = np.nanargmax(popvec_cc_matrix_ss_pearsonr[i,:,current_iter])
            except ValueError:
                popvec_cc_reconstruction_ss[1,i,current_iter] = 0

            popvec_cc_reconstruction_sl[0,i,current_iter] = i

        for i,row in enumerate(range(popvec_cc_matrix_long_size)):
            popvec_cc_reconstruction_ll[0,i,current_iter] = i
            try:
                popvec_cc_reconstruction_ll[1,i,current_iter] = np.nanargmax(popvec_cc_matrix_ll_pearsonr[i,:,current_iter])
            except ValueError:
                popvec_cc_reconstruction_ll[1,i,current_iter] = 0

        # calculate standard deviation (step by step so even an idiot like myself doesn't get confused) of reconstruction vs actual location
        bin_diff = popvec_cc_reconstruction_ss[1,:,current_iter] - popvec_cc_reconstruction_ss[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_ss = np.append(std_prelm_ss, np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_ss = np.append(std_lm_ss,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_ss = np.append(std_pi_ss,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))

        # bin_diff = popvec_cc_reconstruction_sl[1,:,current_iter] - popvec_cc_reconstruction_sl[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_sl = np.append(std_prelm_sl,np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_sl = np.append(std_lm_sl,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_sl = np.append(std_pi_sl,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))

        bin_diff = popvec_cc_reconstruction_ll[1,:,current_iter] - popvec_cc_reconstruction_ll[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_ll = np.append(std_prelm_ll,np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_ll = np.append(std_lm_ll,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_ll = np.append(std_pi_ll,np.sqrt(np.sum(bin_diff[lm_end:end_bin_long])/(end_bin_long-lm_end)))



        # sort by peak activity (naming of variables confusing because the script grew organically...)
        mean_dF_sort_short = np.zeros(mean_dF_short_coll_sig.shape[1])
        for i, row in enumerate(mean_dF_short_coll_sig.T):
            mean_dF_sort_short[i] = np.nanargmax(row)
        histo_short[:,current_iter] = np.histogram(mean_dF_sort_short,np.arange(0,binnr_short+1,1))[0]

        mean_dF_sort_long = np.zeros(mean_dF_long_coll_sig.shape[1])
        for i, row in enumerate(mean_dF_long_coll_sig.T):
            mean_dF_sort_long[i] = np.nanargmax(row)
        histo_long[:,current_iter] = np.histogram(mean_dF_sort_long,np.arange(0,binnr_long+1,1))[0]

        print('--- Number of ROIs ---')
        print('short: ' + str(tot_rois_short))
        print('long: ' + str(tot_rois_long))
        print('----------------------')

    # # calculate mean cc maps
    popvec_cc_matrix_ss_mean = np.nanmean(popvec_cc_matrix_ss_pearsonr,axis=2)
    popvec_cc_matrix_sl_mean = np.nanmean(popvec_cc_matrix_sl_pearsonr,axis=2)
    popvec_cc_matrix_ll_mean = np.nanmean(popvec_cc_matrix_ll_pearsonr,axis=2)
    #
    # calculate reconstructed location estimate from mean cc map
    popvec_cc_reconstruction_ss_mean = np.zeros((2,popvec_cc_matrix_short_size))
    popvec_cc_reconstruction_sl_mean = np.zeros((2,popvec_cc_matrix_short_size))
    popvec_cc_reconstruction_ll_mean = np.zeros((2,popvec_cc_matrix_long_size))

    for i in range(popvec_cc_matrix_short_size):
        popvec_cc_reconstruction_ss_mean[0,i] = i
        popvec_cc_reconstruction_ss_mean[1,i] = np.argmax(popvec_cc_matrix_ss_mean[i,:])

        popvec_cc_reconstruction_sl_mean[0,i] = i
        popvec_cc_reconstruction_sl_mean[1,i] = np.argmax(popvec_cc_matrix_sl_mean[i,:])

    for i in range(popvec_cc_matrix_long_size):
        popvec_cc_reconstruction_ll_mean[0,i] = i
        popvec_cc_reconstruction_ll_mean[1,i] = np.argmax(popvec_cc_matrix_ll_mean[i,:])

    # create figure and axes
    fig = plt.figure(figsize=(30,15))
    ax1 = plt.subplot2grid((6,200),(0,0),rowspan=3, colspan=45)
    ax2 = plt.subplot2grid((6,200),(0,50),rowspan=3, colspan=55)
    # ax3 = plt.subplot2grid((5,200),(0,0),rowspan=1, colspan=40)
    # ax4 = plt.subplot2grid((5,200),(0,50),rowspan=1, colspan=55)
    ax5 = plt.subplot2grid((6,200),(0,110),rowspan=2, colspan=40)
    ax6 = plt.subplot2grid((6,200),(2,110),rowspan=2, colspan=40)
    ax7 = plt.subplot2grid((6,200),(0,155),rowspan=2, colspan=40)
    ax8 = plt.subplot2grid((6,200),(2,155),rowspan=2, colspan=40)

    ax9 = plt.subplot2grid((6,200),(3,0),rowspan=3, colspan=45)
    ax10 = plt.subplot2grid((6,200),(3,50),rowspan=3, colspan=55)

    ax11 = plt.subplot2grid((6,200),(4,110),rowspan=1, colspan=40)
    ax12 = plt.subplot2grid((6,200),(4,155),rowspan=1, colspan=40)

    min_val = np.nanmin(np.nanmin(popvec_cc_matrix_ss_mean))
    max_val = np.nanmax(np.nanmax(popvec_cc_matrix_ss_mean))
    # print(min_val,max_val)
    min_val = -0.2
    max_val = 1.0
    ax5_img = ax5.pcolor(popvec_cc_matrix_ss_mean.T,cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar(ax5_img, ax=ax5)

    ax5.set_xlabel('short')
    ax5.set_ylabel('short')
    ax5.set_xlim([0,popvec_cc_matrix_short_size])
    ax5.set_ylim([0,popvec_cc_matrix_short_size])


    min_val = np.nanmin(np.nanmin(popvec_cc_matrix_ll_mean))
    max_val = np.nanmax(np.nanmax(popvec_cc_matrix_ll_mean))
    print(min_val,max_val)
    min_val = -0.2
    max_val = 1.0
    ax6_img = ax6.pcolormesh(popvec_cc_matrix_ll_mean.T,cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar(ax6_img, ax=ax6)

    mean_dF_sort_short = np.zeros(mean_dF_short_coll_sig.shape[1])
    for i, row in enumerate(mean_dF_short_coll_sig.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_short[i] = np.nanargmax(row)
        else:
            print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
    sort_ind_short = np.argsort(mean_dF_sort_short)

    mean_dF_sort_long = np.zeros(mean_dF_long_coll_sig.shape[1])
    for i, row in enumerate(mean_dF_long_coll_sig.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_long[i] = np.nanargmax(row)
        else:
            # print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
            pass
    sort_ind_long = np.argsort(mean_dF_sort_long)

    mean_dF_sort_short = np.zeros(mean_dF_short_coll_sort.shape[1])
    for i, row in enumerate(mean_dF_short_coll_sort.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_short[i] = np.nanargmax(row)
        else:
            print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
    sort_ind_short_naive = np.argsort(mean_dF_sort_short)

    mean_dF_sort_long = np.zeros(mean_dF_long_coll_sort.shape[1])
    for i, row in enumerate(mean_dF_long_coll_sort.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_long[i] = np.nanargmax(row)
        else:
            # print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
            pass
    sort_ind_long_naive = np.argsort(mean_dF_sort_long)

    # ipdb.set_trace()
    if sort_VR is None:
        if sortby == 'none':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'short':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'long':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'both':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
            sns.heatmap(np.transpose(mean_dF_short_coll_sort[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax9, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sort[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax10, cbar=False)

    else:
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_VR[0]]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_VR[1]]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)

    # histo_short_smoothed = signal.savgol_filter(np.nanmean(histo_short,axis=1), 5, 3)
    # histo_long_smoothed = signal.savgol_filter(np.nanmean(histo_long,axis=1), 5, 3)
    # ax9.plot(histo_short_smoothed,c='k',lw=2)
    # ax10.plot(histo_long_smoothed,c='k',lw=2)
    #
    # print(mean_dF_short_coll_sort)
    # print(mean_dF_short_coll_sort*5)
    # print((mean_dF_short_coll_sort*5)-220)

    # ax11.hist(np.absolute((mean_dF_sort_short*BIN_SIZE)-220))
    # ax12.hist(np.absolute((mean_dF_sort_long*BIN_SIZE)-220))

    if align_point == 'landmark':
        # ax1.axvline((102/BIN_SIZE)-start_bin, lw=8, c='r')
        ax1.axvline((220/BIN_SIZE)-start_bin, lw=8, c='#FF0000')
        ax1.axvline((320/BIN_SIZE)-start_bin, lw=8, c='#2EA7DF')
        ax9.axvline((220/BIN_SIZE)-start_bin, lw=8, c='#FF0000')
        ax9.axvline((320/BIN_SIZE)-start_bin, lw=8, c='#2EA7DF')

        # ax2.axvline((102/BIN_SIZE)-start_bin, lw=8, c='r')
        ax2.axvline((220/BIN_SIZE)-start_bin, lw=8, c='#FF0000')
        ax2.axvline((380/BIN_SIZE)-start_bin, lw=8, c='#2EA7DF')
        ax10.axvline((220/BIN_SIZE)-start_bin, lw=8, c='#FF0000')
        ax10.axvline((380/BIN_SIZE)-start_bin, lw=8, c='#2EA7DF')

    if align_point == 'trialonset':
        ax1.axvline(0/BIN_SIZE, lw=8, c='#39B54A')
        ax2.axvline(0/BIN_SIZE, lw=8, c='#39B54A')

    # print(roi_selection)
    # ax1.set_yticklabels([])
    # ax2.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax2.set_xticklabels([])

    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax9.spines['left'].set_visible(False)
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.spines['bottom'].set_visible(False)
    ax9.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax10.spines['left'].set_visible(False)
    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.spines['bottom'].set_visible(False)
    ax10.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')
#
#     #ax3.set_xlim([0,23])
#     #ax4.set_xlim([0,35])
#     #ax4.set_xlim([0, end_bin_long-start_bin+1])
#
#     #fig.suptitle(fname + '_' + trials + '_' + str([''.join(str(r) for r in ri) for ri in rec_info]),wrap=True)
#
# #    plt.tight_layout()
#     # list_test = popvec_cc_reconstruction_ss[0,:,]
#     # print(len(popvec_cc_reconstruction_ss.tolist()[0]),len(popvec_cc_reconstruction_ss.tolist()[1]))
    f_value_ss = np.nan
    p_value_ss = np.nan
    f_value_ll = np.nan
    p_value_ll = np.nan
    posthoc_res_ss = np.nan
    posthoc_res_sl = np.nan
    posthoc_res_ll = np.nan
    # ipdb.set_trace()
    if NUM_ITER > 1:
        popvloc_results = {
            # 'included_datasets' : roi_param_list,
            'short_short' : [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)],
            'long_long' : [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)],
            'short_short_SEM' : [stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],
            'long_long_SEM' : [stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],
            'short_short_f_value' : f_value_ss,
            'short_short_p_value' : p_value_ss,
            'short_short_multicomparison' : [posthoc_res_ss,posthoc_res_ss,posthoc_res_ss],
            'long_long_f_value' : f_value_ll,
            'long_long_p_value' : p_value_ll,
            'long_long_multicomparison' : [posthoc_res_ll,posthoc_res_ll,posthoc_res_ll],
            'iterations:' : [NUM_ITER],
            'popvec_cc_reconstruction_ss' : popvec_cc_reconstruction_ss.tolist(),
            'popvec_cc_reconstruction_ll' : popvec_cc_reconstruction_ll.tolist(),
            'popvec_cc_matrix_ss_mean' : popvec_cc_matrix_ss_mean.tolist(),
            'popvec_cc_matrix_ll_mean' : popvec_cc_matrix_ll_mean.tolist(),
            'popvec_zscore_matrix_ss_mean' : popvec_cc_matrix_ss_mean.tolist(),
            'popvec_zscore_matrix_ll_mean' : popvec_cc_matrix_ll_mean.tolist()
        }
    else:
        popvloc_results = {}
#


    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)

    if write_to_dict:
        with open(loc_info['figure_output_path'] + 'popvloc' + os.sep + fname + '_popvloc_results_naiveexpert_' + str(NUM_ITER) + '.json','w') as f:
            json.dump(popvloc_results,f)
            print(loc_info['figure_output_path'] + 'popvloc' + os.sep + fname + '_popvloc_results_naiveexpert_' + str(NUM_ITER) + '.json')

    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '_' + str(NUM_ITER) + '.' + fformat
    print(fname)
    # try:
    fig.savefig(fname, format=fformat)
    # except:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     traceback.print_exception(exc_type, exc_value, exc_traceback,
    #                           limit=2, file=sys.stdout)

    return sort_ind_short, sort_ind_long


def fig_popvloc_s5(roi_param_list, celltypes=['trialonset','lmcenter','reward'], reward_distance=0, trials='both', sortby='both', split_data=True, align_point = 'landmark', fname='', subfolder='', write_to_dict=False, sort_VR=None):
    maxlength_short = TRACKLENGTH_SHORT - TRACK_CROP + reward_distance
    maxlength_long = TRACKLENGTH_LONG - TRACK_CROP + reward_distance
    # define spatial bins for analysis
    binnr_short = int(maxlength_short/BIN_SIZE)
    binnr_long = int(maxlength_long/BIN_SIZE)
    reward_bins = int(reward_distance/BIN_SIZE)

    # set up track parameters based on alignment
    if align_point == 'trialonset':
        track_start = 5
    elif align_point == 'landmark':
        track_start = TRACK_CROP
    else:
        print('ERROR: alignment point does not exist, Terminating.')
        return

    start_bin = int(track_start / BIN_SIZE)
    lm_start = int((100)/BIN_SIZE)
    lm_end = int((140)/BIN_SIZE)
    end_bin_short = int((maxlength_short / BIN_SIZE) + reward_bins)
    end_bin_long = int((maxlength_long / BIN_SIZE) + reward_bins)

    popvec_cc_matrix_short_size = int(binnr_short)
    popvec_cc_matrix_long_size = int(binnr_long)

    histo_short = np.zeros((popvec_cc_matrix_short_size, NUM_ITER))
    histo_long = np.zeros((popvec_cc_matrix_long_size, NUM_ITER))

    # pull out location bins that exist in both track types
    popvec_cc_matrix_ss = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_ll = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))
    popvec_cc_matrix_sl = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_sl_stretched = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))

    popvec_cc_matrix_ss_pearsonr = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_matrix_ll_pearsonr = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,NUM_ITER))
    popvec_cc_matrix_sl_pearsonr = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,NUM_ITER))

    popvec_cc_reconstruction_ss = np.zeros((2,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_reconstruction_sl = np.zeros((2,popvec_cc_matrix_short_size,NUM_ITER))
    popvec_cc_reconstruction_ll = np.zeros((2,popvec_cc_matrix_long_size,NUM_ITER))

    std_prelm_ss = []
    std_lm_ss = []
    std_pi_ss = []

    std_prelm_sl = []
    std_lm_sl = []
    std_pi_sl = []

    std_prelm_ll = []
    std_lm_ll = []
    std_pi_ll = []

    for current_iter in range(NUM_ITER):
        print('CURRENT ITERATION: ' + str(current_iter))

        tot_rois_short = 0
        tot_rois_long = 0
        mean_dF_short_coll_sig = np.empty((binnr_short,0))
        mean_dF_short_coll_sort = np.empty((binnr_short,0))
        mean_dF_long_coll_sig = np.empty((binnr_long,0))
        mean_dF_long_coll_sort = np.empty((binnr_long,0))
        mean_dF_short_coll_all = np.empty((binnr_short,0))
        mean_dF_long_coll_all = np.empty((binnr_long,0))

        overlapping_rois_all = []
        only_short_rois_all = []
        only_long_rois_all = []

        trial_error_slopes_short = []
        trial_corr_slopes_short = []
        trial_error_pearsonr_pval_short = []
        trial_error_pearsonp_pval_short = []
        trial_corr_pearsonr_pval_short = []
        trial_corr_pearsonp_pval_short = []

        trial_error_slopes_long = []
        trial_corr_slopes_long = []
        trial_error_pearsonr_pval_long = []
        trial_error_pearsonp_pval_long = []
        trial_corr_pearsonr_pval_long = []
        trial_corr_pearsonp_pval_long = []

        # load data
        for r in roi_param_list:

            # calculate regular population activity maps (split trials 50:50)
            mean_dF_short_sig, mean_dF_short_sort, \
            mean_dF_long_sig, mean_dF_long_sort, \
            tot_rois_short, tot_rois_long, rois_overlap, \
            _,_,_,_,_,_ = \
                popvloc_individual(r, trials, celltypes, split_data, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname, tot_rois_short, tot_rois_long, TRIAL_BY_TRIAL, False, subfolder)


            if TRIAL_BY_TRIAL:
                # calculate population activity maps trial by trial and don't split for session activity
                mean_dF_short_sig_nosplit, _, \
                mean_dF_long_sig_nosplit, _, \
                _, _, _, \
                roi_trial_data_short, roi_trial_data_long,\
                trial_fl_data_short, trial_fl_data_long,_,_ = \
                    popvloc_individual(r, trials, celltypes, False, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname, tot_rois_short, tot_rois_long, TRIAL_BY_TRIAL, False, subfolder)

                # calculate population activity maps and split for 'best' vs 'worst' trials in terms of how close to the start of the reward one the animal started licking
                # popvloc_individual(r, trials, celltypes, False, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname + '_close', tot_rois_short, tot_rois_long, TRIAL_BY_TRIAL, True, subfolder, ['first_lick_distance',25,'<'])
                # popvloc_individual(r, trials, celltypes, False, align_point, binnr_short, binnr_long, maxlength_short, maxlength_long, sortby, fname + '_far', tot_rois_short, tot_rois_long, TRIAL_BY_TRIAL, True, subfolder, ['first_lick_distance',25,'>'])


                trial_cc_short, error_slope_short, corr_slope_short, error_r_short, error_p_short, corr_r_short, corr_p_short = \
                    calc_trial_cc(mean_dF_short_sig_nosplit, roi_trial_data_short, trial_fl_data_short, binnr_short, 'short', r[1], r[2])
                trial_cc_long, error_slope_long, corr_slope_long, error_r_long, error_p_long, corr_r_long, corr_p_long = \
                    calc_trial_cc(mean_dF_long_sig_nosplit, roi_trial_data_long, trial_fl_data_long, binnr_long, 'long', r[1], r[2])

                trial_error_slopes_short.append(error_slope_short)
                trial_corr_slopes_short.append(corr_slope_short)
                trial_error_pearsonr_pval_short.append(error_r_short)
                trial_error_pearsonp_pval_short.append(error_p_short)
                trial_corr_pearsonr_pval_short.append(corr_r_short)
                trial_corr_pearsonp_pval_short.append(corr_p_short)

                trial_error_slopes_long.append(error_slope_long)
                trial_corr_slopes_long.append(corr_slope_long)
                trial_error_pearsonr_pval_long.append(error_r_long)
                trial_error_pearsonp_pval_long.append(error_p_long)
                trial_corr_pearsonr_pval_long.append(corr_r_long)
                trial_corr_pearsonp_pval_long.append(corr_p_long)

            # ipdb.set_trace()
            # keep track of all rois
            overlapping_rois_all.extend(rois_overlap[0])
            only_short_rois_all.extend(rois_overlap[1])
            only_long_rois_all.extend(rois_overlap[2])
            # ipdb.set_trace()

            mean_dF_short_coll_sig = np.append(mean_dF_short_coll_sig, mean_dF_short_sig, axis=1)
            mean_dF_short_coll_sort = np.append(mean_dF_short_coll_sort, mean_dF_short_sort, axis=1)
            mean_dF_long_coll_sig = np.append(mean_dF_long_coll_sig, mean_dF_long_sig, axis=1)
            mean_dF_long_coll_sort = np.append(mean_dF_long_coll_sort, mean_dF_long_sort, axis=1)
            mean_dF_short_coll_all = np.append(mean_dF_short_coll_all, (mean_dF_short_sig + mean_dF_short_sort) / 2, axis=1)
            mean_dF_long_coll_all = np.append(mean_dF_long_coll_all, (mean_dF_long_sig + mean_dF_long_sort) / 2, axis=1)

        if TRIAL_BY_TRIAL:

            fig = plt.figure(figsize=(10,10))
            ax1 = plt.subplot(231)
            ax2 = plt.subplot(232)
            ax3 = plt.subplot(233)
            ax4 = plt.subplot(234)
            ax5 = plt.subplot(235)
            ax6 = plt.subplot(236)

            ax1.scatter(np.full_like(trial_error_slopes_short,0), trial_error_slopes_short, s=80, facecolors='none', edgecolors=SHORT_COLOR)
            ax1.scatter(np.full_like(trial_error_slopes_long,1), trial_error_slopes_long, s=80, facecolors='none', edgecolors=LONG_COLOR)
            ax1.set_title('trial error slopes')
            ax1.set_ylim([-1,1])
            ax1.set_xlim([-1,2])

            ax2.scatter(np.full_like(trial_corr_slopes_short,0), trial_corr_slopes_short, s=80, facecolors='none', edgecolors=SHORT_COLOR)
            ax2.scatter(np.full_like(trial_corr_slopes_long,1), trial_corr_slopes_long, s=80, facecolors='none', edgecolors=LONG_COLOR)
            ax2.set_title('trial corr slopes')
            ax2.set_ylim([-1,1])
            ax2.set_xlim([-1,2])

            ax3.scatter(np.full_like(trial_error_pearsonr_pval_short,0), trial_error_pearsonr_pval_short, s=80, facecolors='none', edgecolors=SHORT_COLOR)
            ax3.scatter(np.full_like(trial_error_pearsonr_pval_long,1), trial_error_pearsonr_pval_long, s=80, facecolors='none', edgecolors=LONG_COLOR)
            ax3.set_title('trial error r')
            ax3.set_ylim([-1,1])
            ax3.set_xlim([-1,2])

            ax4.scatter(np.full_like(trial_error_pearsonp_pval_short,0), trial_error_pearsonp_pval_short, s=80, facecolors='none', edgecolors=SHORT_COLOR)
            ax4.scatter(np.full_like(trial_error_pearsonp_pval_long,1), trial_error_pearsonp_pval_long, s=80, facecolors='none', edgecolors=LONG_COLOR)
            ax4.set_title('trial error p')
            ax4.set_ylim([0,1])
            ax4.set_xlim([-1,2])

            ax5.scatter(np.full_like(trial_corr_pearsonr_pval_short,0), trial_corr_pearsonr_pval_short, s=80, facecolors='none', edgecolors=SHORT_COLOR)
            ax5.scatter(np.full_like(trial_corr_pearsonr_pval_long,1), trial_corr_pearsonr_pval_long, s=80, facecolors='none', edgecolors=LONG_COLOR)
            ax5.set_title('trial corr r')
            ax5.set_ylim([-1,1])
            ax5.set_xlim([-1,2])

            ax6.scatter(np.full_like(trial_corr_pearsonp_pval_short,0), trial_corr_pearsonp_pval_short, s=80, facecolors='none', edgecolors=SHORT_COLOR)
            ax6.scatter(np.full_like(trial_corr_pearsonp_pval_long,1), trial_corr_pearsonp_pval_long, s=80, facecolors='none', edgecolors=LONG_COLOR)
            ax6.set_title('trial corr p')
            ax6.set_ylim([0,1])
            ax6.set_xlim([-1,2])

            plt.show()
        # ipdb.set_trace()

        #if there are fewer neuron for short trials: just take the first n=number of neurons on short trials of the long trials for comparison.
        num_short_neurons = mean_dF_short_coll_sort.shape[1]
        num_long_neurons = mean_dF_long_coll_sort.shape[1]
        if num_short_neurons < num_long_neurons:
            short_vec_rois = np.arange(num_short_neurons)
            # long_vec_rois = np.sort(np.random.choice(num_long_neurons, num_short_neurons, replace=False))
            long_vec_rois = np.arange(num_short_neurons)
        elif num_short_neurons > num_long_neurons:
            short_vec_rois = np.sort(np.random.choice(num_short_neurons, num_long_neurons, replace=False))
            long_vec_rois = np.arange(num_long_neurons)

        # create shuffled maps
        # popvec_ss_shuffled = np.zeros((popvec_cc_matrix_short_size,popvec_cc_matrix_short_size,100))
        # for k in range(100):
        #     vector_dimensions = np.random.permutation(np.arange(mean_dF_short_coll_sig.shape[1]))
        #     for row_shuffled in range(popvec_cc_matrix_short_size):
        #         for col_shuffled in range(popvec_cc_matrix_short_size):
        #             popvec_ss_shuffled[row_shuffled,col_shuffled,k] = stats.pearsonr(mean_dF_short_coll_sig[row_shuffled,vector_dimensions],mean_dF_short_coll_sort[col_shuffled,:])[0]
        #
        # popvec_ll_shuffled = np.zeros((popvec_cc_matrix_long_size,popvec_cc_matrix_long_size,100))
        # for k in range(100):
        #     vector_dimensions = np.random.permutation(np.arange(mean_dF_long_coll_sig.shape[1]))
        #     for row_shuffled in range(popvec_cc_matrix_long_size):
        #         for col_shuffled in range(popvec_cc_matrix_long_size):
        #             popvec_ll_shuffled[row_shuffled,col_shuffled,k] = stats.pearsonr(mean_dF_long_coll_sig[row_shuffled,vector_dimensions],mean_dF_long_coll_sort[col_shuffled,:])[0]

        for row in range(popvec_cc_matrix_short_size):
            for col in range(popvec_cc_matrix_short_size):
                # ss_corr_val = stats.pearsonr(mean_dF_short_coll_sig[row,:],mean_dF_short_coll_sort[col,:])[0]
                # ss_zscore = (ss_corr_val - np.mean(popvec_ss_shuffled[row,col,:]))/np.std(popvec_ss_shuffled[row,col,:])
                # if np.isinf(ss_zscore):
                #     print('INF!', str(popvec_ss_shuffled[row,col,:]), str(np.std(popvec_ss_shuffled[row,col,:])))

                popvec_cc_matrix_ss_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_short_coll_sig[row,:],mean_dF_short_coll_sort[col,:])[0]
                # popvec_cc_matrix_sl_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_short_coll_sig[row,:],mean_dF_long_coll_sig[col,:])[0]
                # pass


        # ipdb.set_trace()


        for row in range(popvec_cc_matrix_long_size):
            for col in range(popvec_cc_matrix_long_size):
                # ll_corr_val = stats.pearsonr(mean_dF_long_coll_sig[row,:],mean_dF_long_coll_sort[col,:])[0]
                # ll_zscore = (ll_corr_val - np.mean(popvec_ll_shuffled[row,col,:]))/np.std(popvec_ll_shuffled[row,col,:])
                popvec_cc_matrix_ll_pearsonr[row,col,current_iter] = stats.pearsonr(mean_dF_long_coll_sig[row,:],mean_dF_long_coll_sort[col,:])[0]

        # run through every row and find element with largest correlation coefficient
        for i,row in enumerate(range(popvec_cc_matrix_short_size)):
            popvec_cc_reconstruction_ss[0,i,current_iter] = i
            try:
                popvec_cc_reconstruction_ss[1,i,current_iter] = np.nanargmax(popvec_cc_matrix_ss_pearsonr[i,:,current_iter])
            except ValueError:
                popvec_cc_reconstruction_ss[1,i,current_iter] = 0

            popvec_cc_reconstruction_sl[0,i,current_iter] = i
            # try:
            #     popvec_cc_reconstruction_sl[1,i,current_iter] = np.nanargmax(popvec_cc_matrix_sl_pearsonr[i,:,current_iter])
            # except ValueError:
            #     print('WARNING: All-NaN slice encountered. Value set to 0')
            #     popvec_cc_reconstruction_sl[1,i,current_iter] = 0

        for i,row in enumerate(range(popvec_cc_matrix_long_size)):
            popvec_cc_reconstruction_ll[0,i,current_iter] = i
            try:
                popvec_cc_reconstruction_ll[1,i,current_iter] = np.nanargmax(popvec_cc_matrix_ll_pearsonr[i,:,current_iter])
            except ValueError:
                popvec_cc_reconstruction_ll[1,i,current_iter] = 0

        # calculate standard deviation (step by step so even an idiot like myself doesn't get confused) of reconstruction vs actual location
        bin_diff = popvec_cc_reconstruction_ss[1,:,current_iter] - popvec_cc_reconstruction_ss[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_ss = np.append(std_prelm_ss, np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_ss = np.append(std_lm_ss,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_ss = np.append(std_pi_ss,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))

        # bin_diff = popvec_cc_reconstruction_sl[1,:,current_iter] - popvec_cc_reconstruction_sl[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_sl = np.append(std_prelm_sl,np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_sl = np.append(std_lm_sl,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_sl = np.append(std_pi_sl,np.sqrt(np.sum(bin_diff[lm_end:end_bin_short])/(end_bin_short-lm_end)))

        bin_diff = popvec_cc_reconstruction_ll[1,:,current_iter] - popvec_cc_reconstruction_ll[0,:,current_iter]
        bin_diff = bin_diff * bin_diff
        std_prelm_ll = np.append(std_prelm_ll,np.sqrt(np.sum(bin_diff[0:lm_start])/(lm_start)))
        std_lm_ll = np.append(std_lm_ll,np.sqrt(np.sum(bin_diff[lm_start:lm_end])/(lm_end-lm_start)))
        std_pi_ll = np.append(std_pi_ll,np.sqrt(np.sum(bin_diff[lm_end:end_bin_long])/(end_bin_long-lm_end)))

    #print(np.mean(std_prelm_ss), np.mean(std_lm_ss), np.mean(std_pi_ss))

        # sort by peak activity (naming of variables confusing because the script grew organically...)
        mean_dF_sort_short = np.zeros(mean_dF_short_coll_sort.shape[1])
        for i, row in enumerate(mean_dF_short_coll_sort.T):
            mean_dF_sort_short[i] = np.nanargmax(row)
        histo_short[:,current_iter] = np.histogram(mean_dF_sort_short,np.arange(0,binnr_short+1,1))[0]

        mean_dF_sort_long = np.zeros(mean_dF_long_coll_sort.shape[1])
        for i, row in enumerate(mean_dF_long_coll_sort.T):
            mean_dF_sort_long[i] = np.nanargmax(row)
        histo_long[:,current_iter] = np.histogram(mean_dF_sort_long,np.arange(0,binnr_long+1,1))[0]

        print('--- Number of ROIs ---')
        print('short: ' + str(tot_rois_short))
        print('long: ' + str(tot_rois_long))
        print('----------------------')

    # perform statistical tests (one-way anova followed by pairwise tests).
    if NUM_ITER > 1:
        f_value_ss, p_value_ss = stats.f_oneway(std_prelm_ss,std_lm_ss,std_pi_ss)
        group_labels = ['prelm'] * len(std_prelm_ss) + ['lm'] * len(std_lm_ss) + ['pi'] * len(std_pi_ss)
        mc_res_ss = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ss,std_lm_ss,std_pi_ss)),group_labels)
        posthoc_res_ss = mc_res_ss.tukeyhsd()

        f_value_sl, p_value_sl = stats.f_oneway(std_prelm_sl,std_lm_sl,std_pi_sl)
        group_labels = ['prelm'] * len(std_prelm_sl) + ['lm'] * len(std_lm_sl) + ['pi'] * len(std_pi_sl)
        mc_res_sl = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_sl,std_lm_sl,std_pi_sl)),group_labels)
        posthoc_res_sl = mc_res_sl.tukeyhsd()

        f_value_ll, p_value_ll = stats.f_oneway(std_prelm_ll,std_lm_ll,std_pi_ll)
        group_labels = ['prelm'] * len(std_prelm_ll) + ['lm'] * len(std_lm_ll) + ['pi'] * len(std_pi_ll)
        mc_res_ll = sm.stats.multicomp.MultiComparison(np.concatenate((std_prelm_ll,std_lm_ll,std_pi_ll)),group_labels)
        posthoc_res_ll = mc_res_ll.tukeyhsd()
    else:
        posthoc_res_ss = np.nan
        posthoc_res_sl = np.nan
        posthoc_res_ll = np.nan
        f_value_ss = np.nan
        p_value_ss = np.nan
        f_value_ll = np.nan
        p_value_ll = np.nan


    # # calculate mean cc maps
    popvec_cc_matrix_ss_mean = np.nanmean(popvec_cc_matrix_ss_pearsonr,axis=2)
    popvec_cc_matrix_sl_mean = np.nanmean(popvec_cc_matrix_sl_pearsonr,axis=2)
    popvec_cc_matrix_ll_mean = np.nanmean(popvec_cc_matrix_ll_pearsonr,axis=2)
    #
    # calculate reconstructed location estimate from mean cc map
    popvec_cc_reconstruction_ss_mean = np.zeros((2,popvec_cc_matrix_short_size))
    popvec_cc_reconstruction_sl_mean = np.zeros((2,popvec_cc_matrix_short_size))
    popvec_cc_reconstruction_ll_mean = np.zeros((2,popvec_cc_matrix_long_size))

    for i in range(popvec_cc_matrix_short_size):
        popvec_cc_reconstruction_ss_mean[0,i] = i
        popvec_cc_reconstruction_ss_mean[1,i] = np.argmax(popvec_cc_matrix_ss_mean[i,:])

        popvec_cc_reconstruction_sl_mean[0,i] = i
        popvec_cc_reconstruction_sl_mean[1,i] = np.argmax(popvec_cc_matrix_sl_mean[i,:])

    for i in range(popvec_cc_matrix_long_size):
        popvec_cc_reconstruction_ll_mean[0,i] = i
        popvec_cc_reconstruction_ll_mean[1,i] = np.argmax(popvec_cc_matrix_ll_mean[i,:])

    # create figure and axes
    fig = plt.figure(figsize=(30,15))
    ax1 = plt.subplot2grid((5,200),(0,0),rowspan=4, colspan=45)
    ax2 = plt.subplot2grid((5,200),(0,50),rowspan=4, colspan=55)
    # ax3 = plt.subplot2grid((5,200),(0,0),rowspan=1, colspan=40)
    # ax4 = plt.subplot2grid((5,200),(0,50),rowspan=1, colspan=55)
    ax5 = plt.subplot2grid((5,200),(0,110),rowspan=2, colspan=40)
    ax6 = plt.subplot2grid((5,200),(2,110),rowspan=2, colspan=40)
    ax7 = plt.subplot2grid((5,200),(0,155),rowspan=2, colspan=40)
    ax8 = plt.subplot2grid((5,200),(2,155),rowspan=2, colspan=40)
    ax9 = plt.subplot2grid((5,200),(4,0),rowspan=1, colspan=45)
    ax10 = plt.subplot2grid((5,200),(4,50),rowspan=1, colspan=55)
    ax11 = plt.subplot2grid((5,200),(4,110),rowspan=1, colspan=40)
    ax12 = plt.subplot2grid((5,200),(4,155),rowspan=1, colspan=40)
    # ax6 = plt.subplot2grid((5,2),(4,1),rowspan=1)

    overlapping_rois_all.extend(rois_overlap[0])
    only_short_rois_all.extend(rois_overlap[1])
    only_long_rois_all.extend(rois_overlap[2])
    venn2_circles(subsets=(len(only_short_rois_all), len(only_long_rois_all), len(overlapping_rois_all)), ax=ax11)
    print(len(only_short_rois_all), len(only_long_rois_all), len(overlapping_rois_all))

    # ax7.pcolormesh(popvec_cc_matrix_sl)
    ax7_img = ax7.pcolor(popvec_cc_matrix_sl_mean.T,cmap='viridis')
    # plt.colorbar(ax7_img, ax=ax7)
    # # ax7.plot(popvec_cc_reconstruction_sl_mean[0,:],popvec_cc_reconstruction_sl_mean[1,:],c='r')
    # ax7.plot(popvec_cc_reconstruction_ll_mean[0,:],popvec_cc_reconstruction_ll_mean[0,:],c='k',ls='--')
    # ax7.axvline((200/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    # ax7.axvline((240/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    # ax7.axhline((200/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    # ax7.axhline((240/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    # ax7.set_xlabel('long')
    # ax7.set_ylabel('short')
    # ax7.set_xlim([0,popvec_cc_matrix_short_size])
    # ax7.set_ylim([0,popvec_cc_matrix_short_size])
    # ax7.set_xticklabels([])
    # ax7.set_yticklabels([])

    # ax5.pcolormesh(popvec_cc_matrix_ss)
    min_val = np.nanmin(np.nanmin(popvec_cc_matrix_ss_mean))
    max_val = np.nanmax(np.nanmax(popvec_cc_matrix_ss_mean))
    print(min_val,max_val)
    ax5_img = ax5.pcolor(popvec_cc_matrix_ss_mean.T,cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar(ax5_img, ax=ax5)
    # # ax5.plot(popvec_cc_reconstruction_ss_mean[0,:],popvec_cc_reconstruction_ss_mean[1,:],c='r')
    # ax5.plot(popvec_cc_reconstruction_ss_mean[0,:],popvec_cc_reconstruction_ss_mean[0,:],lw=4,c='k',ls='--')
    # ax5.axvline((200/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    # ax5.axvline((240/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    # ax5.axhline((200/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)
    # ax5.axhline((240/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["windows blue"],ls='--',lw=3)

    # ax5.axvline((102/BIN_SIZE)-start_bin,c='r',lw=3)
    # ax5.axhline((102/BIN_SIZE)-start_bin,c='r',lw=3)

    # ax5.axvline((220/BIN_SIZE)-start_bin,c='#39B54A',lw=4)
    # ax5.axhline((220/BIN_SIZE)-start_bin,c='#39B54A',lw=4)
    #
    # ax5.axvline((320/BIN_SIZE)-start_bin,c='#FF00FF',lw=4)
    # ax5.axhline((320/BIN_SIZE)-start_bin,c='#FF00FF',lw=4)

    ax5.set_xlabel('short')
    ax5.set_ylabel('short')
    ax5.set_xlim([0,popvec_cc_matrix_short_size])
    ax5.set_ylim([0,popvec_cc_matrix_short_size])


    # ax1.axvline((320/BIN_SIZE)-start_bin, lw=8, c='#FF00FF')

    # ax6.pcolormesh(popvec_cc_matrix_ll)
    min_val = np.nanmin(np.nanmin(popvec_cc_matrix_ll_mean))
    max_val = np.nanmax(np.nanmax(popvec_cc_matrix_ll_mean))
    print(min_val,max_val)
    ax6_img = ax6.pcolormesh(popvec_cc_matrix_ll_mean.T,cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar(ax6_img, ax=ax6)
    # # ax6.plot(popvec_cc_reconstruction_ll_mean[0,:],popvec_cc_reconstruction_ll_mean[1,:],c='r')
    # ax6.plot(popvec_cc_reconstruction_ll_mean[0,:],popvec_cc_reconstruction_ll_mean[0,:],lw=4,c='k',ls='--')
    # ax6.axvline((200/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    # ax6.axvline((240/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    # ax6.axhline((200/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)
    # ax6.axhline((240/BIN_SIZE)-start_bin,c=sns.xkcd_rgb["dusty purple"],ls='--',lw=3)

    # ax6.axvline((220/BIN_SIZE)-start_bin,c='#39B54A',lw=4)
    # ax6.axhline((220/BIN_SIZE)-start_bin,c='#39B54A',lw=4)
    #
    # ax6.axvline((380/BIN_SIZE)-start_bin,c='#FF00FF',lw=4)
    # ax6.axhline((380/BIN_SIZE)-start_bin,c='#FF00FF',lw=4)

    ax6.set_xlabel('long')
    ax6.set_ylabel('long')
    ax6.set_xlim([0,popvec_cc_matrix_long_size])
    ax6.set_ylim([0,popvec_cc_matrix_long_size])

    if align_point == 'trialonset':
        ax5.set_xticks([(0/BIN_SIZE),(100/BIN_SIZE),(200/BIN_SIZE)])
        ax5.set_yticks([(0/BIN_SIZE),(100/BIN_SIZE),(200/BIN_SIZE)])
        ax6.set_xticks([(0/BIN_SIZE),(100/BIN_SIZE),(200/BIN_SIZE)])
        ax6.set_yticks([(0/BIN_SIZE),(100/BIN_SIZE),(200/BIN_SIZE)])

        ax5.set_xticklabels(['0','100','200'])
        ax5.set_yticklabels(['0','100','200'])
        ax6.set_xticklabels(['0','100','200'])
        ax6.set_yticklabels(['0','100','200'])

    elif align_point == 'landmark':
        ax5.set_xticks([(200/BIN_SIZE)-start_bin,(240/BIN_SIZE)-start_bin,(320/BIN_SIZE)-start_bin])
        ax5.set_yticks([(200/BIN_SIZE)-start_bin,(240/BIN_SIZE)-start_bin,(320/BIN_SIZE)-start_bin])
        ax5.set_xticklabels(['200','240','320'])
        ax5.set_yticklabels(['200','240','320'])

        ax6.set_xticks([(200/BIN_SIZE)-start_bin,(240/BIN_SIZE)-start_bin,(380/BIN_SIZE)-start_bin])
        ax6.set_yticks([(200/BIN_SIZE)-start_bin,(240/BIN_SIZE)-start_bin,(380/BIN_SIZE)-start_bin])
        ax6.set_xticklabels(['200','240','380'])
        ax6.set_yticklabels(['200','240','380'])


    ax5.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=4, \
        left='on', \
        bottom='on', \
        right='on', \
        top='on')

    ax6.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=4, \
        left='on', \
        bottom='on', \
        right='on', \
        top='on')
#
    ax8.set_xlim([0,1])
    # ax8.bar([0.25,0.5,0.75],[np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)],-0.05, color=sns.xkcd_rgb["windows blue"],lw=0,yerr=[stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],ecolor='k')
    # ax8.bar([0.25,0.5,0.75],[np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], 0.05, color=sns.xkcd_rgb["dusty purple"],lw=0,yerr=[stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],ecolor='k')
    # ax8.bar([0.3,0.55,0.8],[np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], 0.05, color='0.5',lw=0,yerr=[stats.sem(std_prelm_sl),stats.sem(std_lm_sl),stats.sem(std_pi_sl)],ecolor='k')

    ax8.scatter([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], s=200,color=sns.xkcd_rgb["windows blue"], linewidths=0, zorder=2)


    ax8.scatter([0.27,0.52,0.77], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], s=200,color=sns.xkcd_rgb["dusty purple"], linewidths=0, zorder=2)
    # ax8.scatter([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], s=120,color='0.5', linewidths=0, zorder=2)

    ax8.errorbar([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], yerr=[stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],fmt='none',ecolor='k', zorder=1)
    ax8.errorbar([0.27,0.52,0.77], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], yerr=[stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],fmt='none',ecolor='k', zorder=1)
    # ax8.errorbar([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], yerr=[stats.sem(std_prelm_sl),stats.sem(std_lm_sl),stats.sem(std_pi_sl)],fmt='none',ecolor='k', zorder=1)

    ax8.plot([0.25,0.5,0.75], [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)], lw=2, c=sns.xkcd_rgb["windows blue"])
    ax8.plot([0.27,0.52,0.77], [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)], lw=2, c=sns.xkcd_rgb["dusty purple"])
    # ax8.plot([0.29,0.54,0.79], [np.mean(std_prelm_sl),np.mean(std_lm_sl),np.mean(std_pi_sl)], lw=2, c='0.5')

    ax8.tick_params(length=5,width=2,bottom=False,left=True,top=False,right=False,labelsize=14)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['bottom'].set_linewidth(2)
    ax8.spines['left'].set_linewidth(2)

    # ax8.set_ylim([0,5])

    ax8.set_xticks([0.27,0.52,0.77])
    ax8.set_xticklabels([], rotation=45, fontsize=20)
#
#     # add roi stats info
#     fig.text(0.1,0.1,'total number of rois: ' + str(tot_rois))
#
    # sort by peak activity (naming of variables confusing because the script grew organically...)
    mean_dF_sort_short = np.zeros(mean_dF_short_coll_sort.shape[1])
    for i, row in enumerate(mean_dF_short_coll_sort.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_short[i] = np.nanargmax(row)
        else:
            print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
    sort_ind_short = np.argsort(mean_dF_sort_short)

    plt.tight_layout()
    #start_bin=10
    #ax3.set_ylim([0,30])
    #ax3.set_xlim([start_bin,end_bin_short])
    #ax3.set_ylim([0,40])
#    sns.heatmap(np.transpose(mean_dF_short_coll[start_bin:end_bin_short, sort_ind_short]), cmap='jet', vmin=0.0, vmax=1.0, ax=ax1, cbar=False)

    mean_dF_sort_long = np.zeros(mean_dF_long_coll_sort.shape[1])
    for i, row in enumerate(mean_dF_long_coll_sort.T):
        if not np.all(np.isnan(row)):
            mean_dF_sort_long[i] = np.nanargmax(row)
        else:
            # print('WARNING: sort signal with all NaN encountered. ROI not plotted.')
            pass
    sort_ind_long = np.argsort(mean_dF_sort_long)
#
#     # sns.distplot(mean_dF_sort_short,color=sns.xkcd_rgb["windows blue"],bins=22,hist=True, kde=False,ax=ax3)
#     # sns.distplot(mean_dF_sort_long,color=sns.xkcd_rgb["dusty purple"],bins=28,hist=True, kde=False,ax=ax4)
#     #
#     # ax3.spines['top'].set_visible(False)
#     # ax3.spines['right'].set_visible(False)
#     # ax3.spines['left'].set_visible(False)
#     # ax4.spines['top'].set_visible(False)
#     # ax4.spines['right'].set_visible(False)
#     # ax4.spines['left'].set_visible(False)
#
    if sort_VR is None:
        if sortby == 'none':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,:]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'short':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'long':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
        elif sortby == 'both':
            sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_ind_short]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
            sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_ind_long]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)
    else:
        sns.heatmap(np.transpose(mean_dF_short_coll_sig[:,sort_VR[0]]), cmap='viridis', vmin=0.0, vmax=1, ax=ax1, cbar=False)
        sns.heatmap(np.transpose(mean_dF_long_coll_sig[:,sort_VR[1]]), cmap='viridis', vmin=0.0, vmax=1, ax=ax2, cbar=False)

    histo_short_smoothed = signal.savgol_filter(np.nanmean(histo_short,axis=1), 5, 3)
    histo_long_smoothed = signal.savgol_filter(np.nanmean(histo_long,axis=1), 5, 3)
    ax9.plot(histo_short_smoothed,c='k',lw=2)
    ax10.plot(histo_long_smoothed,c='k',lw=2)
    #
    # print(mean_dF_short_coll_sort)
    # print(mean_dF_short_coll_sort*5)
    # print((mean_dF_short_coll_sort*5)-220)

    # ax11.hist(np.absolute((mean_dF_sort_short*BIN_SIZE)-220))
    # ax12.hist(np.absolute((mean_dF_sort_long*BIN_SIZE)-220))

    if align_point == 'landmark':
        # ax1.axvline((102/BIN_SIZE)-start_bin, lw=8, c='r')
        ax1.axvline((220/BIN_SIZE)-start_bin, lw=8, c='#FF0000')
        ax1.axvline((320/BIN_SIZE)-start_bin, lw=8, c='#2EA7DF')
        # ax2.axvline((102/BIN_SIZE)-start_bin, lw=8, c='r')
        ax2.axvline((220/BIN_SIZE)-start_bin, lw=8, c='#FF0000')
        ax2.axvline((380/BIN_SIZE)-start_bin, lw=8, c='#2EA7DF')

    if align_point == 'trialonset':
        ax1.axvline(0/BIN_SIZE, lw=8, c='#39B54A')
        ax2.axvline(0/BIN_SIZE, lw=8, c='#39B54A')

    # print(roi_selection)
    # ax1.set_yticklabels([])
    # ax2.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax2.set_xticklabels([])

    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')

    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='off', \
        bottom='off', \
        right='off', \
        top='off')
#
#     #ax3.set_xlim([0,23])
#     #ax4.set_xlim([0,35])
#     #ax4.set_xlim([0, end_bin_long-start_bin+1])
#
#     #fig.suptitle(fname + '_' + trials + '_' + str([''.join(str(r) for r in ri) for ri in rec_info]),wrap=True)
#
# #    plt.tight_layout()
#     # list_test = popvec_cc_reconstruction_ss[0,:,]
#     # print(len(popvec_cc_reconstruction_ss.tolist()[0]),len(popvec_cc_reconstruction_ss.tolist()[1]))
    if NUM_ITER > 1:
        popvloc_results = {
            'included_datasets' : roi_param_list,
            'short_short' : [np.mean(std_prelm_ss),np.mean(std_lm_ss),np.mean(std_pi_ss)],
            'long_long' : [np.mean(std_prelm_ll),np.mean(std_lm_ll),np.mean(std_pi_ll)],
            'short_short_SEM' : [stats.sem(std_prelm_ss),stats.sem(std_lm_ss),stats.sem(std_pi_ss)],
            'long_long_SEM' : [stats.sem(std_prelm_ll),stats.sem(std_lm_ll),stats.sem(std_pi_ll)],
            'short_short_f_value' : f_value_ss,
            'short_short_p_value' : p_value_ss,
            'short_short_multicomparison' : [posthoc_res_ss.reject[0].item(),posthoc_res_ss.reject[1].item(),posthoc_res_ss.reject[2].item()],
            'long_long_f_value' : f_value_ll,
            'long_long_p_value' : p_value_ll,
            'long_long_multicomparison' : [posthoc_res_ll.reject[0].item(),posthoc_res_ll.reject[1].item(),posthoc_res_ll.reject[2].item()],
            'iterations:' : [NUM_ITER],
            'popvec_cc_reconstruction_ss' : popvec_cc_reconstruction_ss.tolist(),
            'popvec_cc_reconstruction_ll' : popvec_cc_reconstruction_ll.tolist(),
            'popvec_cc_matrix_ss_mean' : popvec_cc_matrix_ss_mean.tolist(),
            'popvec_cc_matrix_ll_mean' : popvec_cc_matrix_ll_mean.tolist(),
            'popvec_zscore_matrix_ss_mean' : popvec_cc_matrix_ss_mean.tolist(),
            'popvec_zscore_matrix_ll_mean' : popvec_cc_matrix_ll_mean.tolist()
        }
    else:
        popvloc_results = {}
#


    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)

    if write_to_dict:
        with open(loc_info['figure_output_path'] + 'popvloc' + os.sep + fname + '_popvloc_results_' + str(NUM_ITER) + '.json','w') as f:
            json.dump(popvloc_results,f)
            print(loc_info['figure_output_path'] + 'popvloc' + os.sep + fname + '_popvloc_results_' + str(NUM_ITER) + '.json')

    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '_' + str(NUM_ITER) + '.' + fformat
    print(fname)

    fig.savefig(fname, format=fformat)

    return sort_ind_short, sort_ind_long

if __name__ == "__main__":
    # all
    suffix = ''
    write_to_dict = True

    # track paramaters (cm)
    reward_distance = 10

    # TEST: match ROI IDs exactly (rather than by column)
    roi_param_list = [
                      [[loc_info['figure_output_path'] + 'LF191022_3' + os.sep + 'LF191022_3_20191119' + suffix + '.json','LF191022_3','20191119', np.arange(0,136)], [loc_info['figure_output_path'] + 'LF191022_3' + os.sep + 'LF191022_3_20191204' + suffix + '.json','LF191022_3','20191204', np.arange(0,136)]],
                     ]

    # fig_popvloc_naiveexpert(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_naiveexpert_union', 'popvloc', write_to_dict, None, 'expert_naive')

    roi_param_list = [
                      [[loc_info['figure_output_path'] + 'LF191022_3' + os.sep + 'LF191022_3_20191119' + suffix + '.json','LF191022_3','20191119', np.arange(0,136)], [loc_info['figure_output_path'] + 'LF191022_3' + os.sep + 'LF191022_3_20191204' + suffix + '.json','LF191022_3','20191204', np.arange(0,136)]],
                      [[loc_info['figure_output_path'] + 'LF191023_blue' + os.sep + 'LF191023_blue_20191119' + suffix + '.json','LF191023_blue','20191119',np.arange(0,134)],[loc_info['figure_output_path'] + 'LF191023_blue' + os.sep + 'LF191023_blue_20191204' + suffix + '.json','LF191023_blue','20191204',np.arange(0,134)]],
                      [[loc_info['figure_output_path'] + 'LF191024_1' + os.sep + 'LF191024_1_20191115' + suffix + '.json','LF191024_1','20191115', np.arange(0,81)], [loc_info['figure_output_path'] + 'LF191024_1' + os.sep + 'LF191024_1_20191204' + suffix + '.json','LF191024_1','20191204', np.arange(0,81)]]
                     ]

    # fig_popvloc_naiveexpert(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_naiveexpert_union', 'popvloc', write_to_dict, None, 'expert_naive')
    # fig_popvloc_naiveexpert(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_naiveexpert_union_expexp', 'popvloc', write_to_dict, None, 'expert_expert')
    # fig_popvloc_naiveexpert(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_naiveexpert_union_nainai', 'popvloc', write_to_dict, None, 'naive_naive')

    roi_param_list = [
                      [loc_info['figure_output_path'] + 'LF191022_3' + os.sep + 'LF191022_3_20191204' + suffix + '.json','LF191022_3','20191204'],
                      [loc_info['figure_output_path'] + 'LF191023_blue' + os.sep + 'LF191023_blue_20191204' + suffix + '.json','LF191023_blue','20191204'],
                      [loc_info['figure_output_path'] + 'LF191024_1' + os.sep + 'LF191024_1_20191204' + suffix + '.json','LF191024_1','20191204']
                     ]

    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_matchedexpert', 'popvloc', write_to_dict)

    roi_param_list = [
                      [loc_info['figure_output_path'] + 'LF191022_3' + os.sep + 'LF191022_3_20191119' + suffix + '.json','LF191022_3','20191119'],
                      [loc_info['figure_output_path'] + 'LF191023_blue' + os.sep + 'LF191023_blue_20191119' + suffix + '.json','LF191023_blue','20191119'],
                      [loc_info['figure_output_path'] + 'LF191024_1' + os.sep + 'LF191024_1_20191115' + suffix + '.json','LF191024_1','20191115']
                     ]

    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_matchednaive', 'popvloc', write_to_dict)


    roi_param_list = [
                      [loc_info['figure_output_path'] + 'LF191022_3' + os.sep + 'LF191022_3_20191119' + suffix + '.json','LF191022_3','20191119'],
                      [loc_info['figure_output_path'] + 'LF191023_blue' + os.sep + 'LF191023_blue_20191119' + suffix + '.json','LF191023_blue','20191119'],
                      [loc_info['figure_output_path'] + 'LF191024_1' + os.sep + 'LF191024_1_20191115' + suffix + '.json','LF191024_1','20191115']
                      # [loc_info['figure_output_path'] + 'LF191023_blank' + os.sep + 'LF191023_blank_20191116' + suffix + '.json','LF191023_blank','20191116'],
                      # [loc_info['figure_output_path'] + 'LF191022_2' + os.sep + 'LF191022_2_20191116' + suffix + '.json','LF191022_2','20191116']
                     ]

    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_naive', 'popvloc', write_to_dict)

    roi_param_list = [
                      [loc_info['figure_output_path'] + 'LF170613_1' + os.sep + 'LF170613_1_Day20170804' + suffix + '.json','LF170613_1','Day20170804'],
                      [loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day2017719' + suffix + '.json','LF170420_1','Day2017719'],
                      [loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day201783' + suffix + '.json','LF170420_1','Day201783'],
                      [loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day20170719' + suffix + '.json','LF170421_2','Day20170719'],
                      # [loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day2017720' + suffix + '.json','LF170421_2','Day2017720'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_1' + suffix + '.json','LF170110_2','Day201748_1'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_2' + suffix + '.json','LF170110_2','Day201748_2'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_3' + suffix + '.json','LF170110_2','Day201748_3'],
                      [loc_info['figure_output_path'] + 'LF170222_1' + os.sep + 'LF170222_1_Day201776' + suffix + '.json','LF170222_1','Day201776'],
                      [loc_info['figure_output_path'] + 'LF171212_2' + os.sep + 'LF171212_2_Day2018218_2' + suffix + '.json','LF171212_2','Day2018218_2'],
                      [loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l23' + suffix + '.json','LF161202_1','Day20170209_l23'],
                      [loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l5' + suffix + '.json','LF161202_1','Day20170209_l5']
                     ]
                     # [loc_info['figure_output_path'] + 'LF171211_1' + os.sep + 'LF171211_1_Day2018321_2' + suffix + '.json','LF171211_1','Day2018321_2']
    #
    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark_expert', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'trialonset', 'all_onset_expert', 'popvloc', write_to_dict)

    fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'short', True, 'landmark', 'all_landmark_short', 'popvloc', write_to_dict)
    fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'long', True, 'landmark', 'all_landmark_long', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list, ['trialonset'], reward_distance, 'both', 'both', False, 'landmark', 'landmark_trialonset', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list, ['lmcenter'], reward_distance, 'both', 'both', False, 'landmark', 'landmark_lmcenter', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list, ['reward'], reward_distance, 'both', 'both', False, 'landmark', 'landmark_reward', 'popvloc', write_to_dict)

    # fig_popvloc_s5(roi_param_list, ['trialonset'], reward_distance, 'both', 'both', True, 'trialonset', 'trialonset_trialonset', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list, ['lmcenter'], reward_distance, 'both', 'both', True, 'trialonset', 'trialonset_lmcenter', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list, ['reward'], reward_distance, 'both', 'both', True, 'trialonset', 'trialonset_reward', 'popvloc', write_to_dict)

    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'short', True, 'landmark', 'active_all_short', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'long', True, 'landmark', 'active_all_long', 'popvloc', write_to_dict)

    # SPECIAL CASE get the sorting index for short and long trials so we can use it for openloop. NOTE: SESSIONS INCLUDED MUST MATCH THE OPENLOOP ROI PARAM LIST
    # sort_short, sort_long = fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'all_landmark', 'popvloc', False)

    roi_param_list_openloop = [
                      [loc_info['figure_output_path'] + 'LF170613_1' + os.sep + 'LF170613_1_Day20170804' + suffix + '.json','LF170613_1','Day20170804_openloop'],
                      [loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day2017719' + suffix + '.json','LF170420_1','Day2017719_openloop'],
                      [loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day201783' + suffix + '.json','LF170420_1','Day201783_openloop'],
                      [loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day20170719' + suffix + '.json','LF170421_2','Day20170719_openloop'],
                      # [loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day2017720' + suffix + '.json','LF170421_2','Day2017720_openloop'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_1' + suffix + '.json','LF170110_2','Day201748_openloop_1'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_2' + suffix + '.json','LF170110_2','Day201748_openloop_2'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_3' + suffix + '.json','LF170110_2','Day201748_openloop_3'],
                      [loc_info['figure_output_path'] + 'LF170222_1' + os.sep + 'LF170222_1_Day201776' + suffix + '.json','LF170222_1','Day201776_openloop'],
                      [loc_info['figure_output_path'] + 'LF171212_2' + os.sep + 'LF171212_2_Day2018218_2' + suffix + '.json','LF171212_2','Day2018218_openloop_2'],
                      # [loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l23' + suffix + '.json','LF161202_1','Day20170209_l23'],
                      # [loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l5' + suffix + '.json','LF161202_1','Day20170209_l5']
                      # [loc_info['figure_output_path'] + 'LF171211_1' + os.sep + 'LF171211_1_Day2018321_2' + suffix + '.json','LF171211_1','Day2018321_2']
                     ]

    # NUM_ITER = 100
    # fig_popvloc_s5(roi_param_list_openloop, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'active_all_openloop', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list_openloop, ['trialonset'], reward_distance, 'both', 'both', True, 'landmark', 'trialonset_openloop', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list_openloop, ['lmcenter'], reward_distance, 'both', 'both', True, 'landmark', 'lmcenter_openloop', 'popvloc', write_to_dict)
    # fig_popvloc_s5(roi_param_list_openloop, ['reward'], reward_distance, 'both', 'both', True, 'landmark', 'reward_openloop', 'popvloc', write_to_dict)

    roi_param_list = [
                      [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day201777.json','LF170214_1','Day201777'],
                      [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day2017714.json','LF170214_1','Day2017714'],
                      [loc_info['figure_output_path'] + 'LF171211_2' + os.sep + 'LF171211_2_Day201852.json','LF171211_2','Day201852'],
                      [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_1'],
                      [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_2'],
                      [loc_info['figure_output_path'] + 'LF180219_1' + os.sep + 'LF180219_1_Day2018424_0025.json','LF180219_1','Day2018424_0025']
                     ]

    # fig_popvloc_s5(roi_param_list, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'active_v1', 'popvloc', write_to_dict)

    roi_param_list_openloop = [
                      [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day201777.json','LF170214_1','Day201777_openloop'],
                      [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day2017714.json','LF170214_1','Day2017714_openloop'],
                      [loc_info['figure_output_path'] + 'LF171211_2' + os.sep + 'LF171211_2_Day201852.json','LF171211_2','Day201852_openloop'],
                      [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_openloop_1'],
                      [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_openloop_2'],
                      [loc_info['figure_output_path'] + 'LF180219_1' + os.sep + 'LF180219_1_Day2018424_0025.json','LF180219_1','Day2018424_openloop_0025']
                     ]

    # fig_popvloc_s5(roi_param_list_openloop, ['trialonset','lmcenter','reward'], reward_distance, 'both', 'both', True, 'landmark', 'active_v1_openloop', 'popvloc', write_to_dict)
