"""
Template matching classifier: calculate how well we can identify the current trial type from the population of landmark neurons

@author: lukasfischer


"""

# load local settings file
import matplotlib, sys, yaml, h5py, json
import numpy as np
import warnings; warnings.simplefilter('ignore')
sys.path.append("./Analysis")
from analysis_parameters import MIN_FRACTION_ACTIVE, MIN_ZSCORE, MIN_MEAN_AMP
# from analysis_parameters import MIN_MEAN_AMP_BOUTONS as MIN_MEAN_AMP
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from filter_trials import filter_trials
from scipy import stats, signal, io
import statsmodels.api as sm
import statsmodels as sm_all
import seaborn as sns
import ipdb
from write_dict import write_dict
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
            loc_info = yaml.load(f)

# execution parameters
fformat = 'png'
write_to_dict = False


# stdev treshold above which roi response has to be in a given trial to be considered for determining its receptive field center
trial_std_threshold = 3

# set binning
BIN_SIZE = 5
TRACKLENGTH_SHORT = 400
TRACKLENGTH_LONG = 500
binnr_short = int(TRACKLENGTH_SHORT/BIN_SIZE)
binnr_long = int(TRACKLENGTH_LONG/BIN_SIZE)

# track numbers
TRACK_SHORT = 3
TRACK_LONG = 4

# amount of blackbox after a reward to be added to each trial
PRE_TRIAL_TIME = 1.5
REWARD_TIME = 1.5

# fraction of trials that have to be present for a given location to be included in the mean trace
MEAN_TRACE_FRACTION = 0.5

# colors for short and long trial traces
SHORT_COLOR = '#FF8000'
LONG_COLOR = '#0025D0'

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

    return result_max_peak[align_event + '_roi_number_' + peak_trialtype]

def return_valid_rois(roi_params, trial_type, align_point):

    lm_rois = get_eventaligned_rois(roi_params, [trial_type], align_point)

    roi_selection = np.array(roi_params['valid_rois'])
    roi_mean_trace = np.array(roi_params['space_mean_trace_'+trial_type])
    roi_active_lmcenter = np.array(roi_params['lmcenter_active_'+trial_type])
    roi_zscore_lmcenter = np.array(roi_params['lmcenter_peak_zscore_'+trial_type])

    zscore_roi = roi_zscore_lmcenter >= MIN_ZSCORE
    active_roi = roi_active_lmcenter >= MIN_FRACTION_ACTIVE
    pass_rois = np.logical_and(zscore_roi, active_roi)

    # ipdb.set_trace()
    for i in range(len(roi_selection)):
            if (np.nanmax(roi_mean_trace[i]) - np.nanmin(roi_mean_trace[i])) < MIN_MEAN_AMP:
                pass_rois[i] = False

    pass_rois = np.where(pass_rois==False)[0]
    roi_selection = np.delete(roi_selection, pass_rois)
    roi_selection = np.intersect1d(roi_selection, lm_rois)

    return roi_selection


def calc_spacevloc(behav_ds, roi_dF, included_trials, align_point, binnr, speed_col, maxlength):
    # this may be counterintuitive, but when we align by landmark we don't care too much about the very first bin, aligned to trialonset through
    # we want to skip it to avoid artifactual values at the VERY beginning
    if align_point == 'trialonset':
        start_loc = -10
    else:
        start_loc = 0

    bin_edges = np.linspace(start_loc, maxlength+start_loc, binnr+1)

    # intilize matrix to hold data for all trials of this roi
    mean_dF_trials = np.zeros((np.size(included_trials,0),binnr))
    mean_speed_trials = np.zeros((np.size(included_trials,0),binnr))
    # calculate mean dF vs location for each ROI on short trials
    for k,t in enumerate(included_trials):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        # get indeces for the first <REWARD_TIME> sec after a reward
        cur_trial_rew_loc_idx = np.where((behav_ds[:,0] > behav_ds[behav_ds[:,6]==t,0][-1]) & (behav_ds[:,0] < (behav_ds[behav_ds[:,6]==t,0][-1]+REWARD_TIME)))[0]
        cur_trial_rew_loc = behav_ds[cur_trial_rew_loc_idx,1]

        # get indeces for <PRE_TRIAL_TIME> sec prior to trial onset
        cur_trial_pretrial_loc_idx = np.where((behav_ds[:,0] < behav_ds[behav_ds[:,6]==t,0][0]) & (behav_ds[:,0] > (behav_ds[behav_ds[:,6]==t,0][0]-PRE_TRIAL_TIME)))[0]
        cur_trial_pretrial_loc = behav_ds[cur_trial_pretrial_loc_idx,1]

        # subtract starting location from location samples to align all to trial start
        if align_point == 'trialonset':
            cur_trial_rew_loc = cur_trial_rew_loc - cur_trial_loc[0]
            cur_trial_loc = cur_trial_loc - cur_trial_loc[0]

        if np.size(cur_trial_pretrial_loc) > 0:
            if align_point == 'landmark':
                cur_trial_pretrial_loc = cur_trial_pretrial_loc - cur_trial_pretrial_loc[-1] + cur_trial_loc[0]

            elif align_point == 'trialonset':
                cur_trial_pretrial_loc = cur_trial_pretrial_loc - cur_trial_pretrial_loc[-1]

        cur_trial_dF_roi = roi_dF[behav_ds[:,6]==t]
        cur_trial_speed = behav_ds[behav_ds[:,6]==t,speed_col]

        if np.size(cur_trial_rew_loc) > 0:
            cur_trial_loc = np.append(cur_trial_loc, cur_trial_rew_loc)
            cur_trial_dF_roi = np.append(cur_trial_dF_roi, roi_dF[cur_trial_rew_loc_idx])
            cur_trial_speed = np.append(cur_trial_speed, behav_ds[cur_trial_rew_loc_idx,speed_col])

        if np.size(cur_trial_pretrial_loc) > 0:
            cur_trial_loc = np.insert(cur_trial_loc, 0, cur_trial_pretrial_loc)
            cur_trial_dF_roi = np.insert(cur_trial_dF_roi, 0, roi_dF[cur_trial_pretrial_loc_idx])
            cur_trial_speed = np.insert(cur_trial_speed, 0, behav_ds[cur_trial_pretrial_loc_idx,speed_col])

        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', bin_edges, (start_loc, maxlength+start_loc))[0]
        mean_speed_trial = stats.binned_statistic(cur_trial_loc, cur_trial_speed, 'mean', bin_edges, (start_loc, maxlength+start_loc))[0]
        # mean_dF_trial /= np.nanmax(np.abs(mean_dF_trial[start_bin:end_bin_short]))
        mean_dF_trials[k,:] = mean_dF_trial
        mean_speed_trials[k,:] = mean_speed_trial
        #print(mean_dF_trial)

    return mean_dF_trials, mean_speed_trials

def get_trial_dF(behav_ds, dF_ds, landmark_rois, trials, binnr, tracklength, align_point):
    """ calculate Rtheta """
    speed_col = 3
    # hold index of mean trace peak
    meanpeak_idx = np.full(len(landmark_rois),np.nan)
    # hold all dimensions of the Rtheta_short vector (before calculating its norm)
    Rtheta_vec = np.full(len(landmark_rois), np.nan)
    # create matrix that will hold data of all spatial bins for all trials of all neurons
    trial_dF = np.full((len(trials),binnr,len(landmark_rois)), np.nan)
    # loop through each roi to get trial-by-trial data, binned by space
    for i,sr in enumerate(landmark_rois):
        roi_dF = dF_ds[:,sr]
        trial_dF[:,:,i],_ = calc_spacevloc(behav_ds, roi_dF, trials, align_point, binnr, speed_col, tracklength)
        # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot and find peak response value
        mean_valid_indices = []
        for j,trace in enumerate(trial_dF[:,:,i].T):
            if np.count_nonzero(np.isnan(trace))/trace.shape[0] < MEAN_TRACE_FRACTION:
                mean_valid_indices.append(j)
        mean_trace = np.nanmean(trial_dF[:,mean_valid_indices[0]:mean_valid_indices[-1],i],0)
        # append value of current roi to Rtheta
        Rtheta_vec[i] = mean_trace[np.nanargmax(mean_trace).astype(int)]
        # store meanpeak value (add offset of where mean_trace starts so its valid for individual trials)
        meanpeak_idx[i] = mean_valid_indices[0] + np.nanargmax(mean_trace)
    # calculate vector norm of template vector
    Rtheta_vec_norm = np.linalg.norm(Rtheta_vec)

    return Rtheta_vec_norm, Rtheta_vec, trial_dF, meanpeak_idx

def calc_similarity_index(landmark_rois, trial_dF, meanpeak_idx, Rtheta, Rtheta_vec):
    """ calculate the similarity index for a given landmark """
    sim_idx = np.full(trial_dF.shape[0],np.nan)
    # loop through every trial to calculate the template matching value
    for i in range(trial_dF.shape[0]):
        # holds the value for the top half of our calculation
        stim_theta = 0
        # hold all Rstim values
        Ristim_roi_values = np.full(len(landmark_rois), np.nan)

        for j in range(len(landmark_rois)):
            # print(meanpeak_idx[j], i, j, trial_dF[i,meanpeak_i    dx[j].astype(int),j])
            Ristim_roi_values[j] = trial_dF[i,meanpeak_idx[j].astype(int),j]


            # safeguard to avoid spurious negative values having a disproportionate impact on the calculation
            if Ristim_roi_values[j] < 0:
                Ristim_roi_values[j] = 0
            Ritheta = Rtheta_vec[j]
            if not np.isnan(Ristim_roi_values[j]):
                stim_theta += Ristim_roi_values[j] * Ritheta
            else:
                Ristim_roi_values[j] = 0
        # calculate vector norm for a given trial
        Rstim_norm = np.linalg.norm(Ristim_roi_values)
        # return the similiarity index
        sim_idx[i] = stim_theta / (Rtheta * Rstim_norm)
    return sim_idx


def fig_template_decoding(roi_param_list, align_point, fformat='png', fname='', subfolder='', write_to_dict=False, load_raw=False):
    """ main function to calculate similarity index """

    print('### Align point: ' + align_point + ' ###')
    correct_decoding = np.full(len(roi_param_list),np.nan)
    num_neurons = np.zeros(len(roi_param_list))
    for k,rpm in enumerate(roi_param_list):
        print(rpm)
        # load data from HDF5 file
        mouse = rpm[1]
        session = rpm[2]

        if load_raw == False:
            h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
            h5dat = h5py.File(h5path, 'r')
            behav_ds = np.copy(h5dat[session + '/behaviour_aligned'])
            dF_ds = np.copy(h5dat[session + '/dF_win'])
            h5dat.close()
        else:
            processed_data_path = loc_info['raw_dir'] + mouse + os.sep + session + os.sep + 'aligned_data.mat'
            loaded_data = io.loadmat(processed_data_path)
            behav_ds = loaded_data['behaviour_aligned']
            dF_ds = loaded_data['dF_aligned']

        # get trialnumbers for short and long trials
        short_trials = filter_trials(behav_ds, [], ['tracknumber',TRACK_SHORT])
        long_trials = filter_trials(behav_ds, [], ['tracknumber',TRACK_LONG])
        # speed_col = 3

        # load .json data
        with open(rpm[0]) as f:
            roi_params = json.load(f)

        # get landmark rois for short and long trials
        short_rois = return_valid_rois(roi_params, 'short', align_point)
        long_rois = return_valid_rois(roi_params, 'long', align_point)
        landmark_rois = np.union1d(short_rois,long_rois)

        # keep track of number of neurons per animal
        num_neurons[k] = len(landmark_rois)

        # get Rtheta vector, its vector norm and the dF/F data for every trial
        Rtheta_norm_short, Rtheta_vec_short, trial_dF_short, meanpeak_idx_short = get_trial_dF(behav_ds, dF_ds, landmark_rois, short_trials, binnr_short, TRACKLENGTH_SHORT, align_point)
        Rtheta_norm_long, Rtheta_vec_long, trial_dF_long, meanpeak_idx_long = get_trial_dF(behav_ds, dF_ds, landmark_rois, long_trials, binnr_long, TRACKLENGTH_LONG, align_point)

        # get the similarity index for each trial type
        similiarity_index_ss = calc_similarity_index(landmark_rois, trial_dF_short, meanpeak_idx_short, Rtheta_norm_short, Rtheta_vec_short)
        similiarity_index_sl = calc_similarity_index(landmark_rois, trial_dF_short, meanpeak_idx_short, Rtheta_norm_long, Rtheta_vec_long)
        similiarity_index_ll = calc_similarity_index(landmark_rois, trial_dF_long, meanpeak_idx_long, Rtheta_norm_long, Rtheta_vec_long)
        similiarity_index_ls = calc_similarity_index(landmark_rois, trial_dF_long, meanpeak_idx_long, Rtheta_norm_short, Rtheta_vec_short)

        # calculate fraction of correctly decoded trials
        # print(similiarity_index_ss)
        # print(similiarity_index_sl)
        correct_decoding[k] = (np.sum(similiarity_index_ss > similiarity_index_sl)+np.sum(similiarity_index_ll > similiarity_index_ls))/(len(similiarity_index_ss)+len(similiarity_index_ll))
        # print(correct_decoding[k])

    # calculate means per animal

    if fname is 'rsc':
        correct_decoding_grouped = [correct_decoding[0], np.mean(correct_decoding[1:3]), np.mean(correct_decoding[3:5]), np.mean(correct_decoding[5:8]), np.mean(correct_decoding[8]), np.mean(correct_decoding[9]), np.mean(correct_decoding[10:12])]
        num_neurons_grouped = [num_neurons[0], np.sum(num_neurons[1:3]), np.sum(num_neurons[3:5]), np.sum(num_neurons[5:8]), np.sum(num_neurons[8]), np.sum(num_neurons[9]), np.sum(num_neurons[10:12])]
    if fname is 'rsc_2':
        correct_decoding_grouped = [correct_decoding[0], np.mean(correct_decoding[1:3]), np.mean(correct_decoding[3]), np.mean(correct_decoding[4:7]), np.mean(correct_decoding[7]), np.mean(correct_decoding[8]), np.mean(correct_decoding[9:11])]
        num_neurons_grouped = [num_neurons[0], np.sum(num_neurons[1:3]), np.sum(num_neurons[3]), np.sum(num_neurons[4:7]), np.sum(num_neurons[7]), np.sum(num_neurons[8]), np.sum(num_neurons[9:11])]
    elif fname is 'v1':
        correct_decoding_grouped = [np.mean(correct_decoding[0:2]), correct_decoding[2], np.mean(correct_decoding[3:5]), correct_decoding[5]]
        num_neurons_grouped = [np.sum(num_neurons[0:2]), num_neurons[2], np.sum(num_neurons[3:5]), num_neurons[5]]
    elif fname is 'naive':
        correct_decoding_grouped = [np.mean(correct_decoding[0]), correct_decoding[1], np.mean(correct_decoding[2]), correct_decoding[3]]
        num_neurons_grouped = [np.sum(num_neurons[0]), num_neurons[1], np.sum(num_neurons[2]), num_neurons[3]]
    # print(correct_decoding_grouped)
    # create figure to later plot on
    # print(correct_decoding_grouped)
    # fig = plt.figure(figsize=(4,4))
    # ax1 = plt.subplot(111)
    # ax1.scatter(np.zeros(len(correct_decoding_grouped)),correct_decoding_grouped)
    # ax1.set_ylim([0,1])
    # # ax1.axvline((roi_meanpeak_short_idx+mean_valid_indices[0]),c='b')
    # fname = loc_info['figure_output_path'] + 'testplots' + os.sep + fname + '_' + align_point + '.' + fformat
    # print(fname)
    # fig.savefig(fname, format=fformat)

    return correct_decoding_grouped, num_neurons_grouped


if __name__ == "__main__":
    # Slightly more involved across trials
    # 1) get all landmark neurons of a given recording
    # 2) get mean peak responses for all neurons for  and long trials (each creating one population template vector)
    # 3) get responses for each trial of each neuron
    # 4) Calculate vector norm for vector consisting of all neuron templates (Rtheta) (each dimension being the response of one neuron) for short and long trials
    # 5) Calculate vector norm for vector consisting of all neuron responses in a given trial (Rstim) (each dimension being the response in a given trial)
    # 6) Calculate sum (template response of a neuron times trial response of a neuron) for all neurons
    # 7) Divide (6) by ((4) times (5))
    # 8) Repeat for each stimulus condition (/long)
    # 9) Repeat for every trial
    # 10) calculate the fraction of correctly decoded trials based on which got the higher similarity score

    load_raw = False
    roi_param_list = [
                      [loc_info['figure_output_path'] + 'LF170613_1' + os.sep + 'LF170613_1_Day20170804.json','LF170613_1','Day20170804'],
                      [loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day2017719.json','LF170420_1','Day2017719'],
                      [loc_info['figure_output_path'] + 'LF170420_1' + os.sep + 'LF170420_1_Day201783.json','LF170420_1','Day201783'],
                      [loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day20170719.json','LF170421_2','Day20170719'],
                      # [loc_info['figure_output_path'] + 'LF170421_2' + os.sep + 'LF170421_2_Day2017720.json','LF170421_2','Day2017720'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_1.json','LF170110_2','Day201748_1'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_2.json','LF170110_2','Day201748_2'],
                      [loc_info['figure_output_path'] + 'LF170110_2' + os.sep + 'LF170110_2_Day201748_3.json','LF170110_2','Day201748_3'],
                      [loc_info['figure_output_path'] + 'LF170222_1' + os.sep + 'LF170222_1_Day201776.json','LF170222_1','Day201776'],
                      [loc_info['figure_output_path'] + 'LF171212_2' + os.sep + 'LF171212_2_Day2018218_2.json','LF171212_2','Day2018218_2'],
                      [loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l23.json','LF161202_1','Day20170209_l23'],
                      [loc_info['figure_output_path'] + 'LF161202_1' + os.sep + 'LF161202_1_Day20170209_l5.json','LF161202_1','Day20170209_l5']
                     ]

    decoder_to, n_neurons_to = fig_template_decoding(roi_param_list, 'trialonset', fformat, 'rsc_2', 'testplots', write_to_dict, load_raw)
    decoder_lm, n_neurons_lm = fig_template_decoding(roi_param_list, 'lmcenter', fformat, 'rsc_2', 'testplots', write_to_dict, load_raw)
    decoder_rw, n_neurons_rw = fig_template_decoding(roi_param_list, 'reward', fformat, 'rsc_2', 'testplots', write_to_dict, load_raw)

    # roi_param_list = [
    #                   [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day201777.json','LF170214_1','Day201777'],
    #                   [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day2017714.json','LF170214_1','Day2017714'],
    #                   [loc_info['figure_output_path'] + 'LF171211_2' + os.sep + 'LF171211_2_Day201852.json','LF171211_2','Day201852'],
    #                   [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_1'],
    #                   [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_2'],
    #                   [loc_info['figure_output_path'] + 'LF180219_1' + os.sep + 'LF180219_1_Day2018424_0025.json','LF180219_1','Day2018424_0025']
    #                  ]

    # load_raw = True
    # roi_param_list = [
    #                   ['E:\\MTH3_figures\\LF191022_3\\LF191022_3_20191119.json','LF191022_3','20191119'],
    #                   ['E:\\MTH3_figures\\LF191023_blue\\LF191023_blue_20191119.json','LF191023_blue','20191119'],
    #                   ['E:\\MTH3_figures\\LF191023_blank\\LF191023_blank_20191116.json','LF191023_blank','20191116'],
    #                   ['E:\\MTH3_figures\\LF191022_2\\LF191022_2_20191116.json','LF191022_2','20191116'],
    #                  ]
    #
    # decoder_to, n_neurons_to = fig_template_decoding(roi_param_list, 'trialonset', fformat, 'naive', 'testplots', write_to_dict, load_raw)
    # decoder_lm, n_neurons_lm = fig_template_decoding(roi_param_list, 'lmcenter', fformat, 'naive', 'testplots', write_to_dict, load_raw)
    # decoder_rw, n_neurons_rw = fig_template_decoding(roi_param_list, 'reward', fformat, 'naive', 'testplots', write_to_dict, load_raw)
    #
    #

    print('--- number of neurons per animal ---')
    print('trial onset: ' + str(np.mean(np.array(n_neurons_to))) + ' +/- ' + str(stats.sem(np.array(n_neurons_to))))
    print('landmark: ' + str(np.mean(np.array(n_neurons_lm))) + ' +/- ' + str(stats.sem(np.array(n_neurons_lm))))
    print('reward: ' + str(np.mean(np.array(n_neurons_rw))) + ' +/- ' + str(stats.sem(np.array(n_neurons_rw))))
    print('--- decoding results ---------------')
    print('trial onset: ' + str(np.median(np.array(decoder_to))) + ' +/- ' + str(stats.sem(np.array(decoder_to))))
    print('landmark: ' + str(np.median(np.array(decoder_lm))) + ' +/- ' + str(stats.sem(np.array(decoder_lm))))
    print('reward: ' + str(np.median(np.array(decoder_rw))) + ' +/- ' + str(stats.sem(np.array(decoder_rw))))
    print('------------------------------------')

    fig = plt.figure(figsize=(4,8))
    ax1 = plt.subplot(111)
    ax1.scatter(np.full(len(decoder_to),0),decoder_to,s=80,c='0.8')
    ax1.scatter(np.full(len(decoder_lm),1),decoder_lm,s=80,c='0.8')
    ax1.scatter(np.full(len(decoder_rw),2),decoder_rw,s=80,c='0.8')

    ax1.scatter([0],[np.median(decoder_to)],marker='s',s=250,c='#39B54A',linewidths=0,zorder=4)
    ax1.scatter([1],[np.median(decoder_lm)],marker='s',s=250,c='#FF0000',linewidths=0,zorder=4)
    ax1.scatter([2],[np.median(decoder_rw)],marker='s',s=250,c='#29ABE2',linewidths=0,zorder=4)

    # ax1.errorbar([0],[np.median(decoder_to)],yerr=[sp.stats.sem(np.array(num_trialonset_short))],elinewidth=4,capsize=6,capthick=4,zorder=3,c='#39B54A',lw=5,ls='-')

    # loop through every animal
    for i in range(len(decoder_to)):
        ax1.plot([0,1,2], [decoder_to[i],decoder_lm[i],decoder_rw[i]],lw=2,c='0.8')
    ax1.plot([0,1,2], [np.median(decoder_to),np.median(decoder_lm),np.median(decoder_rw)],lw=4,c='k')

    ax1.set_ylim([0,1])
    ax1.set_xlim([-0.5,2.5])

    ax1.set_ylabel('decoder accuracy (%)')

    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(['trial onset','landmark','reward'], rotation=45)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)

    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    ax1.axhline(0.5,lw=2,c='0.9',ls='--')

    # do stats comparison
    print(stats.kruskal(np.array(decoder_to),np.array(decoder_lm),np.array(decoder_rw)))
    _, p_to_lm = stats.mannwhitneyu(decoder_to,decoder_lm)
    _, p_to_rw = stats.mannwhitneyu(decoder_to,decoder_rw)
    _, p_lm_rw = stats.mannwhitneyu(decoder_lm,decoder_rw)
    p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_to_lm,p_to_rw,p_lm_rw],alpha=0.05,method='bonferroni')
    # print(p_sit_run,p_sit_vr,p_run_vr)
    print('CORRECTED P-VALUES:')
    print('trial onset vs. landmark: ' + str(p_corrected[1][0]))
    print('trial onset vs. reward:   ' + str(p_corrected[1][1]))
    print('landmark run vs. reward:  ' + str(p_corrected[1][2]))

    fig.tight_layout()

    fname = loc_info['figure_output_path'] + 'testplots' + os.sep + 'decoder_all.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat)

    # V1
    # roi_param_list = [
    #                   [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day201777.json','LF170214_1','Day201777'],
    #                   [loc_info['figure_output_path'] + 'LF170214_1' + os.sep + 'LF170214_1_Day2017714.json','LF170214_1','Day2017714'],
    #                   [loc_info['figure_output_path'] + 'LF171211_2' + os.sep + 'LF171211_2_Day201852.json','LF171211_2','Day201852'],
    #                   [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_1'],
    #                   [loc_info['figure_output_path'] + 'LF180112_2' + os.sep + 'LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_2'],
    #                   [loc_info['figure_output_path'] + 'LF180219_1' + os.sep + 'LF180219_1_Day2018424_0025.json','LF180219_1','Day2018424_0025']
    #                  ]
    #
    # decoder_to = fig_template_decoding(roi_param_list, 'trialonset', fformat, 'v1', 'testplots', write_to_dict)
    # decoder_lm = fig_template_decoding(roi_param_list, 'lmcenter', fformat, 'v1', 'testplots', write_to_dict)
    # decoder_rw = fig_template_decoding(roi_param_list, 'reward', fformat, 'v1', 'testplots', write_to_dict)
    # #
    # fig = plt.figure(figsize=(6,8))
    # ax1 = plt.subplot(111)
    # ax1.scatter(np.full(len(decoder_to),0),decoder_to,s=80,c='0.8')
    # ax1.scatter(np.full(len(decoder_lm),1),decoder_lm,s=80,c='0.8')
    # ax1.scatter(np.full(len(decoder_rw),2),decoder_rw,s=80,c='0.8')
    #
    # ax1.scatter([0,1,2],[np.median(decoder_to),np.median(decoder_lm),np.median(decoder_rw)],s=120,c='k')
    #
    # # loop through every animal
    # for i in range(len(decoder_to)):
    #     ax1.plot([0,1,2], [decoder_to[i],decoder_lm[i],decoder_rw[i]],lw=2,c='0.8')
    # ax1.plot([0,1,2], [np.median(decoder_to),np.median(decoder_lm),np.median(decoder_rw)],lw=4,c='k')
    #
    # ax1.set_ylim([0,1])
    # ax1.set_xlim([-0.5,2.5])
    #
    # ax1.set_ylabel('decoder accuracy (%)')
    #
    # ax1.set_xticks([0,1,2])
    # ax1.set_xticklabels(['trial onset','landmark','reward'], rotation=45)
    #
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_linewidth(2)
    # ax1.spines['left'].set_linewidth(2)
    #
    # ax1.tick_params( \
    #     axis='both', \
    #     direction='out', \
    #     labelsize=16, \
    #     length=4, \
    #     width=2, \
    #     bottom='on', \
    #     right='off', \
    #     top='off')
    #
    # ax1.axhline(0.5,lw=2,c='0.9',ls='--')
    #
    # # do stats comparison
    # print(stats.kruskal(np.array(decoder_to),np.array(decoder_lm),np.array(decoder_rw)))
    # _, p_to_lm = stats.mannwhitneyu(decoder_to,decoder_lm)
    # _, p_to_rw = stats.mannwhitneyu(decoder_to,decoder_rw)
    # _, p_lm_rw = stats.mannwhitneyu(decoder_lm,decoder_rw)
    # p_corrected = sm_all.sandbox.stats.multicomp.multipletests([p_to_lm,p_to_rw,p_lm_rw],alpha=0.05,method='bonferroni')
    # # print(p_sit_run,p_sit_vr,p_run_vr)
    # print('CORRECTED P-VALUES:')
    # print('trial onset vs. landmark:   ' + str(p_corrected[1][0]))
    # print('trial onset vs. reward:            ' + str(p_corrected[1][1]))
    # print('landmark run vs. reward:            ' + str(p_corrected[1][2]))
    #
    # fname = loc_info['figure_output_path'] + 'testplots' + os.sep + 'decoder_all_v1.' + fformat
    # print(fname)
    # fig.savefig(fname, format=fformat)
