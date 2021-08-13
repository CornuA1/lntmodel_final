"""
Plot the deccelearation onset ahead of the landmark in the visual discrimination
task.

"""


import numpy as np
import h5py
import warnings
import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, filtfilt
from scipy.stats import uniform
import seaborn as sns
sns.set_style("white")

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

with open(loc_info['yaml_file'], 'r') as f:
    project_metainfo = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')
from filter_trials import filter_trials


def fig_decc_onset(h5path, sess, fname, fformat='png', subfolder=[]):
    # set location of reward zone
    rz = [205, 245]
    # location of default reward and tracklength
    default = 245
    tracklength = 245

    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    h5dat.close()

    fig = plt.figure(figsize=(4,4))
    #ax1 = plt.subplot2grid((2,1),(0,0))
    ax3 = plt.subplot2grid((1,1),(0,0))
    #ax1.set_ylabel('d2/f speed')
    ax3.set_xlabel('Location')

    # only plot trials where either a lick and/or a reward were detected
    # therefore: pull out trial numbers from licks and rewards dataset and map to
    # a new list of rows used for plotting
    short_trials = filter_trials( raw_ds, [], ['tracknumber',7])
    long_trials = filter_trials( raw_ds, [], ['tracknumber',9])

    # filter requirements.
    order = 6
    fs = int(np.size(raw_ds,0)/raw_ds[-1,0])       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz

    #raw_ds[:,3] = butter_lowpass_filter(raw_ds[:,3], cutoff, fs, order)
    # number of shuffles
    shuffles = 100

    # plot running speed
    bin_size = 5
    binnr_short = tracklength/bin_size
    mean_speed_short = np.empty((np.size(short_trials,0),int(binnr_short)))
    mean_speed_short[:] = np.NAN
    # mean_speed_short_shuffled = np.zeros((np.size(short_trials,0),int(binnr_short),shuffles))
    # mean_speed_short_shuffled[:] = np.NAN
    max_y_short = 0
    for i,t in enumerate(short_trials):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_bins = np.round(cur_trial[-1,1]/bin_size,0)
        cur_trial_start = raw_ds[raw_ds[:,6]==t,1][0]
        cur_trial_start_bin = np.round(cur_trial[0,1]/bin_size,0)
        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_speed_trial = stats.binned_statistic(cur_trial[:,1], cur_trial[:,3], 'mean', cur_trial_bins-cur_trial_start_bin, [cur_trial[0,1], cur_trial[-1,1]])[0]
            mean_speed_short[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
            #ax1.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8', lw=1)
            max_y_short = np.nanmax([max_y_short,np.nanmax(mean_speed_trial)])

    sem_speed = stats.sem(mean_speed_short,0,nan_policy='omit')
    mean_speed_sess_short = np.nanmedian(mean_speed_short,0)
    # ax1.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c='g',zorder=3)
    # ax1.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color='g',alpha=0.2)

    binnr_long = tracklength/bin_size
    mean_speed_long = np.empty((np.size(long_trials,0),int(binnr_long)))
    mean_speed_long[:] = np.NAN
    # mean_speed_long_shuffled = np.zeros((np.size(long_trials,0),int(binnr_long),shuffles))
    # mean_speed_long_shuffled[:] = np.NAN
    max_y_long = 0
    for i,t in enumerate(long_trials):
        cur_trial = raw_ds[raw_ds[:,6]==t,:]
        cur_trial_bins = np.round(cur_trial[-1,1]/bin_size,0)
        cur_trial_start = raw_ds[raw_ds[:,6]==t,1][0]
        cur_trial_start_bin = np.round(cur_trial[0,1]/bin_size,0)
        if cur_trial_bins-cur_trial_start_bin > 0:
            mean_speed_trial = stats.binned_statistic(cur_trial[:,1], cur_trial[:,3], 'mean', cur_trial_bins-cur_trial_start_bin, [cur_trial[0,1], cur_trial[-1,1]])[0]
            mean_speed_long[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
            #ax1.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
            max_y_long = np.amax([max_y_long,np.amax(mean_speed_trial)])


    sem_speed = stats.sem(mean_speed_long,0,nan_policy='omit')
    mean_speed_sess_long = np.nanmedian(mean_speed_long,0)
    # ax1.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c='r',zorder=3)
    # ax1.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color='r',alpha=0.2)

    mean_speed_sess_short = np.nanmedian(mean_speed_short,0)
    mean_speed_sess_long = np.nanmedian(mean_speed_long,0)

    # calculate the confidence intervals for each spatial bin by bootstrapping the samples in each bin
    # each bootstrap distribution is the same size as the original sample
    bootstrapdists = 1000
    nr_rew_trials = short_trials.shape[0]
    # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
    mean_speed_bootstrap = np.empty((nr_rew_trials,mean_speed_sess_short.shape[0],bootstrapdists))
    mean_speed_bootstrap[:] = np.nan
    # vector holding bootstrap variance estimate
    bt_mean_diff = np.empty((mean_speed_sess_short.shape[0], bootstrapdists))
    bt_mean_diff[:] = np.nan
    # vector holding confidence interval boundaries
    bt_CI_5_short = np.empty((mean_speed_sess_short.shape[0]))
    bt_CI_95_short = np.empty((mean_speed_sess_short.shape[0]))
    bt_CI_5_short[:] = np.nan
    bt_CI_95_short[:] = np.nan
    # loop through each spatial bin
    for i,m in enumerate(mean_speed_sess_short):
        if not np.isnan(mean_speed_sess_short[i]):
            # generate bootstrapped samples for each spatial bin
            for j in range(bootstrapdists):
                # draw random sample from individual speed traces
                mean_speed_bootstrap[:,i,j] = np.random.choice(mean_speed_short[:,i], nr_rew_trials)
                bt_mean_diff[i,j] = np.nanmedian(mean_speed_bootstrap[:,i,j]) - mean_speed_sess_short[i]
            bt_CI_5_short[i] = np.percentile(bt_mean_diff[i,:],5)
            bt_CI_95_short[i] = np.percentile(bt_mean_diff[i,:],95)

    nr_nor_trials = long_trials.shape[0]
    # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
    mean_speed_bootstrap = np.empty((nr_nor_trials,mean_speed_sess_long.shape[0],bootstrapdists))
    mean_speed_bootstrap[:] = np.nan
    # vector holding bootstrap variance estimate
    bt_mean_diff = np.empty((mean_speed_sess_long.shape[0], bootstrapdists))
    bt_mean_diff[:] = np.nan
    # vector holding confidence interval boundaries
    bt_CI_5_long = np.empty((mean_speed_sess_long.shape[0]))
    bt_CI_95_long = np.empty((mean_speed_sess_long.shape[0]))
    bt_CI_5_long[:] = np.nan
    bt_CI_95_long[:] = np.nan
    # loop through each spatial bin
    for i,m in enumerate(mean_speed_sess_long):
        if not np.isnan(mean_speed_sess_long[i]):
            # generate bootstrapped samples for each spatial bin
            for j in range(bootstrapdists):
                mean_speed_bootstrap[:,i,j] = np.random.choice(mean_speed_long[:,i], nr_nor_trials)
                bt_mean_diff[i,j] = np.nanmedian(mean_speed_bootstrap[:,i,j]) - mean_speed_sess_long[i]
            bt_CI_5_long[i] = np.percentile(bt_mean_diff[i,:],5)
            bt_CI_95_long[i] = np.percentile(bt_mean_diff[i,:],95)

    ax3.plot(np.linspace(0,binnr_short-1,binnr_short),mean_speed_sess_short,c=sns.xkcd_rgb["windows blue"],lw=2,zorder=3)
    ax3.fill_between(np.linspace(0,binnr_short-1,binnr_short), mean_speed_sess_short+bt_CI_5_short, mean_speed_sess_short+bt_CI_95_short, color=sns.xkcd_rgb["windows blue"],alpha=0.2)
    ax3.plot(np.linspace(0,binnr_long-1,binnr_long),mean_speed_sess_long,c='r',lw=2,zorder=3)
    ax3.fill_between(np.linspace(0,binnr_long-1,binnr_long), mean_speed_sess_long+bt_CI_5_long, mean_speed_sess_long+bt_CI_95_long, color='r',alpha=0.2)

    # loop through and check where running speeds are significantly different (determined by whether the mean of one track type is outside the CI of the other)
    sig_diff = np.empty(mean_speed_sess_short.shape[0])
    sig_diff[:] = np.nan
    ylims = ax3.get_ylim()
    for i in range(mean_speed_sess_short.shape[0]):
        if (not np.isnan(mean_speed_sess_short[i])) and (not np.isnan(mean_speed_sess_long[i] == np.nan)):
            if mean_speed_sess_short[i] > mean_speed_sess_long[i]:
                if (mean_speed_sess_short[i] + bt_CI_5_short[i]) > mean_speed_sess_long[i]:
                    sig_diff[i] = 1
                    ax3.axvline(i, ymin=mean_speed_sess_long[i]/ylims[1], ymax=mean_speed_sess_short[i]/ylims[1], c=sns.xkcd_rgb["windows blue"], lw=2)
            if mean_speed_sess_long[i] > mean_speed_sess_short[i]:
                if (mean_speed_sess_long[i] + bt_CI_5_long[i]) > mean_speed_sess_short[i]:
                    sig_diff[i] = -1
                    ax3.axvline(i, ymin=mean_speed_sess_short[i]/ylims[1], ymax=mean_speed_sess_long[i]/ylims[1], c='r', lw=2)

    #ax1.set_xlim([10,binnr_short])
    ax3.set_xlim([15,41])
    ax3.set_ylim([0,ylims[1]])

    speed_diff = 0
    diff_seq = 0
    diff_val = 0
    for i,sdiff in enumerate(sig_diff):
        if sig_diff[i] == 1:
            if diff_val == 1:
                diff_seq += 1
            else:
                diff_val = 1
                diff_seq = 1
        elif sig_diff[i] == -1:
            if diff_val == -1:
                diff_seq += 1
            else:
                diff_val = -1
                diff_seq = 1
        else:
            diff_seq = 0
        if diff_seq >= 3:
            speed_diff = i-3
            break

    plt.tight_layout()
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat, dpi=200)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
    #plt.show()
    if speed_diff == 0:
        speed_diff = np.nan
    return speed_diff

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    fformat = 'svg'

    # MOUSE = 'LF170811_2'
    # SESSION = 'Day20171024'
    # h5path = loc_info['vd_control_datafile'] + MOUSE + '/' + MOUSE + '.h5'
    # fig_decc_onset(h5path, SESSION, 'vd_dec_'+MOUSE+SESSION, fformat, 'vd_dec')

    MICE = ['LF170811_1','LF170811_2','LF170811_3','LF170811_6']
    # SESSIONS = ['Day2017831','Day201791','Day201793','Day201794','Day201795',
    # 	'Day201796','Day201797','Day201799','Day2017918','Day2017919','Day2017920','Day2017924','Day2017925'
    #     ,'Day2017926','Day2017927','Day2017104','Day2017105','Day2017106','Day2017108','Day2017109',
    #SESSIONS = ['Day20171021','Day20171022','Day20171023','Day20171024','Day20171025','Day20171026','Day20171027']
    SESSIONS = ['Day20171023','Day20171024','Day20171025']

    # MICE = ['LF170811_2']
    # SESSIONS = ['Day20171025']
    speed_differences = np.zeros((len(MICE), len(SESSIONS)))
    speed_differences[:] = np.nan
    for i,m in enumerate(MICE):
        for j,s in enumerate(SESSIONS):
            h5path = loc_info['vd_control_datafile'] + m + '/' + m + '.h5'
            speed_differences[i,j] = fig_decc_onset(h5path, s, 'vd_dec_'+m+s, fformat, 'vd_dec')

    print(speed_differences)
    fig = plt.figure(figsize=(2,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    for i,sdi in enumerate(speed_differences):
        ax1.scatter([0,1,2],sdi, c='0.8', linewidth=0, s=40)
        ax1.plot(sdi, c='0.8', lw=2)
    ax1.scatter([0,1,2],np.nanmean(speed_differences,0), c='k', linewidth=0, s=40)
    ax1.plot(np.nanmean(speed_differences,0),c='k', lw=3)
    ax1.set_xticks([0,1,2])
    ax1.set_xlim([-0.5,2.5])
    ax1.set_xticklabels(SESSIONS, rotation = 45)
    ax1.set_yticks([0,5,10,15,20,25,30])
    ax1.set_yticklabels(['205','180','165','140','115','90','65'])
    ax1.set_ylim(0,30)
    ax1.set_ylabel('Distance from landmark (cm)')
    ax1.set_xlabel('Day')
    fname = loc_info['figure_output_path'] + 'VD_speed_diff' + '.' + fformat
    fig.savefig(fname, format=fformat, dpi=200)
###### DUMP ######
