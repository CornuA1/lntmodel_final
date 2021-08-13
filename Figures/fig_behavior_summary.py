"""
Plot summary behavior data for Figure 1

"""

import h5py, os, sys, yaml, warnings
import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
import statsmodels.api as sm
import ipdb


warnings.filterwarnings('ignore')

sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + 'Analysis')
sys.path.append(loc_info['base_dir'] + 'Figures')
sys.path.append(loc_info['base_dir'] + 'OptoExp')

COLOR_SHORT = '#F58020'
COLOR_LONG = '#374D9E'

from filter_trials import filter_trials
from smzscore_LNT import smzscore
# from fig_behavior_stage5 import fig_behavior_stage5
# from fig_behavior_opto_s5 import fig_behavior_stage5

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_first_licks(licks, rewards, trials):
    # plot location of first trials on short and long trials
    first_lick = np.empty((0,4))
    first_lick_trials = np.empty((0))
    for r in trials:
        licks_all = licks[licks[:,2]==r,:]
        # licks_all = licks_all[licks_all[:,1]>101,:]
        if licks_all.size == 0:
             rew_lick = rewards[rewards[:,3]==r,:]
             if rew_lick.size > 0:
                 if rew_lick[0,5] == 1:
                     licks_all = np.asarray([[rew_lick[0,4], rew_lick[0,1], rew_lick[0,3], rew_lick[0,2]]])
                     first_lick = np.vstack((first_lick, licks_all[0,:].T))
                     first_lick_trials = np.append(first_lick_trials, r)
        else:
            if licks_all[0,3] == 3:
                licks_all = licks_all[licks_all[:,1]<338,:]
            elif licks_all[0,3] == 4:
                licks_all = licks_all[licks_all[:,1]<398,:]
            first_lick = np.vstack((first_lick, licks_all[0,:].T))
            first_lick_trials = np.append(first_lick_trials, r)


    return first_lick

def fig_behavior_stage5(h5path, sess, fname, fformat='png', subfolder=[], skip_earlylick=True):
    # load data

    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    rz = [320, 340]
    # location of default reward and tracklength
    default = 340
    tracklength = 340
    # number of shuffles for smz calculation
    shuffles = 1000
    session_smz_short,_ = smzscore( raw_ds, [320,340], 340, shuffles, 340, 0 )
    session_smz_long,_ = smzscore( raw_ds, [380,400], 400, shuffles, 400, 2 )
    smz = np.mean([session_smz_short,session_smz_long])
    print('SMZ: ' + str(np.round(smz,2)))
    # bin size (cm) for speed bins
    bin_size = 5

    # print('--- FRACTION REWARDS TRIGGERED/DEFAULT ---')
    # print('total number of rewards: ', reward_ds.shape[0])
    # print('total number TRIGGERED: ', reward_ds[reward_ds[:,5]==1.0,5].shape[0])
    # print('total number DEFAULT: ', reward_ds[reward_ds[:,5]==2.0,5].shape[0])
    # print('------------------------------------------')

    # calculate the fraction of default trials
    default_rewards_fraction = reward_ds[reward_ds[:,5]==2.0,5].shape[0]/reward_ds.shape[0]

    # create figure to later plot on
    fig = plt.figure(figsize=(14,10))
    fig.suptitle(fname)
    ax1 = plt.subplot2grid((10,2),(0,0), rowspan=6)
    ax2 = plt.subplot2grid((10,2),(6,0), rowspan=2)
    ax3 = plt.subplot2grid((10,2),(0,1), rowspan=6)
    ax4 = plt.subplot2grid((10,2),(6,1), rowspan=2)
    ax5 = plt.subplot2grid((10,2),(8,0), rowspan=2)
    ax6 = plt.subplot2grid((10,2),(8,1), rowspan=2)

    ax1.set_xlim([50,340])
    ax1.set_ylabel('Trial #')
    ax1.set_xticks([50,200,240,320])
    ax1.set_xticklabels(['50','200','240','320'])
    ax1.set_xlabel('Location (cm)')
    ax1.set_title('Short trials')

    ax2.set_xlim([10,80])
    ax2.set_ylabel('Speed (cm/sec)')
    ax2.set_xlabel('Location (cm)')
    ax2.set_xticks([50/bin_size,200/bin_size,240/bin_size,320/bin_size,380/bin_size])
    ax2.set_xticklabels(['50','200','240','320','380'])

    ax3.set_xlim([50,400])
    ax3.set_ylabel('Trial #')
    ax3.set_xticks([50,200,240,320,380])
    ax3.set_xticklabels(['50','200','240','320','380'])
    ax3.set_xlabel('Location (cm)')
    ax3.set_title('Short trials')

    # plot landmark and rewarded area as shaded zones
    ax1.axvspan(200,240,color='0.9',zorder=0)
    ax1.axvspan(320,340,color=COLOR_SHORT,alpha=0.3,zorder=9, **{"linewidth":0.0})

    ax2.axvspan(200/bin_size,240/bin_size,color='0.9',zorder=0)
    ax2.axvspan(320/bin_size,340/bin_size,color=COLOR_SHORT,alpha=0.3,zorder=9, **{"linewidth":0.0})
    ax2.axvspan(380/bin_size,400/bin_size,color=COLOR_LONG,alpha=0.3,zorder=9, **{"linewidth":0.0})

    ax3.axvspan(200,240,color='0.9',zorder=0)
    ax3.axvspan(380,400,color=COLOR_LONG,alpha=0.3,zorder=9, **{"linewidth":0.0})

    fl_diff = 0
    t_score = 0

    # make array of y-axis locations for licks. If clause to check for empty arrays
    if np.size(licks_ds) > 0 and np.size(reward_ds) > 0:
        # only plot trials where either a lick and/or a reward were detected
        # therefore: pull out trial numbers from licks and rewards dataset and map to
        # a new list of rows used for plotting

        short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
        long_trials = filter_trials( raw_ds, [], ['tracknumber',4])
        if skip_earlylick:
            short_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[0,240]],short_trials)
            long_trials = filter_trials( raw_ds, [], ['exclude_earlylick_trials',[0,240]],long_trials)
        #long_trials = filter_trials( raw_ds, [], ['opto_stim_on'],long_trials)

        # get trial numbers to be plotted
        lick_trials = np.unique(licks_ds[:,2])
        # below is a correction for the trial a reward was provided in based on whether the rewards dataset was already corrected earlier in the processing pipeline
        if np.array_equal(np.unique(reward_ds[:,2]), np.array([5.])):
            reward_correction = 1
        else:
            reward_correction = 0

        reward_trials = np.unique(reward_ds[:,3])-reward_correction
        scatter_rowlist_map = np.union1d(lick_trials,reward_trials)
        scatter_rowlist_map_short = np.intersect1d(scatter_rowlist_map, short_trials)
        scatter_rowlist_short = np.arange(np.size(scatter_rowlist_map_short,0))
        scatter_rowlist_map_long = np.intersect1d(scatter_rowlist_map, long_trials)
        scatter_rowlist_long = np.arange(np.size(scatter_rowlist_map_long,0))

        ax1.set_ylim([-1,len(np.unique(scatter_rowlist_short))])
        ax3.set_ylim([-1,len(np.unique(scatter_rowlist_long))])

        # scatterplot of licks/rewards in order of trial number
        for i,r in enumerate(scatter_rowlist_map_short):
            plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
            plot_rewards_x = reward_ds[reward_ds[:,3]-reward_correction==r,1]
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            if reward_ds[reward_ds[:,3]-reward_correction==r,5] == 1:
                col = '#00C40E'
            else:
                col = 'r'

            # if reward location is recorded at beginning of track, set it to end of track
            if plot_rewards_x < 300:
                plot_rewards_x = 340

            # plot licks and rewards
            if np.size(plot_licks_x) > 0:
                plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_short[i])
                ax1.scatter(plot_licks_x, plot_licks_y,c='k',lw=0)
            if np.size(plot_rewards_x) > 0:
                plot_rewards_y = scatter_rowlist_short[i]
                ax1.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
            if np.size(cur_trial_start) > 0:
                plot_starts_y = scatter_rowlist_short[i]
                ax1.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)



        # scatterplot of licks/rewards in order of trial number
        for i,r in enumerate(scatter_rowlist_map_long):
            plot_licks_x = licks_ds[licks_ds[:,2]==r,1]
            plot_rewards_x = reward_ds[reward_ds[:,3]-reward_correction==r,1]
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            if reward_ds[reward_ds[:,3]-reward_correction==r,5] == 1:
                col = '#00C40E'
            else:
                col = 'r'

            # if reward location is recorded at beginning of track, set it to end of track
            if plot_rewards_x < 300:
                plot_rewards_x = 400

            # plot licks and rewards
            if np.size(plot_licks_x) > 0:
                plot_licks_y = np.full(plot_licks_x.shape[0],scatter_rowlist_long[i])
                ax3.scatter(plot_licks_x, plot_licks_y,c='k',lw=0)
            if np.size(plot_rewards_x) > 0:
                plot_rewards_y = scatter_rowlist_long[i]
                ax3.scatter(plot_rewards_x, plot_rewards_y,c=col,lw=0)
            if np.size(cur_trial_start) > 0:
                plot_starts_y = scatter_rowlist_long[i]
                ax3.scatter(cur_trial_start, plot_starts_y,c='b',marker='>',lw=0)

        # plot running speed
        # filter requirements.
        if True:
            order = 6
            fs = int(np.size(raw_ds,0)/raw_ds[-1,0])       # sample rate, Hz
            cutoff = 1 # desired cutoff frequency of the filter, Hz
            speed_vector = butter_lowpass_filter(raw_ds[:,3], cutoff, fs, order)
        else:
            speed_vector = raw_ds[:,3]

        all_speed_vals_short = np.empty(0)
        binnr_short = 400/bin_size
        mean_speed = np.empty((np.size(scatter_rowlist_map_short,0),int(binnr_short)))
        mean_speed[:] = np.NAN
        max_y_short = 0

        for i,t in enumerate(scatter_rowlist_map_short):
            cur_trial = raw_ds[raw_ds[:,6]==t,:]
            cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)

            if cur_trial_bins-cur_trial_start_bin > 0:
                cur_trial_speed = speed_vector[raw_ds[:,6]==t]
                cur_trial_speed[cur_trial_speed>80] = np.nan
                all_speed_vals_short = np.concatenate((all_speed_vals_short,cur_trial_speed[cur_trial_speed>3]))
                mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], cur_trial_speed, 'mean', \
                                                          cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
                mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
                #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
                max_y_short = np.amax([max_y_short,np.amax(mean_speed_trial)])

        # ipdb.set_trace()
        sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
        mean_speed_sess_short = np.nanmean(mean_speed,0)
        ax2.plot(np.arange(0,binnr_short,1),mean_speed_sess_short,c=COLOR_SHORT,zorder=3)
        ax2.fill_between(np.arange(0,binnr_short,1), mean_speed_sess_short-sem_speed, mean_speed_sess_short+sem_speed, color=COLOR_SHORT,alpha=0.2)

        # plot running speed
        all_speed_vals_long = np.empty(0)
        binnr_long = 400/bin_size
        mean_speed = np.empty((np.size(scatter_rowlist_map_long,0),int(binnr_long)))
        mean_speed[:] = np.NAN
        max_y_long = 0
        for i,t in enumerate(scatter_rowlist_map_long):
            cur_trial = raw_ds[raw_ds[:,6]==t,:]
            cur_trial_bins = np.round(cur_trial[-1,1]/5,0)
            cur_trial_start = raw_ds[raw_ds[:,6]==r,1][0]
            cur_trial_start_bin = np.round(cur_trial[0,1]/5,0)

            if cur_trial_bins-cur_trial_start_bin > 0:
                # print(np.size(mean_speed_trial))
                cur_trial_speed = raw_ds[raw_ds[:,6]==t,3]
                cur_trial_speed[cur_trial_speed>80] = np.nan
                all_speed_vals_long = np.concatenate((all_speed_vals_long,cur_trial_speed[cur_trial_speed>3]))
                mean_speed_trial = stats.binned_statistic(raw_ds[raw_ds[:,6]==t,1], cur_trial_speed, 'mean',
                                                          cur_trial_bins-cur_trial_start_bin, (cur_trial_start_bin*bin_size, cur_trial_bins*bin_size))[0]
                mean_speed[i,int(cur_trial_start_bin):int(cur_trial_bins)] = mean_speed_trial
                #ax2.plot(np.linspace(cur_trial_start_bin,cur_trial_bins,cur_trial_bins-cur_trial_start_bin),mean_speed_trial,c='0.8')
                max_y_long = np.amax([max_y_long,np.amax(mean_speed_trial)])

        sem_speed = stats.sem(mean_speed,0,nan_policy='omit')
        mean_speed_sess_long = np.nanmean(mean_speed,0)
        ax2.plot(np.arange(0,binnr_long,1),mean_speed_sess_long,c=COLOR_LONG,zorder=3)
        ax2.fill_between(np.arange(0,binnr_long,1), mean_speed_sess_long-sem_speed, mean_speed_sess_long+sem_speed, color=COLOR_LONG,alpha=0.2)

        ax5.hist(all_speed_vals_short, bins=np.arange(0,70,10), histtype='step', color=COLOR_SHORT, **{"linewidth":4.0})
        ax5.hist(all_speed_vals_long, bins=np.arange(0,70,10), histtype='step', color=COLOR_LONG, **{"linewidth":4.0})

        # plot location of first trials on short and long trials
        first_lick_short = []
        first_lick_short_trials = []
        first_lick_long = []
        first_lick_long_trials = []
        for r in scatter_rowlist_map:
            licks_all = licks_ds[licks_ds[:,2]==r,:]

            if not licks_all.size == 0:
                # licks_all = licks_all[licks_all[:,1]>101,:]
                pass
            else:
                rew_lick = reward_ds[reward_ds[:,3]-reward_correction==r,:][0]
                # print(rew_lick)
                if rew_lick[2] == 3:
                    licks_all = np.asarray([[0, rew_lick[1], rew_lick[3], 3]])
                elif rew_lick[2] == 4:
                    licks_all = np.asarray([[0, rew_lick[1], rew_lick[3], 4]])

            if licks_all.shape[0]>0:
                lick = licks_all[0]
                if lick[3] == 3:
                    first_lick_short.append(lick[1])
                    first_lick_short_trials.append(r)

                elif lick[3] == 4:
                    first_lick_long.append(lick[1])
                    first_lick_long_trials.append(r)

        first_lick_short_pairs = np.vstack((first_lick_short,first_lick_short_trials))
        first_lick_short_pairs = first_lick_short_pairs[:,np.in1d(first_lick_short_pairs[1,:],short_trials)]
        first_lick_long_pairs = np.vstack((first_lick_long,first_lick_long_trials))
        first_lick_long_pairs = first_lick_long_pairs[:,np.in1d(first_lick_long_pairs[1,:],long_trials)]

        ax4.scatter(first_lick_short_pairs[0,:],first_lick_short_pairs[1,:],c=COLOR_SHORT,lw=0)
        ax4.scatter(first_lick_long_pairs[0,:],first_lick_long_pairs[1,:],c=COLOR_LONG,lw=0)
        ax4.axvline(np.median(first_lick_short), c=COLOR_SHORT, lw=2)
        ax4.axvline(np.median(first_lick_long), c=COLOR_LONG, lw=2)

        # if np.size(first_lick_short) > 10:
        #     fl_short_running_avg = np.convolve(first_lick_short,np.ones(10),'valid')/10
        #     ax4.plot(fl_short_running_avg, first_lick_short_trials[5:len(first_lick_short_trials)-4], c=COLOR_SHORT, lw=2)
        #
        # if np.size(first_lick_long) > 10:
        #     fl_long_running_avg = np.convolve(first_lick_long,np.ones(10),'valid')/10
        #     ax4.plot(fl_long_running_avg, first_lick_long_trials[5:len(first_lick_long_trials)-4], c=COLOR_LONG, lw=2)

        # bootstrap differences between pairs of first lick locations
        if np.size(first_lick_short) > 5 and np.size(first_lick_long) > 5:
            num_shuffles = 10000
            short_bootstrap = np.random.choice(first_lick_short,num_shuffles)
            long_bootstrap = np.random.choice(first_lick_long,num_shuffles)
            bootstrap_diff = long_bootstrap - short_bootstrap
            # tval,pval = stats.ttest_1samp(bootstrap_diff,0)
            # pval = np.size(np.where(bootstrap_diff < 0))/num_shuffles
            fl_diff = np.mean(bootstrap_diff)/np.std(bootstrap_diff)

            # sns.distplot(bootstrap_diff,ax=ax2)
            # vl_handle = ax2.axvline(np.mean(bootstrap_diff),c='b')
            # vl_handle.set_label('z-score = ' + str(fl_diff))
            # ax2.legend()

        sns.distplot(first_lick_short,bins=np.arange(200,410,5),color=COLOR_SHORT,ax=ax2)
        sns.distplot(first_lick_long,bins=np.arange(200,410,5),color=COLOR_LONG,ax=ax2)
        # ax2.set_xlim([200,400])

        # print(stats.ttest_ind(first_lick_short,first_lick_long))

        # calculate the confidence intervals for first licks from a bootstrapped distribution
        # number of resamples
        bootstrapdists = 1000
        # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
        fl_short_bootstrap = np.empty((len(first_lick_short),bootstrapdists))
        fl_short_bootstrap[:] = np.nan
        # vector holding bootstrap variance estimate
        bt_mean_diff = np.empty((bootstrapdists,))
        bt_mean_diff[:] = np.nan

        for j in range(bootstrapdists):
            fl_short_bootstrap[:,j] = np.random.choice(first_lick_short, len(first_lick_short))
            bt_mean_diff[j] = np.nanmedian(fl_short_bootstrap[:,j]) - np.nanmedian(first_lick_short)

        bt_CI_5_short = np.percentile(bt_mean_diff[:],5)
        bt_CI_95_short = np.percentile(bt_mean_diff[:],95)
        ax4.axvspan(np.nanmedian(first_lick_short)+bt_CI_5_short,np.nanmedian(first_lick_short), color=COLOR_SHORT, ls='--',alpha=0.2)
        ax4.axvspan(np.nanmedian(first_lick_short)+bt_CI_95_short,np.nanmedian(first_lick_short), color=COLOR_SHORT, ls='--',alpha=0.2)

        # calculate the confidence intervals for first licks from a bootstrapped distribution
        # create array with shape [nr_trials,nr_bins_per_trial,nr_bootstraps]
        fl_long_bootstrap = np.empty((len(first_lick_long),bootstrapdists))
        fl_long_bootstrap[:] = np.nan
        # vector holding bootstrap variance estimate
        bt_mean_diff = np.empty((bootstrapdists,))
        bt_mean_diff[:] = np.nan

        for j in range(bootstrapdists):
            if len(first_lick_long) > 0:
                fl_long_bootstrap[:,j] = np.random.choice(first_lick_long, len(first_lick_long))
                bt_mean_diff[j] = np.nanmedian(fl_long_bootstrap[:,j]) - np.nanmedian(first_lick_long)
            else:
                bt_mean_diff[j] = np.nan
        bt_CI_5_long = np.percentile(bt_mean_diff[:],2.5)
        bt_CI_95_long = np.percentile(bt_mean_diff[:],97.5)

        ax4.axvspan(np.nanmedian(first_lick_long)+bt_CI_5_long,np.nanmedian(first_lick_long), color=COLOR_LONG, ls='--',alpha=0.2)
        ax4.axvspan(np.nanmedian(first_lick_long)+bt_CI_95_long,np.nanmedian(first_lick_long), color=COLOR_LONG, ls='--',alpha=0.2)

        # ax4.set_xlim([240,380])
        ax4.set_xlim([50,180])

        if np.size(first_lick_short) > 0 and np.size(first_lick_long):
            t_score = np.median(first_lick_long) - np.median(first_lick_short)
            # print(stats.mannwhitneyu(first_lick_short, first_lick_long))
            ax4.text(320, 10, 't-score: '+str(t_score))

        if np.nanmedian(first_lick_long)+bt_CI_5_long > np.nanmedian(first_lick_short) or np.nanmedian(first_lick_short)+bt_CI_95_short < np.nanmedian(first_lick_long):
            print('significant! Task score: ' + str(np.round(t_score,2)))

    plt.tight_layout()

    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
    print(fname)

    return t_score, default_rewards_fraction, [all_speed_vals_short, all_speed_vals_long], smz

def fig_first_lick_distribution(h5path, sess, fname, trialtype, fformat='png', subfolder=[]):
   # load data
    h5dat = h5py.File(h5path, 'r')
    raw_ds = np.copy(h5dat[sess + '/raw_data'])
    licks_ds = np.copy(h5dat[sess + '/licks_pre_reward'])
    reward_ds = np.copy(h5dat[sess + '/rewards'])
    h5dat.close()

    # create figure to later plot on
    fig = plt.figure(figsize=(6,4))
    # ax1 = plt.subplot(111)
    ax2 = plt.subplot(111)

    short_trials = filter_trials( raw_ds, [], ['tracknumber',3])
    long_trials = filter_trials( raw_ds, [], ['tracknumber',4])
    # mask_off_trials = filter_trials( raw_ds, [], ['opto_mask_light_off'], short_trials)
    mask_off_trials_short = short_trials #filter_trials( raw_ds, [], [trialtype], short_trials)
    mask_off_trials_long = long_trials #filter_trials( raw_ds, [], [trialtype], long_trials)

    first_licks_mask_off_short = get_first_licks(licks_ds, reward_ds, mask_off_trials_short)
    first_licks_mask_off_long = get_first_licks(licks_ds, reward_ds, mask_off_trials_long)

    mask_off_trial_start_short = []
    mask_off_lmend_short = []
    mask_off_loc_pairs_short = np.empty((0,2))
    for m in first_licks_mask_off_short[:,2]:
        cur_trial_start = raw_ds[raw_ds[:,6]==m,1][0]
        first_lick_loc = first_licks_mask_off_short[first_licks_mask_off_short[:,2] == m,1]
        mask_off_trial_start_short.append(first_lick_loc-cur_trial_start)
        mask_off_lmend_short.append(first_lick_loc-240)
        if first_lick_loc > 200:
            mask_off_loc_pairs_short = np.vstack((mask_off_loc_pairs_short,np.asarray([cur_trial_start,first_lick_loc]).T))

    mask_off_trial_start_long = []
    mask_off_lmend_long = []
    mask_off_loc_pairs_long = np.empty((0,2))
    for m in first_licks_mask_off_long[:,2]:
        cur_trial_start = raw_ds[raw_ds[:,6]==m,1][0]
        first_lick_loc = first_licks_mask_off_long[first_licks_mask_off_long[:,2] == m,1]
        mask_off_trial_start_long.append(first_lick_loc-cur_trial_start)
        mask_off_lmend_long.append(first_lick_loc-240)
        if first_lick_loc > 200:
            mask_off_loc_pairs_long = np.vstack((mask_off_loc_pairs_long,np.asarray([cur_trial_start,first_lick_loc]).T))


    # ax1.scatter(mask_off_lmend[:,0], mask_off_lmend[:,1], c='r', s=80)
    bin_edges = np.arange(-150,160,10)

    mask_off_trial_start_short = np.array(mask_off_trial_start_short)-np.median(mask_off_trial_start_short)
    mask_off_trial_start_long = np.array(mask_off_trial_start_long)-np.median(mask_off_trial_start_long)
    mask_off_lmend_short = np.array(mask_off_lmend_short)-np.median(mask_off_lmend_short)
    mask_off_lmend_long = np.array(mask_off_lmend_long)-np.median(mask_off_lmend_long)

    mask_off_trial_start_all = np.concatenate((mask_off_trial_start_short,mask_off_trial_start_long))
    mask_off_lmend_all = np.concatenate((mask_off_lmend_short,mask_off_lmend_long))

    # sns.distplot(mask_off_trial_start_all, bins=bin_edges, color='k', ax=ax1)
    # sns.distplot(mask_off_lmend_all, bins=bin_edges, color='r', ax=ax1)

    mask_off_loc_pairs_starts_all = np.concatenate((mask_off_loc_pairs_short[:,0],mask_off_loc_pairs_long[:,0]))
    mask_off_loc_pairs_fl_all = np.concatenate((mask_off_loc_pairs_short[:,1]-np.median(mask_off_loc_pairs_short[:,1]),mask_off_loc_pairs_long[:,1]-np.median(mask_off_loc_pairs_long [:,1])))

    ax2.scatter(mask_off_loc_pairs_short[:,0], mask_off_loc_pairs_short[:,1]-np.median(mask_off_loc_pairs_short[:,1]),linewidths=0, c=COLOR_SHORT, s=40, label='short trials')
    ax2.scatter(mask_off_loc_pairs_long[:,0], mask_off_loc_pairs_long[:,1]-np.median(mask_off_loc_pairs_long [:,1]),linewidths=0, c=COLOR_LONG, s=40, label='long trials')
    # ax2.scatter(mask_off_loc_pairs_starts_all,mask_off_loc_pairs_fl_all, c='k', s=80)

    mask_off_linregress = stats.linregress(mask_off_loc_pairs_starts_all,mask_off_loc_pairs_fl_all)
    # print(mask_off_linregress)

    ax2.plot(np.arange(45,105,1),(np.arange(45,105,1)*mask_off_linregress[0])-np.mean(np.arange(45,105,1)*mask_off_linregress[0]),c='k',lw=2,ls='--')

    # print(np.mean(np.arange(45,105,1)*mask_off_linregress[0]))

    pil_raw_F_exog = sm.add_constant(mask_off_loc_pairs_starts_all)
    rlmres = sm.RLM(mask_off_loc_pairs_fl_all,pil_raw_F_exog,M=sm.robust.norms.TukeyBiweight()).fit()
    # ax2.plot(mask_off_loc_pairs_starts_all,rlmres.fittedvalues,label='TukeyBiweight',c='r',lw=2,ls='--')
    # print(rlmres.summary())

    # print(mask_off_linregress[0])

    # ax1.set_xlim([-150,150])
    #
    ax2.set_ylim([-100,100])
    ax2.set_xlim([40,110])

    ax2.set_xlabel('starting location (cm)', fontsize=18)
    ax2.set_ylabel('first lick location (cm)', fontsize=18)

    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(2)
    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax2.text(45,88,'slope: ' + str(np.round(mask_off_linregress[0],4)), fontsize=16)

    plt.tight_layout()

    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    fig.savefig(fname, format=fformat,dpi=150)

    print(fname)

    return mask_off_linregress[0]


def slope_summary():

    fformat = 'png'
    start_slope_mask = []
    start_slope_imgaging = []
    start_slope_stim = []

    datasets_opto = [['LF180728_1','Day2018105'],['LF180515_1','Day2018105'],['LF180514_1','Day2018105'],['LF180919_1','Day2018122'],['LF180920_1','Day2018126']]

    datasets_imaging = [['LF170613_1','Day20170804'],['LF170421_2','Day20170719'],['LF170420_1','Day20170719'],['LF170420_1','Day201783'], \
                        ['LF170110_2','Day201748_1'], ['LF170222_1','Day201776'],['LF171212_2','Day2018218_2'],['LF170214_1','Day201777'],['LF170214_1','Day2017714'],\
                        ['LF171211_2','Day201852'],['LF180112_2','Day2018424_1'],['LF180219_1','Day2018424_0025']]

    for ds in datasets_opto:
        # print(ds[0], ds[1])
        h5path = loc_info['imaging_dir'] + ds[0] + '/' + ds[0] + '.h5'
        start_slope_mask.append(fig_first_lick_distribution(h5path, ds[1], ds[0]+ds[1]+'_fl_mask', 'opto_mask_on_stim_off', fformat, subfolder))
        # start_slope_stim.append(fig_first_lick_distribution(h5path, ds[1], ds[0]+ds[1]+'_fl_stim', 'opto_stim_on', fformat, subfolder))

    for ds in datasets_imaging:
        print(ds[0], ds[1])
        h5path = loc_info['imaging_dir'] + ds[0] + '/' + ds[0] + '.h5'
        start_slope_imgaging.append(fig_first_lick_distribution(h5path, ds[1], ds[0]+ds[1]+'_fl_mask', 'opto_mask_on_stim_off', fformat, subfolder))
        # start_slope_stim.append(fig_first_lick_distribution(h5path, s, MOUSE+s+'_fl_stim', 'opto_stim_on', fformat, subfolder))

    print('--- RESULTS FIRTS LICK SLOPE ---')
    print('mean slope +/- SEM imaging: ', str(np.round(np.mean(np.array(start_slope_imgaging)),2)), ' +/- ', str(np.round(stats.sem(np.array(start_slope_imgaging)),4)))
    print('mean slope +/- SEM opto mas: ', str(np.round(np.mean(np.array(start_slope_mask)),2)), ' +/- ', str(np.round(stats.sem(np.array(start_slope_mask)),4)))
    # print('mean slope +/- SEM opto stim: ', str(np.round(np.mean(np.array(start_slope_stim)),2)), ' +/- ', str(np.round(stats.sem(np.array(start_slope_stim)),4)))
    print('--------------------------------')

    fig = plt.figure(figsize=(2,4))
    ax1 = plt.subplot(111)

    ax1.scatter(np.zeros((len(start_slope_imgaging))),np.array(start_slope_imgaging),s=120,linewidths=3,facecolor='none',edgecolor='k', zorder=2)
    ax1.scatter(np.ones((len(start_slope_mask))),np.array(start_slope_mask),s=120,linewidths=3,facecolor='none',edgecolor='#0A8FCF', zorder=3)
    # ax1.scatter(np.full((len(start_slope_stim)),2),np.array(start_slope_stim),s=120,linewidths=3,facecolor='#0A8FCF',edgecolor='#0A8FCF', zorder=3)
    # ax1.scatter([1,1,1,1,1],np.array(start_slope_stim),s=120,linewidths=3,edgecolor='#0A8FCF', color='#0A8FCF',zorder=3)

    # for i in range(len(start_slope_mask)):
        # ax1.plot([0,1],[np.array(start_slope_mask)[i],np.array(start_slope_stim)[i]],lw=2,c='k',zorder=2)

    ax1.set_xlim([-0.5,2.5])
    ax1.set_ylim([-1,1])

    ax1.set_ylabel('slope values', fontsize=18)

    ax1.set_xticks([])
    ax1.set_xticklabels([])

    ax1.set_yticks([-1,0,1])
    ax1.set_yticklabels(['-1','0','1'])

    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    plt.tight_layout()

    fname = 'trialstart slopes'
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    fig.savefig(fname, format=fformat,dpi=150)

    print(fname)

    plt.close(fig)

def task_score_imaging():
    ### START TASK SCORE HISTOGRAM FIGURE ###
    fig = plt.figure(figsize=(2,4))
    ax1 = plt.subplot(111)

    datasets = [['LF170613_1','Day20170804'],['LF170421_2','Day20170719'],['LF170420_1','Day20170719'],['LF170420_1','Day201783'], \
                ['LF170110_2','Day201748_1'], ['LF170222_1','Day201776'],['LF171212_2','Day2018218_2'],['LF170214_1','Day201777'],['LF170214_1','Day2017714'],\
                ['LF171211_2','Day201852'],['LF180112_2','Day2018424_1'],['LF180219_1','Day2018424_0025']]
    # datasets = [['LF170222_1','Day201776']]
    # datasets = [['LF171212_2','Day2018218_2']]
        #,['LF170222_1','Day20170615'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170420_1','Day20170719'],
        #        ['LF170110_2','Day20170331'],['LF170214_1','Day201777'],['LF170214_1','Day2017714']]


    fl_all = []
    default_fraction = []
    smz_all = []
    for ds in datasets:
        # print(ds[0], ds[1])
        h5path = loc_info['imaging_dir'] + ds[0] + '/' + ds[0] + '.h5'
        fldiff, def_frac, _, smz = fig_behavior_stage5(h5path, ds[1], ds[0]+ds[1], fformat, subfolder)
        fl_all.append(fldiff)
        default_fraction.append(def_frac)
        smz_all.append(smz)

    print('--- SMZ SCORES ---')
    # print(smz_all)
    print(str(np.round(np.mean(smz_all),2)), str(np.round(stats.sem(smz_all),2)))
    print('------------------')

    ax1.scatter(np.zeros((len(fl_all))),np.array(fl_all),s=120,linewidths=3,facecolor='none',edgecolor='k', zorder=3)

    ax1.set_xlim([-0.1,0.1])
    ax1.set_ylim([0,80])

    ax1.set_ylabel('Task Score', fontsize=18)

    ax1.set_xticks([])
    ax1.set_xticklabels([])

    ax1.set_yticks([0,20,40,60,80])
    ax1.set_yticklabels(['0','20','40','60','80'])

    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    # sns.distplot(fl_all, kde=False, ax=ax1, bins=np.arange(0,80,10),color="k", hist_kws={"linewidth": 0})
    # ax1.set_yticks([0,1,2,3])
    # ax1.set_yticklabels(['0','1','2','3'])
    #
    # ax1.set_xlim([0,60])
    # ax1.set_xticks([0,20,40,60])
    # ax1.set_xticklabels(['0','20','40','60'])
    #
    # ax1.tick_params(length=5,width=2,bottom=True,left=True,top=False,right=False,labelsize=16)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    #
    # ax1.spines['left'].set_linewidth(2)
    # ax1.spines['bottom'].set_linewidth(2)

    print('--- RESULTS TASK SCORE ---')
    print('mean +/- SEM task score: ', str(np.round(np.mean(np.array(fl_all)),2)), ' +/- ', str(np.round(stats.sem(np.array(fl_all)), 2)))
    print('mean fraction DEFAULT +/- SEM: ', str(np.round(np.mean(np.array(default_fraction)),4)), ' +/- ', str(np.round(stats.sem(np.array(default_fraction)), 4)))
    print('--------------------------')
    # print(default_fraction)


    fig.tight_layout()

    fname = 'tscore all'
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    fig.savefig(fname, format=fformat,dpi=150)

    print(fname)

def running_speed_analysis():
    ### START TASK SCORE HISTOGRAM FIGURE ###
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,10),(0,0), rowspan=1, colspan=3)
    ax2 = plt.subplot2grid((1,10),(0,3), rowspan=1, colspan=6)

    datasets = [['LF170613_1','Day20170804'],['LF170421_2','Day20170719'],['LF170421_2','Day20170720'],['LF170420_1','Day20170719'],['LF170420_1','Day201783'], \
                ['LF170110_2','Day201748_1'], ['LF170222_1','Day201776'],['LF171212_2','Day2018218_2'],['LF170214_1','Day201777'],['LF170214_1','Day2017714'],\
                ['LF171211_2','Day201852'],['LF180112_2','Day2018424_1'],['LF180219_1','Day2018424_0025']]

    # datasets = [['LF170613_1','Day20170804'],['LF170222_1','Day201776']]
    speed_mean_short = []
    speed_mean_long = []
    speed_all = []
    for ds in datasets:
        # print(ds[0], ds[1])
        h5path = loc_info['imaging_dir'] + ds[0] + '/' + ds[0] + '.h5'
        fldiff, def_frac, speed_vals, _ = fig_behavior_stage5(h5path, ds[1], ds[0]+ds[1], fformat, subfolder)
        sns.distplot(speed_vals[0],hist=False,kde_kws={"color": COLOR_SHORT,"alpha":1.0, "lw":2}, ax=ax2)
        sns.distplot(speed_vals[1],hist=False,kde_kws={"color": COLOR_LONG,"alpha":1.0, "lw":2}, ax=ax2)
        speed_mean_short.append(np.mean(speed_vals[0]))
        speed_mean_long.append(np.mean(speed_vals[1]))
        speed_all.append(speed_vals)

        ax1.scatter([0],[np.mean(speed_vals[0])],s=120,linewidths=3,facecolor='w',edgecolor=COLOR_SHORT, zorder=3)
        ax1.scatter([0.5],[np.mean(speed_vals[1])],s=120,linewidths=3,facecolor='w',edgecolor=COLOR_LONG, zorder=3)
        ax1.plot([0,0.5],[np.mean(speed_vals[0]), np.mean(speed_vals[1])], lw=2, c='0.5')
    # ax1.scatter([0],[np.mean(speed_mean)],s=80,marker='.',linewidths=5,facecolor='none',edgecolor='r', zorder=4)

    ax2.set_xlim([0,80])

    print('--- MEAN SPEED ---')
    print('mean +/- SEM (cm/sec) SHORT: ', str(np.round(np.mean(np.array(speed_mean_short)),2)), ' +/- ',  str(np.round(stats.sem(np.array(speed_mean_short)),2)))
    print('mean +/- SEM (cm/sec) LONG: ', str(np.round(np.mean(np.array(speed_mean_long)),2)), ' +/- ',  str(np.round(stats.sem(np.array(speed_mean_long)),2)))
    print(stats.wilcoxon(speed_mean_short, speed_mean_long))
    print('------------------')

    ax1.set_xlim([-0.3,0.8])
    ax1.set_ylim([0,60])

    ax1.set_xticks([])
    ax1.set_xticklabels([])

    ax1.set_yticks([0,20,40,60])
    ax1.set_yticklabels(['0','20','40','60'])
    ax1.set_ylabel('mean speed (cm/sec)', fontsize=16)

    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(2)
    ax2.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    fig.tight_layout()
    fname = 'meanspeed all'
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    fig.savefig(fname, format=fformat,dpi=150)

    print(fname)

if __name__ == '__main__':
    fformat = 'png'
    subfolder = 'behavior summary'

    slope_summary()
    # task_score_imaging()
    # running_speed_analysis()
