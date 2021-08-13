"""
identify transients

@author: Lukas Fischer

"""

import matplotlib.pyplot as plt
import matplotlib.cbook
plt.rcParams['svg.fonttype'] = 'none'
import warnings, sys, os, yaml, time, threading, json
# warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
# os.chdir('C:/Users/Lou/Documents/repos/LNT')
warnings.filterwarnings("ignore")
from scipy.signal import butter, filtfilt
import seaborn as sns
import numpy as np
import scipy as sp
import scipy.io as sio
from multiprocessing import Process
#  from skimage.filters import threshold_otsu
from oasis.functions import deconvolve, estimate_parameters
from oasis.oasis_methods import oasisAR1
# import statsmodels.api as sm
#import ipdb


with open('../' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from licks import licks
from rewards import rewards
from event_ind import event_ind
from write_dict import write_dict
from filter_trials import filter_trials
import seaborn as sns
sns.set_style('white')

fformat = 'png'


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_ind_trace_behavior(mouse, sess, roi, subfolder=[], baseline_percentile=70, plot_interactive=False,s_min=0.55, make_figure=False, corr_neuropil=True, load_raw=False):

    # h5path = loc_info['raw_dir'] + mouse + '/' + mouse + '.h5'
    processed_data_path = loc_info['raw_dir'] + mouse + os.sep + sess + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    behav_ds = loaded_data['behaviour_aligned']
    dF_ds = loaded_data['dF_aligned']
    # remove times when mouse was in the black box
    # no_blackbox_trials = filter_trials(behav_ds, [], ['tracknumber', 3])
    # no_blackbox_trials = np.union1d(no_blackbox_trials, filter_trials(behav_ds, [], ['tracknumber', 4]))
    # behav_ds = behav_ds[np.in1d(behav_ds[:, 4], [3, 4]), :]
    # rewards_ds = rewards(behav_ds)
    # lick_ds,_ = licks(behav_ds, rewards_ds)
    # licks_ds = loaded_data['/licks_pre_reward']

    if make_figure:
        fig = plt.figure(figsize=(30,15))
        # fig.suptitle(fname)
        ax1 = plt.subplot2grid((4,100),(0,0), rowspan=2, colspan=50)
        ax2 = plt.subplot2grid((4,100),(2,0), rowspan=2, colspan=50, sharex=ax1)
        ax3 = plt.subplot2grid((4,100),(0,50), rowspan=2, colspan=25)
        ax4 = plt.subplot2grid((4,100),(0,75), rowspan=2, colspan=25)
        ax5 = plt.subplot2grid((4,100),(2,50), rowspan=1, colspan=25)
        ax6 = plt.subplot2grid((4,100),(3,75), rowspan=1, colspan=25)
        ax7 = plt.subplot2grid((4,100),(3,50), rowspan=1, colspan=25)
        ax8 = plt.subplot2grid((4,100),(2,75), rowspan=1, colspan=25)

        ax1.spines['left'].set_linewidth(2)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            bottom='on', \
            right='off', \
            top='off')

        ax2.spines['left'].set_linewidth(2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            bottom='on', \
            right='off', \
            top='off')

        ax6.spines['left'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        # ax6.set_xticklabels([''])
        # ax6.set_yticklabels([''])
        ax6.tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            bottom='off', \
            right='off', \
            top='off')

    # set standard deviation threshold above which traces have to be to be considered a transient
    std_threshold = 6
    # set minimum transient length in seconds that has to be above threshold
    min_transient_length = 0.5

    # calculate frame to frame latency
    frame_latency = 1/(dF_ds.shape[0]/(behav_ds[-1,0] - behav_ds[0,0]))

    # min transient length in number of frames
    min_transient_length = min_transient_length/frame_latency

    # low pass filter trace
    order = 6
    fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz
    roi_filtered = butter_lowpass_filter(dF_ds[:,roi], cutoff, fs, order)
    roi_unfiltered = dF_ds[:,roi]

    # get standard deviation of lower 80% of samples
    percentile_low_idx = np.where(roi_filtered < np.percentile(roi_filtered,baseline_percentile))[0]

    roi_std = np.std(roi_filtered[percentile_low_idx])
    roi_mean = np.mean(roi_filtered[percentile_low_idx])
    # ax1.axhline((std_threshold*roi_std)+roi_mean,ls='--',c='0.8',lw=2)
    # ax1.axhline(roi_mean,ls='--',c='g',lw=2,alpha=0.5)

    # calculate correlation between speed and dF/F
    # moving_idx = np.where(behav_ds[:,3] > 3)[0]
    # speed_slope, speed_intercept, lo_slope, up_slope = sp.stats.theilslopes(dF_ds[moving_idx,roi], behav_ds[moving_idx,3])
    # print('--- DF/F:SPEED PEARSONR ---')
    # print(speed_slope, speed_intercept)
    # print(sp.stats.pearsonr(behav_ds[moving_idx,3],dF_ds[moving_idx,roi]))
    # print('---------------------------')
    # ax5.scatter(behav_ds[moving_idx,3],dF_ds[moving_idx,roi])
    # ax5.plot(behav_ds[moving_idx,3], speed_intercept+speed_slope * behav_ds[moving_idx,3], lw=2,c='r')
    # ax5.set_xlim([-1,90])

    # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,roi],c='r',lw=1)
    roi_idx = np.arange(0,len(dF_ds[:,roi]),1)
    if make_figure:
        ax1.plot(roi_idx, roi_unfiltered,label='dF/F',c='k',lw=1)

    # get indeces above speed threshold
    transient_high = np.where(roi_filtered > (std_threshold*roi_std)+roi_mean)[0]
    if transient_high.size == 0:
        if make_figure:
            fname = 'ind_trace' + mouse + '_' + sess + '_' + str(roi)
            # fig.suptitle(fname, wrap=True)
            if subfolder != []:
                if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
                    os.mkdir(loc_info['figure_output_path'] + subfolder)
                fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
            else:
                fname = loc_info['figure_output_path'] + fname + '.' + fformat
            print(fname)
            try:
                fig.savefig(fname, format=fformat,dpi=150)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # ax1.plot(roi_idx[transient_high], roi_filtered[transient_high],c='r',lw=1)
    if make_figure:
        ax1.plot(roi_idx[percentile_low_idx], roi_filtered[percentile_low_idx],c='g',lw=1)

    # # use diff to find gaps between episodes of high speed
    idx_diff = np.diff(transient_high)
    idx_diff = np.insert(idx_diff,0,transient_high[0])

    # convert gap tolerance from cm to number of frames
    gap_tolerance_frames = int(0.1/frame_latency)
    # find indeces where speed exceeds threshold. If none are found, return

    # find transient onset and offset points. Colapse transients that briefly dip below threshold
    onset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]]-1
    offset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]-1]+1
    # this is necessary as the first index of the offset is actually the end of the last one (this has to do with indexing conventions on numpy)
    offset_idx = np.roll(offset_idx, -1)

    # calculate the length of each transient and reject those too short to be considered
    index_adjust = 0
    for i in range(len(onset_idx)):
        temp_length = offset_idx[i-index_adjust] - onset_idx[i-index_adjust]
        if temp_length < min_transient_length:
            onset_idx = np.delete(onset_idx,i-index_adjust)
            offset_idx = np.delete(offset_idx,i-index_adjust)
            index_adjust += 1

    # find the onset by looking for the point where the transient drops below 1 std
    above_1_std = np.where(roi_filtered > (2*roi_std)+roi_mean)[0]
    one_std_idx_diff = np.diff(above_1_std)
    one_std_idx_diff = np.insert(one_std_idx_diff,0,one_std_idx_diff[0])

    one_std_onset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]]-1
    one_std_offset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]-1]+1

    onsets = []
    offsets = []
    for oi in onset_idx:
        closest_idx = one_std_onset_idx - oi
        closest_idx_neg = np.where(closest_idx < 0)[0]
        if closest_idx_neg.size == 0:
            closest_idx_neg = [-1]
            one_std_idx = 0
        else:
            one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
        # one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
        onsets.append(oi-one_std_idx)
        # ax1.axvline(oi-one_std_idx, c='0.6', lw=0.5, ls='--')

    for oi in offset_idx:
        closest_idx = one_std_offset_idx - oi
        closest_idx_neg = np.where(closest_idx > 0)[0]
        if closest_idx_neg.size == 0:
            closest_idx_neg = [-1]
            one_std_idx = 0
        else:
            one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
        offsets.append(oi+one_std_idx)
        # ax1.axvline(oi+one_std_idx, c='0.6', lw=0.5, ls='--')

    # find max transient length
    max_transient_length = 0
    for i in range(len(onsets)):
        if offsets[i]-onsets[i] > max_transient_length:
            max_transient_length = offsets[i]-onsets[i]

    all_transients = np.full((len(onsets),max_transient_length), np.nan)
    all_transients_norm = np.full((len(onsets),max_transient_length), np.nan)
    transient_speed = np.full((len(onsets),max_transient_length), np.nan)
    for i in range(len(onsets)):
        all_transients[i,0:offsets[i]-onsets[i]] = roi_unfiltered[onsets[i]:offsets[i]]
        all_transients_norm[i,0:offsets[i]-onsets[i]] = (roi_unfiltered[onsets[i]:offsets[i]]-roi_unfiltered[onsets[i]])/(np.nanmax(roi_unfiltered[onsets[i]:offsets[i]]-roi_unfiltered[onsets[i]]))
        transient_speed[i,0:offsets[i]-onsets[i]] = behav_ds[onsets[i]:offsets[i],3]
        # print((roi_unfiltered[onsets[i]:offsets[i]]-roi_unfiltered[onsets[i]])/(np.nanmax(roi_unfiltered[onsets[i]:offsets[i]]-roi_unfiltered[onsets[i]])))

    all_transient_values = np.empty(0)
    all_peak_transient_values = np.empty(0)
    all_avg_speed_values = np.empty(0)
    all_speed_values = np.empty(0)
    if make_figure:
        for i in range(len(onsets)):
            ax1.plot(np.arange(onsets[i],offsets[i],1),roi_unfiltered[onsets[i]:offsets[i]],c='r',lw=1.5)

        # for i in range(all_transients.shape[0]):
        #     ax3.plot(all_transients[i],c='0.8',lw=0.5)
        #     ax4.plot(all_transients_norm[i],c='0.8',lw=0.5)
        #     all_transient_values = np.concatenate((all_transient_values,all_transients[i][~np.isnan(all_transients[i])]))
        #     all_speed_values = np.concatenate((all_speed_values,transient_speed[i][~np.isnan(transient_speed[i])]))
        #     # ipdb.set_trace()
        #     all_peak_transient_values = np.concatenate((all_peak_transient_values, np.array([np.nanmax(all_transients[i])])))
        #     all_avg_speed_values = np.concatenate((all_avg_speed_values, np.array([np.nanmean(transient_speed[i])])))
        #
        # ax3.plot(np.nanmean(all_transients,0),c='k', lw=2)
        # ax4.plot(np.nanmean(all_transients_norm,0),c='k', lw=1)
        #
        # ax7.scatter(all_speed_values, all_transient_values)
        # ax8.scatter(all_avg_speed_values, all_peak_transient_values)
        # peak_slope, peak_intercept, lo_slope, up_slope = sp.stats.theilslopes(all_peak_transient_values, all_avg_speed_values)
        # ax8.plot(all_avg_speed_values, peak_intercept+peak_slope * all_avg_speed_values, lw=2,c='r')
        # # ipdb.set_trace()
        # transient_slope, transient_intercept, lo_slope, up_slope = sp.stats.theilslopes(all_transient_values, all_speed_values)
        # print('--- TRANSIENT DF/F:SPEED PEARSONR ---')
        # print(transient_intercept, transient_intercept)
        # print(sp.stats.pearsonr(all_speed_values, all_transient_values))
        # print(sp.stats.pearsonr(all_avg_speed_values, all_peak_transient_values))
        # print('---------------------------')
        # ax7.plot(all_speed_values, transient_intercept+transient_slope * all_speed_values, lw=2,c='r')
        # ax5.set_xlim([-1,90])
        # ax7.set_xlim([-1,90])

        if corr_neuropil:
            # calculate roi to neuropil correlation
            rio_raw_F = roi_raw[:,roi]
            pil_raw_F = pil_raw[:,roi]
            pil_raw_F_exog = sm.add_constant(pil_raw_F)

            # olsres = sm.OLS(rio_raw_F,pil_raw_F_exog,M=sm.robust.norms.TukeyBiweight()).fit()
            # rlmres = sm.RLM(rio_raw_F,pil_raw_F_exog,M=sm.robust.norms.RamsayE()).fit()
            # rlmresAW = sm.RLM(rio_raw_F,pil_raw_F_exog,M=sm.robust.norms.AndrewWave()).fit()
            # rlmresHT = sm.RLM(rio_raw_F,pil_raw_F_exog,M=sm.robust.norms.HuberT()).fit()
            # rlmresHM = sm.RLM(rio_raw_F,pil_raw_F_exog,M=sm.robust.norms.Hampel()).fit()
            # rlmresBI = sm.RLM(rio_raw_F,pil_raw_F_exog,M=sm.robust.norms.TukeyBiweight()).fit()
            # rlmresTM = sm.RLM(rio_raw_F,pil_raw_F_exog,M=sm.robust.norms.TrimmedMean()).fit()
            #
            # ax5.plot(pil_raw_F,rlmres.fittedvalues,label='RamsayE')
            # ax5.plot(pil_raw_F,rlmresAW.fittedvalues,label='AndrewWave')
            # ax5.plot(pil_raw_F,rlmresHT.fittedvalues,label='HuberT')
            # ax5.plot(pil_raw_F,rlmresHM.fittedvalues,label='Hampel')
            # ax5.plot(pil_raw_F,rlmresBI.fittedvalues,label='TukeyBiweight')
            # ax5.plot(pil_raw_F,rlmresTM.fittedvalues,label='TrimmedMean')
            # ax5.legend(loc="best")

    # if we want to plot interactive, only take a random subsample of datapoints, otherwise the display gets really slow
    if make_figure and corr_neuropil:
        if plot_interactive:
            dF_idx = np.random.choice(len(rio_raw_F),1000)
            ax5.scatter(pil_raw_F[dF_idx],rio_raw_F[dF_idx])
        else:
            ax5.scatter(pil_raw_F,rio_raw_F)

        max_xy_val = np.amax([pil_raw_F,rio_raw_F])
        ax5.set_xlim([0,max_xy_val])
        ax5.set_ylim([0,max_xy_val])
        # ax5.set_xlabel('neuropil values')
        # ax5.set_ylabel('roi values')

    # calculate full width half maximum (FWHM), peak of transients and AUC
    fwhm = []
    transient_peaks = []
    auc = []
    for i in range(len(onsets)):
        # get half max value
        transient_min = np.min(roi_filtered[onsets[i]:offsets[i]])
        transient_max = np.max(roi_filtered[onsets[i]:offsets[i]])
        half_amplitude = (transient_max - transient_min)/2
        above_half_amp_idx = np.where(roi_filtered[onsets[i]:offsets[i]] > half_amplitude)[0]
        fwhm.append(above_half_amp_idx.shape[0] * frame_latency)
        transient_peaks.append(np.amax(roi_filtered[onsets[i]:offsets[i]]))
        auc.append(sp.integrate.simps(roi_filtered[onsets[i]:offsets[i]], behav_ds[onsets[i]:offsets[i],0]))


    # estimate spike times by using deconvolution
    # ax2.plot(roi_filtered)
    # m = np.mean(roi_filtered,axis=0)
    # m02 = np.mean(np.power(roi_filtered,2),axis=0)
    # m12 = np.mean(np.multiply(roi_filtered[1:-2],roi_filtered[0:-3]),axis=0)
    # a = ((m**2)-m12)/((m**2)-m02)                   # alpha
    # uhat = [roi_filtered[1:-1] - a*roi_filtered[0:-2]]
    # uhat = np.insert(uhat,0,0)                      # spike train
    # shat = np.where(uhat > threshold_otsu(uhat))    # estimated spike train
    #
    # for l in shat[0]:
    #     ax2.axvline(l)
    #
    # print(shat[0].shape[0]/1800)

    # estimate parameters of AR process and noise power, the apply oasisAR2

    # deconvolve applies oasisAR1 of len(g)==1 and oasisAR2 if len(g)==2
    # calling deconvolve like this will call constrained_oasisAR1() with some default paramters. Here we use it to get a baseline estimate (b)
    # fudge_factor = .98
    # g,sn = estimate_parameters(roi_unfiltered, p=1, fudge_factor=fudge_factor)
    # deconvolve (which in turn runs constrained_AR1) just to estimate the baseline (b)
    # c_g, s, b, _, lam = deconvolve(roi_unfiltered,g,sn)
    c_g, s, b, _, lam = deconvolve(roi_unfiltered,g=(None,None),penalty=0, optimize_g=5, max_iter=5)
    # now we call oasisAR1 manually so we can set the s_min parameter
    # c_AR1,s_AR1 = oasisAR1(roi_unfiltered, g[0], lam,  s_min)
    # to apply oasisAR2 we have to estimate g
    # c_AR2,s_AR2 = oasisAR1(roi_unfiltered, g, s_min=0.2)
    # c_foopsi,s_foopsi = foopsi(roi_unfiltered-b, g)

    if make_figure:
        if plot_interactive:
            ax1.plot(b+c_g, lw=2, label='denoised constrained AR1',c='b')
        # ax1.plot(b+c_nog, lw=2, label='denoised constrained AR1',c='b',ls='--')
    #    ax1.plot(b+c_AR2, lw=2, label='denoised AR2',c='b')
        ax1.legend()
        # ax2.plot(s_AR1, lw=1, label='AR1', c='b',zorder=2)
        ax2.plot(s, lw=1, label='deconvolved', c='g',zorder=2)
        # ax2.plot(s_AR2, lw=2, label='AR2', c='g', ls='--',zorder=3)
        # ax2.plot(s_foopsi, lw=2, label='foopsi', c='m', ls='--',zorder=2)

    spikes_idx = np.where(s > 0.1)[0]
    spiketrain = np.zeros(len(roi_unfiltered))
    spiketrain[spikes_idx] = 1
    if make_figure:
        ax2.plot(spiketrain, lw=1, label='spikes', c='r',zorder=2)

    # make sliding window (sec)
    sliding_window_size = [1,0]
    sliding_window_time = sliding_window_size[0] + sliding_window_size[1]
    sliding_window_idx = [int(np.round(sliding_window_size[0]/frame_latency,0)),int(np.round(sliding_window_size[1]/frame_latency,0))]
    inst_spikerate = np.zeros(len(roi_unfiltered))
    for i in range(len(spiketrain)):
        if i - sliding_window_idx[0] < 0:
            num_spikes = np.sum(spiketrain[0:i])
        elif i + sliding_window_idx[1] > len(spiketrain):
            num_spikes = np.sum(spiketrain[i:-1])
        else:
            num_spikes = np.sum(spiketrain[i-sliding_window_idx[0]:i+sliding_window_idx[1]])

        inst_spikerate[i] = num_spikes/sliding_window_time

    if make_figure:
        ax2.plot(inst_spikerate, label='spikerate')
        ax2.legend()


        #one_sec = (t_stop_idx-t_start_idx)/(t_stop - t_start)
        # ax1.set_xticks([0,5*one_sec])
        # ax1.set_xticklabels(['0','5'])

        # ax1.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
        # ax1.set_yticklabels(['0','2','4','6'])
        ax1.set_ylabel('dF/F', fontsize=16)

        # ax2.set_yticks([0,10,20,30,40])
        # ax2.set_yticklabels(['0','10','20','30','40'])
        # ax2.set_ylabel('speed (cm/sec)', fontsize=16)

        # ax1.set_ylim([-0.1,0.6])
        # ax2.set_ylim([-5,40])

    transients_per_min = (len(onsets)/(behav_ds[-1,0] - behav_ds[0,0]))*60
    mean_spikerate = sum(s)/(behav_ds[-1,0] - behav_ds[0,0])
    norm_value = np.amax(roi_filtered)

    if make_figure:
        # print(sum(s))
        ax6.text(0.1,0.9,'transients per minute: ' + str(round(transients_per_min,2)))
        ax6.text(0.1,0.875,'estimated mean spikerate (Hz): ' + str(round(mean_spikerate,2)))
        ax6.text(0.1,0.85,'transient FWHM (s): ' + str(round(np.mean(fwhm),2)) + ' +/- ' + str(round(np.std(fwhm)/np.sqrt(len(fwhm)),2)) + ' (SEM)')
        # if corr_neuropil:
            # ax6.text(0.1,0.825,'neurpil regression results - slope: ' + str(round(rlmresBI.params[1],2)) + ' intercept: ' + str(round(rlmresBI.params[0],2)))
        ax6.text(0.1,0.8,'normalization value: ' + str(round(norm_value,2)))
        fig.tight_layout()

        fname = 'ind_trace' + mouse + '_' + sess + '_' + str(roi)


        # fig.suptitle(fname, wrap=True)
        if subfolder != []:
            if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
                os.mkdir(loc_info['figure_output_path'] + subfolder)
            fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
        else:
            fname = loc_info['figure_output_path'] + fname + '.' + fformat
        print(fname)
        try:
            fig.savefig(fname, format=fformat,dpi=150)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

        if plot_interactive:
            plt.show(fig)

        #
        plt.close(fig)
    # print(s_AR1.shape,spiketrain.shape,inst_spikerate.shape,len(fwhm), np.mean(fwhm),np.std(fwhm)/np.sqrt(len(fwhm)),len(onset_idx)/30)
    return s,spiketrain,inst_spikerate,fwhm, np.mean(fwhm), np.std(fwhm)/np.sqrt(len(fwhm)), transients_per_min, transient_peaks, auc, np.mean(auc), norm_value

def run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=False, skip_rois=[]):
    # load datasets to determine some parameters
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    behav_ds = loaded_data['behaviour_aligned']
    dF_ds = loaded_data['dF_aligned']


    write_to_dict = True
    make_figure = True
    write_to_h5 = False
    write_to_raw = True
    corr_neuropil = False
    plot_interactive = False

    frame_latency = 1/(dF_ds.shape[0]/(behav_ds[-1,0] - behav_ds[0,0]))
    total_num_rois = dF_ds.shape[1]

    if rois == 'all':
        # write_to_dict = True
        rois = total_num_rois
        rois = np.arange(0,total_num_rois,1)
        print('ALL rois processed. Number of rois: ' + str(rois))
    else:
        rois = np.array(rois)

    # set up dictionary and empty matrices to store data
    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'roi_numbers' : rois.tolist(),
        'FWHM' : [],
        'FWHM_mean' : [],
        'FWHM_sem' : [],
        'norm_value_all' : [],
        'transient_rate' : [],
        'transient_peaks' : [],
        'transient_AUC' : [],
        'transient_AUC_mean' : []
    }
    all_transient_rates = []
    df_deconvolved = np.zeros(dF_ds.shape)
    spiketrain = np.zeros(dF_ds.shape)
    spikerate = np.zeros(dF_ds.shape)
    # run through ROIs and store data of analysis
    for roi in rois:
        if not roi in skip_rois:
            print(MOUSE + ' ' + SESSION + ' ' + 'roi: ' + str(roi))
            df_deconvolved[:,roi],spiketrain[:,roi],spikerate[:,roi],fwhm, fwhm_mean, fwhm_sem, transient_rate, transient_peaks, auc, auc_mean, norm_value = \
                plot_ind_trace_behavior(MOUSE, SESSION, roi, MOUSE+'_'+SESSION+'_idtrace', baseline_percentage, plot_interactive, s_min, make_figure, corr_neuropil, load_raw)
            roi_result_params['FWHM'].append(fwhm)
            roi_result_params['FWHM_mean'].append(fwhm_mean)
            roi_result_params['FWHM_sem'].append(fwhm_sem)
            roi_result_params['transient_rate'].append(transient_rate)
            roi_result_params['transient_peaks'].append(transient_peaks)
            roi_result_params['norm_value_all'].append(norm_value)
            roi_result_params['transient_AUC'].append(auc)
            roi_result_params['transient_AUC_mean'].append(auc_mean)
            all_transient_rates.append(transient_rate)

    if write_to_raw:
        loaded_data['spiketrain'] = spiketrain
        loaded_data['spikerate'] = spikerate
        loaded_data['df_deconvolved'] = df_deconvolved
        sio.savemat(processed_data_path + '.mat', mdict=loaded_data)
        print('saved ' + processed_data_path + '.mat')

    # write to dict
    if write_to_dict:
        write_dict(MOUSE, SESSION, roi_result_params)

def run_SIM_1_Day1():
    MOUSE = 'SIM_1'
    SESSION = 'Day1'
    rois = np.arange(0,2,1) #'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, True)

def run_sim_glm_Day1():
    MOUSE = 'SIM_2'
    SESSION = 'Day1'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF180119_1_Day2018424_1():
    MOUSE = 'LF180119_1'
    SESSION = 'Day2018424_1'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF180119_1_Day2018424_2():
    MOUSE = 'LF180119_1'
    SESSION = 'Day2018424_2'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF170110_1_Day20170215_l23():
    MOUSE = 'LF170110_1'
    SESSION = 'Day20170215_l23'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF170110_1_Day20170215_l5():
    MOUSE = 'LF170110_1'
    SESSION = 'Day20170215_l5'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF170110_2_Day20170209_l23():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170209_l23'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF170110_2_Day20170209_l5():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170209_l5'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF170110_2_Day20170213_l23():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170213_l23'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF170110_2_Day20170213_l5():
    MOUSE = 'LF170110_2'
    SESSION = 'Day20170213_l5'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF161202_1_Day20170209_l23():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l23'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF161202_1_Day20170209_l5():
    MOUSE = 'LF161202_1'
    SESSION = 'Day20170209_l5'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)

def run_LF170613_1_Day20170804():
    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170420_1_Day2017719():
    MOUSE = 'LF170420_1'
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = 'Day2017719_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170420_1_Day201783():
    MOUSE = 'LF170420_1'
    SESSION = 'Day201783'
    SESSION_OPENLOOP = 'Day201783_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.3
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170421_2_Day20170719():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170719'
    SESSION_OPENLOOP = 'Day20170719_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170421_2_Day20170720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day20170720'
    SESSION_OPENLOOP = 'Day20170720_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170421_2_Day2017720():
    MOUSE = 'LF170421_2'
    SESSION = 'Day2017720'
    SESSION_OPENLOOP = 'Day2017720_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170222_1_Day201776():
    MOUSE = 'LF170222_1'
    SESSION = 'Day201776'
    SESSION_OPENLOOP = 'Day201776_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170222_1_Day2017615():
    MOUSE = 'LF170222_1'
    SESSION = 'Day2017615'
    SESSION_OPENLOOP = 'Day201776_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170110_2_Day201748_1():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_1'
    SESSION_OPENLOOP = 'Day201748_openloop_1'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170110_2_Day201748_2():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_2'
    SESSION_OPENLOOP = 'Day201748_openloop_2'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF170110_2_Day201748_3():
    MOUSE = 'LF170110_2'
    SESSION = 'Day201748_3'
    SESSION_OPENLOOP = 'Day201748_openloop_3'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF171212_2_Day2018218_1():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_1'
    SESSION_OPENLOOP = 'Day2018218_openloop_1'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF171212_2_Day2018218_2():
    MOUSE = 'LF171212_2'
    SESSION = 'Day2018218_2'
    SESSION_OPENLOOP = 'Day2018218_openloop_2'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)


def run_LF171211_1_Day2018321_2():
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    SESSION_OPENLOOP = 'Day2018321_openloop_2'
    rois = 'all'
    baseline_percentage = 30
    s_min = 0.6
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)


def run_LF180112_2_Day2018322_2():

    MOUSE = 'LF180112_2'
    SESSION = 'Day2018322_2'
    ROI = np.arange(0,283,1)

    roi_result_params = {
        'mouse_session' : MOUSE+'_'+SESSION,
        'FWHM' : [],
        'FWHM_mean' : [],
        'FWHM_sem' : []
    }

    for r in ROI:
        fwhm, fwhm_mean, fwhm_sem, transient_rate = plot_ind_trace_behavior(MOUSE, SESSION, r, 0, 100000, MOUSE+'_'+SESSION+'_idtrace') #1285, 1367
        roi_result_params['FWHM'].append(fwhm)
        roi_result_params['FWHM_mean'].append(fwhm_mean)
        roi_result_params['FWHM_sem'].append(fwhm_sem)

    update_dict(MOUSE, SESSION, roi_result_params)

def run_LF180112_2_Day2018424_1():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_1'
    SESSION_OPENLOOP = 'Day2018424_openloop_1'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF180112_2_Day2018424_2():
    MOUSE = 'LF180112_2'
    SESSION = 'Day2018424_2'
    SESSION_OPENLOOP = 'Day2018424_openloop_2'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)


def run_LF171211_2_Day201852():
    MOUSE = 'LF171211_2'
    SESSION = 'Day201852'
    SESSION_OPENLOOP = 'Day201852_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF180219_1_Day2018424_0025():
    MOUSE = 'LF180219_1'
    SESSION = 'Day2018424_0025'
    SESSION_OPENLOOP = 'Day2018424_openloop_0025'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_LF171211_2_Day201852_matched():
    MOUSE = 'LF171211_2'
    SESSION = 'Day201852_matched'
    SESSION_OPENLOOP = 'Day201852_openloop_matched'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_20170214_1_Day201777():
    MOUSE = 'LF170214_1'
    SESSION = 'Day201777'
    SESSION_OPENLOOP = 'Day201777_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_20170214_1_Day2017714():
    MOUSE = 'LF170214_1'
    SESSION = 'Day2017714'
    SESSION_OPENLOOP = 'Day2017714_openloop'
    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2
    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min)

def run_20191022_3_20191119():
    MOUSE = 'LF191022_3'
    SESSION = '20191119'
    SESSION_OPENLOOP = '20191119_ol'
    use_data = 'aligned_data.mat'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    # run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191022_3_20191204():
    MOUSE = 'LF191022_3'
    SESSION = '20191204'
    SESSION_OPENLOOP = None
    use_data = 'aligned_data.mat'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    # run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191023_blue_20191119():
    MOUSE = '20191023_blue'
    SESSION = '20191119'
    SESSION_OPENLOOP = None
    use_data = 'aligned_data.mat'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    # run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191023_blue_20191204():
    MOUSE = 'LF191023_blue'
    SESSION = '20191204'
    SESSION_OPENLOOP = None
    use_data = 'aligned_data.mat'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    # run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191023_blue_20191208():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    SESSION_OPENLOOP = None
    use_data = 'aligned_data.mat'

    rois = [16,17,18] #'all'
    skip_rois = [17]
    baseline_percentage = 50
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, True, skip_rois)
    # run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191024_1_20191115():
    MOUSE = '20191024_1'
    SESSION = '20191115'
    SESSION_OPENLOOP = None
    use_data = 'aligned_data.mat'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    # run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191024_1_20191204():
    MOUSE = '20191024_1'
    SESSION = '20191204'
    SESSION_OPENLOOP = None
    use_data = 'aligned_data.mat'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    # run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191023_blank_20191116():
    MOUSE = 'LF191023_blank'
    SESSION = '20191116'
    SESSION_OPENLOOP = '20191116_ol'
    use_data = 'aligned_data.mat'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191022_2_20191116():
    MOUSE = 'LF191022_2'
    SESSION = '20191116'
    use_data = 'aligned_data.mat'
    SESSION_OPENLOOP = '20191116_ol'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)


def run_20191022_1_20191115():
    MOUSE = 'LF191022_1'
    SESSION = '20191115'
    use_data = 'aligned_data.mat'
    SESSION_OPENLOOP = ''

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    
def run_20191022_1_20191204():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = 'aligned_data.mat'
    SESSION_OPENLOOP = SESSION + '_ol'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)

def run_20191022_1_20191210():
    MOUSE = 'LF191023_blue'
    SESSION = '20191210'
    use_data = 'aligned_data.mat'
    SESSION_OPENLOOP = SESSION + '_ol'

    rois = 'all'
    baseline_percentage = 70
    s_min = 0.2

    run_analysis(MOUSE, SESSION, rois, baseline_percentage, s_min, load_raw=True)
    run_analysis(MOUSE, SESSION_OPENLOOP, rois, baseline_percentage, s_min, load_raw=True)


def summary_plot():
    roi_param_list = ['/Users/lukasfischer/Work/exps/MTH3/figures/LF170214_1_Day201777/roi_FWHM.json',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF170214_1_Day2017714/roi_FWHM.json',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF171211_2_Day201852/roi_FWHM.json',
                      '/Users/lukasfischer/Work/exps/MTH3/figures/LF180112_2_Day2018322_2/roi_FWHM.json'
                     ]


    # create figure and axes to later plot on
    fig = plt.figure(figsize=(6,6))
    ax1 = plt.subplot(111)

    # load all fwhm mean value
    fwhm_all = []
    for rpl in roi_param_list:
        with open(rpl,'r') as f:
            roi_params = json.load(f)
        fwhm_all.append(roi_params['FWHM_mean'])

    # flatten list
    fwhm_all = [item for sublist in fwhm_all for item in sublist]
    fwhm_all = np.array(fwhm_all)
    fwhm_all = fwhm_all[~np.isnan(fwhm_all)]
    sns.distplot(fwhm_all, bins=np.arange(0,6.2,0.2), kde=False, color='red')

    subfolder = 'fwhm'
    fname = 'v1_fwhm'

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params( \
        reset='on',
        axis='both', \
        direction='in', \
        labelsize=32, \
        length=4, \
        width=2, \
        bottom='off', \
        right='off', \
        top='off')

    ax1.set_xlabel('Transient FWHM (sec)', fontsize=24)
    ax1.set_ylabel('', fontsize=24)


    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    plt.close(fig)

def plot_single():
    # MOUSE = 'LF170613_1'
    # SESSION = 'Day20170804'
    MOUSE = 'LF171211_1'
    SESSION = 'Day2018321_2'
    # MOUSE = 'LF170421_2'
    # SESSION = 'Day20170719'
    # MOUSE = 'LF170421_2'
    # SESSION = 'Day20170720'
    # MOUSE = 'LF170420_1'
    # SESSION = 'Day201783'
    # MOUSE = 'LF170222_1'
    # SESSION = 'Day201776'
    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_3'
    # MOUSE = 'LF171212_2'
    # SESSION = 'Day2018218_1'
    r = 5
    s_min = 0.2
    baseline_percentage = 70

    plot_ind_trace_behavior(MOUSE, SESSION, r, MOUSE+'_'+SESSION+'_idtrace',baseline_percentage, True,s_min)


if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    fformat = 'png'

#    run_SIM_1_Day1()

    # run_20191022_3_20191119()
    # run_20191023_blue_20191119()
    # run_20191023_blue_20191204()
    # run_20191022_1_20191115()
    # run_20191022_2_20191116()
    # run_20191023_blank_20191116()

    # run_20191024_1_20191115()
    # run_20191024_1_20191204()

    # run_20191022_1_20191210()
    run_20191023_blue_20191208()
    
#    flist = [
#            run_20191023_blue_20191119, \
#            run_20191023_blue_20191204, \
#            run_20191024_1_20191115, \
#            run_20191024_1_20191204, \
#            run_20191022_3_20191119, \
#            run_20191022_3_20191204
#            ]

    # flist = [run_20191022_3_20191119, \
    #          run_20191023_blue_20191119, \
    #          run_20191022_2_20191116, \
    #          run_20191023_blank_20191116
    #         ]

    # run_LF170110_2_Day20170213_l23()
    # run_LF170110_2_Day20170213_l5()

    # run_LF170110_1_Day20170215_l23()
    # run_LF170110_1_Day20170215_l5()

    # run_LF170110_2_Day20170209_l23()
    # run_LF170110_2_Day20170209_l5()

    # run_LF161202_1_Day20170209_l23()
    # run_LF161202_1_Day20170209_l5()

    #
    # test2
    # RUN:
    # run_LF170613_1_Day20170804()
    # run_LF170421_2_Day20170719()
    # run_LF170421_2_Day2017720()
    # run_LF170420_1_Day2017719()
    # run_LF170420_1_Day201783()
    # run_LF170222_1_Day2017615()
    # run_LF170222_1_Day201776()
    # run_LF170110_2_Day201748_1()
    # run_LF170110_2_Day201748_2()
    # run_LF170110_2_Day201748_3()
    # run_LF171212_2_Day2018218_2()
    # run_LF171211_1_Day2018321_2()

    # V1 BOUTONS
    # run_20170214_1_Day201777()
    # run_20170214_1_Day2017714()
    # run_LF171211_2_Day201852()
    # run_LF171211_2_Day201852_matched()
    # run_LF180119_1_Day2018424_1()
    # run_LF180119_1_Day2018424_2()
    # flist = [run_LF180112_2_Day2018424_1,
    #          run_LF180112_2_Day2018424_2,
    #          run_LF171211_2_Day201852,
    #          run_20170214_1_Day201777,
    #          run_20170214_1_Day2017714
    #         ]


    # LAYER 2/3 AND LAYER 5 SIMULTANEOUS
    # flist = [#run_LF161202_1_Day20170209_l23,
    #          #run_LF161202_1_Day20170209_l5,
    #          run_LF170110_2_Day20170209_l23,
    #          run_LF170110_2_Day20170209_l5,
    #          run_LF170110_1_Day20170215_l23,
    #          run_LF170110_1_Day20170215_l5,
    #         ]

    # run_LF180219_1_Day2018424_0025()

    # LAYER 2/3 AND LAYER 5 INDIVIDUAL
    # flist = [
    #          run_LF170613_1_Day20170804, \
    #          run_LF170421_2_Day2017720, \
    #          run_LF170421_2_Day20170719, \
    #          run_LF170110_2_Day201748_1, \
    #          run_LF170110_2_Day201748_2, \
    #          run_LF170110_2_Day201748_3, \
    #          run_LF170420_1_Day2017719, \
    #          run_LF170420_1_Day201783, \
    #          run_LF170222_1_Day201776, \
    #          # run_LF171211_1_Day2018321_2, \
    #          run_LF171212_2_Day2018218_2, \
    #          run_LF161202_1_Day20170209_l23, \
    #          run_LF161202_1_Day20170209_l5, \
    #          run_LF170222_1_Day2017615
    # ]

    # jobs = []
    # for fl in flist:
    #     p = Process(target=fl)
    #     jobs.append(p)
    #     p.start()
        #
    # for j in jobs:
    #     j.join()

    # run_LF170421_2_Day20170720()
    # plot_single()

    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # r = 40
    #
    # plot_ind_trace_behavior(MOUSE, SESSION, r, 1000, 1100, MOUSE+'_'+SESSION+'_idtrace') #1285, 1367
    #
    # SESSION = 'Day2017714'
    # r = 41
    #
    # plot_ind_trace_behavior(MOUSE, SESSION, r, 1020, 1120, MOUSE+'_'+SESSION+'_idtrace') #1285, 1367

    # run_LF171211_2_Day201852()
    # run_LF180112_2_Day2018322_2()
    # summary_plot()


### DUMP
# shade areas corresponding to the landmark
# landmark = [200,240]
# lm_temp = behav_ds[:,1]
# lm_start_idx = np.where(lm_temp > landmark[0])[0]
# lm_end_idx = np.where(lm_temp < landmark[1])[0]
# lm_idx = np.intersect1d(lm_start_idx,lm_end_idx)
# lm_diff = np.diff(lm_idx)
# lm_end = np.where(lm_diff>1)[0]
# lm_start = np.insert(lm_end,0,0)+1
# lm_end = np.append(lm_end,lm_idx.size-1)
# if lm_start.size > lm_end.size:
#     lm_end.append(np.size(behav_ds),0)
#
# for i,lm in enumerate(lm_start):
#     if behav_ds[lm_idx[lm],4]!=5:
#         if lm_idx[lm_start[i]] > t_start_idx and lm_idx[lm_start[i]] < t_stop_idx:
#             if behav_ds[lm_idx[lm],4] == 3:
#                 ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='0.9')
#             else:
#                 ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='0.7')
