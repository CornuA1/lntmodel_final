"""
plot section of individual traces during behaviour

@author: Lukas Fischer

"""

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import seaborn as sns
import numpy as np
import warnings
import h5py
import sys, os
import yaml

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

from filter_trials import filter_trials
from event_ind import event_ind
from load_filelist_model import load_filelist
import seaborn as sns
sns.set_style('white')

fformat = 'svg'

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_ind_trace_behavior(mouse, sess, roi, t_start, t_stop, ylims =[], plot_task_feature='landmark_shaded'):
    filedict = load_filelist()
    h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
    print(h5path)
    print(mouse, sess)
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    # sess = 'Day201748_2'
    # h5dat = h5py.File(h5path, 'r')
    # behav_ds2 = np.copy(h5dat[sess + '/behaviour_aligned'])
    # dF_ds2 = np.copy(h5dat[sess + '/dF_win'])
    # h5dat.close()

    # set up figure
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
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
        left='on', \
        bottom='on', \
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
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    t_start_idx = (np.abs(behav_ds[:,0] - t_start)).argmin()
    t_stop_idx = (np.abs(behav_ds[:,0] - t_stop)).argmin()


    # ax1.plot(dF_ds2[t_start_idx:t_stop_idx,roi],c='r',lw=1)

    # filter and plot running speed trace
    order = 6
    fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz
    if sess.find('openloop') > 0:
        speed_filtered = butter_lowpass_filter(behav_ds[:,8], cutoff, fs, order)
    else:
        speed_filtered = butter_lowpass_filter(behav_ds[:,3], cutoff, fs, order)
    ax2.plot(speed_filtered[t_start_idx:t_stop_idx],c='g',lw=2)

    # plot dF trace
    cutoff = 3
    dF_ds[:,roi] = butter_lowpass_filter(dF_ds[:,roi], cutoff, fs, order)
    ax1.plot(dF_ds[t_start_idx:t_stop_idx,roi],c='k',lw=2,zorder=3)

    if plot_task_feature is 'landmark_shaded':
        # shade areas corresponding to the landmark
        landmark = [200,240]
        lm_temp = behav_ds[:,1]
        lm_start_idx = np.where(lm_temp > landmark[0])[0]
        lm_end_idx = np.where(lm_temp < landmark[1])[0]
        lm_idx = np.intersect1d(lm_start_idx,lm_end_idx)
        lm_diff = np.diff(lm_idx)
        lm_end = np.where(lm_diff>1)[0]
        lm_start = np.insert(lm_end,0,0)+1
        lm_end = np.append(lm_end,lm_idx.size-1)
        if lm_start.size > lm_end.size:
            lm_end.append(np.size(behav_ds),0)

        for i,lm in enumerate(lm_start):
            if behav_ds[lm_idx[lm],4]!=5:
                if lm_idx[lm_start[i]] > t_start_idx and lm_idx[lm_start[i]] < t_stop_idx:
                    if behav_ds[lm_idx[lm],4] == 3:
                        ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='0.9')
                    else:
                        ax1.axvspan(lm_idx[lm_start[i]]-t_start_idx,lm_idx[lm_end[i]]-t_start_idx,color='0.7')

    elif plot_task_feature is 'trialonset':
        trials_short = filter_trials( behav_ds, [], ['tracknumber',3])
        trials_long = filter_trials( behav_ds, [], ['tracknumber',4])
        trials_all = np.union1d(trials_short,trials_long)
        events = event_ind(behav_ds, ['trial_transition'], trials_all)
        for e in events:
            ax1.axvline(e[0]-t_start_idx,c='#39B54A',lw=3,zorder=2)

    elif plot_task_feature is 'lmcenter':
        trials_short = filter_trials( behav_ds, [], ['tracknumber',3])
        trials_long = filter_trials( behav_ds, [], ['tracknumber',4])
        trials_all = np.union1d(trials_short,trials_long)
        events = event_ind(behav_ds, ['at_location', 220], trials_all)
        for e in events:
            ax1.axvline(e[0]-t_start_idx,c='r',lw=3,zorder=2)

    elif plot_task_feature is 'rewards':
        trials_short = filter_trials( behav_ds, [], ['tracknumber',3])
        trials_long = filter_trials( behav_ds, [], ['tracknumber',4])
        trials_all = np.union1d(trials_short,trials_long)
        events = event_ind(behav_ds, ['rewards_all', -1], trials_all)
        for e in events:
            ax1.axvline(e[0]-t_start_idx,c='#29ABE2',lw=3,zorder=2)


    ax1.set_xlim([0,t_stop_idx-t_start_idx])
    ax2.set_xlim([0,t_stop_idx-t_start_idx])

    one_sec = (t_stop_idx-t_start_idx)/(t_stop - t_start)
    ax1.set_xticks([0,one_sec,5*one_sec])
    ax1.set_xticklabels(['0','1','5'])

    ax1.set_yticks([0,0.1,0.5,2,4,6])
    ax1.set_yticklabels(['0','0.1','0.5','2','4','6'])
    ax1.set_ylabel('dF/F', fontsize=16)

    ax2.set_yticks([0,10,20,30,40])
    ax2.set_yticklabels(['0','10','20','30','40'])
    ax2.set_ylabel('speed (cm/sec)', fontsize=16)

    if ylims is []:
        ax1.set_ylim([-0.1,10])
    else:
        ax1.set_ylim(ylims)

    ax2.set_ylim([-5,80])


    fname = 'ind_trace' + mouse + '_' + sess + '_' + str(roi)

    subfolder = 'ind_traces'
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:

        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
    print(fname)
    print('done')


if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline



    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_1'
    # plot_task_feature = 'trialonset'
    # ROI = 32
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 535, 615, [-0.3,3.5], plot_task_feature) #1285, 1367


    # plot_task_feature = 'lmcenter'
    # ROI = 74
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 805, 885, [-0.3,3], plot_task_feature) #1285, 1367
    # #
    # MOUSE = 'LF170613_1'
    # SESSION = 'Day20170804'
    # plot_task_feature = 'lmcenter'
    # ROI = 15
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 140, 215, [-0.5,5], plot_task_feature) #1285, 1367
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 1240, 1315, [-0.5,3], plot_task_feature) #1285, 1367
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 500, 700, [-0.5,5], plot_task_feature) #1285, 1367
    # SESSION = 'Day20170804_openloop'
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 45, 120, [-0.5,5], plot_task_feature) #1285, 1367
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 000, 1800, [-0.5,4], plot_task_feature) #1285, 1367
    #
    # MOUSE = 'LF170421_2'
    # SESSION = 'Day20170719'
    # plot_task_feature = 'lmcenter'
    # ROI = 43
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 0, 900, [-0.3,3], plot_task_feature) #1285, 1367
    # SESSION = 'Day20170719_openloop'
    # plot_task_feature = 'lmcenter'
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 0, 900, [-0.3,3], plot_task_feature) #1285, 1367

    MOUSE = 'LF171211_2'
    SESSION = 'Day201852'
    plot_task_feature = 'trialonset'
    ROI = 36
    plot_ind_trace_behavior(MOUSE, SESSION, ROI, 726, 753, [-0.5,6], plot_task_feature) #1285, 1367

    plot_task_feature = 'lmcenter'
    ROI = 44
    plot_ind_trace_behavior(MOUSE, SESSION, ROI, 727, 754, [-0.5,4], plot_task_feature) #1285, 1367

    plot_task_feature = 'rewards'
    ROI = 24
    plot_ind_trace_behavior(MOUSE, SESSION, ROI, 741, 768, [-0.5,7], plot_task_feature) #1285, 1367

    # SESSION = 'Day201852_openloop'
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 546, 568, [-0.5,7], plot_task_feature) #1285, 1367
    #
    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # plot_task_feature = 'lmcenter'
    # ROI = 66
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 930, 953, [-0.05,0.3], plot_task_feature) #1285, 1367
    #
    # SESSION = 'Day201777_openloop'
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 437, 460, [-0.05,0.3], plot_task_feature) #1285, 1367


    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 150, 300)
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 11, 120)
    #
    # SESSION = 'Day201784'
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 461, 570)

    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # ROI = 40
    # plot_ind_trace_behavior(MOUSE, SESSION, ROI, 1030, 1090) #1285, 1367
    #
    # SESSION = 'Day2017714'
    # r = 41
    #
    # plot_ind_trace_behavior(MOUSE, SESSION, r, 1050, 1110) #1285, 1367
