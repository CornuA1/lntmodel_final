"""
compare overall brightness in regular vs openloop and grating sessions

@author: Lukas Fischer

"""


import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from scipy.signal import butter, filtfilt
import seaborn as sns
import scipy as sp
import numpy as np
import warnings
import h5py
import sys, os
import yaml
import json

with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
from write_dict import write_dict
import seaborn as sns
sns.set_style('white')

def compare_brightness(mouse, sess, sess_ol, subfolder=[]):
    h5path = loc_info['imaging_dir'] + mouse + '/' + mouse + '.h5'
    print(h5path)
    print(mouse, sess)
    h5dat = h5py.File(h5path, 'r')
    roi_raw = np.copy(h5dat[sess + '/roi_raw'])
    bri_ds = np.copy(h5dat[sess + '/FOV_bri'])
    bri_ol_ds = np.copy(h5dat[sess_ol + '/FOV_bri'])
    h5dat.close()

    fig = plt.figure(figsize=(16,4))
    ax1 = plt.subplot2grid((2,6),(0,0), rowspan=1, colspan=4)
    ax2 = plt.subplot2grid((2,6),(1,0), rowspan=1, colspan=4)
    ax3 = plt.subplot2grid((2,6),(0,4), rowspan=1, colspan=2)
    ax4 = plt.subplot2grid((2,6),(1,4), rowspan=1, colspan=2)

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

    ax3.spines['left'].set_linewidth(2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=4, \
        width=2, \
        bottom='on', \
        right='off', \
        top='off')

    # fit regression to PERCENTILE of datapoints to check for trends
    PERCENTILE = 50
    percentile_low_idx = np.where(bri_ds < np.percentile(bri_ds,PERCENTILE))[0]
    bri_mean = np.mean(bri_ds[percentile_low_idx])
    ax1.plot(bri_ds, lw=1)
    ax1.plot(percentile_low_idx, bri_ds[percentile_low_idx], c='g', lw=1)
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(percentile_low_idx, np.squeeze(bri_ds[percentile_low_idx]))
    ax1.plot(percentile_low_idx, intercept + slope*percentile_low_idx, 'r', label='fitted line')
    ax1.text(100, np.max(bri_ds)-50, 'slope' + str(np.round(slope,5)) + ' p: ' + str(np.round(p_value,5)) + ' mean: ' + str(np.round(bri_mean,2)))

    percentile_low_ol_idx = np.where(bri_ol_ds < np.percentile(bri_ol_ds,PERCENTILE))[0]
    bri_ol_mean = np.mean(bri_ol_ds[percentile_low_ol_idx])
    ax2.plot(bri_ol_ds, lw=1)
    ax2.plot(percentile_low_ol_idx, bri_ol_ds[percentile_low_ol_idx], c='g', lw=1)
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(percentile_low_ol_idx, np.squeeze(bri_ol_ds[percentile_low_ol_idx]))
    ax2.plot(percentile_low_ol_idx, intercept + slope*percentile_low_ol_idx, 'r', label='fitted line')
    ax2.text(100, np.max(bri_ol_ds)-50, 'slope' + str(np.round(slope,5)) + ' p: ' + str(np.round(p_value,5)) + ' mean: ' + str(np.round(bri_ol_mean,2)))

    min_b = np.min([np.min(bri_ds[percentile_low_idx]), np.min(bri_ol_ds[percentile_low_ol_idx])])
    max_b = np.max([np.max(bri_ds[percentile_low_idx]), np.max(bri_ol_ds[percentile_low_ol_idx])])

    bin_edges = np.linspace(min_b,max_b,40)
    sns.distplot(bri_ds[percentile_low_idx],kde=False,bins=bin_edges,ax=ax3,color='b')
    sns.distplot(bri_ol_ds[percentile_low_ol_idx],kde=False,bins=bin_edges,ax=ax3,color='g')

    # calculate baseline rundown for each roi
    PERCENTILE = 50
    baseline_r_all = np.zeros((roi_raw.shape[1],))
    for i,r in enumerate(roi_raw.T):
        r_baseline_idx = np.where(r < np.percentile(r,PERCENTILE))[0]
        if r_baseline_idx.shape[0] > 0:
            slope_r, intercept, r_value, p_value, std_err = sp.stats.linregress(r_baseline_idx, np.squeeze(r[r_baseline_idx]))
            baseline_r_all[i] = slope_r
        else:
            baseline_r_all[i] = np.nan

    sns.distplot(baseline_r_all, kde=False, bins=np.arange(-0.2,0.2,0.01), color='k', ax=ax4)
    ax4.axvline(slope, color='r', lw=2, label='slope FOV')
    ax4.legend()

    fname = 'compare_bri' + mouse + '_' + sess
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    plt.close()
    return slope.item(), baseline_r_all.tolist()

def summary_figure(roi_param_list):
    """ load brightness data from dictionary and plot """

    fig = plt.figure(figsize=(2.7,5))
    ax1 = plt.subplot(111)

    ax1.spines['left'].set_linewidth(4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(False)
    ax1.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=16, \
        length=6, \
        width=4, \
        bottom='off', \
        right='off', \
        top='off')

    fov_slopes = np.zeros((len(roi_param_list),))
    for i,rpl in enumerate(roi_param_list):
        # load roi parameters for given session
        # print(rpl)
        with open(rpl,'r') as f:
            roi_params = json.load(f)
        fov_slopes[i] = roi_params['FOV_slope']
        # print(fov_slopes[i])

    # sns.distplot(fov_slopes, kde=False, bins=np.arange(-0.2,0.2,0.01), color='k', ax=ax1)
    print(np.mean(np.array(fov_slopes)), sp.stats.sem(np.array(fov_slopes)))
    ax1.scatter(np.zeros((fov_slopes.shape[0])), fov_slopes, c='', linewidths=2,s=300,edgecolors='k')
    ax1.plot([-0.06,0.06], [np.nanmean(fov_slopes), np.nanmean(fov_slopes)], c='r',lw=6, solid_capstyle='butt')
    ax1.set_xlim([-0.2,0.2])
    ax1.set_ylim([-0.1,0.1])
    ax1.set_xticklabels([])

    ax1.set_ylabel('FOV brightness slope', fontsize=24)
    fformat = 'svg'
    subfolder = 'summary'
    fname = 'summary_brightness'
    fig.tight_layout()
    # fig.suptitle(fname, wrap=True)
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=300)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    print(fname)
    plt.close()

if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    fformat = 'png'
    write_to_dict = False

    roi_param_list = [
                      'D:\\Lukas\\Figures\\LF191022_1\\LF191022_1_20191204.json',
                      'D:\\Lukas\\Figures\\LF191022_1\\LF191022_1_20191204_ol.json'
                      ]

    summary_figure(roi_param_list)

    # set up dictionary and empty matrices to store data
    roi_result_params = {
        'FOV_slope' : [],
        'roi_slopes' : []
    }
    # MOUSE = 'LF170613_1'
    # SESSION = 'Day20170804'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF170222_1'
    # SESSION = 'Day201776'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    # #
    # MOUSE = 'LF170421_2'
    # SESSION = 'Day20170719'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF170421_2'
    # SESSION = 'Day2017720'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    # #
    # MOUSE = 'LF170420_1'
    # SESSION = 'Day201783'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF170420_1'
    # SESSION = 'Day2017719'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    # #
    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_1'
    # SESSION_OPENLOOP = 'Day201748_openloop_1'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_2'
    # SESSION_OPENLOOP = 'Day201748_openloop_2'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF170110_2'
    # SESSION = 'Day201748_3'
    # SESSION_OPENLOOP = 'Day201748_openloop_3'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF170214_1'
    # SESSION = 'Day201777'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF170214_1'
    # SESSION = 'Day2017714'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF171211_2'
    # SESSION = 'Day201852'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # MOUSE = 'LF171212_2'
    # SESSION = 'Day2018218_1'
    # SESSION_OPENLOOP = 'Day2018218_openloop_1'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
    # SESSION = 'Day2018218_2'
    # SESSION_OPENLOOP = 'Day2018218_openloop_2'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    # #
    # MOUSE = 'LF171211_1'
    # SESSION = 'Day2018321_2'
    # SESSION_OPENLOOP = 'Day2018321_openloop_2'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)

    # MOUSE = 'LF180112_2'
    # SESSION = 'Day2018322_1'
    # SESSION_OPENLOOP = SESSION + '_openloop'

    # compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')

    #
    # MOUSE = 'LF170222_1'
    # SESSION = 'Day2017615'
    # SESSION_OPENLOOP = SESSION + '_openloop'
    # fov_slope, all_roi_slopes = compare_brightness(MOUSE, SESSION, SESSION_OPENLOOP, MOUSE+'_'+SESSION+'_compare_bri')
    # roi_result_params['FOV_slope'] = fov_slope
    # roi_result_params['roi_slopes'] = all_roi_slopes
    # # write to dict
    # if write_to_dict:
    #     write_dict(MOUSE, SESSION, roi_result_params)
    #
