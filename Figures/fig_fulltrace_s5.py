"""
Plot full trace of ROI

@author: lukasfischer

"""

import numpy as np
import h5py
import sys
import yaml
import os
import json
import warnings; warnings.simplefilter('ignore')
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from event_ind import event_ind
from filter_trials import filter_trials
from scipy import stats
from scipy import signal

def fig_fulltrace(h5path, sess, roi, fname, ylims=[], fformat='png', subfolder=[]):
    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    # create figure and axes to later plot on
    fig = plt.figure(figsize=(20,2))
    ax1 = plt.subplot(111)

    ax1.plot(dF_ds[:,roi],lw=1,c='k',zorder=2)
    ax1.axhline(0,c='0.8',lw=2,ls='--',zorder=3)

    # get indices of rewards
    rews_s = event_ind(behav_ds,['reward_successful',-1])
    rews_us = event_ind(behav_ds,['reward_unsuccessful',-1])

    for rline in rews_s[:,0]:
        ax1.axvline(rline,c='g',lw=1,zorder=1)
    for rline in rews_us[:,0]:
        ax1.axvline(rline,c='r',lw=1,zorder=1)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticklabels([])

    if not ylims == []:
        ax1.set_ylim(ylims)

    fig.tight_layout()

    fig.suptitle(str(roi), wrap=True)
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

    return ax1.get_ylim()

def run_LF170110_2_Day201748_1():
    fformat = 'png'
    MOUSE = 'LF170110_2'
    NUM_ROIS = 152
    SESSION = 'Day201748_1'
    SESSION_OPENLOOP = 'Day201748_openloop_1'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170110_2_Day201748_2():
    fformat = 'png'
    MOUSE = 'LF170110_2'
    NUM_ROIS = 171
    SESSION = 'Day201748_2'
    SESSION_OPENLOOP = 'Day201748_openloop_2'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)


def run_LF170110_2_Day201748_3():
    fformat = 'png'
    MOUSE = 'LF170110_2'
    NUM_ROIS = 50
    SESSION = 'Day201748_3'
    SESSION_OPENLOOP = 'Day201748_openloop_3'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170110_2_Day2017331():
    fformat = 'png'
    MOUSE = 'LF170110_2'
    NUM_ROIS = 184
    SESSION = 'Day2017331'
    SESSION_OPENLOOP = 'Day2017331_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)


def run_LF170421_2_Day2017719():
    fformat = 'png'
    MOUSE = 'LF170421_2'
    NUM_ROIS = 68
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = 'Day2017719_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170421_2_Day2017720():
    fformat = 'png'
    MOUSE = 'LF170421_2'
    NUM_ROIS = 45
    SESSION = 'Day2017720'
    SESSION_OPENLOOP = 'Day2017720_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170420_1_Day201783():
    fformat = 'png'
    MOUSE = 'LF170420_1'
    NUM_ROIS = 81
    SESSION = 'Day201783'
    SESSION_OPENLOOP = 'Day201783_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170420_1_Day2017719():
    fformat = 'png'
    MOUSE = 'LF170420_1'
    NUM_ROIS = 91
    SESSION = 'Day2017719'
    SESSION_OPENLOOP = 'Day2017719_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170613_1_Day201784():
    fformat = 'png'
    MOUSE = 'LF170613_1'
    NUM_ROIS = 77
    SESSION = 'Day201784'
    SESSION_OPENLOOP = 'Day201784_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170613_1_Day20170804():
    fformat = 'png'
    MOUSE = 'LF170613_1'
    NUM_ROIS = 3#105
    SESSION = 'Day20170804'
    SESSION_OPENLOOP = 'Day20170804_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF171212_2_Day2017218():
    fformat = 'png'
    MOUSE = 'LF171212_2'
    NUM_ROIS = 335
    SESSION = 'Day2018218_2'
    SESSION_OPENLOOP = 'Day2018218_openloop_2'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

# def run_LF170214_1_Day201777():
#     fformat = 'png'
#     MOUSE = 'LF170214_1'
#     NUM_ROIS = 163
#     SESSION = 'Day201777'
#     SESSION_OPENLOOP = 'Day201777_openloop'
#     SUBNAME = 'fulltrace'
#     SUBNAME_OPENLOOP = 'fulltraceol'
#     SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
#     SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
#     h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'
#
#     for r in range(NUM_ROIS):
#         print(SUBFOLDER + ': ' + str(r))
#         fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
#         fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170214_1_Day2017714():
    fformat = 'png'
    MOUSE = 'LF170214_1'
    NUM_ROIS = 140
    SESSION = 'Day2017714'
    SESSION_OPENLOOP = 'Day2017714_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF180112_2_Day2018424_1():
    fformat = 'png'
    MOUSE = 'LF180112_2'
    NUM_ROIS = 73
    SESSION = 'Day2018424_1'
    SESSION_OPENLOOP = 'Day2018424_openloop_1'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF180112_2_Day2018424_2():
    fformat = 'png'
    MOUSE = 'LF180112_2'
    NUM_ROIS = 43
    SESSION = 'Day2018424_2'
    SESSION_OPENLOOP = 'Day2018424_openloop_2'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170222_1_Day201776():
    fformat = 'png'
    MOUSE = 'LF170222_1'
    NUM_ROIS = 120
    SESSION = 'Day201776'
    SESSION_OPENLOOP = 'Day201776_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF171211_1_Day2018321_2():
    fformat = 'png'
    MOUSE = 'LF171211_1'
    NUM_ROIS = 170
    SESSION = 'Day2018321_2'
    SESSION_OPENLOOP = 'Day2018321_openloop_2'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

def run_LF170421_2_Day20170719():
    fformat = 'png'
    MOUSE = 'LF170421_2'
    NUM_ROIS = 123
    SESSION = 'Day20170719'
    SESSION_OPENLOOP = 'Day20170719_openloop'
    SUBNAME = 'fulltrace'
    SUBNAME_OPENLOOP = 'fulltraceol'
    SUBFOLDER = MOUSE+'_'+SESSION+'_'+SUBNAME
    SUBFOLDER_OPENLOOP = MOUSE+'_'+SESSION+'_'+SUBNAME_OPENLOOP
    h5path = loc_info['imaging_dir'] +  MOUSE + '/' + MOUSE + '.h5'

    for r in range(NUM_ROIS):
        print(SUBFOLDER + ': ' + str(r))
        fig_ylims = fig_fulltrace(h5path, SESSION, r, MOUSE+'_'+SESSION+'_roi_'+str(r), [], fformat, SUBFOLDER)
        fig_fulltrace(h5path, SESSION_OPENLOOP, r, MOUSE+'_'+SESSION+'_roi_'+str(r), fig_ylims, fformat, SUBFOLDER_OPENLOOP)

if __name__ == '__main__':
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    fformat = 'png'
    # run_LF170110_2_Day201748_1()
    # run_LF170110_2_Day201748_2()
    # run_LF170110_2_Day201748_3()
    # run_LF170110_2_Day2017331()
    # run_LF170421_2_Day2017719()
    # run_LF170421_2_Day2017720()
    # run_LF170420_1_Day2017719()
    # run_LF170420_1_Day201783()
    # run_LF170613_1_Day201784()
    # run_LF170613_1_Day20170804()
    # run_LF171212_2_Day2017218()
    # run_LF170222_1_Day201776()
    run_LF171211_1_Day2018321_2()
    # run_LF170421_2_Day20170719()

    # run_LF170214_1_Day201777()
    # run_LF170214_1_Day2017714()

    # run_LF180112_2_Day2018424_1()
    # run_LF180112_2_Day2018424_2()
