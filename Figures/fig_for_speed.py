"""
Do you feel the fig?
The fig for speed?

@author: rmojica@mit.edu
"""

def fig_speed(h5path, mouse, sess, fname, fformat= 'png'):
    import matplotlib
    from matplotlib import pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set_style("white")

    import h5py
    import os
    import sys
    import yaml
    import warnings
    warnings.filterwarnings('ignore')

    with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')

    try:
        behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    except:
        behav_ds = np.copy(h5dat[sess + '/raw_data'])

    h5dat.close()

    raw_speed = behav_ds[:,3]

    for i in range(len(raw_speed)):
        if raw_speed[i] <= 0.7:
            raw_speed[i] = np.nan
        else:
            pass

    raw_speed = raw_speed[~np.isnan(raw_speed)]

    # set up axes
    fig = plt.figure(figsize=(8,10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    # bin the speeds into discrete values for histo
    speed_bins = []
    discrete_bins = []

    for i in range(raw_speed.shape[0]):
        # if raw_speed[i] >= 0 and raw_speed[i] < 1:
        #     speed_bins.append([1,raw_speed[i]])
        #     discrete_bins.append(1)
        if raw_speed[i] >= 1 and raw_speed[i] <= 10:
            speed_bins.append([2,raw_speed[i]])
            discrete_bins.append(2)
        elif raw_speed[i] > 10 and raw_speed[i] <= 20:
            speed_bins.append([3,raw_speed[i]])
            discrete_bins.append(3)
        elif raw_speed[i] > 20 and raw_speed[i] <= 30:
            speed_bins.append([4,raw_speed[i]])
            discrete_bins.append(4)
        elif raw_speed[i] > 30 and raw_speed[i] <= 40:
            speed_bins.append([5,raw_speed[i]])
            discrete_bins.append(5)
        elif raw_speed[i] > 40:
            speed_bins.append([6,raw_speed[i]])
            discrete_bins.append(6)

    binned = sns.distplot(discrete_bins, kde=False, ax=ax1)
    binned.set_title(mouse)

    raw = sns.distplot(raw_speed, ax=ax2)
    raw.set_title('Raw')


def dF_speed_scatter(h5path, mouse, sess, roi, fname, fformat= 'png',subfolder=[]):
    import matplotlib
    from matplotlib import pyplot as plt
    from scipy import stats
    import numpy as np
    import seaborn as sns
    sns.set_style("white")

    import h5py
    import os
    import sys
    import yaml
    import json
    import warnings
    warnings.filterwarnings('ignore')

    with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[sess + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[sess + '/dF_win'])
    h5dat.close()

    fig = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    # substitute subthresh values for nans for easy cleanup
    speed_thresh = 0.7

    raw_speed = behav_ds[:,3]
    above_thresh = np.copy(raw_speed)
    under_thresh = np.copy(raw_speed)

    for i in range(above_thresh.shape[0]):
        if above_thresh[i] <= speed_thresh:
            above_thresh[i] = np.nan
        if under_thresh[i] > speed_thresh:
            under_thresh[i] = np.nan

    # eliminate nans for plotting, statistics
    speed = above_thresh[~np.isnan(above_thresh)]
    dF_moving = dF_ds[:,roi]                       # take only current roi dF
    dF_speed = dF_moving[~np.isnan(above_thresh)]

    r,p = stats.pearsonr(speed, dF_speed)
    r = round(r,2)
    p = round(p,5)

    avg_dF = np.mean(dF_speed)
    sem_dF = stats.sem(dF_speed)
    avg_speed = np.mean(speed)
    sem_speed = stats.sem(speed)

    # setup data for json output
    roi_stats = {
    'pearson' : r,
    'p_value' : p,
    'mean_dF' : avg_dF,
    'sem_dF' : sem_dF,
    'mean_speed' : avg_speed,
    'sem_speed' : sem_speed
    }

    session_stats = ['Moving', roi, roi_stats]

    with open(content['figure_output_path'] + mouse+sess + os.sep + 'speed_stats.json','a+') as f:
        json.dump(session_stats,f)

    plt.sca(ax1)
    the_scatman = sns.regplot(x=speed, y=dF_speed, lowess=False, marker='1',ax=ax1)
    plt.xlim(0,100)
    plt.ylabel('∆F/F')
    plt.text(0.9, 0.9, 'r = '+str(r), ha='center', va='center', transform = the_scatman.transAxes)
    plt.title('ROI ' + str(roi))

    stationary = under_thresh[~np.isnan(under_thresh)]
    dF_stat = dF_ds[:,roi]
    dF_speed = dF_stat[~np.isnan(under_thresh)]

    s,q = stats.pearsonr(stationary, dF_speed)
    s = round(r,2)
    q = round(p,5)

    avg_dF = np.mean(dF_speed)
    sem_dF = stats.sem(dF_speed)
    avg_speed = np.mean(stationary)
    sem_speed = stats.sem(stationary)

    roi_stats = {
    'pearson' : r,
    'p_value' : p,
    'mean_dF' : avg_dF,
    'sem_dF' : sem_dF,
    'mean_speed' : avg_speed,
    'sem_speed' : sem_speed
    }

    session_stats = ['Stationary', roi, roi_stats]

    with open(content['figure_output_path'] + mouse+sess + os.sep + 'speed_stats.json','a+') as f:
        json.dump(session_stats,f)

    plt.sca(ax2)
    the_other = sns.regplot(x=stationary, y=dF_speed, lowess=False, color='orange', marker='1',ax=ax2)
    plt.xlim(-1,1)
    plt.xlabel('Running speed (cm/s)')
    plt.ylabel('∆F/F')
    plt.text(0.9, 0.9, 'r = '+str(s), ha='center', va='center', transform = the_other.transAxes)
    plt.title('Stationary')

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


# ----
import yaml
import h5py
import os

with open('/Users/Raul/coding/github/harnett_lab/in_vivo/MTH3/loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

MOUSE = 'LF171211_1'
SESSION = 'Day2018314_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

# # fig_speed(h5path, SESSION, 'speed_test')
#
# # dF_speed_scatter(h5path, SESSION, 0, 'dF_speed_roi'+str(0), MOUSE+SESSION)
#
# for i in range(130):
#     dF_speed_scatter(h5path, SESSION, i, 'dF_speed_roi'+str(i), subfolder=MOUSE+SESSION)

MOUSE = 'LF180119_1'
SESSION = 'Day2018316_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

MOUSE = 'LF180119_1'
SESSION = 'Day2018315'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

# dF_speed_scatter(h5path, MOUSE, SESSION, 0, 'dF_speed_roi'+str(0), subfolder=MOUSE+SESSION)

# for i in range(271):
#     dF_speed_scatter(h5path, MOUSE, SESSION, i, 'dF_speed_roi'+str(i), subfolder=MOUSE+SESSION)

MOUSE = 'LF171211_2'
SESSION = 'Day2018321'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

MOUSE = 'LF171211_2'
SESSION = 'Day2018320_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

MOUSE = 'LF170613_1'
SESSION = 'Day20170718'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

MOUSE = 'LF171211_2'
SESSION = 'Day2018321'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

MOUSE = 'LF180112_2'
SESSION = 'Day2018314'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

MOUSE = 'LF171211_1'
SESSION = 'Day2018312'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')

MOUSE = 'LF171211_2'
SESSION = 'Day2018321'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

fig_speed(h5path, MOUSE, SESSION, 'speed_test')
