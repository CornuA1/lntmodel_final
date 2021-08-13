"""
Plot result of Muscimol experiment group result for MTH3

@author: lukasfischer


"""

import h5py
import yaml
import os
import sys

def fig_muscimol_group(h5path, fname, fformat='agg'):
    # load local settings file
    import matplotlib
    import numpy as np
    #matplotlib.use(fformat,force=True)
    from matplotlib import pyplot as plt
    import warnings; warnings.simplefilter('ignore')
    import sys

    import yaml
    import h5py
    from scipy import stats
    import seaborn as sns
    sns.set_style('white')
    import os
    with open('../loc_settings.yaml', 'r') as f:
                content = yaml.load(f)

    with open(content['yaml_file'], 'r') as f:
                project_metainfo = yaml.load(f)

    # load animal group data from project YAML file
    m = project_metainfo['muscimol_mice']

    days_ACSF = project_metainfo['mus_mice_CONTROL']
    days_MUS = project_metainfo['mus_mice_MUSCIMOL']

    fig = plt.figure(figsize=(16, 16))
    ax1 = plt.subplot2grid((14, 4), (0, 0), rowspan=3, colspan=2)
    ax2 = plt.subplot2grid((14, 4), (0, 2), rowspan=3, colspan=2)
    ax3 = plt.subplot2grid((14, 4), (3, 0), rowspan=3, colspan=2)
    ax4 = plt.subplot2grid((14, 4), (3, 2), rowspan=3, colspan=1)
    ax10 = plt.subplot2grid((14, 4), (3, 3), rowspan=3, colspan=1)

    ax5 = plt.subplot2grid((14, 4), (7, 0), rowspan=3, colspan=2)
    ax6 = plt.subplot2grid((14, 4), (7, 2), rowspan=3, colspan=2)
    ax7 = plt.subplot2grid((14, 4), (10, 0), rowspan=3, colspan=2)
    ax8 = plt.subplot2grid((14, 4), (10, 2), rowspan=3, colspan=1)
    ax9 = plt.subplot2grid((14, 4), (10, 3), rowspan=3, colspan=1)


    ### PLOT DATA FROM MUSCIMOL EXPERIMENT ###


    # load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous
    h5dat = h5py.File(content['muscimol_file'], 'r')
    for mouse in m:
        # load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous
        for da in days_ACSF[mouse]:
            try:
                licks_ACSF_ds = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                licks_ACSF_ds[:,2] = licks_ACSF_ds[:,2]+last_trial_ACSF
                licks_ACSF = np.append(licks_ACSF,licks_ACSF_ds,axis=0)
                last_trial_ACSF = licks_ACSF[-1,2]
            except NameError:
                licks_ACSF = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                last_trial_ACSF = licks_ACSF[-1,2]

        # load MUSCIMOL datasets - when datasets are appended, re-enumerate trial numbers to be continuous
        for da in days_MUS[mouse]:
            try:
                licks_MUS_ds = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                licks_MUS_ds[:,2] = licks_MUS_ds[:,2]+last_trial_MUS
                licks_MUS = np.append(licks_MUS,licks_MUS_ds,axis=0)
                last_trial_MUS = licks_MUS[-1,2]
            except NameError:
                licks_MUS = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                last_trial_MUS = licks_MUS[-1,2]
    h5dat.close()
    # determine location of first licks - ACSF
    first_lick_short_ACSF = []
    first_lick_short_trials_ACSF = []
    first_lick_long_ACSF = []
    first_lick_long_trials_ACSF = []
    for r in np.unique(licks_ACSF[:,2]):
        licks_trial = licks_ACSF[licks_ACSF[:,2]==r,:]
        licks_trial = licks_trial[licks_trial[:,1]>240,:]
        if licks_trial.shape[0]>0:
            if licks_trial[0,3] == 3:
                first_lick_short_ACSF.append(licks_trial[0,1])
                first_lick_short_trials_ACSF.append(r)
            elif licks_trial[0,3] == 4:
                first_lick_long_ACSF.append(licks_trial[0,1])
                first_lick_long_trials_ACSF.append(r)

    # determine location of first licks - MUSCIMOL
    first_lick_short_MUS = []
    first_lick_short_trials_MUS = []
    first_lick_long_MUS = []
    first_lick_long_trials_MUS = []
    for r in np.unique(licks_MUS[:,2]):
        licks_trial = licks_MUS[licks_MUS[:,2]==r,:]
        licks_trial = licks_trial[licks_trial[:,1]>240,:]
        if licks_trial.shape[0]>0:
            if licks_trial[0,3] == 3:
                first_lick_short_MUS.append(licks_trial[0,1])
                first_lick_short_trials_MUS.append(r)
            elif licks_trial[0,3] == 4:
                first_lick_long_MUS.append(licks_trial[0,1])
                first_lick_long_trials_MUS.append(r)

    #print(first_lick_short_trials_ACSF)
    #ax1.scatter(first_lick_short_ACSF, range(len(first_lick_short_trials_ACSF)), c=sns.xkcd_rgb["windows blue"], lw=0)
    #ax1.scatter(first_lick_long_ACSF, range(len(first_lick_long_trials_ACSF)), c=sns.xkcd_rgb["dusty purple"], lw=0)
    ax1_1 = ax1.twinx()
    if len(first_lick_short_ACSF) > 0:
        # sns.kdeplot(np.asarray(first_lick_short_ACSF),c=sns.xkcd_rgb["windows blue"],label='short',shade=True,ax=ax1_1)
        sns.distplot(np.asarray(first_lick_short_ACSF),color=sns.xkcd_rgb["windows blue"],kde=False,bins=80,ax=ax1,hist_kws={"range": [200,450]})
    if len(first_lick_long_ACSF) > 0:
        #sns.kdeplot(np.asarray(first_lick_long_ACSF),c=sns.xkcd_rgb["dusty purple"],label='long',shade=True,ax=ax1_1)
        sns.distplot(np.asarray(first_lick_long_ACSF),color=sns.xkcd_rgb["dusty purple"],kde=False,bins=80,ax=ax1,hist_kws={"range": [200,450]})
    ax1.set_xlim([230,400])
    ax1.set_title('Location of first licks in CONTROL condition')
    #ax1.set_ylim([0,len(first_lick_long_trials_ACSF)])
    ax1.set_ylim([0,35])

    #ax2.scatter(first_lick_short_MUS,range(len(first_lick_short_trials_MUS)),c=sns.xkcd_rgb["windows blue"],lw=0)
    #ax2.scatter(first_lick_long_MUS,range(len(first_lick_long_trials_MUS)),c=sns.xkcd_rgb["dusty purple"],lw=0)
    ax2_1 = ax2.twinx()
    if len(first_lick_short_ACSF) > 0:
        #sns.kdeplot(np.asarray(first_lick_short_MUS),c=sns.xkcd_rgb["windows blue"],label='short',shade=True,ax=ax2_1)
        sns.distplot(np.asarray(first_lick_short_MUS),color=sns.xkcd_rgb["windows blue"],kde=False,bins=80,ax=ax2,hist_kws={"range": [200,450]})
    if len(first_lick_long_ACSF) > 0:
        #sns.kdeplot(np.asarray(first_lick_long_MUS),c=sns.xkcd_rgb["dusty purple"],label='long',shade=True,ax=ax2_1)
        sns.distplot(np.asarray(first_lick_long_MUS),color=sns.xkcd_rgb["dusty purple"],kde=False,bins=80,ax=ax2,hist_kws={"range": [200,450]})
    ax2.set_xlim([230,400])
    ax2.set_title('Location of first licks in MUSCIMOL condition')
    #ax2.set_ylim([0,len(first_lick_long_trials_MUS)+50])
    ax2.set_ylim([0,35])

    # bootstrap differences between pairs of first lick locations
    short_bootstrap_ACSF = np.random.choice(first_lick_short_ACSF,10000)
    long_bootstrap_ACSF = np.random.choice(first_lick_long_ACSF,10000)
    bootstrap_diff_ACSF = long_bootstrap_ACSF - short_bootstrap_ACSF
    sns.distplot(bootstrap_diff_ACSF,color='b',ax=ax3)

    short_bootstrap_MUS = np.random.choice(first_lick_short_MUS,10000)
    long_bootstrap_MUS = np.random.choice(first_lick_long_MUS,10000)
    bootstrap_diff_MUS = long_bootstrap_MUS - short_bootstrap_MUS
    sns.distplot(bootstrap_diff_MUS,color='r',ax=ax3)

    ax3.axvline(np.mean(bootstrap_diff_ACSF),c='b',ls='--',lw=2)
    ax3.axvline(np.mean(bootstrap_diff_MUS),c='r',ls='--',lw=2)

    first_lick_ACSF_diff = np.mean(first_lick_long_ACSF) - np.mean(first_lick_short_ACSF)
    first_lick_MUS_diff = np.mean(first_lick_long_MUS) - np.mean(first_lick_short_MUS)

    print('ASCF EXP. 95%: ', np.percentile(bootstrap_diff_ACSF,95))
    print('MUS EXP. 95%: ', np.percentile(bootstrap_diff_MUS,95))
    print(['Diff Mean EXP: ', np.mean(bootstrap_diff_ACSF)-np.mean(bootstrap_diff_MUS)])

    ### PLOT DATA FROM CONTROL EXPERIMENT ###

    # load animal group data from project YAML file
    m = project_metainfo['muscmiol_control_mice']

    days_ACSF = project_metainfo['muscontrol_CONTROL']
    days_MUS = project_metainfo['muscontrol_MUSCIMOL']

    # load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous
    h5dat = h5py.File(content['muscimol_control_datafile'], 'r')
    for mouse in m:
        # load ACSF datasets - when datasets are appended, re-enumerate trial numbers to be continuous
        # for each mice, concatenate all licks from all days of ACSF condition
        for da in days_ACSF[mouse]:
            try:
                licks_ACSF_ds = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                licks_ACSF_ds[:,2] = licks_ACSF_ds[:,2]+last_trial_ACSF #re-enumerate to trial numbers becaome continuous
                licks_ACSF = np.append(licks_ACSF,licks_ACSF_ds,axis=0) #update licks_ACSF
                last_trial_ACSF = licks_ACSF[-1,2]

                raw_ACSF_ds = np.copy(h5dat['/Day' + da + '/'+ mouse + '/raw_data'])
                raw_ACSF_ds[:,6] = raw_ACSF_ds[:,6]+last_ACSF #trial numbers
                raw_ACSF = np.append(raw_ACSF,raw_ACSF_ds,axis=0) #
                last_ACSF = raw_ACSF[-1,6]

            except NameError:
                licks_ACSF = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                last_trial_ACSF = licks_ACSF[-1,2]#trial number of last line

                raw_ACSF = np.copy(h5dat['/Day' + da + '/'+ mouse + '/raw_data'])
                last_ACSF = raw_ACSF[-1,6]
                #raw_ds = np.copy(h5dat['/' + day_widget.value + '/'+mouse_widget.value + '/raw_data'])

        # load MUSCIMOL datasets - when datasets are appended, re-enumerate trial numbers to be continuous
         # for each mice, concante all licks from all days of Muscimol condition
        for da in days_MUS[mouse]:
            try:
                licks_MUS_ds = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                licks_MUS_ds[:,2] = licks_MUS_ds[:,2]+last_trial_MUS
                licks_MUS = np.append(licks_MUS,licks_MUS_ds,axis=0)
                last_trial_MUS = licks_MUS[-1,2]

                raw_MUS_ds = np.copy(h5dat['/Day' + da + '/'+ mouse + '/raw_data'])
                raw_MUS_ds[:,6] = raw_MUS_ds[:,6]+last_MUS #trial numbers
                raw_MUS = np.append(raw_MUS,raw_MUS_ds,axis=0)
                last_MUS = raw_MUS[-1,6]

            except NameError:
                licks_MUS = np.copy(h5dat['/Day' + da + '/'+ mouse + '/licks_pre_reward'])
                last_trial_MUS = licks_MUS[-1,2]

                raw_MUS = np.copy(h5dat['/Day' + da + '/'+ mouse + '/raw_data'])
                last_MUS = raw_MUS[-1,6]


    h5dat.close()

    licks_MUS=licks_MUS[licks_MUS[:,3]!=5]
    licks_ACSF=licks_ACSF[licks_ACSF[:,3]!=5]
    # determine location of first licks - ACSF
    first_lick_short_ACSF = []
    first_lick_short_trials_ACSF = []
    first_lick_long_ACSF = []
    first_lick_long_trials_ACSF = []
    for r in np.unique(licks_ACSF[:,2]):
        #get start location of each trial
        start_location = raw_ACSF[raw_ACSF[:,6]==r][0][1]
        #rows in whcih it licked this trial
        licks_trial = licks_ACSF[licks_ACSF[:,2]==r,:]
        # discard licks that happened within x cm from start
        licks_trial = licks_trial[licks_trial[:,1]>(start_location+10),:]
        if licks_trial.shape[0]>0:
            if licks_trial[0,3] == 7:
                first_lick_short_ACSF.append(licks_trial[0,1])
                first_lick_short_trials_ACSF.append(r)
            elif licks_trial[0,3] == 8:
                first_lick_long_ACSF.append(licks_trial[0,1])
                first_lick_long_trials_ACSF.append(r)

    # determine location of first licks - MUSCIMOL
    first_lick_short_MUS = []
    first_lick_short_trials_MUS = []
    first_lick_long_MUS = []
    first_lick_long_trials_MUS = []

    for r in np.unique(licks_MUS[:,2]):
        #get start location of each trial
        start_location = raw_MUS[raw_MUS[:,6]==r][0][1]
        licks_trial = licks_MUS[licks_MUS[:,2]==r,:]
        licks_trial = licks_trial[licks_trial[:,1]>(start_location+0),:]
        if licks_trial.shape[0]>0:
            if licks_trial[0,3] == 7:
                first_lick_short_MUS.append(licks_trial[0,1])
                first_lick_short_trials_MUS.append(r)
            elif licks_trial[0,3] == 8:
                first_lick_long_MUS.append(licks_trial[0,1])
                first_lick_long_trials_MUS.append(r)

    #print(first_lick_short_trials_ACSF)
    ax5.scatter(first_lick_short_ACSF,range(len(first_lick_short_trials_ACSF)),c=sns.xkcd_rgb["windows blue"],lw=0)
    ax5.scatter(first_lick_long_ACSF,range(len(first_lick_long_trials_ACSF)),c=sns.xkcd_rgb["dusty purple"],lw=0)
    ax5_1 = ax5.twinx()
    if len(first_lick_short_ACSF) > 0:
        sns.kdeplot(np.asarray(first_lick_short_ACSF),c=sns.xkcd_rgb["windows blue"],label='short',shade=True,ax=ax5_1)
    if len(first_lick_long_ACSF) > 0:
        sns.kdeplot(np.asarray(first_lick_long_ACSF),c=sns.xkcd_rgb["dusty purple"],label='long',shade=True,ax=ax5_1)
    ax5.set_xlim([0,250])
    ax5.set_title('Location of first licks in CONTROL condition')

    ax6.scatter(first_lick_short_MUS,range(len(first_lick_short_trials_MUS)),c=sns.xkcd_rgb["windows blue"],lw=0)
    ax6.scatter(first_lick_long_MUS,range(len(first_lick_long_trials_MUS)),c=sns.xkcd_rgb["dusty purple"],lw=0)


    ax6_1 = ax6.twinx()
    if len(first_lick_short_ACSF) > 0:
        sns.kdeplot(np.asarray(first_lick_short_MUS),c=sns.xkcd_rgb["windows blue"],label='short',shade=True,ax=ax6_1)
    if len(first_lick_long_ACSF) > 0:
        sns.kdeplot(np.asarray(first_lick_long_MUS),c=sns.xkcd_rgb["dusty purple"],label='long',shade=True,ax=ax6_1)
    ax6.set_xlim([0,250])
    ax6.set_title('Location of first licks in MUSCIMOL condition')


    # bootstrap differences between pairs of first lick locations
    short_bootstrap_ACSF = np.random.choice(first_lick_short_ACSF,10000)
    long_bootstrap_ACSF = np.random.choice(first_lick_long_ACSF,10000)
    bootstrap_diff_ACSF = long_bootstrap_ACSF - short_bootstrap_ACSF
    sns.distplot(bootstrap_diff_ACSF,color='b',ax=ax7)

    short_bootstrap_MUS = np.random.choice(first_lick_short_MUS,10000)
    long_bootstrap_MUS = np.random.choice(first_lick_long_MUS,10000)
    bootstrap_diff_MUS = long_bootstrap_MUS - short_bootstrap_MUS
    sns.distplot(bootstrap_diff_MUS,color='r',ax=ax7)


    ax7.axvline(np.mean(bootstrap_diff_ACSF),c='b',ls='--',lw=2)
    ax7.axvline(np.mean(bootstrap_diff_MUS),c='r',ls='--',lw=2)

    print('ASCF CONT. 95%: ', np.percentile(bootstrap_diff_ACSF,95))
    print('MUS CONT. 95%: ', np.percentile(bootstrap_diff_MUS,95))
    print(['Diff Mean EXP: ', np.mean(bootstrap_diff_ACSF)-np.mean(bootstrap_diff_MUS)])

    first_lick_CONTROL_ACSF_diff = np.mean(first_lick_long_ACSF) - np.mean(first_lick_short_ACSF)
    first_lick_CONTROL_MUS_diff = np.mean(first_lick_long_MUS) - np.mean(first_lick_short_MUS)

    ax4.scatter([0, 1], [first_lick_CONTROL_ACSF_diff, first_lick_CONTROL_MUS_diff],s=80,c=['w','k'],edgecolors='k',zorder=1)
    ax4.scatter([0, 1], [first_lick_ACSF_diff, first_lick_MUS_diff],s=80,c=['w','k'],edgecolors='k',zorder=1)
    ax4.plot([0, 1], [first_lick_CONTROL_ACSF_diff,first_lick_CONTROL_MUS_diff],c='k',zorder=0)
    ax4.plot([0, 1], [first_lick_ACSF_diff,first_lick_MUS_diff],c='k',zorder=0)

    fl_short_MUS_mean = np.mean(first_lick_short_MUS)
    fl_short_MUS_stderr = stats.sem(first_lick_short_MUS)
    fl_long_MUS_mean = np.mean(first_lick_long_MUS)
    fl_long_MUS_stderr = stats.sem(first_lick_long_MUS)

    fl_short_ACSF_mean = np.mean(first_lick_short_ACSF)
    fl_short_ACSF_stderr = stats.sem(first_lick_short_ACSF)
    fl_long_ACSF_mean = np.mean(first_lick_long_ACSF)
    fl_long_ACSF_stderr = stats.sem(first_lick_long_ACSF)

    ax10.scatter(0, fl_short_MUS_mean, c='k')
    ax10.errorbar(0, fl_short_MUS_mean, yerr=fl_short_MUS_stderr, c='k')
    ax10.scatter(1, fl_long_MUS_mean, c='k')
    ax10.errorbar(1, fl_long_MUS_mean, yerr=fl_long_MUS_stderr, c='k')

    ax10.scatter(0, fl_short_ACSF_mean, c='k')
    ax10.errorbar(0, fl_short_ACSF_mean, yerr=fl_short_ACSF_stderr, c='k')
    ax10.scatter(1, fl_long_ACSF_mean, c='k')
    ax10.errorbar(1, fl_long_ACSF_mean, yerr=fl_long_ACSF_stderr, c='k')

    #ax10.scatter(np.zeros(len(first_lick_short_MUS)), first_lick_short_MUS, s=80, c=['w'], edgecolors='k', zorder=1)
    #ax10.scatter(np.ones(len(first_lick_long_MUS)), first_lick_long_MUS, s=80, c=['w'], edgecolors='k', zorder=1)
    #ax10.plot([0, 1], [first_lick_ACSF_diff,first_lick_MUS_diff],c='k',zorder=0)
    #ax10.set_ylim([25,45])

    ax8.bar([0.2],[first_lick_ACSF_diff-first_lick_MUS_diff],color='k',lw=4,width=0.4)
    ax8.bar([0.8],[first_lick_CONTROL_ACSF_diff-first_lick_CONTROL_MUS_diff],color='w',lw=4,width=0.4)
    ax8.set_xlim([0,1.4])
    ax8.set_ylim([0,12])
    ax8.set_xticks([0.4,1.0])
    ax8.set_xticklabels(['Landmark + path integration', 'Visually guided'],rotation='horizontal')

    plt.tight_layout()

    fig.suptitle(fname, wrap=True)
    fname = content['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat)

if __name__ == "__main__":

    # load local settings file
    sys.path.append("C:\\Users\\The_mothership\\Documents\\GitHub\\LNT\\Analysis")
    sys.path.append("C:\\Users\\The_mothership\\Documents\\GitHub\\LNT\\Figures")

    from fig_muscimol_group import fig_muscimol_group
    os.chdir('C:\\Users\\The_mothership\\Documents\\GitHub\\LNT\\Analysis')

    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    h5path = content['muscimol_file']

    fig_muscimol_group(h5path, 'muscimol_group_MTH3', 'png')
