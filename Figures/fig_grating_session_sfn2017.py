"""
Plot the entire grating session (including black screen), indicating grating presentations


"""


def fig_grating_session(h5path, sess, roi, fname, ylims=[], fformat='png', subfolder=[]):
    import numpy as np
    import h5py
    import sys
    import os
    import yaml
    from yaml_mouselist import yaml_mouselist
    import warnings; warnings.simplefilter('ignore')

    import matplotlib
    from matplotlib import pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal
    from event_ind_gratings import event_ind_gratings

    import seaborn as sns
    sns.set_style("white")
    sns.set_style("ticks")
    sns.despine

    # get local settings
    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)

    # load HDF5 data
    try:
        h5dat = h5py.File(h5path, 'r')
        behav_ds = np.copy(h5dat['/' + sess + '/behaviour_aligned'])
        dF_ds = np.copy(h5dat['/' + sess + '/dF_win'])
        data_grating = True;
        h5dat.close()
    except KeyError:
        print('NO GRATING DATA FOUND.')
        h5dat.close()
        return 0.0, 0.0, 0.0

    # only keep column with desired ROI trace
    dF_ds = dF_ds[:,roi]

    # check if dF data is full of NaNs
    if any(np.isnan(dF_ds)):
        print('DF DATA FULL OF NANS')
        return 0.0, 0.0, 0.0

    y_min = np.amin(dF_ds)
    y_max = np.amax(dF_ds)

    # calculate R values
    r_vals = np.zeros((16,))

    # create figure and subplots for each direction x spatial frequency
    fig = plt.figure(figsize=(12,12))
    ax = [
          plt.subplot2grid((10,4),(0,0)),
          plt.subplot2grid((10,4),(1,0)),
          plt.subplot2grid((10,4),(2,0)),
          plt.subplot2grid((10,4),(3,0)),
          plt.subplot2grid((10,4),(4,0)),
          plt.subplot2grid((10,4),(5,0)),
          plt.subplot2grid((10,4),(6,0)),
          plt.subplot2grid((10,4),(7,0)),
          plt.subplot2grid((10,4),(0,1)),
          plt.subplot2grid((10,4),(1,1)),
          plt.subplot2grid((10,4),(2,1)),
          plt.subplot2grid((10,4),(3,1)),
          plt.subplot2grid((10,4),(4,1)),
          plt.subplot2grid((10,4),(5,1)),
          plt.subplot2grid((10,4),(6,1)),
          plt.subplot2grid((10,4),(7,1)),
          plt.subplot2grid((10,4),(8,0)),
          plt.subplot2grid((10,4),(8,1)),
          plt.subplot2grid((10,4),(9,0)),
          plt.subplot2grid((10,4),(9,1)),
          plt.subplot2grid((10,4),(0,2), colspan=2, rowspan=2, projection="polar"),
          plt.subplot2grid((10,4),(3,2), colspan=2, rowspan=2, projection="polar"),
          plt.subplot2grid((10,4),(6,2), colspan=2, rowspan=2, projection="polar")
         ]

    ax_midline = np.zeros((16,))
    seq_reps = np.zeros((16,))

    # get matrix for orientations and spatial frequencies
    orientations = np.linspace(0, 315, 8)
    spatial_frequencies = np.array([0.01, 0.005])
    repetitions = 9

    # create a counter that increments after each episode of either a grating or black box
    i = 1
    seq_counter = 0
    seq_nr = np.zeros((np.size(behav_ds,0),))
    while i < np.size(behav_ds,0):
        if behav_ds[i,1] != behav_ds[i-1,1]:
            seq_counter += 1
        seq_nr[i] = seq_counter
        i += 1

    # create an array that holds all invdividual traces (with some room to spare to account
    # for fluctuation of the number of samples/trace/sequence)
    trace_length = np.ceil(dF_ds.shape[0]/(np.unique(seq_nr).shape[0]/2)) + 20
    all_traces = np.empty((int(np.ceil(dF_ds.shape[0]/(np.unique(seq_nr).shape[0]/2))) + 20,16,repetitions))
    all_traces[:] = np.NAN

    all_traces_gb = np.empty((int(np.ceil(dF_ds.shape[0]/(np.unique(seq_nr).shape[0]/2))) + 20,16,repetitions))
    all_traces_gb[:] = np.NAN

    # get transition points between sequences, add first and last index of array to have boundaries
    seq_transitions = np.where(np.diff(seq_nr) > 0)[0]
    seq_transitions = np.insert(seq_transitions,0,0)
    seq_transitions = np.append(seq_transitions,behav_ds.shape[0])

    # plot traces in respective subplots, assuming the regular grating sequence. -1 to account for odd number of sequences
    for i in range(seq_transitions.shape[0]-1):
        # determine current sequence, and current repition
        cur_seq = int((np.mod(i,32)-1)/2)
        cur_rep = int(i/32)
        # only grab every 2nd sequence transition as we plot 1 episode of black box + 1 grating together
        if np.mod(i,2) == 1:
            cur_seq_start = seq_transitions[i-1] + 1
            cur_seq_end = seq_transitions[i+1]
            cur_trace = dF_ds[cur_seq_start:cur_seq_end]
            ax[cur_seq].plot(cur_trace,c='0.5',lw=0.5)
            ax[18].plot(cur_trace,c='0.5',lw=0.5)
            # store all traces so we can later calculate the mean
            try:
                all_traces[0:cur_seq_end-cur_seq_start,cur_seq,cur_rep] = cur_trace
            except:
                print('WARNING')
            # record the length of each plotted graph as there can be some fluctuation due to framerate variation
            ax_midline[cur_seq] = ax_midline[cur_seq] + (cur_seq_end - cur_seq_start)
            seq_reps[cur_seq] = seq_reps[cur_seq] + 1

        # grab every 2nd sequence but such that the blackbox is in the second of the two sequences
        if np.mod(i,2) == 0 and i > 0:
            cur_seq_start = seq_transitions[i-1] + 1
            cur_seq_end = seq_transitions[i+1]
            cur_trace = dF_ds[cur_seq_start:cur_seq_end]
            ax[19].plot(cur_trace,c='0.5',lw=0.5)
            # store all traces so we can later calculate the mean
            try:
                all_traces_gb[0:cur_seq_end-cur_seq_start,cur_seq,cur_rep] = cur_trace
            except:
                print('WARNING')

    responses = np.zeros(16)

    # calculate and draw center of each subplot, and average trace and set up plot appearance
    for i in range(16):
        # get location tickmarks so they correspond with the timing of the sequence
        ax_xmax = ax_midline[i] / seq_reps[i]
        xticks_locs = np.linspace(0,ax_xmax,7)
        ax_mid = ax_midline[i] / seq_reps[i] / 2
        # plot mean trace
        ax[i].axvline(ax_mid,c='r')
        ax[i].plot(np.nanmean(all_traces[:,i,:],1),c='k',lw=2)

        responses[i] = np.nanmean(all_traces[int(ax_mid):,i,:]) - np.nanmean(all_traces[int(ax_mid/3)*2:int(ax_mid),i,:])
        r_vals[i] = np.nanmean(all_traces[int(ax_mid):,i,:]) - np.nanmean(all_traces[int(ax_mid/3)*2:int(ax_mid),i,:])
        if r_vals[i] < 0:
            r_vals[i] = 0

        # set up axis appearance
        ax[i].set_xlim([0,ax_midline[i] / seq_reps[i]])
        ax[i].set_xticks(xticks_locs.astype(int))
        ax[i].set_xticklabels(['-3','','','','','','3'])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')
        # force custom y lims if provided
        if ylims == []:
            ax[i].set_ylim([y_min,y_max])
        else:
            ax[i].set_ylim(ylims)
        # set title for each subplot
        if i < 8:
            ax[i].set_title(str(int(orientations[np.mod(i,8)]))+'/0.005 cycles/deg')
        else:
            ax[i].set_title(str(int(orientations[np.mod(i,8)]))+'/0.01 cycles/deg')

    theta = np.radians(np.linspace(0,360,8, endpoint=False))
    ax[20].plot(theta,r_vals[0:8])
    ax[21].plot(theta,r_vals[8:])
    ax[20].set_ylim([0,1])
    ax[21].set_ylim([0,1])
    r_mean = np.nanmean([r_vals[0:8],r_vals[8:]],0)
    ax[22].plot(theta,r_mean)
    ax[22].set_ylim([0,1])

    # calculate and draw center for summary plots, and average trace and set up plot appearance
    ax_xmax = np.sum(ax_midline) / np.sum(seq_reps)
    xticks_locs = np.linspace(0,ax_xmax,7)
    ax_mid = np.sum(ax_midline) / np.sum(seq_reps) / 2
    ax[18].axvline(ax_mid,c='r')
    ax[18].set_xlim([0,np.sum(ax_midline) / np.sum(seq_reps)])
    ax[18].set_xticks(xticks_locs.astype(int))
    ax[18].set_xticklabels(['-3','','','0','','','3'])
    ax[18].plot(np.nanmean(np.nanmean(all_traces[:,:,:],1),1),c='k',lw=2)
    ax[18].spines['top'].set_visible(False)
    ax[18].spines['right'].set_visible(False)
    ax[18].yaxis.set_ticks_position('left')
    ax[18].xaxis.set_ticks_position('bottom')
    ax[18].set_title('all traces black -> grating')

    ax[19].axvline(ax_mid,c='r')
    ax[19].set_xlim([0,np.sum(ax_midline) / np.sum(seq_reps)])
    ax[19].set_xticks(xticks_locs.astype(int))
    ax[19].set_xticklabels(['-3','','','0','','','3'])
    ax[19].plot(np.nanmean(np.nanmean(all_traces_gb[:,:,:],1),1),c='k',lw=2)
    ax[19].spines['top'].set_visible(False)
    ax[19].spines['right'].set_visible(False)
    ax[19].yaxis.set_ticks_position('left')
    ax[19].xaxis.set_ticks_position('bottom')
    ax[19].set_title('all traces grating -> black')

    gOSI = np.zeros(2)

    for i in range(16):
        if i < 8:
            gOSI[0] += responses[i]*np.exp(2j*orientations[i % 8]*np.pi/90.0)
        else:
            gOSI[1] += responses[i]*np.exp(2j*orientations[i % 8]*np.pi/90.0)

    gOSI[0] = gOSI[0]/np.sum(responses[0:8])
    gOSI[1] = gOSI[1]/np.sum(responses[8:])

    ### MOVEMENT ONSET GRAPH (ONLY IN SESSION THAT HAVE MOVEMENT DATA) ###

    if np.size(behav_ds,1) > 10:
        # timewindow (minus and plus time in seconds)
        EVENT_TIMEWINDOW = [3,3]
        # pull out indices for movement onset
        events = event_ind_gratings(behav_ds,['movement_onset',1,1])

        temp = events.astype('int')
        events = temp

        # grab peri-event dF trace for each event
        trial_dF = np.zeros((np.size(events[:,0]),2))
        for i,cur_ind in enumerate(events):
            # determine indices of beginning and end of timewindow
            if behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0] > behav_ds[0,0]:
                trial_dF[i,0] = np.where(behav_ds[:,0] < behav_ds[cur_ind[0],0]-EVENT_TIMEWINDOW[0])[0][-1]
            else:
                trial_dF[i,0] = 0
            if behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1] < behav_ds[-1,0]:
                trial_dF[i,1] = np.where(behav_ds[:,0] > behav_ds[cur_ind[0],0]+EVENT_TIMEWINDOW[1])[0][0]
            else:
                trial_dF[i,1] = np.size(behav_ds,0)
        # determine longest peri-event sweep (necessary due to sometimes varying framerates)
        t_max = int(np.amax(trial_dF[:,1] - trial_dF[:,0]))
        cur_sweep_resampled = np.zeros((np.size(events[:,0]),int(t_max)))
        cur_sweep_resampled_speed = np.zeros((np.size(events[:,0]),int(t_max)))

        # resample every sweep to match the longest sweep
        for i in range(np.size(trial_dF,0)):
            cur_sweep = dF_ds[int(trial_dF[i,0]):int(trial_dF[i,1])]
            cur_sweep_resampled[i,:] = signal.resample(cur_sweep, t_max, axis=0)
            ax[16].plot(cur_sweep_resampled[i,:],c='0.65',lw=1)

            cur_sweep_speed = behav_ds[int(trial_dF[i,0]):int(trial_dF[i,1]),7]
            cur_sweep_resampled_speed[i,:] = signal.resample(cur_sweep_speed, t_max, axis=0)
            ax[17].plot(cur_sweep_resampled_speed[i,:],lw=1,c='g')

        dF_mean = np.mean(cur_sweep_resampled,axis=0)
        speed_mean = np.mean(cur_sweep_resampled_speed,axis=0)

        ax[16].plot(dF_mean,c='k',lw=2)
        ax[16].set_xlim(0,t_max)
        ax[16].axvline(int(t_max/2),c='r')
        xticks_locs = np.linspace(0,t_max,7)
        ax[16].set_xticks(xticks_locs.astype(int))
        ax[16].set_xticklabels(['-3','-2','-1','0','1','2','3'])
        ax[16].spines['top'].set_visible(False)
        ax[16].spines['right'].set_visible(False)
        ax[16].yaxis.set_ticks_position('left')
        ax[16].xaxis.set_ticks_position('bottom')
        ax[16].set_title('dF/F movement onset centered')

        ax[17].plot(speed_mean,c='k',lw=2)
        ax[17].set_xlim(0,t_max)
        ax[17].axvline(int(t_max/2),c='r')
        xticks_locs = np.linspace(0,t_max,7)
        ax[17].set_xticks(xticks_locs.astype(int))
        ax[17].set_xticklabels(['-3','-2','-1','0','1','2','3'])
        ax[17].spines['top'].set_visible(False)
        ax[17].spines['right'].set_visible(False)
        ax[17].yaxis.set_ticks_position('left')
        ax[17].xaxis.set_ticks_position('bottom')
        ax[17].set_title('running speed (cm/sec)')

    ###

    #fig.suptitle('Grating' + fname, wrap=True)
    plt.tight_layout()
    # create subfolder if it doesn't exist
    if not os.path.isdir(content['figure_output_path'] + subfolder):
        os.mkdir(content['figure_output_path'] + subfolder)
    fname = content['figure_output_path'] + subfolder + os.sep + 'gsess_' + fname + '.' + fformat
    print(fname)
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)


    if True:


        number_of_shuffles = 10

        shuffled_gOSIs = np.zeros((number_of_shuffles, 2))

        for shuffle in range(number_of_shuffles):

            # create a counter that increments after each episode of either a grating or black box
            i = 1
            seq_counter = 0
            seq_nr = np.zeros((np.size(behav_ds,0),))
            while i < np.size(behav_ds,0):
                if behav_ds[i,1] != behav_ds[i-1,1]:
                    seq_counter += 1
                seq_nr[i] = seq_counter
                i += 1

            # seq_nr = np.roll(seq_nr, np.random.randint(0, np.size(behav_ds,0)))

            # create an array that holds all individual traces (with some room to spare to account
            # for fluctuation of the number of samples/trace/sequence)
            all_traces = np.empty((int(np.ceil(dF_ds.shape[0]/(np.unique(seq_nr).shape[0]/2))) + 20,16,repetitions))
            all_traces[:] = np.NAN

            all_traces_gb = np.empty((int(np.ceil(dF_ds.shape[0]/(np.unique(seq_nr).shape[0]/2))) + 20,16,repetitions))
            all_traces_gb[:] = np.NAN

            # get transition points between sequences, add first and last index of array to have boundaries
            seq_transitions = np.where(np.diff(seq_nr) > 0)[0]
            seq_transitions = np.insert(seq_transitions,0,0)
            seq_transitions = np.append(seq_transitions,behav_ds.shape[0])

            # plot traces in respective subplots, assuming the regular grating sequence. -1 to account for odd number of sequences
            for i in range(seq_transitions.shape[0]-1):
                # determine current sequence, and current repition
                cur_seq = int((np.mod(i,32)-1)/2)
                cur_rep = int(i/32)
                # only grab every 2nd sequence transition as we plot 1 episode of black box + 1 grating together
                if np.mod(i,2) == 1:
                    cur_seq_start = seq_transitions[i-1] + 1
                    cur_seq_end = seq_transitions[i+1]
                    cur_trace = dF_ds[cur_seq_start:cur_seq_end]

                # store all traces so we can later calculate the mean
                try:
                    all_traces[0:cur_seq_end-cur_seq_start,cur_seq,cur_rep] = cur_trace
                except:
                    pass # traces cropped at beginning and end of the session don't fit and I don't have time to fix this right now
                    # print('WARNING')

                # record the length of each plotted graph as there can be some fluctuation due to framerate variation
                ax_midline[cur_seq] = ax_midline[cur_seq] + (cur_seq_end - cur_seq_start)
                seq_reps[cur_seq] = seq_reps[cur_seq] + 1

                # grab every 2nd sequence but such that the blackbox is in the second of the two sequences
                if np.mod(i,2) == 0 and i > 0:
                    cur_seq_start = seq_transitions[i-1] + 1
                    cur_seq_end = seq_transitions[i+1]
                    cur_trace = dF_ds[cur_seq_start:cur_seq_end]

                    # store all traces so we can later calculate the mean
                    try:
                        all_traces_gb[0:cur_seq_end-cur_seq_start,cur_seq,cur_rep] = cur_trace
                    except:
                        pass # traces cropped at beginning and end of the session don't fit and I don't have time to fix this right now


            shuffled_responses = np.zeros((16))

            for i in range(16):
                ax_mid = ax_midline[i] / seq_reps[i] / 2

                shuffled_responses[i] = np.nanmean(all_traces[int(ax_mid):,i,:]) - np.nanmean(all_traces[int(ax_mid/3)*2:int(ax_mid),i,:])

            for i in range(16):
                if i < 8:
                    shuffled_gOSIs[shuffle, 0] += shuffled_responses[i]*np.exp(2j*orientations[i % 8]*np.pi/90.0)
                else:
                    shuffled_gOSIs[shuffle, 1] += shuffled_responses[i]*np.exp(2j*orientations[i % 8]*np.pi/90.0)

            shuffled_gOSIs[shuffle, 0] = shuffled_gOSIs[shuffle, 0]/np.sum(shuffled_responses[0:8])
            shuffled_gOSIs[shuffle, 1] = shuffled_gOSIs[shuffle, 1]/np.sum(shuffled_responses[8:])

        z_score = (gOSI - np.mean(shuffled_gOSIs))/np.std(shuffled_gOSIs)

        return gOSI, shuffled_gOSIs, z_score
    else:
        return 0.0,0.0,0.0
