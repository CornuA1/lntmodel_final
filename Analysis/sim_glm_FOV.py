"""
generate simulated FOV data. This function uses the data from a behavior session.

NOTE: calcium convolution code is based on:
    Lütcke, H., Gerhard, F., Zenke, F., Gerstner, W. & Helmchen, F.
    Inference of neuronal network spike dynamics and topology from calcium imaging data. Front. Neural Circuits 7, 1–20 (2013).

"""


import os, yaml, sys
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from multiprocessing import Pool, Process
import multiprocessing
from scipy.signal import butter, filtfilt, fftconvolve
import scipy.io as sio

sns.set_style("white")

# os.chdir('C:/Users/Lou/Documents/repos/LNT')
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')
sys.path.append(loc_info['base_dir'] + '/Figures')
# import fig_id_transients
# import fig_dfloc_trace_roiparams

from write_dict import write_dict
from filter_trials import filter_trials
from event_ind import event_ind
from load_behavior_data import load_data
from rewards import rewards
from licks import licks_nopost as licks

TRACK_SHORT = 3
TRACK_LONG = 4

def write_h5(h5dat, day, dset, dset_name):
    """ Write dataset to HDF-5 file. Overwrite if it already exists. """
    try:  # check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
        h5dat.create_dataset('Day' + str(day) + '/' + dset_name,
                             data=dset, compression='gzip')
    except:
        # if we want to overwrite: delete old dataset and then re-create with
        # new data
        del h5dat['Day' + str(day) + '/' + dset_name]
        h5dat.create_dataset('Day' + str(day) + '/' + dset_name,
                             data=dset, compression='gzip')

def load_raw_data(raw_filename):
    raw_data = load_data(raw_filename, 'vr')
    all_licks = licks(raw_data)
    trial_licks = all_licks[np.in1d(all_licks[:, 3], [TRACK_SHORT, TRACK_LONG]), :]
    reward =  rewards(raw_data)
    return raw_data, trial_licks, reward

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def spkTimes2Calcium(tauOn,ampFast,tauFast,ampSlow,tauSlow,frameRate,duration):
    x1 = np.arange(0,duration+(1/frameRate),1/frameRate)
    # y = (1-(np.exp(-(x1-spkT) / tauOn))) * (ampFast * np.exp(-(x1-spkT) / tauFast))+(ampSlow * np.exp(-(x1-spkT) / tauSlow))
    y = (1-(np.exp(-(x1) / tauOn))) * (ampFast * np.exp(-(x1) / tauFast))+(ampSlow * np.exp(-(x1) / tauSlow))
     # y = (1-(ex1p(-(x1-spkT)./tauOn))).*(ampFast*ex1p(-(x1-spkT)./tauFast));
    # y[x1 < spkT] = 0
    # y[np.isnan(y)] = 0

    return y

def place_fields(behav_ds, loc):
    """ return binary vector indicating whenever an animal was at a given location """
    # pull out behavior events and remove all events detected in the blackbox
    events_all = event_ind(behav_ds, ['at_location', loc,20])
    blackbox_trials = filter_trials(behav_ds, [], ['tracknumber',5])
    events_loc = np.zeros((0,2))
    for ev in events_all:
        if not ev[1] in blackbox_trials:
            events_loc = np.vstack([events_loc, ev])
    return events_loc

def reward_points(behav_ds, relative_point, relative_point_type='time'):
    """ return binary vector indicating whenever an animal got a reward. relative point allows to provide either time or distance from the reward point """
    # pull out behavior events and remove all events detected in the blackbox
    events_all = event_ind(behav_ds, ['rewards_all', -1])
    blackbox_trials = filter_trials(behav_ds, [], ['tracknumber',5])
    events_loc = np.zeros((0,2))
    for ev in events_all:
        if not ev[1] in blackbox_trials:
            events_loc = np.vstack([events_loc, ev])

    # find idx closest to reward point + relative_point
    if relative_point_type is 'time':
        for i,ev in enumerate(events_loc):
            ev_time = behav_ds[ev[0].astype(int),0]
            ev_time = ev_time + relative_point
            ev_idx = np.abs(behav_ds[:,0] - ev_time).argmin()
            events_loc[i,0] = ev_idx
    elif relative_point_type is 'space':
        for i,ev in enumerate(events_loc):
            ev_loc = behav_ds[ev[0].astype(int),1]
            ev_loc = ev_loc + relative_point
            cur_trial_start_idx = np.where(behav_ds[:,6]==ev[1].astype(int))[0][0]
            ev_idx = np.abs(behav_ds[behav_ds[:,6]==ev[1].astype(int),1] - ev_loc).argmin()
            ev_idx = ev_idx + cur_trial_start_idx
            events_loc[i,0] = ev_idx

    return events_loc

def track_point(behav_ds, loc, loc_cutoff, relative_point, E, relative_point_type='space'):
    """ return binary vector indicating whenever is at a given point. relative point allows to provide either time or distance from the specified point """
    # pull out behavior events and remove all events detected in the blackbox
    events_all = event_ind(behav_ds, ['at_location', loc, loc_cutoff])
    blackbox_trials = filter_trials(behav_ds, [], ['tracknumber',5])
    events_loc = np.zeros((0,2))
    for ev in events_all:
        if not ev[1] in blackbox_trials:
            events_loc = np.vstack([events_loc, ev])

    # find idx closest to reward point + relative_point
    if relative_point_type is 'time':
        for i,ev in enumerate(events_loc):
            ev_time = behav_ds[ev[0].astype(int),0]
            ev_time = ev_time + relative_point
            ev_idx = np.abs(behav_ds[:,0] - ev_time).argmin()
            events_loc[i,0] = ev_idx
    elif relative_point_type is 'space':
        for i,ev in enumerate(events_loc):
            ev_loc = behav_ds[ev[0].astype(int),1]
            ev_loc = ev_loc + relative_point + (np.random.randn(1) * E['RF_center_noise'])
            cur_trial_start_idx = np.where(behav_ds[:,6]==ev[1].astype(int))[0][0]
            ev_idx = np.abs(behav_ds[behav_ds[:,6]==ev[1].astype(int),1] - ev_loc).argmin()
            ev_idx = ev_idx + cur_trial_start_idx
            events_loc[i,0] = ev_idx

    return events_loc



def create_sim_cell(behav_ds, cell_type, place_cell_loc, S, E, fname, cell_index):
    # pull parameters for estimating calcium signal
    ca_genmode = S['ca_genmode'];
    spk_recmode = S['spk_recmode'];
    tauOn = S['tauOn'];
    A1 = S['A1'];
    tau1 = S['tau1'];
    A2 = S['A2'];
    tau2 = S['tau2'];
    samplingRate = S['samplingRate']
    snr = S['snr']
    # set up vectors to store behavior events and remove blackbox times
    # upsampling_factor = 1/((behav_ds[-1,0] - behav_ds[0,0])/behav_ds.shape[0])
    time_vector = behav_ds[:,0] - behav_ds[0,0]
    orig_samplingRate = 1/(time_vector[-1]/time_vector.shape[0])

    # print(cell_type)
    # ipdb.set_trace()
    if cell_type == 'place_cell':
        events_all = place_fields(behav_ds, place_cell_loc)
    elif cell_type == 'reward_cell':
        events_all = reward_points(behav_ds, place_cell_loc, 'space')
    elif cell_type == 'track_point':
        events_all = track_point(behav_ds, place_cell_loc[0], 20, place_cell_loc[1], E, 'space')
    else:
        print("ERROR: cell type not implemented")
        return np.nan

    # Grab running speed and and upsample it
    order = 6
    cutoff = 0.5 # desired cutoff frequency of the filter, Hz
    speed_filtered = butter_lowpass_filter(behav_ds[:,3], cutoff, orig_samplingRate, order)

    event_times = behav_ds[events_all[:,0].astype('int'),0]
    # create interpolation function to sample event_vector to a higher rate
    # ipdb.set_trace()
    print('upsampling behavior data...')
    x_points_interp = np.arange(0,time_vector[-1],1/samplingRate)
    x_points_orig = np.arange(0,time_vector[-1],1/orig_samplingRate)

    # if the interpolated vector is longer (can be a result of rounding errors), just crop
    if x_points_orig.shape[0] > time_vector.shape[0]:
        x_points_orig = x_points_orig[0:(time_vector.shape[0]-x_points_orig.shape[0])]

    time_vector_upsampled = np.interp(x_points_interp,x_points_orig,time_vector)
    trialtype_vector_upsampled = np.interp(x_points_interp,x_points_orig,behav_ds[:,4])
    speed_filtered_upsampled = np.interp(x_points_interp,x_points_orig,speed_filtered)


    # set binary events to nearest timepoints (may not be exactly the same anymore after interpolation)
    event_vector = np.zeros((np.ceil(time_vector[-1]*samplingRate).astype(int),))
    for ev in event_times:
        event_vector[np.abs(time_vector_upsampled-ev).argmin()] = 1
    # make convolution kernel to convolve binary events with
    # event_std = int(0.5*samplingRate)
    # gauss_kernel_len = int(4*samplingRate)
    # event_kernel = sp.signal.gaussian(gauss_kernel_len,event_std)
    e_duration = E['duration']
    e_tauOn = E['tauOn']
    e_tauFast = E['tauFast']
    e_ampFast = E['ampFast']
    e_scale_jitter = E['scale_jitter']
    x1 = np.arange(0,4+(1/samplingRate),1/samplingRate)
    event_kernel = (1-(np.exp(-x1/e_tauOn))) * (e_ampFast * np.exp(-x1/e_tauFast))
    event_kernel = event_kernel * (E['scale_amp'] + np.random.randint(-e_scale_jitter, e_scale_jitter))

    peak_idx = np.argmax(event_kernel)
    # convolve binary event vector with kernel and subsequently crop to original length
    n_samples = event_vector.shape[0]
    # rsigt = np.convolve(event_vector,event_kernel,mode='full')
    rsigt = fftconvolve(event_vector,event_kernel,mode='full')
    #peak_idx shifts the convolution kernel such that its peak coincides with the specified RF center
    rsigt = rsigt[peak_idx:n_samples+peak_idx]
    # convolve signal with speed signal to simulate
    speed_filtered_upsampled[speed_filtered_upsampled<0]=0
    speed_filtered_upsampled /= np.nanmax(speed_filtered_upsampled)
    # rsigt = rsigt + (speed_filtered_upsampled * S['run_speed_modulation'])
    rsigt = rsigt + (1 * (S['uniform_spike_noise']))

    rsigt = rsigt / samplingRate

    # create vector containing spike train with probability modulated by events
    spike_vector = np.zeros((event_vector.shape[0],))
    for i,lv in enumerate(rsigt):
        xi = np.random.uniform(0.0,1.0,1)
        if xi < rsigt[i]:
            spike_vector[i] = 1

    # convolve binary spiketrain with calcium kernel
    modelTransient = spkTimes2Calcium(tauOn,A1,tau1,A2,tau2,samplingRate,4);
    dff_vector_upsampled = np.convolve(spike_vector, modelTransient)
    dff_vector_upsampled = dff_vector_upsampled[0:time_vector_upsampled.shape[0]]
    # add white noise
    sdnoise = np.amax(dff_vector_upsampled) / snr
    whiteNoise = sdnoise * np.random.randn(dff_vector_upsampled.shape[0])
    dff_vector_upsampled = dff_vector_upsampled + whiteNoise
    dff_vector = sp.signal.resample(dff_vector_upsampled, time_vector.shape[0])


    # xlim = [0,50]

    # fig = plt.figure(figsize=(10,15))
    # plt.subplot(411)
    # plt.plot(time_vector, behav_ds[:,1])
    # plt.xlim(xlim)
    # plt.subplot(412)
    # plt.plot(time_vector_upsampled, event_vector,c='r')
    # plt.xlim(xlim)
    # plt.twinx()
    # plt.plot(time_vector_upsampled, rsigt,c='g',ls='--')
    # plt.ylim([0,np.amax(rsigt)])
    #
    # plt.xlim(xlim)
    #
    # plt.subplot(413)
    # plt.plot(time_vector_upsampled,spike_vector)
    # plt.xlim(xlim)
    #
    # plt.subplot(414)
    # plt.plot(time_vector,dff_vector)
    # plt.xlim(xlim)
    #
    # subfolder = 'SIM_1'
    # fformat = 'png'
    # if subfolder != []:
    #     if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
    #         os.mkdir(loc_info['figure_output_path'] + subfolder)
    #     fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    # else:
    #     fname = loc_info['figure_output_path'] + fname + '.' + fformat
    #
    # fig.savefig(fname, format=fformat,dpi=300)
    print(str(fname) + ' done')

    return [dff_vector, cell_index]

def create_sim_cell_worker(input_args):
    # print(input_args[0], input_args[1], input_args[2], input_args[3], input_args[4], input_args[5])
    res = create_sim_cell(input_args[0], input_args[1], input_args[2], input_args[3], input_args[4], input_args[5], input_args[6])
    return res

def sim_FOV(behav_ds, cell_types, S):
    animal_loc = behav_ds[:,1]

    if 'place_cell' in cell_types:
        for pc in cell_types['place_cell']:
            cell_dff = create_sim_cell(behav_ds, 'place_cell', pc, S, E)
            roi_dffs = np.vstack((roi_dffs,cell_dff))

    if 'reward_cell' in cell_types:
        for pc in cell_types['reward_cell']:
            cell_dff = create_sim_cell(behav_ds, 'reward_cell', pc, S, E)
            roi_dffs = np.vstack((roi_dffs,cell_dff))

    # ipdb.set_trace()

    if 'track_point' in cell_types:
        for pc in cell_types['track_point']:
            cell_dff = create_sim_cell(behav_ds, 'track_point', pc, S, pc[2], pc[3], pc[4])
            roi_dffs = np.vstack((roi_dffs,cell_dff))

    # if 'track_point' in cell_types:
    #     # ipdb.set_trace()
    #     worker_args = []
    #     for pc in cell_types['track_point']:
    #         worker_args.append([behav_ds, 'track_point', pc[0:2], S, pc[2], pc[3], pc[4]])
    #
    #     # ipdb.set_trace()
    #     p = multiprocessing.Pool()
    #     res_dff = p.map(create_sim_cell_worker, worker_args)
        # for pc in cell_types['track_point']:
        #     cell_dff = create_sim_cell(behav_ds, 'track_point', pc[0:2], S, pc[2], pc[3])
        #     roi_dffs = np.vstack((roi_dffs,cell_dff))

    # ipdb.set_trace()
    roi_dffs = np.zeros((animal_loc.shape[0],))
    cell_order = []
    for rd in res_dff:
        roi_dffs = np.vstack((roi_dffs,rd[0]))
        cell_order.append(rd[1])
    roi_dffs = roi_dffs[1:roi_dffs.shape[1],:]
    roi_dffs = roi_dffs[np.argsort(cell_order),:]
    return roi_dffs.T

if __name__ == "__main__":
    # set up model parameters
    S = {
        'ca_genmode' : 'linDFF',
        'spk_recmode' : 'linDFF',
        'ca_onsettau' : 0.02,
        'ca_amp' : 7600,
        'ca_gamma' : 400,
        'ca_amp1' : 0,
        'ca_tau1' : 0,
        'ca_kappas' : 100,
        'ca_rest' : 50,
        'dffmax' : 93,
        'kd' : 250,
        'conc' : 50000,
        'kappab' : 138.8889,
        'A1' : 1,
        'tau1' : 0.5,
        'A1sigma' : [],
        'tau1sigma' : [],
        'A2' : 0,
        'tau2' : 1,
        'tauOn' : 0.01,
        'dur' : 10,
        'spikeRate' : 0.2,
        'snr' : 7,
        'samplingRate' : 1000,
        'frameRate' : 15.5,
        'offset' : 1,
        'maxdt' : 0.5,
        'spkTimes' : [],
        'data_dff' : [],
        'data_ca' : [],
        'data_noisyDFF' : [],
        'uniform_spike_noise' : 0.05,
        'run_speed_modulation' : 0.002
    }

    E = {
        'duration' : 4,
        'tauOn' : 0.5,
        'tauFast' : 0.2,
        'ampFast' : 1,
        'scale_amp' : 20,
        'scale_jitter' : 15,
        'RF_center_noise' : 0
    }

    E_jitter = {
        'duration' : 4,
        'tauOn' : 0.5,
        'tauFast' : 0.2,
        'ampFast' : 1,
        'scale_amp' : 8,
        'scale_jitter' : 6,
        'RF_center_noise' : 60
    }

    E_wide_RF = {
        'duration' : 4,
        'tauOn' : 0.5,
        'tauFast' : 1.2,
        'ampFast' : 1,
        'scale_amp' : 1.35,
        'scale_jitter' : 1.2,
        'RF_center_noise' : 0
    }

    E_transients = {
        'duration' : 4,
        'tauOn' : 0.5,
        'tauFast' : 0.2,
        'ampFast' : 1,
        'scale_amp' : 5,
        'RF_center_noise' : 0
    }

    E_jitter_transients = {
        'duration' : 4,
        'tauOn' : 0.5,
        'tauFast' : 0.9,
        'ampFast' : 1,
        'scale_amp' : 1.5,
        'RF_center_noise' : 0
    }


    # behavior session for which simulated neurons are created for
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    BEHAVIOR_FILE  = 'MTH3_vr1_s5r2_2019128_2141.csv'
    data_path = loc_info['raw_dir'] + MOUSE
    raw_filename = data_path + os.sep + SESSION + os.sep + BEHAVIOR_FILE
    behav_ds, licks_ds, reward_ds = load_raw_data(raw_filename)

    # set basic parameters
    sim_mouse = 'SIM_2'
    sim_day = 'Day1'

    # number of neurons to simulate
    n = 1

    valid_rois = np.arange(0,n,1)
    invalid_rois = []

    # create dict with vali roi list
    roi_result_params = {
        'valid_rois' : valid_rois.tolist(),
        'invalid_rois' : invalid_rois
    }
    write_dict(sim_mouse, sim_day, roi_result_params, False, True, 'glm')

    # cell types
    cell_types = {
        # 'place_cell' : [200] #np.arange(50,320,50).tolist(),
        # 'reward_cell' : np.arange(-40,10,5).tolist(),
        # 'reward_cell' : np.arange(-5,1,0.5).tolist(),
        'track_point' : [[220,0,E,'lm1',1]]#,[220,-35,E,'lm2',2],[220,-30,E,'lm3',3]] #,[220,-25],[220,-20],[220,-15],[220,-10],[220,-5],[220,0],[220,5],[220,10],[220,15],[220,20],[220,30],[220,40],[220,50],[220,60],[220,70],[220,80]]
        # 'track_point' : [[220,-5],[220,-4.5],[220,-4],[220,-3.5],[220,-3],[220,-2.5],[220,-2],[220,-1.5],[220,-1],
        #                 [220,-0.5],[220,0],[220,0.5],[220,1.0],[220,1.5],[220,2],[220,2.5],[220,3],[220,3.5],
        #                 [220,4],[220,4.5],[220,5]]
        }





    # simulate dF/F traces
    dffs_vr = sim_FOV(behav_ds, cell_types, S)

    # save data
    sio.savemat(loc_info['raw_dir'] + sim_mouse + os.sep + sim_day + os.sep + 'aligned_data.mat', mdict={'dF_aligned' : dffs_vr, 'behaviour_aligned' : behav_ds})
    print('saved simulated data...')

    fig_id_transients.run_SIM_1_Day1()

    # cell_types_jitter = {
    #     'track_point' : [[220,-50,E_jitter,'lm0j'],[220,-40,E_jitter,'lm1j'],[220,-30,E_jitter,'lm2j'],[220,-20,E_jitter,'lm3j'],[220,-10,E_jitter,'lm4j'],[220,0,E_jitter,'lm5j']] #,
    #                      # [220,10,E_jitter,'lm6'],[220,20,E_jitter,'lm7'],[220,30,E_jitter,'lm8'],[220,40,E_jitter,'lm9'],[220,50,E_jitter,'lm10']]
    # }
    #
    # dffs = sim_FOV(behav_ds, cell_types_jitter, S)
    # sim_mouse = 'SIM_1'
    # sim_day = 'Day1'
    # sio.savemat(loc_info['raw_dir'] + sim_mouse + os.sep + sim_day + os.sep + 'aligned_data_jitter.mat', mdict={'dF_aligned' : dffs, 'behaviour_aligned' : behav_ds})
    # print('saved simulated data...')
    #
    #
    # cell_types_wider_RF = {
    #     'track_point' : [[220,-50,E_wider_RF,'lm0wrf'],[220,-40,E_wider_RF,'lm1wrf'],[220,-30,E_wider_RF,'lm2wrf'],[220,-20,E_wider_RF,'lm3wrf'],[220,-10,E_wider_RF,'lm4wrf'],[220,0,E_wider_RF,'lm5wrf']] #,
    #                      # [220,10,E_jitter,'lm6'],[220,20,E_jitter,'lm7'],[220,30,E_jitter,'lm8'],[220,40,E_jitter,'lm9'],[220,50,E_jitter,'lm10']]
    # }
    #
    # dffs = sim_FOV(behav_ds, cell_types_wider_RF, S)
    # sim_mouse = 'SIM_1'
    # sim_day = 'Day1'
    # sio.savemat(loc_info['raw_dir'] + sim_mouse + os.sep + sim_day + os.sep + 'aligned_data_wider_RF.mat', mdict={'dF_aligned' : dffs, 'behaviour_aligned' : behav_ds})
    # print('saved simulated data...')




    # cell_types_transients = {
    #     'track_point' : [[220,0,E_transients,'lm1t'],[220,-20,E_transients,'lm2t'],[220,-10,E_transients,'lm3t'],[220,10,E_transients,'lm4t'],[220,20,E_transients,'lm5t']]
    # }
    #
    # cell_types_jitter_transients = {
    #     'track_point' : [[220,0,E_jitter_transients,'lm1tj'],[220,-20,E_jitter_transients,'lm2tj'],[220,-10,E_jitter_transients,'lm3tj'],[220,10,E_jitter_transients,'lm4tj'],[220,20,E_jitter_transients,'lm5tj']]
    # }


    # dffs = sim_FOV(behav_ds, cell_types_jitter, S)
    # dffs = sim_FOV(behav_ds, cell_types_transients, S)
    # dffs = sim_FOV(behav_ds, cell_types_jitter_transients, S)

    # ipdb.set_trace()

    # sim_mouse = 'SIM_1'
    # sim_day = 'Day1'
    # sio.savemat(loc_info['raw_dir'] + sim_mouse + os.sep + sim_day + os.sep + 'aligned_data.mat', mdict={'dF_aligned' : dffs, 'behaviour_aligned' : behav_ds})
    # print('saved simulated data...')

    # write to hdf5 file
    # MOUSE = 'SIM_1'
    # DAY = '1'
    # sim_h5path_dir = loc_info['imaging_dir'] + MOUSE
    # sim_h5path_file = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
    # if not os.path.isdir(sim_h5path_dir):
    #     os.mkdir(sim_h5path_dir)

    # f = h5py.File(sim_h5path_file, 'w')
    # write_h5(f, DAY, behav_ds, 'behaviour_aligned')
    # write_h5(f, DAY, dffs, 'dF_win')
    # write_h5(f, DAY, licks_ds, 'licks_pre_reward')
    # write_h5(f, DAY, reward_ds, 'rewards')
    # f.close()
    #

    # flist = [
    #     fig_dfloc_trace_roiparams.run_SIM_1_Day1_jitter,
    #     fig_dfloc_trace_roiparams.run_SIM_1_Day1_wide_RF
    # ]
    # jobs = []
    # for fl in flist:
    #     p = Process(target=fl)
    #     jobs.append(p)
    #     p.start()
    #
    # for j in jobs:
    #     j.join()
    # fig_dfloc_trace_roiparams.run_SIM_1_Day1('Day1_wider_RF')
