"""
plot section of individual traces of individual, subcellular components

@author: Lukas Fischer

"""

import sys, os, yaml
with open('..' + os.sep + 'dendrite_imaging_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + '/Analysis')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy.io as sio
from align_sig import calc_dF, process_and_align_sigfile
import seaborn as sns
sns.set_style('white')



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_dF_and_movies(dF_signal, rois, roi_ids, frame_times, movie_types, movie_times, quadrature_data, fs, subfolder='unsorted transients', fname='transients', baseline_percentile=70):
    
    plot_figure = True
    # set standard deviation threshold above which traces have to be to be considered a transient
    std_threshold = 8
    # calculate frame to frame latency
    frame_latency = 1/fs
    # set minimum transient length in seconds that has to be above threshold
    min_transient_length_sec = 0.2
    # set max time between transients below which they nget combined into one transient
    gap_tolerance = 0.5
    # get the difference in quadrature data (i.e. movement frame to frame) and convert to cm/sec
    plot_quad_data = True
    quad_diff = (np.diff(quadrature_data) * 0.012566) * fs
    quad_diff[0] = 0
    quad_diff = np.insert(quad_diff,0,quad_diff[0])
    if quad_diff.shape[0] != dF_signal.shape[0]:
        quad_diff = sp.interpolate.griddata(np.arange(len(quad_diff)), quad_diff, np.arange(len(dF_signal[:,0])))
        
    animal_moving_idx = np.where(quad_diff > 5)[0]
    
    num_rois = len(rois)
    axrows_per_roi = int(100/num_rois)
    roi_ax = []
    
    if plot_figure:
        fig = plt.figure(figsize=(30,15))

    # keep track of all transient onsets and their classification
    transient_onsets = []
    independent_transients = []
    movie_transients = []
    dark_transients = []
    running_transients = []
    stationary_transients = []
    
    independent_fractions = []
    movie_independent_fractions = []
    dark_independent_fractions = []
    # determine indeces of where movies started and stopped playing
    movies_idx = []
    for i,mt in enumerate(movie_times):
        movies_idx.append((np.abs(frame_times - mt)).argmin())
        
    for i,roi in enumerate(rois):
        if plot_figure:
            if i > 0:
                roi_ax.append(plt.subplot2grid((100,100),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=100,sharex=roi_ax[0]))
            else:
                roi_ax.append(plt.subplot2grid((100,100),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=100))
            
            #print(roi_ids[i])
            roi_ax[-1].set_ylabel(roi_ids[i])
            roi_ax[-1].spines['left'].set_visible(False)
            roi_ax[-1].spines['top'].set_visible(False)
            roi_ax[-1].spines['right'].set_visible(False)
            roi_ax[-1].spines['bottom'].set_visible(False)
            roi_ax[-1].tick_params( \
                axis='both', \
                direction='out', \
                labelsize=8, \
                length=4, \
                width=2, \
                bottom='off', \
                right='off', \
                top='off')
        # get roi data to plot
        plot_roi = dF_signal[:,roi]
        order = 6
        cutoff = 4 # desired cutoff frequency of the filter, Hz
        roi_filtered = butter_lowpass_filter(plot_roi, cutoff, fs, order)
        plot_roi = roi_filtered
        
        if plot_figure:
            # get frame indeces
            frame_idx = np.arange(0,len(dF_signal[:,roi]),1)
            # plot rois
            roi_ax[i].plot(frame_idx, plot_roi,label='dF/F',c='k',lw=1)
        
        # shade movie areas
        if plot_figure:
            for j in range(len(movies_idx)):
                if j%2 == 0:
                    roi_ax[i].axvspan(movies_idx[j], movies_idx[j+1],color='#BABABA',alpha=0.2)
                
        # get roi parameters for transient id
        percentile_low_idx = np.where(plot_roi < np.percentile(plot_roi,baseline_percentile))[0]
        rois_std = np.std(plot_roi[percentile_low_idx])
        rois_mean = np.mean(plot_roi[percentile_low_idx])
        
#        if plot_figure:
#            roi_ax[i].axhspan(rois_mean, (std_threshold * rois_std) + rois_mean,color='#99D9EA',alpha=1.0)
     
        if i == 0:
            parent_transients = id_transients(plot_roi, std_threshold, rois_std, rois_mean, min_transient_length_sec, gap_tolerance, frame_latency)        
#            movie_parent_transients_idx = np.where((parent_transients[0] > movies_idx[0]) & (parent_transients[0] < movies_idx[1]))[0]
#            dark_parent_transients_idx = np.where((parent_transients[0] > movies_idx[1]))[0]
            # the syntactic abominations below just pull our transient onsets that happen when the animal is running or stationary
#            running_parent_transients_idx = np.array(parent_transients[0])[np.in1d(np.array(parent_transients[0]), animal_moving_idx)]
#            station__parent_transients_idx = np.array(parent_transients[0])[np.in1d(np.array(parent_transients[0]), animal_moving_idx, invert=True)]

                
            if plot_figure:
                for j in range(len(parent_transients[0])):
#                    try:
#                        if parent_transients[1][j] > len(plot_roi):
#                            print('test')
                    roi_ax[i].plot(np.arange(parent_transients[0][j],parent_transients[1][j],1),plot_roi[parent_transients[0][j]:parent_transients[1][j]],c='r',lw=1.5)
#                    except:
#                        print('ahoi')
        else:
            child_transients = id_transients(plot_roi, std_threshold, rois_std, rois_mean, min_transient_length_sec, gap_tolerance, frame_latency)        
#            child_transients = np.array(child_transients[0])[np.in1d(np.array(parent_transients[0]), animal_moving_idx)]
            
            # we crop what happens before the movie onset to get rid of epoch onset artifacts
            child_transients_after_movie_start_idx = np.where((child_transients[0] > movies_idx[0]))[0]
            
            movie_child_transients_idx = np.where((child_transients[0] > movies_idx[0]) & (child_transients[0] < movies_idx[1]))[0]
            dark_child_transients_idx = np.where((child_transients[0] > movies_idx[1]))[0]
            
            # get running and stationary and filter the ones before the movie
            running_child_transients_idx = np.nonzero(np.in1d(np.array(child_transients[0]), animal_moving_idx))[0]
            running_child_transients_idx = np.nonzero(np.in1d(running_child_transients_idx, child_transients_after_movie_start_idx))[0]
            station_child_transients_idx = np.nonzero(np.in1d(np.array(child_transients[0]), animal_moving_idx, invert=True))[0]
            station_child_transients_idx = np.nonzero(np.in1d(station_child_transients_idx, child_transients_after_movie_start_idx))[0]
            
            transient_onsets.append(child_transients_after_movie_start_idx)
            movie_transients.append(movie_child_transients_idx)
            dark_transients.append(dark_child_transients_idx)
            running_transients.append(running_child_transients_idx)
            stationary_transients.append(station_child_transients_idx)
            
            if len(child_transients[0]) > 0:
                ind_trans, fraction_independent = check_transient_overlap(child_transients, parent_transients)
                independent_transients.append(ind_trans)
                
                independent_fractions.append(fraction_independent)
                if len(child_transients[0]) is not None:
                    for j in range(len(child_transients[0])):
                        if not any(np.in1d(j, ind_trans)):
                            plot_col = 'r'
                        else:
                            plot_col = 'b'
#                        try:
                        if plot_figure:
                            roi_ax[i].plot(np.arange(child_transients[0][j],child_transients[1][j],1),plot_roi[child_transients[0][j]:child_transients[1][j]],c=plot_col,lw=1.5)
#                        except:
#                            pass    
            else:
                independent_transients.append([])
#            return
            
        # plot quadrature data
        
        if plot_quad_data and plot_figure:
            speed_ax = roi_ax[i].twinx()
            speed_ax.fill_between(frame_idx,np.zeros((frame_idx.shape[0])),quad_diff,**{'alpha':'0.3','color':'green', 'linewidth':'0'})
            speed_ax.set_ylim([0, np.nanmax(quad_diff)])
            speed_ax.spines['left'].set_visible(False)
            speed_ax.spines['top'].set_visible(False)
            speed_ax.spines['right'].set_visible(False)
            speed_ax.spines['bottom'].set_visible(False)
            speed_ax.tick_params( \
                axis='both', \
                direction='out', \
                labelsize=8, \
                length=4, \
                width=2, \
                bottom='off', \
                right='off', \
                top='off')
        
    if plot_figure:    
        fig.suptitle(fname, wrap=True)
        if subfolder != []:
            if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
                os.mkdir(loc_info['figure_output_path'] + subfolder)
            fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
        else:
            fname = loc_info['figure_output_path'] + fname + '.' + fformat
        fig.savefig(fname, format=fformat,dpi=150)
    print(fname)
#    
    
    # some basic quantification
    total_length = len(frame_times[frame_times > movie_times[0]])/fs
    movie_length = len(frame_times[(frame_times < movie_times[1]) & (frame_times > movie_times[0])])/fs
    dark_length = len(frame_times[frame_times > movie_times[1]])/fs
    moving_length = len(animal_moving_idx[animal_moving_idx > movies_idx[0]])/fs
    
    animal_station_idx = np.where(quad_diff <= 5)[0]
    station_length = len(animal_station_idx[animal_station_idx > movies_idx[0]])/fs
    
    tot_num = np.nansum([len(x) for x in transient_onsets])/total_length/num_rois
    tot_num_independent = np.nansum([len(x) for x in independent_transients])/total_length/num_rois
    tot_num_movie = np.nansum([len(x) for x in movie_transients])/movie_length/num_rois
    tot_num_dark = np.nansum([len(x) for x in dark_transients])/dark_length/num_rois
    if moving_length > 0.0:
        tot_num_running = np.nansum([len(x) for x in running_transients])/moving_length/num_rois
    else:
        tot_num_running = 0
    if station_length > 0.0:
        tot_num_station = np.nansum([len(x) for x in stationary_transients])/station_length/num_rois
    else:
        tot_num_station = 0
    
    independent_movie = []
    independent_dark = []
    independent_running = []
    independent_station = []
    
    for i in range(len(independent_transients)):
        independent_movie.append(np.count_nonzero(np.in1d(independent_transients[i], movie_transients[i])))
    for i in range(len(independent_transients)):
        independent_dark.append(np.count_nonzero(np.in1d(independent_transients[i], dark_transients[i])))
    for i in range(len(independent_transients)):
        independent_running.append(np.count_nonzero(np.in1d(independent_transients[i], running_transients[i])))
    for i in range(len(independent_transients)):
        independent_station.append(np.count_nonzero(np.in1d(independent_transients[i], stationary_transients[i])))    

    independent_movie = np.sum(independent_movie)/movie_length/num_rois
    independent_dark = np.sum(independent_dark)/dark_length/num_rois
    if moving_length > 0.0:
        independent_running = np.sum(independent_running)/moving_length/num_rois
    else:
        independent_running = 0
    if station_length > 0.0:
        independent_station = np.sum(independent_station)/station_length/num_rois
    else:
        independent_station = 0
    
    return {'total_number': tot_num, \
            'total_independent' : tot_num_independent, \
            'total_movie' : tot_num_movie, \
            'total_dark' : tot_num_dark, \
            'total_running' : tot_num_running, \
            'total_stationary' : tot_num_station, \
            'total_independent_movie' : independent_movie, \
            'total_independent_dark' : independent_dark, \
            'total_independent_running' : independent_running, \
            'total_independent_stationary' : independent_station
           }
    

def check_transient_overlap(child_transients, parent_transients):
    """ check whether transients of a given ROI overlap with the transients of a 'parent' ROI """
    
    # get transient idx of parent
    parent_transient_idx = []
    for i in range(len(parent_transients[0])):
        parent_transient_idx.append(np.arange(parent_transients[0][i],parent_transients[1][i]+1))
    
    # run through every child transient and test whether it overlaps
    if child_transients is not None:
        independent_transient = np.ones((len(child_transients[0])))
        for i in range(len(child_transients[0])):
            transient_idx = np.arange(child_transients[0][i],child_transients[1][i]+1)
            for j in range(len(parent_transients[0])):
                if np.any(np.in1d(transient_idx,parent_transient_idx[j])):
                    independent_transient[i] = 0
                    
        # calculate fraction of independent events
        if len(np.unique(independent_transient)) > 1:
            num_independent = np.unique(independent_transient, return_counts=True)[1][1]
        else:
            if independent_transient[0] == 0:
                num_independent = 0
            else:
                num_independent = len(independent_transient)

                
        fraction_independent = num_independent/len(child_transients[0])
        
    else:
        independent_transient = None
        fraction_independent = None
    
    independent_transient = np.where(independent_transient == 1)[0]
    return independent_transient, fraction_independent
    
            
    
def id_transients(roi_data, std_threshold, rois_std, rois_mean, min_transient_length_sec, gap_tolerance, frame_latency):
    # get indeces above speed threshold
    transient_high = np.where(roi_data > (std_threshold * rois_std) + rois_mean)[0]
    onsets = []
    offsets = []
    
    if transient_high.size != 0:
        # use diff to find gaps between episodes of high speed
        idx_diff = np.diff(transient_high)
        idx_diff = np.insert(idx_diff,0,transient_high[0])
    
        # min transient length in number of frames
        min_transient_length = min_transient_length_sec/frame_latency
        gap_tolerance_frames = int(gap_tolerance/frame_latency)
        # find indeces where speed exceeds threshold. If none are found, return
        min_transient_length = 4
        # find transient onset and offset points.
        onset_idx = []
        offset_idx = []
        is_transient = False
        for j in range(roi_data.shape[0]):
            if roi_data[j] > (std_threshold * rois_std) + rois_mean:
                if not is_transient:
                    is_transient = True
                    onset_idx.append(j)
            else:
                if is_transient:
                    is_transient = False
                    offset_idx.append(j)
                    
        if len(offset_idx) < len(onset_idx):
            offset_idx.append(len(roi_data.shape))
    
        # find transient onset and offset points. Colapse transients that briefly dip below threshold
        #gap_tolerance_frames
#        onset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]]-1
#        offset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]-1]+1
        # this is necessary as the first index of the offset is actually the end of the last one (this has to do with indexing conventions on numpy)
#        offset_idx = np.roll(offset_idx, -1)
#        offset_idx[-1] = transient_high[-1]

        
        # calculate the length of each transient and reject those too short to be considered
        index_adjust = 0
        for j in range(len(onset_idx)):
            temp_length = offset_idx[j-index_adjust] - onset_idx[j-index_adjust]
            if temp_length < min_transient_length:
                onset_idx = np.delete(onset_idx,j-index_adjust)
                offset_idx = np.delete(offset_idx,j-index_adjust)
                index_adjust += 1
                
        # combine transients that are close together
        index_adjust = 0
        for j in range(len(onset_idx)-1):
            temp_length = offset_idx[j+1-index_adjust] - onset_idx[j-index_adjust]
            if (offset_idx[j+1-index_adjust] - onset_idx[j]-index_adjust) < gap_tolerance_frames:
                onset_idx = np.delete(onset_idx,j-index_adjust)
                offset_idx = np.delete(offset_idx,j+1-index_adjust)
                index_adjust += 1

        # find the onset by looking for the point where the transient drops below 1 std
        above_1_std = np.where(roi_data > (rois_std*2)+rois_mean)[0]
        one_std_idx_diff = np.diff(above_1_std)
        one_std_idx_diff = np.insert(one_std_idx_diff,0,one_std_idx_diff[0])
    
        one_std_onset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]]-1
        one_std_offset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]-1]+1

        # run through every transient and find where signal drops below 1 std
        for j,oi in enumerate(onset_idx):
            closest_onset_idx = one_std_onset_idx - oi
            closest_onset_idx_neg = np.where(closest_onset_idx < 0)[0]
            closest_offset_idx = one_std_offset_idx - offset_idx[j]
            closest_offset_idx_neg = np.where(closest_offset_idx > 0)[0]
            
            if closest_onset_idx_neg.size == 0:
                closest_onset_idx_neg = [-1]
                one_onset_std_idx = 0
            else:
                one_onset_std_idx = np.min(np.abs(closest_onset_idx[closest_onset_idx_neg])) 
                
            if closest_offset_idx_neg.size == 0:
                closest_offset_idx_neg = [-1]
                one_offset_std_idx = 0
            else:
                one_offset_std_idx = np.min(np.abs(closest_offset_idx[closest_offset_idx_neg]))
            
            # only append if the onset/offset is not the same onset as the previous transient
            actual_onset_idx = oi-one_onset_std_idx
            actual_offset_idx = offset_idx[j]+one_offset_std_idx
            
            if len(onsets) > 0:
                if not actual_onset_idx == onsets[-1]:
                    onsets.append(actual_onset_idx)
                    offsets.append(actual_offset_idx)
            else:
                onsets.append(actual_onset_idx)
                offsets.append(actual_offset_idx)

        # find max transient length
        max_transient_length = 0
        for j in range(len(onsets)):
            if offsets[j]-onsets[j] > max_transient_length:
                max_transient_length = offsets[j]-onsets[j]
    
            
    return [onsets, offsets]

def read_movie_times(fpath):
    movie_type = []
    movie_time = []
    with open(fpath,'r') as movietimes_file:
        for line in movietimes_file:
            movie_type.append(int(float(line.split(' ')[0])))
            movie_time.append(float(line.split(' ')[1]))
    
    return movie_type, movie_time

def get_roi_children(roiIDs, parent_roi):    
    # loop through every roi in the list and determine if its a child of the provided parent
    children_rois_id = []
    children_rois_index = []
    for i,roi in enumerate(roiIDs):
        # check if leading 0 is missing, append if so
        if(len(str(roi[0])) % 2 > 0):
            r_id = '0' + str(roi[0])
        if r_id[0:2] == parent_roi:
            children_rois_id.append(r_id)
            children_rois_index.append(i)
                
    # convert to integer and return
    return children_rois_id, children_rois_index
            

def run_LF190409_1():
    MOUSE= 'LF190409_1'
    sess = '190514_1'
    sigfile = 'M01_000_001.sig'
    meta_file = 'M01_000_001.extra'
    data_path = loc_info['raw_dir'] + MOUSE
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    dF_signal = loaded_data['dF_data']

    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10], fs=15.5, subfolder='LF190409_1 transients', fname='transients_0', baseline_percentile=70)
    id_and_plot_transients(dF_signal, [0,11,12,13,14,15,16,17,18,19,20], fs=15.5, subfolder='LF190409_1 transients', fname='transients_1', baseline_percentile=70)
    id_and_plot_transients(dF_signal, [0,21,22,23,24,25,26,27,28,29,30], fs=15.5, subfolder='LF190409_1 transients', fname='transients_2', baseline_percentile=70)
#    id_and_plot_transients(dF_signal, [0,31,32,33,34,35,36,37,38,39,40], fs=15.5, subfolder='LF190409_1 transients', fname='transients_3', baseline_percentile=70)
#    id_and_plot_transients(dF_signal, [0,41,42,43,44,45,46,47,48,49,50], fs=15.5, subfolder='LF190409_1 transients', fname='transients_4', baseline_percentile=70)
#    id_and_plot_transients(dF_signal, [0,51,52,53,54,55,56], fs=15.5, subfolder='LF190409_1 transients', fname='transients_5', baseline_percentile=70)
    
def run_Jimmy():
    MOUSE= 'Jimmy'
    sess = '20190719_004'
#    sigfile = 'M01_000_000.sig'
#    meta_file = 'M01_000_000.extra'
#    data_path = loc_info['raw_dir'] + MOUSE
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    dF_signal = loaded_data['dF_data']

    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10,12], fs=15.5, subfolder='Jimmy spine transients', fname='transients_0', baseline_percentile=70)

def run_LF190716_1():
    MOUSE= 'LF190716_1'
#    sess = '20190722_003'
#    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
#    loaded_data = sio.loadmat(processed_data_path)
#    dF_signal = loaded_data['dF_data']
#    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10,12], fs=31, subfolder='Jimmy spine transients', fname='transients_0', baseline_percentile=70)

    sess = '20190722_002'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
    loaded_data = sio.loadmat(processed_data_path)
    dF_signal = loaded_data['dF_data']

    id_and_plot_transients(dF_signal, [0,1,2,3,4,5,6,7,8,9,10,12,13], fs=31, subfolder='Jimmy spine transients', fname='transients_0', baseline_percentile=50)

def run_Buddha_0802():
    MOUSE= 'Buddha'
    sess = '190802_2'
    
    loaded_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat')
    dF_signal = loaded_data['dF_data']
    
    metadata = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'Buddha_000_002.mat')
    frame_times = metadata['info'][0][0][25][0]
    
    movietimes_file_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'Buddha_000_002 movie times.txt'
    movie_types, movie_times = read_movie_times(movietimes_file_path)
    
    quadrature_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'Buddha_000_002_quadrature.mat')

    
    plot_dF_and_movies(dF_signal, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], frame_times, movie_types, movie_times, fs=15.5, subfolder='Buddha transients', fname='transients_0', baseline_percentile=70)
#    id_and_plot_transients(dF_signal, [10,11,12,13,14,15,16,17,18,19,20], fs=15.5, subfolder='Buddha transients', fname='transients_1', baseline_percentile=50)
#    id_and_plot_transients(dF_signal, [20,21,22,23,24,25,26,27,28,29,30], fs=15.5, subfolder='Buddha transients', fname='transients_2', baseline_percentile=50)
#    id_and_plot_transients(dF_signal, [30,31,32,33,34,35], fs=15.5, subfolder='Buddha transients', fname='transients_3', baseline_percentile=50)

def run_Anakin_0816_41():
    # data info
    MOUSE= 'Buddha'
    sess = '190816'
    fname = 'tufts'
    recording_numbers = ['41','42','43','44','45','46','47','48','49','50']
    datafile_names = ['sig_data01.mat','sig_data02.mat','sig_data03.mat','sig_data04.mat','sig_data05.mat','sig_data06.mat',
                 'sig_data07.mat','sig_data08.mat','sig_data09.mat','sig_data10.mat']
    roi_parents = ['01','02','03','04','05']
    run_multiple_epochs(MOUSE, sess, fname, recording_numbers, datafile_names, roi_parents)
        
def run_Anakin_0816_51():
    # data info
    MOUSE= 'Buddha'
    sess = '190816'
    fname = 'basals'
    recording_numbers = ['51','52','53','54','55','56','57','58','59','60']
    datafile_names = ['sig_data01.mat','sig_data02.mat','sig_data03.mat','sig_data04.mat','sig_data05.mat','sig_data06.mat',
                 'sig_data07.mat','sig_data08.mat','sig_data09.mat','sig_data10.mat']
    roi_parents = ['01','02','03']
    run_multiple_epochs(MOUSE, sess, fname, recording_numbers, datafile_names, roi_parents)

def run_Anakin_0816_61():
    # data info
    MOUSE= 'Buddha'
    sess = '190816'
    fname = 'obliques'
    recording_numbers = ['61','62','63','64','65','66','67','68','69','70']
    datafile_names = ['sig_data01.mat','sig_data02.mat','sig_data03.mat','sig_data04.mat','sig_data05.mat','sig_data06.mat',
                 'sig_data07.mat','sig_data08.mat','sig_data09.mat','sig_data10.mat']
    roi_parents = ['01']
    run_multiple_epochs(MOUSE, sess, fname, recording_numbers, datafile_names, roi_parents)     
   
def run_multiple_epochs(mousename, sess, fname, recording_numbers, datafile_names, roi_parents):
    # data info
    MOUSE= mousename
    
    # load data
    datasets = []    
    quadrature_data = []
    frame_times = []
    movie_types = []
    movie_times = []
    all_rois = []
    for i in range(len(datafile_names)):
        movietimes_file_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + '_' + recording_numbers[i] + os.sep + 'Buddha_000_0%s movie times.txt'%recording_numbers[i]
        m_types, m_times = read_movie_times(movietimes_file_path)
        movie_types.append(m_types)
        movie_times.append(m_times)
        
        metadata = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + '_' + recording_numbers[i]  + os.sep + 'Buddha_000_0%s_rigid.extra'%recording_numbers[i])
        frame_times.append(metadata['timestamps'][0])
        roiIDs = metadata['roiIDs']
        rois = []
        for rp in roi_parents:
            roi_nm, rois_id = get_roi_children(roiIDs, rp)
            rois.append([roi_nm,rois_id])
        all_rois.append(rois)
        
        parents_only = [[],[]]
        for ar in all_rois[0]:
            parents_only[0].append(ar[0][0])
            parents_only[1].append(ar[1][0])
            

        quad_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + '_' + recording_numbers[i]  + os.sep + 'Buddha_000_0%s_quadrature.mat'%recording_numbers[i])['quad_data']
        quadrature_data.append(quad_data[0])
        loaded_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + '_' + recording_numbers[i]  + os.sep + datafile_names[i])
        datasets.append(loaded_data['dF_data'])
    
    transient_results = {'total_number': [], 
                         'total_independent' : [], 
                         'total_movie' : [], 
                         'total_dark' : [], 
                         'total_running' : [], 
                         'total_stationary' : [], 
                         'total_independent_movie' : [], 
                         'total_independent_dark' : [], 
                         'total_independent_running' : [], 
                         'total_independent_stationary' : []
                        }
    if len(parents_only[0]) > 1:
        for j,ds in enumerate(datasets):
            res_data = plot_dF_and_movies(ds, parents_only[1], parents_only[0], frame_times[j], movie_types[j], movie_times[j], quadrature_data[j], fs=15.5, subfolder='Anakin transients', fname=fname+'_parents_'+datafile_names[j].split('.')[0], baseline_percentile=30)
                 
    for i,rs in enumerate(rois):
        for j,ds in enumerate(datasets):
            res_data = plot_dF_and_movies(ds, rs[1], rs[0], frame_times[j], movie_types[j], movie_times[j], quadrature_data[j], fs=15.5, subfolder='Anakin transients', fname=fname+'_'+roi_parents[i]+'_'+datafile_names[j].split('.')[0], baseline_percentile=30)
            if j == 0:
                transient_results['total_number'].append(res_data['total_number'])
                transient_results['total_independent'].append(res_data['total_independent'])
                transient_results['total_movie'].append(res_data['total_movie'])
                transient_results['total_dark'].append(res_data['total_dark'])
                transient_results['total_running'].append(res_data['total_running'])
                transient_results['total_stationary'].append(res_data['total_stationary'])
                transient_results['total_independent_movie'].append(res_data['total_independent_movie'])
                transient_results['total_independent_dark'].append(res_data['total_independent_dark'])
                transient_results['total_independent_running'].append(res_data['total_independent_running'])
                transient_results['total_independent_stationary'].append(res_data['total_independent_stationary'])
            else:
                transient_results['total_number'][i] = transient_results['total_number'][i] + res_data['total_number']
                transient_results['total_independent'][i] = transient_results['total_independent'][i] + res_data['total_independent']
                transient_results['total_movie'][i] = transient_results['total_movie'][i] + res_data['total_movie']
                transient_results['total_dark'][i] = transient_results['total_dark'][i] + res_data['total_dark']
                transient_results['total_running'][i] = transient_results['total_running'][i] + res_data['total_running']
                transient_results['total_stationary'][i] = transient_results['total_stationary'][i] + res_data['total_stationary']
                transient_results['total_independent_movie'][i] = transient_results['total_independent_movie'][i] + res_data['total_independent_movie']
                transient_results['total_independent_dark'][i] = transient_results['total_independent_dark'][i] + res_data['total_independent_dark']
                transient_results['total_independent_running'][i] = transient_results['total_independent_running'][i] + res_data['total_independent_running']
                transient_results['total_independent_stationary'][i] = transient_results['total_independent_stationary'][i] + res_data['total_independent_stationary']
        
            
    print(transient_results)
    print('total number: ' + str(np.round(np.mean(transient_results['total_number']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_number']),4)))
    print('total independent: ' + str(np.round(np.mean(transient_results['total_independent']))) + ' ' + str(np.round(sp.stats.sem(transient_results['total_independent']),4)))
    print('total movie: ' + str(np.round(np.mean(transient_results['total_movie']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_movie']),4)))
    print('total dark: ' + str(np.round(np.mean(transient_results['total_dark']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_dark']),4)))
    print('total running: ' + str(np.round(np.mean(transient_results['total_running']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_running']),4)))
    print('total stationary: ' + str(np.round(np.mean(transient_results['total_stationary']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_stationary']),4)))
    print('total independent movie: ' + str(np.round(np.mean(transient_results['total_independent_movie']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_independent_movie']),4)))
    print('total independent dark: ' + str(np.round(np.mean(transient_results['total_independent_dark']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_independent_dark']),4)))
    print('total independent running: ' + str(np.round(np.mean(transient_results['total_independent_running']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_independent_running']),4)))
    print('total independent stationary: ' + str(np.round(np.mean(transient_results['total_independent_stationary']),4)) + ' ' + str(np.round(sp.stats.sem(transient_results['total_independent_stationary']),4)))
    
    
    num_branches = len(transient_results['total_number'])
    fig = plt.figure(figsize=(8,4))
    sum_ax = plt.subplot(121)
    sum_ax.scatter(np.full(num_branches,0), transient_results['total_number'])
    sum_ax.scatter(np.full(num_branches,1), transient_results['total_movie'])
    sum_ax.scatter(np.full(num_branches,2), transient_results['total_dark'])
    sum_ax.scatter(np.full(num_branches,3), transient_results['total_running'])
    sum_ax.scatter(np.full(num_branches,4), transient_results['total_stationary'])
    sum_ax.scatter([0,1,2,3,4],[np.mean(transient_results['total_number']),np.mean(transient_results['total_movie']),np.mean(transient_results['total_dark']),np.mean(transient_results['total_running']),np.mean(transient_results['total_stationary'])], c='k')
    sum_ax.plot([0,1,2,3,4],[np.mean(transient_results['total_number']),np.mean(transient_results['total_movie']),np.mean(transient_results['total_dark']),np.mean(transient_results['total_running']),np.mean(transient_results['total_stationary'])], c='k')
    sum_ax.set_xticks([0,1,2,3,4])
    sum_ax.set_xticklabels(['total','movie','dark','running','stationary'], rotation=45)
    sum_ax.set_ylabel('transients/min/roi')
#    sum_ax.set_ylim([0.35,0.9])
    
    sum_ax2 = plt.subplot(122)
    sum_ax2.scatter(np.full(num_branches,0), transient_results['total_independent'])
    sum_ax2.scatter(np.full(num_branches,1), transient_results['total_independent_movie'])
    sum_ax2.scatter(np.full(num_branches,2), transient_results['total_independent_dark'])
    sum_ax2.scatter(np.full(num_branches,3), transient_results['total_independent_running'])
    sum_ax2.scatter(np.full(num_branches,4), transient_results['total_independent_stationary'])
    sum_ax2.scatter([0,1,2,3,4],[np.mean(transient_results['total_independent']),np.mean(transient_results['total_independent_movie']),np.mean(transient_results['total_independent_dark']),np.mean(transient_results['total_independent_running']),np.mean(transient_results['total_independent_stationary'])], c='k')
    sum_ax2.plot([0,1,2,3,4],[np.mean(transient_results['total_independent']),np.mean(transient_results['total_independent_movie']),np.mean(transient_results['total_independent_dark']),np.mean(transient_results['total_independent_running']),np.mean(transient_results['total_independent_stationary'])], c='k')
    sum_ax2.set_xticks([0,1,2,3,4])
    sum_ax2.set_xticklabels(['independent','ind movie','ind dark','ind running','ind stationary'], rotation=45)
#    sum_ax2.set_ylim([0.05,0.2])
    
    sum_ax2.set_ylabel('independent transients/min/roi')
    plt.tight_layout()
    
    fname = fname + ' summary plot'
    subfolder = 'Anakin transients'
    fformat = 'png'
    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    fig.savefig(fname, format=fformat,dpi=150)
    
    print(fname)
    

if __name__ == '__main__':
    
    fformat = 'png'

#    run_LF170613_1()
#    run_LF190409_1()
#    run_Jimmy()
#    run_LF190716_1()
#    run_Buddha_0802()
#    run_Anakin_0816_41()
#    run_Anakin_0816_51()
    run_Anakin_0816_61()

   