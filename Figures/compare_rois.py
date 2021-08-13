# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:53:16 2019

@author: Lou
"""

import sys, os, yaml
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f, Loader=yaml.FullLoader)
sys.path.append(loc_info['base_dir'] + '/Analysis')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as scistat
import seaborn as sns
sns.set_style('white')

def read_movie_times(fpath):
    movie_type = []
    movie_time = []
    with open(fpath,'r') as movietimes_file:
        for line in movietimes_file:
            movie_type.append(int(float(line.split(' ')[0])))
            movie_time.append(float(line.split(' ')[1]))
    
    return movie_type, movie_time

def plot_trans(dF_signal, baseline_percentile, rois_list, roi_ids, movies_idx, on_rois, off_rois, transients, sess, subfolder, fname):
#    roid_list = [101,104,10602,14,16]
#    index_list = []
#    roi_ids = [x[0] for x in roi_ids]
#    for ri in roid_list:
#        index_list.append(roi_ids.index(ri))
#    rois_list = index_list
    num_rois = len(rois_list)
    if num_rois == 0:
        return
    axrows_per_roi = int(100/num_rois)
    roi_ax = []
    std_threshold = 6
    fig = plt.figure(figsize=(15,15))
    for i,r in enumerate(rois_list):
        if i > 0:
            roi_ax.append(plt.subplot2grid((100,50),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50,sharex=roi_ax[0]))
            plt.ylabel(r)
        else:
            roi_ax.append(plt.subplot2grid((100,50),(i*axrows_per_roi,0), rowspan=axrows_per_roi, colspan=50))
            plt.title(sess)
            plt.ylabel(str(roi_ids[r]))
        roi_ax[-1].spines['left'].set_linewidth(2)
        roi_ax[-1].spines['top'].set_visible(False)
        roi_ax[-1].spines['right'].set_visible(False)
        roi_ax[-1].spines['bottom'].set_visible(False)
        roi_ax[-1].tick_params( \
            axis='both', \
            direction='out', \
            labelsize=16, \
            length=4, \
            width=2, \
            bottom='off', \
            right='off', \
            top='off')
        
    for i,roi in enumerate(rois_list):
        rois_unfiltered = dF_signal[:,roi]
        percentile_low_idx = np.where(rois_unfiltered < np.percentile(rois_unfiltered,baseline_percentile))[0]
        rois_std = np.std(rois_unfiltered[percentile_low_idx])
        rois_mean = np.mean(rois_unfiltered[percentile_low_idx])
        roi_ax[i].axhspan(rois_mean, (std_threshold*rois_std)+rois_mean, color='0.9',zorder=0)
        rois_idx = np.arange(0,len(dF_signal[:,roi]),1)
        roi_ax[i].plot(rois_idx, rois_unfiltered,label='dF/F',c='k',lw=1)
        for tran in transients:
            if tran[0] == roi:
                for on_df in range(len(on_rois[roi])):
                    if on_rois[roi][on_df] < tran[2] < off_rois[roi][on_df]:
                        if tran[6] > 0.6:
                            roi_ax[i].plot(np.arange(on_rois[roi][on_df],off_rois[roi][on_df],1),rois_unfiltered[on_rois[roi][on_df]:off_rois[roi][on_df]],c='red',lw=1.5)
                        else:
                            roi_ax[i].plot(np.arange(on_rois[roi][on_df],off_rois[roi][on_df],1),rois_unfiltered[on_rois[roi][on_df]:off_rois[roi][on_df]],c='blue',lw=1.5)
        for j in range(len(movies_idx)):
            if j%2 == 0:
                roi_ax[i].axvspan(movies_idx[j], movies_idx[j+1],color='#BABABA',alpha=0.2)
    
    plt.tight_layout()    
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)
    
def roi_id_composite(roi_list, rois_approved):
    new_roi_list = []
    for numi in roi_list:
        new_roi_list.append(numi[0])
    empty_list = []
    final_list = []
    for i in roi_list:
        if i < 10:
            empty_list.append([])
    for i in roi_list:
        first_num = int(str(i)[1])
#        #Exception Code
#        if first_num == 9:
#            first_num = 8
#        #End Exception Code
        empty_list[first_num-1].append(new_roi_list.index(i))
    for i in empty_list:
        new_list = []
        for num in i:
            if num in rois_approved:
                new_list.append(num)
        final_list.append(new_list)
    return final_list

def compare_rois(dF_signal, sess, movie_times, frame_times, roi_ids, quad_data, fs, subfolder, fname, baseline_percentile):
    movies_idx = []
    if movie_times != None:
        for i,mt in enumerate(movie_times):
            movies_idx.append((np.abs(frame_times - mt)).argmin())
    std_threshold = 6
    frame_latency = 1/fs
    rois_list = range(len(dF_signal[0]))
    rois_rejected = []
    rois_approved = []
    transients = []
    on_rois = []
    off_rois = []
    for roi in rois_list:
        rois_unfiltered = dF_signal[:,roi]
        min_transient_length_sec = 0.4
        min_transient_length = min_transient_length_sec/frame_latency
        percentile_low_idx = np.where(rois_unfiltered < np.percentile(rois_unfiltered,baseline_percentile))[0]
        rois_std = np.std(rois_unfiltered[percentile_low_idx])
        rois_mean = np.mean(rois_unfiltered[percentile_low_idx])
        transient_high = np.where(rois_unfiltered > (std_threshold*rois_std)+rois_mean)[0]
        if transient_high.size == 0:
            on_rois.append([])
            off_rois.append([])
            rois_rejected.append(roi)
        else:
            idx_diff = np.diff(transient_high)
            idx_diff = np.insert(idx_diff,0,transient_high[0])
            gap_tolerance_frames = int(0.2/frame_latency)    
            onset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]]-1
            offset_idx = transient_high[np.where(idx_diff > gap_tolerance_frames)[0]-1]+1
            offset_idx = np.roll(offset_idx, -1)
            index_adjust = 0
            for j in range(len(onset_idx)):
                temp_length = offset_idx[j-index_adjust] - onset_idx[j-index_adjust]
                if temp_length < min_transient_length:
                    onset_idx = np.delete(onset_idx,j-index_adjust)
                    offset_idx = np.delete(offset_idx,j-index_adjust)
                    index_adjust += 1
            above_1_std = np.where(rois_unfiltered > (2*rois_std)+rois_mean)[0]
            one_std_idx_diff = np.diff(above_1_std)
            one_std_idx_diff = np.insert(one_std_idx_diff,0,one_std_idx_diff[0])
            one_std_onset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]]-1
            one_std_offset_idx = above_1_std[np.where(one_std_idx_diff > 1)[0]-1]+1
            onsets_pre = []
            offsets_pre = []
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
                onsets_pre.append(oi-one_std_idx)
            for oi in offset_idx:
                closest_idx = one_std_offset_idx - oi
                closest_idx_neg = np.where(closest_idx > 0)[0]
                if closest_idx_neg.size == 0:
                    closest_idx_neg = [-1]
                    one_std_idx = 0
                else:
                    one_std_idx = np.min(np.abs(closest_idx[closest_idx_neg]))
                offsets_pre.append(oi+one_std_idx)
            for j in range(len(onsets_pre)):
                if j == 0 or (onsets_pre[j] != onsets_pre[j-1] and offsets_pre[j] != offsets_pre[j-1]):
                    onsets.append(onsets_pre[j])
                    offsets.append(offsets_pre[j])
            on_rois.append(onsets)
            off_rois.append(offsets)
#            parent_roi = int(str(roi_ids[roi])[1]) - 1
#            pearson_value = scistat.pearsonr(dF_signal[:,parent_roi],rois_unfiltered)[0]
            pearson_value = scistat.pearsonr(dF_signal[:,0],rois_unfiltered)[0]
            if pearson_value >= 0.10 and len(onsets) > 0:
                rois_approved.append(roi)
            else:
                rois_rejected.append(roi)
    
    for roi in rois_approved:
        rois_unfiltered = dF_signal[:,roi]
        cur_on_roi = on_rois[roi]
        cur_off_roi = off_rois[roi]
        for tran_0 in range(len(cur_on_roi)):
            count = 0
#            if quad_data.all() != None:
            if quad_data != None:
                if quad_data[cur_off_roi[tran_0]] - quad_data[cur_on_roi[tran_0]] > 30:
                    move = 'y'
                else:
                    move = 'n'
            else:
                move = 'n'
            ave_tran = (cur_on_roi[tran_0] + cur_off_roi[tran_0]) / 2
            in_movie = 'n'
            for time in range(int(len(movies_idx)/2)):
                if movies_idx[time*2] < ave_tran < movies_idx[time*2 + 1]:
                    in_movie = 'y'
            try:
                transients.append([roi, 'a', ave_tran, in_movie, move, np.max(rois_unfiltered[cur_on_roi[tran_0]:cur_off_roi[tran_0]])])
            except ValueError:
                pass
            for tran_1 in range(len(on_rois)):
                if roi != tran_1 and tran_1 in rois_approved:
                    comp_on_roi = on_rois[tran_1]
                    comp_off_roi = off_rois[tran_1]
                    for tran_2 in range(len(comp_on_roi)):
                        ave_comp_tran = (comp_on_roi[tran_2] + comp_off_roi[tran_2]) / 2
                        if (cur_on_roi[tran_0] < ave_comp_tran < cur_off_roi[tran_0]) or (comp_on_roi[tran_2] < ave_tran < comp_off_roi[tran_2]):
                            count += 1
                            break
            transients[-1].append(count / len(rois_approved))
        
    for roi in rois_rejected:
        rois_unfiltered = dF_signal[:,roi]
        cur_on_roi = on_rois[roi]
        cur_off_roi = off_rois[roi]
        for tran_0 in range(len(cur_on_roi)):
            count = 0
#            if quad_data.all() != None:
            if quad_data != None:
                if quad_data[cur_off_roi[tran_0]] - quad_data[cur_on_roi[tran_0]] > 30:
                    move = 'y'
                else:
                    move = 'n'
            else:
                move = 'n'
            ave_tran = (cur_on_roi[tran_0] + cur_off_roi[tran_0]) / 2
            in_movie = 'n'
            for time in range(int(len(movies_idx)/2)):
                if movies_idx[time*2] < ave_tran < movies_idx[time*2 + 1]:
                    in_movie = 'y'
            transients.append([roi, 'r', ave_tran, in_movie, move, np.max(rois_unfiltered[cur_on_roi[tran_0]:cur_off_roi[tran_0]])])
            for tran_1 in range(len(on_rois)):
                if roi != tran_1 and tran_1 in rois_rejected:
                    comp_on_roi = on_rois[tran_1]
                    comp_off_roi = off_rois[tran_1]
                    for tran_2 in range(len(comp_on_roi)):
                        ave_comp_tran = (comp_on_roi[tran_2] + comp_off_roi[tran_2]) / 2
                        if (cur_on_roi[tran_0] < ave_comp_tran < cur_off_roi[tran_0]) or (comp_on_roi[tran_2] < ave_tran < comp_off_roi[tran_2]):
                            count += 1
                            break
            transients[-1].append(count / len(rois_rejected))

    plot_list = roi_id_composite(roi_ids, rois_approved)
    for thing in plot_list:
        if len(thing) > 0:
            plot_trans(dF_signal, baseline_percentile, thing, roi_ids, movies_idx, on_rois, off_rois, transients, sess, subfolder, fname=fname + str(count))
    plot_trans(dF_signal, baseline_percentile, rois_approved, roi_ids, movies_idx, on_rois, off_rois, transients, sess, subfolder, fname=fname + str(1))

    if len(rois_rejected) != 0:
        plot_trans(dF_signal, baseline_percentile, rois_rejected, roi_ids, movies_idx, on_rois, off_rois, transients, sess, subfolder, fname=fname + 'reject')
#    
    return transients

def run_Buddha():
    MOUSE = 'Buddha'
    ttiile = 'Budda 190816 Epochs 31-40'
#    session = ['190816_1','190816_2','190816_3','190816_4','190816_5','190816_6','190816_7','190816_8','190816_9']
    session = ['190816_1','190816_2']
#    session = ['190816_11','190816_12','190816_13','190816_14','190816_15','190816_16','190816_17','190816_18','190816_19','190816_20']
#    session = ['190816_21','190816_22','190816_23','190816_24','190816_25','190816_26','190816_27','190816_28','190816_29','190816_30']
#    session = ['190816_31','190816_32','190816_33','190816_34','190816_35','190816_36','190816_37','190816_38','190816_39','190816_40']
#    session = ['190816_41','190816_42','190816_43','190816_44','190816_45','190816_46','190816_47','190816_48','190816_49','190816_50']
    for sess in session:
        if sess[-1] != '0':
            sessvar = '0' + sess[-1]
        else:
            sessvar = '10'
#        processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'mask1' + os.sep + 'sig_data.mat'
#        processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data' + sessvar + '.mat'
        processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'sig_data.mat'
        loaded_data = sio.loadmat(processed_data_path)
        dF_signal = loaded_data['dF_data']
        if sess[-2] == '_':
            construct = '_000_00' + sess[-1]
        else:
            construct = '_000_0' + sess[-2] + sess[-1]
        movietimes_file_path = loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + MOUSE + construct + ' movie times.txt'
        movie_types, movie_times = read_movie_times(movietimes_file_path)
        metadata = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + MOUSE + construct + '.mat')
        frame_times = metadata['info'][0][0][25][0]
#        extra = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + 'mask1' + os.sep + MOUSE + construct + '_rigid.extra')
        extra = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + MOUSE + construct + '_rigid.extra')
        roi_ids = extra['roiIDs']
        quad_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + sess + os.sep + MOUSE + construct + '_quadrature.mat')
        quad_data = quad_data['quad_data'][0]
        if sess == session[0]:
            transient_list = [[] for elem in range(len(dF_signal[0]))]
        pos_trans_data = compare_rois(dF_signal, sess, movie_times, frame_times, roi_ids, quad_data, fs=15.5, subfolder='Buddha transients', fname='compare', baseline_percentile=60)
        for tran_num in pos_trans_data:
            transient_list[tran_num[0]].append(tran_num)
            
    tran_value_m_y = np.array([])
    tran_value_o_y = np.array([])
    tran_value_m_n = np.array([])
    tran_value_o_n = np.array([])
#    for tran_it in transient_list:
#        count = 0
#        average_df = 0
#        average_per = 0
#        for tran in tran_it:
#            if tran[1] == 'a':
#                average_df += tran[5]
#                average_per += tran[6]
#                count += 1
#        if len(tran_it) != 0 and count != 0:
#            tran_value.append([average_df/count,average_per/count])
##        if len(tran_it) != 0:
##            tran_value.append([average_df/len(tran_it),average_per/len(tran_it)])
    for tran_great in transient_list:
        if tran_great != []:
            count_a = 0
            count_ind_m_y = 0
            count_ind_m_n = 0
            count_ind_o_y = 0
            count_ind_o_n = 0
            for tran_lit in tran_great:
                if tran_lit[1] == 'a':
                    count_a += 1
                    if tran_lit[-1] < 0.66 and tran_lit[3] == 'y':
                        if tran_lit[4] == 'y':
                            count_ind_m_y += 1
                        else:
                            count_ind_m_n += 1
                    elif tran_lit[-1] < 0.66 and tran_lit[3] == 'n':
                        if tran_lit[4] == 'y':
                            count_ind_o_y += 1
                        else:
                            count_ind_o_n += 1
            if count_a != 0:
                tran_value_m_y = np.append(tran_value_m_y, count_ind_m_y/count_a)
                tran_value_o_y = np.append(tran_value_o_y, count_ind_o_y/count_a)
                tran_value_m_n = np.append(tran_value_m_n, count_ind_m_n/count_a)
                tran_value_o_n = np.append(tran_value_o_n, count_ind_o_n/count_a)
    tran_val_myav = np.average(tran_value_m_y)
    tran_val_oyav = np.average(tran_value_o_y)
    tran_val_mnav = np.average(tran_value_m_n)
    tran_val_onav = np.average(tran_value_o_n)
    subfolder='Buddha transients'
    fname='compare_all'
    fig = plt.figure(figsize=(15,15))
    ax1 = plt.subplot2grid((100,100),(0,0), rowspan=100, colspan=100)
    ax1.bar(np.arange(4), [tran_val_myav,tran_val_oyav,tran_val_mnav,tran_val_onav], yerr=[np.std(tran_value_m_y),np.std(tran_value_o_y),np.std(tran_value_m_n),np.std(tran_value_o_n)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.set_ylabel('% of Independent Transients out of All Transients')
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(['During Movie and Moving','During Grey Screen and Moving','During Movie, Not Moving','During Grey Screen, Not Moving'])
    ax1.set_title(ttiile)
#    for data in tran_value_m:
#        ax1.plot(data[0],data[1],color='red',marker='o')
#    for data in tran_value_o:
#        ax1.plot(data[0],data[1],color='blue',marker='o')
#    line_value = []
#    average_val = []
#    for data in range(len(tran_value_m)):
#        line_value.append(tran_value_m[data][1] - tran_value_o[data][1])
#        if tran_value_o[data][1] != 0:
#            average_val.append(tran_value_m[data][1] / tran_value_o[data][1])
#        else:
#            average_val.append(tran_value_m[data][1] / 1)
#    ax1.plot(line_value, color='blue')
#    ax1.plot(average_val, color='green')
#    print(sum(line_value)/len(line_value))
#    print(sum(average_val)/len(line_value))
#    ax1.plot([0 for elem in range(len(transient_list))], color='red')
#    plt.tight_layout()    
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)
    
def run_Isabel():
    MOUSE = 'Isabel'
    folder = '190920_0_19'
#    folder = '190920_20_39'
    session = ['Isabel_000_000','Isabel_000_001','Isabel_000_002','Isabel_000_003','Isabel_000_004','Isabel_000_005','Isabel_000_006',
               'Isabel_000_007','Isabel_000_008','Isabel_000_009','Isabel_000_010','Isabel_000_011','Isabel_000_012','Isabel_000_013',
               'Isabel_000_014','Isabel_000_015','Isabel_000_016','Isabel_000_017','Isabel_000_018','Isabel_000_019']
#    session = ['Isabel_000_020','Isabel_000_021','Isabel_000_022','Isabel_000_023','Isabel_000_024','Isabel_000_025','Isabel_000_026',
#               'Isabel_000_027','Isabel_000_028','Isabel_000_029','Isabel_000_030','Isabel_000_031','Isabel_000_032','Isabel_000_033',
#               'Isabel_000_034','Isabel_000_035','Isabel_000_036','Isabel_000_037','Isabel_000_038','Isabel_000_039']
    for sess in session:
        processed_data_path = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + folder + os.sep + 'sig_data_' + sess[11:] + '.mat'
        loaded_data = sio.loadmat(processed_data_path)
        dF_signal = loaded_data['dF_data']
#        movietimes_file_path = loc_info['raw_dir'] + MOUSE + os.sep + folder + os.sep + sess + ' movie times.txt'
#        movie_types, movie_times = read_movie_times(movietimes_file_path)
        movie_times = None
        metadata = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder + os.sep + sess + '.mat')
        frame_times = metadata['info'][0][0][25][0]
        extra = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder + os.sep + sess + '_rigid.extra')
        roi_ids = extra['roiIDs']
        quad_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder + os.sep + sess + '_quadrature.mat')
        quad_data = quad_data['quad_data'][0]
        if sess == session[0]:
            transient_list = [[] for elem in range(len(dF_signal[0]))]
        pos_trans_data = compare_rois(dF_signal, sess, movie_times, frame_times, roi_ids, quad_data, fs=15.5, subfolder='Isabel transients', fname='compare', baseline_percentile=60)    
        for tran_num in pos_trans_data:
            transient_list[tran_num[0]].append(tran_num)
            
    tran_value_m_y = np.array([])
    tran_value_o_y = np.array([])
    tran_value_m_n = np.array([])
    tran_value_o_n = np.array([])
    for tran_great in transient_list:
        if tran_great != []:
            count_a = 0
            count_ind_m_y = 0
            count_ind_m_n = 0
            count_ind_o_y = 0
            count_ind_o_n = 0
            for tran_lit in tran_great:
                if tran_lit[1] == 'a':
                    count_a += 1
                    if tran_lit[-1] < 0.66 and tran_lit[3] == 'y':
                        if tran_lit[4] == 'y':
                            count_ind_m_y += 1
                        else:
                            count_ind_m_n += 1
                    elif tran_lit[-1] < 0.66 and tran_lit[3] == 'n':
                        if tran_lit[4] == 'y':
                            count_ind_o_y += 1
                        else:
                            count_ind_o_n += 1
            if count_a != 0:
                tran_value_m_y = np.append(tran_value_m_y, count_ind_m_y/count_a)
                tran_value_o_y = np.append(tran_value_o_y, count_ind_o_y/count_a)
                tran_value_m_n = np.append(tran_value_m_n, count_ind_m_n/count_a)
                tran_value_o_n = np.append(tran_value_o_n, count_ind_o_n/count_a)
    tran_val_myav = np.average(tran_value_m_y)
    tran_val_oyav = np.average(tran_value_o_y)
    tran_val_mnav = np.average(tran_value_m_n)
    tran_val_onav = np.average(tran_value_o_n)
    subfolder='Isabel transients'
    fname='compare_all'
    fig = plt.figure(figsize=(15,15))
    ax1 = plt.subplot2grid((100,100),(0,0), rowspan=100, colspan=100)
#    ax1.bar(np.arange(4), [tran_val_myav,tran_val_oyav,tran_val_mnav,tran_val_onav], yerr=[np.std(tran_value_m_y),np.std(tran_value_o_y),np.std(tran_value_m_n),np.std(tran_value_o_n)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.bar(np.arange(2), [tran_val_oyav,tran_val_onav], yerr=[np.std(tran_value_o_y),np.std(tran_value_o_n)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.set_ylabel('% of Independent Transients out of All Transients')
#    ax1.set_xticks(np.arange(4))
    ax1.set_xticks(np.arange(2))
#    ax1.set_xticklabels(['During Movie and Moving','During Grey Screen and Moving','During Movie, Not Moving','During Grey Screen, Not Moving'])
    ax1.set_xticklabels(['Running','Not Running'])
    ax1.set_title('Isabel, 9/20, 1st Recording')
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)


def run_Rambo():
    MOUSE = 'Rambo'
    folder1 = '190925_0_19_0000'
    folder2 = '190925_0_19_0087'
    session1 = ['Rambo_000_000_0000','Rambo_000_001_0000','Rambo_000_002_0000','Rambo_000_003_0000','Rambo_000_004_0000','Rambo_000_005_0000','Rambo_000_006_0000',
               'Rambo_000_007_0000','Rambo_000_008_0000','Rambo_000_009_0000','Rambo_000_010_0000','Rambo_000_011_0000','Rambo_000_012_0000','Rambo_000_013_0000',
               'Rambo_000_014_0000','Rambo_000_015_0000','Rambo_000_016_0000','Rambo_000_017_0000','Rambo_000_018_0000','Rambo_000_019_0000']
    session2 = ['Rambo_000_000_0087','Rambo_000_001_0087','Rambo_000_002_0087','Rambo_000_003_0087','Rambo_000_004_0087','Rambo_000_005_0087','Rambo_000_006_0087',
               'Rambo_000_007_0087','Rambo_000_008_0087','Rambo_000_009_0087','Rambo_000_010_0087','Rambo_000_011_0087','Rambo_000_012_0087','Rambo_000_013_0087',
               'Rambo_000_014_0087','Rambo_000_015_0087','Rambo_000_016_0087','Rambo_000_017_0087','Rambo_000_018_0087','Rambo_000_019_0087']
#    folder1 = '190925_21_40_0000'
#    folder2 = '190925_21_40_0103'
#    session1 = ['Rambo_000_021_0000','Rambo_000_022_0000','Rambo_000_023_0000','Rambo_000_024_0000','Rambo_000_025_0000','Rambo_000_026_0000','Rambo_000_027_0000',
#               'Rambo_000_028_0000','Rambo_000_029_0000','Rambo_000_030_0000','Rambo_000_031_0000','Rambo_000_032_0000','Rambo_000_033_0000','Rambo_000_034_0000',
#               'Rambo_000_035_0000','Rambo_000_036_0000','Rambo_000_037_0000','Rambo_000_038_0000','Rambo_000_039_0000','Rambo_000_040_0000']
#    session2 = ['Rambo_000_021_0103','Rambo_000_022_0103','Rambo_000_023_0103','Rambo_000_024_0103','Rambo_000_025_0103','Rambo_000_026_0103','Rambo_000_027_0103',
#               'Rambo_000_028_0103','Rambo_000_029_0103','Rambo_000_030_0103','Rambo_000_031_0103','Rambo_000_032_0103','Rambo_000_033_0103','Rambo_000_034_0103',
#               'Rambo_000_035_0103','Rambo_000_036_0103','Rambo_000_037_0103','Rambo_000_038_0103','Rambo_000_039_0103','Rambo_000_040_0103']
#    folder1 = '190925_41_80_0000'
#    folder2 = '190925_41_80_0132'
#    session1 = ['Rambo_000_041_0000','Rambo_000_042_0000','Rambo_000_043_0000','Rambo_000_044_0000','Rambo_000_045_0000','Rambo_000_046_0000','Rambo_000_047_0000',
#               'Rambo_000_048_0000','Rambo_000_049_0000','Rambo_000_050_0000','Rambo_000_051_0000','Rambo_000_052_0000','Rambo_000_053_0000','Rambo_000_054_0000',
#               'Rambo_000_055_0000','Rambo_000_056_0000','Rambo_000_057_0000','Rambo_000_058_0000','Rambo_000_059_0000','Rambo_000_060_0000',
#               'Rambo_000_061_0000','Rambo_000_062_0000','Rambo_000_063_0000','Rambo_000_064_0000','Rambo_000_065_0000','Rambo_000_066_0000','Rambo_000_067_0000',
#               'Rambo_000_068_0000','Rambo_000_069_0000','Rambo_000_070_0000','Rambo_000_071_0000','Rambo_000_072_0000','Rambo_000_073_0000','Rambo_000_074_0000',
#               'Rambo_000_075_0000','Rambo_000_076_0000','Rambo_000_077_0000','Rambo_000_078_0000','Rambo_000_079_0000','Rambo_000_080_0000']
#    session2 = ['Rambo_000_041_0132','Rambo_000_042_0132','Rambo_000_043_0132','Rambo_000_044_0132','Rambo_000_045_0132','Rambo_000_046_0132','Rambo_000_047_0132',
#               'Rambo_000_048_0132','Rambo_000_049_0132','Rambo_000_050_0132','Rambo_000_051_0132','Rambo_000_052_0132','Rambo_000_053_0132','Rambo_000_054_0132',
#               'Rambo_000_055_0132','Rambo_000_056_0132','Rambo_000_057_0132','Rambo_000_058_0132','Rambo_000_059_0132','Rambo_000_060_0132',
#               'Rambo_000_061_0132','Rambo_000_062_0132','Rambo_000_063_0132','Rambo_000_064_0132','Rambo_000_065_0132','Rambo_000_066_0132','Rambo_000_067_0132',
#               'Rambo_000_068_0132','Rambo_000_069_0132','Rambo_000_070_0132','Rambo_000_071_0132','Rambo_000_072_0132','Rambo_000_073_0132','Rambo_000_074_0132',
#               'Rambo_000_075_0132','Rambo_000_076_0132','Rambo_000_077_0132','Rambo_000_078_0132','Rambo_000_079_0132','Rambo_000_080_0132']
    for sessn in range(len(session1)):
        sess = session1[sessn] + ' :: ' + session2[sessn]
        processed_data_path1 = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + folder1 + os.sep + 'sig_data_' + session1[sessn][10:13] + '.mat'
        processed_data_path2 = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + folder1 + os.sep + 'sig_data_' + session2[sessn][10:13] + '.mat'
        loaded_data1 = sio.loadmat(processed_data_path1)
        dF_signal1 = loaded_data1['dF_data']
        loaded_data2 = sio.loadmat(processed_data_path2)
        dF_signal2 = loaded_data2['dF_data']
        dF_signal = []
        for x in range(len(dF_signal1)):
            place_holder = [n for n in dF_signal1[x]]
            for y in dF_signal2[x]:
                place_holder.append(y)
            dF_signal.append(place_holder)
        dF_signal = np.array(dF_signal)
        movietimes_file_path = loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + session1[sessn][:13] + ' movie times.txt'
        movie_types, movie_times = read_movie_times(movietimes_file_path)
        metadata = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + session1[sessn] + '.mat')
        frame_times = metadata['info'][0][0][25][0]
        extra1 = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + session1[sessn] + '_rigid.extra')
        extra2 = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder2 + os.sep + session2[sessn] + '_rigid.extra')
        roi_ids1 = extra1['roiIDs']
        roi_ids2 = extra2['roiIDs']
        roi_ids = []
        for x in roi_ids1:
            roi_ids.append(x)
        for x in roi_ids2:
            roi_ids.append(x)
#        quad_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder + os.sep + sess + '_quadrature.mat')
#        quad_data = quad_data['quad_data'][0]
        quad_data = None
        if sessn == 0:
            transient_list = [[] for elem in range(len(dF_signal[0]))]
        pos_trans_data = compare_rois(dF_signal, sess, movie_times, frame_times, roi_ids, quad_data, fs=15.5, subfolder='Isabel transients', fname='compare', baseline_percentile=60)    
        for tran_num in pos_trans_data:
            transient_list[tran_num[0]].append(tran_num)
            
    tran_value_m_y = np.array([])
    tran_value_o_y = np.array([])
    tran_value_m_n = np.array([])
    tran_value_o_n = np.array([])
    for tran_great in transient_list:
        if tran_great != []:
            count_a = 0
            count_ind_m_y = 0
            count_ind_m_n = 0
            count_ind_o_y = 0
            count_ind_o_n = 0
            for tran_lit in tran_great:
                if tran_lit[1] == 'a':
                    count_a += 1
                    if tran_lit[-1] < 0.66 and tran_lit[3] == 'y':
                        if tran_lit[4] == 'y':
                            count_ind_m_y += 1
                        else:
                            count_ind_m_n += 1
                    elif tran_lit[-1] < 0.66 and tran_lit[3] == 'n':
                        if tran_lit[4] == 'y':
                            count_ind_o_y += 1
                        else:
                            count_ind_o_n += 1
            if count_a != 0:
                tran_value_m_y = np.append(tran_value_m_y, count_ind_m_y/count_a)
                tran_value_o_y = np.append(tran_value_o_y, count_ind_o_y/count_a)
                tran_value_m_n = np.append(tran_value_m_n, count_ind_m_n/count_a)
                tran_value_o_n = np.append(tran_value_o_n, count_ind_o_n/count_a)
    tran_val_myav = np.average(tran_value_m_y)
    tran_val_oyav = np.average(tran_value_o_y)
    tran_val_mnav = np.average(tran_value_m_n)
    tran_val_onav = np.average(tran_value_o_n)
    subfolder='Rambo transients'
    fname='compare_all'
    fig = plt.figure(figsize=(15,15))
    ax1 = plt.subplot2grid((100,100),(0,0), rowspan=100, colspan=100)
#    ax1.bar(np.arange(4), [tran_val_myav,tran_val_oyav,tran_val_mnav,tran_val_onav], yerr=[np.std(tran_value_m_y),np.std(tran_value_o_y),np.std(tran_value_m_n),np.std(tran_value_o_n)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.bar(np.arange(2), [tran_val_mnav,tran_val_onav], yerr=[np.std(tran_value_m_n),np.std(tran_value_o_n)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.set_ylabel('% of Independent Transients out of All Transients')
#    ax1.set_xticks(np.arange(4))
    ax1.set_xticks(np.arange(2))
#    ax1.set_xticklabels(['During Movie and Moving','During Grey Screen and Moving','During Movie, Not Moving','During Grey Screen, Not Moving'])
    ax1.set_xticklabels(['During Movie','During Grey Screen'])
    ax1.set_title('Rambo, 9/25, 1st Recording, 0-87 um')
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)
    
def run_Isabel_prime():
    MOUSE = 'Isabel'
    folder1 = '190926_0_39_0000'
    folder2 = '190926_0_39_0068'
#    session1 = ['Isabel_000_012_0000']
#    session2 = ['Isabel_000_012_0068']
    session1 = ['Isabel_000_000_0000','Isabel_000_001_0000','Isabel_000_002_0000','Isabel_000_003_0000','Isabel_000_004_0000','Isabel_000_005_0000','Isabel_000_006_0000',
               'Isabel_000_007_0000','Isabel_000_008_0000','Isabel_000_009_0000','Isabel_000_010_0000','Isabel_000_011_0000','Isabel_000_012_0000','Isabel_000_013_0000',
               'Isabel_000_014_0000','Isabel_000_015_0000','Isabel_000_016_0000','Isabel_000_017_0000','Isabel_000_018_0000','Isabel_000_019_0000','Isabel_000_021_0000',
               'Isabel_000_022_0000','Isabel_000_023_0000','Isabel_000_024_0000','Isabel_000_025_0000','Isabel_000_026_0000','Isabel_000_027_0000','Isabel_000_028_0000',
               'Isabel_000_029_0000','Isabel_000_030_0000','Isabel_000_031_0000','Isabel_000_032_0000','Isabel_000_033_0000','Isabel_000_034_0000','Isabel_000_035_0000',
               'Isabel_000_036_0000','Isabel_000_037_0000','Isabel_000_038_0000','Isabel_000_039_0000']
    session2 = ['Isabel_000_000_0068','Isabel_000_001_0068','Isabel_000_002_0068','Isabel_000_003_0068','Isabel_000_004_0068','Isabel_000_005_0068','Isabel_000_006_0068',
               'Isabel_000_007_0068','Isabel_000_008_0068','Isabel_000_009_0068','Isabel_000_010_0068','Isabel_000_011_0068','Isabel_000_012_0068','Isabel_000_013_0068',
               'Isabel_000_014_0068','Isabel_000_015_0068','Isabel_000_016_0068','Isabel_000_017_0068','Isabel_000_018_0068','Isabel_000_019_0068','Isabel_000_021_0068',
               'Isabel_000_022_0068','Isabel_000_023_0068','Isabel_000_024_0068','Isabel_000_025_0068','Isabel_000_026_0068','Isabel_000_027_0068','Isabel_000_028_0068',
               'Isabel_000_029_0068','Isabel_000_030_0068','Isabel_000_031_0068','Isabel_000_032_0068','Isabel_000_033_0068','Isabel_000_034_0068','Isabel_000_035_0068',
               'Isabel_000_036_0068','Isabel_000_037_0068','Isabel_000_038_0068','Isabel_000_039_0068']

    for sessn in range(len(session1)):
        sess = session1[sessn] + ' :: ' + session2[sessn]
        processed_data_path1 = loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + 'sig_data_' + session1[sessn][11:14] + '.mat'
        processed_data_path2 = loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + 'sig_data_' + session2[sessn][11:14] + '.mat'
        loaded_data1 = sio.loadmat(processed_data_path1)
        dF_signal1 = loaded_data1['dF_data']
        loaded_data2 = sio.loadmat(processed_data_path2)
        dF_signal2 = loaded_data2['dF_data']
        dF_signal = []
        for x in range(len(dF_signal1)):
            place_holder = [n for n in dF_signal1[x]]
            for y in dF_signal2[x]:
                place_holder.append(y)
            dF_signal.append(place_holder)
        dF_signal = np.array(dF_signal)
        movietimes_file_path = loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + session1[sessn][:14] + ' movie times.txt'
        movie_types, movie_times = read_movie_times(movietimes_file_path)
        metadata = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + session1[sessn][:14] + '.mat')
        frame_times = metadata['info'][0][0][22][0]
        extra1 = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + session1[sessn] + '_rigid.extra')
        extra2 = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder2 + os.sep + session2[sessn] + '_rigid.extra')
        roi_ids1 = extra1['roiIDs']
        roi_ids2 = extra2['roiIDs']
        roi_ids = []
        for x in roi_ids1:
            roi_ids.append(x)
        for x in roi_ids2:
            roi_ids.append(x)
        quad_data = sio.loadmat(loc_info['raw_dir'] + MOUSE + os.sep + folder1 + os.sep + session1[sessn][:14] + '_quadrature.mat')
        quad_data = quad_data['quad_data'][0]
        if sessn == 0:
            transient_list = [[] for elem in range(len(dF_signal[0]))]
        pos_trans_data = compare_rois(dF_signal, sess, movie_times, frame_times, roi_ids, quad_data, fs=15.5, subfolder='Isabel transients', fname='compare', baseline_percentile=60)    
        for tran_num in pos_trans_data:
            transient_list[tran_num[0]].append(tran_num)
            
    tran_value_m_y = np.array([])
    tran_value_o_y = np.array([])
    tran_value_m_n = np.array([])
    tran_value_o_n = np.array([])
    for tran_great in transient_list:
        if tran_great != []:
            count_a = 0
            count_ind_m_y = 0
            count_ind_m_n = 0
            count_ind_o_y = 0
            count_ind_o_n = 0
            for tran_lit in tran_great:
                if tran_lit[1] == 'a':
                    count_a += 1
                    if tran_lit[-1] < 0.66 and tran_lit[3] == 'y':
                        if tran_lit[4] == 'y':
                            count_ind_m_y += 1
                        else:
                            count_ind_m_n += 1
                    elif tran_lit[-1] < 0.66 and tran_lit[3] == 'n':
                        if tran_lit[4] == 'y':
                            count_ind_o_y += 1
                        else:
                            count_ind_o_n += 1
            if count_a != 0:
                tran_value_m_y = np.append(tran_value_m_y, count_ind_m_y/count_a)
                tran_value_o_y = np.append(tran_value_o_y, count_ind_o_y/count_a)
                tran_value_m_n = np.append(tran_value_m_n, count_ind_m_n/count_a)
                tran_value_o_n = np.append(tran_value_o_n, count_ind_o_n/count_a)
    tran_val_myav = np.average(tran_value_m_y)
    tran_val_oyav = np.average(tran_value_o_y)
    tran_val_mnav = np.average(tran_value_m_n)
    tran_val_onav = np.average(tran_value_o_n)
    subfolder='Isabel transients'
    fname='compare_all'
    fig = plt.figure(figsize=(15,15))
    ax1 = plt.subplot2grid((100,100),(0,0), rowspan=100, colspan=100)
    ax1.bar(np.arange(4), [tran_val_myav,tran_val_oyav,tran_val_mnav,tran_val_onav], yerr=[np.std(tran_value_m_y),np.std(tran_value_o_y),np.std(tran_value_m_n),np.std(tran_value_o_n)], align='center', alpha=0.5, ecolor='black', capsize=10)
#    ax1.bar(np.arange(2), [tran_val_mnav,tran_val_onav], yerr=[np.std(tran_value_m_n),np.std(tran_value_o_n)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.set_ylabel('% of Independent Transients out of All Transients')
    ax1.set_xticks(np.arange(4))
#    ax1.set_xticks(np.arange(2))
    ax1.set_xticklabels(['During Movie and Moving','During Grey Screen and Moving','During Movie, Not Moving','During Grey Screen, Not Moving'])
#    ax1.set_xticklabels(['During Movie','During Grey Screen'])
    ax1.set_title('Isabel, 9/26, 1st Recording, 0-68 um')
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)
    
if __name__ == '__main__':
    fformat = 'svg'
#    run_Buddha()
#    run_Isabel()
    run_Rambo()
#    run_Isabel_prime()