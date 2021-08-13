"""
Calculate ROI cross-correlation for bouton data.

"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import yaml, os, sys
sys.path.append("..\\General\\Imaging")
from dF_win_mpi import dF_win
import scipy.io as sio
from scipy import stats
import warnings; warnings.simplefilter('ignore')
from scipy.signal import butter, filtfilt
from sklearn.cluster.bicluster import SpectralCoclustering
import ipdb
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

fformat = 'svg'

with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

def traceplot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_roi_group(dF_signal_sess, dF_signal_ol, roi_idx, subfolder):
    print(roi_idx)
    fig = plt.figure(figsize=(18,10), dpi=300)
    ax = []
    for i in range(len(roi_idx)*2):
        ax.append(plt.subplot(len(roi_idx),2,i+1))

    max_y = 0
    for i,ri in enumerate(roi_idx):
        ax[i*2].plot(dF_signal_sess[:,ri], c='k')
        ax[i*2].set_xlim([0,dF_signal_sess[:,ri].shape[0]])
        ax[i*2].set_ylabel(str(ri))
        if np.nanmax(dF_signal_sess[:,ri]) > max_y:
            max_y = np.nanmax(dF_signal_sess[:,ri])

        ax[(i*2)+1].plot(dF_signal_ol[:,ri], c='r')
        ax[(i*2)+1].set_xlim([0,dF_signal_sess[:,ri].shape[0]])
        ax[(i*2)+1].set_ylabel('')
        ax[(i*2)+1].set_yticks([0])

        if np.nanmax(dF_signal_ol[:,ri]) > max_y:
            max_y = np.nanmax(dF_signal_sess[:,ri])

        ax[i*2].axhline(0,c='b')
        ax[(i*2)+1].axhline(0,c='b')

        ax[i*2].spines['top'].set_visible(False)
        ax[i*2].spines['right'].set_visible(False)
        ax[i*2].spines['bottom'].set_visible(False)
        ax[i*2].tick_params( \
            axis='both', \
            direction='out', \
            labelsize=8, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')

        ax[(i*2)+1].spines['top'].set_visible(False)
        ax[(i*2)+1].spines['right'].set_visible(False)
        ax[(i*2)+1].spines['bottom'].set_visible(False)
        ax[(i*2)+1].tick_params( \
            axis='both', \
            direction='out', \
            labelsize=8, \
            length=4, \
            width=2, \
            left='on', \
            bottom='on', \
            right='off', \
            top='off')

    # set max value for y-axes
    for i in range(len(roi_idx)*2):
        ax[i].set_ylim([-1,max_y])

    subfolder = 'bouton_cc' + os.sep + subfolder
    fname = 'cc_group_' + str(roi_idx[0])

    # ipdb.set_trace()

    if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
        os.mkdir(loc_info['figure_output_path'] + subfolder)
    fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat

    fig.savefig(fname, format=fformat,dpi=150)
    plt.close(fig)


def make_cc_plot(ROI_gcamp_cc, PIL_gcamp_cc, ROI_gcamp_sess, PIL_gcamp_sess, ROI_gcamp_ol, PIL_gcamp_ol, extra_fname_cc, extra_fname_sess, extra_fname_ol, rois_from_axon,cc_thresh, fname, subfolder):

    rec_info_cc = sio.loadmat( extra_fname_cc )
    frame_brightness_cc = rec_info_cc['meanBrightness']

    rec_info_sess = sio.loadmat( extra_fname_sess )
    frame_brightness_sess = rec_info_sess['meanBrightness']

    rec_info_ol = sio.loadmat( extra_fname_ol )
    frame_brightness_ol = rec_info_ol['meanBrightness']

    # plot some representative
    fig = plt.figure(figsize=(18,20), dpi=300)
    ax1 = plt.subplot2grid((100,100),(0,0), rowspan=10, colspan=100)
    ax2 = plt.subplot2grid((100,100),(10,0), rowspan=10, colspan=100)
    ax3 = plt.subplot2grid((100,100),(20,0), rowspan=10, colspan=100)
    ax7 = plt.subplot2grid((100,100),(35,0), rowspan=30, colspan=30)
    ax8 = plt.subplot2grid((100,100),(35,55), rowspan=30, colspan=30)
    ax9 = plt.subplot2grid((100,100),(75,0), rowspan=30, colspan=30)
    ax10 = plt.subplot2grid((100,100),(75,55), rowspan=30, colspan=30)

    ax1 = traceplot(ax1)
    ax2 = traceplot(ax2)
    ax3 = traceplot(ax3)

    print('calculating dF/F for session')
    mean_frame_brightness_sess = np.mean(frame_brightness_sess[0])
    dF_signal_sess, f0_sig_sess = dF_win((ROI_gcamp_sess-PIL_gcamp_sess)+mean_frame_brightness_sess)

    print('calculating dF/F for openloop')
    mean_frame_brightness_ol = np.mean(frame_brightness_ol[0])
    dF_signal_ol, f0_sig_ol = dF_win((ROI_gcamp_ol-PIL_gcamp_ol)+mean_frame_brightness_ol)

    print('calculating dF/F for cc')
    mean_frame_brightness = np.mean(frame_brightness_cc[0])
    dF_signal, f0_sig = dF_win((ROI_gcamp_cc-PIL_gcamp_cc)+mean_frame_brightness)

    # with plt.style.context(('dark_background')):
    #     ax1.plot(dF_signal_sess[:,8])
    #     ax2.plot(dF_signal_sess[:,9])
    #     ax3.plot(dF_signal_sess[:,10])

    # empty matrix holding correlation coefficients
    roi_CC_matrix = np.zeros((dF_signal.shape[1],dF_signal.shape[1]))
    roi_CC_matrix_sess = np.zeros((dF_signal_sess.shape[1],dF_signal_sess.shape[1]))

    # filter parameters
    order = 6
    fs = 15.5       # sample rate, Hz
    cutoff = 1 # desired cutoff frequency of the filter, Hz

    # create cc matrix for session data
    for j,roi in enumerate(range(roi_CC_matrix_sess.shape[0])):
        df_roi = butter_lowpass_filter(dF_signal_sess[:,roi], cutoff, fs, order)
        for k,roi_corr in enumerate(range(roi_CC_matrix_sess.shape[0])):
            dF_roi_corr = butter_lowpass_filter(dF_signal_sess[:,roi_corr], cutoff, fs, order)
            roi_CC_matrix_sess[j,k], _ = stats.pearsonr(df_roi, dF_roi_corr)

    # collect rois with high cc values and store in separate lists
    consumed_idx = []
    grouped_rois = []
    # run through every row and grab rois above cc_thresh
    for j in range(roi_CC_matrix_sess.shape[0]):
        corr_rois_idx = np.where(roi_CC_matrix_sess[j,:] > cc_thresh)[0]
        # if there are correlated ROIs, check if those rois have in turn other ROI that they are correlated to
        if len(corr_rois_idx) > 1:
            for k in corr_rois_idx:
                sub_corr_rois_idx = np.where(roi_CC_matrix_sess[k,:] > cc_thresh)[0]
                corr_rois_idx = np.union1d(corr_rois_idx, sub_corr_rois_idx)
        # print(corr_rois_idx)
        new_roi_idx = np.setdiff1d(corr_rois_idx, consumed_idx)
        consumed_idx = np.union1d(consumed_idx,new_roi_idx)
        if len(new_roi_idx) > 0:
            grouped_rois.append(new_roi_idx)

            # print(new_roi_idx)

    grouped_rois_flattened = np.array([item for sublist in grouped_rois for item in sublist])

    # model = SpectralCoclustering(n_clusters=15)
    # model.fit(roi_CC_matrix_sess)
    # fit_data = roi_CC_matrix_sess[np.argsort(model.row_labels_)]
    # fit_data = fit_data[:, np.argsort(model.column_labels_)]
    # ax9.pcolormesh(fit_data, cmap='viridis')

    # print(roi_CC_matrix_sess.shape, grouped_rois_flattened.shape)
    ordered_cc_matrix = np.copy(roi_CC_matrix_sess[grouped_rois_flattened,:])
    ordered_cc_matrix = ordered_cc_matrix[:,grouped_rois_flattened]
    ax9.pcolormesh(ordered_cc_matrix, cmap='viridis')
    # ax9.pcolormesh(np.copy(roi_CC_matrix_sess), cmap='viridis')
    ax9.set_xlim([0,dF_signal_sess.shape[1]])
    ax9.set_ylim([0,dF_signal_sess.shape[1]])

    # remove autocorrelation values
    for j in range(roi_CC_matrix_sess.shape[0]):
        roi_CC_matrix_sess[j,j] = np.nan

    # sns.distplot(roi_CC_matrix_sess.flatten()[roi_CC_matrix_sess.flatten()<0.5],bins=np.arange(0,1.1,0.1),
    #              kde=True,hist=False,color='b',ax=ax10,kde_kws={"color": "#460B5E","alpha":1.0, "lw":2},hist_kws={"edgecolor":"w","alpha":1.0})

    sns.kdeplot(roi_CC_matrix_sess.flatten()[roi_CC_matrix_sess.flatten()<0.5],color='#453882',gridsize=100,bw=.05,shade=True,ax=ax10,**{"linewidth":2})
    sns.kdeplot(roi_CC_matrix_sess.flatten()[roi_CC_matrix_sess.flatten()>=0.5],color='#FDE725',gridsize=100,bw=.05,shade=True,ax=ax10,**{"linewidth":2})
    # ax10_1 = ax10.twinx()
                # kde=True,hist=False,color='y',kde_kws={"color": "#FFFF00","alpha":1.0, "lw":2},ax=ax10,hist_kws={"edgecolor":"w","alpha":1.0})
                # sns.distplot(roi_CC_matrix_sess.flatten()[roi_CC_matrix_sess.flatten()>=0.5],bins=np.arange(0,1.1,0.1),

    # create cc matrix for cc-data
    for j,roi in enumerate(range(roi_CC_matrix.shape[0])):
        df_roi = butter_lowpass_filter(dF_signal[:,roi], cutoff, fs, order)
        for k,roi_corr in enumerate(range(roi_CC_matrix.shape[0])):
            dF_roi_corr = butter_lowpass_filter(dF_signal[:,roi_corr], cutoff, fs, order)
            roi_CC_matrix[j,k],_ = stats.pearsonr(df_roi, dF_roi_corr)

    ax7.pcolormesh(roi_CC_matrix, cmap='viridis')
    ax7.set_xlim([0,dF_signal.shape[1]])
    ax7.set_ylim([0,dF_signal.shape[1]])

    # calculate cc pairs
    same_axon_pairs = []
    diff_axon_pairs = []

    # get all cc values from same axon
    for axon in rois_from_axon:
        axon = np.array(axon)
        axon_cc_vals = []
        non_axon_cc_vals = []
        # print(axon[0],axon[-1]+1)
        # axon = axon-axon[0]

        for a in axon:
            # first, grab cc values of the same axon, then from different axon
            row_cc_vals = np.ones((roi_CC_matrix.shape[0],),dtype=bool)
            non_axon_idx = np.arange(0,roi_CC_matrix.shape[0],1)
            non_axon_idx = np.setdiff1d(non_axon_idx,axon)
            row_cc_vals[a] = False
            row_cc_vals[non_axon_idx] = False
            axon_cc_vals.append(roi_CC_matrix[a,row_cc_vals])
            #
            row_diff_cc_vals = np.zeros((roi_CC_matrix.shape[0],),dtype=bool)
            row_diff_cc_vals[non_axon_idx] = True
            non_axon_cc_vals.append(roi_CC_matrix[a,row_diff_cc_vals])

        # non_axon_idx = np.arange(0,roi_CC_matrix.shape[0],1)
        # non_axon_idx = np.setdiff1d(non_axon_idx,axon)
        # for a in non_axon_idx:
        #     non_axon_cc_vals.append()

        # flatten list so all the values are in one dimension
        axon_cc_vals = [item for sublist in axon_cc_vals for item in sublist]
        non_axon_cc_vals = [item for sublist in non_axon_cc_vals for item in sublist]
        same_axon_pairs.append(axon_cc_vals)
        diff_axon_pairs.append(non_axon_cc_vals)

    same_axon_pairs = np.array([item for sublist in same_axon_pairs for item in sublist])
    diff_axon_pairs = np.array([item for sublist in diff_axon_pairs for item in sublist])

    # sns.distplot(same_axon_pairs,bins=np.arange(0,1.1,0.1),kde=True, hist=True,color='g',ax=ax8,hist_kws={"edgecolor":"w","alpha":0.5})
    # sns.distplot(diff_axon_pairs,bins=np.arange(0,1.1,0.1),kde=True, hist=True,color='b',ax=ax8,hist_kws={"edgecolor":"w","alpha":0.5})
    same_axon_pairs[same_axon_pairs<0]=0
    diff_axon_pairs[diff_axon_pairs<0]=0
    sns.kdeplot(same_axon_pairs,color='#FDE725',gridsize=100,bw=.05,shade=True,ax=ax8,**{"linewidth":2})
    sns.kdeplot(diff_axon_pairs,color='#453882',gridsize=100,bw=.05,shade=True,ax=ax8,**{"linewidth":2})
    # ax8.hist(
    # ax8.hist(diff_axon_pairs[0],bins=np.arange(0,1.1,0.1),density=True,facecolor='0.8', edgecolor='w')
    ax8.set_xlim([-0.4,1.2])
    ax8.set_ylim([0,6])
    ax8.axvline(cc_thresh,ls='--',lw=3,c='r')

    ax8.set_xlabel('correlation coefficient', fontsize=24)
    ax8.set_ylabel('probability density', fontsize=24)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['bottom'].set_linewidth(2)
    ax8.spines['left'].set_linewidth(2)

    ax8.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    ax10.set_xlim([-0.4,1.2])
    ax10.set_ylim([0,6])
    ax10.set_xlabel('correlation coefficient', fontsize=24)
    ax10.set_ylabel('probability density', fontsize=24)
    ax10.axvline(cc_thresh,ls='--',lw=3,c='r')
    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    ax10.spines['bottom'].set_linewidth(2)
    ax10.spines['left'].set_linewidth(2)
    ax10.tick_params( \
        axis='both', \
        direction='out', \
        labelsize=20, \
        length=4, \
        width=2, \
        left='on', \
        bottom='on', \
        right='off', \
        top='off')

    folder = 'bouton_cc'
    # fname = 'bouton_cc_LF171211_2_Day201852'

    if folder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + folder):
            os.mkdir(loc_info['figure_output_path'] + folder)
        fname = loc_info['figure_output_path'] + folder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat,dpi=150)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

    print(fname)

    plt.close(fig)
    for gr in grouped_rois:
        plot_roi_group(dF_signal_sess, dF_signal_ol, gr, subfolder)



def run_LF171211_2_Day201852():

    # cc threshold for determining which ROIs belong to the same axons and which belong to different axons
    cc_thresh = 0.5

    subfolder = 'LF171211_2_Day201852'

    sig_filename_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF171211_2\\201852 new\\roi_cc\\M01_000_000.sig'
    extra_fname_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF171211_2\\201852 new\\roi_cc\\M01_000_000.extra'

    sig_filename_sess =  'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF171211_2\\201852 new\\M01_000_000.sig'
    extra_fname_sess = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF171211_2\\201852 new\\M01_000_000.extra'

    sig_filename_ol =  'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF171211_2\\201852 new\\M01_000_002.sig'
    extra_fname_ol = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF171211_2\\201852 new\\M01_000_002.extra'

    rois_from_axon = [[0,1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15,16,17],[18,19,20,21,21,22]]
    # rois_from_axon = [[18,19,20,21,21,22]] # <-- its somewhere in he re!

    # load gcamp data from .sig file
    print('importing session data from .sig file')
    gcamp_raw_sess = np.genfromtxt(sig_filename_sess, delimiter=',' )
    PIL_gcamp_sess = gcamp_raw_sess[:, int(np.size(gcamp_raw_sess, 1) / 2):int(np.size(gcamp_raw_sess, 1))]
    ROI_gcamp_sess = gcamp_raw_sess[:, (int(np.size(gcamp_raw_sess, 1) / np.size(gcamp_raw_sess, 1))-1):int(np.size(gcamp_raw_sess, 1) / 2)]

    print('importing openloop data from .sig file')
    gcamp_raw_ol = np.genfromtxt(sig_filename_ol, delimiter=',' )
    PIL_gcamp_ol = gcamp_raw_ol[:, int(np.size(gcamp_raw_ol, 1) / 2):int(np.size(gcamp_raw_ol, 1))]
    ROI_gcamp_ol = gcamp_raw_ol[:, (int(np.size(gcamp_raw_ol, 1) / np.size(gcamp_raw_ol, 1))-1):int(np.size(gcamp_raw_ol, 1) / 2)]

    # load gcamp data from .sig file
    print('importing cc data from .sig file')
    gcamp_raw_cc = np.genfromtxt(sig_filename_cc, delimiter=',' )
    PIL_gcamp_cc = gcamp_raw_cc[:, int(np.size(gcamp_raw_cc, 1) / 2):int(np.size(gcamp_raw_cc, 1))]
    ROI_gcamp_cc = gcamp_raw_cc[:, (int(np.size(gcamp_raw_cc, 1) / np.size(gcamp_raw_cc, 1))-1):int(np.size(gcamp_raw_cc, 1) / 2)]

    # delete bad rois
    PIL_gcamp_cc = np.delete(PIL_gcamp_cc,[17],1)
    ROI_gcamp_cc = np.delete(ROI_gcamp_cc,[17],1)

    fname = 'bouton_cc_' + subfolder
    make_cc_plot(ROI_gcamp_cc, PIL_gcamp_cc, ROI_gcamp_sess, PIL_gcamp_sess, ROI_gcamp_ol, PIL_gcamp_ol, extra_fname_cc, extra_fname_sess, extra_fname_ol, rois_from_axon, cc_thresh, fname, subfolder)

def run_LF170214_1_Day201777():

    # cc threshold for determining which ROIs belong to the same axons and which belong to different axons
    cc_thresh = 0.5

    subfolder = 'LF170214_1_Day201777'

    sig_filename_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0707\\roi_cc\\M01_000_009.sig'
    extra_fname_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0707\\roi_cc\\M01_000_009.extra'

    sig_filename_sess =  'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0707\\M01_000_009.sig'
    extra_fname_sess = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0707\\M01_000_009.extra'

    sig_filename_ol =  'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0707\\M01_000_010.sig'
    extra_fname_ol = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0707\\M01_000_010.extra'

    rois_from_axon = [[0,1,2,3],[4,5,6,7],[8,9,10],[11,12,13,14],[15,16,17,18,19,20],[21,22,23,24,25]]
    # rois_from_axon = [[18,19,20,21,21,22]] # <-- its somewhere in he re!

    # load gcamp data from .sig file
    print('importing session data from .sig file')
    gcamp_raw_sess = np.genfromtxt(sig_filename_sess, delimiter=',' )
    PIL_gcamp_sess = gcamp_raw_sess[:, int(np.size(gcamp_raw_sess, 1) / 2):int(np.size(gcamp_raw_sess, 1))]
    ROI_gcamp_sess = gcamp_raw_sess[:, (int(np.size(gcamp_raw_sess, 1) / np.size(gcamp_raw_sess, 1))-1):int(np.size(gcamp_raw_sess, 1) / 2)]

    print('importing openloop data from .sig file')
    gcamp_raw_ol = np.genfromtxt(sig_filename_ol, delimiter=',' )
    PIL_gcamp_ol = gcamp_raw_ol[:, int(np.size(gcamp_raw_ol, 1) / 2):int(np.size(gcamp_raw_ol, 1))]
    ROI_gcamp_ol = gcamp_raw_ol[:, (int(np.size(gcamp_raw_ol, 1) / np.size(gcamp_raw_ol, 1))-1):int(np.size(gcamp_raw_ol, 1) / 2)]

    # load gcamp data from .sig file
    print('importing cc data from .sig file')
    gcamp_raw_cc = np.genfromtxt(sig_filename_cc, delimiter=',' )
    PIL_gcamp_cc = gcamp_raw_cc[:, int(np.size(gcamp_raw_cc, 1) / 2):int(np.size(gcamp_raw_cc, 1))]
    ROI_gcamp_cc = gcamp_raw_cc[:, (int(np.size(gcamp_raw_cc, 1) / np.size(gcamp_raw_cc, 1))-1):int(np.size(gcamp_raw_cc, 1) / 2)]

    # delete bad rois
    PIL_gcamp_cc = np.delete(PIL_gcamp_cc,[15],1)
    ROI_gcamp_cc = np.delete(ROI_gcamp_cc,[15],1)

    fname = 'bouton_cc_' + subfolder
    make_cc_plot(ROI_gcamp_cc, PIL_gcamp_cc, ROI_gcamp_sess, PIL_gcamp_sess, ROI_gcamp_ol, PIL_gcamp_ol, extra_fname_cc, extra_fname_sess, extra_fname_ol, rois_from_axon, cc_thresh, fname, subfolder)

def run_LF170214_1_Day2017714():

    # cc threshold for determining which ROIs belong to the same axons and which belong to different axons
    cc_thresh = 0.4

    subfolder = 'LF170214_1_Day2017714'

    sig_filename_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0714\\roi_cc\\M01_000_000.sig'
    extra_fname_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0714\\roi_cc\\M01_000_000.extra'

    sig_filename_sess =  'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0714\\M01_000_000.sig'
    extra_fname_sess = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0714\\M01_000_000.extra'

    sig_filename_ol =  'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0714\\M01_000_001.sig'
    extra_fname_ol = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF170214_1\\new sig data\\0714\\M01_000_001.extra'

    rois_from_axon = [[0,1,2,3,4],[5,6,7],[8,9,10,11,12],[13,14,15],[16,17,18,19],[20,21,22,23,24]]
    # rois_from_axon = [[18,19,20,21,21,22]] # <-- its somewhere in he re!

    # load gcamp data from .sig file
    print('importing session data from .sig file')
    gcamp_raw_sess = np.genfromtxt(sig_filename_sess, delimiter=',' )
    PIL_gcamp_sess = gcamp_raw_sess[:, int(np.size(gcamp_raw_sess, 1) / 2):int(np.size(gcamp_raw_sess, 1))]
    ROI_gcamp_sess = gcamp_raw_sess[:, (int(np.size(gcamp_raw_sess, 1) / np.size(gcamp_raw_sess, 1))-1):int(np.size(gcamp_raw_sess, 1) / 2)]

    print('importing openloop data from .sig file')
    gcamp_raw_ol = np.genfromtxt(sig_filename_ol, delimiter=',' )
    PIL_gcamp_ol = gcamp_raw_ol[:, int(np.size(gcamp_raw_ol, 1) / 2):int(np.size(gcamp_raw_ol, 1))]
    ROI_gcamp_ol = gcamp_raw_ol[:, (int(np.size(gcamp_raw_ol, 1) / np.size(gcamp_raw_ol, 1))-1):int(np.size(gcamp_raw_ol, 1) / 2)]

    # load gcamp data from .sig file
    print('importing cc data from .sig file')
    gcamp_raw_cc = np.genfromtxt(sig_filename_cc, delimiter=',' )
    PIL_gcamp_cc = gcamp_raw_cc[:, int(np.size(gcamp_raw_cc, 1) / 2):int(np.size(gcamp_raw_cc, 1))]
    ROI_gcamp_cc = gcamp_raw_cc[:, (int(np.size(gcamp_raw_cc, 1) / np.size(gcamp_raw_cc, 1))-1):int(np.size(gcamp_raw_cc, 1) / 2)]

    # delete bad rois
    PIL_gcamp_cc = np.delete(PIL_gcamp_cc,[2,6,7,8,9,10,22,23,26,34],1)
    ROI_gcamp_cc = np.delete(ROI_gcamp_cc,[2,6,7,8,9,10,22,23,26,34],1)

    fname = 'bouton_cc_' + subfolder
    make_cc_plot(ROI_gcamp_cc, PIL_gcamp_cc, ROI_gcamp_sess, PIL_gcamp_sess, ROI_gcamp_ol, PIL_gcamp_ol, extra_fname_cc, extra_fname_sess, extra_fname_ol, rois_from_axon, cc_thresh, fname, subfolder)

def run_LF180112_2_Day20180424_1():

    # cc threshold for determining which ROIs belong to the same axons and which belong to different axons
    cc_thresh = 0.5

    subfolder = 'LF180112_2_Day20180424_1'

    sig_filename_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\roi_cc\\M01_000_019_0000.sig'
    extra_fname_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\roi_cc\\M01_000_019_0000.extra'

    sig_filename_sess = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_019_0000.sig'
    extra_fname_sess = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_019_0000.extra'

    sig_filename_ol = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_023_0000.sig'
    extra_fname_ol = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_023_0000.extra'

    rois_from_axon = [[0,1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18,19]]
    # rois_from_axon = [[18,19,20,21,21,22]] # <-- its somewhere in he re!

    # load gcamp data from .sig file
    print('importing session data from .sig file')
    gcamp_raw_sess = np.genfromtxt(sig_filename_sess, delimiter=',' )
    PIL_gcamp_sess = gcamp_raw_sess[:, int(np.size(gcamp_raw_sess, 1) / 2):int(np.size(gcamp_raw_sess, 1))]
    ROI_gcamp_sess = gcamp_raw_sess[:, (int(np.size(gcamp_raw_sess, 1) / np.size(gcamp_raw_sess, 1))-1):int(np.size(gcamp_raw_sess, 1) / 2)]

    print('importing openloop data from .sig file')
    gcamp_raw_ol = np.genfromtxt(sig_filename_ol, delimiter=',' )
    PIL_gcamp_ol = gcamp_raw_ol[:, int(np.size(gcamp_raw_ol, 1) / 2):int(np.size(gcamp_raw_ol, 1))]
    ROI_gcamp_ol = gcamp_raw_ol[:, (int(np.size(gcamp_raw_ol, 1) / np.size(gcamp_raw_ol, 1))-1):int(np.size(gcamp_raw_ol, 1) / 2)]

    # load gcamp data from .sig file
    print('importing cc data from .sig file')
    gcamp_raw_cc = np.genfromtxt(sig_filename_cc, delimiter=',' )
    PIL_gcamp_cc = gcamp_raw_cc[:, int(np.size(gcamp_raw_cc, 1) / 2):int(np.size(gcamp_raw_cc, 1))]
    ROI_gcamp_cc = gcamp_raw_cc[:, (int(np.size(gcamp_raw_cc, 1) / np.size(gcamp_raw_cc, 1))-1):int(np.size(gcamp_raw_cc, 1) / 2)]

    # delete bad rois
    PIL_gcamp_cc = np.delete(PIL_gcamp_cc,[10,11,12],1)
    ROI_gcamp_cc = np.delete(ROI_gcamp_cc,[10,11,12],1)

    fname = 'bouton_cc_' + subfolder
    make_cc_plot(ROI_gcamp_cc, PIL_gcamp_cc, ROI_gcamp_sess, PIL_gcamp_sess, ROI_gcamp_ol, PIL_gcamp_ol, extra_fname_cc, extra_fname_sess, extra_fname_ol, rois_from_axon, cc_thresh, fname, subfolder)

def run_LF180112_2_Day20180424_2():

    # cc threshold for determining which ROIs belong to the same axons and which belong to different axons
    cc_thresh = 0.5

    subfolder = 'LF180112_2_Day20180424_2'

    sig_filename_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\roi_cc\\M01_000_019_0025.sig'
    extra_fname_cc = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\roi_cc\\M01_000_019_0025.extra'

    sig_filename_sess = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_019_0025.sig'
    extra_fname_sess = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_019_0025.extra'

    sig_filename_ol = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_023_0025.sig'
    extra_fname_ol = 'E:\\MTH3_data\\MTH3_data\\animals_h5\\LF180112_2\\180424 new\\M01_000_023_0025.extra'

    rois_from_axon = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15,16]]
    # rois_from_axon = [[18,19,20,21,21,22]] # <-- its somewhere in he re!

    # load gcamp data from .sig file
    print('importing session data from .sig file')
    gcamp_raw_sess = np.genfromtxt(sig_filename_sess, delimiter=',' )
    PIL_gcamp_sess = gcamp_raw_sess[:, int(np.size(gcamp_raw_sess, 1) / 2):int(np.size(gcamp_raw_sess, 1))]
    ROI_gcamp_sess = gcamp_raw_sess[:, (int(np.size(gcamp_raw_sess, 1) / np.size(gcamp_raw_sess, 1))-1):int(np.size(gcamp_raw_sess, 1) / 2)]

    print('importing openloop data from .sig file')
    gcamp_raw_ol = np.genfromtxt(sig_filename_ol, delimiter=',' )
    PIL_gcamp_ol = gcamp_raw_ol[:, int(np.size(gcamp_raw_ol, 1) / 2):int(np.size(gcamp_raw_ol, 1))]
    ROI_gcamp_ol = gcamp_raw_ol[:, (int(np.size(gcamp_raw_ol, 1) / np.size(gcamp_raw_ol, 1))-1):int(np.size(gcamp_raw_ol, 1) / 2)]

    # load gcamp data from .sig file
    print('importing cc data from .sig file')
    gcamp_raw_cc = np.genfromtxt(sig_filename_cc, delimiter=',' )
    PIL_gcamp_cc = gcamp_raw_cc[:, int(np.size(gcamp_raw_cc, 1) / 2):int(np.size(gcamp_raw_cc, 1))]
    ROI_gcamp_cc = gcamp_raw_cc[:, (int(np.size(gcamp_raw_cc, 1) / np.size(gcamp_raw_cc, 1))-1):int(np.size(gcamp_raw_cc, 1) / 2)]

    # delete bad rois
    ROI_gcamp_cc = np.delete(ROI_gcamp_cc,[12,15],1)
    PIL_gcamp_cc = np.delete(PIL_gcamp_cc,[12,15],1)

    fname = 'bouton_cc_' + subfolder
    make_cc_plot(ROI_gcamp_cc, PIL_gcamp_cc, ROI_gcamp_sess, PIL_gcamp_sess, ROI_gcamp_ol, PIL_gcamp_ol, extra_fname_cc, extra_fname_sess, extra_fname_ol, rois_from_axon, cc_thresh, fname, subfolder)


if __name__ == "__main__":
    run_LF171211_2_Day201852()
    # run_LF170214_1_Day201777()
    # run_LF170214_1_Day2017714()
    # run_LF180112_2_Day20180424_1()
    # run_LF180112_2_Day20180424_2()
