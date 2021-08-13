"""
Calculate cross correlation between ROIs. This is intended for testing the cross correlation between bouton recordings.


"""

# %matplotlib inline

# load local settings file
import matplotlib
import numpy as np
import warnings; warnings.simplefilter('ignore')
import sys, yaml, h5py, json, os
sys.path.append("./Analysis")
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from scipy.signal import butter, filtfilt
from sklearn.cluster.bicluster import SpectralCoclustering
import statsmodels.api as sm
plt.rcParams['svg.fonttype'] = 'none'
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_style("white")
with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

fformat='png'

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def bouton_CC(roi_param_list_all, subfolder=''):
    # run through all roi_param files
    for i,rpl in enumerate(roi_param_list_all):
        fig = plt.figure(figsize=(15,15), dpi=200)
        all_ax = []
        for j in range(36):
            all_ax.append(plt.subplot(6,6,j+1))

        # print(rpl)
        # load roi parameters for given session
        with open(rpl[0],'r') as f:
            roi_params = json.load(f)

        # grab roi numbers to be included
        roi_list_all = roi_params['valid_rois']

        h5path = loc_info['imaging_dir'] + rpl[1] + '/' + rpl[1] + '.h5'
        h5dat = h5py.File(h5path, 'r')
        behav_ds = np.copy(h5dat[rpl[2] + '/behaviour_aligned'])
        dF_ds = np.copy(h5dat[rpl[2] + '/dF_win'])
        h5dat.close()

        # empty matrix holding correlation coefficients
        roi_CC_matrix = np.zeros((len(roi_list_all),len(roi_list_all)))

        # filter specs
        order = 6
        fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
        cutoff = 1 # desired cutoff frequency of the filter, Hz

        # print(len(roi_list_all))
        for j,roi in enumerate(roi_list_all):
            # print(j)
            # filter and normalize signal
            df_roi = butter_lowpass_filter(dF_ds[:,roi], cutoff, fs, order)
            df_roi = df_roi / np.amax(df_roi)
            for k,roi_corr in enumerate(roi_list_all):
                dF_roi_corr = butter_lowpass_filter(dF_ds[:,roi_corr], cutoff, fs, order)
                dF_roi_corr = dF_roi_corr / np.amax(dF_roi_corr)
                roi_CC_matrix[j,k],_ = stats.pearsonr(df_roi, dF_roi_corr)

            # roi_corrcoeff[k], _ = stats.pearsonr(df_roi, dF_roi_corr)

            # if roi_corrcoeff[k] > 0.5:
            #     print(roi_corr)

        # sns.distplot(roi_corrcoeff, ax=all_ax[0])
        all_ax[0].pcolormesh(roi_CC_matrix, cmap='viridis')
        # plt.colorbar(all_ax[0]_cmap, ax=all_ax[0])

        all_ax[0].set_xlim([0,len(roi_list_all)])
        all_ax[0].set_ylim([0,len(roi_list_all)])

        for j in np.arange(2,37,1):
            model = SpectralCoclustering(n_clusters=j)
            model.fit(roi_CC_matrix)

            fit_data = roi_CC_matrix[np.argsort(model.row_labels_)]
            fit_data = fit_data[:, np.argsort(model.column_labels_)]

            all_ax[j-1].pcolormesh(fit_data, cmap='viridis')
            # plt.colorbar(all_ax[1]_cmap, ax=all_ax[1])

            all_ax[j-1].set_xlim([0,len(roi_list_all)])
            all_ax[j-1].set_ylim([0,len(roi_list_all)])

    # roi_CC_matrix = np.zeros((len(roi_selection),len(roi_selection)))
    #
    # for i in range(len(roi_selection)):
    #     for j in range(len(roi_selection)):
    #         # roi_CC_matrix[i,j] = signal.correlate(dF_ds[:,i], dF_ds[:,j], mode='valid')
    #         roi_CC_matrix[i,j] = stats.pearsonr(dF_ds[:,i], dF_ds[:,j])[0]
    #


    # ax1.set_xlim([0,len(roi_selection)])
    # ax1.set_ylim([0,len(roi_selection)])
    # ax1.set_xlabel('ROI')
    # ax1.set_ylabel('ROI')
    # plt.suptitle(str(dset)+' '+rois)
    # plt.show()
    #
    # roi_CC_res = {
    #     'included_datasets' : dset,
    #     'rois' : rois,
    #     'roi_CC_matrix' : roi_CC_matrix.tolist()
    # }
    #
    # if not os.path.isdir(content['figure_output_path'] + subfolder):
    #     os.mkdir(content['figure_output_path'] + subfolder)
    #
    # with open(content['figure_output_path'] + subfolder + os.sep + dset[0]+dset[1] + '.json','w+') as f:
    #     json.dump(roi_CC_res,f)
    #
        fname = rpl[1]+rpl[2]
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
        plt.close(fig)


if __name__ == "__main__":
    roi_param_list = [
                      ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day201777.json','LF170214_1','Day201777',38.49],
                      ['E:\\MTH3_figures\\LF170214_1\\LF170214_1_Day2017714.json','LF170214_1','Day2017714',38.92],
                      ['E:\\MTH3_figures\\LF171211_2\\LF171211_2_Day201852.json','LF171211_2','Day201852',31.07],
                      ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_1.json','LF180112_2','Day2018424_1',22.63],
                      ['E:\\MTH3_figures\\LF180112_2\\LF180112_2_Day2018424_2.json','LF180112_2','Day2018424_2',22.63]
                     ]

    bouton_CC(roi_param_list, 'bouton_cc')
