"""
GLM: make regressors (called filters in Driscoll paper), and regress neural activity against them. E.g. location on track, distance from landmark vs distance from reward.
Regularization helps dealing with parameters where we don't have a ton of data. L1 typically snaps values to 0 or 1, while L2 provides a closer approximation of the actual
neural response.

For example: lmcenter vs reward: for each trial take distance from landmark (binned) and distance from reward (binned) and regress against them. The regressor should be a gaussian

Step 1: make regressors (e.g. space: boxcar convolved with gaussian for spatial bins that tile the track)
Step 2: regress neuronal response against regressors (Qs: how do you do that? Which regression method do you pick?)
Step 3: Apply regulisarization method?

Predictors (list of things to be considered)
- location
- running speed
- running vs. non-running
- distance to/from point of interest (e.g. landmark vs reward)
- lick onset (index of every first lick x gaussian)
- reward point
- distance since trial start
- trial type

- bootstrap by taking a random (sub)sample for datapoints and X coordinates --> recalculate

"""
import sys,os,json,matplotlib,yaml #h5py,
from sklearn.metrics import r2_score
import numpy as np
from scipy import signal, stats
import scipy.io as sio
import warnings; warnings.simplefilter('ignore')
from matplotlib import pyplot as plt
import glmnet
import ipdb


import seaborn as sns
sns.set_style("white")

with open('../' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

from filter_trials import filter_trials
from event_ind import event_ind


fformat = 'png'

TRACK_START = 0
BINNR_SHORT = 80
BINNR_LONG = 100
TRACK_END_SHORT = 400
TRACK_END_LONG = 500
TRACK_SHORT = 3
TRACK_LONG = 4
NUM_PREDICTORS = 20

def make_spatial_predictors():
    # set up spatial predictors
    binsize_short = TRACK_END_SHORT/NUM_PREDICTORS
    # matrix that holds predictors
    predictor_short_x = np.zeros((TRACK_END_SHORT, NUM_PREDICTORS))
    # calculate bin edges and centers
    boxcar_short_centers = np.linspace(TRACK_START+binsize_short/2, TRACK_END_SHORT-binsize_short/2, NUM_PREDICTORS)

    location_vector = np.arange(0,TRACK_END_SHORT,1)

    # gaussian kernel to convolve boxcar spatial predictors with
    gauss_kernel = signal.gaussian(100,5) * 2

    # create boxcar vectors and convolve with gaussian
    for i,bcc in enumerate(boxcar_short_centers):
        # calculate single predictor
        predictor_x = np.zeros((TRACK_END_SHORT,))
        predictor_x[int(boxcar_short_centers[i]-binsize_short/2) : int(boxcar_short_centers[i]+binsize_short/2)] = 1
        predictor_x = np.convolve(predictor_x,gauss_kernel,mode='same')
        predictor_x = predictor_x/np.amax(predictor_x)
        predictor_short_x[:,i] = predictor_x
        # plt.plot(predictor_x)
        # predictor_short_x[:,i] = signal.resample(predictor_x,BINNR_SHORT)

    plt.figure(figsize=(10,5))
    plt.subplot(111)
    for px in predictor_short_x.T:
        plt.plot(px)

    plt.axvline(200,c='k')
    plt.axvline(240,c='k')
    plt.axvline(320,c='k')


    # add constant predictor
    # predictor_short_x = np.insert(predictor_short_x,0,np.ones((predictor_short_x.shape[0])), axis=1)

    combine_spatial_predictors(predictor_short_x)

    return predictor_short_x, location_vector

def combine_spatial_predictors(predictor_short_x):
    dual_RF_predictors = np.zeros((predictor_short_x.shape[0],predictor_short_x.shape[1] * predictor_short_x.shape[1]))


    i = 0
    for npx_1 in predictor_short_x.T:
        for npx_2 in predictor_short_x.T:
            dual_RF_predictors[:,i] = npx_1 + npx_2
            i = i + 1

    # ipdb.set_trace()
    # pass



def binned_trials(behav_ds, dF_ds, roi):


    # run through each trial and plot corresponding imaging data
    plt.figure()
    trials_short = filter_trials( behav_ds, [], ['tracknumber',TRACK_SHORT])
    mean_dF_short = np.zeros((np.size(trials_short,0),BINNR_SHORT))
    # run through SHORT trials and calculate avg dF/F for each bin and trial
    for i,t in enumerate(trials_short):
        # pull out current trial and corresponding dF data and bin it
        cur_trial_loc = behav_ds[behav_ds[:,6]==t,1]
        cur_trial_dF_roi = dF_ds[behav_ds[:,6]==t,roi]
        mean_dF_trial = stats.binned_statistic(cur_trial_loc, cur_trial_dF_roi, 'mean', BINNR_SHORT,
                                               (0.0, TRACK_END_SHORT))[0]
        mean_dF_trial = np.nan_to_num(mean_dF_trial)
        mean_dF_short[i,:] = mean_dF_trial
        plt.plot(mean_dF_trial,c='0.8')

    # calculate mean trace by evaluating which datapoints contain data for at least half the trials included in the plot
    mean_valid_indices = []
    for i,trace in enumerate(mean_dF_short.T):
        if np.count_nonzero(np.isnan(trace))/trace.shape[0] < 0.5:
            mean_valid_indices.append(i)
    roi_meanpeak_short = np.nanmax(np.nanmean(mean_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_short_idx = np.nanargmax(np.nanmean(mean_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0))
    roi_meanpeak_short_location = (roi_meanpeak_short_idx+mean_valid_indices[0]) * (TRACK_END_SHORT/BINNR_SHORT)
    mean_trace_short = np.nanmean(mean_dF_short[:,mean_valid_indices[0]:mean_valid_indices[-1]],0)
    plt.plot(np.arange(mean_valid_indices[0], mean_valid_indices[-1],1), mean_trace_short,c=sns.xkcd_rgb["windows blue"],lw=3)
    plt.axvline((roi_meanpeak_short_idx+mean_valid_indices[0]),c='k', ls='--')
    mean_trace_short_start = mean_valid_indices[0]
    plt.xticks([0,5,10,15,20,25,30,35,40],['0','50','100','150','200','250','300','350','400'])
    plt.axvline(22,c='0.5',ls='--')
    # plt.xticklabels()
    # plt.show()
    # plt.close()



    # X = make_spatial_predictors()
    # y = np.zeros((BINNR_SHORT,))
    # y[mean_valid_indices[0]:mean_valid_indices[-1]] = mean_trace_short
    #
    #
    # print(reg.score(X, mean_dF_short.T))
    # print(np.mean(reg.coef_,axis=0))
    # print(mean_dF_short.T.shape)
    # plt.figure(figsize=(5,10))
    # plt.subplot(211)
    # reg = LinearRegression().fit(X, mean_dF_short.T)
    # plt.plot(np.mean(reg.coef_,axis=0))
    # print(np.mean(reg.coef_,axis=0))
    # plt.subplot(212)
    # reg = LinearRegression().fit(X, y.T)
    # plt.plot(reg.coef_)
    # print(reg.coef_)
    # plt.show()

def fit_test(behav_ds, roi_gcamp):
    """ sandbox for fitting """

    # set up predictors
    x_predictors, location_vector = make_spatial_predictors()
    gauss_kernel = signal.gaussian(100,10)
    boxcar_short_edges = np.linspace(TRACK_START, TRACK_END_SHORT, NUM_PREDICTORS+1)
    animal_loc = behav_ds[:40000,1]
    animal_loc_test = behav_ds[40000:,1]
    roi_gcamp_test = roi_gcamp[40000:]
    roi_gcamp = roi_gcamp[:40000]
    predictor_short_x = np.zeros((animal_loc.shape[0], NUM_PREDICTORS))
    predictor_short_x_test = np.zeros((animal_loc_test.shape[0], NUM_PREDICTORS))

    # ipdb.set_trace()
    for i,al in enumerate(animal_loc):
        for j,spx in enumerate(x_predictors.T):
            predictor_short_x[i,j] = 1 * spx[(np.abs(location_vector - animal_loc[i])).argmin()]

    for i,al in enumerate(animal_loc_test):
        for j,spx in enumerate(x_predictors.T):
            predictor_short_x_test[i,j] = 1 * spx[(np.abs(location_vector - animal_loc_test[i])).argmin()]


    glm_obj = glmnet.ElasticNet(alpha=0.1,n_lambda=100)
    glm_obj.fit(predictor_short_x.copy(), roi_gcamp.copy())
    print(glm_obj.coef_)
    roi_gcamp_pred = glm_obj.predict(predictor_short_x_test)
    print(r2_score(roi_gcamp_test,roi_gcamp_pred))

    plt.figure()
    plt.subplot(511)
    plt.plot(predictor_short_x)
    plt.subplot(512)
    plt.plot(glm_obj.coef_)
    plt.subplot(513)
    plt.plot(animal_loc,c='k')
    plt.twinx()
    plt.plot(roi_gcamp,c='g')
    plt.subplot(514)
    plt.plot(animal_loc_test,c='k')
    plt.twinx()
    plt.plot(roi_gcamp_test,c='g')
    plt.plot(roi_gcamp_pred,c='r')
    plt.subplot(515)
    plt.plot(roi_gcamp_test-roi_gcamp_pred)

    return []


if __name__ == "__main__":
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    # MOUSE = 'SIM_2'
    # SESSION = 'Day1'
    # ROI = 0

    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    ROI = 13

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
    loaded_data = sio.loadmat(processed_data_path)
    behav_ds = loaded_data['behaviour_aligned']
    dF_ds = loaded_data['spikerate']

    fit_test(behav_ds, dF_ds[:,ROI])

    plt.show()
