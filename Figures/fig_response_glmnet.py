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
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, ElasticNet
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

load_raw = True
load_h5 = False

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


def fit_distances_reward(behav_ds, roi_gcamp):
    """ fit to distances from certain points """
    bins_short = 20
    bins_long = 25
    # print(reward_locs)
    behav_ds_short = behav_ds[behav_ds[:,4] == 3,:]
    trials_short = filter_trials( behav_ds_short, [], ['tracknumber',3])
    reward_locs = event_ind(behav_ds_short, ['rewards_all', -1], trials_short)
    animal_location_short_idx = np.where(behav_ds_short[:,4] == 3)[0]
    animal_location = behav_ds_short[animal_location_short_idx,1]
    roi_gcamp = roi_gcamp[animal_location_short_idx]

    # determine size (cm) of individual spatial bins
    binsize_short = TRACK_END_SHORT/bins_short
    binsize_long = TRACK_END_LONG/bins_long
    # matrix that holds predictors
    reward_predictor = np.zeros((animal_location.shape[0], bins_short))
    reward_loc = 220
    # calculate bin edges and centers
    boxcar_short_centers = np.linspace(-320+binsize_short/2, 80-binsize_short/2, bins_short)
    boxcar_short_edges = np.linspace(-320, 80, bins_short+1)

    gauss_kernel = signal.gaussian(100,20)
    plt.figure(figsize=(15,10))
    plt.subplot(4,1,1)
    plt.plot(animal_location)
    # plt.xlim([200,1000])
    # plot predictors
    plt.subplot(4,1,2)
    for i,bcc in enumerate(boxcar_short_centers):
        # calculate single predictor
        predictor_x = np.zeros((TRACK_END_SHORT,))
        # print('bin: ' + str(i) + ' range: ' + str(int(boxcar_short_centers[i]-binsize_short/2)) + ' - ' + str(int(boxcar_short_centers[i]+binsize_short/2)))
        predictor_x[int(boxcar_short_centers[i]-binsize_short/2) : int(boxcar_short_centers[i]+binsize_short/2)] = 1
        # predictor_x = np.convolve(predictor_x,gauss_kernel,mode='same')
        predictor_x = predictor_x/np.amax(predictor_x)
        plt.plot(predictor_x)

    # loop through all bins and set predictors
    plt.subplot(4,1,3)
    for i,tn in enumerate(reward_locs):
        for bs in range(bins_short):
            trial_idx_range = [np.where(behav_ds_short[:,6] == tn[1])[0][0], np.where(behav_ds_short[:,6] == tn[1])[0][-1]]
            reward_loc_trial = behav_ds_short[int(tn[0]),1]
            # set all datapoints where animal is in a given spatial bin to 1
            reward_predictor[np.logical_and((animal_location-reward_loc_trial) > boxcar_short_edges[bs], (animal_location-reward_loc_trial) < boxcar_short_edges[bs+1]),bs] = 1
            # print(boxcar_short_edges[bs], boxcar_short_edges[bs+1], np.sum(reward_predictor[:,bs]))
            # reward_predictor[:,bs] = np.convolve(reward_predictor[:,bs],gauss_kernel,mode='same')
            plt.plot(reward_predictor[:,bs])

    ### RUNNING PREDICTOR
    animal_speed = behav_ds_short[animal_location_short_idx,3]
    # split running speed into slow and fast
    speed_bins = 2
    # below speed_threshold (cm/sec) = slow
    speed_threshold = 5
    speed_short_x = np.zeros((animal_speed.shape[0], speed_bins))

    # set all datapoints where animal is
    speed_short_x[animal_speed < speed_threshold,0] = 1
    speed_short_x[animal_speed > speed_threshold,1] = 1

    # plt.xlim([200,1000])
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    reward_predictor = np.insert(reward_predictor,0,np.ones((reward_predictor.shape[0])), axis=1)
    reward_predictor = np.hstack((reward_predictor,speed_short_x))
    fit = glmnet_python.glmnet(x = reward_predictor.copy(), y = roi_gcamp.copy(), family = 'gaussian', alpha = 0.9, nlambda = 100)
    # cvfit = glmnet_python.cvglmnet(x = reward_predictor.copy(), y = roi_gcamp.copy(), family = 'gaussian', alpha = 0.9, nlambda = 100, ptype='mse', nfolds=20)
    # glmnet_python.glmnetPrint(fit)
    glmnet_python.glmnetPlot(fit, xvar = 'dev', label = True);
    # glmnet_python.glmnetPlot(cvfit)
    coeffs = (glmnet_python.glmnetCoef(fit, s = np.float64([0.01]), exact = False))
    # plt.subplot(212)
    # plt.plot(coeffs)
    # plt.suptitle('reward distance fit')
    print(np.argmax(coeffs)-1, boxcar_short_centers[np.argmax(coeffs)-1])
    # plt.show()
    return coeffs

def fit_distances_lmcenter(behav_ds, roi_gcamp):
    """ fit to distances from certain points """
    bins_short = 20
    bins_long = 25
    animal_location_short_idx = np.where(behav_ds[:,4] == 3)[0]
    animal_location = behav_ds[animal_location_short_idx,1]
    roi_gcamp = roi_gcamp[animal_location_short_idx]
    # determine size (cm) of individual spatial bins
    binsize_short = TRACK_END_SHORT/bins_short
    binsize_long = TRACK_END_LONG/bins_long
    # matrix that holds predictors
    lmcenter_predictor = np.zeros((animal_location.shape[0], bins_short))
    lmcenter_loc = 220
    # calculate bin edges and centers
    boxcar_short_centers = np.linspace(-220+binsize_short/2, 180-binsize_short/2, bins_short)
    boxcar_short_edges = np.linspace(-220, 180, bins_short+1)

    gauss_kernel = signal.gaussian(100,20)
    plt.figure(figsize=(15,10))
    plt.subplot(4,1,1)
    plt.plot(animal_location)
    # plt.xlim([200,1000])
    # plot predictors
    plt.subplot(4,1,2)
    for i,bcc in enumerate(boxcar_short_centers):
        # calculate single predictor
        predictor_x = np.zeros((TRACK_END_SHORT,))
        # print('bin: ' + str(i) + ' range: ' + str(int(boxcar_short_centers[i]-binsize_short/2)) + ' - ' + str(int(boxcar_short_centers[i]+binsize_short/2)))
        predictor_x[int(boxcar_short_centers[i]-binsize_short/2) : int(boxcar_short_centers[i]+binsize_short/2)] = 1
        predictor_x = np.convolve(predictor_x,gauss_kernel,mode='same')
        predictor_x = predictor_x/np.amax(predictor_x)
        plt.plot(predictor_x)

    # loop through all bins and set predictors
    plt.subplot(4,1,3)
    for bs in range(bins_short):
        # set all datapoints where animal is in a given spatial bin to 1
        lmcenter_predictor[np.logical_and((animal_location-lmcenter_loc) > boxcar_short_edges[bs], (animal_location-lmcenter_loc) < boxcar_short_edges[bs+1]),bs] = 1
        # lmcenter_predictor[:,bs] = np.convolve(lmcenter_predictor[:,bs],gauss_kernel,mode='same')
        plt.plot(lmcenter_predictor[:,bs])

    ### RUNNING PREDICTOR
    animal_speed = behav_ds[animal_location_short_idx,3]
    # cap max running speed (to avoid artifacts being a problem)
    animal_speed[animal_speed>60] = 60
    # split running speed into slow and fast
    speed_bins = 3
    # below speed_threshold (cm/sec) = slow
    speed_threshold = 5
    speed_short_x = np.zeros((animal_speed.shape[0], speed_bins))
    # set all datapoints where animal is
    speed_short_x[animal_speed < speed_threshold,0] = 1
    speed_short_x[animal_speed > speed_threshold,1] = 1
    speed_short_x[:,2] = animal_speed / np.nanmax(animal_speed)

    plt.subplot(4,1,4)
    plt.plot(speed_short_x[:,2])
    # plt.show()

    # plt.xlim([200,1000])
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    lmcenter_predictor = np.insert(lmcenter_predictor,0,np.ones((lmcenter_predictor.shape[0])), axis=1)
    lmcenter_predictor = np.hstack((lmcenter_predictor,speed_short_x))
    reg = LinearRegression().fit(lmcenter_predictor, roi_gcamp)
    ipdb.set_trace()
    # reg = LassoCV(alphas=[0,0.001,0.01,0.1,0.5,1.0,2.5,5,10,100]).fit(predictor_short_x, roi_gcamp)
    # reg = ElasticNet(alpha=0.01, l1_ratio=0.1).fit(predictor_short_x, roi_gcamp)
    # fit = glmnet_python.glmnet(x = lmcenter_predictor.copy(), y = roi_gcamp.copy(), family = 'gaussian', alpha = 0.9, nlambda = 100)
    # cvfit = glmnet_python.cvglmnet(x = lmcenter_predictor.copy(), y = roi_gcamp.copy(), family = 'gaussian', alpha = 0.9, nlambda = 100, ptype='mse', nfolds=20)
    # glmnet_python.glmnetPrint(fit)
    # glmnet_python.glmnetPlot(fit, xvar = 'dev', label = True);
    # glmnet_python.glmnetPlot(cvfit)
    # coeffs = (glmnet_python.glmnetCoef(fit, s = np.float64([0.01]), exact = False))
    # plt.subplot(212)
    # plt.plot(coeffs)
    # plt.suptitle('lmcenter distance fit')
    # print(np.argmax(coeffs)-1, boxcar_short_centers[np.argmax(coeffs)-1])
    # plt.show()
    return coeffs



def fit_location(behav_ds, roi_gcamp):
    """ make predictors for X matrix """

    ### SPATIAL PREDICTOR
    bins_short = 20
    # remove locations where animal is in blackbox
    # animal_location = behav_ds[behav_ds[:,4]!=5,:]
    animal_location = behav_ds[animal_location_short_idx,1]
    roi_gcamp = roi_gcamp[animal_location_short_idx]
    # determine size (cm) of individual spatial bins
    binsize_short = TRACK_END_SHORT/bins_short
    # matrix that holds predictors
    predictor_short_x = np.zeros((animal_location.shape[0], bins_short))
    # calculate bin edges and centers
    boxcar_short_centers = np.linspace(TRACK_START+binsize_short/2, TRACK_END_SHORT-binsize_short/2, bins_short)
    boxcar_short_edges = np.linspace(TRACK_START, TRACK_END_SHORT, bins_short+1)

    gauss_kernel = signal.gaussian(100,20)
    plt.figure(figsize=(15,10))
    plt.subplot(4,1,1)
    plt.plot(animal_location)
    # plot predictors
    plt.subplot(4,1,2)
    for i,bcc in enumerate(boxcar_short_centers):
        # calculate single predictor
        predictor_x = np.zeros((TRACK_END_SHORT,))
        predictor_x[int(boxcar_short_centers[i]-binsize_short/2) : int(boxcar_short_centers[i]+binsize_short/2)] = 1
        predictor_x = np.convolve(predictor_x,gauss_kernel,mode='same')
        predictor_x = predictor_x/np.amax(predictor_x)
        plt.plot(predictor_x)
        # predictor_short_x[:,i] = signal.resample(predictor_x,BINNR_SHORT)
    # loop through all bins and set predictors
    plt.subplot(4,1,3)
    for bs in range(bins_short):
        # set all datapoints where animal is in a given spatial bin to 1
        predictor_short_x[np.logical_and(animal_location > boxcar_short_edges[bs], animal_location < boxcar_short_edges[bs+1]),bs] = 1
        # predictor_short_x[:,bs] = np.convolve(predictor_short_x[:,bs],gauss_kernel,mode='same')

        plt.plot(predictor_short_x[:,bs])

    ### RUNNING PREDICTOR
    animal_speed = behav_ds[animal_location_short_idx,3]
    # split running speed into slow and fast
    speed_bins = 3
    # below speed_threshold (cm/sec) = slow
    speed_threshold = 5
    speed_short_x = np.zeros((animal_speed.shape[0], speed_bins))

    # set all datapoints where animal is
    speed_short_x[animal_speed < speed_threshold,0] = 1
    speed_short_x[animal_speed > speed_threshold,1] = 1
    speed_short_x[:,3] = animal_speed / np.nanmax(animal_speed)

    plt.subplot(4,1,4)
    plt.plot(speed_short_x[:,3])
    # plt.show()
    # add constant predictor
    predictor_short_x = np.insert(predictor_short_x,0,np.ones((predictor_short_x.shape[0])), axis=1)
    predictor_short_x = np.hstack((predictor_short_x,speed_short_x))
    print(boxcar_short_centers)
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    # reg = LinearRegression().fit(predictor_short_x, roi_gcamp)
    reg = LassoCV(alphas=[0,0.001,0.01,0.1,0.5,1.0,2.5,5,10,100]).fit(predictor_short_x, roi_gcamp)
    reg = ElasticNet(alpha=0.01, l1_ratio=0.1).fit(predictor_short_x, roi_gcamp)
    # plt.plot(reg.coef_)
    print(reg.coef_, reg.alpha)

    fit = glmnet_python.glmnet(x = predictor_short_x.copy(), y = roi_gcamp.copy(), family = 'poisson', alpha = 0.1, nlambda = 100)
    # cvfit = glmnet_python.cvglmnet(x = predictor_short_x.copy(), y = roi_gcamp.copy(), family = 'poisson', alpha = 0.9, nlambda = 100, ptype='mse', nfolds=20)
    glmnet_python.glmnetPrint(fit)
    glmnet_python.glmnetPlot(fit, xvar = 'lambda', label = True);
    glmnet_python.glmnetPlot(fit, xvar = 'dev', label = True);

    coeffs = (glmnet_python.glmnetCoef(fit, s = np.float64([0.001]), exact = False))
    plt.subplot(212)
    plt.plot(coeffs)

def plot_dff_v_loc(behav_ds, dF_ds, ROI):
    animal_location_short_idx = np.where(behav_ds[:,4] == 3)[0]
    animal_location = behav_ds[animal_location_short_idx,1]
    roi_gcamp = dF_ds[animal_location_short_idx, ROI]



    plt.figure()
    plt.scatter(animal_location, roi_gcamp, facecolors='none', edgecolors='0.5', alpha=0.5)
    # plt.show()



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

    # for nps in range(NUM_PREDICTORS):
        # set all datapoints where animal is in a given spatial bin to 1
        # predictor_short_x[np.logical_and(animal_loc > boxcar_short_edges[nps], animal_loc < boxcar_short_edges[nps+1]),nps] = 1
        # predictor_short_x[:,nps] = np.convolve(predictor_short_x[:,nps],gauss_kernel,mode='same')

    # ipdb.set_trace()
    for i,al in enumerate(animal_loc):
        for j,spx in enumerate(x_predictors.T):
            predictor_short_x[i,j] = 1 * spx[(np.abs(location_vector - animal_loc[i])).argmin()]

    for i,al in enumerate(animal_loc_test):
        for j,spx in enumerate(x_predictors.T):
            predictor_short_x_test[i,j] = 1 * spx[(np.abs(location_vector - animal_loc_test[i])).argmin()]

    # regr = LinearRegression()
    # regr = Ridge(alpha=.5)
    # regr = Lasso(alpha=0.02)
    # regr = ElasticNet(alpha=0.01,l1_ratio=0.5)
    # regr.fit(predictor_short_x, roi_gcamp)
    
    # ipdb.set_trace()
    # roi_gcamp_pred = regr.predict(predictor_short_x_test)


    # dmodel = 2 * np.nansum( roi_gcamp_test * np.log(roi_gcamp_test/roi_gcamp_pred) - (roi_gcamp_test-roi_gcamp_pred) )
    # roi_gcamp_pred_null = np.nanmean(roi_gcamp_pred)
    # dmodel_null = 2 * np.nansum( roi_gcamp_test * np.log(roi_gcamp_test/roi_gcamp_pred_null) - (roi_gcamp_test-roi_gcamp_pred_null) )
    # print(dmodel, dmodel_null)
    # print(1 - (dmodel/dmodel_null))

    glm_obj = glmnet.ElasticNet(alpha=0.1,n_lambda=100)
    glm_obj.fit(predictor_short_x.copy(), roi_gcamp.copy())
    print(glm_obj.coef_)
    roi_gcamp_pred = glm_obj.predict(predictor_short_x_test)
    print(r2_score(roi_gcamp_test,roi_gcamp_pred))
    # cvfit = glmnet_python.cvglmnet(x = predictor_short_x.copy(), y = roi_gcamp.copy(), family = 'poisson', alpha = 0.9, nlambda = 100, ptype='mse', nfolds=20)
    # glmnet_python.glmnetPrint(fit)
    # glmnet_python.glmnetPlot(fit, xvar = 'lambda', label = True);
    # glmnet_python.glmnetPlot(fit, xvar = 'dev', label = True);

    # coeffs = (glmnet_python.glmnetCoef(fit, s = np.float64([0.001]), exact = False))
    # print(coeffs)

    plt.figure()
    plt.subplot(511)
    plt.plot(predictor_short_x)
    plt.subplot(512)
    plt.plot(glm_obj.coef_)
    # plt.plot(coeffs)
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
    # plt.scatter(animal_loc_test,roi_gcamp_test, facecolors='none', edgecolors='0.5', alpha=0.3)
    # plt.scatter(animal_loc_train,roi_gcamp_train, facecolors='none', edgecolors='g', alpha=0.3)
    # plt.plot(animal_loc_test,gcamp_pred,c='r')



    return []


if __name__ == "__main__":
    # %load_ext autoreload
    # %autoreload
    # %matplotlib inline

    # MOUSE = 'SIM_2'
    # SESSION = 'Day1'
    # ROI = 0
    #
    MOUSE = 'LF191024_1'
    SESSION = '20191204'
    ROI = 105

    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    ROI = 13

    if load_raw == True:
        processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + 'aligned_data'
        loaded_data = sio.loadmat(processed_data_path)
        behav_ds = loaded_data['behaviour_aligned']
        dF_ds = loaded_data['spikerate']

    # elif load_h5 == True:
    #     h5path = loc_info['imaging_dir'] + MOUSE + os.sep + MOUSE + '.h5'
    #     print(h5path)
    #     h5dat = h5py.File(h5path, 'r')
    #     behav_ds = np.copy(h5dat[SESSION + '/behaviour_aligned'])
    #     dF_ds = np.copy(h5dat[SESSION + '/spikerate'])
    #     h5dat.close()

    # binned_trials(behav_ds, dF_ds, ROI)
    # plot_dff_v_loc(behav_ds, dF_ds, ROI)
    fit_test(behav_ds, dF_ds[:,ROI])
    # fit_location(behav_ds, dF_ds[:,ROI])
    # lmcenter_coeffs = fit_distances_lmcenter(behav_ds, dF_ds[:,ROI])
    # reward_coeffs = fit_distances_reward(behav_ds, dF_ds[:,ROI])
    # plt.figure(figsize=(10,10))
    # plt.subplot(211)
    # plt.plot(lmcenter_coeffs)
    # plt.subplot(212)
    # plt.plot(reward_coeffs)
    plt.show()
