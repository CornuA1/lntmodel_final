"""
Script to test the contribution of neuropil to the ROI signal and fitting a regression

"""

% matplotlib inline

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv
import tkinter
from tkinter import filedialog
import statsmodels.api as sm

root = tkinter.Tk()

root.update()
sigfile = filedialog.askopenfilenames(title='Please pick .sig file')[0]
root.update()
if not sigfile:
    raise NameError('Please select .sig file.')

gcamp_raw = np.genfromtxt( sigfile, delimiter=',' )
print(sigfile)

# ROI_gcamp = gcamp_raw[:, (np.size(gcamp_raw, 1) / 3) * 2:np.size(gcamp_raw, 1)]
# PIL_gcamp = gcamp_raw[:, np.size(gcamp_raw, 1) / 3:(np.size(gcamp_raw, 1) / 3) * 2]

PIL_gcamp = gcamp_raw[:, int(np.size(gcamp_raw, 1) / 2):int(np.size(gcamp_raw, 1))]
ROI_gcamp = gcamp_raw[:, (int(np.size(gcamp_raw, 1) / np.size(gcamp_raw, 1))-1):int(np.size(gcamp_raw, 1) / 2)]

PIL_gcamp = PIL_gcamp[:,0:1]
ROI_gcamp = ROI_gcamp[:,0:1]


for i in range(np.size(ROI_gcamp,1)):
    print(i)
# for i in range(1):
    roi1 = ROI_gcamp[:,i]
    npil1 = PIL_gcamp[:,i]


    fig = plt.figure(figsize=(16,16))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    #(a_s, b_s, r, tt, stderr) = stats.linregress(roi1,npil1)
    ax1.scatter(npil1,roi1)
    npil1_exog = sm.add_constant(npil1)

    ax2.plot(roi1)
    ax2.plot(npil1)

    olsres = sm.OLS(roi1,npil1_exog,M=sm.robust.norms.TukeyBiweight()).fit()
    rlmres = sm.RLM(roi1,npil1_exog,M=sm.robust.norms.RamsayE()).fit()
    rlmresAW = sm.RLM(roi1,npil1_exog,M=sm.robust.norms.AndrewWave()).fit()
    rlmresHT = sm.RLM(roi1,npil1_exog,M=sm.robust.norms.HuberT()).fit()
    rlmresHM = sm.RLM(roi1,npil1_exog,M=sm.robust.norms.Hampel()).fit()
    rlmresBI = sm.RLM(roi1,npil1_exog,M=sm.robust.norms.TukeyBiweight()).fit()
    rlmresTM = sm.RLM(roi1,npil1_exog,M=sm.robust.norms.TrimmedMean()).fit()

    # ax1.plot(roi1[:,1],roi1[:,1]*a_s+b_s,c='g',label='OLS')
    # ax1.plot(npil1,olsres.fittedvalues,c='g',ls='--',label='OLS')
    ax1.plot(npil1,rlmres.fittedvalues,label='RamsayE')
    ax1.plot(npil1,rlmresAW.fittedvalues,label='AndrewWave')
    ax1.plot(npil1,rlmresHT.fittedvalues,label='HuberT')
    ax1.plot(npil1,rlmresHM.fittedvalues,label='Hampel')
    ax1.plot(npil1,rlmresBI.fittedvalues,label='TukeyBiweight')
    ax1.plot(npil1,rlmresTM.fittedvalues,label='TrimmedMean')
    ax1.legend(loc="best")

    print(rlmresBI.fittedvalues)
    print(rlmresBI.fittedvalues.shape)

    ax1.set_xlabel('neuropil values')
    ax1.set_ylabel('roi values')

    # window size in seconds
    win = 60
    # for unidirectional imaging
    fs = 15.5
    # calculate the number of samples in the window (dependent on fs)
    win_samples = win * fs
    # which percentile of samples is used to calculate baseline
    baseline_pctl = 20
    # vector that will hold the result
    dF_F = np.zeros(np.size(roi1))
    npil_dF_F = np.zeros(np.size(npil1))

    # calculate dF using sliding window
    for j,ft in enumerate(roi1):
        # calculate f0, if-clauses are for boundary conditions at beginning and end of trace
        if j-int(win_samples/2) < 0:
            f0 = np.percentile(roi1[0:j+int(win_samples/2)],baseline_pctl)
            f0_npil = np.percentile(npil1[0:j+int(win_samples/2)],baseline_pctl)
        elif j+int(win_samples/2) > roi1.size:
            f0 = np.percentile(roi1[j-int(win_samples/2):roi1.size-1],baseline_pctl)
            f0_npil = np.percentile(npil1[j-int(win_samples/2):npil1.size-1],baseline_pctl)
        else:
            f0 = np.percentile(roi1[j-int(win_samples/2):j+int(win_samples/2)],baseline_pctl)
            f0_npil = np.percentile(npil1[j-int(win_samples/2):j+int(win_samples/2)],baseline_pctl)

        # calculate difference between dF and f0(t)
        dF_F[j] = (roi1[j] - f0)/f0
        npil_dF_F[j] = (npil1[j] - f0_npil)/f0_npil

    # max subtraction modulator is 1
    if rlmresTM.params[1] > 1:
        npil_mod = 1
        title_str = 'neuropil fit capped at 1, original fit: ' + str(rlmresTM.params[1])
    else:
        npil_mod = rlmresTM.params[1]
        title_str = 'neuropil fit: ' + str(rlmresTM.params[1])

    ax3.plot(dF_F)
    ax3.plot(dF_F - (npil_dF_F*npil_mod))
    ax3.plot(npil_dF_F)
    ax3.set_title(title_str)

    ax4.plot(dF_F)
    ax4.plot(dF_F - npil_dF_F)
    ax4.plot(npil_dF_F)
    ax4.set_title('neuropil fit: 1x')

    # ax3.plot(dF_F - (npil_dF_F*rlmres.params[1]))
    # ax3.plot(dF_F - npil_dF_F)

    ax3.set_xlim([4000,8000])
    ax4.set_xlim([4000,8000])
