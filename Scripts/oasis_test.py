import numpy as np
from scipy.signal import butter, filtfilt, lfilter
import matplotlib.pyplot as plt, mpld3
from sys import path
from os import sep
import os
import h5py
import yaml
from skimage.filters import threshold_otsu
path.append('..')

with open('.' + sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters, foopsi
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter_causal(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == '__main__':
    %load_ext autoreload
    %autoreload
    %matplotlib inline

    MOUSE = 'LF170613_1'
    SESSION = 'Day20170804'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    MOUSE = 'LF170421_2'
    SESSION = 'Day20170719'
    h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    # MOUSE = 'LF171211_1'
    # SESSION = 'Day2018321_2'
    # h5path = loc_info['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

    h5dat = h5py.File(h5path, 'r')
    behav_ds = np.copy(h5dat[SESSION + '/behaviour_aligned'])
    dF_ds = np.copy(h5dat[SESSION + '/dF_original'])
    h5dat.close()

    y = dF_ds[:,2]
    # y = roi_raw[:,0] - pil_raw[:,0]

    # use very simple deconvolution
    order = 6
    fs = int(np.size(behav_ds,0)/behav_ds[-1,0])       # sample rate, Hz
    cutoff = 4 # desired cutoff frequency of the filter, Hz
    roi_filtered = butter_lowpass_filter(dF_ds[:,0], cutoff, fs, order)
    m = np.mean(roi_filtered,axis=0)
    m02 = np.mean(np.power(roi_filtered,2),axis=0)
    m12 = np.mean(np.multiply(roi_filtered[1:-2],roi_filtered[0:-3]),axis=0)
    a = ((m**2)-m12)/((m**2)-m02)                   # alpha
    uhat = [roi_filtered[1:-1] - a*roi_filtered[0:-2]]
    uhat = np.insert(uhat,0,0)                      # spike train
    shat = np.where(uhat > threshold_otsu(uhat))    # estimated spike train


    true_b = 2
    # y, true_c, true_s = map(np.squeeze, gen_data(N=1, b=true_b, seed=0))
    c, s, b, g, lam = deconvolve(y, penalty=0, optimize_g=5)
    c,s = oasisAR1(y-b, g, s_min=0.4)

    cutoff = 10
    s_filtered = butter_lowpass_filter_causal(s, cutoff, fs, order)

    c_foopsi,s_foopsi = foopsi(y-b, [g])

    # fig = plt.figure(figsize=(20,6))
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True, figsize=(20,6))
    # ax1 = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313)
    #
    plot_simple = np.zeros((len(y)))
    plot_simple[shat[0]] = 1
    ax4.plot(plot_simple)
    # for l in shat[0]:
    #     ax3.axvline(l)

    ax1.plot(b+c, lw=2, label='denoised')
    ax1.plot(y, label='data', zorder=-12, c='y')
    ax2.plot(s, lw=2, label='deconvolved', c='g')
    # ax2.plot(s_filtered, lw=2, label='deconvolved', c='r')
    ax3.plot(s_foopsi, lw=2, label='deconvolved', c='g')
    ax1.legend(ncol=3, frameon=False, loc=(.02,.85))
    ax2.set_ylim(0,1.3)
    ax2.legend(ncol=3, frameon=False, loc=(.02,.85));
    simpleaxis(plt.gca())

    mpld3.show()
    # plt.show()

    # ax1.set_xlim([5000,7000])
    # ax2.set_xlim([5000,7000])
    # ax3.set_xlim([5000,7000])


    # plot_trace()
