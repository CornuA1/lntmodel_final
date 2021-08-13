# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 22:06:23 2020

@author: Lou
"""

import sys, yaml, os
#os.chdir('C:/Users/Keith/Documents/GitHub/LNT')
## with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
##     loc_info = yaml.load(f)
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT'+ os.sep + "Analysis")
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT'+ os.sep + "Imaging")
#sys.path.append('C:/Users/Keith/Documents/GitHub/OASIS-master')

os.chdir(r'C:\Users\Lou\Documents\repos\LNT')
sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Analysis")
sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Imaging")
sys.path.append(r'C:\Users\Lou\Documents\repos\OASIS-master')

import warnings; warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy import stats
from align_dF_interp_mpi import align_dF
from load_behavior_data import load_data
from oasis.functions import deconvolve
from sklearn import linear_model

def process_and_align_caiman(data_path, sess, dff_file, behavior_file):
    eye_data = None
    session_crop=[0,1]
    
    print('Loading df/f data...')
    df_signal_path = data_path + os.sep + dff_file
    dF_signal_bef =  h5py.File(df_signal_path, 'r+')
    
    try:
        del dF_signal_bef['dF_aligned']
    except KeyError:
        pass
    try:
        del dF_signal_bef['behaviour_aligned']
    except KeyError:
        pass
    
    dF_signal = np.copy(dF_signal_bef['F_dff'])
    frame_brightness = np.zeros(len(dF_signal))
    
    print('Loading behavior data for alignment...')
    fname = data_path + os.sep + behavior_file
    raw_data = np.genfromtxt(fname, delimiter=';')
	
    dF_aligned, behaviour_aligned, bri_aligned = align_dF(raw_data, dF_signal, frame_brightness,[-1, -3], [2, -4], True, True, session_crop, eye_data)
    
    # del dF_signal_bef['dF_aligned']
    dF_signal_bef.create_dataset('dF_aligned', data=dF_aligned)
    # del dF_signal_bef['behaviour_aligned']
    dF_signal_bef.create_dataset('behaviour_aligned', data=behaviour_aligned)
    dF_signal_bef.close()
                                        
    print('done...')

    return dF_aligned, behaviour_aligned


def factor_out_baseline(data_path, sess, dff_file):

    print('Loading df/f data...')
    df_signal_path = data_path + os.sep + dff_file
    dF_signal_bef =  h5py.File(df_signal_path, 'r+')
    try:
        del dF_signal_bef['new_dF']
    except KeyError:
        pass
    try:
        del dF_signal_bef['calcium_dF']
    except KeyError:
        pass
    dF_signal = np.copy(dF_signal_bef['dF_aligned'])
    new_dF = np.zeros((len(dF_signal[:,0]),len(dF_signal[0,:])))
    calcium_dF = np.zeros((len(dF_signal[:,0]),len(dF_signal[0,:])))
    print('Calculate baselines.')
    for num in range(len(dF_signal[0,:])):
        test_sig = dF_signal[:,num]
        rep_sig = np.zeros(len(test_sig))
        ker_leg = 1000
        x_up_lim = len(test_sig) - ker_leg
        for x in range(len(test_sig)):
            if x < ker_leg:
                cur_vals = test_sig[:x + ker_leg]
            elif x > x_up_lim:
                cur_vals = test_sig[x-ker_leg:]
            else:
                cur_vals = test_sig[x-ker_leg:x+ker_leg]
            sort_val = np.sort(cur_vals)
            bottom_20 = int(len(cur_vals)*0.2)
            cur_val = np.sum(sort_val[:bottom_20])
            rep_sig[x] = cur_val/bottom_20
        current_dF = test_sig-rep_sig

        xx = np.fft.fft(current_dF)
        xx[0] = 0
        yy = np.fft.ifft(xx)
        final_dF = yy.astype('float64')
        new_dF[:,num] = final_dF
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(np.arange(len(current_dF)).reshape(-1, 1), current_dF.reshape(-1, 1))
        prediction_ransac = ransac.predict(np.arange(len(current_dF)).reshape(-1, 1))
        calcium_dF[:,num] = current_dF-prediction_ransac[int(len(current_dF)/2),0]

    dF_signal_bef.create_dataset('new_dF', data=new_dF)
    dF_signal_bef.create_dataset('calcium_dF', data=calcium_dF)
    dF_signal_bef.close()

    print('done...')

    return None


def calculate_detrend_and_deconvolve(data_path, sess, dff_file):

    print('Loading df/f data...')
    df_signal_path = data_path + os.sep + dff_file
    dF_signal_bef =  h5py.File(df_signal_path, 'r+')
    dF_signal = np.copy(dF_signal_bef['new_dF'])
    try:
        del dF_signal_bef['deconv']
    except KeyError:
        pass
    try:
        del dF_signal_bef['spikes']
    except KeyError:
        pass
    deconv = np.zeros((len(dF_signal[:,0]),len(dF_signal[0,:])))
    spikes = np.zeros((len(dF_signal[:,0]),len(dF_signal[0,:])))    

    print('Calculate detrends and convolutions.')
    for num in range(len(dF_signal[0,:])):
        current_dF = dF_signal[:,num]
        c, s, b, g, lam = deconvolve(current_dF, penalty=0)
        deconv[:,num] = c
        spikes[:,num] = s

    dF_signal_bef.create_dataset('deconv', data=deconv)
    dF_signal_bef.create_dataset('spikes', data=spikes)
    dF_signal_bef.close()
    print('done...')

    return None

    
def run_test():
    MOUSE= 'LF191022_1'
    sess = '20191209'
    dff_file = 'M01_000_000_results.mat'
    data_path = 'Q:\Documents\Harnett UROP' + os.sep + MOUSE + os.sep + sess
    factor_out_baseline(data_path, sess, dff_file)

if __name__ == '__main__':
    run_test()







        # ransac = linear_model.RANSACRegressor()
        # ransac.fit(np.arange(len(current_dF)).reshape(-1, 1), current_dF.reshape(-1, 1))
        # offset_predict = ransac.predict(np.arange(len(current_dF)).reshape(-1, 1))




