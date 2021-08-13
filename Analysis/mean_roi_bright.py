# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:10:16 2019

@author: Lou
"""

import sys, yaml, os
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + "Analysis")
sys.path.append(loc_info['base_dir'] + "Imaging")
import warnings; warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean as mean

def gather_mean_intensity(data_path, sess, sigfile):
    sig_filename = data_path + os.sep + sess + os.sep + sigfile
    raw_sig_mat = np.genfromtxt( sig_filename, delimiter=',' )
    mean_intensity_list = []
    for i in range(len(raw_sig_mat)):
        for ii in range(len(raw_sig_mat[0])):
            mean_intensity_list.append(raw_sig_mat[i][ii])
    return mean(mean_intensity_list)

def fig_mean_intensity(data_path, sess_list, sigfile_list, MOUSE, tufts_trunks, subfolder=[], fname='bright'):
    mean_val_list = []
    fig = plt.figure()
    for sess in range(len(sess_list)):
        mean_val_list.append(gather_mean_intensity(data_path, sess_list[sess], sigfile_list[sess]))
    int_val_gra = plt.subplot2grid((1,1),(0,0))
    int_val_gra.plot(sess_list, mean_val_list)
    int_val_gra.set_title(MOUSE + ': ' + tufts_trunks)
    int_val_gra.set_ylabel('Mean Brightness')
    int_val_gra.set_xlabel('Session')
    plt.tight_layout()
    print('done...')
    
    if subfolder != []:
        if not os.path.isdir(loc_info['figure_output_path'] + subfolder):
            os.mkdir(loc_info['figure_output_path'] + subfolder)
        fname = loc_info['figure_output_path'] + subfolder + os.sep + fname + '.' + fformat
    else:
        fname = loc_info['figure_output_path'] + fname + '.' + fformat
    print(fname)
    fig.savefig(fname, format=fformat,dpi=150)
    
def run_Jimmy():
    MOUSE = 'Jimmy'
    data_path = loc_info['raw_dir'] + MOUSE
    sess_list = ['190507_3','190508_8','190514_10','190627_0','190709_0','190715_1']
    sigfile_list = ['M01_000_003_0120.sig','M01_000_008_0120.sig','M01_000_010_0120.sig','M01_000_000_0086.sig','M01_000_000_0086.sig','M01_000_001_0080.sig']
    fig_mean_intensity(data_path, sess_list, sigfile_list, MOUSE, 'Tufts')
    sigfile_list = ['M01_000_003_0000.sig','M01_000_008_0000.sig','M01_000_010_0000.sig','M01_000_000_0000.sig','M01_000_000_0000.sig','M01_000_001_0000.sig']
    fig_mean_intensity(data_path, sess_list, sigfile_list, MOUSE, 'Trunks')

def run_Pumba():
    MOUSE = 'Pumba'
    data_path = loc_info['raw_dir'] + MOUSE
    sess_list = ['190503_2','190508_9','190514_11','190627_3','190709_2','190715_3']
    sigfile_list = ['M01_000_002_0120.sig','M01_000_009_0120.sig','M01_000_011_0120.sig','M01_000_003_0080.sig','M01_000_002_0080.sig','M01_000_003_0080.sig']
    fig_mean_intensity(data_path, sess_list, sigfile_list, MOUSE, 'Tufts')
    sigfile_list = ['M01_000_002_0000.sig','M01_000_009_0000.sig','M01_000_011_0000.sig','M01_000_003_0000.sig','M01_000_002_0000.sig','M01_000_003_0000.sig']    
    fig_mean_intensity(data_path, sess_list, sigfile_list, MOUSE, 'Trunks')

def run_Buddha():
    MOUSE = 'Buddha'
    data_path = loc_info['raw_dir'] + MOUSE
    sess_list = ['190802_16','190802_17','190802_18','190802_19','190802_20']
    sigfile_list = ['Buddha_000_016_rigid.sig','Buddha_000_017_rigid.sig','Buddha_000_018_rigid.sig','Buddha_000_019_rigid.sig','Buddha_000_020_rigid.sig']
    fig_mean_intensity(data_path, sess_list, sigfile_list, MOUSE, 'Spines')

if __name__ == '__main__':

    fformat = 'png'

#    run_Jimmy()
#    run_Pumba()
    run_Buddha()
