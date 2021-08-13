# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 02:45:25 2020

@author: Lou
"""

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import warnings, sys, os, yaml
# warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
# os.chdir('C:/Users/Lou/Documents/repos/LNT')
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import scipy as sp
import scipy.io as sio


with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

sns.set_style('white')

fformat = 'png'


def run_LF191022_1_20191114_s():
    MOUSE = 'LF191022_1'
    SESSION = '20191114'
    use_data = 'M01_000_002_heatmaps.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191022_1_20191114_l():
    MOUSE = 'LF191022_1'
    SESSION = '20191114'
    use_data = 'M01_000_002_heatmaps.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191022_1_20191209_s():
    MOUSE = 'LF191022_1'
    SESSION = '20191209'
    use_data = 'M01_000_000_heatmaps.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191022_1_20191209_l():
    MOUSE = 'LF191022_1'
    SESSION = '20191209'
    use_data = 'M01_000_000_heatmaps.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191022_3_20191113_s():
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = 'M01_000_000_heatmaps.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191022_3_20191113_l():
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = 'M01_000_000_heatmaps.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191022_3_20191207_s():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = 'M01_000_002_heatmaps.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191022_3_20191207_l():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = 'M01_000_002_heatmaps.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191023_blue_20191113_s():
    MOUSE = 'LF191023_blue'
    SESSION = '20191113'
    use_data = 'M01_000_003_heatmaps.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191023_blue_20191113_l():
    MOUSE = 'LF191023_blue'
    SESSION = '20191113'
    use_data = 'M01_000_003_heatmaps.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191023_blue_20191208_s():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = 'M01_000_002_heatmaps.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
def run_LF191023_blue_20191208_l():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = 'M01_000_002_heatmaps.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    ax1 = sns.heatmap(loaded_data['coef_'+leng])
    ax1.figure.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    

def run_LF191022_1_cellreg_heat():
    MOUSE = 'LF191022_1'
    use_data = 'LF191022_1_coef_cellreg_heatmap'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)['coef_matrix_'+leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx)
    axx.set_xlabel('Coefficients')
    axx.set_ylabel('Neurons')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'.png')
    
def run_LF191022_3_cellreg_heat():
    MOUSE = 'LF191022_3'
    use_data = 'LF191022_3_coef_cellreg_heatmap'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)['coef_matrix_'+leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx)
    axx.set_xlabel('Coefficients')
    axx.set_ylabel('Neurons')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'.png')

def run_LF191023_blue_cellreg_heat():
    MOUSE = 'LF191023_blue'
    use_data = 'LF191023_blue_coef_cellreg_heatmap'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)['coef_matrix_'+leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx)
    axx.set_xlabel('Coefficients')
    axx.set_ylabel('Neurons')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'.png')
    
def run_LF191022_1_coef():
    MOUSE = 'LF191022_1'
    use_data = 'LF191022_1_layer_2_3_'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)[leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx, vmin=0,vmax=1)
    axx.set_xlabel('Days of Recording')
    axx.set_ylabel('Days of Recording')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'_2_3_coef.png')
    
def run_LF191022_3_coef():
    MOUSE = 'LF191022_3'
    use_data = 'LF191022_3_layer_2_3_'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)[leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx, vmin=0,vmax=1)
    axx.set_xlabel('Days of Recording')
    axx.set_ylabel('Days of Recording')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'_2_3_coef.png')

def run_LF191023_blue_coef():
    MOUSE = 'LF191023_blue'
    use_data = 'LF191023_blue_layer_2_3_'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)[leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx, vmin=0,vmax=1)
    axx.set_xlabel('Days of Recording')
    axx.set_ylabel('Days of Recording')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'_2_3_coef.png')
    
def run_LF191022_1_coef_naive():
    MOUSE = 'LF191022_1'
    use_data = 'LF191022_1naive'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)[leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx, vmin=0,vmax=1)
    axx.set_xlabel('Days of Recording')
    axx.set_ylabel('Days of Recording')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'_5_coef.png')
    
def run_LF191022_3_coef_naive():
    MOUSE = 'LF191022_3'
    use_data = 'LF191022_3naive'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)[leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx, vmin=0,vmax=1)
    axx.set_xlabel('Days of Recording')
    axx.set_ylabel('Days of Recording')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'_5_coef.png')

def run_LF191023_blue_coef_naive():
    MOUSE = 'LF191023_blue'
    use_data = 'LF191023_bluenaive'
#    leng = 'long_x'
#    lenlevel = ' Expert Long '

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)[leng].T
    
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data, ax=axx, vmin=0,vmax=1)
    axx.set_xlabel('Days of Recording')
    axx.set_ylabel('Days of Recording')
    axx.set_title('Heatmap of ' +MOUSE+ lenlevel + 'Track Neurons from Cellreg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\'+ use_data +'_'+leng+'_5_coef.png')
    
def run_a_lot_of_files():
    MOUSE = 'LF191023_blue'
    SESSION = '20191217'
    use_data = 'M01_000_006_heatmaps.mat'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)

    leng = 'short'
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data['coef_'+leng], ax=axx)
    axx.set_xlabel('Coefficients')
    axx.set_ylabel('Neurons')
    axx.set_title('Heatmap of ' + MOUSE + leng + 'Track Neurons from ' + SESSION)
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
    leng = 'long'
    plt.clf()
    fig, axx = plt.subplots()
    ax1 = sns.heatmap(loaded_data['coef_'+leng], ax=axx)
    axx.set_xlabel('Coefficients')
    axx.set_ylabel('Neurons')
    axx.set_title('Heatmap of ' + MOUSE + leng + 'Track Neurons from ' + SESSION)
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'.png')
    
if __name__ == '__main__':

#    leng = 'long'
#    lenlevel = ' Layer 2/3 Naive Long '
    # run_a_lot_of_files()
    run_LF191022_1_coef_naive()
#    run_LF191022_3_coef_naive()
#    run_LF191023_blue_coef_naive()
#    plt.clf()
#    run_LF191022_1_20191114_s()
#    plt.clf()
#    run_LF191022_1_20191114_l()
#    plt.clf()
    # run_LF191022_1_20191209_s()
#    plt.clf()
    # run_LF191022_1_20191209_l()
#    plt.clf()
#    run_LF191022_3_20191113_s()
#    plt.clf()
#    run_LF191022_3_20191113_l()
#    plt.clf()
#    run_LF191022_3_20191207_s()
#    plt.clf()
#    run_LF191022_3_20191207_l()
#    plt.clf()
#    run_LF191023_blue_20191113_s()
#    plt.clf()
#    run_LF191023_blue_20191113_l()
#    plt.clf()
#    run_LF191023_blue_20191208_s()
#    plt.clf()
#    run_LF191023_blue_20191208_l()
    
    
    