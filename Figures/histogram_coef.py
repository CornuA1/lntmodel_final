# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:38:02 2020

@author: Lou
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 02:45:25 2020

@author: Lou
"""

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import warnings, sys, os, yaml
# warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
os.chdir('C:/Users/Lou/Documents/repos/LNT')
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import scipy as sp
import scipy.io as sio


with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

sys.path.append(loc_info['base_dir'] + '/Analysis')

sns.set_style('white')

fformat = 'png'


def run_LF191022_1_20191114_s():
    MOUSE = 'LF191022_1'
    SESSION = '20191114'
    use_data = 'M01_000_002_new_histo.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Short Track Coefficients: 22_1, 1114')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191022_1_20191114_l():
    MOUSE = 'LF191022_1'
    SESSION = '20191114'
    use_data = 'M01_000_002_new_histo.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Long Track Coefficients: 22_1, 1114')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191022_1_20191209_s():
    MOUSE = 'LF191022_1'
    SESSION = '20191209'
    use_data = 'M01_000_000_new_histo.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Short Track Coefficients: 22_1, 1209')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191022_1_20191209_l():
    MOUSE = 'LF191022_1'
    SESSION = '20191209'
    use_data = 'M01_000_000_new_histo.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Long Track Coefficients: 22_1, 1209')   
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191022_3_20191113_s():
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = 'M01_000_000_new_histo.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Short Track Coefficients: 22_3, 1113')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191022_3_20191113_l():
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = 'M01_000_000_new_histo.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Long Track Coefficients: 22_3, 1113')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191022_3_20191207_s():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = 'M01_000_002_new_histo.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Short Track Coefficients: 22_3, 1207')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191022_3_20191207_l():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = 'M01_000_002_new_histo.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Long Track Coefficients: 22_3, 1207')    
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191023_blue_20191113_s():
    MOUSE = 'LF191023_blue'
    SESSION = '20191113'
    use_data = 'M01_000_003_new_histo.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Short Track Coefficients: 22_blue, 1113')   
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191023_blue_20191113_l():
    MOUSE = 'LF191023_blue'
    SESSION = '20191113'
    use_data = 'M01_000_003_new_histo.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Long Track Coefficients: 22_blue, 1113')   
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191023_blue_20191208_s():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = 'M01_000_002_new_histo.mat'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Short Track Coefficients: 22_blue, 1208')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191023_blue_20191208_l():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = 'M01_000_002_new_histo.mat'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Long Track Coefficients: 22_blue, 1208')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    

def run_naive_short_coef_data():
    leng = 'short'
    short_coef = []
    
    MOUSE = 'LF191022_1'
    SESSION = '20191115'
    use_data = 'M01_000_004_new_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = 'M01_000_000_new_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = short_coef + loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191023_blue'
    SESSION = '20191113'
    use_data = 'M01_000_003_new_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = short_coef + loaded_data['save_'+leng][0]
    
    plt.clf()
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Naive Short Track Coefficients')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + 'naive_short_coef_data_hist.png')
    
def run_naive_long_coef_data():
    leng = 'long'
    long_coef = []
    
    MOUSE = 'LF191022_1'
    SESSION = '20191114'
    use_data = 'M01_000_002_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    long_coef = loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = 'M01_000_000_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    long_coef = long_coef + loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191023_blue'
    SESSION = '20191113'
    use_data = 'M01_000_003_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    long_coef = long_coef + loaded_data['save_'+leng][0]
    
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),long_coef,width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Naive Long Track Coefficients')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + 'naive_long_coef_data_hist.png')
    
def run_expert_short_coef_data():
    leng = 'short'
    short_coef = []
    
    MOUSE = 'LF191022_1'
    SESSION = '20191209'
    use_data = 'M01_000_000_new_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = 'M01_000_002_new_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = short_coef + loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = 'M01_000_002_new_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = short_coef + loaded_data['save_'+leng][0]
    
    plt.clf()
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Short Track Coefficients')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + 'expert_short_coef_data_hist.png')
    
def run_expert_long_coef_data():
    leng = 'long'
    long_coef = []
    
    MOUSE = 'LF191022_1'
    SESSION = '20191209'
    use_data = 'M01_000_000_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    long_coef = loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = 'M01_000_002_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    long_coef = long_coef + loaded_data['save_'+leng][0]
    
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = 'M01_000_002_histo.mat'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    long_coef = long_coef + loaded_data['save_'+leng][0]
    
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),long_coef,width=.95,color=ccc)
    ax.set_xlabel('Top 3 Coefficients per Neuron')
    ax.set_ylabel('Histogram Count')
    ax.set_title('Histogram of Expert Long Track Coefficients')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + 'expert_long_coef_data_hist.png')

def run_LF191022_3_20191113_s_28():
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = '10'
    leng = 'short'

    plt.clf()
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_xlabel('Coefficients')
    ax.set_title('Coefficients for ' +MOUSE + ' :: ' + SESSION + ' :: ' + use_data + ' :: ' + leng + ' track')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+use_data+'_'+leng+'_alpha_bar.png')
    
def run_LF191022_3_20191113_l_28():
    MOUSE = 'LF191022_3'
    SESSION = '20191113'
    use_data = '10'
    leng = 'long'

    plt.clf()
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm_poisson_c\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_xlabel('Coefficients')
    ax.set_title('Coefficients for ' +MOUSE + ' :: ' + SESSION + ' :: ' + use_data + ' :: ' + leng + ' track')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+use_data+'_'+leng+'_bar.png')
    
def run_LF191022_3_20191207_s_31():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = '18'
    leng = 'short'

    plt.clf()
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_xlabel('Coefficients')
    ax.set_title('Coefficients for ' +MOUSE + ' :: ' + SESSION + ' :: ' + use_data + ' :: ' + leng + ' track')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+use_data+'_'+leng+'_alpha_bar.png')
    
def run_LF191022_3_20191207_l_31():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = '18'
    leng = 'long'

    plt.clf()
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm_poisson_c\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_xlabel('Coefficients')
    ax.set_title('Coefficients for ' +MOUSE + ' :: ' + SESSION + ' :: ' + use_data + ' :: ' + leng + ' track')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+use_data+'_'+leng+'_bar.png')
    
def run_LF191022_3_20191207_s_18():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = '18'
    leng = 'short'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm_poisson\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_title('Short Trials')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+use_data+'_'+leng+'_old_bar.png')
    
def run_LF191022_3_20191207_l_18():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = '18'
    leng = 'long'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm_poisson\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_title('Long Trials')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+use_data+'_'+leng+'_old_bar.png')
    
def run_LF191022_1_20191209_roi():
    MOUSE = 'LF191022_1'
    SESSION = '20191209'
    use_data = '85'
    leng = 'short'
    
    plt.clf()
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm_poisson_c\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_xlabel('Coefficients')
    ax.set_title('Coefficients for ' +MOUSE + ' :: ' + SESSION + ' :: ' + use_data + ' :: ' + leng + ' track')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+use_data+'_'+leng+'_bar.png')
    
def run_LF191022_3_20191207_roi():
    MOUSE = 'LF191022_3'
    SESSION = '20191207'
    use_data = '10'
    leng = 'short'
    
    plt.clf()
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm_poisson_c\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_xlabel('Coefficients')
    ax.set_title('Coefficients for ' +MOUSE + ' :: ' + SESSION + ' :: ' + use_data + ' :: ' + leng + ' track')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+use_data+'_'+leng+'_bar.png')
    
def run_LF191023_blue_20191208_roi():
    MOUSE = 'LF191023_blue'
    SESSION = '20191208'
    use_data = '10'
    leng = 'short'
    
    plt.clf()
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + '\\data_glm_poisson_c\\' + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    fig, ax = plt.subplots()
    ax1 = ax.bar(np.arange(28),loaded_data['coef_'+leng].T[0][1:],width=.95,color=ccc)
    ax.set_xlabel('Coefficients')
    ax.set_title('Coefficients for ' +MOUSE + ' :: ' + SESSION + ' :: ' + use_data + ' :: ' + leng + ' track')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+use_data+'_'+leng+'_bar.png')
    
def run_LF191022_1_cellreg_histo():
    MOUSE = 'LF191022_1'
    use_data = 'LF191022_1_coef_matched_histo'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = loaded_data['save_'+leng][0]
    
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of'+lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
def run_LF191022_3_cellreg_histo():
    MOUSE = 'LF191022_3'
    use_data = 'LF191022_3_coef_matched_histo'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    short_coef = loaded_data['save_'+leng][0]
    
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
def run_LF191023_blue_cellreg_histo():
    MOUSE = 'LF191023_blue'
    use_data = 'LF191023_blue_coef_matched_23_x_histo'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    
    leng = 'long'
    lenlevel = ' Layer 23 Long Day 1 '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'long_x'
    lenlevel = ' Layer 23 Long Day 2 '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'long_x3'
    lenlevel = ' Layer 23 Long Day 3 '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'short'
    lenlevel = ' Layer 23 Short Day 1 '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'short_x'
    lenlevel = ' Layer 23 Short Day 2 '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'short_x3'
    lenlevel = ' Layer 23 Short Day 3 '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
def run_layer_5():
    MOUSE = 'LF191023_blue'
    SESSION = '20191215'
    use_data = 'M01_000_003_new_histo.mat'

    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + SESSION + os.sep + use_data
    loaded_data = sio.loadmat(processed_data_path)
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),loaded_data['save_'+leng][0],width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + MOUSE+'_'+SESSION+'_'+leng+'_hist.png')
    
def run_LF191023_blue_20191208_8_reg_ol():
    MOUSE = 'LF191023_blue'
    use_data = 'LF191023_blue_coef_matched_23_04_x_histo'
    processed_data_path = loc_info['raw_dir'] + MOUSE + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    
    leng = 'short'
    lenlevel = ' Layer 23 04 Short Regular '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'long'
    lenlevel = ' Layer 23 04 Long Regular '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'short_x'
    lenlevel = ' Layer 23 04 Short OL '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
    leng = 'long_x'
    lenlevel = ' Layer 23 04 Long OL '
    short_coef = loaded_data['save_'+leng][0]
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+'Track Coefficients in CellReg')
    fig.savefig('C:\\Users\\Lou\\Documents\\check-up\\' + use_data + '_' + leng + '.png')
    
if __name__ == '__main__':

    ccc = ['r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','b','b','b','b','b','b','b','b','b','b',]
#    leng = 'long'
#    lenlevel = ' Layer 2/3 Long Expert '
#    run_layer_5()
#    lenlevel = ' Layer 5 Short '
#    run_LF191022_1_cellreg_histo()
#    run_LF191022_3_cellreg_histo()
    run_LF191023_blue_20191208_8_reg_ol()
#    run_LF191022_3_cellreg_histo()
#    plt.clf()
#    run_LF191022_1_20191114_s()
#    plt.clf()
#    run_LF191022_1_20191114_l()
#    plt.clf()
#    run_LF191022_1_20191209_s()
#    plt.clf()
#    run_LF191022_1_20191209_l()
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
#    plt.clf()
#    run_naive_short_coef_data()
#    plt.clf()
#    run_naive_long_coef_data()
#    plt.clf()
#    run_expert_short_coef_data()
#    plt.clf()
#    run_expert_long_coef_data()
#    plt.clf()
#    run_LF191022_3_20191113_s_28()
#    run_LF191022_3_20191113_l_28()
#    run_LF191022_3_20191207_s_31()
#    run_LF191022_3_20191207_l_31()
#    plt.clf()
#    run_LF191022_3_20191207_s_18()
#    plt.clf()
#    run_LF191022_3_20191207_l_18()
    