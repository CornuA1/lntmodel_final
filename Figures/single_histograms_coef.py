# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 01:21:02 2020

@author: Keith
"""

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import warnings, sys, os, yaml
# warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
# os.chdir('C:/Users/Lou/Documents/repos/LNT')
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import heapq
import scipy.io as sio


# with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
#     loc_info = yaml.load(f)

# sys.path.append(loc_info['base_dir'] + '/Analysis')

sns.set_style('white')

fformat = 'png'

def goldwater_Figs():
    MOUSE = 'LF191022_1'
    leng = 'short'
    try:
        os.mkdir('Q:\\Documents\\Harnett UROP\\pics\\essay_figs\\')
    except:
        pass
    
    use_data = 'coef_19'
    date = '20191115'
    processed_data_path = 'Q:\\Documents\\Harnett UROP\\' + MOUSE + os.sep + date + os.sep + 'histo_files' + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)

    short_coef_1 = loaded_data['coef_'+leng][:,0]
    naive_cf = reversed(heapq.nlargest(4,short_coef_1))
    naive_cf = [x for x in naive_cf]
    # fig.savefig('Q:\\Documents\\Harnett UROP\\pics\\essay_figs\\' + use_data + '_' + leng + '.svg', format="svg")
    
    '''Above is naive, below is expert.'''
    
    use_data = 'coef_4'
    date = '20191209'
    processed_data_path = 'Q:\\Documents\\Harnett UROP\\' + MOUSE + os.sep + date + os.sep + 'histo_files' + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    
    short_coef_2 = loaded_data['coef_'+leng][:,0]
    expert_cf = reversed(heapq.nlargest(4,short_coef_2))
    expert_cf = [x for x in expert_cf]
    plt.clf()
    fig, ax = plt.subplots()
    ax.barh(np.arange(8),expert_cf+naive_cf,color=['r','r','r','r','b','b','r','b'])
    ax.set_title('Naive and Expert')
    fig.savefig('Q:\\Documents\\Harnett UROP\\pics\\essay_figs\\example.png')
    fig.savefig('Q:\\Documents\\Harnett UROP\\pics\\essay_figs\\example.svg', format="svg")


def single_cell_stuff(nnum):
    str_num = str(nnum)
    MOUSE = 'LF191024_1'
    date = '20191210'
    use_data = 'coef_'+str_num
    processed_data_path = 'Q:\\Documents\\Harnett UROP\\' + MOUSE + os.sep + date + os.sep + 'histo_files' + os.sep + use_data + '.mat'
    loaded_data = sio.loadmat(processed_data_path)
    
    try:
        os.mkdir('Q:\\Documents\\Harnett UROP\\pics\\' + MOUSE + '_' + date + '_histos\\')
    except:
        pass
    
    leng = 'short'
    lenlevel = ' Naive Short '
    short_coef = loaded_data['coef_'+leng][:,0]
    r2_short = str(round(loaded_data['r2_'+leng][0,0],2))
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+ use_data +' Track Coef|| ' + r2_short)
    fig.savefig('Q:\\Documents\\Harnett UROP\\pics\\' + MOUSE + '_' + date + '_histos\\' + use_data + '_' + leng + '.png')
    
    leng = 'long'
    lenlevel = ' Naive Long '
    short_coef = loaded_data['coef_'+leng][:,0]
    r2_long = str(round(loaded_data['r2_'+leng][0,0],2))
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(np.arange(28),short_coef,width=.95,color=ccc)
    ax.set_title('Histogram of '+ MOUSE +lenlevel+ use_data +' Track Coef|| ' + r2_long)
    fig.savefig('Q:\\Documents\\Harnett UROP\\pics\\' + MOUSE + '_' + date + '_histos\\' + use_data + '_' + leng + '.png')

    
if __name__ == '__main__':

    ccc = ['r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','b','b','b','b','b','b','b','b','b','b',]
    """
    Naive is below
    """
    # file_int = [1,2,3,4,5,6,9,10,11,13,15,16,17,18,19,21,22,24,25,26,32,33,37,39,42,43,44,46,48,49,53,55,57,59,62,66,68,69,72,76,83,94,96,99]
    # file_int = [1,2,3,5,6,7,8,10,11,13,14,15,16,18,21,23,25,28,29,30,33,35,38,41,45,53,56,58,79,81,100,109,112]
    # file_int = [1,4,7,8,9,10,17,20,22,25,28,30,31,33,40,42,47,50,51,60,61,66,68,71,98]
    # file_int = [1,3,6,9,12,13,16,22,24,26,34,38,48,64,71,76,90,92,96,100,102,105,125]
    # file_int = [1,2,3,4,7,13,19,20,24,35,42,58,68,69,72,73,75,78,87,101]
    # file_int = [1,6,12,18,28,29,56,58]
    """
    Expert is below
    """
    # file_int = [2,3,4,5,6,8,11,12,13,15,16,17,19,20,22,23,24,26,28,30,31,32,33,44,45,46,47,49,51,55,60,63,66,67,73,76,84,89,92,95,96,98,104,109]
    # for x in file_int:
    #     single_cell_stuff(x)
    # file_int = [1,2,3,6,7,9,10,12,13,16,17,18,24,28,32,33,37,39,44,49,52,56,60,64,67,69,72,74,80,81,83,84,87]
    # for x in file_int:
    #     single_cell_stuff(x)
    # file_int = [4,8,13,14,16,17,18,22,31,37,55,63,75,78,87,101,113,115,117,119,123,124,130,136,139]
    # for x in file_int:
    #     single_cell_stuff(x)
    # file_int = [8,19,22,24,27,31,32,35,38,43,52,85,94,108,118,124,125,128,131,134,135,141,142]
    # for x in file_int:
    #     single_cell_stuff(x)
    # file_int = [2,3,4,6,7,9,16,44,59,66,74,77,100,102,105,113,118,119,134,142]
    # for x in file_int:
    #     single_cell_stuff(x)
    # file_int = [7,9,10,14,38,47,67,107]
    # for x in file_int:
    #     single_cell_stuff(x)
    goldwater_Figs()








    