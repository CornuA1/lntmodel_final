"""
Plot licking task-score (from shuffled distribution) for STAGE 5 of the
muscimol2 experiment

this ONLY works with blocks of 5 short/long trial structure, NOT randomized

"""

%load_ext autoreload
%autoreload
%matplotlib inline

import numpy as np
import warnings
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


def fig_lumafluor_red_beads(fname):
    ldat = np.loadtxt(fname, delimiter=',', skiprows=1)
    return np.mean(ldat[:,1])

if __name__ == "__main__":
    datapath = '/Users/lukasfischer/Work/exps/MTH3/beads linescan/'

    mean_dat_green = np.zeros((10,))
    mean_dat_red = np.zeros((10,))
    for i in range(1,11):
        fname = '{}{}{:02d}{}{:02d}{}'.format(datapath, 'LineScan-12072017-1258-0',i,'/LineScan-12072017-1258-0',i,'_Cycle00001_LineProfileData.csv')
        ldat = np.loadtxt(fname, delimiter=',', skiprows=1)
        mean_dat_green[i-1] = np.mean(ldat[:,1])
        mean_dat_red[i-1] = np.mean(ldat[:,3])

    print(mean_dat)
    fig = plt.figure(figsize=(4,4))
    ax1 = plt.subplot2grid((1,1),(0,0),colspan=2)
    ax1.plot(mean_dat_red,lw=2,c='r')
    ax1.plot(mean_dat_green,lw=2,c='g')
    ax1.set_xticklabels(['790','800','810','820','830','840','850','860','870','880'], rotation=45)
    ax1.set_xlabel('wavelength (nm)')
    ax1.set_ylabel('fluoresence (A.U.)')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(datapath + 'result_curve.png', format='png')
