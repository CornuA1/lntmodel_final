# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:24 2021

@author: Keith
"""
import sys, yaml, os
#os.chdir('C:/Users/Keith/Documents/GitHub/LNT')
## with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
##     loc_info = yaml.load(f)
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT'+ os.sep + "Analysis")
#sys.path.append('C:/Users/Keith/Documents/GitHub/LNT'+ os.sep + "Imaging")

os.chdir(r'C:\Users\Lou\Documents\repos\LNT')
sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Analysis")
sys.path.append(r'C:\Users\Lou\Documents\repos\LNT'+ os.sep + "Imaging")
sys.path.append(r'C:\Users\Lou\Documents\repos\OASIS-master')

import warnings; warnings.simplefilter('ignore')

from t_score_calculation import run_tscores, run_all_mice, runAnalysisAllMiceAllSessions

def run_LF191022_1():
    mouse = 'LF191022_1'
    sessionNaive = ['20191115','MTH3_vr1_s5r_20191115_2225.csv']
    sessionExpert = ['20191209','MTH3_vr1_s5r2_2019129_1735.csv']
    run_tscores(mouse,sessionNaive,sessionExpert)
    
def run_LF191022_2():
    mouse = 'LF191022_2'
    sessionNaive = ['20191116','MTH3_vr1_s5r_20191116_1815.csv']
    sessionExpert = ['20191210','MTH3_vr1_s5r2_20191210_2335.csv']
    run_tscores(mouse,sessionNaive,sessionExpert)
    
def run_LF191022_3():
    mouse = 'LF191022_3'
    sessionNaive = ['20191113','MTH3_vr1_s5r_20191113_1618.csv']
    sessionExpert = ['20191207','MTH3_vr1_s5r2_2019128_111.csv']
    run_tscores(mouse,sessionNaive,sessionExpert)
    
def run_LF191023_blank():
    mouse = 'LF191023_blank'
    sessionNaive = ['20191114','MTH3_vr1_s5r_20191114_225.csv']
    sessionExpert = ['20191210','MTH3_vr1_s5r2_20191210_214.csv']
    run_tscores(mouse,sessionNaive,sessionExpert)
    
def run_LF191023_blue():
    mouse = 'LF191023_blue'
    sessionNaive = ['20191119','MTH3_vr1_s5r_20191119_1857.csv']
    sessionExpert = ['20191208','MTH3_vr1_s5r2_2019128_2141.csv']
    run_tscores(mouse,sessionNaive,sessionExpert)
    
def run_LF191024_1():
    mouse = 'LF191024_1'
    sessionNaive = ['20191114','MTH3_vr1_s5r_20191114_1756.csv']
    sessionExpert = ['20191210','MTH3_vr1_s5r2_20191210_1844.csv']
    run_tscores(mouse,sessionNaive,sessionExpert)
    
    
if __name__ == '__main__':
    
#    run_LF191022_1()
#    run_LF191022_2()
#    run_LF191022_3()
#    run_LF191023_blank()
#    run_LF191023_blue()
#    run_LF191024_1()
#    
    runAnalysisAllMiceAllSessions()    