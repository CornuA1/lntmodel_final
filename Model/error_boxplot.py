#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 01:29:17 2021

@author: lukasfischer
"""

import csv, os, yaml
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
sns.set_style("white")

# load yaml file with local filepaths
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

fname_sim = "dsim"
fname_bothsoma = "both_soma"
fname_nospatial = "no spatial"

file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + "error_boxplot" + os.sep + fname_sim + ".csv"
file_path_bothsoma = loc_info["raw_dir"] + "figure_sample_data" + os.sep + "error_boxplot" + os.sep + fname_bothsoma + ".csv"
file_path_nospatial = loc_info["raw_dir"] + "figure_sample_data" + os.sep + "error_boxplot" + os.sep + fname_nospatial + ".csv"

sim_error = np.zeros((0,1))
bothsoma_error = np.zeros((0,1))
nospatial_error = np.zeros((0,1))

with open(file_path) as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    headers = next(datareader)
       
    for row in datareader:
        # x = np.array((next(datareader)))
        sim_error = np.vstack((sim_error,row[2]))
sim_error = sim_error.astype('float64')
        
with open(file_path_bothsoma) as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    headers = next(datareader)
       
    for row in datareader:
        # x = np.array((next(datareader)))
        bothsoma_error = np.vstack((bothsoma_error,row[2]))   
bothsoma_error = bothsoma_error.astype('float64')  
        
with open(file_path_nospatial) as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    headers = next(datareader)
       
    for row in datareader:
        # x = np.array((next(datareader)))
        nospatial_error = np.vstack((nospatial_error,row[2]))  
     
nospatial_error = nospatial_error.astype('float64')  
     
fig = plt.figure(figsize=(5,5))
ax = fig.subplots()  
ax.errorbar([0,1,2], [np.mean(sim_error), np.mean(bothsoma_error), np.mean(nospatial_error)], 
            yerr=[np.std(sim_error)/np.sqrt(sim_error.shape[0]), np.std(bothsoma_error)/np.sqrt(bothsoma_error.shape[0]), np.std(nospatial_error)/np.sqrt(nospatial_error.shape[0])])

ax.set_ylim([0,12])