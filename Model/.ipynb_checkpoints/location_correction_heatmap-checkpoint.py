#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:14:04 2021

@author: lukasfischer
"""

import csv, os, yaml
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# load yaml file with local filepaths
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)

# file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + "location_correction_heatmap.csv"

# plot_data = np.zeros((0,3))

# print(file_path)
# with open(file_path) as csvfile:
#     datareader = csv.reader(csvfile, delimiter=',')
#     headers = next(datareader)
       
#     for row in datareader:
#         # x = np.array((next(datareader)))
#         plot_data = np.vstack((plot_data,row))
        
# print(plot_data.shape)

# np.save(file_path, plot_data)

file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + "location_correction_heatmap.npy"
plot_data = np.load(file_path)
print(plot_data.shape)
plt.plot(plot_data[:,1])
