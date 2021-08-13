#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:14:04 2021

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

fname = "location_correction_heatmap"
fname = "data error heatmap bothsoma"
fname = "data error heatmap nospatial"

# file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + fname + ".csv"

# plot_data = np.zeros((0,3))

# print(file_path)
# with open(file_path) as csvfile:
#     datareader = csv.reader(csvfile, delimiter=',')
#     headers = next(datareader)
       
#     for row in datareader:
#         # x = np.array((next(datareader)))
#         plot_data = np.vstack((plot_data,row))
        
# print(plot_data.shape)

file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + fname + ".mat"
plot_data = sp.io.loadmat(file_path)
np.save(file_path, plot_data["datanospat"])
# np.save(file_path, plot_data["databothsoma"])
# np.save(file_path, plot_data["dataerrorheatmap"])

file_path = loc_info["raw_dir"] + "figure_sample_data" + os.sep + fname + ".npy"
plot_data = np.load(file_path, allow_pickle=True).astype('float64')
print(plot_data.shape)
plt.plot(plot_data[:,0], plot_data[:,1])

locs = plot_data[:,1].astype('int32')
t = plot_data[:,0].astype('int32')
error = plot_data[:,1] - plot_data[:,2]
error = error.astype('int32')

min_loc = np.amin(locs)
max_loc = np.amax(locs)
loc_binsize = (max_loc - min_loc) / 200
loc_bins = np.arange(min_loc, max_loc+loc_binsize, loc_binsize)
loc_nbins = np.floor((max_loc - min_loc)/loc_binsize).astype('int32')

t_min = np.amin(t)
t_max = np.amax(t)
t_binsize = (t_max - t_min) / 50
t_bins = np.arange(t_min, t_max+t_binsize, t_binsize)
t_nbins = np.floor((t_max - t_min)/t_binsize).astype('int32')

error_colormap = np.full((loc_nbins, t_nbins), np.nan)
for i in range(t_nbins):
    timepoints = plot_data[plot_data[:,0] > t_bins[i],:]
    timepoints = timepoints[timepoints[:,0] < t_bins[i+1],:]
    for j in range(loc_nbins):
        datapoints = timepoints[timepoints[:,1] > loc_bins[j],:]
        datapoints = datapoints[datapoints[:,1] < loc_bins[j+1],:]
        error_colormap[j,i] = np.abs(np.nanmean(datapoints[:,1] - datapoints[:,2]))

fig = plt.figure(figsize=(8,5))
ax = fig.subplots()  


# error_colormap = error_colormap / np.nanmax(error_colormap)      

ax2 = ax.twinx().twiny()
im = ax.imshow(error_colormap, aspect='auto', origin='lower', cmap='OrRd', zorder=5)
cbar = fig.colorbar(im)
cbar.set_label('error (cm)')
ax.set_yticks([])
ax.set_yticklabels([])

ax2.plot(plot_data[:,0], plot_data[:,1], linewidth=0.5, color='k', zorder=4)
ax2.set_xlim([0,np.amax(plot_data[:,0])])
ax2.set_ylim([np.amin(plot_data[:,1]),np.amax(plot_data[:,1])])





# ax2.set_xticks([])
# ax2.set_yticks([])

sns.despine(top=True, right=True, left=False, bottom=False)

fig.savefig(file_path + "_fig.svg", format='svg', dpi=300)

print(file_path + "_fig.svg saved")