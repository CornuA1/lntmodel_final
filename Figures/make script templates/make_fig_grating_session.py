"""
Plot traces for individual gratings reward for individual ROIs

@author: lukasfischer


"""

#%load_ext autoreload
#%autoreload

# %% init block

import sys
import os

# load local settings file
sys.path.append('..' + os.sep + 'Analysis')
sys.path.append('..' + os.sep + 'Figures')

os.chdir('..' + os.sep + '..' + os.sep + 'Analysis')

from fig_grating_session_sfn2017 import fig_grating_session
from yaml_mouselist import yaml_mouselist
from filter_trials import filter_trials
from matplotlib import pyplot as plt
from ruamel import yaml
import warnings
import numpy
import h5py
import numpy as np

warnings.filterwarnings('ignore')

# this file contains machine-specific info
try:
    with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
        content = yaml.load(yaml_file)
except OSError:
    print('        Could not read local settings .yaml file.')

# %% LF170110_2

#MOUSE = 'LF170110_2'
#SESSION = 'Day20170331_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#fig_grating_session(h5path, SESSION, 59, MOUSE+SESSION+'59', [0,1.5], 'svg', 'LF170110_2_gratings')

#for i in range(126):
#    fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [-1,4], 'png', 'LF170110_2_gratings')

# %% LF170222_1

#MOUSE = 'LF170222_1'
#SESSION = 'Day20170612_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'

#fig_roi_gratings(h5path, SESSION, 23, MOUSE+SESSION+'23', [-1,4], 'png')
#fig_grating_session(h5path, SESSION, 23, MOUSE+SESSION+'23', [-1,2], 'svg', 'LF170222_1_gratings')
#for i in range(54):
#    fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_gratings')

# %% LF170222_1

#MOUSE = 'LF170222_1'
#SESSION = 'Day20170615_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#fig_grating_session(h5path, SESSION, 23, MOUSE+SESSION+'23', [], 'svg', 'LF170222_1_gratings')
#fig_roi_gratings(h5path, SESSION, 0, MOUSE+SESSION+'0', [-1,4], 'png')
#for i in range(107):
#    fig_roi_gratings(h5path, SESSION, i, MOUSE+SESSION+str(i), [-1,4], 'png')

# %% LF170612_1

#MOUSE = 'LF170612_1'
#SESSION = 'Day20170719_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#fig_grating_session(h5path, SESSION, 16, MOUSE+SESSION+'16', [], 'png', MOUSE+'_gratings')
#for i in range(85):
#    fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_gratings')

#MOUSE = 'LF170613_1'
#SESSION = 'Day201784_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# for i in range(129):
#     fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_'+SESSION+'_gratings')

#SESSION = 'Day201783_gratings'
# for i in range(126):
#     fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_'+SESSION+'_gratings')


# %% LF170525_1

#MOUSE = 'LF170525_1'
#SESSION = 'Day20170801_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#fig_roi_gratings(h5path, SESSION, 0, MOUSE+SESSION+'0', [-1,4], 'png')
#for i in range(215):
#    fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_gratings')

# %% LF170829_1
#MOUSE = 'LF170829_1'
#SESSION = 'Day2017919_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#fig_grating_session(h5path, SESSION, 2, MOUSE+SESSION+str(2), [], 'png', MOUSE+'_gratings')
#for i in range(42):
#    fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_gratings')

# %% LF170801_1
#MOUSE = 'LF170808_1'
#SESSION = 'Day2017919_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# r,rand_r, z = fig_grating_session(h5path, SESSION, 47, MOUSE+SESSION+str(47), [], 'png', MOUSE+SESSION)
#for i in range(177):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)

# fig = plt.figure(figsize=(4,4))
# ax1 = plt.subplot(111)
#plt.hist(rand_r)
# print(rand_r)
#ax1.axvline(r,c='r')
#ax1.set_xlim([-0.5,1])
# plt.show

#li = np.zeros((177, 2))
#z = np.zeros((177, 2))
     
#for i in range(177): #178
#    temp_li, temp_randli, temp_z = fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+'roi'+str(i), [0,1], 'png', MOUSE+SESSION)
#    
#    if np.size(temp_li) > 1:
#        if ~any(np.isnan(temp_li)):
#            li[i, :] = temp_li
#            z[i, :] = temp_z

#fig = plt.figure(figsize=(4,4))
#ax1 = plt.subplot(111)
#plt.hist(z, bins=40)
#fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_Z_res' + '.csv'
#np.savetxt(fname, z, delimiter=';')
#fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_res' + '.csv'
#np.savetxt(fname, li, delimiter=';')
#fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_Z_dist' + '.' + 'svg'
#fig.savefig(fname, format='svg')

# %% LF170420_1
#MOUSE = 'LF170420_1'
#SESSION = 'Day20170719_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#fig_grating_session(h5path, SESSION, 2, MOUSE+SESSION+str(2), [], 'png', MOUSE+'_gratings')
# for i in range(109):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_'+SESSION)
#
# SESSION = 'Day20170804_gratings'
# for i in range(108):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_'+SESSION)

# %%

#MOUSE = 'LF170214_1'
#SESSION = 'Day201777_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# fig_grating_session(h5path, SESSION, 24, MOUSE+SESSION+str(24), [], 'svg', MOUSE+SESSION)
# fig_grating_session(h5path, SESSION, 32, MOUSE+SESSION+str(32), [], 'svg', MOUSE+SESSION)
# fig_grating_session(h5path, SESSION, 77, MOUSE+SESSION+str(77), [], 'svg', MOUSE+SESSION)
#for i in range(111):
#	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)

#SESSION = 'Day2017714_gratings'
# for i in range(164):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)

# %%

#MOUSE = 'LF170829_1'
#SESSION = 'Day2017919_gratings_1'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# for i in range(140):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)
#
# SESSION = 'Day2017919_gratings_2'
# for i in range(156):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)
# fig_grating_session(h5path, SESSION, 18, MOUSE+SESSION+str(18), [], 'svg', MOUSE+SESSION)

MOUSE = 'LF170829_1'
SESSION = 'Day2017919_gratings_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# r,rand_r, z = fig_grating_session(h5path, SESSION, 47, MOUSE+SESSION+str(47), [], 'png', MOUSE+SESSION)
#for i in range(177):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)

# fig = plt.figure(figsize=(4,4))
# ax1 = plt.subplot(111)
#plt.hist(rand_r)
# print(rand_r)
#ax1.axvline(r,c='r')
#ax1.set_xlim([-0.5,1])
# plt.show

li = np.zeros((140, 2))
z = np.zeros((140, 2))
     
for i in range(140): #178
    temp_li, temp_randli, temp_z = fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+'roi'+str(i), [0,1], 'png', MOUSE+SESSION)
    
    if np.size(temp_li) > 1:
        if ~any(np.isnan(temp_li)):
            li[i, :] = temp_li
            z[i, :] = temp_z

fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot(111)
plt.hist(z, bins=40)
fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_Z_res' + '.csv'
np.savetxt(fname, z, delimiter=';')
fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_res' + '.csv'
np.savetxt(fname, li, delimiter=';')
fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_Z_dist' + '.' + 'svg'
fig.savefig(fname, format='svg')
SESSION = 'Day20170919_gratings_1'
h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
# r,rand_r, z = fig_grating_session(h5path, SESSION, 47, MOUSE+SESSION+str(47), [], 'png', MOUSE+SESSION)
#for i in range(177):
# 	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)

# fig = plt.figure(figsize=(4,4))
# ax1 = plt.subplot(111)
#plt.hist(rand_r)
# print(rand_r)
#ax1.axvline(r,c='r')
#ax1.set_xlim([-0.5,1])
# plt.show

SESSION = 'Day2017919_gratings_2'

li = np.zeros((156, 2))
z = np.zeros((156, 2))
     
for i in range(156): #178
    temp_li, temp_randli, temp_z = fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+'roi'+str(i), [0,1], 'png', MOUSE+SESSION)
    
    if np.size(temp_li) > 1:
        if ~any(np.isnan(temp_li)):
            li[i, :] = temp_li
            z[i, :] = temp_z

fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot(111)
plt.hist(z, bins=40)
fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_Z_res' + '.csv'
np.savetxt(fname, z, delimiter=';')
fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_res' + '.csv'
np.savetxt(fname, li, delimiter=';')
fname = content['figure_output_path'] + MOUSE+SESSION + os.sep + 'OSI_Z_dist' + '.' + 'svg'
fig.savefig(fname, format='svg')

# %%

#MOUSE = 'LF170421_2'
#SESSION = 'Day20170719_gratings'
#h5path = content['imaging_dir'] + MOUSE + '/' + MOUSE + '.h5'
#fig_grating_session(h5path, SESSION, 2, MOUSE+SESSION+str(2), [], 'png', MOUSE+'_gratings')
#for i in range(95):
#	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+'_gratings')
#fig_grating_session(h5path, SESSION, 2, MOUSE+SESSION+str(2), [0,2.5], 'svg', MOUSE+'_'+SESSION)
#fig_grating_session(h5path, SESSION, 15, MOUSE+SESSION+str(15), [0,2.5], 'svg', MOUSE+'_'+SESSION)
#fig_grating_session(h5path, SESSION, 22, MOUSE+SESSION+str(22), [0,2.5], 'svg', MOUSE+'_'+SESSION)
#fig_grating_session(h5path, SESSION, 48, MOUSE+SESSION+str(48), [0,2.5], 'svg', MOUSE+'_'+SESSION)

#SESSION = 'Day20170720_gratings'
#for i in range(59):
#	fig_grating_session(h5path, SESSION, i, MOUSE+SESSION+str(i), [], 'png', MOUSE+SESSION)
