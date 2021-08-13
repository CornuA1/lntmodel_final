# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:32:25 2019

@author: Lou
"""

import sys, os, yaml
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f, Loader=yaml.FullLoader)
sys.path.append(loc_info['base_dir'] + '/Analysis')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as scistat
import seaborn as sns
from PIL import Image
sns.set_style('white')


def draw_image_roi(rois, tif):
    list_of_val = []
    dict_of_val = {}
    dict_of_out = []
    list_of_pix_val = []
    for row in rois:
        for val in row:
            if val not in list_of_val and val != 0:
                list_of_val.append(val)
                dict_of_val[str(val)] = []
    for val in list_of_val:
        for row in range(len(rois)):
            for pos in range(len(rois[row])):
                if rois[row][pos] == val:
                    dict_of_val[str(val)].append([pos,row])
                    list_of_pix_val.append([pos,row])
    for val in list_of_val:
        lid = dict_of_val[str(val)]
        for pos in lid:
            x_val = pos[0]
            y_val = pos[1]
            if [x_val-1,y_val] not in lid or [x_val+1,y_val] not in lid or [x_val,y_val-1] not in lid or [x_val,y_val+1] not in lid:
                dict_of_out.append(pos)
    tif_load = tif.load()
    rev_tif = Image.new(tif.mode,tif.size)
    rev_load = rev_tif.load()
    for i in range(rev_tif.size[0]):
        for j in range(rev_tif.size[1]):
            if [i,j] in dict_of_out:
                rev_load[i,j] = 255
            else:
                rev_load[i,j] = tif_load[i,j]
    rev_tif.show()
    
def run_Buddha():
    MOUSE = 'Buddha'
    sess = '190816_21'
    processed_data_path = loc_info['raw_dir'] + '\\' + MOUSE + os.sep + sess + os.sep
    roi_data_path = processed_data_path + 'Buddha_000_021_rigid.rois'
    tif_pic_path = processed_data_path + 'Buddha_000_021_rigid_max.tif'
    rois = sio.loadmat(roi_data_path)['roiMask']
    tif = Image.open(tif_pic_path)
    draw_image_roi(rois, tif)

if __name__ == '__main__':
    run_Buddha()