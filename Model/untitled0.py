#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:36:14 2021

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
    
fname = "total analysis"