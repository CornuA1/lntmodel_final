""" sandbox for trying things """

import sys, yaml, os
with open('..' + os.sep + 'loc_settings.yaml', 'r') as f:
    loc_info = yaml.load(f)
sys.path.append(loc_info['base_dir'] + "Analysis")

import numpy as np
import matplotlib.pyplot as plt
from load_behavior_data import load_data
from licks import licks_nopost as licks
from rewards import rewards

#TRACK_SHORT = 3
#TRACK_LONG = 4
#
#raw_filename = loc_info['raw_dir'] + 'LF170613_1' + os.sep + '20170804' + os.sep + 'MTH3_vr1_20170804_1708.csv'
#
#print('start loading...')
#
#raw_data = load_data(raw_filename, 'vr')
#all_licks = licks(raw_data)
#trial_licks = all_licks[np.in1d(all_licks[:, 3], [TRACK_SHORT, TRACK_LONG]), :]
#rewards =  rewards(raw_data)
#
#plt.figure()
#plt.plot([1,2,3,4])
#
#print('done')

x = np.arange(-10,10,0.1)


plt.figure()
plt.plot(x,1/(1+np.exp(-x)))
#plt.show()