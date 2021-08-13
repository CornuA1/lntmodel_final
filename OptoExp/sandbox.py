"""
Sandbox for opto related analysis, or really just anything. Also not really a sandbox.

"""
%load_ext autoreload
%autoreload
%matplotlib inline

import numpy as np
import csv
import sys
sys.path.append("/Users/lukasfischer/github/in_vivo/MTH3/Analysis")
from rewards import rewards_legacy, rewards

raw_datafilename = '/Users/lukasfischer/Work/exps/MTH3/opto/vr2/raw/MTH3_vr2_opto_201883_1747.csv'

datfile = open(raw_datafilename, 'r')
raw_data = np.genfromtxt(raw_datafilename, delimiter=';')
datfile.close()


# raw = np.copy(f['Day' + str(day) + folder_suffix + '_' + osuff + '/raw_data'])
rews = rewards_legacy(raw_data)
rews2 = rewards2(raw_data)
print(np.shape(rews))
print(np.shape(rews2))

print('done')
