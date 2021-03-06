"""
manually align imaging and behavior data

its currently a little clumsy to use. First you check the aligned brightness
signal against the transitions to the black box by setting dF_start and dF_end. Its important to check the beginning and the end
of each recording to make sure there is no drift within the recording.

Once you are happy with the alignment, set the bottom block to True - it will
run the resampling for all ROIs and write it back to the HDF5-file.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import h5py
import sys
import os
import yaml
from scipy.interpolate import griddata
import multiprocessing

sys.path.append("./Analysis")
with open('./loc_settings.yaml', 'r') as f:
            content = yaml.load(f)

from filter_trials import filter_trials
from scipy import stats
from scipy import signal
