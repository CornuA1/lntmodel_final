"""
import_dF
Import dF/F traces from imaging analysis. The input filetype must be in .csv .
Each row must represent one ROI and each column one datapoint as sampled at
40 Hz.

Parameters
-------
dF_file : String
          Path including filename to .csv file containing dF/F traces. If 
          invalid, and open file dialog will be opened.
          

Outputs
-------
dF_mat : ndarray
     raw dF/F traces. 
"""

import numpy as np
from tkinter import filedialog

def import_dF( filename='' ):
    try:
        dF_mat = np.genfromtxt( filename, delimiter=',' )

    # if illegal filename or no argument was provided, just ask for hdf5 file
    except IOError:	
        dF_mat = np.genfromtxt( filedialog.askopenfilename(title = 'Please select datafile'), delimiter=',' )        
        
    return dF_mat