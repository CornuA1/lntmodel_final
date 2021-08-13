"""
Create a ratemap of either licks or stops (depending on the dataset that is 
passed to the function). Applies a 5-point gaussian filter with stdev=1


Parameters
-------
dset :  ndarray
        dataset containing stop or lick information
binnr : int
        number of bins
mode :  int
        calculate {-1: all, 0: non mode, 1: mode, 2: probe (trialnr%10==0), 3 non-probe} trials
tracklength : int
        total tracklength
dset : int
        gain modulation factor
trialnr_column : int
        column in dataset containing the trial number        
   
Outputs
-------
ratemap_smoothed :  ndarray
                    smoothed ratemap with length=binnr          

"""

import numpy as np
from scipy import ndimage
from scipy import signal


if __name__ == "__main__":
	"""This script can not be executed stand-alone"""
	print("This script can not be executed stand-alone")

def ratemap( dset, binnr, tracklength, trialnr_column=2 ):
    # check its not an empty dataset
    if(dset.shape[0] > 0):
        # create histogram
        stop_histo, bin_edges = np.histogram(dset[:,1], bins=binnr, range=(0,tracklength))
        # divide total dset by total number of trials to get average stopping rate
        stop_rate = (stop_histo/np.unique(dset[:,trialnr_column]).shape[0])/(tracklength/binnr)
        # create a 5-point gaussian window with a standard deviation of 1
        gk = signal.gaussian(5, 1.0)
        
        # convolve gaussian kernel with signal. Window is wrapped around at the edges.
        ratemap_smoothed = ndimage.filters.convolve1d(stop_rate, gk/gk.sum(), mode='wrap')
		
        return ratemap_smoothed
    else:
        return np.zeros((100,))

