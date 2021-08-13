"""
Bin track into the specified number of bins and calculate the average running
speed in each. Exclude datapoints where the running speed is below the 
specified threshold

Parameters
-------
raw :       ndarray 
            raw data
binnr : int
        number of bins
speed_thresh :  int
        Speed values below this threshold will be considered stationary and disregarded
tracklength : 
            ndarray
            total length of the track
gainmod :   float
            specifiy gain modulation parameter. Normally locked at 1
"""

import numpy as np

if __name__ == "__main__":
	"""This script can not be executed stand-alone"""
	print("This script can not be executed stand-alone")

def bincentres( bins ):
	"""
	Return a vector of length len(bins)-1 with the centre values of the bins
	
	Keyword arguments:
	bins -- vector of bins
	
	"""
	
	bincentres = np.zeros( len(bins)-1 )
	for i in range( len( bins-1 )):
		bincentres[i-1] = ( bins[i-1] + bins[i] ) / 2
	return bincentres
	
def speed_v_loc( raw, binnr, speed_thresh, tracklength, gainmod=1 ):
	# Remove episodes where mouse is below the movement threshold
	raw = np.delete(raw, np.where(raw[:,3] < speed_thresh), 0)

	# create empty array to store speed values in
	speed_histo = np.zeros( binnr )	
	
	# make location histogram (number of sample-point in each bin along the track) so we can later calculate the average	
	loc_histo, bins_loc = np.histogram( raw[:, 1], binnr, (0.0, tracklength) )	
	
	# get array with bincentres
	loc_bincentres = bincentres( bins_loc )	
	
	# iterate through every row, add the speed-value to the corresponding location bin and the calculate the average
	for i in range(len(raw[:, 2])):	
		speed_histo[ (np.abs(loc_bincentres - raw[i, 1])).argmin() ] += raw[i,3]
	speed_histo = speed_histo / loc_histo	# divide accumulated speed by number of sample-points in each bin

	return speed_histo	
	
