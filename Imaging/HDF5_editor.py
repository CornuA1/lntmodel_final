#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:18:08 2017

@author: Nicolas Meirhaeghe
"""

import numpy as np
import h5py
import tkinter as tk
import os
import csv, sys
sys.path.append("/Users/Nico/Cours/MIT/Research/Rotations/Harnett_Lab/Coding/in_vivo/MTH1/analysis") 
sys.path.append("/Users/Nico/Cours/MIT/Research/Rotations/Harnett_Lab/Coding/in_vivo/General/Imaging") 
from import_dF import import_dF
from dF_win import dF_win
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import askquestion 
from tkinter.messagebox import showerror
from align_dF_interp import align_dF
from itertools import islice

## datadir, this is for convenience
#datdir = '/Users/Nico/Cours/MIT/Research/Rotations/Harnett_Lab/MTH3_NM/imaging/'
#
root = Tk()

root.withdraw()

h5filename = filedialog.asksaveasfilename(title = 'Select an existing hdf5 file or Create a new one', defaultextension = '.h5')

from PyQt5.QtWidgets import (QFileDialog, QAbstractItemView, QListView,
                             QTreeView, QApplication, QDialog)

class getExistingDirectories(QFileDialog):
    def __init__(self, *args):
        super(getExistingDirectories, self).__init__(*args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.Directory)
        self.setOption(self.ShowDirsOnly, True)
        self.findChildren(QListView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.findChildren(QTreeView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)

qapp = QApplication(sys.argv)
dlg = getExistingDirectories()
if dlg.exec_() == QDialog.Accepted:
    allDir = dlg.selectedFiles()

for i in range(0, len(allDir)):
    
    # holds the trials-list from the config file
    trial_list = []
    # store the maximum number of characters used to specify trials. Required to store in HDF5 later
    trial_def_len = -1
    # length of script ID
    len_scrid = -1
    # number of rows that contain script IDs at the beginning of each datafile
    SCRID_ROWS = 0
    
    #dirname = filedialog.askdirectory();
    dirname = allDir[i]
    
    #if len(os.listdir(dirname)) != 3:
    #    sys.exit("There should only be 3 files in the target folder")
    
    for file in os.listdir(dirname):
        if file.endswith(".txt"):
            configfile= file
        elif file.endswith(".csv"):
            behavfile= file
        elif file.endswith(".sig"):
            imagfile= file
    
    #filelist = os.listdir(dirname)
    #configfile = filelist[1];
    
    config_data = csv.reader(open((dirname + '/' + configfile), 'rt'), delimiter=';', quoting=csv.QUOTE_NONE)
    conf_info = next(islice(config_data, 0,1))
    exp = conf_info[1]
    explen = conf_info[3]
    day = conf_info[5]
    group = conf_info[7]                  
    mouse = conf_info[9]	
    dob = conf_info[11]
    strain = conf_info[13]	
    stop_thresh = conf_info[15]
    valve_open = conf_info[17]
    comments = conf_info[19]

    print('processing: Mouse ', str(mouse), ' Day ', str(day))
    
    # read the list of trial specifications and convert to strings
    for trials in config_data:
    	trial_str = ' '.join(trials) 
    	trial_list.append(np.string_(trial_str))
    	# store length of longest trial specification line
    	if( len(trial_str) >  trial_def_len):
    		trial_def_len = len(trial_str)
      
      
    # open datafile. Store the filepointer separately as we need it 
    	datfile = open((dirname + '/' + behavfile), 'r')
    	raw_data = csv.reader(open(dirname + '/' + behavfile), delimiter=';', quoting=csv.QUOTE_NONE)
    
    	# count number of rows in raw-datafile. Unfortunately there is no better way to do this.
    	row_count = sum(1 for row in raw_data) - SCRID_ROWS
    	
    	# set filepointer to the beginning of the file
    	datfile.seek(0)
    	datfile.close()
    	datfile = open((dirname + '/' + behavfile), 'r')
    	raw_data = csv.reader(open(dirname + '/' + behavfile), delimiter=';', quoting=csv.QUOTE_NONE)
    	
    	# read the rows containing script IDs first
    	for row in islice(raw_data, 0, SCRID_ROWS):
    		scrid_str = ' '.join(row) 
    		scrid.append(np.string_(scrid_str))
    		# store length of longest script ID line
    		if( len(scrid_str) >  len_scrid):
    			len_scrid = len(scrid_str)
    	
    	# make ndarray and then write from the list into this array. This is necessary to store it in HDF5 format
    	datcont = np.ndarray(shape = (row_count,8), dtype = float)
    	
    	# offsets added if a raw datafile had been concatenated as well as how many lines have been skipped
    	time_offset = 0
    	trial_offset = 0
    	skiplines = 0
    	
    	# running index to loop through raw datafile
    	i = 0
    	for line in raw_data:	# there is probably a much more efficient way to do this, but for some one-time parsing it will do
    		# skip the first second of every datafile (even if there has been a re-set within the file)
    		# ass readings from the rotary sensor can be erratic in the initial period
    		datcont[i-skiplines][0] = float(line[0]) + time_offset
    		datcont[i-skiplines][1] = line[1]
    		datcont[i-skiplines][2] = line[2]
    		datcont[i-skiplines][3] = line[3]
    		datcont[i-skiplines][4] = line[4]
    		datcont[i-skiplines][5] = line[5]
    		datcont[i-skiplines][6] = float(line[6]) + trial_offset
    		datcont[i-skiplines][7] = line[7]
    			
    		# a datafile reset is being detected by checking if a timestamp is
    		# smaller than the preceeding one. If yes, add the last timestamp to
    		# the offset
    		if datcont[i - skiplines][0] < datcont[i - skiplines - 1][0] and i > 0:
    			print(datcont[i - skiplines][0], datcont[i - skiplines - 1][0])
    			time_offset += datcont[i - skiplines - 1][0]	# take the last time
    			trial_offset += datcont[i - skiplines - 1][6]
    			datcont[i-skiplines][0] = float(line[i][0]) + time_offset
    			datcont[i-skiplines][6] = float(line[i][6]) + trial_offset
    		
    		i += 1
    		
    	# this copy is necessary so that the new array owns the data (lengthy numpy issue), otherwise we can't resize it
    	tarr = np.copy(datcont)	
    	# remove skipped lines at the end (which are just zeros)
    	tarr.resize((row_count - skiplines, 8))	
      
    
    h5file = h5py.File(h5filename, 'a')	
    
    daygrp = h5file.require_group(day)
    daygrp.attrs['Session_length'] = explen.encode('utf8')
    daygrp.attrs['Day'] = day.encode('utf8')
    daygrp.attrs['Mouse'] = mouse.encode('utf8')
    daygrp.attrs['Group'] = group.encode('utf8')
    daygrp.attrs['DOB'] = dob.encode('utf8')
    daygrp.attrs['Strain'] = strain.encode('utf8')
    daygrp.attrs['Stop_Threshold'] = stop_thresh.encode('utf8')
    daygrp.attrs['Valve_Open_Time'] = valve_open.encode('utf8')
    daygrp.attrs['Comments'] = comments.encode('utf8')
    
    #check if dataset exists, if yes: ask if user wants to overwrite. If no, create it
    dt_t = 'S'+str(trial_def_len)
    dt_s = 'S'+str(len_scrid)
    
    gcamp_data = np.genfromtxt( (dirname + '/' + imagfile), delimiter=',' )
    
    try:	
    	# retrive dataset (just check for first one)
    	test_dset = daygrp['behav_raw' ]
    	# if we want to overwrite: delete old dataset and then re-create with new data
    	del daygrp['behav_raw']	
    	dset = daygrp.create_dataset('behav_raw' , data = tarr, compression = 'gzip')
    	del daygrp ['trial_list']
    	trials_list = daygrp.create_dataset("trial_list", data = trial_list, dtype=dt_t)
    	del daygrp['gcamp_raw']	
    	gcamp_raw = daygrp.create_dataset('gcamp_raw', data = import_dF(imagfile), compression = 'gzip')
    	
    except KeyError:
    	dset = daygrp.create_dataset('behav_raw' , data = tarr, compression = 'gzip')
    	trials_list = daygrp.create_dataset("trial_list", data = trial_list, dtype=dt_t)
    	gcamp_raw = daygrp.create_dataset('gcamp_raw', data = gcamp_data, compression = 'gzip')
    
    ROI_gcamp = gcamp_data[:,(np.size(gcamp_data,1)/3)*2:np.size(gcamp_data,1)]
    PIL_gcamp = gcamp_data[:,np.size(gcamp_data,1)/3:(np.size(gcamp_data,1)/3)*2]
    dF_sig_dF = dF_win( ROI_gcamp )
    dF_pil_dF = dF_win( PIL_gcamp )
                
    dF_signal = dF_sig_dF - dF_pil_dF
    
    dF_aligned, behaviour_aligned = align_dF( dset, dF_signal, -1, [-1,-1], True, False, [-1,-1]) 
    write_h5( h5dat, d, m, dF_aligned,'/dF_win')
    write_h5( h5dat, d, m, behaviour_aligned,'/behaviour_aligned')
    
    try: 
    	# if we want to overwrite: delete old dataset and then re-create with new data
    	del daygrp['dF_win']	
    	df_win = daygrp.create_dataset('dF_win' , data = dF_aligned, compression = 'gzip')
    	del daygrp ['behaviour_aligned']
    	behav_aligned = daygrp.create_dataset('behaviour_aligned', data = behaviour_aligned, compression = 'gzip')
   
    except KeyError:
    	df_win = daygrp.create_dataset('dF_win' , data = dF_aligned, compression = 'gzip')
    	behav_aligned = daygrp.create_dataset('behaviour_aligned', data = behaviour_aligned, compression = 'gzip')
     
 
h5file.close()