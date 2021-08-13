"""
Retrieve and combine datasets from HDF5 files based on search criteria for the project .yaml file

Parameters
--------
group : list
		which experimental groups from the .yaml file to include

filterprops : list of tuples
		each tuple within the list has the format ['KEYWORD','VALUE']
		example [['TASK','STAGE 5'],['RECTYPE','OPENLOOP']]


Outputs
--------
behav_ds : list of ndarrays
		each element of the list is one retrieved dataset

dF_ds : list of ndarrays
		each element of the list is one retrieved dataset

"""

import h5py
import numpy as np
import yaml
from searchyaml import searchyaml


def load_h5_data(group, filterprops):
	# get settings for local machine
	with open('../loc_settings.yaml', 'r') as f:
			content = yaml.load(f)
	data_dir = content['imaging_dir']
	yaml_file = content['yaml_file']
	# define which sessions to plot
	rec_info = searchyaml(yaml_file, group, filterprops)
	mice = rec_info[0]
	dates = rec_info[1]
	rectypes = rec_info[2]
	tasks = rec_info[3]

	behav_ds = []
	dF_ds = []

	# retrieve datasets
	for i, m in enumerate(mice):
		# construct path to HDF5 file based on assumption of relative path to notebook
		filepath = data_dir + m + '/' + m + '.h5'
		# create path to dataset
		# create path to dataset
		if rectypes[i] == 'REGULAR':
			ds_path = '/Day' + str(dates[i]) + '/'
		elif rectypes[i] == 'OPENLOOP':
			ds_path = '/Day' + str(dates[i]) + '_openloop/'
		elif rectypes[i] == 'DARK':
			raise KeyError('DARK sessions not supported by this function')
		elif rectypes[i] == 'GRATINGS':
			raise KeyError('GRATINGS sessions not supported by this function')

		h5dat = h5py.File(filepath, 'r')
		behav_ds.append(np.copy(h5dat[ds_path + 'behaviour_aligned']))
		dF_ds.append(np.copy(h5dat[ds_path + 'dF_win']))
		h5dat.close()

	return behav_ds, dF_ds, rec_info

def load_h5_dslist(dslist, rectype=''):
	"""
	dslist are mousename, dataset name and ROI list triplets
	e.g.:
	[['LF160901_1','Day20161214',[-1]], ['LF170110_2','Day2017331',[-1]]]

	The third item is a list that contains the number of ROIs to be included. If it
	is -1, all ROIs are included

	rectypes: suffix that will be appended to session name

	"""

	# get settings for local machine
	with open('../loc_settings.yaml', 'r') as f:
			content = yaml.load(f)
	data_dir = content['imaging_dir']
	yaml_file = content['yaml_file']

	behav_ds = []
	dF_ds = []

	# append a _ to the beginning of the rectype (if present), to conform with session naming convention
	if rectype is not '':
		rectype = '_' + rectype

	for ds in dslist:
		# construct path to HDF5 file based on assumption of relative path to notebook
		filepath = data_dir + ds[0] + '/' + ds[0] + '.h5'
		try:
			print(ds[0], ds[1])
			h5dat = h5py.File(filepath, 'r')
			behav_ds.append(np.copy(h5dat[ds[1] + rectype + '/behaviour_aligned']))
			dF_data = np.copy(h5dat[ds[1] + rectype + '/dF_win'])
			if ds[2][0] != -1:
				dF_ds.append(dF_data[:,ds[2]])
			else:
				dF_ds.append(dF_data)
			h5dat.close()
		except KeyError:
			pass


	return behav_ds, dF_ds

def load_recinfo(group, filterprops):
	""" return recording info only, instead of entire dataset """
	# get settings for local machine
	with open('../loc_settings.yaml', 'r') as f:
			content = yaml.load(f)
	data_dir = content['imaging_dir']
	yaml_file = content['yaml_file']
	print(yaml_file)
	# define which sessions to plot
	rec_info = searchyaml(yaml_file, group, filterprops)

	return rec_info
