

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:08:49 2018

""
do PCA for short and long trials based on the mean dF signal of the ROIs
--> reconstruct short and long track (trial-by-trial and mean) based on the linear combination of ROI responses 
""

@author: katya
"""

#plot reconstructed track for mean dF responses for all trials
def PCA_mean_trials(plt, short_dF_temp, long_dF_temp,transformer_long_temp, transformer_short_temp):
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("./Analysis")
    
    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal
    import statsmodels.api as sm

    import seaborn as sns
    sns.set_style('white')
    
    import pandas as pd
    from numpy.linalg import svd
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    from mpl_toolkits.mplot3d import Axes3D
   
    #standardize dF by all ROIs (required for PCA analysis) 
    short_dF_temp = short_dF_temp.T - np.mean(short_dF_temp,1)
    long_dF_temp = long_dF_temp.T - np.mean(long_dF_temp,1)
    
    
    track_start = 100
    bin_size = 5
    
    # define spatial bins for analysis
    track_start = 100
    lm_start = (200-track_start)/bin_size
    lm_end = (240-track_start)/bin_size
    lm_start = int(lm_start)
    lm_end = int(lm_end)
  
   
    reduce_short = transformer_short_temp.transform(short_dF_temp.T)
    reduce_long = transformer_long_temp.transform(long_dF_temp.T)
    
    
    ax = fig.gca(projection='3d')
    lm_start = int(lm_start)
    lm_end = int(lm_end)
   
    #plot reconstructed short track based on reduced ROI responses 
    ax.plot(reduce_short[:,0], reduce_short[:,1], reduce_short[:,2], c='blue' )
    
    ax.scatter(reduce_short[lm_start,0],reduce_short[lm_start,1],reduce_short[lm_start,2],s=100, c = 'blue')
    ax.text(reduce_short[lm_start,0],reduce_short[lm_start,1],reduce_short[lm_start,2],  '1', size=20, zorder=1, color='k' )
    
    ax.scatter(reduce_short[lm_end,0],reduce_short[lm_end,1],reduce_short[lm_end,2],s=100, c = 'blue')
    ax.text(reduce_short[lm_end,0],reduce_short[lm_end,1],reduce_short[lm_end,2],  '2', size=20, zorder=1, color='k')
    
    #plot reconstructed short track based on reduced ROI responses 
    ax.plot(reduce_long[:,0], reduce_long[:,1],reduce_long[:,2], c='orange')
    ax.scatter(reduce_long[lm_start,0],reduce_long[lm_start,1],reduce_long[lm_start,2],s=100, c = 'orange')
    ax.text(reduce_long[lm_start,0],reduce_long[lm_start,1],reduce_long[lm_start,2], '1', size=20, zorder=1, color='k')
    
    ax.scatter(reduce_long[lm_end,0],reduce_long[lm_end,1],reduce_long[lm_end,2],s=100, c = 'orange')
    ax.text(reduce_long[lm_end,0],reduce_long[lm_end,1],reduce_long[lm_end,2], '2', size=20, zorder=1, color='k')

#plot reconstructed track for dF responses of all trials
def PCA_all_trials(ax,all_trials_short,all_trials_long,trials_short_size,trials_long_size, count, transformer_long_temp, transformer_short_temp):
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    sys.path.append("../Analysis")

    import matplotlib.pyplot as plt
    from filter_trials import filter_trials
    from scipy import stats
    from scipy import signal
    import statsmodels.api as sm

    import seaborn as sns
    sns.set_style('white')
    
    import pandas as pd
    from numpy.linalg import svd
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    from mpl_toolkits.mplot3d import Axes3D
   
    # track numbers used in this analysis
    bin_size = 5
    reward_distance = 0
    tracklength_short = 320 + reward_distance
    tracklength_long = 380 + reward_distance
    bin_size = 5
    
    
    
    # define spatial bins for analysis
    track_start = 100
    lm_start = (200-track_start)/bin_size
    lm_end = (240-track_start)/bin_size
    binnr_short = int((tracklength_short-track_start)/bin_size)
    binnr_long = int((tracklength_long-track_start)/bin_size)


    #standardize dF by all ROIs (required for PCA analysis) 
    short_dF_temp = all_trials_short.T - np.mean(all_trials_short,1)
    long_dF_temp = all_trials_long.T - np.mean(all_trials_long,1)
    
    
 
    #filter out bins where there are NaN values
    ind_nans_short = np.where(np.isnan(short_dF_temp.T).any(axis=1))
    ind_nans_long = np.where(np.isnan(long_dF_temp.T).any(axis=1))
    
    short_dF_temp = short_dF_temp.T[~np.isnan(short_dF_temp.T).any(axis=1)]
    long_dF_temp = long_dF_temp.T[~np.isnan(long_dF_temp.T).any(axis=1)]
    
    
    reduce_short = transformer_short_temp.transform(short_dF_temp)
    reduce_long = transformer_long_temp.transform(long_dF_temp)
    
   
   
    lm_start = int(lm_start)
    lm_end = int(lm_end)
    
    count_short = 0 
    ind_short = 0
    count_long = 0
    ind_long = 0
    ax = plt.gca(projection='3d')
    
    #plot reconstructed tracks for each trial for reduced ROI responses
    for i in range(trials_short_size):
         sub_short = np.where((ind_nans_short[0] >= ind_short) & (ind_nans_short[0] < (ind_short+binnr_short)))
         binnr_short_sub = binnr_short - sub_short[0].shape[0]
         short_tp = reduce_short[count_short:count_short+binnr_short_sub] 
         ax.plot(short_tp[:,0], short_tp[:,1], short_tp[:,2], c = 'blue' )
         count_short = count_short + binnr_short_sub
         ind_short = ind_short + binnr_short
    for k in range(trials_long_size):      
        sub_long = np.where((ind_nans_long[0] >= count_long) & (ind_nans_long[0] <= (count_long+binnr_long)))
        binnr_long_sub = binnr_long - sub_long[0].shape[0]
        
        long_tp = reduce_long[count_long:count_long+binnr_long_sub]
            
        ax.plot(long_tp[:,0], long_tp[:,1], long_tp[:,2], c = 'orange' )
        count_long = count_long + binnr_long_sub
        ind_long = ind_long + binnr_long
   
    
    

if __name__ == "__main__":
  
    import json
    import os
    from sklearn.decomposition import PCA
    import sys
    sys.path.append("./Analysis")
    import yaml
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings; warnings.simplefilter('ignore')
    import sys
    
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    sns.set_style('white')
    sys.path.append("./Analysis")
    import os
    with open('../loc_settings.yaml', 'r') as f:
        content = yaml.load(f)
    with open(content['figure_output_path'] + 'all_trials_dF_results' + os.sep + '_alltrials_andmean_dF_results_allfiles_norm_entire_sess.json','r') as f:
        all_trials_results = json.load(f)
    with open(content['figure_output_path']  + 'recordings_with_behav_inc420.json','r') as f:
        recordings = json.load(f)
    
   
    good_recordings = recordings['good_recordings']
    del good_recordings[2]
    fig = plt.figure(figsize=(25,12))
    cmap = plt.cm.get_cmap('viridis')
    count = 1
    
    k = 3 #number of PCs returned
    inds_start = 0
    for i in range(len(good_recordings)):
        r = good_recordings[i]
        mouse = r[0]
        
        #get data for mean trial and all trial responses for each mouse 
        mean_trials_short = np.array(all_trials_results['%s_mean_dF_short' % mouse])
        mean_trials_long = np.array(all_trials_results['%s_mean_dF_long' % mouse])
        all_trials_short = all_trials_results['%s_all_trial_dF_short' % mouse]
        all_trials_long = all_trials_results['%s_all_trial_dF_long' % mouse]
        trials_short_size = all_trials_results['%s_num_trials_short' % mouse]
        trials_long_size = all_trials_results['%s_num_trials_long' % mouse]
         
        plt.subplot(4,2,count, projection = '3d')
        tf_temp = PCA(n_components=k, random_state=0)
        transformer_short_temp = tf_temp.fit(mean_trials_short)
        transformer_long_temp = tf_temp.fit(mean_trials_long)
      
        PCA_all_trials(plt,np.asarray(all_trials_short),np.asarray(all_trials_long),trials_short_size[0],trials_long_size[0], count,transformer_long_temp, transformer_short_temp)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('%s' % mouse)
        
        plt.subplot(4,2,count+1, projection = '3d')
        transformer_short_temp = tf_temp.fit(mean_trials_short)
        transformer_long_temp = tf_temp.fit(mean_trials_long)
        PCA_mean_trials(plt, mean_trials_short, mean_trials_long,transformer_long_temp, transformer_short_temp)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('%s' % mouse)
        count = count + 2
    
    subfolder = 'pca_dtw_figures'
    fformat = 'png'
    ffname = 'pca_analysis_all_trials_withmean'
    
    fname = content['figure_output_path'] + subfolder + os.sep + ffname + '.' + fformat
    try:
        fig.savefig(fname, format=fformat)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
