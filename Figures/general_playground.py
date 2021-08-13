""" Sandbox for testing functions, should be on .gitignore """


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Analysis")
import seaborn as sns
sns.set_style('white')

# load local settings file
import matplotlib
import numpy as np
import warnings; warnings.simplefilter('ignore')
#from filter_trials import filter_trials
from scipy import stats
from scipy import signal
from scipy import math
import statsmodels.api as sm
import yaml
#import h5py
import json

import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


import os
with open('../loc_settings.yaml', 'r') as f:
            content = yaml.load(f)

#Simulation window parameters

tot_sim = 1000

# total time (msec)
t_all = 1000

# instantenous firing rate (Hz)
#r = 10
#rsigt = r/t_all
#
#num_spikes = []
#for j in range(tot_sim):
#    t = np.zeros((t_all))
#    for i in range(len(t)):
#        xi = np.random.uniform(0.0,1.0,1)
#        if xi < rsigt:
#            t[i] = 1
#    num_spikes.append(np.sum(t))
#        
#plt.figure()
#plt.hist(num_spikes)
#plt.show()

frameRate = 10000
duration = 4
tauOn = 0.5
ampFast = 1
tauFast = 0.5
ampSlow = 0
tauSlow = 1

#x1 = np.arange(0,duration+(1/frameRate),1/frameRate)
#    # y = (1-(np.exp(-(x1-spkT) / tauOn))) * (ampFast * np.exp(-(x1-spkT) / tauFast))+(ampSlow * np.exp(-(x1-spkT) / tauSlow))
#y = (1-(np.exp(-(x1) / tauOn))) * (ampFast * np.exp(-(x1) / tauFast))+(ampSlow * np.exp(-(x1) / tauSlow))
#plt.figure()
#plt.plot(x1, y)
#plt.show()

samplingRate = 1000
x1 = np.arange(0,4+(1/samplingRate),1/samplingRate)
e_tauOn = 0.5
e_tauFast = 0.2
e_ampFast = 1
event_kernel = (1-(np.exp(-x1/e_tauOn))) * (e_ampFast * np.exp(-x1/e_tauFast))  
event_kernel = (event_kernel/samplingRate) * 30
event_kernel_AUC = np.trapz(event_kernel,x1)

e_tauOn_slow = 0.5
e_tauFast_slow = 1.5
event_kernel_slow = (1-(np.exp(-x1/e_tauOn_slow))) * (e_ampFast * np.exp(-x1/e_tauFast_slow))  
event_kernel_slow = (event_kernel_slow/samplingRate)

scale_factor = np.arange(0,100,0.1)
auc_diff = np.zeros((scale_factor.shape[0],))
for i,sf in enumerate(scale_factor):
    auc_diff[i] = event_kernel_AUC - (np.trapz(event_kernel_slow,x1)*sf)
    
resolved_scale_factor = scale_factor[np.argmin(np.abs(auc_diff))]
event_kernel_slow = event_kernel_slow * resolved_scale_factor

# put in solver here to change the value above (30) until our AUC is the same as original curve

#event_kernel =(e_ampFast * np.exp(-x1/e_tauFast))  

print(np.trapz(event_kernel,x1))
print(np.trapz(event_kernel_slow,x1))
print(scale_factor[np.argmin(np.abs(auc_diff))])

plt.figure()
plt.plot(x1, event_kernel)
plt.plot(x1, event_kernel_slow)
plt.show()

#noise_vals = np.random.randn(1000)      
#plt.figure()
#plt.hist(noise_vals*10)
#plt.plot(x1, event_kernel_slow)
#plt.show()


            
#p_lambda = 3
#
#k = [0,1,2,3,4,5,6,7,8,9,10]
#p_k = np.zeros((len(k)))
#
#for kn in k:
#    p_k[kn] = np.exp(-p_lambda) * ((p_lambda**kn)/math.factorial(kn))
#
#plt.figure()
#plt.plot(p_k)    
#plt.show()

#tMax = 1
#dt = 0.01
#k = 3
#r = p_lambda/dt
#p_t = np.zeros((int(tMax/dt)))
#
#for i,tn in enumerate(np.arange(0,tMax,dt)):
#    p_t[i] = np.exp(-p_lambda) * ((p_lambda**k)/math.factorial(k))
#
#plt.plot(p_t)
#plt.show()
    
#Point process parameters
#lambda0=10; #intensity (ie mean density) of the Poisson process
# 
##Simulate Poisson point process
#for i in range(10):
#    numbPoints = stats.poisson( lambda0*xDelta ).rvs()#Poisson number of points
#    print(numbPoints)
#xx = xDelta*stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
#Plotting
#plt.plot(np.arange(len(xx)), xx[:])

#plt.show()

#x = [1,1.2,1.4,2.4,4.0]
#y = [1088,998,784,478,278]
#yw = [1462,1384,1090,674,387]
#
#plt.figure(figsize=(4,4))
#ax1 = plt.subplot(111)
#
#plt.plot(x,y)
#plt.plot(x,yw)
#
#print(np.interp(2, x, yw))

# x = np.arange(1,10,1)
#
# x = x / 2
#
# print(np.random.rand(2))

#
# plt.figure(figsize=(4,4))
# ax1 = plt.subplot(111)
#
# fs = 15.5
# t = np.arange(0,10,1/fs) # sec
#
#
#
# plt.plot(10*(np.exp(-2*t)-np.exp(-3*t)))


# x = np.arange(10)
#
# print(x)
# print(np.sqrt(x))

# x = np.linspace(-np.pi, np.pi, 1005)
# xs = np.sin(x)
# xc = np.cos(x)

# x = np.array([0,0,0,0,1,0,0,0,0,0])
# y = np.array([0,0,0,0,0.1,0,0,0,0,0])# + np.random.rand(len(x)))
#
#
#
#
# corr = signal.correlate(x, y, mode='valid')# / np.amax(signal.correlate(x, y, mode='valid'))
# print(corr)
# # print(x)
# plt.figure(figsize=(4,8))
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(212)
# ax1.plot(x)
# ax1.plot(y)
# ax2.plot(corr)

# x = np.random.random(1000)
# bin_edges = np.linspace(0.0,1,9)

# x = np.array([1,2,3,4,5,6,7,8,9])
# y = np.array([1,2,3])
#
# print(~np.in1d(x,y))
#
# z = x[~np.in1d(x,y)]
#
# print(z)

#plt.figure()
#plt.hist(x)

# y,be,bc = stats.binned_statistic(x, x, 'mean', 15, [1.1,2])
#
# print(bin_edges,be)
# print(y)
# print(np.sum(y))



#x = np.arange(16).reshape(4,4)
# num_elem = 100
# g1 = np.random.randn(num_elem)
# g2 = np.random.randn(num_elem) + 0.2
# g3 = np.random.randn(num_elem) + 0.4
#
# gg1 = ['g1'] * num_elem
# gg2 = ['g2'] * num_elem
# gg3 = ['g3'] * num_elem
#
# print(np.mean(g1),np.mean(g2),np.mean(g3))
#
# #print(stats.f_oneway(g1,g2,g3))
# #sm.multicomp.pairwise_tukeyhsd([g1,g2,g3],[gg1,gg2,gg3])
# #stats.multicomp.MultiComparison()
# #sm.sandbox.stats.multicomp.MultiComparison.kruskal()
# mc_res = sm.stats.multicomp.MultiComparison(np.concatenate((g1,g2,g3)),gg1+gg2+gg3)
# print(mc_res.tukeyhsd())


# plt.figure(figsize=(16,8))
# sns.distplot(g1, color='b')
# sns.distplot(g2, color='r')
# sns.distplot(g3, color='g')
# plt.pcolormesh(x)
# plt.colorbar()

# print(x)
