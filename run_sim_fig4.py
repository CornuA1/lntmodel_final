# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:25:52 2020

@author: Liane, Lukas
"""

# from lif_python import *
# from two_comp_lif_python import background_input
# from attractor_network2 import *
# from scipy.integrate import solve_ivp
import math, csv, random, os, time
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import binned_statistic
import seaborn as sns
import numpy as np
from multiprocessing import Pool
from functools import partial
# from make_folder import make_folder
from scipy.interpolate import interp1d
from itertools import groupby


import yaml
try:
    with open('.' + os.sep + 'loc_settings.yaml', 'r') as f:
        loc_info = yaml.safe_load(f)
    fformat = '.png'
except:
    print('Can\'t load loc_info. Server execution?')
    loc_info = {
            'figure_output_path' : './sims/',
            'raw_dir' : './data/',
        }
    fformat = '.svg'

sns.set_style("white")
sns.set_style("ticks")
plt.rcParams['svg.fonttype'] = 'none'

# fformat = '.svg'

# parameters for 2-compartment LIF
model_params = {
    'tau_s' : 16, #ms
    'tau_d' : 7, #ms
    'C_s' : 370, #pF
    'C_d' : 170, #pF
    'E_L' : -70, #mV
    'V_T' : -50, #mV
    'tau_ws' : 100, #ms
    'tau_wd' : 30.0, #ms
    'a_wd' : -13, #nS
    'b_ws' : -200, #pA
    'c_d' : 2600, #pA
    'g_s' : 1300, #pA
    'g_d' : 1200, #pA
    'E_d' : -38, #mV
    'D_d' : 6, #mV

    # input current 1 = fixed, input current 2 = gaussian; mu = recep_field
    'I1' : 250, 
    'stdev2' : 5, 
    'I2_max' : 800, 
    
    'dend_input' : 150,  #400 # if spatial is true for 2 comp lif
    'dend_input_off' : 0, 

    }

x_offset = 0



supralinear_I_cap = 2000
supralinear_I_thresh = 1000
supralinear_I_sorder = 10

# attractor network variables
tau_ = 10
# thres = 1.6
# thres = 1.75
# w = .01

def make_folder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

def save_data(filename, fields, rows):
    '''
    save data to csv

    Parameters
    ----------
    filename : string
        where to save data to.
    fields : list (of strings)
        fields for columns.
    rows : list of lists (i forget what they're actually called...)
        rows.

    Returns
    -------
    None.

    '''
    with open(filename, 'w', newline = '') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

def savefig(fname):
    fformat = 'png'
    if len(fname) > 0:
        fname = loc_info['figure_output_path'] + os.sep + fname + '.' + fformat
        plt.savefig(fname, dpi=150)

def one_comp_lif(y, t, I, k):
    v, w_s = y
    
    eq1 = -(v-model_params['E_L'])/model_params['tau_s'] + (I + w_s)/model_params['C_s']
    eq2 = -w_s/model_params['tau_ws']
    
    return [eq1, eq2]

def two_comp_lif(y, t, I, k):
    v, w_s, v_d, w_d = y
    I_s, I_d = I
    f = 1/(1 + math.exp(-(v_d-model_params['E_d'])/model_params['D_d']))
    
    if k >= 0.5 and k < 2.5:
        K = 1
    else:
        K = 0
        
    eq1 = -(v-model_params['E_L'])/model_params['tau_s'] + (model_params['g_s']*f + I_s + w_s)/model_params['C_s']
    eq2 = - w_s/model_params['tau_ws']
    eq3 = -(v_d-model_params['E_L'])/model_params['tau_d'] + (model_params['g_d']*f + model_params['c_d']*K + I_d + w_d)/model_params['C_d']
    eq4 = (-w_d + model_params['a_wd'] *(v_d-model_params['E_L']))/model_params['tau_wd']
    
    return [eq1, eq2, eq3, eq4]

def solver(var, I, k, dt, single_comp):
    '''
    

    Parameters
    ----------
    var : list
        [v, w, v_d, w_d].
    I : list or float
        [I_s, I_d]/I_s
    k : float
        timer for bAP.
    dt : float
        length of timestep.

    Returns
    -------
    sol : list
        updated [v, w, v_d, w_d].

    '''
    # sol = solve_ivp(two_comp_lif, [0, dt], var, args = (I,k), t_eval = [dt])
    # return sol.y.flatten()
    
    if single_comp:
        sol = odeint(one_comp_lif, var, [0, dt], args = (I, k))
    else:
        sol = odeint(two_comp_lif, var, [0, dt], args = (I, k))
    return sol[1,:].flatten()


def run_1comp_neuron(n, neurons, lmd_rfs, mouse_loc, coh, control, I_s_bg, t, I_d_bg, dt, model_params, nonlin = False):
    '''
    run neuron n for 1 timestep

    Parameters
    ----------
    n : int
        neuron index.

    Returns
    -------
    updated [v, w]

    '''
    
    recep_field = lmd_rfs[n] # mean of receptive field of landmark
    # check if neuron has a receptive field, if not, just keep input current at 0
    if recep_field is not None:
        I2 = model_params['I2_max'] * math.exp(-(mouse_loc-recep_field)**2/(2*model_params['stdev2']**2)) # gaussian to represent receptive field
    else:
        I2 = 0.0
    var = neurons[:2, n]
    # refrac = neurons[4, n] # refractory timer
    k = neurons[5, n] # bAP timer

    # recep_field = lmd_rfs[n] # mean of receptive field of landmark
    # I2 = I2_max * math.exp(-(mouse_loc-recep_field)**2/(2*stdev2**2)) # gaussian to represent receptive field
    # var = neurons[:2, n]
    # refrac = neurons[2, n] # refractory timer
            
    # dendritic input
    # if spatial[n]:
    #     I_d = dend_input
    # else:
    #     I_d = dend_input_off

    # dendritic input
    if type(coh) == int or type(coh) == np.float64:
        I_d = coh* model_params['dend_input'] # if coh is a scalar
    else:
        I_d = coh[n]*model_params['dend_input'] # if coh is a list
    
    I = model_params['I1'] + I2 + I_s_bg[n] + I_d + I_d_bg[n]
    
    # if the neuron is meant to do supralinear integration
    if nonlin:
        sig = supralinear_I_cap/(1+ math.exp(-(I-supralinear_I_thresh)/supralinear_I_sorder))
        I = I + sig + I_d_bg[n]
        # if sig > I:
        #     I = sig + I_s_bg[n] + I_d_bg[n]
     
    # update
    sol = solver(var, I, k, dt, True)

    return (np.transpose(sol), I)

def run_neuron(n, neurons, lmd_rfs, mouse_loc, coh, control, I_s_bg, t, I_d_bg, dt, model_params):
    '''
    run neuron n for 1 timestep

    Parameters
    ----------
    n : int
        neuron index.

    Returns
    -------
    updated [v, w, v_d, w_d]

    '''

    recep_field = lmd_rfs[n] # mean of receptive field of landmark
    # check if neuron has a receptive field, if not, just keep input current at 0
    if recep_field is not None:
        I2 = model_params['I2_max'] * math.exp(-(mouse_loc-recep_field)**2/(2*model_params['stdev2']**2)) # gaussian to represent receptive field
    else:
        I2 = 0.0
    var = neurons[:4, n]
    # refrac = neurons[4, n] # refractory timer
    k = neurons[5, n] # bAP timer
            
    # dendritic input
    # if spatial[n]:
    #     I_d = dend_input
    # else:
    #     I_d = dend_input_off
    
    if type(coh) == int or type(coh) == np.float64:
        I_d = coh* model_params['dend_input'] # if coh is a scalar
    else:
        I_d = coh[n]*model_params['dend_input'] # if coh is a list
              
    if control:
        I_s = model_params['I1'] + I2 + I_s_bg[n]
        # I = [I_s + I_d, 0]
        I = [I_s + I_d, I_d_bg[n]]
    else:
        I_s = model_params['I1'] + I2 + I_s_bg[n]
        I_d =  I_d + I_d_bg[n]
        I = [I_s, I_d] # somatic and dendritic inputs
     
    # update
    sol = solver(var, I, k, dt, False)

    return (np.transpose(sol), [I_s,I_d])
    # return np.transpose(sol)

def sim_neuron_current_injections_1comp(n, neurons, coh, control, I_s_bg, I_d_bg, t, dt, I_inj, model_params, nonlin = False):
    '''
    Inject step currents into a neuron at given times

    Parameters
    ----------
    neuron : ndarray
        vector containing neuron's current state
        
    coh : scalar or list
        coherence signal
        
    control : bool
        flag whether both currents go into the soma, or into soma and dendrite
        
    I_s_bg : ?
        somatic background current
        
    I_d_bg : ?
        dendritic background current
        
    t : scalar
        current timestep in the execution
        
    dt : scalar
        timestep (ms)
        
    I_inj : ndarray
        matrix containing timing and amplitude information for current injections

    Returns
    -------
    updated [v, w, v_d, w_d]

    '''
    
    # grab current neuron's state
    var = neurons[:2, n]
    # refrac = neurons[4, n] # refractory timer
    # k = neurons[5, n] # bAP timer
    k = 0
    
    # determine somatic injection
    t_idx = (np.abs(I_inj[:,0] - t)).argmin() # find current timestep
    
    
    # if control:
    #    # I_s = model_params['I1'] + I2 + I_s_bg[n]
    #    I_s = I_inj[t_idx,1] + I_s_bg[n] 
    #    I_d = I_inj[t_idx,2]
    #    I = [I_s + I_d, I_d_bg[n]]
    # else:
    #    I_s = I_inj[t_idx,1] + I_s_bg[n]
    #    I_d = I_inj[t_idx,2] + I_d_bg[n]
    #    I = [I_s, I_d]
       
    I = I_inj[t_idx,1] + I_inj[t_idx,2] + I_s_bg[n] + I_d_bg[n]
    
    # if the neuron is meant to do supralinear integration
    if nonlin:
        sig = supralinear_I_cap/(1+ math.exp(-(I-supralinear_I_thresh)/supralinear_I_sorder))
        I = I + sig
        # if sig > I:
        #     I = sig + I_d_bg[n]
    
    # update neuron state
    sol = solver(var, I, k, dt, True)

    return (np.transpose(sol), I)

def sim_neuron_current_injections(n, neurons, coh, control, I_s_bg, I_d_bg, t, dt, I_inj, model_params):
    '''
    Inject step currents into a neuron at given times

    Parameters
    ----------
    neuron : ndarray
        vector containing neuron's current state
        
    coh : scalar or list
        coherence signal
        
    control : bool
        flag whether both currents go into the soma, or into soma and dendrite
        
    I_s_bg : ?
        somatic background current
        
    I_d_bg : ?
        dendritic background current
        
    t : scalar
        current timestep in the execution
        
    dt : scalar
        timestep (ms)
        
    I_inj : ndarray
        matrix containing timing and amplitude information for current injections

    Returns
    -------
    updated [v, w, v_d, w_d]

    '''
    
    # grab current neuron's state
    var = neurons[:4, n]
    # refrac = neurons[4, n] # refractory timer
    k = neurons[5, n] # bAP timer
    
    # determine somatic injection
    t_idx = (np.abs(I_inj[:,0] - t)).argmin() # find current timestep
    
    
    if control:
       # I_s = model_params['I1'] + I2 + I_s_bg[n]
       I_s = I_inj[t_idx,1] + I_s_bg[n] 
       I_d = I_inj[t_idx,2]
       I = [I_s + I_d, I_d_bg[n]]
    else:
       I_s = I_inj[t_idx,1] + I_s_bg[n]
       I_d = I_inj[t_idx,2] + I_d_bg[n]
       I = [I_s, I_d]
    
    # update neuron state
    sol = solver(var, I, k, dt, False)

    return np.transpose(sol)

    

def upd_bg_input(I_bg, dt, sigma, mu = 0, tau = 2):
    return I_bg + ((mu - I_bg)/tau)*dt + sigma*np.random.normal(size = np.size(I_bg))*np.sqrt(dt)

def run_sim_parallel(total_time, dt, model_params, lmd_rfs, mouse_start, att_start, 
                     vr_vel = .03, mouse_vel = None, sigma = 200, control = False, 
                     graph_loc_v_time = True, no_spatial = False, fname = "", show_plot = False, 
                     thresh=1.75, single_comp = False, nonlin = False):
    '''
    parallelized simulation

    Parameters
    ----------
    total_time : float
        time. (ms)
    dt : float
        length of timestep. (ms)
    lmd_rfs : list
        list of receptive field centers (max current).
    mouse_start : float
        mouse's real start location.
    att_start : float
        where the attractor network starts.
    vr_vel : float or function, optional
        mouse's velocity. The default is .06. (cm/ms)
    sigma : float or tuple, optional
        background noise injected into soma. The default is 200.
        if single scalar: equal noise injected into soma and dendrite
        if tuple: sigma[0] = soma noise, sigma[1] = dendritic noise
    control : boolean, optional
        True: both currents to soma. The default is False.
    graph_loc_v_time : boolean, optional
        True: plot location vs time. The default is True.
    no_spatial : boolean, optional
        True: no spatial signal. The default is False.
    fname : string, optional
        where to save figure to. The default is "".
    thresh : float
        threshold (unit-less) for when spikes at short intervals are considered bursts and influence CAN
    single_comp : bool
        treat neuron as single-compartment
    nonlin : bool
        whether or not a single-compartment neuron integrates linearly or supralinearly

    Returns
    -------
    None.

    '''
    total_spikes = 0
    # time1 = time.time()
    
    num_neurons = len(lmd_rfs)
    w = 1/num_neurons # for normalization
    neurons = np.transpose(np.tile(np.array([-70, 0, -70, 0, 1000, 1000], dtype = np.float64), (num_neurons,1))) # row 1: v, 2: w, 3: v_d, 4: w_d, 5: refrac, 6: k
    # refrac and k can be initialized to any large number
    times = np.arange(total_time, step = dt)
    timesteps = len(times) # number of timesteps
    
    if single_comp:
        neuron_I = np.zeros((timesteps,num_neurons))
    else:
        neuron_I = np.zeros((timesteps,num_neurons,2))
    
    mouse_loc = mouse_start # mouse's true start location
    mouse_locs = np.zeros(timesteps) # mouse's true location
    mouse_locs[0] = mouse_loc
    
    can = np.zeros(num_neurons) # input s for each cell (will have exp filter)
    can_locs = np.zeros(timesteps) # attractor locations
    can_locs[0] = att_start
    
    force = np.zeros(timesteps) # force
    
    if isinstance(sigma, tuple):
        sigma_s = sigma[0]
        sigma_d = sigma[1]
    else:
        sigma_s = sigma
        sigma_d = sigma
    
    # spatial = np.zeros(num_neurons, dtype = bool) # spatial signals for each cell
    if mouse_vel is None:
        coh = 1
    else:
        coh = 0

    I_s_bg = np.zeros(num_neurons)
    I_d_bg = np.zeros(num_neurons)
    
    base = np.full(num_neurons, -70) # resting potential
    adap_curr = np.full(num_neurons, model_params['b_ws']) # adaptive current
    time_upd = np.full(num_neurons, dt) # to update timers
    
    v = np.zeros((timesteps,num_neurons))
    spikes = np.zeros((timesteps,num_neurons))
    
    # for plotting mouse velocity, vr velocity, coherence
    all_mouse_vel = np.zeros(timesteps)
    all_vr_vel = np.zeros(timesteps)
    all_coh = np.zeros(timesteps)
    
    # time2 = time.time()
    # print("Initialization time: " + str(time2-time1))
    
    p = Pool()
    for t in range(timesteps):
        if type(vr_vel) == float:
            # if vr_vel is a float
            vel = vr_vel
        else:
            # if vr_vel is a function
            vel = vr_vel(t*dt)
            
        if mouse_vel is None:
            # mouse velocity == VR velocity
            coh = 1
            all_mouse_vel[t] = vel
        else:
            if type(mouse_vel) == float:
                # if mouse_vel is a float
                mouse_vel_temp = mouse_vel
            else:
                # if mouse_vel is a function
                mouse_vel_temp = mouse_vel(t*dt)

            coh = max(1 - abs(mouse_vel_temp - vel)/vel, 0) # coherence
            # coh = 1 - abs(mouse_vel_temp - vel)/vel
        
            all_mouse_vel[t] = mouse_vel_temp
        
        all_vr_vel[t] = vel
        all_coh[t] = coh
        # print(coh)
        # parallel processing
        # time3 = time.time()

        if single_comp:
            update = p.map(partial(run_1comp_neuron, neurons = neurons, lmd_rfs = lmd_rfs, mouse_loc = mouse_loc, coh = coh, control = control, I_s_bg = I_s_bg, t = t, I_d_bg = I_d_bg, dt = dt, model_params = model_params, nonlin = nonlin), np.arange(num_neurons))
            neuron_I_update = [u[1] for u in update]
            update = [u[0] for u in update]
            # update = run1te.reshape((update.shape[0],1)).T
        else:
            update = p.map(partial(run_neuron, neurons = neurons, lmd_rfs = lmd_rfs, mouse_loc = mouse_loc, coh = coh, control = control, I_s_bg = I_s_bg, t = t, I_d_bg = I_d_bg, dt = dt, model_params = model_params), np.arange(num_neurons))
            neuron_I_update = [u[1] for u in update]
            update = [u[0] for u in update]
            # update = run_neuron(0, neurons = neurons, lmd_rfs = lmd_rfs, mouse_loc = mouse_loc, coh = coh, control = control, I_s_bg = I_s_bg, t = t, I_d_bg = I_d_bg, dt = dt, model_params = model_params)
            # update = update.reshape((update.shape[0],1)).T
        # time4 = time.time()
        # print("Parallel processing time: " + str(time4-time3))
        
        # update values
        update = np.transpose(update)
        neurons[1:4, :] = update[1:] # update w, v_d, w_d
        # refrac = neurons[4, :] < 2 # True if neuron in refractory period
        # neurons[0, :] = np.multiply(refrac, base) + np.multiply(~refrac, update[0]) # update voltage
        neurons[0, :] = update[0]
        
        # update values if spike
        spike = np.array([v > model_params['V_T'] for v in neurons[0, :]])
        neurons[0, :] = np.multiply(spike, base) + np.multiply(~spike, neurons[0, :])
        neurons[1, :] = neurons[1, :] + np.multiply(spike, adap_curr)
        neurons[4, :] = np.multiply(~spike, neurons[4, :])
        neurons[5, :] = np.multiply(~spike, neurons[5, :])
        
        total_spikes += np.sum(spike)
        
        # update timers
        neurons[4, :] = neurons[4, :] + time_upd
        neurons[5, :] = neurons[5, :] + time_upd
        
        # update attractor network
        # exponential decay
        can = can * (1 - dt/tau_)
        
        # self motion input
        if t > 0:
            if mouse_vel is None:
                # mouse_vel == vr_vel
                can_locs[t] = can_locs[t-1] + vel*dt
            else:
                # mouse velocity is different than vr velocity, use mouse velocity/self-motion signals to update attractor network
                can_locs[t] = can_locs[t-1] + mouse_vel_temp*dt
        
        # landmark input
        can = can + spike
        # temp = can * (can > thres) # threshold
        # thresh = 1.25
        temp = can * (2.5 / (1 + np.exp(-30*(can-thresh)))) # threshold

        for n in np.arange(num_neurons):
            s = temp[n]
            lmd_strength = min(w * s, 1)
            if lmd_rfs[n] is not None:
                f_direction = 1 if (lmd_rfs[n] - can_locs[t]) > 0 else -1
                f = lmd_strength * f_direction * dt
                # f = lmd_strength * ((lmd_rfs[n]-x_offset) - can_locs[t]) * dt # -5 for offset between spiking and current
            else:
                f = 0.0
            # f = lmd_strength * (lmd_rfs[n] - can_locs[t])
            force[t] += np.abs(f)
            can_locs[t] += f
            
            # # update spatial signals
            # if not no_spatial:
            #     spatial[n] = (abs(lmd_rfs[n] - x_offset - can_locs[t]) < 20)
            #     # spatial[n] = (abs(lmd_rfs[n] - can_locs[t]) < 20)
            
        # update mouse location (based on vr speed)
        mouse_locs[t] = mouse_loc
        mouse_loc += vel * dt
        
        # update background input
        I_s_bg = upd_bg_input(I_s_bg, dt, sigma_s)
        I_d_bg = upd_bg_input(I_d_bg, dt, sigma_d)
        
        # voltage of neuron 0
        # if spike[0]:
        #     v[t] = 20
        # else:
        #     v[t,:] = neurons[0, :]
        v[t,:] = neurons[0, :]
        v[t,spike] = 20
        spikes[t,:] = spike
        if single_comp:
            neuron_I[t,:] = neuron_I_update
        else:
            neuron_I[t,:,:] = neuron_I_update
        
    p.close()
    p.join()
 
    print("number of neurons: " + str(num_neurons) + " noise: " + str(sigma) + " total spikes: " + str(total_spikes))
    return (times, mouse_locs, can_locs, v, spikes, force, neuron_I, all_coh, all_mouse_vel, all_vr_vel)

def plot_trace(v, dt, fname = "", show_plot = True):
    fig, ax = plt.subplots(figsize = (24,6))
    ax.plot(np.arange(0, dt*np.size(v), step = dt), v, linewidth = 1, color = 'black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params( \

        reset=True,

        axis='both', \

        direction='out', \

        length=5, \

        right=False, \

        top=False, \
            
        labelsize = 20)
    # ax.set_xlim([2000, 3000])
    ax.set_xlabel("Time (ms)", fontsize = 20)
    ax.set_ylabel("Voltage (mV)", fontsize = 20)
    if len(fname) > 0:
        savefig(fname + "_trace")
    if show_plot:
        plt.show()
    plt.close()

def random_landmarks(num_neurons, distribution = 'uniform', mu = 220, lm_sigma = 30):
    '''
    generates random landmarks from a uniform distribution

    Parameters
    ----------
    num_neurons : float
        number of neurons.
    mu : float
        mean of gaussian distribution from which we select receptive fields
    sigma : float
        standard deviation of gaussian distribution from which we select receptive fields
    distribution : string
        'uniform' ... returns landmarks with receptive fields drawn from a uniform distribution
        'gaussian' ... return landmarks with receptive fields drawn from a gaussian distribution
    

    Returns
    -------
    landmarks : list
        list of landmarks, randomly generated with uniform distribution.

    '''
    if distribution == 'gaussian':
        landmark = np.random.normal(mu, lm_sigma, num_neurons)
            # for n in range(num_neurons):
            #     landmark.append(random.gauss(mu, sigma))
    elif distribution == 'uniform':
        landmark = 50 + 300*np.random.random_sample(num_neurons)
    return landmark

def load_data(mouse, session, ol = False):
    '''
    loads behavior data for a mouse and session

    Parameters
    ----------
    mouse : str
        mouse id.
    session : str
        session.

    Returns
    -------
    behav_data : array
        behavioral data.

    '''
    try:
        if ol:
            filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'old' + os.sep + 'aligned_data.mat'
        else:
            filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'old' + os.sep + 'aligned_data.mat'
        # filename_ol = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data_ol.mat'
        aligned_data = loadmat(filename)
    except:
        try:
            if ol:
                filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data.mat'
            else:
                filename = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data.mat'
            # filename_ol = loc_info['raw_dir'] + os.sep + mouse + os.sep + session + os.sep + 'aligned_data_ol.mat'
            aligned_data = loadmat(filename)
        except:
            if ol:
                filename = loc_info['raw_dir'] + os.sep + session + os.sep + 'aligned_data_ol.mat'
            else:
                filename = loc_info['raw_dir'] + os.sep + 'aligned_data.mat'
            aligned_data = loadmat(filename)
    behav_data = aligned_data['behaviour_aligned']
    return behav_data

def filter_trials(behav_data, trial_type):
    '''
    get behavioral data for short or long trials

    Parameters
    ----------
    behav_data : array
        behavioral data.
    trial_type : str
        'short' or 'long'.

    Returns
    -------
    trials : array
        data from either short or long trials.

    '''
    
    all_trials = False
    ol_sess = False
    
    if trial_type == 'short':
        trial_type = 3
    elif trial_type == 'long':
        trial_type = 4
    elif trial_type == 'both':
        all_trials = True
        trial_type = -1
    elif trial_type == 'ol_fast':
        ol_sess = True
        
    trials = [] # list of trials; each entry is a trial
    # trial_temp = [] # list of data points for a trial, which are then appended to trials
    # trial_num = 1 # trial number, the aligned data starts at trial 1
    trial_start = np.zeros((0,))
    
    sess_trials = np.unique(behav_data[:,6]) # get all trial numbers in a session
    
    for t in sess_trials:
        cur_trial = behav_data[behav_data[:,6]==t,:]
        if cur_trial[0,4] == trial_type and not ol_sess:
            trials.append(cur_trial)
            if cur_trial[0,1] > 40: # the threshold is just a cutoff to toss out the very first trial of a session that starts at 0
                trial_start = np.hstack((trial_start,cur_trial[0,1]))
        elif all_trials and cur_trial[0,4] != 5 and not ol_sess:
            trials.append(cur_trial)
            if cur_trial[0,1] > 40: # the threshold is just a cutoff to toss out the very first trial of a session that starts at 0
                trial_start = np.hstack((trial_start,cur_trial[0,1]))
        elif ol_sess and trial_type == 'ol_fast' and cur_trial[0,4] != 5:
            if cur_trial[0,3] > 20:
                trials.append(cur_trial)
                if cur_trial[0,1] > 40: # the threshold is just a cutoff to toss out the very first trial of a session that starts at 0
                    trial_start = np.hstack((trial_start,cur_trial[0,1]))
                
            
    
    # for n in range (len(behav_data[:,0])):
    #     if trial_num != behav_data[n,6]:
    #     # if trial_num != current trial number, this means it's the start of a new trial
    #         if behav_data[n-1,4] == trial_type:
    #             # add to trials if it's the right trial type
    #             trials.append(trial_temp)
            
    #         # reset
    #         trial_temp = []
    #         trial_num = behav_data[n,6]

    #     if behav_data[n,4] == trial_type:
    #     # if it's the right trial type, add data points to trial_temp
    #         trial_temp.append(behav_data[n,:])

    return trials, np.mean(trial_start)

def run_nolm_sim():
    ''' run a simulation in which neurons receive background noise only '''
    mouse_start = 60
    att_start = 40
    num_neurons = [1] #np.arange(25, 26, step = 25)
    landmarks = [None]
    for n in num_neurons:
        # landmarks = random_landmarks(n, 'uniform')
        start_time = time.time()
        # run_sim_parallel(10000, .5, landmarks, mouse_start, att_start, sigma = 100, show_plot = True)
        run_sim_parallel(10000, 1, model_params, landmarks, mouse_start, att_start, control=True, sigma = (300,0), fname='nolm')
        end_time = time.time()
    
    print("Time taken = " + str(end_time - start_time))

def run_spikerate_burstrate():
    ''' Run simulation with soma + dendrite and soma only inputs and compare spikerate vs. burstrate '''
    
    # burst detection threshold
    total_time = 10000 # ms
    dt = 1 # ms
    temporal_binning = 100 # ms
    burst_ISI_thresh = 10 # ms
    random.seed(234)
    mouse_start = 60
    att_start = 40
    num_neurons = [10] #np.arange(25, 26, step = 25)
    landmarks = np.ones((num_neurons[0],1)) * 200 # landmarks = random_landmarks(n)
    sigma = (300,100)
    # landmarks = [None,None,None,None,None]
    # landmarks = [100,150,200,250,300]
    # start_time = time.time()
    # run_sim_parallel(10000, .5, landmarks, mouse_start, att_start, sigma = 100, show_plot = True)
    timestamps, mouse_locs, can_locs, v, spikes, force, neuron_I, coh, mouse_vel, vr_vel = run_sim_parallel(total_time, dt, model_params, landmarks, mouse_start, att_start, control=False, sigma = sigma, fname='sd')
    # timestamps, mouse_locs_control, can_locs_control, v_control, spikes_control, force_control, neuron_I_control = run_sim_parallel(1000, 1, landmarks, mouse_start, att_start, control=True, sigma = (300,0), fname='nolm')
    
    # run through the spiking activity of each neuron and detect bursts
    t_vec = np.arange(0, spikes.shape[0], 1)# vector with timestamps
    spikes_all = []
    bursts_all = []
    for i,sp in enumerate(spikes.T):
        t_spikes = np.zeros((0,3))
        t_bursts = np.zeros((0,3))
        cur_neuron = np.vstack((np.arange(0,sp.shape[0],1),t_vec,sp)).T # create matrix where we have index and timestamps of spikes for each neuron
        cur_spikes = cur_neuron[cur_neuron[:,2]>0,:]
        spike_ISI = np.diff(cur_spikes[:,1])
        isburst = np.hstack((False, spike_ISI<burst_ISI_thresh))
        # pull out times of spikes and bursts
        curburst = False
        for j,ib in enumerate(isburst):
            if not ib:
                t_spikes = np.vstack((t_spikes,cur_spikes[j,:]))
                curburst = False
            elif not curburst: 
                t_bursts = np.vstack((t_bursts,cur_spikes[j,:]))
                curburst = True
        spikes_all.append(t_spikes)
        bursts_all.append(t_bursts)
    
    fig, ax = plt.subplots(10)
    for i,v_n in enumerate(v.T):
        ax[i].plot(v_n)
    
    fig = plt.figure(figsize=(10,15))
    gs = fig.add_gridspec(15, 1)
    ax3 = fig.add_subplot(gs[4:7,:])
    ax2 = fig.add_subplot(gs[0:2,:])
    ax1 = fig.add_subplot(gs[2:4,:])
    ax4 = fig.add_subplot(gs[7:10,:])
    ax5 = fig.add_subplot(gs[10:13,:])
    
    for v_n in v.T:
        ax1.plot(v_n)
        
    
    # create bins to count spikes and bursts for a given amount of time
    hist_bin_edges = np.arange(0,total_time + temporal_binning, temporal_binning)
    spike_count = np.zeros((0,len(hist_bin_edges)-1))
    burst_count = np.zeros((0,len(hist_bin_edges)-1))
    for i,s_t in enumerate(spikes.T):
        x_coord = t_vec[s_t > 0]
        y_coord = s_t[s_t > 0] * i
        ax3.scatter(x_coord,y_coord, marker='|', c='k' )
        if len(spikes_all[i]) > 0:
            ax3.scatter(spikes_all[i][:,1],spikes_all[i][:,2] * i, marker='|', c='b' )
            spike_hist,_ = np.histogram(spikes_all[i][:,1], bins=hist_bin_edges)
            spike_count = np.vstack((spike_count, spike_hist))
            
        if len(bursts_all[i]) > 0:
            ax3.scatter(bursts_all[i][:,1],bursts_all[i][:,2] * i, marker='|', c='r' )
            burst_hist,_ = np.histogram(bursts_all[i][:,1], bins=hist_bin_edges)
            burst_count = np.vstack((burst_count, burst_hist))
    
    x_coords = np.arange(temporal_binning/2,total_time + temporal_binning/2, temporal_binning)
    mean_spikes = np.mean(spike_count,0) * 10
    spike_count_sem = np.std(spike_count,0) / np.sqrt(spike_count.shape[0]) * 10
    mean_bursts = np.mean(burst_count,0) * 10
    burst_count_sem = np.std(burst_count,0) / np.sqrt(burst_count.shape[0]) * 10
    
    ax4.plot(x_coords,mean_spikes)
    ax4.fill_between(x_coords, mean_spikes - spike_count_sem, mean_spikes + spike_count_sem, alpha = .2)
    ax5.plot(x_coords, mean_bursts)
    ax5.fill_between(x_coords, mean_bursts - burst_count_sem, mean_bursts + burst_count_sem, alpha = .2)
    
    ax4.set_ylabel("spike rate (Hz)")
    ax5.set_ylabel("burst rate (Hz)")
    
    # ax4.set_ylim([0,1])
    # ax5.set_ylim([0,1])
    
    ax3.set_ylim([-1,spikes.shape[1]])
    
    # ax2.plot(I_inj[:,0], I_inj[:,1], c='#EC008C', ls='-')
    # ax2_2 = ax2.twinx()
    # ax2_2.plot(I_inj[:,0], I_inj[:,2], c='#EC008C', ls='--')
    
    ax1.set_xlim([0,total_time])
    ax2.set_xlim([0,total_time])
    # ax2_2.set_xlim([0,total_time])
    ax3.set_xlim([0,total_time])
    
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])
    
    
        
    # timestamps, mouse_locs, can_locs, v, spikes, force, neuron_I = run_sim_parallel(10000, 1, model_params, landmarks, mouse_start, att_start, control=True, sigma = (300,0), fname='ss')
    # fig, ax = plt.subplots(5)
    # for i,v_n in enumerate(v.T):
    #     ax[i].plot(v_n)
    # plt.show()
    
        
    # end_time = time.time()
    
def run_neuron_range_simulation():
    random.seed(234)
    mouse_start = 60
    att_start = 40
    # num_neurons = np.arange(25, 26, step = 25)
    num_neurons = [1,5]
    # landmarks = np.ones((num_neurons[0],1)) * 150 # landmarks = random_landmarks(n)
    
    # landmarks = [100,150,200,250,300]
    
    for n in num_neurons:
        landmarks = random_landmarks(n, 'uniform')
        # run_sim_parallel(10000, .5, landmarks, mouse_start, att_start, sigma = 100, show_plot = True)
        # start_time = time.time()
        timestamps, mouse_locs, can_locs, v, spikes, force, neuron_I, coh, mouse_vel, vr_vel = run_sim_parallel(10000, 1, model_params, landmarks, mouse_start, att_start, control=False, sigma = (300,0), fname='sd')
        # end_time = time.time()
        # timestamps, mouse_locs_control, can_locs_control, v_control, spikes_control, force_control, neuron_I_control = run_sim_parallel(1000, 1, landmarks, mouse_start, att_start, control=True, sigma = (300,0), fname='nolm')
        # fig, ax = plt.subplots(5)
        # for i,v_n in enumerate(v.T):
        #     ax[i].plot(v_n)
        # plt.show()
    
def sim_mouse_behavior(MOUSE, SESSION, ol, no_spatial, control, sigma, show_plot, trial_type, num_neurons, select_trials, landmarks, subfolder='', thresh=1.75, single_comp=False, nonlin=False, trial_nr_offset = 0):
    
    select_trials_all = select_trials # if we grab trials from multiple sessions, keep track which trials we have in total
    final_sess = False # if we grab trials from multiple sesssions, indicate when all trials have been run
    sess_trial_nr = 0
    
    random.seed(234)
    out_folder = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise_' + 'thresh_' + str(thresh)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # set up vectors to keep track of individual trial simulation results
    init_error = np.empty(0)
    final_error = np.empty(0)
    # ind = 0 # temp variable just for indexing init_error and final_error    
    
    # if only a single session is provided, turn it into 
    if isinstance(SESSION, str):
        SESSION = [SESSION]
    
    # run through every provided session
    for s in SESSION:
        #load data
        behav_data = load_data(MOUSE, s, ol = ol)
        trials, ave_mouse_start = filter_trials(behav_data, trial_type)
        
        # if openloop is true, we assume that we use all trials in sequence (rather than a specific list of trial numbers)
        # and we also need to split it up over multiple sessions
        if select_trials.shape[0] > len(trials) and ol:
            sess_trial_nr = len(trials)
            sess_trials = np.arange(0,sess_trial_nr,1)
            select_trials = select_trials[sess_trial_nr:]
            
            print('full block ' + str(sess_trial_nr) + ' sess trials: ' + str(sess_trials) + ' select_trials ' + str(select_trials) + ' trialnr offset: ' + str(trial_nr_offset))
        elif ol:
            if final_sess: # this condition is only true once we ran through the final set of trials
                sess_trials = []
                print('nothing left to do!')
            else:
                sess_trial_nr = len(select_trials)
                sess_trials = np.arange(0,sess_trial_nr,1)
                # trial_nr_offset = trial_nr_offset + sess_trial_nr
                final_sess = True
                print('final block ' + str(sess_trial_nr) + ' sess trials: ' + str(sess_trials) + ' select_trials ' + str(select_trials) + ' trialnr offset: ' + str(trial_nr_offset))
        else:
            sess_trials = select_trials

        # run simulation for each trial
        for i,tn in enumerate(sess_trials):
        # for tn,trial in enumerate(trials):       
            trial = trials[tn]
            trial[:,0] = (trial[:,0] - trial[0,0]) *1000 # start the trial timer at 0 for each trial and convert to milliseconds
            end_time = trial[-1,0]
            if not ol:
                trial[trial[:,3]>120,3] = 120 # clip running speeds at 120 to remove artifacts
                velocity = interp1d(trial[:,0],trial[:,3]/1000)
                mouse_velocity = None
            else:
                trial[trial[:,8]>120,8] = 120 # clip running speeds at 120 to remove artifacts
                velocity = interp1d(trial[:,0],trial[:,3]/1000)
                mouse_velocity = interp1d(trial[:,0],trial[:,8]/1000)
            
            if len(out_folder) > 0:
                fname = 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "sim_" + str(i + trial_nr_offset)
            else:
                fname = ""
                
            timestamps, mouse_locs, can_locs, v, spikes, force, neuron_I, coh, mouse_vel, vr_vel = run_sim_parallel(end_time, .5, model_params, landmarks, trial[0][1], ave_mouse_start, vr_vel = velocity, mouse_vel = mouse_velocity, no_spatial = no_spatial, fname = fname, control = control, sigma = sigma, show_plot = show_plot, thresh = thresh, single_comp = single_comp, nonlin = nonlin)
            
            init_error = np.append(init_error, mouse_locs[0] - can_locs[0])
            final_error = np.append(final_error, mouse_locs[-1] - can_locs[-1])
            # init_error.append(mouse_locs[0] - can_locs[0])
            # final_error.append(mouse_locs[-1] - can_locs[-1])
            
            fig = plt.figure(figsize=(5,7.5))
            gs = fig.add_gridspec(16, 1)
            ax1 = fig.add_subplot(gs[0:3,:])
            ax2 = fig.add_subplot(gs[3:6,:])
            ax4 = fig.add_subplot(gs[6:9,:])
            ax3 = fig.add_subplot(gs[14:16,:])
            ax5 = fig.add_subplot(gs[10:13,:])
            
            # fig = plt.figure(figsize=(5,5))
            # ax1 = fig.add_subplot(111)
            ax1.plot(timestamps, mouse_locs, label = "Mouse location")
            ax1.plot(timestamps, can_locs, label = "CAN location")
            ax1.set(xlabel = "Time (ms)", ylabel = "Location (cm)")
            ax1.legend()
            # ax1.set_xlim([0, total_time])
            # ax1.set_ylim([0, .1])
            
            for j,s_t in enumerate(spikes.T):
                x_coord = timestamps[s_t > 0]
                y_coord = s_t[s_t > 0] * j
                ax2.scatter(x_coord,y_coord, marker='|', c='k' )
                
            # plots error
            loc_diff = abs(can_locs - mouse_locs)
            ax3.plot(mouse_locs, loc_diff, label = "Error")
            ax3.axvline(220,ls = '--', c = 'r', lw = 3)
            ax3.set_xlabel("Mouse location (cm)")
            ax3.set_ylabel("Difference from CAN (cm)")
            # ax3.set_xlim([50,350])
            ax3.set_ylim([0,40])
            
            # plots force
            ax3_3 = ax3.twinx()
        #     # ax4 = plt.subplot(122)
            ax3.set_zorder(ax3_3.get_zorder()+1)
            ax3.patch.set_visible(False)
            ax3_3.set_ylabel("Force")
            ax3_3.plot(mouse_locs, force, label = "Force", color = "lavender")
            # ax3_3.set_ylim([-.4,.4])
            ax3_3.legend()
            
            ax4_2 = ax4.twinx()
            ax4.set_zorder(ax4_2.get_zorder()+1)
            ax4.patch.set_visible(False)
            ax4.plot(timestamps, loc_diff,  label = "Error")
            ax4.set_xlabel("Time (ms)")
            ax4.set_ylabel("Difference from CAN (cm)")
            
            ax4_2.plot(timestamps, force, label = "Force", color = "lavender")
            
            ax4_lims = ax4.get_ylim()
            ax4.set_ylim([0,ax4_lims[1]+5])
            
            if single_comp:
                ax5.plot(timestamps, np.mean(neuron_I,1))
            
            fname = out_folder + os.sep + "trial_" + str(i + trial_nr_offset) + fformat
            plt.savefig(fname, dpi=100)
            print('saved ' + fname)
            plt.close()
            
            np.savez(out_folder + os.sep + "results_trial_" + str(i + trial_nr_offset) + ".npz", timestamps=timestamps, mouse_locs=mouse_locs, can_locs=can_locs, v=v, spikes=spikes, force=force, neuron_I=neuron_I, coh=coh, mouse_vel=mouse_vel, vr_vel=vr_vel)
            
        trial_nr_offset = trial_nr_offset + sess_trial_nr # add the offset
        
    print(np.abs(init_error))
    print(np.abs(final_error))
    
    fields = ["Trial Number", "Initial Error", "Final Error"]
    # rows = np.transpose(np.array([mouse_locs, can_locs, trial_num])).tolist()
    rows = np.transpose(np.array([select_trials_all, init_error, final_error])).tolist()
    save_data(out_folder + os.sep + "data.csv", fields, rows)
    
    
    return init_error, final_error

def run_neuron_n_range():
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    subfolder = 'neuron_range'
    ol = False
    no_spatial = False
    control = False
    sigma = (300,50)
    show_plot = True
    trial_type = 'short'
    num_neurons_range = [10]
    select_trials = np.arange(10)
    
    for num_neurons in num_neurons_range:
        make_folder(loc_info['figure_output_path'] + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        print(num_neurons)
        landmarks = random_landmarks(num_neurons, 'gaussian')
        landmarks = np.ones((num_neurons)) * 220
        print(np.sort(landmarks))
        # landmarks = [None] * num_neurons
        init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = subfolder)
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)  
        for i in range(len(init_error)):
            ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        fname = loc_info['figure_output_path'] + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "error"+ fformat
        plt.savefig(fname, dpi=300)
        plt.close()
        # plt.scatter( np.hstack(( np.zeros(( init_error.shape[0],1 )), np.ones(( final_error.shape[0],1 ))  )), np.hstack(( np.hstack((init_error,final_error)) )) )

def run_thresh_range():
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    subfolder = 'thresh_range'
    ol = False
    no_spatial = False
    control = False
    sigma = (300,50)
    show_plot = True
    trial_type = 'short'
    num_neurons = 100
    select_trials = np.arange(50)
    thresh_range = [1,1.25,1.50,1.75,2,2.25,2.5]
    thresh_range = [1.125,1.375,1.625,1.875,2.125,2.375]
    thresh_range = [1.625,1.875]
    
    for thresh in thresh_range:
        make_folder(loc_info['figure_output_path'] + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise_' + 'thresh_' + str(thresh) )
        print(num_neurons)
        landmarks = random_landmarks(num_neurons, 'gaussian')
        print(np.sort(landmarks))
        # landmarks = [None] * num_neurons
        init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, thresh=thresh, landmarks=landmarks, subfolder = subfolder)
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)  
        for i in range(len(init_error)):
            ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        fname = loc_info['figure_output_path'] + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise_'  + 'thresh_' + str(thresh)  + os.sep + "error"+ fformat
        plt.savefig(fname, dpi=300)
        plt.close()
        # plt.scatter( np.hstack(( np.zeros(( init_error.shape[0],1 )), np.ones(( final_error.shape[0],1 ))  )), np.hstack(( np.hstack((init_error,final_error)) )) )
    

def run_uniform_vs_gaussian():
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    subfolder = 'ug'
    ol = False
    no_spatial = False
    control = False
    sigma = (300,50)
    show_plot = True
    trial_type = 'short'
    num_neurons_range = [1,10,50]
    select_trials = np.arange(2)    
    
    for num_neurons in num_neurons_range:
        make_folder(loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        
        print(num_neurons)
        
        
        landmarks = random_landmarks(num_neurons, 'uniform')
        print(np.sort(landmarks))
        # landmarks = [None] * num_neurons
        init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'ug_uni')
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)  
        for i in range(len(init_error)):
            ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "uni_error"+ '.png'
        plt.savefig(fname, dpi=300)
        plt.close()
        # plt.scatter( np.hstack(( np.zeros(( init_error.shape[0],1 )), np.ones(( final_error.shape[0],1 ))  )), np.hstack(( np.hstack((init_error,final_error)) )) )
        
    for num_neurons in num_neurons_range:
        print(num_neurons)
        landmarks = random_landmarks(num_neurons, 'gaussian', 220, 30)
        print(np.sort(landmarks))
        # landmarks = [None] * num_neurons
        init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'ug_gauss')
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)  
        for i in range(len(init_error)):
            ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "gauss_error"+ '.png'
        plt.savefig(fname, dpi=300)
        plt.close()            
    
def run_sigma_range():
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    subfolder = 'sr_ug'
    ol = False
    no_spatial = False
    control = False
    sigma_range = [(300,50),(300,150)]
    show_plot = True
    trial_type = 'short'
    num_neurons_range = [1,10,50]
    select_trials = np.arange(10)
    
    for sigma in sigma_range:
        for num_neurons in num_neurons_range:
            make_folder(loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
            
            print(num_neurons)
            # print(num_neurons)
            landmarks = random_landmarks(num_neurons, 'uniform')
            print(np.sort(landmarks))
            # landmarks = [None] * num_neurons
            init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'srug_uni')
            
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)  
            for i in range(len(init_error)):
                ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
            
            fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "uni_error"+ '.png'
            plt.savefig(fname, dpi=300)
            plt.close()
            # plt.scatter( np.hstack(( np.zeros(( init_error.shape[0],1 )), np.ones(( final_error.shape[0],1 ))  )), np.hstack(( np.hstack((init_error,final_error)) )) )
        
    for sigma in sigma_range:
        for num_neurons in num_neurons_range:
            print(num_neurons)
            landmarks = random_landmarks(num_neurons, 'gaussian', 220, 30)
            print(np.sort(landmarks))
            # landmarks = [None] * num_neurons
            init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'srug_gauss')
            
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)  
            for i in range(len(init_error)):
                ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
            
            fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "gauss_error"+ '.png'
            plt.savefig(fname, dpi=300)
            plt.close()   
       
def run_sim_with_control():
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    subfolder = 'sr_ug_rc'
    ol = False
    no_spatial = False
    control = False
    sigma_range = [(300,50)]
    show_plot = True
    trial_type = 'both'
    num_neurons_range = [100]
    select_trials = np.arange(100)
    
    for sigma in sigma_range:
        for num_neurons in num_neurons_range:
            control = False
            make_folder(loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
            
            # print(num_neurons)
            # # landmarks = np.ones((num_neurons)) * 220
            landmarks = random_landmarks(num_neurons, 'gaussian', 220, 60)
            
            print(np.sort(landmarks))
            # # landmarks = [None] * num_neurons
            # init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'srug_real')
            
            # fig = plt.figure(figsize=(5,5))
            # ax = fig.add_subplot(1,1,1)  
            # for i in range(len(init_error)):
            #     ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
            
            # fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "real_error"+ '.png'
            # plt.savefig(fname, dpi=300)
            # plt.close()   

            
            control = True
            print(num_neurons)
            # landmarks = [None] * num_neurons
            init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'srug_cont')
            
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)  
            for i in range(len(init_error)):
                ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
            
            fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "control_error"+ '.png'
            plt.savefig(fname, dpi=300)
            plt.close()  
        
def run_openloop():
    MOUSE = 'LF191022_1'
    SESSION = ['20191207_ol','20191204_ol','20191213_ol','20191209_ol','20191215_ol']
    subfolder = 'sr_ug_rc_ol'
    ol = True
    no_spatial = False
    control = False
    sigma_range = [(300,50)]
    show_plot = True
    trial_type = 'ol_fast'
    num_neurons_range = [100]
    select_trials = np.arange(20)
    
    for sigma in sigma_range:
        for num_neurons in num_neurons_range:
            control = False
            make_folder(loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
            
            print(num_neurons)
            landmarks = random_landmarks(num_neurons, 'gaussian', 220, 60)
            # landmarks = np.ones((num_neurons)) * 220
            print(np.sort(landmarks))
            # landmarks = [None] * num_neurons
            init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'srug_ol_real')
            
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)  
            for i in range(len(init_error)):
                ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
            
            fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "real_error"+ '.png'
            plt.savefig(fname, dpi=300)
            plt.close()   

            
            # control = True
            # print(num_neurons)
            # # landmarks = [None] * num_neurons
            # init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = 'srug_ol_cont')
            
            # fig = plt.figure(figsize=(5,5))
            # ax = fig.add_subplot(1,1,1)  
            # for i in range(len(init_error)):
            #     ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
            
            # fname = loc_info['figure_output_path'] + os.sep + subfolder + os.sep + 'behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "control_error"+ '.png'
            # plt.savefig(fname, dpi=300)
            # plt.close()  

def neuron_current_injections(num_neurons, total_time, dt, I_inj, control=False, fname_suffix = '', sigma = None, thresh = 1.75, single_comp = False, nonlin = False):
    ''' Run neuron simulation (without CAN) with different currents '''
    # set up neurons
    neurons = np.transpose(np.tile(np.array([-70, 0, -70, 0, 1000, 1000], dtype = np.float64), (num_neurons,1)))
     
    # burst detection threshold
    burst_ISI_thresh = 10 # ms
    
    # bin size for evaluating spike vs burstrate
    temporal_binning = 100 # ms
    
    # normalizing landmark force
    w = 1/num_neurons # for normalization
    
    # times = np.arange(total_time, step = dt)
    timesteps = int(total_time/dt) # number of timesteps
    v = np.zeros((timesteps,num_neurons))
    v_d = np.zeros((timesteps,num_neurons))
    spikes = np.zeros((timesteps,num_neurons))
    neuron_I = np.zeros((timesteps,num_neurons))
    
    # calculate the force that would be transmitted to the CAN
    can = np.zeros(num_neurons) # input s for each cell (will have exp filter)
    force = np.zeros(timesteps) # force
    
    # set up parameters for background noise
    if sigma != None:
        sigma_s = sigma[0]
        sigma_d = sigma[1]
    else:
        sigma_s = 200
        sigma_d = 50
        
        
    I_s_bg = np.zeros(num_neurons)
    I_d_bg = np.zeros(num_neurons)
    
    # additional neuron parameters
    base = np.full(num_neurons, -70) # resting potential
    adap_curr = np.full(num_neurons, model_params['b_ws']) # adaptive current
    time_upd = np.full(num_neurons, dt) # to update timers
    
    # keep track of number of spikes
    total_spikes = 0
    
    # coherence value. Right now we just lock it at 1, might change later
    coh = 1.0
    
    # keep track of spiketimes/bursts of each neuron
    spiketimes = [[] for n in range(num_neurons)]
    isburst = [[] for n in range(num_neurons)]
    
    p = Pool()
    for i,t in enumerate(I_inj[:,0]):
        if single_comp:
            update = p.map(partial(sim_neuron_current_injections_1comp, neurons = neurons, coh = coh, control = control, I_s_bg = I_s_bg, I_d_bg = I_d_bg, t = t, dt = dt, I_inj = I_inj, model_params = model_params, nonlin = nonlin), np.arange(num_neurons))
            # when we run a single compartment neuron, we also return the actual current injected into the soma
            neuron_I_update = [u[1] for u in update]
            update = [u[0] for u in update]
        else:
            update = p.map(partial(sim_neuron_current_injections, neurons = neurons, coh = coh, control = control, I_s_bg = I_s_bg, I_d_bg = I_d_bg, t = t, dt = dt, I_inj = I_inj, model_params = model_params), np.arange(num_neurons))
            # update = sim_neuron_current_injections(0, neurons, coh, control, I_s_bg, I_d_bg, t, dt, I_inj, model_params = model_params)
            # update = update.reshape((update.shape[0],1)).T

        # update values
        update = np.transpose(update)
        neurons[1:4, :] = update[1:] # update w, v_d, w_d
        neurons[0, :] = update[0]
        
        # update values if spike
        spike = np.array([v > model_params['V_T'] for v in neurons[0, :]])
        neurons[0, :] = np.multiply(spike, base) + np.multiply(~spike, neurons[0, :])
        neurons[1, :] = neurons[1, :] + np.multiply(spike, adap_curr)
        neurons[4, :] = np.multiply(~spike, neurons[4, :])
        neurons[5, :] = np.multiply(~spike, neurons[5, :])
        
        # record spike times
        for j,s in enumerate(spike):
            if s:
                spiketimes[j].append(t)
        
        
        # keep track of the number of spikes fired
        total_spikes += np.sum(spike)
    
        # update timers
        neurons[4, :] = neurons[4, :] + time_upd
        neurons[5, :] = neurons[5, :] + time_upd
    
        # update background input
        I_s_bg = upd_bg_input(I_s_bg, dt, sigma_s)
        I_d_bg = upd_bg_input(I_d_bg, dt, sigma_d)
        
        # voltage of neuron 0
        v[i,:] = neurons[0, :]
        v[i,spike] = 20
        v_d[i,:] = neurons[2,:]
        spikes[i,:] = spike
        if single_comp:
            neuron_I[i,:] = neuron_I_update
        
        # exponential decay
        can = can * (1 - dt/tau_)
        
        # landmark input
        can = can + spike
        # temp = can * (can > thres) # threshold
        thresh = 1.75
        temp = can * (2 / (1 + np.exp(-30*(can-thresh)))) # threshold

        for n in range(num_neurons):
            s = temp[n]
            lmd_strength = min(w * s, 1)
            # f = lmd_strength * (lmd_rfs[n] - can_locs[t])
            force[i] += lmd_strength
     
    p.close()
    p.join()
            
    # run through the spiking activity of each neuron and detect bursts
    t_vec = I_inj[:,0] # vector with timestamps
    spikes_all = []
    bursts_all = []
    spikes_all_all = [] # intuitive naming, I know...
    for i,sp in enumerate(spikes.T):
        t_spikes = np.zeros((0,3)) # keep track of individual spikes
        t_bursts = np.zeros((0,4)) # keep track of bursts
        t_spikes_all = np.zeros((0,3)) # keep track of all spikes, regardless of whether they are in a burst or not
        cur_neuron = np.vstack((np.arange(0,sp.shape[0],1),t_vec,sp)).T # create matrix where we have index and timestamps of spikes for each neuron
        cur_spikes = cur_neuron[cur_neuron[:,2]>0,:]
        spike_ISI = np.diff(cur_spikes[:,1])
        isburst = np.hstack((False, spike_ISI<burst_ISI_thresh))
        
        # pull out times of spikes and bursts. Only count bursts with 3 or more spikes, otherwise record them as individual spikes
        curburst = 0
        burst_id = 0    
        curburst_spikes = np.zeros((0,4))
        for j,ib in enumerate(isburst):
            if not ib:
                # only consider spikes as a burst if at least 3 consecutive spikes are below the threshold
                if curburst >= 3:
                    t_bursts = np.vstack((t_bursts,curburst_spikes))
                    t_spikes_all = np.vstack((t_spikes_all,curburst_spikes[:,:3]))
                    curburst_spikes = np.zeros((0,4))
                    curburst = 0
                    burst_id = burst_id + 1
                elif curburst > 0:
                    t_spikes = np.vstack((t_spikes,curburst_spikes[:,:3]))
                    t_spikes_all = np.vstack((t_spikes_all,curburst_spikes[:,:3]))
                    curburst_spikes = np.zeros((0,4))
                    curburst = 0
                    
                t_spikes = np.vstack((t_spikes,cur_spikes[j,:]))
                t_spikes_all = np.vstack((t_spikes_all,cur_spikes[j,:]))
            else:
                curburst = curburst + 1
                curburst_spikes = np.vstack((curburst_spikes, np.hstack((cur_spikes[j,:],burst_id))))
            
        # this final block just makes sure that the last burst in a simulation gets counted too
        if curburst >= 3:
            t_bursts = np.vstack((t_bursts,curburst_spikes))
            t_spikes_all = np.vstack((t_spikes_all,curburst_spikes[:,:3]))
            curburst_spikes = np.zeros((0,4))
            curburst = 0
            burst_id = burst_id + 1
        elif curburst > 0:
            t_spikes = np.vstack((t_spikes,curburst_spikes[:,:3]))
            t_spikes_all = np.vstack((t_spikes_all,curburst_spikes[:,:3]))
            curburst_spikes = np.zeros((0,4))
            curburst = 0
            
        spikes_all.append(t_spikes)
        bursts_all.append(t_bursts)
        spikes_all_all.append(t_spikes_all)
                    
    
    fig = plt.figure(figsize=(10,15))
    gs = fig.add_gridspec(22, 4)
    ax3 = fig.add_subplot(gs[4:7,:])
    ax2 = fig.add_subplot(gs[0:2,:])
    ax1 = fig.add_subplot(gs[2:4,:])
    ax4 = fig.add_subplot(gs[7:10,:])
    ax5 = fig.add_subplot(gs[10:13,:])
    ax6 = fig.add_subplot(gs[13:16,:])
    ax7 = fig.add_subplot(gs[16:19,:])
    ax8 = fig.add_subplot(gs[19:22,0:1])
    ax9 = fig.add_subplot(gs[19:22,1:2])
    
    if single_comp:
        # for n_I in neuron_I.T:
        ax1.plot(I_inj[:,0], np.mean(neuron_I,1))
    else:
        for v_n in v.T:
            ax1.plot(I_inj[:,0], v_n)
        
    # create bins to count spikes and bursts for a given amount of time
    hist_bin_edges = np.arange(0,total_time + temporal_binning, temporal_binning)
    spike_count = np.zeros((0,len(hist_bin_edges)-1))
    burst_count = np.zeros((0,len(hist_bin_edges)-1))
    spike_all_count = np.zeros((0,len(hist_bin_edges)-1))
    spikes_in_bursts_count = np.zeros((0,len(hist_bin_edges)-1))
    spikes_non_bursts_count = np.zeros((0,len(hist_bin_edges)-1))
    
    inj_I_binned,_,inj_I_idx = binned_statistic(I_inj[:,0],I_inj[:,1],'mean',bins=hist_bin_edges)
    for i,s_t in enumerate(spikes.T):
        x_coord = I_inj[s_t > 0,0]
        y_coord = s_t[s_t > 0] * i
        ax3.scatter(x_coord,y_coord, marker='|', c='b' )
        
        # calculate spikes per temporal bin
        if len(spikes_all[i]) > 0:
            ax3.scatter(spikes_all[i][:,1],spikes_all[i][:,2] * i, marker='|', c='k' )
            spike_hist,_ = np.histogram(spikes_all[i][:,1], bins=hist_bin_edges)
            spike_count = np.vstack((spike_count, spike_hist))
        
        # calculate bursts per temporal bin
        if len(bursts_all[i]) > 0:
            ax3.scatter(bursts_all[i][:,1],bursts_all[i][:,2] * i, marker='|', c='r' )
            # the abomination of a line below just pulls out the line of the first spike in each burst
            unique_bursts = bursts_all[i][np.hstack((True,np.diff(bursts_all[i][:,3])>0)),:]
            burst_hist,_ = np.histogram(unique_bursts[:,1], bins=hist_bin_edges)
            burst_count = np.vstack((burst_count, burst_hist))
        
        # calculate total spikes, non-burst spikes, and spikes in bursts per temporal bin
        if len(spikes_all_all[i]) > 0:
            burst_hist,_ = np.histogram(bursts_all[i][:,1], bins=hist_bin_edges)
            spike_hist,_ = np.histogram(spikes_all[i][:,1], bins=hist_bin_edges)
            spikes_in_bursts_count = np.vstack((spikes_in_bursts_count, burst_hist))
            spikes_non_bursts_count = np.vstack((spikes_non_bursts_count, spike_hist))
            
            spike_all_hist,_ = np.histogram(spikes_all_all[i][:,1], bins=hist_bin_edges)
            spike_all_count = np.vstack((spike_all_count, spike_all_hist))
    
    x_coords = np.arange(temporal_binning/2,total_time + temporal_binning/2, temporal_binning)
    mean_spikes = np.mean(spike_count,0) * 10
    spike_count_sem = np.std(spike_count,0) / np.sqrt(spike_count.shape[0]) * 10
    mean_bursts = np.mean(burst_count,0) * 10
    burst_count_sem = np.std(burst_count,0) / np.sqrt(burst_count.shape[0]) * 10
    
    ax4.plot(x_coords,mean_spikes)
    ax4.fill_between(x_coords, mean_spikes - spike_count_sem, mean_spikes + spike_count_sem, alpha = .2)
    ax5.plot(x_coords, mean_bursts)
    ax5.fill_between(x_coords, mean_bursts - burst_count_sem, mean_bursts + burst_count_sem, alpha = .2)
    
    ax4.set_ylabel("spike rate (Hz)")
    ax5.set_ylabel("burst rate (Hz)")
    
    ax6.plot(v[:,0])
    if not single_comp:
        ax6.plot(v_d[:,0])
    
    # ax4.set_ylim([0,1])
    # ax5.set_ylim([0,1])
    
    spike_all_count[spike_all_count==0] = 1 # set zeros to ones to avoid division by zero
    frac_burst_spikes = np.divide(spikes_in_bursts_count, spike_all_count)
    frac_burst_spikes[np.isnan(frac_burst_spikes)] = 0
    ax8.plot(np.arange(0,len(hist_bin_edges)-1,1),np.mean(frac_burst_spikes,0))
      
    ax3.set_ylim([-1,spikes.shape[1]])
    
    ax2.plot(I_inj[:,0], I_inj[:,1], c='#EC008C', ls='-')
    # ax2_2 = ax2.twinx()
    ax2.plot(I_inj[:,0], I_inj[:,2], c='#EC008C', ls='--')
    
    ax1.set_xlim([0,total_time])
    ax2.set_xlim([0,total_time])
    # ax2_2.set_xlim([0,total_time])
    ax3.set_xlim([0,total_time])
    
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])
    
    ax7.plot(I_inj[:,0],force, c='lavender')
    
    # if fname_suffix == 'ramp_real' or fname_suffix == 'ramp_control':
    #     ax4.set_ylim([0,50])
    #     ax5.set_ylim([0,6])
    
    sns.despine(fig=fig, top=True, right=True, left=False, bottom=False)
    sns.despine(ax=ax1, bottom=True)
    sns.despine(ax=ax2, bottom=True)
    # sns.despine(ax=ax2_2, bottom=True)
    sns.despine(ax=ax3, bottom=True)
    sns.despine(ax=ax4, bottom=True)
    sns.despine(ax=ax5, bottom=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'single_neuron_trace_' + fname_suffix + '.svg'
    plt.savefig(fname, dpi=300)
    print('saved: ' + fname)
    
    return spike_count*(1000/temporal_binning), burst_count*(1000/temporal_binning), frac_burst_spikes, inj_I_binned
    
def run_step_currents():  
    num_neurons = 10
    # set up timesteps
    total_time = 5000 # ms
    dt = 0.5 # ms
    
    # set up injection current. This is where we define the actual pattern of injections into soma (and dendrite)
    I_inj = np.zeros((int(total_time/dt),3))
    I_inj[:,0] = np.linspace(0,total_time,num=int(total_time/dt), endpoint=False)
    I_inj[:,1] = 300
    I_inj[:,2] = 100
    I_inj[500:1500,1] = 350 #pa
    I_inj[1000:2000,2] = 250 #pa
    
    I_inj[3000:4000,1] = 350 #pa
    I_inj[3500:4500,2] = 250 #pa
    
    I_inj[5500:6500,1] = 350 #pa
    I_inj[6000:7000,2] = 250 #pa
    
    neuron_current_injections(num_neurons, total_time, dt, I_inj)
    
def run_single_neuron_step_current():  
    num_neurons = 1
    # set up timesteps
    total_time = 150 # ms
    dt = 0.5 # ms
    
    
    I_inj = np.zeros((int(total_time/dt),3))
    I_inj[:,0] = np.linspace(0,total_time,num=int(total_time/dt), endpoint=False)
    I_inj[:,1] = 0
    I_inj[:,2] = 0
    
    # set up injection current. This is where we define the actual pattern of injections into soma (and dendrite)
    
    # I_inj[100:115,1] = 1500 #pa
    # I_inj[115:130,2] = 300 #pa
    # I_inj[130:145,1] = 1500 #pa
    # I_inj[145:160,2] = 300 #pa
    
    I_inj[100:170,1] = 800 #pa
    I_inj[200:270,2] = 500 #pa
    
    control = False
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'control')
    
    I_inj[:,1] = 0
    I_inj[:,2] = 0
    # I_inj[100:115,1] = 1500 #pa
    # I_inj[100:115,2] = 300 #pa
    # I_inj[130:145,1] = 1500 #pa
    # I_inj[130:145,2] = 300 #pa
    
    I_inj[100:170,1] = 800 #pa
    I_inj[100:170,2] = 500 #pa
    control = False
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'real')
    
def run_single_neuron_step_current_1comp():  
    num_neurons = 1
    # set up timesteps
    total_time = 200 # ms
    dt = 0.5 # ms
    
    
    I_inj = np.zeros((int(total_time/dt),3))
    I_inj[:,0] = np.linspace(0,total_time,num=int(total_time/dt), endpoint=False)
    I_inj[:,1] = 0
    I_inj[:,2] = 0
    
    # set up injection current. This is where we define the actual pattern of injections into soma (and dendrite)
    
    I_inj[100:200,1] = 4000 #pa
    # I_inj[130:145,1] = 2000 #pa
    
    control = False
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'control', single_comp = True, nonlin = False)
    
    control = False
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'control', single_comp = True, nonlin = True)
    
    # I_inj[:,1] = 0
    # I_inj[:,2] = 0
    # I_inj[100:115,1] = 1500 #pa
    # I_inj[100:115,2] = 300 #pa
    # I_inj[130:145,1] = 1500 #pa
    # I_inj[130:145,2] = 300 #pa
    # control = False
    # neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'real')

def run_comp_single_multi_comp_neurons():
    # set up timesteps and neurons
    num_neurons = 1
    total_time = 135 # ms
    dt = 0.5 # ms
    
    I_inj = np.zeros((int(total_time/dt),3))
    I_inj[:,0] = np.linspace(0,total_time,num=int(total_time/dt), endpoint=False)
    I_inj[:,1] = 0
    I_inj[:,2] = 0
    
    I_inj[100:130,1] = 1500 #pa
    I_inj[100:130,2] = 300 #pa

    control = True
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'control_mcomp', single_comp = False, nonlin = False)
    
    control = False
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'real_mcomp', single_comp = True, nonlin = False)
    
    I_inj[100:130,1] = 1800 #pa
    # I_inj[130:145,1] = 2000 #pa
    
    control = False
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'supralin_1comp', single_comp = True, nonlin = True)
    neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'linear_1comp', single_comp = True, nonlin = False)

    
    
def run_current_ramp():      
    num_neurons = 200
    # set up timesteps
    total_time = 2000 # ms
    dt = 0.5 # ms

    # set up injection current. This is where we define the actual pattern of injections into soma (and dendrite)
    I_inj = np.zeros((int(total_time/dt),3))
    I_inj[:,0] = np.linspace(0,total_time,num=int(total_time/dt), endpoint=False)
    I_inj[:,1] = 0
    I_inj[:,2] = 0
    I_inj[600:3600,2] = 150
    I_inj[600:3600,1] = np.linspace(0,1200,3000) #pa
    # I_inj[500:3500,2] = np.linspace(0,1000,3000) #pa
    control = False
    spikes_sd, bursts_sd, frac_burst_spikes_real, inj_I_binned_real = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'ramp_real', (350,75))
    mean_spikes_sd = np.mean(spikes_sd,0)
    mean_bursts_sd = np.mean(bursts_sd,0)
    
    control = True
    spikes_ctl, bursts_ctl, frac_burst_spikes_ctl, inj_I_binned_ctl = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'ramp_control', (350,75))
    mean_spikes_ctl = np.mean(spikes_ctl,0)
    mean_bursts_ctl = np.mean(bursts_ctl,0)
    
    fig = plt.figure(figsize=(3,12))
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    
    ax1.plot(mean_spikes_sd, c='k')
    ax1.plot(mean_spikes_ctl, c='k', ls='--')
    
    ax2.plot(mean_bursts_sd, c='r')
    ax2.plot(mean_bursts_ctl, c='r', ls='--')
     
    ax3.plot(inj_I_binned_real[0:18], np.mean(frac_burst_spikes_ctl,0)[0:18], c='r', ls='--', label='2-comp, control')
    ax3.plot(inj_I_binned_real[0:18], np.mean(frac_burst_spikes_real,0)[0:18], c='r', label='2-comp, soma+dendrite')
    
    sd_mean = np.mean(frac_burst_spikes_ctl[:,0:18],0)
    sd_sem = np.std(frac_burst_spikes_ctl[:,0:18],0) / np.sqrt(frac_burst_spikes_ctl.shape[0]) 
    
    nonlin_mean = np.mean(frac_burst_spikes_real[:,0:18],0)
    nonlin_sem = np.std(frac_burst_spikes_real[:,0:18],0) / np.sqrt(frac_burst_spikes_real.shape[0]) 
    
    ax3.fill_between(inj_I_binned_real[0:18], sd_mean - sd_sem, sd_mean + sd_sem, color='r', alpha = .2, linewidth=0)
    ax3.fill_between(inj_I_binned_real[0:18], nonlin_mean - nonlin_sem, nonlin_mean + nonlin_sem, color='r', alpha = .2, linewidth=0)
    ax3.legend()
    
    ax3.set_xticks([0,400,800,1200])
    ax3.set_xticklabels(['0','400','800','1200'])
    
    ax3.set_ylim([0,1])
    ax3.set_yticks([0, 0.5 ,1])
    ax3.set_yticklabels(['0', '0.5' ,'1.0'])
    
    ax4.plot(inj_I_binned_real[0:18], np.mean(spikes_sd,0)[0:18], c='k', label='2-comp, soma+dend')
    ax4.plot(inj_I_binned_real[0:18], np.mean(spikes_ctl,0)[0:18], c='k', ls='--', label='2-comp, control')
    
    sd_mean = np.mean(spikes_sd[:,0:18],0)
    sd_sem = np.std(spikes_sd[:,0:18],0) / np.sqrt(spikes_sd.shape[0]) 
    
    nonlin_mean = np.mean(spikes_ctl[:,0:18],0)
    nonlin_sem = np.std(spikes_ctl[:,0:18],0) / np.sqrt(spikes_ctl.shape[0]) 
    
    ax4.fill_between(inj_I_binned_real[0:18], sd_mean - sd_sem, sd_mean + sd_sem, color='k', alpha = .2, linewidth=0)
    ax4.fill_between(inj_I_binned_real[0:18], nonlin_mean - nonlin_sem, nonlin_mean + nonlin_sem, color='k', alpha = .2, linewidth=0)
    ax4.legend()
    
    ax4.set_xticks([0,400,800,1200])
    ax4.set_xticklabels(['0','400','800','1200'])
    
    ax4.set_ylim([0,50])
    ax4.set_yticks([0, 25 ,50])
    ax4.set_yticklabels(['0', '25' ,'50'])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    sns.despine(ax=ax4, right=True, top=True)
    
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace' + os.sep + 'ramp_comparison.svg'
    plt.savefig(fname, dpi=300)

def run_current_ramp_1comp():      
    num_neurons = 3
    # set up timesteps
    total_time = 2000 # ms
    dt = 0.5 # ms

    # set up injection current. This is where we define the actual pattern of injections into soma (and dendrite)
    I_inj = np.zeros((int(total_time/dt),3))
    I_inj[:,0] = np.linspace(0,total_time,num=int(total_time/dt), endpoint=False)
    I_inj[:,1] = 0
    I_inj[:,2] = 0
    I_inj[600:3600,1] = np.linspace(0,1200,3000) #pa
    # I_inj[500:3500,2] = np.linspace(0,1000,3000) #pa
    control = False
    mean_spikes_nonlin, mean_bursts_nonlin,_,_ = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, '1comp_ramp_control', (350,75), single_comp = True, nonlin = True)
    mean_spikes_lin, mean_bursts_lin,_,_ = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, '1comp_ramp_real', (350,75), single_comp = True, nonlin = False)    
    mean_spikes_nonlin = np.mean(mean_spikes_nonlin,0)
    mean_bursts_nonlin = np.mean(mean_bursts_nonlin,0)
    
    
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    ax1.plot(mean_spikes_nonlin, c='k')
    ax1.plot(mean_spikes_lin, c='k', ls='--')
    
    ax2.plot(mean_bursts_nonlin, c='r')
    ax2.plot(mean_bursts_lin, c='r', ls='--')
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'single_neuron_trace_1comp')
    fname = loc_info['figure_output_path'] + os.sep + 'single_neuron_trace_1comp' + os.sep + 'ramp_comparison.svg'
    plt.savefig(fname, dpi=300)

def run_comp_current_ramp():
    # set up timesteps
    num_neurons = 10
    total_time = 2000 # ms
    dt = 0.5 # ms

    # set up injection current. This is where we define the actual pattern of injections into soma (and dendrite)
    I_inj = np.zeros((int(total_time/dt),3))
    I_inj[:,0] = np.linspace(0,total_time,num=int(total_time/dt), endpoint=False)
    I_inj[:,1] = 0
    I_inj[:,2] = 150
    I_inj[600:3600,1] = np.linspace(0,1200,3000) #pa
    # I_inj[500:3500,2] = np.linspace(0,1000,3000) #pa
    
    control = False
    mean_spikes_sd, mean_bursts_sd, frac_burst_spikes_sd, inj_I_binned_sd = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'ramp_real', (350,50))
    mean_spikes_sd = np.mean(mean_spikes_sd,0)
    mean_bursts_sd = np.mean(mean_bursts_sd,0)
    control = True
    spikes_ctl, bursts_ctl, frac_burst_spikes_ctl, inj_I_binned_ctl = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, 'ramp_control', (350,50))
    mean_spikes_ctl = np.mean(spikes_ctl,0)
    mean_bursts_ctl = np.mean(bursts_ctl,0)
    
    control = False
    mean_spikes_nonlin, mean_bursts_nonlin, frac_burst_spikes_nonlin, inj_I_binned_nonlin = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, '1comp_ramp_control', (350,50), single_comp = True, nonlin = True)
    mean_spikes_nonlin = np.mean(mean_spikes_nonlin,0)
    mean_bursts_nonlin = np.mean(mean_bursts_nonlin,0)
    
    spikes_lin, bursts_lin, frac_burst_spikes_lin, inj_I_binned_lin = neuron_current_injections(num_neurons, total_time, dt, I_inj, control, '1comp_ramp_real', (350,50), single_comp = True, nonlin = False)    
    mean_spikes_lin = np.mean(spikes_lin,0)
    mean_bursts_lin = np.mean(bursts_lin,0)
    
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    
    ax1.plot(mean_spikes_ctl, c='k', label='2-comp, control')
    ax1.plot(mean_spikes_lin, c='k', ls='--', label='1-comp, linear')
    ax1.legend()
    
    ax2.plot(mean_spikes_sd, c='k', label='2-comp, soma+dend')
    ax2.plot(mean_spikes_nonlin, c='k', ls='--', label='1-comp, supralinear')
    ax2.legend()
    
    ax3.plot(mean_bursts_ctl, c='r', label='2-comp, control')
    ax3.plot(mean_bursts_lin, c='r', ls='--', label='1-comp, linear')
    ax3.legend()
    
    ax4.plot(mean_bursts_sd, c='r', label='2-comp, control')
    ax4.plot(mean_bursts_nonlin, c='r', ls='--', label='1-comp, linear')
    ax4.legend()
    
    ax5.plot(inj_I_binned_sd[0:18], np.mean(frac_burst_spikes_sd,0)[0:18], c='r', label='2-comp, soma+dend')
    ax5.plot(inj_I_binned_sd[0:18], np.mean(frac_burst_spikes_nonlin,0)[0:18], c='r', ls='--', label='1-comp, supralinear')
    
    sd_mean = np.mean(frac_burst_spikes_sd[:,0:18],0)
    sd_sem = np.std(frac_burst_spikes_sd[:,0:18],0) / np.sqrt(frac_burst_spikes_sd.shape[0]) 
    
    nonlin_mean = np.mean(frac_burst_spikes_nonlin[:,0:18],0)
    nonlin_sem = np.std(frac_burst_spikes_nonlin[:,0:18],0) / np.sqrt(frac_burst_spikes_nonlin.shape[0]) 
    
    ax5.fill_between(inj_I_binned_sd[0:18], sd_mean - sd_sem, sd_mean + sd_sem, color='r', alpha = .2, linewidth=0)
    ax5.fill_between(inj_I_binned_sd[0:18], nonlin_mean - nonlin_sem, nonlin_mean + nonlin_sem, color='r', alpha = .2, linewidth=0)
    ax5.legend()
    
    ax5.set_ylim([0,0.5])
    ax5.set_yticks([0, 0.25, 0.5])
    ax5.set_yticklabels(['0', '0.25', '0.5'])
    
    ax5.set_xticks([0,400,800,1200])
    ax5.set_xticklabels(['0','400','800','1200'])
    
    ctl_mean = np.mean(spikes_ctl,0)[0:18]
    ctl_sem = np.std(spikes_ctl[:,0:18],0) / np.sqrt(spikes_ctl.shape[0]) 
    
    lin_mean = np.mean(spikes_lin,0)[0:18]
    lin_sem = np.std(spikes_lin[:,0:18],0) / np.sqrt(spikes_lin.shape[0]) 
    
    ax6.plot(inj_I_binned_sd[0:18], ctl_mean, c='k', label='2-comp, soma')
    ax6.plot(inj_I_binned_sd[0:18], lin_mean, c='k', ls='--', label='1-comp, linear')
    
    ax6.fill_between(inj_I_binned_sd[0:18], ctl_mean - ctl_sem, ctl_mean + ctl_sem, color='k', alpha = .2, linewidth=0)
    ax6.fill_between(inj_I_binned_sd[0:18], lin_mean - lin_sem, lin_mean + lin_sem, color='k', alpha = .2, linewidth=0)
    ax6.legend()
    
    ax6.set_ylim([0,50])
    ax6.set_yticks([0, 25, 50])
    ax6.set_yticklabels(['0', '25', '50'])
    
    ax6.set_xticks([0,400,800,1200])
    ax6.set_xticklabels(['0','400','800','1200'])
    
    sns.despine(ax=ax1, right=True, top=True)
    sns.despine(ax=ax2, right=True, top=True)
    sns.despine(ax=ax3, right=True, top=True)
    sns.despine(ax=ax4, right=True, top=True)
    sns.despine(ax=ax5, right=True, top=True)
    sns.despine(ax=ax6, right=True, top=True)
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'ramp_1comp_comparison')
    fname = loc_info['figure_output_path'] + os.sep + 'ramp_1comp_comparison' + os.sep + 'ramp_comparison.svg'
    plt.savefig(fname, dpi=300)

    
    # ax5.set_ylim([0,0.5])

def run_single_comp():
    MOUSE = 'LF191022_1'
    SESSION = '20191213'
    
    ol = False
    no_spatial = False
    control = False
    sigma = (300,50)
    show_plot = True
    trial_type = 'short'
    thresh = 1.75
    num_neurons_range = [100]
    select_trials = np.arange(3)
    
    for num_neurons in num_neurons_range:
        
        # landmarks = random_landmarks(num_neurons, 'gaussian')
        # landmarks = np.ones((num_neurons)) * 220
        landmarks = random_landmarks(num_neurons, 'gaussian', 220, 60)
        
        # linear integration
        make_folder(loc_info['figure_output_path'] + os.sep + 'single_comp_l_behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        print(num_neurons)
        
        print(np.sort(landmarks))
        nonlin = False
        subfolder = 'single_comp_l'
        init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, thresh=thresh, select_trials=select_trials, landmarks=landmarks, subfolder = subfolder, single_comp = True, nonlin = nonlin)
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)  
        for i in range(len(init_error)):
            ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        fname = loc_info['figure_output_path'] + os.sep + 'single_comp_l_behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "error"+ fformat
        plt.savefig(fname, dpi=300)
        plt.close()
        print("saved: " + fname)
        
        #supralinear integration
        # make_folder(loc_info['figure_output_path'] + os.sep + 'single_comp_s_behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        # nonlin = True
        # subfolder = 'single_comp_s'
        # init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, thresh=thresh, select_trials=select_trials, landmarks=landmarks, subfolder = subfolder, single_comp = True, nonlin = nonlin)
        
        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(1,1,1)  
        # for i in range(len(init_error)):
        #     ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        # fname = loc_info['figure_output_path'] + os.sep + 'single_comp_s_behav_trials_' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "error"+ fformat
        # plt.savefig(fname, dpi=300)
        # plt.close()
        # print("saved: " + fname)

def run_single_comp_openloop_l():
    MOUSE = 'LF191022_1'
    SESSION = ['20191209_ol','20191215_ol']#['20191204_ol','20191213_ol']#['20191207_ol']#['20191207_ol'] #,,
    subfolder = 'single_comp_l_ol'
    ol = True
    no_spatial = False
    control = False
    sigma = (300,50)
    show_plot = True
    trial_type = 'ol_fast'
    num_neurons_range = [100]
    select_trials = np.arange(40)
    trial_nr_offset = 40
    
    for num_neurons in num_neurons_range:
        
        # landmarks = random_landmarks(num_neurons, 'gaussian')
        # landmarks = np.ones((num_neurons)) * 220
        landmarks = random_landmarks(num_neurons, 'gaussian', 220, 60)
        
        # linear integration
        make_folder(loc_info['figure_output_path'] + os.sep + 'single_comp_ol_l_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        print(num_neurons)
        
        print(np.sort(landmarks))
        nonlin = False
        init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = subfolder, single_comp = True, nonlin = nonlin, trial_nr_offset=trial_nr_offset)
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)  
        for i in range(len(init_error)):
            ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        fname = loc_info['figure_output_path'] + os.sep + 'single_comp_ol_l_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "error"+ fformat
        plt.savefig(fname, dpi=300)
        plt.close()
        print("saved: " + fname)
        
        #supralinear integration
        # make_folder(loc_info['figure_output_path'] + os.sep + 'single_comp_ol_s_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        # nonlin = True
        # subfolder = 'single_comp_s'
        # init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = subfolder, single_comp = True, nonlin = nonlin)
        
        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(1,1,1)  
        # for i in range(len(init_error)):
        #     ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        # fname = loc_info['figure_output_path'] + os.sep + 'single_comp_ol_s_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "error"+ fformat
        # plt.savefig(fname, dpi=300)
        # plt.close()
        # print("saved: " + fname)

def run_single_comp_openloop_s():
    MOUSE = 'LF191022_1'
    SESSION = ['20191209_ol','20191215_ol']#['20191204_ol','20191213_ol']#['20191207_ol']#['20191207_ol'] #,,
    subfolder = 'single_comp_s_ol'
    ol = True
    no_spatial = False
    control = False
    sigma = (300,50)
    show_plot = True
    trial_type = 'ol_fast'
    num_neurons_range = [100]
    select_trials = np.arange(40)
    trial_nr_offset = 40
    
    for num_neurons in num_neurons_range:
        
        # landmarks = random_landmarks(num_neurons, 'gaussian')
        # landmarks = np.ones((num_neurons)) * 220
        landmarks = random_landmarks(num_neurons, 'gaussian', 220, 60)
        
        # linear integration
        # make_folder(loc_info['figure_output_path'] + os.sep + 'single_comp_ol_l_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        # print(num_neurons)
        
        print(np.sort(landmarks))
        # nonlin = False
        # subfolder = 'single_comp_l'
        # init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = subfolder, single_comp = True, nonlin = nonlin)
        
        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(1,1,1)  
        # for i in range(len(init_error)):
        #     ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        # fname = loc_info['figure_output_path'] + os.sep + 'single_comp_ol_l_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "error"+ fformat
        # plt.savefig(fname, dpi=300)
        # plt.close()
        # print("saved: " + fname)
        
        #supralinear integration
        make_folder(loc_info['figure_output_path'] + os.sep + 'single_comp_ol_s_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise')
        nonlin = True
        init_error, final_error = sim_mouse_behavior(MOUSE=MOUSE, SESSION=SESSION, ol=ol, no_spatial=no_spatial, control=control, sigma=sigma, show_plot=show_plot, trial_type=trial_type, num_neurons=num_neurons, select_trials=select_trials, landmarks=landmarks, subfolder = subfolder, single_comp = True, nonlin = nonlin, trial_nr_offset=trial_nr_offset)
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)  
        for i in range(len(init_error)):
            ax.plot([0,1],[np.abs(init_error),np.abs(final_error)], c='k', lw='2', marker='o')
        
        fname = loc_info['figure_output_path'] + os.sep + 'single_comp_ol_s_behav_trials' + str(num_neurons) + 'neurons_' + str(sigma) + 'noise' + os.sep + "error"+ fformat
        plt.savefig(fname, dpi=300)
        plt.close()
        print("saved: " + fname)
        

def landmark_distribution():
    fig = plt.figure(figsize=(2.5,3))
    ax = fig.add_subplot(1,1,1)  
    landmarks = random_landmarks(100, 'gaussian', 220, 60)
    ax.hist(landmarks,  cumulative=True,histtype='step' , color='r', linewidth=2) #
    ax.set_xticks([0,100,200,300,400])
    ax.set_xticklabels([0,100,200,300,400])
    ax.set_xlabel('Location (cm)')
    ax.set_ylabel('Neuron distribution (cum.)')
    
    ax.set_xlim([0,350])
    
    sns.despine(ax=ax, right=True, top=True)
    plt.tight_layout()
    
    make_folder(loc_info['figure_output_path'] + os.sep + 'Landmark neuron distribution')
    fname = loc_info['figure_output_path'] + os.sep + 'Landmark neuron distribution' + os.sep + "Landmark neuron distribution"+ fformat
    plt.savefig(fname, dpi=300)
    print('saved ' + fname)

if __name__ == '__main__':
    # run_nolm_sim()
    # run_spikerate_burstrate()
    # run_neuron_range_simulation()
    # run_sim_mouse_behavior()
    # run_neuron_step_currents()
    
    # run_neuron_n_range()
    # run_sigma_range()
    # run_uniform_vs_gaussian()
    # run_step_currents()
    # run_single_neuron_step_current()
    # run_thresh_range()
    
    # run_single_neuron_step_current_1comp()
    
    
    # landmark_distribution()
    
    
    # run_current_ramp_1comp()
    
    # run_comp_single_multi_comp_neurons()
    
    run_current_ramp()
    # run_comp_current_ramp()
    
    # run_single_comp()
    
    # run_sim_with_control()
    # run_openloop()
    # run_single_comp_openloop_l()
    # run_single_comp_openloop_s()
    
    # I_bg = np.zeros(100)
    # for n in np.arange(1, len(I_bg)):
    #     I_bg[n] = upd_bg_input(I_bg[n-1], 1, 200)
        
    # plt.plot(np.arange(100), I_bg)
    # plt.show()
    # pdb.set_trace()
    
    # random.seed(2020)
    
    # random.seed(123)
    # random.seed(234)
    
    