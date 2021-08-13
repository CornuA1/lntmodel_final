#!/usr/bin/env python
# coding: utf-8

# In[19]:


from brian2 import *
get_ipython().run_line_magic('matplotlib', 'inline')

# variables
tau_s = 16*ms
tau_d = 7*ms
C_s = 370*pF
C_d = 170*pF
E_L = -70*mV
V_T = -50*mV
tau_ws = 100*ms
tau_wd = 30.0*ms
a_wd = -13*nS
b_ws = -200*pA
c_d = 2600*pA
g_s = 1300*pA
g_d = 1200*pA
E_d = -38*mV
D_d = 6*mV

# input
input_eqs = '''
I_s : amp
I_d: amp
'''

# equations

soma_eqs = '''
dv/dt = -(v-E_L)/tau_s + (g_s*f + I_s + w_s)/C_s : volt (unless refractory)
dw_s/dt = - w_s/tau_ws : amp
'''
dend_eqs = '''
dv_d/dt = -(v_d-E_L)/tau_d + (g_d*f + c_d*K + I_d + w_d)/C_d : volt
dw_d/dt = (-w_d + a_wd *(v_d-E_L))/tau_wd : amp
'''
coupling_eq = '''
f = 1/(1+exp(-(v_d-E_d)/D_d)) : 1
'''
# K is a rectangular kernel of amplitude 1 from 0.5-2.5 ms after a spike
# int converts boolean to either 0 or 1, timestep prevents floating point impreciseness
# t0 and t1 are the times of the last 2 spikes, which is needed since the refractory period is 2ms and there is a possibility of overlap 
bap_eqs = '''
t0 : second
t1 : second
K = int(timestep(t-t0,dt) >= timestep(.5*ms,dt))*int(timestep(t-t0, dt)<= timestep(2.5*ms, dt)) + int(timestep(t-t1,dt)>= timestep(.5*ms,dt))*int(timestep(t-t1, dt)<= timestep(2.5*ms,dt)) : 1
bAP = c_d*K : amp
'''

eqs = input_eqs + soma_eqs + dend_eqs + coupling_eq + bap_eqs
# spike triggered adaptation current included in reset_eqs

reset_eqs = '''
v = E_L
w_s += b_ws
t1 = t0
t0 = t

'''

# neuron
G = NeuronGroup(1, eqs, threshold = 'v > V_T', reset = reset_eqs, refractory = 2*ms, method = 'euler')

# initial conditions
G.v = E_L
G.v_d = E_L
G.t0 = -3*ms # time of last spike, -3ms is arbitrary, can be anything < -2.5*ms to start
G.t1 = -3*ms # time of second to last spike
G.I_s = 2000*pA
G.I_d = 1000*pA

# monitors
soma_mon = StateMonitor(G, 'v', record = True)
dend_mon = StateMonitor(G, 'v_d', record = True)
adap_i_mon = StateMonitor(G, 'w_s', record = True)
bap_mon = StateMonitor(G, 'bAP', record = True)
t0_mon = StateMonitor(G, 't0', record = True)
t1_mon = StateMonitor(G, 't1', record = True)
spikemon = SpikeMonitor(G) # need to add this to spike train

run(20*ms)
# graph
figure(figsize = (12,4))
# subplot(131)
plot(soma_mon.t/ms, soma_mon.v[0]*10**3)
plot(dend_mon.t/ms, dend_mon.v_d[0]*10**3)
for t in spikemon.t:
    axvline(t/ms, ls = '--', c = 'C1', lw = 3)
xlabel('Time (ms)')
ylabel('v (mV)')
# subplot(132)
#plot(adap_i_mon.t/ms, adap_i_mon.w_s[0])
# plot(bap_mon.t/ms, bap_mon.bAP[0])
# for t in spikemon.t:
#     axvline(t/ms, ls = '--', c = 'C1', lw = 3)
# xlabel('Time (ms)')
# ylabel('Backpropagation current(A)')
# subplot(133)
# plot(t0_mon.t/ms, t0_mon.t0[0])
# plot(t1_mon.t/ms, t1_mon.t1[0])
# for t in spikemon.t:
#     axvline(t/ms, ls = '--', c = 'C1', lw = 3)
# xlabel('Time (ms)')
#ylabel('Adaptation current (A)')

# In[ ]:




