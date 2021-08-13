"""
Create simulated dataset to test analysis functions

NOTE: most of this code is translated from Matlab provided by:
    Lütcke, H., Gerhard, F., Zenke, F., Gerstner, W. & Helmchen, F.
    Inference of neuronal network spike dynamics and topology from calcium imaging data. Front. Neural Circuits 7, 1–20 (2013).

"""

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

def spkTimes2Calcium(spkT,tauOn,ampFast,tauFast,ampSlow,tauSlow,frameRate,duration):
    x1 = np.arange(0,duration+(1/frameRate),1/frameRate)
    y = (1-(np.exp(-(x1-spkT) / tauOn))) * (ampFast * np.exp(-(x1-spkT) / tauFast))+(ampSlow * np.exp(-(x1-spkT) / tauSlow))
     # y = (1-(ex1p(-(x1-spkT)./tauOn))).*(ampFast*ex1p(-(x1-spkT)./tauFast));
    y[x1 < spkT] = 0
    y[np.isnan(y)] = 0

    return y

def PoissonSpikeTrain(rate, dur):
    # Generate Poisson Spike Train with firing rate and duration
    dt = 0.0001
    timestamps = np.arange(0,dur+dt,dt)
    spikeTimes=[]

    # Generating spikes from a exponential distribution
    for ts in timestamps:
        if (rate*dt) >= np.random.rand(1):
            spikeTimes = np.append(spikeTimes,ts)

    return spikeTimes

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def spkTimes2FreeCalcium(spkT,Ca_amp,Ca_gamma,Ca_onsettau,Ca_rest, kappaS, Kd, Conc,frameRate,duration):
    """ CURRENTLY NOT FULLY CONVERTED FROM MATLAB LF06/12/18"""
    pass

    # returns modeled free calcium trace derived from list of spike times (spkT)
    # calculated based on buffered increment Ca_amp and taking indicator
    # binding into account
    #
    # x = np.arange(0,duration+(1/frameRate),1/frameRate)
    # y = np.zeros((len(x)))
    # y.fill(Ca_rest)
    # unfilt = np.zeros((1,length(x)))
    # unfilt.fill(Ca_rest)
    #
    # for i in range(len(spkT)):
    #     if i < len(spkT):
    #         ind = np.where(x >= spkT[i])[0][0]
    #         lastind = np.where(x >= spkT[i+1])[0][0]
    #         if (lastind-ind) <= 2:
    #             lastind = ind+2  # have at least 3 points to process
    #     else:
    #         ind = np.where(x >= spkT(i))[0][0]
    #         lastind = np.where(x >= spkT(i))[0][0]
    #         if (lastind-ind) <= 2:
    #             ind = lastind-2  # have at least 3 points to process
    #
    # tspan = x[ind:lastind]
    #
    # # currentCa = y(ind);
    # currentCa = unfilt(ind);
    #
    # Y0 = currentCa;   # current ca conc following increment due to next spike
    # [~,ylow] = CalciumDecay(Ca_gamma,Ca_rest,Y0, kappaS, Kd, Conc, tspan);   # solving ODE for single comp model
    #
    # kappa = Kd.*Conc./(currentCa+Kd).^2;
    # Y0 = currentCa + Ca_amp./(1+kappaS+kappa);   # current ca conc following increment due to next spike
    # [~,yout] = CalciumDecay(Ca_gamma,Ca_rest,Y0, kappaS, Kd, Conc, tspan);   # solving ODE for single comp model
    #
    # unfilt(ind:lastind) = yout;
    #
    # % now onset filtering with rising exponential
    # % caonset = (1 - exp(-(tspan-tspan(1))./Ca_onsettau));
    # caonset = (1 - exp(-(tspan-spkT(i))./Ca_onsettau));
    # caonset( caonset < 0) = 0;
    # difftmp = yout - ylow;
    # yout = difftmp.*caonset' + ylow;
    #
    # y(ind:lastind) = yout;


if __name__ == "__main__":
    # set up parameters
    S = {
        'ca_genmode' : 'linDFF',
        'spk_recmode' : 'linDFF',
        'ca_onsettau' : 0.02,
        'ca_amp' : 7600,
        'ca_gamma' : 400,
        'ca_amp1' : 0,
        'ca_tau1' : 0,
        'ca_kappas' : 100,
        'ca_rest' : 50,
        'dffmax' : 93,
        'kd' : 250,
        'conc' : 50000,
        'kappab' : 138.8889,
        'A1' : 8.5,
        'tau1' : 0.5,
        'A1sigma' : [],
        'tau1sigma' : [],
        'A2' : 0,
        'tau2' : 1,
        'tauOn' : 0.01,
        'dur' : 30,
        'spikeRate' : 0.2,
        'snr' : 5,
        'samplingRate' : 1000,
        'frameRate' : 15.5,
        'offset' : 1,
        'maxdt' : 0.5,
        'spkTimes' : [],
        'data_dff' : [],
        'data_ca' : [],
        'data_noisyDFF' : []
    }
    # number of cells to be simulated
    cellNo = 1
    # add offset to total duration
    dur = S['dur'] + S['offset']

    # spike times for a Poisson spike train
    S['spkTimes'] = PoissonSpikeTrain(S['spikeRate'], dur)
    # DEBUG times
    # S['spkTimes'] = [10,15,20]

    # get sampling point times
    x = np.arange(1/S['samplingRate'],dur+1/S['samplingRate'],1/S['samplingRate'])

    ca = np.zeros((cellNo,len(x)));
    dff = np.zeros((cellNo,len(x)));
    spkTCell = S['spkTimes'];

    ca_genmode = S['ca_genmode'];
    spk_recmode = S['spk_recmode'];
    tauOn = S['tauOn'];
    A1 = S['A1'];
    tau1 = S['tau1'];
    A2 = S['A2'];
    tau2 = S['tau2'];

    fig = plt.figure(figsize=(8,12))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)


    if ca_genmode == 'linDFF':
        PeakA = A1 * (tau1 / tauOn * (tau1 / tauOn+1) ** -(tauOn/tau1+1))
    elif ca_genmode == 'satDFF':
        S['ca_amp1'] = S['ca_amp'] / (1+S['ca_kappas']+S['kappab'])             # set for consistency
        S['ca_tau1'] = (1+S['ca_kappas']+S['kappab']) /S['ca_gamma']
        PeakCa = S['ca_amp1'] + S['ca_rest']
        PeakDFF = S['dffmax'] * (PeakCa - S['ca_rest'])/(PeakCa + S['kd'])
        PeakA = PeakDFF * (S['ca_tau1'] / S['ca_onsettau'] * (S['ca_tau1'] / S['ca_onsettau']+1) ** -(S['ca_onsettau'] / S['ca_tau1']+1))
    else:
        raise ValueError('Calcium trace generation mode illdefined')

    sdnoise = PeakA / S['snr'];

    samplingRate = S['samplingRate'];
    offset = S['offset'];
    spkTCell2 = np.zeros((1,cellNo))

    #
    m=0
    currentSpkT = spkTCell

    if ('A1sigma' in S) and S['A1sigma'] and ('tau1sigma' in S) and S['tau1sigma']:
        # convolution for each spike (slower, allows variable model calcium transient)
        # for n = 1:numel(currentSpkT)
        #     currentA1 = random('normal',A1(m),A1(m).*S.A1sigma);
        #     currentTau1 = random('normal',tau1(m),tau1(m).*S.tau1sigma);
        #     y = spkTimes2Calcium(currentSpkT(n),tauOn(m),currentA1,currentTau1,A2(m),...
        #         tau2(m),samplingRate,dur);
        #     dff(m,:) = dff(m,:) + y(1:numel(x));
        # end
        print('not implemented')
        pass
    else:
        if ca_genmode == 'linDFF':
            # convolution over all spikes (faster, same model calcium transient)
            modelTransient = spkTimes2Calcium(0,tauOn,A1,tau1,A2,tau2,samplingRate,dur);
            spkVector = np.zeros((len(x)))
            for i in range(len(currentSpkT)):
                idx = np.argmin(np.abs(currentSpkT[i]-x))
                spkVector[idx] = spkVector[idx]+1
            dffConv = np.convolve(spkVector,modelTransient)
            dff[m,:] = dffConv[0:len(x)]
        elif ca_genmode == 'satDFF':
            print('not implemented')
            # taking saturation into account by solving the single comp model differential equation
            # piecewise, then applying nonlinear transformation from ca to dff
            # ca = spkTimes2FreeCalcium(currentSpkT,S['ca_amp'],S['ca_gamma'],S['ca_onsettau'],S['ca_rest'], S['ca_kappas'],S['kd'], S['conc'],S['samplingRate'],dur);
            # dff[m,:] = Calcium2Fluor(ca,S['ca_rest'],S['kd'],S['dffmax']);

    # plot DFF
    xPlot = x - S['offset']
    dff = dff[:,xPlot > 0]
    xPlot = xPlot[xPlot > 0]
    yOffset = 0
    # only continue if there is at least one spike for this cell
    if len(spkTCell) > 0:
        currentDff = dff[m,:]
        ax1.plot(xPlot,currentDff)
        yOffset = np.max(currentDff) + np.max(dff[:])/5
        S['data_dff'] = dff
        S['data_ca'] = ca

    ax1.set_ylabel('DFF / %')

    # add some noise and plot noisy DFF
    noisyDFF = np.zeros((1,np.size(dff,1)))
    whiteNoise = sdnoise * np.random.randn(np.size(dff,1))
    noisyDFF[m,:] = dff[m,:] + whiteNoise
    yOffset = 0
    currentDff = noisyDFF[m,:]

    ax2.plot(xPlot,currentDff)
    yOffset = np.max(currentDff) + np.max(noisyDFF[:])/5
    ax2.set_ylabel('DFF / %')
    S['data_noisyDFF'] = noisyDFF

    # no sample signal at normal imaging framerate
    lowResT = np.arange(1/S['frameRate'],np.max(xPlot),1/S['frameRate'])
    lowResT = lowResT - (0.5 * 1/S['frameRate'])
    idxList = []
    for ts in lowResT:
        idxList.append(np.abs(xPlot - ts).argmin())

    noisyDFFlowRes = np.zeros((m,len(idxList)))
    print(noisyDFFlowRes.shape,noisyDFF[m,idxList].shape,noisyDFF.shape)
    noisyDFFlowRes = noisyDFF[m,idxList][:]

    yOffset = 0
    currentDff = noisyDFFlowRes
    ax3.plot(lowResT,currentDff)
    yOffset = np.max(currentDff) + np.max(noisyDFFlowRes[:])/5
    ax3.set_ylabel('DFF / %')

    plt.show()
