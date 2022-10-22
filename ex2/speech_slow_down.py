import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import sounddevice as sd
import math

from matplotlib.pyplot import pie  # in order to use pi


# %% Functions


def FourierCoeffGen(signal):
    # TODO: Implement the FourierCoeffGen function.
    # This function compute signal's Fourier Coefficients.
    # Fourier Coefficients for discrete signals are periodic a_k(k)=a_k(k+N)
    # so we will compute only N
    pie = math.pi
    w_0 = (2 * pie) / N  # 2pi divided by the period
    i = complex(0, 1)  # put 0+i inside i
    FourierCoeff = []  # array of coeef

    for k in range(N):  # get N FourierCoeff, each one from sigma
        a_k = 0
        for n in range(N):  # sigma of DFT
            a_k = a_k + signal[n] * np.exp(-i * k * w_0 * n)  # sum of signal*exponent
        a_k = a_k / N  # divided by N (period) (normalization)
        FourierCoeff.append(a_k)  # put new coef to FourierCoeff Array

    return FourierCoeff

    ############# Your code here ############

    #########################################


def DiscreteFourierSeries(FourierCoeff):
    # TODO: Implement the FourierSeries function.
    # This function compute the Discrete Fourier Series from Fourier Coefficients.    
    # make FourierSeries out of the FourierCoeff
    signal = 0
    pie = math.pi
    signal = []
    i = complex(0, 1)  # put 0+i inside i
    w_0 = (2 * pie) / N  # 2pi divided by the period
    for n in range(N):  # FourierSeries Periodic x[N+1]=x[1]
        x_n = 0  # init x_n
        for k in range(N):  # add all K coeef together (n=0,1,2...)
            x_n = x_n + FourierCoeff[k] * np.exp(i * k * w_0 * n)
        signal.append(x_n)  # put new spot (n) to signal Array
    return signal


# %% import wav file
wav_path = "C:/Users/yaniv/PycharmProjects/systemsAndSignals/yaniv9.wav"  # Insert your path here, you can pick another wav file!
signal, fs = sf.read(wav_path)  # give an array of samples from the somg using sf finc
signal = signal[:10 * fs]  # 10 seconds

plt.figure(1)  #
plt.title("Input signal Wave")
plt.plot(signal)

plt.show()
# %% Parameters
N = int(512)  # time period
step = int(N / 4)  # jump every 128
kk = 0
M = 3
signal_out = np.zeros(M * len(signal))  # output length
phase_pre = np.ones(N)  # מערך של 1 בגדול N
last_phase = np.ones(N)
current_phase = 0
b_k = 0
# %%

for k in range(0, signal.shape[0] + 1 - N, step):  # one paket
    print(k)  # the program take a while so we print 128,256,512...,end to see if the program is about to finish its run
    # Analysis
    x = np.multiply(signal[k:k + N], np.hamming(N))
    a_k = FourierCoeffGen(x)  #list of coeef
    # TODO: 1. Extract the Frame's phase.
    #       2. Find the diff phase 
    phase = 0
    phase_diff = 0

    ############# Your code here ############
    phase = [coeef / abs(coeef) for coeef in a_k] # phase is the coeef divided by its absolute value
    phase_diff = np.divide(phase, phase_pre)  # divided current phase by the old phase
    #########################################

    for n in range(M):  # constant delay M
        # Synthesis
        # TODO: 1. Compute the current signal's phase.
        #       2. Compute the output b_k
        #       3. Save the last phase for the next frame 

        ############# Your code here ############
        current_phase = last_phase * phase_diff  # we Compute phase_diff,no multiply it by last phase
        b_k = [abs(coeef) for coeef in a_k] * current_phase  # new coeff is current phase multiply by absoulte value of a_k
        last_phase = current_phase  #save the current phase for next round
        #########################################

        w = np.real(DiscreteFourierSeries(b_k))
        z = np.multiply(w, np.hamming(N))
        signal_out[kk:kk + N] = signal_out[kk:kk + N] + z
        kk = kk + step

    phase_pre = phase  #save the current phase for next round

# %% cheack your results
plt.figure(2)
plt.title("Output signal Wave")
plt.plot(signal_out)

plt.show()

output_path = "C:/Users/yaniv/PycharmProjects/systemsAndSignals/yanivSlow.wav"  # write your path here!
sf.write(output_path, signal_out, fs)

sd.play(signal_out, fs)
