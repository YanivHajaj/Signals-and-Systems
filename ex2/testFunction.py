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
    signal = []
    i = complex(0, 1)  # put 0+i inside i
    w_0 = (2 * pie) / N  # 2pi divided by the period
    for n in range(N):  # FourierSeries Periodic x[N+1]=x[1]
        x_n = 0  # init x_n
        for k in range(N):  # add all K coeef together (n=0,1,2...)
            x_n = x_n + FourierCoeff[k] * np.exp(i * k * w_0 * n)
        signal.append(x_n)  # put new spot (n) to signal Array
    return signal


########################################################################## cos signal
C = 200  # yaniv hajaj last ID digit
N = C  # one period equal C
pie = math.pi  # use pi from library math
e = math.e
x_1 = []  # array of x1 points
for n in range(N):
    x_1.append(math.cos((2 * pie * n) / C))  # N element of x1 samples W=2pi/c -> N=2pi/w -> N=c
plt.plot(x_1)  # X_1 on a graph
plt.title("first signal cos(2pi*n/c)")  # X_1 title
plt.show()
cos_coeff = FourierCoeffGen(x_1)
plt.plot(cos_coeff, 'g')
plt.title("cos_coeff")
plt.show()
reConstructedSignal = DiscreteFourierSeries(cos_coeff)
plt.plot(reConstructedSignal, 'g')
plt.title("reconstructed signal in time - cos")
plt.show()
##########################################################################

##########################################################################
N_1 = 20  # ori glam last ID digit
N = 20 * N_1
x_2 = []
for n in range(int(-N / 2), int(N / 2)):  # sample from -N/2 to N/2
    if abs(n) < 5 * N_1:  # |n|<5*N_1 put 1
        x_2.append(1)
    else:  # |n|>5*N_1 put 0
        x_2.append(0)
plt.plot(x_2, 'g')
plt.title("window signal")
plt.show()
coe = FourierCoeffGen(x_2)
plt.plot(coe, 'g')
plt.title("window_coeff")
plt.show()
reconstructedSignal = DiscreteFourierSeries(coe)
plt.plot(reconstructedSignal, 'r.')
plt.title("reconstructed window signal")
plt.show()
##########################################################################
