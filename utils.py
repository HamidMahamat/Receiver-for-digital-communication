import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import sounddevice as sd

from scipy.special import erfc


def bit2Symbol_Mapping_QPSK_Gray(A, m) :
    """
    Parameters
    ----------
    A : Amplitude of the symbols
        float.
    m : np.ndarray (N,2)
        containing the list of bit given 2by2.

    Returns
    -------
    np.ndarray (N,1)
        containing the list of qpsk symbols.

    """
    # QPSK table
    qpsk_symbols = A * np.exp(1j * (np.pi/4) * np.array([1, 3, 7, 5]))
    
    m_dec = m[:,0]*2 + m[:,1]
    return qpsk_symbols[m_dec]




def symbol2Bit_Demapping_QPSK_Gray(d_est) :
    """
    Parameters
    ----------
    d_est : np.ndarray monodimensional (N,1)
        array of qpsk symbols.

    Returns
    -------
    np.ndarray (N,2)
        containing the list of bit given 2by2.
    """
    
    R = np.real(d_est)
    I = np.imag(d_est)
    
    return (-.5 * (np.sign(np.vstack((I,R))) - 1).T).astype(int)


def seqSymbEch(d,F) : 
    """
    Parameters
    ----------
    d : np.ndarray (N,1)
        array of symbols.
    F : integer
        oversampling factor e.g if
        Ts is the payload data sampling periode 
        then Ts = F * Te is the digital processing 
        sampling period.

    Returns
    -------
    res : np.ndarray (F * (N-1) + N, 1)
        oversampled sequence of symbols where
        zeros have been inserted between each value
        of the sequence symbols.
    """
    N = len(d) * F
    res = np.zeros(N, dtype=complex)
    res[::F] = d
    return res

    
      

def filterootcos(β, F, Ω) :
    he = []
    for n in range(int(np.floor(2 * Ω * F))+1) :
        if np.abs(n + Ω*F) == F/(4 * β) :
            s = (2*β/(np.pi*np.sqrt(F)))*np.sin(np.pi*(1-β)/(4*β)) + (β/np.sqrt(F))*np.cos(np.pi*(1+β)/(4*β)) 
            he.append(s)
            
        else:
            s = 4*β/(np.pi*np.sqrt(F))*(np.cos((1+β)*np.pi*(n-Ω*F)/F) + ((1-β)*np.pi/(4*β))*np.sinc((n-Ω*F)*(1-β)/F))/(1-(4*β*(n-Ω*F)/F)**2)
            he.append(s)
    return np.array(he)

                

def eye_diagram(signal, oversampling_factor, filter_sample_number, window_limit) :
    """
    Parameters
    ----------
    signal : np.ndarray (N,1)
        array of symbols.
    oversampling_factor : integer
        oversampling factor e.g if
        Te is the payload data sampling periode 
        then Ts = F * Te is the digital processing 
        sampling period.
    filter_sample_number : integer
    window_limit : integer
        number of symbols * F to consider in the eye diagram.

    Returns
    -------
    None.
    """

    N_filter = filter_sample_number
    signal = signal[N_filter-1 : -N_filter-4]
    limit_xx = int(np.floor(window_limit * oversampling_factor))
    
    
    y_max = np.max(np.real(signal)) * 1.05
    plt.xlim(0, limit_xx-1)
    plt.ylim(-y_max, y_max)
    plt.grid()
    for i in range(int(np.floor(len(signal)/limit_xx))) :
        plt.plot(np.arange(limit_xx), np.real(signal[i*limit_xx:(i+1)*limit_xx]), 'b')
    
    # the last part
    len_last_part = len(np.real(signal[(i+1)*limit_xx:])) 
    plt.plot(np.arange(len_last_part), np.real(signal[(i+1)*limit_xx:]), 'b') 
    plt.xlabel(r'$t/T_e$')
    plt.ylabel("Amplitude")


def sychronization(s_hr, idx_start, oversampling_factor, filter_sample_number, d) :
    """
    Parameters
    ----------
    s_hr : np.ndarray complex (N,1)
        output of reception filter.
    idx_start : int
        index to start the clock.
    oversampling_factor : int
        oversampling factor.
    filter_sample_number : int
        number of samples of the filter.
    d : np.ndarray complex (L,1)
        sequence of mapped symbols.
    Returns
    -------
    return synchronized sampled symbols.
    """    
    F = oversampling_factor
    N_filter = filter_sample_number
    s_hr = s_hr[N_filter-1 : -N_filter-4]
    s_hr_ech = s_hr[idx_start::F]
    return s_hr_ech[:len(d)]

    
def  MLSymbolDetectorQPSKlowCPLX(A, Delta, z) :
    d_j = A * np.exp(1j * (np.pi/4) * np.array([5, 7, 1, 3]))
    
    # Maximum Likelihood metric
    metric = np.abs( z - ((Delta**2/2)*np.ones((len(z),len(d_j)))*d_j).T )
    return d_j[np.argmin(metric, axis=0)]
    
    
    
def AWGN(Delta, var, d) :
    noise = np.sqrt(var/2) * (np.random.randn(len(d)) + 1j * np.random.randn(len(d)))
    return (Delta**2)/2 * d + noise
    
    
def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = sg.correlate(x, y, mode="full")
    lags = sg.correlation_lags(len(x), len(y), mode="full")
    return lags, corr

    