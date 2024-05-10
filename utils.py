import numpy as np
import matplotlib.pyplot as plt


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
        Te is the payload data sampling periode 
        then Ts = F * Te is the digital processing 
        sampling period.

    Returns
    -------
    res : np.ndarray (F * (N-1) + N, 1)
        oversampled sequence of symbols where
        zeros have been inserted between each value
        of the sequence symbols.
    """
    zero = np.zeros(F)  
    res = np.zeros(F * (len(d)-1) + len(d)).astype(complex)
    for k in range(len(d)):
        res[k*F] = d[k]
    return res
      

def filterootcos(beta, F, omega) :
    he = []
    for n in range(int(np.floor(2 * omega * F))+1) :
        if np.abs(n + omega*F) == F/(4 * beta) :
            s = (2*beta/(np.pi*np.sqrt(F)))*np.sin(np.pi*(1-beta)/(4*beta)) + (beta/np.sqrt(F))*np.cos(np.pi*(1+beta)/(4*beta)) 
            he.append(s)
            
        else:
            s = 4*beta/(np.pi*np.sqrt(F))*(np.cos((1+beta)*np.pi*(n-omega*F)/F) + ((1-beta)*np.pi/(4*beta))*np.sinc((n-omega*F)*(1-beta)/F))/(1-(4*beta*(n-omega*F)/F)**2)
            he.append(s)
    return np.array(he)
                
    

    
def  MLSymbolDetectorQPSKlowCPLX(A, Delta, z) :
    d_j = A * np.exp(1j * (np.pi/4) * np.array([5, 7, 1, 3]))
    
    # Maximum Likelihood metric
    metric = np.abs( z - ((Delta**2/2)*np.ones((len(z),len(d_j)))*d_j).T )
    return d_j[np.argmin(metric, axis=0)]
    
    
    
def AWGN(Delta, var, d) :
    noise = np.sqrt(var/2) * (np.random.randn(len(d)) + 1j * np.random.randn(len(d)))
    return (Delta**2)/2 * d + noise
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    