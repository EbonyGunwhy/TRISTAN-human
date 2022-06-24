import math
import numpy as np
from scipy.special import gamma


def expconv(T, time, a):
    """Convolve a 1D-array with a normalised exponential.

    expconv() uses an efficient and accurate numerical formula to calculate the convolution,
    as detailed in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006.

    Note (1): by definition, expconv preserves the area under a(time)
    Note (2): if T=0, expconv returns a copy of a

    Arguments
    ---------
    a : numpy array
        the 1D array to be convolved.
    time : numpy array
        the time points where the values of ca are defined
        these do not have to to be equally spaced.
    T : float
        the characteristic time of the the exponential function.
        time and T must be in the same units.

    Returns
    -------
    a numpy array of the same shape as ca.

    Example
    -------
    coming soon..

    """
    if T==0: return a

    n = len(time)
    f = np.zeros(n)
    x = (time[1:n] - time[0:n-1])/T
    da = (a[1:n] - a[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = a[0:n-1]*E0 + da*E1
    for i in range(0,n-1):
        f[i+1] = E[i]*f[i] + add[i]      
    return f

def convolve(t, ci, H):

    co = np.empty(len(t))
    ci = np.flip(ci)
    for k in range(t):
        co[k] = np.trapz(H[0:k]*ci[-k:], t[0:k]) #needs checking
    return co   

def propagate_compartment(t, c, MTT):

    return expconv(MTT, t, c)

def propagate_dd(t, c, MTT, TTD):
    """
    Propagate concentration through a serial arrangement of a plug flow and a compartment.

    Arguments
    ---------
    TTD : Transit Time Dispersion of the system
        This is the mean transit time of the compartment
    MTT : Mean Transit Time of the system
        This is the sum of delay and MTT of the compartment

    Returns
    -------
    Concentration at the outlet
    """

    delay = MTT - TTD 
    c = expconv(TTD, t, c)
    c = np.interp(t-delay, t, c, left=0)
    return c

def propagate_chain(t, ci, MTT, n): # 1 <= n 

    # MTT = n * Tx
    # TTD = sqrt(n) * Tx = MTT / sqrt(n)
    # n = (MTT/TTD)^2
    if MTT == 0:
        return ci
    Tx = MTT/n
    H = (np.exp(-t/Tx)/Tx) * (t/Tx)^(n-1)/gamma(n)  
    return convolve(t, ci, H)

def propagate_delay(t, c, delay):

    return np.interp(t-delay, t, c, left=0) 

def propagate_2cxm(t, ca, KP, KE, KB):
    """Calculate the propagators for the individual compartments in the 2CXM 
    
    For details and notations see appendix of 
    Sourbron et al. Magn Reson Med 62:672–681 (2009)

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    ca : numpy array
        input function (mmol/mL)
    KP : float
        inverse plasma MTT (sec) = VP/(FP+PS)
    KE : float
        inverse extracellular MTT (sec) = VE/PS
    KB : float
        inverse blood MTT (sec) = VP/FP

    Returns
    -------
    cp : numpy array
        concentration in the plasma compartment (mmol/mL)
    ce : numpy array
        concentration in the extracellular compartment (mmol/mL)

    Examples
    --------
    coming soon..

    """

    KT = KP + KE
    sqrt = math.sqrt(KT**2-4*KE*KB)

    Kpos = 0.5*(KT + sqrt)
    Kneg = 0.5*(KT - sqrt)

    cpos = expconv(1/Kpos, t, ca)
    cneg = expconv(1/Kneg, t, ca)

    Eneg = (Kpos - KB)/(Kpos - Kneg)

    cp = (1-Eneg)*cpos + Eneg*cneg
    ce = (cneg*Kpos - cpos*Kneg) / (Kpos -  Kneg) 

    return cp, ce

def propagate_simple_body(t, c_vena_cava, 
    MTTlh, Eint, MTTe, MTTo, TTDo, Eext):
    """Propagation through a 2-site model of the body."""

    dose0 = np.trapz(c_vena_cava, t)
    dose = dose0
    min_dose = 10**(-3)*dose0

    c_vena_cava_total = 0*t
    c_aorta_total = 0*t

    while dose > min_dose:
        c_aorta = expconv(MTTlh, t, c_vena_cava)
        c_aorta_total += c_aorta
        c_vena_cava_total += c_vena_cava
        c = propagate_dd(t, c_aorta, MTTo, TTDo)
        c = (1-Eint)*c + Eint*expconv(MTTe, t, c) 
        c_vena_cava = c*(1-Eext)
        dose = np.trapz(c_vena_cava, t)

    return c_vena_cava_total, c_aorta_total


def injection(t, weight, conc, dose, rate, start1, start2=None):
    """dose injected per unit time (mM/sec)"""

    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J1 = propagate_delay(t, J, start1)
    if start2 is None:
        return J1
    else:
        J2 = propagate_delay(t, J, start2)
        return J1 + J2

def signalSPGRESS(TR, FA, R1, S0):

    E = np.exp(-TR*R1)
    cFA = np.cos(FA*math.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def signal_genflash(TR, R1, S0, a, A):
    """Steady-state model of a spoiled gradient echo but
    parametrised with cos(FA) instead of FA and generalised to include rate.
    0<S0
    0<a
    -1<A<+1
    """
    E = np.exp(-a*TR*R1)
    return S0 * (1-E) / (1-A*E)

def signal_hyper(TR, R1, S0, a, b):
    """
    Descriptive bi-exponentional model for SPGRESS sequence.

    S = S0 (e^(+ax) - e^(-bx)) / (e^(+ax) + e^(-bx))
    with x = TR*R1
    0 < S
    0 < a
    0 < b
    """
    x = TR*R1
    Ea = np.exp(+a*x)
    Eb = np.exp(-b*x)
    return S0 * (Ea-Eb)/(Ea+Eb)

def signalBiExp(TR, R1, S0, A, a, b):
    """
    Descriptive bi-exponentional model for SPGRESS sequence.

    S = S0 (1 - A e^(-ax) - (1-A) e^(-bx))
    with x = TR*R1
    0 < A < 1
    0 < S
    0 < a
    0 < b
    """
    x = TR*R1
    Ea = np.exp(-a*x)
    Eb = np.exp(-b*x)
    return S0 * (1 - A*Ea - (1-A)*Eb)

def quadratic(x, x1, x2, x3, y1, y2, y3):
    """returns a quadratic function of x 
    that goes through the three points (xi, yi)"""

    a = x1*(y3-y2) + x2*(y1-y3) + x3*(y2-y1)
    a /= (x1-x2)*(x1-x3)*(x2-x3)
    b = (y2-y1)/(x2-x1) - a*(x1+x2)
    c = y1-a*x1**2-b*x1
    return a*x**2+b*x+c

def linear(x, x1, x2, y1, y2):
    """returns a linear function of x 
    that goes through the three points (xi, yi)"""

    b = (y2-y1)/(x2-x1)
    c = y1-b*x1
    return b*x+c

def concentrationSPGRESS(S, S0, T10, FA, TR, r1):
    """
    Calculates the tracer concentration from a spoiled gradient-echo signal.

    Arguments
    ---------
        S: Signal S(C) at concentration C
        S0: Precontrast signal S(C=0)
        FA: Flip angle in degrees
        TR: Repetition time TR in msec (=time between two pulses)
        T10: Precontrast T10 in msec
        r1: Relaxivity in Hz/mM

    Returns
    -------
        Concentration in mM
    """
    
    E = math.exp(-TR/T10)
    c = math.cos(FA*math.pi/180)
    Sn = (S/S0)*(1-E)/(1-c*E)	#normalized signal
    R1 = -np.log((1-Sn)/(1-c*Sn))/TR	#relaxation rate in 1/msec
    return (R1 - 1/T10)/r1

def sample(t, S, ts, dts): 
    """Sample the signal assuming sample times are at the start of the acquisition"""

    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t > tk) & (t < tk+dts)
        Ss[k] = np.average(S[np.nonzero(tacq)[0]])
    return Ss 