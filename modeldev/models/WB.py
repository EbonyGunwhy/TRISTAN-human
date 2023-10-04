import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

# HELPER FUNCTIONS
def expconv(T: float,
            time: np.ndarray,
            a: np.ndarray
            ) -> np.ndarray:
    if T == 0:
        return a

    n = len(time)
    f = np.zeros(n)
    x = (time[1:n] - time[0:n-1])/T
    da = (a[1:n] - a[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = a[0:n-1]*E0 + da*E1
    for i in range(0, n-1):
        f[i+1] = E[i]*f[i] + add[i]

    return f


def interp_timepoints(simulated_time: np.ndarray,
                      simulated_signal: np.ndarray,
                      observed_time: np.ndarray) -> np.ndarray:
    """Interpolates sampled timepoints to observed signal."""
    # x=time and y=signal
    # Interpolate with same number of timepoints as observed signal
    interp_func = interp1d(simulated_time,
                           simulated_signal,
                           fill_value="extrapolate")
    interp_signal = interp_func(observed_time)
    return interp_signal


# MODELS
def propagate_2cxm(t: np.ndarray,
                   Jin: np.ndarray,
                   Tp: int,
                   Te: int,
                   Erb: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the propagators for the individual compartments in the 2CXM 
    
    For details and notations see appendix of 
    Sourbron et al. Magn Reson Med 62:672–681 (2009)

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    Tp : float
        plasma MTT (sec) = Vp/(Fp+PS)
    Te : float
        extracellular MTT (sec) = Ve/PS
    Erb : float
        extraction fraction to rest of body = PS/(Fp+PS)

    Returns
    -------
    Jp : numpy array
        flux in the plasma compartment (mmol/sec)
    Je : numpy array
        flux in the extracellular compartment (mmol/sec)

    Examples
    --------
    coming soon..

    """
    Kp = (1/Tp)
    Ke = (1/Te)
    Kt = Kp + Ke
    KB = Kp*(1-Erb)
    sqrt = math.sqrt(Kt**2-4*Ke*KB)

    Kpos = 0.5*(Kt + sqrt)
    Kneg = 0.5*(Kt - sqrt)

    Jpos = expconv(1/Kpos, t, Jin)
    Jneg = expconv(1/Kneg, t, Jin)
    #cpos = expconv(1/Kpos, t, ca)
    #cneg = expconv(1/Kneg, t, ca)

    Eneg = (Kpos - KB)/(Kpos - Kneg)

    Jp = (1-Eneg)*Jpos + Eneg*Jneg
    Je = ((Erb/(1-Erb))*(Jneg*Kpos - Jpos*Kneg)) / (Kpos -  Kneg)
    #Je = ((PS/Fp)*(Jneg*Kpos - Jpos*Kneg)) / (Kpos -  Kneg)
    #cp = (1-Eneg)*cpos + Eneg*cneg
    #ce = (cneg*Kpos - cpos*Kneg) / (Kpos -  Kneg)
    # PS/Fp = (Erb/(1-Erb))
    
    return Jp, Je


def propagate_hl(t: np.ndarray,
                 Jin: np.ndarray,
                 THLu: int) -> np.ndarray:
    """Calculates propagator for heart and lungs compartment.
    
    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    THLu : float
        heart and lung MTT (sec)

    Returns
    -------
    JHLu : numpy array
        flux in the heart and lung compartment (mmol/sec)
    """
    return expconv(THLu, t, Jin)


def propagate_gut(t: np.ndarray,
                  Jin: np.ndarray,
                  Tgut: int) -> np.ndarray:
    """Calculates propagator for gut compartment.
    
    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    Tgut : float
        gut MTT (sec)

    Returns
    -------
    Jgut : numpy array
        flux in the gut compartment (mmol/sec)
    """
    return expconv(Tgut, t, Jin)


def propagate_liver(t: np.ndarray,
                    Jin: np.ndarray,
                    TeL: int,
                    Th: int,
                    Eh: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculates propagators for liver compartments.
    
    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    TeL : float
        liver extracellular MTT (sec)
    Th : float
        hepatocytes MTT (sec)

    Returns
    -------
    JeL : numpy array
        flux in the liver extracellular compartment (mmol/sec)
    Jh : numpy array
        flux in the hepatocytes compartment (mmol/sec)
    """
    JeL = (1-Eh)*expconv(TeL, t, Jin) # extracellular
    Jh = (Eh/(1-Eh))*expconv(Th, t, JeL) # hepatocytes
    return JeL, Jh


def propagate_kidneys(t: np.ndarray,
                      Jin: np.ndarray,
                      TpG: int,
                      Tt: int,
                      Et: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculates propagators for kidney compartments.
    
    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    TpG : float
        glomerular plasma MTT (sec)
    Tt : float
        tubular MTT (sec)

    Returns
    -------
    JpG : numpy array
        flux in the glomerular plasma compartment (mmol/sec)
    Jt : numpy array
        flux in the tubular compartment (mmol/sec)
    """
    JpG = (1-Et)*expconv(TpG, t, Jin) # glomerular plasma
    Jt = (Et/(1-Et))*expconv(Tt, t, JpG) # tubular
    return JpG, Jt


def residue_2cxm(t: np.ndarray,
                 Jin: np.ndarray,
                 Tp: int,
                 Te: int,
                 Erb: int) -> np.ndarray:
    """Calculates residue function for 2cxm.

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    Tp : float
        plasma MTT (sec) = Vp/(Fp+PS)
    Te : float
        extracellular MTT (sec) = Ve/PS
    Erb : float
        extraction fraction to rest of body = PS/(Fp+PS)

    Returns
    -------
    Ct : numpy array
        concentrtaion in whole tissue compartment (mM)
    """
    Kp = (1/Tp) # plasma inverse MTT
    Ke = (1/Te) # extracellular inverse MTT
    Kt = Kp + Ke # total inverse MTT
    # 1-Erb = Fp/(Fp+PS)
    sqrt = math.sqrt(Kt**2-4*Ke*Kp*(1-Erb))    

    Kpos = 0.5*(Kt + sqrt)
    Kneg = 0.5*(Kt - sqrt)

    cpos = expconv(1/Kpos, t, Jin) / Kpos
    cneg = expconv(1/Kneg, t, Jin) / Kneg

    Eneg = (Kpos - (Kp*(1-Erb)))/(Kpos - Kneg)
    Ct = (1-Eneg)*cpos + Eneg*cneg

    return Ct


def residue_liver(t: np.ndarray,
                  Jin: np.ndarray,
                  TeL: int,
                  Th: int,
                  Eh: int) -> tuple[np.ndarray,
                                    np.ndarray,
                                    np.ndarray]:
    """Calculates residue function for liver 2cum.

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    TeL : float
        liver extracellular MTT (sec) = VeL/(Fp+Khe)
    Th : float
        hepatocytes MTT (sec) = Vh/Kbh
    Eh : float
        extraction fraction to hepatocytes = Khe/(Fp+Khe)

    Returns
    -------
    CtL : numpy array
        concentrtaion in whole liver tissue compartment (mM)
    CeL : numpy array
        concentration in extracellular compartment (mM)
    Ch : numpy array
        concentration in hepatocytes compartment (mM)
    """
    
    JeL = expconv(TeL, t, Jin) # flux through liver extracellular
    CeL = TeL*JeL
    Ch = Th*expconv(Th, t, Eh*JeL)
    CtL = CeL + Ch
    return CtL, CeL, Ch


def residue_kidneys(t: np.ndarray,
                    Jin: np.ndarray,
                    TpG: int,
                    Tt: int,
                    Et: int) -> np.ndarray:
    """Calculates residue function for kidney 2cfm.

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    TpG : float
        liver extracellular MTT (sec) = VpG/(Fp+GFR)
    Tt : float
        hepatocytes MTT (sec) = Vt/Ft
    Et : float
        extraction fraction to hepatocytes = GFR/(Fp+GFR)

    Returns
    -------
    CtK : numpy array
        concentration in whole kidney tissue compartment (mM)
    """
    JpG = expconv(TpG, t, Jin) # flux through glomerular plasma
    CpG = TpG*JpG
    Ct = Tt*expconv(Tt, t, Et*JpG)
    CtK = CpG + Ct
    return CtK


def propagate_all(t: np.ndarray,
                  Jin: np.ndarray,
                  THLu: int,
                  Tgut: int,
                  E1: int,
                  TeL: int,
                  Th: int,
                  Eh: int,
                  E2: int,
                  TpG: int,
                  Tt: int,
                  Et: int,
                  Tp: int,
                  Te: int,
                  Erb: int) -> tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray]:
    """Calculates propagators for whole body (WB) model.

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    Jin : numpy array
        input function (mmol/sec)
    THLu : float
        heart and lungs MTT (sec)
    Tgut : float
        gut MTT (sec)
    E1 : float
        extraction fraction to liver and gut
    TeL : float
        liver extracellular MTT (sec) = VeL/(Fp+Khe)
    Th : float
        hepatocytes MTT (sec) = Vh/Kbh
    Eh : float
        extraction fraction to hepatocytes = Khe/(Fp+Khe)
    E2 : float
        extraction fraction to kidneys
    TpG : float
        liver extracellular MTT (sec) = VpG/(Fp+GFR)
    Tt : float
        hepatocytes MTT (sec) = Vt/Ft
    Et : float
        extraction fraction to hepatocytes = GFR/(Fp+GFR)
    Tp : float
        plasma MTT (sec) = Vp/(Fp+PS)
    Te : float
        extracellular MTT (sec) = Ve/PS
    Erb : float
        extraction fraction to rest of body = PS/(Fp+PS)

    Returns
    -------
    Jout : numpy array
        flux out of all compartments (mmol/sec) = JeL + JpG + Jp
    JHLu : numpy array
        flux in the heart and lung compartment (mmol/sec)
    Jgut : numpy array
        flux in the gut compartment (mmol/sec)
    JeL : numpy array
        flux in the liver extracellular compartment (mmol/sec)
    Jh : numpy array
        flux in the hepatocytes compartment (mmol/sec)
    JpG : numpy array
        flux in the glomerular plasma compartment (mmol/sec)
    Jt : numpy array
        flux in the tubular compartment (mmol/sec)
    Jp : numpy array
        flux in the plasma compartment (mmol/sec)
    Je : numpy array
        flux in the extracellular compartment (mmol/sec)
    """
    JHLu = propagate_hl(t, Jin, THLu)
    Jgut = propagate_gut(t, E1*JHLu, Tgut)
    JeL, Jh = propagate_liver(t, Jgut, TeL, Th, Eh)
    JpG, Jt = propagate_kidneys(t, E2*(1-E1)*JHLu, TpG, Tt, Et)
    Jp, Je = propagate_2cxm(t, ((1-E1)*(1-E2))*JHLu, Tp, Te, Erb)

    Jout = JeL + JpG + Jp
    return Jout, JHLu, Jgut, JeL, Jh, JpG, Jt, Jp, Je


# SIGNAL CONVERSIONS
def get_R1(R10: int,
           r1: int,
           C: np.ndarray) -> np.ndarray:
    """Converts concentrations to R1s."""
    # Convert concentration to R1
    R1 = R10 + r1*C
    return R1

def get_R1Liver(R10Liver: int,
                r_bl: int,
                rh: int,
                CeL: np.ndarray,
                Ch: np.ndarray):
    """Converts liver concentrations to liver R1s."""
    # Convert concentration to R1
    R1 = R10Liver + r_bl*CeL + rh*Ch # liver
    return R1


def get_signal(amplitude: int,
               R1: np.ndarray,
               TR: int,
               FA: int) -> np.ndarray:
    """Converts R1s to signals."""
    # Convert R1 to MRI signal, S
    E = np.exp(-TR*R1)
    cFA = math.cos(FA*math.pi/180)
    S = amplitude*(1-E)/(1-cFA*E)
    return S


def simulate_signals(t_obs: np.ndarray,
                     J0max: int,
                     t0: int,
                     duration: int,
                     TR: int,
                     FA: int,
                     r_bl: int,
                     rh: int,
                     Fp: int,
                     R10Aorta: int,
                     R10Liver: int,
                     R10Kidneys: int,
                     R10Body: int,
                     S0Aorta: int,
                     S0Liver: int,
                     S0Kidneys: int,
                     S0Body: int,
                     params: dict,
                     iterations=40,
                     dt=0.1,) -> np.ndarray:
    """Simulates MRI signals for whole body model."""
    #J0max = conc*rate
    t = np.arange(0, np.amax(t_obs), dt)
    t_inject = (t > t0) & (t < t0 + duration) # time of injection
    Jin = np.zeros([iterations+1, t.size])
    Jin[0,t_inject] = J0max
    # Propagate fluxes
    for n in range(0, iterations):
        Jin[n+1,:] = propagate_all(t,
                                   Jin[n,:],
                                   **params)[0]
    Jfinal = np.sum(Jin, axis=0)
    Jout, JHLu, Jgut, JeL, Jh, JpG, Jt, Jp, Je = propagate_all(t,
                                                               Jfinal,
                                                               **params)

    # Simulate concentrations
    CAorta = JHLu/Fp # mM when F in L, M when F in mL
    CLiver, Ce, Ch = residue_liver(t,
                                   Jgut,
                                   params['TeL'],
                                   params['Th'],
                                   params['Eh'])

    CKidneys = residue_kidneys(t,
                               (params['E2']*(1-params['E1']))*JHLu,
                               params['TpG'],
                               params['Tt'],
                               params['Et'])

    CBody = residue_2cxm(t,
                         ((1-params['E1'])*(1-params['E2']))*JHLu,
                         params['Tp'],
                         params['Te'],
                         params['Erb'])

    # Convert simulated concentrations to relaxations
    R1Aorta = get_R1(R10Aorta, r_bl, CAorta)
    R1Liver = get_R1Liver(R10Liver, r_bl, rh, Ce, Ch)
    R1Kidneys = get_R1(R10Kidneys, r_bl, CKidneys)
    R1Body = get_R1(R10Body, r_bl, CBody)

    # Convert relaxations to signals
    SAorta = get_signal(S0Aorta, R1Aorta, TR, FA)
    SLiver = get_signal(S0Liver, R1Liver, TR, FA)
    SKidneys = get_signal(S0Kidneys, R1Kidneys, TR, FA)
    SBody = get_signal(S0Body, R1Body, TR, FA)
    
    Saorta = interp_timepoints(t, SAorta, t_obs)
    Sliver = interp_timepoints(t, SLiver, t_obs)
    Skidney = interp_timepoints(t, SKidneys, t_obs)
    Sbody = interp_timepoints(t, SBody, t_obs)
    
    AllSignals = np.concatenate([Saorta, Sliver, Skidney, Sbody])

    return AllSignals

