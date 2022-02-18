# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:55:31 2021

@author: Robert James
"""
import pandas as pd
import numpy as np 
from scipy import signal 

def _single_optimal_block(x):
    """
    Find optimal block size for block boostrap. 
    See Patton, A., Politis, D.N. and White, H., 2009. 
    Correction to “Automatic block-length selection for the dependent bootstrap” 
    D. Politis and H. White. Econometric Reviews, 28(4), pp.372-375.
    
    Parameters
    ----------
    x : Numpy array
        The time series sequence to be warped.

    Returns
    -------
    b_sb : float
        block size for stationary block bootstrap
    b_cb : float
        block size for the circular block bootstrap

    """
    
    nobs = x.shape[0]
    eps = x - x.mean(0)
    b_max = np.ceil(min(3 * np.sqrt(nobs), nobs / 3))
    kn = max(5, int(np.log10(nobs)))
    m_max = int(np.ceil(np.sqrt(nobs))) + kn
    
    # Find first collection of kn autocorrelations that are insignificant
    cv = 2 * np.sqrt(np.log10(nobs) / nobs)
    acv = np.zeros(m_max + 1)
    abs_acorr = np.zeros(m_max + 1)
#    opt_m: Optional[int] = None
    opt_m = None
    for i in range(m_max + 1):
        v1 = eps[i + 1 :] @ eps[i + 1 :]
        v2 = eps[: -(i + 1)] @ eps[: -(i + 1)]
        cross_prod = eps[i:] @ eps[: nobs - i]
        acv[i] = cross_prod / nobs
        abs_acorr[i] = np.abs(cross_prod) / np.sqrt(v1 * v2)
        if i >= kn:
            if np.all(abs_acorr[i - kn : i] < cv) and opt_m is None:
                opt_m = i - kn
                if not type(opt_m) == int:
                    opt_m = None
    m = 2 * max(opt_m, 1) if opt_m is not None else m_max
    m = min(m, m_max)

    g = 0.0
    lr_acv = acv[0]
    for k in range(1, m + 1):
        lam = 1 if k / m <= 1 / 2 else 2 * (1 - k / m)
        g += 2 * lam * k * acv[k]
        lr_acv += 2 * lam * acv[k]
    d_sb = 2 * lr_acv ** 2
    d_cb = 4 / 3 * lr_acv ** 2
    b_sb = ((2 * g ** 2) / d_sb) ** (1 / 3) * nobs ** (1 / 3)
    b_cb = ((2 * g ** 2) / d_cb) ** (1 / 3) * nobs ** (1 / 3)
    b_sb = min(b_sb, b_max)
    b_cb = min(b_cb, b_max)
    return b_sb, b_cb

def add_random_warping(sequence, warping_percentage):
    """
    Randomly warps a sequence by removing a random selection of data points, 
    re-sampling and interpolating the remaining data points back to the original 
    length of the series. Uses endpoint stuffing to ensure that the warping does 
    not add eronoeous end points to the data. 
    
    Parameters
    ----------
    sequence : Numpy array
        The time series sequence to be warped.
    warping_percentage : float
        Degree of warping to add to the sequence.

    Returns
    -------
    warped_sequence : Numpy array 
        Warped multivariate sequence. 

    """
    n = len(sequence)
    i = np.random.permutation(n)
    #Select the random data points from the instance array. use the same points across all channels of the multivariate sequence
    i_new = sequence[sorted(i[[x for x in range(0,int(n - n*warping_percentage))]]), :]
    
    warped_sequence = np.zeros(sequence.shape)
    for _feature_ in range(sequence.shape[1]):
        sequence_channel = i_new[:,_feature_]
        #Create start and end padding
        n_points_to_stuff = int(np.ceil(0.1*n))
        start_stuff = np.array([sequence_channel[0] for x in range(n_points_to_stuff)])
        end_stuff = np.array([sequence_channel[-1] for x in range(n_points_to_stuff)]) 
        
        i_new_stuffed = np.append(np.append(start_stuff, sequence_channel),end_stuff)
        i_new_stuffed = np.vstack([np.array([0 for x in range(len(i_new_stuffed))]), i_new_stuffed])
        i_new_stuffed = pd.DataFrame(i_new_stuffed.T)
    
        resampled = signal.resample(i_new_stuffed, n + 2*n_points_to_stuff)
        warped_sequence_channel = resampled[n_points_to_stuff:n + n_points_to_stuff,1]
        warped_sequence[:,_feature_] = warped_sequence_channel
    return warped_sequence  
    