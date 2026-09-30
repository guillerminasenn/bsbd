"""Derivatives for the collapsed HMC gradient in Euclidean mode.

Wavelet parameters represent the blur kernel and reflectivity parameters
represent the image.

Functions:
---------
derivative_w_star_j
derivative_W0_j
derivative_Sigma_dw_j
derivative_inv_Sigma_dw_j
derivative_det_Sigma_dw_j
"""

# Third-party library imports
import numpy as np
import pandas as pd
from scipy import linalg

pp = 4

def derivative_w_star_j(j, _w, constr_w=True, verbose=False):
    """Derivative of w* with respect to coordinate j.
    Params:
    -------
    j: coordinate.
    _lik:
    """
    if verbose: print('Inside derivative_w_star_i')
    
    unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']

    d_w_star_j = np.zeros((_w.lattice.nv, 1))
    if j in unconstrained_indices:
        d_w_star_j[j, 0] = 1
            
    if verbose: print(f'd_w_star_j =\n{d_w_star_j}\n')
    return d_w_star_j

def derivative_W0_j(j, _lik, constr_w=True, verbose=False):
    """Derivative of W0 and W0.T with respect to coordinate j in w*, -l <= j <= r. 
    Must be stored in the correct positions of the gradient vector.
    Also called D0_j and D0T_j.
    
    Params:
    -------
    _w: Wavelet object containing the lattice where it was created.
    _lik: containing the 2D lattice.
    """
    if verbose: print('Inside derivative_W0_i')

    nv = _lik.lattice.nv # because the convolutional matrix is for the full axis

    # Case: coordinate j corresponds to non-zero elements of w*
    DW0_j = np.diag(np.ones(nv - abs(-j)), k=-j)
    DW0T_j = np.diag(np.ones(nv - abs(j)), k=j)

    if verbose: 
        print(f'Derivative of W0 wrt coordinate j={j}:\n{DW0_j}')
     
    return DW0_j, DW0T_j

def derivative_Sigma_dw_j(W0, d_W0_j, d_W0T_j, Sigma_c, _lik, verbose=False):
    """Derivative of Sigma_dw = W * Sigma_c* * W' + Sigma_d with respect 
    to coordinate j.
    
    Params:
    -------
    W0: Column convolutional matrix based on w*.
    d_W0_j: Derivative of the W0 matrix wrt the j-th coordinate.
    Sigma_c: the reflectivity covariance matrix (incorporating refl constraints) 
    _lik:

    Return:
    -------
    d_Sigma_dw_j: The derivative wrt coordinate j of Sigma_dw.
    """
    if verbose: print('Inside derivative_Sigma_dw_i')
    
    # Build lattice wavelet convolutional matrix
    nh = _lik.lattice.nh
    W0 = W0.toarray()
    W = linalg.kron(np.eye(nh), W0)

    # Traditional formulas, and without splitting Sigma_c as a kron
    DW_j = linalg.kron(np.eye(nh), d_W0_j)
    DWT_j = linalg.kron(np.eye(nh), d_W0T_j)
    if verbose: 
        print(f'Derivative of W:\n{np.round(DW_j, pp)}')

    d_Sigma_dw_j_1 = DW_j @ Sigma_c @ (W.T)
    d_Sigma_dw_j_2 = W @ Sigma_c @ DWT_j 
    d_Sigma_dw_j = d_Sigma_dw_j_1 + d_Sigma_dw_j_2 
    if verbose: 
        print(f'Term1 of Derivative of Sigma_dw wrt i:\n{pd.DataFrame(d_Sigma_dw_j_1)}')
        print(f'Term2 of Derivative of Sigma_dw wrt i:\n{pd.DataFrame(d_Sigma_dw_j_2)}')

    # # Testing; only when no constraints on c -> ok 28.03

    if verbose: 
        print(f'derivative_Sigma_dw_j =\n{pd.DataFrame(d_Sigma_dw_j)}\n')        
    return d_Sigma_dw_j   

def derivative_inv_Sigma_dw_j(d_Sigma_dw_j, inv_Sigma_dw, verbose=False):
    """Derivative of Sigma_dw^{-1} with respect to coordinate j, with
        Sigma_dw = W * Sigma_c* * W' + Sigma_d 
    the collapsed likelihood covariance matrix.
    
    Params:
    -------
    d_Sigma_dw_j: derivative of Sigma_dw wrt coordinate j.

    Return:
    -------
    d_inv_Sigma_dw_j: The derivative wrt coordinate j of inv_Sigma_dw.
    """
    d_inv_Sigma_dw_j = -inv_Sigma_dw @ (d_Sigma_dw_j @ inv_Sigma_dw)
    if verbose: 
        print('Inside derivative_inv_Sigma_dw_i')
        print(f'inv_Sigma_dw (passed) = \n{pd.DataFrame(inv_Sigma_dw)}\n')
        print(f'Derivative of inv_Sigma_dw wrt coordinate j:\n {pd.DataFrame(d_inv_Sigma_dw_j)}.\n')
    return d_inv_Sigma_dw_j

def derivative_logdet_Sigma_dw_j(d_Sigma_dw_j, inv_Sigma_dw, verbose=False):
    """Derivative of log|Sigma_dw| with respect to coordinate j, with
        Sigma_dw = W * Sigma_c* * W' + Sigma_d 
    the collapsed likelihood covariance matrix.
    
    Params:
    -------
    d_Sigma_dw_j: derivative of Sigma_dw wrt coordinate j.
    inv_Sigma_dw: Inverse of Sigma_dw.

    Return:
    -------
    d_logdet_Sigma_dw_j: The derivative wrt coordinate j of log|Sigma_dw|.
    """
    if verbose: print('Inside derivative_det_Sigma_dw_i')
   
    # Jacobi's formula
    d_logdet_Sigma_dw_j = np.trace(inv_Sigma_dw @ d_Sigma_dw_j)
    if verbose: 
        print(f'derivative of log|Sigma_dw| wrt coord. j ={d_logdet_Sigma_dw_j}\n')   
    
    return d_logdet_Sigma_dw_j