"""Some derivatives that work for any topology and computing strategy. 

Functions:
---------
derivative_W0_i
"""

# Standard library imports

# Third-party library imports
import numpy as np
from scipy import linalg

# Local library imports

pp = 4

def derivative_w_star_i(i, _lik, constr_w=True, verbose=False):
    """Derivative of w* with respect to coordinate i.
    Params:
    -------
    i: coordinate.
    _lik_col:
    """
    if verbose: print('Inside derivative_w_star_i')
    
    

    if _lik.lattice.topology == 'E':
        d_w_star_i = np.zeros((_lik.lattice.k, 1))
        d_w_star_i[i, 0] = 1
        if constr_w and ((i == 0) or (i == _lik.lattice.k - 1)):
            d_w_star_i[i, 0] = 0
            
    elif _lik.lattice.topology == 'C':
        d_w_star_i = np.zeros((_lik.lattice.nv, 1))
        
        # Is position i a constrained element?
        wavelet_start_v = _lik.lattice.wavelet_positions['wavelet_start_v']
        wavelet_end_v = _lik.lattice.wavelet_positions['wavelet_end_v']
        i_is_endpoint = (constr_w and ((i == wavelet_start_v) or (i == wavelet_end_v - 1)))
        i_is_padding = (i < wavelet_start_v or i >= wavelet_end_v)
        constrained = i_is_endpoint or i_is_padding
        
        # If wavelet element i is a free random variable, the derivative is one
        if not constrained:
            d_w_star_i[i, 0] = 1
            
    if verbose: 
        print(f'd_w_star_i =\n{d_w_star_i}\n')
    
    return d_w_star_i

def derivative_W0_i(i, _lik, constr_w=True, verbose=False):
    """
    Compute the derivative of circulant matrix W0 with base w 
    with respect to w*_i.
    
    Parameters:
    -----------
    i (int): Index in the base vector (0 to n_v-1)
    _lik: contains the 2D lattice.
    
    Returns:
    --------
    numpy.ndarray: The derivative matrices dW0/dw*_i and d(W0.T)/dw*_i.
    """
    if verbose: print('Inside derivative_W0_i')

    nv = _lik.lattice.nv # because the convolutional matrix is for the full axis

    # By default, the derivative is a zero matrix
    base = np.zeros(nv)
    baseT = np.zeros(nv)

    # Does index i correspond to a constrained element in the wavelet?
    wavelet_start_v = _lik.lattice.wavelet_positions['wavelet_start_v']
    wavelet_end_v = _lik.lattice.wavelet_positions['wavelet_end_v']
    i_is_endpoint = (constr_w and ((i == wavelet_start_v) or (i == wavelet_end_v - 1)))
    i_is_padding = (i < wavelet_start_v or i >= wavelet_end_v)
    constrained = i_is_endpoint or i_is_padding
    if verbose and constrained: 
        print('Wavelet endpoint or padding.')
        
    # Change the corresponding element in the base vector to 1
    if not constrained:
        base[nv//2 - i] = 1
        baseT[-nv // 2 + i] = 1
    
    # Construct the derivative matrices
    DW0_i = linalg.circulant(base).T 
    DW0T_i = linalg.circulant(baseT).T 
    
    if verbose: 
        print(f'D_W0_i:\n{DW0_i}')
        print(f'D_W0.T_i:\n{DW0T_i}\n')
     
    return DW0_i, DW0T_i