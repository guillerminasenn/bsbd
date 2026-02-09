"""This module contains the functions to perform matrix algebra with the FFT and kronecker products. 

List of functions:
build_matrix_circ
build_base_from_circ
transpose_base_circ
invert_circ
multiply_circ
multiply_matrix_vector_circ
compute_eigenvalues_circ
compute_log_det_circ
"""

# Third-party library imports
import numpy as np
from scipy import linalg
import scipy as scipy

pp = 4 

def build_matrix_circ(base, explicit=False):
    """Build the explicit form of a circulant matrix from its base.
    
    Parameters
    ----------
    base : ndarray
        nv x nh base array used to build the circulant matrix.
        
    Returns
    -------
    ndarray
        n x n matrix.
    
    Notes
    -----
    Works for both level-1 and level-2 circulants.
    """
    # Extract dimensions
    nv, nh = base.shape
    n = nh * nv
    
    # Initialize result
    matrix = None
    
    # Handle single column case
    if nh == 1:
        matrix = linalg.circulant(base.reshape(-1)).T
        return matrix
        
    # Handle multi-column case
    if nh > 1:
        if n > 300 and explicit==False:
            print(f'Building the {n}x{n} matrix is computationally intensive, returning None.')
            return matrix
        
        elif n > 100:
            print(f'Be aware that building the {n}x{n} matrix might be computationally intensive.')
        
        # Create first block row
        first_brow = np.hstack([linalg.circulant(base[:, j]) for j in np.arange(nh)])
        
        # Build the full matrix by rolling the first block row
        matrix = np.zeros((n, n))
        for i in np.arange(0, n, nv):
            matrix[i:(i + nv), :] = np.roll(first_brow, i, axis=1)
                
        return matrix

def build_base_from_circ(matrix, nh, nv):
    """Given a circulant matrix, return its base.

    Params:
    ------
    matrix: Circulant whose base we want. 
    nh: Number of nv x nv blocks
    nv: Block size
    
    Return:
    -------
    base: (nv, nh) array.
    """
    n = nh * nv
    first_brow = matrix[:nv, :]
    blocks = [first_brow[:nv, i:(i + nv)] for i in np.arange(0, n, nv)]
    base = np.array([block[0, :].T for block in blocks]).T
    return base

def transpose_base_circ(base):
    """Receives the base of the circulant A and returns the base 
    of A transpose.
    
    Parameters:
    -----------
    base: np.array nv x nh
    """
    nh = base.shape[1]
    
    if nh == 1: # circulant
        base_rev = base[::-1] # reverse the base
        baseT = np.roll(base_rev, 1) # shift 1 position  

    elif nh > 1: # BCCB
        # First, shift-reverse the base of each block
        base_rev = base[::-1] # reverse all the columns in the base matrix
        base_rev_shift = np.roll(base_rev, 1, axis=0) # shift all the columns 1 position

        # Then, shift-reverse the columns in the base matrix
        base_rev_shift_rev = base_rev_shift[:, ::-1]
        baseT = np.roll(base_rev_shift_rev, 1, axis=1)
        
    return baseT

def transpose_base_circ_vectorized(base_array):
    """Vectorized version of transpose_base_circ.
    
    Parameters:
    -----------
    base_array: np.array (n_matrices, nv, nh)
        Array of bases for multiple circulant matrices
        
    Returns:
    --------
    baseT_array: np.array (n_matrices, nv, nh)
        Array of bases for the transposed matrices
    """
    baseT_array = np.zeros_like(base_array)

    if base_array.ndim == 2: # circulant case
        # Reverse along axis 1 (nv dimension) for all matrices
        base_rev = base_array[:, ::-1]  # shape: (n_matrices, nv)
        # Shift 1 position for all matrices
        baseT_array = np.roll(base_rev, 1, axis=1)
        
    elif base_array.ndim == 3:  # BCCB case
        # First, shift-reverse the base of each block (reverse along nv dimension)
        base_rev = base_array[:, ::-1, :]  # shape: (n_matrices, nv, nh)
        base_rev_shift = np.roll(base_rev, 1, axis=1)  # shift along nv dimension
        
        # Then, shift-reverse the columns (reverse along nh dimension)
        base_rev_shift_rev = base_rev_shift[:, :, ::-1]  # reverse along nh dimension
        baseT_array = np.roll(base_rev_shift_rev, 1, axis=2)  # shift along nh dimension


    # if nh == 1:  # circulant case
    #     # Reverse along axis 1 (nv dimension) for all matrices
    #     base_rev = base_array[:, ::-1, :]  # shape: (n_matrices, nv, 1)
    #     # Shift 1 position for all matrices
    #     baseT_array = np.roll(base_rev, 1, axis=1)
        
    # elif nh > 1:  # BCCB case
    #     # First, shift-reverse the base of each block (reverse along nv dimension)
    #     base_rev = base_array[:, ::-1, :]  # shape: (n_matrices, nv, nh)
    #     base_rev_shift = np.roll(base_rev, 1, axis=1)  # shift along nv dimension
        
    #     # Then, shift-reverse the columns (reverse along nh dimension)
    #     base_rev_shift_rev = base_rev_shift[:, :, ::-1]  # reverse along nh dimension
    #     baseT_array = np.roll(base_rev_shift_rev, 1, axis=2)  # shift along nh dimension
        
    return baseT_array

# With 2D-FFT even if nh==1
def invert_circ(base):
    """Compute the base of the inverse of a circulant, from the
    base of the circulant. 

    Return:
    -------
    q: (nv, nh) array 
        The base of the inverse matrix.
    """
    nh = base.shape[1]
    nv = base.shape[0]
    n = nh * nv
    internal_term = np.power(np.real(scipy.fft.fft2(base, norm='ortho')), -1)
    q = scipy.fft.ifft2(internal_term, norm='ortho') / n
    q = np.real(q)
    return q

def decompose_circ(base):
    """Compute the base of the decomposition of a circulant. 

    Return:
    -------
    q: (nv, nh) array 
        The base of the inverse square root factor.
    """
    nh = base.shape[1]
    nv = base.shape[0]
    n = nh * nv
    internal_term = np.power(np.real(scipy.fft.fft2(base, norm='ortho')), -1/2)
    q = scipy.fft.ifft2(internal_term, norm='ortho') / n
    q = np.real(q)
    return q

def multiply_circ(base_A, base_B, verbose=False): 
    """Compute the base of the product of two circulant matrices A and B 
    of the same order, with bases base_A and base_B.

    Return:
    -------
    base_ab: (nv, nh) array 
        The base of the product AB.
    """

    if base_A.shape != base_B.shape:
        raise Exception("Circulants not of the same order.")
        
    nh = base_A.shape[1]
    nv = base_A.shape[0]
    n = nh * nv
    norm = np.sqrt(n)
    
    internal_term = scipy.fft.fft2(base_A, norm='ortho', ) * scipy.fft.fft2(base_B, norm='ortho')
    base_AB = norm * scipy.fft.ifft2(internal_term, norm='ortho')
    base_AB = base_AB.reshape((nv, nh))
    
    # Verification against explicit multiplication
    if verbose:
        print(f'in multiply circ:\nbase_A =\n{base_A}\n')
        print(f'base_B =\n{base_B}\n')
        print(f'base_AB =\n{base_AB}\n')
        if nh == 1:
            A = linalg.circulant(base_A.reshape(-1)).T
            B = linalg.circulant(base_B.reshape(-1)).T
            AB = linalg.circulant(base_AB.reshape(-1)).T
            print(f'A =\n{A}\n')
            print(f'B =\n{B}\n')
            print(f'AB =\n{AB}\n')
    base_AB = np.real(base_AB)
    return base_AB

def multiply_matrix_vector_circ(base, b): 
    """Return Ab, with A circulant or BCCB, and b a vector.
    
    Parameters:
    ----------
    base: np.array nv x nh 
        Base of the matrix A. n = nv x nh
    b: np.array n x 1 
        Vector.
        
    Return:
    -------
    u: np.array n x 1 
        u = Ab.
    """
    
    nh = base.shape[1]
    nv = base.shape[0]
    n = nh * nv

    B = b.reshape((nv, nh), order='F')
    internal_term = scipy.fft.fft2(base, norm='ortho') * scipy.fft.ifft2(B, norm='ortho')
    u = scipy.fft.fft2(internal_term).reshape((n, 1), order='F')

    u = np.real(u)
    return u

def compute_eigenvalues_circ(base):
    """Compute the eigenvalues of the circulant with the FFT.
    
    Parameters
    ----------
    base : ndarray
        nv x nh base array used to build the circulant matrix.
        
    Returns
    -------
    ndarray
        Eigengalues.
        
    Notes
    -----
    Works for both level-1 and level-2 circulants."""
    # Extract dimensions
    nv, nh = base.shape
    n = nh * nv
    
    # Compute eivengalues 
    eigenvalue_matrix = np.sqrt(n) * scipy.fft.fft2(base, norm='ortho')
    eigenvalues = eigenvalue_matrix.reshape((n, 1), order='F')
    
#     # Old fn for 1D
#     F = scipy.fft.fft2(np.eye(n))
#     eigenvalues = F @ base
    
    # # Old fn for 2D
    # Fh = scipy.fft.fft(np.eye(nh))
    # Fv = scipy.fft.fft(np.eye(nv))
    # eigenvalue_matrix_col = Fv @ base # col application
    # eigenvalue_matrix = eigenvalue_matrix_col @ (Fh.T) # row application
    # eigenvalues = eigenvalue_matrix.reshape((n, 1), order='F')

    return eigenvalues 

# # # With 1D FFT when nh==1
# def invert_circ(base):
#     """Compute the base of the inverse of a circulant, from the
#     base of the circulant. 

#     Return:
#     -------
#     q: (nv, nh) array 
#         The base of the inverse matrix.
#     """
#     nh = base.shape[1]
#     nv = base.shape[0]
#     n = nh * nv
    
#     if nh == 1:
#         # Use 1D FFT for single column case
#         base_1d = base.reshape(-1)
#         internal_term = np.power(scipy.fft.rfft(base_1d, norm='ortho'), -1)
#         q_1d = scipy.fft.irfft(internal_term, norm='ortho') / n
#         q = q_1d.reshape((nv, 1))
#     else:
#         # Use 2D FFT for multi-column case
#         internal_term = np.power(scipy.fft.rfft2(base, norm='ortho'), -1)
#         q = scipy.fft.irfft2(internal_term, norm='ortho') / n
    
#     return q

# def multiply_circ(base_A, base_B, verbose=False): 
#     """Compute the base of the product of two circulant matrices A and B 
#     of the same order, with bases base_A and base_B.

#     Return:
#     -------
#     base_ab: (nv, nh) array 
#         The base of the product AB.
#     """

#     if base_A.shape != base_B.shape:
#         raise Exception("Circulants not of the same order.")
        
#     nh = base_A.shape[1]
#     nv = base_A.shape[0]
#     n = nh * nv
#     norm = np.sqrt(n)
    
#     if nh == 1:
#         # Use 1D FFT for single column case
#         base_A_1d = base_A.reshape(-1)
#         base_B_1d = base_B.reshape(-1)
#         internal_term = scipy.fft.rfft(base_A_1d, norm='ortho') * scipy.fft.rfft(base_B_1d, norm='ortho')
#         base_AB = norm * scipy.fft.irfft(internal_term, norm='ortho').reshape((nv, 1))
#         # base_AB = np.real(base_AB_1d)
#     else:
#         # Use 2D FFT for multi-column case
#         internal_term = scipy.fft.rfft2(base_A, norm='ortho') * scipy.fft.rfft2(base_B, norm='ortho')
#         base_AB = norm * scipy.fft.irfft2(internal_term, norm='ortho')
#         base_AB = base_AB.reshape((nv, nh)) # why dont i use ordr here
    
#     # # Verification against explicit multiplication
#     # if verbose:
#     #     print(f'in multiply circ:\nbase_A =\n{base_A}\n')
#     #     print(f'base_B =\n{base_B}\n')
#     #     print(f'base_AB =\n{base_AB}\n')
#     #     if nh == 1:
#     #         A = linalg.circulant(base_A.reshape(-1)).T
#     #         B = linalg.circulant(base_B.reshape(-1)).T
#     #         AB = linalg.circulant(base_AB.reshape(-1)).T
#     #         print(f'A =\n{A}\n')
#     #         print(f'B =\n{B}\n')
#     #         print(f'AB =\n{AB}\n')
#     return base_AB
# 
# def multiply_matrix_vector_circ(base, b): 
#     """Return Ab, with A circulant or BCCB, and b a vector.
    
#     Parameters:
#     ----------
#     base: np.array nv x nh 
#         Base of the matrix A. n = nv x nh
#     b: np.array n x 1 
#         Vector.
        
#     Return:
#     -------
#     u: np.array n x 1 
#         u = Ab.
#     """
    
#     nh = base.shape[1]
#     nv = base.shape[0]
#     n = nh * nv

#     if nh == 1:
#         # Use 1D FFT for single column case
#         base_1d = base.reshape(-1)
#         b_1d = b.reshape(-1)
#         internal_term = scipy.fft.rfft(base_1d, norm='ortho') * scipy.fft.irfft(b_1d, norm='ortho')
#         u_1d = scipy.fft.rfft(internal_term)
#         u = u_1d.reshape((n, 1))
#     else:
#         # Use 2D FFT for multi-column case
#         B = b.reshape((nv, nh), order='F')
#         internal_term = scipy.fft.rfft2(base, norm='ortho') * scipy.fft.irfft2(B, norm='ortho')
#         u = scipy.fft.rfft2(internal_term).reshape((n, 1), order='F')
# 
#     return u

# def compute_eigenvalues_circ(base):
#     """Compute the eigenvalues of the circulant with the FFT.
    
#     Parameters
#     ----------
#     base : ndarray
#         nv x nh base array used to build the circulant matrix.
        
#     Returns
#     -------
#     ndarray
#         Eigengalues.
        
#     Notes
#     -----
#     Works for both level-1 and level-2 circulants."""
#     # Extract dimensions
#     nv, nh = base.shape
#     n = nh * nv
#     norm = np.sqrt(n)
    
#     if nh == 1:
#         # Use 1D FFT for single column case
#         base_1d = base.reshape(-1)
#         eigenvalues_1d = norm * scipy.fft.fft(base_1d, norm='ortho')
#         eigenvalues = eigenvalues_1d.reshape((n, 1))
#     else:
#         # Use 2D FFT for multi-column case
#         eigenvalue_matrix = norm * scipy.fft.fft2(base, norm='ortho')
#         eigenvalues = eigenvalue_matrix.reshape((n, 1), order='F')
    
#     return eigenvalues


# Log-det
def compute_log_det_circ(a, discard=True, epsilon=1e-10, verbose=False):
    """Given A circulant or BCM with base a, compute log(det(A)) 
    from the eigenvalues of A.
    
    Params:
    -------
    a: base(A)
    discard: If the max. abs value of the imaginary eigenvalues is smaller
    than epsilon, discard all imaginary parts. Else, 
    epsilon: Tolerance level."""
    
    # Compute eigenvalues 
    eigenvalues = compute_eigenvalues_circ(a)
    
    if discard:
        # Discard the imaginary term of the eigs (if trivial)
        nr_imag_significative = np.sum(np.abs(np.imag(eigenvalues)) > epsilon)
        if nr_imag_significative == 0:
            # print('Discarding imag part')
            eigenvalues_real = np.real(eigenvalues)
            # Compute determinant
            log_det = np.sum(np.log(eigenvalues_real))
            # print('compute log done.')
            return(log_det)
        else:
            raise Exception(f'There were {nr_imag_significative} eigenvalues whose imaginary part was greater than {epsilon}, so det=complex.')
            # print('compute log done.')
            # log_det = np.sum(np.log(eigenvalues))
            # return('error.')
    log_det = np.sum(np.log(eigenvalues))
    # det = np.prod(eigenvalues)
    # if verbose:
    #     print(f'In compute_log_det_circ, det = {det} (computed as prod of eigs)')
    if np.abs(np.imag(log_det)) > epsilon:
        print(f'in efficient_algebra_utils.compute_log_det_circ: The imaginary part of the determinant was not trivial:{np.abs(np.imag(log_det))}')
        print(f"The logdet = {log_det}")
    return(np.real(log_det))

def check_pd_circulant(a, epsilon=0.000000001, verbose=False):
    """Check if a circulant/block-circulan matrix A with base a is positive-
    definite. Numerical errors are accounted at tolerance level epsilon."""

    # print(f'Tolerance level: {epsilon}.')
    # Compute eigenvalues of A
    if len(a.shape) == 1:
        # norm = np.sqrt(a.shape[0])
        F = scipy.fft.fft(np.eye(a.shape[0]))
        eigenvalues = F @ a
    if len(a.shape) == 2:
        norm = np.sqrt(a.shape[0] * a.shape[1])
        eigenvalues = norm * scipy.fft.fft2(a, norm='ortho')
    else:
        return('The base does not correspond to a 1D or 2D DFT.')
    
    # Check whether A is real
    nr_imag_elems = np.sum(np.imag(a) != 0)
    nr_significative_imag_elems = np.sum(np.abs(np.imag(a)) > epsilon)
    is_real_exact = nr_imag_elems == 0
    is_real_tol = nr_significative_imag_elems == 0

    # Check whether all eigenvalues are real
    nr_imag_eig = np.sum(np.imag(eigenvalues) != 0)
    nr_significative_imag_eig = np.sum(np.abs(np.imag(eigenvalues)) > epsilon)
    all_eig_real_exact = nr_imag_eig == 0
    all_eig_real_tol = nr_significative_imag_eig == 0
    
    # Check whether all (real) eigenvalues are positive
    nr_nonpos_eig = np.sum(np.real(eigenvalues) <= 0)
    nr_significative_nonpos_eig = np.sum(np.real(eigenvalues) <= -epsilon)
    all_eig_pos_exact = nr_nonpos_eig == 0
    all_eig_pos_tol = nr_significative_nonpos_eig == 0

    # Is positive definite?
    pos_def = (is_real_tol*all_eig_real_tol*all_eig_pos_tol) == 1
    if verbose:
        print(f'Exact: {nr_imag_elems} imaginary elements, {nr_imag_eig} imaginary eigenvalues, {nr_nonpos_eig} real non positive eigenvalues.')
        print(f'Imaginary eigenvalues: min. magnitude: {np.min(np.abs(np.imag(eigenvalues)))}, max. magnitude: {np.max(np.abs(np.imag(eigenvalues)))}')
        print(f'Real eigenvalues: min: {np.min(np.real(eigenvalues))}, max: {np.max(np.real(eigenvalues))}')
        print(f'With epsilon tolerance: {nr_significative_imag_elems} imaginary elements')
        print(f'{nr_significative_imag_eig} imaginary eigenvalues, {nr_significative_nonpos_eig} real non positive eigenvalues.')
        # print(f'is_real_tol:{is_real_tol}, all_eig_real_tol:{all_eig_real_tol}, all_eig_pos_tol:{all_eig_pos_tol}')
        print(f'Pos. def? {pos_def}\n')
    return(pos_def)
