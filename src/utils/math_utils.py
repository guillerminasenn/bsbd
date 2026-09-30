"""This module contains the functions to perform matrix algebra with the FFT and 
kronecker products. 
List of functions:
1. base_kron_block_circulant
2. base_block_circulant
2. build_bcm
2. circulant_product_fft
2.5 circulant_matrix_vector_product_1d
3. invert_circ
7. compute_eigenvalues_circ
5. compute_log_det
6. compute_log_det_circ
6. check_pd_circulant
6. pseudo_inverse_RH
7. Euclidean_dist
8. great_circle_dist
"""

# Standard library imports
import math

# Third-party library imports
import numpy as np
from scipy import linalg


pp = 4 

###################### ###################### ###################### 
######################    Circulant algebra   ######################
###################### ###################### ###################### 
    
def base_kron_block_circulant(A, B):
    """Return the base of the block circulant matrix formed by the Kronecker 
    product of two circulant matrices A and B.
    
    Return:
    -------
    base: (nv, nh) array.
    """
    a = A[0, :].reshape((1, A.shape[1])) #  base of A
    # print(f"shape of base a: {a.shape}")
    b = B[0, :].reshape((1, B.shape[1])) # base of B
    # print(f"shape of base b: {b.shape}")
    # kron = linalg.kron(a, b)
    # c = kron.reshape((B.shape[1], A.shape[1]), order='F')
    base = linalg.kron(a, b.T)
    return(base)

def base_block_circulant(A, nv, nh, **kwargs):
    """Given a bcm matrix A and its order, return its base.
        
    Return:
    -------
    base: (nv, nh) array.
    """
    n = nv * nh
    first_brow = A[:nv, :]
    blocks = [first_brow[:nv, i:(i + nv)] for i in np.arange(0, n, nv)]
    base = np.array([block[0, :].T for block in blocks]).T
    return base

def build_bcm(base):
    """Given the base of a block-circulant with circulant blocks matrix A, 
    return the matrix.
    Params:
    -------
    base: (nv, nh) array.
    """
    
    nv = base.shape[0]
    nh = base.shape[1]
    n = nv * nh
    # print('base:\n', base)
    first_brow = np.hstack([linalg.circulant(base[:, j]) for j in np.arange(nh)])
    A = np.zeros((n, n))
    for i in np.arange(0, n, nv):
        A[i:(i + nv), :] = np.roll(first_brow, i, axis=1)
    return A

# # in Circulant class
# def circulant_product_fft(a, b):
#     """Compute the product of two circulant matrices A and B with bases a and b, 
#     with the same dimensions."""

#     # print('inside circulant_product_fft')
#     norm = np.sqrt(len(a))
#     internal_term = np.fft.fft(a, norm='ortho', ) * np.fft.fft(b, norm='ortho')
#     base_ab = norm * np.fft.ifft(internal_term, norm='ortho')
#     # print(f'length of base_ab: {base_ab.shape}')
#     # print(f'Nr. of complex elements: {np.sum(np.imag(base_ab) > 0.0000000001)}')
#     # print('DONE: circulant_product_fft.')
#     return(np.real(base_ab))

# # in Circulant class
# def circulant_matrix_vector_product_1d(a, b):
#     """Compute the matrix-vector product of circulant matrix A with base a, 
#     with the vector b."""
#     internal_term = np.fft.fft(a, norm='ortho') * np.fft.ifft(b, norm='ortho')
#     u = np.fft.fft(internal_term)
#     return(u)

# # in Circulant class
# def invert_circ(c):
#     """Compute the base of the inverse of the circulant matrix C with base c.
    
#     Params:
#     ------
#     c: base of C (first row or block row).
    
#     Return:
#     -------
#     q: (nv, nh) array which is the base of the inverse of C.
#     """
#     nv = c.shape[0]
#     if len(c.shape) == 1: # level-1 circulant
#         internal_term = np.power(np.fft.fft(c, norm='ortho'), -1)
#         q = np.real(np.fft.ifft(internal_term, norm='ortho')) / nv
#         return q
    
#     if len(c.shape) == 2: # level-2 circulant
#         nh = c.shape[1]
#         n = nv * nh
#         internal_term = np.power(np.real(np.fft.fft2(c, norm='ortho')), -1)
#         q = np.real(np.fft.ifft2(internal_term, norm='ortho')) / n
#         return q
#     else:
#         print('Implement the case for a circulant not of level {1, 2}.')
        
# in Circulant class - but kept as a function too
def compute_eigenvalues_circ(a, epsilon=0.000000001):
    """Compute the eigenvalues of a circulant or block-circulant 
    matrix A with base a."""

    # print(f'Tolerance level: {epsilon}.')
    # Compute eigenvalues 
    if len(a.shape) == 1:
        # norm = np.sqrt(a.shape[0])
        F = np.fft.fft(np.eye(a.shape[0]))
        eigenvalues = F @ a
        return(eigenvalues)
    if len(a.shape) == 2:
        norm = np.sqrt(a.shape[0] * a.shape[1])
        eigenvalues = norm * np.fft.fft2(a, norm='ortho')
        return(eigenvalues)
    else:
        return('The base does not correspond to a 1D or 2D DFT.')
    
def compute_log_det(A, epsilon=1e-6):
    """Compute log(det(A)) from the eigenvalues of A."""
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(A)
    # print(f'In compute_log_det: \n Eigenvalues = {np.round(eigenvalues, pp)}\nEigenvectors = {np.round(eigenvectors, pp)}')

    log_det = np.sum(np.log(eigenvalues))
    

    # Check for negative eigenvalues (nonPD)
    if np.min(np.real(eigenvalues)) < -epsilon:
        print(f'In math_utils.compute_log_det_circ, the smallest real part in the eigenvalues was {np.min(np.real(eigenvalues))} < tol={-epsilon}, the matrix might not be PD.')
        print(f'Logdet before discarding imaginary part: {log_det}.')
        raise Exception(f"Interrupting iteration because nonPD.")
    
    # Check for imaginary eigenvalues (non symmetric)
    if np.max(np.abs(np.imag(eigenvalues))) > epsilon:
        print(f'In math_utils.compute_log_det, the largest imaginary component in the eigenvalues was {np.max(np.abs(np.imag(eigenvalues)))} > tol={epsilon}.')
        print(f'Logdet before discarding imaginary part: {log_det}.')
        raise Exception(f"Interrupting iteration because non-symmetric.")

    return np.real(log_det)

# in Circulant class - but kept as a function too
def compute_log_det_circ(a, epsilon=1e-6):
    """Given A circulant or BCM with base a, compute log(det(A)) 
    from the eigenvalues of A.
    
    Params:
    -------
    a: base(A)
    discard: If the max. abs value of the imaginary eigenvalues is smaller
    than epsilon, discard all imaginary parts. Else, 
    epsilon: Tolerance level."""
    
    # Compute eigenvalues     
    eigenvalues = compute_eigenvalues_circ(a, epsilon)
    log_det = np.sum(np.log(eigenvalues))
    
    # Check for negative eigenvalues (nonPD)
    if np.min(np.real(eigenvalues)) < -epsilon:
        print(f'In math_utils.compute_log_det_circ, the smallest real part in the eigenvalues was {np.min(np.real(eigenvalues))} < tol={-epsilon}, the matrix might not be PD.')
        print(f'Logdet before discarding imaginary part: {log_det}.')
        raise Exception(f"Interrupting iteration because nonPD.")
    
    # Check for imaginary eigenvalues (lack of symmetry)
    if np.max(np.abs(np.imag(eigenvalues))) > epsilon:
        print(f'Logdet before discarding imaginary part: {log_det}.')
        print(f'In math_utils.compute_log_det_circ, the largest imaginary component in the eigenvalues was {np.max(np.abs(np.imag(eigenvalues)))} > tol={epsilon}.')
        raise Exception(f"Interrupting iteration because non-symmetric.")
    return(np.real(log_det))


def check_pd_circulant(a, epsilon=0.000000001, verbose=False):
    """Check if a circulant/block-circulan matrix A with base a is positive-
    definite. Numerical errors are accounted at tolerance level epsilon."""

    # print(f'Tolerance level: {epsilon}.')
    # Compute eigenvalues of A
    if len(a.shape) == 1:
        # norm = np.sqrt(a.shape[0])
        F = np.fft.fft(np.eye(a.shape[0]))
        eigenvalues = F @ a
    if len(a.shape) == 2:
        norm = np.sqrt(a.shape[0] * a.shape[1])
        eigenvalues = norm * np.fft.fft2(a, norm='ortho')
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

def pseudo_inverse_RH(Sigma_star, tolerance=1e-10, verbose=False):
    """Compute the eigenvalues, eigenvectors, and pseudoinverse of a
    singular covariance matrix Sigma_star with decomposition
        Sigma_star = V \Lambda V', 
    pseudoinverse given as 
        Q* = V \Lambda^{-} V'.
        
    Params:
    -------
    tolerance: Tolerance to treat small eigenvalues as zero.
    """
    # Sigma_star: your singular covariance matrix
    eigvals, eigvecs = np.linalg.eig(Sigma_star)
    
    # If the eigenvalues are complex but small, take only the real part
    if np.max(np.abs(np.imag(eigvals))) > tolerance:
        raise Exception(f"In math_utils.pseudo_inverse_RH, the largest imaginary component in the eigenvalues was {np.max(np.abs(np.imag(eigvals)))} > tol={tolerance}.")
    else:
        eigvals = np.real(eigvals)
        print(f"In math_utils.pseudo_inverse_RH, the largest imaginary component in the eigenvalues was {np.max(np.abs(np.imag(eigvals)))} < tol={tolerance}, so only the real part was kept.") if verbose else None
        print(f"eigvals = {eigvals}") if verbose else None

    # Calculate log of non-zero eigenvalues
    log_eigvals = np.zeros_like(eigvals)
    log_eigvals[eigvals > tolerance] = np.log(eigvals[eigvals > tolerance])
    print(f"In math_utils.pseudo_inverse_RH, the number of imaginary log eigenvalues was {np.sum(np.abs(np.imag(log_eigvals)) > 0)}.") if verbose else None
    print(f"log_eigvals = {log_eigvals}") if verbose else None
    
    # Invert only the non-zero eigenvalues
    pseudo_inv_eigvals = np.zeros_like(eigvals)
    pseudo_inv_eigvals[eigvals > tolerance] = 1.0 / eigvals[eigvals > tolerance]
    pseudo_inv_eigvals[eigvals <= tolerance] = 0 # make exactly zero
    print(f"In math_utils.pseudo_inverse_RH, the number of imaginary pseudo-inverse eigenvalues was {np.sum(np.abs(np.imag(pseudo_inv_eigvals)) > 0)}.") if verbose else None
    print(f"pseudo_inv_eigvals = {pseudo_inv_eigvals}") if verbose else None

    # Create the pseudo-inverse of the eigenvalue matrix
    # Sigma_pseudo_inv_old = eigvecs @ np.diag(pseudo_inv_eigvals) @ (eigvecs.T) # old
    eigvecs_diag_root = eigvecs @ np.diag(np.sqrt(pseudo_inv_eigvals))
    Sigma_pseudo_inv = eigvecs_diag_root @ eigvecs_diag_root.T
    # print(f"Max diff in pseudo-inverse old and new: {np.max(np.abs(Sigma_pseudo_inv - Sigma_pseudo_inv_old))}") if verbose else None
    # Sigma_pseudo_inv_rounded = copy.deepcopy(Sigma_pseudo_inv)
    # Sigma_pseudo_inv_rounded[np.abs(Sigma_pseudo_inv_rounded) < tolerance] = 0
    # Sigma_pseudo_inv_rounded = sparse.csr_matrix(Sigma_pseudo_inv_rounded)

    # Return only real values or raise exception
    if np.max(np.abs(np.imag(Sigma_pseudo_inv))) > tolerance:
        raise Exception(f"In math_utils.pseudo_inverse_RH, the largest imaginary component in the pseudo-inverse was {np.max(np.abs(np.imag(Sigma_pseudo_inv)))} > tol={tolerance}.")
    else:
        Sigma_pseudo_inv = np.real(Sigma_pseudo_inv)
        print(f"In math_utils.pseudo_inverse_RH, the largest imaginary component in the pseudo-inverse was {np.max(np.abs(np.imag(Sigma_pseudo_inv)))} < tol={tolerance}, so only the real part was kept.") if verbose else None

    return(log_eigvals, Sigma_pseudo_inv)

###################### ###################### ###################### 
######################      Distances         ######################
###################### ###################### ###################### 

def Euclidean_dist(n, verbose=False):
    """Return a distance matrix H
    computed with Euclidean distance."""
    positions = np.arange(n)  # Positions: [0, 1, ..., n]
    H = np.abs(positions[:, None] - positions)
    if verbose:
        print('H.shape:', H.shape)
        print('H:\n', H)
    return {'H':H}

def great_circle_dist(n, verbose=False):
    """Return a circulant distance matrix H and its base, 
    computed with great-circle distance."""
    if n == 1:
        base_H_temp = np.array([0])
    if n > 1 and n % 2 == 0:
        base_H_temp = np.array(list(range(math.floor(n/2))) + list(range(math.floor(n/2), 0, -1))) 
    if n > 1 and n % 2 != 0:
        base_H_temp = np.array(list(range(math.floor(n/2) + 1)) + list(range(math.floor(n/2), 0, -1))) 
        
    base_H = base_H_temp.reshape((n, 1)) 
    H = linalg.circulant(base_H.reshape(-1)).T 
    if verbose:
        print('base_H_temp:', base_H_temp)
        print('H.shape:', H.shape)
        print('In the great-circle distance calculation in the Lattice class.\nH base:', base_H.T)
        print('H:\n', H)        
    return {'base_H':base_H, 'H':H}

