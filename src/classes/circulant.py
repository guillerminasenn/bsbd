"""Circulant matrix utilities.

Supports level-1 and level-2 circulants with FFT-based operations.
"""

import numpy as np
from scipy import linalg


###################### ###################### ###################### 
######################      Circulant         ######################
###################### ###################### ###################### 

class Circulant():
    """Level 1 or 2 circulant.
    Params:
    -------
    base: (nx1) array, where n is the order of the circulant.
    -------
     
    """
    def __init__(self, base):
        self.nv = base.shape[0]
        self.nh = base.shape[1]
        self.n = self.nv * self.nh
        self.base = base.reshape((self.nv, self.nh))
    
    def build_matrix(self, base=None):
        """Build the explicit form of the circulant matrix from its base.
        Works for both level-1 and level-2 circulants.
        """
        if base is None:
            base = self.base
            
        if self.nh == 1:
            matrix = linalg.circulant(base.reshape(-1)).T
        if self.nh >= 1:
            if self.n > 300:
                print('Circulant class: Be aware that building the {0}x{1} matrix might be computationally intensive.'.format(self.n, self.n))
            first_brow = np.hstack([linalg.circulant(base[:, j]) for j in np.arange(self.nh)])
            A = np.zeros((self.n, self.n))
            for i in np.arange(0, self.n, self.nv):
                A[i:(i + self.nv), :] = np.roll(first_brow, i, axis=1)
            matrix = A
        if base is self.base:
            self.matrix = matrix
        return matrix
        
    def build_base(self, matrix=None):
        """Given a matrix which is an instance of the Circulant class, return its base.
        
        **Don't overwrite the base because it's been required when the class was created.**
        
        Params:
        ------
        matrix: Circulant whose basis we want. If None, return the base of self.
        
        Return:
        -------
        base: (nv, nh) array.
        """
        if matrix is None:
            matrix = self
            print('matrix:\n', matrix)
        if matrix.shape != (self.n, self.n):
            raise Exception('Create instance of Circulant with the correct matrix order.')
            
        first_brow = matrix[:(self.nv), :]
        blocks = [first_brow[:(self.nv), i:(i + (self.nv))] for i in np.arange(0, self.n, self.nv)]
        base = np.array([block[0, :].T for block in blocks]).T
        return base
        
    def invert_circ(self, base=None):
        """Compute the base of the inverse of a circulant with base.
        Works for both level-1 and level-2 circulants.

        Return:
        -------
        q: (nv, nh) array which is the base of the inverse matrix.
        """
        if base is None:
            base = self.base
        internal_term = np.power(np.real(np.fft.fft2(base, norm='ortho')), -1)
        q = np.real(np.fft.ifft2(internal_term, norm='ortho')) / self.n
        
        if base is self.base:
            self.base_inv = q
        return q

    def multiply_circ(self, base_b): 
        """Compute the product of two circulant matrices A and B of the same
        order, with bases "base" and base_b.
        
        Return:
        -------
        base_ab: (nv, nh) array which is the base of the product AB.
        """
        
        if base_b.shape != self.base.shape:
            raise Exception("Circulants not of the same order.")
            
        norm = np.sqrt(self.n)
        internal_term = np.fft.fft2(self.base, norm='ortho', ) * np.fft.fft2(base_b, norm='ortho')
        base_ab = norm * np.fft.ifft2(internal_term, norm='ortho')
        base_ab = np.real(base_ab).reshape((self.nv, self.nh))
        return base_ab

    def multiply_matrix_vector_circ(self, b): # test 
        """Compute the matrix-vector product of the circulant matrix
        with the vector b.
        NOTE: only implemented for circulant.
        Q: don't i need this for the level-2 circulant? in that case i could adapt."""
        internal_term = np.fft.fft(self.base, norm='ortho') * np.fft.ifft(b, norm='ortho')
        u = np.fft.fft(internal_term).reshape((self.n, 1))
        return u
    
    def compute_eigenvalues_circ(self):
        """Compute the eigenvalues of the circulant.
        NOTE: only implemented for circulant.
        Q: don't i need this for the level-2 circulant? in that case i could adapt."""
        F = np.fft.fft(np.eye(self.n))
        eigenvalues = F @ self.base
        return eigenvalues 
    
    def check_pd_circ(self, epsilon=0.000000001, verbose=False):
        """Check if a circulant C is positive-definite. 
        Numerical errors are accounted at tolerance level epsilon.
        NOTE: only implemented for circulant.
        Q: don't i need this for the level-2 circulant? in that case i could adapt."""

        eigenvalues = compute_eigenvalues_circ(self)

        # Is C real?
        nr_imag_elems = np.sum(np.imag(self.base) != 0)
        nr_significative_imag_elems = np.sum(np.abs(np.imag(self.base)) > epsilon)
        is_real_exact = nr_imag_elems == 0
        is_real_tol = nr_significative_imag_elems == 0

        # Are all eigenvalues real?
        nr_imag_eig = np.sum(np.imag(eigenvalues) != 0)
        nr_significative_imag_eig = np.sum(np.abs(np.imag(eigenvalues)) > epsilon)
        all_eig_real_exact = nr_imag_eig == 0
        all_eig_real_tol = nr_significative_imag_eig == 0

        # Are all (real) eigenvalues positive?
        nr_nonpos_eig = np.sum(np.real(eigenvalues) <= 0)
        nr_significative_nonpos_eig = np.sum(np.real(eigenvalues) <= -epsilon)
        all_eig_pos_exact = nr_nonpos_eig == 0
        all_eig_pos_tol = nr_significative_nonpos_eig == 0

        # Is C positive definite?
        if verbose:
            print(f'Exact: {nr_imag_elems} imaginary elements, {nr_imag_eig} imaginary eigenvalues, {nr_nonpos_eig} real non positive eigenvalues.')
            print(f'Imaginary eigenvalues: min. magnitude: {np.min(np.abs(np.imag(eigenvalues)))}, max. magnitude: {np.max(np.abs(np.imag(eigenvalues)))}')
            print(f'Real eigenvalues: min: {np.min(np.real(eigenvalues))}, max: {np.max(np.real(eigenvalues))}')
            print(f'With epsilon tolerance: {nr_significative_imag_elems} imaginary elements, {nr_significative_imag_eig} imaginary eigenvalues, {nr_significative_nonpos_eig} real non positive eigenvalues.')
        pos_def = (is_real_tol * all_eig_real_tol * all_eig_pos_tol) == 1
        return pos_def
