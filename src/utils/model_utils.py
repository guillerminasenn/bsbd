"""This module contains helper functions to implement the MCMC sampler bp1
on the torus.

New Functions:
-------------
vertical_margin
exp_cor
recover_1d_lattice
embed_omega
create_W
create_C
build_c_vector
subset_AB
subset_ABA

-------------
create_reflectivity_constraint_matrices
create_data_constraint_matrices
create_wavelet_constraints

"""

# Standard library imports
import math
import copy

# Third-party library imports
import numpy as np
import pandas as pd
from scipy import linalg, sparse

# Local library imports
from src.utils import math_utils
from src.classes.lattice import Lattice
from src.utils.efficient_algebra_utils import transpose_base_circ

###################### ###################### ###################### 
######################       New functions    ######################
###################### ###################### ###################### 

def vertical_margin(k, phi, thres=0.05, verbose=False):
    """Choose vertical margin based on the length of the wavelet
    and the correlation range in the wavelet covariance matrix.
    
    Parameters:
    -----------
    k: int
        Wavelet length.
    phi: float
        Wavelet correlation range parameter.
    thres: float
        Correlation strength at which we consider the correlation as non significant. 
        
    Returns:
    --------
    mv: int
        The number of nodes to add in the vertical direction of the lattice.
    """
    
    # Distance at which the correlation decays to 0.05, given phi and squared exp. corr. fn
    distance_005 = np.sqrt(- phi**2 * np.log(0.05))
    
    # Vertical margin
    mv = int(np.max([k, 2 * distance_005]))
    
    if verbose: 
        print(f'distance_005={distance_005}; margin={mv}')
    return mv

def exp_cor(rho, p, H=None, h=None):
    """Return a correlation matrix R = exp{(-H/rho)^p}. If only H is passed, return R.
    If h is passed, assume circulant case with h = base(R) and return r and R = circ(r).

    Params:
    -------
    rho: Correlation range.
    p: Governs smoothness. Use p=1.98 for squared exponential and p=1 for exponential.
    H: Distance matrix.
    h: Base of the distance matrix.
    explicit: True to return both correlation matrix and base of the matrix in the circ. case.

    Return:
    -------
    R: Correlation matrix.
    r: base of R (only in circulant case)."""

    # Basic checks
    if H is None and h is None:
        raise Exception('I need the distance matrix or its base.')
    if np.imag(rho) != 0 or rho <= 0:
        raise Exception('The correlation range must be real and positive.')

    # Non-circulant 
    if H is not None:
        R = np.exp(-(H / rho) ** p)
        return {'R':R, 'base_R':None}

    # Circulant
    if h is not None:
        r = np.exp(-(h / rho) ** p)
        R = linalg.circulant(r.reshape(-1)).T
        return {'R':R, 'base_R':r}

# def recover_1d_lattice_kdim(lattice, ava=False):
#     """Given a lattice object, keep only the vertical lattice (and without horizontal margins).
#     Params:
#     -------
#     lattice: Lattice object.
#     (ava: if True, will recover the vertical 1D lattice only for the AVA rows.) (not implemented)
#     """
#     nv_ava = lattice.nv_ava
#     nh_ava = 1
#     k = lattice.k
#     l = lattice.l
#     r = lattice.r
#     mv = lattice.mv
    
#     # The 1d lattice will have k rows and be Euclidean
#     lattice_1d = Lattice(k, nh_ava, k, l=l, r=r, mv=0, mh=0, topology='E')
#     lattice_1d.wavelet_positions = lattice.wavelet_positions

#     return lattice_1d

def recover_1d_lattice(lattice):
    """Given a 2D lattice object, extract the 1D lattice needed to create the wavelet.
    
    Params:
    -------
    lattice: Lattice object.
    
    Return:
    ------
    lattice_1d: Vertical Euclidean (length-k) or Cyclic (length-nv) lattice.
    """
    k = lattice.k
    l = lattice.l
    r = lattice.r
    
    if lattice.topology == 'E':
        # The 1d lattice will have k rows and be Euclidean
        lattice_1d = Lattice(lattice.nv, 1, k, l=l, r=r, mv=0, mh=0, topology='E')
        lattice_1d.wavelet_positions = lattice.wavelet_positions
    
    if lattice.topology == 'C':
        # The 1d lattice will have nv rows and be Cyclic
        lattice_1d = Lattice(lattice.nv, 1, k, l=l, r=r, mv=0, mh=0, topology='C')
        lattice_1d.wavelet_positions = lattice.wavelet_positions

    # back up
    # if lattice.topology == 'E':
    #     # The 1d lattice will have k rows and be Euclidean
    #     lattice_1d = Lattice(k, 1, k, l=l, r=r, mv=0, mh=0, topology='E')
    #     lattice_1d.wavelet_positions = lattice.wavelet_positions
    
    # if lattice.topology == 'C':
    #     # The 1d lattice will have nv rows and be Cyclic
    #     lattice_1d = Lattice(lattice.nv, 1, k, l=l, r=r, mv=0, mh=0, topology='C')
    #     lattice_1d.wavelet_positions = lattice.wavelet_positions

    return lattice_1d

def embed_omega_old(lattice, omega):
    """Embed the original wavelet omega with length k into a lengh-nv vector of zeros, 
    to get an extended vector w of length nv."""
    if lattice.nv % 2 != 0:
        raise Exception('The pad omega function requires even number of rows in the lattice (nv).')
        
    # omega = np.array(omega)
    w = np.zeros((lattice.nv, 1))
    start = lattice.nv // 2 - lattice.l
    end = lattice.nv // 2 + lattice.r + 1
    
    # w[start:end, 0] = np.squeeze(omega)
    w[start:end, 0] = omega.reshape(-1)
    return w

def embed_wu(wu, _w, verbose=False):
    """Embed the random part wu of the wavelet, with length K=k-nr_constraints, 
    into a vector of zeros with length equal to the length of the 1d lattice used
    for the wavelet. In the Euclidean case this will mean extended w will have 
    length k, and in the cyclic, it'll have length  nv.
    
    Params:
    -------
    wu: Random vector with wavelet elements. 
    _w: Wavelet object (containing info about the constraints and the lattice).
    
    Return:
    -------
    wu: A length-k vector with the purely random wu wavelet weights embedded.
    """
    n = _w.lattice.n
    nv = _w.lattice.nv
    if n % 2 != 0:
        raise Exception('The pad omega function requires even number of rows in the lattice (nv).')
        
    # Create extended wavelet vector and fill with zeros
    w = np.zeros((n, 1))
    
    # If the wavelet is constrained, consider the endpoints
    if _w.wavelet_constraints['constr']:
        start = n // 2 - _w.lattice.l + 1
        end = n // 2 + _w.lattice.r + 1 - 1
    else:
        start = nv // 2 - _w.lattice.l
        end = nv // 2 + _w.lattice.r + 1
    
    w[start:end, 0] = wu.reshape(-1)
    return w

def pad(lattice, w_star, verbose=False):
    """Pad a length-k w* = w|Aw=0 with zeros to obtain a w_ext with length nv,
    the nr of rows in the 2D lattice.
    
    Params:
    -------
    lattice: The 2D lattice.
    w_star: Constrained wavelet w*|Ax=0 with length the size of the wavelet lattice.

    Return:
    -------
    w_star_pad: Extended constrained wavelet with length the size of the 2D vertical dimension.
    """
    
    w_star_arr = np.asarray(w_star).reshape(-1, 1)
    if w_star_arr.shape[0] == lattice.nv:
        w_star_pad = w_star_arr
    elif w_star_arr.shape[0] == lattice.k:
        w_star_pad = np.zeros((lattice.nv, 1))
        start = lattice.nv // 2 - lattice.l
        end = lattice.nv // 2 + lattice.r + 1
        w_star_pad[start:end, 0] = w_star_arr.reshape(-1)
    else:
        raise ValueError(
            f"pad: unexpected wavelet length {w_star_arr.shape[0]} (expected {lattice.k} or {lattice.nv})."
        )
            
    if verbose:
        print(f'l = {lattice.l}, r = {lattice.r}')
        print(f'w_star = \n{np.round(w_star_arr, 4)}')
        print(f'w_star_pad = \n{np.round(w_star_pad, 4)}')
        
    return w_star_pad
    
def create_W(lattice, w_star, verbose=False, explicit=False):
    """
    Create wavelet convolutional matrices depending on the type of convolution.
    Choosing zero-padded convolution for Euclidean lattices and 
    circular convolution for cyclic lattices. 

    Params:
    -------
    lattice: The 2D lattice.
    w_star: Constrained wavelet w*|Ax=0 with length the size of the wavelet lattice.
    explicit: If True, create and store matrices W0 and W in the circulant case and not only their bases.

    Return:
    -------
    W0: nv x nv matrix that performs the (circulant or edge) convolution on one column. 
    W: sparse n x n matrix to perform the convolution on the lattice (None for cyclic).
    base_W0: For cyclic ('C') topology, the base vector used to construct W0 (None for Euclidean).
    """

    # Extract dimensions
    nv = lattice.nv
    nh = lattice.nh
    l = lattice.l
    r = lattice.r
    
    # Make 0 the 0
    w_star[np.abs(w_star) < 0.00000000001] = 0
    
    if lattice.topology == 'C': # Circular convolution 
        W0 = None
        W = None
            
        # Reverse and shift w* to build base(W0)
        w_rev = w_star[::-1] # reverse w
        w_rev_shift = np.roll(w_rev, nv // 2 + 1) # Shift: 1 position so omega_0 is at position nv/2, and then nv/2 positions 
        if verbose:
            print(f'w_rev_shift (base) = \n{np.round(w_rev_shift, 4)}')

        # Build and store bases for W0 and W
        base_W0 = w_rev_shift
        base_W0T = transpose_base_circ(base_W0)
        if verbose:
            print(f'base_W0T = \n {base_W0T}')
            
        one_vector = np.zeros((1, lattice.nh))
        one_vector[0, 0] = 1      
        base_W = linalg.kron(one_vector, base_W0)       
        base_WT = linalg.kron(one_vector, base_W0T)
        
        # Create (sparse) explicit matrices W0 and W
        if explicit and lattice.n < 400:
            W0 = linalg.circulant(w_rev_shift.reshape(-1)).T
            W0 = sparse.csr_matrix(W0)
            W = sparse.kron(sparse.identity(nh, format='csr'), W0, format='csr')
            if verbose:
                print(f'W0 (sparse) = \n{pd.DataFrame(W0.toarray())}')
                print(f'W (sparse) = \n{pd.DataFrame(W.toarray())}')

    elif lattice.topology == 'E': # Zero-edge convolution
        if lattice.n >400:
            raise Exception('Euclidean lattice too big for explicit wavelet convolutional matrices.')
            
        base_W0 = base_W0T = None  # No base in Euclidean
        base_W = base_WT = None

        # Pad the wavelet
        w_star_pad = pad(lattice, w_star, verbose=verbose)

        # Ensure w_star_pad is 1D
        w_star_pad = np.asarray(w_star_pad).reshape(-1)
        
        # Reverse and shift
        w_rev = w_star_pad[::-1]
        w_rev_shift = np.roll(w_rev, nv // 2 + 1)

        # Create dense circulant matrix
        W0_temp = linalg.circulant(w_rev_shift.reshape(-1)).T

        # Zero out wrapping-around elements
        W0_zero_edge = np.zeros_like(W0_temp)
        for i in range(nv):
            valid_indices = range(max(0, i - r), min(nv, i + l + 1))
            W0_zero_edge[i, valid_indices] = W0_temp[i, valid_indices]

        # Make W0 sparse
        W0 = sparse.csr_matrix(W0_zero_edge)
        
        # Kronecker product for the 2D lattice convolution
        W = sparse.kron(sparse.identity(nh, format='csr'), W0, format='csr')
                
        if verbose:
            print(f'w_star_pad = \n{np.round(w_star_pad, 4)}')
            print(f'w_rev_shift = \n{np.round(w_rev_shift, 4)}')
            print(f'W0_zero_edge (sparse) = \n{W0.toarray()}')
            
    else:
        raise ValueError(f"Unknown topology {lattice.topology}")

    return (W0, W, base_W0, base_W, base_W0T, base_WT)
    
def create_C(lattice, c, verbose=False):
    """
    Compute Gamma, the n x nv rectangular matrix that performs the convolution such that
        Gamma @ w = W @ c, 
    being W the nxn convolutional matrix with the wavelet w in each row. Note that k 
    is the length of omega, the useful part of the extended wavelet w with length nv. 

    Params:
    ------
    c: nx1 reflectivity vector.
    
    Return:
    ------
    bases: nv x nh matrix that at column j contains the first row of the circulant 
        block j in the matrix C.
    Gamma: n x nv matrix that performs the convolution such that Gamma @ w = W @ c,
    """
    # Rename for readability
    nv = lattice.nv
    nh = lattice.nh

    # Create the bases from the rc vector
    bases_orig = c.reshape((nv, nh), order='F')
    bases_inv = bases_orig[::-1] # invert all bases
    bases_shifted = np.roll(bases_inv, 1, axis=0) # shift (center + margin)
    bases = np.vstack(
        [bases_shifted[(math.floor(nv / 2)):], 
        bases_shifted[0:(math.floor(nv / 2))]])   
    
    if lattice.n < 400:
        # Create Gamma for Euclidean topology 
        blocks = [linalg.circulant(bases[:, j]).T for j in range(nh)] # create circulant blocks
        if verbose:
            print(f'blocks:\n{blocks}')
        if lattice.topology == 'E':
            l = lattice.nv // 2
            r = l - 1
            Gamma_zeros = [np.zeros((nv, nv)) for _ in range(nh)]

            # Fill in the blocks of Gamma using elements of the circulant blocks, but making zero the edge elements 
            # NOTE: This approach is very unefficient
            for j in range(nh):
                for i in range(nv):
                    valid_indices = range(max(0, i - r), min(nv, i + l + 1))
                    Gamma_zeros[j][i, valid_indices] = blocks[j][i, valid_indices]
            Gamma = np.vstack(Gamma_zeros)

            # # If k < nv, remove the last nv - k columns of Gamma
            # if lattice.k < nv:
            #     Gamma = Gamma[:, :lattice.k]
            #     if verbose:
            #         print(f'Gamma (Euclidean) with {lattice.k} columns:\n{Gamma}')

        # Create Gamma for Cyclic topology (with the circulant blocks)
        if lattice.topology == 'C':
            Gamma = np.vstack(blocks)
    else:
        # print(f'Lattice is too big to create the convolution matrix explicitly.')
        Gamma = None
        
    return(bases, Gamma)

def build_c_vector(lattice, cu, b):
    """Helper method to build c' = {c_u', c_o}, to pass the complete vector c 
    to the likelihood.
    Params:
    ------
    lattice: the 2D lattice.
    cu: The stochastic portion of c.
    b: the constraints co.
    """
    n = lattice.n
    c_vector = np.zeros((n, 1))
    mask = np.ones(n, dtype=bool)  
    mask[lattice.well_positions['well_coords_vec']] = False  # set False for indices in well_coords_vec
    c_vector[mask] = copy.deepcopy(cu)
    c_vector[mask == False] = b
    return c_vector 
    
def subset_ABA(base_B, _c, verbose=False):
    """Compute the matrix product A_co @ B @ A_co.T, with A_co the reflectivity 
    constraint matrix and B a BCCB matrix, by subsetting the correct 
    elements of the base of B. 
    """
    rows_cyclic = _c.lattice.well_positions['rows_cyclic']

    # Compute A @ B @ At by subsetting the relevant elements of Rv
    first_block = linalg.circulant(base_B[:, 0]).T
    ABAt_temp = first_block[rows_cyclic, :] 
    ABAt = ABAt_temp[:, rows_cyclic]
    if verbose:
        print(f'ABAt =\n{ABAt}')
        
    return ABAt

def subset_AB(base_B, _c, verbose=False):
    """Compute the matrix product A_co @ B, with A_co the reflectivity 
    constraint matrix and B a BCCB matrix, by subsetting the correct 
    elements of the base of B. 
    """
    nh = _c.lattice.nh
    well_column = _c.lattice.well_positions['well_column']
    rows_cyclic = _c.lattice.well_positions['rows_cyclic']

    # Build the first block-row of B, shifted to start at the well column
    brow1_base_indexes = list(range(nh))
    brow1_base_indexes_shifted = np.roll(brow1_base_indexes, well_column)
    blocks = [linalg.circulant(base_B[:, j]).T for j in brow1_base_indexes_shifted]
    block_row = np.hstack(blocks)

    # Compute A @ B by subsetting the relevant rows of the block-row
    AB = block_row[rows_cyclic, :] # old new, allows partial and whole column
    if verbose:
        print(f'ASigma =\n{AB}')
        
    return AB

    
    
    
###################### ###################### ###################### 
######################    Old functions       ######################
###################### ###################### ###################### 

def create_reflectivity_constraint_matrices(lattice, pos):
    """Create constraint matrices used to select the observed (c_o) or
    unobserved (c_u) reflectivities. 
    
    If multiple wells, they must be observed in the same rows. 
    
    Params:
    -------
    lattice: Object with the configuration of the lattice.
    pos: Object with the positions of the well on the lattice.
    
    Return:
    -------
    A_co_h: Selection matrix for the columns where the well is located.
    A_co_v: Selection matrix for the rows where the well is located.
    A_co: Contraint matrix to select the well c_o from the refl. vector c.
    A_co_ext: Constraint matrix to select the well from (c, d).
    A_cu: Constraint matrix to subset c_u from c.
    A_co_one: Matrix that has a 1 for the elements in c corresponding to c_o.
    A_co_h_one:
    A_co_v_one:
    """
    # Extract variables for readability
    nh_ava = lattice.nh_ava
    nv_ava = lattice.nv_ava
    nh = lattice.nh
    nv = lattice.nv
    n = lattice.n
    
    # Constraint matrix to select c_o from c (can be built as a kron. prod)
    A_co_h = sparse.csr_matrix((1, nh)) # 1 x nh
    A_co_h[0, pos.well_location_h] = 1
    A_co_v = sparse.csr_matrix((nv_ava, nv)) # nv_ava x nv
    for i in range(pos.well_length):
        A_co_v[i, i + pos.well_start_v] = 1
    A_co = sparse.kron(A_co_h, A_co_v) # (nv_ava x n),

    # Constraint matrix to select c_o from (d, c)
    zeros = sparse.csr_matrix((nv_ava, n)) # Dont select elements of (d)
    A_co_ext = sparse.hstack([zeros, A_co]) # (nv_ava x 2n)
    
    # Matrix that has a 1 in the diagonal for elements in c corresponding to c_o
    A_co_h_one = np.zeros((nh, nh))
    A_co_h_one[pos.well_location_h, pos.well_location_h] = 1
    A_co_v_one = np.zeros((nv, nv))
    for i in np.arange(pos.well_start_v, pos.well_end_v):
        A_co_v_one[i, i] = 1
    # A_co_one = linalg.kron(A_co_h_one, A_co_v_one)

    # Matrix with rows equal to zero for elements in c corresponding to c_o
    A_zero_co = sparse.identity(n, format='csr') # n x n
    for i in pos.well_coords_vec: 
        A_zero_co[i, i] = 0 # method 1
    # A_zero_co2 = I_n.copy() - A_co_one # method 2
    
    # Constraint matrix to subset c_u from c
    # A_cu = np.delete(A_zero_co, pos.well_coords_vec, axis=0) 
    keep_rows = [i for i in range(A_zero_co.shape[0]) if i not in pos.well_coords_vec]
    A_cu = A_zero_co[keep_rows, :]

    return(A_co_h, A_co_v, A_co, A_co_ext, A_cu, A_co_h_one, A_co_v_one)

def create_data_constraint_matrices(lattice, pos):
    """Create constraint matrices to select the unobserved (d_u) or 
    observed (d_o) portions of the extended data vector d. 

    Params:
    -------
    lattice: Object with the configuration of the lattice.
    pos: Object with the positions of the well on the lattice.
    
    Return:
    -------
    X_matrix: Selection matrix for the columns occupied by the AVA lattice. 
    U_matrix: Selection matrix for the rows occupied by the AVA lattice. 
    A_do: Selects the elements corresponding to the observed AVA data. 
    A_do_ext: Constraint matrix to select d_o from (d, c_u).
    A_du: Constraint matrix to select d_u from d.
    """

    # Extract variables for readability
    nh_ava = lattice.nh_ava
    nv_ava = lattice.nv_ava
    n_ava = lattice.n_ava
    nh = lattice.nh
    nv = lattice.nv
    n = lattice.n
    ava_start_h = pos.ava_start_h
    ava_end_h = pos.ava_end_h
    ava_start_v = pos.ava_start_v
    ava_end_v = pos.ava_end_v
    
    # Constraint matrix to select d_o from d
    X_matrix = sparse.lil_matrix((nh_ava, nh))
    X_matrix[:, ava_start_h:ava_end_h] = sparse.identity(nh_ava)
    U_matrix = sparse.lil_matrix((nv_ava, nv))
    U_matrix[:, ava_start_v:ava_end_v] = sparse.identity(nv_ava)
    A_do = sparse.kron(X_matrix, U_matrix)
    
    # Constraint matrix to select d_o from (d, c_u)
    zeros = sparse.csr_matrix((n_ava, n - nv_ava)) # (n_ava x n-nv_ava)
    # zeros = np.zeros((n_ava, n - nv_ava)) # (n_ava x n-nv_ava)
    A_do_ext = sparse.hstack([A_do, zeros]) # n_ava x (2n - nv_ava)
    
    # Matrix that zeroes-out positions in d corresponding to d_o
    NonAVArows = sparse.identity(nh, format='csr') # nh x nh
    for i in np.arange(ava_start_h, ava_end_h):
        NonAVArows[i, i] = 0
    AVArows = np.abs(NonAVArows - sparse.identity(nh)) # nh x nh
    I_v = sparse.identity(nv) # nv x nv
    I_v_ava = sparse.identity(nv, format='csr') # nv x nv
    
    for i in np.arange(ava_start_v, ava_end_v):
        I_v_ava[i, i] = 0
    A_du_with_zeros = sparse.kron(NonAVArows, I_v) + sparse.kron(AVArows, I_v_ava) # n x n
    
    # Constraint matrix to select d_u from d
    # A_du = A_du_with_zeros.copy()[~np.all(A_du_with_zeros == 0, axis=1)] # (n - n_ava) x n
    non_zero_row_indices = A_du_with_zeros.getnnz(axis=1) > 0
    A_du = A_du_with_zeros[non_zero_row_indices]

    return(X_matrix, U_matrix, A_do, A_do_ext, A_du)

def create_wavelet_constraints(nv, wavelet_start_v, wavelet_end_v, 
                               wavelet_margin_left, wavelet_margin_right,
                               sum_one=False, verbose=True):
    """Create the constraint matrix and independent term for the wavelet.
    
    Params:
    -------
    sum_one: (Default is False) To activate the sum-to-one constraint for the wavelet weights.

    Return:
    -------
    A_w_t: constraint matrix.
    b_w_t: independent term.
    """

    # Constraint matrix and independent term 
    A_w_t_temp = np.zeros((nv, nv))
    A_w_t_temp[0:wavelet_start_v, 0:wavelet_start_v] = np.identity(wavelet_margin_left)
    A_w_t_temp[wavelet_end_v:, wavelet_end_v:] = np.identity(wavelet_margin_right)
    A_w_t = np.vstack([A_w_t_temp[0:wavelet_start_v, :], A_w_t_temp[wavelet_end_v:, :]])
    b_w_t = np.zeros(A_w_t.shape[0])
    
#     # Coordinates where the wave is located
#     wave_coords_vec = np.arange(wavelet_start_v, wavelet_end_v)
    
    # Constraint matrix and independent term if sum-to-one condition is enabled
    if sum_one:
        A_w_t_temp = np.zeros((nv + 1, nv)) # with sum-to-zero constraint
        A_w_t_temp[0:wavelet_start_v, 0:wavelet_start_v] = np.identity(wavelet_margin_left)
        A_w_t_temp[wavelet_end_v:-1, wavelet_end_v:] = np.identity(wavelet_margin_right)
        A_w_t_temp[-1, :] = 1 # sum-to-0 constraint
        A_w_t = np.vstack([A_w_t_temp[0:wavelet_start_v, :], A_w_t_temp[wavelet_end_v:, :]])
        b_w_t = np.zeros(A_w_t.shape[0])
        b_w_t[-1] = 1 # sum-to-0 constraint

    # Print
    if verbose:
        print(f'Constraint matrix for wavelet:')
        print(A_w_t)
        print(f'Independent constraint termx for wavelet:')
        print(b_w_t)
    return(A_w_t, b_w_t)

def create_cor_matrix(one_step_cor_h, one_step_cor_v, 
                      nh, nv,
                      unit_dist_h, unit_dist_v,
                      cor_f, ratio_nugget_h, ratio_nugget_v,
                      verbose=False):    
    """ 
    Create the horizontal and vertical correlation matrices, 
    and the spatial correlation matrix as their Kronecker product. 
    
    Params:
    ------
    cor_f: Correlation function.
    one_step_cor_h: Correlation at unit distance in the h. direction.
    one_step_cor_v: Correlation at unit distance in the v. direction.
    unit_distance_h: Unit distance matrix in the h. direction.
    unit_distance_v: Unit distance matrix in the v. direction.
    ratio_nugget_h: Ratio nugget/variance in the h direction.
    ratio_nugget_v: Analogous.
    
    Return:
    ------
    base_R: Base of the spatial correlation matrix.
    R_h: Correlation matrix in the Horizontal direction.
    R_v: Correlation matrix in the Vertical direction.
    """
    
    # Exponential correlation function in both directions
    if cor_f == 'exponential':
        
        # Calculate correlation range parameters.
        phi_h = - 1 / np.log(one_step_cor_h)
        phi_v = - 1 / np.log(one_step_cor_v) 
        
        # Build correlation matrices in both directions.
        R_h = np.exp(-np.abs(unit_dist_h) / phi_h) / (1 + ratio_nugget_h)
        np.fill_diagonal(R_h, 1)
        R_v = np.exp(-np.abs(unit_dist_v) / phi_v) / (1 + ratio_nugget_v)   
        np.fill_diagonal(R_v, 1)
        
        # Build base of the spatial correlation matrix.
        base_R = math_utils.base_kron_block_circulant(R_h, R_v)

    # Squared exponential correlation function in both directions - under dev.
    if cor_f == 'sq_exponential':
        # Calculate correlation range parameters
        phi_h = - 1 / np.log(one_step_cor_h)
        phi_v = - 1 / np.log(one_step_cor_v) 
        # phi_h = 1 / np.sqrt(-np.log(one_step_cor_h * (1 + ratio_nugget_h)))
        # phi_v = 1 / np.sqrt(-np.log(one_step_cor_v * (1 + ratio_nugget_v)))

        # Build correlation matrices
        R_h = np.exp(-(np.abs(unit_dist_h) / phi_h)**2) / (1 + ratio_nugget_h)
        np.fill_diagonal(R_h, 1)
        R_v = np.exp(-(np.abs(unit_dist_v) / phi_v)**2) / (1 + ratio_nugget_v)   
        np.fill_diagonal(R_v, 1)
        base_R = math_utils.base_kron_block_circulant(R_h, R_v)
    
    # Second option for correlation function - under dev.
    if cor_f == '17':
        # Calculate correlation range parameters
        phi_h = 1 / (-np.log(one_step_cor_h))**(1/1.7)
        phi_v = 1 / (-np.log(one_step_cor_v))**(1/1.7)
        # phi_h = - 1 / np.log(one_step_cor_h * (1 + ratio_nugget_h))
        # phi_v = - 1 / np.log(one_step_cor_v * (1 + ratio_nugget_v)) 

        # Build correlation matrices
        R_h = np.exp(-(np.abs(unit_dist_h) / phi_h)**1.7) / (1 + ratio_nugget_h)
        np.fill_diagonal(R_h, 1)
        R_v = np.exp(-(np.abs(unit_dist_v) / phi_v)**1.7) / (1 + ratio_nugget_v)   
        np.fill_diagonal(R_v, 1)
        base_R = math_utils.base_kron_block_circulant(R_h, R_v)

    # Print 
    print(f'Horizontal correlation range: {phi_h}; Vertical correlation range: {phi_v}.')
    print(f"\nThe correlations in the Vertical direction are\n {np.round(R_v[0, :5], 4)}")
    print(f"\nThe correlations in the Horizontal direction are\n {np.round(R_h[0, :5], 4)}")
    print(f"\nThe spatial correlation matrix base is \n {np.round(base_R[:5, :5], 4)}\n")
    if verbose:
        print(f'R_h is PD? {math_utils.check_pd_circulant(R_h, epsilon=0.00000001)}')
        print(f'R_v is PD? {math_utils.check_pd_circulant(R_v, epsilon=0.00000001)}')
        print(f"\nCorrelation type={cor_f}.")
        print(f"\nRatio nugget mer={ratio_nugget_h} y lat={ratio_nugget_v}.")
    
    return(base_R, R_h, R_v)


def select_cor_well_block_circ(base_R, param_drc=None, **kwargs):
    """
    Subset the correlations corresponding to locations at the well
    from the BCCB correlation matrix R with base base_R. 
    In other words, build the matrix 
        subset_R = A @ R @ A ^T
    by choosing the right elements of base_R.
    
    Params:
    -------
    base_R: 
    c_base: Base of the nxn original covariance matrix.
    
    Return:
    -------
    RAt: The dot product R @ A_c^T.
    ARAt: The square desired subset.
    """
    
    # Get values for the parameters
    params = {}
    if param_drc is not None:
        params.update(param_drc)
    params.update(kwargs)
    Params = namedtuple('Params', params.keys()) 
    params = Params(**params) # Put all arguments in the params object
    
    # Rename variables for readability
    well_start = params.well_margin_v
    well_stop = well_start + params.well_length

    # Compute R @ At
    bases = (list(range(params.well_location_h, params.nh)) 
             + list(range(params.well_location_h)))
    blocks = [linalg.circulant(base_R[:, base]).T for base in bases]
    block_row = np.hstack(blocks)
    AR = block_row[well_start:well_stop, :]
    RAt = AR.T 

    # Compute A @ R @ At
    temp = linalg.circulant(base_R[:, 0]).T # this is just Rv
    ARAt = temp[well_start:well_stop, well_start:well_stop]
    return(RAt, ARAt)


def compute_psi(Model, verbose=False):
    nv = Model.lattice.nv
    if verbose:
        r = getattr(Model.lattice, 'r', None)
        l = getattr(Model.lattice, 'l', None)
        print(f"In compute_psi, nv={nv}, r={r}, l={l}.")

    # Vertical correlation of reflectivity (stationary prior correlation).
    if Model.lattice.topology == 'E':
        R_cv = Model.model['c'].Sigma.Rv
    elif Model.lattice.topology == 'C':
        R_cv = linalg.circulant(Model.model['c'].Sigma.base_Rv.reshape(-1)).T
    else:
        raise ValueError(f"Unknown lattice topology: {Model.lattice.topology}")

    # Wavelet conditional correlation (accounts for padding/endpoints constraints).
    w_constraints = Model.model['w'].wavelet_constraints
    unconstrained_indices = w_constraints.get('unconstrained_indices', np.arange(nv))

    R_w_star = w_constraints.get('R_w_star', None)
    if R_w_star is not None:
        R_w = R_w_star.toarray()
    else:
        # Fall back to the unconstrained vertical correlation for w.
        if Model.lattice.topology == 'E':
            R_w = Model.model['w'].Sigma.Rv
        else:
            R_w = linalg.circulant(Model.model['w'].Sigma.base_Rv.reshape(-1)).T

    idx = np.asarray(unconstrained_indices, dtype=int)
    R_cv_subset = R_cv[np.ix_(idx, idx)]
    R_w_subset = R_w[np.ix_(idx, idx)]
    psi = float(np.sum(R_cv_subset * R_w_subset))
    if verbose:
        print(f"psi (sum over unconstrained wavelet indices): {psi}")
        R_sum_temp_endpoints = R_cv_subset * R_w_subset_unc
        np.fill_diagonal(R_sum_temp_endpoints, 0)
        psi_old_endpoints = k + np.sum(R_sum_temp_endpoints)
        print(f"psi (computed from R_w), considering endpoints: {psi_old_endpoints}")
    

    # Compute the psi 
    # R_w_subset = Model.model['w'].wavelet_constraints['R_w_star'].toarray()[start:stop, start:stop]
    # print(f"R_w_subset = {R_w_subset}") if verbose else None
    # R_cv_subset = R_cv[start:stop, start:stop]
    # R_sum_temp = R_cv_subset * R_w_subset
    # psi = np.sum(R_sum_temp)
    # print(f"psi (new way) = {psi}") if verbose else None
    
    return(psi)


def compute_sigma_d_sq(param_w, param_drc, theta):
    """Compute the variance in the data based on the formula
    sigma_d_sq = Var(w*c|phi)*h^2."""
    var_signal = compute_var_signal(param_w, param_drc, theta)
    sigma_d_sq = var_signal * theta['zeta']
    return(sigma_d_sq)


def compute_var_signal(param_w, param_drc, theta):
    """Compute the variance of the signal, Var(w*c|phi)."""

    # Load fixed parameters
    nv = param_drc['nv']
    start = math.floor(nv / 2) - param_w['r']
    stop = math.floor(nv / 2) + param_w['l'] + 1    
    gaussian_exp = param_w['gaussian_exp']
    start = math.floor(nv / 2) - param_w['r']
    stop = math.floor(nv / 2) + param_w['l'] + 1    

    # Load current states
    sigma_u_sq = theta['sigma_u_sq']
    sigma_c_sq = theta['sigma_c_sq']

    # Compute the variance
    R_u_subset = (np.exp(-(np.abs(param_w['unit_dist_v'][start:stop, start:stop]
                                  / theta['phi_u'])**gaussian_exp))) # 'Gaussian' cor
    R_sum_temp = param_drc['R_c_subset'] * R_u_subset
    np.fill_diagonal(R_sum_temp, 0)
    R_sum = np.sum(R_sum_temp)
    # var_signal = sigma_u_sq * sigma_c_sq * (param_drc['nv'] + R_sum) # old
    var_signal = sigma_u_sq * sigma_c_sq * (param_w['k'] + R_sum) # old
    return(var_signal)

