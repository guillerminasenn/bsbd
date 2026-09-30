"""Lattice utilities for AVA grids and wavelet supports.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.
"""

# Standard library imports
import math

# Third-party library imports
import numpy as np

# Local library imports
from src.utils.math_utils import Euclidean_dist, great_circle_dist

###################### ###################### ###################### 
######################       Lattice          ######################
###################### ###################### ###################### 

class Lattice():
    """Lattice geometry for AVA grids.

    Supports both Euclidean and cyclic topologies and tracks wavelet support
    needed for blur/wavelet `w` and image/reflectivity `c` updates.

    Attributes:
        nv_ava (int): Number of rows in the observed AVA lattice.
        nh_ava (int): Number of columns in the observed AVA lattice.
        k (int): Number of elements in the wavelet (including endpoints).
        topology (str): Topology of lattice, 'E' for Euclidean or 'C' for cyclic.
    """
    def __init__(self, nv_ava, nh_ava, k, mv=None, mh=None, l=None, r=None, topology=None):
        """
        nv_ava: number of rows in the observed AVA lattice.
        nh_ava: number of columns. 
        k: Number of elements in the wavelet (including the endpoints).
        topology: Topology of lattice, E for Euclidean or C for cyclic."""
        
        if topology not in np.array(['E', 'C']):
            raise Exception('Unknown topology.')
        self.topology = topology
        
        # Initialize dimensions of the AVA lattice
        self.nv_ava = nv_ava
        self.nh_ava = nh_ava
        self.n_ava = nv_ava * nh_ava
        
        # Initialize length of the wavelet
        if k <= nv_ava:
            self.k = k
            
            # If side lobe lenghts not specified, assume symmetric wavelet
            if l is None or r is None:
                if k % 2 == 0:
                    self.l = k // 2
                    self.r = self.l - 1
                else:
                    self.l = k // 2
                    self.r = self.l
                    
            # When both side lobes lengths are specified
            if (l is not None) and (r is not None):
                if (l + r + 1 == k):
                    self.l = l
                    self.r = r     
                else:
                    raise Exception('l + k + 1 != k.')
        else: 
            raise Exception('The wavelet cannot be longer than the number of rows in the lattice.')
        
        # Initialize margins to passed or default argument
        self.mv = mv if mv is not None else math.floor(2 * self.k) 
        self.mh = mh if mh is not None else self.nh_ava
        
        # Make margins even, extend lattice and create distance matrices
        self._initialize_lattice()
        
    def _initialize_lattice(self):
        """Encapsulate initialization logic for margins and dependent properties."""
        self._even_margins()
        self._extend_lattice()
        self._create_distance_matrices()
        
    def _even_margins(self):
        """Ensure margins are even."""
        if self.mv % 2 != 0: # make it even
            self.mv += 1
        if self.mh % 2 != 0: # make it even
            self.mh += 1
    
    def _extend_lattice(self):
        """The new lattice dimensions are the original dimensions plus the margins
        in each direction."""
        self.nv = self.nv_ava + self.mv
        self.nh = self.nh_ava + self.mh
        self.n = self.nv * self.nh
        
    def _create_distance_matrices(self):
        """Create distance matrices on the lattice."""
        if self.topology == 'E':
            self.dist = {'dist_v':Euclidean_dist(self.nv), 'dist_h':Euclidean_dist(self.nh)}
        if self.topology == 'C':
            self.dist = {'dist_v':great_circle_dist(self.nv), 'dist_h':great_circle_dist(self.nh)}
       
    def modify_margins(self, mv=None, mh=None, verbose=False):
        """Change the margins for the extended lattice and recreate the lattice
        dimensions and distance matrices."""
        
        if verbose: print('Modifying margins.')
        self.mv = mv if mv is not None else self.mv
        self.mh = mh if mh is not None else self.mh
        self._initialize_lattice()

    def create_ava_positions(self, verbose=False):
        """Create the ava_positions dict with the location 
        of the observed data in the extended lattice."""
        
        # The horizontal and vertical margins are split in two
        half_margin_h = math.floor(self.mh / 2)
        half_margin_v = math.floor(self.mv / 2)
        
        # The AVA lattice starts after the left half of the horizontal margin
        ava_start_h = half_margin_h # because python starts from 0
        ava_end_h = ava_start_h + self.nh_ava

        # The AVA lattice starts after the top half of the vertical margin
        ava_start_v = half_margin_v
        ava_end_v = ava_start_v + self.nv_ava

        # This is to select the AVA lattice from a 2D array representing the extended lattice
        ava_coords = np.meshgrid(
            np.arange(start=ava_start_h, stop=ava_end_h, dtype='int'),
            np.arange(start=-self.nv_ava / 2, stop=self.nv_ava / 2, dtype='int')
        )

        if verbose:
            print(f'Position of AVA lattice in the ext. lattice: from column={ava_start_h} up and to column={ava_end_h - 1}')
        
        self.ava_positions = {
            'ava_start_h':ava_start_h,
            'ava_end_h':ava_end_h,
            'ava_start_v':ava_start_v,
            'ava_end_v':ava_end_v,
            'ava_coords':ava_coords
        }
            
    def create_obs_reflectivity_positions(self, well_column_ava=None, rows_ava=None, verbose=True):
        """Create quantities necessary to locate and subset the observed reflectivity.
        Right now allows for observing reflectivity in a few or all positions within a single column.
        
        Params:
        ------
        well_column_ava: Column of the AVA lattice where the well is located, counting from 0. 
                If None, assume the well is located at column nh/2 in the AVA lattice
                (the first column of the second half of the AVA lattice).
        rows_ava: rows of the column where the reflectivity is observed. If 'all', assumes 
            all rows have been observed. Else, pass a np.array with the observed rows.
        """
        
        # Initialize to None
        whole_column = False
        well_column_ava = None
        well_column = None
        well_start_v = None
        well_length = None
        well_end_v = None
        well_start_vec = None
        well_end_vec = None
        well_coords_vec = None
        well_coord = None
        
        # By default, the only well is located at column nh/2 of the AVA lattice
        if well_column_ava is None:
            well_column_ava = math.floor(self.nh_ava / 2)
            well_column = self.ava_positions['ava_start_h'] + well_column_ava
        else:
            well_column = self.ava_positions['ava_start_h'] + well_column_ava
            
        # By default, the only well occupies the whole well_column
        if rows_ava is None:
            well_start_v = self.ava_positions['ava_start_v']
            well_length = self.nv_ava
            well_end_v = well_start_v + well_length
            rows_ava = np.arange(well_length)

            # To subset the well from the vectorized reflectivity field
            well_start_vec = well_start_v + well_column * self.nv
            well_end_vec = well_start_vec + well_length
            well_coords_vec = np.arange(start=well_start_vec, stop=well_end_vec, dtype='int')

            # To subset the well from the 2d array representing the extended lattice
            well_coords = np.meshgrid(
                np.arange(start=well_column, stop=well_column + 1, dtype='int'),
                np.arange(start=-self.nv_ava / 2, stop=self.nv_ava / 2, dtype='int')
            )
            
        # But the user can choose which nodes in well_column are observed
        else:  
            # Check that the passed row indexes are allowed by the dimensions of the lattice
            if np.max(rows_ava) > self.nv_ava:
                raise Exception('You are passing rows indexes that exceed the nr of rows in the lattice.')
                
            rows_cyclic = rows_ava + self.mv // 2
            
            # To subset the well from the vectorized reflectivity field
            well_coords_vec = rows_ava + well_column * self.nv + self.mv // 2
            
            # To subset the well from the 2d array representing the extended lattice
            well_coords = np.meshgrid(
                np.arange(start=well_column, stop=well_column + 1, dtype='int'),
                rows_ava + self.nv + self.mv // 2
            )      
            
        if verbose:
            print("Whole column? {3}. \nThe well is located at column {0} of the AVA lattice, which is column {1} of the extended lattice, and occupies rows {2}".format(well_column_ava, well_column, rows_ava, whole_column))
        
        self.well_positions = {
            'well_column_ava':well_column_ava,
            'well_column':well_column,
            'rows_ava':rows_ava, 
            'rows_cyclic':rows_cyclic,
            'well_start_v':well_start_v,
            'well_length':well_length,
            'well_end_v':well_end_v,
            'well_start_vec':well_start_vec,
            'well_end_vec':well_end_vec,
            'well_coords_vec':well_coords_vec,
            'well_coords':well_coords
        }

    def create_wavelet_positions(self, verbose=False):
        """Position the wavelet (including endpoints) in a vertical column of the lattice.
        The center element of omega is positioned at the nv/2-element of the vertical lattice.
        Then the first element of omega is positioned at element nv/2 - l of the vertical lattice."""
            
        # The first element of omega
        wavelet_start_v = math.floor(self.nv / 2) - self.l
        wavelet_end_v = math.floor(self.nv / 2) + self.r + 1 # the +1 is because Python starts at 0
        
        # Do I need this? Change the name to something more intuitive, like nr_wavelet_constraints_left
        wavelet_margin_left = wavelet_start_v
        wavelet_margin_right = self.nv - wavelet_end_v

        if verbose:
            print(f'Original wavelet length = {self.k}; left and right lobes hav {self.l} and {self.r} weights.')
            print(f'The extended wavelet vector with {self.nv} points embeds the original wavelet, '
                  f'from element index {wavelet_start_v} up and to element {wavelet_end_v - 1},')
            print(f'leaving {wavelet_margin_left} constrained elements to the left and {wavelet_margin_right} to the right.')

        self.wavelet_positions = {
            'wavelet_start_v': wavelet_start_v,
            'wavelet_end_v': wavelet_end_v,
            'wavelet_margin_left': wavelet_margin_left,
            'wavelet_margin_right': wavelet_margin_right
        }
            
           