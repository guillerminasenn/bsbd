"""Convolutional model configuration and simulation helpers.

Terminology: `w` is the blur/wavelet and `c` is the image/reflectivity.
"""

# Standard library imports
import copy
import pickle

# Third-party library imports
import numpy as np
import uuid

# Local library imports
from src.classes.covariance import Covariance
from src.classes.gaussian import Wavelet, Reflectivity, SeismicData
from src.classes.inverse_gamma import WaveletVariance, ReflectivityVariance, InverseSNR

from src.utils.model_utils import recover_1d_lattice, create_W, compute_psi
from src.utils.efficient_algebra_utils import multiply_matrix_vector_circ

###################### ###################### ###################### 
######################   ConvolutionalModel   ######################
###################### ###################### ######################

class ConvolutionalModel():
    """To simulate a dataset d based on the convolutional forward model, i.e. 
        d = w * c + e, 
    with w, c, and e Gaussian. 

    Terminology: `w` is the blur/wavelet and `c` is the image/reflectivity.
    
    Attributes:
    -----------
    lattice: The mandatory Lattice object passed when instancing.
    theta: Dictionary containing the simulated values sampled from the Wavelet, 
        Reflectivity and SeismicData classes.
    model: Dictionary containing the Wavelet, Reflectivity and SeismicData objects
        themselves.
    """
    
    def __init__(self, lattice):
        self.lattice = lattice
        self.model = {}
        
        # Default value for the hyperparameters
        self.theta = {
            'sigma2c': 1,
            'sigma2w': 1,
            'zeta': 0.025
        }

        self.aux = {}

    def setup_wavelet_variance(self, alpha=None, beta=None):
        """Setup the wavelet variance by creating a WaveletVariance object.
        
        Params:
        -------
        alpha: Shape parameter for the InverseGamma distribution.
        beta: Scale parameter for the InverseGamma distribution.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self:
                model['sigma2w']: WaveletVariance object that is stored as attribute of the ConvolutionalModel.
        """
        # Create the WaveletVariance object
        self.model['sigma2w'] = WaveletVariance(alpha, beta)

    def initialize_wavelet_variance(self, init=None):
        """Initialize the wavelet variance by sampling from the WaveletVariance object.
        
        Params:
        -------
        sigma2w: If provided, use this value instead of sampling.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self.theta:
                sigma2w: Sampled wavelet variance.
        """
        # Extract the WaveletVariance object
        _sigma2w = self.model['sigma2w']

        # Sample or set the wavelet variance
        if init is None:
            self.theta['sigma2w'] = _sigma2w.sample()
        else:
            self.theta['sigma2w'] = init

    def setup_reflectivity_variance(self, alpha=None, beta=None):
        """Setup the reflectivity variance by creating a ReflectivityVariance object.
        
        Params:
        -------
        alpha: Shape parameter for the InverseGamma distribution.
        beta: Scale parameter for the InverseGamma distribution.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self:
                model['sigma2c']: ReflectivityVariance object that is stored as attribute of the ConvolutionalModel.
        """
        # Create the ReflectivityVariance object
        self.model['sigma2c'] = ReflectivityVariance(alpha, beta)

    def initialize_reflectivity_variance(self, init=None):
        """Initialize the reflectivity variance by sampling from the ReflectivityVariance object.
        
        Params:
        -------
        sigma2c: If provided, use this value instead of sampling.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self.theta:
                sigma2c: Sampled reflectivity variance.
        """
        # Extract the ReflectivityVariance object
        _sigma2c = self.model['sigma2c']

        # Sample or set the reflectivity variance
        if init is None:
            self.theta['sigma2c'] = _sigma2c.sample()
        else:
            self.theta['sigma2c'] = init

    def setup_inverse_snr(self, alpha=None, beta=None):
        """Setup the inverse SNR by creating an InverseSNR object.
        
        Params:
        -------
        alpha: Shape parameter for the InverseGamma distribution.
        beta: Scale parameter for the InverseGamma distribution.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self:
                model['zeta']: InverseSNR object that is stored as attribute of the ConvolutionalModel.
        """
        # Create the InverseSNR object
        self.model['zeta'] = InverseSNR(alpha, beta)

    def initialize_inverse_snr(self, init=None):
        """Initialize the inverse SNR by sampling from the InverseSNR object.
        
        Params:
        -------
        zeta: If provided, use this value instead of sampling.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self.theta:
                zeta: Sampled inverse SNR.
        """
        # Extract the InverseSNR object
        _zeta = self.model['zeta']

        # Sample or set the inverse SNR
        if init is None:
            self.theta['zeta'] = _zeta.sample()
        else:
            self.theta['zeta'] = init

    def setup_wavelet_prior(self, mean=None, rho=5, p=1.98, constr=False, verbose=False):
        """Setup the wavelet prior by creating a Wavelet object and sampling from it.
        
        Params:
        -------
        mean: Mean of the wavelet.
        rho: Correlation length in the vertical direction.
        p: Power parameter for the exponential covariance.
        constr: If True, endpoints are zero.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self:
                model['w']: Wavelet object that is stored as attribute of the ConvolutionalModel object.
        """
        # Create the 1D vertical lattice 
        lattice_1d = recover_1d_lattice(self.lattice) # topology logic is inside this function
        
        # Create and store the wavelet marginal variance
        sigma2w = self.theta['sigma2w'] # load from the dict with true values for the parameteres
            
        # Create the wavelet covariance matrix
        print(f"setup_wavelet_prior: Create the wavelet correlation matrix:") if verbose else None
        Sigma = Covariance(lattice=lattice_1d, sigma2=sigma2w) # Create a k x k or nv x nv covariance matrix, depends on topology
        Sigma.create_exp_cor_matrix(rho, p, coord='v', explicit=True)
        Sigma.create_exp_cor_matrix(rho, p, coord='h', explicit=True) # this is just =1

        # Create and store the Wavelet object 
        w = Wavelet(mean, Sigma, constr, nh=self.lattice.nh, verbose=verbose)
        self.model['w'] = w

    def initialize_wavelet(self, init=None, init_unc=None, verbose=False):
        """Sample a wavelet from a Wavelet object.
        
        Params:
        -------
        y_unc: An unconstrained, fixed vector, length-k (Euclidean) or length-nv (cyclic).
        """
        # Extract the Wavelet object
        _w = self.model['w']

        # If no wavelet is passed
        if init is None:
            sample = _w.sample() # sample
            self.aux['w'] = sample
            if _w.wavelet_constraints['constr'] or (self.lattice.topology == 'C' and _w.wavelet_constraints['nr_constraints'] > 0):
                sample = _w.correct_sample(sample)
            self.theta['w_star'] = sample
        
        # When an initial wavelet is passed, if available, store both the unc and constr wavelets,
        else:
            self.aux['w'] = copy.deepcopy(init_unc) if init_unc is not None else copy.deepcopy(init)
            self.theta['w_star'] = copy.deepcopy(init)

    def setup_reflectivity_prior(self, mean=None, rho_v=None, rho_h=None, p=1, 
                                 b=None, set_c=False, verbose=False): 
        """Setup the reflectivity prior by creating a Reflectivity object.
        Params:
        -------
        mean: Mean of the reflectivity.
        rho_v: Correlation length in the vertical direction.
        rho_h: Correlation length in the horizontal direction.
        p: Power parameter for the exponential covariance.
        b: Constraints vector, if any.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self:
                model['c']: Reflectivity object that is stored as attribute of the ConvolutionalModel.
        """
        sigma2c = self.theta['sigma2c'] # load from the dict with true values for the parameteres
                
        # Create the covariance matrix
        print(f"setup_reflectivity_prior: Create the reflectivity correlation matrix:") if verbose else None
        Sigma = Covariance(lattice=self.lattice, sigma2=sigma2c)
        Sigma.create_exp_cor_matrix(rho=rho_v, p=p, coord='v', explicit=False)
        Sigma.create_exp_cor_matrix(rho=rho_h, p=p, coord='h', explicit=False)
        # Create and store the Reflectivity object 
        c = Reflectivity(mean=mean, Sigma=Sigma, b=b, verbose=verbose)
        self.model['c'] = c

    def initialize_reflectivity(self, init=None, verbose=False):
        """Sample a reflectivity from a Reflectivity object.
        
        Params:
        -------
        init: A manually specified reflectivity vector obeying the constraints.

        Return:
        ------
            Stores in self.theta:
                c_star: Constrained reflectivity sample w* ~ p(w*|Aw=b).
            
            (if unconstrained) Stores in self.aux:
                c: Unconstrained reflectivity sample c ~ p(c).
        """
        # Extract the Reflectivity object
        _c = self.model['c']

        # Create or load a reflectivity sample from the Reflectivity object
        if init is None:
            sample = _c.sample() # sample
            self.aux['c'] = sample
            if _c.reflectivity_constraints['constr']:
                sample = _c.correct_sample(sample)
        else:
            sample = copy.deepcopy(init)
            self.aux['c'] = sample
        
        # Store the sample
        self.theta['c_star'] = sample

    def setup_seismic_model(self, rho_v=None, rho_h=None, p=1, b=None, verbose=False):
        """Setup the seismic model by creating a SeismicData object.
        
        Params:
        -------
        rho_v: Correlation length in the vertical direction.
        rho_h: Correlation length in the horizontal direction.
        p: Power parameter for the exponential covariance.
        b: An observed seismic data sample. If None, it is simulated.
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self:
                model['d']: SeismicData object that is stored as attribute of the ConvolutionalModel.
        """

        # Create the mean vector as the convolution between w_star and c_star stored in self.theta
        W0, W, base_W0, base_W, base_W0T, base_WT = create_W(self.lattice, self.theta['w_star'])
        if self.lattice.topology == 'E':
            try:
                mean = W.dot(self.theta['c_star'])
            except Exception as e:
                print(f'In Euclidean topology, trying to create the mean of the seismic data, \n{e}')
        if self.lattice.topology == 'C':
            try:
                print(f"Multiplying in cyclic topology with efficient algebra. W shape: {base_W.shape}, c_star shape: {self.theta['c_star'].shape}")
                mean = multiply_matrix_vector_circ(base_W, self.theta['c_star'])
            except Exception as e:
                raise Exception(f'In cyclic topology, trying to create the mean of the seismic data, \n{e}')

        # Create and store the observational noise marginal variance 
        # sigma2d_emp = np.var(mean) * zeta
        psi = float(compute_psi(self))
        self.aux['psi'] = psi
        sigma2d = psi * self.theta['sigma2c'] * self.theta['sigma2w'] * self.theta['zeta']
        self.aux['sigma2d'] = sigma2d
        if verbose:
            print(f'psi = {psi}\n')
            print(f"sigma2d = psi * sigma2c * sigma2w * zeta = {sigma2d}\n")

        # Create the observational covariance matrix
        print(f"setup_seismic_model: Create the reflectivity correlation matrix:") if verbose else None
        Sigma = Covariance(lattice=self.lattice, sigma2=sigma2d)
        print(f"Creating exp. cor. matrix for seismic data:") if verbose else None
        Sigma.create_exp_cor_matrix(rho=rho_v, p=p, coord='v', explicit=False)
        Sigma.create_exp_cor_matrix(rho=rho_h, p=p, coord='h', explicit=False)

        if b is None:
            d = SeismicData(mean=mean, Sigma=Sigma)
            sample_d = d.sample()

            if self.lattice.n > self.lattice.n_ava:
                print('The lattice is extended, and b was not passed. Creating a b.') 
                b_extended_2d = sample_d.reshape((self.lattice.nv, self.lattice.nh), order='F')
                b_2d = d.data_constraints['U_matrix'].dot(b_extended_2d @ (d.data_constraints['X_matrix'].T))
                b = b_2d.reshape((self.lattice.n_ava, 1), order='F')
            
            if self.lattice.n == self.lattice.n_ava:
                print('The lattice is not extended, and b was not passed. Creating a b.') 
                b = sample_d

        # Create and store the Seismic data object
        print(f"Create and store the Seismic data object") if verbose else None
        d = SeismicData(mean=mean, Sigma=Sigma, b=b)
        self.model['d'] = d

    def initialize_seismic(self, init=None, verbose=False):
        """Initialize the seismic data by creating a sample from the SeismicData object.
        
        Params:
        -------
        verbose: If True, print additional information.

        Return:
        ------
            Stores in self.theta:
                d_star: Sampled seismic data d* ~ p(d*|w*, c*).
            
            (if unconstrained) Stores in self.aux:
                d: Unconstrained seismic data sample d ~ p(d|w*, c*).
        """
        
        # Extract the SeismicData object
        _d = self.model['d']
        
        # If the lattice is extended, create the auxiliary data, with the observed data in the center
        if init is None:
            if self.lattice.n > self.lattice.n_ava:
                
                # Create unconstrained and constrained samples from the SeismicData object
                d = _d.sample()
                d_star = d
                if _d.data_constraints['constr']:
                    if verbose:
                        print(f"Correcting d when initializing seismic data.")
                    d_star = _d.correct_sample(d)
                    
            # If the lattice is not extended, the data is passed as b
            if self.lattice.n == self.lattice.n_ava:
                d_star = _d.data_constraints['b']
                d = d_star
        else:
            d_star = copy.deepcopy(init)
            d = copy.deepcopy(init)
            
        # Save samples
        self.theta['d_star'] = d_star
        self.aux['d'] = d
    
    # Saving the model instance
    def save_model(self, filename=None, folder=None, path=None):
        
        # Create filename with info about the lattice and the refl constraints
        if filename is None:
            filename = "{0}_nv{1}_nh{2}_mv{3}_mh{4}_k{5}".format(
                self.lattice.topology, self.lattice.nv, self.lattice.nh, 
                self.lattice.mv, self.lattice.mh, self.lattice.k)
            if hasattr(self.lattice, 'well_positions'):
                filename = filename + '_well'
            snr = np.round(1 / self.theta['zeta'], 0)
            filename = filename + '_snr' + str(snr)
            random_id = str(uuid.uuid4())
            filename = filename + '_' + random_id
            self.model['filename'] = filename + '.pkl'
        
        # Path to file
        if folder is None:
            folder = 'datasets/'
        if path is None:
            path = '/lustre01/other/6884uc/bpwave2/'
        full_filename = path + folder + self.model['filename'] 

        # Save
        # In notebooks, module reloading can create multiple class objects with the same
        # qualified name, which breaks standard pickle ("not the same object" errors).
        # Fall back to cloudpickle when that happens.
        try:
            with open(full_filename, 'wb') as file:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001
            try:
                import cloudpickle  # type: ignore
            except Exception as cloud_exc:  # pragma: no cover
                raise exc from cloud_exc

            with open(full_filename, 'wb') as file:
                cloudpickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'The Model object has been saved to {full_filename}.')

        return full_filename