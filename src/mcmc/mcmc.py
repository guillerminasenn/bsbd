"""Core MCMC driver and configuration.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.
"""

# Standard library import
import os
import time
import random
import pickle
import copy

# Third-party library imports
import numpy as np

# Import algorithms
from src.mcmc.mwg.mwg import mwg

class MCMC():
    """MCMC sampler for joint updates of blur `w` and image `c`.

    Supports Gibbs and collapsed HMC updates with optional adaptation.
    """
    
    def __init__(self, model, theta_init, estimate=None, data=None, theta_true=None,
                 chunk_size=5000, model_filename=None, folder=None, path=None):
        """
        Initialize MCMC sampler.
        
        Params:
        -------
        model: ConvolutionalModel
            Object containing the model configuration (priors, likelihood, etc.)
        data: dict
            Observations and constraints
        theta_init: dict
            Dictionary with initial values for all parameters
        estimate: list, optional
            Which parameters to estimate, a subset of ['w', 'c', 'sigma2c', 'sigma2w', 'zeta', 'd'].
            If None, estimates all. The true value for those not estimated need 
            to be provided in theta_true
        theta_true: dict, optional
            True parameter values (only for simulated data assessment/plotting)
        chunk_size: 
            Iterations will be stored in files with chunk_size iterations each.
        model_filename: str, optional   
            Name of the model file to save the MCMC results.
        folder: str, optional   
            Folder where the MCMC results will be saved.
        path: str, optional
            Root path to the file where the folder with the MCMC results will be saved.
        """
        if not hasattr(model, 'lattice'):
            raise ValueError('Model must be a ConvolutionalModel object with lattice attribute.')
        
        self.model = model
        self.lattice = self.model.lattice
        
        # Store initial values (deep copy to avoid modification)
        self.theta_init = copy.deepcopy(theta_init)

        # Store true values for assessment (optional, only for simulated data)
        self.theta_true = copy.deepcopy(theta_true) if theta_true is not None else None

        # If not all paremeters are to be estimated, replace fixed params in theta_init with their true value
        all_pars = ['w_star', 'c_star', 'sigma2c', 'sigma2w', 'zeta', 'd_star']

        if estimate is None:
            estimate = all_pars

        else:
            if not isinstance(estimate, list):
                raise ValueError("Parameter 'estimate' must be a list of parameter names.")
            
            if not set(estimate).issubset(all_pars):
                raise ValueError("Parameter 'estimate' must be a subset of ['w_star', 'c_star', 'sigma2c', 'sigma2w', 'zeta', 'd_star'].")

            if 'd_star' in estimate and self.lattice.n == self.lattice.n_ava:
                raise ValueError("Cannot estimate 'd_star' when all data is known (lattice.n == lattice.n_ava).")
            
            not_estimate = set(all_pars) - set(estimate)
            if not set(not_estimate).issubset(set(theta_true.keys())):
                raise ValueError("All parameters not in 'estimate' must be present in 'theta_true'.") 

            # Replace fixed params in theta_init with their true value
            for param in not_estimate:
                print(f"{param} is fixed. Replacing its init value with the true value in theta_true.")
                self.theta_init[param] = self.theta_true.get(param)
                
        self.estimate = estimate

        # Store file saving configuration
        self.file_config = {
            'chunk_size': chunk_size,
            'model_filename': model_filename,
            'folder': folder,
            'path': path,  # Root path to file
        }

        
    def set_initial_values(self, **kwargs):
        """
        Override initial values for specific parameters.
        
        Params:
        -------
        **kwargs: Parameter name and value pairs to override
        
        Example:
        --------
        sampler.set_initial_values(w=custom_w_values, c=custom_c_values)
        """
        for param, value in kwargs.items():
            if param in self.theta_init:
                self.theta_init[param] = copy.deepcopy(value)
                # Also update the _star versions if they exist
                star_param = f"{param}_star"
                if star_param in self.theta_init:
                    self.theta_init[star_param] = copy.deepcopy(value)
            else:
                raise ValueError(f"Parameter '{param}' not found in initial values")

    def _initialize_sample_aux_dicts(self, i=0, verbose=False):
        """Initialize dictionaries to store samples and auxiliary quantities.
        Params:
        -------
        i: int
            Current global iteration index (default: 0). Will initialize
            with values in theta_init. Else, will initialize with the last 
            sampled state.
        N: int, optional
            Total number of iterations (default: self.mcmc_config['N']).
        """
        
        if i == 0:
            
            # Initialize the samples dictionaries with the initial values
            for param in self.estimate:
                try:
                    self.theta[param][:, 0] = self.theta_init.get(param)
                except:
                    self.theta[param][:, 0] = self.theta_init.get(param).reshape(-1)

            # Fill the whole dictionary with the true value if the param is not estimated
            for param in set(self.theta.keys()) - set(self.estimate):
                try:
                    self.theta[param][:, ] = self.theta_true.get(param)
                except:
                    self.theta[param][:, ] = self.theta_true.get(param).reshape(-1)


            # Initialize sigma2d
            psi = self.model.aux['psi']
            self.aux['sigma2d'][:, 0] = psi * self.theta['sigma2c'][:, 0] * self.theta['sigma2w'][:, 0] * self.theta['zeta'][:, 0] 
            
            # Initialize the unconstrained d, c, w
            self.aux['d'][:, 0] = self.model.aux['d'].reshape(-1)  
            self.aux['c'][:, 0] = self.model.aux['c'].reshape(-1)  
            self.aux['w'][:, 0] = self.model.aux['w'].reshape(-1) 

        elif i > 0:
            
            # Initialize the samples dictionaries with the last state
            verbose = self.mcmc_config['verbose']
            print(f"Initializing the samples dictionary at iteration {i}.")                           
            for param in self.estimate:
                print(f"param={param}") if verbose else None
                try:
                    print(f"current param value: {self.theta_temp.get(param)}") if verbose else None
                    self.theta[param][:, 0] = self.theta_temp.get(param)
                except:
                    self.theta[param][:, 0] = self.theta_temp.get(param).reshape(-1)

            # Fill the whole dictionary with the true value if the param is not estimated
            for param in set(self.theta.keys()) - set(self.estimate):
                try:
                    self.theta[param][:, ] = self.theta_true.get(param)
                except:
                    self.theta[param][:, ] = self.theta_true.get(param).reshape(-1)

            # Initialize sigma2d
            psi = self.model.aux['psi']
            self.aux['sigma2d'][:, 0] = psi * self.theta['sigma2c'][:, 0] * self.theta['sigma2w'][:, 0] * self.theta['zeta'][:, 0] 
            
            # Initialize the unconstrained d, c, w; This is wrong, because should be init if param is estimated.
            self.aux['d'][:, 0] = self.aux_temp['d'].reshape(-1)  
            self.aux['c'][:, 0] = self.aux_temp['c'].reshape(-1)   
            self.aux['w'][:, 0] = self.aux_temp['w'].reshape(-1)    
        
    def _create_sample_aux_dicts(self, N=None, verbose=False):
        """Create dictionaries to store samples and auxiliary quantities
        that change in each iteration after each MCMC update."""

        if N is None: N = self.mcmc_config['N']

        # Create dictionary to store the samples
        nv = self.lattice.nv
        n = self.lattice.n
        k = self.lattice.k
        theta = {
            'sigma2c': np.zeros((1, N + 1)),
            'sigma2w': np.zeros((1, N + 1)),
            'zeta': np.zeros((1, N + 1)),
            'w_star': np.zeros((nv, N + 1)),
            'c_star': np.zeros((self.lattice.n, N + 1)),
            'd_star': np.zeros((self.lattice.n, N + 1))
        }
        
        # Create dictionary to store auxiliary quantities that are re-computed at each iteration
        aux = {
            # 'psi': np.zeros((1, N + 1)), 
            'sigma2d': np.zeros((1, N + 1)),
            'd': np.zeros((n, N + 1)),
            'c': np.zeros((n, N + 1)),
            'w': np.zeros((nv, N + 1)),
            'likelihood_ss_unconstrained': np.zeros((1, N + 1)),
            'likelihood_ss_subset': np.zeros((1, N + 1)),
            'likelihood_ss_constrained': np.zeros((1, N + 1)),
            'reflectivity_ss_unconstrained': np.zeros((1, N + 1)),
            'reflectivity_ss_subset': np.zeros((1, N + 1)),
            'reflectivity_ss_constrained': np.zeros((1, N + 1)),
            'wavelet_ss_unconstrained': np.zeros((1, N + 1)),
            'wavelet_ss_subset': np.zeros((1, N + 1)),
            'wavelet_ss_constrained': np.zeros((1, N + 1))
        }

        # Initialize psi; # fixed for all iterations when the correlation ranges in w, c are fixed
        # self.aux['psi'][:, ] = self.model.aux['psi'] 

        # Save the samples and aux dictionaries as attributes 
        self.theta = theta
        self.aux = aux

    def run(
        self,
        N,
        algorithm='gibbs',
        adapt_config=None,
        verbose=False,
        save_stats=False,
        **kwargs,
    ):

        """Run the Metropolis-within-Gibbs (MwG) with the update for (w, c) as 
        specified by the algorithm parameter.

        Params:
        -------
        N: int
            Nr. of iterations.
        algorithm: str (default='gibbs')
            Algorithm to update (w, c), one of ['gibbs', 'hmc', 'collapsed_hmc'].
        adapt_config: dict
            Optional dictionary with adaptation configuration for the (w, c) update.
        verbose: bool
        **kwargs: 
            Pass algorithm-specific arguments.
        """
    
        # Check that algorithm is implemented    
        if algorithm not in ['gibbs', 'hmc', 'collapsed_hmc']:
            raise Exception(f'Algorithm {algorithm} is not implemented.')
        print(f'Will run {N} iterations of MwG with {algorithm} for the (w, c) update.')

        # Store options as attributes of class MCMC
        self.mcmc_config = {
            'algorithm':algorithm,
            'N': N,
            'verbose': verbose,
            'save_stats': save_stats,
            'constr_SSD': False,
            'constr_SSW': True,
            'constr_SSC': False,
        }
    
        # Instance objects that we'll use to sample from priors and likelihood
        self._create_par_objects(verbose)

        # Create dictionary to store information about adaptation
        self.adapt_config = adapt_config or {}  # default: no adaptation

        # Create dictionary to track adaptation
        self.adapt_tracking = {
            method: {
                param: {
                    'history': np.zeros(N + 1),
                    'batch_ar': []
                } for param in param_dict
            } for method, param_dict in adapt_config.items()
        }

        # Create dictionary to store algorithm statistics
        self.stats = {
            'mwg': {
                'init_time': time.time(),  # initial time
                'exec_time': 0, # total execution time
                'error': np.zeros(N + 1) # flag to indicate if an error occurred and to discard the iteration
                }, 
            'wc_update': {
                'exec_time': None, # execution time of the (w, c) update
                'iter_with_chmc': np.zeros(N + 1), # iteration indices
                'nr_iters': np.zeros(N + 1), # number of iterations in each (w, c) update
                'loglik': np.zeros(N + 1), 
                'log_prior_w': np.zeros(N + 1),
                'log_prior_c': np.zeros(N + 1),
                'log_posterior_dens': np.zeros(N + 1),
                'acceptance': {
                    'log_acc_prob': np.zeros(N + 1), # iteration acceptance probability
                    'acc_iter': np.zeros(N + 1), # was iteration accepted?
                    'cum_ar': np.zeros(N + 1), # cumulative acceptance rate 
                } 
            },
            'log_post': {
                'likelihood': np.zeros((1, N + 1)),
                'prior_reflectivity': np.zeros((1, N + 1)),
                'prior_wavelet': np.zeros((1, N + 1)),
                'prior_sigma2c': np.zeros((1, N + 1)),
                'prior_sigma2w': np.zeros((1, N + 1)),
                'prior_zeta': np.zeros((1, N + 1)),
                'joint': np.zeros((1, N + 1))
            }
        }

        # Save the MCMC object at the beginning of the run
        filenames = self.save_sampler_object()

        # Finally, call the mcmc algorithm
        mwg(self, **kwargs)

        # # Save the MCMC object at the end of the run
        # filenames = self.save(save_sampler=True)
        return filenames

    def _create_par_objects(self, verbose=False):
        """Helper method to instance prior wavelet and reflectivity objects for sampling
        and seismic data objects for computing the likelihood of the current and proposed 
        values."""
        
        _w = copy.deepcopy(self.model.model['w'])
        _c = copy.deepcopy(self.model.model['c'])
        _d = copy.deepcopy(self.model.model['d'])

        _sigma2w = copy.deepcopy(self.model.model['sigma2w'])
        _sigma2c = copy.deepcopy(self.model.model['sigma2c'])
        _zeta = copy.deepcopy(self.model.model['zeta'])

        if verbose:
            psi = self.model.aux['psi']
            zeta = _d.Sigma.sigma2 / (psi * _c.Sigma.sigma2 * _w.Sigma.sigma2)
            print(f"When creating par_objs, they are a copy of the model passed (probably the true model), and this model has:\n sigma2w = {_w.Sigma.sigma2}, sigma2c = {_c.Sigma.sigma2}, zeta = {zeta}, sigma2d = {_d.Sigma.sigma2}, psi={psi}")
        self.par_objs = {'w': _w, 'c': _c, 'd': _d, 'sigma2w': _sigma2w, 'sigma2c': _sigma2c, 'zeta': _zeta}

    def save_sampler_object(self):
        """Create and store in the sampler object the path, folder, and model_filename.
        Then save the MCMC sampler object to a file.

        Returns:
        -------
        path: str
            Root path to the file where the folder with the MCMC results will be saved.
        folder: str
            Folder where the MCMC results will be saved.
        model_filename: str
            Name of the model file to save the MCMC results.
        """

        # Extract 
        model_filename = self.file_config.get('model_filename', None)
        folder = self.file_config.get('folder', None)
        path = self.file_config.get('path', None)

        # Extract algorithm name
        algorithm = self.mcmc_config['algorithm']

        # Dataset name, algorithm name, and iterations
        if model_filename is None:
            
            # Extract the name of the object containing the model, without the extension
            model_filename = self.model.model['filename'].replace('.pkl', '')
            print(f"model filename as stored in mcmc obj: {model_filename}")

            # Build the folder name using the model_filename, algorithm, and nr of iterations
            N_str = str(self.mcmc_config['N'])
            rnd_nr = str(random.randrange(1, 10000))
            # model_filename_runid = model_filename + '_runID_' + rnd_nr
            model_filename = model_filename + '_' + algorithm + '_N' + N_str + '_runID_' + rnd_nr
            self.file_config['model_filename'] = model_filename

        # Folder
        if folder is None:

            # Algorithm version
            version = self.mcmc_config.get('version')
            
            # Constraints type
            constraints_w = 'unc_w'
            constraints_c = 'unc_c'
            if self.model.model['c'].reflectivity_constraints['constr']:
                constraints_c = 'constr_c'
            if self.model.model['w'].wavelet_constraints['constr']:
                constraints_w = 'constr_w'
            constraints_folder = constraints_w + '_' + constraints_c

            # Create folder name

            # extract run ID from model_filename
            # runID = model_filename.split('_runID_')[-1] if '_runID_' in model_filename else None
            # rnd_nr = str(random.randrange(1, 10000))
            # model_filename_runid = model_filename + '_runID_' + rnd_nr
            if version is not None:
                folder = (algorithm + '/v' + str(version) + '/' + constraints_folder 
                          + '/' + model_filename + '/' ) # if algm has versions
            else:
                folder = algorithm + '/' + constraints_folder + '/' + model_filename + '/'
            self.file_config['folder'] = folder

        # Root path to file
        if path is None:
            path = '/lustre01/other/6884uc/bpwave2/'
            self.file_config['path'] = path
        absolute_path = path + folder 

        # Create filename
        absolute_filename_sampler_object = absolute_path + model_filename + '.pkl'
        ensure_dir_exists(absolute_filename_sampler_object)

        # Create a dictionary to save the interesting qunatities in the sampler object 
        sampler_save = {
            'model': self.model,
            'lattice': self.lattice,
            'mcmc_config': self.mcmc_config,
            'adapt_config': self.adapt_config,
            'file_config': self.file_config,
            'estimate': self.estimate,
            'theta_init': self.theta_init,
            'theta_true': self.theta_true
        }

        # Save sampler dict. In notebooks, hot-reloads can create duplicate class
        # identities, which makes stdlib pickle fail. Fall back to cloudpickle.
        try:
            with open(absolute_filename_sampler_object, 'wb') as file:
                pickle.dump(sampler_save, file, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            try:
                import cloudpickle

                with open(absolute_filename_sampler_object, 'wb') as file:
                    cloudpickle.dump(sampler_save, file)
            except Exception:
                raise e

        print(f'The Sampler dict has been saved to {absolute_filename_sampler_object}.')

        return path, folder, model_filename

    def save_stats(self):
        """Save the current stats of the MCMC run to a file.
        
        Returns:
        -------
        """

        # Create filename
        model_filename = self.file_config.get('model_filename', None)
        folder = self.file_config.get('folder', None)
        path = self.file_config.get('path', None)
        absolute_filename_stats_object = path + folder + model_filename + '_stats.pkl'
        ensure_dir_exists(absolute_filename_stats_object)
        
        # Create stats dictionary to save
        stats_save = {
            'stats': self.stats,
            'adapt_tracking': self.adapt_tracking
        }

        # Save stats dict
        with open(absolute_filename_stats_object, 'wb') as file:
            pickle.dump(stats_save, file, pickle.HIGHEST_PROTOCOL)
        print(f'The stats dict has been saved to {absolute_filename_stats_object}.')

    def save_samples(self, chunk_index):
        """Save the current chunk of samples.
        
        Params:
        -------
        chunk_index: int
            Index of the chunk to save."""

        # Build the chunk filename
        model_filename = self.file_config.get('model_filename', None)
        folder = self.file_config.get('folder', None)
        path = self.file_config.get('path', None)
        chunk_filename = model_filename + '_chunk_' + str(chunk_index) + '.pkl'
        absolute_filename_chunk = path + folder + 'samples/' + chunk_filename
        ensure_dir_exists(absolute_filename_chunk)

        # Modify theta to save only the first chunk_size columns
        theta_to_save = {k: v[:, :self.file_config['chunk_size']] for k, v in self.theta.items()}

        # Add aux['sigma2d'] to theta_to_save
        theta_to_save['sigma2d'] = self.aux['sigma2d'][:, :self.file_config['chunk_size']]

        # Save the chunk to file
        with open(absolute_filename_chunk, 'wb') as file:
            pickle.dump(theta_to_save, file, pickle.HIGHEST_PROTOCOL)
        print(f'Chunk {chunk_index} has been saved to {absolute_filename_chunk}.')
        
        return path, folder, model_filename

def ensure_dir_exists(filepath):
    """Create directory if it doesn't exist."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

