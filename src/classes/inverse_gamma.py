"""Inverse-Gamma priors for variance hyperparameters.

Terminology: `w` is the blur/wavelet and `c` is the image/reflectivity.
"""

# Third-party library imports
from scipy import stats

###################### ###################### ###################### 
######################     Inverse Gamma      ######################
###################### ###################### ######################

class InverseGamma():
    """My own inverse-Gamma class for Euclidean and cyclic lattices. 
    Contains methods to evaluate the pdf and to sample.
    """
    
    def __init__(self, alpha=None, beta=None, verbose=False):
        """
        Params:
        -------
        alpha: float
            Shape.
        beta: float 
            Scale
        """

        # Save shape and scale hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.stats_object = stats.invgamma(a=alpha, scale=beta)

    def update_alpha(self, alpha):
        """Replace the shape parameter."""
        self.alpha = alpha
        
    def update_beta(self, beta):
        """Replace the scale parameter."""
        self.beta = beta
    
    def pdf(self, x, alpha=None, beta=None, verbose=False):
        """Evaluate the pdf of the inverse-Gamma distribution."""
        
        if alpha is None and beta is None:
            alpha = self.alpha
            beta = self.beta
        
        ig = stats.invgamma(a=alpha, scale=beta)
        if verbose:
            print(f"InverseGamma pdf (stats ig): {ig.pdf(x)}")
        return ig.pdf(x)
    
    def logpdf(self, x, alpha=None, beta=None, verbose=False):
        """Evaluate the logpdf of the inverse-Gamma distribution."""
        
        if alpha is None and beta is None:
            alpha = self.alpha
            beta = self.beta
        
        ig = stats.invgamma(a=alpha, scale=beta)
        logpdf = ig.logpdf(x)
        if verbose:
            print(f"InverseGamma logpdf (stats ig): {logpdf}")
        return logpdf

    def sample(self, alpha=None, beta=None, verbose=False):
        """Draw 1 random sample from an inverse-Gamma."""

        # Use manually passed hyperparameters if provided
        if alpha is None and beta is None:
            alpha = self.alpha
            beta = self.beta
        
        y = stats.invgamma(a=alpha, scale=beta).rvs(1).item()

        # Print
        if verbose:
            print(f"InverseGamma sample (stats): {y}")
        return y

###################### ###################### ###################### 
######################  Reflectivity variance ######################
###################### ###################### ###################### 

class ReflectivityVariance(InverseGamma):
    """Inverse-Gamma prior for image/reflectivity variance. 
    The class includes methods to compute the conditional parameters 
    of the reflectivity variance full conditional with the DFT."""
    
    def __init__(self, alpha=None, beta=None):
        """
        Params:
        -------
        alpha: float
            Shape parameter.
        beta: float 
            Scale parameter.
        verbose: bool
            If True, print additional information.
        """
        # Initialize the InverseGamma class    
        super().__init__(alpha, beta)  
    
    def recompute_alpha_beta(self, sampler, i):
        """Recompute the hyperparameters alpha and beta in the 
        inverse-gamma reflectivity full conditionals using the 
        current state stored in the sampler object. 

        Params:
        -------
        sampler: Sampler object
            The sampler containing the current state and model.
        i: int
            The current iteration index.
        """
        verbose = sampler.mcmc_config.get('verbose', False)
        print("Recomputing alpha and beta for ReflectivityVariance...") if verbose else None

        # Extract the Gaussian objects
        n = sampler.lattice.n
        _c = sampler.model.model['c']
        constr_SSD = sampler.mcmc_config.get('constr_SSD', False)
        nr_constr_d = sampler.model.model['d'].data_constraints['nr_constraints']
        constr_SSC = sampler.mcmc_config.get('constr_SSC', False)
        nr_constr_c = _c.reflectivity_constraints['nr_constraints']
    
        # Extract current states
        sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
        zeta = sampler.theta['zeta'][:, i + 1].item()
        # psi = sampler.aux['psi'][:, i + 1].item() # is this in theta?
        psi = sampler.model.aux['psi']

        # Likelihood sums of squares
        if constr_SSD:
            # Extract the constrained likelihood sums of squares -> 9.08.2025 decided this was never correct
            likelihood_ss = sampler.aux['likelihood_ss_constrained'][:, i + 1].item()
            data_dof = n - nr_constr_d

        else:
            # Extract the unconstrained likelihood sums of squares
            likelihood_ss = sampler.aux['likelihood_ss_unconstrained'][:, i + 1].item()
            data_dof = n

        # Reflectivity sums of squares
        if 'c_star' in sampler.estimate:
            
            if constr_SSC:
                # Constrained reflectivity sums of squares -> 9.08.2025 decided this was never correct
                reflectivity_ss = sampler.aux['reflectivity_ss_constrained'][:, i + 1].item()
                reflectivity_dof = n - nr_constr_c
            else:
                # Unconstrained reflectivity sums of squares
                reflectivity_ss = sampler.aux['reflectivity_ss_unconstrained'][:, i + 1].item() 
                reflectivity_dof = n

        else:
            # Reflectivity is fixed
            reflectivity_ss = 0
            reflectivity_dof = 0
        
        # Conditional hyperparameters
        alpha = 0.5 * (data_dof + reflectivity_dof)
        beta = 0.5 * (likelihood_ss / (psi * sigma2w * zeta) + reflectivity_ss)
        alpha_post = sampler.model.model['sigma2c'].alpha + alpha
        beta_post = sampler.model.model['sigma2c'].beta + beta
        if verbose:
            print(f"likelihood_ss (constr?{constr_SSD}): {likelihood_ss}")
            print(f"reflectivity_ss (constr?{constr_SSC}): {reflectivity_ss}")
            print(f"n= {n}, nr_constr_c={nr_constr_c}, nr_constr_d={nr_constr_d}")
            print(f"data_dof: {data_dof}, reflectivity_dof: {reflectivity_dof}")
            print(f"Original alpha: {self.alpha}, alpha stored in model: {sampler.model.model['sigma2c'].alpha}")
            print(f"Updated alpha: {alpha_post}")
            print(f"psi: {psi}, sigma2w: {sigma2w}, zeta: {zeta}, psi* sigma2w * zeta: {psi * sigma2w * zeta}")
            print(f"Original beta: {self.beta}, beta stored in model: {sampler.model.model['sigma2c'].beta}")
            print(f"Updated beta: {beta_post}\n")


        # Return updated alpha and beta 
        return alpha_post, beta_post

###################### ###################### ###################### 
######################   Wavelet variance     ######################
###################### ###################### ###################### 

class WaveletVariance(InverseGamma):
    """Inverse-Gamma prior for blur/wavelet variance. 
    The class includes methods to compute the conditional parameters 
    of the reflectivity variance full conditional with the DFT."""
    
    def __init__(self, alpha=None, beta=None):
        """
        Params:
        -------
        alpha: float
            Shape parameter.
        beta: float 
            Scale parameter.
        verbose: bool
            If True, print additional information.
        """
        # Initialize the InverseGamma class    
        super().__init__(alpha, beta) 

    def recompute_alpha_beta(self, sampler, i):
        """Recompute the hyperparameters alpha and beta in the 
        inverse-gamma reflectivity full conditionals using the 
        current state stored in the sampler object. 

        Params:
        -------
        sampler: Sampler object
            The sampler containing the current state and model.
        i: int
            The current iteration index.
        """
        verbose = sampler.mcmc_config.get('verbose', False)
        if verbose:
            print("Recomputing alpha and beta for WaveletVariance...")

        # Extract the Gaussian objects
        n = sampler.lattice.n
        nv = sampler.lattice.nv
        constr_SSD = sampler.mcmc_config.get('constr_SSD', False)
        nr_constr_d = sampler.model.model['d'].data_constraints['nr_constraints']
        constr_SSW = sampler.mcmc_config.get('constr_SSW', False)
        nr_constr_w = sampler.model.model['w'].wavelet_constraints['nr_constraints']

        # Extract current states
        sigma2c = sampler.theta['sigma2c'][:, i + 1].item()
        zeta = sampler.theta['zeta'][:, i + 1].item()
        # psi = sampler.aux['psi'][:, i + 1].item() # is this in theta?
        psi = sampler.model.aux['psi']

        # Likelihood sums of squares
        if constr_SSD:
            # Extract the constrained likelihood sums of squares -> 9.08.2025 decided this was never correct
            likelihood_ss = sampler.aux['likelihood_ss_constrained'][:, i + 1].item()
            data_dof = n - nr_constr_d

        else:
            # Extract the unconstrained likelihood sums of squares
            likelihood_ss = sampler.aux['likelihood_ss_unconstrained'][:, i + 1].item()
            data_dof = n

        # Wavelet sums of squares
        if 'w_star' in sampler.estimate:
            
            if constr_SSW:
                # Constrained wavelet sums of squares
                wavelet_ss = sampler.aux['wavelet_ss_constrained'][:, i + 1].item()
                wavelet_dof = nv - nr_constr_w
            
            else:
                # Unconstrained wavelet sums of squares
                wavelet_ss = sampler.aux['wavelet_ss_unconstrained'][:, i + 1].item() 
                wavelet_dof = nv

        else:
            # Wavelet is fixed -> what is this if/else used for?
            wavelet_ss = 0
            wavelet_dof = 0
        
        # Conditional hyperparameters
        alpha = 0.5 * (data_dof + wavelet_dof)
        beta = 0.5 * (likelihood_ss / (psi * sigma2c * zeta) + wavelet_ss)
        alpha_post = sampler.model.model['sigma2w'].alpha + alpha
        beta_post = sampler.model.model['sigma2w'].beta + beta
        if verbose:
            print(f"likelihood_ss (constr?{constr_SSD}): {likelihood_ss}")
            print(f"wavelet_ss (constr?{constr_SSW}): {wavelet_ss}")
            print(f"n= {n}, nr_constr_w={nr_constr_w}, nr_constr_d={nr_constr_d}")
            print(f"data_dof: {data_dof}, wavelet_dof: {wavelet_dof}")
            print(f"Original alpha: {self.alpha}, alpha stored in model: {sampler.model.model['sigma2w'].alpha}")
            print(f"Updated alpha: {alpha_post}")
            print(f"psi: {psi}, sigma2c: {sigma2c}, zeta: {zeta}, psi* sigma2c * zeta: {psi * sigma2c * zeta}")
            print(f"Original beta: {self.beta}, beta stored in model: {sampler.model.model['sigma2w'].beta}")
            print(f"Updated beta: {beta_post}\n")
            

        # Return updated alpha and beta 
        return alpha_post, beta_post

###################### ###################### ###################### 
######################     Inverse SNR        ######################
###################### ###################### ###################### 

class InverseSNR(InverseGamma):
    """Inverse-Gamma prior for inverse SNR.
    The class includes methods to compute the conditional parameters 
    of the reflectivity variance full conditional with the DFT."""
    
    def __init__(self, alpha=None, beta=None):
        """
        Params:
        -------
        alpha: float
            Shape parameter.
        beta: float 
            Scale parameter.
        verbose: bool
            If True, print additional information.
        """
        # Initialize the InverseGamma class    
        super().__init__(alpha, beta) 


    def recompute_alpha_beta(self, sampler, i):
        """Recompute the hyperparameters alpha and beta in the 
        inverse-gamma reflectivity full conditionals using the 
        current state stored in the sampler object. 

        Params:
        -------
        sampler: Sampler object
            The sampler containing the current state and model.
        i: int
            The current iteration index.
        constr_SSD: bool
            If True, use constrained sums of squares for data.
        """
        verbose = sampler.mcmc_config.get('verbose', False)
        constr_SSD = sampler.mcmc_config.get('constr_SSD', False)
        if verbose:
            print("Recomputing alpha and beta for InverseSNR...")

        # Extract the Gaussian objects
        n = sampler.lattice.n
        nr_constr_d = sampler.model.model['d'].data_constraints['nr_constraints']

        # Extract current states
        sigma2c = sampler.theta['sigma2c'][:, i + 1].item()
        sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
        psi = sampler.model.aux['psi']

        # Likelihood sums of squares
        if constr_SSD:

            # Extract the constrained likelihood sums of squares -> 9.08.2025 decided this was never correct
            likelihood_ss = sampler.aux['likelihood_ss_constrained'][:, i + 1].item()
            data_dof = n - nr_constr_d

        else:
            # Extract the unconstrained likelihood sums of squares
            likelihood_ss = sampler.aux['likelihood_ss_unconstrained'][:, i + 1].item()
            data_dof = n
        
        # Conditional hyperparameters
        alpha = 0.5 * data_dof
        beta = 0.5 * likelihood_ss / (psi * sigma2c * sigma2w)
        alpha_post = sampler.model.model['zeta'].alpha + alpha
        beta_post = sampler.model.model['zeta'].beta + beta
        if verbose:
            print(f"likelihood_ss (constr?{constr_SSD}): {likelihood_ss}")
            print(f"n= {n}, nr_constr_d={nr_constr_d}")
            print(f"data_dof: {data_dof}")
            print(f"Original alpha: {self.alpha}, alpha stored in model: {sampler.model.model['zeta'].alpha}")
            print(f"Updated alpha: {alpha_post}")
            print(f"psi: {psi}, sigma2c: {sigma2c}, sigma2w: {sigma2w}, psi* sigma2c * sigma2w: {psi * sigma2c * sigma2w}")
            print(f"Original beta: {self.beta}, beta stored in model: {sampler.model.model['zeta'].beta}")
            print(f"Updated beta: {beta_post}\n")
            

        # Return updated alpha and beta 
        return alpha_post, beta_post