#!/usr/bin/python

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from scipy.interpolate import interp1d
import emcee
from multiprocessing import Pool
from functools import partial

from .Spectrum import Spectrum
import time

# TODO arg pos is not even used vv
def sort_on_runtime(pos):
    """
    Description.

    Args:
        pos (): ?? 

    Returns:
        ndarray (ndarray): ?
        ndarray (ndarray): ? 
    """

    p = np.atleast_2d(p)
    idx = np.argsort(p[:, 0])[::-1]
    
    return p[idx], idx

#-----------------------------------------------------------------------------#

def prior(params, components):
    
    """
    Calculate the sum of the natural logarithm of the priors (ln_priors) for all components in the model.

    Args:
        params (np.array): An array of parameter values for the model.

    Returns:
        total_ln_prior (float): The sum of the ln_priors for all components in the model.
    """

    total_ln_prior = 0.
    current_index = 0

    for component in components:
        # Calculate the end index for the current component's parameters.
        end_index = current_index + component.parameter_count

        # Sum all ln_priors for the current component's parameters and add to the total.
        total_ln_prior += sum(component.ln_priors(params=params[current_index:end_index]))

        current_index = end_index

    return total_ln_prior

#-----------------------------------------------------------------------------#

def likelihood(data_spectrum, model_spectrum_flux):
    """
    Calculate the natural logarithm of the likelihood (ln(L)) of the given model spectrum.

    The likelihood is calculated based on a Gaussian distribution, with the model 
    spectrum interpolated over the data wavelength grid. The formula used is:

    ln(L) = -0.5 * sum( (flux_observed - flux_model)^2 / sigma^2 + ln(2 * pi * sigma^2) )

    where:
    - flux_observed is the observed data flux
    - flux_model is the model flux interpolated over the data wavelength grid
    - sigma is the observed data flux error

    Args:
        model_spectrum_flux (np.array): The model spectrum flux values, represented as a numpy array.

    Returns:
        float: The sum of the natural logarithm of the likelihood values. This represents the total
        likelihood of the model given the observed data. If any likelihood values are NaN, they are 
        replaced with zero before summing.

    Note:
        This method assumes that the model flux changes linearly between the points in the model spectrum.
    """

    # Create an interpolation function. #TODO WARNING: SHOULDNT THIS BE MODEL_SPECTRUM INSTEAD OF DATA_SPECTRUM?
    interp_func = interp1d(data_spectrum.spectral_axis, model_spectrum_flux)

    # Interpolate the model flux over the data spectral axis.
    model_flux_interp = interp_func(data_spectrum.spectral_axis)

    # Calculate the ln(likelihood).
    first_term = np.power((data_spectrum.flux - model_flux_interp) / data_spectrum.flux_error, 2)
    second_term = np.log(2 * np.pi * np.power(data_spectrum.flux_error, 2))
    ln_likelihood = -0.5 * np.sum(first_term + second_term)

    # Replace any NaN values with zero.
    ln_likelihood = np.nan_to_num(ln_likelihood)
    
    return ln_likelihood

#-----------------------------------------------------------------------------#

def ln_posterior(args, params):

    data_spectrum, components = args[0], args[1]

    ln_prior = prior(params=params, components=components)
    if not np.isfinite(ln_prior):
        return -np.inf
    
    model_spectrum_flux = model_flux(params=params, data_spectrum=data_spectrum, components=components)
    ln_likelihood = likelihood(data_spectrum=data_spectrum, model_spectrum_flux=model_spectrum_flux)

    return ln_prior + ln_likelihood

def model_flux(params, data_spectrum, components):
    """
    Given the parameters in the model, generate a spectrum. This method is
    called by multiple MCMC walkers at the same time, so edit with caution.

    Args:
        params (ndarray): 1D numpy array of all parameters of all 
            components of the model.

    Returns:
        self.model_spectrum.flux (ndarray): Array of flux values?
    """

    # Make a copy since we'll delete elements.
    # Note: np.copy does a deepcopy
    params2 = np.copy(params)

    # Initialize the model spectrum flux
    model_spectrum_flux = np.zeros(len(data_spectrum.spectral_axis))

    for component in components:
        # Extract parameters from full array for each component.
        p = params2[0:component.parameter_count]

        # Add the flux of each component to the model spectrum, 
        # except for extinction
        if component.name != "Extinction":
            component_flux = component.flux(spectrum=data_spectrum, parameters=p)
            model_spectrum_flux += component_flux
        else:
            extinction = component.extinction(spectrum=data_spectrum, parameters=p)
            extinct_spectra = np.array(model_spectrum_flux)*extinction
            model_spectrum_flux = extinct_spectra

        # Remove the parameters for this component from the list
        params2 = params2[component.parameter_count:]

    return model_spectrum_flux

#-----------------------------------------------------------------------------#

#TODO do we need all these commented attributes??
class Model(object):
    """
    The Model class holds the data spectrum and a list of components that make up the model. 
    It provides methods for setting the data spectrum, adding components, and running 
    an MCMC process to fit the model to the data.

    Attributes:
        _mask (np.array): Mask for the data spectrum.
        _data_spectrum (Spectrum object): Observed data spectrum.
        z (float): Redshift of the model.
        components (list): List of Component objects in the model.
        parallel (bool): Flag for using local parallel processing on one machine
        sampler (emcee.EnsembleSampler object): MCMC sampler.
        model_spectrum (Spectrum object): Generated model spectrum.
        downsample_data_if_needed (bool): Flag for downsampling data spectrum.
        upsample_components_if_needed (bool): Flag for upsampling components.
    """
    
    def __init__(self, wavelength_start=1000, wavelength_end=10000, 
                 wavelength_delta=0.05, parallel=True):
        """
        Args:
            wavelength_start (): 
            wavelength_end (): 
            wavelength_delta (float):
            parallel (Bool):
        """

        self._mask = None
        self._data_spectrum = None
        
        self.z = None
        self.components = []
        self.parallel = parallel

        self.sampler = None
        #self.sampler_output = None

        wl_init = np.arange(wavelength_start, wavelength_end, wavelength_delta)
        self.model_spectrum = Spectrum(spectral_axis = wl_init,
                                       flux = np.zeros(len(wl_init)),
                                       flux_error = np.zeros(len(wl_init)))

        # Flag to allow Model to interpolate components' wavelength grid to 
        # match data if component grid is more course than data.
        # TODO - document better!
        self.downsample_data_if_needed = False
        self.upsample_components_if_needed = False

# TODO is this needed? vvvv

#        self.reddening = None
#        self.model_parameters = {}
#        self.mcmc_param_vector = None
#    @property
#    def mask(self):
#        """
#
#        """
#        if self.data_spectrum is None:
#            print("Attempting to read the bad pixel mask before a spectrum was defined.")
#            sys.exit(1)
#        if self._mask is None:
#            self._mask = np.ones(len(self.data_spectrum.spectral_axis))
#
#        return self._mask
#
#    @mask.setter
#    def mask(self, new_mask):
#        """
#        Document me.
#
#        :params mask: A numpy array representing the mask.
#        """
#        self._mask = new_mask

#-----------------------------------------------------------------------------#

    @property
    def data_spectrum(self):
        """
        All components of the model must be set before setting the data.

        Returns:
            _data_spectrum (Spectrum object): ?
        """

        return self._data_spectrum

#-----------------------------------------------------------------------------#

    @data_spectrum.setter
    def data_spectrum(self, new_data_spectrum):
        """
        Args:
            new_data_spectrum (Spectrum object): ?
        """

        self._data_spectrum = new_data_spectrum

        if len(self.components) == 0:
            raise Exception("Components must be added before defining the data spectrum.")

        # The data spectrum defines the model wavelength grid.
        self.model_spectrum.spectral_axis = np.array(new_data_spectrum.spectral_axis)
        self.model_spectrum.flux = np.zeros(len(self.model_spectrum.spectral_axis))

        # Check that all components are on the same wavelength grid.
        # If they are not, *and* the flag to interpolate them has been set, 
        # *and* they are not more coarse than the data, interpolate. 
        # If not, fail.
        need_to_downsample_data = False
        components_to_upsample = {}

        # Keep track of the maximum grid spacing found so far
        # and the component with that grid spacing
        max_gs = 0
        worst_component = None

        for component in self.components:
            component.initialize(data_spectrum=new_data_spectrum)
            
            # Get the grid spacing of the component.
            component_gs = component.grid_spacing() 
            
            # If the component's grid spacing is not None and is greater than the current 
            # maximum, update the maximum grid spacing and the worst component
            if (component_gs is not None) and (component_gs > max_gs):
                max_gs = component_gs
                worst_component = component

        if max_gs > new_data_spectrum.grid_spacing():
            if self.downsample_data_if_needed:
                downsampled_spectrum = new_data_spectrum.copy()

                # Adjust the spectral axis of the downsampled spectrum to the maximum grid spacing
                downsampled_spectrum.spectral_axis = np.arange(new_data_spectrum[0], new_data_spectrum[-1], 
                                                               max_gs)
                
                # Interpolate the flux of the downsampled spectrum to the new spectral axis                                                 
                downsampled_spectrum.flux = interp1d(x=downsampled_spectrum.spectral_axis, y=new_data_spectrum.flux, 
                                                     kind="linear")
                
                # Update the spectral axis of the model spectrum to match the downsampled spectrum
                self.model_spectrum.spectral_axis = np.array(downsampled_spectrum.spectral_axis)

                # Reinitialize all components with the downsampled data spectrum
                for component in self.components:
                    component.initialize(data_spectrum=downsampled_spectrum)
            else:
                raise ValueError(f"""Component '{worst_component}' has coarser grid spacing than data. 
                                     Increase component spacing or use 'upsample_components_if_needed' 
                                     or 'downsample_data_if_needed' flags in Model class.""")
        

#-----------------------------------------------------------------------------#

    def run_mcmc(self, data_spectrum, components, n_walkers=100, n_iterations=100):
        """
        Run emcee MCMC.
    
        Args:
            n_walkers (int): Number of walkers to pass to the MCMC.
            n_iteratins (int): Number of iterations to pass to the MCMC. 
        """

        # Initialize walker matrix with initial parameters
        walkers_matrix = []
        for _ in range(n_walkers):
            walker_params = []
            for component in self.components:
                walker_params += component.initial_values(self.data_spectrum)
            walkers_matrix.append(walker_params)

        # Create MCMC sampler. To enable multiproccessing, set threads > 1.
        # If using multiprocessing, the "log_prob_fn" and "args" parameters 
        # must be pickleable.

        wrapped_posterior = partial(ln_posterior, (data_spectrum, components))
        
        if self.parallel:
            with Pool() as pool:
                self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=len(walkers_matrix[0]),
                                                        log_prob_fn=wrapped_posterior, pool=pool)
                self.sampler.run_mcmc(walkers_matrix, n_iterations, progress=True)
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=len(walkers_matrix[0]),
                                                    log_prob_fn=wrapped_posterior)
            self.sampler.run_mcmc(walkers_matrix, n_iterations, progress=True)

#-----------------------------------------------------------------------------#

# TODO should there be a getter without a setter? vv
    @property
    def total_parameter_count(self):
        """
        Return the total number of parameters of all components.
        
        Returns:
            total_no_parameters (int): Total number of parameters for 
                all components.
        """
        
        total_no_parameters = 0
        for c in self.components:
            total_no_parameters += c.parameter_count
        
        return total_no_parameters

#-----------------------------------------------------------------------------#

    def parameter_vector(self):
        """
        Return the names? of all component parameters..

        Returns:
            param_vector (list): List of all component parameter names?
        """

        param_vector = []
        for component in self.components:
            param_vector.append(component.parameters())

        return param_vector

#-----------------------------------------------------------------------------#


    def model_parameter_names(self):
        """
        Return a list of all component parameter names.

        Returns:
            labels (list): List of component names.
        """

        labels = []
        for c in self.components:
            labels = labels + [x for x in c.model_parameter_names]
        return labels
    

