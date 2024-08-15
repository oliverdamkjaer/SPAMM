#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u

from utils.parse_pars import parse_pars
from utils import draw_from_sample
from spamm.components.NarrowComponent import NarrowComponent
from spamm.Spectrum import Spectrum
from astropy.modeling.functional_models import Gaussian1D

PARS = parse_pars()["power_law"]
TEST_WL = parse_pars()["testing"]
WL = np.arange(TEST_WL["wl_min"], TEST_WL["wl_max"], TEST_WL["wl_step"])

#-----------------------------------------------------------------------------#

def create_ne(ne_params=None):

    wl = ne_params["wl"]
    del ne_params["wl"]
    
    print(f"ne params: {ne_params}")
    
    # Make a Spectrum object with dummy flux and flux error
    spectrum = Spectrum(wl, wl, wl)
    ne_flux = np.zeros(len(spectrum.spectral_axis))
    
    width = ne_params['width']
    for line in ne_params['lines']:
        amp = line[0]
        center = line[1]
        gaussian = Gaussian1D(amplitude=amp, mean=center, stddev=width)
        ne_flux += gaussian(spectrum.spectral_axis)
        
    ne_err = ne_flux * 0.05

    return wl, ne_flux, ne_err, ne_params

#-----------------------------------------------------------------------------#
