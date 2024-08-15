import numpy as np

def _wave_convert(lam):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param lam - Wavelength in Angstroms
    :return: conversion factor

    """
    lam = np.asarray(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return fact

###############################################################################

def vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms

    """
    return lam_vac/_wave_convert(lam_vac)

###############################################################################

def air_to_vac(lam_air):
    """
    Convert air to vacuum wavelengths

    :param lam_air - Wavelength in Angstroms
    :return: lam_vac - Wavelength in Angstroms

    """
    return lam_air*_wave_convert(lam_air)

def gaussian(ln_lam_temp, line_wave, FWHM_gal, pixel=True):
    """
    Instrumental Gaussian line spread function (LSF), optionally analytically
    integrated within the pixels. When used as template for pPXF it is
    rigorously insensitive to undersampling.

    It implements equations (14) and (15) `of Westfall et al. (2019)
    <https://ui.adsabs.harvard.edu/abs/2019AJ....158..231W>`_

    The function is normalized in such a way that::
    
            line.sum(0) = 1
    
    When the LSF is not severey undersampled, and when pixel=False, the output
    of this function is nearly indistinguishable from a normalized Gaussian::
    
      x = (ln_lam_temp[:, None] - np.log(line_wave))/dx
      gauss = np.exp(-0.5*(x/xsig)**2)
      gauss /= np.sqrt(2*np.pi)*xsig

    However, to deal rigorously with the possibility of severe undersampling,
    this Gaussian is defined analytically in frequency domain and transformed
    numerically to time domain. This makes the convolution within pPXF exact
    to machine precision regardless of sigma (including sigma=0).

    :param ln_lam_temp: np.log(wavelength) in Angstrom
    :param line_wave: Vector of lines wavelength in Angstrom
    :param FWHM_gal: FWHM in Angstrom. This can be a scalar or the name of
        a function wich returns the instrumental FWHM for given wavelength.
        In this case the sigma returned by pPXF will be the intrinsic one,
        namely the one corrected for instrumental dispersion, in the same
        way as the stellar kinematics is returned.
      - To measure the *observed* dispersion, ignoring the instrumental
        dispersison, one can set FWHM_gal=0. In this case the Gaussian
        line templates reduce to Dirac delta functions. The sigma returned
        by pPXF will be the same one would measure by fitting a Gaussian
        to the observed spectrum (exept for the fact that this function
        accurately deals with pixel integration).
    :param pixel: set to True to perform analytic integration over the pixels.
    :return: LSF computed for every ln_lam_temp

    """
    line_wave = np.asarray(line_wave)

    if callable(FWHM_gal):
        FWHM_gal = FWHM_gal(line_wave)

    n = ln_lam_temp.size
    npad = 2**int(np.ceil(np.log2(n)))
    nl = npad//2 + 1  # Expected length of rfft

    dx = (ln_lam_temp[-1] - ln_lam_temp[0])/(n - 1)   # Delta\ln\lam
    x0 = (np.log(line_wave) - ln_lam_temp[0])/dx      # line center
    w = np.linspace(0, np.pi, nl)[:, None]            # Angular frequency

    # Gaussian with sigma=xsig and center=x0,
    # optionally convolved with an unitary pixel UnitBox[]
    # analytically defined in frequency domain
    # and numerically transformed to time domain
    xsig = FWHM_gal/2.355/line_wave/dx    # sigma in pixels units
    rfft = np.exp(-0.5*(w*xsig)**2 - 1j*w*x0)
    if pixel:
        rfft *= np.sinc(w/(2*np.pi))

    line = np.fft.irfft(rfft, n=npad, axis=0)

    return line[:n, :]


def emission_lines(ln_lam_temp, lam_range_gal, FWHM_gal, pixel=True,
                   tie_balmer=False, limit_doublets=False, vacuum=False):
    """
    Generates an array of Gaussian emission lines to be used as gas templates in PPXF.

    ****************************************************************************
    ADDITIONAL LINES CAN BE ADDED BY EDITING THE CODE OF THIS PROCEDURE, WHICH 
    IS MEANT AS A TEMPLATE TO BE COPIED AND MODIFIED BY THE USERS AS NEEDED.
    ****************************************************************************

    Generally, these templates represent the instrumental line spread function
    (LSF) at the set of wavelengths of each emission line. In this case, pPXF
    will return the intrinsic (i.e. astrophysical) dispersion of the gas lines.

    Alternatively, one can input FWHM_gal=0, in which case the emission lines
    are delta-functions and pPXF will return a dispersion which includes both
    the intrumental and the intrinsic disperson.

    For accuracy the Gaussians are integrated over the pixels boundaries.
    This can be changed by setting `pixel`=False.

    The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    The [OII] and [SII] doublets can be restricted to physical range of ratios.

    The Balmer Series can be fixed to the theoretically predicted decrement.

    Input Parameters
    ----------------

    ln_lam_temp: array_like
        is the natural log of the wavelength of the templates in Angstrom.
        ``ln_lam_temp`` should be the same as that of the stellar templates.
    lam_range_gal: array_like
        is the estimated rest-frame fitted wavelength range. Typically::

            lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),

        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial rough estimate of the galaxy redshift.
    FWHM_gal: float or func
        is the instrumantal FWHM of the galaxy spectrum under study in Angstrom.
        One can pass either a scalar or the name "func" of a function
        ``func(wave)`` which returns the FWHM for a given vector of input
        wavelengths.
    pixel: bool, optional
        Set this to ``False`` to ignore pixels integration (default ``True``).
    tie_balmer: bool, optional
        Set this to ``True`` to tie the Balmer lines according to a theoretical
        decrement (case B recombination T=1e4 K, n=100 cm^-3).

        IMPORTANT: The relative fluxes of the Balmer components assumes the
        input spectrum has units proportional to ``erg/(cm**2 s A)``.
    limit_doublets: bool, optional
        Set this to True to limit the rato of the [OII] and [SII] doublets to
        the ranges allowed by atomic physics.

        An alternative to this keyword is to use the ``constr_templ`` keyword
        of pPXF to constrain the ratio of two templates weights.

        IMPORTANT: when using this keyword, the two output fluxes (flux_1 and
        flux_2) provided by pPXF for the two lines of the doublet, do *not*
        represent the actual fluxes of the two lines, but the fluxes of the two
        input *doublets* of which the fit is a linear combination.
        If the two doublets templates have line ratios rat_1 and rat_2, and
        pPXF prints fluxes flux_1 and flux_2, the actual ratio and flux of the
        fitted doublet will be::

            flux_total = flux_1 + flux_1
            ratio_fit = (rat_1*flux_1 + rat_2*flux_2)/flux_total

        EXAMPLE: For the [SII] doublet, the adopted ratios for the templates are::

            ratio_d1 = flux([SII]6716/6731) = 0.44
            ratio_d2 = flux([SII]6716/6731) = 1.43.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([SII]6731_d1) = flux_1
            flux([SII]6731_d2) = flux_2

        the total flux and true lines ratio of the [SII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([SII]6716/6731) = (0.44*flux_1 + 1.43*flux_2)/flux_total

        Similarly, for [OII], the adopted ratios for the templates are::

            ratio_d1 = flux([OII]3729/3726) = 0.28
            ratio_d2 = flux([OII]3729/3726) = 1.47.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([OII]3726_d1) = flux_1
            flux([OII]3726_d2) = flux_2

        the total flux and true lines ratio of the [OII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([OII]3729/3726) = (0.28*flux_1 + 1.47*flux_2)/flux_total

    vacuum:  bool, optional
        set to ``True`` to assume wavelengths are given in vacuum.
        By default the wavelengths are assumed to be measured in air.

    Output Parameters
    -----------------

    emission_lines: ndarray
        Array of dimensions ``[ln_lam_temp.size, line_wave.size]`` containing
        the gas templates, one per array column.

    line_names: ndarray
        Array of strings with the name of each line, or group of lines'

    line_wave: ndarray
        Central wavelength of the lines, one for each gas template'

    """
    #        Balmer:     H10       H9         H8        Heps    Hdelta    Hgamma    Hbeta     Halpha
    balmer = np.array([3798.983, 3836.479, 3890.158, 3971.202, 4102.899, 4341.691, 4862.691, 6564.632])  # vacuum wavelengths

    if tie_balmer:

        # Balmer decrement for Case B recombination (T=1e4 K, ne=100 cm^-3)
        # from Storey & Hummer (1995) https://ui.adsabs.harvard.edu/abs/1995MNRAS.272...41S
        # In electronic form https://cdsarc.u-strasbg.fr/viz-bin/Cat?VI/64
        # See Table B.7 of Dopita & Sutherland 2003 https://www.amazon.com/dp/3540433627
        # Also see Table 4.2 of Osterbrock & Ferland 2006 https://www.amazon.co.uk/dp/1891389343/
        wave = balmer
        if not vacuum:
            wave = vac_to_air(wave)
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal, pixel)
        ratios = np.array([0.0530, 0.0731, 0.105, 0.159, 0.259, 0.468, 1, 2.86])
        ratios *= wave[-2]/wave  # Account for varying log-sampled pixel size in Angstrom
        emission_lines = gauss @ ratios
        line_names = ['Balmer']
        w = (lam_range_gal[0] < wave) & (wave < lam_range_gal[1])
        line_wave = np.mean(wave[w]) if np.any(w) else np.mean(wave)

    else:

        line_wave = balmer
        if not vacuum:
            line_wave = vac_to_air(line_wave)
        line_names = ['H10', 'H9', 'H8', 'Heps', 'Hdelta', 'Hgamma', 'Hbeta', 'Halpha']
        emission_lines = gaussian(ln_lam_temp, line_wave, FWHM_gal, pixel)

    if limit_doublets:

        # The line ratio of this doublet lam3727/lam3729 is constrained by
        # atomic physics to lie in the range 0.28--1.47 (e.g. fig.5.8 of
        # Osterbrock & Ferland 2006 https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #       -----[OII]-----
        wave = [3727.092, 3729.875]    # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[OII]3726_d1', '[OII]3726_d2']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal, pixel)
        doublets = gauss @ [[1, 1], [0.28, 1.47]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

        # The line ratio of this doublet lam6717/lam6731 is constrained by
        # atomic physics to lie in the range 0.44--1.43 (e.g. fig.5.8 of
        # Osterbrock & Ferland 2006 https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #        -----[SII]-----
        wave = [6718.294, 6732.674]    # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[SII]6731_d1', '[SII]6731_d2']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal, pixel)
        doublets = gauss @ [[0.44, 1.43], [1, 1]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    else:

        # Here the two doublets are free to have any ratio
        #         -----[OII]-----     -----[SII]-----
        wave = [3727.092, 3729.875, 6718.294, 6732.674]  # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal, pixel)
        emission_lines = np.column_stack([emission_lines, gauss])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    # Here the lines are free to have any ratio
    #       -----[NeIII]-----    HeII      HeI      OI
    wave = [3968.59, 3869.86, 4687.015, 5877.243, 8446.5]  # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    names = ['[NeIII]3968', '[NeIII]3869', 'HeII4687', 'HeI5876', 'OI8446']
    gauss = gaussian(ln_lam_temp, wave, FWHM_gal, pixel)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, wave)

    ######### Doublets with fixed ratios #########

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OIII]-----
    wave = [4960.295, 5008.240]    # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OIII]5007_d')  # single template for this doublet
    line_wave = np.append(line_wave, wave[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OI]-----
    wave = [6302.040, 6365.535]    # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal, pixel) @ [1, 0.33]
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OI]6300_d')  # single template for this doublet
    line_wave = np.append(line_wave, wave[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #       -----[NII]-----
    wave = [6549.860, 6585.271]    # air wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NII]6583_d')  # single template for this doublet
    line_wave = np.append(line_wave, wave[1])

    # Only include lines falling within the estimated fitted wavelength range.
    #
    w = (lam_range_gal[0] < line_wave) & (line_wave < lam_range_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave