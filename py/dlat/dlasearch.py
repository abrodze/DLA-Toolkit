#!/usr/bin/env python

"""
Identify DLAs in data sample using template plus viogt fits
"""

import numpy as np
import os
import fitsio
from astropy.table import Table, vstack

import time 
import datetime

from scipy.optimize import curve_fit

# desi packages - TO DO : remove or isolate desi dependencies
import desispec.io
from desispec.interpolation import resample_flux
from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

from . import constants
from . import dlaprofile
from .fitwarning import DLAFLAG

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)

# all spectra use the same column density range to search for DLAs
# define globally here
nhiscan = np.arange(constants.nhimin, constants.nhimax, 0.2) # must allow subDLA NHI to avoid FPs
nhirefine = np.arange(constants.nhimin, constants.nhimax, 0.05)
nhi_factors = 10.**nhiscan
nhi_refine_factors = 10.**nhirefine


def timestamp():
    """ 
    return current time in YYYY-MM-DD HH:MM:SS format
    
    Arguments
    ---------
    None

    Returns
    -------
    (str) : current timestamp formatted as YYYY-MM-DD HH:MM:SS
    """
    now = datetime.datetime.now()
    return(now.strftime("%Y-%m-%d %H:%M:%S"))


def dlasearch_hpx(healpix, survey, program, datapath, pixcat, model):
    """
    Find the best fitting DLA profile(s) for spectra in hpx catalog

    Arguments
    ---------
    healpix (int) : healpixel number, agnostic to nside
    survey (str) : e.g., main, sv1, sv2, etc.
    program (str) : e.g., bright, dark, etc.
    datapath (str) : path to coadd files
    pixcat (table) : collection of spectra to search for DLAs, all belonging to
                     single healpix
    model (dict) : flux model dictionary containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys

    Returns
    -------
    fitresults (table) : attributes of detected DLAs
    """

    t0 = time.time()
   
    # read spectra from healpixel
    coaddname = f'coadd-{survey}-{program}-{str(healpix)}.fits'
    coadd = os.path.join(datapath, str(healpix//100), str(healpix), coaddname)

    if os.path.exists(coadd):

        fitresults = process_spectra_group(coadd, pixcat, model, False)

    else:
        print(f'{timestamp()} - Warning: could not locate coadd file for healpix {healpix}')
        return()        


    t1 = time.time()
    total = np.round(t1-t0,2)
    print(f'{timestamp()} - Completed processing of {len(pixcat)} spectra from healpix {healpix} in {total}s')

    return fitresults
    

def dlasearch_mock(specfile, catalog, model):
    """
    Find the best fitting DLA profile(s) for spectra in mock spectra file

    Arguments
    ---------
    specfile (str) : path to mock spectra-16-X.fits file
    catalog (table) : catalog of spectra to search for DLAs
    model (dict) : flux model dictionary containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys

    Returns
    -------
    fitresults (table) : fit attributes for detected DLAs
    """
    
    t0 = time.time()

    if os.path.exists(specfile):
    
        # open spectra file fibermap only
        fm = desispec.io.read_fibermap(specfile)
    
        # pare catalog to match spectra file fibermap
        tidmask = np.isin(catalog['TARGETID'], fm['TARGETID'])
        catalog = catalog[tidmask]
        if len(catalog) < 1:
            # no objects
            return()

        fitresults = process_spectra_group(specfile, catalog, model, True)

    else:
        print(f'{timestamp()} - Warning: could not locate coadd file: {specfile}')
        return()


    t1 = time.time()
    total = np.round(t1-t0,2)
    print(f'{timestamp()} - Completed processing of {len(catalog)} spectra from {specfile} in {total}s')

    return fitresults


def process_spectra_group(coaddpath, catalog, model, is_mock=False):
    """
    pre-process group of spectra in same file and run DLA searching tools

    Arguments
    ---------
    coaddpath (str) : path to file containing spectra
    catalog (table) : collection of spectra in file to search for DLAs
    model (dict) : flux model containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys
    is_mock (bool) : True if processing mock spectra file

    Returns
    -------
    fitresults (table) : attributes of detected DLAs
    """

    specobj = desispec.io.read_spectra(coaddpath, targetids=catalog['TARGETID'], skip_hdus=['EXP_FIBERMAP', 'SCORES', 'EXTRA_CATALOG'])
    if not(is_mock):
        specobj = coadd_cameras(specobj)
    else:
        if specobj.resolution_data is not None: 
            # resample on linear grid
            wave_min = np.min(specobj.wave['b'])
            wave_max = np.max(specobj.wave['z'])
            specobj = resample_spectra_lin_or_log(specobj, linear_step=0.8, wave_min = wave_min, 
                                                  wave_max = wave_max, fast = True)
            specobj = coadd_cameras(specobj)
        else:
            # check if mock truth file exists
            truthfile = coaddpath.replace('spectra-16-', 'truth-16-')
            if not(os.path.exists(truthfile)):
                print(f"{timestamp()} - Error: cannot process {coaddpath}; no mock truth file or resolution data")
            specobj.resolution_data = {}
            for cam in ['b', 'r', 'z']:
                tres = fitsio.read(truthfile, ext=f'{cam}_RESOLUTION')
                tresdata = np.empty([specobj.flux[cam].shape[0], tres.shape[0], specobj.flux[cam].shape[1]], dtype=float)
                for i in range(specobj.flux[cam].shape[0]):
                    tresdata[i] = tres
                specobj.resolution_data[cam] = tresdata
            specobj = resample_spectra_lin_or_log(specobj, linear_step=0.8, wave_min = np.min(specobj.wave['b']), 
                                                  wave_max = np.max(specobj.wave['z']), fast = True)

    wave = specobj.wave['brz']

    # var_lss term for Lya and Lyb+ regions
    varlss_lya = model['VAR_FUNC_LYA'](wave)
    varlss_lyb = model['VAR_FUNC_LYB'](wave)

    tidlist, ralist, declist, zqsolist, bluesnrlist, redsnrlist, dlaidlist = [], [], [], [], [], [], []
    zlist, nhilist, dchi2list, zerrlist, nhierrlist, fitwarnlist, coefflist = [], [], [], [], [], [], []

    # create tid to index mapping
    tid_to_idx = {tid: i for i, tid in enumerate(np.asarray(specobj.fibermap['TARGETID']))}

    # for each entry in passed catalog, fit spectrum with intrinsic model + N DLA
    for entry in range(len(catalog)):

        tid = catalog['TARGETID'][entry]
        try:
            ra = catalog['TARGET_RA'][entry]
            dec = catalog['TARGET_DEC'][entry]
        except:
            # mock catalog
            ra = catalog['RA'][entry]
            dec = catalog['DEC'][entry]
        zqso = catalog['Z'][entry]

        idx = tid_to_idx.get(tid)
        if idx is None:
            print(f'{timestamp()} - Error: Targetid {tid} NOT FOUND on {coaddpath}')
            continue

        flux = specobj.flux['brz'][idx]
        ivar = specobj.ivar['brz'][idx]
        wave_rf = wave / (1 + zqso)

        # only searching to rest frame 912 A
        fitmask = wave_rf > constants.search_minlam

        # limit our bestfit comparision w/ and w/o DLAs to search region of spectrum
        searchmask = (wave_rf[fitmask] >= constants.search_minlam) & (wave_rf[fitmask] <= constants.search_maxlam)
        
        # apply mask to BAL features, if available
        # initialize as no BALs
        has_bal_info = 'NCIV_450' in catalog.columns
        bal_locs, nbal = [], 0
        if has_bal_info:
            nbal = catalog['NCIV_450'][entry]
            for n in range(nbal):
                
                # Compute velocity ranges
                v_max = -catalog[entry]['VMAX_CIV_450'][n] / constants.c + 1.
                v_min = -catalog[entry]['VMIN_CIV_450'][n] / constants.c + 1.

                for line, lam in constants.bal_lines.items():
                    # Mask wavelengths within the velocity ranges
                    mask = (wave_rf <= lam*v_min) & (wave_rf >= lam*v_max)
                    if (line == 'Lya') or (line == 'NV'):
                        rededge = (lam*v_min)*(1+zqso)
                        blueedge = (lam*v_max)*(1+zqso)
                        bal_locs.append((rededge,blueedge))

                    # Update ivar = 0
                    ivar[mask] = 0

        # check if too much of the spectrum is masked
        if np.sum(ivar[fitmask][searchmask] != 0)/np.sum(searchmask) < 0.2:
            print(f'{timestamp()} - Warning: Targetid {tid} skipped - SEARCH WINDOW >80% MASKED')
            continue

        # resample model to observed wave grid
        fitmodel = np.zeros([model['PCA_COMP'].shape[0], np.sum(fitmask)])
        for i in range(model['PCA_COMP'].shape[0]):
            fitmodel[i] = resample_flux(wave[fitmask], model['PCA_WAVE']*(1 + zqso), model['PCA_COMP'][i])

        # apply mean transmission correction for lyman alpha forest
        for transition, values in constants.Lyman_series[model['IGM']].items():
            lam_range = wave_rf[fitmask] < values['line']
            zpix = wave[fitmask][lam_range] / values['line'] - 1
            T = np.exp( -values['A']*(1+zpix)**values['B'] )
            fitmodel[:,lam_range] *= T

        # determine var_lss array
        lyaregion = (wave_rf < constants.Lya_line) & (wave_rf > constants.Lyb_line)
        lybregion = wave_rf < constants.Lyb_line # assuming N>3 transition minimal impact
        varlss = np.zeros(len(ivar))
        varlss[lyaregion] = varlss_lya[lyaregion]
        varlss[lybregion] = varlss_lyb[lybregion]

        var_pipe = np.zeros_like(ivar)
        var_pipe[ivar != 0] = 1.0 / ivar[ivar != 0]

        # model w/o DLAs
        coeff_null, chi2dof_null  = fit_spectrum(wave[fitmask], flux[fitmask], ivar[fitmask], var_pipe[fitmask], fitmodel, 
                                                 varlss[fitmask], searchmask)

        # add up to 3 DLAs to fit, no detections have [z, zerr, nhi, nhierr, dchi2] = [-1, inf, 0, inf, 0]
        zdla, zerr, nhi, nhierr, dchi2, fitwarn, coeff_dla  = fit_spectrum_DLA(wave[fitmask], flux[fitmask], ivar[fitmask], var_pipe[fitmask],
                                                            fitmodel, varlss[fitmask], searchmask, zqso, chi2dof_null)

        # check for potential BAL contamination in solution
        # false positive should only come from Lya and NV - all other lines too weak
        if (nbal > 0) & np.any(zdla != -1):
            lam_center_dla = constants.Lya_line*(1+zdla)
            for window in bal_locs:
                balflag = (lam_center_dla < window[0]) & (lam_center_dla > window[1])
                fitwarn[balflag] |= DLAFLAG.POTENTIAL_BAL

        # average signal to noise computation
        mask = (ivar != 0) & ((wave_rf >= constants.bluesnr_min) & (wave_rf <= constants.bluesnr_max))
        # check for insufficent coverage and set to -1
        if np.any(mask):
            bluesnr = np.mean((flux[mask]*np.sqrt(ivar[mask])))
        else:
            bluesnr = -1
        
        mask = (ivar != 0) & ((wave_rf >= constants.redsnr_min) & (wave_rf <= constants.redsnr_max))
        if np.any(mask):
            redsnr = np.mean((flux[mask]*np.sqrt(ivar[mask])))
        else:
            redsnr = -1
            
        ndla = np.sum(zdla != -1)
        for n in range(ndla):
            tidlist.append(tid)
            dlaid = str(tid)+'00'+str(n)
            dlaidlist.append(dlaid)
            ralist.append(ra)
            declist.append(dec)
            zqsolist.append(zqso)
            zlist.append(zdla[n])
            zerrlist.append(zerr[n])
            nhilist.append(nhi[n])
            nhierrlist.append(nhierr[n])
            dchi2list.append(dchi2[n])
            fitwarnlist.append(fitwarn[n])
            coefflist.append(coeff_dla[n])
            bluesnrlist.append(bluesnr)
            redsnrlist.append(redsnr)

    if len(tidlist) == 0:
        # avoid vstack error for empty tables
        return()

    fitresults = Table(data =(tidlist, ralist, declist, zqsolist, bluesnrlist, redsnrlist, dlaidlist, 
                              zlist, zerrlist, nhilist, nhierrlist, coefflist, dchi2list, fitwarnlist),
                names=['TARGETID', 'RA', 'DEC', 'Z_QSO', 'SNR_FOREST', 'SNR_REDSIDE', 'DLAID', 
                       'Z_DLA', 'Z_DLA_ERR', 'NHI', 'NHI_ERR', 'COEFF', 'DELTACHI2', 'DLAFLAG'],
                dtype=('int', 'float64', 'float64', 'float64', 'float64', 'float64', 'str', 
                       'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int'))
        
    return(fitresults)

def PCA_reconstruction_DLA(x, dla_params, eigvec, lam_obs):
    """
    reconstruct spectrum with eigenspectra + DLA absorption model

    Arguments
    ---------
    x (array of floats) : coeff on eigenspectra, length of nvec
    dla_params (array of tuples) : list of tuples (zdla, log10(NHI)) for all DLAs;
                                  if None, treated as no DLAs
    eigvec (2D array of floats) : array of eigenspectra, nvec x nlam
    lam_obs (array of floats) : observed wave array in angstroms, length nlam

    Returns
    -------
    recon_spec (array of floats) : reconstructed spectrum, length of nlam

    """

    if dla_params is None:
        dla_params = []
    trans = np.ones_like(lam_obs)
    for (z_s, nhi_s) in dla_params:
        trans = trans * np.exp(
            -(10.0 ** nhi_s) * dlaprofile.dla_tau_from_zdla(lam_obs, z_s))

    recon_spec = np.dot(x, trans*eigvec)

    return(recon_spec)

def parabola(x, a, b, c):
    """
    Parabola model for curve_fit, of the form y = a + ((x-b)/c)**2

    Arguments
    ---------
    x (array of floats) : independent variable values
    a (float) : minimum value of parabola (y-offset)
    b (float) : x-position of parabola minimum
    c (float) : width parameter (error-like scaling on x)

    Returns
    -------
    y (array of floats) : parabola evaluated at x
    """
    return( a + ((x-b)/c)**2 )


def _solve_DLA_with_transmission(ivar, flux, model_flux, var_pipe, varlss, wave, searchmask, return_coeff, 
                                 DLA_transmission, NDLA):
    """
    fit spectrum with eigenspectra given a fixed DLA profile

    Arguments
    ---------
    ivar (array of floats) : inverse variance on observed flux
    flux (array of floats) : observed flux
    model_flux (2D array of floats) : eigenspectra model with dimension nvec X nlam
    var_pipe (array of floats) : pipeline variance estimated from ivar
    varlss (array of floats) : LSS variance
    wave (array of floats) : observed wavelength (Angstroms)
    searchmask (array of bool) : DLA search window mask, dimensions nlam
    return_coeff (bool) : return coefficients of bestfit
    DLA_transmission (array of floats) :  precomputed transmission array to multiplied against model
    NDLA (int) : number of DLAs being fit

    Returns
    -------
    coeff (array of floats, optional) : coefficients on eigenspectra, dimensions nvec
    chi2dof (float) : reduced chi2 of best fit over DLA search range defined
                      by searchmask argument
    """

    # mask ivar = 0 entries to avoid divide by zero error
    mask = ivar != 0
    
    # compute model with DLAs
    M = (DLA_transmission*model_flux)[:,mask]
    
    # assume only pipeline error contributes for first fit
    # lss contribution is model flux dependent - solved for below
    w_m = ivar.copy()
    flux_m = flux[mask]
    var_pipe_m = var_pipe[mask]
    varlss_m = varlss[mask]
    dw_max = 1.0

    niter = 0
    while (dw_max > 10e-4) and (niter < 5):
        # linalg requires a and b are square matrices
        # also taking ivar weights into account
        b = M.dot( w_m*flux_m )
        a = M.dot( (M * w_m).T )

        coeff = np.linalg.solve(a,b)
        bestfit = np.dot(coeff,M)

        # adjust weights for LSS contribution
        nw_m = 1.0 / (var_pipe_m + varlss_m * bestfit**2)
        dw_max = np.max(np.abs(w_m - nw_m) / w_m)
        w_m = nw_m
        niter += 1

    # get the chi2 of fit just in the region we are searching for DLAs
    smask = mask&searchmask
    dof = np.sum(mask[searchmask]) - model_flux.shape[0] - float(NDLA)
    bestfit = coeff.dot((DLA_transmission*model_flux)[:, smask])
    w = 1./(var_pipe[smask]  + varlss[smask]*bestfit**2)
    chi2 = np.sum(w * (flux[smask] - bestfit)**2)

    if return_coeff:
        return(coeff, chi2/dof)
    else:
        return(chi2/dof)

def _refined_scan_z(wave, ivar, flux, model_flux, var_pipe, varlss, searchmask,
                    z_transmission_grid, fixed_nhi, fixed_trans, ndla_total):
    """
    Refined 1D chi2 scan over DLA redshift at fixed NHI, on top of a fixed
    background transmission from previously solved DLAs

    Arguments
    ---------
    wave (array of floats) : observed wavelength in Angstroms
    ivar (array of floats) : inverse variance on observed flux
    flux (array of floats) : observed flux
    model_flux (2D array of floats) : eigenspectra model with dimensions nvec x nlam
    varlss (array of floats) : LSS variance
    searchmask (array of bool) : DLA search window mask, dimensions nlam
    z_transmission_grid (2D array of floats) : precomputed DLA optical depth (tau) per
                                           redshift on observed wavelength grid,
                                           dimensions nz x nlam
    fixed_nhi (float) : log10 column density (cm^-2) of the DLA being scanned, held fixed
    fixed_trans (array of floats) : background transmission from already-solved DLAs,
                                    dimensions nlam
    ndla_total (int) : total number of DLAs in the fit (including the one being scanned),
                       used in degrees-of-freedom calculation

    Returns
    -------
    chi2 (array of floats) : reduced chi2 evaluated at each redshift in zrefine
    """
    nhi_factor = 10.**fixed_nhi
    trans = np.exp(-nhi_factor * z_transmission_grid) * fixed_trans
    chi2 = np.empty(z_transmission_grid.shape[0])
    for iz in range(z_transmission_grid.shape[0]):
        chi2[iz] = _solve_DLA_with_transmission(
            ivar, flux, model_flux, var_pipe, varlss, wave, searchmask,
            False, trans[iz], ndla_total)
    return(chi2)


def _refined_scan_nhi(wave, ivar, flux, model_flux, var_pipe, varlss, searchmask,
                      fixed_z, fixed_trans, ndla_total):
    """
    Refined 1D chi2 scan over DLA log10(NHI) at fixed redshift, on top of a fixed
    background transmission from previously solved DLAs

    Arguments
    ---------
    wave (array of floats) : observed wavelength in Angstroms
    ivar (array of floats) : inverse variance on observed flux
    flux (array of floats) : observed flux
    model_flux (2D array of floats) : eigenspectra model with dimensions nvec x nlam
    varlss (array of floats) : LSS variance
    searchmask (array of bool) : DLA search window mask, dimensions nlam
    fixed_z (float) : redshift of the DLA being scanned, held fixed
    fixed_trans (array of floats) : background transmission from already-solved DLAs,
                                    dimensions nlam
    ndla_total (int) : total number of DLAs in the fit (including the one being scanned),
                       used in degrees-of-freedom calculation

    Returns
    -------
    chi2 (array of floats) : reduced chi2 evaluated at each NHI value in nhirefine
    """
    tau_at_z = dlaprofile.dla_tau_from_zdla(wave, fixed_z)
    chi2 = np.empty(len(nhirefine))
    trans = np.exp(-nhi_refine_factors[:, None] * tau_at_z[None, :]) * fixed_trans
    for inhi in range(len(nhirefine)):
        chi2[inhi] = _solve_DLA_with_transmission(
            ivar, flux, model_flux, var_pipe, varlss, wave, searchmask,
            False, trans[inhi], ndla_total)
    return(chi2)

def _build_fixed_trans(wave, solved_dlas):
    """
    Compute combined background transmission from an arbitrary number of DLAs

    Arguments
    ---------
    wave (array of floats) : observed wavelength in Angstroms, length nlam
    solved_dlas (list of tuples) : list of (zdla, log10(NHI)) tuples for previously
                                   solved DLAs; may be empty

    Returns
    -------
    fixed_trans (array of floats) : combined transmission profile from all DLAs in
                                    solved_dlas, dimensions nlam; ones everywhere if
                                    solved_dlas is empty
    """
    fixed_trans = np.ones_like(wave)
    for (z_s, nhi_s) in solved_dlas:
        fixed_trans = fixed_trans * np.exp(
            -(10.0 ** nhi_s) * dlaprofile.dla_tau_from_zdla(wave, z_s))
    return(fixed_trans)


def fit_spectrum_DLA(wave, flux, ivar, var_pipe, model_flux, varlss, searchmask, zqso, chi2null):
    """
    fit spectrum with eigenspectra model with up to 3 free DLA (zdla, nhi) profiles

    Arguments
    ---------
    wave (array of floats) : observed wavelength in Angstroms
    flux (array of floats) : observed flux
    ivar (array of floats) : inverse variance on observed flux
    var_pipe (array of floats) : pipeline variance estimated from ivar
    model_flux (2D array of floats) : eigenspectra model with dimensions nvec x nlam
    varlss (array of floats) : LSS variance
    searchmask (array of bool) : DLA search window mask, dimensions nlam
    zqso (float) : quasar redshift
    chi2null (float) : reference reduced chi2 of the DLA-free model fit

    Returns
    -------
    zdla_soln (array of floats) : best fit DLA redshifts, length 3; -1 for no detection
    zerr_soln (array of floats) : error on zdla estimated with parabola fit, length 3;
                                  np.inf for no detection
    nhi_soln (array of floats) : best fit log10 DLA column density (cm^-2), length 3;
                                 0 for no detection
    nhierr_soln (array of floats) : error on nhi estimated with parabola fit, length 3;
                                    np.inf for no detection
    dchi2_soln (array of floats) : chi2 improvement from each DLA relative to the
                                   previous best fit (null fit for the 1st DLA),
                                   evaluated over the searchmask region, length 3
    fitwarning (array of int) : bitmask flags on each DLA solution, length 3
    coeff_soln (2D array of floats) : coefficients on eigenspectra for each DLA solution,
                                      dimensions 3 x nvec
    """

    # no detection will give results of [z, zerr, nhi, nhierr, dchi2] = [-1, inf, 0, inf, 0]
    zdla_soln = np.full(3,-1.)
    zerr_soln = np.full(3,np.inf)
    nhi_soln = np.full(3,0.)
    nhierr_soln = np.full(3,np.inf)
    dchi2_soln = np.full(3,0.)
    fitwarning = np.full(3,0)
    coeff_soln = np.full((3,model_flux.shape[0]), 0. )

    # redshift and log10(NHI) search values
    # zmin = max(912/lya * (1 + zqso) - 1 + 3000 km/s, min(lam_ob)/lya)
    # zmax = zqso - 3000 km/s
    zmin = max((constants.Lyinf/constants.Lya_line)*(1 + zqso) - 1 + 0.01, min(wave[searchmask])/constants.Lya_line - 1)
    zscan = np.linspace(zmin,zqso-0.01,int(np.ceil((zqso-zmin)/0.01)))
    # nhiscan and nhiscan_refine are the same for all objects, defined as global variable

    # precompute transmission grid for zscan and re-use throughout solves
    tau_grid = np.empty((len(zscan), len(wave)))
    for iz, z in enumerate(zscan):
        tau_grid[iz] = dlaprofile.dla_tau_from_zdla(wave, z)

    def refined_fit(z_bf_idx, nhi_bf_idx, ini_chi2, solved_dlas):
        """
        Iteratively refine the best fit (z, NHI) for a single new DLA on top of any
        previously solved DLAs, using alternating 1D parabola fits in z and NHI.

        Arguments
        ---------
        z_bf_idx (int) : index of coarse-scan best fit redshift in zscan
        nhi_bf_idx (int) : index of coarse-scan best fit log10(NHI) in nhiscan
        ini_chi2 (float) : reduced chi2 value at the coarse-scan minimum, used as
                           the initial guess for the parabola minimum/offset parameter
        solved_dlas (list of tuples) : list of (z, log10(NHI)) tuples for already-fixed
                                       DLAs; may be empty

        Returns
        -------
        bestz (float) : refined best fit DLA redshift
        zerr (float) : error on bestz estimated with parabola fit; np.inf if fit failed
        bestnhi (float) : refined best fit log10 DLA column density (cm^-2)
        nhierr (float) : error on bestnhi estimated with parabola fit; np.inf if fit failed
        chi2dof (float) : reduced chi2 of the final refined fit over the searchmask region
        fitwarn (int) : bitmask flags raised during the refined fit
        coeff (array of floats) : coefficients on eigenspectra for the final fit,
                                  dimensions nvec
        """

        # initial flag with no warning
        fitwarn = 0
        ndla_total = len(solved_dlas) + 1
        # transmission vector from DLAs previously detected
        fixed_trans = _build_fixed_trans(wave, solved_dlas)

        # check if fit solution relaxed to redshift boundary
        if z_bf_idx < (len(zscan)-1):
            # define refined search grid, allowing for slight buffer for parabola fitting if at z \approx zqso
            zrefine = np.arange(zscan[np.max([z_bf_idx-2, 0])], zqso, 0.0025)
        else: 
            # define refined search grid
            zrefine = np.arange(zscan[np.max([z_bf_idx-2, 0])], zscan[np.min([z_bf_idx+2, len(zscan)-1])]+0.001, 0.0025)

        # precompute tau_refined grid since zrefine will not change
        tau_refine_grid = np.empty((len(zrefine), len(wave)))
        for iz, z in enumerate(zrefine):
            tau_refine_grid[iz] = dlaprofile.dla_tau_from_zdla(wave, z)

        # define curve_fit boundaries
        zbounds = ([0, zrefine[0], 0], [np.inf, zrefine[-1], 10e4])
        nhibounds = ([0, nhirefine[0], 0], [np.inf, nhirefine[-1], 10e4])

        ### Iteratively solving 2 1D parabolas
        ### repeat until parameters converge or niter = 5
        niter = 0
        param_frac_change = [1., 1.]
        last_bestz = zscan[z_bf_idx]
        last_bestnhi = nhiscan[nhi_bf_idx]

        while (niter < 5) and (max(param_frac_change) > 10e-5):

            # refined solve for z, assuming nhi value
            zzchi2 = _refined_scan_z(wave, ivar, flux, model_flux, var_pipe, varlss, searchmask,
                                     tau_refine_grid, last_bestnhi, fixed_trans, ndla_total)

                
            # fit minima of refined search with parabola to find best fit z+zerr
            iniguess = np.array([ini_chi2, last_bestz, 0.1])
            minpos = np.argmin(zzchi2)
            ini_idx = np.max([minpos-2, 0])
            fin_idx = np.min([minpos+3,len(zzchi2)])
            try:
                popt, pcov = curve_fit(parabola, zrefine[ini_idx:fin_idx],
                                       zzchi2[ini_idx:fin_idx],
                                       p0=iniguess, bounds=zbounds)
                bestz = popt[1]
                zerr = popt[2]
            except(ValueError,OptimizeWarning,RuntimeError):
                fitwarn |= DLAFLAG.BAD_ZFIT
                bestz = last_bestz
                zerr = np.inf

            # refined solve for nhi, assuming z value
            zzchi2 = _refined_scan_nhi(wave, ivar, flux, model_flux, var_pipe, varlss, searchmask,
                                       bestz, fixed_trans, ndla_total)
                
            # fit minima of refined search with parabola to find best fit nhi+nhierr
            iniguess = np.array([ini_chi2, last_bestnhi, 1.])
            minpos = np.argmin(zzchi2)
            ini_idx = np.max([minpos-2, 0])
            fin_idx = np.min([minpos+3,len(zzchi2)])
            try:
                popt, pcov = curve_fit(parabola, nhirefine[ini_idx:fin_idx], 
                                       zzchi2[ini_idx:fin_idx],
                                       p0=iniguess, bounds=nhibounds)
                bestnhi = popt[1]
                nhierr = popt[2]
            except(ValueError, OptimizeWarning,RuntimeError):
                fitwarn |= DLAFLAG.BAD_NHIFIT
                bestnhi = last_bestnhi
                nhierr = np.inf

            # increase niter, reset last_* vars, set fractional change in parameters for iteration
            param_frac_change[0] = abs(bestz - last_bestz) / last_bestz
            param_frac_change[1] = abs(bestnhi - last_bestnhi) / last_bestnhi
            last_bestz = bestz
            last_bestnhi = bestnhi
            niter += 1
            
        # trigger warning if solution \approx boundary
        if (np.round(bestnhi - nhibounds[0][1],6) == 0) or (np.round(bestnhi - nhibounds[1][1],6) == 0):
            fitwarn |= DLAFLAG.NHIBOUNDARY
        if (np.round(bestz - zbounds[0][1],6) == 0) or (np.round(bestz - zbounds[1][1],6) == 0):
            fitwarn |= DLAFLAG.ZBOUNDARY
        # also trigger a warning if the redshift solution is too close to ZQSO
        if bestz >= np.max(zscan):
            fitwarn |= DLAFLAG.ZBOUNDARY

        # fit final solution 
        all_dlas = list(solved_dlas) + [(bestz, bestnhi)]
        final_trans = _build_fixed_trans(wave, all_dlas)
        coeff, chi2dof = _solve_DLA_with_transmission(ivar, flux, model_flux, var_pipe, varlss, wave, 
                                                      searchmask, True, final_trans, ndla_total)
        
        return(bestz, zerr, bestnhi, nhierr, chi2dof, fitwarn, coeff)


    # coarse solve - vectorized over NHI for each z
    zchi2 = np.empty((len(zscan), len(nhiscan)))
    
    # Precompute fixed DLA transmission 
    # For the 1st-DLA coarse scan, start with transmission = 1 everywhere
    fixed_trans_0 = _build_fixed_trans(wave, [])
    for iz in range(len(zscan)):
        trans_nhi = np.exp(-nhi_factors[:, None] * tau_grid[iz][None, :]) * fixed_trans_0
        for inhi in range(len(nhiscan)):
            zchi2[iz, inhi] = _solve_DLA_with_transmission(
                ivar, flux, model_flux, var_pipe, varlss, wave, searchmask, False, trans_nhi[inhi], 1)

    # find minimum of coarse search
    bf = np.unravel_index(zchi2.argmin(), zchi2.shape)

    # if all of chi2 surface is > null chi2, do not bother with refined solve
    if zchi2[bf] > chi2null:
        return(zdla_soln, zerr_soln, nhi_soln, nhierr_soln, dchi2_soln, fitwarning, coeff_soln)

    # otherwise let's continue with solving
    bestz, zerr, bestnhi, nhierr, chi2dof, fitwarning[0], coeff = refined_fit(bf[0], bf[1], zchi2[bf], [])
   

    # check if DLA is detected using chi2 detection threshold AND didn't fail on coarse solve
    if ((chi2null - chi2dof) > constants.detection) & ((fitwarning[0] & DLAFLAG.ZBOUNDARY) == 0):

        # store best fit attributes for 1st DLA
        zdla_soln[0] = bestz
        zerr_soln[0] = zerr
        nhi_soln[0] = bestnhi
        nhierr_soln[0] = nhierr
        dchi2_soln[0] = chi2null - chi2dof
        coeff_soln[0] = coeff
                
        # fix first solution
        # coarse solve - vectorized over NHI for each z
        zchi2 = np.empty((len(zscan), len(nhiscan)))
        
        # Precompute fixed DLA transmission 
        fixed_trans_1 = _build_fixed_trans(wave, [(zdla_soln[0], nhi_soln[0])])
        for iz in range(len(zscan)):
            trans_nhi = np.exp(-nhi_factors[:, None] * tau_grid[iz][None, :]) * fixed_trans_1
            for inhi in range(len(nhiscan)):
                zchi2[iz, inhi] = _solve_DLA_with_transmission(
                    ivar, flux, model_flux, var_pipe, varlss, wave, searchmask, False, trans_nhi[inhi], 2)
                
        # find minimum of coarse search
        bf = np.unravel_index(zchi2.argmin(), zchi2.shape)

        bestz, zerr, bestnhi, nhierr, chi2dof, fitwarning[1], coeff = refined_fit(bf[0], bf[1], zchi2[bf], [(zdla_soln[0], nhi_soln[0])])

        # TO DO : should subsequent DLAs be held to higher detection thresholds?
        if (((chi2null-dchi2_soln[0]) - chi2dof) > constants.detection) & ((fitwarning[1] & DLAFLAG.ZBOUNDARY) == 0):

            # store best fit attributes for 2nd DLA
            zdla_soln[1] = bestz
            zerr_soln[1] = zerr
            nhi_soln[1] = bestnhi
            nhierr_soln[1] = nhierr
            dchi2_soln[1] = (chi2null-dchi2_soln[0]) - chi2dof
            coeff_soln[1] = coeff

            # fix first two solutions
            zchi2 = np.empty((len(zscan), len(nhiscan)))
            
            # Precompute fixed DLA transmission 
            fixed_trans_2 = _build_fixed_trans(wave, [(zdla_soln[0], nhi_soln[0]),
                                                      (zdla_soln[1], nhi_soln[1])])
            for iz in range(len(zscan)):
                trans_nhi = np.exp(-nhi_factors[:, None] * tau_grid[iz][None, :]) * fixed_trans_2
                for inhi in range(len(nhiscan)):
                    zchi2[iz, inhi] = _solve_DLA_with_transmission(
                        ivar, flux, model_flux, var_pipe, varlss, wave, searchmask, False, trans_nhi[inhi], 3)
                    
            # find minimum of coarse search
            bf = np.unravel_index(zchi2.argmin(), zchi2.shape)

            bestz, zerr, bestnhi, nhierr, chi2dof, fitwarning[2], coeff = refined_fit(bf[0], bf[1], zchi2[bf], 
                                                                                      [(zdla_soln[0], nhi_soln[0]), 
                                                                                       (zdla_soln[1], nhi_soln[1])])

            # TO DO : should subsequent DLAs be held to higher detection thresholds?
            if (((chi2null-dchi2_soln[0]-dchi2_soln[1]) - chi2dof) > constants.detection) & ((fitwarning[2] & DLAFLAG.ZBOUNDARY) == 0):

                # store best fit attributes for 3rd DLA
                zdla_soln[2] = bestz
                zerr_soln[2] = zerr
                nhi_soln[2] = bestnhi
                nhierr_soln[2] = nhierr
                dchi2_soln[2] = (chi2null-dchi2_soln[0]-dchi2_soln[1]) - chi2dof
                coeff_soln[2] = coeff

    return(zdla_soln, zerr_soln, nhi_soln, nhierr_soln, dchi2_soln, fitwarning, coeff_soln)

def PCA_reconstruction(coeff, eigvec):
    """
    reconstruct spectrum with PCA eigenspectra

    Arguments
    ---------
    coeff (array of floats) : coefficients on eigenspectra, length of nvec
    eigvec (2D array of floats) : array of eigenspectra, nvec x nlam

    Returns
    -------
    recon_spec (array of floats) : reconstructed spectrum, length of nlam

    """

    recon_spec = np.dot(coeff, eigvec)

    return( recon_spec )

def fit_spectrum(wave, flux, ivar, var_pipe, model_flux, varlss, searchmask):
    """
    fit full spectrum with intrinsic flux model

    Arguments
    ---------
    wave (array of floats) : observer frame wavelength in Angstroms
    flux (array of floats) : observed flux
    ivar (array of floats) : inverse variance on observed flux
    var_pipe (array of floats) : pipeline variance estimated from ivar
    model_flux (2D array of floats) : eigenspectra model with dimension nvec X nlam,
                                    resampled with observed wave array at quasar's redshift
    varlss (array of floats) : LSS variance 
    searchmask (array of boolean) : search window mask, dimensions nlam

    Returns
    -------
    coeff (array of floats) : coefficients on the eigenspectra model, dimension nvec
    chi2dof (float) : reduced chi2 of fit over DLA searchrange

    """

    # mask ivar = 0 entries to avoid divide by zero error
    mask = ivar != 0
    
    # assume only pipeline error contributes for first fit
    # lss contribution is model flux dependent - solved for below
    var_pipe_masked = var_pipe[mask]
    varlss_m = varlss[mask]
    wm = ivar[mask].copy()
    Mm = model_flux[:,mask]
    fm = flux[mask]
    dw_max = 1.0

    niter = 0
    while (dw_max > 10e-4) and (niter < 5):
        # linalg requires a and b are square matrices
        # also taking ivar weights into account
        b = Mm.dot( wm*fm )
        a = Mm.dot( (Mm * wm).T )

        coeff = np.linalg.solve(a,b)
        bestfit = np.dot(coeff,Mm)
        
        # adjust weights for LSS contribution
        nw_m = 1.0 / (var_pipe_masked + varlss_m * bestfit**2)
        dw_max = np.max(np.abs(wm - nw_m) / wm)
        wm = nw_m
        niter += 1

    # get the chi2 of fit just in the region we are searching for DLAs
    smask = mask&searchmask
    dof = np.sum(mask[searchmask]) - model_flux.shape[0]
    bestfit = coeff.dot(model_flux[:, smask])
    w = 1./(var_pipe[smask] + varlss[smask]*bestfit**2)
    chi2 = np.sum(w * (flux[smask] - bestfit)**2)

    return( coeff, chi2/dof)

