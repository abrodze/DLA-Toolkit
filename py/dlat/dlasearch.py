#!/usr/bin/env python

"""
Identify DLAs in data sample using template plus viogt fits
"""

import numpy as np
import os
import fitsio
from astropy.table import Table, vstack

import multiprocessing as mp
from functools import partial
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


def timestamp():
    """ 
    return current time in YYYY-MM-DD HH:MM:SS format
    """
    now = datetime.datetime.now()
    return(now.strftime("%Y-%m-%d %H:%M:%S"))


def dlasearch_hpx(healpix, survey, program, datapath, hpxcat, model, nproc):
    """
    Find the best fitting DLA profile(s) for spectra in hpx catalog

    Arguments
    ---------
    healpix (int) : N64 healpix
    survey (str) : e.g., main, sv1, sv2, etc.
    program (str) : e.g., bright, dark, etc.
    datapath (str) : path to coadd files
    hpxcat (table) : collection of spectra to search for DLAs, all belonging to
                     single healpix
    model (dict) : flux model dictionary containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys
    nproc (int) : number of multiprocessing processes for solve_DLA, default=64

    Returns
    -------
    fitresults (table) : attributes of detected DLAs
    """

    t0 = time.time()
   
    # read spectra from healpix
    coaddname = f'coadd-{survey}-{program}-{str(healpix)}.fits'
    coadd = os.path.join(datapath, str(healpix//100), str(healpix), coaddname)

    if os.path.exists(coadd):

        # set up pool
        if nproc > 1:
            pool = mp.Pool(nproc)
        else:
            pool = None

        fitresults = process_spectra_group(coadd, hpxcat, model, False, pool)

        if nproc > 1:
            pool.close()

    else:
        print(f'{timestamp()} - Warning: could not locate coadd file for healpix {healpix}')
        return()        


    t1 = time.time()
    total = np.round(t1-t0,2)
    print(f'{timestamp()} - Completed processing of {len(hpxcat)} spectra from healpix {healpix} in {total}s')

    return fitresults
    

def dlasearch_tile(tileid, datapath, tilecat, model, nproc):
    """
    Find the best fitting DLA profile(s) for spectra in hpx catalog

    Arguments
    ---------
    tileid (int) : tile no.
    datapath (str) : path to coadd files
    tilecat (table) : collection of spectra to search for DLAs, all belonging to
                     single tile
    model (dict) : flux model dictionary containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys
    nproc (int) : number of multiprocessing processes for solve_DLA, default=64

    Returns
    -------
    fitresults (table) : fit attributes for detected DLAs
    """

    t0 = time.time()

    # do tile based search, will need to save tileid in catalog since targetid is not unique to a tile
    # call process_spectra_group, append tileid and petal id to fitresults
    # loop over petal number

    # e.g. for petal in np.unique(tilecat['PETAL_LOC']):
    #           petcat = tilecat[tilecat['PETAL_LOC'] == petal]
    #           coadd = 'path to tile-petal coadd'
    #           #check if pool should be set up
    #           process_spectr_group(coadd, petcat, model, pool)
    #           # apeend tile and petal columns

    t1 = time.time()
    total = t1-t0
    print(f'{timestamp()} - Completed processing of {len(tilecat)} spectra from tile {tileid} in {total}s')

def dlasearch_mock(specfile, catalog, model, nproc):
    """
    function description

    Arguments
    ---------
    specfile (str) : path to mock spectra-16-X.fits file
    catalog (table) : catalog of spectra to search for DLAs
    model (dict) : flux model dictionary containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys
    nproc (int) : number of multiprocessing processes for solve_DLA, default=64

    Returns
    -------
    fitresults (table) : fit attributes for detected DLAs
    """
    
    t0 = time.time()

    if os.path.exists(specfile):
    
        # open spectra file fibermap only
        fm = desispec.io.read_fibermap(specfile)
    
        # pare catalog to match spectra file fibermap
        tidmask = np.in1d(catalog['TARGETID'], fm['TARGETID'])
        catalog = catalog[tidmask]
        if len(catalog) < 1:
            # no objects
            return()

        # set up pool
        if nproc > 1:
            pool = mp.Pool(nproc)
        else:
            pool = None

        fitresults = process_spectra_group(specfile, catalog, model, True, pool)

        if nproc > 1:
            pool.close()

    else:
        print(f'{timestamp()} - Warning: could not locate coadd file: {specfile}')
        return()


    t1 = time.time()
    total = np.round(t1-t0,2)
    print(f'{timestamp()} - Completed processing of {len(catalog)} spectra from {specfile} in {total}s')

    return fitresults


def process_spectra_group(coaddpath, catalog, model, is_mock=False, pool=None):
    """
    pre-process group of spectra in same file and run DLA searching tools

    Arguments
    ---------
    coaddpath (str) : path to file containing spectra
    catalog (table) : collection of spectra in file to search for DLAs
    model (dict) : flux model containing 'PCA_WAVE', 'PCA_COMP', and 'IGM' keys
    pool : shared mp pool

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

        try:
            idx = np.nonzero(specobj.fibermap['TARGETID']==tid)[0][0]
        except:
            print(f'{timestamp()} - Error: Targetid {tid} NOT FOUND on healpix {healpix}')
            continue

        flux = specobj.flux['brz'][idx]
        ivar = specobj.ivar['brz'][idx]
        wave_rf = wave / (1 + zqso)

        # only searching to rest frame 912 A
        fitmask = wave_rf > constants.search_minlam

        # limit our bestfit comparision w/ and w/o DLAs to search region of spectrum
        searchmask = np.ma.masked_inside(wave_rf[fitmask], constants.search_minlam, constants.search_maxlam).mask
        
        # apply mask to BAL features, if available
        if 'NCIV_450' in catalog.columns:
            nbal = catalog['NCIV_450'][entry]
            bal_locs = []
            for n in range(nbal):
                # Compute velocity ranges
                v_max = -catalog[entry]['VMAX_CIV_450'][n] / constants.c + 1.
                v_min = -catalog[entry]['VMIN_CIV_450'][n] / constants.c + 1.

                for line, lam in constants.bal_lines.items():
                    # Mask wavelengths within the velocity ranges
                    mask = np.ma.masked_inside(wave_rf, lam*v_min, lam*v_max).mask
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

        # model w/o DLAs
        coeff_null, chi2dof_null  = fit_spectrum(wave[fitmask], flux[fitmask], ivar[fitmask], fitmodel, varlss[fitmask], searchmask)

        # add up to 3 DLAs to fit, no detections have [z,nhi,dchi] = [-1,0,0]
        zdla, zerr, nhi, nhierr, dchi2, fitwarn, coeff_dla  = fit_spectrum_DLA(wave[fitmask], flux[fitmask], ivar[fitmask],
                                                            fitmodel, varlss[fitmask], searchmask, zqso, chi2dof_null, pool)

        # check for potential BAL contamination in solution
        # false positive should only come from Lya and NV - all other lines too weak
        if ('nbal' in locals()) & np.any(zdla != -1):
            lam_center_dla = constants.Lya_line*(1+zdla)
            for window in bal_locs:
                balflag = (lam_center_dla < window[0]) & (lam_center_dla > window[1])
                fitwarn[balflag] |= DLAFLAG.POTENTIAL_BAL

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
            
            # average signal to noise computation
            mask = np.logical_and(ivar != 0, np.ma.masked_inside(wave_rf, constants.bluesnr_min, constants.bluesnr_max).mask)
            bluesnr = np.mean((flux[mask]*np.sqrt(ivar[mask])))
            
            mask = np.logical_and(ivar != 0, np.ma.masked_inside(wave_rf, constants.redsnr_min, constants.redsnr_max).mask)
            redsnr = np.mean((flux[mask]*np.sqrt(ivar[mask])))

            # check for insufficent coverage and set to -1
            if np.isinf(bluesnr) or np.isnan(bluesnr):
                bluesnr = -1
            if np.isinf(redsnr) or np.isnan(redsnr):
                redsnr = -1

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
    reconstruct spectrum with eigenspectra + DLA model

    Arguments
    ---------
    x (array of floats) : coeff on eigenspectra, length of nvec
    dla_params (array of floats) : (zdla, log10(NHI)) per DLA, length N_DLA X 2
    eigvec (2D array of floats) : array of eigenspectra, nvec x nlam
    lam_obs (array of floats) : observed wave array in angstroms

    Returns
    -------
    recon_spec (array of floats) : reconstructed spectrum, length of nlam

    """

    # determine how many DLAs are in dla_param arguments
    NDLA = int(len(dla_params)/2)

    # account for mean transmission of each DLA
    z_idx, nhi_idx = 0, 1
    DLA_transmission = np.ones(len(eigvec[0]))
    for dla in range(NDLA):
        DLA_transmission *= dlaprofile.dla_profile(lam_obs, dla_params[z_idx], dla_params[nhi_idx])
        z_idx += 2
        nhi_idx += 2

    recon_spec = np.dot(x, DLA_transmission*eigvec)

    return(recon_spec)

def parabola(x, a, b, c):
    """
     z = z0 + ((x-x0)/xerr)^2
    """
    return( a + ((x-b)/c)**2 )


def _solve_DLA(ivar, flux, model_flux, varlss, wave, searchmask, return_coeff, dlaparams):
    """
    fit spectrum with eigenspectra given a fixed DLA profile

    Arguments
    ---------
    ivar (array of floats) : inverse variance on observed flux
    flux (array of floats) : observed flux
    model_flux (2D array of floats) : eigenspectra model with dimension nvec X nlam
    varlss (array of floats) : LSS variance
    wave (array of floats) : observed wavelength (Angstroms)
    searchmask (array of bool) : DLA search window mask, dimensions nlam
    return_coeff (bool) : return coefficients of bestfit
    dlaparams (array of floats) :  (zdla, log10(NHI)) per DLA, length N_DLA X 2

    Returns
    -------
    coeff (array of floats, optional) : coefficients on eigenspectra, dimensions nvec
    chi2dof (float) : reduced chi2 of best fit over DLA search range defined
                      by searchmask argument
    """

    # determine how many DLAs are in dla_param arguments
    NDLA = int(len(dlaparams)/2)

    # account for mean transmission of each DLA
    z_idx, nhi_idx = 0, 1
    DLA_transmission = np.ones(len(model_flux[0]))
    for dla in range(NDLA):
        DLA_transmission *= dlaprofile.dla_profile(wave, dlaparams[z_idx], dlaparams[nhi_idx])
        z_idx += 2
        nhi_idx += 2

    M = DLA_transmission*model_flux

    # mask ivar = 0 entries to avoid divide by zero error
    mask = ivar != 0
    # assume only pipeline error contributes for first fit
    # lss contribution is model flux dependent - solved for below
    w = ivar
    nw = np.zeros(len(w))
    dw = np.ones(len(w))

    niter = 0
    while (np.max(dw) > 10e-4) and (niter < 5):
        # linalg requires a and b are square matrices
        # also taking ivar weights into account
        b = M[:,mask].dot( w[mask]*flux[mask] )
        a = M[:,mask].dot( (M[:,mask] * w[mask]).T )

        coeff = np.linalg.solve(a,b)
        bestfit = np.dot(coeff,M)

        # adjust weights for LSS contribution
        nw[mask] = 1./(ivar[mask]**-1 + varlss[mask]*bestfit[mask]**2)
        dw[mask] = np.abs(w-nw)[mask]/w[mask]
        dw[~mask] = 0.
        w = nw
        niter += 1

    # get the chi2 of fit just in the region we are searching for DLAs
    dof = np.sum(mask[searchmask]) - M.shape[0] - float(NDLA)
    bestfit = np.dot(coeff,M)[mask&searchmask]
    w = 1./(ivar[mask&searchmask]**-1  + varlss[mask&searchmask]*bestfit**2)
    chi2 = np.sum(w * (flux[mask&searchmask] - bestfit)**2)

    if return_coeff:
        return(coeff, chi2/dof)
    else:
        return(chi2/dof)


def fit_spectrum_DLA(wave, flux, ivar, model_flux, varlss, searchmask, zqso, chi2null, pool=None):
    """
    fit spectrum with eigenspectra model with up to 3 free DLA (zdla, nhi) profiles

    Arguments
    ---------
    wave (array of floats) : observed wavelength (Angstroms)
    flux (array of floats) : observed flux
    ivar (array of floats) : inverse variance on observed flux
    model_flux (2D array of floats) : eigenspectra model with dimension nvec X nlam
    varlss (array of floats) : LSS variance
    searchmask (array of bool) : DLA search window mask, dimensions nlam
    zqso (float) : quasar redshift
    chi2null (float) : reference chi2 of DLA-free model
    pool : shared mp pool

    Returns
    -------
    zdla_soln (array floats) : best fit DLA redshifts
    zerr_soln (array of floats) : error on zdla estimated with parabola fit
    nhi_soln (array of floats) : best fit log10 DLA column density (cm^-2)
    nhierr_soln (array of floats) : error on nhi estimated with parabola fit
    dchi_soln (array of floats) : difference in chi2 from best fit to previous best fit (null fit for 
                        1st DLA) over range defined by searchmask argument
    fitwarning (bitmask) : flags on solution
    coeff_soln (array of floats) : coefficients on eigenspectra, dimensions nvec
    """

    # no detection will give results of [z,zerr,nhi,nhierr,dchi2] = [-1,0,0,0,0]
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
    nhiscan = np.arange(constants.nhimin, constants.nhimax, 0.2) # must allow subDLA NHI to avoid FPs

    # for multiprocessing
    DLA_chi2_matrix = partial(_solve_DLA, ivar, flux, model_flux, varlss, wave, searchmask, False)

    def refined_fit(z_bf_idx, nhi_bf_idx, ini_chi2, solved_dla):

        # initial flag with no warning
        fitwarn = 0

        # check if fit solution relaxed to redshift boundary
        if z_bf_idx < (len(zscan)-1):
            # define refined search grid, allowing for slight buffer for parabola fitting if at z \approx zqso
            zrefine = np.arange(zscan[np.max([z_bf_idx-2, 0])], zqso, 0.0025)
        else: 
            # define refined search grid
            zrefine = np.arange(zscan[np.max([z_bf_idx-2, 0])], zscan[np.min([z_bf_idx+2, len(zscan)-1])]+0.001, 0.0025)
        nhirefine = np.arange(constants.nhimin,constants.nhimax,0.05)

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
            if solved_dla is not None:
                test_params = [(z, last_bestnhi, *solved_dla) for z in zrefine]
            else:
                test_params = [(z, last_bestnhi) for z in zrefine]

            if pool is None:
                zzchi2 = []
                for tp in test_params:
                    zzchi2.append(DLA_chi2_matrix(tp))
            else:
                zzchi2 = pool.map(DLA_chi2_matrix, test_params)
                
            # fit minima of refined search with parabola to find best fit z+zerr
            iniguess = np.array([ini_chi2, last_bestz, 0.1])
            minpos = np.argmin(zzchi2)
            ini_idx = np.max([minpos-2, 0])
            fin_idx = np.min([minpos+3,len(zzchi2)])
            try:
                popt, pcov = curve_fit(parabola, np.array(test_params)[:,0][ini_idx:fin_idx], zzchi2[ini_idx:fin_idx],
                                  p0=iniguess, bounds=zbounds)
                bestz = popt[1]
                zerr = popt[2]
            except(ValueError,OptimizeWarning,RuntimeError):
                fitwarn |= DLAFLAG.BAD_ZFIT
                bestz = last_bestz
                zerr = np.inf

            # refined solve for nhi, assuming z value
            if solved_dla is not None:
                test_params = [(bestz, nhi, *solved_dla) for nhi in nhirefine]
            else:
                test_params = [(bestz, nhi) for nhi in nhirefine]

            if pool is None:
                zzchi2 = []
                for tp in test_params:
                    zzchi2.append(DLA_chi2_matrix(tp))
            else:
                zzchi2 = pool.map(DLA_chi2_matrix, test_params)
                
            # fit minima of refined search with parabola to find best fit nhi+nhierr
            iniguess = np.array([ini_chi2, last_bestnhi, 1.])
            minpos = np.argmin(zzchi2)
            ini_idx = np.max([minpos-2, 0])
            fin_idx = np.min([minpos+3,len(zzchi2)])
            try:
                popt, pcov = curve_fit(parabola, np.array(test_params)[:,1][ini_idx:fin_idx], zzchi2[ini_idx:fin_idx],
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
                
        if solved_dla is not None:
            coeff, chi2dof = _solve_DLA(ivar, flux, model_flux, varlss, wave, searchmask, True, 
                                        (*solved_dla, bestz, bestnhi))
        else:
            coeff, chi2dof = _solve_DLA(ivar, flux, model_flux, varlss, wave, searchmask, True, (bestz, bestnhi))
        
        return(bestz, zerr, bestnhi, nhierr, chi2dof, fitwarn, coeff)


    # coarse solve
    test_params = []
    for z in zscan:
        for nhi in nhiscan:
            test_params.append((z,nhi))

    if pool is None:
        zchi2 = []
        for tp in test_params:
            zchi2.append(DLA_chi2_matrix(tp))
    else:
        zchi2 = pool.map(DLA_chi2_matrix, test_params)

    # reshape to correspond 1-1 with [z, nhi]
    zchi2 = np.array(zchi2).reshape(len(zscan), len(nhiscan))

    # find minimum of course search
    bf = np.unravel_index(zchi2.argmin(), zchi2.shape)

    bestz, zerr, bestnhi, nhierr, chi2dof, fitwarning[0], coeff = refined_fit(bf[0], bf[1], zchi2[bf], None)
   

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
        # TO DO : how should first solution be allowed to vary? vary only under certain criteria?
        test_params = []
        for z in zscan:
            for nhi in nhiscan:
                test_params.append((zdla_soln[0], nhi_soln[0], z, nhi))
        
        if pool is None:
            zchi2 = []
            for tp in test_params:
                zchi2.append(DLA_chi2_matrix(tp))
        else:
            zchi2 = pool.map(DLA_chi2_matrix, test_params) 
            
        # reshape to correspond 1-1 with [z, nhi]
        zchi2 = np.array(zchi2).reshape(len(zscan), len(nhiscan))
        # find minimum of course search
        bf = np.unravel_index(zchi2.argmin(), zchi2.shape)

        bestz, zerr, bestnhi, nhierr, chi2dof, fitwarning[1], coeff = refined_fit(bf[0], bf[1], zchi2[bf], (zdla_soln[0], nhi_soln[0]))

        # TO DO : should subsequent DLAs be held to higher detection thresholds?
        if (((chi2null-dchi2_soln[0]) - chi2dof) > constants.detection) & ((fitwarning[1] & DLAFLAG.ZBOUNDARY) == 0):

            # store best fit attributes for 2nd DLA
            zdla_soln[1] = bestz
            zerr_soln[1] = zerr
            nhi_soln[1] = bestnhi
            nhierr_soln[1] = nhierr
            dchi2_soln[1] = (chi2null-dchi2_soln[0]) - chi2dof
            coeff_soln[1] = coeff

            # fix first solution
            # TO DO : how should first two solutions be allowed to vary? vary only under certain criteria like wings overlap?
            test_params = []
            for z in zscan:
                for nhi in nhiscan:
                    test_params.append((zdla_soln[0], nhi_soln[0], zdla_soln[1], nhi_soln[1], z, nhi))

            if pool is None:
                zchi2 = []
                for tp in test_params:
                    zchi2.append(DLA_chi2_matrix(tp))
            else:
                zchi2 = pool.map(DLA_chi2_matrix, test_params)

            # reshape to correspond 1-1 with [z, nhi]
            zchi2 = np.array(zchi2).reshape(len(zscan), len(nhiscan))
            # find minimum of course search
            bf = np.unravel_index(zchi2.argmin(), zchi2.shape)

            bestz, zerr, bestnhi, nhierr, chi2dof, fitwarning[2], coeff = refined_fit(bf[0], bf[1], zchi2[bf], (zdla_soln[0], nhi_soln[0], zdla_soln[1], nhi_soln[1]))

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

def fit_spectrum(wave, flux, ivar, model_flux, varlss, searchmask):
    """
    fit full spectrum with intrinsic flux model

    Arguments
    ---------
    wave (array of floats) : observer frame wavelength in Angstroms
    flux (array of floats) : observed flux
    ivar (array of floats) : inverse variance on observed flux
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
    w = ivar
    nw = np.zeros(len(w))
    dw = np.ones(len(w))

    niter = 0
    while (np.max(dw) > 10e-4) and (niter < 5):
        # linalg requires a and b are square matrices
        # also taking ivar weights into account
        b = model_flux[:,mask].dot( w[mask]*flux[mask] )
        a = model_flux[:,mask].dot( (model_flux[:,mask] * w[mask]).T )

        coeff = np.linalg.solve(a,b)
        bestfit = np.dot(coeff,model_flux)
        
        # adjust weights for LSS contribution
        nw[mask] = 1./(ivar[mask]**-1 + varlss[mask]*bestfit[mask]**2)
        dw[mask] = np.abs(w-nw)[mask]/w[mask]
        dw[~mask] = 0.
        w = nw
        niter += 1

    # get the chi2 of fit just in the region we are searching for DLAs
    dof = np.sum(mask[searchmask]) - model_flux.shape[0]
    bestfit = np.dot(coeff,model_flux)[mask&searchmask]
    w = 1./(ivar[mask&searchmask]**-1 + varlss[mask&searchmask]*bestfit**2)
    chi2 = np.sum(w * (flux[mask&searchmask] - bestfit)**2)

    return( coeff, chi2/dof)

