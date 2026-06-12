#!/usr/bin/env python

"""
script for running DLA Toolkit on DESI spectra
"""

from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d
import healpy
import fitsio

import os
import datetime
import time
import argparse
import importlib
import multiprocessing as mp

from dlat import dlasearch
from dlat import constants

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

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""search for DLAs in DESI quasar spectra""")

    parser.add_argument('-q','--qsocat', type = str, default = None, required = True,
                        help='path to quasar catalog')

    parser.add_argument('-r', '--release', type = str, default = None, required = True,
                        help='DESI redux version (e.g. iron)')

    parser.add_argument('-p', '--program', type = str, default = 'dark', required = False,
                        help='observing program, default is dark')

    parser.add_argument('-s', '--survey', type = str, default = 'main', required = False,
                        help='survey, default is main')
    
    parser.add_argument('--mocks', default = False, required = False, action = 'store_true',
                        help='is this a mock catalog? Default is False')
    
    parser.add_argument('--mockdir', type = str, default = None, required = False,
                        help='path to mock directory')
    
    parser.add_argument('-m', '--model', type = str, default = None , required = False,
                        help='path to intrinsic flux model, defaults to v1.1 Redrock QSO_HIZ')

    parser.add_argument('--varlss', type = str, default = None, required = False,
                        help='path to LSS variance input files, defaults to jura LSS variance')

    parser.add_argument('--balmask', default = False, required = False, action='store_true',
                        help='should BALs be masked using AI_CIV? Default is False but recommended setting is True')

    parser.add_argument('-o', '--outdir', type = str, default = None, required = True,
                        help='output directory for DLA catalog')

    parser.add_argument('--outfile', type = str, default = None, required = True,
                        help='name for output FITS file containing DLA catalog')

    parser.add_argument('-n', '--nproc', type = int, default=128, required=False, 
                        help='number of multiprocressing processes to use, default is 128')

    parser.add_argument('--dir_type', type = str, default = None, required = True,
                        help='explicit directory tree type (healpix or uniqpix)')
    
    parser.add_argument('--datapath', type = str, default = None, required = False,
                        help='explicit directory path (release/survey/program will NOT be used to construct data path)')

    parser.add_argument('--hpx_nside', type= int, default = None, required = False,
                        help='healpix nside; if dir_type=HEALPIX then HEALPIX is calculated from RA/DEC using nside=hpx_nside if provided')

    if options is None:
        args  = parser.parse_args()
    else:
        args  = parser.parse_args(options)

    return args


def main(args=None):

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    # Check is catalog exists
    if not os.path.isfile(args.qsocat):
        print(f"{timestamp()} - Critical Error: {args.qsocat} does not exist")
        exit(1)
    # if catalog is healpix based, we must have program & survey
    if not(args.mocks):
        print(f"{timestamp()} - Warning: expecting healpix catalog for redux={args.release}, survey={args.survey}, program={args.program}; confirm this matches the catalog provided!")
    # confirm bal masking choice
    if not(args.balmask):
        print(f"{timestamp()} - Warning: BALs will not be masked! The only good reason to do this is if you do not have a BAL catalog, set --balmask to turn on masking.")
    # check the model file exits
    if (args.model is not None) and not(os.path.isfile(args.model)):
        print(f"{timestamp()} - Critical Error: cannot not find flux model file, looking for {args.model}")
        exit(1)
    # check if LSS file exits
    if (args.varlss is not None) and not(os.path.isfile(args.varlss)):
        print(f"{timestamp()} - Critical Error: cannot find LSS variance file, looking for {args.varlss}")
        exit(1)
    # check if mock data
    if args.mocks and (args.mockdir is None):
        print(f"{timestamp()} - Critical Error: mocks argument set to true but no mock data path provided")
        exit(1)
    elif args.mocks and not(os.path.exists(args.mockdir)):
        print(f"{timestamp()} - Critical Error: {args.mockdir} does not exist")
        exit(1)
    # check if output directory already exists
    os.makedirs(args.outdir, exist_ok=True)
    # remove "fits" if present
    if args.outfile.endswith(".fits"):
        args.outfile = args.outfile.removesuffix(".fits")

    tini = time.time()

    # read in quasar catalog and intrinsic flux model
    if args.mocks:
        catalog = read_mock_catalog(args.qsocat, args.balmask, args.mockdir)
    else:
        # set data path and expected healpix keyword 
        args.dir_type = args.dir_type.upper()
        # boolean to use healpix from catalog or to calcuate from nside
        calculate_healpix_num = False
        if (args.dir_type == 'HPXPIXEL') or (args.dir_type == 'HEALPIXEL') or (args.dir_type == 'HEALPIX'):
            pix_keyword = 'HPXPIXEL'
            datapath = f'/global/cfs/cdirs/desi/spectro/redux/{args.release}/healpix/{args.survey}/{args.program}'
            if args.hpx_nside is not None:
                # need to explicitly map RA and DEC to nside healpix
                print(f"{timestamp()} - HEALPIX number will be determined from RA/DEC using nside={args.hpx_nside}")
                calculate_healpix_num = True
            else:
                f(f"{timestamp()} - HEALPIX numbers will be pulled from {pix_keyword} column in catalog to find coadd files")
        elif args.dir_type == 'UNIQPIX':
            pix_keyword = args.dir_type
            datapath = f'/global/cfs/cdirs/desi/spectro/redux/{args.release}/spectra/{args.survey}/{args.program}'
        else:
            print(f"{timestamp()} - Critical Error: data directory tree argument is not understood. \
                                    Must be HPXPIXEL, HEALPIX, HEALPIXEL, or UNIQPIX.")
            exit(1)

        # if explicit data path was passed, overwrite assume data path structure (used for QSO-only Lya redux)
        if args.datapath is not None:
            datapath = args.datapath
            
        catalog = read_catalog(args.qsocat, pix_keyword, args.balmask)

        if calculate_healpix_num:
            # redetermine healpix number and overwrite pix_keyword column
            print(f"{timestamp()} - calculating HEALPIX numbers with nside={args.hpx_nside}")
            catalog[pix_keyword] = healpy.ang2pix(args.hpx_nside,
                                catalog['TARGET_RA'], catalog['TARGET_DEC'],
                                lonlat=True, nest=True)
            print(f"{timestamp()} - completed calculation of healpix numbers")
    
    if args.model is None:
        # locate default model file in repo
        args.model = str(importlib.resources.files('dlat').joinpath('models/QSO-HIZv1.1_RR.npz'))

    model = np.load(args.model)
    fluxmodel = dict()
    fluxmodel['PCA_COMP'] = model['PCA_COMP']
    fluxmodel['PCA_WAVE'] = 10**model['LOGLAM']
    fluxmodel['IGM'] = model['IGM'][0]

    # add lss variance info to dictionary for forest fitting
    if args.varlss is None:
        # locate default file in repo
        args.varlss = str(importlib.resources.files('dlat').joinpath('lss_variance/jura-var-lss.fits'))
    fluxmodel = read_varlss(args.varlss, fluxmodel)
    
    if args.nproc == 1:
        print(f"{timestamp()} - Warning: nproc set to {args.nproc}, multiprocessing disabled")

    if not(args.mocks):

        # create uniqpix/healpix list
        unihpx, counts = np.unique(catalog[pix_keyword], return_counts=True)
        # sort the catalog by expected workload per healpix
        unihpx = unihpx[np.argsort(-counts)]
        
        # process in batches to allow intermediate caching
        if len(unihpx) < 3:
            groups = 1
            group_step = len(unihpx)
        else:
            groups = 3
            group_step = int(np.ceil(len(unihpx)/groups))

        # track whether file was written
        group_exists = np.zeros(groups, dtype=bool)
   
        if args.nproc == 1:
            
            for g in np.arange(0, groups):
                
                # check if group has already been processed and confirm it is not empty
                outfile = os.path.join(args.outdir,f'{args.outfile}-chunk{g}-tmp.fits')
                group_results = []
                iini = g*group_step
                ifin = min((g+1)*group_step, len(unihpx))

                if os.path.exists(outfile):
                    group_exists[g] = True
                    continue
                if unihpx[iini:ifin].shape[0] == 0:
                    continue
                    
                for hpx in unihpx[iini:ifin]:
                    group_results.append(dlasearch.dlasearch_hpx(hpx, args.survey, args.program, datapath,
                                                                 catalog[catalog[pix_keyword] == hpx], 
                                                                 fluxmodel))

                # remove extra column from hpx with no detections
                group_results = vstack(group_results)
                if 'col0' in group_results.columns:
                    group_results.remove_column('col0')

                # write tmp file
                if len(group_results) != 0:
                    group_results.write(outfile)
                    group_exists[g] = True

        if args.nproc > 1: 

            # check if nproc is under-subscribed and adjust if necessary
            if group_step < args.nproc:
                args.nproc = group_step
            
            arguments = [ {"healpix": hpx , \
                       "survey": args.survey, \
                       "program": args.program, \
                       "datapath": datapath, \
                       "pixcat": catalog[catalog[pix_keyword] == hpx], \
                       "model": fluxmodel, \
                       } for ih,hpx in enumerate(unihpx)]
        
            with mp.Pool(args.nproc) as pool:

                for g in np.arange(0, groups):
            
                    # check if group has already been processed and confirm it is not empty
                    outfile = os.path.join(args.outdir,f'{args.outfile}-chunk{g}-tmp.fits')
                    iini = g*group_step
                    ifin = min((g+1)*group_step, len(unihpx))

                    if os.path.exists(outfile):
                        group_exists[g] = True
                        continue
                    if unihpx[iini:ifin].shape[0] == 0:
                        continue

                    results = pool.map(_dlasearchhpx, arguments[iini:ifin])
                    
                    # remove extra column from hpx with no detections
                    results = vstack(results)
                    if 'col0' in results.columns:
                        results.remove_column('col0')
                    
                    # write tmp file
                    if len(results) != 0:
                        results.write(outfile)
                        group_exists[g] = True

        # combine all batches into final catalog
        fin_results = None
        for g in np.arange(groups):
            chunkfile = os.path.join(args.outdir, f'{args.outfile}-chunk{g}-tmp.fits')
            if not group_exists[g]:
                continue  # group was empty, nothing to combine
            if not os.path.exists(chunkfile):
                print(f'{timestamp()} - Warning: temporary file for group {g} does not exist')
                continue
            gresults = Table(fitsio.read(chunkfile, ext=1))
            fin_results = gresults if fin_results is None else vstack([fin_results, gresults])

        if fin_results is None:
            print(f'{timestamp()} - Warning: no DLA detections found across any group; no catalog written')
        else:
    
            # split into good and flagged catalogs
            good_mask = fin_results['DLAFLAG'] == 0
            fin_results_good = fin_results[good_mask]
            fin_results_flagged = fin_results[~good_mask]
    
            # set extension name
            fin_results_good.meta['EXTNAME'] = 'DLACAT'
            fin_results_flagged.meta['EXTNAME'] = 'DLACAT'
    
            # remove DLAFLAG column from good catalogs
            fin_results_good.remove_column('DLAFLAG')
    
            outfile = f"{os.path.join(args.outdir, args.outfile)}-good.fits"
            if os.path.isfile(outfile):
                print(f'{timestamp()} - Warning: {args.outfile}-good.fits already exists in {args.outdir}, overwriting')
            fin_results_good.write(outfile, overwrite=True)
            print(f'{timestamp()} - wrote DLA catalog of good detections to {outfile}')
    
            outfile = f"{os.path.join(args.outdir, args.outfile)}-flagged.fits"
            if os.path.isfile(outfile):
                print(f'{timestamp()} - Warning: {args.outfile}-flagged.fits already exists in {args.outdir}, overwriting')
            fin_results_flagged.write(outfile, overwrite=True)
            print(f'{timestamp()} - wrote DLA catalog of flagged detections to {outfile}')
    
            # remove temporary files
            for g in range(groups):
                chunkfile = os.path.join(args.outdir, f'{args.outfile}-chunk{g}-tmp.fits')
                if not group_exists[g]:
                    continue  # nothing to clean up
                if os.path.exists(chunkfile):
                    os.remove(chunkfile)
                else:
                    print(f'{timestamp()} - Warning: temporary file for group {g} does not exist')

    elif args.mocks:
        
        datapath = f'{args.mockdir}/spectra-16'

        mock_healpix_list = healpy.ang2pix(16, catalog['RA'], catalog['DEC'] , lonlat=True, nest=True)
        unihpx, counts = np.unique(mock_healpix_list, return_counts=True)
        # sort the catalog by expected workload per healpix
        unihpx = unihpx[np.argsort(-counts)]
        speclist = [f'{datapath}/{u//100}/{u}/spectra-16-{u}.fits' for u in unihpx]
        
        # process in batches to allow intermediate caching
        if len(speclist) < 3:
            groups = 1
            group_step = len(speclist)
        else:
            groups = 3
            group_step = int(np.ceil(len(speclist)/groups))

        # track whether file was written
        group_exists = np.zeros(groups, dtype=bool)
        
        if args.nproc == 1:
            for g in np.arange(0, groups):
            
                # check if group has already been processed and confirm it is not empty
                outfile = os.path.join(args.outdir,f'{args.outfile}-mockcat-chunk{g}-tmp.fits')
                group_results = []
                iini = g*group_step
                ifin = min((g+1)*group_step, len(speclist))

                if os.path.exists(outfile):
                    group_exists[g] = True
                    continue
                if len(speclist[iini:ifin]) == 0:
                    continue

                for i,specfile in enumerate(speclist[iini:ifin]):
                    u = unihpx[iini:ifin][g*i+i]
                    cat_mask = mock_healpix_list == u
                    group_results.append(dlasearch.dlasearch_mock(specfile, catalog[cat_mask], fluxmodel))

                # remove extra column from spec groups with no detections
                group_results = vstack(group_results)
                if 'col0' in group_results.columns:
                    group_results.remove_column('col0')

                # write tmp file
                if len(group_results) != 0:
                    group_results.write(outfile)
                    group_exists[g] = True

        if args.nproc > 1:

            # check if nproc is under-subscribed and adjust if necessary
            if group_step < args.nproc:
                args.nproc = group_step
                
            arguments = [ {"specfile": specfile , \
                       "catalog": catalog[mock_healpix_list == unihpx[ih]] , \
                       "model": fluxmodel, \
                       } for ih,specfile in enumerate(speclist) ]
            
            with mp.Pool(args.nproc) as pool:

                for g in np.arange(0, groups):
                    
                    # check if group has already been processed and confirm it is not empty
                    outfile = os.path.join(args.outdir,f'{args.outfile}-mockcat-chunk{g}-tmp.fits')
                    iini = g*group_step
                    ifin = min((g+1)*group_step, len(speclist))

                    if os.path.exists(outfile):
                        group_exists[g] = True
                        continue
                    if len(speclist[iini:ifin]) == 0:
                        continue

                    results = list(pool.map(_dlasearchmock, arguments[iini:ifin]))

                    # remove extra column from hpx with no detections
                    results = vstack(results)
                    if 'col0' in results.columns:
                        results.remove_column('col0')

                    # write tmp file
                    if len(results) != 0:
                        results.write(outfile)                            
                        group_exists[g] = True

        # combine all batches into final catalog
        fin_results = None
        for g in np.arange(groups):
            chunkfile = os.path.join(args.outdir, f'{args.outfile}-mockcat-chunk{g}-tmp.fits')
            if not group_exists[g]:
                continue  # group was empty, nothing to combine
            if not os.path.exists(chunkfile):
                print(f'{timestamp()} - Warning: temporary file for group {g} does not exist')
                continue
            gresults = Table(fitsio.read(chunkfile, ext=1))
            fin_results = gresults if fin_results is None else vstack([fin_results, gresults])

        if fin_results is None:
            print(f'{timestamp()} - Warning: no DLA detections found across any group; no catalog written')
        else:

            # split into good and flagged catalogs
            good_mask = fin_results['DLAFLAG'] == 0
            fin_results_good = fin_results[good_mask]
            fin_results_flagged = fin_results[~good_mask]
            
            # remove DLAFLAG column from good catalogs
            fin_results_good.remove_column('DLAFLAG')
    
            # set extension name
            fin_results_good.meta['EXTNAME'] = 'DLACAT'
            fin_results_flagged.meta['EXTNAME'] = 'DLACAT'
    
            outfile = f"{os.path.join(args.outdir, args.outfile)}-good.fits"
            if os.path.isfile(outfile):
                print(f'{timestamp()} - Warning: {args.outfile}-good.fits already exists in {args.outdir}, overwriting')
            fin_results_good.write(outfile, overwrite=True)
            print(f'{timestamp()} - wrote DLA catalog of good detections to {outfile}')
    
            outfile = f"{os.path.join(args.outdir, args.outfile)}-flagged.fits"
            if os.path.isfile(outfile):
                print(f'{timestamp()} - Warning: {args.outfile}-flagged.fits already exists in {args.outdir}, overwriting')
            fin_results_flagged.write(outfile, overwrite=True)
            print(f'{timestamp()} - wrote DLA catalog of flagged detections to {outfile}')
    
            # remove temporary files
            print(f'{timestamp()} - removing tmp files')
            for g in range(groups):
                chunkfile = os.path.join(args.outdir, f'{args.outfile}-mockcat-chunk{g}-tmp.fits')
                if not group_exists[g]:
                    continue  # nothing to clean up
                if os.path.exists(chunkfile):
                    os.remove(chunkfile)
                else:
                    print(f'{timestamp()} - Warning: temporary file for group {g} does not exist')
                    
    tfin = time.time()
    total_time = tfin-tini

    print(f'{timestamp()} - SUCCESS')
    print(f'total run time: {np.round(total_time/60,1)} minutes')

def read_catalog(qsocat, pix_keyword, balmask):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    pix_keyword (str) : name of healpix column; must be either UNIQ or HPXPIXEL
    balmask (bool) : should BAL attributes from baltools be read in?
    
    Returns
    -------
    table of relevant attributes for quasars in default redshift range set by constants.py

    """

    if not(np.any(pix_keyword == np.array(['UNIQPIX','HPXPIXEL']))):
        print(f'{timestamp()} - Critical Error: unexpected healpix keyword passed to read_catalog')
        exit(1)

    if balmask:
        try:
            # read the following columns from qsocat
            cols = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z', pix_keyword, 'AI_CIV', 'NCIV_450', 'VMIN_CIV_450', 'VMAX_CIV_450']
            catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
        except:
            print(f'{timestamp()} - Critical Error: cannot find {cols} in quasar catalog')
            exit(1)
    else:
        # read the following columns from qsocat
        cols = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z', pix_keyword]
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    print(f'{timestamp()} - Successfully read quasar catalog: {qsocat}')

    # Apply redshift cuts
    zmask = (catalog['Z'] > constants.zmin_qso) & (catalog['Z'] < constants.zmax_qso)
    print(f'{timestamp()} - objects in catalog: {len(catalog)} ')
    print(f'{timestamp()} - restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain')

    catalog = catalog[zmask]

    return( catalog )

def read_mock_catalog(qsocat, balmask, mockpath):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes be read in?
    mockpath (str) : path to mock data

    Returns
    -------
    table of relevant attributes for z>2 quasars

    """
    # read the following columns from qsocat
    cols = ['TARGETID', 'RA', 'DEC', 'Z']
    try:
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
    except:
        cols = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z']
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
    print(f'{timestamp()} - Successfully read mock quasar catalog: {qsocat}')

    # Apply redshift cuts
    zmask = (catalog['Z'] > constants.zmin_qso) & (catalog['Z'] < constants.zmax_qso)
    print(f'{timestamp()} - objects in catalog: {len(catalog)} ')
    print(f'{timestamp()} - restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain')

    catalog = catalog[zmask]

    if balmask:

        # define columns to read
        cols = ['TARGETID', 'AI_CIV', 'NCIV_450', 'VMIN_CIV_450', 'VMAX_CIV_450']
        balcat_path = os.path.join(mockpath,'bal_cat.fits')
        balcat = Table(fitsio.read(balcat_path, ext=1, columns=cols))

        # add columns to catalog
        ai = np.full(len(catalog), 0.)
        nciv = np.full(len(catalog), 0)
        vmin = np.full((len(catalog), balcat['VMIN_CIV_450'].shape[1]), -1.)
        vmax = np.full((len(catalog), balcat['VMIN_CIV_450'].shape[1]), -1.)
            
        # sort BAL catalog by TARGETID for searchsorted
        bal_tids = np.asarray(balcat['TARGETID'])
        order = np.argsort(bal_tids)
        bal_tids_sorted = bal_tids[order]
    
        cat_tids = np.asarray(catalog['TARGETID'])
        # locate each catalog TID in the sorted BAL TID array
        idx = np.searchsorted(bal_tids_sorted, cat_tids)
        # guard against out-of-range and confirm exact match
        in_range = idx < len(bal_tids_sorted)
        valid = np.zeros(len(cat_tids), dtype=bool)
        valid[in_range] = bal_tids_sorted[idx[in_range]] == cat_tids[in_range]
    
        match_rows = order[idx[valid]]  # rows in original balcat
        ai[valid] = balcat['AI_CIV'][match_rows]
        nciv[valid] = balcat['NCIV_450'][match_rows]
        vmin[valid] = balcat['VMIN_CIV_450'][match_rows]
        vmax[valid] = balcat['VMAX_CIV_450'][match_rows]
    
        catalog.add_columns([ai, nciv, vmin, vmax],
                            names=['AI_CIV','NCIV_450','VMIN_CIV_450','VMAX_CIV_450'])
        
    
    return( catalog )

def read_varlss(varlss_path, fluxmodel):
    """
    add sigma_lss function to flux model dictionary

    Arguments
    ---------
    varlss_path (str) : path to file containing LSS variance function
    fluxmodel (dict) : dictionary for PCA flux model

    Returns
    -------
    fluxmodel (dict) : flux model dictionary with var_lss entry appended

    """

    print(f'{timestamp()} - reading sigma_lss function from {varlss_path}')

    # read in data
    varlss_lya = Table(fitsio.read(varlss_path, ext='VAR_FUNC_LYA'))
    varlss_lyb = Table(fitsio.read(varlss_path, ext='VAR_FUNC_LYB'))

    # map lambda_obs -> var_lss
    fluxmodel['VAR_FUNC_LYA'] = interp1d(10**varlss_lya['LOGLAM'], varlss_lya['VAR_LSS'], bounds_error=False, fill_value=0.)
    fluxmodel['VAR_FUNC_LYB'] = interp1d(10**varlss_lyb['LOGLAM'], varlss_lyb['VAR_LSS'], bounds_error=False, fill_value=0.)

    return(fluxmodel)

# for parallelization over hpx
def _dlasearchhpx(arguments):
    return( dlasearch.dlasearch_hpx(**arguments) )
   
# for parallelization over mock spectra files
def _dlasearchmock(arguments):
    return( dlasearch.dlasearch_mock(**arguments) )

if __name__ == "__main__":
    main()

