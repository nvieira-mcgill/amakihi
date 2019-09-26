#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:46:08 2019
@author: Nicholas Vieira
@psfphotom.py
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from subprocess import run
from timeit import default_timer as timer

from astropy.io import fits
from astropy import wcs
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats


def make_source_list(image_file, sigma=8.0, output=None):
    """
    Input: 
        - filename for **background-subtracted** image 
        - sigma to use as the threshold for source detection (optional; 
          default 8.0)
        - name for the output fits file (optional; default set below)
        
    Uses astrometry.net to detect sources in the image and write them to a 
    table for PSF photometry.
    
    Output: the source list 
    """
    if not(output):
        output = image_file.replace(".fits", "_sourcelist.fits")
        
    # overwrite existing files, do NOT do bg subtraction    
    options = "-O -b -o "+output+" -p "+str(sigma)
    # BANDAID FIX: can't locate image2xy at the moment 
    run("/usr/local/astrometry/bin/image2xy "+options+" "+image_file, 
        shell=True) 
    
    print("\nSource list written to file "+output)
    sources_data = fits.getdata(output)
    return sources_data


def PSF_photometry(image_file, source_file=None, source_sigma=8.0,
                   plot_ePSF=True, plot_resid=False, plot_corr=True,
                   plot_source_offsets=True, plot_field_offsets=False, 
                   source_lim=None, gaussian_blur_sigma=30.0, 
                   cat_num=None):
    """
    Inputs:
        - filename for **background-subtracted** image
        - a list of sources obtained with astrometry.net (optional; default 
          None, in which case a new list is obtained and used)
        - the sigma for source detection in astrometry.net (optional; default
          8.0)
        - whether to plot the derived ePSF (optional; default True)
        - whether to plot the residuals of the iterative PSF fitting (optional;
          default False)
        - whether to plot the instrumental magnitude versus the catalogue 
          magnitude when obtaining the zero point (optional; default True)
        - whether to plot the offsets between the image WCS and catalogue WCS 
          (optional; default True)
        - whether to plot the offsets across the field with a Gaussian blur to 
          visualize large-scale structure in the offsets if any is present 
          (optional; default False)
        - the limit on the number of sources to fit with the PSF (optional; 
          default None)
        - the sigma to use for the Gaussian blur, if relevant (optional; 
          default 30.0)
        - the Vizier identifier for the catalog to use when calibrating 
          magnitudes (optional; defaults set below)
          
    Write description later. 
    
    Output: a table of PSF-fit sources with calibrated magnitudes 
    """
    
    psf_sources = __fit_PSF(image_file, source_file, source_sigma, plot_ePSF, 
                            plot_resid, source_lim)
    
    if not(psf_sources == None): # if the PSF Was properly fit 
        psf_sources = __zero_point(image_file, source_file, psf_sources, 
                                   plot_corr, plot_source_offsets, 
                                   plot_field_offsets, gaussian_blur_sigma, 
                                   cat_num)
        
    return psf_sources


def __fit_PSF(image_file, source_file=None, source_sigma=8.0,
              plot_ePSF=True, plot_residuals=False, source_lim=None):
    """
    Input: the image file, the file containing a list of sources (optional; 
    default None; in which case a new source file is produced and used), 
    whether to plot the empirically determined effective Point-Spread Function 
    (ePSF) (optional; default True), whether to plot the residuals of the 
    iterative PSF fitting (optional; default False) and a limit on the no. of 
    sources to fit with the ePSF (optional; default no limit)
    
    Uses a **background-subtracted** image to obtain the ePSF and fits this 
    function to all of the sources previously detected by astrometry. Builds a 
    table containing the instrumental magnitudes and corresponding 
    uncertainties to be used in obtaining the zero point for PSF calibration.
    
    Output: None
    """
    
    from astropy.modeling.fitting import LevMarLSQFitter
    from photutils.psf import (BasicPSFPhotometry, DAOGroup)
    from astropy.nddata import NDData
    from photutils.psf import extract_stars
    from photutils import EPSFBuilder
        
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    
    if source_file:
        sources_data = fits.getdata(source_file)
    else: 
        print("\nSince no source list was provided, a list will be obtained "+
              "now...")
        sources_data = make_source_list(image_file, source_sigma)
    
    instrument = image_header["INSTRUME"]
    pixscale = image_header["PIXSCAL1"]
    ### SETUP
    # get source WCS coords
    x = np.array(sources_data['X'])
    y = np.array(sources_data['Y'])
    w = wcs.WCS(image_header)
    wcs_coords = np.array(w.all_pix2world(x,y,1))
    ra = Column(data=wcs_coords[0], name='ra')
    dec = Column(data=wcs_coords[1], name='dec')
    
    sources = Table() # build a table 
    sources['x_mean'] = sources_data['X'] # for BasicPSFPhotometry
    sources['y_mean'] = sources_data['Y']
    sources['x'] = sources_data['X'] # for EPSFBuilder 
    sources['y'] = sources_data['Y']
    sources.add_column(ra)
    sources.add_column(dec)
    sources['flux'] = sources_data['FLUX']  # already bkg-subtracted 
 
    # mask out edge sources:
    # a bounding circle for WIRCam, rectangle for MegaPrime
    xsize = image_data.shape[1]
    ysize = image_data.shape[0]    
    if "WIRCam" in instrument:
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((sources['x_mean']-xsize/2.0)**2 + 
                         (sources['y_mean']-ysize/2.0)**2)
        mask = dist_to_center <= rad_limit
        sources = sources[mask]
    else: 
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        mask = (sources['x_mean']>x_lims[0]) & (
                sources['x_mean']<x_lims[1]) & (
                sources['y_mean']>y_lims[0]) & (
                sources['y_mean']<y_lims[1])
        sources = sources[mask]
        

    ### EMPIRICALLY DETERMINED ePSF
    start = timer() # timing ePSF building time
    
    nddata = NDData(image_data)# NDData object
    stars = extract_stars(nddata, sources, size=25) # extract stars
    
    # use only the stars with fluxes between two percentiles
    stars_tab = Table() # temporary table 
    stars_col = Column(data=range(len(stars.all_stars)), name="stars")
    stars_tab["stars"] = stars_col # column of indices of each star
    fluxes = [s.flux for s in stars]
    fluxes_col = Column(data=fluxes, name="flux")
    stars_tab["flux"] = fluxes_col # column of fluxes
    
    # get percentiles
    per_low = np.percentile(fluxes, 80) # 80th percentile flux 
    per_high = np.percentile(fluxes, 90) # 90th percentile flux
    mask = (stars_tab["flux"] >= per_low) & (stars_tab["flux"] <= per_high)
    stars_tab = stars_tab[mask] # include only stars between these fluxes
    idx_stars = (stars_tab["stars"]).data # indices of these stars
    nstars_epsf = len(idx_stars) # no. of stars used in ePSF building
    print(str(nstars_epsf)+" stars used in building the ePSF\n")
    
    # update stars object and then build the ePSF
    # have to manually update all_stars AND _data attributes
    stars.all_stars = [stars[i] for i in idx_stars]
    stars._data = stars.all_stars
    epsf_builder = EPSFBuilder(oversampling=1, maxiters=10, # build it
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    
    # compute 90% radius of the ePSF to determine appropriate aperture size
    # for aperture photometry 
    epsf_data = epsf.data
    y, x = np.indices(epsf_data.shape)
    x_0 = epsf.data.shape[1]/2.0
    y_0 = epsf.data.shape[0]/2.0
    r = np.sqrt((x-x_0)**2 + (y-y_0)**2) # radial distances from source
    r = r.astype(np.int) # round to ints 
    
    # bin the data, obtain and normalize the radial profile 
    tbin = np.bincount(r.ravel(), epsf_data.ravel()) 
    norm = np.bincount(r.ravel())  
    profile = tbin/norm 
    
    # find radius at 10% of max 
    limit = np.min(profile[0:20]) 
    limit += 0.1*(np.max(profile[0:20])-np.min(profile[0:20]))
    for i in range(len(profile)):
        if profile[i] >= limit:
            continue
        else: # if below the 10% of max 
            epsf_radius = i # radius in pixels 
            break
    print("\nePSF 90% radius: "+str(epsf_radius)+" pix")
    
    end = timer() # timing 
    time_elaps = end-start
    print("Time required for ePSF building: %.2f s\n"%time_elaps)
    
    psf_model = epsf # set the model
    psf_model.x_0.fixed = True # fix centroids (known beforehand) 
    psf_model.y_0.fixed = True
    
    # initial guesses for centroids, fluxes
    pos = Table(names=['x_0', 'y_0','flux_0'], data=[sources['x_mean'],
               sources['y_mean'], sources['flux']]) 

    ### FIT THE ePSF TO ALL DETECTED SOURCES 
    start = timer() # timing the fit 
    
    # sources separated by less than this critical separation are grouped 
    # together when fitting the PSF via the DAOGROUP algorithm
    sigma_psf = 2.0 # 2 pix
    crit_sep = 2.0*sigma_psf*gaussian_sigma_to_fwhm  # twice the PSF FWHM
    daogroup = DAOGroup(crit_sep) 

    # an astropy fitter, does Levenberg-Marquardt least-squares fitting
    fitter_tool = LevMarLSQFitter()
    
    # if we have a limit on the number of sources
    if source_lim:
        try: 
            import random # pick a given no. of random sources 
            source_rows = random.choices(sources, k=source_lim)
            sources = Table(names=['x_mean', 'y_mean', 'x', 'y', 'ra', 
                                   'dec', 'flux'], rows=source_rows)
            pos = Table(names=['x_0', 'y_0','flux_0'], 
                        data=[sources['x_mean'], sources['y_mean'], 
                              sources['flux']])
            
            
        except IndexError:
            print("The input source limit exceeds the number of sources"+
                  " detected by astrometry, so no limit is imposed.\n")
    
    photometry = BasicPSFPhotometry(group_maker=daogroup,
                            bkg_estimator=None, # bg subtract already done
                            psf_model=psf_model,
                            fitter=fitter_tool,
                            fitshape=(11,11))
    
    result_tab = photometry(image=image_data, init_guesses=pos) # results
    residual_image = photometry.get_residual_image() # residuals of PSF fit
    
    end = timer() # timing 
    time_elaps = end - start
    print("Time required to fit ePSF to all sources: %.2f s\n"%time_elaps)
    
    # include previous WCS results
    result_tab.add_column(sources['ra'])
    result_tab.add_column(sources['dec'])
    
    # mask out negative flux_fit values in the results 
    mask_flux = (result_tab['flux_fit'] >= 0.0)
    psf_sources = result_tab[mask_flux]
    
    # compute magnitudes and their errors and add to the table
    # error = (2.5/(ln(10)*flux_fit))*flux_unc
    mag_fit = -2.5*np.log10(psf_sources['flux_fit']) # instrumental mags
    mag_fit.name = 'mag_fit'
    mag_unc = 2.5/(psf_sources['flux_fit']*np.log(10))
    mag_unc *= psf_sources['flux_unc']
    mag_unc.name = 'mag_unc' 
    psf_sources['mag_fit'] = mag_fit
    psf_sources['mag_unc'] = mag_unc
    
    # mask entries with large magnitude uncertainties 
    mask_unc = psf_sources['mag_unc'] < 0.4
    psf_sources = psf_sources[mask_unc]
    
    if plot_ePSF: # if we wish to see the ePSF
        plt.figure(figsize=(10,9))
        plt.imshow(epsf.data, origin='lower', aspect=1, cmap='magma',
                   interpolation="nearest")
        plt.xlabel("Pixels", fontsize=16)
        plt.ylabel("Pixels", fontsize=16)
        plt.title("Effective Point-Spread Function (1 pixel = "
                                                    +str(pixscale)+
                                                    '")', fontsize=16)
        plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        plt.rc("xtick",labelsize=16) # not working?
        plt.rc("ytick",labelsize=16)
        plt.savefig(image_file.replace(".fits", "_ePSF.png"), 
                    bbox_inches="tight")
        plt.close()
    
    if plot_residuals: # if we wish to see a plot of the residuals
        if "WIRCam" in instrument:
            plt.figure(figsize=(10,9))
        else:
            plt.figure(figsize=(12,14))
        ax = plt.subplot(projection=w)
        plt.imshow(residual_image, cmap='magma', aspect=1, 
                   interpolation='nearest', origin='lower')
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("PSF residuals", fontsize=16)
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        plt.savefig(image_file.replace(".fits", "_PSF_resid.png"),
                    bbox_inches="tight")
        plt.close()
    
    return psf_sources     
    
    
def __zero_point(image_file, source_file, psf_sources, plot_corr=True, 
                 plot_source_offsets=True, 
                 plot_field_offsets=False, gaussian_blur_sigma=30.0, 
                 cat_num=None):
    """
    Input: a bool indicating whether or not to plot the correlation with 
    linear fit (optional; default True), whether to plot the offsets in 
    RA and Dec of each catalog-matched source (optional; default True), 
    whether to show the overall offsets as an image with a Gaussian blur 
    to visualize large-scale structure (optional; default False), the 
    sigma to apply to the Gaussian filter (optional; default 30.0), and 
    a Vizier catalog number to choose which catalog to cross-match 
    (optional; defaults are PanStarrs 1, SDSS DR12, and 2MASS for relevant 
    filters)
    
    Uses astroquery and Vizier to query an online catalog for sources 
    which match those detected by astrometry. Computes the offset between
    the apparent and instrumental magnitudes of the queried sources for 
    photometric calibration. Computes the mean, median and standard 
    deviation.
    
    Output: None
    """
    
    from astroquery.vizier import Vizier

    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    
    #if source_file:
    #    sources_data = fits.getdata(source_file)
    
    filt = image_header["FILTER"][0]
    instrument = image_header["INSTRUME"]
    pixscale = image_header["PIXSCAL1"]
    date_isot = image_header["DATE"] # full ISOT time
    t_isot = Time(date_isot, format='isot', scale='utc')
    t_MJD = t_isot.mjd # convert ISOT in UTC to MJD
    
    # determine the catalog to compare to for photometry
    if cat_num: # if a Vizier catalog number is given 
        ref_cat = cat_num
        ref_cat_name = cat_num
    else:  
        if filt in ['g','r','i','z','Y']:
            zp_filter = (filt).lower() # lowercase needed for PS1
            ref_cat = "II/349/ps1" # PanStarrs 1
            ref_cat_name = "PS1" 
        elif filt == 'u':
            zp_filter = 'u' # closest option right now 
            ref_cat = "V/147" 
            ref_cat_name = "SDSS DR12"
        else: 
            zp_filter = filt[0] # Ks must be K for 2MASS 
            ref_cat = "II/246/out" # 2MASS
            ref_cat_name = "2MASS"
        
    w = wcs.WCS(image_header) # WCS object and coords of centre 
    xsize = image_data.shape[1]
    ysize = image_data.shape[0]          
    wcs_centre = np.array(w.all_pix2world(
            xsize/2.0, ysize/2.0, 1)) 

    ra_centre = wcs_centre[0]
    dec_centre = wcs_centre[1]
    radius = pixscale*np.max([xsize,ysize])/60.0 #arcmins
    minmag = 13.0 # magnitude minimum
    maxmag = 20.0 # magnitude maximum
    max_emag = 0.4 # maximum allowed error 
    nd = 5 # minimum no. of detections for a source (across all filters)
     
    # actual querying 
    # internet connection needed 
    print("\nQuerying Vizier %s (%s) "%(ref_cat, ref_cat_name)+
          "around RA %.4f, Dec %.4f "%(ra_centre, dec_centre)+
          "with a radius of %.4f arcmin"%radius)
    
    v = Vizier(columns=["*"], column_filters={
            zp_filter+"mag":str(minmag)+".."+str(maxmag),
            "e_"+zp_filter+"mag":"<"+str(max_emag),
            "Nd":">"+str(nd)}, row_limit=-1) # no row limit 
    Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                        unit = (u.deg, u.deg)), radius = str(radius)+'m', 
                        catalog=ref_cat, cache=False)

    if len(Q) == 0: # if no matches
        print("\nNo matches were found in the "+ref_cat_name+
              " catalog. The requested region may be in an unobserved"+
              " region of this catalog. Exiting.")
        return 
        
    
    # pixel coords of found sources
    cat_coords = w.all_world2pix(Q[0]['RAJ2000'], Q[0]['DEJ2000'], 1)
    
    # mask out edge sources
    # a bounding circle for WIRCam, rectangle for MegaPrime
    if "WIRCam" in instrument:
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((cat_coords[0]-xsize/2.0)**2 + 
                                 (cat_coords[1]-ysize/2.0)**2)
        mask = dist_to_center <= rad_limit
        good_cat_sources = Q[0][mask]
    else:
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        mask = (cat_coords[0] > x_lims[0]) & (
                cat_coords[0] < x_lims[1]) & (
                cat_coords[1] > y_lims[0]) & (
                cat_coords[1] < y_lims[1])
        good_cat_sources = Q[0][mask] 
    
    # cross-matching coords of sources found by astrometry
    source_coords = SkyCoord(ra=psf_sources['ra'], 
                             dec=psf_sources['dec'], 
                             frame='icrs', unit='degree')
    # and coords of valid sources in the queried catalog 
    cat_source_coords = SkyCoord(ra=good_cat_sources['RAJ2000'], 
                                 dec=good_cat_sources['DEJ2000'], 
                                 frame='icrs', unit='degree')
    
    # indices of matching sources (within 2*(pixel scale) of each other) 
    idx_image, idx_cat, d2d, d3d = cat_source_coords.search_around_sky(
            source_coords, 2*pixscale*u.arcsec)
    
    nmatches = len(idx_image) # store number of matches 
    sep_mean = np.mean(d2d.value*3600.0) # store mean separation in "
    print('\nFound %d sources in %s within 2.0 pix of'%(nmatches, 
                                                        ref_cat_name)+
          ' sources detected by astrometry, with average separation '+
          '%.3f" '%sep_mean)
    
    # get coords for sources which were matched
    source_matches = source_coords[idx_image]
    cat_matches = cat_source_coords[idx_cat]
    source_matches_ra = [i.ra.value for i in source_matches]
    cat_matches_ra = [i.ra.value for i in cat_matches]
    source_matches_dec = [i.dec.value for i in source_matches]
    cat_matches_dec = [i.dec.value for i in cat_matches]
    # compute offsets 
    ra_offsets = np.subtract(source_matches_ra, cat_matches_ra)*3600.0 # in arcsec
    dec_offsets = np.subtract(source_matches_dec, cat_matches_dec)*3600.0
    ra_offsets_mean = np.mean(ra_offsets)
    dec_offsets_mean = np.mean(dec_offsets)

    # plot the correlation
    if plot_corr:
        # fit a straight line to the correlation
        from scipy.optimize import curve_fit
        def f(x, m, b):
            return b + m*x
        
        xdata = good_cat_sources[zp_filter+'mag'][idx_cat] # catalog
        xdata = [float(x) for x in xdata]
        ydata = psf_sources['mag_fit'][idx_image] # instrumental 
        ydata = [float(y) for y in ydata]
        popt, pcov = curve_fit(f, xdata, ydata) # obtain fit
        m, b = popt # fit parameters
        perr = np.sqrt(np.diag(pcov))
        m_err, b_err = perr # errors on parameters 
        fitdata = [m*x + b for x in xdata] # plug fit into data 
        
        # plot correlation
        fig, ax = plt.subplots(figsize=(10,10))
        ax.errorbar(good_cat_sources[zp_filter+'mag'][idx_cat], 
                 psf_sources['mag_fit'][idx_image], 
                 psf_sources['mag_unc'][idx_image],
                 marker='.', mec="#fc5a50", mfc="#fc5a50", ls="",color='k', 
                 markersize=12, label="Data ["+filt+"]", zorder=1) 
        ax.plot(xdata, fitdata, color="blue", 
                 label=r"$y = mx + b $"+"\n"+r"$ m=$%.3f$\pm$%.3f, $b=$%.3f$\pm$%.3f"%(
                         m, m_err, b, b_err), zorder=2) # the linear fit 
        ax.set_xlabel("Catalog magnitude ["+ref_cat_name+"]", 
                      fontsize=15)
        ax.set_ylabel('Instrumental PSF-fit magnitude', fontsize=15)
        ax.set_title("PSF Photometry", fontsize=15)
        ax.legend(loc="upper left", fontsize=15, framealpha=0.5)
        plt.savefig(image_file.replace(".fits", "_PSF_photometry.png"),
                    bbox_inches="tight")
        plt.close()        
    
    # plot the RA, Dec offset for each matched source 
    if plot_source_offsets:             
        # plot
        plt.figure(figsize=(10,10))
        plt.plot(ra_offsets, dec_offsets, marker=".", linestyle="", 
                color="#ffa62b")
        plt.xlabel('RA (J2000) offset ["]', fontsize=15)
        plt.ylabel('Dec (J2000) offset ["]', fontsize=15)
        plt.title("Source offsets from %s catalog"%
                     ref_cat_name, fontsize=15)
        plt.axhline(0, color="k", linestyle="--", alpha=0.3) # (0,0)
        plt.axvline(0, color="k", linestyle="--", alpha=0.3)
        plt.plot(ra_offsets_mean, dec_offsets_mean, marker="X", 
                 color="blue", label = "Mean", linestyle="") # mean
        plt.legend(fontsize=15)
        plt.rc("xtick",labelsize=14)
        plt.rc("ytick",labelsize=14)
        plt.savefig(image_file.replace(".fits", 
                                       "_source_offsets_astrometry.png"),
                    bbox_inches="tight")        
        plt.close()
    
    # plot the overall offset across the field 
    if plot_field_offsets:
        from scipy.ndimage import gaussian_filter
        # add offsets to a 2d array
        offsets_image = np.zeros(image_data.shape)
        for i in range(len(d2d)): 
            x = psf_sources[idx_image][i]["x_0"]
            y = psf_sources[idx_image][i]["y_0"]
            intx, inty = int(x), int(y)
            offsets_image[inty, intx] = d2d[i].value*3600.0    
        # apply a gaussian blur to visualize large-scale structure
        blur_sigma = gaussian_blur_sigma
        offsets_image_gaussian = gaussian_filter(offsets_image, blur_sigma)
        offsets_image_gaussian *= np.max(offsets_image)
        offsets_image_gaussian *= np.max(offsets_image_gaussian)
        
        # plot
        if "WIRCam" in instrument:
            plt.figure(figsize=(10,9))
        else:
            plt.figure(figsize=(9,13))                
        ax = plt.subplot(projection=w)
        plt.imshow(offsets_image_gaussian, cmap="magma", 
                   interpolation="nearest", origin="lower")
        # textbox indicating the gaussian blur and mean separation
        textstr = r"Gaussian blur: $\sigma = %.1f$"%blur_sigma+"\n"
        textstr += r'$\overline{offset} = %.3f$"'%sep_mean
        box = dict(boxstyle="square", facecolor="white", alpha=0.8)
        if "WIRCam" in instrument:
            plt.text(0.6, 0.91, transform=ax.transAxes, s=textstr, 
                     bbox=box, fontsize=15)
        else:
            plt.text(0.44, 0.935, transform=ax.transAxes, s=textstr, 
                     bbox=box, fontsize=15)    
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("Field offsets from %s catalog"%ref_cat_name, 
                  fontsize=15)
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        plt.savefig(image_file.replace(".fits", 
                                       "_field_offsets_astrometry.png"),
                    bbox_inches="tight")        
        plt.close()
    
    # compute magnitude differences and zero point mean, median and error
    mag_offsets = ma.array(good_cat_sources[zp_filter+'mag'][idx_cat] - 
                  psf_sources['mag_fit'][idx_image])

    zp_mean, zp_med, zp_std = sigma_clipped_stats(mag_offsets)
    
    # add these to the header of the image file 
    f = fits.open(image_file, mode="update")
    f[0].header["ZP_MEAN"] = zp_mean
    f[0].header["ZP_MED"] = zp_med
    f[0].header["ZP_STD"] = zp_std
    f.close()
    
    # add a mag_calib and mag_calib_unc column to psf_sources
    mag_calib = psf_sources['mag_fit'] + zp_mean
    mag_calib.name = 'mag_calib'
    # propagate errors 
    mag_calib_unc = np.sqrt(psf_sources['mag_unc']**2 + zp_std**2)
    mag_calib_unc.name = 'mag_calib_unc'
    psf_sources['mag_calib'] = mag_calib
    psf_sources['mag_calib_unc'] = mag_calib_unc
    
    # add flag indicating if source is in a catalog and which catalog 
    in_cat = []
    for i in range(len(psf_sources)):
        if i in idx_image:
            in_cat.append(True)
        else:
            in_cat.append(False)
    in_cat_col = Column(data=in_cat, name="in_catalog")
    psf_sources["in "+ref_cat_name] = in_cat_col
    
    # add new columns 
    nstars = len(psf_sources)
    col_filt = Column([filt for i in range(nstars)], "filter",
                       dtype = np.dtype("U2"))
    col_mjd = Column([t_MJD for i in range(nstars)], "MJD")
    psf_sources["filter"] = col_filt
    psf_sources["MJD"] = col_mjd
    
    # compute magnitude differences between catalog and calibration 
    # diagnostic for quality of zero point determination 
    sources_mags = psf_sources[idx_image]["mag_calib"]
    cat_mags = good_cat_sources[idx_cat][zp_filter+"mag"]
    mag_diff_mean = np.mean(sources_mags - cat_mags)
    print("\nMean difference between calibrated magnitudes and "+
          ref_cat_name+" magnitudes = "+str(mag_diff_mean))
    
    return psf_sources