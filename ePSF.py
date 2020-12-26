#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:59:35 2020
@author: Nicholas Vieira
@ePSF.py
"""

# misc
from subprocess import run
import numpy as np
import re

# scipy
from scipy.ndimage import zoom

# astropy
from astropy.io import fits
from astropy import wcs
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import convolve_fft, Gaussian2DKernel, Moffat2DKernel
from astropy.table import Table, Column
from photutils import make_source_mask, detect_sources, source_properties

# amakihi
from plotting import __plot_ePSF, __plot_convolve_self

## for speedy FFTs
#import pyfftw
#import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs
#pyfftw.interfaces.cache.enable()

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
#### EFFECTIVE POINT-SPREAD FUNCTION ##########################################

def ePSF_FWHM(epsf_data, verbose=False):
    """Estimate the full width at half max (FWHM) of a 2D ePSF. 
    
    Arguments
    ---------
    epsf_data : str OR np.ndarray
        Filename for a fits file containing the ePSF data **OR** the 2D array
        representing the ePSF directly
    verbose : bool, optional
        Whether to be verbose (default Faulse)
    
    Returns
    -------
    float
        FWHM of the input ePSF
    
    """
    
    if (type(epsf_data) == str): # if a filename, open it 
        epsf_data = fits.getdata(epsf_data)
    
    # enlarge the ePSF by a factor of 100 
    epsf_data = zoom(epsf_data, 10)
    
    # compute FWHM of ePSF 
    y, x = np.indices(epsf_data.shape)
    x_0 = epsf_data.shape[1]/2.0
    y_0 = epsf_data.shape[0]/2.0
    r = np.sqrt((x-x_0)**2 + (y-y_0)**2) # radial distances from source
    r = r.astype(np.int) # round to ints 
    
    # bin the data, obtain and normalize the radial profile 
    tbin = np.bincount(r.ravel(), epsf_data.ravel()) 
    norm = np.bincount(r.ravel())  
    profile = tbin/norm 
    
    # find radius at FWHM
    limit = np.min(profile) 
    limit += 0.5*(np.max(profile)-np.min(profile)) # limit: half of maximum
    for i in range(len(profile)):
        if profile[i] >= limit:
            continue
        else: # if below 50% of max 
            epsf_radius = i # radius in pixels 
            break

    if verbose:
        print(f"ePSF FWHM = {epsf_radius*2.0/10.0} pix")
    return epsf_radius*2.0/10.0


## using image segmentation ###################################################

def build_ePSF_imsegm(image_file, mask_file=None, nstars=40,                              
                      thresh_sigma=5.0, pixelmin=20, etamax=1.4, areamax=500,             
                      cutout=35, 
                      write=True, output=None, 
                      plot=False, output_plot=None, 
                      verbose=False):
    """Build the effective Point-Spread Function using a sample of stars from
    some image acquired via image segmentation.

    Arguments
    ---------
    image_file : str
        Filename for a **background-subtracted** image
    mask_file : str, optional
        Filename for a mask file (default None)
    nstars : int, optional
        *Maximum* number of stars to use in building the ePSF (default 40;
        set to None to impose no limit)
    thresh_sigma : float, optional
        Sigma threshold for source detection with image segmentation (default
        5.0)
    pixelmin : float, optional
        *Minimum* pixel area of an isophote to be considered a good source for 
        building the ePSF (default 20)
    etamax : float, optional
        *Maximum* allowed elongation for an isophote to be considered a good 
        source for building the ePSF (default 1.4)
    areamax : float, optional
        *Maximum* allowed area (in square pixels) for an isophote to be 
        considered a good source for building the ePSF (default 500)
    cutout : int, optional
        Cutout size around each star in pixels (default 35; must be **odd**; 
        rounded **down** if even)
    write : bool, optional
        Whether to write the ePSF to a new fits file (default True)
    output : str, optional
        Name for the output ePSF data fits file (default
        `image_file.replace(".fits", "_ePSF.fits")`)
    plot : bool, optional
        Whether to plot the newly-built ePSF (default False)
    output_plot : str, optional
        Name for the output figure (default 
        `image_file.replace(".fits", "_ePSF.png")`)
    verbose : bool, optional
        Whether to be verbose (default False)

    Returns
    -------
    np.ndarray
        The ePSF data in a 2D array
    
    Notes
    -----
    Uses image segmentation via `photutils` to obtain a list of sources in the 
    image with their x, y coordinates, flux, and background at their 
    location. Then uses `EPSFBuilder` to empirically obtain the ePSF of these 
    stars. Optionally writes and/or plots the obtained ePSF.
    
    **The ePSF obtained here should not be used in convolutions.** Instead, it 
    can serve as a tool for estimating the seeing of an image. 
    """
    
    # ignore annoying warnings from photutils
    from astropy.utils.exceptions import AstropyWarning
    warnings.simplefilter('ignore', category=AstropyWarning)
    
    # imports
    from astropy.nddata import NDData
    from photutils.psf import extract_stars
    from photutils import EPSFBuilder
    
    # load in data 
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file) 
    try:
        instrument = image_header["INSTRUME"]
    except KeyError:
        instrument = "Unknown"
        
    ## source detection
    # add mask to image_data
    image_data = np.ma.masked_where(image_data==0.0, image_data) 
    
    # build an actual mask
    mask = (image_data==0)
    if mask_file:
        mask = np.logical_or(mask, fits.getdata(mask_file))

    # set detection standard deviation
    try:
        std = image_header["BKGSTD"] # header written by bkgsub function
    except KeyError:
        # make crude source mask, get standard deviation of background
        source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                       dilate_size=15, mask=mask)
        final_mask = np.logical_or(mask, source_mask)
        std = np.std(np.ma.masked_where(final_mask, image_data))
    
    # use the segmentation image to get the source properties 
    segm = detect_sources(image_data, thresh_sigma*std, npixels=pixelmin,
                          mask=mask) 
    cat = source_properties(image_data, segm, mask=mask)

    ## get the catalog and coordinate/fluxes for sources, do some filtering
    try:
        tbl = cat.to_table()
    except ValueError:
        print("SourceCatalog contains no sources. Exiting.")
        return
    
    # restrict elongation and area to obtain only unsaturated stars 
    tbl = tbl[(tbl["elongation"] <= etamax)]
    tbl = tbl[(tbl["area"].value <= areamax)]
    # build a table
    sources = Table() # build a table 
    sources['x'] = tbl['xcentroid'] # for EPSFBuilder 
    sources['y'] = tbl['ycentroid']
    sources['flux'] = tbl['source_sum'].data/tbl["area"].data   
    sources.sort("flux")
    sources.reverse()    
    # restrict number of stars (if requested)
    if nstars: sources = sources[:min(nstars, len(sources))]

    ## get WCS coords for all sources 
    w = wcs.WCS(image_header)
    sources["ra"], sources["dec"] = w.all_pix2world(sources["x"],
                                                    sources["y"], 1)    
    ## mask out edge sources: 
    # a bounding circle for WIRCam, rectangle for MegaPrime
    xsize = image_data.shape[1]
    ysize = image_data.shape[0]
    if "WIRCam" in instrument: # bounding circle
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((sources['x']-xsize/2.0)**2 + 
                                 (sources['y']-ysize/2.0)**2)
        mask = dist_to_center <= rad_limit
        sources = sources[mask]
    else: # rectangle
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        mask = (sources['x']>x_lims[0]) & (sources['x']<x_lims[1]) & (
                sources['y']>y_lims[0]) & (sources['y']<y_lims[1])
        sources = sources[mask]
        
    ## empirically obtain the effective Point Spread Function (ePSF)  
    nddata = NDData(image_data) # NDData object
    if mask_file: # supply a mask if needed 
        nddata.mask = fits.getdata(mask_file)
    if cutout%2 == 0: # if cutout even, subtract 1
        cutout -= 1
    stars = extract_stars(nddata, sources, size=cutout) # extract stars

    ## build the ePSF
    nstars_epsf = len(stars.all_stars) # no. of stars used in ePSF building
    
    if nstars_epsf == 0:
        print("\nNo valid sources were found to build the ePSF with the given"+
              " conditions. Exiting.")
        return    
    if verbose:
        print(f"{nstars_epsf} stars used in building the ePSF")
        
    epsf_builder = EPSFBuilder(oversampling=1, maxiters=7, # build it
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    epsf_data = epsf.data       
    
    if write: # write, if desired
        epsf_hdu = fits.PrimaryHDU(data=epsf_data)
        if not(output):
            output = image_file.replace(".fits", "_ePSF.fits")
            
        epsf_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    if plot: # plot, if desired
        if not(output_plot): # set output name if not given
            output_plot = image_file.replace(".fits", "_ePSF.png")
        __plot_ePSF(epsf_data=epsf_data, 
                    output=output_plot)
    
    return epsf_data


## using astrometry.net #######################################################

def build_ePSF_astrometry(image_file, mask_file=None, nstars=40, 
                          image_source_file=None, 
                          astrom_sigma=5.0, psf_sigma=5.0, alim=10000, 
                          lowper=0.6, highper=0.9, 
                          keep=False,
                          cutout=35, 
                          write=True, output=None, 
                          plot=False, output_plot=None, 
                          verbose=False):
    """Build the effective Point-Spread Function using a sample of stars from
    some image acquired via the `image2xy` tool of `astrometry.net`.

    Arguments
    ---------
    image_file : str
        Filename for a **background-subtracted** image
    mask_file : str, optional
        Filename for a mask file (default None)
    nstars : int, optional
        *Maximum* number of stars to use in building the ePSF (default 40;
        set to None to impose no limit)
    image_source_file : str, optional
        Filename for a `.xy.fits` file containing detected sources with their 
        pixel coordinates and **background-subtracted** flux (default None, in 
        which case a new such file is produced)
    astrom_sigma : float, optional
        Detection significance when using `image2xy` in `astrometry.net` to 
        find sources (default 5.0)
    psf_sigma : float, optional
        Sigma of the approximate Gaussian PSF of the images (default 5.0)
    alim : int, optional
        *Maximum* allowed source area in square pixels for `astrometry.net`, 
        above which sources will be deblended (default 10000)
    lowper, highper : float, optional
        Lower and upper flux percentiles (as a fraction between 0 and 1) such
        that sources outside the corresponding flux range will be excluded from 
        ePSF building (default 0.6 and 0.9, respectively)
    keep : bool, optional
        Whether to keep the source list file (`.xy.fits` files; default False)
    cutout : int, optional
        Cutout size around each star in pixels (default 35; must be **odd**; 
        rounded **down** if even)
    write : bool, optional
        Whether to write the ePSF to a new fits file (default True)
    output : str, optional
        Name for the output ePSF data fits file (default
        `image_file.replace(".fits", "_ePSF.fits")`)
    plot : bool, optional
        Whether to plot the newly-built ePSF (default False)
    output_plot : str, optional
        Name for the output figure (default 
        `image_file.replace(".fits", "_ePSF.png")`)
    verbose : bool, optional
        Whether to be verbose (default False)

    Returns
    -------
    np.ndarray
        The ePSF data in a 2D array
    
    Notes
    -----
    Uses `astrometry.net` to obtain a list of sources in the image with their 
    x, y coordinates, flux, and background at their location. (If a list of 
    sources has already been obtained `solve-field` or `image2xy`, this can 
    be input). Finally, selects stars between the `lowper` th and `highper`  
    percentile fluxes.
    
    Finally, uses `EPSFBuilder` to empirically obtain the ePSF of these stars. 
    Optionally writes and/or plots the obtained ePSF.
    
    **The ePSF obtained here should not be used in convolutions.** Instead, it 
    can serve as a tool for estimating the seeing of an image. 
    """
    
    # ignore annoying warnings from photutils
    from astropy.utils.exceptions import AstropyWarning
    warnings.simplefilter('ignore', category=AstropyWarning)
    
    from astropy.nddata import NDData
    from photutils.psf import extract_stars
    from photutils import EPSFBuilder
    
    # load in data 
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file) 
    try:
        instrument = image_header["INSTRUME"]
    except KeyError:
        instrument = "Unknown"
        
    ### source detection
    ## use pre-existing file obtained by astrometry.net, if supplied
    if image_source_file:
        image_sources = np.logical_not(fits.getdata(image_source_file))
        
    ## use astrometry.net to find the sources 
    # -b --> no background-subtraction
    # -O --> overwrite
    # -p <astrom_sigma> --> signficance
    # -w <psf_sigma> --> estimated PSF sigma 
    # -m <alim> --> max object size for deblending is <alim>    
    else:
        options = f" -b -O -p {astrom_sigma} -w {psf_sigma} -m {alim}"  
        run(f"image2xy {options} {image_file}", shell=True) 
        image_sources_file = image_file.replace(".fits", ".xy.fits")
        image_sources = fits.getdata(image_sources_file)
        if not(keep):
            run(f"rm {image_sources_file}", shell=True) # file is not needed
        print(f'\n{len(image_sources)} stars at >{astrom_sigma} sigma found '+
              f'in image {re.sub(".*/", "", image_file)} with astrometry.net')  

        sources = Table() # build a table 
        sources['x'] = image_sources['X'] # for EPSFBuilder 
        sources['y'] = image_sources['Y']
        sources['flux'] = image_sources['FLUX']

        if nstars:
            sources = sources[:min(nstars, len(sources))]

    ## get WCS coords for all sources 
    w = wcs.WCS(image_header)
    sources["ra"], sources["dec"] = w.all_pix2world(sources["x"],
                                                    sources["y"], 1)    
    ## mask out edge sources: 
    # a bounding circle for WIRCam, rectangle for MegaPrime
    xsize = image_data.shape[1]
    ysize = image_data.shape[0]
    if "WIRCam" in instrument:
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((sources['x']-xsize/2.0)**2 + 
                                 (sources['y']-ysize/2.0)**2)
        mask = dist_to_center <= rad_limit
        sources = sources[mask]
    else: 
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        mask = (sources['x']>x_lims[0]) & (sources['x']<x_lims[1]) & (
                sources['y']>y_lims[0]) & (sources['y']<y_lims[1])
        sources = sources[mask]
        
    ## empirically obtain the effective Point Spread Function (ePSF)  
    nddata = NDData(image_data) # NDData object
    if mask_file: # supply a mask if needed 
        nddata.mask = fits.getdata(mask_file)
    if cutout%2 == 0: # if cutout even, subtract 1
        cutout -= 1
    stars = extract_stars(nddata, sources, size=cutout) # extract stars
    
    ## use only the stars with fluxes between two percentiles if using 
    ## astrometry.net
    if image_source_file:
        stars_tab = Table() # temporary table 
        stars_col = Column(data=range(len(stars.all_stars)), name="stars")
        stars_tab["stars"] = stars_col # column of indices of each star
        fluxes = [s.flux for s in stars]
        fluxes_col = Column(data=fluxes, name="flux")
        stars_tab["flux"] = fluxes_col # column of fluxes
        
        # get percentiles
        per_low = np.percentile(fluxes, lowper*100) # lower percentile flux 
        per_high = np.percentile(fluxes, highper*100) # upper percentile flux
        mask = (stars_tab["flux"] >= per_low) & (stars_tab["flux"] <= per_high)
        stars_tab = stars_tab[mask] # include only stars between these fluxes
        idx_stars = (stars_tab["stars"]).data # indices of these stars
        
        # update stars object 
        # have to manually update all_stars AND _data attributes
        stars.all_stars = [stars[i] for i in idx_stars]
        stars._data = stars.all_stars

    ## build the ePSF
    nstars_epsf = len(stars.all_stars) # no. of stars used in ePSF building
    
    if nstars_epsf == 0:
        print("\nNo valid sources were found to build the ePSF with the given"+
              " conditions. Exiting.")
        return    
    if verbose:
        print(f"{nstars_epsf} stars used in building the ePSF")
        
    epsf_builder = EPSFBuilder(oversampling=1, maxiters=7, # build it
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    epsf_data = epsf.data       
    
    if write: # write, if desired
        epsf_hdu = fits.PrimaryHDU(data=epsf_data)
        if not(output):
            output = image_file.replace(".fits", "_ePSF.fits")
            
        epsf_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    if plot: # plot, if desired
        if not(output_plot): # set output name if not given
            output_plot = image_file.replace(".fits", "_ePSF.png")
        __plot_ePSF(epsf_data=epsf_data, 
                    output=output_plot)
    
    return epsf_data

###############################################################################
### SELF-CONVOLUTION ##########################################################

def convolve_self(image_file, mask_file=None,                                
                  thresh_sigma=5.0, pixelmin=20, etamax=1.4, areamax=500, 
                  cutout=35, 
                  psf_fwhm_max=7.0, 
                  kernel='gauss', alpha=1.5,
                  write=True, output=None, 
                  plot=False, output_plot=None,
                  verbose=True):
    """Using a sample of stars detected by image segmentation, construct the 
    ePSF of an image (using `build_ePSF_imsegm()`), fit the ePSF with a 
    Gaussian or Moffat kernel, and then convolve the image with this kernel.
    
    Arguments
    ---------
    image_file : str
        Filename for a **background-subtracted** image
    mask_file : str, optional
        Filename for a mask file (default None)
    nstars : int, optional
        *Maximum* number of stars to use in building the ePSF (default 40;
        set to None to impose no limit)
    thresh_sigma : float, optional
        Sigma threshold for source detection with image segmentation (default
        5.0)
    pixelmin : float, optional
        *Minimum* pixel area of an isophote to be considered a good source for 
        building the ePSF (default 20)
    etamax : float, optional
        *Maximum* allowed elongation for an isophote to be considered a good 
        source for building the ePSF (default 1.4)
    areamax : float, optional
        *Maximum* allowed area (in square pixels) for an isophote to be 
        considered a good source for building the ePSF (default 500)
    cutout : int, optional
        Cutout size around each star in pixels (default 35; must be **odd**; 
        rounded **down** if even)
    psf_fwhm_max : float, optional
        *Maximum* allowed ePSF FWHM in pixels such that function will exit if 
        the FWHM exceeds this value (default 7.0)
    kernel : {"gauss", "moffat"}, optional
        Type of kernel to fit to the ePSF, where this kernel will then be 
        convolved with the input image (default "gauss")
    alpha : float, optional
        Alpha parameter (power) for te Moffat kernel (default 1.5)
    write : bool, optional
        Whether to write the newly-convolved image to a new fits file (default 
        True)
    output : str, optional
        Name for the output convolved image (default
        `image_file.replace(".fits", "_selfconv.fits")`)
    plot : bool, optional
        Whether to plot the newly-convolved image (default False)
    output_plot : str, optional
        Name for the output figure (default 
        `image_file.replace(".fits", "_selfconv.png")`)
    verbose : bool, optional
        Whether to be verbose (default True)
    
    Returns
    -------
    astropy.io.fits.PrimaryHDU
        Fits HDU for the newly-convolved image
    
    Notes
    -----
    Finds the ePSF of the input image and finds the FWHM (and thus sigma) of 
    the ePSF. Then builds a Gaussian with this sigma and convolves the input
    image with the Gaussian, **or** builds a Moffat distribution with half the 
    ePSF FWHM as the core width and convolves the image with the Moffat. 
    
    Effectively tries to increase the size of the image's ePSF by sqrt(2). 
    Useful as a preparation for image differencing when sigma_template > 
    sigma_science.
    
    Output: PrimaryHDU for the science image convolved with the kernel
    """
    
    ## ePSF
    # get the ePSF for the image 
    epsf_data = build_ePSF_imsegm(image_file, mask_file, 
                                  thresh_sigma=thresh_sigma, 
                                  pixelmin=pixelmin, 
                                  etamax=etamax, areamax=areamax, 
                                  cutout=cutout, 
                                  write=False, verbose=verbose)     
    
    # get the sigma of the ePSF, assuming it is Gaussian
    fwhm = ePSF_FWHM(epsf_data, verbose)
    
    # if too large, exit 
    if fwhm > psf_fwhm_max:
        print(f"\nePSF_FWHM = {fwhm} > ePSF_FWHM_max = {psf_fwhm_max}")
        print("--> Exiting.")
        return
    
    # if size OK, build a kernel 
    if kernel.lower() == 'gauss': # gaussian with sigma = sigma of ePSF
        sigma = fwhm*gaussian_fwhm_to_sigma
        print("\nBuilding Gaussian kernel...")
        print(f"sigma = {sigma:.2f} pix")
        kern = Gaussian2DKernel(sigma, x_size=511, y_size=511)
    elif kernel.lower() == 'moffat':
        gamma = 0.5*fwhm/(2.0*((2**(1.0/alpha) - 1.0)**0.5))
        print("\nBuilding Moffat kernel...")
        print(f"gamma [core width] = {gamma:.2f} pix")
        kern = Moffat2DKernel(gamma, alpha=alpha, factor=1, x_size=127, 
                              y_size=127)
    else:
        raise ValueError("\nInvalid kernel selected; options are 'gauss' or "+
                         f"'moffat' but argument supplied was {kernel}")
    
    ## convolving
    # convolve the image data with its own ePSF, approximated as a Gaussian or
    # as a Moffat distribution
    hdr = fits.getheader(image_file)
    data = fits.getdata(image_file)
    mask = np.logical_or(data==0, np.isnan(data))
    if mask_file: 
        mask = np.logical_or(fits.getdata(mask_file), mask)
    data = np.ma.masked_where(mask, data) # mask the bad pixels 
    print("\nConvolving...")
    conv = convolve_fft(data, kernel=kern, boundary='fill', fill_value=0,
                        nan_treatment='fill', preserve_nan=True,
                        fft_pad=False, psf_pad=False)
    conv[mask] = 0 # set masked values to zero again
    
    ## plotting and/or writing    
    # plot, if desired
    if plot: 
        topfile = re.sub(".*/", "", image_file) # set title for plot
        title = topfile.replace(".fits","")
        title = r"$\mathtt{"+title.replace("_","\_")+"}$"
        title = f"{title} convolved with self"
        
        if not(output_plot): # set name for output plot if not given
            output_plot = image_file.replace(".fits", "_selfconv.png")
            
        __plot_convolve_self(conv=conv, mask=mask, hdr=hdr, 
                             title=title, output=output_plot)
    # final new HDU
    hdu = fits.PrimaryHDU(data=conv, header=hdr)  
    # write, if desired
    if write:     
        if not(output): # set name for output file if not given
            output = image_file.replace(".fits", "_selfconv.fits")          
        hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return hdu