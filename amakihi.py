#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:58:03 2019
@author: Nicholas Vieira
@amakihi.py 

**This script contains a library of functions for:**

- Image registration (i.e. image alignment)
- Masking out bad pixels (including saturated stars)

It also acts as a crude python wrapper for hotpants 
(https://github.com/acbecker/hotpants).

**Sections:**

- Image registraton (alignment)
- Mask building (boxmask, saturation mask)
- ePSF building 
- Image differencing with hotpants
- Transient detection, triplets 
- Miscellaneous plotting


**Essential dependencies:**

- ``astropy`` (used extensively)
- ``photutils`` (used extensively)
- ``astrometry.net`` (used extensively, but can be ignored in favour of 
  source detection with `photutils`' `image_segmentation` instead)
- ``image_registration`` (used in `image_align_morph()`)
- ``hotpants`` (essential for image subtraction via `hotpants`, duh)

    
**Non-essential dependencies:**

- ``astroscrappy`` (OPTIONAL cosmic ray rejection during background removal)


**Important:** This software makes use of a slightly modified version of the 
`astroalign` software developed by Martin Beroiz and the TOROS Dev Team 
(https://github.com/toros-astro/astroalign) in the form of my own script 
`astroalign_mod.py`. I claim absolutely no ownership of this software. All
modifications are described in that script.
"""

# misc
import os
import sys
from subprocess import run, PIPE, CalledProcessError
import numpy as np
import re

# scipy
from scipy.ndimage import zoom, binary_dilation, gaussian_filter

# astropy
from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.stats import (sigma_clipped_stats, SigmaClip, 
                           gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm)
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft, Gaussian2DKernel, Moffat2DKernel
from astropy.table import Table, Column
from photutils import Background2D, MedianBackground
from photutils import make_source_mask, detect_sources, source_properties

## for speedy FFTs
#import pyfftw
#import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs
#pyfftw.interfaces.cache.enable()

# amakihi function for cropping by WCS
from crop import crop_WCS

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
try: plt.switch_backend('Qt5Agg')
except ImportError: pass # for Compute Canada server

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

# if hotpants cannot be called directly from the command line, the path to the
# executable can be put modified with the function hotpants_path() below
# by default, set to None --> assume can be called directly from the cmd line
HOTPANTS_PATH = "" 


###############################################################################
#### MASK BUILDING ####
    
def box_mask(image_file, pixx, pixy, mask_file=None, plot=False, 
             write=True, output=None):
    """
    Input: image file, the x pixel range and y pixel range to mask, an 
    already existing mask file to combine with the new box mask (optional; 
    default None), whether to plot the mask (optional; default False), whether
    to write the mask file (optional; default True), and the name for the
    output mask file (optional; default set below)
    
    Creates a simple box-shaped mask delimited by (pixx[0],pixx[1]) and 
    (pixy[0],pixy[1]). If an existing mask is supplied, the output mask will be 
    a combination of the previous mask and the box mask. 
    
    Output: the mask file HDU 
    """
    data = fits.getdata(image_file)
    hdr = fits.getheader(image_file)
    
    newmask = np.zeros(data.shape)
    newmask[pixy[0]:pixy[1], pixx[0]:pixx[1]] = 1.0
    newmask = newmask.astype(bool)
    
    if mask_file: # combine with another mask 
        mask = fits.getdata(mask_file)
        newmask = np.logical_or(mask, newmask)
    
    hdr = fits.getheader(image_file)
    mask_hdu = fits.PrimaryHDU(data=newmask.astype(int), header=hdr)
    
    if plot:
        plt.figure(figsize=(14,13))
        # show WCS      
        w = wcs.WCS(hdr)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        plt.imshow(newmask, cmap='binary_r', aspect=1, interpolation='nearest', 
                   origin='lower')
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("boxmask", fontsize=15)
        plt.savefig(image_file.replace(".fits","_boxmask.png")) 
        plt.close()
    
    if write:
        if not(output):
            output = image_file.replace(".fits", "_boxmask.fits")
            
        mask_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return mask_hdu


def saturation_mask(image_file, mask_file=None, sat_ADU=40000, 
                    sat_area_min=500, 
                    ra_safe=None, dec_safe=None, rad_safe=None, dilation_its=5, 
                    blursigma=2.0, write=True, output=None, plot=True, 
                    plotname=None):
    """
    Input: 
        - image file
        - a mask file to merge with the created saturation mask (optional; 
          default None)
        - saturation ADU above which all pixels will be masked (optional; 
          default 40000, which is a bit below the limit for MegaCam)
        - *minimum* and maximum areas in square pixels for a saturated source 
          (optional; default 500 and 100000, respectively)
        - the RA (deg), DEC (deg), and radius (arcsec) denoting a "safe zone"
          in which no sources will be masked (optional; defaults are None)
        - the number of iterations of binary dilation to apply to the mask, 
          where binary dilation is a process which "dilates" the detected 
          sources to make sure all pixels are adequately masked (optional; 
          default 5)
        - sigma of the gaussian filter to apply to the mask to blur it 
          (optional; default 2.0)
        - whether to write the mask file (optional; default True)
        - name for the output mask file (optional; default set below)
        - whether to plot the mask in greyscale (optional; default True)
        - name for the output plot (optional; default set below; only relevant
          if plot=True)
        
    Uses image segmentation to find all sources in the image. Then, looks for 
    sources which have a maximum flux above the saturation ADU and within the 
    minimum and maximum saturation area and creates a mask of these sources. 
    Binary dilation is applied to to enlarge features in the mask and Gaussian 
    blurring is applied to smooth out the mask. 
    
    If a "safe zone" is supplied, any sources within the safe zone will be 
    labelled as NON-saturated. This is useful if you know the coordinates of 
    some galaxy/nebulosity in your image which should not be masked, as it is 
    sometimes difficult to distinguish between a saturated source and a galaxy.
    
    If an existing mask is supplied, the output mask will be a combination of 
    the previous mask and saturation mask.
    
    Output: a table of source properties and the mask file HDU 
    """    
    
    data = fits.getdata(image_file)
    hdr = fits.getheader(image_file)
    
    ## set the threshold for image segmentation
    try:
        bkg_rms = hdr["BKGSTD"] # header written by bkgsub function
    except KeyError:
        # use crude image segmentation to find sources above SNR=3, build a 
        # source mask, and estimate the background RMS 
        if mask_file: # load a bad pixel mask if one is present 
            bp_mask = fits.getdata(mask_file).astype(bool)
            source_mask = make_source_mask(data, snr=3, npixels=5, 
                                       dilate_size=15, mask=bp_mask)
            # combine the bad pixel mask and source mask 
            rough_mask = np.logical_or(bp_mask,source_mask)
        else: 
            source_mask = make_source_mask(data, snr=3, npixels=5, 
                                       dilate_size=15)
            rough_mask = source_mask
        
        # estimate the background standard deviation
        try:
            sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
        except TypeError: # in old astropy, "maxiters" was "iters"
            sigma_clip = SigmaClip(sigma=3, iters=5)
        
        bkg = Background2D(data, (10,10), filter_size=(5,5), 
                           sigma_clip=sigma_clip, 
                           bkg_estimator=MedianBackground(), 
                           mask=rough_mask)
        bkg_rms = bkg.background_rms

    threshold = 3.0*bkg_rms # threshold for proper image segmentation 
    
    ## get the segmented image and source properties
    ## only detect sources composed of at least sat_area_min pixels
    segm = detect_sources(data, threshold, npixels=sat_area_min)
    labels = segm.labels 
    cat = source_properties(data, segm) 
    
    ## if any sources are found
    if len(cat) != 0:
        tbl = cat.to_table() # catalogue of sources as a table    
        mask = tbl["max_value"] >= sat_ADU # must be above this ADU
        sat_labels = labels[mask]
        tbl = tbl[mask] 
        
        # eliminate sources within the "safe zone", if given
        if (ra_safe and dec_safe and rad_safe):
            # get coordinates
            w = wcs.WCS(hdr)
            tbl["ra"], tbl["dec"] = w.all_pix2world(tbl["xcentroid"],
                                                    tbl["ycentroid"], 1) 
            safe_coord = SkyCoord(ra_safe*u.deg, dec_safe*u.deg, frame="icrs")
            source_coords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                                     frame="icrs")
            sep = safe_coord.separation(source_coords).arcsecond # separations 
            tbl["sep"] = sep # add a column for sep from safe zone centre
            mask = tbl["sep"] > rad_safe # only select sources outside this rad
            sat_labels = sat_labels[mask]
            tbl = tbl[mask]  
            
        # keep only the remaining saturated sources
        segm.keep_labels(sat_labels)        
        # build the mask, where masked=1 and unmasked=0
        newmask = segm.data_ma
        
        # combine with existing mask, if given
        if mask_file: 
            mask = fits.getdata(mask_file)
            newmask = np.logical_or(mask, newmask)             
        newmask[newmask >= 1] = 1 # masked pixels are labeled with 1
        newmask = newmask.filled(0) # unmasked labeled with 0 

        # mask pixels equal to 0, nan, or above the saturation ADU in the data
        newmask[data==0] = 1
        newmask[np.isnan(data)] = 1
        newmask[data>=sat_ADU] = 1
        
        # use binary dilation to fill holes, esp. near diffraction spikes
        newmask = (binary_dilation(newmask, 
                                   iterations=dilation_its)).astype(float)
        # use gaussian blurring to smooth out the mask 
        newmask = gaussian_filter(newmask, sigma=blursigma, mode="constant", 
                                  cval=0.0)
        newmask[newmask > 0] = 1
      
    ## if no sources are found 
    else: 
        tbl = Table() # empty table

        # use existing mask, if given
        newmask = np.zeros(shape=data.shape)
        if mask_file:
            mask = fits.getdata(mask_file)
            newmask[mask] = 1
            
        # mask pixels equal to 0, nan, or above the saturation ADU in the data
        newmask[data==0] = 1
        newmask[np.isnan(data)] = 1
        newmask[data>=sat_ADU] = 1
    
    ## construct the mask PrimaryHDU object    
    hdr = fits.getheader(image_file)
    mask_hdu = fits.PrimaryHDU(data=newmask.astype(int), header=hdr)
   
    if plot: # plot, if desired
        plt.figure(figsize=(14,13))
        w = wcs.WCS(hdr)
        ax = plt.subplot(projection=w) # show WCS
        ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        plt.imshow(newmask, cmap='gray', aspect=1, interpolation='nearest', 
                   origin='lower')
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("saturation mask", fontsize=15)
        
        if not(plotname):
            plotname = image_file.replace(".fits","_satmask.png")
        plt.savefig(plotname, bbox_inches="tight") 
        plt.close()
    
    if write:
        if not(output):
            output = image_file.replace(".fits", "_satmask.fits")          
        mask_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return tbl, mask_hdu

###############################################################################
#### ePSF BUILDING/USAGE ####

def ePSF_FWHM(epsf_data, verbose=False):
    """
    Input: 
        - fits file containing ePSF data OR the data array itself
        - be verbose (optional; default False)
    
    Output: the FWHM of the input ePSF
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


def build_ePSF(image_file, mask_file=None, nstars=40,              
               astrometry=False,                 
               thresh_sigma=5.0, pixelmin=20, etamax=1.4, area_max=500,             
               image_source_file=None, astrom_sigma=5.0, psf_sigma=5.0, 
               alim=10000, lowper=60, highper=90, clean=True,
               cutout=35, 
               write=True, output=None, plot=False, output_plot=None, 
               verbose=False):
    """             
    Input: 
        general:
        - filename for a **BACKGROUND-SUBTRACTED** image
        - filename for a mask image (optional; default None)
        - maximum number of stars to use (optional; default 40; set to None
          to impose no limit)
          
        source detection:
        - use astrometry.net for source detection if True, use simple image 
          segmentation to detect sources if False (optional; default False)

          if astrometry = False (default):
        - sigma threshold for source detection with image segmentation 
          (optional; default 5.0)
        - *minimum* number of isophotal pixels (optional; default 20)
        - *maximum* allowed elongation for sources found by image segmentation 
          (optional; default 1.4)
        - *maximum* allowed area for sources found by image segmentation 
          (optional; default 500 pix**2)
         
          if astrometry=True:
        - filename for a .xy.fits file containing detected sources with their
          x, y coords and **BACKGROUND-SUBTRACTED** fluxes (optional; default 
          None, in which case a new list will be made)
        - sigma threshold for astrometry.net source detection image (optional; 
          default 5.0)
        - sigma of the Gaussian PSF of the image (optional; default 5.0)
        - maximum allowed source area in pix**2 for astrometry.net for 
          deblending (optional; default 10000)
        - LOWER flux percentile such that sources below this flux will be 
          rejected when building the ePSF (optional; default 60 [%])
        - UPPER flux percentile such that sources above this flux will be 
          rejected when building the ePSF (optional; default 90 [%])
        - whether to remove files output by image2xy once finished with them 
          (optional; default True)
        
        - cutout size around each star in pix (optional; default 35 pix; must 
          be ODD, rounded down if even)
        
        writing, plotting, verbosity:
        - whether to write the ePSF to a new fits file (optional; default True)
        - name for the output fits file (optional; default set below)
        - whether to plot the ePSF (optional; default False) 
        - name for the output plot (optional; default set below)
        - be verbose (optional; default False)
    
    Uses image segmentation OR astrometry.net to obtain a list of sources in 
    the image with their x, y coordinates, flux, and background at their 
    location. (If a list of sources has already been obtained with astrometry, 
    this can be input). If astrometry is used for source detetion, then selects 
    stars between the <lowper>th and <highper>th percentile flux.
    
    Finally, uses EPSFBuilder to empirically obtain the ePSF of these stars. 
    Optionally writes and/or plots the obtaind ePSF.
    
    The ePSF obtained here should NOT be used in convolutions. Instead, it can 
    serve as a tool for estimating the seeing of an image. 
    
    Output: the ePSF data array
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

    ### OPTION 1: use image segmentation to find sources with an area > 
    ### pixelmin pix**2 which are above the threshold sigma*std (DEFAULT)
    if not(astrometry):
        image_data = fits.getdata(image_file) # subfile data
        image_data = np.ma.masked_where(image_data==0.0, 
                                        image_data) # mask bad pixels
        
        ## build an actual mask
        mask = (image_data==0)
        if mask_file:
            mask = np.logical_or(mask, fits.getdata(mask_file))

        ## set detection standard deviation
        try:
            std = image_header["BKGSTD"] # header written by bkgsub function
        except KeyError:
            # make crude source mask, get standard deviation of background
            source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                           dilate_size=15, mask=mask)
            final_mask = np.logical_or(mask, source_mask)
            std = np.std(np.ma.masked_where(final_mask, image_data))
        
        ## use the segmentation image to get the source properties 
        segm = detect_sources(image_data, thresh_sigma*std, npixels=pixelmin,
                              mask=mask) 
        cat = source_properties(image_data, segm, mask=mask)

        ## get the catalog and coordinates for sources
        try:
            tbl = cat.to_table()
        except ValueError:
            print("SourceCatalog contains no sources. Exiting.")
            return
        
        # restrict elongation and area to obtain only unsaturated stars 
        tbl = tbl[(tbl["elongation"] <= etamax)]
        tbl = tbl[(tbl["area"].value <= area_max)]

        sources = Table() # build a table 
        sources['x'] = tbl['xcentroid'] # for EPSFBuilder 
        sources['y'] = tbl['ycentroid']
        sources['flux'] = tbl['source_sum'].data/tbl["area"].data   
        sources.sort("flux")
        sources.reverse()
        
        if nstars:
            sources = sources[:min(nstars, len(sources))]

    ### OPTION 2: use pre-existing file obtained by astrometry.net, if supplied
    elif image_source_file:
        image_sources = np.logical_not(fits.getdata(image_source_file))
        
    ### OPTION 3: use astrometry.net to find the sources 
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
        if clean:
            run(f"rm {image_sources_file}", shell=True) # this file is not needed
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
    if astrometry and image_source_file:
        stars_tab = Table() # temporary table 
        stars_col = Column(data=range(len(stars.all_stars)), name="stars")
        stars_tab["stars"] = stars_col # column of indices of each star
        fluxes = [s.flux for s in stars]
        fluxes_col = Column(data=fluxes, name="flux")
        stars_tab["flux"] = fluxes_col # column of fluxes
        
        # get percentiles
        per_low = np.percentile(fluxes, lowper) # lower percentile flux 
        per_high = np.percentile(fluxes, highper) # upper percentile flux
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
        plt.figure(figsize=(10,9))
        plt.imshow(epsf_data, origin='lower', aspect=1, cmap='bone',
                   interpolation="nearest")
        plt.xlabel("Pixels", fontsize=16)
        plt.ylabel("Pixels", fontsize=16)
        plt.title("effective Point-Spread Function", fontsize=16)
        plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        plt.rc("xtick",labelsize=16) # not working?
        plt.rc("ytick",labelsize=16)
    
        if not(output_plot):
            output_plot = image_file.replace(".fits", "_ePSF.png")
        plt.savefig(output_plot, bbox_inches="tight")
        plt.close()
    
    return epsf_data


def convolve_self(image_file, mask_file=None,                                
                  thresh_sigma=5.0, pixelmin=20, etamax=1.4, area_max=500, 
                  cutout=35, verbose=True,
                  psf_fwhm_max=7.0, 
                  kernel='gauss', alpha=1.5, print_new_epsf=False,
                  write=True, output=None, 
                  plot=False, output_plot=None):
    """
    Input:
        - science image 
        - mask image (optional; default None)
        ...
        ...
        args to pass to build_ePSF(see above)
        ...
        ...
        - maximum allowed ePSF FWHM (pix); if FWHM is larger than this value 
          the function will exit (optional; default 7.0 pix)
        - type of kernel to build using ePSF (optional; default 'gauss'; 
          options are 'gauss' and 'moffat')
        - alpha parameter (power) for the Moffat kernel (optional; default 1.5,
          only relevant if kernel='moffat')
        -
        - whether to write the output image (optional; default True)
        - output filename (optional; default set below)
        - whether to plot the output image (optional; default False)
        - output plot filename (optional; default set below)
    
    Finds the ePSF of the input image and finds the FWHM (and thus sigma) of 
    the ePSF. Then builds a Gaussian with this sigma and convolves the science
    image with the Gaussian, OR builds a Moffat distribution with half the ePSF 
    FWHM as the core width and convolves the image with the Moffat. 
    
    Effectively tries to increase the size of the image's ePSF by ~sqrt(2). 
    Useful as a preparation for image differencing when sigma_template > 
    sigma_science.
    
    Output: PrimaryHDU for the science image convolved with the kernel
    """
    
    # get the ePSF for the image 
    epsf_data = build_ePSF(image_file, mask_file, thresh_sigma=thresh_sigma, 
                           pixelmin=pixelmin, etamax=etamax, 
                           area_max=area_max, cutout=cutout, write=False, 
                           verbose=verbose)     
    
    # get the sigma of the ePSF, assuming it is Gaussian
    fwhm = ePSF_FWHM(epsf_data, verbose)
    
    # if too large, exit 
    if fwhm > psf_fwhm_max:
        print(f"\nePSF_FWHM = {fwhm} > ePSF_FWHM_max = {psf_fwhm_max}")
        print("--> Exiting.")
        return
    
    # if size OK, build a kernel 
    if kernel == 'gauss': # gaussian with sigma = sigma of ePSF
        sigma = fwhm*gaussian_fwhm_to_sigma
        print("\nBuilding Gaussian kernel...")
        print(f"sigma = {sigma:.2f} pix")
        kern = Gaussian2DKernel(sigma, x_size=511, y_size=511)
    elif kernel == 'moffat':
        gamma = 0.5*fwhm/(2.0*((2**(1.0/alpha) - 1.0)**0.5))
        print("\nBuilding Moffat kernel...")
        print(f"gamma [core width] = {gamma:.2f} pix")
        kern = Moffat2DKernel(gamma, alpha=alpha, factor=1, x_size=127, 
                              y_size=127)
    else:
        print("\nInvalid kernel selected; options are 'gauss' or 'moffat'")
        print("--> Exiting.")
        return
    
    ## for bugtesting
    #plt.imshow(kern)
    
    # convolve the image data with its own ePSF, approximated as a Gaussian
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
    
    if plot: # plot, if desired
        plt.figure(figsize=(14,13))
        w = wcs.WCS(hdr)
        ax = plt.subplot(projection=w) # show WCS
        mean, med, std = sigma_clipped_stats(conv, mask=mask)
        ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        plt.imshow(conv, vmin=med-8*std, vmax=mean+8*std, cmap="bone",
                   aspect=1, interpolation='nearest', origin='lower')

        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        cb.set_label(label="ADU", fontsize=16)
        cb.ax.tick_params(which='major', labelsize=15)   
        
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        
        # set the title
        topfile = re.sub(".*/", "", image_file)
        title = topfile.replace(".fits","")
        title = r"$\mathtt{"+title.replace("_","\_")+"}$"
        title = f"{title} convolved with self"
        plt.title(title, fontsize=15)
    
        if not(output_plot):
            output_plot = image_file.replace(".fits", "_selfconv.png")
        plt.savefig(output_plot, bbox_inches="tight")
        plt.close()

    hdu = fits.PrimaryHDU(data=conv, header=hdr)    
    if write: # write, if desired     
        if not(output):
            output = image_file.replace(".fits", "_selfconv.fits")          
        hdu.writeto(output, overwrite=True, output_verify="ignore")

        if print_new_epsf:
            # get the ePSF for the convolved image
            epsf_data = build_ePSF(output, mask_file, 
                                   thresh_sigma=thresh_sigma, 
                                   pixelmin=pixelmin, 
                                   etamax=etamax, area_max=area_max, 
                                   cutout=cutout, write=False, verbose=verbose)     
        
            # get the sigma of the ePSF, assuming it is Gaussian
            fwhm = ePSF_FWHM(epsf_data, verbose)
        
    return hdu

###############################################################################
#### IMAGE DIFFERENCING WITH HOTPANTS ####

def hotpants_path(path):
    """
    Input: a path to the hotpants executable
    Output: None
    
    A bandaid fix for when I haven't correctly setup my paths to find hotpants
    on the command line. Can explicitly set the path here.
    
    """
    global HOTPANTS_PATH
    HOTPANTS_PATH = path


def get_substamps(source_file, template_file, 
                  sci_mask_file=None, tmp_mask_file=None, 
                  sigma=3.0, etamax=2.0, area_max=400.0, sep_max=5.0, 
                  coords="mean",
                  output=None, verbose=False):
    """    
    Input: 
        - science image (the source)
        - template to match to
        - mask of bad pixels in science image (optional; default None)
        - mask of bad pixels in template image (optional; default None)
        - sigma to use in setting the threshold for image segmentation 
          (optional; default 3.0)
        - maximum elongation allowed for the sources (optional; default 2.0)
        - maximum pixel area allowed for the sources (optional; default 400.0
          pix**2)
        - maximum allowed separation for sources to be designated as common to 
          both the science image and template image, in arcsec (optional; 
          default 5.0")
        - coordinate system for output (optional; default 'mean' of science and
          template; options are 'mean', 'source', 'template')
        - name for the output ascii txt file of substamps 'x y' (optional; 
          default set below)
        - whether to be verbose (optional; default False)
    
    For both the science and template image, finds sources via image 
    segmentation and selects those which are not overly elongated (to filter 
    out galaxies) or overly large (to filter out saturated stars). Then 
    compares the science and template images, finds sources in common, and 
    writes these to an ascii text file of the form 
    'x1 y1'
    'x2 y2'
    'x3 y3'
     ...
     
    Output: 
        - properties of all sources which were detected in both the science and 
          template image, as obtained by image segmentation
    """    
    # load in data
    source_header = fits.getheader(source_file)
    source_data = fits.getdata(source_file)
    tmp_header = fits.getheader(template_file)
    tmp_data = fits.getdata(template_file)
    data = [source_data, tmp_data]
    hdrs = [source_header, tmp_header]

    ## masks
    if sci_mask_file and tmp_mask_file: # if both masks are supplied        
        mask = np.logical_or(fits.getdata(sci_mask_file).astype(bool),
                             fits.getdata(tmp_mask_file).astype(bool))
    elif sci_mask_file and not(tmp_mask_file): # if only science        
        mask = fits.getdata(sci_mask_file).astype(bool)
    elif not(sci_mask_file) and tmp_mask_file: # if only template        
        mask = fits.getdata(sci_mask_file).astype(bool)
    else:
        mask = None
        
    ## find sources in both the science and template image
    obj_props = []
    for i in range(2):
        image_data = data[i]
        image_header = hdrs[i]
        ## set the threshold for image segmentation
        try: # if bkg subtraction has already been obtained 
            std = image_header["BKGSTD"] # header written by bkgsub function
            threshold = sigma*std 
            
        except KeyError: # else, do it
            # use crude image segmentation to find sources above SNR=3
            # allows us to ignore sources during background estimation
            if type(mask) == np.ndarray: # use bad pixel mask if one is present 
                source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                               dilate_size=15, mask=mask)
                # combine the bad pixel mask and source mask 
                final_mask = np.logical_or(mask, source_mask)
            else: 
                source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                               dilate_size=15)
                final_mask = source_mask
            # estimate the background
            try:
                sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
            except TypeError: # in old astropy, "maxiters" was "iters"
                sigma_clip = SigmaClip(sigma=3, iters=5)
            
            bkg = Background2D(image_data, (10,10), filter_size=(5,5), 
                               sigma_clip=sigma_clip, 
                               bkg_estimator=MedianBackground(), 
                               mask=final_mask)
            bkg_rms = bkg.background_rms       
            threshold = sigma*bkg_rms # threshold for proper image segmentation 
        
        # get the segmented image 
        segm = detect_sources(image_data, threshold, npixels=5, mask=mask)
        # get the source properties
        cat = source_properties(image_data, segm, mask=mask) 
        try:
            tbl = cat.to_table()
        except ValueError:
            print("SourceCatalog contains no sources. Exiting.")
            return
        # restrict elongation and area to obtain only unsaturated stars 
        tbl_mask = (tbl["elongation"] <= etamax)
        tbl = tbl[tbl_mask]
        tbl_mask = tbl["area"].value <= area_max
        tbl = tbl[tbl_mask]
        obj_props.append(tbl)

    ## get RA, Dec of sources after this restriction
    # science image
    w = wcs.WCS(source_header)
    source_coords = w.all_pix2world(obj_props[0]["xcentroid"], 
                                    obj_props[0]["ycentroid"], 1)
    source_skycoords = SkyCoord(ra=source_coords[0], dec=source_coords[1],
                                frame="icrs", unit="degree")
    # template
    w = wcs.WCS(tmp_header)
    tmp_coords = w.all_pix2world(obj_props[1]["xcentroid"], 
                                 obj_props[1]["ycentroid"], 1)   
    tmp_skycoords = SkyCoord(ra=tmp_coords[0], dec=tmp_coords[1], frame="icrs", 
                             unit="degree")
    
    ## find sources which are present in both the science image and template
    ## sources must be <= 1.0" away from each other 
    idx_sci, idx_tmp, d2d, d3d = tmp_skycoords.search_around_sky(
                                 source_skycoords, sep_max*u.arcsec) 
    obj_props[0] = obj_props[0][idx_sci]
    obj_props[1] = obj_props[1][idx_tmp]
    
    ## informative prints
    if verbose:
        print("\nsources in science image:\n")
        obj_props[0]["xcentroid","ycentroid"].pprint()
        print("\nsources in template image:\n")
        obj_props[1]["xcentroid","ycentroid"].pprint()    
        print(f'\nmean separation = {np.mean(d2d).arcsec:.3f}"')
    
    ## write x, y to a substamps ascii file 
    nmatch = len(obj_props[0])
    if coords == "mean":
        x = [np.mean([obj_props[0]["xcentroid"][i].value+1, 
                obj_props[1]["xcentroid"][i].value+1]) for i in range(nmatch)]
        y = [np.mean([obj_props[0]["ycentroid"][i].value+1, 
                obj_props[1]["ycentroid"][i].value+1]) for i in range(nmatch)]
    elif coords == "source":
        x = [obj_props[0]["xcentroid"][i].value for i in range(nmatch)]
        y = [obj_props[0]["ycentroid"][i].value for i in range(nmatch)]
    elif coords == "template":
        x = [obj_props[1]["xcentroid"][i].value for i in range(nmatch)]
        y = [obj_props[1]["ycentroid"][i].value for i in range(nmatch)]  
              
    if not(output):
        output = source_file.replace(".fits", "_substamps.txt")
    lines = np.array([[x[n], y[n]] for n in range(nmatch)])
    lines.sort(axis=0)
    #lines = [f"{int(x[n])} {int(y[n])}" for n in range(nmatch)]
    np.savetxt(output, lines, fmt="%.3f", encoding="ascii", newline="\n")
    
    return (obj_props[0]["xcentroid","ycentroid"], 
            obj_props[1]["xcentroid", "ycentroid"])
    

def param_estimate(source_file, template_file, mask_file=None,
                            source_epsf_file=None, tmp_epsf_file=None, 
                            thresh_sigma=3.0, pixelmin=20, etamax=1.4,
                            area_max=400, cutout=35, verbose=True):
    """   
    WIP:
        - Not stable right now; messes up when science and reference have 
          very similar PSFs (which is often the case)
    
    Inputs:        
        general:
        - science image 
        - template image
        - mask image (optional; default None)
        - source ePSF in fits file (optional; default None, in which case the 
          ePSF is obtained)
        - template ePSF in fits file (optional; default None, in which case the
          ePSF is obtained)
          ...
          ...
          args to pass to build_ePSF [see above]
          ...
          ...
        - be verbose (optional; default False)
     
    Given some science and template image, determines the optimal parameters 
    to pass to hotpants for successful subtraction.
          
    Outputs:
        - whether to convolve the image or template 
        - estimate 3 Gaussian terms with polynomial orders (as a dictionary)
    """
    
    ## load in OR build the source image ePSF, get FWHM
    if source_epsf_file:
        source_epsf = fits.getdata(source_epsf_file)
    else:
        source_epsf = build_ePSF(source_file, mask_file, 
                                 thresh_sigma=thresh_sigma, pixelmin=pixelmin, 
                                 etamax=etamax, 
                                 area_max=area_max, cutout=cutout, 
                                 write=False, verbose=verbose)
    
    ## load in OR build the template image ePSF
    if tmp_epsf_file:
        tmp_epsf = fits.getdata(tmp_epsf_file)
    else:
        tmp_epsf = build_ePSF(template_file, mask_file, 
                              thresh_sigma=thresh_sigma, pixelmin=pixelmin, 
                              etamax=etamax, area_max=area_max,                                
                              cutout=cutout, write=False, verbose=verbose)
        
    
    ## get the FWHM and sigma of each PSF, assuming roughly Gaussian
    source_epsf_sig = gaussian_fwhm_to_sigma*ePSF_FWHM(source_epsf, verbose)
    tmp_epsf_sig = gaussian_fwhm_to_sigma*ePSF_FWHM(tmp_epsf, verbose)
    
    ## compare ePSFS
    if tmp_epsf_sig > source_epsf_sig: 
        ## sharpening the template leads to false positives, SO --> 
        # need to either convolve the science image with convolution kernel OR 
        # convolve the science image with its own ePSF prior to template 
        # matching, which increases the sigma of the science image ePSF by a
        # factor of ~sqrt(2)
        if verbose:
            print("\n FWHM_tmp > FWHM_sci")        
        # trying the following for now 
        psf_match_sig = (tmp_epsf_sig**2 - source_epsf_sig**2)**(0.5)
        widths = (0.5*psf_match_sig, psf_match_sig, 2.0*psf_match_sig)
        poly_orders = (6,4,2)
        gaussian_terms = dict(zip(poly_orders, widths))        
        convolve="i" # convolve the template
    
    else:
        ## smudge the template
        # the sigma that matches the image and template is approx.
        # sqrt(sig_im**2 - sig_tmp**2)
        if verbose:
            print("\n FWHM_sci > FWHM_tmp")
        psf_match_sig = (source_epsf_sig**2 - tmp_epsf_sig**2)**(0.5)
        widths = (0.5*psf_match_sig, psf_match_sig, 2.0*psf_match_sig)
        poly_orders = (6,4,2)
        gaussian_terms = dict(zip(poly_orders, widths))        
        convolve="t" # convolve the template
    
    return convolve, gaussian_terms


def hotpants(source_file, template_file, 
             sci_mask_file=None, 
             tmp_mask_file=None, 
             substamps_file=None, param_estimate=False,
             iu=None, tu=None, il=None, tl=None, 
             lsigma=5.0, hsigma=5.0,
             gd=None, 
             ig=None, ir=None, tg=None, tr=None, 
             ng=None, rkernel=10, 
             nrx=1, nry=1, nsx=10, nsy=10, nss=3, rss=None,
             ssig=3.0,
             norm="i",
             convi=False, convt=False, 
             bgo=0, ko=1, 
             output=None, mask_write=False, conv_write=False, 
             kern_write=False, noise_write=False, noise_scale_write=False,
             maskout=None, convout=None, kerout=None, noiseout=None, 
             noisescaleout=None,
             v=1, log=None,
             plot=True, plotname=None, scale=None, 
             target=None, target_small=None,
             thresh_sigma=3.0, pixelmin=20, etamax=1.4, area_max=500):
    """       
    hotpants args OR hotpants-related: 
        basic inputs:
        - the science image 
        - the template to match to
        - a mask of which pixels to ignore in the science image (optional; 
          default None)
        - a mask of which pixels to ignore in the template image (optional; 
          default None)
        - a text file containing the substamps 'x y' (optional; default None)
        - whether to compare the science and template ePSFS to estimate the 
          optimal Gaussian terms, convolution kernel half width, substamp FWHM 
          to extract around stars, and whether to convolve the science image or 
          template (optional; default False; overrides the following parameters 
          if True: ng, rkernel, convi, convt)
        
        ADU limits:
        - the upper (u) and lower (l) ADU limits for the image (i) and 
          template (t) (optional; defaults set below by lsigma/hsigma)
        - no. of std. devs away from background at which to place the LOWER 
          ADU limit (optional; default 5.0)
        - no. of std. devs away from background at which to place the UPPER
          ADU limit (optional; default 5.0)
        
        good pixels:
        - good pixels coordinates (optional; default is full image)
            format: xmin xmax ymin ymax 
            e.g. gd="150 1000 200 800" 
        
        gain, noise:
        - the gain (g) and readnoise (r) for the image (i) and template (t) 
          (optional; default is to extract from headers or set gain=1, noise=0
          if no relevant headers are found)
        
        Gaussian terms:
        - gaussian terms (optional; default set below)
            format: ngauss degree0 sigma0 ... degreeN sigmaN, N=ngauss-1
            e.g. 3 6.0 0.7 4.0 1.5 2.0 3.0 (default)
            where ngauss is the no. of gaussians which compose the kernel, 
            degreeI is the degree of the Ith polynomial, and sigmaI is the 
            width of the Ith gaussian
        
        convolution kernel:
        - convolution kernel FWHM (optional; default is 10.0)
        
        regions, stamps and substamps:
        - no. of regions in x direction (optional; default 1)
        - no. of regions in y direction (optional; default 1)
        - no. of stamps in each region in x direction (optional; default 10)
        - no. of stamps in each region in y direction (optional; default 10)
        - no. of centroids to extract from each stamp (optional; default 3)
        - half width of substamp around centroids (optional; default is 2.5*
          rkernel = 25.0 for default rkernel) 
        
        sigma clipping, normalization:
        - sigma threshold for sigma clipping (optional; default 3.0)
        - normalization term (optional; default 'i' for image; options are 
          'i' for image, 't' for template, 'u' for unconvolved image/template)
        
        convolution of image or template:
        - force convolve the image (optional; default False)
        - force convolve the template (optional; default False)
        
        background/kernel variation
        - spatial background variation order (optional; default 0)
        - spatial kernel variation order (optional; default 1)
        
        outputs:
        - name for the output subtracted image (optional; default set below)
        - whether to output the bad pixel mask (optional; default False)
        - whether to output the convolved image (optional; default False)
        - whether to output the kernel image (optional; default False)
        - whether to output the noise image (optional; default False)
        - whether to output the noise *scaled difference* image (optional; 
          default False)
        - name for the output bad pixel mask (optional; default set below; only
          relevant if mask_write=True)
        - name for the output convolved image (optional; default set below; 
          only relevant if conv_write=True)
        - name for the output kernel (optional; default set below; only 
          relevant if kern_write=True)
        - name for the output noise image (optional; default set below; only 
          relevant if noise_write=True)
        - name for the output noise *scaled difference* image (optional; 
          default set below, only relevant if noise_scale_write=True)
        
        verbosity:
        - verbosity (optional; default 1; options are 0 - 2 where 0 is least
          verbose)
        
        log file:
        - a name for a logfile to store STDERR (optional; default None)
        
    other args:
        - plot the subtracted image data (optional; default False)
        - name for the plot (optional; default set below)
        - scale to apply to the plot (optional; default None (linear); 
          options are "linear, "log", "asinh")
        - target Ra, Dec at which to place crosshair (optional, default None)
        - target Ra, Dec at which to place smaller crosshair (optional, default 
          None)
        
    args for build_ePSF (iff param_estimate = True):
        - sigma threshold for source detection with image segmentation 
          (optional; default 5.0)
        - *minimum* number of isophotal pixels (optional; default 20)
        - *maximum* allowed elongation for sources found by image segmentation 
          (optional; default 1.8)
        - *maximum* allowed area for sources found by image segmentation 
          (optional; default 500 pix**2)
    
    A wrapper for hotpants. 
    
    https://github.com/acbecker/hotpants
    HOTPANTS: High Order Transformation of Psf ANd Template Subtraction
    Based on >>> Alard, C., & Lupton, R. H. 1998, ApJ, 503, 325 <<<
    https://iopscience.iop.org/article/10.1086/305984/pdf 
    
    Output: the subtracted image HDU 
    """

    ## load in data 
    source_header = fits.getheader(source_file)
    source_data = fits.getdata(source_file)
    tmp_header = fits.getheader(template_file)
    tmp_data = fits.getdata(template_file)
        
    ########################## OPTIONS FOR HOTPANTS ###########################
    
    ### input/output files and masks ##########################################
    hp_options = f"-inim {source_file} -tmplim {template_file} "#-fi 0"
    
    if not(output):
        output = source_file.replace(".fits", "_subtracted.fits")
    hp_options = f"{hp_options} -outim {output}" # output subtracted file
    
    if mask_write:
        if not(maskout):
            maskout = source_file.replace(".fits", "_submask.fits")
        hp_options = f"{hp_options} -omi {maskout}" # output bad pixel mask        
    if conv_write:
        if not(convout):
            convout = source_file.replace(".fits", "_conv.fits")        
        hp_options = f"{hp_options} -oci {convout}" # output convolved image       
    if kern_write:
        if not(kerout):
            kerout = source_file.replace(".fits", "_kernel.fits")
        hp_options = f"{hp_options} -oki {kerout}" # output kernel image
    if noise_write:
        if not(noiseout):
            noiseout = source_file.replace(".fits", "_noise.fits")
        hp_options = f"{hp_options} -oni {noiseout}" # output noise image
    if noise_scale_write:
        if not(noisescaleout):
            noisescaleout = source_file.replace(".fits", "_noise_scaled.fits")
        hp_options = f"{hp_options} -ond {noisescaleout}" # noise *scaled diff* 
     
    # masks
    if sci_mask_file and tmp_mask_file: # if both masks are supplied        
        mask = np.logical_or(fits.getdata(sci_mask_file).astype(bool),
                             fits.getdata(tmp_mask_file).astype(bool))
        hp_options = f"{hp_options} -imi {sci_mask_file}" # mask for sci 
        hp_options = f"{hp_options} -tmi {tmp_mask_file}" # for tmp 
    elif sci_mask_file and not(tmp_mask_file): # if only science        
        mask = fits.getdata(sci_mask_file).astype(bool)
        hp_options = f"{hp_options} -imi {sci_mask_file}"  
        hp_options = f"{hp_options} -tmi {sci_mask_file}"  
    elif not(sci_mask_file) and tmp_mask_file: # if only template        
        mask = fits.getdata(sci_mask_file).astype(bool)
        hp_options = f"{hp_options} -imi {tmp_mask_file}" 
        hp_options = f"{hp_options} -tmi {tmp_mask_file}" 
    else:
        mask = None
            
    # substamps file
    if substamps_file: # if a file of substamps X Y is supplied
        hp_options = f"{hp_options} -ssf {substamps_file} -afssc 1"
        #hp_options += f' -savexy {source_file.replace(".fits", "_conv")}'

    ### if requested, compare sci/tmp ePSF to estimate optimal params #########    
    if param_estimate:
        c, g = param_estimate(source_file, template_file, 
                              thresh_sigma=thresh_sigma, 
                              pixelmin=pixelmin, etamax=etamax, 
                              area_max=area_max, verbose=True)
    
        ng = f"3 6 {g[6]:.2f} 4 {g[4]:.2f} 2 {g[2]:.2f}"
        rkernel = 2.5*g[4]*gaussian_sigma_to_fwhm
        print(f"rkernel = {rkernel}")
        if c == "t": convi = False; convt = True
        elif c == "i": convi = True; convt = False
            
    ### upper/lower limits for SCIENCE image ##################################
    # impose no limits if none are given
    if iu: hp_options = f"{hp_options} -iu {iu}"
    else: hp_options = f"{hp_options} -iu {np.max(source_data)}"
        
    if il: hp_options = f"{hp_options} -il {il}"
    else: hp_options = f"{hp_options} -il {np.min(source_data)}"
        
    #if iu and il: # if both limits given
    #    hp_options = f"{hp_options} -iu {iu} -il {il}"
    #    
    #else: # if values are not 
    #    try:
    #        std = source_header["BKGSTD"] # header written by bkgsub function
    #        med = 0
    #    except KeyError: 
    #        # use image segmentation to find sources above SNR=3 and mask them 
    #        if sci_mask_file: # load the bad pixel mask if one is present 
    #            sci_bp_mask = fits.getdata(sci_mask_file).astype(bool)
    #            source_mask = make_source_mask(source_data, snr=3, npixels=5, 
    #                                           dilate_size=15, 
    #                                           mask=sci_bp_mask)
    #            # combine the bad pixel mask and source mask 
    #            final_mask = np.logical_or(sci_bp_mask, source_mask)
    #        else: 
    #            source_mask = make_source_mask(source_data, snr=3, npixels=5, 
    #                                           dilate_size=15)
    #            final_mask = source_mask 
    #            
    #        # estimate the background as just the median 
    #        try: 
    #            mean, med, std = sigma_clipped_stats(source_data, 
    #                                                 mask=final_mask,
    #                                                 maxiters=1)
    #        except TypeError: # in old astropy, "maxiters" was "iters"
    #            mean, med, std = sigma_clipped_stats(source_data, 
    #                                                 mask=final_mask, 
    #                                                 iters=1)
    #        
    #    #  set upper/lower thresholds to median +/- hsigma*std/lsigma*std
    #    hp_options = f"{hp_options} -iu {med+hsigma*std}"
    #    hp_options = f"{hp_options} -il {med-lsigma*std}"
    #    print(f"\n\nSCIENCE UPPER LIMIT = {med+hsigma*std}")
    #    print(f"SCIENCE LOWER LIMIT = {med-lsigma*std}\n") 

    ### upper/lower limits for TEMPLATE image #################################  
    # if no limits given, first look for headers 
    # if none found, impose no limits 
    if tu: 
        hp_options = f"{hp_options} -tu {tu}"
    else: 
        try: # PS1's image saturation ADU header
            hp_options += f' -tu {tmp_header["HIERARCH CELL.SATURATION"]}' 
        except KeyError: # header not found
            hp_options = f"{hp_options} -tu {np.max(tmp_data)}"

    if tl: hp_options = f"{hp_options} -tl {tl}"
    else: hp_options = f"{hp_options} -tl {np.min(tmp_data)}"
        
    #if tu and tl: # if both limits given
    #    hp_options = f"{hp_options} -tu {tu} -tl {tl}"
    #    
    #else: # if values are not given
    #    try:
    #        std = tmp_header["BKGSTD"] # header written by bkgsub function
    #        med = 0
    #    except KeyError: 
    #        # use image segmentation to find sources above SNR=3 and mask them 
    #        if tmp_mask_file: # load the bad  
    #            tmp_bp_mask = fits.getdata(tmp_mask_file).astype(bool)
    #            source_mask = make_source_mask(tmp_data, snr=3, npixels=5, 
    #                                           dilate_size=15, 
    #                                           mask=tmp_bp_mask)
    #            # combine the bad pixel mask and source mask 
    #            final_mask = np.logical_or(tmp_bp_mask,source_mask)
    #        else: 
    #            source_mask = make_source_mask(tmp_data, snr=3, npixels=5, 
    #                                           dilate_size=15)
    #            final_mask = source_mask    
    #            
    #        # estimate the background as just the median 
    #        try: 
    #            mean, med, std = sigma_clipped_stats(tmp_data, mask=final_mask,
    #                                                 maxiters=1)
    #        except TypeError: # in old astropy, "maxiters" was "iters"
    #            mean, med, std = sigma_clipped_stats(tmp_data, mask=final_mask, 
    #                                                 iters=1)
    #        
    #    #  set upper/lower thresholds to median +/- hsigma/lsigma*std
    #    hp_options = f"{hp_options} -tu {med+hsigma*std}"
    #    hp_options = f"{hp_options} -tl {med-lsigma*std}"
    #   print(f"\n\nTEMPLATE UPPER LIMIT = {med+hsigma*std}")
    #   print(f"TEMPLATE LOWER LIMIT = {med-lsigma*std}\n") 
    
    ### x, y limits ###########################################################
    if gd: hp_options = f"{hp_options} -gd {gd}"
        
    ### gain (e-/ADU) and readnoise (e-) for SCIENCE image ####################    
    if ig: # if argument is supplied for image gain
        hp_options = f"{hp_options} -ig {ig}"
    else: # if no argument given, extract from header
        try:
            hp_options = f'{hp_options} -ig {source_header["GAIN"]}'
        except KeyError: # if no keyword GAIN found in header 
            pass
    if ir: # image readnoise
        hp_options = f"{hp_options} ir {ir}"
    else:
        try:
            hp_options = f'{hp_options} -ir {source_header["RDNOISE"]}'
        except KeyError: 
            pass    

    ### gain (e-/ADU) and readnoise (e-) for TEMPLATE image ###################       
    if tg: # if argument is supplied for template gain
        hp_options = f"{hp_options} -tg {tg}"
    else: # if no argument given, extract from header
        try:
            hp_options = f'{hp_options} -tg {tmp_header["GAIN"]}' # CFHT header
        except KeyError: 
            # if no keyword GAIN found in header, look for PS1's gain header
            try:
                hp_options += f' -tg {tmp_header["HIERARCH CELL.GAIN"]}' 
            except KeyError:
                pass           
    if tr: # template readnoise
        hp_options = f"{hp_options} -tr {tr}"
    else:
        try:
            hp_options = f'{hp_options} -tr {tmp_header["RDNOISE"]}'
        except KeyError: 
            try:
                hp_options += f' -tr {tmp_header["HIERARCH CELL.READNOISE"]}'
            except KeyError:
                pass
    
    ### Gaussian and convolution kernel parameters ############################
    if ng: hp_options = f"{hp_options} -ng {ng}"        
    hp_options = f"{hp_options} -r {rkernel}" # convolution kernel FWHM 

    ### regions, stamps and substamps #########################################
    # no. of regions, stamps per region 
    hp_options += f" -nrx {nrx:d} -nry {nry:d} -nsx {nsx:d} -nsy {nsy:d}"
    
    # half-width substamp around centroids 
    if rss: hp_options = f"{hp_options} -rss {rss}"     
    else: hp_options = f"{hp_options} -rss {2.5*rkernel}" # default 2.5*rkernel
    
    ### normalization, sigma clipping, convolution ############################    
    hp_options = f"{hp_options} -n {norm}" # normalization
    hp_options = f"{hp_options} -ssig {ssig}" # sigma clipping threshold
    
    # if convi and convt are both true, convi overrules  
    if convi: hp_options = f"{hp_options} -c i" # convolve the science image
    elif convt: hp_options = f"{hp_options} -c t" # convolve the template
        
    ### spatial kernel/background variation ###################################
    hp_options = f"{hp_options} -bgo {bgo} -ko {ko}" # bg order, kernel order     

    ### misc ##################################################################
    hp_options = f"{hp_options} -v {v}" # verbosity 
    
    ### writing stderr to a log ###############################################
    if log: hp_options = f"{hp_options} 2> {log}"

    ######################### CALL HOTPANTS  ##################################
    
    # if a special path to hotpants is supplied
    if HOTPANTS_PATH: hp_cmd = f"{HOTPANTS_PATH} {hp_options}"
    # if not, use the system default
    else: hp_cmd = f"hotpants {hp_options}"
    
    # try calling hotpants
    try:
        hp = run(f"{hp_cmd}", shell=True, stderr=PIPE, check=True) 
        
        # check STDERR to see if hotpants failed without throwing an error code
        hp_stderr = str(hp.stderr, "utf-8")
        if "0 stamps built (0.00%)" in hp_stderr[-33:-10]:
            print(f"{hp_stderr[:-10]}\n\nFAILED, EXITING")
            return  
        else:
            print(f"{hp_stderr}")
                          
    # if an error code is returned by hotpants, exit
    except CalledProcessError: # if an error code is returned, exit 
        print("\n\n\nError, exiting")
        return 
    
    # if file successfully produced and non-empty
    outfile = re.sub(".*/", "", output) # for a file /a/b/c, extract the "c"
    topdir = output[:-len(outfile)] # extract the "/a/b/"
    if (outfile in os.listdir(topdir)) and (os.stat(output).st_size!=0):
        sub = fits.getdata(output) # load it in 
        sub_header = fits.getheader(output)
    else:
        return
    
    # mask bad pixels (update the difference image)
    if type(mask) == np.ndarray:
        sub[mask] = 0 # set all bad pixels to 0 in image
        hdul = fits.open(output, mode="update")
        hdul[0].data = sub
        hdul.close()
    
    try:
        mean_diff = float(sub_header["DMEAN00"]) # mean of diff img good pixels
        std_diff = float(sub_header["DSIGE00"]) # stdev of diff img good pixels
        print(f"\nMEAN OF DIFFERENCE IMAGE = {mean_diff:.2f} +/- "
              f"{std_diff:.2f}\n")
              
    except KeyError:
        print("\nCould not find DMEAN00 and DSIGE00 (mean and standard dev. "+
              "of difference image good pixels) headers in difference image. "+
              "The difference image was probably not correctly produced. "+
              "Exiting.\n")
        return 
        
    ### PLOTTING (OPTIONAL) ##############################################
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS      
        w = wcs.WCS(source_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        
        if not scale or (scale == "linear"): # if no scale to apply 
            scale = "linear"
            plt.imshow(sub, cmap='coolwarm', vmin=mean_diff-3*std_diff, 
                       vmax=mean_diff+3*std_diff, aspect=1, 
                       interpolation='nearest', origin='lower')
            #plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            
        elif scale == "log": # if we want to apply a log scale 
            sub_log = np.log10(sub)
            lognorm = simple_norm(sub_log, "log", percent=99.0)
            plt.imshow(sub_log, cmap='bone', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            #plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            
        elif scale == "asinh":  # asinh scale
            sub_asinh = np.arcsinh(sub)
            asinhnorm = simple_norm(sub, "asinh")
            plt.imshow(sub_asinh, cmap="viridis", aspect=1, norm=asinhnorm,
                       interpolation="nearest", origin="lower")
            
        cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        cb.set_label(label="ADU", fontsize=16)
        cb.ax.tick_params(which='major', labelsize=15)
            
        if target:
            ra, dec = target
            plt.gca().plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
                   transform=plt.gca().get_transform('icrs'), linewidth=2, 
                   color="black", marker="")
            plt.gca().plot([ra, ra], [dec+10.0/3600.0, dec+5.0/3600.0], 
                   transform=plt.gca().get_transform('icrs'),  linewidth=2, 
                   color="black", marker="")

        if target_small:
            ra, dec = target_small
            plt.gca().plot([ra-5.0/3600.0, ra-2.5/3600.0], [dec,dec], 
                   transform=plt.gca().get_transform('icrs'), linewidth=2, 
                   color="#fe019a", marker="")
            plt.gca().plot([ra, ra], [dec-5.0/3600.0, dec-2.5/3600.0], 
                   transform=plt.gca().get_transform('icrs'),  linewidth=2, 
                   color="#fe019a", marker="")
        
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title(r"$\mathtt{hotpants}$"+" difference image", fontsize=15)
        
        if not(plotname):
            plotname = outfile.replace(".fits", "_hotpants.png")
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()
        
    return sub

###############################################################################
#### TRANSIENT DETECTION ####

def __rejection_plot(sub_file, 
                     dipoles, elongated_sources, large_sources, vetted, 
                     dipole_width, etamax, area_max, nsource_max,
                     toi=None, #toi_sep_max=None, 
                     dipole_color="black", 
                     elong_color="#be03fd", 
                     large_color="#fcc006", 
                     vetted_color="#53fca1",
                     cmap="coolwarm", 
                     output=None):
    """    
    Input:
        - filename for the difference image
        - table of candidates rejected as dipoles
        - table of candidates rejected due to elongation
        - table of candidates rejected due to area 
        - table of candidates after all vetting 
        - maximum dipole width 
        - maximum allowed elongation 
        - maximum allowed pixel area 
        - [ra,dec] for some target of interest (optional; default None)
        - radius (in arcsec) around the toi to probe (optional; default None)
        - color of circle flagging dipoles (optional; default black)
        - color of square flagging elongated sources (optional; default bright
          purple)
        - color of triangle flagging overly large sources (optional; default 
          marigold)
        - color of diamond flagging sources which passed all vetting (optional;
          default sea green)
        - colourmap for image (optional; default "coolwarm")
        - output plotname (optional; default set below)
    
    Plot a difference image with marker flagging likely dipoles, overly 
    elongated sources, overly large sources, and sources which passed all these
    vetting steps, as determined by transient_detect()
    
    Output: None
    """

    from collections import OrderedDict

    ## load in data    
    data = fits.getdata(sub_file)
    hdr = fits.getheader(sub_file)

    plt.figure(figsize=(14,13))
    w = wcs.WCS(hdr) # show WCS
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)

    ## plot the image
    mean, median, std = sigma_clipped_stats(data)
    plt.imshow(data, vmin=mean-5*std, vmax=mean+5*std, origin='lower', 
               cmap=cmap)
    cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cb.set_label(label="ADU", fontsize=16)
    cb.ax.tick_params(which='major', labelsize=15)

    # markers over dipoles, overly elongated sources, & overly large sources
    for d in dipoles: # circles
        ra, dec = d["ra"], d["dec"]
        plt.plot(ra, dec, transform=ax.get_transform('icrs'), ms=10,
                 mec=dipole_color, mfc="None", mew=2, marker="o", ls="", 
                 label='dipole '+r'$\leqslant{ }$'+' '+str(dipole_width)+'"')

    for e in elongated_sources: # squares
        ra, dec = e["ra"], e["dec"]
        plt.plot(ra, dec, transform=ax.get_transform('icrs'), ms=10,
                 mec=elong_color, mfc="None", mew=2, marker="s", ls="", 
                 label=r"$\eta$"+" "+r"$\geqslant$"+" "+str(etamax))
        
    for l in large_sources: # triangles
        ra, dec = l["ra"], l["dec"]
        plt.plot(ra, dec, transform=ax.get_transform('icrs'), ms=10,
                 mec=large_color, mfc="None", mew=2, marker="^", ls="", 
                label=r"$A$"+" "+r"$\geqslant$"+" "+str(area_max)+
                " pix"+r"${}^2$")
        
    # marker over accepted sources
    for v in vetted: # diamonds
        ra, dec = v["ra"], v["dec"]
        plt.plot(ra, dec, transform=ax.get_transform('icrs'), ms=10,
                 mec=vetted_color, mfc="None", mew=2, marker="D", ls="", 
                 label="accepted")

    # crosshair denoting target of interest
    if (toi != None):
        # circle?
        #circ = ptc.Circle((toi[0],toi[1]), radius=toi_sep_max/3600.0, 
        #                  transform=ax.get_transform('icrs'), fill=False,
        #                  ec="black", lw=2, ls="-.")
        #plt.gca().add_patch(circ)
        # or a crosshair?
        plt.plot([ra-10.0/3600.0, ra-5.0/3600.0], [dec,dec], 
                 transform=ax.get_transform('icrs'), linewidth=2, 
                 color="black", marker="")
        plt.plot([ra, ra], [dec-10.0/3600.0, dec-5.0/3600.0], 
                 transform=ax.get_transform('icrs'),  linewidth=2, 
                 color="black", marker="")
    
    # title
    topfile = re.sub(".*/", "", sub_file)
    title = topfile.replace(".fits","")
    title = r"$\mathtt{"+title.replace("_","\_")+"}$"+" transient candidates"
    plt.title(title, fontsize=16)
    # remove duplicates from legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))
    # add the legend
    ax.legend(by_label.values(), by_label.keys(), loc="lower center",
              bbox_to_anchor=[0.5,-0.1], fontsize=16, ncol=len(by_label), 
              fancybox=True)  

    if not(output):
        output = sub_file.replace(".fits", "_rejections.png")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def __triplet_plot(og_file, sub_hdu, og_hdu, ref_hdu, n, ntargets,  
                   wide=True, cmap="bone", title=None, plotdir=None):
    """
    Input:
        - difference image PrimaryHDU
        - science image PrimaryHDU
        - reference image PrimaryHDU
        - id of the candidate (i.e., for candidate #14, n=14)
        - total number of vetted candidates in the difference image 
        - whether to plot the triplets as 3 columns, 1 row (horizontally wide)
          or 3 rows, 1 column (vertically tall) (optional; default wide=True)
        - colourmap to apply to all images in the triplet (optional; default 
          "bone")
        - a title to include in all plots AND to append to all filenames 
          (optional; default None)
        - directory in which to store plots (optional; default is the location
          of the science image file)
    
    Plots a single [science image, reference image, difference image] triplet.
    
    Output: None
    """
    
    # load in data
    sub_data = sub_hdu.data
    og_data = og_hdu.data
    ref_data = ref_hdu.data

    # parameters for the plots and subplots based on whether wide=True
    plot_dict = {True:[(14,5), 131, 132, 133], 
                 False:[(5, 14), 311, 312, 313]}
    
    ## plot
    fig = plt.figure(figsize=plot_dict[wide][0])   

    # science image
    w = wcs.WCS(og_hdu.header) # wcs of science image
    ax = fig.add_subplot(plot_dict[wide][1], projection=w)
    mean, median, std = sigma_clipped_stats(og_data)
    ax.imshow(og_data, vmin=mean-5*std, vmax=mean+5*std, origin='lower', 
              cmap=cmap)
    
    # reference image 
    w2 = wcs.WCS(ref_hdu.header) # wcs of reference image
    ax2 = fig.add_subplot(plot_dict[wide][2], projection=w2)
    mean, median, std = sigma_clipped_stats(ref_data)
    ax2.imshow(ref_data, vmin=mean-5*std, vmax=mean+9*std, origin='lower', 
               cmap=cmap)
    
    # difference image 
    w3 = wcs.WCS(sub_hdu.header) # wcs of difference image
    ax3 = fig.add_subplot(plot_dict[wide][3], projection=w3)
    mean, median, std = sigma_clipped_stats(sub_data)
    ax3.imshow(sub_data, vmin=mean-5*std, vmax=mean+9*std, origin='lower', 
               cmap=cmap)
    
    ## express the candidate number as a string
    if ntargets < 100:
        if n < 10: nstr = "0"+str(n)
        else: nstr = str(n)
    else:
        if n < 10: nstr = "00"+str(n)
        elif n < 100: nstr = "0"+str(n)
        else: nstr = str(n)

    ## ticks, tick labels
    if wide: # RA under middle image, DEC left of leftmost image 
        ax.coords["ra"].set_ticklabel_visible(False)
        ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        ax.set_ylabel("Dec (J2000)", fontsize=16)
        ax2.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax2.coords["dec"].set_ticklabel_visible(False)
        ax2.set_xlabel("RA (J2000)", fontsize=16)
        ax3.coords["ra"].set_ticklabel_visible(False)
        ax3.coords["dec"].set_ticklabel_visible(False)
    else: # RA under bottom image, DEC left of middle image
        ax.coords["ra"].set_ticklabel_visible(False)
        ax.coords["dec"].set_ticklabel_visible(False)        
        ax2.coords["ra"].set_ticklabel_visible(False)        
        ax2.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        ax2.set_ylabel("Dec (J2000)", fontsize=16)
        ax3.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax3.coords["dec"].set_ticklabel_visible(False)    
        ax3.set_xlabel("RA (J2000)", fontsize=16)
    
    ## titles, output filenames
    if title:
        if wide: # title above the middle image 
            ax2.set_title(title, fontsize=15)
        else: # title above the topmost image
            ax.set_title(title, fontsize=15)
        figname = og_file.replace(".fits", f"_{title}_candidate{nstr}.png")
    else:
        figname = og_file.replace(".fits", f"_candidate{nstr}.png")   
        
    if plotdir: 
        figname = f'{plotdir}/{re.sub(".*/", "", figname)}'
    
    # save and close the figure
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    

def transient_detect(sub_file, og_file, ref_file, mask_file=None, 
                     thresh_sigma=5.0, pixelmin=20, 
                     dipole_width=2.0, dipole_fratio=5.0,
                     etamax=1.8, area_max=300, nsource_max=50, 
                     toi=None, toi_sep_min=None, toi_sep_max=None,
                     write=True, output=None, 
                     plot_distributions=False,
                     plot_rejections=False,
                     plots=["zoom og", "zoom ref", "zoom diff"], 
                     pixcoords=False,
                     sub_scale=None, og_scale=None, stampsize=200.0, 
                     crosshair_og="#fe019a", crosshair_sub="#5d06e9",
                     title=None, plotdir=None):
    """        
    Inputs:
        general:
        - subtracted image file
        - original science image file (can be background-subtracted or not)
        - original reference image file (can be background-subtracted or not)
        - a mask file (optional; default None)
        
        source detection and candidate rejection:
        - sigma threshold for source detection with image segmentation 
          (optional; default 3.0)
        - *minimum* number of isophote pixels, i.e. area (optional; default 20)
        - *maximum* allowed separation for any potential dipoles to be 
          considered real sources (optional; default 2.0"; setting this to None
          sets no maximum) 
        - *maximum* allowed flux ratio for dipoles (optional; default 5.0)
        - *maximum* allowed elongation for sources found by image segmentation 
          (optional; default 1.8; setting this to None sets no maximum)
        - *maximum* allowed pixel area for sources (optional; default 300)
        - *maximum* allowed total number of transients (optional; default 50; 
          setting this to None sets no maximum)
        - [ra,dec] for some target of interest (e.g., a candidate host galaxy)
          such that the distance to this target will be computed for every 
          candidate transient (optional; default None)
        - *minimum* separation between the target of interest and the transient
          (optional; default None; only relevant if TOI is supplied)
        - *maximum* separation between the target of interest and the transient 
          (optional; default None, only relevant if TOI is supplied)
          
        writing and plotting:
        - whether to write the source table (optional; default True)
        - name for the output source table (optional; default set below; only 
          relevant if write=True)
        - whether to plot the histograms of elongation, area, and a scatter 
          plot of elongation versus area (optional; default False)
        - whether to plot an image showing all objects in the difference image
          which did not pass candidate vetting and those which did, with 
          different markers indicating the reason for rejection (optional; 
          default False)
        - an array of which plots to produce, where valid options are:
              (1) "full" - the full-frame subtracted image
              (2) "zoom og" - postage stamp of the original science image 
              (3) "zoom ref" - postage stamp of the original reference image
              (4) "zoom diff" - postage stamp of the subtracted image 
              (optional; default is ["zoom og", "zoom ref", "zoom diff"]) 
        
        
        the following are relevant only if plots are requested:
        - whether to use the pixel coord
        - scale to apply to the difference images (optional; default "asinh"; 
          options are "linear", "log", "asinh")
        - scale to apply to the science/reference images (optional; default
          "asinh"; options are "linear", "log", "asinh")
        - size of the transient stamp in pixels (optional; default 200.0)
        - colour for crosshair on transient in science/ref images (optional; 
          default ~hot pink)
        - colour for crosshair on transient in difference images (optional;
          default ~purple-blue)     
        - a title to include in the titles of all plots AND filenames of all 
          plots, except the rejection plot (optional; default None)
        - name for the directory in which to store all plots (optional; 
          default set below)
    
    Looks for sources with flux > sigma*std, where std is the standard 
    deviation of the good pixels in the subtracted image. Sources must also be 
    made up of at least pixelmin pixels. From these, selects sources below some 
    elongation limit to try to prevent obvious residuals from being detected as 
    sources. For each candidate transient source, optionally plots the full 
    subtracted image and "postage stamps" of the transient on the original 
    science image, reference image and/or subtracted image. 
    
    Output: a table of sources with their properties (coordinates, area, 
    elongation, separation from a target of interest (if relevant), etc.)
    """
    
    data = fits.getdata(sub_file) # subfile data
    hdr = fits.getheader(sub_file)
    tmp_data = fits.getdata(ref_file)
    
    # build a mask, including the template's bad pixels too
    mask = np.logical_or(data==0, np.isnan(data))
    tmpmask = np.logical_or(tmp_data==0, np.isnan(tmp_data))
    mask = np.logical_or(mask, tmpmask)
    if mask_file:
        mask = np.logical_or(mask, fits.getdata(mask_file))
        
    data = np.ma.masked_where(data==0.0, data) # mask all bad pix

    ### SOURCE DETECTION ######################################################
    ## use image segmentation to find sources with an area > pixelmin pix**2 
    ## which are above the threshold sigma*std 
    try: 
        std = float(hdr['DSIGE00']) # good pixels standard deviation 
    except KeyError:
        std = np.std(data)

    segm = detect_sources(data, thresh_sigma*std, npixels=pixelmin,
                          mask=mask)          
    # use the segmentation image to get the source properties 
    cat = source_properties(data, segm, mask=mask)
    
    # do the same with the inverse of the image to look for dipoles
    segm_inv = detect_sources((-1.0)*data, thresh_sigma*std, 
                              npixels=pixelmin, mask=mask)
    cat_inv = source_properties((-1.0)*data, segm_inv, mask=mask)
    # get the catalog and coordinates for sources
    try:
        tbl = cat.to_table()
    except ValueError:
        print("SourceCatalog contains no sources. Exiting.")
        return
    w = wcs.WCS(hdr)
    tbl["ra"], tbl["dec"] = w.all_pix2world(tbl["xcentroid"], 
                                            tbl["ycentroid"], 1)    
    # save a copy of the original, unvetted table
    tbl["source_sum/area"] = tbl["source_sum"]/tbl["area"] # flux per pixel
    tbl_novetting = tbl.copy()
        
    ### INFORMATIVE PLOTS #####################################################
    if plot_distributions:
        ## histogram of elongation distribution
        plt.figure()
        elongs = tbl["elongation"].data
        nbelow = len(tbl[tbl["elongation"]<etamax])
        nabove = len(tbl[tbl["elongation"]>etamax])
        mean, med, std = sigma_clipped_stats(elongs) 
        plt.hist(elongs, bins=18, range=(min(1,mean-std),max(10,mean+std)), 
                 color="#90e4c1", alpha=0.5)
                 
        plt.axvline(mean, color="blue", label=r"$\mu$")
        plt.axvline(mean+std, color="blue", ls="--", 
                    label=r"$\mu$"+"±"+r"$\sigma$")
        plt.axvline(mean-std, color="blue", ls="--")
        plt.axvline(etamax, color="red", lw=2.5, label=r"$\eta_{max}$")
        plt.xlabel("Elongation", fontsize=15)
        plt.xlim(min(1,mean-std), max(11,mean+std+1))
        plt.ylabel("Counts", fontsize=15)
        plt.gca().tick_params(which='major', labelsize=10)
        plt.grid()
        
        textboxstr = r"$\mu - \eta_{max} = $"+"%.2f"%(mean-etamax)+"\n"
        textboxstr += r"$\eta < \eta_{max} = $"+str(nbelow)+"\n"
        textboxstr += r"$\eta > \eta_{max} = $"+str(nabove)+"\n"
        textboxstr += r"$f_{used} = $"+"%.2f"%(nbelow/(nbelow+nabove))
        plt.text(1.05, 0.66, textboxstr,transform=plt.gca().transAxes, 
                 fontsize=13, bbox=dict(facecolor="white", alpha=0.5))
        plt.text(0.92, 0.5, r"$\dots$", transform=plt.gca().transAxes, 
                 fontsize=20)
        plt.legend(loc=[1.03, 0.32], fancybox=True, fontsize=13)
        
        if title:
            plt.title(f"{title}: elongation distribution", fontsize=13)
            plt.savefig(og_file.replace(".fits", f"_{title}_elongs.png"), 
                        bbox_inches="tight")
        else:
            plt.title("elongation distribution", fontsize=13)
            plt.savefig(og_file.replace(".fits", "_elongs.png"), 
                        bbox_inches="tight")            
        plt.close()
    
        ## histogram of area distribution   
        plt.figure()
        areas = tbl["area"].value
        nbelow = len(tbl[areas<area_max])
        nabove = len(tbl[areas>area_max])
        mean, med, std = sigma_clipped_stats(areas) 
        plt.hist(areas, bins=20, color="#c875c4", alpha=0.5)
                 
        plt.axvline(mean, color="red", label=r"$\mu$")
        plt.axvline(mean+std, color="red", ls="--", 
                    label=r"$\mu$"+"±"+r"$\sigma$")
        plt.axvline(mean-std, color="red", ls="--")
        plt.axvline(area_max, color="blue", lw=2.5, label=r"$A_{max}$")
        plt.xlabel("Area [pix"+r"${}^2$"+"]", fontsize=15)
        plt.ylabel("Counts", fontsize=15)
        plt.gca().tick_params(which='major', labelsize=10)
        plt.grid()
        plt.xscale("log")
        plt.yscale("log")
        
        textboxstr = r"$\mu - A_{max} = $"+"%.2f"%(mean-area_max)+"\n"
        textboxstr += r"$A < A_{max} = $"+str(nbelow)+"\n"
        textboxstr += r"$A > A_{max} = $"+str(nabove)+"\n"
        textboxstr += r"$f_{used} = $"+"%.2f"%(nbelow/(nbelow+nabove))
        plt.text(1.05, 0.66, textboxstr,transform=plt.gca().transAxes, 
                 fontsize=13, bbox=dict(facecolor="white", alpha=0.5))
        plt.legend(loc=[1.03, 0.32], fancybox=True, fontsize=13)
        
        if title:
            plt.title(f"{title}: area distribution", fontsize=13)
            plt.savefig(og_file.replace(".fits", f"_{title}_areas.png"), 
                        bbox_inches="tight")
        else:
            plt.title("area distribution")
            plt.savefig(og_file.replace(".fits", "_areas.png"), 
                        bbox_inches="tight") 
        plt.close()
    
        ## elongation versus pixel area
        plt.figure()    
        elongsgood = [elongs[i] for i in range(len(elongs)) if (
                      elongs[i]<etamax and areas[i]<area_max)]
        areasgood = [areas[i] for i in range(len(elongs)) if (
                     elongs[i]<etamax and areas[i]<area_max)]
    
        elongsbad = [elongs[i] for i in range(len(elongs)) if (
                     elongs[i]>etamax or areas[i]>area_max)]
        areasbad = [areas[i] for i in range(len(elongs)) if (
                    elongs[i]>etamax or areas[i]>area_max)]
        
        # elongation on x axis, area on y axis            
        plt.scatter(elongsgood, areasgood, marker="o", color="#5ca904", s=12)
        plt.scatter(elongsbad, areasbad, marker="s", color="#fe019a", s=12)
        # means and maxima
        mean, med, std = sigma_clipped_stats(elongs)              
        plt.axvline(mean, ls="--", color="black", label=r"$\mu$")
        plt.axvline(etamax, color="#030aa7", lw=2.5, label=r"$\eta_{max}$")
        mean, med, std = sigma_clipped_stats(areas) 
        plt.axhline(mean, ls="--", color="black")
        plt.axhline(area_max, color="#448ee4", lw=2.5, label=r"$A_{max}$")
        # allowed region of parameter space 
        rect = ptc.Rectangle((0,0), etamax, area_max, fill=False, 
                             hatch="//", lw=0.5, color="black")
        plt.gca().add_patch(rect)
        # labels, scales 
        plt.gca().tick_params(which='major', labelsize=10)
        plt.xlabel("Elongation", fontsize=15)
        plt.ylabel("Area [pix"+r"${}^2$"+"]", fontsize=15)    
        plt.xscale("log")
        plt.yscale("log")
        
        # fraction of sources which are used 
        f_used = len(elongsgood)/(len(elongsgood)+len(elongsbad))
        textboxstr = r"$f_{used} = $"+"%.2f"%(f_used)
        plt.text(1.03, 0.93, textboxstr, transform=plt.gca().transAxes, 
                 fontsize=13, bbox=dict(facecolor="white", alpha=0.5))
            
        plt.legend(loc=[1.03,0.62], fancybox=True, fontsize=13) # legend 

        if title:
            plt.title(f"{title}: elongations and areas", fontsize=13)
            plt.savefig(og_file.replace(".fits", 
                                        f"_{title}_elongs_areas.png"), 
                        bbox_inches="tight")
        else:
            plt.title("elongations and areas", fontsize=13)
            plt.savefig(og_file.replace(".fits", "_elongs_areas.png"), 
                        bbox_inches="tight")            
        plt.close()
    
    ### CANDIDATE VETTING #####################################################
    ## look for dipoles by looking for sources in (-1)*difference and cross-  
    ## matching to the segmentation image 
    if dipole_width:
        idx_rem = [] # indices to remove
        try:
            inv = cat_inv.to_table()
            inv["ra"], inv["dec"] = w.all_pix2world(inv["xcentroid"], 
                                                    inv["ycentroid"], 1)
            inv["source_sum/area"] = inv["source_sum"]/inv["area"] # flux/pixel
            coords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, frame="icrs")
            inv_coords = SkyCoord(inv["ra"]*u.deg, inv["dec"]*u.deg, 
                                  frame="icrs")        
            # indices of sources within <dipole_width> arcsec of each other
            idx_inv, idx, d2d, d3d = coords.search_around_sky(inv_coords, 
                                                        dipole_width*u.arcsec)
            
            # if they correspond to the same peak, the amplitude of one part 
            # of the dipole should be no more than fratio times the amplitude 
            # of the other part of the dipole
            for n in range(len(idx)):
                i, j = idx[n], idx_inv[n]
                
                ratio = tbl[i]["source_sum/area"]/inv[j]["source_sum/area"]
                if ((1.0/dipole_fratio) <= ratio <= dipole_fratio): 
                    idx_rem.append(i)
            
            if len(idx_rem) > 0:
                print(f'\n{len(idx_rem)} likely dipole(s) (width < '+
                      f'{dipole_width}", fratio < {dipole_fratio}) '+
                      'detected and removed')
                tbl.remove_rows(idx_rem)
            tbl_nodipoles = tbl.copy() # tbl with no dipoles
            
        except ValueError:
            print("The inversion of the difference image, (-1.0)*diff, does "+
                  "not contain any sources. Will continue without searching "+
                  "for dipoles.")
            tbl_nodipoles = tbl.copy() # tbl with no dipoles
    
    ## restrict based on elongation 
    if etamax:
        premasklen = len(tbl)
        mask = tbl["elongation"] < etamax 
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print(f"\n{premasklen-postmasklen} source(s) with "+
                  f"elongation >{etamax} removed")
        tbl_noelong = tbl.copy()
            
    ## restrict based on maximum pixel area 
    if area_max:
        premasklen = len(tbl)
        mask = tbl["area"].value < area_max
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print(f"\n{premasklen-postmasklen} source(s) with "+
                  f"area >{area_max} pix**2 removed")
    
    vetted = tbl.copy()

    ## plot rejected/non-rejected candidates, if desired 
    if plot_rejections:
        dipoles = tbl_novetting[idx_rem]
        elongated_sources = tbl_nodipoles[tbl_nodipoles["elongation"] >= 
                                          etamax]
        large_sources = tbl_noelong[tbl_noelong["area"].value >= area_max]
        
        # colours below might be out of date (could be improved)
        __rejection_plot(sub_file,
                         dipoles, elongated_sources, large_sources, vetted,
                         dipole_width, etamax, area_max, nsource_max,
                         toi, #toi_sep_max, 
                         dipole_color="black", 
                         elong_color="#be03fd", 
                         large_color="#fcc006", 
                         vetted_color="#53fca1",
                         cmap="coolwarm",
                         output=None)

    ## check that the number of detected sources is believable 
    if nsource_max:
        if len(tbl) > nsource_max:
           print(f"\nSourceCatalog contains {len(tbl)} sources across the "+
                 f"entire image, which is above the limit of {nsource_max}. "+
                 "The subtraction may have large image_registration errors. "+
                 "Exiting.")
           return    

    ## if provided, only look for sources within a certain angular separation
    ## of the target of interest 
    if toi != None: 
        toi_coord = SkyCoord(toi[0]*u.deg, toi[1]*u.deg, frame="icrs")
        trans_coord = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, frame="icrs")
        sep = toi_coord.separation(trans_coord).arcsecond
        tbl["sep"] = sep # add a column for separation
        mask = tbl["sep"] < toi_sep_max
        tbl = tbl[mask]
        mask = tbl["sep"] > toi_sep_min
        tbl = tbl[mask]  
      
    if len(tbl) == 0:
        print("\nAfter filtering sources, no viable transient candidates "+
              "remain. No table will be written.")
        return
 
    ### PREPARE CANDIDATE TABLE ###############################################       
    ## sort by flux per pixel and print out number of sources found
    tbl["source_sum/area"] = tbl["source_sum"]/tbl["area"]
    tbl.sort("source_sum/area")  
    tbl.reverse()
    print(f"\n{len(tbl)} candidate(s) found and passed vetting.")
    
    ## get rid of "id" column and empty columns: 
    # (sky_centroid, sky_centroid_icrs, source_sum_err, background_sum, 
    # background_sum_err, background_mean, background_at_centroid) 
    tbl.remove_column("id")
    colnames = tbl.colnames
    colnames = [c for c in colnames.copy() if not(tbl[c][0] == 'None')]
    tbl = tbl[colnames]
    
    ## add in other useful columns 
    tbl["MJD"] = [float(hdr["MJDATE"]) for i in range(len(tbl))]
    tbl["science"] = [re.sub(".*/", "", og_file) for i in range(len(tbl))]
    tbl["template"] = [re.sub(".*/", "", ref_file) for i in range(len(tbl))]
    tbl["difference"] = [re.sub(".*/", "", sub_file) for i in range(len(tbl))]
    
    if write:
        if not(output):
            output = og_file.replace(".fits", "_candidates.fits")
        tbl.write(output, overwrite=True, format="ascii")
    
    ### PLOT ##################################################################
    if type(plots) in (list, np.ndarray, np.array):
        transient_plot(sub_file=sub_file, og_file=og_file, ref_file=ref_file, 
                       tbl=tbl, 
                       pixcoords=pixcoords, 
                       toi=toi, 
                       plots=plots, 
                       sub_scale=sub_scale, og_scale=og_scale, 
                       stampsize=stampsize, 
                       crosshair_og=crosshair_og, crosshair_sub=crosshair_sub, 
                       title_append=title, plotdir=plotdir)    
    return tbl


def transient_plot(sub_file, og_file, ref_file, tbl, 
                   pixcoords=False,
                   toi=None, 
                   plots=["zoom og", "zoom ref", "zoom diff"], 
                   sub_scale=None, og_scale=None, 
                   stampsize=200.0, 
                   crosshair_og="#fe019a", crosshair_sub="#5d06e9", 
                   title_append=None, plotdir=None):
    """
    Inputs:
        general:￼
        - difference image file
        - original science image file (can be background-subtracted or not)
        - original reference image file (can be background-subtracted or not)
        - a table of candidate transient source(s) found with 
          transient_detect() or some other tool (can be the name of a table 
          file or the table itself)
          note: if pixcoords=False, the table must have at least the columns 
          "ra" and "dec"; if pixcoords=True, the table must have at least the 
          columns "xcentroid" and "ycentroid"
        - whether to use the pixel coordinates of the transients when plotting 
          rather than the ra, dec (optional; default False; recommended to use
          in cases where it is not clear that the WCS of the science and 
          reference images matches exactly)
        - [ra,dec] for some target of interest (e.g., a candidate host galaxy)
          such that a crosshair is also plotted at its location (optional; 
          default None)
        
        plotting specifics:
        - an array of which plots to produce, where valid options are:
              (1) "full" - the full-frame subtracted image
              (2) "zoom og" - postage stamp of the original science image 
              (3) "zoom ref" - postage stamp of the original reference image
              (4) "zoom diff" - postage stamp of the subtracted image 
              (optional; default is ["zoom og", "zoom ref", "zoom diff"])
              
        - scale to apply to the difference images (optional; default "asinh"; 
          options are "linear", "log", "asinh")
        - scale to apply to the science/reference images (optional; default
          "asinh"; options are "linear", "log", "asinh")
        - size of the transient stamp in pixels (optional; default 200.0)
        - colour for crosshair on transient in science/ref images (optional; 
          default hot pink)
        - colour for crosshair on transient in difference images (optional;
          default purple-blue)
        - a title to include in all plots AND all filenames for the plots 
          (optional; default None)
        - name for the directory in which to store all plots (optional; 
          default set below)   
        
    Output: None
    """

    # check if table is a filename or pre-loaded table
    if type(tbl) == str:
        tbl = Table.read(tbl, format="ascii")
    
    if not(pixcoords):
        targets_sci= [tbl["ra"].data, tbl["dec"].data]
        targets_ref = [tbl["ra"].data, tbl["dec"].data]
    else:
        pix = [tbl["xcentroid"].data, tbl["ycentroid"].data]
        wsci = wcs.WCS(fits.getheader(sub_file))
        wref = wcs.WCS(fits.getheader(ref_file))
        rasci, decsci = wsci.all_pix2world(pix[0], pix[1], 1)
        raref, decref = wref.all_pix2world(pix[0], pix[1], 1)
        targets_sci = [rasci, decsci]
        targets_ref = [raref, decref]
    
    if not(type(plots) in (list, np.ndarray, np.array)):
        print("\ntransient_plot() was called, but no plots were requested "+
              "via the <plots> arg. Exiting.")
        return
    elif len(plots) == 0:
        print("\ntransient_plot() was called, but no plots were requested "+
              "via the <plots> arg. Exiting.")
        return
    
    ntargets = len(targets_sci[0])
    for n in range(ntargets):
        if ntargets < 100:
            if n < 10: nstr = "0"+str(n)
            else: nstr = str(n)
        else:
            if n < 10: nstr = "00"+str(n)
            elif n < 100: nstr = "0"+str(n)
            else: nstr = str(n)
                
        ### set titles 
        if title_append:
            title = f"{title_append} difference image: candidate {nstr}"
            title_og = f"{title_append} original image: candidate {nstr}"
            title_ref = f"{title_append} reference image: candidate {nstr}"
        else:
            title = f"difference image: candidate {nstr}"
            title_og = f"original image: candidate {nstr}"
            title_ref = f"reference image: candidate {nstr}"
    
        # setting output figure directory
        if plotdir:
            if plotdir[-1] == "/": plotdir = plotdir[-1]        
        
        ### full-frame difference image #######################################
        if "full" in plots:
            
            # set output figure title
            if title_append:
                full_output = sub_file.replace(".fits", 
                             f"_{title_append}_diff_candidate{nstr}.png")                
            else:
                full_output = sub_file.replace(".fits", 
                                               f"_diff_candidate{nstr}.png")                
            if plotdir:
                full_output = f'{plotdir}/{re.sub(".*/", "", full_output)}'
                
            make_image(sub_file, 
                       scale=sub_scale, cmap="coolwarm", label="", title=title, 
                       output=full_output, 
                       target=toi,
                       target_small=[targets_sci[0][n], targets_sci[1][n]],
                       crosshair_small=crosshair_sub)
        
        ### small region of science image ####################################
        if "zoom og" in plots:
            # set output figure name
            if title_append:
                zoom_og_output = og_file.replace(".fits", 
                        f"_{title_append}_zoomed_sci_candidate{nstr}.png")
            else:
                zoom_og_output = og_file.replace(".fits", 
                                        f"_zoomed_sci_candidate{nstr}.png")                
            if plotdir:
                zoom_og_output = f'{plotdir}/{re.sub(".*/","",zoom_og_output)}'
                
            transient_stamp(og_file, [targets_sci[0][n], targets_sci[1][n]], 
                            stampsize, 
                            scale=og_scale, cmap="viridis", 
                            crosshair=crosshair_og, title=title_og, 
                            output=zoom_og_output, toi=toi)

        ### small region of reference image ###################################
        if "zoom ref" in plots:
            # set output figure name
            if title_append:
                zoom_ref_output = og_file.replace(".fits", 
                        f"_{title_append}_zoomed_ref_candidate{nstr}.png")
            else:
                zoom_ref_output = og_file.replace(".fits", 
                                        f"_zoomed_ref_candidate{nstr}.png")
            if plotdir:
                zoom_ref_output = f'{plotdir}/'
                zoom_ref_output += f'{re.sub(".*/", "", zoom_ref_output)}'
                
            transient_stamp(ref_file, [targets_ref[0][n], targets_ref[1][n]], 
                            stampsize, 
                            scale=og_scale, cmap="viridis", 
                            crosshair=crosshair_og, title=title_ref, 
                            output=zoom_ref_output, toi=toi)
        
        ### small region of difference image ##################################
        if "zoom diff" in plots:
            # set output figure name
            if title_append:
                zoom_diff_output = sub_file.replace(".fits", 
                        f"_{title_append}_zoomed_diff_candidate{nstr}.png")
            else:
                zoom_diff_output = sub_file.replace(".fits", 
                                        f"_zoomed_diff_candidate{nstr}.png")
            if plotdir:
                zoom_diff_output = f'{plotdir}/'
                zoom_diff_output += f'{re.sub(".*/", "", zoom_diff_output)}'
                
            transient_stamp(sub_file, [targets_sci[0][n], targets_sci[1][n]], 
                            stampsize, scale=sub_scale, cmap="coolwarm", 
                            label="ADU", crosshair=crosshair_sub, title=title, 
                            output=zoom_diff_output, toi=toi)    


def transient_triplets(sub_file, og_file, ref_file, tbl, pixcoords=False, 
                       size=200, 
                       cropmode="extend", write=True, output=None,
                       plot=False, wide=True, cmap="bone", title=None, 
                       plotdir=None):
    """    
    Inputs:
        - difference image file
        - original science image file (can be background-subtracted or not)
        - original reference image file (can be background-subtracted or not)
        - a table of candidate transient source(s) found with 
          transient_detect() or some other tool (can be the name of a table 
          file or the table itself)
          note: if pixcoords=False, the table must have at least the columns 
          "ra" and "dec"; if pixcoords=True, the table must have at least the 
          columns "xcentroid" and "ycentroid"
        - whether to use the pixel coordinates of the transients when plotting 
          rather than the ra, dec (optional; default False; recommended to use
          in cases where it is not clear that the WCS of the science and 
          reference images matches exactly)         
        - the size of each 2D array in the triplet, i.e. the size of the 
          stamp to obtain around the transient(s) (optional; default 200 pix)
        - mode to use for crop_WCS (optional; default "extend"; options are
          "truncate" and "extend")
        - whether to write the produced triplet(s) to a .npy file (optional; 
          default True)
        - name for output .npy file (or some other file) (optional; default 
          set below)
        - whether to plot all of the triplets (optional; default False)
        
        only relevant if plot=True:
        - whether to plot the triplets as 3 columns, 1 row (horizontally wide)
          or 3 rows, 1 column (vertically tall) (optional; default wide=True)
        - colourmap to apply to all images in the triplet (optional; default 
          "bone")
        - a title to include in all plots AND to append to all filenames 
          (optional; default None)
        - name for the directory in which to store all plots (optional; 
          default set below)
    
    From a table of candidate transients and the corresponding difference, 
    science, and reference images, builds a triplet: three "postage stamps"
    around the candidate showing the science image, reference image, and 
    difference image. Optionally writes the triplet to a .npy file.
    
    Output: a numpy array with shape (N, 3, size, size), where N is the number 
    of rows in the input table (i.e., no. of candidate transients) and the 3 
    sub-arrays represent cropped sections of the science image, reference 
    image, and difference image
    """

    # check if table is a filename or pre-loaded table
    if type(tbl) == str:
        tbl = Table.read(tbl, format="ascii")
    
    # get the targets and count how many in total
    if not(pixcoords):
        targets_sci= [tbl["ra"].data, tbl["dec"].data]
        targets_ref = [tbl["ra"].data, tbl["dec"].data]
    else:
        pix = [tbl["xcentroid"].data, tbl["ycentroid"].data]
        wsci = wcs.WCS(fits.getheader(sub_file))
        wref = wcs.WCS(fits.getheader(ref_file))
        rasci, decsci = wsci.all_pix2world(pix[0], pix[1], 1)
        raref, decref = wref.all_pix2world(pix[0], pix[1], 1)
        targets_sci = [rasci, decsci]
        targets_ref = [raref, decref]
          
    ntargets = len(targets_sci[0])
    triplets = []  
    
    ## iterate through all of the targets
    for n in range(ntargets):
        
        # get cropped data   
        sub_hdu = crop_WCS(sub_file, targets_sci[0][n], targets_sci[1][n], 
                           size=size, mode=cropmode, write=False)
        sub_data = sub_hdu.data        
        og_hdu = crop_WCS(og_file, targets_sci[0][n], targets_sci[1][n], 
                          size=size, mode=cropmode, write=False)
        og_data = og_hdu.data        
        ref_hdu = crop_WCS(ref_file, targets_ref[0][n], targets_ref[1][n], 
                           size=size, mode=cropmode, write=False)
        ref_data = ref_hdu.data       
        
        # make the triplet and add it to the list of triplets
        trip = np.array([og_data, ref_data, sub_data])
        triplets.append(trip)
        
        if plot: # plot this triplet, if desired
            if plotdir[-1] == "/":
                plotdir = plotdir[-1]
                
            __triplet_plot(og_file=og_file, 
                           sub_hdu=sub_hdu, og_hdu=og_hdu, ref_hdu=ref_hdu, 
                           n=n, ntargets=ntargets, 
                           wide=wide, cmap=cmap, title=title, plotdir=plotdir)

    ## build the final triplet stack, write it (optional)
    triplets = np.stack(triplets) # (3, size, size) --> (N, 3, size, size)   
    if write:
        if not(output):
            output = og_file.replace(".fits", "_candidates_triplets.npy")
        np.save(output, triplets)
        
    return triplets


def im_contains(im_file, ra, dec, exclusion=None, mute=False):
    """
    Input:
        - image of interest
        - ra, dec of interest for a single source 
        - fraction of edge pixels to ignore (optional; default None; 
          e.g. if exclusion=0.05 and image is 1000x1000, will only consider pix
          in range [50, 950] valid for both axes)
        - mute the warnings (optional; default False)
        
    Check if some coordinate at ra, dec is within the bounds of the image. 
    
    Output: bool indicating whether coordinate is in image 
    """
    data = fits.getdata(im_file)
    w = wcs.WCS(fits.getheader(im_file))
    
    xlims = [0, data.shape[1]]
    ylims = [0, data.shape[0]]
    
    if exclusion:
        xlims = [xlims[1]*exclusion, xlims[1]*(1.0-exclusion)]
        ylims = [ylims[1]*exclusion, ylims[1]*(1.0-exclusion)]
    
    try: 
        x, y = w.all_world2pix(ra, dec, 1)
    except wcs.NoConvergence:
        if not(mute): print("\nWCS could not converge, exiting")
        return 
    
    if (xlims[0] < x < xlims[1]) and (ylims[0] < y < ylims[1]): return True
    else: return  
        
###############################################################################
#### MISCELLANEOUS PLOTTING ####
    
def make_image(im_file, mask_file=None, scale=None, cmap="bone", label=None,
               title=None, output=None, target=None, target_small=None,
               crosshair_big="black", crosshair_small="#fe019a"):
    """
    Input: 
        - image of interest
        - bad pixels mask (optional; default None)
        - scale to use for the image (optional; default None (linear), options 
          are "linear", "log", "asinh")
        - colourmap to use for the image (optional; default is "bone")
        - label to apply to the colourbar (optional; default set below)
        - title for the image (optional; default None)
        - name for the output file (optional; default set below)
        - [ra,dec] for a target at which to place a crosshair (optional; 
          default None)
        - [ra,dec] for a second target at which to place a smaller crosshair 
          (optional; default None)
        - colour for the big crosshair (optional; default "black")
        - colour for the small crosshair (optional; default hot pink)
          
    Output: None
    """
    
    image_data = fits.getdata(im_file)
    image_header = fits.getheader(im_file)
    
    plt.figure(figsize=(14,13))
    
    # show WCS     
    w = wcs.WCS(image_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    if mask_file:
        mask = fits.getdata(mask_file)
        image_data_masked = np.ma.masked_where(mask, image_data)
        image_data = np.ma.filled(image_data_masked, 0)
    
    if not scale or (scale == "linear"): # if no scale to apply 
        scale = "linear"
        mean, median, std = sigma_clipped_stats(image_data)
        plt.imshow(image_data, vmin=mean-5*std, vmax=mean+9*std, cmap=cmap, 
                   aspect=1, interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        if label == None:
            cb.set_label(label="ADU", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    elif scale == "log": # if we want to apply a log scale 
        image_data_log = np.log10(image_data)
        lognorm = simple_norm(image_data_log, "log", percent=99.0)
        plt.imshow(image_data_log, cmap=cmap, aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        if label == None:
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    elif scale == "asinh": # asinh scale 
        image_data_asinh = np.arcsinh(image_data)
        asinhnorm = simple_norm(image_data_asinh, "asinh")
        plt.imshow(image_data_asinh, cmap=cmap, aspect=1, norm=asinhnorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        if label == None:
            cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)

    cb.ax.tick_params(which='major', labelsize=15)        
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(title):
        topfile = re.sub(".*/", "", im_file)
        title = topfile.replace(".fits","")
        title = r"$\mathtt{"+title.replace("_","\_")+"}$"+" data"
    if not(output):
        output = im_file.replace(".fits", f"_{scale}.png")

    if target:
        ra, dec = target
        plt.gca().plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
               transform=plt.gca().get_transform('icrs'), linewidth=2, 
               color=crosshair_big, marker="")
        plt.gca().plot([ra, ra], [dec+10.0/3600.0, dec+5.0/3600.0], 
               transform=plt.gca().get_transform('icrs'),  linewidth=2, 
               color=crosshair_big, marker="")
    if target_small:
        ra, dec = target_small
        plt.gca().plot([ra-10.0/3600.0, ra-5.0/3600.0], [dec,dec], 
               transform=plt.gca().get_transform('icrs'), linewidth=2, 
               color=crosshair_small, marker="")
        plt.gca().plot([ra, ra], [dec-10.0/3600.0, dec-5.0/3600.0], 
               transform=plt.gca().get_transform('icrs'),  linewidth=2, 
               color=crosshair_small, marker="")

    plt.title(title, fontsize=15)
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def transient_stamp(im_file, target, size=200.0, cropmode="truncate", 
                    scale=None, cmap="bone", label=None, crosshair="#fe019a", 
                    title=None, output=None, toi=None):
    """
    Inputs: 
        - image of interest
        - [ra, dec] of a candidate transient source
        - size of the zoomed-in region around the transient to plot in pixels 
          (optional; default 200.0)
        - mode to use for crop_WCS (optional; default "truncate"; options are
          "truncate" and "extend")
        - scale to apply to the image (optional; default None (linear); options 
          are "linear", "log", "asinh")
        - colourmap to use for the image (optional; default is "bone")
        - label to apply to the colourbar (optional; defaults set below)
        - colour for the crosshairs (optional; default is ~ hot pink)
        - title for the image (optional; defaults set below)
        - name for the output file (optional; defaults set below)
        - [ra, dec] of some other source of interest (optional; default None)
        
    Output: None
    """
    
    ra, dec = target
    imhdu = crop_WCS(im_file, ra, dec, size, mode=cropmode, write=False)
    if imhdu == None:
        print("\nCropping was unsucessful, so a stamp cannot be produced. "+
              "Exiting. ")
        return
        
    image_data = imhdu.data
    image_header = imhdu.header

    plt.figure(figsize=(14,13))
    
    # show WCS     
    w = wcs.WCS(image_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    if not scale or (scale == "linear"): # if no scale to apply 
        scale = "linear"
        mean, median, std = sigma_clipped_stats(image_data)
        plt.imshow(image_data, vmin=mean-5*std, vmax=mean+9*std, cmap=cmap, 
                   aspect=1, interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        if label == None:
            cb.set_label(label="ADU", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    elif scale == "log": # if we want to apply a log scale 
        image_data_log = np.log10(image_data)
        lognorm = simple_norm(image_data_log, "log", percent=99.0)
        plt.imshow(image_data_log, cmap=cmap, aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        if label == None:
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    elif scale == "asinh": # asinh scale 
        image_data_asinh = np.arcsinh(image_data)
        asinhnorm = simple_norm(image_data_asinh, "asinh")
        plt.imshow(image_data_asinh, cmap=cmap, aspect=1, norm=asinhnorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        if label == None:
            cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)

    cb.ax.tick_params(which='major', labelsize=15)        
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(title):
        topfile = re.sub(".*/", "", im_file)
        title = topfile.replace(".fits","")
        title = r"$\mathtt{"+title.replace("_","\_")+"}$"+" data (stamp)"
    if not(output):
        output = im_file.replace(".fits", f"_stamp_{scale}.png")

    plt.gca().plot([ra-10.0/3600.0, ra-5.0/3600.0], [dec,dec], 
           transform=plt.gca().get_transform('icrs'), linewidth=2, 
           color=crosshair, marker="")
    plt.gca().plot([ra, ra], [dec-10.0/3600.0, dec-5.0/3600.0], 
           transform=plt.gca().get_transform('icrs'),  linewidth=2, 
           color=crosshair, marker="")
    
    # textbox indicating the RA, Dec of the candidate transient source 
    textstr = r"$\alpha = $"+"%.5f\n"%ra+r"$\delta = $"+"%.5f"%dec
    
    if toi: # if a target of interest is given
        toi_coord = SkyCoord(toi[0]*u.deg, toi[1]*u.deg, frame="icrs")
        trans_coord = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
        sep = toi_coord.separation(trans_coord).arcsecond
        textstr+='\n'+r'$s = $'+'%.2f'%sep+'"'
        
    box = dict(boxstyle="square", facecolor="white", alpha=0.8)
    plt.text(0.05, 0.9, transform=ax.transAxes, s=textstr, 
             bbox=box, fontsize=20)    
        
    plt.title(title, fontsize=15)
    plt.savefig(output, bbox_inches="tight")
    plt.close()                    
    
