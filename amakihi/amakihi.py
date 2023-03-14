#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Thu Aug  1 13:58:03 2019
.. @author: Nicholas Vieira
.. @amakihi.py 

Big module which handles all of the actual image differencing. 

**References:**

[1] Alard, C., & Lupton, R. H. 1998, ApJ, 503, 325
    https://ui.adsabs.harvard.edu/abs/1998ApJ...503..325A/abstract

"""

# misc
import os
from subprocess import run, PIPE, CalledProcessError
import numpy as np
import re

# astropy
from astropy.io import fits
from astropy import wcs
import astropy.units as u 
from astropy.stats import (SigmaClip, 
                           gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm)
from astropy.coordinates import SkyCoord
from photutils import Background2D, MedianBackground
from photutils import make_source_mask, detect_sources#, source_properties
from photutils.segmentation import SourceCatalog

## for speedy FFTs
#import pyfftw
#import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs
#pyfftw.interfaces.cache.enable()

# amakihi 
from .ePSF import build_ePSF_imsegm, ePSF_FWHM
from .plotting import __plot_hotpants

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)


###############################################################################
### USED EVERYWHERE ###########################################################

from photutils.segmentation.catalog import DEFAULT_COLUMNS

REQ_COLUMNS = DEFAULT_COLUMNS + ["elongation"]

###############################################################################
#### IMAGE DIFFERENCING WITH HOTPANTS #########################################

def get_substamps(science_file, template_file, 
                  sci_mask_file=None, tmp_mask_file=None, 
                  thresh_sigma=3.0, etamax=2.0, areamax=400.0, sepmax=5.0, 
                  mode="mean",
                  output=None, verbose=False):
    """Get and write a substamps file containing the coordinates of good 
    sources for hotpants which are present in both the science and template
    images. 
    
    Arguments
    ---------
    science_file, template_file : str
        Science and template images fits files 
    sci_mask_file : str, optional
        Fits files for bad pixel masks for the science and template images 
        (default None for both)
    thresh_sigma : float, optional
        Sigma to use in setting the threshold for source detection via image
        segmentation (default 3.0)
    etamax : float, optional
        *Maximum* allowed elongation for an isophote to be considered a good
        source for hotpants (default 2.0)
    areamax : float, optional
        *Maximum* allowed pixel area for an isophote to be considered a good
        source for hotpants (default 400.0)
    sepmax : float, optional
        *Maximum* on-sky separation (arcsec) of sources between the science and 
        template images to be designated as a source present in both (default 
        5.0)
    mode : {"mean", "science", "template"}, optional
        Coordinate system to use for the output (xi, yi) pairs (default "mean",
        see notes for details)
    output : str, optional
        Name for output ascii text file (default 
        `science_file.replace(".fits", "_substamps.txt")`)
    verbose : bool, optional
        Whether to be verbose (default False)

    Returns
    -------
    tuple of astropy.table.Table objects
        A tuple containing the tables of source coordinates ("xcentroid", 
        "ycentroid") for the science and reference images
    
    Notes
    -----   
    For both the science and template image, finds sources via image 
    segmentation and selects those which are not overly elongated (to filter 
    out galaxies) or overly large (to filter out saturated stars). Then 
    compares the science and template images, finds sources in common, and 
    writes these to an ascii text file in the form 
    
    ::
        
        'x1 y1'
        'x2 y2'
        'x3 y3'
        ...
    
    Where the exact coordinates are determined by the setting for `mode`. If 
    `mode == "mean"`, the coordinates will be the average of the science and
    template image; if `mode == "science"` or `mode == "template"`, the 
    coordinates will be those of the science or template image, respectively.
    """    
    # verify input coords mode
    if not(mode in ("mean", "science", "template")):
        raise ValueError('mode must be one of ("mean", "science", "template")'+
                         f' but the supplied argument was {mode}')
    
    # load in data
    sci_header = fits.getheader(science_file)
    sci_data = fits.getdata(science_file)
    tmp_header = fits.getheader(template_file)
    tmp_data = fits.getdata(template_file)
    data = [sci_data, tmp_data]
    hdrs = [sci_header, tmp_header]

    ## masks
    if sci_mask_file and tmp_mask_file: # if both masks are supplied        
        mask = np.logical_or(fits.getdata(sci_mask_file).astype(bool),
                             fits.getdata(tmp_mask_file).astype(bool))
    elif sci_mask_file: # if only science        
        mask = fits.getdata(sci_mask_file).astype(bool)
    elif tmp_mask_file: # if only template        
        mask = fits.getdata(sci_mask_file).astype(bool)
    else:
        mask = None
        
    ## find sources in both the science and template image
    obj_props = []
    for i in range(2):
        # pick the relevant data and header
        image_data = data[i]
        image_header = hdrs[i]
        
        # set the threshold for image segmentation
        try: # if background subtraction has already been obtained 
            std = image_header["BKGSTD"] # header written by bkgsub function
            threshold = thresh_sigma*std 
            
        except KeyError: # else, do it now
            # use crude image segmentation to find sources above SNR=3
            # allows us to ignore sources during background estimation
            if type(mask) == np.ndarray: # use bad pixel mask if one is present 
                source_mask = make_source_mask(image_data, nsigma=3, npixels=5, 
                                               dilate_size=15, mask=mask)
                # combine the bad pixel mask and source mask 
                final_mask = np.logical_or(mask, source_mask)
            else: 
                source_mask = make_source_mask(image_data, nsigma=3, npixels=5, 
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
            threshold = thresh_sigma*bkg_rms
        
        # get the segmented image 
        segm = detect_sources(image_data, threshold, npixels=5, mask=mask)
        
        # get the source properties
        #cat = source_properties(image_data, segm, mask=mask) # photutils 0.8
        cat = SourceCatalog(data=image_data, segment_img=segm, 
                            mask=mask) # photutils >=1.1
        try:
            tbl = cat.to_table(columns=REQ_COLUMNS)
        except ValueError:
            print("SourceCatalog contains no sources. Exiting.")
            return
        
        # restrict elongation and area to obtain only unsaturated stars
        tbl_mask = (tbl["elongation"] <= etamax)
        tbl = tbl[tbl_mask]
        tbl_mask = tbl["area"].value <= areamax
        tbl = tbl[tbl_mask]
        obj_props.append(tbl)

    ## get RA, Dec of sources after this restriction
    # science image
    w = wcs.WCS(sci_header)
    sci_coords = w.all_pix2world(obj_props[0]["xcentroid"], 
                                    obj_props[0]["ycentroid"], 1)
    sci_skycoords = SkyCoord(ra=sci_coords[0], dec=sci_coords[1],
                             frame="icrs", unit="degree")
    # template
    w = wcs.WCS(tmp_header)
    tmp_coords = w.all_pix2world(obj_props[1]["xcentroid"], 
                                 obj_props[1]["ycentroid"], 1)   
    tmp_skycoords = SkyCoord(ra=tmp_coords[0], dec=tmp_coords[1], frame="icrs", 
                             unit="degree")
    
    ## find sources which are present in both the science image and template
    ## condition: sources must be <= <sepmax> arcsec away from each other 
    idx_sci, idx_tmp, d2d, d3d = tmp_skycoords.search_around_sky(
            sci_skycoords, sepmax*u.arcsec) 
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
    if mode == "mean":
        x = [np.mean([obj_props[0]["xcentroid"][i].value+1, 
                obj_props[1]["xcentroid"][i].value+1]) for i in range(nmatch)]
        y = [np.mean([obj_props[0]["ycentroid"][i].value+1, 
                obj_props[1]["ycentroid"][i].value+1]) for i in range(nmatch)]
    elif mode == "science":
        x = [obj_props[0]["xcentroid"][i].value for i in range(nmatch)]
        y = [obj_props[0]["ycentroid"][i].value for i in range(nmatch)]
    elif mode == "template":
        x = [obj_props[1]["xcentroid"][i].value for i in range(nmatch)]
        y = [obj_props[1]["ycentroid"][i].value for i in range(nmatch)]  
              
    if not(output): # set filename if not given
        output = science_file.replace(".fits", "_substamps.txt")
    
    lines = np.array([[x[n], y[n]] for n in range(nmatch)])
    lines.sort(axis=0)
    #lines = [f"{int(x[n])} {int(y[n])}" for n in range(nmatch)]
    np.savetxt(output, lines, fmt="%.3f", encoding="ascii", newline="\n")
    
    return (obj_props[0]["xcentroid", "ycentroid"], 
            obj_props[1]["xcentroid", "ycentroid"])
    

def param_estimate(science_file, template_file, mask_file=None,
                   sci_epsf_file=None, tmp_epsf_file=None, 
                   thresh_sigma=3.0, pixelmin=20, etamax=1.4, areamax=400.0, 
                   cutout=35, verbose=True):
    """Get a rough estimate for the parameters of the Gaussian basis kernel 
    which should be used by `hotpants` given some input science and template 
    images, and whether to convolve the science or template image during 
    differencing. **However, see notes**.
    
    Arguments
    ---------
    science_file, template_file : str
        Science and template images fits files 
    mask_file : str, optional
        Bad pixel mask fits file (default None)
    sci_epsf_file : str, optional
        Fits file containing the science and template images' ePSFs, 
        respectively (default None for both, in which case ePSF will be 
        estimated here with `build_ePSF()`)
    thresh_sigma : float, optional
        Sigma threshold for source detection with image segmentation (default
        3.0)
    pixelmin : float, optional
        *Minimum* pixel area of an isophote to be considered a good source for 
        building the ePSF (default 20; passed to `build_ePSF_imsegm()` only 
        if needed)
    etamax : float, optional
        *Maximum* allowed elongation for an isophote to be considered a good 
        source for building the ePSF (default 1.4; passed to 
        `build_ePSF_imsegm()` only if needed)
    areamax : float, optional
        *Maximum* allowed area (in square pixels) for an isophote to be 
        considered a good source for building the ePSF (default 400.0, passed 
        to `build_ePSF_imsegm()` only if needed)
    cutout : int, optional
        Cutout size around each star in pixels (default 35; must be **odd**; 
        rounded **down** if even; passed to `build_ePSF_imsegm()` only if 
        needed)
    verbose : bool, optional
        Whether to be verbose (default True)
    
    Returns
    -------
    convolve : {"t", "i"}
        String indicating whether to convolve the template (t) or science 
        image (i) during image differencing with `hotpants`
    gaussian : dict
        Dictionary containing the parameters of the Gaussian basis kernel
        to use during image differencing with `hotpants`
        
    Notes
    -----
    **Not stable right now.** Messes up when science and reference have very 
    similar PSFs (which is often the case and in fact what we want).
    
    """
    
    ## load in OR build the source image ePSF, get FWHM
    if sci_epsf_file:
        sci_epsf = fits.getdata(sci_epsf_file)
    else:
        sci_epsf = build_ePSF_imsegm(science_file, mask_file, 
                                        thresh_sigma=thresh_sigma, 
                                        pixelmin=pixelmin, 
                                        etamax=etamax, areamax=areamax, 
                                        cutout=cutout, 
                                        write=False, verbose=verbose)
    
    ## load in OR build the template image ePSF
    if tmp_epsf_file:
        tmp_epsf = fits.getdata(tmp_epsf_file)
    else:
        tmp_epsf = build_ePSF_imsegm(template_file, mask_file, 
                                     thresh_sigma=thresh_sigma, 
                                     pixelmin=pixelmin, 
                                     etamax=etamax, areamax=areamax,                                
                                     cutout=cutout, 
                                     write=False, verbose=verbose)
        
    
    ## get the FWHM and sigma of each PSF, assuming roughly Gaussian
    source_epsf_sig = gaussian_fwhm_to_sigma*ePSF_FWHM(sci_epsf, verbose)
    tmp_epsf_sig = gaussian_fwhm_to_sigma*ePSF_FWHM(tmp_epsf, verbose)
    
    ## compare ePSFs
    if tmp_epsf_sig > source_epsf_sig: 
        ## sharpening the template leads to false positives, SO --> 
        # need to either convolve the science image with convolution kernel OR 
        # convolve the science image with its own ePSF prior to template 
        # matching, which increases the sigma of the science image ePSF by a
        # factor of ~sqrt(2)
        if verbose:
            print("\n FWHM_sci < FWHM_tmp")        
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


def hotpants(science_file, template_file, 
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
             maskout=None, convout=None, kernout=None, noiseout=None, 
             noisescaleout=None,
             v=1, log=None,
             plot=True, plotname=None, scale="linear", 
             target_large=None, target_small=None,
             thresh_sigma=3.0, pixelmin=20, etamax=1.4, areamax=500.0):
    """Wrapper for the hotpants image differencing software [1]_, adapted
    primarily for use with the Canada-France-Hawaii Telescope (CFHT) MegaCam
    instrument.
    
    Arguments
    ---------
    science_file, template_file : str
        Science image and template image fits files
    sci_mask_file, tmp_mask_file : str, optional
        Masks of pixels to ignore in the science and template images (fits 
        files; default None)
    substamps_file : str, optional
        Text file containing the locations of substamps (default None)
    param_estimate : bool, optional
        Whether to try and estimate the optimal parameters of the Gaussian 
        basis kernel and whether to convolve the template or science image 
        (default False; **recommended to keep it this way** - not reliable)
    iu, tu, il, tl : float, optional
        Upper (u) and lower (l) ADU limits for the science image (i) and 
        template image (t) (default None, in which case these limits are set 
        according to args `lsigma`, `hsigma`)
    lsigma, hsigma : float, optional
        Number of standard deviations away from the background at which to 
        place the lower and upper ADU limits (default 5.0, but only used if 
        `tu`, `tl`, `iu`, `il` are not given)
    gd : str, optional
        String of form "xmin xmax ymin ymax" indicating the region in which 
        pixels are good (e.g. "150 1000 200 800" to trim the edges of some 
        1000x1000 image; default None --> use whole image)
    ig, ir, tg, tr : float, optional
        Gain (g) and readnoise (r) for the science image (i) and template (t) 
        (default None, in which case values are extracted from headers if 
        present; if not present, default is `ig = tg = 1`, `ir = tr = 0`)
    ng : str, optional
        String of form "ngauss degree0 sigma0 ... degreeN sigmaN", where 
        N=ngauss-1, "ngauss" is the number of Gaussians in the basis kernel,
        degreej is the degree of the polynomial attached to the jth Gaussian, 
        and sigmaj is the width of the jth Gaussian (default 
        "3 6.0 0.7 4.0 1.5 2.0 3.0", which is a mixture of 3 Gaussians with
        widths 0.7, 1.5, 3.0 respectively and polynomials of order 6, 4, 2, 
        respectively). **This is the most important setting.**
    rkernel : float, optional
        FWHM of the convolution kernel (default, 10.0)
    nrx, nry : int, optional
        Number of regions in the x and y directions, respectively (defaul 1 for 
        both)
    nsx, nsy : int, optional
        Number of stamps in each region in the x and y directions, respectively 
        (default 10 for both)
    nss : int, optional
        Number of centroids (i.e. sources) to extract within each stamp 
        (default 3)
    rss : float, optional
        Half-width of the substamp for centroid extraction (default 
        `2.5*rkernel`, which is 25.0 for default `rkernel`)
    ssig : float, optional
        Sigma threshold for sigma clipping (default 3.0)
    norm : {"i", "t", "u"}, optional
        Normalization to use for final difference image, where "i" is the 
        science image, "t" the template, and "u" the unconvolved science image
        or template depending on which was convolved (default "i")
    convi : bool, optional
        Whether to force convolution of the science image (default False)
    convt : bool, optional
        Whether to force convolution of the template image (default False)
    bgo : int, optional
        Order of spatial variations in the background (default 0)
    ko : int, optional
        Order of spatial variations in the kernel (default 1)
    output : str, optional
        Output difference image file name (default 
        `science_file.replace(".fits", "_subtracted.fits")`)
    mask_write : bool, optional
        Whether to write the bad pixel mask (default False)
    conv_write : bool, optional
        Whether to write the convolved image (default False)
    kern_write : bool, optional
        Whether to write the kernel image (default False)
    noise_write : bool, optional
        Whether to write the noise image (default False)
    noise_scale_write : bool, optional
        Whether to write the *scaled difference* noise image (default False)
    maskout : str, optional
        Name for the output bad pixel mask if `mask_write = True` (default
        `science_file.replace(".fits", "_submask.fits")`)
    convout : str, optional
        Name for output convolved image if `conv_write = True` (default 
        `science_file.replace(".fits", "_conv.fits")`)
    kernout : str, optional
        Name for output kernel if `kern_write = True` (default 
        `science_file.replace(".fits", "_kernel.fits")`)
    noiseout : str, optional
        Name for output noise image if `noise_write = True` (default 
        `science_file.replace(".fits", "_noise.fits")`)
    noisecaleout : str, optional
        Name for output noise *scaled difference* image if 
        `noise_scale_write = True` (default 
        `science_file.replace(".fits", "_noise_scaled.fits")`)
    v : {0, 1, 2}, optional
        Level of verbosity, where 0 is least and 2 is most (default 1)
    log : str, optional
        Name for a logfile in which to store STDERR (default None)
    plot : bool, optional
        Whether to plot the final difference image (default True)
    plotname : str, optional
        Name for the output figure (default 
        `output.replace(".fits", "_hotpants.png")`)
    scale : {"linear", "log", "asinh"}, optional
        Scale to apply to the plot (default "linear")
    target_large : array_like
        [ra, dec] for some target of interest (e.g. a potential transient) 
        at which to draw a large crosshair (default None)
    target_small : array_like
        [ra, dec] for some other target of interest (e.g. the candidate host
        galaxy of some target of interest) at which to draw a small crosshair 
        (default None)
    thresh_sigma : float, optional
        Sigma threshold for source detection with image segmentation (default
        3.0)
    pixelmin : float, optional
        *Minimum* pixel area of an isophote to be considered a good source for 
        building the ePSF (default 20; passed to `build_ePSF_imsegm()` only 
        if `param_estimate = True`)
    etamax : float, optional
        *Maximum* allowed elongation for an isophote to be considered a good 
        source for building the ePSF (default 1.4; passed to 
        `build_ePSF_imsegm()` only if `param_estimate = True`)
    areamax : float, optional
        *Maximum* allowed area (in square pixels) for an isophote to be 
        considered a good source for building the ePSF (default 500.0, passed 
        to `build_ePSF_imsegm()` only if `param_estimate = True`)
    
    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) for the difference image 
    
    Notes
    -----
    For more details, see: https://github.com/acbecker/hotpants
    
    High Order Transformation of Psf ANd Template Subtraction
    Based (hotpants) is based on the method of "optimal image subtraction" of 
    Alard & Lupton 1998 [1]_ .
    
    Note that this wrapper does not implement *all* settings which can be 
    specified by the user when running hotpants directly from the command line,
    but does include most of them (and all of the most important ones).
    
    References
    ----------
    .. [1] Alard, C., & Lupton, R. H. 1998, ApJ, 503, 325
           https://ui.adsabs.harvard.edu/abs/1998ApJ...503..325A/abstract
    """

    ## load in data 
    science_header = fits.getheader(science_file)
    science_data = fits.getdata(science_file)
    tmp_header = fits.getheader(template_file)
    tmp_data = fits.getdata(template_file)
        
    ########################## OPTIONS FOR HOTPANTS ###########################
    
    ### input/output files and masks ##########################################
    hp_options = f"-inim {science_file} -tmplim {template_file} "#-fi 0"
    
    if not(output):
        output = science_file.replace(".fits", "_subtracted.fits")
    hp_options = f"{hp_options} -outim {output}" # output subtracted file
    
    if mask_write:
        if not(maskout):
            maskout = science_file.replace(".fits", "_submask.fits")
        hp_options = f"{hp_options} -omi {maskout}" # output bad pixel mask        
    if conv_write:
        if not(convout):
            convout = science_file.replace(".fits", "_conv.fits")        
        hp_options = f"{hp_options} -oci {convout}" # output convolved image       
    if kern_write:
        if not(kernout):
            kernout = science_file.replace(".fits", "_kernel.fits")
        hp_options = f"{hp_options} -oki {kernout}" # output kernel image
    if noise_write:
        if not(noiseout):
            noiseout = science_file.replace(".fits", "_noise.fits")
        hp_options = f"{hp_options} -oni {noiseout}" # output noise image
    if noise_scale_write:
        if not(noisescaleout):
            noisescaleout = science_file.replace(".fits", "_noise_scaled.fits")
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
        #hp_options += f' -savexy {science_file.replace(".fits", "_conv")}'

    ### if requested, compare sci/tmp ePSF to estimate optimal params #########    
    if param_estimate:
        c, g = param_estimate(science_file, template_file, 
                              thresh_sigma=thresh_sigma, 
                              pixelmin=pixelmin, etamax=etamax, 
                              areamax=areamax, verbose=True)
    
        ng = f"3 6 {g[6]:.2f} 4 {g[4]:.2f} 2 {g[2]:.2f}"
        rkernel = 2.5*g[4]*gaussian_sigma_to_fwhm
        print(f"rkernel = {rkernel}")
        if c == "t": convi = False; convt = True
        elif c == "i": convi = True; convt = False
            
    ### upper/lower limits for SCIENCE image ##################################
    # impose no limits if none are given
    if iu: hp_options = f"{hp_options} -iu {iu}"
    else: hp_options = f"{hp_options} -iu {np.max(science_data)}"
        
    if il: hp_options = f"{hp_options} -il {il}"
    else: hp_options = f"{hp_options} -il {np.min(science_data)}"
        
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
    #            source_mask = make_source_mask(source_data, nsigma=3, npixels=5, 
    #                                           dilate_size=15, 
    #                                           mask=sci_bp_mask)
    #            # combine the bad pixel mask and source mask 
    #            final_mask = np.logical_or(sci_bp_mask, source_mask)
    #        else: 
    #            source_mask = make_source_mask(source_data, nsigma=3, npixels=5, 
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
    #            source_mask = make_source_mask(tmp_data, nsigma=3, npixels=5, 
    #                                           dilate_size=15, 
    #                                           mask=tmp_bp_mask)
    #            # combine the bad pixel mask and source mask 
    #            final_mask = np.logical_or(tmp_bp_mask,source_mask)
    #        else: 
    #            source_mask = make_source_mask(tmp_data, nsigma=3, npixels=5, 
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
            hp_options = f'{hp_options} -ig {science_header["GAIN"]}'
        except KeyError: # if no keyword GAIN found in header 
            pass
    if ir: # image readnoise
        hp_options = f"{hp_options} ir {ir}"
    else:
        try:
            hp_options = f'{hp_options} -ir {science_header["RDNOISE"]}'
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

    # output filenames
    outfile = re.sub(".*/", "", output) # for a file /a/b/c, extract the "c"
    topdir = output[:-len(outfile)] # extract the "/a/b/"
    if topdir == "": # if using current directory without full filepath
        topdir = "./"
    print(f"\namakihi.hotpants(): output = {output}", flush=True)
    print(f"amakihi.hotpants(): outfile = {outfile}", flush=True)
    print(f"amakihi.hotpants(): topdir = {topdir}", flush=True)

    # build the command
    hp_cmd = f"hotpants {hp_options}"
    
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
    except CalledProcessError as cpe: # if an error code is returned, exit 
        print(f"\n\n\nCalledProcessError: {cpe}"+
              "\n--> exiting")
        return 
    
    # if file successfully produced and non-empty
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
        if not(plotname): # set filename for plot if not given
            plotname = outfile.replace(".fits", "_hotpants.png")
        __plot_hotpants(sub=sub, hdr=science_header, 
                        mean_diff=mean_diff, std_diff=std_diff, 
                        scale=scale, 
                        target_large=target_large, 
                        target_small=target_small,
                        output=plotname)
        
    return sub
