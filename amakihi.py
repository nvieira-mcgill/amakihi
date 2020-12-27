#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:58:03 2019
@author: Nicholas Vieira
@amakihi.py 

**Sections:**

- Image differencing with hotpants (https://github.com/acbecker/hotpants)
- Transient detection, triplets


**Essential dependencies:**

- ``astropy`` (used extensively)
- ``photutils`` (used extensively)
- ``astrometry.net`` (used extensively, but can be ignored in favour of 
  source detection with `photutils`' `image_segmentation` instead)
- ``hotpants`` (essential for image subtraction via `hotpants`, duh)

"""

# misc
import os
from subprocess import run, PIPE, CalledProcessError
import numpy as np
import re

# astropy
from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.stats import (sigma_clipped_stats, SigmaClip, 
                           gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm)
from astropy.coordinates import SkyCoord
from astropy.table import Table
from photutils import Background2D, MedianBackground
from photutils import make_source_mask, detect_sources, source_properties

## for speedy FFTs
#import pyfftw
#import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs
#pyfftw.interfaces.cache.enable()

# amakihi 
from crop import crop_WCS
from ePSF import build_ePSF_imsegm, ePSF_FWHM
from plotting import __plot_rejected, __plot_triplet, plot_transient

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
try: plt.switch_backend('Qt5Agg')
except ImportError: pass # for Compute Canada server

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
#### IMAGE DIFFERENCING WITH HOTPANTS ####

def get_substamps(source_file, template_file, 
                  sci_mask_file=None, tmp_mask_file=None, 
                  sigma=3.0, etamax=2.0, areamax=400.0, sepmax=5.0, 
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
        tbl_mask = tbl["area"].value <= areamax
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
                                 source_skycoords, sepmax*u.arcsec) 
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
                   thresh_sigma=3.0, pixelmin=20, etamax=1.4, areamax=400, 
                   cutout=35, verbose=True):
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
        source_epsf = build_ePSF_imsegm(source_file, mask_file, 
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
             thresh_sigma=3.0, pixelmin=20, etamax=1.4, areamax=500):
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
                              areamax=areamax, verbose=True)
    
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

def transient_detect(sub_file, og_file, ref_file, mask_file=None, 
                     thresh_sigma=5.0, pixelmin=20, 
                     dipole_width=2.0, dipole_fratio=5.0,
                     etamax=1.8, areamax=300, nsource_max=50, 
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
        nbelow = len(tbl[areas<areamax])
        nabove = len(tbl[areas>areamax])
        mean, med, std = sigma_clipped_stats(areas) 
        plt.hist(areas, bins=20, color="#c875c4", alpha=0.5)
                 
        plt.axvline(mean, color="red", label=r"$\mu$")
        plt.axvline(mean+std, color="red", ls="--", 
                    label=r"$\mu$"+"±"+r"$\sigma$")
        plt.axvline(mean-std, color="red", ls="--")
        plt.axvline(areamax, color="blue", lw=2.5, label=r"$A_{max}$")
        plt.xlabel("Area [pix"+r"${}^2$"+"]", fontsize=15)
        plt.ylabel("Counts", fontsize=15)
        plt.gca().tick_params(which='major', labelsize=10)
        plt.grid()
        plt.xscale("log")
        plt.yscale("log")
        
        textboxstr = r"$\mu - A_{max} = $"+"%.2f"%(mean-areamax)+"\n"
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
                      elongs[i]<etamax and areas[i]<areamax)]
        areasgood = [areas[i] for i in range(len(elongs)) if (
                     elongs[i]<etamax and areas[i]<areamax)]
    
        elongsbad = [elongs[i] for i in range(len(elongs)) if (
                     elongs[i]>etamax or areas[i]>areamax)]
        areasbad = [areas[i] for i in range(len(elongs)) if (
                    elongs[i]>etamax or areas[i]>areamax)]
        
        # elongation on x axis, area on y axis            
        plt.scatter(elongsgood, areasgood, marker="o", color="#5ca904", s=12)
        plt.scatter(elongsbad, areasbad, marker="s", color="#fe019a", s=12)
        # means and maxima
        mean, med, std = sigma_clipped_stats(elongs)              
        plt.axvline(mean, ls="--", color="black", label=r"$\mu$")
        plt.axvline(etamax, color="#030aa7", lw=2.5, label=r"$\eta_{max}$")
        mean, med, std = sigma_clipped_stats(areas) 
        plt.axhline(mean, ls="--", color="black")
        plt.axhline(areamax, color="#448ee4", lw=2.5, label=r"$A_{max}$")
        # allowed region of parameter space 
        rect = ptc.Rectangle((0,0), etamax, areamax, fill=False, 
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
    ## (1) look for dipoles by looking for sources in (-1)*difference and  
    ## cross-matching to the segmentation image 
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
    
    ## (2) restrict based on source elongation 
    if etamax:
        premasklen = len(tbl)
        mask = tbl["elongation"] < etamax 
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print(f"\n{premasklen-postmasklen} source(s) with "+
                  f"elongation >{etamax} removed")
        tbl_noelong = tbl.copy()
            
    ## (3) restrict based on maximum pixel area 
    if areamax:
        premasklen = len(tbl)
        mask = tbl["area"].value < areamax
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print(f"\n{premasklen-postmasklen} source(s) with "+
                  f"area >{areamax} pix**2 removed")
    
    vetted = tbl.copy()

    ## plot rejected/non-rejected candidates, if desired 
    if plot_rejections:
        dipoles = tbl_novetting[idx_rem]
        elongated_sources = tbl_nodipoles[tbl_nodipoles["elongation"] >= 
                                          etamax]
        large_sources = tbl_noelong[tbl_noelong["area"].value >= areamax]
        
        # colours below might be out of date (could be improved)
        __plot_rejected(sub_file=sub_file,
                        dipoles=dipoles, 
                        elongated_sources=elongated_sources, 
                        large_sources=large_sources, 
                        vetted=vetted,
                        dipole_width=dipole_width, 
                        etamax=etamax, 
                        areamax=areamax, 
                        #nsource_max=nsource_max,
                        toi=toi, #toi_sep_max, 
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
        plot_transient(sub_file=sub_file, og_file=og_file, ref_file=ref_file, 
                       tbl=tbl, 
                       pixcoords=pixcoords, 
                       toi=toi, 
                       plots=plots, 
                       sub_scale=sub_scale, og_scale=og_scale, 
                       stampsize=stampsize, 
                       crosshair_og=crosshair_og, crosshair_sub=crosshair_sub, 
                       title_append=title, plotdir=plotdir)    
    return tbl


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
                
            __plot_triplet(og_file=og_file, 
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
        

    
