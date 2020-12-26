#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:57:05 2020
@author: Nicholas Vieira
@background.py

Background estimation and subtraction.
"""

# misc
import numpy as np

# astropy
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from photutils import make_source_mask

# amakihi
from plotting import __plot_bkg, __plot_bkgsubbed

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
#### BACKGROUND SUBTRACTION ###################################################
    
def bkgsub(im_file, mask_file=None, 
           bkg_box=(5,5), bkg_filt=(5,5),
           plot_bkg=False, plot_bkgsubbed=False,
           scale_bkg="linear", scale_bkgsubbed="linear", 
           write=True, output=None):
    """Subtract the background from an image. 
    
    Arguments
    ---------
    im_file : str
        Filename for image of interest
    mask_file : str, optional
        Filename for bad pixels mask (default None)
    bkg_box : tuple, optional
        Sizes of box along each axis in which background is estimated (default 
        (5,5))
    bkg_filt : tuple, optional
        Size of the median filter pre-applied to the background before 
        background estimation (default (5,5) where (1,1) --> no filtering)
    plot_bkg : bool, optional
        Whether to plot BACKGROUND image (default False)
    plot_bkgsubbed : bool, optional
        Whether to plot background-SUBTRACTED image (default False)
    scale_bkg : {"linear", "log", "asinh"}, optional
        Scale to apply to BACKGROUND image plot (default "linear")
    scale_bkgsubbed : {"linear", "log", "asinh"}, optional
        Scale to apply to background-SUBTRACTED image plot (default "linear")
    write : bool, optional
        Whether to write the background-SUBTRACTED image to a fits file 
        (default True)
    output : str, optional
        Name for output fits file (default 
        `im_file.replace(".fits", "_bkgsub.fits")`)

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) with the image background-subtracted


        
    **TO-DO:**
    
    - Allow naming of output plots
    
    """
    
    ## load in the image data
    image_data = fits.getdata(im_file)
    
    ## source detection and building masks
    if mask_file: # load a bad pixel mask if one is provided
        bp_mask = fits.getdata(mask_file).astype(bool)
        zeromask = image_data == 0 # mask out pixels equal to 0
        nansmask = np.isnan(image_data) # mask out nans
        bp_mask = np.logical_or.reduce((bp_mask, zeromask, nansmask))
        #bp_mask = np.logical_or(bp_mask, zeromask) # old way: chained
        #bp_mask = np.logical_or(bp_mask, nansmask)       
    else: 
        zeromask = image_data == 0 # mask out pixels equal to 0
        nansmask = np.isnan(image_data) # mask out nans
        bp_mask = np.logical_or(nansmask, zeromask)
    # make a crude source mask
    source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                   dilate_size=15, mask=bp_mask)
    # combine the bad pixel mask and source mask for background subtraction
    # make the final mask
    mask = np.logical_or(bp_mask, source_mask)

    
    ## estimate the background
    try:
        sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    except TypeError: # in old astropy, "maxiters" was "iters"
        sigma_clip = SigmaClip(sigma=3, iters=5)
    
    bkg = Background2D(image_data, box_size=bkg_box, filter_size=bkg_filt, 
                       sigma_clip=sigma_clip, bkg_estimator=MedianBackground(), 
                       mask=mask)
    bkg_img = bkg.background
    bkgstd = bkg.background_rms_median   

    
    ## subtract the background 
    bkgsub_img = image_data - bkg_img     
    bkgstd = bkg.background_rms_median # save this quantity to write to header

        
    ## finally, mask bad pixels 
    # all bad pix are then set to 0 for consistency
    bkg_img_masked = np.ma.masked_where(bp_mask, bkg_img)
    bkg_img = np.ma.filled(bkg_img_masked, 0)   
    bkgsub_img_masked = np.ma.masked_where(bp_mask, bkgsub_img)
    bkgsub_img = np.ma.filled(bkgsub_img_masked, 0)

    
    ## plotting (optional)
    if plot_bkg: # plot the background, if desired
        output_bkg = f"background_{scale_bkg}.png"
        __plot_bkg(im_header=fits.getheader(im_file), 
                   bkg_img_masked=bkg_img_masked, 
                   scale_bkg=scale_bkg, 
                   output_bkg=output_bkg)
            
    if plot_bkgsubbed: # plot the background-subtracted image, if desired
        output_bkgsubbed = f"background_subbed_{scale_bkgsubbed}.png"        
        __plot_bkgsubbed(im_header=fits.getheader(im_file), 
                         bkgsub_img_masked=bkgsub_img_masked, 
                         scale_bkgsubbed=scale_bkgsubbed,
                         output_bkgsubbed=output_bkgsubbed)

    
    ## wrapping up
    hdr = fits.getheader(im_file)
    hdr["BKGSTD"] = bkgstd # useful header for later
    bkgsub_hdu = fits.PrimaryHDU(data=bkgsub_img, header=hdr)
    
    if write: # if we want to write the background-subtracted fits file
        if not(output): # if no output name given, set default
            output = im_file.replace(".fits", "_bkgsub.fits")
        bkgsub_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return bkgsub_hdu