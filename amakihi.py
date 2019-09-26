#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:58:03 2019
@author: Nicholas Vieira
@amakihi.py 

SECTIONS:
    - Downloading templates
    - Cropping images
    - Background subtraction
    - Image registraton (alignment)
    - Mask building (boxmask, saturation mask)
    - Image differencing/transient detection with HOTPANTS
    - Image differencing/transient detection with Proper Image Subtraction
    - Miscellaneous plotting
"""


import os
from subprocess import run
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import re
import requests
from timeit import default_timer as timer

from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.convolution import convolve_fft
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.coordinates import SkyCoord
from photutils import (Background2D, 
                       MedianBackground, 
                       BiweightScaleBackgroundRMS)
from photutils import make_source_mask, detect_sources, source_properties

import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

##############################################################################
#### DOWNLOADING TEMPLATES ####

def __downloadtemplate(url, survey, pixscale, output=None):
    """
    Input: the url(s) of interest, the survey the templates come from, the 
    pixel scale used in arcsec per pix, and name(s) for the output file(s) 
    (optional; defaults set below)
    Downloads the fits image(s) at the given url(s). 
    Output: HDU object(s) for the downloaded template image(s) 
    """

    if type(url) == str: # if just one
        if not(output):
            output = re.sub(".*/", "", url)
            output = re.sub(".*=", "", output)
        tmpl = fits.open(url) # download it 
        tmpl.writeto(output, overwrite=True, output_verify="ignore")
        tmpl.close()
        tmpl = fits.open(output, mode="update")
        tmpl[0].header["SURVEY"] = survey # informative headers
        tmpl[0].header["PIXSCAL1"] = pixscale
        tmpl.close()
        return tmpl
    else: # if many
        templates = []
        for i in range(len(url)):
            ur = url[i]
            if not output:
                out = re.sub(".*/", "", ur) 
                out = re.sub(".*=", "", out)
            else:
                out = output[i]
            tmpl = fits.open(ur) # download it 
            templates.append(tmpl)
            tmpl.writeto(out, overwrite=True, output_verify="ignore")
            tmpl.close()
            tmpl = fits.open(out, mode="update")
            tmpl[0].header["SURVEY"] = survey # informative headers
            tmpl[0].header["PIXSCAL1"] = pixscale
            print(tmpl[0].header)
            tmpl.close()
            
        return templates
    

def __downloadtemplate_auth(url, survey, pixscale, auth_user, auth_pass):
    """
    Input: the url(s) of interest, the survey the templates come from, the 
    pixel scale used in arcsec per pix, and the CADC username and password 
    needed to access proprietary CFIS data 
    
    Downloads the fits image at the given url, with authorization, as is 
    required for CFIS images.
    
    Output: HDU object(s) for the downloaded template images
    """
    
    if type(url) == str: # if just one
        output = re.sub(".*/", "", url)
        output = output[:output.find("[")] # clean up filename
        r = requests.get(url, auth=(auth_user, auth_pass))
        open(output, "wb").write(r.content) # download it 
        tmpl = fits.open(output) # download it 
        tmpl.close()
        return tmpl
    else: # if many
        templates = []
        for ur in url:
            output = re.sub(".*/", "", ur)
            output = output[:output.find("[")] # clean up filename
            r = requests.get(ur, auth=(auth_user, auth_pass))
            open(output, "wb").write(r.content) # download it 
            tmpl = fits.open(output) 
            templates.append(tmpl)
            tmpl.close()
        return templates


def download_PS1_template(ra, dec, size=2400, filt="grizy", output=None):
    """
    Input: a RA, Dec of interest, a size for the image in pixels (1 pix ==
    0.25" in PS1), the filter(s) (g, r, i, z, y) desired, and output name(s)
    for the downloaded template(s)
    
    Downloads the relevant PS1 template image(s) at the input RA, Dec. 
    
    Output: HDU object(s) for the downloaded template image(s) 
    """
    if dec < -30: 
        print("\nPanStarrs 1 does not cover regions of the sky below "+
              "DEC = -30.0. Exiting.")
        return
    
    if (output and (type(output) != str)): # if output list too short or long
        if len(filt) != len(output):
            print("\nPlease provide a number of output filenames to match "+
                  "the number of requested template filters. Exiting.")
            return
            
            
    import query_PS1 # script for using PS1's cutout service
    
    size_arcmin = size*0.258/60.0 # cutout size in arcmin
    url = query_PS1.geturl(ra, dec, size, filters=filt, format="fits")
    
    filt_upd = ""
    for f in filt:
        if f in "grizy":
            filt_upd += f
    
    # download the template and tell the user
    # 0.258"/pix = PS1 resolution
    tmps = __downloadtemplate(url, "PS1", 0.258, output) # download template 
    print("\nDownloaded square PanStarrs1 cutout image(s) in the "+filt_upd+
          " band(s), centered on RA, Dec = %.3f, %.3f "%(ra, dec)+" with"+
          " sides of length %.2f'"%size_arcmin+"\n")
    return tmps


def download_DECaLS_template(ra, dec, size=512, pixscale=0.262, filt="grz", 
                             output=None):
    """
    Input: a RA, Dec of interest, a size for the image in pixels, the scale of 
    the image in arcseconds per pixel, the filters to use (files are always 
    downloaded separately when multiple filters are provided; options are 
    (g, r, z)) and output name(s) for the downloaded template(s)
    
    Downloads the relevant DECaLS template image(s) at the input RA, Dec. 
    
    Output: HDU object(s) for the downloaded template image(s) 
    """
    
    # verify the input filters 
    filt_upd = ""
    for f in filt:
        if f in "grz":
            filt_upd += f
        else: 
            print("\nDECaLS does not contain the "+f+" band, so this band "+
                  "will be ignored.")    
    
    # verify other input s
    if size > 512: # if requested image size too big
        print("\nThe maximum image size is 512 pix. The output image "+
              "will have these dimensions.")
        size = 512
    if (output and (type(output) != str)): # if output list too short or long
        if len(filt_upd) != len(output):
            print("\nPlease provide a list of output filenames to match "+
                  "the number of valid requested template filters. Exiting.")
            return
    if (output and (type(output) == str) and (len(filt_upd)>1)): 
        # if only one output string is given and multiple filters are requested
        print("\nPlease provide a list of output filenames to match "+
              "the number of valid requested template filters. Exiting.")
        return
        
       
    import query_DECaLS # script for using DECaLS's cutout service
    # get the url
    url = query_DECaLS.geturl(ra, dec, size, pixscale, filters=filt_upd)
    
    # download the template and tell the user
    tmps = __downloadtemplate(url,"DECaLS",pixscale,output) # download template 
    size_arcmin  = size*pixscale/60.0
    print("\nDownloaded square DECaLS cutout image(s) in the "+filt_upd+
          " band(s), centered on RA, Dec = %.3f, %.3f "%(ra, dec)+" with"+
          " sides of length %.2f'"%size_arcmin+"\n")
    return tmps


def download_CFIS_template(ra, dec, size=1600, filt="ur", 
                           auth_user="nvieira97", auth_pass="iwtg2s"):
    """
    Input: a RA, Dec of interest, a size for the image in pixels (1 pix ==
    0.185" in CFIS), the filter(s) (u, r) desired, and the CADC authorization
    needed to download CFIS data (optional; login is set by default)
    
    Downloads the relevant CFIS template image(s) at the input RA, Dec.
    
    Output: HDU object(s) for the downloaded template image(s) 
    """
    
    import query_CFIS # script for using CFIS' cutout service
    
    size_arcmin = size*0.185/60.0 # cutout size in arcmin
    url = query_CFIS.geturl(ra, dec, size, filters=filt) # url of file(s)
    if len(url) == 0: # if no templates found, exit
        print("\nNo CFIS images were found at the given coordinates. Exiting.")
        return

    filt_upd = ""
    for f in filt:
        if f in "ur":
            filt_upd += f
    
    # download the template and tell the user
    # 0.185"/pix = CFIS resolution
    tmps = __downloadtemplate_auth(url, "CFIS", 0.185, auth_user, auth_pass)
    print("\nDownloaded square CFIS cutout image(s) in the "+filt_upd+
          " band(s), centered on RA, Dec = %.3f, %.3f "%(ra, dec)+" with"+
          " sides of length %.2f'"%size_arcmin+"\n")
    return tmps


def download_2MASS_template(ra, dec, size=150, filt="A"):
    """
    Input: a RA, Dec of interest, a size for the image in pixels (1 pix ==
    4.0" in 2MASS), and the filter(s) (A for all or J, H, K) desired
    
    Downloads the relevant 2MASS template image(s) at the input RA, Dec. 
    
    Output: HDU object(s) for the downloaded template image(s) 
    """
    
    import query_2MASS
    
    if not(filt in ["A","J","H","K"]):
        print("\nAn invalid filter argument was given. Please choose only ONE"+
              "from  the following four (4) options: A (all), J, H, K")
    
    size_arcmin = size*4.0/60.0 # cutout size in arcmin
    url = query_2MASS.geturl(ra, dec, size, filters=filt)
    
    if filt == "A":
        filt_upd = "JHK" # for printing 
    else:
        filt_upd = filt
        
    # download the template and tell the user
    # 4.0"/pix = 2MASS resolution
    tmps = __downloadtemplate(url, "2MASS", 4.0) # download template 
    print("\nDownloaded 2MASS image(s) in the "+filt_upd+
          " band(s), centered on RA, Dec = %.3f, %.3f "%(ra, dec)+" with"+
          " sides of length %.2f'"%size_arcmin+"\n")
    
    return tmps

###############################################################################
#### CROPPING ####

def __get_crop(fits_file, frac_hori=[0,1], frac_vert=[0,1]):
    """
    Input: a single fits file, the horizontal fraction of the fits file's 
    image to crop (default [0,1], which does not crop), and the vertical 
    fraction (default [0,1])
    e.g. __get_crop("foo.fits", [0.5,1], [0,0.5]) would crop the bottom right
    corner of the image 
    Output: a new fits HDU containing the header and the cropped image
    """
        
    # get data 
    data = fits.getdata(fits_file)
    hdr = fits.getheader(fits_file)
    ydim, xdim = data.shape
    
    # get the indices in the data which bound the cropped area
    idx_x = [int(frac_hori[0]*xdim), int(frac_hori[1]*xdim)]
    idx_y = [int(frac_vert[0]*ydim), int(frac_vert[1]*ydim)]

    # get the cropped data, build a new PrimaryHDU object
    cropped = data[idx_y[0]:idx_y[1], idx_x[0]:idx_x[1]]
    hdr["NAXIS1"] = len(idx_x) # adjust NAXIS sizes
    hdr["NAXIS2"] = len(idx_y)
    hdr["CRPIX1"] -= idx_x[0] # update WCS reference pixel 
    hdr["CRPIX2"] -= idx_y[0]
    new_hdu = fits.PrimaryHDU(data=cropped, header=hdr)
    
    return new_hdu


def crop_WCS(source_file, ra, dec, size, write=True, output=None):
    """
    Input: a right ascension, declination (in decimal degrees), and the 
    size of a box (in pixels) to crop, a bool indicating whether to 
    save write the output .fits file (optional; default True) and a name for
    the output fits file (optional; default is set below)
    
    For a single fits file, crops the image to a box of size pixels centered 
    on the given RA and Dec. If the given box extends beyond the bounds of the 
    image, the box will be truncated at these bounds. 
    
    Output: a PrimaryHDU for the cropped image (contains data and header)
    """
    
    hdr = fits.getheader(source_file)
    img = fits.getdata(source_file)
    y_size, x_size = img.shape # total image dims in pix 
    w = wcs.WCS(hdr)
    
    # compute the bounds
#    pix_scale = -1.0 
#    while pix_scale <= 0:
#        try: # look for pixel scale in headers 
#            pix_scale = hdr["PIXSCAL1"] # scale of image in arcsec per pix
#            break
#        except KeyError:
#            pix_scale = -1
#        
#        try: # check if survey is DECaLS
#            survey = hdr["SURVEY"]
#            if "DECaLS" in survey:
#                pix_scale = 0.262
#            break
#        except KeyError:
#            pix_scale = -1

    try:
        pix_scale = hdr["PIXSCAL1"] # scale of image in arcsec per pix
    except KeyError:
        topfile = re.sub(".*/", "", source_file)
        pix_scale = float(input('\nPixel scale not found for '+topfile+'. '+
                                'Please input a scale in arcseconds per '+
                                'pixel \n(NOTE: PS1=0.258"/pix, DECaLS='+
                                '0.262"/pix, CFIS=0.185"/pix, 2MASS=4.0"/pix)'+
                                '\n>>> '))
                                
    size_wcs = pix_scale*size/3600.0 # size of desired box in degrees
    pix_x1 = np.array(w.all_world2pix(ra-size_wcs/2.0, dec, 1))[0]
    pix_x2 = np.array(w.all_world2pix(ra+size_wcs/2.0, dec, 1))[0]
    pix_y1 = np.array(w.all_world2pix(ra, dec-size_wcs/2.0, 1))[1]
    pix_y2 = np.array(w.all_world2pix(ra, dec+size_wcs/2.0, 1))[1]
    x_bounds = np.array(sorted([pix_x1, pix_x2])) # sorted arrays of 
    y_bounds = np.array(sorted([pix_y1, pix_y2])) # pixel boundaries
    # truncate bounds if needed
    x_bounds[x_bounds<0] = 0 
    x_bounds[x_bounds>x_size] = x_size
    y_bounds[y_bounds<0] = 0 
    y_bounds[y_bounds>y_size] = y_size
    # convert to horizontal & vertical fractions, pass to __get_crop()
    frac_hori = x_bounds/x_size
    frac_vert = y_bounds/y_size
    
    # if the crop does not contain the bounds, notify user and exit
    # if the crop's aspect ratio is more skew than 4:1 or 1:4, notify user
    # if the crop is < 50% the width/height of the desired box, notify user 
    if np.all(frac_hori==0) or np.all(frac_hori==1.0) or np.all(
            frac_vert==0.0) or np.all(frac_vert==1.0):
            print("\nDesired cropped image is out of bounds. Exiting.")
            return 
    if not(0.25 < ((frac_hori[1]-frac_hori[0])/
                   (frac_vert[1]-frac_vert[0])) < 4.0):
            print("\nWARNING: the aspect ratio of the image is more skew than"+
                  "1:4 or 4:1.")
    if not((x_bounds[1]-x_bounds[0] > size/2.0) and 
           (y_bounds[1]-y_bounds[0] > size/2.0) ):
            print("\nWARNING: the cropped image is less than 50% the height "+
                  "or width of the desired crop.")
    
    cropped_hdu = __get_crop(source_file, frac_hori, frac_vert)
    
    if write: # if we want to write the cropped .fits file 
        if not(output): # if no output name given, set default
            output = source_file.replace(".fits", "_crop.fits")
        cropped_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    return cropped_hdu


def crop_frac(source_file, frac_hori=[0,1], frac_vert=[0,1], write=True, 
             output=None):
    """
    Input: a right ascension, the horizontal fraction of the fits file's 
    image to crop (default [0,1], which does not crop), the vertical fraction 
    (default [0,1]), a bool indicating whether to save write the output .fits 
    file (optional; default True) and a name for the output fits file 
    (optional; default is set below) 
    
    Output: a PrimaryHDU for the cropped image (contains data and header)
    """
    
    cropped_hdu = __get_crop(source_file, frac_hori, frac_vert)
    if write: # if we want to write the cropped .fits file 
        if not(output): # if no output name given, set default
            output = source_file.replace(".fits", "_crop.fits")
        cropped_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    return cropped_hdu

###############################################################################
#### BACKGROUND SUBTRACTION ####
    
def bkgsub(im_file, mask_file=None, bkg_box=(5,5), bkg_filt=(5,5),
           crreject=False, plot_bkg=False, scale_bkg=None, plot=False, 
           scale=None, write=True, output=None):
    """
    WIP:
        - photutils error?
    
    Input: 
        - image of interest
        - bad pixels mask (optional; default None)
        - size of the box along each axis for background estimation (optional;
          default (5,5))
        - size of the median filter pre-applied to the background before 
          estimation (optional; default (5,5); (1,1) indicates no filtering)
        - whether to do cosmic ray rejection (optional; default False)
        - whether to plot the background (optional; default False)
        - scale to apply to the background plot (optional; default None 
          (linear);  options are ("log" and asinh")
        - whether to plot the background-SUBTRACTED image (optional; default 
          False)
        - scale to apply to the background-SUBTRACTED image (optional; default 
          None (linear);  options are "log", "asinh")
        - whether to write the background-subtracted image (optional; default 
          True) 
        - name for the output background-subtracted image in a FITS file 
          (optional; default set below)
    
    Performs background subtraction and (optionally) cosmic ray rejection on 
    the input image. 
    
    Output: the background-subtracted image data in a fits HDU 
    """

    image_data = fits.getdata(im_file)
    
    ### SOURCE DETECTION ###
    # use image segmentation to find sources above SNR=3 and mask them 
    # for background estimation
    if mask_file: # load a bad pixel mask if one is present 
        bp_mask = fits.getdata(mask_file)
        bp_mask = bp_mask.astype(bool)
        source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                   dilate_size=15, mask=bp_mask)
        # combine the bad pixel mask and source mask 
        final_mask = np.logical_or(bp_mask,source_mask)
    else: 
        source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                   dilate_size=15)
        final_mask = source_mask
    
    ### BACKGROUND SUBTRACTION ###
    # estimate the background
    sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    bkg_estimator = MedianBackground()
    
    try: # try using photutils
        bkg = Background2D(image_data, box_size=bkg_box, filter_size=bkg_filt, 
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
                           mask=final_mask)
        bkg_img = bkg.background
    except ValueError: # error due to resampling during alignment? 
        print("\nCannot estimate background with photutils package. Using "+
              "median estimation with sigma clipping instead.")
        bkg_img = np.ndarray(image_data.shape)
        mean, median, std = sigma_clipped_stats(image_data, mask=final_mask)
        bkg_img.fill(median)
    
    bkgsub_img = image_data - bkg_img # subtract the background from the input
    
    
    ### COSMIC RAY REJECTION (OPTIONAL) ###
    # very large rejection sigma --> only reject most obvious cosmic rays 
    # objlim and sigclip determine this sigma 
    if crreject: # perform cosmic ray rejection
        import astroscrappy
        try:
            gain = fits.getheader(im_file)["GAIN"]
        except KeyError:
            gain = 1.0
        try:
            rdnoise = fits.getheader(im_file)["RDNOISE"]
        except KeyError:
            rdnoise = 0.0
        
        if mask_file: # if bp mask provided, pass to cosmic ray rejection
            crmask, crclean = astroscrappy.detect_cosmics(bkgsub_img, bp_mask, 
                                                          gain=gain,
                                                          readnoise=rdnoise,
                                                          objlim=12,
                                                          sigclip=12,
                                                          sigfrac=0.9)
            bp_mask = np.logical_or(bp_mask, crmask)
        else: 
            crmask, crclean = astroscrappy.detect_cosmics(bkgsub_img,
                                                          gain=gain,
                                                          readnoise=rdnoise,
                                                          objlim=12,
                                                          sigclip=12,
                                                          sigfrac=0.9)
            bp_mask = crmask
        
        # mask cosmic rays and/or bad pixels
        bkg_img_masked = np.ma.masked_where(bp_mask, bkg_img)
        bkg_img = np.ma.filled(bkg_img_masked, 0)   
        bkgsub_img_masked = np.ma.masked_where(bp_mask, bkgsub_img)
        bkgsub_img = np.ma.filled(bkgsub_img_masked, 0)
    
    else: # mask bad pixels only 
        if mask_file: # if bad pixel map provided 
            bkg_img_masked = np.ma.masked_where(bp_mask, bkg_img)
            bkg_img = np.ma.filled(bkg_img_masked, 0)
            
            bkgsub_img_masked = np.ma.masked_where(bp_mask, bkgsub_img)
            bkgsub_img = np.ma.filled(bkgsub_img_masked, 0)
        else: 
            bkg_img_masked = bkg_img
            bkgsub_img_masked = bkgsub_img
    
    ### PLOTTING (OPTIONAL) ###
    # plot the background, if desired
    if plot_bkg: 
        plt.figure(figsize=(14,13))
        # show WCS
        im_header = fits.getheader(im_file) # image header       
        w = wcs.WCS(im_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not scale_bkg: # if no scale to apply 
            scale_bkg = "linear"
            plt.imshow(bkg_img_masked, cmap='magma', aspect=1, 
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label="ADU", fontsize=16)
            
        elif scale_bkg == "log": # if we want to apply a log scale 
            bkg_img_log = np.log10(bkg_img_masked)
            lognorm = simple_norm(bkg_img_log, "log", percent=99.0)
            plt.imshow(bkg_img_log, cmap='magma', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
            
        elif scale_bkg == "asinh":  # asinh scale
            bkg_img_asinh = np.arcsinh(bkg_img_masked)
            asinhnorm = simple_norm(bkg_img_asinh, "asinh")
            plt.imshow(bkg_img_asinh, cmap="viridis", aspect=1, 
                       norm=asinhnorm, interpolation="nearest", origin="lower")
            cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("Image background", fontsize=15)
        plt.savefig("background_"+scale_bkg+".png", bbox_inches="tight")
        
    # plot the background-subtracted image, if desired
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS
        im_header = fits.getheader(im_file) # image header       
        w = wcs.WCS(im_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            plt.imshow(bkgsub_img_masked, cmap='magma', aspect=1, 
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label="ADU", fontsize=16)
            
        elif scale == "log": # if we want to apply a log scale 
            bkgsub_img_masked_log = np.log10(bkgsub_img_masked)
            lognorm = simple_norm(bkgsub_img_masked_log, "log", percent=99.0)
            plt.imshow(bkgsub_img_masked_log, cmap='magma', aspect=1, 
                       norm=lognorm, interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
            
        elif scale == "asinh":  # asinh scale
            bkgsub_img_masked_asinh = np.arcsinh(bkgsub_img_masked)
            asinhnorm = simple_norm(bkgsub_img_masked_asinh, "asinh")
            plt.imshow(bkgsub_img_masked_asinh, cmap="viridis", aspect=1, 
                       norm=asinhnorm, interpolation="nearest", origin="lower")
            cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("Background-subtracted image", fontsize=15)
        plt.savefig("background_sub_"+scale+".png", bbox_inches="tight")
    
    hdr = fits.getheader(im_file)
    bkgsub_hdu = fits.PrimaryHDU(data=bkgsub_img, header=hdr)
    
    if write: # if we want to write the background-subtracted fits file
        if not(output): # if no output name given, set default
            output = im_file.replace(".fits", "_bkgsub.fits")
        bkgsub_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return bkgsub_hdu
    
###############################################################################
#### IMAGE REGISTRATION (ALIGNMENT) ####

def image_align(source_file, template_file, thresh_sigma = 3.0,
                plot=False, scale=None, write=True, output_im=None, 
                output_mask=None):
    """
    Input: the science image (the source), the template to match to, a sigma 
    threshold for source detection (optional; default 3.0), a bool indicating 
    whether to plot the matched image data (optional; default False), a scale 
    to apply to the plot (optional; default None ("linear"), options are "log" 
    and "asinh"), whether to write the output .fits to files (optional; default 
    True) and names for the output aligned image and image mask (both optional; 
    defaults set below)
    
    Calls on astroalign to align the source image with the target to allow for 
    proper image subtraction. Also finds a mask of out of bounds pixels to 
    ignore during subtraction.
    
    *** Uses a slightly modified version of astroalign.
    
    Output: the aligned image data and a pixel mask
    """
    
    # a modified version of astroalign
    # allows the user to set the sigma threshold for source detection, which 
    # sometimes needs to be tweaked
    import astroalign_mod as aa
    
    source = fits.getdata(source_file)
    template = fits.getdata(template_file)
    
    # mask nans (not sure if this does anything)
    mask = np.isnan(template)
    template = np.ma.masked_where(mask, template)
    
    try: 
        img_aligned, footprint = aa.register(source, template, 
                                             thresh=thresh_sigma)
    except aa.MaxIterError: # if cannot match images 
        #print(source)
        source = np.flip(source, axis=0)
        source = np.flip(source, axis=1)
        print("\nMax iterations exceeded; flipping the image...\n")
        try:
            img_aligned, footprint = aa.register(source, template, 
                                                 thresh=thresh_sigma)
        except aa.MaxIterError:
            print("\nMax iterations exceeded while trying to find acceptable "+
                  "transformation. Exiting.\n")
            return
    except aa.TooFewStarsError:
        print("\nReference stars in source/template image are less than the "+
              "minimum value (3). Exiting.")
        return
    
    except Exception:
        print("\nSome error other than MaxIterError or TooFewStarsError was "+
              "raised. Exiting.")
        return
    
    # build the mask 
    # mask pixels==0 from the original source image and mask the footprint of
    # the image registration
    bool_img_aligned = np.logical_not(img_aligned.astype(bool))
    mask = np.logical_or(bool_img_aligned, footprint)
    
    # plot, if desired
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS
        template_header = fits.getheader(template_file) # image header       
        w = wcs.WCS(template_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        img_aligned_masked = np.ma.masked_where(mask, img_aligned)
        img_aligned = np.ma.filled(img_aligned_masked, 0)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            plt.imshow(img_aligned, cmap='magma', aspect=1, 
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label="ADU", fontsize=16)
            
        elif scale == "log": # if we want to apply a log scale 
            img_aligned_log = np.log10(img_aligned)
            lognorm = simple_norm(img_aligned_log, "log", percent=99.0)
            plt.imshow(img_aligned_log, cmap='magma', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
            
        elif scale == "asinh":  # asinh scale
            img_aligned_asinh = np.arcsinh(img_aligned)
            asinhnorm = simple_norm(img_aligned_asinh, "asinh")
            plt.imshow(img_aligned_asinh, cmap="viridis", aspect=1, 
                       norm=asinhnorm, interpolation="nearest", origin="lower")
            cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("Registered image via astroalign", fontsize=15)
        plt.savefig("aligned_source_image_"+scale+".png", bbox_inches="tight")
    
    # set header for new aligned fits file 
    hdr = fits.getheader(template_file)
    align_hdu = fits.PrimaryHDU(data=img_aligned, header=hdr)
    mask_hdu = fits.PrimaryHDU(data=mask.astype(int), header=hdr)
    
    if write: # if we want to write the aligned fits file and the mask 
        if not(output_im): # if no output name given, set default
            output_im = source_file.replace(".fits", "_align.fits")
        if not (output_mask): 
            output_mask = source_file.replace(".fits", "_align_mask.fits")
            
        align_hdu.writeto(output_im, overwrite=True, output_verify="ignore")
        mask_hdu.writeto(output_mask, overwrite=True, output_verify="ignore")
    
    return align_hdu, mask_hdu


def image_align_fine(source_file, template_file, mask_file=None, 
                    flip=False, plot=False, scale=None, write=True, 
                    output_im=None, output_mask=None):
    """
    Input: the science image (the source), the template to match to, a mask of
    bad pixels to ignore (optional; default None), a bool indicating whether to 
    flip (invert along x AND y) the image before trying to align (optional; 
    default False), a bool indicating whether to plot the matched image data 
    (optional; default False), a scale to apply to the plot (optional; default 
    None ("linear"), options are "log" and "asinh"), whether to write output 
    .fits to files (optional; default True) and names for the output aligned 
    image and image mask (both optional; defaults set below)
    
    Calls on image_registration to align the source image with the target to 
    allow for proper image subtraction. Also finds a mask of out of bounds 
    pixels to ignore during subtraction.
    
    Output: the aligned image data and a pixel mask
    """
    if flip:
        source = np.flip(fits.getdata(source_file))
    else: 
        source = fits.getdata(source_file)
    template = fits.getdata(template_file)
    
    
    import warnings # ignore warnings given by image_registration
    warnings.simplefilter('ignore', category=FutureWarning)
    
    from image_registration import chi2_shift
    from scipy import ndimage
    
    # pad the source array so that it has the same shape as the template
    # template must be larger than source
    if source.shape != template.shape:
        ypad = (template.shape[1] - source.shape[1])
        xpad = (template.shape[0] - source.shape[0])
        print("\nXPAD == "+str(xpad))
        print("YPAD == "+str(ypad)+"\n")
        
        if xpad > 0:
            source = np.pad(source, [(0,xpad), (0,0)], mode="constant", 
                                     constant_values=0)
        elif xpad < 0: 
            template = np.pad(template, [(0,abs(xpad)), (0,0)], 
                                         mode="constant", constant_values=0)
        if ypad > 0:
            source = np.pad(source, [(0,0), (0,ypad)], 
                                     mode="constant", constant_values=0)
        elif ypad < 0:
            template = np.pad(template, [(0,0), (0,abs(ypad))], 
                                         mode="constant", constant_values=0)
            
    if mask_file: # if a mask is provided
        mask = fits.getdata(mask_file)
        source = np.ma.masked_where(mask, source)
        masknans = np.isnan(template)
        template_mask = np.logical_or(mask, masknans)
        template = np.ma.masked_where(template_mask, template)
    else: # else, just mask nans in the template 
        masknans = np.isnan(template)
        template = np.ma.masked_where(masknans, template)
        
    # compute the required shift
    xoff, yoff, exoff, eyoff = chi2_shift(template, source, err=None, 
                                          return_error=True, 
                                          upsample_factor="auto",
                                          boundary="constant")
    
    if not(abs(xoff) < 20.0 and abs(yoff) < 20.0): # offsets should be very small  
        print("\nOffsets are both larger than 20.0 pix. Flipping the image "+
              "and trying again.\n") 
        source = np.flip(source) # try flipping the image 
        xoff, yoff, exoff, eyoff = chi2_shift(template, source, err=None, 
                                              return_error=True, 
                                              upsample_factor="auto",
                                              boundary="constant")
        if not(abs(xoff) < 20.0 and abs(yoff) < 20.0):
            print("\nAfter flipping, offsets are still both larger than 20.0 "+
                  "pix. Could not compute the fine alignment offset. Exiting.")
            return 
        
    # shift
    img_aligned = ndimage.shift(source, np.array((-yoff, -xoff)), order=3, 
                                mode='constant', cval=0.0, prefilter=True)   
    if mask_file:
        mask = np.logical_or((img_aligned == 0), mask)
    else: 
        mask = (img_aligned == 0)
    
    print("\nX OFFSET = "+str(xoff)+"+/-"+str(exoff))
    print("Y OFFSET = "+str(yoff)+"+/-"+str(eyoff)+"\n")

    # plot, if desired
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS
        template_header = fits.getheader(template_file) # image header       
        w = wcs.WCS(template_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        img_aligned_masked = np.ma.masked_where(mask, img_aligned)
        img_aligned = np.ma.filled(img_aligned_masked, 0)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            plt.imshow(img_aligned, cmap='magma', aspect=1, 
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label="ADU", fontsize=16)
            
        elif scale == "log": # if we want to apply a log scale 
            img_aligned_log = np.log10(img_aligned)
            lognorm = simple_norm(img_aligned_log, "log", percent=99.0)
            plt.imshow(img_aligned_log, cmap='magma', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
            
        elif scale == "asinh":  # asinh scale
            img_aligned_asinh = np.arcsinh(img_aligned)
            asinhnorm = simple_norm(img_aligned_asinh, "asinh")
            plt.imshow(img_aligned_asinh, cmap="viridis", aspect=1, 
                       norm=asinhnorm, interpolation="nearest", origin="lower")
            cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("Registered image via image_registration", fontsize=15)
        plt.savefig("aligned_source_image_"+scale+".png", bbox_inches="tight")
        
    hdr = fits.getheader(template_file)
    mask_hdu = fits.PrimaryHDU(data=mask.astype(int), header=hdr)
    align_hdu = fits.PrimaryHDU(data=img_aligned, header=hdr)
    
    if write: # if we want to write the aligned fits file and the mask 
        if not(output_im): # if no output name given, set default
            output_im = source_file.replace(".fits", "_align.fits")
        if not (output_mask): 
            output_mask = source_file.replace(".fits", "_align_mask.fits")
            
        align_hdu.writeto(output_im, overwrite=True, output_verify="ignore")
        mask_hdu.writeto(output_mask, overwrite=True, output_verify="ignore")
    
    return align_hdu, mask_hdu


def image_align_manual(source_file, template_file, mask_file, xoff, yoff,
                       plot=False, scale=None, write=True, output_im=None, 
                       output_mask=None):
    """
    WIP. NOT RELIABLE.
    Write description later.
    """
    
    from scipy import ndimage
    
    source = fits.getdata(source_file)
    mask = fits.getdata(mask_file)
    source = np.ma.masked_where(mask, source)
    
    # shift the image by the input xoff, yoff 
    img_aligned = ndimage.shift(source, [yoff, xoff], order=3, 
                                mode='constant', cval=0.0, prefilter=True)
    mask = np.logical_or((img_aligned == 0), mask)
    
    
    # plot, if desired
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS
        template_header = fits.getheader(template_file) # image header       
        w = wcs.WCS(template_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        img_aligned_masked = np.ma.masked_where(mask, img_aligned)
        img_aligned = np.ma.filled(img_aligned_masked, 0)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            plt.imshow(img_aligned, cmap='magma', aspect=1, 
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label="ADU", fontsize=16)
            
        elif scale == "log": # if we want to apply a log scale 
            img_aligned_log = np.log10(img_aligned)
            lognorm = simple_norm(img_aligned_log, "log", percent=99.0)
            plt.imshow(img_aligned_log, cmap='magma', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
            
        elif scale == "asinh":  # asinh scale
            img_aligned_asinh = np.arcsinh(img_aligned)
            asinhnorm = simple_norm(img_aligned_asinh, "asinh")
            plt.imshow(img_aligned_asinh, cmap="viridis", aspect=1, 
                       norm=asinhnorm, interpolation="nearest", origin="lower")
            cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("Registered image via manual alignment", fontsize=15)
        plt.savefig("manual_align_image_"+scale+".png", bbox_inches="tight")
        
    
    hdr = fits.getheader(template_file)
    mask_hdu = fits.PrimaryHDU(data=mask.astype(int), header=hdr)
    align_hdu = fits.PrimaryHDU(data=img_aligned, header=hdr)
    
    if write: # if we want to write the aligned fits file and the mask 
        if not(output_im): # if no output name given, set default
            output_im = source_file.replace(".fits", "_manalign.fits")
        if not (output_mask): 
            output_mask = source_file.replace(".fits", "_manalign_mask.fits")
            
        align_hdu.writeto(output_im, overwrite=True, output_verify="ignore")
        mask_hdu.writeto(output_mask, overwrite=True, output_verify="ignore")
    
    return align_hdu, mask_hdu

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
    
    Creates a simple box-shaped mask delimited by (xpix[0],xpix[1]) and 
    (ypix[0],ypix[1]). If an existing mask is supplied, the output mask will be 
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
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
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
                    sat_area_min=200, sat_area_max=100000, ra_safe=None, 
                    dec_safe=None, rad_safe=None, dilation_its=10, plot=True, 
                    write=True, output=None):
    """
    WIP:
        - Restriction by elongation currently disabled; not sure if a valid 
          discriminator 
    
    Input: 
        - image file
        - a mask file to merge with the created saturation mask (optional; 
          default None)
        - saturation ADU above which all pixels will be masked (optional; 
          default 40000, which is a bit below the limit for MegaCam)
        - minimum and maximum areas in square pixels for a saturated source 
          (optional; default 200 and 100000, respectively)
        - the RA (deg), DEC (deg), and radius (arcsec) denoting a "safe zone"
          in which no sources will be masked (optional; defaults are None)
        - the number of iterations of binary dilation to apply to the mask, 
          where binary dilation is a process which "dilates" the detected 
          sources to make sure all pixels are adequately masked (optional; 
          default 10)
        - whether to plot the mask in greyscale (optional; default True)
        - whether to write the mask file (optional; default True)
        - name for the output mask file (optional; default set below)
    
    Uses image segmentation to find all sources in the image. Then, looks for 
    sources which have a maximum flux above the saturation ADU and within the 
    minimum and maximum saturation area and creates a mask of these sources. 
    
    If a "safe zone" is supplied, any sources within the safe zone will be 
    labelled as NON-saturated. This is useful if you know the coordinates of 
    some galaxy/nebulosity in your image which should not be masked, as it is 
    sometimes difficult to distinguish between a saturated source and a galaxy.
    
    If an existing mask is supplied, the output mask will be a combination of 
    the previous mask and saturation mask.
    
    Output: a table of source properties and the mask file HDU 
    """    
    from scipy.ndimage import binary_dilation
    
    data = fits.getdata(image_file)
    hdr = fits.getheader(image_file)
    
    # image segmentation
    # use crude image segmentation to find sources above SNR=3
    if mask_file: # load a bad pixel mask if one is present 
        bp_mask = fits.getdata(mask_file)
        bp_mask = bp_mask.astype(bool)
        source_mask = make_source_mask(data, snr=3, npixels=5, 
                                   dilate_size=15, mask=bp_mask)
        # combine the bad pixel mask and source mask 
        rough_mask = np.logical_or(bp_mask,source_mask)
    else: 
        source_mask = make_source_mask(data, snr=3, npixels=5, 
                                   dilate_size=15)
        rough_mask = source_mask
    
    # estimate the background
    sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    bkg_estimator = MedianBackground()
    
    bkg = Background2D(data, (10,10), filter_size=(5,5), 
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
                       mask=rough_mask)
    bkg_rms = bkg.background_rms

    threshold = 3.0*bkg_rms # threshold for proper image segmentation 
    
    # get the segmented image and source properties
    # only detect sources composed of at least sat_area_min pixels
    segm = detect_sources(data, threshold, npixels=sat_area_min)
    labels = segm.labels 
    cat = source_properties(data, segm) 
    tbl = cat.to_table() # catalogue of sources as a table    
    # restrict based on area and maximum pixel ADU
    mask = tbl["area"].value <= sat_area_max # must have less than this area
    sat_labels = labels[mask]
    tbl = tbl[mask]
    mask = tbl["max_value"] >= sat_ADU # max value must exceed this ADU
    sat_labels = sat_labels[mask]
    tbl = tbl[mask]    
    # eliminate sources within the "safe zone"
    if (ra_safe and dec_safe and rad_safe):
        # get coordinates
        w = wcs.WCS(hdr)
        tbl["ra"], tbl["dec"] = w.all_pix2world(tbl["xcentroid"],
                                                tbl["ycentroid"], 1) 
        safe_coord = SkyCoord(ra_safe*u.deg, dec_safe*u.deg, frame="icrs")
        source_coords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                                 frame="icrs")
        sep = safe_coord.separation(source_coords).arcsecond # separations 
        tbl["sep"] = sep # add a column for separation from safe zone centre
        mask = tbl["sep"] > rad_safe # only select sources outside this radius
        sat_labels = sat_labels[mask]
        tbl = tbl[mask]       
    # keep only the remaining saturated sources
    segm.keep_labels(sat_labels)
    
    # build the mask
    newmask = segm.data_ma
    if mask_file: # combine with another mask 
        mask = fits.getdata(mask_file)
        newmask = np.logical_or(mask, newmask)            
    newmask[newmask >= 1] = 1 # masked pixels are labeled with 1
    newmask = newmask.filled(0) # unmasked labeled with 0    
    
    # use binary dilation to fill holes in mask, esp. near diffraction spikes
    newmask = binary_dilation(newmask, iterations=dilation_its)
    
    hdr = fits.getheader(image_file)
    mask_hdu = fits.PrimaryHDU(data=newmask.astype(int), header=hdr)
    
    if plot:
        plt.figure(figsize=(14,13))
        w = wcs.WCS(hdr)
        ax = plt.subplot(projection=w) # show WCS
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        plt.imshow(newmask, cmap='binary_r', aspect=1, interpolation='nearest', 
                   origin='lower')
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("saturation mask", fontsize=15)
        plt.savefig(image_file.replace(".fits","_satmask.png")) 
        plt.close()
    
    if write:
        if not(output):
            output = image_file.replace(".fits", "_satmask.fits")
            
        mask_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return tbl, mask_hdu

###############################################################################
#### IMAGE DIFFERENCING WITH HOTPANTS ####

def get_substamps(source_file, template_file, mask_file=None, sigma=3.0, 
                  elongation_max=3.0, area_max=50.0, output=None):
    """
    WIP:
        - Function does what it's asked, but not sure if the output file is 
          successfully understood and used by hotpants 
    
    Input: the science image (the source), the template to match to, a mask of
    bad pixels to ignore (optional; default None), the sigma to use in setting
    the threshold for image segmentation (optional; default 3.0*bkg_rms), 
    the maximum elongation allowed for the sources (optional; default 3.0), 
    the maximum pixel area allowed for the sources (optional; default 50.0
    pix**2), and the name for the output ascii txt file of substamps
    
    For both the science and template image, finds sources via image 
    segmentation and selects those which are not overly elongated (to filter 
    out galaxies) or overly large (to filter out oversaturated stars). Then 
    compares the science and template images, finds sources in common, and 
    writes these to an ascii text file of the form 
    x1 y1
    x2 y2
    x3 y3
     ...
     
    Output: the properties of the selected sources in both the science and 
    template images 
    """
    
    source_header = fits.getheader(source_file)
    source_data = fits.getdata(source_file)
    tmp_header = fits.getheader(template_file)
    tmp_data = fits.getdata(template_file)
    
    obj_props = []
    for image_data in [source_data, tmp_data]:
        # use crude image segmentation to find sources above SNR=3
        # allows us to ignore sources during background estimation
        if mask_file: # load a bad pixel mask if one is present 
            bp_mask = fits.getdata(mask_file)
            bp_mask = bp_mask.astype(bool)
            source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                       dilate_size=15, mask=bp_mask)
            # combine the bad pixel mask and source mask 
            final_mask = np.logical_or(bp_mask,source_mask)
        else: 
            source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                       dilate_size=15)
            final_mask = source_mask
        
        # estimate the background
        sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
        bkg_estimator = MedianBackground()
        
        bkg = Background2D(image_data, (10,10), filter_size=(5,5), 
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
                           mask=final_mask)
        bkg_rms = bkg.background_rms
    
        threshold = sigma*bkg_rms # threshold for proper image segmentation 
        
        # get the segmented image 
        # in practice, deblending leads to false positives --> we don't deblend
        segm = detect_sources(image_data, threshold, npixels=5)
        # get the source properties
        cat = source_properties(image_data, segm) 
        tbl = cat.to_table() # catalogue of sources as a table
        # restrict elongation and area to obtain only unsaturated stars 
        mask = (tbl["elongation"] <= elongation_max)
        tbl = tbl[mask]
        mask = tbl["area"].value <= area_max
        tbl = tbl[mask]
        obj_props.append(tbl)

    # get RA, Dec of sources after this restriction
    w = wcs.WCS(source_header)
    source_coords = w.all_pix2world(obj_props[0]["xcentroid"], 
                                    obj_props[0]["ycentroid"], 1)
    w = wcs.WCS(tmp_header)
    tmp_coords = w.all_pix2world(obj_props[1]["xcentroid"], 
                                 obj_props[1]["ycentroid"], 1)   
    source_skycoords = SkyCoord(ra=source_coords[0], dec=source_coords[1],
                                frame="icrs", unit="degree")
    tmp_skycoords = SkyCoord(ra=tmp_coords[0], dec=tmp_coords[1], frame="icrs", 
                             unit="degree")
    # find sources which are present in both the science image and template
    # sources must be <= 1.0" away from each other 
    idx_sci, idx_tmp, d2d, d3d = tmp_skycoords.search_around_sky(
                source_skycoords, 2.0*u.arcsec) 
    obj_props[0] = obj_props[0][idx_sci]
    obj_props[1] = obj_props[1][idx_tmp]
    
    # write x, y to a substamps ascii file 
    nmatch = len(obj_props[0])
    x = [np.mean([obj_props[0]["xcentroid"][i].value, 
                  obj_props[1]["xcentroid"][i].value]) for i in range(nmatch)]
    y = [np.mean([obj_props[0]["ycentroid"][i].value, 
                  obj_props[1]["ycentroid"][i].value]) for i in range(nmatch)]
    if not(output):
        output = source_file.replace(".fits", "_substamps.txt")
    lines = [str(x[n])+" "+str(y[n]) for n in range(nmatch)]
    np.savetxt(output, lines, fmt="%s", newline="\n")
    
    return obj_props
    

def hotpants(source_file, template_file, mask_file=None, substamps_file=None, 
             iu=None, tu=None, il=None, tl=None, gd=None, ig=None, ir=None, 
             tg=None, tr=None, ng=None, rkernel=None, convi=False, convt=False, 
             bgo=0, ko=1, output=None, maskout=None, convout=None, kerout=None, 
             v=1, plot=True, scale=None, target=None, target_small=None):
    """    
    hotpants args: 
        - the science image 
        - the template to match to
        - a mask of which pixels to ignore (optional; default None)
        - a text file containing the substamps 'x y' (optional; default None) 
        - the upper (u) and lower (l) ADU limits for the image (i) and 
          template (t) (optional; defaults set below)
        - good pixels coordinates (optional; default is full image)
            format: xmin xmax ymin ymax 
            e.g. gd="150 1000 200 800" 
        - the gain (g) and readnoise (r) for the image (i) and template (t) 
          (optional; default is to extract from headers or set gain=1, noise=0
          if no headers are found)
        - gaussian terms (optional; default set below)
            format: ngauss degree0 sigma0 ... degreeN sigmaN, N=ngauss-1
            e.g. 3 6.0 0.7 4.0 1.5 2.0 3.0 (default)
            where ngauss is the no. of gaussians which compose the kernel, 
            degreeI is the degree of the Ith polynomial, and sigmaI is the 
            width of the Ith gaussian
        - convolution kernel FWHM (optional; default if 2.5 times the FWHM of 
          the central Gaussian given in -ng, i.e. 2.5*1.5 = 4.375)
        - force convolve the image (optional; default False)
        - force convolve the template (optional; default False)
        - spatial background variation order (optional; default 0)
        - spatial kernel variation order (optional; default 1)
        - name for the output subtracted image (optional; default set below)
        - name for the output bad pixel mask (optional; default set below)
        - name for the output convolved image (optional; default set below)
        - name for the output kernel (optional; default set below)
        - verbosity (optional; default 1; options are 0 - 2 where 0 is least
          verbose)
        
    other args:
        - plot the subtracted image data (optional; default False)
        - scale to apply to the plot (optional; default None (linear); 
          options are "log" and "asinh")
        - target Ra, Dec at which to place crosshair (optional, default None)
        - target Ra, Dec at which to place smaller crosshair (optional, default 
          None)
    
    A wrapper for hotpants. 
    
    https://github.com/acbecker/hotpants
    HOTPANTS: High Order Transformation of Psf ANd Template Subtraction
    Based on >>> Alard, C., & Lupton, R. H. 1998, ApJ, 503, 325 <<<
    https://iopscience.iop.org/article/10.1086/305984/pdf 
    
    Output: the subtracted image HDU 
    """

    source_header = fits.getheader(source_file)
    source_data = fits.getdata(source_file)
    tmp_header = fits.getheader(template_file)
    tmp_data = fits.getdata(template_file)
    
    if not(output):
        output = source_file.replace(".fits", "_subtracted.fits")
    if not(maskout):
        maskout = source_file.replace(".fits", "_submask.fits")
    if not(convout):
        convout = source_file.replace(".fits", "_conv.fits")
    if not(kerout):
        kerout = source_file.replace(".fits", "_kernel.fits")
        
    ### INPUTS/OUTPUTS FOR HOTPANTS ###
    
    ## Input/output files and masks
    hp_options = " -inim "+source_file+" -tmplim "+template_file
    hp_options += " -outim "+output # output subtracted file
    hp_options += " -omi "+maskout # output bad pixel mask
    hp_options += " -oci "+convout # output convoluted file
    hp_options += " -oki "+kerout # output kernel file
    if mask_file: # if a mask is supplied
            mask = fits.getdata(mask_file)
            hp_options += " -tmi "+mask_file # mask for template 
            hp_options += " -imi "+mask_file # same mask for source 
    if substamps_file: # if a file of substamps X Y is supplied
        hp_options += " -afssc 0 -ssf "+substamps_file
        hp_options += " -savexy "+source_file.replace(".fits", "_conv")
            
    ## ADU upper/lower limits for the science image 
    if iu and il: # if both limits given
        hp_options += " -iu "+str(iu)+" -il "+str(il)
    else: # if values are not given
        # use image segmentation to find sources above SNR=3 and mask them 
        if mask_file: # load a bad pixel mask if one is present 
            bp_mask = fits.getdata(mask_file)
            bp_mask = bp_mask.astype(bool)
            source_mask = make_source_mask(source_data, snr=3, npixels=5, 
                                       dilate_size=15, mask=bp_mask)
            # combine the bad pixel mask and source mask 
            final_mask = np.logical_or(bp_mask,source_mask)
        else: 
            source_mask = make_source_mask(source_data, snr=3, npixels=5, 
                                       dilate_size=15)
            final_mask = source_mask    
        # estimate the background as just the median 
        mean, median, std = sigma_clipped_stats(source_data, mask=final_mask,
                                                maxiters=2)
        #  set the lower upper/lower thresholds to bkg_median +/- 5*bkg_rms
        hp_options += " -iu "+str(median+10*std)
        hp_options += " -il "+str(median-10*std)

        # for debugging
        print("\n\nSCIENCE UPPER LIMIT = "+str(median+10*std))
        print("SCIENCE LOWER LIMIT = "+str(median-10*std)+"\n") 

    ## ADU upper/lower limits for the template image 
    if tu and tl: # if both limits given
        hp_options += " -tu "+str(tu)+" -tl "+str(tl)
    else: # if values are not given
        # use image segmentation to find sources above SNR=3 and mask them 
        if mask_file: # load a bad pixel mask if one is present 
            bp_mask = fits.getdata(mask_file)
            bp_mask = bp_mask.astype(bool)
            source_mask = make_source_mask(tmp_data, snr=3, npixels=5, 
                                       dilate_size=15, mask=bp_mask)
            # combine the bad pixel mask and source mask 
            final_mask = np.logical_or(bp_mask,source_mask)
        else: 
            source_mask = make_source_mask(tmp_data, snr=3, npixels=5, 
                                       dilate_size=15)
            final_mask = source_mask    
        # estimate the background as just the median 
        mean, median, std = sigma_clipped_stats(tmp_data, mask=final_mask,
                                                maxiters=2)
        #  set the lower upper/lower thresholds to bkg_median +/- 5*bkg_rms
        hp_options += " -tu "+str(median+10*std)
        hp_options += " -tl "+str(median-10*std)
        
        # for debugging
        print("\n\nTEMPLATE UPPER LIMIT = "+str(median+10*std))
        print("TEMPLATE LOWER LIMIT = "+str(median-10*std)+"\n")
    
    ## x, y limits 
    if gd: 
        hp_options += " -gd "+gd
        
    ## gain (e-/ADU) and readnoise (e-) for image and template 
    if ig: # if argument is supplied for image gain
        hp_options += " -ig "+str(ig)
    else: # if no argument given, extract from header
        try:
            hp_options += " -ig "+str(source_header["GAIN"])
        except KeyError: # if no keyword GAIN found in header 
            pass      
    if ir: # image readnoise
        hp_options += " ir "+str(ir)
    else:
        try:
            hp_options += " -ir "+str(source_header["RDNOISE"])
        except KeyError: 
            pass
    if tg: # if argument is supplied for template gain
        hp_options += " -tg "+str(tg)
    else: # if no argument given, extract from header
        try:
            hp_options += " -tg "+str(tmp_header["GAIN"])
        except KeyError: # if no keyword GAIN found in header 
            pass     
    if tr: # template readnoise
        hp_options += " tr "+str(ir)
    else:
        try:
            hp_options += " -tr "+str(tmp_header["RDNOISE"])
        except KeyError: 
           pass
    
    ## gaussian parameters, convolution kernel parameters, forced convolutions
    if ng: # gaussian term 
        hp_options += " -ng "+ng
    if rkernel: # convolution kernel FWHM 
        hp_options += " -r "+str(rkernel)
        hp_options += " -rss "+str(2.0*rkernel) # cutout around centroids 
    if convi: # force convolution of the image
        hp_options += " -c i"  
    if convt: # force convolution of the template
        hp_options += " -c t" 
        
    ## spatial kernel/background variation 
    hp_options += " -bgo "+str(bgo)+" -ko "+str(ko) # bg order, kernel order     

    ## misc
    #hp_options += " -fi 0.0" # fill value for bad pixels 
    hp_options += " -v "+str(v) # verbosity 
    
    run("~/hotpants/hotpants"+hp_options, shell=True) # call hotpants
    
    # if file successfully produced and non-empty
    outfile = re.sub(".*/", "", output) # for a file /a/b/c, extract the "c"
    topdir = output[:-len(outfile)] # extract the "/a/b/"
    if (outfile in os.listdir(topdir)) and (os.stat(output).st_size!=0):
        sub = fits.getdata(output) # load it in 
        #sub_bp_mask = fits.getdata(maskout).astype(bool)
        zero_mask = sub==0
    else:
        return
    
    # mask bad pixels
    if mask_file:
        mask = np.logical_or(mask, zero_mask)
        sub = np.ma.masked_where(mask, sub)
        sub = np.ma.filled(sub, 0) # set ALL bad pixels to 0 in image
    else:
        sub = np.ma.masked_where(zero_mask, sub)
        sub = np.ma.filled(sub, 0) # set bad pixels to 0 in image
    
    sub_header = fits.getheader(output)
    try:
        mean_diff = float(sub_header["DMEAN00"]) # mean of diff img good pixels
        std_diff = float(sub_header["DSIGE00"]) # stdev of diff img good pixels
        print("\nMEAN OF DIFFERENCE IMAGE: %.2f ± %.2f \n"%(
                mean_diff, std_diff))
    except KeyError:
        print("\nCould not find DMEAN00 and DSIGE00 (mean and standard dev. "+
              "of difference image good pixels) headers in difference image.")
        print("The difference image was probably not correctly produced. "+
              "Exiting.\n")
        return 
        
    ### PLOTTING (OPTIONAL) ###
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS      
        w = wcs.WCS(source_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            #mean, median, std = sigma_clipped_stats(sub)
            plt.imshow(sub, cmap='coolwarm', vmin=mean_diff-3*std_diff, 
                       vmax=mean_diff+3*std_diff, aspect=1, 
                       interpolation='nearest', origin='lower')
            plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            
        elif scale == "log": # if we want to apply a log scale 
            sub_log = np.log10(sub)
            lognorm = simple_norm(sub_log, "log", percent=99.0)
            plt.imshow(sub_log, cmap='magma', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            
        elif scale == "asinh":  # asinh scale
            sub_asinh = np.arcsinh(sub)
            asinhnorm = simple_norm(sub, "asinh")
            plt.imshow(sub_asinh, cmap="viridis", aspect=1, norm=asinhnorm,
                       interpolation="nearest", origin="lower")
            plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            
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
        plt.title("hotpants image difference", fontsize=15)
        plt.savefig(outfile.replace(".fits", "_hotpants.png"), 
                    bbox_inches="tight")
        plt.close()
        
    return sub


def transient_detect_hotpants(sub_file, og_file, sigma=5, pixelmin=5, 
                              elongation_lim=2.0, plots=True, 
                              sub_scale="asinh", og_scale="asinh", 
                              stampsize=200.0, crosshair="#fe019a",
                              toi=None, toi_sep_min=None, toi_sep_max=None):
    """    
    Inputs:
        - subtracted image file
        - original science image file (can be background subtracted or not)
        - sigma above which to look for transients in difference image 
          (optional; default 5.0)
        - minimum number of pixels to count as a source (optional; default 5)
        - maximum allowed elongation for sources found by image segmentation 
          (optional; default 2.0)
        - whether to plot the candidate transient sources with a crosshair 
          denoting their positions over all of the following:
              (1) the subtracted image 
              (2) "postage stamp" of the original unsubtracted image 
              (3) postage stamp of the subtracted image 
              (optional; default True)
        - scale to apply to the difference images (optional; default "asinh"; 
          options are "linear", "log", "asinh")
        - scale to apply to the original unsubtracted image (optional; default
          "asinh"; options are "linear", "log", "asinh")
        - size of the transient stamp in pixels (optional; default 200.0)
        - colour for crosshair on transient (optiona; default ~ hot pink)
        - [ra,dec] for some target of interest (e.g., a candidate host galaxy)
          such that the distance to this target will be computed for every 
          candidate transient (optional; default None)
        - minimum separation between the target of interest and the transient
          (optional; default None; only relevant if TOI is supplied)
        - maximum separation between the target of interest and the transient 
          (optional; default None, only relevant if TOI is supplied)
    
    Looks for sources with flux > sigma*std, where std is the standard 
    deviation of the good pixels in the subtracted image. Sources must also be 
    made up of at least pixelmin pixels. From these, selects sources below some 
    elongation limit to try to prevent obvious residuals from being detected as 
    sources. For each candidate transient source, optionally plots the full 
    subtracted image and "postage stamps" of the transient on the original 
    science image and subtracted image. 
    
    Output: a table of sources with their properties (coordinates, area, 
    elongation, separation from a target of interest (if relevant), etc.)
    """
    
    data = fits.getdata(sub_file)
    hdr = fits.getheader(sub_file)
    try: 
        std = float(hdr['DSIGE00']) # good pixels standard deviation 
    except KeyError:
        std = np.std(data)

    # use image segmentation to find sources with an area >pixelmin pix**2 
    # which are above the threshold sigma*std
    data = np.ma.masked_where(data==0.0, data) # mask bad pixels
    segm = detect_sources(data, sigma*std, npixels=pixelmin)  
    cat = source_properties(data, segm) 
    try:
        tbl = cat.to_table()
    except ValueError:
        print("SourceCatalog contains no sources. Exiting.")
        return
    mask = tbl["elongation"] < 2.0 # restrict based on elongation
    tbl = tbl[mask]
    
    # get coordinates
    w = wcs.WCS(hdr)
    tbl["ra"], tbl["dec"] = w.all_pix2world(tbl["xcentroid"], 
                                            tbl["ycentroid"], 1)
    
    # if provided, only look for sources within a certain angular separation
    # of the target of interest
    if toi != None: 
        toi_coord = SkyCoord(toi[0]*u.deg, toi[1]*u.deg, frame="icrs")
        trans_coord = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, frame="icrs")
        sep = toi_coord.separation(trans_coord).arcsecond
        tbl["sep"] = sep # add a column for separation
        mask = tbl["sep"] < toi_sep_max
        tbl = tbl[mask]
        mask = tbl["sep"] > toi_sep_min
        tbl = tbl[mask]        
    
    if plots: # plot crosshairs at locations of transients 
        targets = [tbl["ra"].data, tbl["dec"].data]
        ntargets = len(targets[0])
        for i in range(ntargets):
            # set titles
            title = "difference image: candidate "+str(i)
            title_og = "original image: candidate "+str(i)
            # set output filenames
            #output = sub_file.replace(".fits", "_candidate"+str(i)+".png")
            output_og = og_file.replace(".fits", 
                                          "_zoomed_candidate"+str(i)+".png")
            output_zoom = sub_file.replace(".fits", 
                                          "_zoomed_candidate"+str(i)+".png")
            
            # crosshair on full-frame difference image 
            #make_image(sub_file, 
            #           scale=sub_scale, cmap="coolwarm", label="", title=title, 
            #           output=output, 
            #           target=toi,
            #           target_small=[targets[0][i], targets[1][i]])
            
            # crossair on small region of science image and difference image 
            transient_stamp(og_file, 
                            [targets[0][i], targets[1][i]], 
                            stampsize, 
                            scale=og_scale, cmap="magma", 
                            crosshair=crosshair,
                            title=title_og, output=output_og,
                            toi=toi)
            transient_stamp(sub_file, 
                            [targets[0][i], targets[1][i]], 
                            stampsize, 
                            scale=sub_scale, cmap="magma", label="",
                            crosshair=crosshair,
                            title=title, output=output_zoom,
                            toi=toi)
    return tbl
    
###############################################################################
#### IMAGE DIFFERENCING WITH PROPER IMAGE SUBTRACTION ####
#### Based on 2016 work by Barak Zackay, Eran O. Ofek and Avishay Gal-Yam ####
#### This section is a work in progress. ####

def derive_ePSF(image_file, sigma=8.0, write=True, output=None, plot=False, 
                  output_plot=None):
    """    
    Input: 
        - filename for an image
        - sigma to use in source detection with astrometry.net (optional; 
          default 8.0)
        - whether to write the ePSF to a new fits file (optional; default True)
        - name for the output fits file (optional; default set below)
        - whether to plot the ePSF (optional; default False) 
        - name for the output plot (optional; default set below)
    
    Uses astrometry.net to obtain a list of sources in the image with their 
    x, y coordinates, flux, and background at their location. Then selects  
    stars between the 80th and 90th percentile flux and uses EPSFBuilder to 
    empirically obtain the ePSF of these stars. 
    
    Output: the epsf data array
    """
    from astropy.table import Table, Column
    #from astropy.modeling.fitting import LevMarLSQFitter
    #from photutils.psf import (BasicPSFPhotometry, DAOGroup)
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
    
    # setup: get WCS coords for all sources 
    # use astrometry.net to find the sources 
    # BANDAID FIX: can't locate image2xy at the moment 
    options = " -O -p "+str(sigma)+" "
    run("/usr/local/astrometry/bin/image2xy"+options+image_file, shell=True)
    image_sources_file = image_file.replace(".fits", ".xy.fits")
    image_sources = fits.getdata(image_sources_file)
    run("rm "+image_sources_file, shell=True) # this file is not needed
    
    x = np.array(image_sources['X'])
    y = np.array(image_sources['Y'])
    w = wcs.WCS(image_header)
    wcs_coords = np.array(w.all_pix2world(x,y,1))
    ra = Column(data=wcs_coords[0], name='ra')
    dec = Column(data=wcs_coords[1], name='dec')
    print("\n"+str(len(ra))+" stars at >"+str(sigma)+" sigma found with "+
          "astrometry.net")
    
    sources = Table() # build a table 
    sources['x_mean'] = image_sources['X'] # for BasicPSFPhotometry
    sources['y_mean'] = image_sources['Y']
    sources['x'] = image_sources['X'] # for EPSFBuilder 
    sources['y'] = image_sources['Y']
    sources.add_column(ra)
    sources.add_column(dec)
    sources['flux'] = image_sources['FLUX']-image_sources["BACKGROUND"]
 
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
        
    # empirically obtain the effective Point Spread Function (ePSF)  
    nddata = NDData(image_data) # NDData object
    stars = extract_stars(nddata, sources, size=35) # extract stars
    
    # use only the stars with fluxes between two percentiles
    stars_tab = Table() # temporary table 
    stars_col = Column(data=range(len(stars.all_stars)), name="stars")
    stars_tab["stars"] = stars_col # column of indices of each star
    fluxes = [s.flux for s in stars]
    fluxes_col = Column(data=fluxes, name="flux")
    stars_tab["flux"] = fluxes_col # column of fluxes
    
    # get percentiles
    per_low = np.percentile(fluxes, 70) # 70th percentile flux 
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
    
    # get rid of negligibly small values and make sure borders are 0
    epsf_data = epsf.data
    #epsf_data[epsf_data<0.001] = 0.0 
    #epsf_data[0:,0:1], epsf_data[0:1,0:] = 0, 0
    #epsf_data[0:,-1:], epsf_data[-1:,0:] = 0, 0
    
    # set pixels outside a certain radius to 0 
    y, x = np.ogrid[:35, :35]
    center = [17.5, 17.5]
    rad = 10.0
    dist_from_center = np.sqrt((x-center[0])**2 + (y-center[0])**2)
    mask = dist_from_center >= rad
    epsf_data[mask] = 0
    
    epsf_hdu = fits.PrimaryHDU(data=epsf_data)
    if write: # write, if desired
        if not(output):
            output = image_file.replace(".fits", "_ePSF.fits")
            
        epsf_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    if plot:
        plt.figure(figsize=(10,9))
        plt.imshow(epsf_data, origin='lower', aspect=1, cmap='magma',
                   interpolation="nearest")
        plt.xlabel("Pixels", fontsize=16)
        plt.ylabel("Pixels", fontsize=16)
        plt.title("Effective Point-Spread Function", fontsize=16)
        plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        plt.rc("xtick",labelsize=16) # not working?
        plt.rc("ytick",labelsize=16)
    
        if not(output_plot):
            output_plot = image_file.replace(".fits", "_ePSF.png")
        plt.savefig(output_plot, bbox_inches="tight")
        plt.close()
    
    return epsf_data


def __plot_num_denom(filename, quant, data, term, scale=None):
    """
    Inputs: filename, the quantity to plot ("numerator" or "denominator"), 
    the data array, the term ("1" or "2") and the scale to apply (optional; 
    default None ("linear"); options are "linear", "log", "asinh")
    Output: None
    """
    # determine plot title
    if quant=="numterm": # a term in the numerator
        data = fft.ifft2(data)
        titles = [r"$F_n {F_r}^2 \overline{\widehat{P_n}} |\widehat{P_r}|^2$"+
                  r"$\widehat{N}$", r"$F_r {F_n}^2 \overline{\widehat{P_r}}$"+
                  r"$|\widehat{P_n}|^2 \widehat{R}$"]
        titles = [r"$|{ }\mathfrak{F}^{-1}\{$"+t+r"$\}{ }|$" for t in titles]       
    elif quant=="denomterm": # a term in the denominator
        #data = fft.fftshift(data)
        titles = [r"$ \sigma_r^2 {F_n}^2 |\widehat{P_n}|^2$", 
                  r"$ \sigma_n^2 {F_r}^2 |\widehat{P_r}|^2$"]
    elif quant=="denominator": # 1 / the entire denominator 
        data = fft.ifft2(data)
        title = r"$ \sigma_r^2 {F_n}^2 |\widehat{P_n}|^2 + $"
        title += r"$\sigma_n^2 {F_r}^2 |\widehat{P_r}|^2$"
        title = r"$|{ }\mathfrak{F}^{-1}\{ ($"+title+r"$)^{-1} \}{ }|$"
    
    if term == 1:
        title = titles[0]
    elif term == 2:
        title = titles[1]
    
    plt.figure(figsize=(14,13))      
    w = wcs.WCS(fits.getheader(filename))
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    # take the norm of the complex-valued array   
    data = np.sqrt(np.abs(data*np.conj(data))) 
    
    if not scale: # if no scale to apply 
        scale = "linear"
        mean, median, std = sigma_clipped_stats(data)
        plt.imshow(data, cmap='coolwarm', vmin=mean-3*std, 
                   vmax=mean+3*std, aspect=1, 
                   interpolation='nearest', origin='lower')
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        
    elif scale == "log": # if we want to apply a log scale 
        data_log = np.log10(data)
        lognorm = simple_norm(data_log, "log", percent=99.0)
        plt.imshow(data_log, cmap='magma', aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        
    elif scale == "asinh":  # asinh scale
        data_asinh = np.arcsinh(data)
        asinhnorm = simple_norm(data, "asinh")
        plt.imshow(data_asinh, cmap="viridis", aspect=1, norm=asinhnorm,
                   interpolation="nearest", origin="lower")
        plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
    
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    plt.title(title, fontsize=20)
    if quant=="numerator":
        plt.savefig(filename.replace(".fits","_propersub_num"+str(term)+
                                     ".png"), bbox_inches="tight")
    elif quant=="denomterm":
        plt.savefig(filename.replace(".fits","_propersub_denom"+str(term)+
                                     ".png"), bbox_inches="tight")
    elif quant=="denominator":
        plt.savefig(filename.replace(".fits","_propersub_denom.png"), 
                    bbox_inches="tight")
    plt.close()
    

def __solve_beta(nd, rd, pn, pr, denom1, denom2, beta0, its=10, plot=True, 
                 hdr=None):
    """
    Input: 
        - new image data
        - reference image data
        - new image ePSF
        - reference image ePSF
        - first term in the denominator of D_hat
        - second term in the denominator of D_hat
        - initial guess for the beta parameter 
        
        - maximum no. of iterations to use in the solver (optional; default 3)
        - whether the ePSFs are padded beforehand (optional; default False)
        - whether to plot the result (optional; default True)
        - image header to use to obtain WCS axes for the plot (optional; 
          default None, only necessary if plotting result)
    
    Uses Newton-Krylov nonlinear solving go obtain a solution for the beta 
    parameter used in proper image subtraction. 
    
    Output: the beta parameter
    """
    from scipy.optimize import newton_krylov
    
    def F_compute(beta):
        # compute F(beta) to find beta for which F(beta) is minimized
        # nans are not preserved because Newton-Krylov cannot handle them 
        dn = fft.fft2(convolve_fft(nd, pr, fftn=fft.fft2, ifftn=fft.ifft2,
                                   normalize_kernel=False)/np.sqrt(
                                           denom1*beta**2 + denom2))

        dr = fft.fft2(convolve_fft(rd, pn, fftn=fft.fft2, ifftn=fft.ifft2,
                                   normalize_kernel=False)/np.sqrt(
                                           denom1*beta**2 + denom2))
        
        dn = np.sqrt(np.abs(dn*np.conj(dn)))
        dr = np.sqrt(np.abs(dr*np.conj(dr)))
        return dn - beta*dr # should this be allowed to be complex? 
    
    # use Newton-Krylov to find a solution (initial guess: beta=fzero_n)
    beta0 = np.ones(shape=nd.shape)*beta0
    #F0 = F_compute(beta0)
    # at the moment, does not converge very fast...
    soln = newton_krylov(F_compute, beta0, verbose=True, iter=its)
    soln = np.abs(soln)
    
    if plot: # plot the beta parameter
        plt.figure(figsize=(14,13))
        # show WCS      
        w = wcs.WCS(hdr)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        mean, median, std = sigma_clipped_stats(soln)
        plt.imshow(soln, cmap='coolwarm', vmin=mean-3*std, 
                   vmax=mean+3*std, aspect=1, 
                   interpolation='nearest', origin='lower')
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        plt.title(r"$\beta$"+" parameter", fontsize=16)
        plt.savefig("beta.png", bbox_inches="tight")        
        
    return soln
    

def __inter_pix(data, data_std, mask_2replace, dpix=10, k=3):
    """
    Input: data array, std dev of each pixel, the mask to replace via spline
    fitting, size of each row to be spline fit (optional; default 10), and 
    the dimension of the spline (optional; default 3 (cubic)) 
    
    Function to replace mask pixels with spline fit along rows. Taken directly
    from 
    https://github.com/pmvreeswijk/ZOGY/blob/master/zogy.py
    
    Output: the updated data
    """

    from scipy import ndimage, interpolate
    
    data_replaced = np.copy(data)
    # if [data_std] is a scalar, convert it to an array with the same
    # shape as data
    if np.isscalar(data_std):
        data_std *= np.ones(np.shape(data))
        
    # label consecutive pixels in x as distinct region
    regions, nregions = ndimage.label(mask_2replace, 
                                      structure=[[0,0,0],[1,1,1],[0,0,0]])
    # determine slices corresponding to these regions
    obj_slices = ndimage.find_objects(regions)

    # iterate over regions to replace
    ymax, xmax = np.shape(data)    
    for i in range(nregions):
    
        y_slice = obj_slices[i][0]
        x_slice = obj_slices[i][1]
        
        i_start = max(x_slice.start-dpix, 0)
        i_stop = min(x_slice.stop+dpix+1, xmax)
        
        x_row = np.arange(i_start, i_stop)
        print ('x_row: {}'.format(x_row))
        y_index = y_slice.start
        mask_row = mask_2replace[y_index, x_row]
        print ('mask_row: {}'.format(mask_row))
        data_row = data_replaced[y_index, x_row]
        print ('data_row: {}'.format(data_row))
        data_std_row = data_std[y_index, x_row]
        print ('data_std_row: {}\n'.format(data_std_row))

        # data to fit
        x_fit = x_row[~mask_row]
        y_fit = data_row[~mask_row]
        w_fit = 1./data_std_row[~mask_row]

        # spline fit       
        try:
            fit_spline = interpolate.UnivariateSpline(x_fit, y_fit, w=w_fit, 
                                                      k=k, check_finite=True)
        except:
            print('Warning: spline fit in [inter_pix] to region '+str(i)+
                  ' with slice {} failed; pixel values not updated'.format(
                          obj_slices[i]))
            continue
        else:    
            # replace masked entries with interpolated values
            x_fill = x_row[mask_row]
            data_replaced[y_index, x_fill] = fit_spline(x_fill)
        
    return data_replaced


def __kernel_recenter(kernel):
    """
    Input: the kernel (can be complex)
    Output: the re-centered kernel
    """
    from scipy.ndimage import shift, center_of_mass
    
    if type(kernel[0][0]) == np.complex128: # if complex
        kernel_real = np.real(kernel) # real
        kernel_imag = np.imag(kernel) # imaginary
        # shift real component
        arr_center = np.array([kernel_real.shape[0]/2.0, 
                               kernel_real.shape[1]/2.0])
        kern_center = np.array(center_of_mass(kernel_real))
        off = arr_center - kern_center
        yoff, xoff = off[0], off[1]
        recentered_real = shift(kernel_real, [yoff, xoff], order=3, 
                                mode='constant', cval=1e-20, prefilter=False)
        # shift imaginary component
        arr_center = np.array([kernel_imag.shape[0]/2.0, 
                               kernel_imag.shape[1]/2.0])
        kern_center = np.array(center_of_mass(kernel_imag))
        off = arr_center - kern_center
        yoff, xoff = off[0], off[1]
        recentered_imag = shift(kernel_imag, [yoff, xoff], order=3, 
                                mode='constant', cval=1e-20, prefilter=False)
        # combine them 
        recentered = recentered_real + 1j*recentered_imag
        
    else:   
        arr_center = np.array([kernel.shape[0]/2.0, kernel.shape[1]/2.0])
        kern_center = np.array(center_of_mass(kernel))
        off = arr_center - kern_center
        yoff, xoff = off[0], off[1]
        recentered = shift(kernel, [yoff, xoff], order=3, 
                           mode='constant', cval=1e-20, prefilter=False)
    
    return recentered


def __arrays_adjust(arrays, parity="odd"):
    """
    Input: as many arrays as desired and the desired parity ("odd" or "even")
    for the output arrays (optional; default "odd")
    
    Resizes all of the input arrays so that their dimensions are the same by 
    cropping arrays to the smallest dimension present. Then, enforces that the 
    dimensions are either both odd or both even.
    
    Output: the resized arrays
    """
    
    if len(arrays.shape) == 2: # if only a single 2d array
        yshape, xshape = np.array(arrays.shape)
        if parity == "odd": # make it odd
            xshape = xshape + xshape%2 - 1
            yshape = yshape + yshape%2 - 1
        else: # make it even 
            xshape = xshape - xshape%2  
            yshape = yshape - yshape%2
        arrays = arrays[:yshape, :xshape]
    
    else:
        ys = [a.shape[0] for a in arrays]
        xs = [a.shape[1] for a in arrays]
        yshape = min(ys)
        xshape = min(xs)
        if parity == "odd": # make it odd
            xshape = xshape + xshape%2 - 1
            yshape = yshape + yshape%2 - 1
        else: # make it even 
            xshape = xshape - xshape%2  
            yshape = yshape - yshape%2            
        arrays = arrays[:, :yshape, :xshape]
    
    return arrays


def proper_subtraction(new_file, new_og, ref_file, ref_og, 
                       pn_file=None, pr_file=None, mask_file=None, 
                       parity="odd", 
                       bpix_fill=True, interpix=False, 
                       findbeta=True, fzero_n=1, fzero_r=1, 
                       sigma=8.0, ADU_max_new=50000, ADU_max_ref=50000, 
                       pnan=False, nnan="interpolate", 
                       write=True, output=None, 
                       plot="S", plot_numerator=True, plot_denominator=True, 
                       scale=None, target=None, target_small=None, 
                       # following are for debugging
                       pad=False, shift=False, border=True):
    """ 
    WIP:
        - Denominator looks funky for S_hat 
        - Beta parameter takes very long to compute 
        - Not clear if convolution is better than direct matrix multiplication
    
    Inputs: 
        - new (science) file (aligned and background-subtracted)
        - aligned but **NOT** background-subtracted new file
        - reference (template) file (aligned and background-subtracted)
        - aligned but **NOT** background-subtracted reference file

        - file for the new image ePSF (optional; default None, in which case a 
          new one will be created)
        - file for the ref image ePSF (optional; default None, in which case a
          new one will be created)
        - mask file for the new and ref images (optional; default None)
        - the parity to enforce for all of the data arrays (optional; default 
          "odd", can be "odd" or "even")
        - whether to fill masked pixels with the background at their location
          (optional; default True)
        - whether to use a spline fitter to interpolate values for masked 
          pixels (optional; default False; EXPERIMENTAL)
        - whether to solve for the beta parameter (optional; default True)
        - which statistic to plot (optional; default S, options are 'S', 'D', 
          or None)
        - flux zero point for the new image (optional; default 1, only used if 
          beta is not computed)
        - flux zero point for the ref image (optional; default 1, only used if 
          beta is not computed)
        - sigma to use during source detection with astrometry.net (optional; 
          default 8.0, only used if no ePSFs are supplied ) 
        - maximum allowed ADU in new image (optional; default 50000 ADU)
        - maximum allowed ADU in ref image (optional; default 50000 ADU)   
        - whether to preserve nans during convolution or to interpolate/fill 
          them (optional; default False)
        - the treatment for nans (optional; default "interpolate", only 
          relevant if pnan=False)
        - whether to write the subtraction to a fits file (optional; default 
          True) 
        - name for the output fits file (optional; default set below)
        - whether to plot the "S" statistic, "D" statistic, or neither 
          (optional; default "S") 
        - whether to plot the norm of the inverse FFT of both terms of the 
          numerator of S_hat (optional; default True)
        - whether to plot the norm of the inverse FFT of both terms of the 
          denominator of S_hat (optional; default True)
        - scale to apply to the plots (optional; default None (linear); options 
          are "log" and "asinh")
        - [ra,dec] for a target crosshair (optional; default None)
        - [ra,dec] for second smaller target crosshair (optional; default None)
        
        - pad the ePSFs and use direct matrix multiplication rather than 
          convolution to get statistics (optional; default False; for 
          debugging)
        - call fftshift on the padded, FFT'd ePSFs (optional; default False, 
          for debugging)
        - mask border pixels of data arrays (optional; default True; for debug)
    
    Computes the most basic D and S-statistics for transient detection, 
    presented in Zackay et al., 2016. Does not compute S_corr and other 
    quantities. These will be added in the future. 
    
    Output: the D and S statistics
    """
    
    # load in data
    new_data, ref_data = fits.getdata(new_file), fits.getdata(ref_file)
    new_data_og, ref_data_og = fits.getdata(new_og), fits.getdata(ref_og)
    new_header = fits.getheader(new_file)
    print("\nN = "+new_file+" (new image)")
    print("R = "+ref_file+" (reference image)\n")
    
    ### adjusting arrays ###
    print("new data: "+str(new_data.shape))
    print("ref data: "+str(ref_data.shape))
    print("og new data: "+str(new_data_og.shape))
    print("og ref data: "+str(ref_data_og.shape)+"\n") 
    print("bmpask data: "+str(fits.getdata(mask_file).shape)+"\n") 
    
    start = timer()
    if mask_file: 
        bp_mask = fits.getdata(mask_file)
        arrs = __arrays_adjust(np.array([new_data, ref_data, new_data_og, 
                                           ref_data_og, bp_mask]), parity)
        new_data, ref_data, new_data_og, ref_data_og, bp_mask = arrs
    else:
        arrs = __arrays_adjust(np.array([new_data, ref_data, new_data_og, 
                                         ref_data_og]), parity)
        new_data, ref_data, new_data_og, ref_data_og = arrs
    print("new data: "+str(new_data.shape))
    print("ref data: "+str(ref_data.shape))
    print("og new data: "+str(new_data_og.shape))
    print("og ref data: "+str(ref_data_og.shape)+"\n")
    print("bmpask data: "+str(bp_mask.shape)+"\n")
    
    ## find minimum image size in both dimensions
#    if mask_file: # load a bad pixel mask if one is present 
#        bp_mask = fits.getdata(mask_file)       
#        xshape = min(new_data.shape[1], ref_data.shape[1], 
#                        new_data_og.shape[1], ref_data_og.shape[1],
#                        bp_mask.shape[1])
#
#        yshape = min(new_data.shape[0], ref_data.shape[0], 
#                        new_data_og.shape[0], ref_data_og.shape[0],
#                        bp_mask.shape[0])
#    else: 
#        xshape = min(new_data.shape[1], ref_data.shape[1], 
#                     new_data_og.shape[1], ref_data_og.shape[1])
#        yshape = min(new_data.shape[0], ref_data.shape[0], 
#                     new_data_og.shape[0], ref_data_og.shape[0])
#        
    ## ensure all arrays have same dimensions and make sure dimensions are ODD
#    if xshape%2 == 0:
#        xshape -= 1
#    if yshape%2 == 0:
#        yshape -= 1        
#    new_data = new_data[:yshape, :xshape]
#    new_data_og = new_data_og[:yshape, :xshape]
#    ref_data = ref_data[:yshape, :xshape]
#    ref_data_og = new_data_og[:yshape, :xshape]    
#    if mask_file:
#        bp_mask = bp_mask[:yshape, :xshape]
#    if new_data.shape[0]%2 == 0:
#        new_data = new_data[:-1,:] # trim last pixel in y-dim
#        ref_data = ref_data[:-1,:]
#    if new_data.shape[1]%2 == 0:
#        new_data = new_data[:,:-1] # trim last pixel in x-dim
#        ref_data = ref_data[:,:-1]    
#    if new_data_og.shape[0]%2 == 0:
#        new_data_og = new_data_og[:-1,:] 
#        ref_data_og = ref_data_og[:-1,:]
#    if new_data_og.shape[1]%2 == 0:
#        new_data_og = new_data_og[:,:-1] 
#        ref_data_og = ref_data_og[:,:-1]
#    if mask_file:
#        if bp_mask.shape[0]%2 == 0:
#            bp_mask = bp_mask[:-1,:] # trim last pixel in y-dim
#        if bp_mask.shape[1]%2 == 0:
#            bp_mask = bp_mask[:,:-1] # trim last pixel in x-dim
#        bp_mask = bp_mask.astype(bool) 
    
    end = timer()    
    print("\n"+str(end-start)+"s were spent adjusting the array dimensions.\n")
    return
    
    # obtain the ePSFs of the images 
    if not(pn_file): # no file supplied
        print("N = "+new_file+" will be used for ePSF building")
        pn = derive_ePSF(new_file, sigma, plot=True)
    else:
        pn = fits.getdata(pn_file)
    if not(pr_file): # no file supplied
        print("R = "+ref_file+" will be used for ePSF building\n")
        pr = derive_ePSF(ref_file, sigma, plot=True)
    else:
        pr = fits.getdata(pr_file)  
    
    # set 0s to very small value to avoid singularities 
    #mask = (pn == 0)
    #pn[mask] = 1e-20
    #mask = (pr == 0)
    #pr[mask] = 1e-20

    # get the fourier transforms of the ePSFs  
    pn_hat, pr_hat = fft.fft2(pn), fft.fft2(pr)
    
    if pad: # pad the ePSFs to match the size of the data
        print("Padding the ePSFs to match the size of the image data...\n")
        xpad = (new_data.shape[1] - pn.shape[1])//2
        ypad = (new_data.shape[0] - pn.shape[0])//2
        pn = np.pad(pn, [(ypad,ypad), (xpad,xpad)], mode="constant", 
                         constant_values=0)
        pr = np.pad(pr, [(ypad,ypad), (xpad,xpad)], mode="constant",
                         constant_values=0)
        pn_hat, pr_hat = fft.fft2(pn), fft.fft2(pr)

    # get the background-only error on the images 
    # have to use **NOT** background-subtracted images for this ?
    print("Estimating the background on the original images...\n")
    ADU_max = [ADU_max_new, ADU_max_ref]
    bkg_arrays = []
    bkg_rms_arrays = [] # store error arrays
    counter = 0 # counter 

    for image_data in [new_data_og, ref_data_og]:
        # use crude image segmentation to find sources above SNR=3
        # allows us to ignore sources during background estimation
        if mask_file:
            source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                           dilate_size=15, mask=bp_mask)
            # combine the bad pixel mask and source mask 
            source_mask = np.logical_or(bp_mask,source_mask)
        else: 
            source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                       dilate_size=15)
        
        # make a simple ADU maximum mask and combine it with previous mask
        adu_mask = image_data > ADU_max[counter]
        source_mask = np.logical_or(source_mask, adu_mask)
        
        # estimate the background
        #sigma_clip = SigmaClip(sigma=3, maxiters=1) # sigma clipping
        bkg_estimator = MedianBackground(sigma_clip=None)       
        bkg_rms_estimator = BiweightScaleBackgroundRMS(sigma_clip=None)
        bkg = Background2D(image_data, (3,3), filter_size=(1,1), 
                           sigma_clip=None, bkg_estimator=bkg_estimator, 
                           bkgrms_estimator=bkg_rms_estimator, 
                           mask=source_mask)
        bkg_arrays.append(bkg.background)
        
        # doing the following produces correlated noise (I think) 
        #bkg_rms_arrays.append(bkg.background_rms)
        
        # in order for proper image subtraction to work, noise must be Gaussian
        # and uncorrelated --> estimate noise on images as the median of the 
        # RMS image + a Gaussian with width equal to the stdev of the RMS 
        bkg_rms = np.median(bkg.background_rms)
        bkg_rms += np.random.normal(scale=np.std(bkg.background_rms), 
                                    size=new_data.shape)
        bkg_rms_arrays.append(bkg_rms)
        counter += 1 
    
    # background error arrays 
    new_rms2 = bkg_rms_arrays[0]**2 # square of the background error arrays
    ref_rms2 = bkg_rms_arrays[1]**2

    # get/build masks
    print("\nImage dimensions (for debugging):")
    print("new data: "+str(new_data.shape))
    print("ref data: "+str(ref_data.shape))
    print("og new data: "+str(new_data_og.shape))
    print("og ref data: "+str(ref_data_og.shape)+"\n")
    if mask_file: # load a bad pixel mask if one is present 
        # make a simple ADU maximum mask
        adu_mask = np.logical_or((new_data > ADU_max_new),
                                 (ref_data > ADU_max_ref)) 
        # make sure no pixels are equal to 0
        zero_mask = np.logical_or(new_data == 0.0, ref_data == 0.0)
        final_mask = np.logical_or(np.logical_or(bp_mask,adu_mask), zero_mask)
    else: 
        adu_mask = np.logical_or((new_data > ADU_max_new),
                                 (ref_data > ADU_max_ref))
        zero_mask = np.logical_or(new_data == 0.0, ref_data == 0.0) 
        final_mask = np.logical_or(adu_mask, zero_mask)
    
    if interpix: # fit these masked pixels with a spline
        new_data = __inter_pix(new_data, bkg_rms_arrays[0], final_mask)
        ref_data = __inter_pix(ref_data, bkg_rms_arrays[1], final_mask)
    else: # incorporate these masks directly 
        new_data = ma.masked_array(new_data, mask=final_mask)
        ref_data = ma.masked_array(ref_data, mask=final_mask)
        if bpix_fill: 
            # fill masked pixels with the bkg_rms at their location  
            #new_noise = np.random.normal(scale=np.median(bkg_rms_arrays[0]),
            #                             size=new_data.shape)
            #ref_noise = np.random.normal(scale=np.median(bkg_rms_arrays[1]),
            #                             size=ref_data.shape)
            new_data_tofill = final_mask.astype(int)*(bkg_rms_arrays[0])
            new_data = np.ma.filled(new_data,0)+new_data_tofill
            ref_data_tofill = final_mask.astype(int)*(bkg_rms_arrays[1])
            ref_data = np.ma.filled(ref_data,0)+ref_data_tofill
            
    #return new_data, ref_data

    # remove 1% edge pixels in data and error arrays
    if border:
        print("Removing the first/last 1% of the pixels from each of the 4 "+
              "edges of the data arrays...\n")
        borderx = int(0.01*new_data.shape[1])
        bordery = int(0.01*new_data.shape[0])
        new_data[0:,0:bordery], new_data[0:borderx,0:] = 1e30, 1e30 
        new_data[0:,-bordery:], new_data[-borderx:,0:] = 1e30, 1e30   
        ref_data[0:,0:bordery], ref_data[0:borderx,0:] = 1e30, 1e30
        ref_data[0:,-bordery:], ref_data[-borderx:,0:] = 1e30, 1e30
        new_data = np.ma.masked_where(new_data==1e30, new_data)
        ref_data = np.ma.masked_where(ref_data==1e30, ref_data)
        # lots of padding
        #borderx = 4*borderx
        #bordery = 4*bordery
        #new_rms2[0:,0:bordery], new_rms2[0:borderx,0:] = 0, 0 
        #new_rms2[0:,-bordery:], new_rms2[-borderx:,0:] = 0, 0
        #ref_rms2[0:,0:bordery], ref_rms2[0:borderx,0:] = 0, 0
        #ref_rms2[0:,-bordery:], ref_rms2[-borderx:,0:] = 0, 0
        #new_rms2 = np.ma.masked_where(new_rms2==0, new_rms2)
        #ref_rms2 = np.ma.masked_where(ref_rms2==0, ref_rms2)

    ### DENOMINATOR TERMS FOR D_HAT (WITHOUT BETA) ########################
    denom1_ker = fft.ifft2(np.abs(pn_hat*np.conj(pn_hat)))
    denom1_ker = fft.fftshift(denom1_ker)
    denom1_ker = __kernel_recenter(denom1_ker) # looks Gaussian again
    denom1 = fft.fft2(convolve_fft(fft.ifft2(ref_rms2), denom1_ker, 
                                   preserve_nan=pnan, nan_treatment=nnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   normalize_kernel=False))
    denom2_ker = fft.ifft2(np.abs(pr_hat*np.conj(pr_hat)))
    denom2_ker = fft.fftshift(denom2_ker)
    denom2_ker = __kernel_recenter(denom2_ker) # looks Gaussian again
    denom2 = fft.fft2(convolve_fft(fft.ifft2(new_rms2), denom2_ker, 
                                   preserve_nan=pnan, nan_treatment=nnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   normalize_kernel=False))
    
    ### GET BETA (IF DESIRED) #############################################
    if findbeta: # solve for beta = fzero_n/fzero_r parameter
        print("Finding the beta parameter...\n")       
        beta = __solve_beta(new_data, ref_data, pn, pr, denom1, denom2, 
                            fzero_n, hdr=new_header)
        fzero_r, fzero_n = 1, beta

    ### COMPUTE D_HAT #####################################################
    dn_hat = fft.fft2(convolve_fft(new_data, pr, preserve_nan=pnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   nan_treatment=nnan, normalize_kernel=False))
    dn_hat = dn_hat/np.sqrt(denom1*fzero_n**2 + denom2)
    dr_hat = fft.fft2(convolve_fft(ref_data, pn, preserve_nan=pnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   nan_treatment=nnan, normalize_kernel=False))
    dr_hat = dr_hat/np.sqrt(denom1*fzero_n**2 + denom2)
    D_hat = dn_hat - fzero_n*dr_hat    

    ### COMPUTE S_HAT ######################################################
    ## numerator terms 
    num1_ker = fft.ifft2(np.conj(pn_hat)*np.abs(pr_hat*np.conj(pr_hat)))
    num1_ker = __kernel_recenter(num1_ker) # recenter
    num1 = fft.fft2(convolve_fft(new_data, num1_ker, preserve_nan=pnan,
                                 fftn=fft.fft2, ifftn=fft.ifft2,
                                 nan_treatment=nnan,normalize_kernel=False))
    num1 *= fzero_n*fzero_r**2
    num2_ker = fft.ifft2(np.conj(pr_hat)*np.abs(pn_hat*np.conj(pn_hat)))
    num2_ker = __kernel_recenter(num2_ker)
    num2 = fft.fft2(convolve_fft(ref_data, num2_ker, preserve_nan=pnan, 
                                 fftn=fft.fft2, ifftn=fft.ifft2,
                                 nan_treatment=nnan,normalize_kernel=False))
    num2 *= fzero_r*fzero_n**2    
    numerator = num1 - num2  
    
    ## denominator terms 
    denom1_ker = fft.ifft2(np.abs(pn_hat*np.conj(pn_hat)))
    denom1_ker = fft.fftshift(denom1_ker)
    denom1_ker = __kernel_recenter(denom1_ker) # looks Gaussian again
    denom1 = fft.fft2(convolve_fft(fft.ifft2(ref_rms2), denom1_ker,
                                   preserve_nan=pnan, nan_treatment=nnan,
                                   fftn=fft.fft2, ifftn=fft.ifft2,
                                   normalize_kernel=False))*fzero_n**2
    denom1 *= fzero_n**2
    denom2_ker = fft.ifft2(np.abs(pr_hat*np.conj(pr_hat)))
    denom2_ker = fft.fftshift(denom2_ker)
    denom2_ker = __kernel_recenter(denom2_ker) # looks Gaussian again
    denom2 = fft.fft2(convolve_fft(fft.ifft2(new_rms2), denom2_ker,
                                   preserve_nan=pnan, nan_treatment=nnan,
                                   fftn=fft.fft2, ifftn=fft.ifft2,
                                   normalize_kernel=False))*fzero_r**2
    denom2 *= fzero_r**2
    denom = denom1 + denom2

    ### S STATISTIC #######################################################
    #return fft.ifft2(new_rms2)
    S_hat = numerator/denom
    S = fft.ifft2(S_hat)
    sub = np.abs(S)
    #sub = np.real(S)
    sub /= np.std(sub)
    sub_hdu = fits.PrimaryHDU(data=sub, header=new_header)
    
    ### D STATISTIC ####################################################### 
    D = fft.ifft2(D_hat) 
    dsub = np.abs(D)
    #dsub = np.real(D)
    dsub /= np.std(dsub)
    dsub_hdu = fits.PrimaryHDU(data=dsub, header=new_header)
    
    if write:
        if not(output):
            outputS = new_file.replace(".fits", "_propersub_S.fits")
            outputD = new_file.replace(".fits", "_propersub_D.fits")
            
        sub_hdu.writeto(outputS, overwrite=True, output_verify="ignore")
        dsub_hdu.writeto(outputD, overwrite=True, output_verify="ignore")
            
    ### PLOTTING (OPTIONAL) ###
    if plot_numerator: # plot the 2 terms in the numerator of S_hat
        __plot_num_denom(new_file, "numterm", num1, 1, scale)
        __plot_num_denom(ref_file, "numterm", num2, 2, scale)
        
    if plot_denominator: # plot the 2 terms in the denominator of S_hat
        __plot_num_denom(new_file, "denomterm", denom1, 1, scale) # term1
        __plot_num_denom(ref_file, "denomterm", denom2, 2, scale) # term2
        __plot_num_denom(new_file, "denominator", 1.0/denom, 0, scale) # 1/sum
        
    if plot: # plot the difference images
        if plot == "S": # just S
            sub = sub
        elif plot == "D": # just D
            sub = dsub
        else:
            print("\nInvalid statistic selected for plotting. Plotting S.\n")
            sub = sub
        
        plt.figure(figsize=(14,13))
        w = wcs.WCS(new_header) 
        ax = plt.subplot(projection=w) # show WCS 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            mean, median, std = sigma_clipped_stats(sub)
            plt.imshow(sub, cmap='coolwarm', vmin=mean-3*std, 
                       vmax=mean+3*std, aspect=1, 
                       interpolation='nearest', origin='lower')
            plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            
        elif scale == "log": # if we want to apply a log scale 
            sub_log = np.log10(sub)
            lognorm = simple_norm(sub_log, "log", percent=99.0)
            plt.imshow(sub_log, cmap='magma', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            
        elif scale == "asinh":  # asinh scale
            sub_asinh = np.arcsinh(sub)
            asinhnorm = simple_norm(sub, "asinh")
            plt.imshow(sub_asinh, cmap="viridis", aspect=1, norm=asinhnorm,
                       interpolation="nearest", origin="lower")
            plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            
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
        plt.title("Proper image subtraction "+plot+"-statistic", fontsize=15)
        plt.savefig(new_file.replace(".fits", "_propersub_"+plot+".png"), 
                    bbox_inches="tight")
        plt.close()
        
    return dsub, sub


def transient_detect_proper(sub_file, og_file, sigma=5, pixelmin=5, 
                            elongation_lim=2.0, plots=True, 
                            sub_scale="asinh", og_scale="asinh", 
                            stampsize=200.0, crosshair="#0165fc",
                            toi=None, toi_sep_min=None, toi_sep_max=None):
    """    
    Inputs:
        - subtracted image file
        - original science image file (can be background subtracted or not)
        
        - sigma above which to look for transients in difference image 
          (optional; default 5.0)
        - minimum number of pixels to count as a source (optional; default 5)
        - maximum allowed elongation for sources found by image segmentation 
          (optional; default 2.0)
        - whether to plot the candidate transient sources with a crosshair 
          denoting their positions over all of the following:
              (1) the subtracted image 
              (2) "postage stamp" of the original unsubtracted image 
              (3) postage stamp of the subtracted image 
              (optional; default True)
        - scale to apply to the difference images (optional; default "asinh"; 
          options are "linear", "log", "asinh")
        - scale to apply to the original unsubtracted image (optional; default
          "asinh"; options are "linear", "log", "asinh")
        - size of the transient stamp in pixels (optional; default 200.0)
        - colour for crosshair on transient (optiona; default ~ hot pink)
        - [ra,dec] for some target of interest (e.g., a candidate host galaxy)
          such that the distance to this target will be computed for every 
          candidate transient (optional; default None)
        - minimum separation between the target of interest and the transient
          (optional; default None; only relevant if TOI is supplied)
        - maximum separation between the target of interest and the transient 
          (optional; default None, only relevant if TOI is supplied)
    
    Looks for sources with flux > sigma*std, where std is the standard 
    deviation of the good pixels in the subtracted image. Sources must also be 
    made up of at least pixelmin pixels. From these, selects sources below some 
    elongation limit to try to prevent obvious residuals from being detected as 
    sources. For each candidate transient source, optionally plots the full 
    subtracted image and "postage stamps" of the transient on the original 
    science image and subtracted image. 
    
    Output: a table of sources with their properties (coordinates, area, 
    elongation, separation from a target of interest (if relevant), etc.)
    """
    
    data = fits.getdata(sub_file)
    hdr = fits.getheader(sub_file)

    # use image segmentation to find sources with an area >pixelmin pix**2 
    # which are above the threshold sigma*std
    std = np.std(data)
    data = np.ma.masked_where(data==0.0, data) # mask bad pixels
    segm = detect_sources(data, sigma*std, npixels=pixelmin)  
    cat = source_properties(data, segm) 
    try:
        tbl = cat.to_table()
    except ValueError:
        print("SourceCatalog contains no sources. Exiting.")
        return
    mask = tbl["elongation"] < 2.0 # restrict based on elongation
    tbl = tbl[mask]
    
    # get coordinates
    w = wcs.WCS(hdr)
    tbl["ra"], tbl["dec"] = w.all_pix2world(tbl["xcentroid"], 
                                            tbl["ycentroid"], 1)
    
    # if provided, only look for sources within a certain angular separation
    # of the target of interest
    if toi != None: 
        toi_coord = SkyCoord(toi[0]*u.deg, toi[1]*u.deg, frame="icrs")
        trans_coord = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, frame="icrs")
        sep = toi_coord.separation(trans_coord).arcsecond
        tbl["sep"] = sep # add a column for separation
        mask = tbl["sep"] < toi_sep_max
        tbl = tbl[mask]
        mask = tbl["sep"] > toi_sep_min
        tbl = tbl[mask]
        
    
    if plots: # plot crosshairs at locations of transients 
        targets = [tbl["ra"].data, tbl["dec"].data]
        ntargets = len(targets[0])
        for i in range(ntargets):
            # set titles
            title = "difference image: candidate "+str(i)
            title_og = "original image: candidate "+str(i)
            # set output filenames
            output = sub_file.replace(".fits", "_candidate"+str(i)+".png")
            output_og = og_file.replace(".fits", 
                                          "_zoomed_candidate"+str(i)+".png")
            output_zoom = sub_file.replace(".fits", 
                                          "_zoomed_candidate"+str(i)+".png")
            
            # crosshair on full-frame difference image 
            make_image(sub_file, 
                       scale=sub_scale, cmap="coolwarm", label="", title=title, 
                       output=output, 
                       target=toi,
                       target_small=[targets[0][i], targets[1][i]])
            
            # crossair on small region of science image and difference image 
            transient_stamp(og_file, 
                            [targets[0][i], targets[1][i]], 
                            stampsize, 
                            scale=og_scale, cmap="magma", 
                            crosshair=crosshair,
                            title=title_og, output=output_og,
                            toi=toi)
            transient_stamp(sub_file, 
                            [targets[0][i], targets[1][i]], 
                            stampsize, 
                            scale=sub_scale, cmap="magma", label="",
                            crosshair=crosshair,
                            title=title, output=output_zoom,
                            toi=toi)
    return tbl

###############################################################################
#### MISCELLANEOUS PLOTTING ####
    
def make_image(im_file, mask_file=None, scale=None, cmap="magma", label=None,
               title=None, output=None, target=None, target_small=None):
    """
    Input: any image of interest, a bad pixels mask (optional; default None), 
    the scale to use for the image (optional; default None=linear, options are 
    "log" and "asinh"), the colourmap to use for the image (optional; default
    is "magma"), the title to apply to the colourbar (optional; defaults set
    below), a title for the image (optional; default None), name for the output 
    file (optional; default None), [ra,dec] for a target at which to place a 
    black crosshair (optional; default None), [ra,dec] for a second target at 
    which to place a smaller pink crosshair (optional; default None)
    Output: None
    """
    
    image_data = fits.getdata(im_file)
    image_header = fits.getheader(im_file)
    
    plt.figure(figsize=(14,13))
    
    # show WCS     
    w = wcs.WCS(image_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    if mask_file:
        mask = fits.getdata(mask_file)
        image_data_masked = np.ma.masked_where(mask, image_data)
        image_data = np.ma.filled(image_data_masked, 0)
    
    if not scale: # if no scale to apply 
        scale = "linear"
        plt.imshow(image_data, cmap=cmap, aspect=1, 
                   interpolation='nearest', origin='lower')
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
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(title):
        title = im_file.replace(".fits"," data")
    if not(output):
        output = im_file.replace(".fits","_"+scale+".png")

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

    plt.title(title, fontsize=15)
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def transient_stamp(im_file, target, size=200.0, scale=None, cmap="magma", 
                    label=None, crosshair="#fe019a", title=None, 
                    output=None, toi=None):
    """
    Inputs: any image of interest, the [ra, dec] of a candidate transient 
    source, the size of the zoomed-in region around the transient to plot in 
    pixels (optional; default 200.0), the colourmap to use for the image 
    (optional; default is "magma"), the label to apply to the colourbar 
    (optional; defaults set below), the colour for the crosshairs (optional;
    default is ~ hot pink), a title for the image (optional; defaults set 
    below), a name for the output file (optional; defaults set below)
    Output: None
    """
    
    ra, dec = target
    imhdu = crop_WCS(im_file, ra, dec, size, write=False)
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
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    if not scale: # if no scale to apply 
        scale = "linear"
        plt.imshow(image_data, cmap=cmap, aspect=1, 
                   interpolation='nearest', origin='lower')
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
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(title):
        title = im_file.replace(".fits"," data")
    if not(output):
        output = im_file.replace(".fits","_"+scale+".png")

    plt.gca().plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
           transform=plt.gca().get_transform('icrs'), linewidth=2, 
           color=crosshair, marker="")
    plt.gca().plot([ra, ra], [dec+10.0/3600.0, dec+5.0/3600.0], 
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

    
    

        
        
    
    
