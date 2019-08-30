#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:58:03 2019
@author: Nicholas Vieira
@amakihi.py 
"""

import os
from subprocess import run
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import re
import requests

from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats

##############################################################################
#### DOWNLOADING TEMPLATES ####

def __downloadtemplate(url, output=None):
    """
    Input: the url(s) of interest 
    Downloads the fits image at the given url. 
    Output: HDU object(s) for the downloaded template image(s) 
    """

    if type(url) == str: # if just one
        if not(output):
            output = re.sub(".*/", "", url)
            output = re.sub(".*=", "", output)
        tmpl = fits.open(url) # download it 
        tmpl.writeto(output, overwrite=True, output_verify="ignore")
        return tmpl
    else: # if many
        templates = []
        for i in range(len(url)):
            u = url[i]
            if not output:
                out = re.sub(".*/", "", u) 
                out = re.sub(".*=", "", out)
            else:
                out = output[i]
            tmpl = fits.open(u) # download it 
            templates.append(tmpl)
            tmpl.writeto(out, overwrite=True, output_verify="ignore")
            tmpl.close()
        return templates
    

def __downloadtemplate_auth(url, auth_user, auth_pass):
    """
    Input: the url(s) of interest and the CADC username and password needed to 
    access proprietary CFIS data 
    
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
        for u in url:
            output = re.sub(".*/", "", u)
            output = output[:output.find("[")] # clean up filename
            r = requests.get(u, auth=(auth_user, auth_pass))
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
    tmps = __downloadtemplate(url, output) # download template 
    print("\nDownloaded square PanStarrs1 cutout image(s) in the "+filt_upd+
          " band(s), centered on RA, Dec = %.3f, %.3f "%(ra, dec)+" with"+
          " sides of length %.2f'"%size_arcmin+"\n")
    return tmps


def download_DECaLS_template(ra, dec, size=400, pixscale=0.262, filt="grz", 
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
    tmps = __downloadtemplate(url, output) # download template 
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
    tmps = __downloadtemplate_auth(url, auth_user, auth_pass)
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
    tmps = __downloadtemplate(url) # download template 
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
    try:
        pix_scale = hdr["PIXSCAL1"] # scale of image in arcsec per pix
    except KeyError:
        pix_scale = float(input("\nPixel scale not found. Please input a "+
                                "scale in arcseconds per pixel: \n>>> "))
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
    
def bkgsub(im_file, mask_file=None, crreject=False, plot_bkg=False, 
           scale_bkg=None, plot=False, scale=None, write=True, output=None):
    """
    Input: any image of interest, a bad pixels mask (optional; default None),
    bool indicating whether to do reject cosmic rays (optional; default False)
    bool indicating whether to plot the background (optional; default False),
    a scale to apply to the background (optional; default None (linear); 
    options are ("log" and asinh"), the same bools for the backround-subtracted
    image, a bool indicating whether to write the background-subtracted image 
    (optional; default True) and a name for the output background-subtracted 
    image in a FITS file (optional; default set below)
    
    Performs background subtraction and cosmic ray rejection on the input 
    image. 
    
    Output: the background-subtracted image data in a fits HDU 
    """
    
    from astropy.stats import SigmaClip
    from photutils import Background2D, MedianBackground, make_source_mask
    import astroscrappy
    
    image_data = fits.getdata(im_file)
    
    ### SOURCE DETECTION ###
    # use image segmentation to find sources above SNR=3 and mask them 
    # for background estimation
    if mask_file: # load a bad pixel mask if one is present 
        bp_mask = fits.getdata(mask_file)
        bp_mask = bp_mask.astype(bool)
        source_mask = make_source_mask(image_data, snr=2, npixels=3, 
                                   dilate_size=15, mask=bp_mask)
        # combine the bad pixel mask and source mask 
        final_mask = np.logical_or(bp_mask,source_mask)
    else: 
        source_mask = make_source_mask(image_data, snr=2, npixels=3, 
                                   dilate_size=15)
        final_mask = source_mask
    
    ### BACKGROUND SUBTRACTION ###
    # estimate the background
    sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    bkg_estimator = MedianBackground()
    
    try: # try using photutils
        bkg = Background2D(image_data, (5,5), filter_size=(5,5), 
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
        gain = fits.getheader(im_file)["GAIN"]
        rdnoise = fits.getheader(im_file)["RDNOISE"]
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
    INPUT NEEDS TO BE UPDATED
    Input: the science image (the source), the template to match to, a sigma 
    threshold for source detection (optional; default 3.0), a bool indicating 
    whether to plot the matched image data (optional; default False), a scale 
    to apply to the plot (optional; default None (linear), options are "log" 
    and "asinh"), whether to write the output .fits to files (optional; default 
    True) and names for the output aligned image and image mask (both optional; 
    defaults set below )
    
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
    WIP. NOT RELIABLE.
    
    
    Input: the science image (the source), the template to match to, a mask of
    bad pixels to ignore (optional; default None), a bool indicating whether to 
    flip (invert along x AND y) the image before trying to align (optional; 
    default False), a bool indicating whether to plot the matched image data 
    (optional; default False), a scale to apply to the plot (optional; default 
    None (linear), options are "log" and "asinh"), whether to write the output 
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
    
    from image_registration import chi2_shift
    from scipy import ndimage
    
    # pad the source array so that it has the same shape as the template
    # template must be larger than source
    if source.shape != template.shape:
        ypad = (template.shape[1] - source.shape[1])
        xpad = (template.shape[0] - source.shape[0])
        print("xpad == "+str(xpad))
        print("ypad == "+str(ypad))
        
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
        

    xoff, yoff, exoff, eyoff = chi2_shift(template, source, err=None, 
                                          return_error=True, 
                                          upsample_factor="auto",
                                          boundary="constant")
    img_aligned = ndimage.shift(source, [-yoff, -xoff], order=3, 
                                mode='constant', cval=0.0, prefilter=True)
    
    if mask_file:
        mask = np.logical_or((img_aligned == 0), mask)
    else: 
        mask = (img_aligned == 0)
    
    print("x offset = "+str(xoff)+"+/-"+str(exoff))
    print("y offset = "+str(yoff)+"+/-"+str(eyoff))

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
    
def box_mask(source_file, pixx=None, pixy=None, mask_file=None, plot=False, 
             write=True, output=None):
    """
    Manual box-shaped mask addition. Write description later.
    """
    source = fits.getdata(source_file)
    
    if not pixx:
        pixx = [0, int(source.shape[1])]
    if not pixy:
        pixy = [0, int(source.shape[0])]
    
    newmask = np.zeros(source.shape)
    newmask[pixy[0]:pixy[1], pixx[0]:pixx[1]] = 1.0
    newmask = newmask.astype(bool)
    
    if mask_file: # combine with another mask 
        mask = fits.getdata(mask_file)
        newmask = np.logical_or(mask, newmask)
    
    hdr = fits.getheader(source_file)
    mask_hdu = fits.PrimaryHDU(data=newmask.astype(int), header=hdr)
    
    if plot:
        plt.imshow(newmask)
    
    if write:
        if not(output):
            output = source_file.replace(".fits", "_boxmask.fits")
            
        mask_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return mask_hdu

###############################################################################
#### IMAGE DIFFERENCING ####

def hotpants(source_file, template_file, mask_file=None, iu=65535, tu=65535, 
             il=-500, tl=-500, gd=None, convi=False, convt=False, bgo=0, ko=1, 
             output=None, convout=None, v=1, plot=True, scale=None, 
             target=None, target_small=None):
    """
    INPUTS NEED TO BE UPDATED
    
    hotpants args: 
        - the science image (the source)
        - the template to match to
        - a mask of which pixels to ignore during subtraction (optional; 
          default None)
        - the upper(u) and lower(l) ADU limits for the image (i) and 
          template (t) (optional; defaults 65535, 65535, -500, -500)
        - good pixels coordinates (optional; default is full image)
            xmin xmax ymin ymax 
            e.g. gd="150 1000 200 800" 
        - force convolve the image (optional; default False)
        - force convolve the template (optional; default False)
        - spatial background variation order (optional; default 0)
        - spatial kernel variation order (optional; default 1)
        - name for the output subtracted image (optional; default set below)
        - name for the output convolved image (optional; default set below)
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
    
    Output: the subtracted image
    """

    source_header = fits.getheader(source_file)
    
    if not(output):
        output = source_file.replace(".fits", "_subtracted.fits")
    if not(convout):
        convout = source_file.replace(".fits", "_conv.fits")
        
    # setting inputs and options for hotpants 
    hp_options = " -inim "+source_file+" -tmplim "+template_file
    hp_options += " -outim "+output # output subtracted file
    hp_options += " -oci "+convout # output convoluted file
    if mask_file: # if a mask is supplied
            mask = fits.getdata(mask_file)
            hp_options += " -tmi "+mask_file # mask for template 
            hp_options += " -imi "+mask_file # same mask for source       
    hp_options += " -iu "+str(iu)+" -tu "+str(tu) # upper ADU limits
    hp_options += " -il "+str(il)+" -tl "+str(tl) # lower ADU limits
    if gd:
        hp_options += " -gd "+gd
    if convi:
        hp_options += " -c i" # force convolution of the image 
    if convt: 
        hp_options += " -c t" # force convolution of the template
    hp_options += " -bgo "+str(bgo)+" -ko "+str(ko) # bg order, kernel order     

    #hp_options += " -ng 3 6 3.0 4 5.0 2 8.0"
    hp_options += " -v "+str(v)
    
    run("~/hotpants/hotpants"+hp_options, shell=True) # call hotpants
    
    # if file successfully produced and non-empty
    cwd = os.getcwd()
    if (output in os.listdir(cwd)) and (os.stat(output).st_size!=0):
        sub = fits.getdata(output) # load it in 
    else:
        return
    
    # mask bad pixels
    sub_bp_mask = (sub == 1.0e-30)
    if mask_file:
        mask = np.logical_or(mask, sub_bp_mask)
        sub = np.ma.masked_where(mask, sub)
        sub = np.ma.filled(sub, 0) # set bad pixels to 0
    else:
        sub = np.ma.masked_where(sub_bp_mask, sub)
        sub = np.ma.filled(sub, 0) # set bad pixels to 0
    
    sub_header = fits.getheader(output)
    mean_diff = float(sub_header["DMEAN00"]) # mean of diff image good pixels
    std_diff = float(sub_header["DSIGE00"]) # std dev of diff image good pixels
    print("\nMEAN OF DIFFERENCE IMAGE: %.2f ± %.2f \n"%(mean_diff, std_diff))
        
    # plot, if desired
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS      
        w = wcs.WCS(source_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            mean, median, std = sigma_clipped_stats(sub)
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
            #xpix, ypix = w.all_world2pix(ra, dec)
            plt.gca().plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
                   transform=plt.gca().get_transform('icrs'), linewidth=3, 
                   color="black", marker="")
            plt.gca().plot([ra, ra], [dec+10.0/3600.0, dec+5.0/3600.0], 
                   transform=plt.gca().get_transform('icrs'),  linewidth=3, 
                   color="black", marker="")
        if target_small:
            ra, dec = target_small
            #xpix, ypix = w.all_world2pix(ra, dec)
            plt.gca().plot([ra-5.0/3600.0, ra-2.5/3600.0], [dec,dec], 
                   transform=plt.gca().get_transform('icrs'), linewidth=3, 
                   color="#fe019a", marker="")
            plt.gca().plot([ra, ra], [dec-5.0/3600.0, dec-2.5/3600.0], 
                   transform=plt.gca().get_transform('icrs'),  linewidth=3, 
                   color="#fe019a", marker="")
        
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("hotpants image difference", fontsize=15)
        plt.savefig("hotpants_sub_"+scale+".png", bbox_inches="tight")
        plt.show()
        
    return sub

###############################################################################
#### MISCELLANEOUS PLOTTING ####
    
def make_image(im_file, mask_file=None, scale=None, title=None, output=None):
    """
    Input: any image of interest, a bad pixels mask (optional; default None), 
    the scale to use for the image (optional; default None=linear, options are 
    "log" and "asinh"), title for the image (optional) and name for the output 
    file (optional)
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
        plt.imshow(image_data, cmap='magma', aspect=1, 
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        
    elif scale == "log": # if we want to apply a log scale 
        image_data_log = np.log10(image_data)
        lognorm = simple_norm(image_data_log, "log", percent=99.0)
        plt.imshow(image_data_log, cmap='magma', aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        
    elif scale == "asinh": # asinh scale 
        image_data_asinh = np.arcsinh(image_data)
        asinhnorm = simple_norm(image_data_asinh, "asinh")
        plt.imshow(image_data_asinh, cmap="viridis", aspect=1, norm=asinhnorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
        
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(title):
        title = im_file.replace(".fits"," data")
    if not(output):
        output = im_file.replace(".fits","_"+scale+".png")

    plt.title(title, fontsize=15)
    plt.savefig(output, bbox_inches="tight")

        
        
    
    
