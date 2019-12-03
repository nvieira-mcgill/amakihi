#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:58:03 2019
@author: Nicholas Vieira
@amakihi.py 

This script contains a library of functions for downloading templates and all 
other steps required for image differencing/subtraction: cropping, background-
subtraction, registration (alignment) and masking out of bad pixels, including 
saturated stars. Can then perform image subtraction using either the tool 
HOTPANTS (Becker, 2015) or the form of Proper Image Subtraction (ZOGY 
formalism) presented in Zackay et al, 2016. ** 

** Proper Image Subtraction is currently a WIP. Has been cut and pasted into 
a separate script for the moment.

HOTPANTS: https://github.com/acbecker/hotpants
Zackay et al: https://arxiv.org/abs/1601.02655

Templates can be downloaded from each survey in the corresponding filters:
    - PanStarrs 1 (PS1):                            g, r, i, z, y
    - Dark Energy Camera Legacy Survey (DECaLS):    g, r, z
    - Canada-France Imaging Survey (CFIS):          u, r
    - 2MASS:                                        J, H, K
    
IMPORTANT NOTE: this software makes use of a slightly modified version of the 
astroalign software developed by Martin Beroiz and the TOROS Dev Team. The 
original software can be seen here:
    https://github.com/toros-astro/astroalign
I claim absolutely no ownership of this software. The additions I have made to 
the software are mostly superficial and have to do with error handling. 

SECTIONS:
    - Downloading templates
    - Cropping images
    - Background subtraction
    - Image registraton (alignment)
    - Mask building (boxmask, saturation mask)
    - ePSF building 
    - Image differencing with HOTPANTS
    - Transient detection 
    - **Image differencing with Proper Image Subtraction (ZOGY formalism)
    - Miscellaneous plotting
    - **Functions I directly took from elsewhere (with references)
    
    ** Currently not present or usable
    
GENERAL DEPENDENCIES NOT INCLUDED ON THIS SOFTWARE'S GITHUB:
    - astropy (used extensively)
    - photutils (used extensively)
    - astrometry.net (used extensively, but can be ignored in favour of 
      source detection with photutils' image_segmentation instead)
    - image_registration (used in image_align_morph())

HOTPANTS-SPECIFIC:
    - hotpants (essential for image subtraction via hotpants)
    
PROPER IMAGE SUBTRACTION-SPECIFIC:
    - pyfftw (used extensively to speed up FFTs)
    - skimage (bad pixel filling during proper image subtraction)
    
LESS ESSENTIAL DEPENDENCIES:
    - astroscrappy (OPTIONAL cosmic ray rejection during background removal)
    - astroplan (airmass determination during proper subtraction)
"""


import os
import sys
from subprocess import run
import numpy as np
#import random
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
plt.switch_backend('Qt5Agg')
import re
import requests
#from timeit import default_timer as timer

from scipy.ndimage import zoom, binary_dilation, gaussian_filter

from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
import astropy.units as u 
#from astropy.convolution import convolve_fft
from astropy.stats import (sigma_clipped_stats, SigmaClip, 
                           gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm)
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from photutils import Background2D, MedianBackground
from photutils import make_source_mask, detect_sources, source_properties

## for speedy FFTs
#import pyfftw
#import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs
#pyfftw.interfaces.cache.enable()

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

# if hotpants cannot be called directly from the command line, the path to the
# executable can be put modified with the function hotpants_path() below
# by default, set to None --> assume can be called directly from the cmd line
HOTPANTS_PATH = ""

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
    0.258" in PS1), the filter(s) (g, r, i, z, y) desired, and output name(s)
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
    size = int(size) # ensure size is an int in the url
    
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
    Input: a RA, Dec of interest, a size for the image in pixels*, the scale of 
    the image in arcseconds per pixel, the filters to use (files are always 
    downloaded separately when multiple filters are provided; options are 
    (g, r, z)) and output name(s) for the downloaded template(s)
    
    Downloads the relevant DECaLS template image(s) at the input RA, Dec. 
    
    * size must be an int. If it is a float, it will be converted.
    
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
    
    if len(filt_upd) == 0:
        return
    
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
    
    # ensure size is an int
    size = int(size)
       
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
    idx_x = [int(round(frac_hori[0]*xdim)), int(round(frac_hori[1]*xdim))]
    idx_y = [int(round(frac_vert[0]*ydim)), int(round(frac_vert[1]*ydim))]

    # get the cropped data, build a new PrimaryHDU object
    cropped = data[idx_y[0]:idx_y[1], idx_x[0]:idx_x[1]]
    hdr["NAXIS1"] = len(idx_x) # adjust NAXIS sizes
    hdr["NAXIS2"] = len(idx_y)
    hdr["CRPIX1"] -= idx_x[0] # update WCS reference pixel 
    hdr["CRPIX2"] -= idx_y[0]
    new_hdu = fits.PrimaryHDU(data=cropped, header=hdr)
    
    return new_hdu


def crop_WCS(source_file, ra, dec, size, mode="truncate", write=True, 
             output=None):
    """
    Input: 
        - fits filename
        - right ascension, declination (in decimal degrees)
        - size of a box (in pixels) to crop
        - mode to apply if the requested crop lies outside of image bounds:
          "truncate" to truncate the crop and return a crop smaller than the 
          requested one; "extend" to extend the bounds of the image in the 
          direction opposite the exceeded boundary, effectively recentering the 
          crop (optional; default "truncate")
        - whether to write the output to a .fits file (optional; default True) 
        - name for the output fits file (optional; default set below)
    
    For a single fits file, crops the image to a box of size pixels centered 
    on the given RA and Dec. If the given box extends beyond the bounds of the 
    image, the box will either be truncated at these bounds or re-centered to 
    allow for a box of the requested size, depending on the input mode.
    
    Output: a PrimaryHDU for the cropped image (contains data and header)
    """
    
    hdr = fits.getheader(source_file)
    img = fits.getdata(source_file)
    y_size, x_size = img.shape # total image dims in pix 
    w = wcs.WCS(hdr)

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
    
    if mode == "truncate": # truncate bounds if needed
        x_bounds[x_bounds<0] = 0 
        x_bounds[x_bounds>x_size] = x_size
        y_bounds[y_bounds<0] = 0 
        y_bounds[y_bounds>y_size] = y_size
    
    elif mode == "extend": # re-center crop to obtain requested size
        # fix the size
#        if (x_bounds[1] - x_bounds[0]) < size:
#            x_bounds[0] -= (size-(x_bounds[1]-x_bounds[0]))
#        if (y_bounds[1] - y_bounds[0]) < size:
#            y_bounds[0] -= (size-(y_bounds[1]-y_bounds[0]))  
#        if (x_bounds[1] - x_bounds[0]) > size:
#            x_bounds[0] += (size-(x_bounds[1]-x_bounds[0]))
#        if (y_bounds[1] - y_bounds[0]) > size:
#            y_bounds[0] += (size-(y_bounds[1]-y_bounds[0]))
        x_bounds[0] = round(x_bounds[0])
        x_bounds[1] = x_bounds[0] + size
        y_bounds[0] = round(y_bounds[0])
        y_bounds[1] = y_bounds[0] + size        
        
        # check boundaries
        if x_bounds[0] < 0: # left edge beyond boundary
            x_bounds[0] = 0
            x_bounds[1] = size
        if x_bounds[1] > x_size: # right edge beyond boundary
            x_bounds[0] = x_size - size
            x_bounds[1] = x_size
        if y_bounds[0] < 0: # bottom edge beyond boundary
            y_bounds[0] = 0
            y_bounds[1] = size
        if y_bounds[1] > y_size: # top edge beyond boundary
            y_bounds[0] = y_size - size
            y_bounds[1] = y_size
            
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
    elif not((x_bounds[1]-x_bounds[0] > size/2.0) and 
             (y_bounds[1]-y_bounds[0] > size/2.0) ):
            print("\nWARNING: the cropped image is less than 50% the height "+
                  "or width of the desired crop.")
    elif not(0.25 < ((frac_hori[1]-frac_hori[0])/
                     (frac_vert[1]-frac_vert[0])) < 4.0):
            print("\nWARNING: the aspect ratio of the image is more skew than"+
                  " 1:4 or 4:1.")
    
    cropped_hdu = __get_crop(source_file, frac_hori, frac_vert)
    
    if write: # if we want to write the cropped .fits file 
        if not(output): # if no output name given, set default
            output = source_file.replace(".fits", "_crop.fits")
        cropped_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    return cropped_hdu


def crop_frac(source_file, frac_hori=[0,1], frac_vert=[0,1], write=True, 
             output=None):
    """
    Input: the horizontal fraction of the fits file's 
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


def crop_octant(source_file, ra, dec, write=True, output=None):
    """
    Input: the name of the fits file, a right ascension, declination (in 
    decimal degrees), whether to write the output fits file (optional; 
    default True) and the name for the output fits file (optional; default set
    below)
    
    For a given, RA, Dec, finds the octant-sized ~square of the image in which 
    these coords are found. Then crops the image to this octant. Best for 
    images with an aspect ratio of around 2:1 (i.e. MegaCam images) such that 
    the output crop is square-like.
    
    Output: a PrimaryHDU for the cropped image (contains data and header)
    """
    
    # get data 
    data = fits.getdata(source_file)
    hdr = fits.getheader(source_file)
    
    # locate the octant of the image in which the source is found 
    ydim, xdim = data.shape # total image dims in pix 
    w = wcs.WCS(hdr)
    x, y = w.all_world2pix(ra, dec, 1)
    x, y = x/xdim, y/ydim   
    if x < 0.5: # check the x coord
        frac_hori = [0.0, 0.5]
        #if abs(0.5-y) < abs(0.25-y):
        #    frac_hori = [0.25, 0.75]
        #else:
        #    frac_hori = frac_h
    else:
        frac_hori = [0.5, 1.0]
        #if abs(0.5-y) < abs(0.75-y):
        #    frac_hori = [0.25, 0.75]
        #else:
        #    frac_hori = frac_h
        
    if y < 0.25: # check the y coord 
        frac_v = [0, 0.25]
        if abs(0.25-y) < abs(0.125-y):
            frac_vert = [f+0.125 for f in frac_v]
        else:
            frac_vert = frac_v
    elif y < 0.5:
        frac_v = [0.25, 0.5]
        if abs(0.25-y) < abs(0.375-y):
            frac_vert = [f-0.125 for f in frac_v]
        elif abs(0.5-y) < abs(0.375-y):
            frac_vert = [f+0.125 for f in frac_v]
        else:
            frac_vert = frac_v
    elif y < 0.75:
        frac_v = [0.5, 0.75]
        if abs(0.5-y) < abs(0.625-y):
            frac_vert = [f-0.125 for f in frac_v]
        elif abs(0.75-y) < abs(0.625-y):
            frac_vert = [f+0.125 for f in frac_v]
        else:
            frac_vert = frac_v
    else:
        frac_v = [0.75, 1.0]
        if abs(0.75-y) < abs(0.875-y):
            frac_vert = [f-0.125 for f in frac_v]
        else:
            frac_vert = frac_v
    
    cropped_hdu = __get_crop(source_file, frac_hori, frac_vert)
    
    if write: # if we want to write the cropped .fits file 
        if not(output): # if no output name given, set default
            output = source_file.replace(".fits", "_crop.fits")
        cropped_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    return cropped_hdu   


def crop_half(source_file, ra, dec, write=True, output=None):
    """
    Input: the name of the fits file, a right ascension, declination (in 
    decimal degrees), whether to write the output fits file (optional; 
    default True) and the name for the output fits file (optional; default set
    below)
    
    For a given, RA, Dec, finds the half of the image in which these coords
    are found. This can be the top half, bottom half, or a box centered on the 
    center of the overall image. Then crops the image. Best for images with an 
    aspect ratio of around 2:1 (i.e. MegaCam images) such that the output crop 
    is square-like.
    
    Output: a PrimaryHDU for the cropped image (contains data and header)
    """
    
    # get data 
    data = fits.getdata(source_file)
    hdr = fits.getheader(source_file)
    
    # locate the "half" of the image in which the source is found 
    ydim, xdim = data.shape # total image dims in pix 
    w = wcs.WCS(hdr)
    x, y = w.all_world2pix(ra, dec, 1)
    x, y = x/xdim, y/ydim   
    if y < 0.25: # check the y coord 
        frac_vert = [0, 0.5]
    elif y < 0.75:
        frac_vert = [0.25, 0.75]
    else:
        frac_vert = [0.5, 1]
    frac_hori = [0, 1]        
    
    cropped_hdu = __get_crop(source_file, frac_hori, frac_vert)
    
    if write: # if we want to write the cropped .fits file 
        if not(output): # if no output name given, set default
            output = source_file.replace(".fits", "_crop.fits")
        cropped_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    return cropped_hdu   

###############################################################################

def __remove_SIP(header):
    """
    Input: a FITS header
    Output: None
    
    Removes all SIP-related keywords from a FITS header, in-place.
    """
    
    SIP_KW = re.compile('''^[AB]P?_1?[0-9]_1?[0-9][A-Z]?$''')
    
    for key in (m.group() for m in map(SIP_KW.match, list(header))
            if m is not None):
        del header[key]
    try:
        del header["A_ORDER"]
        del header["B_ORDER"]
        del header["AP_ORDER"]
        del header["BP_ORDER"]
    except KeyError:
        pass
    
    header["CTYPE1"] = header["CTYPE1"][:-4]
    header["CTYPE2"] = header["CTYPE2"][:-4]

###############################################################################
#### BACKGROUND SUBTRACTION ####
    
def bkgsub(im_file, mask_file=None, bkg_box=(5,5), bkg_filt=(5,5),
           crreject=False, plot_bkg=False, scale_bkg=None, plot=False, 
           scale=None, write=True, output=None):
    """    
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
          (linear);  options are "linear", "log", "asinh")
        - whether to plot the background-SUBTRACTED image (optional; default 
          False)
        - scale to apply to the background-SUBTRACTED image (optional; default 
          None (linear);  options are "linear", "log", "asinh")
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
    
    ## build masks
    if mask_file: # load a bad pixel mask if one is present 
        bp_mask = fits.getdata(mask_file)
        bp_mask = bp_mask.astype(bool)
        zeromask = image_data == 0 # mask out pixels equal to 0
        nansmask = np.isnan(image_data) # mask out nans
        bp_mask = np.logical_or(bp_mask, zeromask)
        bp_mask = np.logical_or(bp_mask, nansmask)       
    else: 
        zeromask = image_data == 0 # mask out pixels equal to 0
        nansmask = np.isnan(image_data) # mask out nans
        bp_mask = np.logical_or(nansmask, zeromask)

    ## make a crude source mask
    source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                   dilate_size=15, mask=bp_mask)
    # combine the bad pixel mask and source mask for background subtraction
    final_mask = np.logical_or(bp_mask, source_mask)
    
    ### BACKGROUND SUBTRACTION ###
    ## estimate the background
    try:
        sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    except TypeError: # in older versions of astropy, "maxiters" was "iters"
        sigma_clip = SigmaClip(sigma=3, iters=5)
    bkg_estimator = MedianBackground()
    
    bkg = Background2D(image_data, box_size=bkg_box, filter_size=bkg_filt, 
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
                       mask=final_mask)
    bkg_img = bkg.background
    bkgstd = bkg.background_rms_median   
    
    ## subtract the background 
    bkgsub_img = image_data - bkg_img     
    bkgstd = bkg.background_rms_median # save this quantity to write to header
    
    ### COSMIC RAY REJECTION (OPTIONAL) ###
    # very large rejection sigma --> only reject most obvious cosmic rays 
    # objlim and sigclip determine this sigma 
    if crreject: # perform cosmic ray rejection
        import astroscrappy
        try:
            gain = fits.getheader(im_file)["GAIN"]
        except KeyError:
            try:
                gain = fits.getheader(im_file)["HIERARCH CELL.GAIN"]
            except KeyError:
                gain = 1.0
        try:
            rdnoise = fits.getheader(im_file)["RDNOISE"]
        except KeyError:
            try:
                rdnoise = fits.getheader(im_file)["HIERARCH CELL.READNOISE"]
            except KeyError:
                rdnoise = 1.0
        
        crmask, crclean = astroscrappy.detect_cosmics(bkgsub_img, bp_mask, 
                                        gain=gain, readnoise=rdnoise,
                                        objlim=12, sigclip=12, sigfrac=0.9)
        bp_mask = np.logical_or(bp_mask, crmask)
        
    ## finally, mask bad pixels (and cosmic rays if crreject=True)
    ## all bad pix are then set to 0 for consistency
    bkg_img_masked = np.ma.masked_where(bp_mask, bkg_img)
    bkg_img = np.ma.filled(bkg_img_masked, 0)   
    bkgsub_img_masked = np.ma.masked_where(bp_mask, bkgsub_img)
    bkgsub_img = np.ma.filled(bkgsub_img_masked, 0)
    
    ### PLOTTING (OPTIONAL) ###
    if plot_bkg: # plot the background, if desired
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
            cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)

        cb.ax.tick_params(which='major', labelsize=15)            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("image background", fontsize=15)
        plt.savefig("background_"+scale_bkg+".png", bbox_inches="tight")
            
    if plot: # plot the background-subtracted image, if desired
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
            cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)

        cb.ax.tick_params(which='major', labelsize=15)            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("background-subtracted image", fontsize=15)
        plt.savefig("background_sub_"+scale+".png", bbox_inches="tight")
    
    hdr = fits.getheader(im_file)
    hdr["BKGSTD"] = bkgstd # useful header for later
    bkgsub_hdu = fits.PrimaryHDU(data=bkgsub_img, header=hdr)
    
    if write: # if we want to write the background-subtracted fits file
        if not(output): # if no output name given, set default
            output = im_file.replace(".fits", "_bkgsub.fits")
        bkgsub_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return bkgsub_hdu
    
###############################################################################
#### IMAGE ALIGNMENT (REGISTRATION) ####

def image_align(source_file, template_file, mask_file=None, 
                astrometry=True, bkgsub=False, astrom_sigma=8.0, psf_sigma=5.0, 
                nsources=50, keep=False, 
                thresh_sigma=3.0, 
                plot=False, scale=None, write=True, output_im=None, 
                output_mask=None):
    """    
    WIP:
        - WCS header of aligned image doesn't seem correct
    
    Input: 
        general:
        - science image (source) to register
        - template image (target) to match source to
        - mask file for the SOURCE image (optional; default None)
        
        source detection:
        - use astrometry.net (specifically, image2xy) for source detection and 
          provide astroalign with a list of x, y coordinates if True, use 
          simple image segmentation to detect sources if False (optional; 
          default True)
        
        if astrometry=True:
        - whether the input source and template files have been background-
          subtracted (optional; default False; can be set to a list to specify
          that the science image is background-subtracted and the template is
          not or vice-versa)
        - sigma threshold for image2xy source detection in the source/template 
          (optional; default 8.0 which sets the threshold to 8.0 for both; can
          pass a list to specify a different threshold for source and template)
        - the sigma of the Gaussian PSF of the source/template (optional; 
          default 5.0 which sets the width to 5.0 for both; can pass a list to
          specify a different width for source and template)
        - no. of sources to keep for asterism-matching (optional; default 50)
        - whether to keep the source list files after running image2xy 
          (optional; default False)
          
        if astrometry=False:
        - sigma threshold for source detection with image segmentation 
          (optional; default 3.0)
          
        writing and plotting:
        - whether to plot the matched image data (optional; default False)
        - scale to apply to the plot (optional; default None (linear); options 
          are "linear", "log", "asinh")
        - whether to write the output .fits to files (optional; default True) 
        - names for the output aligned image and image mask (both optional; 
          defaults set below)
    
    Optionally uses astrometry.net to get x, y coords for all sources. Then 
    calls on astroalign to align the source image with the target to allow for 
    proper image subtraction. Also finds a mask of out of bounds pixels to 
    ignore during subtraction.
    
    *** Uses a slightly modified version of astroalign.
    
    NOTE: image2xy has a hard time doing basic background subtraction when an 
    image has nans in it. Before trying alignment with an image which contains
    nans, it is suggested to set all nans to 0 instead.
    
    Output: the aligned image data and a pixel mask
    """
    
    # a modified version of astroalign
    # allows the user to set the sigma threshold for source detection, which 
    # sometimes needs to be tweaked
    import astroalign_mod as aa
    
    ## load in data
    source = fits.getdata(source_file)
    template = fits.getdata(template_file)
    template_header = fits.getheader(template_file)

    ## build up masks, apply them
    mask = np.logical_or(source==0, np.isnan(source)) # mask zeros/nans in src 
    if mask_file: # if a mask is provided
        mask = np.logical_or(mask, fits.getdata(mask_file))
    source = np.ma.masked_where(mask, source)  
    
    tmpmask = np.logical_or(template==0, np.isnan(template)) # in template
    template = np.ma.masked_where(tmpmask, template)
    
    ### OPTION 1: use astrometry.net to find the sources, find the transform, 
    ### and then apply the transform (DEFAULT)
    if astrometry:  
        # check if input bkgsub bool is list or single value
        if not(type(bkgsub) == bool):
            source_bkgsub = bkgsub[0]
            tmp_bkgsub = bkgsub[1]
        else:
            source_bkgsub = bkgsub
            tmp_bkgsub = bkgsub
        # check if input astrometry significance sigma is list or single value 
        if not(type(astrom_sigma) in [float,int]):
            source_sig = astrom_sigma[0]
            tmp_sig = astrom_sigma[1]
        else:
            source_sig = astrom_sigma
            tmp_sig = astrom_sigma
        # check if input astrometry PSF sigma is list or single value 
        if not(type(psf_sigma) in [float,int]):
            source_psf = psf_sigma[0]
            tmp_psf = psf_sigma[1]
        else:
            source_psf = psf_sigma
            tmp_psf = psf_sigma
            
        # -O --> overwrite
        # -p --> source significance 
        # -w --> estimated PSF sigma 
        # -s 10 --> size of the median filter to apply to the image is 10x10
        # -m 10000 --> max object size for deblending is 10000 pix**2
        
        ## find control points in source image 
        options = " -O -p "+str(source_sig)+" -w "+str(source_psf)+" -s 10"
        options += " -m 10000 "
        if source_bkgsub: 
            options += "-b " # no need for subtraction
        run("image2xy"+options+source_file, shell=True)    
        source_list_file = source_file.replace(".fits", ".xy.fits")
        source_list = Table.read(source_list_file)
        if len(source_list) == 0: # if no sources found 
            print("\nNo sources found with astrometry.net in the source "+
                  "image, so image alignment cannot be obtained. Exiting.")
            return
        # pick at most <nsources> sources below the 95% flux percentile
        source_list.sort('FLUX') # sort by flux
        source_list.reverse() # biggest to smallest
        start = int(0.05*len(source_list))
        end = min((len(source_list)-1), (start+nsources))
        source_list = source_list[start:end]   
        source_list = np.array([[source_list['X'][i], 
                                 source_list['Y'][i]] for i in 
                               range(len(source_list['X']))]) 
        # remove sources in leftmost/rightmost/topmost/bottomost 5% of image
        source_list = [s for s in source_list.copy() if (
                (min(s[0], source.shape[1]-s[0])>0.05*source.shape[1]) and
                (min(s[1], source.shape[0]-s[1])>0.05*source.shape[0]))]
            
        ## find control points in template image
        options = " -O -p "+str(tmp_sig)+" -w "+str(tmp_psf)+" -s 10 -m 10000 "
        if tmp_bkgsub: 
            options += "-b " # no need for subtraction
        run("image2xy"+options+template_file, shell=True)    
        template_list_file = template_file.replace(".fits", ".xy.fits")        
        template_list = Table.read(template_list_file)
        if len(template_list) == 0: # if no sources found 
            print("\nNo sources found with astrometry.net in the template "+
                  "image, so image alignment cannot be obtained. Exiting.")
            return
        # pick at most <nsources> sources below the 95% flux percentile
        template_list.sort('FLUX') # sort by flux 
        template_list.reverse() # biggest to smallest
        start = int(0.05*len(template_list))
        end = min((len(template_list)-1), (start+nsources))
        template_list = template_list[start:end]  
        template_list = np.array([[template_list['X'][i], 
                                   template_list['Y'][i]] for i in 
                                 range(len(template_list['X']))])
        # remove sources in leftmost/rightmost/topmost/bottomost 5% of image
        template_list = [t for t in template_list.copy() if (
                (min(t[0], template.shape[1]-t[0])>0.05*template.shape[1]) and
                (min(t[1], template.shape[0]-t[1])>0.05*template.shape[0]))]
    
        if keep:
            print("\nKeeping the source list files for the science and "+
                  "template images. They have been written to:")
            print(source_list_file+"\n"+template_list_file)
        else:
            run("rm "+source_list_file, shell=True) # not needed
            run("rm "+template_list_file, shell=True) 

        try: 
            # find the transform using the control points
            tform, __ = aa.find_transform(source_list, template_list)
            # apply the transform
            img_aligned, footprint = aa.apply_transform(tform, source,
                                                        template,
                                                        propagate_mask=True)
        except aa.MaxIterError: # if cannot match images, try flipping 
            print("\nMax iterations exceeded; flipping the image...")
            xsize = fits.getdata(source_file).shape[1]
            ysize = fits.getdata(source_file).shape[0]
            source_list = [[xsize,ysize]-coords for coords in 
                           source_list.copy()]
            source = np.flip(source, axis=0)
            source = np.flip(source, axis=1)
                
            try:
                tform, __ = aa.find_transform(source_list, template_list)
                print(tform)
                img_aligned, footprint = aa.apply_transform(tform, source,
                                                            template,
                                                           propagate_mask=True)
            except aa.MaxIterError: # still too many iterations 
                print("\nMax iterations exceeded while trying to find "+
                      "acceptable transformation. Exiting.\n")
                return
            
        except aa.TooFewStarsError: # not enough stars in source/template
            print("\nReference stars in source/template image are less than "+
                  "the minimum value (3). Exiting.")
            return
        
        except Exception: # any other exceptions
            e = sys.exc_info()
            print("\nWhile calling astroalign, some error other than "+
                  "MaxIterError or TooFewStarsError was raised: "+
                  "\n"+str(e[0])+"\n"+str(e[1]))
            return
    
    ### OPTION 2: use image segmentation to find sources in the source and 
    ### template, find the transformation using these sources as control 
    ### points, and apply the transformation to the source image 
    else: 
        try: 
            # find control points using image segmentation, find the transform,
            # and apply the transform
            img_aligned, footprint = aa.register(source, template,
                                                 propagate_mask=True,
                                                 thresh=thresh_sigma)
        except aa.MaxIterError: # if cannot match images, try flipping 
            print("\nMax iterations exceeded; flipping the image...")
            source = np.flip(source, axis=0)
            source = np.flip(source, axis=1)
                
            try:
                img_aligned, footprint = aa.register(source, template, 
                                                     propagate_mask=True,
                                                     thresh=thresh_sigma)
            except aa.MaxIterError: # still too many iterations 
                print("\nMax iterations exceeded while trying to find "+
                      "acceptable transformation. Exiting.\n")
                return
            
        except aa.TooFewStarsError: # not enough stars in source/template
            print("\nReference stars in source/template image are less than "+
                  "the minimum value (3). Exiting.")
            return
        
        except Exception: # any other exceptions
            e = sys.exc_info()
            print("\nWhile calling astroalign, some error other than "+
                  "MaxIterError or TooFewStarsError was raised: "+
                  "\n"+str(e[0])+"\n"+str(e[1]))
            return
        
    # build the new mask 
    # mask pixels==0 or nan and mask the footprint of the image registration
    # the mask should propagate via astroalign, but not sure if it does atm
    mask = np.logical_or(img_aligned==0, np.isnan(img_aligned))
    mask = np.logical_or(mask, footprint)
    
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
            cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)

        cb.ax.tick_params(which='major', labelsize=15)            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("registered image via "+r"$\mathbf{\mathtt{astroalign}}$", 
                  fontsize=15)
        plt.savefig("aligned_source_image_"+scale+".png", bbox_inches="tight")
    
    # set header for new aligned fits file 
    hdr = fits.getheader(source_file)
    hdr["ASTALIGN"] = "Yes" # make a note that astroalign was successful 
    w = wcs.WCS(template_header)
    
    if not("SIP" in template_header["CTYPE1"]) and ("SIP" in hdr["CTYPE1"]):
        __remove_SIP(hdr)
    hdr.update((w.to_fits(relax=True))[0].header)
        
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


def image_align_morph(source_file, template_file, mask_file=None, 
                      flip=False, maxoffset=30.0, plot=False, scale=None, 
                      write=True, output_im=None, output_mask=None):
    """
    Input:
        - science image (source) to register
        - template image (target) to match source to
        - mask file for the SOURCE image (optional; default None)
        - whether to flip the image (invert along X and Y) before tying to 
          align (optional; default False)
        - maximum allowed pixel offset before deciding that alignment is not
          accurate (optional; default 30.0 pix)
        - whether to plot the matched image data (optional; default False)
        - scale to apply to the plot (optional; default None (linear); options
          are "linear", "log", "asinh")
        - whether to write the output to .fits files (optional; default True)
        - name for output aligned image file (optional; default set below)
        - name for output mask image file (optional; default set below)
    
    Input: the science image (the source), the template to match to, a mask of
    bad pixels to ignore (optional; default None), a bool indicating whether to 
    flip (invert along x AND y) the image before trying to align (optional; 
    default False), the maximum allowed offset before deciding 
    that the alignment is not accurate (optional; default 30.0 pix), a bool 
    indicating whether to plot the matched image data (optional; default 
    False), a scale to apply to the plot (optional; default None (linear); 
    options are "linear", "log", "asinh"), whether to write output .fits to 
    files (optional; default True) and names for the output aligned image and 
    image mask (both optional; defaults set below)
    
    Calls on image_registration to align the source image with the target to 
    allow for proper image subtraction. Also finds a mask of out of bounds 
    pixels to ignore during subtraction. The image registration in this 
    function is based on morphology and edge detection in the image, in 
    contrast with image_align, which uses asterism-matching to align the two
    images. 
    
    For images composed mostly of point sources, use image_align. For images 
    composed mainly of galaxies/nebulae and/or other extended objects, use this
    function.
    
    Output: the aligned image data and a pixel mask
    """
    
    ## load in data
    if flip:
        source = np.flip(fits.getdata(source_file), axis=0)
        source = np.flip(source, axis=1)
    else: 
        source = fits.getdata(source_file)
    template = fits.getdata(template_file)
    template_header = fits.getheader(template_file)
    
    import warnings # ignore warning given by image_registration
    warnings.simplefilter('ignore', category=FutureWarning)
    
    from image_registration import chi2_shift
    from scipy import ndimage
    
    ## pad/crop the source array so that it has the same shape as the template
    if source.shape != template.shape:
        xpad = (template.shape[1] - source.shape[1])
        ypad = (template.shape[0] - source.shape[0])
        
        if xpad > 0:
            print("\nXPAD = "+str(xpad)+
                  " --> padding source")
            source = np.pad(source, [(0,0), (0,xpad)], mode="constant", 
                                     constant_values=0)
        elif xpad < 0: 
            print("\nXPAD = "+str(xpad)+
                  " --> cropping source")
            source = source[:, :xpad]
        else: 
            print("\nXPAD = "+str(xpad)+
                  " --> no padding/cropping source")
            
        if ypad > 0:
            print("YPAD = "+str(ypad)+
                  " --> padding source\n")
            source = np.pad(source, [(0,ypad), (0,0)], mode="constant", 
                                     constant_values=0)
            
        elif ypad < 0:
            print("YPAD = "+str(ypad)+
                  " --> cropping source\n")
            source = source[:ypad, :]
        else: 
            print("YPAD = "+str(ypad)+
                  " --> no padding/cropping source")

    ## build up and apply a mask
    srcmask = np.logical_or(source==0, np.isnan(source)) # zeros/nans in source
    tmpmask = np.logical_or(template==0,np.isnan(template)) # in template 
    mask = np.logical_or(srcmask, tmpmask)
    
    if mask_file: # if a mask is provided
        maskdata = fits.getdata(mask_file) # load it in
        maskdata = maskdata[0:srcmask.shape[0], 0:srcmask.shape[1]] # crop
        mask = np.logical_or(mask, fits.getdata(mask_file))

    source = np.ma.masked_where(mask, source)
    template = np.ma.masked_where(mask, template)
        
    ## compute the required shift
    xoff, yoff, exoff, eyoff = chi2_shift(template, source, err=None, 
                                          return_error=True, 
                                          upsample_factor="auto",
                                          boundary="constant")
    
    # if offsets are too large, try flipping the image 
    if not(abs(xoff) < maxoffset and abs(yoff) < maxoffset):   
        print("\nEither the X or Y offset is larger than "+str(maxoffset)+" "+
              "pix. Flipping the image and trying again...") 
        source = np.flip(source, axis=0) # try flipping the image
        source = np.flip(source, axis=1)
        xoff, yoff, exoff, eyoff = chi2_shift(template, source, err=None, 
                                              return_error=True, 
                                              upsample_factor="auto",
                                              boundary="constant")
        # if offsets are still too large, don't trust them 
        if not(abs(xoff) < maxoffset and abs(yoff) < maxoffset):
            print("\nAfter flipping, either the X or Y offset is still "+
                  "larger than "+str(maxoffset)+" pix. Exiting.")
            return 
        
    ## apply the shift 
    img_aligned = ndimage.shift(source, np.array((-yoff, -xoff)), order=3, 
                                mode='constant', cval=0.0, prefilter=True)   
    if mask_file:
        mask = np.logical_or((img_aligned == 0), mask)
    else: 
        mask = (img_aligned == 0)
    
    print("\nX OFFSET = "+str(xoff)+"+/-"+str(exoff))
    print("Y OFFSET = "+str(yoff)+"+/-"+str(eyoff)+"\n")

    if plot: # plot, if desired
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
            cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)

        cb.ax.tick_params(which='major', labelsize=15)            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("registered image via "+
                  r"$\mathbf{\mathtt{image_registration}}$", fontsize=15)
        plt.savefig("aligned_source_image_"+scale+".png", bbox_inches="tight")
    
    # set header for new aligned fits file    
    hdr = fits.getheader(source_file)
    hdr["IMGREG"] = "Yes" # make a note that image_registration was applied
    w = wcs.WCS(template_header)

    if not("SIP" in template_header["CTYPE1"]) and ("SIP" in hdr["CTYPE1"]):
        __remove_SIP(hdr)
        
    hdr.update((w.to_fits(relax=True))[0].header)
    
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
        
        # estimate the background standard deviation
        try:
            sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
        except TypeError: # in older versions of astropy, "maxiters" was "iters"
            sigma_clip = SigmaClip(sigma=3, iters=5)
        bkg_estimator = MedianBackground()
        
        bkg = Background2D(data, (10,10), filter_size=(5,5), 
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
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
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
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
#### ePSF BUILDING ####

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
        print("ePSF FWHM: "+str(epsf_radius*2.0/10.0)+" pix")
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
          (optional; default 1.8)
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
        options = " -b -O -p "+str(astrom_sigma)+" -w "+str(psf_sigma)
        options += " -m "+str(alim)+" "    
        run("image2xy"+options+image_file, shell=True) 
        image_sources_file = image_file.replace(".fits", ".xy.fits")
        image_sources = fits.getdata(image_sources_file)
        if clean:
            run("rm "+image_sources_file, shell=True) # this file is not needed
        print("\n"+str(len(image_sources))+" stars at >"+str(astrom_sigma)+
              " sigma found in image "+re.sub(".*/", "", image_file)+
              " with astrometry.net")  

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
        print(str(nstars_epsf)+" stars used in building the ePSF")
        
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
        plt.imshow(epsf_data, origin='lower', aspect=1, cmap='magma',
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


def get_substamps(source_file, template_file, mask_file=None, sigma=3.0, 
                  etamax=3.0, area_max=200.0, sep_max=1.0, 
                  output=None):
    """
    WIP:
        - Function does what it's asked, but not sure if the output file is 
          successfully understood and used by hotpants 
    
    Input: 
        - science image (the source)
        - template to match to
        - mask of bad pixels to ignore (optional; default None)
        - sigma to use in setting the threshold for image segmentation 
          (optional; default 3.0)
        - maximum elongation allowed for the sources (optional; default 1.8)
        - maximum pixel area allowed for the sources (optional; default 200.0
          pix**2)
        - maximum allowed separation for sources common to both the science
          image and template image, in arcsec (optional; default 1.0")
        - name for the output ascii txt file of substamps 'x y' (optional; 
          default set below)
    
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
        try:
            sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
        except TypeError: # older versions of astropy, "maxiters" was "iters"
            sigma_clip = SigmaClip(sigma=3, iters=5)
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
        mask = (tbl["elongation"] <= etamax)
        tbl = tbl[mask]
        mask = tbl["area"].value <= area_max
        tbl = tbl[mask]
        obj_props.append(tbl)

    ## get RA, Dec of sources after this restriction
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
    
    ## find sources which are present in both the science image and template
    ## sources must be <= 1.0" away from each other 
    idx_sci, idx_tmp, d2d, d3d = tmp_skycoords.search_around_sky(
                                 source_skycoords, sep_max*u.arcsec) 
    obj_props[0] = obj_props[0][idx_sci]
    obj_props[1] = obj_props[1][idx_tmp]
    
    ## write x, y to a substamps ascii file 
    nmatch = len(obj_props[0])
    x = [np.mean([obj_props[0]["xcentroid"][i].value, 
                  obj_props[1]["xcentroid"][i].value]) for i in range(nmatch)]
    y = [np.mean([obj_props[0]["ycentroid"][i].value, 
                  obj_props[1]["ycentroid"][i].value]) for i in range(nmatch)]
    if not(output):
        output = source_file.replace(".fits", "_substamps.txt")
    lines = ["'"+str(x[n])+" "+str(y[n])+"'" for n in range(nmatch)]
    np.savetxt(output, lines, fmt="%s", newline="\n")
    
    return obj_props
    

def param_estimate_hotpants(source_file, template_file, mask_file=None,
                            source_epsf_file=None, tmp_epsf_file=None, 
                            thresh_sigma=3.0, pixelmin=20, etamax=1.4,
                            area_max=500, cutout=35, verbose=False):
    """   
    WIP:
        - Not stable right now. Messes up when science and reference have 
          very similar PSFs.
    
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
    
    ## load in OR build the source image ePSF. get FWHM
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
             mask_file=None, substamps_file=None, param_estimate=False,
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
             kern_write=False, 
             maskout=None, convout=None, kerout=None, 
             v=1, 
             plot=True, plotname=None, scale=None, 
             target=None, target_small=None,
             thresh_sigma=3.0, pixelmin=20, etamax=1.4, area_max=500):
    """       
    hotpants args OR hotpants-related: 
        basic inputs:
        - the science image 
        - the template to match to
        - a mask of which pixels to ignore (optional; default None)
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
          'i' for image, 't' for template, 'u' for unconvolved)
        
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
        - name for the output bad pixel mask (optional; default set below; only
          relevant if mask_write=True)
        - name for the output convolved image (optional; default set below; 
          only relevant if conv_write=True)
        - name for the output kernel (optional; default set below; only 
          relevant if kern_write=True)
        
        verbosity:
        - verbosity (optional; default 1; options are 0 - 2 where 0 is least
          verbose)
        
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
        
    ######################### INPUTS/OUTPUTS FOR HOTPANTS #####################
    
    ### input/output files and masks ##########################################
    hp_options = " -inim "+source_file+" -tmplim "+template_file
    hp_options += " -outim "+output # output subtracted file
    
    if mask_write:
        if not(maskout):
            maskout = source_file.replace(".fits", "_submask.fits")
        hp_options += " -omi "+maskout # output bad pixel mask        
    if conv_write:
        if not(convout):
            convout = source_file.replace(".fits", "_conv.fits")        
        hp_options += " -oci "+convout # output convolved file       
    if kern_write:
        if not(kerout):
            kerout = source_file.replace(".fits", "_kernel.fits")
        hp_options += " -oki "+kerout # output kernel file
        
    if mask_file: # if a mask is supplied
            mask = fits.getdata(mask_file)
            hp_options += " -tmi "+mask_file # mask for template 
            hp_options += " -imi "+mask_file # same mask for source 
    if substamps_file: # if a file of substamps X Y is supplied
        hp_options += " -afssc 1 -ssf "+substamps_file
        hp_options += " -savexy "+source_file.replace(".fits", "_conv")

    ### if requested, compare sci/tmp ePSF to estimate optimal params #########    
    if param_estimate:
        c, g = param_estimate_hotpants(source_file, template_file, 
                                       thresh_sigma=thresh_sigma, 
                                       pixelmin=pixelmin,
                                       etamax=etamax,
                                       area_max=area_max, verbose=True)
    
        ng = "3 6 %.2f 4 %.2f 2 %.2f"%(g[6], g[4], g[2])
        rkernel = 2.5*g[4]*gaussian_sigma_to_fwhm
        print("rkernel = "+str(rkernel))
        if c == "t":
            convi, convt = False, True
        elif c == "i":
            convi, convt = True, False
            
    ### upper/lower limits for SCIENCE image ##################################    
    if iu and il: # if both limits given
        hp_options += " -iu "+str(iu)+" -il "+str(il)
        
    else: # if values are not 
        try:
            std = source_header["BKGSTD"] # header written by bkgsub function
            median = 0
        except KeyError: 
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
            try: 
                mean, median, std = sigma_clipped_stats(source_data, 
                                                        mask=final_mask,
                                                        maxiters=1)
            except TypeError: # older versions of astropy, "maxiters" was "iters"
                mean, median, std = sigma_clipped_stats(source_data, 
                                                        mask=final_mask, 
                                                        iters=1)
            
        #  set upper/lower thresholds to median +/- hsigma/lsigma*std
        hp_options += " -iu "+str(median+hsigma*std)
        hp_options += " -il "+str(median-lsigma*std)
        print("\n\nSCIENCE UPPER LIMIT = "+str(median+hsigma*std))
        print("SCIENCE LOWER LIMIT = "+str(median-lsigma*std)+"\n") 

    ### upper/lower limits for TEMPLATE image #################################  
    if tu and tl: # if both limits given
        hp_options += " -tu "+str(tu)+" -tl "+str(tl)
        
    else: # if values are not given
        try:
            std = source_header["BKGSTD"] # header written by bkgsub function
            median = 0
        except KeyError: 
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
            try: 
                mean, median, std = sigma_clipped_stats(tmp_data, mask=final_mask,
                                                        maxiters=1)
            except TypeError: # older versions of astropy, "maxiters" was "iters"
                mean, median, std = sigma_clipped_stats(tmp_data, mask=final_mask, 
                                                        iters=1)
            
        #  set upper/lower thresholds to median +/- hsigma/lsigma*std
        hp_options += " -tu "+str(median+hsigma*std)
        hp_options += " -tl "+str(median-lsigma*std)
        print("\n\nTEMPLATE UPPER LIMIT = "+str(median+hsigma*std))
        print("TEMPLATE LOWER LIMIT = "+str(median-lsigma*std)+"\n")
    
    ### x, y limits ###########################################################
    if gd: 
        hp_options += " -gd "+gd
        
    ### gain (e-/ADU) and readnoise (e-) for SCIENCE image ####################    
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

    ### gain (e-/ADU) and readnoise (e-) for TEMPLATE image ###################       
    if tg: # if argument is supplied for template gain
        hp_options += " -tg "+str(tg)
    else: # if no argument given, extract from header
        try:
            hp_options += " -tg "+str(tmp_header["GAIN"]) # CFHT header
        except KeyError: 
            # if no keyword GAIN found in header, look for PS1's gain header
            try:
                hp_options += " -tg "+str(tmp_header["HIERARCH CELL.GAIN"]) 
            except KeyError:
                pass           
    if tr: # template readnoise
        hp_options += " tr "+str(ir)
    else:
        try:
            hp_options += " -tr "+str(tmp_header["RDNOISE"])
        except KeyError: 
            try:
                hp_options += " -tr "
                hp_options += str(tmp_header["HIERARCH CELL.READNOISE"]) 
            except KeyError:
                pass
    
    ### Gaussian and convolution kernel parameters ############################
    if ng: # gaussian terms
        hp_options += " -ng "+ng        
    hp_options += " -r "+str(rkernel) # convolution kernel FWHM 

    ### regions, stamps and substamps #########################################
    # no. of regions, stamps per region
    hp_options += " -nrx %d -nry %d -nsx %d -nsy %d"%(nrx, nry, nsx, nsy)
    
    # half-width substamp around centroids 
    if not rss:        
        hp_options += " -rss "+str(2.5*rkernel) 
    else:
        hp_options += " -rss "+str(rss)
    
    ### sigma clipping, normalization, convolution ############################    
    hp_options += " -n "+norm+" -ssig "+str(ssig) # norm, sigma clipping
    
    if convi: # force convolution of the image
        hp_options += " -c i"  
    if convt: # force convolution of the template
        hp_options += " -c t" 
        
    ### spatial kernel/background variation ###################################
    hp_options += " -bgo "+str(bgo)+" -ko "+str(ko) # bg order, kernel order     

    ### misc ##################################################################
    hp_options += " -v "+str(v) # verbosity 

    ######################### CALL HOTPANTS  ##################################
    
    if HOTPANTS_PATH: # if a special path to hotpants is supplied
        run(HOTPANTS_PATH+hp_options, shell=True) # call hotpants
    else:
        run("hotpants"+hp_options, shell=True) # call hotpants
    
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
        try: 
            print("\nMEAN OF DIFFERENCE IMAGE: %.2f ± %.2f \n"%(
                    mean_diff, std_diff))
        except UnicodeEncodeError: # weird error on irulan when running qsub
            pass
    except KeyError:
        print("\nCould not find DMEAN00 and DSIGE00 (mean and standard dev. "+
              "of difference image good pixels) headers in difference image.")
        print("The difference image was probably not correctly produced. "+
              "Exiting.\n")
        return 
        
    ### PLOTTING (OPTIONAL) ##############################################
    if plot: 
        plt.figure(figsize=(14,13))
        # show WCS      
        w = wcs.WCS(source_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not scale: # if no scale to apply 
            scale = "linear"
            plt.imshow(sub, cmap='coolwarm', vmin=mean_diff-3*std_diff, 
                       vmax=mean_diff+3*std_diff, aspect=1, 
                       interpolation='nearest', origin='lower')
            #plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            
        elif scale == "log": # if we want to apply a log scale 
            sub_log = np.log10(sub)
            lognorm = simple_norm(sub_log, "log", percent=99.0)
            plt.imshow(sub_log, cmap='magma', aspect=1, norm=lognorm,
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
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)

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
    title = r"$\mathtt{"+title+"}$"+" transient candidates"
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


def __triplet_plot(og_file, sub_hdu, og_hdu, ref_hdu, n, ntargets, ra, dec, 
                   crosshair="#fe019a", cmap="bone", plotdir=None):
    """
    WIP:
        - Weird bug with $ appearing on axes which extend beyond the data
    
    Input:
        - difference image PrimaryHDU
        - science image PrimaryHDU
        - reference image PrimaryHDU
        - the id of the candidate (i.e., for candidate #14, n=14)
        - the total number of candidates
        - the RA, Dec of the transient in the triplet 
        - color for the crosshair (optional; default #fe019a ~ hot pink; will
          plot no crosshair if None)
        - colourmap to apply to all images in the triplet (optional; default 
          "bone")
        - directory in which to store plots (optional; default set below)
    
    Plots a single [science image, reference image, difference image] triplet.
    
    Output: None
    """
    
    sub_data = sub_hdu.data
    og_data = og_hdu.data
    ref_data = ref_hdu.data
    
    fig = plt.figure(figsize=(14, 5))

    # science image
    w = wcs.WCS(og_hdu.header)
    ax = fig.add_subplot(131, projection=w)
    #ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["ra"].set_ticklabel_visible(False)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    mean, median, std = sigma_clipped_stats(og_data)
    ax.imshow(og_data, vmin=mean-5*std, vmax=mean+5*std, origin='lower', 
              cmap=cmap)
    ax.set_ylabel("Dec (J2000)", fontsize=16)
    
    # reference image 
    w2 = wcs.WCS(ref_hdu.header)
    ax2 = fig.add_subplot(132, projection=w2)
    ax2.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax2.coords["dec"].set_ticklabel_visible(False)
    mean, median, std = sigma_clipped_stats(ref_data)
    ax2.imshow(ref_data, vmin=mean-5*std, vmax=mean+5*std, origin='lower', 
               cmap=cmap)
    ax2.set_xlabel("RA (J2000)", fontsize=16)
    
    # difference image 
    w3 = wcs.WCS(sub_hdu.header)
    ax3 = fig.add_subplot(133, projection=w3)
    #ax3.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax3.coords["ra"].set_ticklabel_visible(False)
    ax3.coords["dec"].set_ticklabel_visible(False)
    mean, median, std = sigma_clipped_stats(sub_data)
    ax3.imshow(sub_data, vmin=mean-5*std, vmax=mean+5*std, origin='lower', 
               cmap=cmap)

    if crosshair:        
           ax.plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
                   transform=ax.get_transform('icrs'), linewidth=2, 
                   color=crosshair, marker="")
           ax.plot([ra, ra], [dec-10.0/3600.0, dec-5.0/3600.0], 
                   transform=ax.get_transform('icrs'),  linewidth=2, 
                   color=crosshair, marker="")
           ax2.plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
                    transform=ax2.get_transform('icrs'), linewidth=2, 
                    color=crosshair, marker="")
           ax2.plot([ra, ra], [dec-10.0/3600.0, dec-5.0/3600.0], 
                    transform=ax2.get_transform('icrs'),  linewidth=2, 
                    color=crosshair, marker="")  
           ax3.plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
                    transform=ax3.get_transform('icrs'), linewidth=2, 
                    color=crosshair, marker="")
           ax3.plot([ra, ra], [dec-10.0/3600.0, dec-5.0/3600.0], 
                    transform=ax3.get_transform('icrs'),  linewidth=2, 
                    color=crosshair, marker="")
    
    
    if ntargets < 100:
        if n < 10:
            nstr = "0"+str(n)
        else:
            nstr = str(n)
    else:
        if n < 10:
            nstr = "00"+str(n)
        elif n < 100:
            nstr = "0"+str(n)
        else:
            nstr = str(n)

    figname = og_file.replace(".fits", "_candidate"+nstr+".png")
    if plotdir:
        figname = plotdir+"/"+re.sub(".*/", "", figname)
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    

def transient_detect(sub_file, og_file, ref_file, mask_file=None, 
                     thresh_sigma=5.0, pixelmin=20, dipole_width=1.0, 
                     etamax=1.8, area_max=300, nsource_max=50, 
                     toi=None, toi_sep_min=None, toi_sep_max=None,
                     write=True, output=None, 
                     plot_distributions=False,
                     plot_rejections=False,
                     plots=["zoom og", "zoom ref", "zoom diff"], 
                     sub_scale=None, og_scale=None, stampsize=200.0, 
                     crosshair_og="#fe019a", crosshair_sub="#5d06e9",
                     plotdir=None):
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
        - *minimum* required separation for any potential dipoles to be 
          considered real sources (optional; default 1.0"; setting this to None
          sets no minimum) 
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
        - scale to apply to the difference images (optional; default "asinh"; 
          options are "linear", "log", "asinh")
        - scale to apply to the science/reference images (optional; default
          "asinh"; options are "linear", "log", "asinh")
        - size of the transient stamp in pixels (optional; default 200.0)
        - colour for crosshair on transient in science/ref images (optional; 
          default ~hot pink)
        - colour for crosshair on transient in difference images (optional;
          default ~purple-blue)       
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
    
    # build a mask
    mask = np.logical_or(data==0, np.isnan(data))
    if mask_file:
        mask = np.logical_or(mask, fits.getdata(mask_file))
        
    data = np.ma.masked_where(data==0.0, data) # send all bad pix to 0

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
        plt.savefig(og_file.replace(".fits", "_elongs_areas.png"), 
                    bbox_inches="tight") # save the figure 
        plt.close()
    
    ### CANDIDATE VETTING #####################################################
    ## look for dipoles by looking for sources in (-1)*difference and cross-  
    ## matching to the segmentation image 
    if dipole_width:
        try:
            inv = cat_inv.to_table()
            inv["ra"], inv["dec"] = w.all_pix2world(inv["xcentroid"], 
                                                    inv["ycentroid"], 1)
            coords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, frame="icrs")
            inv_coords = SkyCoord(inv["ra"]*u.deg, inv["dec"]*u.deg, 
                                  frame="icrs")        
            # indices of sources within <dipole_width> arcsec of each other
            idx_inv, idx, d2d, d3d = coords.search_around_sky(inv_coords, 
                                                        dipole_width*u.arcsec)
            
            # if they correspond to the same peak, the amplitude of one part 
            # of the dipole should be no more than 5 times the amplitude of the 
            # other part of the dipole
            idx_rem = [] # indices to remove
            for n in range(len(idx)):
                i, j = idx[n], idx_inv[n]
                
                ratio = tbl[i]["source_sum"]/inv[j]["source_sum"]
                if (0.2 <= ratio <= 5): 
                    idx_rem.append(i)
            
            if len(idx_rem) > 0:
                print('\n'+str(len(idx_rem))+' likely dipole(s) (width <'+
                      str(dipole_width)+'") detected and removed')
                tbl.remove_rows(idx_rem)
            tbl_nodipoles = tbl.copy() # tbl with no dipoles
            
        except ValueError:
            print("The inversion of the difference image, (-1.0)*diff, does "+
                  "not contain any sources. Will continue without searching "+
                  "for dipoles.")
    
    ## restrict based on elongation 
    if etamax:
        premasklen = len(tbl)
        mask = tbl["elongation"] < etamax 
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print("\n"+str(premasklen-postmasklen)+" source(s) with "+
                  "elongation >"+str(etamax)+" removed")
        tbl_noelong = tbl.copy()
            
    ## restrict based on maximum pixel area 
    if area_max:
        premasklen = len(tbl)
        mask = tbl["area"].value < area_max
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print("\n"+str(premasklen-postmasklen)+" source(s) with "+
                  "area >"+str(area_max)+" pix**2 removed")
    
    vetted = tbl.copy()

    ## plot rejected/non-rejected candidates, if desired 
    if plot_rejections:
        dipoles = tbl_novetting[idx_rem]
        elongated_sources = tbl_nodipoles[tbl_nodipoles["elongation"] >= 
                                          etamax]
        large_sources = tbl_noelong[tbl_noelong["area"].value >= area_max]
        
        # colours below might be out of date
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
           print("\nSourceCatalog contains "+str(len(tbl))+" sources, which "+
                 "is above the limit of "+str(nsource_max)+". The subtraction"+
                 " is most likely poor in quality and may have large "+
                 "astrometric alignment/image_registration errors. Exiting.")
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
    print("\n"+str(len(tbl))+" candidate(s) found and passed vetting.")
    
    ## get rid of "id" column and empty columns 
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
        transient_plot(sub_file, og_file, ref_file, tbl, toi, plots, 
                       sub_scale, og_scale, stampsize, crosshair_og, 
                       crosshair_sub, plotdir)    
    return tbl


def transient_plot(sub_file, og_file, ref_file, tbl, toi=None, 
                   plots=["zoom og", "zoom ref", "zoom diff"], 
                   sub_scale=None, og_scale=None, 
                   stampsize=200.0, 
                   crosshair_og="#fe019a", crosshair_sub="#5d06e9", 
                   plotdir=None):
    """
    Inputs:
        general:
        - difference image file
        - original science image file (can be background-subtracted or not)
        - original reference image file (can be background-subtracted or not)
        - a table with at least the columns "ra", "dec" of candidate transient 
          source(s) found with transient_detect() or some other tool (can be 
          the name of a table file or the table itself)
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
        - name for the directory in which to store all plots (optional; 
          default set below)   
        
    Output: None
    """

    # check if table is a filename or pre-loaded table
    if type(tbl) == str:
        tbl = Table.read(tbl, format="ascii")
    
    targets = [tbl["ra"].data, tbl["dec"].data]
    
    if not(type(plots) in (list, np.ndarray, np.array)):
        print("\ntransient_plot() was called, but no plots were requested "+
              "via the <plots> arg. Exiting.")
        return
    elif len(plots) == 0:
        print("\ntransient_plot() was called, but no plots were requested "+
              "via the <plots> arg. Exiting.")
        return
    
    ntargets = len(targets[0])
    for n in range(ntargets):
        if ntargets < 100:
            if n < 10:
                nstr = "0"+str(n)
            else:
                nstr = str(n)
        else:
            if n < 10:
                nstr = "00"+str(n)
            elif n < 100:
                nstr = "0"+str(n)
            else:
                nstr = str(n)
        # set titles
        title = "difference image: candidate "+nstr
        title_og = "original image: candidate "+nstr
        title_ref = "reference image: candidate "+nstr

        # setting output figure names
        if plotdir[-1] == "/":
            plotdir = plotdir[-1]
        
        # crosshair on full-frame difference image 
        if "full" in plots:
            # set output figure name
            full_output = sub_file.replace(".fits", 
                                           "_diff_candidate"+nstr+".png")
            if plotdir:
                full_output = plotdir+"/"+re.sub(".*/", "", full_output)
                
            make_image(sub_file, 
                       scale=sub_scale, cmap="coolwarm", label="", title=title, 
                       output=full_output, 
                       target=toi,
                       target_small=[targets[0][n], targets[1][n]],
                       crosshair_small=crosshair_sub)
        
        # crossair on small region of science image 
        if "zoom og" in plots:
            # set output figure name
            zoom_og_output = og_file.replace(".fits", 
                                         "_zoomed_sci_candidate"+nstr+
                                         ".png")
            if plotdir:
                zoom_og_output = plotdir+"/"+re.sub(".*/", "", zoom_og_output)
                
            transient_stamp(og_file, [targets[0][n], targets[1][n]], stampsize, 
                            scale=og_scale, cmap="viridis", 
                            crosshair=crosshair_og, title=title_og, 
                            output=zoom_og_output, toi=toi)

        # crossair on small region of reference image 
        if "zoom ref" in plots:
            # set output figure name
            zoom_ref_output = og_file.replace(".fits", 
                                             "_zoomed_ref_candidate"+nstr+
                                             ".png")
            if plotdir:
                zoom_ref_output = plotdir+"/"+re.sub(".*/", "", 
                                                     zoom_ref_output)
                
            transient_stamp(ref_file, [targets[0][n], targets[1][n]], 
                            stampsize, scale=og_scale, cmap="viridis", 
                            crosshair=crosshair_og, title=title_ref, 
                            output=zoom_ref_output, toi=toi)
        
        # crossair on small region of difference image 
        if "zoom diff" in plots:
            # set output figure name
            zoom_diff_output = sub_file.replace(".fits", 
                                        "_zoomed_diff_candidate"+nstr+
                                        ".png")
            if plotdir:
                zoom_diff_output = plotdir+"/"+re.sub(".*/", "", 
                                                      zoom_diff_output)
                
            transient_stamp(sub_file, [targets[0][n], targets[1][n]], 
                            stampsize, scale=sub_scale, cmap="coolwarm", 
                            label="ADU", crosshair=crosshair_sub, title=title, 
                            output=zoom_diff_output, toi=toi)    


def transient_triplets(sub_file, og_file, ref_file, tbl, size=200, 
                       cropmode="extend", write=True, output=None,
                       plot=False, crosshair="#fe019a", cmap="bone",
                       plotdir=None):
    """    
    Inputs:
        - difference image file
        - original science image file (can be background-subtracted or not)
        - original reference image file (can be background-subtracted or not)
        - a table with at least the columns "ra", "dec" of candidate transient 
          source(s) found with transient_detect() or some other tool (can be 
          the name of a table file or the table itself)
        - the size of each 2D array in the triplet, i.e. the size of the 
          stamp to obtain around the transient(s) (optional; default 200 pix)
        - mode to use for crop_WCS (optional; default "extend"; options are
          "truncate" and "extend")
        - whether to write the produced triplet to a .npy file (optional; 
          default True)
        - name for output .npy file (or some other file) (optional; default 
          set below)
        - whether to plot all of the triplets (optional; default False)
        - color for the crosshair (optional; default #fe019a ~ hot pink; will
          plot no crosshair if None; only relevant if plot=True)
        - colourmap to apply to all images in the triplet (optional; default 
          "bone"; only relevant if plot=True)
        - name for the directory in which to store all plots (optional; 
          default set below; only relevant if plot=True)
    
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
        
    targets = [tbl["ra"].data, tbl["dec"].data]    
    ntargets = len(targets[0])
    
    triplets = []   
    for n in range(ntargets):
        ra = targets[0][n]
        dec = targets[1][n]

        # get cropped data   
        sub_hdu = crop_WCS(sub_file, ra, dec, size=size, mode=cropmode, 
                           write=False)
        sub_data = sub_hdu.data        
        og_hdu = crop_WCS(og_file, ra, dec, size=size, mode=cropmode, 
                          write=False)
        og_data = og_hdu.data        
        ref_hdu = crop_WCS(ref_file, ra, dec, size=size, mode=cropmode, 
                           write=False)
        ref_data = ref_hdu.data
        
        # make the triplet and add it to the list of triplets
        trip = np.array([og_data, ref_data, sub_data])
        triplets.append(trip)
        
        if plot:
            if plotdir[-1] == "/":
                plotdir = plotdir[-1]
                
            __triplet_plot(og_file, sub_hdu, og_hdu, ref_hdu, n, ntargets, 
                           ra, dec, crosshair, cmap, plotdir)

    triplets = np.stack(triplets) # (3, size, size) --> (N, 3, size, size)
    
    if write:
        if not(output):
            output = og_file.replace(".fits", "_candidates_triplets.npy")
        np.save(output, triplets)
        
    return triplets
        
###############################################################################
#### MISCELLANEOUS PLOTTING ####
    
def make_image(im_file, mask_file=None, scale=None, cmap="magma", label=None,
               title=None, output=None, target=None, target_small=None,
               crosshair_big="black", crosshair_small="#fe019a"):
    """
    Input: 
        - image of interest
        - bad pixels mask (optional; default None)
        - scale to use for the image (optional; default None (linear), options 
          are "linear", "log", "asinh")
        - colourmap to use for the image (optional; default is "magma")
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
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    if mask_file:
        mask = fits.getdata(mask_file)
        image_data_masked = np.ma.masked_where(mask, image_data)
        image_data = np.ma.filled(image_data_masked, 0)
    
    if not scale: # if no scale to apply 
        scale = "linear"
        mean, median, std = sigma_clipped_stats(image_data)
        plt.imshow(image_data, vmin=mean-8*std, vmax=mean+8*std, cmap=cmap, 
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
        title = r"$\mathtt{"+title+"}$"+" data"
    if not(output):
        output = im_file.replace(".fits","_"+scale+".png")

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
                    scale=None, cmap="magma", label=None, crosshair="#fe019a", 
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
        - colourmap to use for the image (optional; default is "magma")
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
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    if not scale: # if no scale to apply 
        scale = "linear"
        mean, median, std = sigma_clipped_stats(image_data)
        plt.imshow(image_data, vmin=mean-8*std, vmax=mean+8*std, cmap=cmap, 
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
        title = r"$\mathtt{"+title+"}$"+" data (stamp)"
    if not(output):
        output = im_file.replace(".fits","_stamp_"+scale+".png")

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


    
    

        
        
    
    
