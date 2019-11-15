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

** Proper Image Subtraction is currently a WIP.

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
    - Image differencing with Proper Image Subtraction (ZOGY formalism)
    - Miscellaneous plotting
    - Functions I directly took from elsewhere (with references)
    
GENERAL DEPENDENCIES NOT INCLUDED ON THIS SOFTWARE'S GITHUB:
    - astropy (used extensively)
    - photutils (used extensively)
    - astrometry.net (used extensively, but can be ignored in favour of 
      source detection with photutils' image_segmentation instead)
    - image_registration (used in image_align_fine())

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
import numpy.ma as ma
import random
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import re
import requests
#from timeit import default_timer as timer

from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.convolution import convolve_fft
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from photutils import Background2D, MedianBackground
from photutils import make_source_mask, detect_sources, source_properties

## for speedy FFTs
import pyfftw
import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs
pyfftw.interfaces.cache.enable()

from scipy.ndimage import binary_dilation, gaussian_filter

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
    if mask_file: # load a bad pixel mask if one is present 
        bp_mask = fits.getdata(mask_file)
        bp_mask = bp_mask.astype(bool)
        source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                       dilate_size=15, mask=bp_mask)
        # combine the bad pixel mask and source mask 
        final_mask = np.logical_or(bp_mask, source_mask)
    else: 
        source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                       dilate_size=15)
        final_mask = source_mask    
    
    ### BACKGROUND SUBTRACTION ###
    # estimate the background
    try:
        sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    except TypeError: # for older versions of astropy, "maxiters" was "iters"
        sigma_clip = SigmaClip(sigma=3, iters=5)
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
#### IMAGE ALIGNMENT (REGISTRATION) ####

def image_align(source_file, template_file, mask_file=None, 
                astrometry=True, astrom_sigma=8.0, psf_sigma=5.0, keep=False,
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
        - sigma threshold for image2xy source detection in the 
          source/template (optional; default 8.0 which sets the threshold to 
          8.0 for both; can pass a list to specify a different threshold for 
          source and template; only relevant if astrometry=True)
        - the sigma of the Gaussian PSF of the source/template (optional; 
          default 5.0 which sets the width to 5.0 for both; can pass a list to
          specify a different width for source and template; only relevant if 
          astrometry=True)
        - whether to keep the source list files after running image2xy 
          (optional; default False)
        - sigma threshold for source detection with image segmentation 
          (optional; default 3.0; only relevant if astrometry=False)
          
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
    
    Output: the aligned image data and a pixel mask
    """
    
    # a modified version of astroalign
    # allows the user to set the sigma threshold for source detection, which 
    # sometimes needs to be tweaked
    import astroalign_mod as aa
    
    source = fits.getdata(source_file)
    template = fits.getdata(template_file)
    template_header = fits.getheader(template_file)

    # build up masks, apply them
    mask = (source == 0) # zeros in source
    if mask_file: # if a mask is provided
        mask = np.logical_or(mask, fits.getdata(mask_file))
    source = np.ma.masked_where(mask, source)  
    
    nansmask = np.isnan(template) # nans in template 
    template = np.ma.masked_where(nansmask, template)
    
    ### OPTION 1: use astrometry.net to find the sources, find the transform, 
    ### and then apply the transform (DEFAULT)
    if astrometry:  
        # check if the input astrometry sigma is a list or single value 
        if not(type(astrom_sigma) in [float,int]):
            source_sig = astrom_sigma[0]
            tmp_sig = astrom_sigma[1]
        else:
            source_sig = astrom_sigma
            tmp_sig = astrom_sigma
        # check if the input astrometry sigma is a list or single value 
        if not(type(psf_sigma) in [float,int]):
            source_psf = psf_sigma[0]
            tmp_psf = psf_sigma[1]
        else:
            source_psf = psf_sigma
            tmp_psf = psf_sigma
            
        # -O --> overwrite, -p --> source significance 
        # -w --> estimated PSF width (pixels)
        # -s 10 --> size of the median filter to apply to the image is 10x10
        # -m 1000 --> max object size is 1000 pix**2
        
        ## find control points in source image 
        options = " -O -p "+str(source_sig)+" -w "+str(source_psf)+" -s 10"
        options += " -m 1000 "
        run("image2xy"+options+source_file, shell=True)    
        source_list_file = source_file.replace(".fits", ".xy.fits")
        source_list = Table.read(source_list_file)
        if len(source_list) == 0: # if no sources found 
            print("\nNo sources found with astrometry.net in the source "+
                  "image, so image alignment cannot be obtained. Exiting.")
            return
        # pick 50 sources (or less if not possible) below 95% flux percentile
        source_list.sort('FLUX') # sort by flux
        source_list.reverse() # biggest to smallest
        start = int(0.05*len(source_list))
        end = min((len(source_list)-1), (start+50))
        source_list = source_list[start:end]   
        source_list = np.array([[source_list['X'][i], 
                                 source_list['Y'][i]] for i in 
                               range(len(source_list['X']))]) 
        # remove sources in leftmost/rightmost/topmost/bottomost 5% of image
        source_list = [s for s in source_list.copy() if (
                (min(s[0], source.shape[1]-s[0])>0.05*source.shape[1]) and
                (min(s[1], source.shape[0]-s[1])>0.05*source.shape[0]))]
            
        ## find control points in template image
        options = " -O -p "+str(tmp_sig)+" -w "+str(tmp_psf)+" -s 10 -m 1000 "
        run("image2xy"+options+template_file, shell=True)    
        template_list_file = template_file.replace(".fits", ".xy.fits")        
        template_list = Table.read(template_list_file)
        if len(template_list) == 0: # if no sources found 
            print("\nNo sources found with astrometry.net in the template "+
                  "image, so image alignment cannot be obtained. Exiting.")
            return
        # pick 50 sources (or less if not possible) below 95% flux percentile
        template_list.sort('FLUX') # sort by flux 
        template_list.reverse() # biggest to smallest
        start = int(0.05*len(template_list))
        end = min((len(template_list)-1), (start+50))
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


def image_align_fine(source_file, template_file, mask_file=None, 
                    flip=False, maxoffset=30.0, plot=False, scale=None, 
                    write=True, 
                    output_im=None, output_mask=None):
    """
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
    pixels to ignore during subtraction.
    
    Output: the aligned image data and a pixel mask
    """
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
    
    # pad/crop the source array so that it has the same shape as the template
    # template must be larger than source
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

    # build up a mask
    zeromask = (source == 0) # zeros in source
    nansmask = np.isnan(template) # nans in template 
    mask = np.logical_or(zeromask, nansmask)
    
    if mask_file: # if a mask is provided
        mask = np.logical_or(mask, fits.getdata(mask_file))
    
    # apply the mask
    source = np.ma.masked_where(mask, source)
    template = np.ma.masked_where(mask, template)
        
    # compute the required shift
    xoff, yoff, exoff, eyoff = chi2_shift(template, source, err=None, 
                                          return_error=True, 
                                          upsample_factor="auto",
                                          boundary="constant")
    
    if not(abs(xoff) < maxoffset and abs(yoff) < maxoffset): # offsets should be small  
        print("\nEither the X or Y offset is larger than "+str(maxoffset)+" "+
              "pix. Flipping the image and trying again...") 
        source = np.flip(source, axis=0) # try flipping the image
        source = np.flip(source, axis=1)
        xoff, yoff, exoff, eyoff = chi2_shift(template, source, err=None, 
                                              return_error=True, 
                                              upsample_factor="auto",
                                              boundary="constant")
        if not(abs(xoff) < maxoffset and abs(yoff) < maxoffset):
            print("\nAfter flipping, either the X or Y offset is still "+
                  "larger than "+str(maxoffset)+" pix. Exiting.")
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
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("Registered image via image_registration", fontsize=15)
        plt.savefig("aligned_source_image_"+scale+".png", bbox_inches="tight")
    
    # set header for new aligned fits file    
    hdr = fits.getheader(source_file)
    hdr["IMGREG"] = "Yes" # make a note that image_registration worked
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
                    sat_area_min=500, sat_area_max=100000, ra_safe=None, 
                    dec_safe=None, rad_safe=None, dilation_its=5, 
                    blursigma=2.0, write=True, output=None, plot=True, 
                    plotname=None):
    """
    Input: 
        - image file
        - a mask file to merge with the created saturation mask (optional; 
          default None)
        - saturation ADU above which all pixels will be masked (optional; 
          default 40000, which is a bit below the limit for MegaCam)
        - minimum and maximum areas in square pixels for a saturated source 
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
    try:
        sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    except TypeError: # for older versions of astropy, "maxiters" was "iters"
        sigma_clip = SigmaClip(sigma=3, iters=5)
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
    if len(cat) != 0:
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
            tbl["sep"] = sep # add a column for sep from safe zone centre
            mask = tbl["sep"] > rad_safe # only select sources outside this rad
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
        
        # mask pixels equal to 0
        zeromask = data==0
        newmask = np.logical_or(newmask, zeromask)
        
        # use binary dilation to fill holes, esp. near diffraction spikes
        newmask = (binary_dilation(newmask, 
                                   iterations=dilation_its)).astype(float)
        # use gaussian blurring to smooth out the mask 
        newmask = gaussian_filter(newmask, sigma=blursigma, mode="constant", 
                                  cval=0.0)
        newmask[newmask > 0] = 1
        
        hdr = fits.getheader(image_file)
        mask_hdu = fits.PrimaryHDU(data=newmask.astype(int), header=hdr)
        
    else: # no masking needed 
        hdr = fits.getheader(image_file)
        newmask = np.zeros(shape=data.shape)
        if mask_file:
            mask = fits.getdata(mask_file)
            newmask[mask] = 1.0
        mask_hdu = fits.PrimaryHDU(data=newmask, header=hdr)
        tbl = Table() # empty table
    
    if plot:
        plt.figure(figsize=(14,13))
        w = wcs.WCS(hdr)
        ax = plt.subplot(projection=w) # show WCS
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        plt.imshow(newmask, cmap='bone_r', aspect=1, interpolation='nearest', 
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

def __ePSF_FWHM(epsf_data):
    
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

    return epsf_radius*2.0

def build_ePSF(image_file, image_source_file=None, mask_file=None,
               clean=True,
               sigma=8.0, psfsigma=5.0, alim=1000, lowper=60, highper=90, 
               cutout=35, fill=1e-9, fillrad=None, 
               write=True, output=None, plot=False, output_plot=None):
    """         
    Input: 
        - filename for a **BACKGROUND-SUBTRACTED** image
        - filename for a .xy.fits file containing detected sources with their
          x, y coords and **BACKGROUND-SUBTRACTED** fluxes (optional; default 
          None, in which case a new source list wil be made)
        - whether to remove output source list files (optional; default False;
          only relevant if no source list file is provided)
        - sigma to use in source detection with astrometry.net (optional; 
          default 8.0; only relevant if no source list file is provided)
        - estimate for the Gaussian PSF sigma for astrometry.net (optional; 
          default 5.0; only relevant if no source list file is provided)
        - maximum allowed source area in pix**2 for astrmetry.net (optional;
          default 1000; only relevant if no source list file is provided)
        - LOWER flux percentile such that sources below this flux will be 
          rejected when building the ePSF (optional; default 60 [%])
        - UPPER flux percentile such that sources above this flux will be 
          rejected when building the ePSF (optional; default 90 [%])
        - * cutout size around each star in pix (optional; default 35 pix; must 
          be ODD, rounded down if even)
        - * value that all pixels a distance <fillrad> away from the center of 
          the cutout are sent to (optional; default 1e-9)
        - * radius at which the ePSF should tend to <fill> (optional; default
          is int(cutout)-1) 
        - whether to write the ePSF to a new fits file (optional; default True)
        - name for the output fits file (optional; default set below)
        - whether to plot the ePSF (optional; default False) 
        - name for the output plot (optional; default set below)
        
    * Currently not in use
    
    Uses astrometry.net to obtain a list of sources in the image with their 
    x, y coordinates, flux, and background at their location. (If a list of 
    sources has already been obtained, this can be input). Then selects  
    stars between the <lowper>th and <highper>th percentile flux and uses 
    EPSFBuilder to empirically obtain the ePSF of these stars. Optionally 
    writes and/or plots the obtaind ePSF.
    
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
    
    ## setup: get WCS coords for all sources 

    # use pre-existing file, if supplied
    if image_source_file:
        image_sources = np.logical_not(fits.getdata(image_source_file))
        
    # OR, use astrometry.net to find the sources 
    # -b --> no background-subtraction, -O --> overwrite, -p _ --> signficance,
    # -w --> estimated PSF width, -m <alim> --> max object size is <alim>    
    else:
        options = " -b -O -p "+str(sigma)+" -w "+str(psfsigma)
        options += " -m "+str(alim)+" "    
        run("image2xy"+options+image_file, shell=True) 
        image_sources_file = image_file.replace(".fits", ".xy.fits")
        image_sources = fits.getdata(image_sources_file)
        if clean:
            run("rm "+image_sources_file, shell=True) # this file is not needed

    x = np.array(image_sources['X'])
    y = np.array(image_sources['Y'])
    w = wcs.WCS(image_header)
    wcs_coords = np.array(w.all_pix2world(x,y,1))
    ra = Column(data=wcs_coords[0], name='ra')
    dec = Column(data=wcs_coords[1], name='dec')
    print("\n"+str(len(ra))+" stars at >"+str(sigma)+" sigma found in image "+
          re.sub(".*/", "", image_file)+" with astrometry.net")    
    
    sources = Table() # build a table 
    sources['x_mean'] = image_sources['X'] # for BasicPSFPhotometry
    sources['y_mean'] = image_sources['Y']
    sources['x'] = image_sources['X'] # for EPSFBuilder 
    sources['y'] = image_sources['Y']
    sources.add_column(ra)
    sources.add_column(dec)
    sources['flux'] = image_sources['FLUX']
 
    ## mask out edge sources: 
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
        
    ## empirically obtain the effective Point Spread Function (ePSF)  
    nddata = NDData(image_data) # NDData object
    if mask_file: # supply a mask if needed 
        nddata.mask = fits.getdata(mask_file)
    if cutout%2 == 0: # if cutout even, subtract 1
        cutout -= 1
    stars = extract_stars(nddata, sources, size=cutout) # extract stars
    
    # use only the stars with fluxes between two percentiles
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
    nstars_epsf = len(idx_stars) # no. of stars used in ePSF building
    print(str(nstars_epsf)+" stars used in building the ePSF")
    
    # update stars object and then **build the ePSF**
    # have to manually update all_stars AND _data attributes
    stars.all_stars = [stars[i] for i in idx_stars]
    stars._data = stars.all_stars
    epsf_builder = EPSFBuilder(oversampling=1, maxiters=10, # build it
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    epsf_data = epsf.data
    
    ## set pixels outside radius <fillrad> to <fill>, renormalize
    #y, x = np.ogrid[:cutout, :cutout]
    #center = [cutout/2.0, cutout/2.0]
    #if not fillrad:
    #    fillrad = float(int(cutout/2.0))-1.0
    #dist_from_center = np.sqrt((x-center[0])**2 + (y-center[0])**2)
    #mask = dist_from_center >= fillrad
    #epsf_data[mask] = fill
    #epsf_data = epsf_data/np.sum(epsf_data)
    
    
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

###############################################################################
#### IMAGE DIFFERENCING WITH HOTPANTS ####

def hotpants_path(path):
    """
    Input: a path to the hotpants executable
    Output: None
    """
    global HOTPANTS_PATH
    HOTPANTS_PATH = path


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
    

def param_estimate_hotpants(source_file, template_file, mask_file=None,
                            source_list_file=None, tmp_list_file=None):
    """
    WIP:
        - blah, I am drunk
    
    Inputs:
        - science image 
        - template image
        - mask image (optional; default None)
        - file containing list of sources in science image with **BACKGROUND-
          SUBTRACTED** fluxes (optional; default None)
        - file containing list of sources in template image with **BACKGROUND-
          SUBTRACTED** fluxes (optional; default None)
     
    Given some science and template image, determines the optimal parameters 
    to pass to hotpants for successful subtraction.
          
    Outputs:
        estimates (recommendations) for the follwing hotpants parameters:
        - Gaussian terms
        - convolution kernel FWHM 
        - whether to convolve the image or template 
    """
    
    source_epsf = build_ePSF(source_file, source_list_file, mask_file, 
                             clean=True, write=False)
    tmp_epsf = build_ePSF(source_file, source_list_file, mask_file, 
                          clean=True, write=False)
    
    source_epsf_fwhm = __ePSF_FWHM(source_epsf)
    tmp_epsf_fwhm = __ePSF_FWHM(tmp_epsf)
    
    if tmp_epsf_fwhm > source_epsf_fwhm:
        something = ""
    else:
        something = ""
    
    return something

def hotpants(source_file, template_file, 
             mask_file=None, substamps_file=None, 
             iu=None, tu=None, il=None, tl=None, 
             lsigma=5.0, hsigma=5.0,
             gd=None, 
             ig=None, ir=None, tg=None, tr=None, 
             ng=None, rkernel=10, 
             nrx=1, nry=1, nsx=10, nsy=10, nss=3, rss=None,
             norm="t",
             convi=False, convt=False, 
             bgo=0, ko=1, 
             output=None, mask_write=False, conv_write=False, 
             kern_write=False, 
             maskout=None, convout=None, kerout=None, 
             v=1, 
             plot=True, plotname=None, scale=None, 
             target=None, target_small=None):
    """       
    hotpants args: 
        basic inputs:
        - the science image 
        - the template to match to
        - a mask of which pixels to ignore (optional; default None)
        - a text file containing the substamps 'x y' (optional; default None) 
        
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
          if no headers are found)
        
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
        - half width of substamp around centroids (optional; default is 2.0*
          rkernel) 
        
        normalization:
        - normalization term (optional; default 't' for template; options are 
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
    
    ### input/output files and masks ###########################################
    
    hp_options = " -inim "+source_file+" -tmplim "+template_file
    hp_options += " -outim "+output # output subtracted file
    
    if mask_write:
        if not(maskout):
            maskout = source_file.replace(".fits", "_submask.fits")
        hp_options += " -omi "+maskout # output bad pixel mask        
    if conv_write:
        if not(convout):
            convout = source_file.replace(".fits", "_conv.fits")        
        hp_options += " -oci "+convout # output convoluted file       
    if kern_write:
        if not(kerout):
            kerout = source_file.replace(".fits", "_kernel.fits")
        hp_options += " -oki "+kerout # output kernel file
        
    if mask_file: # if a mask is supplied
            mask = fits.getdata(mask_file)
            hp_options += " -tmi "+mask_file # mask for template 
            hp_options += " -imi "+mask_file # same mask for source 
    if substamps_file: # if a file of substamps X Y is supplied
        hp_options += " -afssc 0 -ssf "+substamps_file
        hp_options += " -savexy "+source_file.replace(".fits", "_conv")
            
    ### upper/lower limits for SCIENCE image ##################################
    
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
        
    ### gain (e-/ADU) and readnoise (e-) for image and template ###############
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
    
    ### Gaussian and convolution kernel parameters ############################
    if ng: # gaussian terms
        hp_options += " -ng "+ng        
    hp_options += " -r "+str(rkernel) # convolution kernel FWHM 

    ### regions, stamps and substamps #########################################
    # no. of regions, stamps per region
    hp_options += " -nrx %d -nry %d -nsx %d -nsy %d"%(nrx, nry, nsx, nsy)
    
    # half-width substamp around centroids 
    if not rss:        
        hp_options += " -rss "+str(2.0*rkernel) 
    else:
        hp_options += " -rss "+str(rss)
    
    ### normalization and convolution #########################################
    
    hp_options += " -n "+norm
    
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
        
        if not(plotname):
            plotname = outfile.replace(".fits", "_hotpants.png")
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()
        
    return sub

###############################################################################
#### TRANSIENT DETECTION ####

def __triplet_plot(og_file, sub_hdu, og_hdu, ref_hdu, N, ra, dec, 
                   crosshair="#fe019a", cmap="bone", plotdir=None):
    """
    WIP:
        - Weird bug with $ appearing on axes which extend beyond the data
    
    Input:
        - difference image PrimaryHDU
        - science image PrimaryHDU
        - reference image PrimaryHDU
        - the id of the candidate (i.e., for candidate14, N=14)
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
    
    figname = og_file.replace(".fits", "_candidate"+str(N)+".png")
    if plotdir:
        figname = plotdir+"/"+re.sub(".*/", "", figname)
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    

def transient_detect(sub_file, og_file, ref_file, mask_file=None, 
                     astrometry=False, astrom_sigma=5.0, psf_sigma=5.0, 
                     clean=True, thresh_sigma=5.0, 
                     pixelmin=20, dipole_width=1.0, elongation_lim=1.8, 
                     nsource_lim=50,
                     toi=None, toi_sep_min=None, toi_sep_max=None,
                     write=True, output=None, 
                     plots=["zoom og", "zoom ref", "zoom diff"], 
                     sub_scale=None, og_scale=None, stampsize=200.0, 
                     crosshair_og="#fe019a", crosshair_sub="#5d06e9",
                     plotdir=None):
    """  
    WIP:
        - astrometry.net doesn't get very accurate isophotes, which are kind of
          needed for source_properties, and doesn't seem to present any real 
          improvement in source count or speed
        - add dipole checking for astrometry.net, eventually
      
    Inputs:
        general:
        - subtracted image file
        - original science image file (can be background-subtracted or not)
        - original reference image file (can be background-subtracted or not)
        - a mask file (optional; default None)
        
        source detection:
        - use astrometry.net for source if True, use simple image segmentation 
          to detect sources if False (optional; default False)
        - sigma threshold for astrometry.net source detection in the difference
          image (optional; default 5.0 only relevant if astrometry=True)
        - sigma of the Gaussian PSF of the difference image (optional; default 
          5.0; only relevant if astrometry=True)
        - whether to remove files output by image2xy once finished with them 
          (optional; default True; only relevant if astrometry=True)
        - sigma threshold for source detection with image segmentation 
          (optional; default 3.0; only relevant if astrometry=False)

        candidate rejection:
        - *minimum* number of isophotal pixels (optional; default 20)
        - *minimum* required separation for any potential dipoles to be 
          considered real sources (optional; default 1.0"; setting this to None
          sets no minimum) 
        - *maximum* allowed elongation for sources found by image segmentation 
          (optional; default 1.8; setting this to None sets no maximum)
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
        - an array of which plots to produce, where valid options are:
              (1) "full" - the full-frame subtracted image
              (2) "zoom og" - postage stamp of the original science image 
              (3) "zoom ref" - postage stamp of the original reference image
              (4) "zoom diff" - postage stamp of the subtracted image 
              (optional; default is ["zoom og", "zoom ref", "zoom diff"]) 
              
        the following are relevant only if plots are requested (i.e., array 
        'plots' is non-empty):     
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
        
        old (unused) args:
        - name for the full frame difference image plot (optional; default set 
          below; only relevant if "full" in array 'plots')
        - name for the zoomed in science image plot (optional; default set 
          below, only relevant if "zoom og" in array 'plots')
        - name for the zoomed in reference image plot (optional; default set 
          below, only relevant if "zoom ref" in array 'plots')
        - name for the zoomed in difference image plot (optional; default set 
          below, only relevant if "zoom diff" in array 'plots')
    
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
    data = np.ma.masked_where(data==0.0, data) # mask bad pixels
    
    # build an actual mask
    mask = (data==0)
    if mask_file:
        mask = np.logical_or(mask, fits.getdata(mask_file))

    # for now, prevent the user from using astrometry.net
    if astrometry:
        print("\nWarning: using astrometry.net for transient detection is "+
              "a work in progress and is *NOT* reliable. Overriding and "+
              "using image segmentation instead.")


    ### OPTION 1: use image segmentation to find sources with an area > 
    ### pixelmin pix**2 which are above the threshold sigma*std (DEFAULT)
    if not(astrometry):
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

        
    ### OPTION 2: use astrometry.net to find the sources
    ### WORK IN PROGRESS, NOT USABLE AT THE MOMENT ###
    else:        
        # -O --> overwrite any output files 
        # -p astrom_sigma --> source significance 
        # -w psf_sigma --> estimated PSF width (pixels)
        # -s 10 --> size of the median filter to apply to the image is 10x10
        # -m 1000 --> max object size is 1000 pix**2
        # -M ... --> name of mask file
 
        source_list_file = sub_file.replace(".fits", ".xy.fits")
        source_mask_file = sub_file.replace(".fits", "_sourcemask.fits")
       
        options = " -O -p "+str(astrom_sigma)+" -w "+str(psf_sigma)+" -s 1"
        options += " -m 1000 -M "+source_mask_file+" "      
        run("image2xy"+options+sub_file, shell=True)  
        
        # get the results 
        source_list = Table.read(source_list_file)
        maskim = fits.getdata(source_mask_file)
        
        if clean:
            run("rm "+source_list_file, shell=True) # not needed 
            run("rm "+source_mask_file, shell=True)
        
        if len(source_list) == 0: # if no sources found 
            print("\nNo sources found with astrometry.net in this difference "+
                  "image. Exiting.")
            return

        maskim[maskim<=0] = 0 # background 
        maskim[maskim>0] = random.randint(1, 100) # sources    
        
        # use the astrometry.net mask image to get the source properties 
        cat = source_properties(data, maskim)
        print(len(cat))


    ## get the catalog and coordinates for sources
    try:
        tbl = cat.to_table()
    except ValueError:
        print("SourceCatalog contains no sources. Exiting.")
        return
    w = wcs.WCS(hdr)
    tbl["ra"], tbl["dec"] = w.all_pix2world(tbl["xcentroid"], 
                                            tbl["ycentroid"], 1)
    
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
            # indices of sources within <dipole_width> " of each other
            idx_inv, idx, d2d, d3d = coords.search_around_sky(inv_coords, 
                                                        dipole_width*u.arcsec)
            
            if len(idx) > 0:
                print('\n'+str(len(idx))+' likely dipole(s) (width <'+
                      str(dipole_width)+'") detected and removed')
                tbl.remove_rows(idx)
            
        except ValueError:
            print("The inversion of the difference image, (-1.0)*diff, does "+
                  "not contain any sources. Will continue without searching "+
                  "for dipoles.")
    
    ## restrict based on elongation   
    if elongation_lim:
        premasklen = len(tbl)
        mask = tbl["elongation"] < elongation_lim 
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print("\n"+str(premasklen-postmasklen)+" source(s) with "+
                  "elongation >"+str(elongation_lim)+" removed")
    
    ## check that the number of detected sources is believable
    if nsource_lim:
        if len(tbl) > nsource_lim:
           print("SourceCatalog contains "+str(len(tbl))+" sources, which is "+
                 "above the limit of "+str(nsource_lim)+". The subtraction "+
                 "is most likely poor in quality and may have large "+
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
        
    ## sort by flux and print out number of sources found 
    tbl.sort("source_sum")  
    tbl.reverse()
    print("\n"+str(len(tbl))+" candidate(s) found.")
    
    if write:
        if not(output):
            output = og_file.replace(".fits", "_candidates.fits")
        tbl.write(output, overwrite=True, format="ascii")
    
    ## plot as desired
    transient_plot(sub_file, og_file, ref_file, tbl, toi, plots, sub_scale, 
                   og_scale, stampsize, crosshair_og, crosshair_sub, plotdir)

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
        
        old (unused) args:
        - name for the full frame difference image plot (optional; default set 
          below; only relevant if "full" in array 'plots')
        - name for the zoomed in science image plot (optional; default set 
          below, only relevant if "zoom og" in array 'plots')
        - name for the zoomed in reference image plot (optional; default set 
          below, only relevant if "zoom ref" in array 'plots')
        - name for the zoomed in difference image plot (optional; default set 
          below, only relevant if "zoom diff" in array 'plots')        
        
    Output: None
    """

    # check if table is a filename or pre-loaded table
    if type(tbl) == str:
        tbl = Table.read(tbl, format="ascii")
    
    targets = [tbl["ra"].data, tbl["dec"].data]
    
    if not(plots): # if no plots requested
        print("\ntransient_plot() was called, but no plots were requested "+
              "via the <plots> arg. Exiting.")
        return
    
    ntargets = len(targets[0])
    for n in range(ntargets):
        # set titles
        title = "difference image: candidate "+str(n)
        title_og = "original image: candidate "+str(n)
        title_ref = "reference image: candidate "+str(n)

        # setting output figure names
        if plotdir[-1] == "/":
            plotdir = plotdir[-1]
        
        # crosshair on full-frame difference image 
        if "full" in plots:
            # set output figure name
            full_output = sub_file.replace(".fits", 
                                           "_diff_candidate"+str(n)+".png")
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
                                         "_zoomed_sci_candidate"+str(n)+
                                         ".png")
            if plotdir:
                zoom_og_output = plotdir+"/"+re.sub(".*/", "", zoom_og_output)
                
            transient_stamp(og_file, 
                            [targets[0][n], targets[1][n]], 
                            stampsize, 
                            scale=og_scale, cmap="viridis", 
                            crosshair=crosshair_og,
                            title=title_og, output=zoom_og_output,
                            toi=toi)

        # crossair on small region of reference image 
        if "zoom ref" in plots:
            # set output figure name
            zoom_ref_output = og_file.replace(".fits", 
                                             "_zoomed_ref_candidate"+str(n)+
                                             ".png")
            if plotdir:
                zoom_ref_output = plotdir+"/"+re.sub(".*/", "", 
                                                     zoom_ref_output)
                
            transient_stamp(ref_file, 
                            [targets[0][n], targets[1][n]], 
                            stampsize, 
                            scale=og_scale, cmap="viridis", 
                            crosshair=crosshair_og,
                            title=title_ref, output=zoom_ref_output,
                            toi=toi)
        
        # crossair on small region of difference image 
        if "zoom diff" in plots:
            # set output figure name
            zoom_diff_output = sub_file.replace(".fits", 
                                        "_zoomed_diff_candidate"+str(n)+
                                        ".png")
            if plotdir:
                zoom_diff_output = plotdir+"/"+re.sub(".*/", "", 
                                                      zoom_diff_output)
                
            transient_stamp(sub_file, 
                            [targets[0][n], targets[1][n]], 
                            stampsize, 
                            scale=sub_scale, cmap="coolwarm", label="",
                            crosshair=crosshair_sub,
                            title=title, output=zoom_diff_output,
                            toi=toi)    


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
          
    Output: a numpy array with shape (N, 3, size, size), where N is the number 
    of rows in the input table (i.e., no. of candidate transients) and the 3 
    sub-arrays represent cropped sections of the science image, template 
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
                
            __triplet_plot(og_file, sub_hdu, og_hdu, ref_hdu, n, ra, dec, 
                           crosshair, cmap, plotdir)

    triplets = np.stack(triplets) # (3, size, size) --> (N, 3, size, size)
    
    if write:
        if not(output):
            output = og_file.replace(".fits", "_candidates_triplets.npy")
        np.save(output, triplets)
        
    return triplets
        
###############################################################################
#### IMAGE DIFFERENCING WITH PROPER IMAGE SUBTRACTION ####
#### Based on 2016 work by Barak Zackay, Eran O. Ofek and Avishay Gal-Yam ####
#### This section is a work in progress. ####

def __arrays_adjust(*arrays, odd=True):
    """
    Input: as many arrays as desired and the desired parity for the output 
    arrays (optional; default True --> odd),  False --> even
    
    Resizes all of the input arrays so that their dimensions are the same by 
    cropping arrays to the smallest dimension present. Then, enforces that the 
    dimensions are either both odd or both even.
    
    Output: the resized arrays and the x, y offsets introduced 
    """
    
    if len(arrays) == 1: # if only a single 2d array
        yshape, xshape = np.array(arrays[0].shape)
        if odd: # make it odd
            xshape = xshape + xshape%2 - 1
            yshape = yshape + yshape%2 - 1
        else: # make it even 
            xshape = xshape - xshape%2  
            yshape = yshape - yshape%2
        ret = arrays[0][:yshape, :xshape]
        xoff = arrays[0].shape[1] - xshape
        yoff = arrays[0].shape[0] - yshape
    
    else:
        ys = []
        xs = []
        for a in arrays:
            ys.append(a.shape[0])
            xs.append(a.shape[1])
        #ys = [a.shape[0] for a in np.copy(arrays)]
        #xs = [a.shape[1] for a in np.copy(arrays)]
        yshape = min(ys)
        xshape = min(xs)
        if odd: # make it odd
            xshape = xshape + xshape%2 - 1
            yshape = yshape + yshape%2 - 1
        else: # make it even 
            xshape = xshape - xshape%2  
            yshape = yshape - yshape%2  
        ret = []
        for a in arrays:
            ret.append(a[:yshape, :xshape])
        xoff = arrays[0].shape[1] - xshape
        yoff = arrays[0].shape[0] - yshape
    
    return ret, xoff, yoff 


def __badpix_fill(image_file, image_data, image_header, sigma_clip, 
                  bkg_estimator, mask, plot=False, odd=True):
    """
    WIP: 
        - Ugly plotting
    
    Input: 
        - image data
        - sigma clipping object
        - background estimator
        - mask data 
        - whether to plot the filled data (optional; default False)
        - the parity of the image dimensions before any pixel fixing (optional;
          default True --> odd; False --> even)
    
    For the raw image, fills the bad pixels identified by the mask. Does so by 
    obtaining the background and background RMS deviation and producing 
    Gaussian noise with mean equal to the median of the background and width
    equal to the median of the RMS deviation. 
    
    Output: the filled array 
    """      
    # get background 
    bkg = Background2D(image_data, (3,3), filter_size=(1,1), 
                       sigma_clip=sigma_clip, 
                       bkg_estimator=bkg_estimator,  
                       mask=mask)
    bkg_array = bkg.background
    bkg_rms = np.nanmedian(bkg.background_rms)
    bkg_rms += np.random.normal(scale=np.nanstd(bkg.background_rms), 
                                size=image_data.shape)      
    # fill masked pixels with Gaussian noise with center = median of 
    # the background and width of order the background noise 
    noise = np.random.normal(loc=np.nanmedian(bkg_array),
                             scale=np.nanmedian(bkg_rms),
                             size=image_data.shape)
    
    # resample the image by taking the median over blocks 
    from skimage.util import view_as_blocks
    from scipy.ndimage import zoom
    bkg_toresamp = bkg_array+noise
    if odd: # if input array is odd, needs to temporarily be made even
        bkg_toresamp = np.pad(bkg_toresamp, [(0,1), (0,1)], mode="constant",
                              constant_values=np.nanmedian(bkg_toresamp))
    block_shape = (2,2)
    view = view_as_blocks(bkg_toresamp, block_shape)
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1)
    bkg_resamp = np.median(flatten_view, axis=2)
    if odd: # if input array was odd, restore to odd 
        bkg_resamp_zoom = zoom(bkg_resamp, zoom=2)[:-1,:-1]
    else:
        bkg_resamp_zoom = zoom(bkg_resamp, zoom=2)
    
    # fill in the array 
    data_tofill = mask.astype(int)*(bkg_resamp_zoom)
    image_data = np.ma.filled(image_data, 0) 
    image_data += data_tofill  
    
    if plot:
        plt.figure(figsize=(14,13))
        # show WCS      
        w = wcs.WCS(image_header)
        ax = plt.subplot(projection=w) 
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        mean, median, std = sigma_clipped_stats(image_data)
        plt.imshow(image_data, vmin=mean-3*std, vmax=mean+3*std,
                   cmap="viridis", aspect=1,
                   interpolation="nearest", origin="lower")
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        cb.set_label("ADU", fontsize=16)
        topfile = re.sub(".*/", "", image_file)
        title = topfile.replace(".fits", " with bad pixels filled")
        plt.title(title, fontsize=15)
        plt.savefig(image_file.replace(".fits", "_bpix_fill.png"), 
                    bbox_inches="tight")
        plt.close()
        
    return image_data


def __flux_ratio(new_file, ref_file, sigma=8.0, psfsigma=5.0, alim=1000,
                 ADU_max_new=50000, ADU_max_ref=50000, plot=False):
    """
    Input:
        - new image file (aligned and **background-subtracted**)
        - reference image file (aligned and **background-subtracted**)
        - sigma for source detection with astrometry.net (optional; default 8)
        - estimated Gaussian PSF width (optional; default 5.0 pix)
        - maximum allowed object size in pix**2 for astrometry.net (optional;
          default 1000)
        - maximum allowed ADU for the new image's sources (optional; default 
          50000)
        - maximum allowed ADU for the ref image's sources (optional; default
          50000)
        - whether to plot the correlation between the new stars' flux and 
          reference stars' flux (optional; default False)
    
    Uses astrometry.net to get the flux of as many sources as possible in each
    image. Then, compares the sources to find those within 1.0" of each
    other and calculates the ratio of their flux. Computes the mean, median 
    and standard deviation across all ratios. 
    
    Output: the mean, median and standard deviation of the flux ratios 
    """

    # load in data 
    new_data = fits.getdata(new_file)
    new_header = fits.getheader(new_file).copy()
    ref_header = fits.getheader(ref_file).copy()
    xsize = new_data.shape[1]
    ysize = new_data.shape[0]
    
    ## setup: get WCS coords for all sources 
    ## use astrometry.net to find the sources 
    # -b --> no background-subtraction, -O --> overwrite, -p _ --> signficance,
    # -w --> estimated PSF width, -m 1000 --> max object size is 1000 pix**2
    options = " -b -O -p "+str(sigma)+" -w "+str(psfsigma)+" -m "+str(alim)+" "
    
    run("image2xy"+options+new_file, shell=True)    
    new_sources_file = new_file.replace(".fits", ".xy.fits")
    new_sources = fits.getdata(new_sources_file)
    run("rm "+new_sources_file, shell=True) # this file is not needed
    
    x = np.array(new_sources['X'])
    y = np.array(new_sources['Y'])
    w = wcs.WCS(new_header)
    wcs_coords = np.array(w.all_pix2world(x,y,1))
    ra = Column(data=wcs_coords[0], name='ra')
    dec = Column(data=wcs_coords[1], name='dec')
    print("\n"+str(len(ra))+" stars at >"+str(sigma)+" sigma found in image "+
          re.sub(".*/", "", new_file)+" with astrometry.net\n")    
    new = Table() # build a table 
    new['x'] = new_sources['X'] # for EPSFBuilder 
    new['y'] = new_sources['Y']
    new.add_column(ra)
    new.add_column(dec)
    new['flux'] = new_sources['FLUX'] 
    # mask out edge sources: circle for WIRCam, rectangle for MegaPrime
    try:
        instrument = new_header["INSTRUME"]
    except KeyError:
        instrument = "Unknown"
    if "WIRCam" in instrument:
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((new['x']-xsize/2.0)**2 + 
                                 (new['y']-ysize/2.0)**2)
        mask = dist_to_center <= rad_limit
        new = new[mask]
    else: 
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        mask = (new['x']>x_lims[0]) & (new['x']<x_lims[1]) & (
                new['y']>y_lims[0]) & (new['y']<y_lims[1])
        new = new[mask]
    
    ## repeat all of the above for the reference image     
    run("image2xy"+options+new_file, shell=True) 
    ref_sources_file = ref_file.replace(".fits", ".xy.fits")
    ref_sources = fits.getdata(ref_sources_file)
    run("rm "+ref_sources_file, shell=True) # this file is not needed
    
    x = np.array(ref_sources['X'])
    y = np.array(ref_sources['Y'])
    w = wcs.WCS(ref_header)
    wcs_coords = np.array(w.all_pix2world(x,y,1))
    ra = Column(data=wcs_coords[0], name='ra')
    dec = Column(data=wcs_coords[1], name='dec')
    print("\n"+str(len(ra))+" stars at >"+str(sigma)+" sigma found in image "+
          re.sub(".*/", "", ref_file)+" with astrometry.net\n")    
    ref = Table() # build a table 
    ref['x'] = ref_sources['X'] # for EPSFBuilder 
    ref['y'] = ref_sources['Y']
    ref.add_column(ra)
    ref.add_column(dec)
    ref['flux'] = ref_sources['FLUX'] 
    # mask out edge sources: circle for WIRCam, rectangle for MegaPrime
    try:
        instrument = ref_header["INSTRUME"]
    except KeyError:
        instrument = "Unknown"
    if "WIRCam" in instrument:
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((ref['x']-xsize/2.0)**2 + 
                                 (ref['y']-ysize/2.0)**2)
        mask = dist_to_center <= rad_limit
        ref = ref[mask]
    else: 
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        mask = (ref['x']>x_lims[0]) & (ref['x']<x_lims[1]) & (
                ref['y']>y_lims[0]) & (ref['y']<y_lims[1])
        ref = ref[mask]       
        
    ## mask out saturated sources
    new = new[new["flux"]<=ADU_max_new]
    ref = ref[ref["flux"]<=ADU_max_ref]
    
    ## find sources common to both images 
    new_coords = SkyCoord(ra=new['ra'], dec=new['dec'], frame='icrs', 
                          unit='degree')
    ref_coords = SkyCoord(ra=ref['ra'], dec=ref['dec'], frame='icrs', 
                          unit='degree')    
    # indices of matching sources (within 1.0 arcsec of each other) 
    idx_new, idx_ref, d2d, d3d = ref_coords.search_around_sky(new_coords, 
                                                              1.0*u.arcsec)
    # flux of matching sources 
    new_flux = new[idx_new]["flux"]
    ref_flux = ref[idx_ref]["flux"]
    
    ## plot the correlation, if desired
    if plot: 
        # fit a straight line to the correlation
        from scipy.optimize import curve_fit
        def f(x, m, b):
            return b + m*x
        
        xdata = ref_flux
        ydata = new_flux
        popt, pcov = curve_fit(f, xdata, ydata) # obtain fit
        m, b = popt # fit parameters
        perr = np.sqrt(np.diag(pcov))
        m_err, b_err = perr # errors on parameters 
        fitdata = [m*x + b for x in xdata] # plug fit into data
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(xdata, ydata, marker='.', mec="#fc5a50", mfc="#fc5a50", ls="", 
                markersize=12, label="Data", zorder=1) 
        ax.plot(xdata, fitdata, color="blue", 
                label=r"$y = mx + b $"+"\n"+r"$ m=$%.3f$\pm$%.3f, $b=$%.3f$\pm$%.3f"%(
                        m, m_err, b, b_err), zorder=2) # the linear fit 
        ax.set_xlabel(r"$F_r$", fontsize=15)
        ax.set_ylabel(r"$F_n$", fontsize=15)
        ax.set_title("Flux ratios", fontsize=15)
        ax.legend(loc="upper left", fontsize=15, framealpha=0.5)
        plt.savefig(new_file.replace(".fits", "_flux_ratios.png"), 
                    bbox_inches="tight")
        plt.close()
    
    ## statistics
    mean, med, std = sigma_clipped_stats(new_flux/ref_flux)
    return mean, med, std
    

def __solve_beta(nd, rd, pn, pr, nr2, rr2, beta0, pnan=False, 
                 nnan="interpolate", its=10, plot=False, hdr=None, 
                 nthreads=4):
    """
    WIP:
        - WAY too slow right now. 
    
    Input: 
        - new image data
        - reference image data
        - new image ePSF
        - reference image ePSF
        - new image noise**2
        - reference image noise**2
        - initial guess for the beta parameter 
        
        - whether to preserve nans during convolution or to interpolate/fill 
          them (optional; default False)
        - the treatment for nans (optional; default "interpolate"; other option
          is "fill" which will fill all nans with zeroes; only relevant if 
          pnan=False)
        - no. of iterations to use in the solver (optional; default 10)
        - whether to plot the result (optional; default False)
        - image header to use to obtain WCS axes for the plot (optional; 
          default None, only necessary if plotting result)
        - no. of threads to use in all FFTs/iFFTs (optional; default 4)
    
    Uses Anderson mixing (nonlinear solving) to obtain a solution for the beta 
    parameter used in proper image subtraction. 
    
    Output: the beta parameter
    """
    from scipy.optimize import root
    from autograd import jacobian
    
    # fourier transforms
    pn_hat = fft.fft2(pn, threads=nthreads)
    pr_hat = fft.fft2(pr, threads=nthreads)
    
    # get rid of most of edges to speed up computation
    xcrop = [int(0.45*nd.shape[1]), int(0.55*nd.shape[1])]
    ycrop = [int(0.45*nd.shape[0]), int(0.55*nd.shape[0])]
    if (xcrop[1]-xcrop[0])%2 == 0:
        xcrop[1] += 1
    if (ycrop[1]-ycrop[0])%2 == 0:
        ycrop[1] += 1       
    #og_shape = nd.shape
    nd = nd[ycrop[0]:ycrop[1], xcrop[0]:xcrop[1]]
    rd = rd[ycrop[0]:ycrop[1], xcrop[0]:xcrop[1]]
    nr2 = nr2[ycrop[0]:ycrop[1], xcrop[0]:xcrop[1]]
    rr2 = rr2[ycrop[0]:ycrop[1], xcrop[0]:xcrop[1]]

    # compute denominator terms for D_hat (without beta)
    d1_ker = fft.ifft2(np.abs(pn_hat*np.conj(pn_hat)), threads=nthreads)
    d1_ker = fft.fftshift(d1_ker)
    d1_ker = __kernel_recenter(d1_ker) 
    d1 = fft.fft2(convolve_fft(fft.ifft2(rr2, threads=nthreads), d1_ker, 
                               preserve_nan=pnan, nan_treatment=nnan, 
                               fftn=fft.fft2, ifftn=fft.ifft2, 
                               normalize_kernel=False),
                  threads=nthreads)  
     
    d2_ker = fft.ifft2(np.abs(pr_hat*np.conj(pr_hat)), threads=nthreads)
    d2_ker = fft.fftshift(d2_ker)
    d2_ker = __kernel_recenter(d2_ker) 
    d2 = fft.fft2(convolve_fft(fft.ifft2(nr2, threads=nthreads), d2_ker, 
                                   preserve_nan=pnan, nan_treatment=nnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   normalize_kernel=False),
                  threads=nthreads)
    
    def F_compute(beta):
        # compute F(beta) to find beta for which F(beta) is minimized
        # nans are not preserved because Newton-Krylov cannot handle them 
        dn = fft.fft2(convolve_fft(nd, pr, fftn=fft.fft2, ifftn=fft.ifft2, 
                                   normalize_kernel=False), threads=nthreads)
        dn = fft.ifft2(dn/np.sqrt(d1*beta**2 + d2), threads=nthreads)

        dr = fft.fft2(convolve_fft(rd, pn, fftn=fft.fft2, ifftn=fft.ifft2,
                                   normalize_kernel=False), threads=nthreads)
        dr = fft.ifft2(dr/np.sqrt(d1*beta**2 + d2), threads=nthreads)
        
        return (np.real(dn) - beta*np.real(dr))
    
    def F_jac(beta): # jacobian of F
        return jacobian(F_compute)
    
    # use least squares to find a soln (initial guess: beta=fzero_n)
    beta0 = np.ones(shape=nd.shape)*beta0
    options = {"nit":its, "disp":True}
    # at the moment, does not converge very fast...
    soln = (root(F_compute, beta0, method="anderson", #jac=F_jac, 
                 options=options)).x
    
#    print("Finding beta for each of the "+str(len(nd))+" rows...")
#    soln = np.ndarray(shape=(len(nd),1))
#    for i in range(len(nd)):
#        lsq = least_squares(F_compute, x0=beta0, bounds=(0, np.inf), 
#                            loss="arctan", verbose=2, args=[i])
#        soln[i] = lsq.x
#        print("\nbeta(row "+str(i)+") = %.7f\n"%soln[i])
    
    if plot: # plot the beta parameter
        plt.figure(figsize=(14,13))    
        w = wcs.WCS(hdr) # show WCS 
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
     
    print(soln)
    soln = np.mean(soln)
    return soln


def __kernel_recenter(kernel):
    """
    Input: the kernel (can be complex)
    Output: the re-centered kernel
    """
    from scipy.ndimage import shift, center_of_mass
    
    if np.iscomplexobj(kernel): # if complex
        kernel_real = np.real(kernel) # real
        kernel_imag = np.imag(kernel) # imaginary
        # shift real component
        arr_center = np.array([kernel_real.shape[0]/2.0, 
                               kernel_real.shape[1]/2.0])
        kern_center = np.array(center_of_mass(kernel_real))
        off = arr_center - kern_center
        yoff, xoff = off[0], off[1]
        recentered_real = shift(kernel_real, [yoff, xoff], order=3, 
                                mode='constant', cval=0, prefilter=False)
        # shift imaginary component
        arr_center = np.array([kernel_imag.shape[0]/2.0, 
                               kernel_imag.shape[1]/2.0])
        kern_center = np.array(center_of_mass(kernel_imag))
        off = arr_center - kern_center
        yoff, xoff = off[0], off[1]
        recentered_imag = shift(kernel_imag, [yoff, xoff], order=3, 
                                mode='constant', cval=0, prefilter=False)
        # combine them 
        recentered = recentered_real + 1j*recentered_imag
        
    else:   
        arr_center = np.array([kernel.shape[0]/2.0, kernel.shape[1]/2.0])
        kern_center = np.array(center_of_mass(kernel))
        off = arr_center - kern_center
        yoff, xoff = off[0], off[1]
        recentered = shift(kernel, [yoff, xoff], order=3, 
                           mode='constant', cval=0, prefilter=False)
    
    return recentered


def __get_airmass_CFHT(image_file):
    """
    Input: an image file from CFHT
    Output: the airmass of every point in the image
    
    ** Currently not in use. Would be useful for a really large-field over 
       which the airmass might actually slightly vary.** 
    """
    
    from astroplan import Observer, FixedTarget
    from astropy.time import Time

    data = fits.getdata(image_file)
    hdr = fits.getheader(image_file)
    w = wcs.WCS(hdr) 
    
    # get RA, DEC of every pixel
    inds = np.indices(data.shape)
    coords = w.all_pix2world((inds[0]).flatten(), (inds[1]).flatten(), 1)    
    ras = coords[0]
    decs = coords[1]
    skycoords = SkyCoord(ras*u.deg, decs*u.deg, frame="icrs")
    
    # build a FixedTarget for each pixel
    targs = FixedTarget(skycoords)
    
    # get the airmass of each target 
    mjd = Time(hdr["MJDATE"], format="mjd")
    cfht = Observer.at_site("Canada-France-Hawaii Telescope")
    targs_altaz = cfht.altaz(mjd, targs)
    airmass = targs_altaz.secz
    airmass = (np.reshape(airmass, newshape=data.shape)).value
    
    return airmass


def __fzero_PS1(hdr):
    """
    Input: header of an image from the PanStarrs 1 (PS1) survey
    Output: the flux-based zero point for the image
    
    Computes the flux-based zero point F as the product of the exposure time 
    and the airmass for a given image. Uses the average airmass of all images
    used in producing the PS1 deep stack.
    """
    
    exptime = hdr["EXPTIME"]
    ninputs = hdr["NINPUTS"] # no. of files in the stack
    airms = []
    for i in range(ninputs):
        if i < 10:
            airms.append(hdr["AIR_000"+str(i)])
        else:
            airms.append(hdr["AIR_00"+str(i)])
            
    airm = np.mean(airms)
    airm_std = np.std(airms)
    
    return exptime*airm, airm, airm_std                    


def __fzero_DECaLS(hdr):
    """
    Input: header of an image from the Dark Energy Cam. Legacy Survey (DECaLS)
    Output: the flux-based zero point for the image
    
    Computes the flux-based zero point F as the product of the exposure time 
    and the airmass for a given image. 
    """
    pass

def __fzero_2MASS(hdr):
    """
    Input: header of an image from the 2-micron All-Sky Survey (2MASS)
    Output: the flux-based zero point for the image
    
    Computes the flux-based zero point F as the product of the exposure time 
    and the airmass for a given image. 
    """
    pass

def __plot_num_denom(filename, quant, data, term, scale=None, nthreads=4):
    """
    Inputs: filename, the quantity to plot ("numerator" or "denominator"), 
    the data array, the term ("1" or "2") the scale to apply (optional; default 
    None (linear); options are "linear", "log", "asinh"), and the number of 
    threads to use in all FFTs/iFFTs (optional; default 4)
    
    A helpful plotting function for proper image subtraction.
    
    Output: None
    """
    # determine plot title
    if quant=="numterm": # a term in the numerator
        data = fft.ifft2(data, threads=nthreads)
        titles = [r"$F_n {F_r}^2 \overline{\widehat{P_n}} |\widehat{P_r}|^2$"+
                  r"$\widehat{N}$", r"$F_r {F_n}^2 \overline{\widehat{P_r}}$"+
                  r"$|\widehat{P_n}|^2 \widehat{R}$"]
        titles = [r"$|{ }\mathfrak{F}^{-1}\{$"+t+r"$\}{ }|$" for t in titles]       
    elif quant=="denomterm": # a term in the denominator
        titles = [r"$ \sigma_r^2 {F_n}^2 |\widehat{P_n}|^2$", 
                  r"$ \sigma_n^2 {F_r}^2 |\widehat{P_r}|^2$"]
    elif quant=="denominator": # 1 / the entire denominator 
        #data = fft.ifft2(data, threads=nthreads)
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
    
    # take the norm of the complex-valued array, or the real part?
    data = np.abs(data)
    
    if not scale: # if no scale to apply 
        scale = "linear"
        mean, median, std = sigma_clipped_stats(data)
        plt.imshow(data, cmap='coolwarm', vmin=mean-8*std, 
                   vmax=mean+8*std, aspect=1, 
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
    
    if quant=="numterm":
        plt.savefig(filename.replace(".fits","_propersub_num"+str(term)+
                                     ".png"), bbox_inches="tight")
    elif quant=="denomterm":
        plt.savefig(filename.replace(".fits","_propersub_denom"+str(term)+
                                     ".png"), bbox_inches="tight")
    elif quant=="denominator":
        plt.savefig(filename.replace(".fits","_propersub_denom.png"), 
                    bbox_inches="tight")
    plt.close()


def proper_subtraction(new_file, new_og, ref_file, ref_og, 
                       pn_file=None, pr_file=None, mask_file=None, 
                       odd=True, 
                       bpix_fill=True, interpix=False, 
                       fzero_n=1, fzero_r=1, findbeta=True, betaits=10, 
                       analyticbeta=False, fluxratio=False,
                       sigma=8.0, psfsigma=5.0, alim=1000,
                       ADU_max_new=50000, ADU_max_ref=50000, 
                       pnan=False, nnan="interpolate", 
                       nthreads=4, pad=True, maskborder=False, 
                       zeroborder=False,
                       write=True, output=None, 
                       plot="S", plot_epsf=False, plot_beta=False, 
                       plot_corr=False, plot_bpixfill=False, 
                       plot_numterms=True, plot_denomterms=True,
                       plot_denominator=True,
                       scale=None, target=None, target_small=None):
    """ 
    WIP:
        - Denominator still looks funky 
        - Beta parameter takes very long to compute using nonlinear solvers 
        - Add beta finding for DECaLS, 2MASS (and CFIS? maybe not needed)
        - Add useful information to headers of difference images 
        - Factor transparency into beta calculations 
    
    Inputs: 
          MANDATORY:
        - new (science) file (aligned and background-subtracted)
        - aligned but **NOT** background-subtracted new file
        - reference (template) file (aligned and background-subtracted)
        - aligned but **NOT** background-subtracted reference file

          OPTIONAL FILES:
        - file for the new image ePSF (optional; default None, in which case a 
          new one will be created)
        - file for the ref image ePSF (optional; default None, in which case a
          new one will be created)
        - mask file for the new and ref images (optional; default None)
        
          PARITY FOR ARRAYS:
        - the parity to enforce for all of the data arrays (optional; default 
          True --> odd;  False --> even)
        
          TREATMEANT OF MASKED PIXELS:
        - whether to fill masked pixels with the background at their location
          (optional; default True)
        - whether to use a spline fitter to interpolate values for masked 
          pixels (optional; default False; EXPERIMENTAL)
          
          FLUX-BASED ZERO POINTS AND/OR BETA PARAMETER:
        - flux zero point for the new image (optional; default 1; overriden if
          any of the beta-finding bools below are True and successful)
        - flux zero point for the ref image (optional; default 1; overriden if 
          any of the beta-finding bools below are True and successful)
        - whether to solve for the beta parameter using nonlinear least-squares
          optimization (optional; default False)
        - the no. of iterations to use in solving for the beta parameter 
          (optional; default 10; onnly relevant if findbeta=True) **
        - whether to compute the fzero_n, fzero_r, and therefore beta
          parameters analytically as the product of exposure time and airmass 
          (optional; default True; only used if findbeta=False)
        - whether to (*much more crudely*) solve for the beta parameter by 
          matching sources detected in each image with astrometry and taking 
          their flux ratios (optional; default False; only used if findbeta= 
          False AND analyticbeta=False) ***

          ASTROMETRY.NET:
        - sigma to use during source detection with astrometry.net (optional; 
          default 8.0; only relevant if no ePSFs are supplied) 
        - estimate for the Gaussian PSF width for astrometry.net (optional; 
          default 5.0 pix)
        - maximum allowed area for sources detected with astrometry.net 
          (optional; defauilt 1000 pix**2)
          
          ADU MAXIMA/MINIMA:
        - maximum allowed ADU in new image (optional; default 50000 ADU)
        - maximum allowed ADU in ref image (optional; default 50000 ADU)   
        
          TREATMENT OF NANS IN convolve_fft():
        - whether to preserve nans during convolution or to interpolate/fill 
          them (optional; default False)
        - the treatment for nans (optional; default "interpolate"; other option
          is "fill" which will fill all nans with zeroes; only relevant if 
          pnan=False)
        
          OTHER:
        - number of threads to use in speedy FFTs/iFFTs (optional; default 4)
        - whether to zero-pad the ePSFs before any convolutions (optional; 
          default True)
        - whether to mask 1% of edge pixels from the input images (optional;
          default False)
        - whether to set 1% of edge pixels from the input images to zero
          (optional; default False)
        
          WRITING & PLOTTING:
        - whether to write the subtraction to a fits file (optional; default 
          True) 
        - name for the output fits file (optional; default set below)
        - which statistic to plot (optional; default S, options are 'S', 'D', 
          or None)       
        - whether to plot the ePSFs (optional; default False; only relevant if 
          no ePSFs are supplied)
        - whether to plot the beta parameter (optional; default False; only 
          relevant if findbeta=True)
        - whether to plot the flux ratios correlation (optional; default False,
          only relevant if fluxratio=True)
        - whether to plot the filled-in bad pixels (optional; default False; 
          only relevant if bpix_fill=True)
        - whether to plot the norm of the inverse FFT of both terms of the 
          numerator of S_hat (optional; default True)
        - whether to plot the norm of the inverse FFT of both terms of the 
          denominator of S_hat (optional; default True)
        - scale to apply to the plots (optional; default None (linear); options 
          are "linear", "log", "asinh")
        - [ra,dec] for a target crosshair (optional; default None)
        - [ra,dec] for second smaller target crosshair (optional; default None)
    
    Computes the most basic D and S-statistics for transient detection, 
    presented in Zackay et al., 2016. Does not compute S_corr and other 
    quantities. These will be added in the future. 
    
    ** This is the preferred way to obtain beta, but is currently not working.
    *** This is probably incorrect.
    
    Output: the D and S statistics
    """
    
    # load in data
    new_data, ref_data = fits.getdata(new_file), fits.getdata(ref_file)
    new_data_og, ref_data_og = fits.getdata(new_og), fits.getdata(ref_og)
    new_header = fits.getheader(new_file).copy()
    ref_header = fits.getheader(ref_file).copy()
    print("\nN = "+re.sub(".*/", "", new_file)+" (new image)")
    print("R = "+re.sub(".*/", "", ref_file)+" (reference image)\n")    

    # speed up all subsequent FFTs (IDK HOW THIS WORKS)
    R = ref_data.astype('complex64')
    R_hat = np.zeros_like(R)
    fft_forward = pyfftw.FFTW(R, R_hat, axes=(0,1), direction='FFTW_FORWARD', 
                              flags=('FFTW_MEASURE', ),
                              threads=nthreads, planning_timelimit=None)
    fft_forward()    
    fft_backward = pyfftw.FFTW(R_hat, R, axes=(0,1), direction='FFTW_BACKWARD', 
                               flags=('FFTW_MEASURE', ), 
                               threads=nthreads, planning_timelimit=None)
    fft_backward()
    
    ### adjusting array dimensions ###    
    if odd:
        print("Adjusting arrays to be odd in all dimensions...\n")
    else:
        print("Adjusting arrays to be even in all dimensions...\n")
    if mask_file: 
        bp_mask = fits.getdata(mask_file)
        arrs, xoff, yoff  = __arrays_adjust(new_data, ref_data, new_data_og, 
                                            ref_data_og, bp_mask, odd=odd)
        new_data, ref_data, new_data_og, ref_data_og, bp_mask = arrs
    else:
        arrs, xoff, yoff = __arrays_adjust(new_data, ref_data, new_data_og, 
                                           ref_data_og, odd=odd)
        new_data, ref_data, new_data_og, ref_data_og, xoff, yoff = arrs
    print("x offset introduced by array adjustment: "+str(xoff))
    print("y offset introduced by array adjustment: "+str(yoff)+"\n")
    
    ### obtain the ePSFs of the images ###
    if not(pn_file): # no file supplied
        print("N = "+re.sub(".*/", "", new_file)+" will be used for ePSF "+
              " building")
        pn = build_ePSF(new_file, sigma, psfsigma, alim, plot=plot_epsf)
    else:
        pn = fits.getdata(pn_file)
    if not(pr_file): 
        print("R = "+re.sub(".*/", "", ref_file)+" will be used for ePSF "+
              " building\n")
        pr = build_ePSF(ref_file, sigma, psfsigma, alim, plot=plot_epsf)
    else:
        pr = fits.getdata(pr_file)      
        
    # get the fourier transforms of the ePSFs  
    pn_hat = fft.fft2(pn, threads=nthreads)
    pr_hat = fft.fft2(pr, threads=nthreads)
    
    if pad: # zero-pad the ePSFs 
        #print("Padding the ePSFs to match the size of the image data...\n")   
        print("Zero-padding the ePSFs...\n")
        # trying to just pad them a bit 
        xpad = (new_data.shape[1] - pn.shape[1])//16
        ypad = (new_data.shape[0] - pn.shape[0])//16
        pn = np.pad(pn, [(ypad,ypad), (xpad,xpad)], mode="constant", 
                         constant_values=0.0)
        pr = np.pad(pr, [(ypad,ypad), (xpad,xpad)], mode="constant",
                         constant_values=0.0)
        pn_hat = fft.fft2(pn, threads=nthreads) 
        pr_hat = fft.fft2(pr, threads=nthreads)

    ### get the background-only error on the images ###
    # have to use **NOT** background-subtracted images for this 
    print("Estimating the noise on the original images using:")
    print("N_0 = "+re.sub(".*/", "", new_og))
    print("R_0 = "+re.sub(".*/", "", ref_og)+"\n")
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
        try:
            sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
        except TypeError: # older versions of astropy, "maxiters" was "iters"
            sigma_clip = SigmaClip(sigma=3, iters=5)
        bkg_estimator = MedianBackground(sigma_clip=sigma_clip)       
        bkg = Background2D(image_data, (5,5), filter_size=(1,1), 
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
                           mask=source_mask)
        bkg_arrays.append(bkg.background)        
        # in order for proper image subtraction to work, noise must be Gaussian
        # and uncorrelated --> estimate noise on images as the median of the 
        # RMS image + a Gaussian with width equal to the stdev of the RMS 
        bkg_rms = np.nanmedian(bkg.background_rms)
        bkg_rms += np.random.normal(scale=np.nanstd(bkg.background_rms), 
                                    size=new_data.shape)
        bkg_rms_arrays.append(bkg_rms)
        counter += 1 
    
    # background error arrays 
    new_rms2 = bkg_rms_arrays[0]**2 # square of the background error arrays
    ref_rms2 = bkg_rms_arrays[1]**2

    ### get/build masks ###
    # leave pixels = 0 alone
#    if mask_file: # load a bad pixel mask if one is present 
#        # make a simple ADU maximum mask
#        adu_mask = np.logical_or((new_data > ADU_max_new),
#                                 (ref_data > ADU_max_ref)) 
#        final_mask = np.logical_or(bp_mask,adu_mask)
#    else: 
#        final_mask = np.logical_or((new_data > ADU_max_new),
#                                   (ref_data > ADU_max_ref))
#    
#    if interpix: # fit these masked pixels with a spline
#        new_data = __inter_pix(new_data, bkg_rms_arrays[0], final_mask)
#        ref_data = __inter_pix(ref_data, bkg_rms_arrays[1], final_mask)
#    else:
#        new_data = ma.masked_array(new_data, mask=final_mask)
#        ref_data = ma.masked_array(ref_data, mask=final_mask)
#        
#        if bpix_fill: # fill bad pixels 
#            new_data = __badpix_fill(new_file, new_data, new_header.copy(), 
#                                     sigma_clip, bkg_estimator, final_mask,
#                                     plot=plot_bpixfill, odd=odd)
#            ref_data = __badpix_fill(ref_file, ref_data, ref_header.copy(), 
#                                     sigma_clip, bkg_estimator, final_mask,
#                                     plot=plot_bpixfill, odd=odd) 
    ### get/build masks ###
    # leave pixels = 0 alone as convolve_fft already handles them 
    if mask_file:
        if interpix: # fit these masked pixels with a spline
            new_data = __inter_pix(new_data, bkg_rms_arrays[0], bp_mask)
            ref_data = __inter_pix(ref_data, bkg_rms_arrays[1], bp_mask)
        else:
            new_data = ma.masked_array(new_data, mask=bp_mask)
            ref_data = ma.masked_array(ref_data, mask=bp_mask)
            
            if bpix_fill: # fill bad pixels 
                new_data = __badpix_fill(new_file, new_data, new_header.copy(), 
                                         sigma_clip, bkg_estimator, bp_mask,
                                         plot=plot_bpixfill, odd=odd)
                ref_data = __badpix_fill(ref_file, ref_data, ref_header.copy(), 
                                         sigma_clip, bkg_estimator, bp_mask,
                                         plot=plot_bpixfill, odd=odd)
    # mask pixels which are still above the ADU limit
    new_data = ma.masked_array(new_data, mask=(new_data>ADU_max_new))
    ref_data = ma.masked_array(ref_data, mask=(ref_data>ADU_max_ref))
    

    # mask 1% edge pixels in data and error arrays
    if maskborder:
        print("\nMasking the first/last 1% of the pixels from each of the 4 "+
              "edges of the data arrays...\n")
        borderx = int(0.01*new_data.shape[1])
        bordery = int(0.01*new_data.shape[0])
        new_data[0:,0:bordery], new_data[0:borderx,0:] = 1e30, 1e30 
        new_data[0:,-bordery:], new_data[-borderx:,0:] = 1e30, 1e30   
        ref_data[0:,0:bordery], ref_data[0:borderx,0:] = 1e30, 1e30
        ref_data[0:,-bordery:], ref_data[-borderx:,0:] = 1e30, 1e30
        new_data = np.ma.masked_where(new_data==1e30, new_data)
        ref_data = np.ma.masked_where(ref_data==1e30, ref_data)
    # or, set them to zero 
    if zeroborder:
        print("\nSetting the first/last 1% of the pixels from each of the 4 "+
              "edges of the data arrays to zero...\n")
        borderx = int(0.01*new_data.shape[1])
        bordery = int(0.01*new_data.shape[0])
        new_data[0:,0:bordery], new_data[0:borderx,0:] = 0, 0
        new_data[0:,-bordery:], new_data[-borderx:,0:] = 0, 0  
        ref_data[0:,0:bordery], ref_data[0:borderx,0:] = 0, 0
        ref_data[0:,-bordery:], ref_data[-borderx:,0:] = 0, 0
        new_data = np.ma.masked_where(new_data==1e30, new_data)
        ref_data = np.ma.masked_where(ref_data==1e30, ref_data)
    
    ### GET BETA (IF DESIRED) #############################################
    if findbeta: # solve for beta = fzero_n/fzero_r parameter
        print("Finding the beta parameter using a non-linear solver")
        print("beta0 = fzero_n = "+str(fzero_n)+"\n")
        beta = __solve_beta(new_data, ref_data, pn, pr, new_rms2, ref_rms2, 
                            beta0=fzero_n, pnan=pnan, nnan=nnan, its=betaits,
                            plot=plot_beta, hdr=new_header, nthreads=nthreads)
        fzero_r, fzero_n = 1, beta
        #print("beta = %.4f"%beta+"\n")
        
    elif analyticbeta: # compute beta analytically
        print("Computing fzero_n and fzero_r analytically as the product of "+
              "exposure time and airmass...\n") 
        try: # check if reference image is from CFHT
            if ref_header["INSTRUME"] in ["WIRCam", "MegaPrime"]:
                fzero_r = ref_header["EXPTIME"]*ref_header["AIRMASS"] # CFHT 
        except: # if not, figure out which survey it is from 
            try: 
                survey = ref_header["SURVEY"]
            except:
                survey = input("\nWhich survey does the reference image "+
                               "belong to? [PS1/DECaLS/2MASS]\n>>> ")
            if "PS1" in survey:
                fzero_r, airm, airm_std = __fzero_PS1(ref_header)
            elif "DECaLS" in survey:
                pass # do something
            elif "2MASS" in survey:
                pass # do something 
            else: 
                print("Input survey is not not a valid option. Could not "+
                      "compute the flux-based zero point for this image. "+
                      "Resetting fzero_n = fzero_r = 1.")        
        if not(fzero_r == 1): # if successful 
            fzero_n = new_header["EXPTIME"]*new_header["AIRMASS"] # CFHT
            beta = fzero_n/fzero_r
            print("fzero_n = %.4f"%fzero_n)
            print("fzero_r = %.4f"%fzero_r)
            print("beta = fzero_n/fzero_r = %.4f"%beta+"\n")        
        
    elif fluxratio: # solve for beta in a much more crude way
        print("Finding the beta parameter by comparing the fluxes for stars "+
              "present in both images...\n")
        betamean, betamed, betastd = __flux_ratio(new_file, ref_file, sigma,  
                                                  psfsigma, alim, ADU_max_new, 
                                                  ADU_max_ref, plot=plot_corr)
        fzero_r, fzero_n = 1, betamed
        print("beta = fzero_n/fzero_r = %.4f"%betamed+"±%.4f"%betastd+"\n")
        
    ### COMPUTE D_HAT ########################     
    print("Computing the D-statistic...\n")
    ## denominator terms
    denom1_ker = fft.ifft2(np.abs(pn_hat*np.conj(pn_hat)), threads=nthreads)
    denom1_ker = fft.fftshift(denom1_ker)
    denom1_ker = __kernel_recenter(denom1_ker) 
    denom1 = fft.fft2(convolve_fft(fft.ifft2(ref_rms2, threads=nthreads), 
                                   denom1_ker, 
                                   preserve_nan=pnan, nan_treatment=nnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   normalize_kernel=False),
                      threads=nthreads)
    #denom1 = fft.fftshift(denom1)
    
    denom2_ker = fft.ifft2(np.abs(pr_hat*np.conj(pr_hat)), threads=nthreads)
    denom2_ker = fft.fftshift(denom2_ker)
    denom2_ker = __kernel_recenter(denom2_ker) 
    denom2 = fft.fft2(convolve_fft(fft.ifft2(new_rms2, threads=nthreads),
                                   denom2_ker, 
                                   preserve_nan=pnan, nan_treatment=nnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   normalize_kernel=False),
                      threads=nthreads)
    #denom2 = fft.fftshift(denom2)
    
    ## dn_hat, dr_hat 
    dn_hat = fft.fft2(convolve_fft(new_data, pr, preserve_nan=pnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   nan_treatment=nnan, normalize_kernel=False),
                      threads=nthreads)
    dn_hat = dn_hat/np.sqrt(denom1*fzero_n**2 + denom2)
    dr_hat = fft.fft2(convolve_fft(ref_data, pn, preserve_nan=pnan, 
                                   fftn=fft.fft2, ifftn=fft.ifft2, 
                                   nan_treatment=nnan, normalize_kernel=False),
                      threads=nthreads)
    dr_hat = dr_hat/np.sqrt(denom1*fzero_n**2 + denom2)
    D_hat = dn_hat - fzero_n*dr_hat    

    ### COMPUTE S_HAT ######################################################
    print("Computing the S-statistic...\n")
    ## numerator terms 
    num1_ker = fft.ifft2(np.conj(pn_hat)*np.abs(pr_hat*np.conj(pr_hat)),
                         threads=nthreads)
    num1_ker = __kernel_recenter(num1_ker) # recenter
    num1 = fft.fft2(convolve_fft(new_data, num1_ker, preserve_nan=pnan,
                                 fftn=fft.fft2, ifftn=fft.ifft2,
                                 nan_treatment=nnan,normalize_kernel=False),
                    threads=nthreads)
    num1 *= fzero_n*fzero_r**2
    num2_ker = fft.ifft2(np.conj(pr_hat)*np.abs(pn_hat*np.conj(pn_hat)),
                         threads=nthreads)
    num2_ker = __kernel_recenter(num2_ker)
    num2 = fft.fft2(convolve_fft(ref_data, num2_ker, preserve_nan=pnan, 
                                 fftn=fft.fft2, ifftn=fft.ifft2,
                                 nan_treatment=nnan,normalize_kernel=False),
                    threads=nthreads)
    num2 *= fzero_r*fzero_n**2    
    numerator = num1 - num2   
    
    ## denominator terms         
    denom1_ker = fft.ifft2(np.abs(pn_hat*np.conj(pn_hat)), threads=nthreads)
    denom1_ker = fft.fftshift(denom1_ker)
    denom1_ker = __kernel_recenter(denom1_ker) 
    denom1 = fft.fft2(convolve_fft(fft.ifft2(ref_rms2, threads=nthreads), 
                                   denom1_ker,
                                   preserve_nan=pnan, nan_treatment=nnan,
                                   fftn=fft.fft2, ifftn=fft.ifft2,
                                   normalize_kernel=False),
                      threads=nthreads)*fzero_n**2   
    denom1 *= fzero_n**2
    
    denom2_ker = fft.ifft2(np.abs(pr_hat*np.conj(pr_hat)), threads=nthreads)
    denom2_ker = fft.fftshift(denom2_ker)
    denom2_ker = __kernel_recenter(denom2_ker) 
    denom2 = fft.fft2(convolve_fft(fft.ifft2(new_rms2, threads=nthreads), 
                                   denom2_ker,
                                   preserve_nan=pnan, nan_treatment=nnan,
                                   fftn=fft.fft2, ifftn=fft.ifft2,
                                   normalize_kernel=False),
                      threads=nthreads)*fzero_r**2
    denom2 *= fzero_r**2
    
    denom = denom1 + denom2

    ### S STATISTIC #######################################################
    S_hat = numerator/denom 
    S = fft.ifft2(S_hat, threads=nthreads)
    sub = np.real(S)
    sub_hdu = fits.PrimaryHDU(data=sub, header=new_header)
    
    ### D STATISTIC ####################################################### 
    D = fft.ifft2(D_hat, threads=nthreads) 
    dsub = np.real(D)
    dsub_hdu = fits.PrimaryHDU(data=dsub, header=new_header)
    
    #### give headers useful information ####
    
    if write:
        if not(output):
            outputS = new_file.replace(".fits", "_propersub_S.fits")
            outputD = new_file.replace(".fits", "_propersub_D.fits")
            
        sub_hdu.writeto(outputS, overwrite=True, output_verify="ignore")
        dsub_hdu.writeto(outputD, overwrite=True, output_verify="ignore")
            
    ### PLOTTING (OPTIONAL) ###############################################
    
    if plot_numterms: # plot the 2 terms in the numerator of S_hat
        __plot_num_denom(new_file, "numterm", num1, 1, scale, 
                         nthreads=nthreads) # term1
        __plot_num_denom(new_file, "numterm", num2, 2, scale,
                         nthreads=nthreads) # term2
        
    if plot_denomterms: # plot the 2 terms in the denominator of S_hat
        __plot_num_denom(new_file, "denomterm", denom1, 1, scale,
                         nthreads=nthreads) # term1
        __plot_num_denom(new_file, "denomterm", denom2, 2, scale,
                         nthreads=nthreads) # term2
        
    if plot_denominator: # plot 1/denominator
        __plot_num_denom(new_file, "denominator", 1.0/denom, 0, scale,
                         nthreads=nthreads) 
       
    if plot: 
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
            plt.imshow(sub, cmap='coolwarm', vmin=mean-8*std, 
                       vmax=mean+8*std, aspect=1, 
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
                   color="#6241c7", marker="")
            plt.gca().plot([ra, ra], [dec-5.0/3600.0, dec-2.5/3600.0], 
                   transform=plt.gca().get_transform('icrs'),  linewidth=2, 
                   color="#6241c7", marker="")     
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        title = "Proper image subtraction "
        title += r"$|%s|$"%plot+"-statistic"
        plt.title(title, fontsize=15)
        plt.savefig(new_file.replace(".fits", "_propersub_"+plot+".png"), 
                    bbox_inches="tight")
        plt.close()    
    print("\nImage subtraction complete.\n")
    
    return dsub, sub

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
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(title):
        topfile = re.sub(".*/", "", im_file)
        title = topfile.replace(".fits"," data")
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
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(title):
        topfile = re.sub(".*/", "", im_file)
        title = topfile.replace(".fits"," data (stamp)")
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

###############################################################################
#### TAKEN FROM ELSEWHERE ####

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


    
    

        
        
    
    
