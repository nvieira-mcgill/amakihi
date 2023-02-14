#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Fri Dec 25 17:14:47 2020
.. @author: Nicholas Vieira
.. @templates.py

This module enables downloading reference images (i.e. templates) to be used in
image differencing. Templates can be downloaded from each survey in the 
corresponding filters:

- **Pan-STARRS 1 (PS1)** :math:`3\\pi` **survey:** *g, r, i, z, y*
- **Dark Energy Camera Legacy Survey (DECaLS):** *g, r, z*
- **Canada-France Imaging Survey (CFIS):** *u, r*
- **2-Micron All-Sky Survey (2MASS):** *J, H, K*

Note that CFIS requires a Canadian Astronomy Data Centre (CADC) username and 
password to verify that you have access to this data.

**TO-DO:**

- Why do some of the `download_x_template()` functions not take an `output` arg
  while others do?
  
"""

# misc
import os
import sys
import numpy as np
import re
import requests

# astropy
from astropy.io import fits
from astropy import wcs

# amakihi modules for querying
from . import query_PS1
from . import query_DECaLS
from . import query_CFIS 
from . import query_2MASS

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
#### ERRORS ###################################################################

class InvalidBandError(Exception):
    """
    Raise this error when a template is requested in a single band/multiple 
    bands and none of the inputs band(s) are available for the requested 
    survey.
    """
    pass

###############################################################################
#### DOWNLOADING TEMPLATES ####################################################

def __downloadtemplate(url, survey, pixscale, output=None):
    """Downloads the fits image(s) at the given url(s). 
    
    Arguments
    ---------
    url : array_like
        URL(s) of interest
    survey : {'PS1', 'DECaLS', 'CFIS', '2MASS'}
        Survey to get the templates from
    pixscale : float
        Pixel scale of image in arcseconds per pixel
    output : str, optional
        Output filename(s) (defaults set automatically)
    
    Returns
    -------
    list of `astropy.io.fits.hdu.hdulist.HDUList` objects
        List of astropy HDUList object(s) for the downloaded template image(s)
    """

    # verify the value of survey
    if not(survey in ["PS1", "DECaLS", "CFIS", "2MASS"]):
        raise ValueError('survey must be one of '+
                         '["PS1", "DECaLS", "CFIS", "2MASS"], '+
                         f'argument provided was {survey}')

    # make the list of HDUList objects
    if type(url) == str: # if just one
        if not(output):
            output = re.sub(".*/", "", url)
            output = re.sub(".*=", "", output)
        tmpl = fits.open(url) # download it 
        tmpl.writeto(output, overwrite=True, output_verify="ignore")
        tmpl.close()
        tmpl = fits.open(output, mode="update")
        tmpl[0].header["SURVEY"] = survey # informative header
        tmpl[0].header["PIXSCAL1"] = pixscale # informative header
        tmpl.close()
        return tmpl
    
    else: # if many
        templates = []
        for i in range(len(url)): # for each url
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
            tmpl[0].header["SURVEY"] = survey # informative header
            tmpl[0].header["PIXSCAL1"] = pixscale # informative header
            print(tmpl[0].header)
            tmpl.close()
            
        return templates
    

def __downloadtemplate_auth(url, survey, pixscale, auth_user, auth_pass):
    """Downloads the fits image(s) at the given url(s), **with authorization**, 
    as is required for Canada France Imaging Survey (CFIS) images, which are 
    downloaded from the Canadian Astronomy Data Centre (CADC).
    
    Arguments
    ---------
    url : array_like
        URL(s) of interest
    survey : {'PS1', 'DECaLS', 'CFIS', '2MASS'}
        Survey to get the templates from
    pixscale : float
        Pixel scale of image in arcseconds per pixel
    auth_user, auth_pass : str
        CADC username and password
    
    Returns
    -------
    list of `astropy.io.fits.hdu.hdulist.HDUList` objects
        List of astropy HDUList object(s) for the downloaded template image(s)
    """

    # verify the value of survey
    if not(survey == "CFIS"):
        raise ValueError(
                f'survey must be "CFIS", argument provided was {survey}')

    # make the list of HDUList objects    
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


def download_PS1_template(ra, dec, size=2400, bands="grizy", output=None):
    """Downloads the relevant template image(s) at the input RA, Dec from the
    Panoramic Survey Telescope and Rapid Response System (Pan-STARRS) archive
    (specifically, the Pan-STARRS 1 (PS1) :math:`3\\pi` survey).
    
    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : int, optional
        Size of the cutout image in pixels (1 pix == 0.25" in PS1; default 
        2400)
    bands : str, optional
        Photometric band(s) of choice (default "grizy" --> g-, r-, i-, z-, and 
        y-band; options are "g", "r", "i", "z", "y" or any combination of them)
    output : str, optional
        Output filename(s) for the downloaded template(s) (default set 
        automatically)
    
    Returns
    -------
    list of `astropy.io.fits.hdu.hdulist.HDUList` objects
        List of astropy HDUList object(s) for the downloaded template image(s)
    """
    
    # verify the input declination dec
    if dec < -30: 
        raise ValueError("\nPan-STARRS 1 does not cover regions of the sky "+
                         "below Dec. = -30.0 deg")
    
    # verify the desired output
    if (output and (type(output) != str)): # if output list too short or long
        if len(bands) != len(output):
            print("\nPlease provide a number of output filenames to match "+
                  "the number of requested template bands. Exiting.")
            return

    # verify the input band(s)
    bands_upd = ""
    for bs in bands:
        if bs in "grizy":
            bands_upd += bs
        else: 
            print(f"\nPS1 does not contain the {bs} band, so this band "+
                  "will be ignored.")     
    if len(bands_upd) == 0:
        raise InvalidBandError("Valid bands for PS1 are grizy")
    
    # get the url(s)      
    url = query_PS1.geturl_PS1(ra, dec, size, bands=bands_upd, fmt="fits")
    
    # download the template and tell the user
    # 0.258"/pix = PS1 resolution
    tmps = __downloadtemplate(url, "PS1", 0.258, output) # download template 
    size_arcmin = size*0.258/60.0 # cutout size in arcmin  
    print("\nDownloaded square Pan-STARRS 1 cutout image(s) in the "+
          f"{bands_upd} band(s), centered on RA, Dec = {ra:.3f}, {dec:.3f} "+
          f"with sides of length {size_arcmin:.2f}'\n")
    return tmps


def download_DECaLS_template(ra, dec, size=512, pixscale=0.262, bands="grz", 
                             output=None):
    """Downloads the relevant Dark Energy Camera Legacy Survey (DECaLS) 
    template image(s) at the input RA, Dec.
    
    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : int, optional
        Size of the cutout image in pixels (1 pix == 0.262" in native DECam
        resolution, but pixel scale can be changed; default 512)
    pixscale : float, optional
        Pixel scale of the images in arcsec per pixel, (default 0.262"/pix, 
        which is the native DECam resolution)
    bands : str, optional
        Photometric band(s) of choice (default "grz" --> g-, r-, and z-band; 
        options are "g", "r", "z", or any combination of them)
    output : str, optional
        Output filename(s) for the downloaded template(s) (default set 
        automatically)
    
    Returns
    -------
    list of `astropy.io.fits.hdu.hdulist.HDUList` objects
        List of astropy HDUList object(s) for the downloaded template image(s)     
    """
    
    # verify the input band(s)
    bands_upd = ""
    for bs in bands:
        if bs in "grz":
            bands_upd += bs
        else: 
            print(f"\nDECaLS does not contain the {bs} band, so this band "+
                  "will be ignored.")    
    if len(bands_upd) == 0:
        raise InvalidBandError("Valid bands for DECaLS are grz")
    
    # verify input size
    if size > 512: # if requested image size too big
        print("\nThe maximum image size is 512 pix. The output image "+
              "will have these dimensions.")
        size = 512
        
    # verify the desired output
    if (output and (type(output) != str)): # if output list too short or long
        if len(bands_upd) != len(output):
            print("\nPlease provide a list of output filenames to match "+
                  "the number of valid requested template filters. Exiting.")
            return
    if (output and (type(output) == str) and (len(bands_upd)>1)): 
        # if only one output string is given and multiple filters are requested
        print("\nPlease provide a list of output filenames to match "+
              "the number of valid requested template filters. Exiting.")
        return
    
    # get the url(s)
    url = query_DECaLS.geturl_DECaLS(ra, dec, size, pixscale, bands=bands_upd)
    
    # download the template and tell the user
    tmps = __downloadtemplate(url,"DECaLS", 
                              pixscale, output) # download template 
    size_arcmin  = size*pixscale/60.0
    print("\nDownloaded square DECaLS cutout image(s) in the "+
          f"{bands_upd} band(s), centered on RA, Dec = {ra:.3f}, {dec:.3f} "+
          f"with sides of length {size_arcmin:.2f}'\n")
    return tmps


def download_CFIS_template(ra, dec, auth_user, auth_pass, 
                           size=1600, bands="ur"):
    """Downloads the relevant Canada France Imaging Survey (CFIS) template 
    image(s) at the input RA, Dec. Requires a Canadian Astronomy Data Centre 
    (CADC) username and password.

    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    auth_user, auth_pass : str
        CADC username and password
    size : float, optional
        Size of the cutout image in pixels (1 pix == 0.185" in CFIS; default
        1600)
    bands : str, optional
        Photometric band(s) of choice (default "ur" --> u- and r-band; 
        options are "u", "r", "ur")
    
    Returns
    -------
    list of `astropy.io.fits.hdu.hdulist.HDUList` objects
        List of astropy HDUList object(s) for the downloaded template image(s)  
    """

    # verify the input band(s)
    bands_upd = ""
    for bs in bands:
        if bs in "grz":
            bands_upd += bs
        else: 
            print(f"\nDECaLS does not contain the {bs} band, so this band "+
                  "will be ignored.")     
    if len(bands_upd) == 0:
        raise InvalidBandError("Valid bands for CFIS are ur")
    
    # get the url(s)
    url = query_CFIS.geturl_CFIS(ra, dec, size, bands=bands_upd)
    if len(url) == 0: # if no templates found, exit
        print("\nNo CFIS images were found at the given coordinates. Exiting.")
        return
    
    # download the template and tell the user
    # 0.185"/pix = CFIS resolution
    tmps = __downloadtemplate_auth(url, "CFIS", 0.185, auth_user, auth_pass)
    size_arcmin = size*0.185/60.0 # cutout size in arcmin
    print("\nDownloaded square CFIS cutout image(s) in the "+
          f"{bands_upd} band(s), centered on RA, Dec = {ra:.3f}, {dec:.3f} "+
          f"with sides of length {size_arcmin:.2f}'\n")
    return tmps


def download_2MASS_template(ra, dec, size=150, bands="A"):
    """Downloads the relevant 2MASS template image(s) at the input RA, Dec.
    
    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : float, optional
        Size of the cutout image in pixels (1 pix == 4.0" in 2MASS; default
        150)
    bands : str, optional
        Photometric band(s) of choice (default "A" --> all; options are "A",
        "J", "H", "K" - see notes for details)
    
    Returns
    -------
    list of `astropy.io.fits.hdu.hdulist.HDUList` objects
        List of astropy HDUList object(s) for the downloaded template image(s) 

    Notes
    -----
    Options for bands are "A" -->  all, **OR** "J" for only J-band, **OR** "H" 
    for only H-band, **OR** "K" for only K-band. Combinations such as e.g. "JH" 
    are not allowed.
    """

    # verify the input band(s)
    if not(bands in ["A", "J", "H", "K"]):
        raise InvalidBandError(
                'Valid bands for 2MASS are "A" (all) OR "J" OR "H" OR "K", '+
                f'argument supplied was {bands}')
    
    # get the url
    url = query_2MASS.geturl_2MASS(ra, dec, size, bands=bands)
        
    # download the template and tell the user
    # 4.0"/pix = 2MASS resolution
    tmps = __downloadtemplate(url, "2MASS", 4.0) # download template 
    size_arcmin = size*4.0/60.0 # cutout size in arcmin
    if bands == "A": bands_toprint = "JHK" # for pretty printing
    else: bands_toprint = bands
    print(f"\nDownloaded 2MASS image(s) in the {bands_toprint} band(s), "+
          f"centered on RA, Dec = {ra:.3f}, {dec:.3f}, "
          f"with sides of length {size_arcmin:.2f}'\n")    
    return tmps


def get_templates(images, survey="PS1", outputs=None):
    """For a directory full of images, downloads a corresponding template from 
    the requested survey, if possible. Currently only implemented for PS1 and
    CFIS. 
    
    Arguments
    ---------
    images : str or array_like
        Directory containing image file(s) for which to download a template
        **OR** a list of individual filenames
    survey : {'PS1', 'CFIS'}, optional
        Survey of interest (default 'PS1')
    output : str, optional
        Output filename(s) for the downloaded template(s) (default set 
        automatically)
    """
    
    # get the image names
    if type(images) == str:
        if images[-1] == "/":
            images = images[:-1]
        images = [images+"/"+f for f in os.listdir(images)]
    images = [i for i in images.copy() if not("template" in i)]
    
    # set output names
    if outputs == None:
        outputs = [i.replace(".fits", "_templates.fits") for i in images]       
    if type(outputs) == str:
        outputs = [outputs]
    
    # dictionary of survey : function to use 
    fdict = {"PS1":download_PS1_template,
             "CFIS":download_CFIS_template}
    
    # loop over every image 
    for i in range(len(images)):
        # get size, central ra and dec, and filter 
        data, hdr = fits.getdata(images[i]), fits.getheader(images[i])
        w = wcs.WCS(hdr)
        ra, dec = w.all_pix2world(data.shape[1]/2.0, data.shape[0]/2.0, 1)
        size = np.max([data.shape[0], data.shape[1]])
        try:
            filt = hdr["FILTER"][0]
        except KeyError:
            print(f"\nCould not find FILTER keyword in header of {images[i]}"+
                  "\nProceeding to next image")
            continue
        
        # download the template 
        try:
            _ = fdict[survey](ra, dec, size, filt, outputs[i]) 
        except: # except any errors
            e = sys.exc_info() 
            print("\nSome error occurred while trying to download a "+
                  f"template for {images[i]}:\n{str(e[0])}\n{str(e[1])}"+
                  "\nProceeding to next image")
            continue 
           
