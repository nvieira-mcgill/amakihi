#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:24:56 2020
@author: Nicholas Vieira
@crop.py

This module enables cropping (and optionally writing) fits images by either WCS 
or pixel coords. 

**TO-DO:**

- Proper warnings in `crop_frac()`
- Run lots of tests in general

"""

# misc
import numpy as np
#import re

# astropy
from astropy.io import fits
from astropy import wcs

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
#### CROPPING #################################################################

def __get_crop(fits_file, frac_hori=[0,1], frac_vert=[0,1]):
    """Crop a .fits file's image according to some horizontal and vertical 
    fraction of interest. 
    
    Arguments
    ---------
    fits_file : str
        Filename for a single fits file
    frac_hori, frac_vert : array_like, optional
        Horizontal and vertical fractions of the image to crop (default [0, 1] 
        for both, in which case no cropping is performed) 

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) with the image cropped and header partially 
        updated (remainder of updating occurs in other functions)
    
    Examples
    --------
    The following would crop to select the bottom right corner of the image:
    
    >>> __get_crop("foo.fits", [0.5, 1], [0.0, 0.5])

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
    """Crop an image based on some central WCS coordinates of interest. 
    
    Arguments
    ---------
    source_file : str
        Filename for a single fits file
    ra, dec : float
        Right Ascension, Declination to center on, in decimal degrees
    size : float
        Size of the box (in pixels) to crop
    mode : {"truncate", "extend"}, optional
        Mode to apply if the requested crop lies outside of the image bounds
        (default "truncate", see notes for details)
    write : bool, optional
        Whether to write the output to a .fits file (default True)
    output : str, optional
        Name for output fits file (default 
        `source_file.replace(".fits", "_crop.fits")`

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) with the image cropped and header partially 
        updated
    
    Notes
    -----
    Crops the image of the fits file to a box of size `pixels` pix centered on
    RA and Dec `ra, dec`. If given box extends beyond bounds of image, there 
    are two possibilities:
    
    - `mode = "truncate"` --> Cropped image will be rectangular rather than 
      square, and truncated at the bounds
    - `mode = "extend"` --> Re-center the crop on a new set of coords such that 
      the output cropped image is still a square

    """
    
    hdr = fits.getheader(source_file)
    img = fits.getdata(source_file)
    y_size, x_size = img.shape # total image dims in pix 
    w = wcs.WCS(hdr)

    try:
        pix_scale = hdr["PIXSCAL1"] # scale of image in arcsec per pix
    except KeyError:
        #topfile = re.sub(".*/", "", source_file)
        pix_scale = float(input('\nPixel scale header PIXSCAL1 not found '+
                                f'for {source_file}. Please input a scale in '+
                                'arcseconds per pixel (note: '+
                                'PS1 = 0.258"/pix, '+
                                'DECaLS = 0.262"/pix, '+
                                'CFIS = 0.185"/pix, '+
                                '2MASS = 4.0"/pix)'+
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
    """Crop a .fits file's image according to some horizontal and vertical 
    fraction of interest, and optionally write it.
    
    Arguments
    ---------
    fits_file : str
        Filename for a single fits file
    frac_hori, frac_vert : array_like, optional
        Horizontal and vertical fractions of the image to crop (default [0, 1] 
        for both, in which case no cropping is performed) 
    write : bool, optional
        Whether to write the output to a .fits file (default True)
    output : str, optional
        Name for output fits file (default 
        `source_file.replace(".fits", "_crop.fits")`

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) with the image cropped and header partially 
        updated
    
    Examples
    --------
    The following would crop to select the bottom right corner of the image and
    write it to a new file "bar.fits":
    
    >>> crop_frac("foo.fits", [0.5, 1], [0, 0.5], write=True, output="bar.fits")

    """
    
    cropped_hdu = __get_crop(source_file, frac_hori, frac_vert)
    if write: # if we want to write the cropped .fits file 
        if not(output): # if no output name given, set default
            output = source_file.replace(".fits", "_crop.fits")
        cropped_hdu.writeto(output, overwrite=True, output_verify="ignore")
    
    return cropped_hdu


def crop_half(source_file, ra, dec, write=True, output=None):
    """Given some Right Ascension and Declination, find the half of the image
    which contains these coordinates, crop to select that half, and optionally
    write it.

    Arguments
    ---------
    fits_file : str
        Filename for a single fits file
    ra, dec : float
        Right Ascension, Declination of interest
    write : bool, optional
        Whether to write the output to a .fits file (default True)
    output : str, optional
        Name for output fits file (default 
        `source_file.replace(".fits", "_crop.fits")`

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) with the image cropped and header partially 
        updated
    
    Notes
    -----
    For a given, RA, Dec, finds the half of the image in which these coords
    are found. This can be the top half, bottom half, or a box centered on the 
    center of the overall image. Then crops the image. Best for images with an 
    aspect ratio of around 2:1 (e.g. MegaCam CCDs) such that the output crop is
    approximately square.
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


def crop_octant(source_file, ra, dec, write=True, output=None):
    """Given some Right Ascension and Declination, find the octant of the image
    which contains these coordinates, crop to select that octant, and 
    optionally write it.
    
    Arguments
    ---------
    fits_file : str
        Filename for a single fits file
    ra, dec : float
        Right Ascension, Declination of interest
    write : bool, optional
        Whether to write the output to a .fits file (default True)
    output : str, optional
        Name for output fits file (default 
        `source_file.replace(".fits", "_crop.fits")`

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) with the image cropped and header partially 
        updated
    
    Notes
    -----
    For a given, RA, Dec, finds the octant-sized ~square of the image in which 
    these coords are found. Then crops the image to this octant. Best for 
    images with an aspect ratio of around 2:1 (e.g. MegaCam CCDs) such that 
    the output crop is approximately square.
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
