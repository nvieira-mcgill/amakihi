#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:38:44 2020
@author: Nicholas Vieira
@plotting.py
"""

# misc
#import os
#import sys
#from subprocess import run, PIPE, CalledProcessError
import numpy as np
#import re

# scipy
#from scipy.ndimage import zoom, binary_dilation, gaussian_filter

# astropy
#from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
#import astropy.units as u 
from astropy.stats import sigma_clipped_stats#, SigmaClip, 
                           #gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm)
#from astropy.coordinates import SkyCoord
#from astropy.convolution import convolve_fft, Gaussian2DKernel, Moffat2DKernel
#from astropy.table import Table, Column
#from photutils import Background2D, MedianBackground
#from photutils import make_source_mask, detect_sources, source_properties

## for speedy FFTs
#import pyfftw
#import pyfftw.interfaces.numpy_fft as fft # for speedy FFTs
#pyfftw.interfaces.cache.enable()

# amakihi function for cropping by WCS
#from crop import crop_WCS

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
### background ################################################################

def __plot_bkg(im_header, bkg_img_masked, scale_bkg, output_bkg):
    """Plot the background image. 
    
    Arguments
    ---------
    im_header : FIX THIS
        FIX THIS
    bkg_img_masked : array_like
        2D array representing the background image 
    scale_bkg : {"linear", "log", "asinh"}
        Scale to apply to the plot    
    """

    # verify scale_bkg
    if not(scale_bkg in ("linear", "log", "asinh")):
        raise ValueError('scale_bkg must be one of ("linear", "log", '+
                         f'"asinh"), but argument supplied was {scale_bkg}')
    
    # set figure dimensions
    plt.figure(figsize=(14,13))
        
    # show WCS     
    w = wcs.WCS(im_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    # plot the background image with desired scaling
    if scale_bkg == "linear": # linear scale
        plt.imshow(bkg_img_masked, cmap='bone', aspect=1, 
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        
    elif scale_bkg == "log": # log scale 
        bkg_img_log = np.log10(bkg_img_masked)
        lognorm = simple_norm(bkg_img_log, "log", percent=99.0)
        plt.imshow(bkg_img_log, cmap='bone', aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        
    elif scale_bkg == "asinh":  # asinh scale
        bkg_img_asinh = np.arcsinh(bkg_img_masked)
        asinhnorm = simple_norm(bkg_img_asinh, "asinh")
        plt.imshow(bkg_img_asinh, cmap="bone", aspect=1, 
                   norm=asinhnorm, interpolation="nearest", origin="lower")
        cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)

    cb.ax.tick_params(which='major', labelsize=15)            
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    plt.title("image background", fontsize=15)
    plt.savefig(output_bkg, bbox_inches="tight")
    plt.close()


def __plot_bkgsubbed(im_header, bkgsub_img_masked, scale_bkgsubbed, 
                     output_bkgsubbed):
    """Plot the background-SUBTRACTED image.

    Arguments
    ---------
    im_header : astropy.io.fits.header.Header
        Image fits header
    bkgsub_img_masked : array_like
        2D array representing the background image 
    scale_bkgsubbed : {"linear", "log", "asinh"}
        Scale to apply to the plot    
    output_bkgsubbed : str
        Name for output figure
    """
    
    # verify scale_bkgsubbed
    if not(scale_bkgsubbed in ("linear", "log", "asinh")):
        raise ValueError('scale_bkgsubbed must be one of ("linear", "log", '+
                         '"asinh"), but argument supplied was '+
                         f'{scale_bkgsubbed}')

    # set figure dimensions
    plt.figure(figsize=(14,13))

    # show WCS     
    w = wcs.WCS(im_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)

    # plot with desired scaling     
    if scale_bkgsubbed == "linear": # linear scale
        plt.imshow(bkgsub_img_masked, cmap='bone', aspect=1, 
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        
    elif scale_bkgsubbed == "log": # log scale
        bkgsub_img_masked_log = np.log10(bkgsub_img_masked)
        lognorm = simple_norm(bkgsub_img_masked_log, "log", percent=99.0)
        plt.imshow(bkgsub_img_masked_log, cmap='bone', aspect=1, 
                   norm=lognorm, interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        
    elif scale_bkgsubbed == "asinh":  # asinh scale
        bkgsub_img_masked_asinh = np.arcsinh(bkgsub_img_masked)
        asinhnorm = simple_norm(bkgsub_img_masked_asinh, "asinh")
        plt.imshow(bkgsub_img_masked_asinh, cmap="bone", aspect=1, 
                   norm=asinhnorm, interpolation="nearest", origin="lower")
        cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)

    cb.ax.tick_params(which='major', labelsize=15)            
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    plt.title("background-subtracted image", fontsize=15)
    plt.savefig(output_bkgsubbed, bbox_inches="tight")
    plt.close()   
    
###############################################################################
### imalign ###################################################################

def __plot_sources(source_data, template_data, source_hdr, template_hdr, 
                   source_list, template_list, scale, color, output):    
    """Given some source image and template image, plot the locations of cross-
    matched sources in each, side-by-side for comparison.
    
    Arguments
    ---------
    source_data, template_data : array_like
        Source image and template image data
    source_hdr, template_hdr : astropy.io.fits.header.Header
        Source fits header and template fits header
    source_list, template_list : array_like
        Lists of source pixel coords in the source image and template_image
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plot  
    color : str
        Color for the circle denoting the positions of sources
    output : str
        Name for output figure        
    """

    # verify scale
    if not(scale in ("linear", "log", "asinh")):
        raise ValueError('scale must be one of ("linear", "log", '+
                         f'"asinh"), but argument supplied was {scale}')

    # get WCS objects    
    wsci = wcs.WCS(source_hdr)
    wtmp = wcs.WCS(template_hdr)
    
    # get miscellaneous coords
    xsci = [s[0] for s in source_list]
    ysci = [s[1] for s in source_list]
    xtmp = [t[0] for t in template_list]
    ytmp = [t[1] for t in template_list]
    rasci, decsci = wsci.all_pix2world(xsci, ysci, 1)
    ratmp, dectmp = wtmp.all_pix2world(xtmp, ytmp, 1)

    ## set figure dimensions
    fig = plt.figure(figsize=(19,11))
    
    ## science image
    ax = fig.add_subplot(121, projection=wsci)
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticks_position('l') # Decs on left
    ax.coords["dec"].set_ticklabel_position('l') 
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)       
    # plot with desired scaling
    if scale == "linear": # linear scale
        mean, median, std = sigma_clipped_stats(source_data)
        ax.imshow(source_data, cmap='bone', vmin=mean-5*std, vmax=mean+9*std,
                  aspect=1, interpolation='nearest', origin='lower')      
    elif scale == "log": # log scale
        source_data_log = np.log10(source_data)
        lognorm = simple_norm(source_data_log, "log", percent=99.0)
        ax.imshow(source_data_log, cmap='bone', aspect=1, norm=lognorm,
                  interpolation='nearest', origin='lower')      
    elif scale == "asinh":  # asinh scale
        source_data_asinh = np.arcsinh(source_data)
        asinhnorm = simple_norm(source_data_asinh, "asinh")
        ax.imshow(source_data_asinh, cmap="bone", aspect=1, 
                  norm=asinhnorm, interpolation="nearest", origin="lower")
    # sources in the science image  
    for i in range(len(rasci)):
        # when using wcs, looks a bit off...
        #circ = ptc.Circle((rasci[i],decsci[i]), radius=4.0/3600.0, 
        #                  transform=ax.get_transform('icrs'), fill=False,
        #                  ec=color, lw=2, ls="-") 
        #ax.add_patch(circ)
        circ = ptc.Circle((xsci[i], ysci[i]), radius=25.0, fill=False,
                          ec=color, lw=2, ls="-") 
        ax.add_patch(circ)

    ## template image
    ax2 = fig.add_subplot(122, projection=wtmp)
    ax2.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax2.coords["dec"].set_ticks_position('r') # Decs on right
    ax2.coords["dec"].set_ticklabel_position('r') 
    ax2.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)     
    # plot with desired scaling
    if scale == "linear": # linear scale
        mean, median, std = sigma_clipped_stats(template_data)
        ax2.imshow(template_data, cmap='bone', 
                   vmin=mean-5*std, vmax=mean+9*std,
                   aspect=1, interpolation='nearest', origin='lower')       
    elif scale == "log": # log scale
        template_data_log = np.log10(template_data)
        lognorm = simple_norm(template_data_log, "log", percent=99.0)
        ax2.imshow(template_data_log, cmap='bone', aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')       
    elif scale == "asinh":  # asinh scale
        template_data_asinh = np.arcsinh(template_data)
        asinhnorm = simple_norm(template_data_asinh, "asinh")
        ax2.imshow(template_data_asinh, cmap="bone", aspect=1, 
                   norm=asinhnorm, interpolation="nearest", origin="lower")
    # sources in the template image
    for i in range(len(ratmp)):
        # when using wcs, looks a bit off...
        #circ = ptc.Circle((ratmp[i],dectmp[i]), radius=4.0/3600.0, 
        #                  transform=ax2.get_transform('icrs'), fill=False,
        #                  ec=color, lw=2, ls="-") 
        #ax2.add_patch(circ)
        circ = ptc.Circle((xtmp[i], ytmp[i]), radius=25.0, fill=False,
                          ec=color, lw=2, ls="-") 
        ax2.add_patch(circ)

    ## final pretty plotting, writing
    ax.set_xlabel("RA (J2000)", fontsize=16)
    ax.set_ylabel("Dec (J2000)", fontsize=16)
    fig.subplots_adjust(wspace=0.07)
    fig.suptitle(r"$\mathbf{\mathtt{astroalign}}$"+" input sources", 
                 fontsize=15, y=0.93)
    fig.savefig(output, bbox_inches="tight")
    plt.close()


def __plot_align(template_hdr, img_aligned, mask, scale, output):
    """Plot the newly-aligned source image.
    
    Arguments
    ---------
    template_hdr : astropy.io.fits.header.Header
        Template fits header
    img_aligned : array_like
        Aligned source image data
    mask : array_like
        Mask data
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plot  
    output : str
        Name for output figure 
    """

    # verify scale
    if not(scale in ("linear", "log", "asinh")):
        raise ValueError('scale must be one of ("linear", "log", '+
                         f'"asinh"), but argument supplied was {scale}')
    
    # set figure dimensions    
    plt.figure(figsize=(14,13))
    
    # show WCS      
    w = wcs.WCS(template_hdr)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    # images
    img_aligned_masked = np.ma.masked_where(mask, img_aligned)
    img_aligned = np.ma.filled(img_aligned_masked, 0)

    # plot with desired scaling          
    if scale == "linear": # linear scale
        mean, median, std = sigma_clipped_stats(img_aligned)
        plt.imshow(img_aligned, cmap='bone', aspect=1, 
                   vmin=mean-5*std, vmax=mean+9*std,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        
    elif scale == "log": # log scale
        img_aligned_log = np.log10(img_aligned)
        lognorm = simple_norm(img_aligned_log, "log", percent=99.0)
        plt.imshow(img_aligned_log, cmap='bone', aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        
    elif scale == "asinh":  # asinh scale
        img_aligned_asinh = np.arcsinh(img_aligned)
        asinhnorm = simple_norm(img_aligned_asinh, "asinh")
        plt.imshow(img_aligned_asinh, cmap="bone", aspect=1, 
                   norm=asinhnorm, interpolation="nearest", origin="lower")
        cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)

    cb.ax.tick_params(which='major', labelsize=15)            
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    plt.title("registered image via "+r"$\mathbf{\mathtt{astroalign}}$", 
              fontsize=15)
    plt.savefig(output, bbox_inches="tight")
    plt.close()

