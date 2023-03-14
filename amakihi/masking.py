#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Fri Dec 25 23:49:21 2020
.. @author: Nicholas Vieira
.. @masking.py

Constructing masks. 

.. **TO-DO:**

.. - Re-think saturated object size limits (can miss cosmic rays? is this OK?)

"""

# misc
import numpy as np

# scipy
from scipy.ndimage import binary_dilation, gaussian_filter

# astropy
from astropy.io import fits
from astropy import wcs
import astropy.units as u 
from astropy.stats import SigmaClip
from astropy.coordinates import SkyCoord
from astropy.table import Table
from photutils import Background2D, MedianBackground
from photutils import make_source_mask, detect_sources#, source_properties
from photutils.segmentation import SourceCatalog

# amakihi
from .plotting import __plot_mask

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
### USED EVERYWHERE ###########################################################

from photutils.segmentation.catalog import DEFAULT_COLUMNS

REQ_COLUMNS = DEFAULT_COLUMNS + ["elongation"]


###############################################################################
#### MASK BUILDING ############################################################
    
def box_mask(image_file, pixx, pixy, mask_file=None, plot=False, 
             write=True, output=None):
    """Create a simple box-shaped mask delimited by input pixel coordinates, 
    optionally combined with an existing mask. 
    
    Arguments
    ---------
    image_file : str
        Image (fits) filename
    pixx, pixy : tuple
        x-pixel pair and y-pixel pair denoting the bounds of the box to build
    mask_file : str, optional
        Existing mask image (fits) filename to combine existing mask with new
        boxmask (default None)
    plot : bool, optional
        Whether to plot the mask (default False)
    write : bool, optional
        Whether to write the new mask to a fits file
    output : str, optional
        Name for output fits file (default 
        `image_file.replace(".fits", "_boxmask.fits")`)
        
    Returns
    -------
    astropy.io.fits.PrimaryHDU
        New HDU (image + header) for the mask (**not** the masked data)
    """
    # get data, header
    data = fits.getdata(image_file)
    hdr = fits.getheader(image_file)
    
    # build the mask
    newmask = np.zeros(data.shape)
    newmask[pixy[0]:pixy[1], pixx[0]:pixx[1]] = 1.0
    newmask = newmask.astype(bool)
    
    # combine with another mask, if desired
    if mask_file: # combine with another mask 
        mask = fits.getdata(mask_file)
        newmask = np.logical_or(mask, newmask)
    
    # build the HDU
    mask_hdu = fits.PrimaryHDU(data=newmask.astype(int), header=hdr)
    
    # plot, if desired
    if plot:
        boxmask_plot = image_file.replace(".fits","_boxmask.png")
        title = "boxmask"
        __plot_mask(hdr=hdr, newmask=newmask, title=title, output=boxmask_plot)
    
    # write, if desired
    if write:
        if not(output):
            output = image_file.replace(".fits", "_boxmask.fits")
            
        mask_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return mask_hdu


def saturation_mask(image_file, mask_file=None, 
                    sat_ADU=40000, sat_area_min=500, 
                    ra_safe=None, dec_safe=None, rad_safe=None, 
                    dilation_its=5, blur_sigma=2.0, 
                    plot=True, 
                    write=True, output=None):
    """Produce a mask of all saturated sources in an image, optionally combined
    with an existing mask.
    
    Arguments
    ---------
    image_file : str
        Image (fits) filename
    mask_file : str, optional
        Existing mask image (fits) filename to combine existing mask with new
        saturation mask (default None)
    sat_ADU : float, optional
        ADU value above which a pixel is considered saturated (default 40000, 
        which is a bit below the limit for MegaCam)
    sat_area_min : float, optional
        *Minimum* pixel area for a source to be considered a saturated source
        (default 500)
    ra_safe, dec_safe : float, optional
        Right Ascension and Declination (in decimal degrees) at the centre of a 
        "safe zone" in which sources will not be masked (default None), see
        `rad_safe`
    rad_safe : float, optional
        Radius (in arcsec) of the "safe zone", a circle centered on 
        (`ra_safe`, `dec_safe`) (default None)
    dilation_its : int, optional
        Number of iterations of binary dilation to apply to the mask (see 
        notes for details; default 5; can be set to 0 turn this off)
    blur_sigma : float, optional
        Sigma of the Gaussian filter to apply to the mask to blur it (default 
        2.0; can be set to 0 to turn this off)
    plot : bool, optional
        Whether to plot the new saturation mask (default True)
    write : bool, optional
        Whether to write the new mask to a fits file (default True)
    output : str, optional
        Name for output fits file (default 
        `image_file.replace(".fits", "_satmask.fits")`)

    Returns
    -------
    tbl : astropy.table.table.Table
        A table containing properties of the sources which were flagged as 
        saturated
    mask_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) for the mask (**not** the masked data)
        
    Notes
    -----
    Uses image segmentation to find all sources in the image. Then, looks for 
    sources which have a maximum flux above the saturation ADU and above the 
    minimum saturation area and creates a mask of these sources. 

    Binary dilation is applied to to "dilate" (i.e., enlarge) features in the 
    mask, and Gaussian blurring is applied, to smooth out the mask and ensure
    that saturated sources are completely masked. Binary dilation is very 
    helpful for catching diffraction spikes which are initially not completely
    masked. **Note** however that both binary dilation and Gaussian blurring
    can be disabled by setting `dilation_its = 0` and `blur_sigma = 0`, 
    respectively.
    
    If a "safe zone" is supplied, any sources within the safe zone will be 
    labelled as NON-saturated. This is useful if you know the coordinates of 
    some galaxy/nebulosity in your image which should not be masked, as it is 
    sometimes difficult to distinguish between a saturated source and a galaxy.
    
    If an existing mask is supplied, the output mask will be a combination of 
    the previous mask and saturation mask.
    
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
            bp_mask = fits.getdata(mask_file).astype(bool)
            source_mask = make_source_mask(data, nsigma=3, npixels=5, 
                                       dilate_size=15, mask=bp_mask)
            # combine the bad pixel mask and source mask 
            rough_mask = np.logical_or(bp_mask,source_mask)
        else: 
            source_mask = make_source_mask(data, nsigma=3, npixels=5, 
                                       dilate_size=15)
            rough_mask = source_mask
        
        # estimate the background standard deviation
        try:
            sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
        except TypeError: # in old astropy, "maxiters" was "iters"
            sigma_clip = SigmaClip(sigma=3, iters=5)
        
        bkg = Background2D(data, (10,10), filter_size=(5,5), 
                           sigma_clip=sigma_clip, 
                           bkg_estimator=MedianBackground(), 
                           mask=rough_mask)
        bkg_rms = bkg.background_rms

    threshold = 3.0*bkg_rms # threshold for proper image segmentation 
    
    ## get the segmented image and source properties
    ## only detect sources composed of at least sat_area_min pixels
    segm = detect_sources(data, threshold, npixels=sat_area_min)
    labels = segm.labels 
    #cat = source_properties(data, segm) # photutils 0.8
    cat = SourceCatalog(data=data, segment_img=segm) # photutils >=1.1
    
    ## if any sources are found
    if len(cat) != 0:
        # catalogue of sources as a table  
        tbl = cat.to_table(columns=REQ_COLUMNS)   
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

        # mask any remaining pixels equal to 0, nan, or above the saturation 
        # ADU in the data
        newmask[data==0] = 1
        newmask[np.isnan(data)] = 1
        newmask[data>=sat_ADU] = 1
        
        # use binary dilation to fill holes, esp. near diffraction spikes
        if dilation_its < 0:
            raise ValueError("dilation_its must be 0 or a positive number, "+
                             f"but the provided argument was {dilation_its}")
        elif dilation_its == 0:
            print("Skipping binary dilation...")
        else:
            newmask = (binary_dilation(newmask, 
                                       iterations=dilation_its)).astype(float)
            
        # use Gaussian blurring to smooth out the mask
        if blur_sigma < 0:
            raise ValueError("blur_sigma must be 0 or a positive number, "+
                             f"but the provided argument was {blur_sigma}")
        elif blur_sigma == 0:
            print("Skipping Gaussian blurring...")
        else:
            newmask = gaussian_filter(newmask, sigma=blur_sigma, 
                                      mode="constant", cval=0.0)
        
        # anything which is above zero should be masked
        newmask[newmask > 0] = 1
      
    ## if no sources are found 
    else: 
        # empty table
        tbl = Table() 

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
   
    ## plot, if desired
    if plot: # plot, if desired
        satmask_plot = image_file.replace(".fits","_satmask.png")
        title = "saturation mask"
        __plot_mask(hdr=hdr, newmask=newmask, title=title, output=satmask_plot)
    
    ## write, if desired
    if write:
        if not(output):
            output = image_file.replace(".fits", "_satmask.fits")          
        mask_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return tbl, mask_hdu

