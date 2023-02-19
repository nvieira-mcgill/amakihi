#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Fri Dec 25 21:44:27 2020
.. @author: Nicholas Vieira
.. @imalign.py

Image alignment, A.K.A. image registration. In all, this is the trickiest part 
of image differencing. Poor alignment will result in all sorts of headaches 
later, and in particular, the appearance of "dipole"-like artifacts in the 
final difference image. 

**Important:** This module makes use of a slightly modified version of the 
`astroalign` software developed by Martin Beroiz and the TOROS Dev Team 
(https://github.com/quatrope/astroalign) in the form of my own script 
`astroalign_mod.py`. I claim absolutely no ownership of this software. All
modifications are described in that script.

"""

# misc
import sys
from subprocess import run, PIPE, CalledProcessError
import numpy as np
import re

# astropy
from astropy.io import fits
from astropy import wcs
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.table import Table
from photutils import detect_sources#, source_properties
from photutils.segmentation import SourceCatalog

# amakihi 
from .background import bkgstd
from .plotting import __plot_sources, __plot_align

# a modified version of astroalign which allows the user to set the sigma 
# threshold for source detection, which sometimes needs to be tweaked
from . import astroalign_mod as aa

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)


###############################################################################
### UTILITIES #################################################################

def __control_points(data, data_thresh, data_std, data_pixmin, data_mask,
                     im_type, eta, exclu, per, nsources):
    """
    Arguments
    ---------
    data
    data_thresh
    data_std
    data_pixmin
    data_mask
    im_type
    eta
    exclu
    per
    nsources
    
    Returns
    -------
    array_like
        List of newly-found control points
    """
    
    segm_data = detect_sources(data, 
                               data_thresh*data_std, 
                               npixels=data_pixmin,
                               mask=data_mask)          
    # use the segmentation image to get the source properties 
    #cat = source_properties(data, segm_data, mask=data_mask) # photutils 0.8
    cat = SourceCatalog(data=data, segment_img=segm_data, 
                        mask=data_mask) # photutils >=1.1
    try:
        tbl = cat.to_table()
    except ValueError:
        print(f"{im_type} image contains no sources. Exiting.")
        return        
    # restrict elongation  
    tbl = tbl[(tbl["elongation"] <= eta)]
    # remove sources at image edges
    xc = tbl["xcentroid"].value; yc = tbl["ycentroid"].value
    xedge_min = [min(x, data.shape[1]-x) for x in xc]
    yedge_min = [min(y, data.shape[0]-y) for y in yc]
    tbl["xedge_min"] = xedge_min
    tbl["yedge_min"] = yedge_min
    keep = [(s["xedge_min"]>exclu*data.shape[1]) and
            (s["yedge_min"]>exclu*data.shape[0]) for s in tbl]
    tbl = tbl[keep]        
    # pick at most <nsources> sources below the <per_upper> flux percentile
    tbl["sum/area"] = tbl["source_sum"].data/tbl["area"].data
    tbl.sort("sum/area") # sort by flux
    tbl.reverse() # biggest to smallest
    start = int((1-per)*len(tbl))
    end = min((len(tbl)-1), (start+nsources))
    tbl = tbl[start:end]   
    # get the list
    source_list = np.array([[tbl['xcentroid'].data[i], 
                             tbl['ycentroid'].data[i]] for i in 
        range(len(tbl['xcentroid'].data))])
    
    return source_list


def __control_points_astrometry():
    """
    """
    pass


def __remove_SIP(header):
    """Removes all SIP-related keywords from a fits header, in-place.
    
    Arguments
    ---------
    header : astropy.io.fits.header.Header
        A fits header    
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
    
    header["CTYPE1"] = header["CTYPE1"][:-4] # get rid of -SIP
    header["CTYPE2"] = header["CTYPE2"][:-4] # get rid of -SIP
    

def __WCS_transfer(header, template_header):
    """Transfer the WCS information from `template_header` to `header`.
    
    Arguments
    ---------
    header : astropy.io.fits.header.Header
        A fits header
    template_header : astropy.io.fits.header.Header
        Fits header for template image 
    
    Returns
    -------
    astropy.io.fits.header.Header
        Fits header with updated WCS information
    """

    mjdobs, dateobs = header["MJD-OBS"], header["DATE-OBS"] # store temporarily
    w = wcs.WCS(template_header)    
    # if no SIP transformations in header, need to update 
    if not("SIP" in template_header["CTYPE1"]) and ("SIP" in header["CTYPE1"]):
        __remove_SIP(header) # remove -SIP and remove related headers 
        # if old-convention headers (used in e.g. PS1), need to update 
        # not sure if this is helping 
        #if 'PC001001' in template_header: 
        #    hdr['PC001001'] = template_header['PC001001']
        #    hdr['PC001002'] = template_header['PC001002']
        #    hdr['PC002001'] = template_header['PC002001']
        #    hdr['PC002002'] = template_header['PC002002']
        #    del hdr["CD1_1"]
        #    del hdr["CD1_2"]
        #    del hdr["CD2_1"]
        #    del hdr["CD2_2"]   
    header.update((w.to_fits(relax=False))[0].header) # update 

    # build the final header
    header["MJD-OBS"] = mjdobs # get MJD-OBS of SCIENCE image
    header["DATE-OBS"] = dateobs # get DATE-OBS of SCIENCE image
    
    return header
    
###############################################################################
### ALIGNMENT WITH ASTROALIGN PACKAGE #########################################

## using image segmentation ###################################################

def image_align_imsegm(science_file, template_file, 
                       mask_file=None,
                       sciexclu=0.05, tmpexclu=0.05, sciper=0.95, tmpper=0.95, 
                       nsources=50,
                       thresh_sigma=3.0, pixelmin=10, etamax=2.0,
                       doublealign=False,
                       sep_max=2.0, ncrossmatch=8,
                       wcs_transfer=True,
                       plot_sources=False, 
                       plot_align=False,
                       scale="linear", circ_color="#fe01b1",
                       write=True, output_im=None, output_mask=None):
    """Align a science image to a template image, using image segmentation (via
    `photutils`) for source extraction and `astroalign` for alignment.
    
    Arguments
    ---------
    science_file, template_file :  str
        Science and template fits file names
    mask_file : str, optional
        Mask fits file name (default None)
    sciexclu, tmpexclu : float, optional
        Fraction of the image edges from which to exclude sources during 
        matching for the science and template images (default 0.05)
    sciper, tmpper : float, optional
        Upper flux percentile beyond which to exclude sources (default 0.95)
    nsources : int, optional
        Maximum number of sources to use in asterism-matching (default 50)
    thresh_sigma : float or array_like, optional
        Sigma threshold for source detection with image segmentation (default
        3.0; can be length-2 array to assign different values for science and
        template)
    pixelmin : float or array_like, optional
        *Minimum* pixel area of an isophote to be considered a source (default
        10; can be length-2 array to assign different values for science and
        template)
    etamax : float or array_like, optional
        *Maximum* allowed elongation for an isophote to be considered a source 
        (default 2.0; can be length-2 array to assign different values for 
        science and template)
    doublealign : bool, optional
        Whether to perform a second iteration of alignment for fine-tuning (see 
        notes for details; default False)
    sep_max : float, optional
        *Maximum* allowed on-sky separation (in arcsec) to cross-match a source 
        in the science image to the template during double alignment (default 
        2.0)
    ncrossmatch : int, optional
        Required number of cross-matched sources to proceed with double 
        alignment (default 8)
    wcs_transfer : bool, optional
        Whether to attempt to transfer WCS coordinates from the template to the 
        newly-aligned image (default True)
    plot_sources : bool, optional
        Whether to plot sources detected in the science and template images, 
        side-by-side, for visual inspection (default False)
    plot_align : bool, optional
        Whether to plot the final aligned science image (default False)
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plots (default "linear")
    circ_color : str, optional
        Color for the circles drawn around detected sources (default "#fe01b1 
        --> bright pink; only relevant if `plot_sources == True`)
    write : bool, optional
        Whether to write *both* outputs (the aligned image and mask image from
        the image registration footprint) (default True)
    output_im : str, optional
        Name for output aligned image fits file (default 
        `science_file.replace(".fits", "_align.fits")`)
    output_mask : str, optional
        Name for output mask fits file (default 
        `science_file.replace(".fits", "_align_mask.fits")`)        
        
    Returns
    -------
    align_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the aligned science image 
    mask_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the final mask       
        
    Notes
    -----
    **Uses a slightly modified version of astroalign.**

    This function does a lot. Step by step: 
        
    1. Uses image segmentation to find "control points" (sources) in the 
       science and template images
    2. Rejects sources very close to the image edges (limits set by user, 
       see `sciexclu` and `tmpexclu`)
    3. Rejects sources below some minimum isophotal area `pixelmin`
    4. Rejects sources with elongation above `etamax`
    5. Rejects sources above some upper flux percentile (set by user, see 
       `sciper` and `tmpper`)
    6. Retains at most `nsources` control points
    7. **Control points are passed to astroalign**, which finds invariants
       between the source and template control point sets and computes the 
       affine transformation which aligns the source image to the template 
       image, and then applies this transformation to the science image
    8. **Optional (if doublealign == True)**: With the newly aligned image 
       and WCS coordinates in both science and template images, constructs
       pairs of sources at the same coordinates, and computes another 
       transformation which respects these pairs to fine-tune the alignment
       
    
    **TO-DO:**
    
    - Allow naming of output plots
    - Fix: During transfer of WCS header from PS1 to CFHT, WCS becomes nonsense

    """
    #########
    ### setup
    #########
    ## load in data
    science = fits.getdata(science_file)
    template = fits.getdata(template_file)
    science_hdr = fits.getheader(science_file)
    template_hdr = fits.getheader(template_file)

    ## build up masks, apply them
    mask = np.logical_or(science==0, np.isnan(science)) # in sci image
    if mask_file: # if a mask is provided
        mask = np.logical_or(mask, fits.getdata(mask_file))
    science = np.ma.masked_where(mask, science)  
    
    template_mask = np.logical_or(template==0, np.isnan(template)) # in tmp
    template = np.ma.masked_where(template_mask, template)

    # check if input thresh_sigma is a list or single value 
    if not(type(thresh_sigma) in [float,int]):
        scithresh = thresh_sigma[0]
        tmpthresh = thresh_sigma[1]
    else:
        scithresh = tmpthresh = thresh_sigma
    # check if input pixelmin is list or single value 
    if not(type(pixelmin) in [float,int]):
        scipixmin = pixelmin[0]
        tmppixmin = pixelmin[1]
    else:
        scipixmin = tmppixmin = pixelmin
    # check if input etamax is list or single value 
    if not(type(etamax) in [float,int]):
        scieta = etamax[0]
        tmpeta = etamax[1]
    else:
        scieta = tmpeta = etamax

    # check input nsources
    if nsources > 50:
        print("The number of sources used for matching cannot exceed 50 --> "+
              "setting limit to 50")
        nsources = 50
    
    ###########################################################################
    ## get background standard deviation for science/template image (if needed)
    ###########################################################################
    try: # science 
        scistd = float(science_hdr['BKGSTD'])
    except KeyError:
        scistd = bkgstd(science, mask)
        if scistd == None:
            return

    try: # template
        tmpstd = float(template_hdr['BKGSTD']) 
    except KeyError:
        tmpstd = bkgstd(template, template_mask)
        if tmpstd == None:
            return
        
    #####################################################
    ## find control points in science and template images  
    #####################################################
    science_list, scitbl = __control_points(data=science, 
                                            data_thresh=scithresh, 
                                            data_std=scistd, 
                                            data_pixmin=scipixmin, 
                                            data_mask=mask, 
                                            im_type="science", 
                                            eta=scieta, exclu=sciexclu, 
                                            per=sciper, nsources=nsources)

    template_list, tmptbl = __control_points(data=template, 
                                             data_thresh=tmpthresh, 
                                             data_std=tmpstd, 
                                             data_pixmin=tmppixmin, 
                                             data_mask=template_mask,
                                             im_type="template", 
                                             eta=tmpeta, exclu=tmpexclu, 
                                             per=tmpper, nsources=nsources)

    ########################################################
    ## plot the sources attempting to be matched, if desired
    ########################################################
    if plot_sources:
        sourceplot = science_file.replace(".fits", 
                                         f"_alignment_sources_{scale}.png")
        __plot_sources(science_data=science, 
                       template_data=template, 
                       science_hdr=science_hdr, 
                       template_hdr=template_hdr, 
                       science_list=science_list, 
                       template_list=template_list, 
                       scale=scale, 
                       color=circ_color, 
                       output=sourceplot)      

    ####################
    ## align the images!
    ####################
    try: 
        print(f"\nAttempting to match {len(science_list)} sources in the "+
              f"science image to {len(template_list)} in the template")                      
        # find the transform using the control points
        tform, __ = aa.find_transform(science_list, template_list)
        # apply the transform
        img_aligned, footprint = aa.apply_transform(tform, science,
                                                    template,
                                                    propagate_mask=True)
        print("\nSUCCESS\n")
    except aa.MaxIterError: # if cannot match images, try flipping 
        print("Max iterations exceeded; flipping the image...")
        xsize = fits.getdata(science_file).shape[1]
        ysize = fits.getdata(science_file).shape[0]
        science_list = [[xsize,ysize]-coords for coords in 
                        science_list.copy()]
        science = np.flip(science, axis=0)
        science = np.flip(science, axis=1)
            
        try:
            tform, __ = aa.find_transform(science_list, template_list)
            img_aligned, footprint = aa.apply_transform(tform, science,
                                                        template,
                                                        propagate_mask=True)
            print("\nSUCCESS\n")
        except aa.MaxIterError: # still too many iterations 
            print("Max iterations exceeded while trying to find "+
                  "acceptable transformation. Exiting.")
            return
        
    except aa.TooFewStarsError: # not enough stars in source/template
        print("Reference stars in source/template image are less than "+
              "the minimum value (3). Exiting.")
        return
    
    except Exception: # any other exceptions
        e = sys.exc_info()
        print("\nWhile calling astroalign, some error other than "+
              "MaxIterError or TooFewStarsError was raised: "+
              f"\n{str(e[0])}\n{str(e[1])}")
        return
    
    
    ##########################################################
    ### align them again, this time cross-matching sources 1:1
    ########################################################## 
    # using the while loop here is bad practice, I think...
    while doublealign:
        print("\nAttempting a second iteration of alignment...")
        
        # build new mask
        mask = np.logical_or(img_aligned==0, np.isnan(img_aligned))
        mask = np.logical_or(mask, footprint)
        
        science_new = img_aligned.copy()
        science_new = np.ma.masked_where(mask, science_new)  
        segm_science = detect_sources(science_new, 
                                      scithresh*scistd, 
                                      npixels=scipixmin,
                                      mask=mask)          
        # use the segmentation image to get the source properties 
        cat_science = source_properties(science_new, segm_science, mask=mask)
        try:
            scitbl = cat_science.to_table()
        except ValueError:
            print("science image contains no sources. Exiting.")
            return        
        # restrict elongation  
        scitbl = scitbl[(scitbl["elongation"] <= scieta)]
        # remove sources in edges of image
        xc = scitbl["xcentroid"].value
        yc = scitbl["ycentroid"].value
        xedge_min = [min(x, science_new.shape[1]-x) for x in xc]
        yedge_min = [min(y, science_new.shape[0]-y) for y in yc]
        scitbl["xedge_min"] = xedge_min
        scitbl["yedge_min"] = yedge_min
        keep = [(s["xedge_min"]>sciexclu*science_new.shape[1]) and
            (s["yedge_min"]>sciexclu*science_new.shape[0]) for s in scitbl]
        scitbl = scitbl[keep]        
        # pick at most <nsources> sources below <per_upper>
        scitbl["sum/area"] = scitbl["source_sum"].data/scitbl["area"].data
        scitbl.sort("sum/area") # sort by flux
        scitbl.reverse() # biggest to smallest
        start = int((1-sciper)*len(scitbl))
        end = min((len(scitbl)-1), (start+nsources))
        scitbl = scitbl[start:end]   
        # get the list
        science_list = np.array([[scitbl['xcentroid'].data[i], 
                                 scitbl['ycentroid'].data[i]] for i in 
                                 range(len(scitbl['xcentroid'].data))])
        
        # get ra, dec for all of the sources to allow cross-matching 
        xsci = [s[0] for s in science_list]
        ysci = [s[1] for s in science_list]
        xtmp = [t[0] for t in template_list]
        ytmp = [t[1] for t in template_list]
        wtmp = wcs.WCS(template_hdr)
        rasci, decsci = wtmp.all_pix2world(xsci, ysci, 1)       
        ratmp, dectmp = wtmp.all_pix2world(xtmp, ytmp, 1) 
        scicoords = SkyCoord(rasci*u.deg, decsci*u.deg, frame="icrs")
        tmpcoords = SkyCoord(ratmp*u.deg, dectmp*u.deg, frame="icrs")            
        # cross-match
        idxs, idxt, d2d, d3d = tmpcoords.search_around_sky(scicoords,
                                                           sep_max*u.arcsec)
        scitbl_new = scitbl[idxs]; tmptbl_new = tmptbl[idxt]
        
        # if no cross-matching was possible...
        if len(scitbl_new) == 0:
            print('\nWas not able to cross-match any sources in the '+
                  'science and template images for separation '+
                  f'< {sep_max:.2f}". New solution not obtained.')
            doublealign = False
            break
        elif len(scitbl_new) < ncrossmatch: # or too few sources 
            # 8 is arbitrary atm
            print(f'\nWas only able to cross-match {len(scitbl_new):d} '+
                  f'< {ncrossmatch:d} sources in the science and template '+
                  f'images for separation < {sep_max:.2f}". '+
                  'New solution not obtained.')
            doublealign = False
            break
        
        # otherwise, keep going
        print(f'\nFound {len(scitbl_new)} sources in the science image '+
              'with a 1:1 match to a source in the template within '+
              f'< {sep_max:.2f}", with average separation '+
              f'{np.mean(d2d.value*3600):.2f}"')
        
        # new list of sources 
        science_list_new = np.array([[scitbl_new['xcentroid'].data[i], 
                scitbl_new['ycentroid'].data[i]] for i in range(
                len(scitbl_new['xcentroid'].data))])            
        template_list_new = np.array([[tmptbl_new['xcentroid'].data[i], 
                tmptbl_new['ycentroid'].data[i]] for i in range(
                len(tmptbl_new['xcentroid'].data))])

        # for bugtesting
        print(science_list_new[:5])
        print(template_list_new[:5])
        
        try: 
            print("\nAttempting to match...")                     
            # find the transform using the control points
            tform = aa.estimate_transform('affine', science_list_new, 
                                          template_list_new)
            # apply the transform
            img_aligned, footprint = aa.apply_transform(tform, science_new,
                                                        template,
                                                        propagate_mask=True)
            print("\nSUCCESS\n")
            break
        except aa.MaxIterError: # if cannot match images, try flipping 
            print("Max iterations exceeded. New solution not obtained.")
            break
            
        except aa.TooFewStarsError: # not enough stars in source/template
            print("Reference stars in source/template image are less "+
                  "than the minimum value (3). New solution not obtained.")
            break
        
        except Exception: # any other exceptions
            e = sys.exc_info()
            print("\nWhile calling astroalign, some error other than "+
                  "MaxIterError or TooFewStarsError was raised: "+
                  f"\n{str(e[0])}\n{str(e[1])}")         
            break

    #######################
    ### make the final mask
    #######################
    # mask 0s / nans AND mask outside the footprint of the image registration
    # (the mask should propagate via astroalign, but not sure if it does...)
    mask = np.logical_or(img_aligned==0, np.isnan(img_aligned))
    mask = np.logical_or(mask, footprint)
    
    ######################################
    ### plot the aligned image, if desired
    ######################################
    if plot_align: 
        alignplot = science_file.replace(".fits", 
                                    f"_astroalign_{scale}.png")
        __plot_align(template_hdr=template_hdr, 
                     img_aligned=img_aligned, 
                     mask=mask, 
                     scale=scale, 
                     output=alignplot)
    
    #####################################################
    ### finish editing the header, transfer WCS if needed
    #####################################################
    # set header for new aligned fits file 
    hdr = fits.getheader(science_file)
    # make a note that astroalign was successful
    hdr["IMREG"] = ("astroalign", "image registration software")  
    try: 
        hdr["PIXSCAL1"] = template_hdr["PIXSCAL1"] # pixscale of TEMPLATE
    except KeyError:
        pass
    # try to transfer the template WCS 
    if wcs_transfer: 
        hdr = __WCS_transfer(hdr, template_hdr)
    # final HDUs
    align_hdu = fits.PrimaryHDU(data=img_aligned, header=hdr)
    mask_hdu = fits.PrimaryHDU(data=mask.astype(int), header=hdr)
    
    #################################
    ### write (if desired) and return
    #################################
    if write: # if we want to write the aligned fits file and the mask 
        if not(output_im): # if no output name given, set default
            output_im = science_file.replace(".fits", "_align.fits")
        if not (output_mask): 
            output_mask = science_file.replace(".fits", "_align_mask.fits")
            
        align_hdu.writeto(output_im, overwrite=True, output_verify="ignore")
        mask_hdu.writeto(output_mask, overwrite=True, output_verify="ignore")
    
    return align_hdu, mask_hdu


## using astrometry.net #######################################################

def image_align_astrometry(science_file, template_file,
                           mask_file=None,
                           sciexclu=0.05, tmpexclu=0.05, 
                           sciper=0.95, tmpper=0.95,
                           nsources=50,
                           bkgsubbed=False,
                           astrom_sigma=8.0,
                           psf_sigma=5.0,
                           keep=False,
                           wcs_transfer=True,
                           plot_sources=False, 
                           plot_align=False,
                           scale="linear", circ_color="#fe01b1",
                           write=True, output_im=None, output_mask=None):
    """Align a science image to a template image, using `astrometry.net` for 
    source extraction and `astroalign` for alignment.

    Arguments
    ---------
    science_file, template_file :  str
        Science and template fits file names
    mask_file : str, optional
        Mask fits file name (default None)
    sciexclu, tmpexclu : float, optional
        Fraction of the image edges from which to exclude sources during 
        matching for the science and template images (default 0.05)
    sciper, tmpper : float, optional
        Upper flux percentile beyond which to exclude sources (default 0.95)
    nsources : int, optional
        Maximum number of sources to use in asterism-matching (default 50)
    bkgsubbed : bool or array_like, optional
        Whether the images have already been background-subtracted (default
        False; can be length-2 array to assign different bools for science 
        and template)
    astrom_sigma : float or array_like, optional
        Detection significance when using `image2xy` in `astrometry.net` to 
        find sources (default 8.0, can be length-2 array to assign different
        values for science and template)
    psf_sigma : float or array_like, optional
        Sigma of the approximate Gaussian PSF of the images (default 5.0; can 
        be length-2 array to assign different values for science and template)
    keep : bool, optional
        Whether to keep the source list files (`.xy.fits` files; default False)
    wcs_transfer : bool, optional
        Whether to attempt to transfer WCS coordinates from the template to the 
        newly-aligned image (default True)
    plot_sources : bool, optional
        Whether to plot sources detected in the science and template images, 
        side-by-side, for visual inspection (default False)
    plot_align : bool, optional
        Whether to plot the final aligned science image (default False)
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plots (default "linear")
    circ_color : str, optional
        Color for the circles drawn around detected sources (default "#fe01b1 
        --> bright pink; only relevant if `plot_sources == True`)
    write : bool, optional
        Whether to write *both* outputs (the aligned image and mask image from
        the image registration footprint) (default True)
    output_im : str, optional
        Name for output aligned image fits file (default 
        `science_file.replace(".fits", "_align.fits")`)
    output_mask : str, optional
        Name for output mask fits file (default 
        `science_file.replace(".fits", "_align_mask.fits")`)    

    Returns
    -------
    align_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the aligned science image 
    mask_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the final mask

    Notes
    -----
    **Uses a slightly modified version of astroalign.**
    
    This function does a lot. Step by step: 
        
    1. Uses `image2xy` of `astrometry.net` to find "control points" (sources) 
       in the science and template images 
    2. Retains at most `nsources` control points
    3. **Control points are passed to astroalign**, which finds invariants
       between the source and template control point sets and computes the 
       affine transformation which aligns the source image to the template 
       image, and then applies this transformation to the science image

    **Note:** `image2xy` has a hard time doing basic background subtraction 
    when an image has nans in it. Before trying alignment with an image which 
    contains nans, it is suggested to set all nans to 0 instead.       
    
    **TO-DO:**
    
    - Allow naming of output plots
    - Fix: During transfer of WCS header from PS1 to CFHT, WCS becomes nonsense

    """

    #########
    ### setup
    #########
    ## load in data
    science = fits.getdata(science_file)
    template = fits.getdata(template_file)
    science_hdr = fits.getheader(science_file)
    template_hdr = fits.getheader(template_file)

    ## build up masks, apply them
    mask = np.logical_or(science==0, np.isnan(science)) # mask zeros/nans in sci
    if mask_file: # if a mask is provided
        mask = np.logical_or(mask, fits.getdata(mask_file))
    science = np.ma.masked_where(mask, science)  
    
    template_mask = np.logical_or(template==0, np.isnan(template)) # in template
    template = np.ma.masked_where(template_mask, template)

    # check if input bkgsubbed bool is list or single value
    if not(type(bkgsubbed) == bool):
        science_bkgsub = bkgsubbed[0]
        tmp_bkgsub = bkgsubbed[1]
    else:
        science_bkgsub = tmp_bkgsub = bkgsubbed
    # check if input astrometry significance sigma is list or single value 
    if not(type(astrom_sigma) in [float,int]):
        science_sig = astrom_sigma[0]
        tmp_sig = astrom_sigma[1]
    else:
        science_sig = tmp_sig = astrom_sigma
    # check if input astrometry PSF sigma is list or single value 
    if not(type(psf_sigma) in [float,int]):
        science_psf = psf_sigma[0]
        tmp_psf = psf_sigma[1]
    else:
        science_psf = tmp_psf = psf_sigma
       
    # -O --> overwrite
    # -p --> source significance 
    # -w --> estimated PSF sigma 
    # -s 10 --> size of the median filter to apply to the image is 10x10
    # -m 10000 --> max object size for deblending is 10000 pix**2
    
    ######################################
    ## find control points in source image 
    ######################################
    options = f" -O -p {science_sig} -w {science_psf} -s 10 -m 10000"
    if science_bkgsub: options = f"{options} -b" # no need for subtraction
    run(f"image2xy {options} {science_file}", shell=True)    
    science_list_file = science_file.replace(".fits", ".xy.fits")
    science_list = Table.read(science_list_file)
    if len(science_list) == 0: # if no sources found 
        print("\nNo sources found with astrometry.net in the source "+
              "image, so image alignment cannot be obtained. Exiting.")
        return
    # pick at most <nsources> sources below the <per_upper> flux percentile
    science_list.sort('FLUX') # sort by flux
    science_list.reverse() # biggest to smallest
    start = int((1-sciper)*len(science_list))
    end = min((len(science_list)-1), (start+nsources))
    science_list = science_list[start:end]   
    science_list = np.array([[science_list['X'][i], 
                              science_list['Y'][i]] for i in 
                            range(len(science_list['X']))]) 
    # remove sources in leftmost/rightmost/topmost/bottommost edge of image
    science_list = [s for s in science_list.copy() if (
            (min(s[0], science.shape[1]-s[0])>sciexclu*science.shape[1]) and
            (min(s[1], science.shape[0]-s[1])>sciexclu*science.shape[0]))]
    
    ########################################    
    ## find control points in template image
    ########################################
    options = f" -O -p {tmp_sig} -w {tmp_psf} -s 10 -m 10000"
    if tmp_bkgsub: options = f"{options} -b" # no need for subtraction
    run(f"image2xy {options} {template_file}", shell=True)    
    template_list_file = template_file.replace(".fits", ".xy.fits")        
    template_list = Table.read(template_list_file)
    if len(template_list) == 0: # if no sources found 
        print("\nNo sources found with astrometry.net in the template "+
              "image, so image alignment cannot be obtained. Exiting.")
        return
    # pick at most <nsources> sources below the <per_upper> flux percentile
    template_list.sort('FLUX') # sort by flux 
    template_list.reverse() # biggest to smallest
    start = int((1-tmpper)*len(template_list))
    end = min((len(template_list)-1), (start+nsources))
    template_list = template_list[start:end]  
    template_list = np.array([[template_list['X'][i], 
                               template_list['Y'][i]] for i in 
                             range(len(template_list['X']))])
    # remove sources in leftmost/rightmost/topmost/bottommost edge of image
    template_list = [t for t in template_list.copy() if (
        (min(t[0], template.shape[1]-t[0])>tmpexclu*template.shape[1]) and
        (min(t[1], template.shape[0]-t[1])>tmpexclu*template.shape[0]))]

    if keep:
        print("\nKeeping the source list files for the science and "+
              "template images. They have been written to:")
        print(f"{science_list_file}\n{template_list_file}")
    else:
        run(f"rm {science_list_file}", shell=True) # not needed
        run(f"rm {template_list_file}", shell=True) 

    ########################################################
    ## plot the sources attempting to be matched, if desired
    ########################################################
    if plot_sources:
        sourceplot = science_file.replace(".fits", 
                                         f"_alignment_sources_{scale}.png")
        __plot_sources(science_data=science, 
                       template_data=template, 
                       science_hdr=science_hdr, 
                       template_hdr=template_hdr, 
                       science_list=science_list, 
                       template_list=template_list, 
                       scale=scale, 
                       color=circ_color, 
                       output=sourceplot)    
        
    ###################
    ## align the images
    ###################
    try: 
        #print(f"image_align_astrometry(): science_list = {science_list}")
        #print(f"image_align_astrometry(): template_list = {template_list}")
        print(f"\nAttempting to match {len(science_list)} sources in the "+
              f"science image to {len(template_list)} in the template")                      
        # find the transform using the control points
        tform, __ = aa.find_transform(science_list, template_list)
        # apply the transform
        print("\nAPPLYING TRANSFORM")
        img_aligned, footprint = aa.apply_transform(tform, science,
                                                    template,
                                                    propagate_mask=True)
        print("\nSUCCESS!\n")
    except aa.MaxIterError: # if cannot match images, try flipping 
        print("Max iterations exceeded; flipping the image...")
        xsize = fits.getdata(science_file).shape[1]
        ysize = fits.getdata(science_file).shape[0]
        source_list = [[xsize,ysize]-coords for coords in 
                       science_list.copy()]
        science = np.flip(science, axis=0)
        science = np.flip(science, axis=1)
            
        try:
            tform, __ = aa.find_transform(source_list, template_list)
            print(tform)
            img_aligned, footprint = aa.apply_transform(tform, science,
                                                        template,
                                                        propagate_mask=True)
            print("\nSUCCESS!\n")
        except aa.MaxIterError: # still too many iterations 
            print("Max iterations exceeded while trying to find "+
                  "acceptable transformation. Exiting.")
            return
        
    except aa.TooFewStarsError: # not enough stars in source/template
        print("Reference stars in source/template image are less than "+
              "the minimum value (3). Exiting.")
        return
    
    # except Exception: # any other exceptions
    #     e = sys.exc_info()
    #     print("\nWhile calling astroalign, some error other than "+
    #           "MaxIterError or TooFewStarsError was raised: "+
    #           f"\n{str(e[0])}\n{str(e[1])}\n\n")
    #     return


    #######################
    ### make the final mask
    #######################
    # mask 0s / nans AND mask outside the footprint of the image registration
    # (the mask should propagate via astroalign, but not sure if it does...)
    mask = np.logical_or(img_aligned==0, np.isnan(img_aligned))
    mask = np.logical_or(mask, footprint)
    
    ######################################
    ### plot the aligned image, if desired
    ######################################
    if plot_align: 
        alignplot = science_file.replace(".fits", 
                                    f"_astroalign_{scale}.png")
        __plot_align(template_hdr=template_hdr, 
                     img_aligned=img_aligned, 
                     mask=mask, 
                     scale=scale, 
                     output=alignplot)
    
    #####################################################
    ### finish editing the header, transfer WCS if needed
    #####################################################
    # set header for new aligned fits file 
    hdr = fits.getheader(science_file)
    # make a note that astroalign was successful
    hdr["IMREG"] = ("astroalign", "image registration software")  
    try: 
        hdr["PIXSCAL1"] = template_hdr["PIXSCAL1"] # pixscale of TEMPLATE
    except KeyError:
        pass
    # try to transfer the template WCS 
    if wcs_transfer: 
        hdr = __WCS_transfer(hdr, template_hdr)
    # final HDUs
    align_hdu = fits.PrimaryHDU(data=img_aligned, header=hdr)
    mask_hdu = fits.PrimaryHDU(data=mask.astype(int), header=hdr)
    
    #################################
    ### write (if desired) and return
    #################################
    if write: # if we want to write the aligned fits file and the mask 
        if not(output_im): # if no output name given, set default
            output_im = science_file.replace(".fits", "_align.fits")
        if not (output_mask): 
            output_mask = science_file.replace(".fits", "_align_mask.fits")
            
        align_hdu.writeto(output_im, overwrite=True, output_verify="ignore")
        mask_hdu.writeto(output_mask, overwrite=True, output_verify="ignore")
    
    return align_hdu, mask_hdu


## "unsupervised", pure astroalign ############################################

def image_align(science_file, template_file, mask_file=None, 
                thresh_sigma=3.0,
                wcs_transfer=True,
                plot_align=None, scale=None, 
                write=True, output_im=None, output_mask=None):
    """Align a science image to a template image using `astroalign` for **all** 
    steps, including source extraction and the final alignment.
    
    Arguments
    ---------
    science_file, template_file :  str
        Science and template fits file names
    mask_file : str, optional
        Mask fits file name (default None)
    thresh_sigma : float or array_like, optional
        Sigma threshold for source detection within astroalign (default 3.0)
    wcs_transfer : bool, optional
        Whether to attempt to transfer WCS coordinates from the template to the 
        newly-aligned image (default True)
    plot_align : bool, optional
        Whether to plot the final aligned science image (default False)
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plots (default "linear")
    write : bool, optional
        Whether to write *both* outputs (the aligned image and mask image from
        the image registration footprint) (default True)
    output_im : str, optional
        Name for output aligned image fits file (default 
        `science_file.replace(".fits", "_align.fits")`)
    output_mask : str, optional
        Name for output mask fits file (default 
        `science_file.replace(".fits", "_align_mask.fits")`)
    
    Returns
    -------
    align_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the aligned science image 
    mask_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the final mask

    Notes
    ------
    **Uses a slightly modified version of astroalign.**
    
    **TO-DO:**
    
    - Allow naming of output plot
    - Fix: During transfer of WCS header from PS1 to CFHT, WCS becomes nonsense

    """
    
    ## load in data
    science = fits.getdata(science_file)
    template = fits.getdata(template_file)
    #science_hdr = fits.getheader(science_file)
    template_hdr = fits.getheader(template_file)

    ## build up masks, apply them
    mask = np.logical_or(science==0, np.isnan(science)) # mask zeros/nans in sci
    if mask_file: # if a mask is provided
        mask = np.logical_or(mask, fits.getdata(mask_file))
    science = np.ma.masked_where(mask, science)  
    
    template_mask = np.logical_or(template==0, np.isnan(template)) # in template
    template = np.ma.masked_where(template_mask, template)

    ###########################################################################
    ## let astroalign handle everything
    try: 
        # find control points using image segmentation, find the transform,
        # and apply the transform
        img_aligned, footprint = aa.register(science, template,
                                             propagate_mask=True,
                                             thresh=thresh_sigma)
    except aa.MaxIterError: # if cannot match images, try flipping 
        print("\nMax iterations exceeded; flipping the image...")
        science = np.flip(science, axis=0)
        science = np.flip(science, axis=1)
            
        try:
            img_aligned, footprint = aa.register(science, template, 
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
              f"\n{str(e[0])}\n{str(e[1])}")
        return

    ###########################################################################        
    ## build the new mask 
    # mask pixels==0 or nan AND mask the footprint of the image registration
    # the mask should propagate via astroalign, but not sure if it does...
    mask = np.logical_or(img_aligned==0, np.isnan(img_aligned))
    mask = np.logical_or(mask, footprint)
    
    ## plot the aligned image, if desired
    if plot_align: 
        alignplot = science_file.replace(".fits", 
                                    f"_astroalign_{scale}.png")
        __plot_align(template_hdr=template_hdr, 
                     img_aligned=img_aligned, 
                     mask=mask, 
                     scale=scale, 
                     output=alignplot)
    
    ## set header for new aligned fits file 
    hdr = fits.getheader(science_file)
    # make a note that astroalign was successful
    hdr["IMREG"] = ("astroalign", "image registration software")  
    try: 
        hdr["PIXSCAL1"] = template_hdr["PIXSCAL1"] # pixscale of TEMPLATE
    except KeyError:
        pass
    # try to transfer the template WCS 
    if wcs_transfer: 
        hdr = __WCS_transfer(hdr, template_hdr)
    # final HDUs
    align_hdu = fits.PrimaryHDU(data=img_aligned, header=hdr)
    mask_hdu = fits.PrimaryHDU(data=mask.astype(int), header=hdr)
    
    if write: # if we want to write the aligned fits file and the mask 
        if not(output_im): # if no output name given, set default
            output_im = science_file.replace(".fits", "_align.fits")
        if not (output_mask): 
            output_mask = science_file.replace(".fits", "_align_mask.fits")
            
        align_hdu.writeto(output_im, overwrite=True, output_verify="ignore")
        mask_hdu.writeto(output_mask, overwrite=True, output_verify="ignore")
    
    return align_hdu, mask_hdu

###############################################################################
### ALIGNMENT WITH IMAGE_REGISTRATION PACKAGE #################################

def image_align_morph(science_file, template_file, mask_file=None, 
                      flip=False, maxoffset=30.0, wcs_transfer=True, 
                      plot_align=False, scale="linear", 
                      write=True, output_im=None, output_mask=None):
    """Align a science image to a template image using `image_registration`, 
    which relies on image morphology rather than cross-matching control points.

    Arguments
    ---------
    science_file, template_file :  str
        Science and template fits file names
    mask_file : str, optional
        Mask fits file name (default None)
    flip : bool, optional
        Whether to flip the science image before attempting to align it 
        (default False)
    maxoffset : float, optional
        *Maximum* allowed offset between science and template images to 
        consider the solution a good alignment (default 30.0)
    wcs_transfer : bool, optional
        Whether to attempt to transfer WCS coordinates from the template to the 
        newly-aligned image (default True)
    plot_align : bool, optional
        Whether to plot the final aligned science image (default False)
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plots (default "linear")
    write : bool, optional
        Whether to write *both* outputs (the aligned image and mask image from
        the image registration footprint) (default True)
    output_im : str, optional
        Name for output aligned image fits file (default 
        `science_file.replace(".fits", "_align.fits")`)
    output_mask : str, optional
        Name for output mask fits file (default 
        `science_file.replace(".fits", "_align_mask.fits")`)

    Returns
    -------
    align_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the aligned science image 
    mask_hdu : astropy.io.fits.PrimaryHDU
        New HDU (image + header) of the final mask
    
    Notes
    -----
    Calls on `image_registration` to align the science image with the target to 
    allow for proper image differencing. Also finds a mask of out of bounds 
    pixels to ignore during differencing. The image registration in this 
    function is based on morphology and edge detection in the image, in 
    contrast with `image_align_segm()`, `image_align_astrometry()`, and 
    `image_align()`, which use asterism-matching to align the two images. 
    
    For images composed mostly of point sources, use image_align. For images 
    composed mainly of galaxies/nebulae and/or other extended objects, use this
    function.
    
    **TO-DO:**
    
    - Fix: WCS header of aligned image doesn't always seem correct for alignment
      with non-CFHT templates
    
    """
 
    # some imports
    import warnings # ignore warning given by image_registration
    warnings.simplefilter('ignore', category=FutureWarning)   
    from image_registration import chi2_shift
    from scipy import ndimage
    
    ## load in data
    science = fits.getdata(science_file)
    if flip:
        science = np.flip(science, axis=0)
        science = np.flip(science, axis=1)
        
    template = fits.getdata(template_file)
    template_hdr = fits.getheader(template_file)
    
    ## pad/crop the science array so that it has the same shape as the template
    if science.shape != template.shape:
        xpad = (template.shape[1] - science.shape[1])
        ypad = (template.shape[0] - science.shape[0])
        
        if xpad > 0:
            print(f"\nXPAD = {xpad} --> padding science")
            science = np.pad(science, [(0,0), (0,xpad)], mode="constant", 
                             constant_values=0)
        elif xpad < 0: 
            print(f"\nXPAD = {xpad} --> cropping science")
            science = science[:, :xpad]
        else: 
            print(f"\nXPAD = {xpad} --> no padding/cropping science")
            
        if ypad > 0:
            print(f"YPAD = {ypad} --> padding science\n")
            science = np.pad(science, [(0,ypad), (0,0)], mode="constant", 
                             constant_values=0)
        elif ypad < 0:
            print(f"YPAD = {ypad} --> cropping science\n")
            science = science[:ypad, :]
        else: 
            print(f"\nYPAD = {ypad} --> no padding/cropping science\n")

    ## build up and apply a mask
    scimask = np.logical_or(science==0, np.isnan(science)) # in science
    template_mask = np.logical_or(template==0,np.isnan(template)) # in template 
    mask = np.logical_or(scimask, template_mask)
    
    if mask_file: # if a mask is provided
        maskdata = fits.getdata(mask_file) # load it in
        maskdata = maskdata[0:scimask.shape[0], 0:scimask.shape[1]] # crop
        mask = np.logical_or(mask, fits.getdata(mask_file))

    science = np.ma.masked_where(mask, science)
    template = np.ma.masked_where(mask, template)
        
    ## compute the required shift
    xoff, yoff, exoff, eyoff = chi2_shift(template, science, err=None, 
                                          return_error=True, 
                                          upsample_factor="auto",
                                          boundary="constant")
    
    # if offsets are too large, try flipping the image 
    if not(abs(xoff) < maxoffset and abs(yoff) < maxoffset):   
        print(f"\nEither the X or Y offset is larger than {maxoffset} "+
              "pix. Flipping the image and trying again...") 
        science = np.flip(science, axis=0) # try flipping the image
        science = np.flip(science, axis=1)
        xoff, yoff, exoff, eyoff = chi2_shift(template, science, err=None, 
                                              return_error=True, 
                                              upsample_factor="auto",
                                              boundary="constant")
        # if offsets are still too large, don't trust them 
        if not(abs(xoff) < maxoffset and abs(yoff) < maxoffset):
            print("\nAfter flipping, either the X or Y offset is still "+
                  f"larger than {maxoffset} pix. Exiting.")
            return 
        
    ## apply the shift 
    img_aligned = ndimage.shift(science, np.array((-yoff, -xoff)), order=3, 
                                mode='constant', cval=0.0, prefilter=True)   
    if mask_file:
        mask = np.logical_or((img_aligned == 0), mask)
    else: 
        mask = (img_aligned == 0)
    
    print(f"\nX OFFSET = {xoff} +/- {exoff}")
    print(f"Y OFFSET = {yoff} +/- {eyoff}\n")
    
    if plot_align: # plot, if desired
        alignplot = science_file.replace(".fits", 
                                         f"_image_morph_{scale}.png")
        __plot_align(template_hdr=template_hdr, 
                     img_aligned=img_aligned, 
                     mask=mask, 
                     scale=scale, 
                     output=alignplot)

    ## set header for new aligned fits file 
    hdr = fits.getheader(science_file)
    # make a note that image_registration was applied
    hdr["IMREG"] = ("image_registration", "image registration software")  
    try: 
        hdr["PIXSCAL1"] = template_hdr["PIXSCAL1"] # pixscale of TEMPLATE
    except KeyError:
        pass
    # try to transfer the template WCS 
    if wcs_transfer: 
        hdr = __WCS_transfer(hdr, template_hdr)
    # final HDUs
    align_hdu = fits.PrimaryHDU(data=img_aligned, header=hdr)
    mask_hdu = fits.PrimaryHDU(data=mask.astype(int), header=hdr)
    
    if write: # if we want to write the aligned fits file and the mask 
        if not(output_im): # if no output name given, set default
            output_im = science_file.replace(".fits", "_align.fits")
        if not (output_mask): 
            output_mask = science_file.replace(".fits", "_align_mask.fits")
            
        align_hdu.writeto(output_im, overwrite=True, output_verify="ignore")
        mask_hdu.writeto(output_mask, overwrite=True, output_verify="ignore")
    
    return align_hdu, mask_hdu


###############################################################################
### ASTROMETRY ################################################################

def solve_field(image_file, remove_PC=True, verify=False, prebkgsub=False, 
                guess_scale=False, read_scale=True, pixscale=None, 
                scale_tol=0.05, 
                verbose=0, output=None):
    """Use the `solve-field` command of `astrometry.net` to find an astrometric 
    solution for an image and output an updated, solved fits file.
    
    Arguments
    ---------
    image_file : str
        Filename for fits image to solve with `astrometry.net`'s 
        `solve-field`
    remove_PC : bool, optional
        Whether to look for "PC00i00j" headers and remove them before solving,
        if present (default True)
    verify : bool, optional
        Whether to try and verify existing WCS headers when solving (default 
        False)
    prebkgsub : bool, optional
        Whether the input image has already been background-subtracted (default 
        False)
    guess_scale : bool, optional
        Whether to try and guess the scale of the image from WCS headers 
        (default False)
    read_scale : bool, optional
        Whether to instead search for a "PIXSCAL1" header containing the image 
        pixel scale (default True; ignored if `guess_scale == True`)
    pixscale : float, optional
        Image scale in arcsec per pixel, if known (default None; will be 
        ignored if `guess_scale == True` *OR* if `read_scale == True` and the 
        "PIXSCAL1" header is successfully found)
    scale_tol : float, optional
        Degree of +/- tolerance for the pixel scale of the image (in arcsec per
        pix), if the header "PIXSCAL1" is found *or* if a scale is given 
        (default 0.05; see notes for details)
    verbose : {0, 1, 2}, optional
        Level of verbosity (default 0)
    output : str, optional
        Name for output updated .fits file (default 
        `image_file.replace(".fits","_solved.fits")`)
    
    Returns
    -------
    str
        The STDOUT produced by `astrometry.net`
        
    Raises
    ------
    ValueError
        If the output filename and input filename are the same
    
    Notes
    -----
    If hdr["PIXSCAL1"] = 0.185 and `scale_tol = 0.05`, `solve-field` will only 
    look for solutions with a pixel scale of 0.185 +/- 0.05 arcsec per pix.
    
    **Note:** Output filename must be different from input filename.
    """
    
    if remove_PC: # check for PC00i00j headers (e.g. in PS1)
        hdul = fits.open(image_file, mode="update")
        try:
            del hdul[0].header["PC001001"]
            del hdul[0].header["PC001002"]
            del hdul[0].header["PC002001"]
            del hdul[0].header["PC002002"]
        except KeyError:
            pass
        hdul.close()

    data = fits.getdata(image_file)
    hdr = fits.getheader(image_file)
    w = wcs.WCS(hdr)
    
    if output == image_file: 
        raise ValueError("Output and input can not have the same name")

    # overwrite, don't plot, input a fits image
    options = "--overwrite --no-plot --fits-image"
    
    # try and verify the WCS headers?
    if not(verify): options = f"{options} --no-verify"

    # don't bother producing these files 
    options = f'{options} --match "none" --solved "none" --rdls "none"'
    options = f'{options} --wcs "none" --corr "none"' 
    options = f'{options} --temp-axy' # only write a temporary augmented xy

    # if no need for background subtraction (already performed)
    if prebkgsub: options = f"{options} --no-background-subtraction" 
    
    # get RA, Dec of center to speed up solving
    centy, centx = [i//2 for i in data.shape]
    ra, dec = w.all_pix2world(centx, centy, 1) 
    rad = 0.5 # look in a radius of 0.5 deg
    options = f"{options} --ra {ra} --dec {dec} --radius {rad}" 

    # try and use headers to guess the image scale
    if guess_scale: 
        options = f"{options} --guess_scale"    
    # get pixel scale (in arcsec per pixel), if present 
    elif read_scale:
        try:
            pixscale = hdr["PIXSCAL1"]
            pixmin = pixscale-scale_tol
            pixmax = pixscale+scale_tol
            options = f"{options} --scale-low {pixmin} --scale-high {pixmax}"
            options = f'{options} --scale-units "app"'
        except KeyError:
            pass
    # manually input the scale
    elif pixscale:
        pixmin = pixscale-scale_tol
        pixmax = pixscale+scale_tol
        options = f"{options} --scale-low {pixmin} --scale-high {pixmax}"
        options = f'{options} --scale-units "app"'
        
    # set level of verbosity
    for i in range(min(verbose, 2)):
        options = f"{options} -v" # -v = verbose, -v -v = very verbose
    
    # set output filenames 
    if not(output):
        output = image_file.replace(".fits","_solved.fits")
    # -C <output> --> stop astrometry when this file is produced 
    options = f"{options} -C {output} --new-fits {output}"
    
    # run astrometry
    try:
        # store the STDOUT 
        a = run(f"solve-field {options} {image_file}", shell=True, check=True,
                stdout=PIPE)   
        #print("\n\n\n\n"+str(a.stdout, "utf-8"))
        # if did not solve, exit
        if not("Field 1: solved with index" in str(a.stdout, "utf-8")):
            print(str(a.stdout, "utf-8"))
            print("\nExiting.")
            return      
    # if an error code is returned by solve-field, exit
    except CalledProcessError: # if an error code is returned, exit 
        print("\n\n\nError, exiting")
        return 
    
    # get rid of xyls file 
    run(f'rm {image_file.replace(".fits","-indx.xyls")}', shell=True)
    
    return str(a.stdout, "utf-8")
