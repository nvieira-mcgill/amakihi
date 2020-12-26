#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:44:27 2020
@author: Nicholas Vieira
@imalign.py

**NOTE:** If using astroalign for everything, can't plot sources. Make note of
this.

**TO-DO:**

- `image_align()` is almost 1000 lines... break it up in 3 based on fact that
  it allows 3 different methods to be used

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
from astropy.stats import SigmaClip
from astropy.coordinates import SkyCoord
from astropy.table import Table
from photutils import Background2D, MedianBackground
from photutils import make_source_mask, detect_sources, source_properties

# amakihi 
from plotting import __plot_sources, __plot_align

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
#### IMAGE ALIGNMENT (REGISTRATION) ###########################################

def __remove_SIP(header):
    """Removes all SIP-related keywords from a FITS header, in-place.
    
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


def image_align(source_file, template_file, mask_file=None, 
                imsegm=True, astrometry=False, doublealign=False,
                per_upper=0.95, exclusion=0.05, nsources=50,
                thresh_sigma=3.0, pixelmin=10, etamax=2.0, 
                bkgsubbed=False, astrom_sigma=8.0, psf_sigma=5.0, 
                keep=False, 
                sep_max=2.0, ncrossmatch=8,
                wcs_transfer=True,
                plot_sources=False, plot_align=None, scale=None, 
                circ_color="#fe01b1",
                write=True, output_im=None, output_mask=None):
    """    
    WIP: during transfer of WCS header from PS1 to CFHT, WCS becomes nonsense 
    
    Input: 
        general:
        - science image (source) to register
        - template image (target) to match source to
        - mask file for the SOURCE image (optional; default None)
        
        source detection:
        - use image segmentation for source detection and provide astroalign 
          with a list of x, y coordinates if True (optional; default True)
        - use astrometry.net (specifically, image2xy) for source detection and 
          provide astroalign with a list of x, y coordinates if True (optional; 
          default False; overriden by imsegm if imsegm=True)
        - attempt a second, more refined alignment by cross-matching sources 
          from the two images and using the correspondence to build a new 
          transform (optional; default False)
        - flux percentile upper limit (optional; default 0.95 = 95%; can be 
          set to a list to specify different values for science and template
          images)
        - no. of sources to keep for asterism-matching (optional; default 50;
          astroalign will not accept any more than 50)
        - the fraction of the top/bottom/left/right-most edges from which to 
          exclude sources during matching (optional; default 0.05; can be 
          set to a list to specify different values for science and 
          template images)
        
        if imsegm=True (default):
        - sigma threshold for source detection with image segmentation 
          (optional; default 3.0; can pass a list to specify a different 
          threshold for source and template)
        - *minimum* number of isophote pixels, i.e. area (optional; default 10;
          can pass a list to specify a different area for source and template)
        - *maximum* allowed elongation for sources found by image segmentation 
          (optional; default 2.0; setting this to None sets no maximum; 
          can pass a list to specify different maximum elongation for source 
          and template)   

        if astrometry=True (and imsegm=False):
        - whether the input source and template files have been background-
          subtracted (optional; default False; can be set to a list to specify
          that the science image is background-subtracted and the template is
          not or vice-versa)
        - sigma threshold for image2xy source detection in the source/template 
          (optional; default 8.0 which sets the threshold to 8.0 for both; can
          pass a list to specify a different threshold for source and template)
        - the sigma of the Gaussian PSF of the source/template (optional; 
          default 5.0 which sets the sigma to 5.0 for both; can pass a list to
          specify a different width for source and template)
        - whether to keep the source list files after running image2xy 
          (optional; default False)   

        if doublealign=True (and imsegm=True):
        - maximum allowed on-sky separation (in arcsec) to cross-match a source 
          in the science image and another in the template (optional; default 
          2.0")
        - required no. of cross-matched sources to do double alignment 
          (optional; default 8)

        transferring WCS:
        - whether to try transferring the WCS solution of the template image 
          to the science image (optional; default True)
          
        plotting:
        - whether to plot the sources detected in the science image and 
          and template image, if astrometry was used (optional; default False)
        - whether to plot the aligned image data (optional; default False)
        - scale to apply to the plot(s) (optional; default None (linear); 
          options are "linear", "log", "asinh")
        - color for the circles around the sources in the source detection 
          image (optional; default "#fe01b1" ~ bright pink; only relevant if 
          plot_sources=True)
        
        writing:
        - whether to write the outputs to .fits files (optional; default True) 
        - names for the output aligned image and image mask (both optional; 
          defaults set below)
    
    Uses either image segmentation (default) or astrometry.net's image2xy to 
    find a set of x, y control points in the source and target images. Then,
    - excludes sources very close to the border as set by the user
    - if using image segmentation, imposes a minimum isophote area and maximum 
      source elongation for each source
    - rejects sources above some upper flux percentile as set by the user
    - selects as many sources as possible (maximum set by user, can not exceed
      50) as control points 
    
    These control points are then passed to astroalign, which finds the 
    invariants between the source control point set and target control point 
    set to compute the affine transformation which aligns the source image with 
    the target image. Applies this transformation to the science image. 
    
    Alternatively, can allow astroalign to do all of the above steps on its own 
    with no supervision. 
    
    *** Uses a slightly modified version of astroalign.
    
    NOTE: image2xy has a hard time doing basic background subtraction when an 
    image has nans in it. Before trying alignment with an image which contains
    nans, it is suggested to set all nans to 0 instead.
    
    Output: the aligned image and a bad pixel mask

    **TO-DO:**
    
    - Allow naming of output plots

    """
    
    # a modified version of astroalign which allows the user to set the sigma 
    # threshold for source detection, which sometimes needs to be tweaked
    import astroalign_mod as aa
    
    ## load in data
    source = fits.getdata(source_file)
    template = fits.getdata(template_file)
    source_hdr = fits.getheader(source_file)
    template_hdr = fits.getheader(template_file)

    ## build up masks, apply them
    mask = np.logical_or(source==0, np.isnan(source)) # mask zeros/nans in src 
    if mask_file: # if a mask is provided
        mask = np.logical_or(mask, fits.getdata(mask_file))
    source = np.ma.masked_where(mask, source)  
    
    tmpmask = np.logical_or(template==0, np.isnan(template)) # in template
    template = np.ma.masked_where(tmpmask, template)

    # check if input upper flux percentile is list or single value 
    if not(type(per_upper) in [float,int]):
        sciper = per_upper[0]
        tmpper = per_upper[1]
    else:
        sciper = tmpper = per_upper
    # check if input exclusion fraction is list or single value 
    if not(type(exclusion) in [float,int]):
        sciexclu = exclusion[0]
        tmpexclu = exclusion[1]
    else:
        sciexclu = tmpexclu = exclusion
        
    # check input nsources
    if nsources > 50:
        print("The number of sources used for matching cannot exceed 50 --> "+
              "setting limit to 50")
        nsources = 50

    ###########################################################################
    ### OPTION 1: use image segmentation to find sources in the source and #### 
    ### template, find the transformation using these sources as control ######
    ### points, and apply the transformation to the source image ##############
    ###########################################################################
    if imsegm:  
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
        
        #################################################################
        ## get background standard deviation for source image (if needed)
        #################################################################
        try: 
            scistd = float(source_hdr['BKGSTD']) 
        except KeyError:
            # use crude image segmentation to find sources above SNR=3, build a 
            # source mask, and estimate the background RMS 
            if mask_file: # load a bad pixel mask if one is present 
                source_mask = make_source_mask(source, snr=3, npixels=5, 
                                               dilate_size=15, mask=mask)
                # combine the bad pixel mask and source mask 
                rough_mask = np.logical_or(mask, source_mask)
            else: 
                source_mask = make_source_mask(source, snr=3, npixels=5, 
                                               dilate_size=15)
                rough_mask = source_mask            
            # estimate the background standard deviation
            try:
                sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
            except TypeError: # in old astropy, "maxiters" was "iters"
                sigma_clip = SigmaClip(sigma=3, iters=5)            
            try:
                bkg = Background2D(source, (50,50), filter_size=(5,5), 
                                   sigma_clip=sigma_clip, 
                                   bkg_estimator=MedianBackground(), 
                                   mask=rough_mask)
            except ValueError:
                e = sys.exc_info()
                print("\nWhile attempting background estimation on the "+
                      "science image, the following error was raised: "+
                      f"\n{str(e[0])}\n{str(e[1])}\n--> exiting.")
                return
            
            scistd = bkg.background_rms    

        ###################################################################
        ## get background standard deviation for template image (if needed)
        ###################################################################
        try: 
            tmpstd = float(template_hdr['BKGSTD']) 
        except KeyError:
            # use crude image segmentation to find sources above SNR=3, build a 
            # source mask, and estimate the background RMS 
            source_mask = make_source_mask(template, snr=3, npixels=5, 
                                           dilate_size=15, mask=tmpmask)
            rough_mask = np.logical_or(source_mask, tmpmask)            
            # estimate the background standard deviation
            try:
                sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
            except TypeError: # in old astropy, "maxiters" was "iters"
                sigma_clip = SigmaClip(sigma=3, iters=5)   
            try:
                bkg = Background2D(template, (50,50), filter_size=(5,5), 
                                   sigma_clip=sigma_clip, 
                                   bkg_estimator=MedianBackground(), 
                                   mask=rough_mask)   
            except ValueError:
                e = sys.exc_info()
                print("\nWhile attempting background estimation on the "+
                      "template image, the following error was raised: "+
                      f"\n{str(e[0])}\n{str(e[1])}\n--> exiting.")
                return

            tmpstd = bkg.background_rms
            
        ######################################
        ## find control points in source image  
        ######################################
        segm_source = detect_sources(source, 
                                     scithresh*scistd, 
                                     npixels=scipixmin,
                                     mask=mask)          
        # use the segmentation image to get the source properties 
        cat_source = source_properties(source, segm_source, mask=mask)
        try:
            scitbl = cat_source.to_table()
        except ValueError:
            print("source image contains no sources. Exiting.")
            return        
        # restrict elongation  
        scitbl = scitbl[(scitbl["elongation"] <= scieta)]
        # remove sources in leftmost/rightmost/topmost/bottommost edge of image
        xc = scitbl["xcentroid"].value; yc = scitbl["ycentroid"].value
        xedge_min = [min(x, source.shape[1]-x) for x in xc]
        yedge_min = [min(y, source.shape[0]-y) for y in yc]
        scitbl["xedge_min"] = xedge_min
        scitbl["yedge_min"] = yedge_min
        keep = [(s["xedge_min"]>sciexclu*source.shape[1]) and
                (s["yedge_min"]>sciexclu*source.shape[0]) for s in scitbl]
        scitbl = scitbl[keep]        
        # pick at most <nsources> sources below the <per_upper> flux percentile
        scitbl["sum/area"] = scitbl["source_sum"].data/scitbl["area"].data
        scitbl.sort("sum/area") # sort by flux
        scitbl.reverse() # biggest to smallest
        start = int((1-sciper)*len(scitbl))
        end = min((len(scitbl)-1), (start+nsources))
        scitbl = scitbl[start:end]   
        # get the list
        source_list = np.array([[scitbl['xcentroid'].data[i], 
                                 scitbl['ycentroid'].data[i]] for i in 
                               range(len(scitbl['xcentroid'].data))]) 

        ########################################
        ## find control points in template image
        ########################################
        segm_tmp = detect_sources(template, 
                                  tmpthresh*tmpstd, 
                                  npixels=tmppixmin,
                                  mask=tmpmask)          
        # use the segmentation image to get the source properties 
        cat_tmp = source_properties(template, segm_tmp, mask=tmpmask)
        try:
            tmptbl = cat_tmp.to_table()
        except ValueError:
            print("template image contains no sources. Exiting.")
            return        
        # restrict elongation  
        tmptbl = tmptbl[(tmptbl["elongation"] <= tmpeta)] 
        # remove sources in leftmost/rightmost/topmost/bottommost edge of image
        xc = tmptbl["xcentroid"].value; yc = tmptbl["ycentroid"].value
        xedge_min = [min(x, template.shape[1]-x) for x in xc]
        yedge_min = [min(y, template.shape[0]-y) for y in yc]
        tmptbl["xedge_min"] = xedge_min
        tmptbl["yedge_min"] = yedge_min
        keep = [(t["xedge_min"]>tmpexclu*template.shape[1]) and
                (t["yedge_min"]>tmpexclu*template.shape[0]) for t in tmptbl]
        tmptbl = tmptbl[keep] 
        # pick at most <nsources> sources below the <per_upper> flux percentile
        tmptbl["sum/area"] = tmptbl["source_sum"].data/tmptbl["area"].data
        tmptbl.sort("sum/area") # sort by flux
        tmptbl.reverse() # biggest to smallest
        start = int((1-tmpper)*len(tmptbl))
        end = min((len(tmptbl)-1), (start+nsources))
        tmptbl = tmptbl[start:end] 
        # get the list
        template_list = np.array([[tmptbl['xcentroid'].data[i], 
                                   tmptbl['ycentroid'].data[i]] for i in 
                                 range(len(tmptbl['xcentroid'].data))]) 

        ########################################################
        ## show the sources attempting to be matched, if desired
        ########################################################
        if plot_sources:
            sourceplot = source_file.replace(".fits", 
                                             f"_alignment_sources_{scale}.png")
            __plot_sources(source_data=source, 
                           template_data=template, 
                           source_hdr=source_hdr, 
                           template_hdr=template_hdr, 
                           source_list=source_list, 
                           template_list=template_list, 
                           scale=scale, 
                           color=circ_color, 
                           output=sourceplot)      

        ###################
        ## align the images
        ###################
        try: 
            print(f"\nAttempting to match {len(source_list)} sources in the "+
                  f"science image to {len(template_list)} in the template")                      
            # find the transform using the control points
            tform, __ = aa.find_transform(source_list, template_list)
            # apply the transform
            img_aligned, footprint = aa.apply_transform(tform, source,
                                                        template,
                                                        propagate_mask=True)
            print("\nSUCCESS\n")
        except aa.MaxIterError: # if cannot match images, try flipping 
            print("Max iterations exceeded; flipping the image...")
            xsize = fits.getdata(source_file).shape[1]
            ysize = fits.getdata(source_file).shape[0]
            source_list = [[xsize,ysize]-coords for coords in 
                           source_list.copy()]
            source = np.flip(source, axis=0)
            source = np.flip(source, axis=1)
                
            try:
                tform, __ = aa.find_transform(source_list, template_list)
                img_aligned, footprint = aa.apply_transform(tform, source,
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
        
        # using the while loop here is bad practice, I think
        while doublealign:
            print("\nAttempting a second iteration of alignment...")
            
            # build new mask
            mask = np.logical_or(img_aligned==0, np.isnan(img_aligned))
            mask = np.logical_or(mask, footprint)
            
            source_new = img_aligned.copy()
            source_new = np.ma.masked_where(mask, source_new)  
            segm_source = detect_sources(source_new, 
                                         scithresh*scistd, 
                                         npixels=scipixmin,
                                         mask=mask)          
            # use the segmentation image to get the source properties 
            cat_source = source_properties(source_new, segm_source, mask=mask)
            try:
                scitbl = cat_source.to_table()
            except ValueError:
                print("source image contains no sources. Exiting.")
                return        
            # restrict elongation  
            scitbl = scitbl[(scitbl["elongation"] <= scieta)]
            # remove sources in edges of image
            xc = scitbl["xcentroid"].value
            yc = scitbl["ycentroid"].value
            xedge_min = [min(x, source_new.shape[1]-x) for x in xc]
            yedge_min = [min(y, source_new.shape[0]-y) for y in yc]
            scitbl["xedge_min"] = xedge_min
            scitbl["yedge_min"] = yedge_min
            keep = [(s["xedge_min"]>sciexclu*source_new.shape[1]) and
                (s["yedge_min"]>sciexclu*source_new.shape[0]) for s in scitbl]
            scitbl = scitbl[keep]        
            # pick at most <nsources> sources below <per_upper>
            scitbl["sum/area"] = scitbl["source_sum"].data/scitbl["area"].data
            scitbl.sort("sum/area") # sort by flux
            scitbl.reverse() # biggest to smallest
            start = int((1-sciper)*len(scitbl))
            end = min((len(scitbl)-1), (start+nsources))
            scitbl = scitbl[start:end]   
            # get the list
            source_list = np.array([[scitbl['xcentroid'].data[i], 
                                     scitbl['ycentroid'].data[i]] for i in 
                                     range(len(scitbl['xcentroid'].data))])
            
            # get ra, dec for all of the sources to allow cross-matching 
            xsci = [s[0] for s in source_list]
            ysci = [s[1] for s in source_list]
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
                doublealign=False
                break
            elif len(scitbl_new) < ncrossmatch: # or too few sources 
                # 8 is arbitrary atm
                print(f'\nWas only able to cross-match {len(scitbl_new):d} '+
                      f'< {ncrossmatch:d} sources in the science and template '+
                      f'images for separation < {sep_max:.2f}". '+
                      'New solution not obtained.')
                doublealign=False
                break
            
            # otherwise, keep going
            print(f'\nFound {len(scitbl_new)} sources in the science image '+
                  'with a 1:1 match to a source in the template within '+
                  f'< {sep_max:.2f}", with average separation '+
                  f'{np.mean(d2d.value*3600):.2f}"')
            
            # new list of sources 
            source_list_new = np.array([[scitbl_new['xcentroid'].data[i], 
                    scitbl_new['ycentroid'].data[i]] for i in range(
                    len(scitbl_new['xcentroid'].data))])            
            template_list_new = np.array([[tmptbl_new['xcentroid'].data[i], 
                    tmptbl_new['ycentroid'].data[i]] for i in range(
                    len(tmptbl_new['xcentroid'].data))])

            # for bugtesting
            print(source_list_new[:5])
            print(template_list_new[:5])
            
            try: 
                print("\nAttempting to match...")                     
                # find the transform using the control points
                tform = aa.estimate_transform('affine', source_list_new, 
                                              template_list_new)
                # apply the transform
                img_aligned, footprint = aa.apply_transform(tform, source_new,
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
              

    ###########################################################################
    ### OPTION 2: use astrometry.net to find the sources, find the transform, #
    ### and then apply the transform ##########################################
    ###########################################################################
    elif astrometry:  
        # check if input bkgsubbed bool is list or single value
        if not(type(bkgsubbed) == bool):
            source_bkgsub = bkgsubbed[0]
            tmp_bkgsub = bkgsubbed[1]
        else:
            source_bkgsub = tmp_bkgsub = bkgsubbed
        # check if input astrometry significance sigma is list or single value 
        if not(type(astrom_sigma) in [float,int]):
            source_sig = astrom_sigma[0]
            tmp_sig = astrom_sigma[1]
        else:
            source_sig = tmp_sig = astrom_sigma
        # check if input astrometry PSF sigma is list or single value 
        if not(type(psf_sigma) in [float,int]):
            source_psf = psf_sigma[0]
            tmp_psf = psf_sigma[1]
        else:
            source_psf = tmp_psf = psf_sigma
           
        # -O --> overwrite
        # -p --> source significance 
        # -w --> estimated PSF sigma 
        # -s 10 --> size of the median filter to apply to the image is 10x10
        # -m 10000 --> max object size for deblending is 10000 pix**2
        
        ######################################
        ## find control points in source image 
        ######################################
        options = f" -O -p {source_sig} -w {source_psf} -s 10 -m 10000"
        if source_bkgsub: options = f"{options} -b" # no need for subtraction
        run(f"image2xy {options} {source_file}", shell=True)    
        source_list_file = source_file.replace(".fits", ".xy.fits")
        source_list = Table.read(source_list_file)
        if len(source_list) == 0: # if no sources found 
            print("\nNo sources found with astrometry.net in the source "+
                  "image, so image alignment cannot be obtained. Exiting.")
            return
        # pick at most <nsources> sources below the <per_upper> flux percentile
        source_list.sort('FLUX') # sort by flux
        source_list.reverse() # biggest to smallest
        start = int((1-sciper)*len(source_list))
        end = min((len(source_list)-1), (start+nsources))
        source_list = source_list[start:end]   
        source_list = np.array([[source_list['X'][i], 
                                 source_list['Y'][i]] for i in 
                               range(len(source_list['X']))]) 
        # remove sources in leftmost/rightmost/topmost/bottommost edge of image
        source_list = [s for s in source_list.copy() if (
                (min(s[0], source.shape[1]-s[0])>sciexclu*source.shape[1]) and
                (min(s[1], source.shape[0]-s[1])>sciexclu*source.shape[0]))]
        
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
            print(f"{source_list_file}\n{template_list_file}")
        else:
            run(f"rm {source_list_file}", shell=True) # not needed
            run(f"rm {template_list_file}", shell=True) 

        ########################################################
        ## show the sources attempting to be matched, if desired
        ########################################################
        if plot_sources:
            if not scale: scale = "linear"
            sourceplot = source_file.replace(".fits", 
                                             f"_alignment_sources_{scale}.png")
            __plot_sources(source_data=source, 
                           template_data=template, 
                           source_hdr=source_hdr, 
                           template_hdr=template_hdr, 
                           source_list=source_list, 
                           template_list=template_list, 
                           scale=scale, 
                           color=circ_color, 
                           output=sourceplot)    
            
        ###################
        ## align the images
        ###################
        try: 
            print(f"\nAttempting to match {len(source_list)} sources in the "+
                  f"science image to {len(template_list)} in the template")                      
            # find the transform using the control points
            tform, __ = aa.find_transform(source_list, template_list)
            # apply the transform
            img_aligned, footprint = aa.apply_transform(tform, source,
                                                        template,
                                                        propagate_mask=True)
            print("\nSUCCESS\n")
        except aa.MaxIterError: # if cannot match images, try flipping 
            print("Max iterations exceeded; flipping the image...")
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

    ###########################################################################
    ### OPTION 3: let astroalign handle everything ############################
    ###########################################################################
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
                  f"\n{str(e[0])}\n{str(e[1])}")
            return
        
    # build the new mask 
    # mask pixels==0 or nan AND mask the footprint of the image registration
    # the mask should propagate via astroalign, but not sure if it does...
    mask = np.logical_or(img_aligned==0, np.isnan(img_aligned))
    mask = np.logical_or(mask, footprint)
    
    if plot_align: # plot the aligned image, if desired
        alignplot = source_file.replace(".fits", 
                                    f"_astroalign_{scale}.png")
        __plot_align(template_hdr=template_hdr, 
                     img_aligned=img_aligned, 
                     mask=mask, 
                     scale=scale, 
                     output=alignplot)
    
    ## set header for new aligned fits file 
    hdr = fits.getheader(source_file)
    # make a note that astroalign was successful
    hdr["IMREG"] = ("astroalign", "image registration software")  
    try: 
        hdr["PIXSCAL1"] = template_hdr["PIXSCAL1"] # pixscale of TEMPLATE
    except KeyError:
        pass
    
    if wcs_transfer: # try to transfer the template WCS 
        mjdobs, dateobs = hdr["MJD-OBS"], hdr["DATE-OBS"] # store temporarily
        w = wcs.WCS(template_hdr)    
        # if no SIP transformations in header, need to update 
        if not("SIP" in template_hdr["CTYPE1"]) and ("SIP" in hdr["CTYPE1"]):
            __remove_SIP(hdr) # remove -SIP and remove related headers 
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
        hdr.update((w.to_fits(relax=False))[0].header) # update     

        # build the final header
        hdr["MJD-OBS"] = mjdobs # get MJD-OBS of SCIENCE image
        hdr["DATE-OBS"] = dateobs # get DATE-OBS of SCIENCE image
        
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
                      flip=False, maxoffset=30.0, wcs_transfer=True, 
                      plot_align=False, scale=None, 
                      write=True, output_im=None, output_mask=None):
    """
    WIP: WCS header of aligned image doesn't always seem correct for alignment
         with non-CFHT templates
         
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
    
    Output: the aligned image and a bad pixel mask
    """
    
    ## load in data
    if flip:
        source = np.flip(fits.getdata(source_file), axis=0)
        source = np.flip(source, axis=1)
    else: 
        source = fits.getdata(source_file)
    template = fits.getdata(template_file)
    template_hdr = fits.getheader(template_file)
    
    import warnings # ignore warning given by image_registration
    warnings.simplefilter('ignore', category=FutureWarning)
    
    from image_registration import chi2_shift
    from scipy import ndimage
    
    ## pad/crop the source array so that it has the same shape as the template
    if source.shape != template.shape:
        xpad = (template.shape[1] - source.shape[1])
        ypad = (template.shape[0] - source.shape[0])
        
        if xpad > 0:
            print(f"\nXPAD = {xpad} --> padding source")
            source = np.pad(source, [(0,0), (0,xpad)], mode="constant", 
                                     constant_values=0)
        elif xpad < 0: 
            print(f"\nXPAD = {xpad} --> cropping source")
            source = source[:, :xpad]
        else: 
            print(f"\nXPAD = {xpad} --> no padding/cropping source")
            
        if ypad > 0:
            print(f"YPAD = {ypad} --> padding source\n")
            source = np.pad(source, [(0,ypad), (0,0)], mode="constant", 
                                     constant_values=0)
            
        elif ypad < 0:
            print(f"YPAD = {ypad} --> cropping source\n")
            source = source[:ypad, :]
        else: 
            print(f"\nYPAD = {ypad} --> no padding/cropping source\n")

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
        print(f"\nEither the X or Y offset is larger than {maxoffset} "+
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
                  f"larger than {maxoffset} pix. Exiting.")
            return 
        
    ## apply the shift 
    img_aligned = ndimage.shift(source, np.array((-yoff, -xoff)), order=3, 
                                mode='constant', cval=0.0, prefilter=True)   
    if mask_file:
        mask = np.logical_or((img_aligned == 0), mask)
    else: 
        mask = (img_aligned == 0)
    
    print(f"\nX OFFSET = {xoff} +/- {exoff}")
    print(f"Y OFFSET = {yoff} +/- {eyoff}\n")

    template_header = fits.getheader(template_file) # image header
    
    if plot_align: # plot, if desired
        alignplot = source_file.replace(".fits", 
                                        f"_image_morph_{scale}.png")
        __plot_align(template_hdr=template_hdr, 
                     img_aligned=img_aligned, 
                     mask=mask, 
                     scale=scale, 
                     output=alignplot)

    ## set header for new aligned fits file 
    hdr = fits.getheader(source_file)
    # make a note that image_registration was applied
    hdr["IMREG"] = ("image_registration", "image registration software")  
    try: 
        hdr["PIXSCAL1"] = template_hdr["PIXSCAL1"] # pixscale of TEMPLATE
    except KeyError:
        pass
    
    if wcs_transfer:
        mjdobs, dateobs = hdr["MJD-OBS"], hdr["DATE-OBS"] # store temporarily
        w = wcs.WCS(template_header)    
        # if no SIP transformations in header, need to update 
        if not("SIP" in template_hdr["CTYPE1"]) and ("SIP" in hdr["CTYPE1"]):
            __remove_SIP(hdr) # remove -SIP and remove related headers 
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
        hdr.update((w.to_fits(relax=False))[0].header) # update     

        # build the final header
        hdr["MJD-OBS"] = mjdobs # get MJD-OBS of SCIENCE image
        hdr["DATE-OBS"] = dateobs # get DATE-OBS of SCIENCE image
        
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


def solve_field(image_file, remove_PC=True, verify=False, prebkgsub=False, 
                guess_scale=False, read_scale=True, pixscale=None, 
                scale_tolerance=0.05, 
                verbose=0, output=None):
    """
    Input:
        general:
        - image file to solve with astrometry.net's solve-field
        - whether to look for PC00i00j headers and remove them in the image 
          before solving, if present (optional; default True)
        - try to verify WCS headers when solving (optional; default False)
        - whether input file is previously background-subtracted (optional; 
          default False)
        
        pixel scale:
        - try to guess the scale of the image from WCS headers (optional; 
          default False)
        - whether to search the headers for a PIXSCAL1 header containing the 
          image pixel scale (optional; default True)
          will be ignored if guess_scale=True
        - image scale (in arcsec per pixel), if known (optional; default None)
          will be ignored unless :
              guess_scale=False AND read_scale=True AND no header "PIXSCAL1",
              OR guess_scale=False AND read_scale=False
        - degree of +/- tolerance for the pixel scale of the image, if the 
          header PIXSCAL1 is found in the image file OR if a scale is given 
          (optional; default 0.05 arcsec per pix)
          e.g., if hdr["PIXSCAL1"] = 0.185, astrometry.net will only look for 
          solutions with a pixel scale of 0.185 +/- 0.05 arcsec per pix by 
          default
          or, if this header is not found OR read_scale=False AND scale=0.185,
          the same 
          
        other:
        - level of verbosity (optional; default 0; options are 0, 1, 2)
        - name for output updated .fits file (optional; default set below)
        
    NOTE: the output filename MUST be different from the input filename. 
    astrometry.net will exit if this is not the case.
        
    Output: the STDOUT from solve-field, as text
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
        print("Output and input can not have the same name. Exiting.")
        return

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
            pixmin = pixscale-scale_tolerance
            pixmax = pixscale+scale_tolerance
            options = f"{options} --scale-low {pixmin} --scale-high {pixmax}"
            options = f'{options} --scale-units "app"'
        except KeyError:
            pass
    # manually input the scale
    elif pixscale:
        pixmin = pixscale-scale_tolerance
        pixmax = pixscale+scale_tolerance
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