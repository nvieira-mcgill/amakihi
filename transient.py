#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:33:42 2020
@author: Nicholas Vieira
@transient.py

Transient detection and converting candidate transients to "triplets" for 
further vetting with bogus-real adversarial artificial intelligence (braai).

**TO-DO:**

- Proper docstrings for `transient_detect()`, `transient_triplets()`

"""

# misc
import numpy as np
import re

# astropy
from astropy.io import fits
from astropy import wcs
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.table import Table
from photutils import detect_sources, source_properties

# amakihi 
from crop import crop_WCS
from plotting import (__plot_rejected, __plot_triplet, __plot_distributions, 
                      plot_transient)

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

###############################################################################
#### TRANSIENT DETECTION ######################################################   

def transient_detect(sub_file, og_file, ref_file, mask_file=None, 
                     thresh_sigma=5.0, pixelmin=20, 
                     dipole_width=2.0, dipole_fratio=5.0,
                     etamax=1.8, areamax=300, nsource_max=50, 
                     toi=None, toi_sep_min=None, toi_sep_max=None,
                     write=True, output=None, 
                     plot_distributions=False,
                     plot_rejections=False,
                     plots=["zoom og", "zoom ref", "zoom diff"], 
                     pixcoords=False,
                     sub_scale=None, og_scale=None, stampsize=200.0, 
                     crosshair_og="#fe019a", crosshair_sub="#5d06e9",
                     title=None, plotdir=None):
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
        - *maximum* allowed separation for any potential dipoles to be 
          considered real sources (optional; default 2.0"; setting this to None
          sets no maximum) 
        - *maximum* allowed flux ratio for dipoles (optional; default 5.0)
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
        - whether to use the pixel coord
        - scale to apply to the difference images (optional; default "asinh"; 
          options are "linear", "log", "asinh")
        - scale to apply to the science/reference images (optional; default
          "asinh"; options are "linear", "log", "asinh")
        - size of the transient stamp in pixels (optional; default 200.0)
        - colour for crosshair on transient in science/ref images (optional; 
          default ~hot pink)
        - colour for crosshair on transient in difference images (optional;
          default ~purple-blue)     
        - a title to include in the titles of all plots AND filenames of all 
          plots, except the rejection plot (optional; default None)
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
    tmp_data = fits.getdata(ref_file)
    
    # build a mask, including the template's bad pixels too
    mask = np.logical_or(data==0, np.isnan(data))
    tmpmask = np.logical_or(tmp_data==0, np.isnan(tmp_data))
    mask = np.logical_or(mask, tmpmask)
    if mask_file:
        mask = np.logical_or(mask, fits.getdata(mask_file))
        
    data = np.ma.masked_where(data==0.0, data) # mask all bad pix

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
    tbl["source_sum/area"] = tbl["source_sum"]/tbl["area"] # flux per pixel
    tbl_novetting = tbl.copy()
        
    ### INFORMATIVE PLOTS #####################################################
    if plot_distributions:
        __plot_distributions(og_file=og_file, tbl=tbl, 
                             etamax=etamax, areamax=areamax, 
                             title=title)

    
    ### CANDIDATE VETTING #####################################################
    ## (1) look for dipoles by looking for sources in (-1)*difference and  
    ## cross-matching to the segmentation image 
    if dipole_width:
        idx_rem = [] # indices to remove
        try:
            inv = cat_inv.to_table()
            inv["ra"], inv["dec"] = w.all_pix2world(inv["xcentroid"], 
                                                    inv["ycentroid"], 1)
            inv["source_sum/area"] = inv["source_sum"]/inv["area"] # flux/pixel
            coords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, frame="icrs")
            inv_coords = SkyCoord(inv["ra"]*u.deg, inv["dec"]*u.deg, 
                                  frame="icrs")        
            # indices of sources within <dipole_width> arcsec of each other
            idx_inv, idx, d2d, d3d = coords.search_around_sky(inv_coords, 
                                                        dipole_width*u.arcsec)
            
            # if they correspond to the same peak, the amplitude of one part 
            # of the dipole should be no more than fratio times the amplitude 
            # of the other part of the dipole
            for n in range(len(idx)):
                i, j = idx[n], idx_inv[n]
                
                ratio = tbl[i]["source_sum/area"]/inv[j]["source_sum/area"]
                if ((1.0/dipole_fratio) <= ratio <= dipole_fratio): 
                    idx_rem.append(i)
            
            if len(idx_rem) > 0:
                print(f'\n{len(idx_rem)} likely dipole(s) (width < '+
                      f'{dipole_width}", fratio < {dipole_fratio}) '+
                      'detected and removed')
                tbl.remove_rows(idx_rem)
            tbl_nodipoles = tbl.copy() # tbl with no dipoles
            
        except ValueError:
            print("The inversion of the difference image, (-1.0)*diff, does "+
                  "not contain any sources. Will continue without searching "+
                  "for dipoles.")
            tbl_nodipoles = tbl.copy() # tbl with no dipoles
    
    ## (2) restrict based on source elongation 
    if etamax:
        premasklen = len(tbl)
        mask = tbl["elongation"] < etamax 
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print(f"\n{premasklen-postmasklen} source(s) with "+
                  f"elongation >{etamax} removed")
        tbl_noelong = tbl.copy()
            
    ## (3) restrict based on maximum pixel area 
    if areamax:
        premasklen = len(tbl)
        mask = tbl["area"].value < areamax
        tbl = tbl[mask]    
        postmasklen = len(tbl)       
        if premasklen-postmasklen > 0:
            print(f"\n{premasklen-postmasklen} source(s) with "+
                  f"area >{areamax} pix**2 removed")
    
    vetted = tbl.copy()

    ## plot rejected/non-rejected candidates, if desired 
    if plot_rejections:
        dipoles = tbl_novetting[idx_rem]
        elongated_sources = tbl_nodipoles[tbl_nodipoles["elongation"] >= 
                                          etamax]
        large_sources = tbl_noelong[tbl_noelong["area"].value >= areamax]
        
        # colours below might be out of date (could be improved)
        __plot_rejected(sub_file=sub_file,
                        dipoles=dipoles, 
                        elongated_sources=elongated_sources, 
                        large_sources=large_sources, 
                        vetted=vetted,
                        dipole_width=dipole_width, 
                        etamax=etamax, 
                        areamax=areamax, 
                        toi=toi,  
                        dipole_color="black", 
                        elong_color="#be03fd", 
                        large_color="#fcc006", 
                        vetted_color="#53fca1",
                        cmap="coolwarm",
                        output=None)

    ## check that the number of detected sources is believable 
    if nsource_max:
        if len(tbl) > nsource_max:
           print(f"\nSourceCatalog contains {len(tbl)} sources across the "+
                 f"entire image, which is above the limit of {nsource_max}. "+
                 "The subtraction may have large image_registration errors. "+
                 "Exiting.")
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
    print(f"\n{len(tbl)} candidate(s) found and passed vetting.")
    
    ## get rid of "id" column and empty columns: 
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
        plot_transient(sub_file=sub_file, og_file=og_file, ref_file=ref_file, 
                       tbl=tbl, 
                       pixcoords=pixcoords, 
                       toi=toi, 
                       plots=plots, 
                       sub_scale=sub_scale, og_scale=og_scale, 
                       stampsize=stampsize, 
                       crosshair_og=crosshair_og, crosshair_sub=crosshair_sub, 
                       title_append=title, plotdir=plotdir)    
    return tbl


def transient_triplets(sub_file, og_file, ref_file, tbl, pixcoords=False, 
                       size=200, 
                       cropmode="extend", write=True, output=None,
                       plot=False, wide=True, cmap="bone", title=None, 
                       plotdir=None):
    """    
    Inputs:
        - difference image file
        - original science image file (can be background-subtracted or not)
        - original reference image file (can be background-subtracted or not)
        - a table of candidate transient source(s) found with 
          transient_detect() or some other tool (can be the name of a table 
          file or the table itself)
          note: if pixcoords=False, the table must have at least the columns 
          "ra" and "dec"; if pixcoords=True, the table must have at least the 
          columns "xcentroid" and "ycentroid"
        - whether to use the pixel coordinates of the transients when plotting 
          rather than the ra, dec (optional; default False; recommended to use
          in cases where it is not clear that the WCS of the science and 
          reference images matches exactly)         
        - the size of each 2D array in the triplet, i.e. the size of the 
          stamp to obtain around the transient(s) (optional; default 200 pix)
        - mode to use for crop_WCS (optional; default "extend"; options are
          "truncate" and "extend")
        - whether to write the produced triplet(s) to a .npy file (optional; 
          default True)
        - name for output .npy file (or some other file) (optional; default 
          set below)
        - whether to plot all of the triplets (optional; default False)
        
        only relevant if plot=True:
        - whether to plot the triplets as 3 columns, 1 row (horizontally wide)
          or 3 rows, 1 column (vertically tall) (optional; default wide=True)
        - colourmap to apply to all images in the triplet (optional; default 
          "bone")
        - a title to include in all plots AND to append to all filenames 
          (optional; default None)
        - name for the directory in which to store all plots (optional; 
          default set below)
    
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
    
    # get the targets and count how many in total
    if not(pixcoords):
        targets_sci= [tbl["ra"].data, tbl["dec"].data]
        targets_ref = [tbl["ra"].data, tbl["dec"].data]
    else:
        pix = [tbl["xcentroid"].data, tbl["ycentroid"].data]
        wsci = wcs.WCS(fits.getheader(sub_file))
        wref = wcs.WCS(fits.getheader(ref_file))
        rasci, decsci = wsci.all_pix2world(pix[0], pix[1], 1)
        raref, decref = wref.all_pix2world(pix[0], pix[1], 1)
        targets_sci = [rasci, decsci]
        targets_ref = [raref, decref]
          
    ntargets = len(targets_sci[0])
    triplets = []  
    
    ## iterate through all of the targets
    for n in range(ntargets):
        
        # get cropped data   
        sub_hdu = crop_WCS(sub_file, targets_sci[0][n], targets_sci[1][n], 
                           size=size, mode=cropmode, write=False)
        sub_data = sub_hdu.data        
        og_hdu = crop_WCS(og_file, targets_sci[0][n], targets_sci[1][n], 
                          size=size, mode=cropmode, write=False)
        og_data = og_hdu.data        
        ref_hdu = crop_WCS(ref_file, targets_ref[0][n], targets_ref[1][n], 
                           size=size, mode=cropmode, write=False)
        ref_data = ref_hdu.data       
        
        # make the triplet and add it to the list of triplets
        trip = np.array([og_data, ref_data, sub_data])
        triplets.append(trip)
        
        if plot: # plot this triplet, if desired
            if plotdir[-1] == "/":
                plotdir = plotdir[-1]
                
            __plot_triplet(og_file=og_file, 
                           sub_hdu=sub_hdu, og_hdu=og_hdu, ref_hdu=ref_hdu, 
                           n=n, ntargets=ntargets, 
                           wide=wide, cmap=cmap, title=title, plotdir=plotdir)

    ## build the final triplet stack, write it (optional)
    triplets = np.stack(triplets) # (3, size, size) --> (N, 3, size, size)   
    if write:
        if not(output):
            output = og_file.replace(".fits", "_candidates_triplets.npy")
        np.save(output, triplets)
        
    return triplets


def im_contains(im_file, ra, dec, exclusion=None):
    """Check if some coordinates [ra, dec] lie within the bounds of an image.
    
    Arguments
    ---------
    im_file : str
        Image of interest 
    ra, dec : float
        Right Ascension, Declination of a single source
    exclusion : float, optional
        Fraction of edge pixels to ignore (default None; see examples)

    Returns
    -------
    bool
        Whether the coordinates are in the image
        
    Examples
    --------    
    For a 1000 x 1000 image, to look for a source at (ra, dec) = (180.0, 30.0)
    and ignore the outermost 5% of edge pixels:
    
    >>> im_contains("foo.fits", 180.0, 30.0, 0.05)
    
    To **not** ignore any pixels (i.e. consider a set of coordinates to be 
    contained even if on the edgemost pixels):
    
    >>> im_contains("foo.fits", 180.0, 30.0)
    
    """
    data = fits.getdata(im_file)
    w = wcs.WCS(fits.getheader(im_file))
    
    xlims = [0, data.shape[1]]
    ylims = [0, data.shape[0]]
    
    if exclusion:
        xlims = [xlims[1]*exclusion, xlims[1]*(1.0-exclusion)]
        ylims = [ylims[1]*exclusion, ylims[1]*(1.0-exclusion)]
    
    try: 
        x, y = w.all_world2pix(ra, dec, 1)
    except wcs.NoConvergence:
        print("\nWCS could not converge, exiting")
        return 
    
    if (xlims[0] < x < xlims[1]) and (ylims[0] < y < ylims[1]): 
        return True
    else: 
        return

