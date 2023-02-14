#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Sat Dec 26 23:33:42 2020
.. @author: Nicholas Vieira
.. @transient.py

Transient detection, vetting, and converting candidate transients to 
"triplets". This represents the final step in an image differencing and 
transient detection pipeline. You can then use your favourite machine learning
algorithm to further vet the candidates. I use bogus-real adversarial 
artificial intelligence ``braai`` (https://github.com/dmitryduev/braai), which
is an example of a "real-bogus" classifier.
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
from photutils import detect_sources#, source_properties
from photutils.segmentation import SourceCatalog

# amakihi 
from .crop import crop_WCS
from .plotting import (__plot_rejected, __plot_triplet, __plot_distributions, 
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
                     etamax=1.8, areamax=300.0, nsource_max=50, 
                     toi=None, toi_sep_min=None, toi_sep_max=None,
                     write=True, output=None, 
                     plot_distributions=False,
                     plot_rejections=False,
                     plots=["zoom og", "zoom ref", "zoom diff"], 
                     pixcoords=False,
                     sub_scale=None, og_scale=None, stampsize=200.0, 
                     crosshair_og="#fe019a", crosshair_sub="#5d06e9",
                     title_append=None, plotdir=None):
    """

    Arguments
    ---------
    sub_file : str
        Difference image fits filename
    og_file : str
        Original science image fits filename (does not need to be background-
        subtracted)
    ref_file : str
        Reference image fits filename (does not need to be background-
        subtracted)
    mask_file : str, optional
        Mask image fits filename (default None)
    thresh_sigma : float, optional
        Sigma threshold for source detection with image segmentation (default
        5.0)
    pixelmin : float, optional
        *Minimum* pixel area of an isophote to be considered a potential 
        transient (default 20)   
    dipole_width : float, optional
        *Maximum* allowed width (in arcsec) for a source to be considered a 
        potential dipole residual from image differencing (default 2.0)
    dipole_fratio : float, optional
        *Maximum* allowed ratio of brightness between the brighter and dimmer
        of the two "poles" for a source to be a considered a potential dipole
        residual from image differencing (default 5.0)
    etamax : float, optional
        *Maximum* allowed elongation for an isophote to be considered a 
        potential transient (default 1.8)
    areamax : float, optional
        *Maximum* allowed area (in square pixels) for an isophote to be 
        considered a potential transient (default 300.0) 
    nsource_max : int, optional
        If more than `nsource_max` sources pass all vetting, the image 
        registration was probably of poor quality, and so the function exits 
        (default 50)
    toi : array_like, optional
        [ra, dec] for some target of interest (e.g. a candidate host galaxy) 
        such that only transients within some distance of this target are 
        considered (see `toi_sep_min`, `toi_sep_max`; default None)
    toi_sep_min, toi_sep_max : float, optional
        *Minimum* and *maximum* allowed separation from the `toi` for a source
        to be considered valid (default None)
    write : bool, optional
        Whether to write the table of candidate sources (default True)
    output : str, optional
        Name for output table (default 
        `og_file.replace(".fits", "_candidates.fits")`) 
    plot_distributions : bool, optional
        Whether to plot informative distributions of the elongation and area of
        the candidate transients (default False)
    plot_rejections : bool, optional
        Whether to plot a version of the difference image where accepted/
        rejected sources are labelled with markers (default False)
    plots : array_like, optional
        Array indicating which plots to produce, where options are "full", 
        "zoom og", "zoom ref", "zoom diff" (default 
        ["zoom og", "zoom ref", "zoom diff"], see notes)         
    pixcoords : bool, optional
        Whether to use the *pixel* coordinates of the transient(s) when 
        plotting rather than the WCS coordinates (default False)    
    sub_scale : {"asinh", "linear", "log"}, optional
        Scale to use for the difference image (default "asinh")
    og_scale : {"asinh", "linear", "log"}, optional
        Scale to use for the science/reference images (default "asinh")
    stampsize : float, optional
        Pixel size of the box to crop around the transient(s) (default 200.0)
    crosshair_og : str, optional
        Color for the crosshair in the science/reference image (default 
        "#fe019a" --> ~ hot pink)   
    crosshair_sub : str, optional
        Color for the crosshair in the difference image (default 
        "#5d06e9" --> ~ purple-blue) 
    title_append : str, optional
        A "title" to include in all plots' titles **and** the filenames of all
        saved figures (default None)
    plotdir : str, optional
        Name of directory in which to save plots (default None, in which case
        the stamps will be saved in the directory containing the corresponding
        fits image)

    Returns
    -------
    astropy.table.Table
        An astropy table containing vetted sources with their properties 
        (coordinates, area, elongation, separation from some target of 
        interest, etc.)

    Notes
    -----    
    Finds sources with flux > `thresh_sigma` * std, where std is the standard 
    deviation of the good pixels in the subtracted image. Sources must also be 
    made up of at least `pixelmin` pixels. From these, selects sources within 
    some elongation limit `etamax` to try to prevent obvious residuals from 
    being detected as sources. Also vets according to some maximum allowed 
    pixel area `areamax`. Finally, if more than `nsource_max` sourcs remain, 
    the image registration was probably faulty, and so the function terminates.
    
    Optionally plots informative figures showing the elongation distribution
    and area distribution of all sources (included those which are rejected).
    
    For each candidate transient source, also optionally plots the full 
    subtracted image and "postage stamps" of the transient on the original 
    science image, reference image and/or subtracted image. When indicating 
    which plots to produce via the `plots` argument (an `array_like`), the 
    options are: 
        
    1. "full" - the full-frame subtracted image
    2. "zoom og" - postage stamp of the original science image 
    3. "zoom ref" - postage stamp of the original reference image
    4. "zoom diff" - postage stamp of the subtracted image 

    If performing any of this plotting, it is recommended to use the pixel 
    coordinates (i.e. set `pixcoords = True`) when the quality of the 
    astrometric solution for the science or reference images cannot be assured.
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
    #cat = source_properties(data, segm, mask=mask) # photutils 0.8
    cat = SourceCatalog(data=data, segment_image=segm, 
                        mask=mask) # photutils >=1.1
    
    # do the same with the inverse of the image to look for dipoles
    segm_inv = detect_sources((-1.0)*data, thresh_sigma*std, 
                              npixels=pixelmin, mask=mask)
    #cat_inv = source_properties((-1.0)*data, segm_inv, mask=mask) # photutils 0.8
    cat_inv = SourceCatalog(data=(-1.0)*data, segment_image=segm_inv, 
                            mask=mask) # photutils >=1.1
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
                             title_append=title_append)

    
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
                 "The subtraction may have large image registration errors. "+
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
                       title_append=title_append, plotdir=plotdir)    
    return tbl


def transient_triplets(sub_file, og_file, ref_file, tbl, pixcoords=False, 
                       size=200, 
                       cropmode="extend", write=True, output=None,
                       plot=False, wide=True, cmap="bone", 
                       title_append=None, plotdir=None):
    """From some detected and preliminarily vetted transients, produce and 
    optionally write the final set of triplets.
    
    Arguments
    ---------
    sub_file : str
        Difference image fits filename
    og_file : str
        Original science image fits filename (does not need to be background-
        subtracted)
    ref_file : str
        Reference image fits filename (does not need to be background-
        subtracted)   
    tbl : str or astropy.table.Table
        Table of candidate transient source(s) found with `transient_detect()` 
        or some other tool (can be a filename or the table itself)
    pixcoords : bool, optional
        Whether to use the *pixel* coordinates of the transient(s) when 
        plotting rather than the WCS coordinates (default False; see notes)
    size : float, optional
        Pixel size of the box to crop around the transient(s) (default 200.0)
    cropmode : {"extend", "truncate"}, optional
        Mode to apply when cropping around the transient (default "extend")
    write : bool, optional
        Whether to write the produced triplets to a .npy file (default True)
    output : str, optional
        Name for output .npy file (default
        `og_file.replace(".fits", "_candidates_triplets.npy")`)
    plot : bool, optional
        Whether to plot **all** of the triplets (default False)
    wide : bool, optional
        Whether to plot each triplet as 1 row, 3 columns (`wide == True`) or 3
        rows, 1 column (`wide == False`) (default True)
    cmap : str, optional
        Colormap to apply to all images in the triplets (default "bone")
    title_append : str, optional
        A "title" to include in all plots' titles **and** filenames (default 
        None)
    plotdir : str, optional
        Directory in which to save the plots (default is the directory of 
        `og_file`)
    
    Returns
    -------
    np.ndarray
        Array with shape `(N, 3, size, size)`, where `N` is the number of rows 
        in the input `tbl` (i.e., number of candidate transients), `size` is 
        the dimension of the box used when cropping around candidates, and the 
        3 sub-arrays represent cropped sections of the science, reference, 
        and difference image, in that order
    
    Notes
    ------
    If `pixcoords == False`, `tbl` must contain *at least* the columns *ra* and
    *dec*. If true, must contain *at least* the columns *xcentroid* and 
    *ycentroid*. It is recommended to use the pixel coordinates (i.e. set 
    `pixcoords = True`) when the quality of the astrometric solution for the 
    science or reference images cannot be assured.
    
    When indicating which plots to produce via the `plots` argument (an 
    `array_like`), the options are: 
        
    1. "full" - the full-frame subtracted image
    2. "zoom og" - postage stamp of the original science image 
    3. "zoom ref" - postage stamp of the original reference image
    4. "zoom diff" - postage stamp of the subtracted image 
    
    From a table of candidate transients and the corresponding difference, 
    science, and reference images, builds a triplet: three "postage stamps"
    around the candidate showing the science image, reference image, and 
    difference image. Optionally writes the triplet to a .npy file.
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
                           wide=wide, cmap=cmap, 
                           title_append=title_append, plotdir=plotdir)

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

