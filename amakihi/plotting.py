#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Fri Dec 25 21:38:44 2020
.. @author: Nicholas Vieira
.. @plotting.py

Functions for plotting images, triplets, etc.

Also contains several functions for plotting machine learning diagnostics.

See the source code for utility plotting functions called by:

- :obj:`background` module
- :obj:`masking` module
- :obj:`imalign` module
- :obj:`ePSF` module
- :obj:`amakihi` (image differencing) module
- :obj:`transient` module

"""

# misc
import numpy as np
import re

# astropy
from astropy.io import fits
from astropy import wcs
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.table import Table#, Column

# amakihi 
from .crop import crop_WCS

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
### CHANGING PLOTTING BACKEND (needs to be "agg" on remote servers) ###########

def to_agg():
    """Switch matplotlib backend to 'agg'"""
    plt.switch_backend("agg")
    
def to_Qt4Agg():
    """Switch matplotlib backend to 'Qt4Agg'"""
    plt.switch_backend('Qt4Agg')
    

###############################################################################
#### MISCELLANEOUS PLOTTING ###################################################
    
def plot_image(im_file, mask_file=None, 
               scale="linear", cmap="bone", 
               label=None, title=None, 
               output=None, 
               target_large=None, target_small=None,
               crosshair_large="black", crosshair_small="#fe019a"):
    """Plot an image of any kind, with lots of options.
    
    Arguments
    ---------
    im_file : str
        Filename for fits image of interest
    mask_file : str, optional
        Filename for mask fits image (default None)
    scale : {"linear", "log", "asinh"}, optional
        Scale to use for the image (default "linear")
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colourmap to use for the image (default "bone")
    label : str, optional
        Label to apply to the colorbar (default None, in which case will be 
        ADU, log(ADU), or asinh(ADU) depending on `scale`)
    title : str, optional
        Title for the image (default None)
    output : str, optional
        Name for output figure (default 
        `im_file.replace(".fits", f"_{scale}.png")`)
    target_large : array_like, optional
        [ra, dec] for a target at which to place a large crosshair (default 
        None)
    target_small : array_like, optional
        [ra, dec] for a target at which to place a small crosshair (default 
        None)
    crosshair_large : str, optional
        Color for the large crosshair (default "black")
    crosshair_small : str, optional
        Color for the small crosshair (default "#fe019a" --> ~ hot pink)
    """
    
    # load in data
    image_data = fits.getdata(im_file)
    image_header = fits.getheader(im_file)
    
    # set figure dimensions
    plt.figure(figsize=(14,13))
    
    # show WCS     
    w = wcs.WCS(image_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    if mask_file:
        mask = fits.getdata(mask_file)
        image_data_masked = np.ma.masked_where(mask, image_data)
        image_data = np.ma.filled(image_data_masked, 0)
    
    if scale == "linear": # linear scale
        mean, median, std = sigma_clipped_stats(image_data)
        plt.imshow(image_data, vmin=mean-5*std, vmax=mean+9*std, cmap=cmap, 
                   aspect=1, interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        if label == None:
            cb.set_label(label="ADU", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    elif scale == "log": # log scale 
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
    
    if not(title): # set title if not given
        topfile = re.sub(".*/", "", im_file)
        title = topfile.replace(".fits","")
        title = r"$\mathtt{"+title.replace("_","\_")+"}$"+" data"
    if not(output): # set output figure name if not given
        output = im_file.replace(".fits", f"_{scale}.png")

    # crosshairs
    if target_large: # large crosshair
        ra, dec = target_large
        plt.gca().plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
               transform=plt.gca().get_transform('icrs'), linewidth=2, 
               color=crosshair_large, marker="")
        plt.gca().plot([ra, ra], [dec+10.0/3600.0, dec+5.0/3600.0], 
               transform=plt.gca().get_transform('icrs'),  linewidth=2, 
               color=crosshair_large, marker="")
    if target_small: # small crosshair
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


def plot_transient_stamp(im_file, target, size=200.0, cropmode="truncate", 
                         scale="asinh", cmap="bone", crosshair="#fe019a",
                         label=None, title=None, 
                         output=None, 
                         toi=None):
    """Plot a "postage stamp" around a transient source of interest, where the
    input image can be the science, reference, difference, or whatever image.
    
    Arguments
    ---------
    im_file : str
        Filename for fits image of interest
    target : array_like
        [ra, dec] of transient source of interest
    size : float, optional
        Pixel size of the box to crop around the transient (default 200.0)
    cropmode : {"truncate", "extend"}
        Mode to use for `crop_WCS()` (when cropping to select the transient) if 
        the crop goes out of bounds (default "truncate")
    scale : {"linear", "log", "asinh"}, optional
        Scale to use for the image (default "linear")
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colourmap to use for the image (default "bone")
    crosshair : str, optional
        Color for the crosshair(s) (default "#fe019a" --> ~ hot pink)
    label : str, optional
        Label to apply to the colorbar (default None, in which case will be 
        ADU, log(ADU), or asinh(ADU) depending on `scale`)
    title : str, optional
        Title for the image (default None)
    output : str, optional
        Name for output figure (default 
        `im_file.replace(".fits", f"_{scale}.png")`)
    toi : array_like, optional
        [ra, dec] of some other source of interest such as a candidate host 
        galaxy for the transient such that another crosshair is plotted at its
        location (default None)
    """
    
    # crop to target
    ra, dec = target
    imhdu = crop_WCS(im_file, ra, dec, size, mode=cropmode, write=False)
    if imhdu == None:
        print("\nCropping was unsucessful, so a stamp cannot be produced. "+
              "Exiting. ")
        return
    
    # load in data    
    image_data = imhdu.data
    image_header = imhdu.header

    # set figure dimensions
    plt.figure(figsize=(14,13))
    
    # show WCS     
    w = wcs.WCS(image_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    if scale == "linear": # linear scale 
        mean, median, std = sigma_clipped_stats(image_data)
        plt.imshow(image_data, vmin=mean-5*std, vmax=mean+9*std, cmap=cmap, 
                   aspect=1, interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        if label == None:
            cb.set_label(label="ADU", fontsize=16)
        else:
            cb.set_label(label=label, fontsize=16)
        
    elif scale == "log": # log scale
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
    
    if not(title): # set title if not given
        topfile = re.sub(".*/", "", im_file)
        title = topfile.replace(".fits","")
        title = r"$\mathtt{"+title.replace("_","\_")+"}$"+" data (stamp)"
    if not(output): # set output figure name if not given
        output = im_file.replace(".fits", f"_stamp_{scale}.png")

    # crosshair
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


def plot_transient(sub_file, og_file, ref_file, tbl, 
                   pixcoords=False,
                   toi=None, 
                   plots=["zoom og", "zoom ref", "zoom diff"], 
                   sub_scale="asinh", og_scale="asinh", 
                   stampsize=200.0, 
                   crosshair_og="#fe019a", crosshair_sub="#5d06e9", 
                   title_append=None, plotdir=None):
    """Plot a candidate transient source, with tons of options. 
    
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
    toi : array_like, optional
        [ra, dec] of some other source of interest such as a candidate host 
        galaxy for the transient such that another crosshair is plotted at its 
        location (default None)
    plots : array_like, optional
        Array indicating which plots to produce, where options are "full", 
        "zoom og", "zoom ref", "zoom diff" (default 
        ["zoom og", "zoom ref", "zoom diff"]; see notes)    
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

    Notes
    -----
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
    
    """

    # check if table is a filename or pre-loaded table
    if type(tbl) == str:
        tbl = Table.read(tbl, format="ascii")
    
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
    
    if not(type(plots) in (list, np.ndarray, np.array)):
        print("\ntransient_plot() was called, but no plots were requested "+
              "via the <plots> arg. Exiting.")
        return
    elif len(plots) == 0:
        print("\ntransient_plot() was called, but no plots were requested "+
              "via the <plots> arg. Exiting.")
        return
    
    ntargets = len(targets_sci[0])
    for n in range(ntargets):
        if ntargets < 100:
            if n < 10: nstr = "0"+str(n)
            else: nstr = str(n)
        else:
            if n < 10: nstr = "00"+str(n)
            elif n < 100: nstr = "0"+str(n)
            else: nstr = str(n)
                
        ### set titles 
        if title_append:
            title = f"{title_append} difference image: candidate {nstr}"
            title_og = f"{title_append} original image: candidate {nstr}"
            title_ref = f"{title_append} reference image: candidate {nstr}"
        else:
            title = f"difference image: candidate {nstr}"
            title_og = f"original image: candidate {nstr}"
            title_ref = f"reference image: candidate {nstr}"
    
        # setting output figure directory
        if plotdir:
            if plotdir[-1] == "/": plotdir = plotdir[-1]        
        
        ### full-frame difference image #######################################
        if "full" in plots:
            
            # set output figure title
            if title_append:
                full_output = sub_file.replace(".fits", 
                             f"_{title_append}_diff_candidate{nstr}.png")                
            else:
                full_output = sub_file.replace(".fits", 
                                               f"_diff_candidate{nstr}.png")                
            if plotdir:
                full_output = f'{plotdir}/{re.sub(".*/", "", full_output)}'
                
            plot_image(im_file=sub_file, 
                       scale=sub_scale, 
                       cmap="coolwarm", 
                       label="", 
                       title=title, 
                       output=full_output, 
                       target_large=toi,
                       target_small=[targets_sci[0][n], targets_sci[1][n]],
                       crosshair_small=crosshair_sub)
        
        ### small region of science image ####################################
        if "zoom og" in plots:
            # set output figure name
            if title_append:
                zoom_og_output = og_file.replace(".fits", 
                        f"_{title_append}_zoomed_sci_candidate{nstr}.png")
            else:
                zoom_og_output = og_file.replace(".fits", 
                                        f"_zoomed_sci_candidate{nstr}.png")                
            if plotdir:
                zoom_og_output = f'{plotdir}/{re.sub(".*/","",zoom_og_output)}'
                
            plot_transient_stamp(im_file=og_file, 
                                 target=[targets_sci[0][n], targets_sci[1][n]], 
                                 size=stampsize, scale=og_scale, 
                                 cmap="viridis", crosshair=crosshair_og, 
                                 title=title_og, 
                                 output=zoom_og_output, 
                                 toi=toi)

        ### small region of reference image ###################################
        if "zoom ref" in plots:
            # set output figure name
            if title_append:
                zoom_ref_output = og_file.replace(".fits", 
                        f"_{title_append}_zoomed_ref_candidate{nstr}.png")
            else:
                zoom_ref_output = og_file.replace(".fits", 
                                        f"_zoomed_ref_candidate{nstr}.png")
            if plotdir:
                zoom_ref_output = f'{plotdir}/'
                zoom_ref_output += f'{re.sub(".*/", "", zoom_ref_output)}'
                
            plot_transient_stamp(im_file=ref_file, 
                                 target=[targets_ref[0][n], targets_ref[1][n]], 
                                 size=stampsize, scale=og_scale, 
                                 cmap="viridis", crosshair=crosshair_og, 
                                 title=title_ref, 
                                 output=zoom_ref_output, 
                                 toi=toi)
        
        ### small region of difference image ##################################
        if "zoom diff" in plots:
            # set output figure name
            if title_append:
                zoom_diff_output = sub_file.replace(".fits", 
                        f"_{title_append}_zoomed_diff_candidate{nstr}.png")
            else:
                zoom_diff_output = sub_file.replace(".fits", 
                                        f"_zoomed_diff_candidate{nstr}.png")
            if plotdir:
                zoom_diff_output = f'{plotdir}/'
                zoom_diff_output += f'{re.sub(".*/", "", zoom_diff_output)}'
                
            plot_transient_stamp(im_file=sub_file, 
                                 target=[targets_sci[0][n], targets_sci[1][n]], 
                                 size=stampsize, scale=sub_scale, 
                                 cmap="coolwarm", label="ADU", 
                                 crosshair=crosshair_sub, 
                                 title=title, 
                                 output=zoom_diff_output, 
                                 toi=toi)    


def plot_triplets(tripfile, 
                  outdir, 
                  tabfile=None, 
                  images=None,
                  
                  wide=True, 
                  
                  SRDlabel=False,
                  titles=None,
                  
                  ticks_axes=True, 
                  scalebar=True, 
                  pixscale=0.185,
                  
                  crosshair_from_table=False,
                  crosshair_peak=False, 
                  crosshair_color="#fe01b1",
                  crosshair_thick=2.5,
                  
                  cmap="bone", fmt="png"):
    """Plot a set of triplets.

    Arguments
    ---------
    tripfile : str
        .npy file containing triplets
    outdir : str
        Directory in which to store output plots
    tabfile : str, optional
        Table of candidate transients corresponding to `tripfile` (must be a 
        .csv or .fits file) (default None)
    images : list, optional
        **List** of .fits images from which to extract WCS information for 
        plotting, **if** `tabfile` was provided (default None --> use pixel 
        coordinates)
    wide : bool, optional
        Plot the triplets as 3 columns, 1 row (horizontally wide)? Else, plot 
        as 3 rows, 1 column (vertically tall) (default True)
    SRDlabel : bool, optional
        Label the science, reference, and difference images with 'science', 
        'reference', 'difference'? (default False)
    titles : list, optional
        **List** of titles to include in corresponding plots; also used for 
        setting output filenames (default None)
    ticks_axes : bool, optional
        Include ticks, tick labels, and axis labels? (default True)
    scalebar : bool, optional
        Include a bar showing the scale? (default True)
    pixscale : float, optional
        Pixel scale of the image in arcseconds per pixel; only relevant if 
        `scalebar = True` (default 0.185 = pixel scale for MegaCam)
    crosshair_from_table : bool, optional
        Label the science, reference, and difference images with a crosshair 
        using the location of the source given via `images` and `tabfile`? 
        (default False)
    crosshair_peak : bool, optional
        Label the science, reference, and difference images with a crosshair 
        using the peak of the image? (default False)
    crosshair_color : str, optional
        Color for crosshair (default "#fe01b1" = bright pink)
    crosshair_thick : float, optional
        Thickness of crosshairs (default 2.5)
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colormap to use in plots (default "bone")
    fmt : {"png", "pdf"}, optional
        Filetype for the plots (default "png")
        
    """

    from matplotlib_scalebar.scalebar import ScaleBar

    ## load in triplets    
    triplets = np.load(tripfile, mmap_mode="r")
    
    ## load in corresponding table if one is given    
    if tabfile:
        if ".fits" in tabfile:
            tbl = Table.read(tabfile, format="ascii")
        elif ".csv" in tabfile:
            tbl = Table.read(tabfile, format="ascii.csv")
        else:
            raise ValueError("tabfile must be of filetype .csv or .fits, "+
                             f"did not recognize {tabfile}")
        
    
    # parameters for the plots and subplots based on whether wide=True
    plot_dict = {True:[(17,7), 131, 132, 133],
                 False:[(7, 17), 311, 312, 313]}

    ## write candidate number as a string for filenaming (if needed)
    if type(titles) == type(None):
        nstrs = []
        if len(triplets) < 100:
            for i in range(len(triplets)):
                if i<10: nstrs.append(f"0{i}")
                else: nstrs.append(str(i)) 
        elif len(triplets) < 1000:
            for i in range(len(triplets)):
                if i<10: nstrs.append(f"00{i}")
                elif i<100: nstrs.append(f"0{i}")
                else: nstrs.append(str(i)) 
        else:
            for i in range(len(triplets)):
                if i<10: nstrs.append(f"000{i}")
                elif i<100: nstrs.append(f"00{i}")
                elif i<1000: nstrs.append(f"0{i}")
                else: nstrs.append(str(i)) 

    ## set crosshair width/offset from coordinate center (if needed)
    if crosshair_from_table or crosshair_peak:
        if triplets.shape[1] == 3:
            triplet_temp = triplets[0][0]
        else:
            triplet_temp = triplets[0][:,:,0]
        crosshair_width = 0.2*triplet_temp.shape[0] # width of crosshair
        crosshair_offset = 0.1*triplet_temp.shape[0] # offset from center 
    
    ## loop over each triplet 
    for i in range(len(triplets)):        
        # check shape of the array ((N, 3, X, Y) or (N, X, Y, 3))
        if triplets.shape[1] == 3:
            og_data, ref_data, sub_data = triplets[i]
        else:
            og_data = triplets[i][:,:,0]
            ref_data = triplets[i][:,:,1]
            sub_data = triplets[i][:,:,2]
            
        ## see if an image *and* table were provided 
        if not(type(images) == type(None) or type(tabfile) == type(None)):
            ra = tbl["ra"][i]; dec = tbl["dec"][i]
            new_hdu = crop_WCS(images[i], ra, dec, size=og_data.shape[0],
                               write=False)
            w = wcs.WCS(new_hdu.header)
        else:
            w = None
            
        ## plot
        fig = plt.figure(figsize=plot_dict[wide][0])      
        # science image
        ax = fig.add_subplot(plot_dict[wide][1], projection=w)
        mean, median, std = sigma_clipped_stats(og_data)
        #og_norm = simple_norm(og_data, "asinh")
        ax.imshow(og_data, vmin=mean-5*std, vmax=mean+9*std, origin='lower', 
                  cmap=cmap)                   
        # reference image 
        ax2 = fig.add_subplot(plot_dict[wide][2], projection=w)
        mean, median, std = sigma_clipped_stats(ref_data)
        #ref_norm = simple_norm(ref_data, "asinh")
        ax2.imshow(ref_data, vmin=mean-5*std, vmax=mean+9*std, origin='lower', 
                   cmap=cmap)        
        # difference image 
        ax3 = fig.add_subplot(plot_dict[wide][3], projection=w)
        mean, median, std = sigma_clipped_stats(sub_data)
        #sub_norm = simple_norm(sub_data, "asinh")
        ax3.imshow(sub_data, vmin=mean-5*std, vmax=mean+9*std, origin='lower', 
                   cmap=cmap) 
        
        ## ticks and axis labels? ticks are forced if <images>, <table> given
        if not(type(images) == type(None) or type(tabfile) == type(None)):
            if wide: # RA under middle image, DEC left of leftmost image 
                ax.coords["ra"].set_ticklabel_visible(False)
                ax.coords["dec"].set_ticklabel(size=26, 
                         exclude_overlapping=True)
                ax.set_ylabel("Dec (J2000)", fontsize=24)
                ax2.coords["ra"].set_ticklabel(size=26, 
                          exclude_overlapping=True)
                ax2.coords["dec"].set_ticklabel_visible(False)
                ax2.set_xlabel("RA (J2000)", fontsize=24)
                ax2.coords["ra"].set_major_formatter("dd:mm:ss")
                ax3.coords["ra"].set_ticklabel_visible(False)
                ax3.coords["dec"].set_ticklabel_visible(False)
            else: # RA under bottom image, DEC left of middle image
                ax.coords["ra"].set_ticklabel_visible(False)
                ax.coords["dec"].set_ticklabel_visible(False)        
                ax2.coords["ra"].set_ticklabel_visible(False)        
                ax2.coords["dec"].set_ticklabel(size=26, 
                          exclude_overlapping=True)
                ax2.set_ylabel("Dec (J2000)", fontsize=24)
                ax3.coords["ra"].set_ticklabel(size=26, 
                          exclude_overlapping=True)
                ax3.coords["dec"].set_ticklabel_visible(False)    
                ax3.set_xlabel("RA (J2000)", fontsize=24)
        elif ticks_axes:
            if wide:
                ax.set_ylabel("Pixels", fontsize=24)
                ax2.set_xlabel("Pixels", fontsize=24)
            else:
                ax2.set_ylabel("Pixels", fontsize=24)
                ax3.set_xlabel("Pixels", fontsize=24)
            ax.tick_params(which='major', labelsize=26)
            ax2.tick_params(which='major', labelsize=26) 
            ax3.tick_params(which='major', labelsize=26)
            ax.set_yticks(ax.get_xticks()[1:-1])
            ax2.set_yticks(ax2.get_xticks()[1:-1])
            ax3.set_yticks(ax3.get_xticks()[1:-1])
        else: 
            ax.set_xticks([])
            ax.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3.set_xticks([])
            ax3.set_yticks([])
            
        ## scalebar?  
        if scalebar:
            scalebar = ScaleBar(pixscale, units="''", dimension='angle', 
                                color="k", box_alpha=1.0, 
                                label_loc="top", 
                                height_fraction=0.015,
                                font_properties={"size":23})
            ax2.add_artist(scalebar)

        ## crosshair at coords supplied by table?
        if crosshair_from_table: 
            if not(type(images) == type(None) or type(tabfile) == type(None)):
                x, y = w.all_world2pix(tbl["ra"][i], tbl["dec"][i], 1)
                crosshair_x = [x-crosshair_offset-crosshair_width,
                               x-crosshair_offset] # xmin, xmax of crosshair
                crosshair_y = [y+crosshair_offset, # ymin, ymax of crosshair
                               y+crosshair_offset+crosshair_width]

                ax.hlines(y, xmin=crosshair_x[0], xmax=crosshair_x[1], 
                          colors=crosshair_color, lw=crosshair_thick, 
                          transform=ax.get_transform('pixel'))
                ax.vlines(x, ymin=crosshair_y[0], ymax=crosshair_y[1], 
                          colors=crosshair_color, lw=crosshair_thick,
                          transform=ax.get_transform('pixel'))
                ax2.hlines(y, xmin=crosshair_x[0], xmax=crosshair_x[1], 
                          colors=crosshair_color, lw=crosshair_thick, 
                          transform=ax2.get_transform('pixel'))
                ax2.vlines(x, ymin=crosshair_y[0], ymax=crosshair_y[1], 
                          colors=crosshair_color, lw=crosshair_thick,
                          transform=ax2.get_transform('pixel'))
                ax3.hlines(y, xmin=crosshair_x[0], xmax=crosshair_x[1], 
                          colors=crosshair_color, lw=crosshair_thick, 
                          transform=ax3.get_transform('pixel'))
                ax3.vlines(x, ymin=crosshair_y[0], ymax=crosshair_y[1], 
                          colors=crosshair_color, lw=crosshair_thick,
                          transform=ax3.get_transform('pixel'))
            else:
                print("\nEither a table or a list of .fits images was not "+
                      "provided, so no crosshair can be drawn.")
                
        ## OR, crosshair at peak of diff image?
        elif crosshair_peak:            
            ind = np.unravel_index(np.argmax(sub_data, axis=None), 
                                   sub_data.shape) # find the peak's indices
            if not(type(images) == type(None) or type(tabfile) == type(None)):
                crosshair_x = [ind[1]-crosshair_offset-crosshair_width,
                               ind[1]-crosshair_offset] 
                crosshair_y = [ind[0]+crosshair_offset, 
                               ind[0]+crosshair_offset+crosshair_width]
                ax.hlines(ind[0], xmin=crosshair_x[0], xmax=crosshair_x[1], 
                          colors=crosshair_color, lw=crosshair_thick, 
                          transform=ax.get_transform('pixel'))
                ax.vlines(ind[1], ymin=crosshair_y[0], ymax=crosshair_y[1], 
                          colors=crosshair_color, lw=crosshair_thick,
                          transform=ax.get_transform('pixel'))
                ax2.hlines(ind[0], xmin=crosshair_x[0], xmax=crosshair_x[1], 
                          colors=crosshair_color, lw=crosshair_thick, 
                          transform=ax2.get_transform('pixel'))
                ax2.vlines(ind[1], ymin=crosshair_y[0], ymax=crosshair_y[1], 
                          colors=crosshair_color, lw=crosshair_thick,
                          transform=ax2.get_transform('pixel'))
                ax3.hlines(ind[0], xmin=crosshair_x[0], xmax=crosshair_x[1], 
                          colors=crosshair_color, lw=crosshair_thick, 
                          transform=ax3.get_transform('pixel'))
                ax3.vlines(ind[1], ymin=crosshair_y[0], ymax=crosshair_y[1], 
                          colors=crosshair_color, lw=crosshair_thick,
                          transform=ax3.get_transform('pixel'))
                
            else:
                ax.axhline(ind[0], 0.2, 0.4, color=crosshair_color, 
                           lw=crosshair_thick)
                ax.axvline(ind[1], 0.6, 0.8, color=crosshair_color, 
                           lw=crosshair_thick)
                ax2.axhline(ind[0], 0.2, 0.4, color=crosshair_color, 
                            lw=crosshair_thick)
                ax2.axvline(ind[1], 0.6, 0.8, color=crosshair_color, 
                            lw=crosshair_thick)        
                ax3.axhline(ind[0], 0.2, 0.4, color=crosshair_color, 
                            lw=crosshair_thick)
                ax3.axvline(ind[1], 0.6, 0.8, color=crosshair_color, 
                            lw=crosshair_thick)
            
        ## titles/filenames
        # if "science", "reference", "difference" titles are requested
        if SRDlabel:
            ax.set_title('$science$', fontsize=24)
            ax2.set_title('$reference$', fontsize=24)
            ax3.set_title('$difference$', fontsize=24)            
            if not(type(titles) == type(None)): # title can be a supertitle above the others
                plt.suptitle(titles[i], fontsize=24)
                # get rid of illegal chars
                title_nosp = [t.replace(" ","_") for t in titles.copy()]
                title_nosp = [t.replace(",","") for t in title_nosp.copy()]
                title_nosp = [t.replace("\n","_") for t in title_nosp.copy()]
                title_nosp = [t.replace("$","") for t in title_nosp.copy()]
                plotname = tripfile.replace(".npy", 
                                             f"_{title_nosp[i]}.{fmt}")                          
            else: # no supertitle will be used
                plotname = tripfile.replace(".npy", 
                                            f"_candidate{nstrs[i]}.{fmt}")                
                
        # OR, if a title is given but no SRD label is requested 
        elif not(type(titles) == type(None)):
            if wide:
                ax2.set_title(titles[i], fontsize=24)
            else:
                ax.set_title(titles[i], fontsize=24)
            # get rid of illegal chars
            title_nosp = [t.replace(" ","_") for t in titles.copy()]
            title_nosp = [t.replace(",","") for t in title_nosp.copy()]
            title_nosp = [t.replace("\n","_") for t in title_nosp.copy()]
            title_nosp = [t.replace("$","") for t in title_nosp.copy()]
            plotname = tripfile.replace(".npy", f"_{title_nosp[i]}.{fmt}") 
               
        # no table nor title nor SRDlabel requested
        else:
            plotname = tripfile.replace(".npy", f"_candidate{nstrs[i]}.{fmt}")
        
        ## save and close the figure 
        plotname = f'{outdir}/{re.sub(".*/", "", plotname)}' 
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()



###############################################################################
### background ################################################################

def __plot_bkg(im_header, bkg_img_masked, scale, output):
    """Plot the background image. 
    
    Arguments
    ---------
    im_header : astropy.io.fits.header.Header
        Image fits header
    bkg_img_masked : array_like
        2D array representing the background image 
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plot    
    output : str
        Name for output figure
    """

    # verify scale
    if not(scale in ("linear", "log", "asinh")):
        raise ValueError('scale must be one of ("linear", "log", "asinh"), '+
                         'but argument supplied was {scale}')
    
    # set figure dimensions
    plt.figure(figsize=(14,13))
        
    # show WCS     
    w = wcs.WCS(im_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    # plot the background image with desired scaling
    if scale == "linear": # linear scale
        plt.imshow(bkg_img_masked, cmap='bone', aspect=1, 
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        
    elif scale == "log": # log scale 
        bkg_img_log = np.log10(bkg_img_masked)
        lognorm = simple_norm(bkg_img_log, "log", percent=99.0)
        plt.imshow(bkg_img_log, cmap='bone', aspect=1, norm=lognorm,
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        
    elif scale == "asinh":  # asinh scale
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
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def __plot_bkgsubbed(im_header, bkgsub_img_masked, scale, output):
    """Plot the background-SUBTRACTED image.

    Arguments
    ---------
    im_header : astropy.io.fits.header.Header
        Image fits header
    bkgsub_img_masked : array_like
        2D array representing the background image 
    scale : {"linear", "log", "asinh"}
        Scale to apply to the plot    
    output : str
        Name for output figure
    """
    
    # verify scale
    if not(scale in ("linear", "log", "asinh")):
        raise ValueError('scale must be one of ("linear", "log", "asinh"), '+
                         'but argument supplied was {scale}')

    # set figure dimensions
    plt.figure(figsize=(14,13))

    # show WCS     
    w = wcs.WCS(im_header)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)

    # plot with desired scaling     
    if scale == "linear": # linear scale
        plt.imshow(bkgsub_img_masked, cmap='bone', aspect=1, 
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        
    elif scale == "log": # log scale
        bkgsub_img_masked_log = np.log10(bkgsub_img_masked)
        lognorm = simple_norm(bkgsub_img_masked_log, "log", percent=99.0)
        plt.imshow(bkgsub_img_masked_log, cmap='bone', aspect=1, 
                   norm=lognorm, interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label=r"$\log(ADU)$", fontsize=16)
        
    elif scale == "asinh":  # asinh scale
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
    plt.savefig(output, bbox_inches="tight")
    plt.close()   


###############################################################################
### masking ###################################################################
    
def __plot_mask(hdr, newmask, title, output):
    """Plot a mask (box mask or saturation mask) as a binary image (black or 
    white).
    
    Arguments
    ---------
    hdr : astropy.io.fits.header.Header
        Header for the image (needed for WCS info)
    newmask : array_like
        Mask data as 2D array
    title : str
        Title to give the plot
    output : str
        Name for output figure 
    """

    # set figure dimensions
    plt.figure(figsize=(14,13))
    
    # show WCS      
    w = wcs.WCS(hdr)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    # plot mask as a binary image (black or white)
    plt.imshow(newmask, cmap='binary_r', aspect=1, interpolation='nearest', 
               origin='lower')
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    plt.title(title, fontsize=15)
    plt.savefig(output) 
    plt.close()
    
      
###############################################################################
### imalign ###################################################################

def __plot_sources(science_data, template_data, science_hdr, template_hdr, 
                   science_list, template_list, scale, color, output):    
    """Given some source image and template image, plot the locations of cross-
    matched sources in each, side-by-side for comparison.
    
    Arguments
    ---------
    science_data, template_data : array_like
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
    wsci = wcs.WCS(science_hdr)
    wtmp = wcs.WCS(template_hdr)
    
    # get miscellaneous coords
    xsci = [s[0] for s in science_list]
    ysci = [s[1] for s in science_list]
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
        mean, median, std = sigma_clipped_stats(science_data)
        ax.imshow(science_data, cmap='bone', vmin=mean-5*std, vmax=mean+9*std,
                  aspect=1, interpolation='nearest', origin='lower')      
    elif scale == "log": # log scale
        science_data_log = np.log10(science_data)
        lognorm = simple_norm(science_data_log, "log", percent=99.0)
        ax.imshow(science_data_log, cmap='bone', aspect=1, norm=lognorm,
                  interpolation='nearest', origin='lower')      
    elif scale == "asinh":  # asinh scale
        science_data_asinh = np.arcsinh(science_data)
        asinhnorm = simple_norm(science_data_asinh, "asinh")
        ax.imshow(science_data_asinh, cmap="bone", aspect=1, 
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


###############################################################################
### ePSF ######################################################################
    
def __plot_ePSF(epsf_data, output):
    """Plot the effective Point-Spread Function.
    
    Arguments
    ---------
    epsf_data : np.ndarray
        ePSF as a 2D array
    output : str
        Name for output figure
    """
    
    # set figure dimensions
    plt.figure(figsize=(10,9))
    
    # plot it
    plt.imshow(epsf_data, origin='lower', aspect=1, cmap='bone',
               interpolation="nearest")
    plt.xlabel("Pixels", fontsize=16)
    plt.ylabel("Pixels", fontsize=16)
    plt.title("effective Point-Spread Function", fontsize=16)
    plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
    plt.rc("xtick",labelsize=16) # not working?
    plt.rc("ytick",labelsize=16)

    plt.savefig(output, bbox_inches="tight")
    plt.close()


def __plot_convolve_self(conv, mask, hdr, title, output):
    """Plot an image which has just been convolved with its own effective PSF 
    (very crude, only really useful for bug-testing at the moment).
    
    Arguments
    ---------
    conv : np.ndarray
        Image which has been convolved with its own effective PSF 
    mask : np.ndarray
        Mask to apply when plotting
    hdr : astropy.io.fits.header.Header
        Header for the original fits image
    title : str
        Title to give to the plot
    output : str
        Name for output figure
    """
    
    # set figure dimensions
    plt.figure(figsize=(14,13))
    
    # show WCS
    w = wcs.WCS(hdr)
    ax = plt.subplot(projection=w)
    
    # plot it
    mean, med, std = sigma_clipped_stats(conv, mask=mask)
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    plt.imshow(conv, vmin=med-8*std, vmax=mean+8*std, cmap="bone",
               aspect=1, interpolation='nearest', origin='lower')

    # pretty plotting
    cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cb.set_label(label="ADU", fontsize=16)
    cb.ax.tick_params(which='major', labelsize=15)       
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    plt.title(title, fontsize=15)

    plt.savefig(output, bbox_inches="tight")
    plt.close()


###############################################################################
### amakihi ###################################################################

def __plot_hotpants(sub, hdr, mean_diff, std_diff, scale,
                    target_large, target_small, output):
    """Plot the difference image produced by hotpants.
    
    Arguments
    ---------
    sub : np.ndarray
        Difference image
    hdr : astropy.io.fits.header.Header
        Science image's header
    mean_diff : float
        Mean of the difference image's good pixels
    std_diff : float
        Standard deviation of the difference image's good pixels
    target_large : array_like
        [ra, dec] for some target of interest (e.g. a potential transient) 
        at which to draw a large crosshair
    target_small : array_like
        [ra, dec] for some other target of interest (e.g. the candidate host
        galaxy of some target of interest) at which to draw a small crosshair
    output : str
        Name for output figure
    """
    
    # set figure dimensions
    plt.figure(figsize=(14,13))
    
    # show WCS      
    w = wcs.WCS(hdr)
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
    
    if scale == "linear": # linear scale
        plt.imshow(sub, cmap='coolwarm', vmin=mean_diff-3*std_diff, 
                   vmax=mean_diff+3*std_diff, aspect=1, 
                   interpolation='nearest', origin='lower')
        #plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        
    elif scale == "log": # log scale
        sub_log = np.log10(sub)
        lognorm = simple_norm(sub_log, "log", percent=99.0)
        plt.imshow(sub_log, cmap='bone', aspect=1, norm=lognorm,
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
        
    if target_large: # large crosshair
        ra, dec = target_large
        plt.gca().plot([ra+10.0/3600.0, ra+5.0/3600.0], [dec,dec], 
               transform=plt.gca().get_transform('icrs'), linewidth=2, 
               color="black", marker="")
        plt.gca().plot([ra, ra], [dec+10.0/3600.0, dec+5.0/3600.0], 
               transform=plt.gca().get_transform('icrs'),  linewidth=2, 
               color="black", marker="")

    if target_small: # small crosshair
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

    plt.savefig(output, bbox_inches="tight")
    plt.close()


###############################################################################
### transient #################################################################
    
def __plot_rejected(sub_file, 
                    dipoles, elongated_sources, large_sources, vetted, 
                    dipole_width, etamax, areamax, 
                    toi=None,  
                    dipole_color="black", elong_color="#be03fd", 
                    large_color="#fcc006", vetted_color="#53fca1",
                    cmap="coolwarm", 
                    output=None):
    """Plot a difference image with markers flagging sources which were 
    rejected during preliminary vetting in `transient_detect()`.
    
    Arguments
    ---------
    sub_file : str
        Filename for the difference image
    dipoles : astropy.table.Table
        Table of candidates rejected as dipoles
    elongated_sources : astropy.table.Table
        Table of candidates rejected due to elongation
    large_sources : astropy.table.Table 
        Table of candidates rejected due to size
    vetted : astropy.table.Table
        Table of candidates which passed all vetting
    dipole_width : float
        *Maximum* dipole width imposed during vetting
    etamax : float
        *Maximum* elongation imposed during vetting
    areamax : float
        *Maximum* pixel area imposed during vetting
    toi : array_like, optional
        [ra, dec] for some target of interest (default None)
    dipole_color : str, optional
        Color of circle around dipoles (default "black")
    elong_color : str, optional
        Color of square flagging overly elongated sources (default "#be03fd" 
        --> bright purple)
    large_color : str, optional
        Color of triangle flagging overly large sources (default "#fcc006" --> 
        marigold)
    vetted_color : str, optional
        Color of diamond flagging sources which passed all vetting (default 
        "#53fca1" --> sea green)
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colormap for image (default "coolwarm")
    output : str, optional
        Name for output figure (default 
        `sub_file.replace(".fits", "_rejections.png")`)
        
    Notes
    -----
    Plot a difference image with markers flagging likely dipoles, overly 
    elongated sources, overly large sources, and sources which passed all these
    vetting steps, as determined by `transient_detect()`.
    """
    from collections import OrderedDict

    ## load in data    
    data = fits.getdata(sub_file)
    hdr = fits.getheader(sub_file)

    plt.figure(figsize=(14,13))
    w = wcs.WCS(hdr) # show WCS
    ax = plt.subplot(projection=w) 
    ax.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
    ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)

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
                label=r"$A$"+" "+r"$\geqslant$"+" "+str(areamax)+
                " pix"+r"${}^2$")
        
    # marker over accepted sources
    for v in vetted: # diamonds for accepted
        ra, dec = v["ra"], v["dec"]
        plt.plot(ra, dec, transform=ax.get_transform('icrs'), ms=10,
                 mec=vetted_color, mfc="None", mew=2, marker="D", ls="", 
                 label="accepted")

    # crosshair denoting target of interest
    if (toi != None):
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
    title = r"$\mathtt{"+title.replace("_","\_")+"}$"+" transient candidates"
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


def __plot_distributions(og_file, tbl, etamax, areamax, title_append):
    """Produces informative plots showing the distributions of elongation
    and area (size) for candidates identified by `transient_detect()`.
    
    Arguments
    ---------
    og_file : str
        Science image fits filename
    tbl : astropy.table.Table
        Table of all candidate sources, with no vetting performed yet
    etamax : float
        *Maximum* elongation imposed during vetting
    areamax : float
        *Maximum* pixel area imposed during vetting
    title_append : str
        A "title" to include in the plot's title **and** filename
    """
    
    ## candidates ELONGATION distribution as a histogram
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
    plt.xlabel(r"elongation $\eta$", fontsize=15)
    plt.xlim(min(1,mean-std), max(11,mean+std+1))
    plt.ylabel("counts", fontsize=15)
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
    
    if title_append:
        plt.title(f"{title_append}: elongation distribution", fontsize=13)
        plt.savefig(og_file.replace(".fits", f"_{title_append}_elongs_histo.png"), 
                    bbox_inches="tight")
    else:
        plt.title("elongation distribution", fontsize=13)
        plt.savefig(og_file.replace(".fits", "_elongs_histo.png"), 
                    bbox_inches="tight")            
    plt.close()

    ## candidates AREA distribution as a histogram  
    plt.figure()
    areas = tbl["area"].value
    nbelow = len(tbl[areas<areamax])
    nabove = len(tbl[areas>areamax])
    mean, med, std = sigma_clipped_stats(areas) 
    plt.hist(areas, bins=20, color="#c875c4", alpha=0.5)
             
    plt.axvline(mean, color="red", label=r"$\mu$")
    plt.axvline(mean+std, color="red", ls="--", 
                label=r"$\mu$"+"±"+r"$\sigma$")
    plt.axvline(mean-std, color="red", ls="--")
    plt.axvline(areamax, color="blue", lw=2.5, label=r"$A_{max}$")
    plt.xlabel("Area [pix"+r"${}^2$"+"]", fontsize=15)
    plt.ylabel("Counts", fontsize=15)
    plt.gca().tick_params(which='major', labelsize=10)
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    
    textboxstr = r"$\mu - A_{max} = $"+"%.2f"%(mean-areamax)+"\n"
    textboxstr += r"$A < A_{max} = $"+str(nbelow)+"\n"
    textboxstr += r"$A > A_{max} = $"+str(nabove)+"\n"
    textboxstr += r"$f_{used} = $"+"%.2f"%(nbelow/(nbelow+nabove))
    plt.text(1.05, 0.66, textboxstr,transform=plt.gca().transAxes, 
             fontsize=13, bbox=dict(facecolor="white", alpha=0.5))
    plt.legend(loc=[1.03, 0.32], fancybox=True, fontsize=13)
    
    if title_append:
        plt.title(f"{title_append}: area distribution", fontsize=13)
        plt.savefig(og_file.replace(".fits", f"_{title_append}_areas_histo.png"), 
                    bbox_inches="tight")
    else:
        plt.title("area distribution")
        plt.savefig(og_file.replace(".fits", "_areas_histo.png"), 
                    bbox_inches="tight") 
    plt.close()

    ## elongation versus pixel area
    plt.figure()    
    elongsgood = [elongs[i] for i in range(len(elongs)) if (
                  elongs[i]<etamax and areas[i]<areamax)]
    areasgood = [areas[i] for i in range(len(elongs)) if (
                 elongs[i]<etamax and areas[i]<areamax)]

    elongsbad = [elongs[i] for i in range(len(elongs)) if (
                 elongs[i]>etamax or areas[i]>areamax)]
    areasbad = [areas[i] for i in range(len(elongs)) if (
                elongs[i]>etamax or areas[i]>areamax)]
    
    # elongation on x axis, area on y axis            
    plt.scatter(elongsgood, areasgood, marker="o", color="#5ca904", s=12)
    plt.scatter(elongsbad, areasbad, marker="s", color="#fe019a", s=12)
    # means and maxima
    mean, med, std = sigma_clipped_stats(elongs)              
    plt.axvline(mean, ls="--", color="black", label=r"$\mu$")
    plt.axvline(etamax, color="#030aa7", lw=2.5, label=r"$\eta_{max}$")
    mean, med, std = sigma_clipped_stats(areas) 
    plt.axhline(mean, ls="--", color="black")
    plt.axhline(areamax, color="#448ee4", lw=2.5, label=r"$A_{max}$")
    # allowed region of parameter space 
    rect = ptc.Rectangle((0,0), etamax, areamax, fill=False, 
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

    if title_append:
        plt.title(f"{title_append}: elongations and areas", fontsize=13)
        plt.savefig(og_file.replace(".fits", 
                                    f"_{title_append}_elongs_areas.png"), 
                    bbox_inches="tight")
    else:
        plt.title("elongations and areas", fontsize=13)
        plt.savefig(og_file.replace(".fits", "_elongs_areas.png"), 
                    bbox_inches="tight")            
    plt.close()


def __plot_triplet(og_file, sub_hdu, og_hdu, ref_hdu, n, ntargets,  
                   wide=True, cmap="bone", title_append=None, plotdir=None):
    """Plot a single [science image, reference image, difference image] 
    triplet.
    
    Arguments
    ---------
    og_file : str
        Science image file name
    sub_hdu : astropy.io.fits.PrimaryHDU
        Difference image fits HDU
    og_hdu : astropy.io.fits.PrimaryHDU
        Science image fits HDU
    ref_hdu : astropy.io.fits.PrimaryHDU
        Reference image fits HDU
    n : int
        ID of the candidate (first candidate is n=0, next n=1, etc.)
    ntargets : int
        Total number of candidates (which passed all preliminary vetting) in 
        the difference image
    wide : bool, optional
        Whether to plot the triplet as 1 row, 3 columns (`wide == True`) or 3
        rows, 1 column (`wide == False`) (default True)
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colormap to apply to all images in the triplet (default "bone")
    title_append : str, optional
        A "title" to include in the plot's title **and** filename (default 
        None)
    plotdir : str, optional
        Directory in which to save the plot (default is the directory of 
        `og_file`)
    """
    
    # load in data
    sub_data = sub_hdu.data
    og_data = og_hdu.data
    ref_data = ref_hdu.data

    # parameters for the plots and subplots based on whether wide=True
    plot_dict = {True:[(14,5), 131, 132, 133], 
                 False:[(5, 14), 311, 312, 313]}
    
    ## plot
    fig = plt.figure(figsize=plot_dict[wide][0])   

    # science image
    w = wcs.WCS(og_hdu.header) # wcs of science image
    ax = fig.add_subplot(plot_dict[wide][1], projection=w)
    mean, median, std = sigma_clipped_stats(og_data)
    ax.imshow(og_data, vmin=mean-5*std, vmax=mean+5*std, origin='lower', 
              cmap=cmap)
    
    # reference image 
    w2 = wcs.WCS(ref_hdu.header) # wcs of reference image
    ax2 = fig.add_subplot(plot_dict[wide][2], projection=w2)
    mean, median, std = sigma_clipped_stats(ref_data)
    ax2.imshow(ref_data, vmin=mean-5*std, vmax=mean+9*std, origin='lower', 
               cmap=cmap)
    
    # difference image 
    w3 = wcs.WCS(sub_hdu.header) # wcs of difference image
    ax3 = fig.add_subplot(plot_dict[wide][3], projection=w3)
    mean, median, std = sigma_clipped_stats(sub_data)
    ax3.imshow(sub_data, vmin=mean-5*std, vmax=mean+9*std, origin='lower', 
               cmap=cmap)
    
    ## express the candidate number as a string
    if ntargets < 100:
        if n < 10: nstr = "0"+str(n)
        else: nstr = str(n)
    else:
        if n < 10: nstr = "00"+str(n)
        elif n < 100: nstr = "0"+str(n)
        else: nstr = str(n)

    ## ticks, tick labels
    if wide: # RA under middle image, DEC left of leftmost image 
        ax.coords["ra"].set_ticklabel_visible(False)
        ax.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        ax.set_ylabel("Dec (J2000)", fontsize=16)
        ax2.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax2.coords["dec"].set_ticklabel_visible(False)
        ax2.set_xlabel("RA (J2000)", fontsize=16)
        ax3.coords["ra"].set_ticklabel_visible(False)
        ax3.coords["dec"].set_ticklabel_visible(False)
    else: # RA under bottom image, DEC left of middle image
        ax.coords["ra"].set_ticklabel_visible(False)
        ax.coords["dec"].set_ticklabel_visible(False)        
        ax2.coords["ra"].set_ticklabel_visible(False)        
        ax2.coords["dec"].set_ticklabel(size=15, exclude_overlapping=True)
        ax2.set_ylabel("Dec (J2000)", fontsize=16)
        ax3.coords["ra"].set_ticklabel(size=15, exclude_overlapping=True)
        ax3.coords["dec"].set_ticklabel_visible(False)    
        ax3.set_xlabel("RA (J2000)", fontsize=16)
    
    ## titles, output filenames
    if title_append:
        if wide: # title above the middle image 
            ax2.set_title(title_append, fontsize=15)
        else: # title above the topmost image
            ax.set_title(title_append, fontsize=15)
        output = og_file.replace(".fits", 
                                 f"_{title_append}_candidate{nstr}.png")
    else:
        output = og_file.replace(".fits", f"_candidate{nstr}.png")   
        
    if plotdir: 
        output = f'{plotdir}/{re.sub(".*/", "", output)}'
    
    # save and close the figure
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    
    

###############################################################################
### TRAINING SET PROPERTIES ###################################################

def histogram_elongation(tabfile, title=None, output=None):
    """Plot a histogram of the elongations for sources generated by transient 
    detection. 
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    title : str, optional
        Title for output plot (default None)
    output : str, optional
        Filename for output figure (default set by function)

    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # plot
    elongs = tbl["elongation"].data
    mean, med, std = sigma_clipped_stats(elongs) 
    plt.figure(figsize=(11,10))
    plt.hist(elongs, bins=18, color="#90e4c1", alpha=0.8)
    
    # label the mean and 1sigma deviation
    plt.axvline(mean, color="black", label=r"$\mu$")
    plt.axvline(mean+std, color="black", ls="--", 
                label=r"$\mu$"+"±"+r"$\sigma$")
    plt.axvline(mean-std, color="black", ls="--")
    
    plt.xlabel("Elongation "+r"$\eta$", fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.gca().tick_params(axis="both", which='major', labelsize=15)    
    plt.legend(fancybox=True, fontsize=16)
    plt.grid()

    # title
    if title:
        plt.title(title, fontsize=15)
    
    # write it
    if type(output) == type(None):
        output = tabfile.replace(filext, "_elongs_histo.png")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def histogram_area(tabfile, title=None, output=None):
    """Plot a histogram of the area for sources generated by transient 
    detection. 
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    title : str, optional
        Title for output plot (default None)
    output : str, optional
        Filename for output figure (default set by function)

    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # plot
    areas = tbl["area"].data     
    mean, med, std = sigma_clipped_stats(areas)
    plt.figure(figsize=(11,10)) 
    plt.hist(areas, bins=18, range=(0,np.max(areas)), color="#c875c4", 
             alpha=0.8)
    
    # label the mean and 1sigma deviations
    plt.axvline(mean, color="black", label=r"$\mu$")
    plt.axvline(mean+std, color="black", ls="--", 
                label=r"$\mu$"+"±"+r"$\sigma$")
    plt.axvline(mean-std, color="black", ls="--")
    
    plt.xlabel("Area [pix"+r"${}^2$"+"]", fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().tick_params(axis="both", which='major', labelsize=15)    
    plt.legend(fancybox=True, fontsize=16)
    plt.grid()

    # title
    if title:
        plt.title(title, fontsize=15)
    
    # write it
    if type(output) == type(None):
        output = tabfile.replace(filext, "_areas_histo.png")
    plt.savefig(output, bbox_inches="tight")
    plt.close()



###############################################################################
### MACHINE LEARNING DIAGNOSTICS ##############################################

def plot_train_vs_valid_accuracy(train_acc, val_acc, 
                                 output=None, figsize=(20,9)):
    """Plot the training and validation set **accuracy** over the course of 
    training. 
    
    Arguments
    ---------
    train_acc : array_like
        Training set accuracy as a function of epoch
    val_acc : array_like 
        Validation set accuracy as a function of epoch
    output : str, optional
        Name for output figure (defualt set by function)
    figsize : tuple, optional
        Figure dimensions, in inches (default (20,9))
  
    """

    # plot the accuracy of the training and validation sets    
    plt.figure(figsize=figsize)
    plt.plot(train_acc, label='Training', linewidth=2.0, color="#ff5b00")
    plt.plot(val_acc, label='Validation', linewidth=2.0, color="#5539cc")
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend(loc='best', fontsize=18, fancybox=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    
    # save the figure 
    if type(output) == type(None):
        output = "train_vs_valid_accuracy.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    

def plot_train_vs_valid_loss(train_loss, val_loss, 
                             output=None, figsize=(20,9)):
    """Plot the training and validation set **loss** over the course of 
    training. 
    
    Arguments
    ---------
    train_loss : array_like
        Training set loss as a function of epoch
    val_loss : array_like 
        Validation set loss as a function of epoch
    output : str, optional
        Name for output figure (defualt set by function)
    figsize : tuple, optional
        Figure dimensions, in inches (default (20,9))
  
    """

    # plot the accuracy of the training and validation sets    
    plt.figure(figsize=figsize)
    plt.plot(train_loss, label='Training', linewidth=2.0, color="#ff5b00")
    plt.plot(val_loss, label='Validation', linewidth=2.0, color="#5539cc")
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(loc='best', fontsize=18, fancybox=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    
    # save the figure 
    if type(output) == type(None):
        output = "train_vs_valid_loss.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(label_true, label_pred, 
                          normalize=True, 
                          classes=["bogus","real"],
                          cmap="Purples",
                          output=None):
    """Plot the confusion matrix given some true labels and predicted labels. 
    
    Arguments
    ---------
    label_true : array_like
        **Flattened** array of the ground truth labels (0 for bogus, 1 for 
        real) for a test set
    label_pred : np.ndarray
        Array of **predicted** labels for a test set, as returned by some 
        classifier 
    normalize : bool, optional
        Return percentages? or return actual numbers (default True)
    classes : array_like, optional
        Array / tuple with the names the classes 
        (default ["bogus", "real"], which assigns "bogus" to 0 and "real" to 
        1)
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colormap for the plot (default "Purples")
    output : str, optional
        Name for output figure (default set by function)

    Returns
    -------
    np.ndarray
        Confusion matrix ((2,2) array)

    Notes
    -----
    E.g., if `label_true` has shape `(100,)`, `label_pred` must have shape 
    `(100,1)`
    
    """

    from sklearn.metrics import confusion_matrix
    
    ## compute confusion matrix 
    cm = confusion_matrix(label_true, label_pred)
    # output has shape [n_classes, n_classes] = [2,2] for real-bogus
    
    # normalize?
    if normalize: # express as percentages 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
        print("\nNormalized confusion matrix")
    else: # express as raw counts 
        print('\nConfusion matrix [without normalization]')
    print(cm)
    
    ## plot the confusion matrix
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # set ticks and tick labels
    ax.set_xticks(np.arange(cm.shape[1])) # no. of ticks = no. of labels
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(labels=classes, fontdict={"fontsize":14}) # ["bogus", "real"]
    ax.set_yticklabels(labels=classes, fontdict={"fontsize":14})
    ax.set_xlabel("Prediction", fontsize=16) 
    ax.set_ylabel("Truth", fontsize=16)
    ax.axhline(0.5, color="black", lw=2) # lines for clarity
    ax.axvline(0.5, color="black", lw=2) 

    # rotate the y tick labels by 90 degrees
    plt.setp(ax.get_yticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    
    ## label each entry in the matrix
    # above this threshold, text should be white, below, should be black
    thresh = cm.max()/2.0 
    # for each entry in the matrix...
    for i in range(cm.shape[0]): 
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.1f}%' if normalize else f'{cm[i, j]:d}', 
                    ha="center", va="center", # center of box
                    color="white" if cm[i, j] > thresh else "black",
                    size=20)
            
    # save the figure 
    if type(output) == type(None):
        if normalize:
            output = "confusion_matrix_normalized.png"
        else:
            output = "confusion_matrix.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    
    return cm
    

def plot_FPR_FNR_RBscore(preds, 
                         bool_bogus_true, bool_real_true, 
                         thresholds=[0.5], 
                         output=None):
    """Plot the False Positive Rate (FPR) and False Negative Rate (FNR) of 
    some predictions given some RB score threshold.
    
    Arguments
    ---------
    preds : array_like
        Predicted element for each element in the test set
    bool_bogus_true : array_like
        Array where True if ground truth is **bogus**, for each element in the 
        test set
    bool_real_true : array_like
        Array where True if ground truth is **real**, for each element in the 
        test set
    thresholds : array_like
        Array of thresholds for which the FPR and FNR will be annotated 
        (default [0.5]; can be set to None to be disabled)
    output : str, optional
        Name for output figure (default set by function)

    """
    
    # bins for histogram
    rbbins = np.arange(-0.0001, 1.0001, 0.0001)
    
    # values and edges of histogram, normalized such that binwidth*sum = 1
    # b = bogus, r = real
    h_b, e_b = np.histogram(preds[tuple(bool_bogus_true)], bins=rbbins, 
                                  density=True)
    h_b_c = np.cumsum(h_b) # same as h_b --  not sure why?    
    h_r, e_r = np.histogram(preds[tuple(bool_real_true)], bins=rbbins, 
                                  density=True)
    h_r_c = np.cumsum(h_r) # same as h_r --  not sure why?

    # real bogus thresholds
    rb_thres = np.array(list(range(len(h_b)))) / len(h_b)   
    
    ## plot
    # FNR and FPR
    plt.figure(figsize=(13, 9))    
    plt.plot(rb_thres, h_r_c/np.max(h_r_c), color="#e17701",
             label='False Negative Rate (FNR)', linewidth=3.0)
    plt.plot(rb_thres, 1 - h_b_c/np.max(h_b_c), color="#632de9",
             label='False Positive Rate (FPR)', linewidth=3.0) 
    
    # mean misclassification error
    mmce = (h_r_c/np.max(h_r_c) + 1 - h_b_c/np.max(h_b_c))/2
    plt.plot(rb_thres, mmce, '--', label='Mean misclassification error', 
             color='gray', linewidth=2.5)
          
    # x and y scales/ticks/labels
    plt.yscale("log") # log scaling  
    plt.xlim([-0.05, 1.05])    
    plt.ylim([5e-4, 1.05])
    plt.xticks(np.arange(0, 1.1, 0.1))   
    ytick_values = plt.gca().get_yticks()
    # make y ticks into percentages
    plt.gca().set_yticklabels(['{:,.1%}'.format(x) if x < 0.01 else 
                               '{:,.0%}'.format(x) for x in ytick_values])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('RB score threshold', fontsize=18)
    plt.ylabel('Cumulative percentage', fontsize=18)  
    
    # mark the desired RB score threshold(s) (default is just 0.5)
    if thresholds != None:
        for t in thresholds:
            mask_thresh = rb_thres < t
            # compute FPR, FNR for this threshold
            fnr = np.array(h_r_c/np.max(h_r_c))[mask_thresh][-1]
            fpr = np.array(1 - h_b_c/np.max(h_b_c))[mask_thresh][-1]
            print(f"\nAt RB threshold = {t}:")
            print(f"FPR = {(fpr*100):.2f}%\tFNR = {(fnr*100):.2f}%")
            # draw a line at the RB score threshold and write the FPR and FNR
            # above the point where the line hits max(FPR, FNR)
            plt.vlines(t, ymin=5e-4, ymax=max(fnr, fpr), linestyle="-", 
                        color="black",linewidth=2.5)
            t_label = f' {fnr*100:.1f}% FNR\n {fpr*100:.1f}% FPR'
            plt.text(t-0.04, max(fnr, fpr)+0.01, t_label, fontsize=16)
    
    # grid lines     
    plt.gca().grid(True, which='major', linewidth=1.0)
    plt.gca().grid(True, which='minor', linewidth=0.5)    
        
    # legend
    plt.legend(loc="best", fontsize=16, fancybox=True)    
            
    # save the figure 
    if type(output) == type(None):
        output = "FPR_FNR_RBscore.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_ROC(test_labels, preds, output=None):
    """Plot the receiver operating characteristic (ROC) curve, for some test 
    data set.
    
    Arguments
    ---------
    test_labels : array_like
        Array of labels (0 or 1) given to sources in the test set, for some 
        RB score threshold
    preds : array_like
        Array of **predicted** RB scores given some threshold
    output : str, optional
        Name for output figure (default set by function)
    
    Notes
    -----
    The ROC curve shows the False Positive Rate (FPR; Contamination) versus 
    the True Positive Rate (TPR; Sensitivity). Shows the entire plot as well 
    as a zoomed-in section near where (ideally) FPR~0, TPR~1. 

    """
    from sklearn.metrics import roc_curve, auc    

    # compute False Pos. Rate (FPR), True Pos. Rate (TPR), and thresholds
    # for a Receiver Operating Characteristic (ROC) curve     
    fpr, tpr, thresholds = roc_curve(test_labels, preds) # curve
    roc_auc = auc(fpr, tpr) # area under the curve 
 
    ## plot
    fig = plt.figure(figsize=(14, 5))
    fig.subplots_adjust(bottom=0.09, left=0.05, right=0.70, top=0.98, 
                        wspace=0.2, hspace=0.2)            
    # ROC
    ax = fig.add_subplot(1,2,1)                 
    ax.plot(fpr, tpr, lw=3.0, color="#632de9") # ROC curve      
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate [Contamination]', fontsize=16)
    ax.set_ylabel('True Positive Rate [Sensitivity]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.grid(True)    
    # zoomed ROC
    ax2 = fig.add_subplot(1,2,2)     
    ax2.plot(fpr, tpr, lw=3.0, color="#632de9",
             label=f'ROC (AUC = {roc_auc:.2f})') # ROC curve    
    ax2.set_xlim([-0.005, 0.1])
    ax2.set_ylim([0.9, 1.005])
    #ax2.set_xlabel('False Positive Rate [Contamination]', fontsize=16)
    #ax2.set_ylabel('True Positive Rate [Sensitivity]', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    ax2.grid(True)
    ax2.legend(loc="best", fontsize=16, fancybox=True)
    
    # make edges red, dashed lines with thickness 2.5
    for i in ["bottom", "top", "right", "left"]:
        ax2.spines[i].set_color('red')
        ax2.spines[i].set_linewidth(2.5)
        ax2.spines[i].set_linestyle("--")

    # straight line with slope 1, intercept 0
    ax.plot([0, 1], [0, 1], color='#333333', lw=1.5, linestyle='--')
    
    # rectangle indicating zoomed region
    rect = ptc.Rectangle((0.0, 0.9), width=0.1, height=0.1, fill=False, 
                         lw=2.5, ls="--", color="red")
    ax.add_patch(rect)
    
    ### CONNECT INSET AXES HERE?

    # save the figure 
    if type(output) == type(None):
        output = "ROC_curve.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_histogram_RB_score(tabfile, 
                            key="label_preds", rb_thresh=0.5, 
                            title=None, 
                            output=None):
    """Plot a histogram showing the RB scores predicted by some model. 
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients which have been assigned an RB score by 
        some model, or have been manually labelled (must be a .csv or .fits 
        file)
    key : str, optional
        Key (column name) to use to extract the predicted RB score (default 
        "label_preds")
    rb_thresh : float, optional
        RB score to set as the threshold for real sources (default 0.5)
    title : str, optional
        Title for the plot (default None)
    output : str, optional
        Name for the output figure (default set by function)

    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # labels 
    labels = tbl[key]
    real_labels = [l for l in labels if l>rb_thresh] # real sources
    bogus_labels = [l for l in labels if l<rb_thresh] # bogus sources 
    # bins
    real_bins = np.arange(rb_thresh, 1.05, 0.05)
    bogus_bins = np.arange(0, rb_thresh+0.05, 0.05)
    
    # plot
    plt.figure(figsize=(11,10))
    plt.hist(real_labels, bins=real_bins, color="#0652ff", 
             label=f"Real [{len(real_labels)}]")
    plt.hist(bogus_labels, bins=bogus_bins, color="#fc5a50", 
             label=f"Bogus [{len(bogus_labels)}]")   
    
    plt.xlabel("RB score", fontsize=18)
    plt.ylabel("Counts", fontsize=18)
    xticks = np.arange(0, 1.1, 0.1)
    plt.xticks(xticks)
    plt.yscale("log")
    plt.gca().tick_params(axis="both", which='major', labelsize=20)    
    plt.legend(fancybox=True, fontsize=20)
    plt.gca().grid(True, which='major', linewidth=1.0)
    plt.gca().grid(True, which='minor', linewidth=0.3)  

    # a line at the RB score threshold
    plt.axvline(rb_thresh, lw=2.0, color="black")    

    # add a title to the plot?
    if not(type(title) == type(None)):
        plt.title(title, fontsize=15)

    # save the figure 
    if type(output) == type(None):
        output = tabfile.replace(filext, f"_histo_RBscore_{rb_thresh}.png")
    plt.savefig(output, bbox_inches="tight")
    plt.close()
