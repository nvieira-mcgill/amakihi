#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:13:15 2019
@author: Nicholas Vieira
@apphotom.py 
"""

import numpy as np
import matplotlib.pyplot as plt
from subprocess import run

from astropy.io import fits
from astropy import wcs
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats
from photutils import Background2D, MMMBackground
from photutils import (SkyCircularAperture, aperture_photometry,
                       SkyCircularAnnulus)

def make_source_mask(image_file, sigma=8.0, output=None):
    """
    Input: 
        - filename for **NOT background-subtracted** image 
        - sigma to use as the threshold for source detection (optional; 
          default 8.0)
        - name for the output fits file (optional; default set below)
        
    Uses astrometry.net to detect sources in the image and write them to a 
    file to be used as a source mask in aperture photometry
    
    Output: the source mask data
    """
    if not(output):
        output = image_file.replace(".fits", "_sourcemask.fits")
    
    # overwrite existing files
    options = "-O -M "+output
    # BANDAID FIX: can't locate image2xy at the moment 
    run("/usr/local/astrometry/bin/image2xy "+options+" "+image_file, 
        shell=True) 
    image_source_file = image_file.replace(".fits", ".xy.fits")
    run("rm "+image_source_file, shell=True) # this file not needed 
    
    print("\nSource mask written to file "+output)
    source_mask = fits.getdata(output)
    return source_mask
    

def error_array(image_file, mask_file=None, astrom_sigma=8.0, write=True, 
                output=None):
    """
    Input: 
        - filename for **NOT background-subtracted** image
        - filename for source mask (optional; default None, in which case a 
          source mask is made)
        - the detection sigma to use in astrometry.net (optional; default 8.0,
          only relevant if a mask file is not provided)
        - whether to write the data to a file (optional; default True)
        - the name for the output file (optional; default set below)
        
    Computes the error on the background-only image as the RMS deviation 
    of the background, and then computes the total image error including 
    the contribution of the Poisson noise for detected sources. Necessary 
    for error propagation in aperture photometry. 
    
    Output: the image error array
    """
    from photutils.utils import calc_total_error
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    
    if "WIRCam" in image_header["INSTRUME"]:
        eff_gain = 3.8 # effective gain (e-/ADU) for WIRCam
    else: 
        eff_gain = image_header["GAIN"] # effective gain for MegaPrime

    # mask out sources and convert to bool for background estimation
    if mask_file:
        source_mask = fits.getdata(mask_file)
        source_mask = np.logical_not(source_mask)
    else: 
        print("\nSince no mask was provided, a source mask will be obtained "+
              "now...")
        source_mask = make_source_mask(image_file, astrom_sigma)
        
    # mask out 0 regions near borders/corners
    zero_mask = image_data <= 0 
    # combine the masks with logical OR 
    source_mask = np.ma.mask_or(source_mask, zero_mask)
    
    # estimate background 
    bkg_est = MMMBackground()
    bkg = Background2D(image_data, (10,10), filter_size=(3,3), 
                       bkg_estimator=bkg_est, mask=source_mask)
    # compute sum of Poisson error and background error  
    ### currently, this seems to overestimate unless the input data is 
    ### background-subtracted
    err = calc_total_error(image_data-bkg.background, 
                           bkg.background_rms, eff_gain)
    
    if write:
        if not(output):
            output = image_file.replace(".fits", "_error.fits")
        err_hdu = fits.PrimaryHDU(data=err, header=image_header)
        err_hdu.writeto(output, overwrite=True, output_verify="ignore")
            
    return err


def aperture_photom(image_file, ra_list, dec_list, mask_file=None, 
                    error_file=None, astrom_sigma=8.0, ap_radius=1.2, r1=2.0, 
                    r2=5.0, plot_annulus=True, plot_aperture=True, 
                    ann_output=None, ap_output=None, bkgsub_verify=True):
    """
    Inputs: 
        - filename for **NOT background-subtracted** image 
        - list OR single float/int of ra, dec of interest
        - filename for the source mask (optional; default None, in 
          which case a source mask is made) 
        - filename for the image error array (optional; default None, in which
          case an error array is made)
        - sigma to use as the threshold for astrometry.net (optional; default 
          8.0; only relevant if a mask file is not provided)
        - aperture radius (in arcsec; optional; default 1.2") 
        - inner and outer radii for the annulus (in arcsec; optional; default 
          2.0" and 5.0") 
        - whether to plot the annulus (optional; default True)
        - whether to plot the aperture (optional; default True), 
        - name for the output annulus plot (optional; defaults set below) 
        - name for the output aperture plot (optional; defaults set below)
        - whether to verify that the background-subtracted flux is non-negative 
          (optional; default True)
    
    Finds the total flux in a defined aperture, computes the background in an 
    annulus around this aperture, and computes the background-subtracted flux 
    of the "source" defined by the aperture. Can be called multiple times if a 
    list of RA/Decs is given. 
    
    If the background-subtracted flux at some location is negative, make sure 
    that no sources remain in the annulus of the data, or consider 
    getting a limiting magnitude at the RA, Dec of interest instead. 
    
    Output: a table containing the results of aperture photometry 
    """        
    
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    if mask_file:
        source_mask = fits.getdata(mask_file)
        source_mask = np.logical_not(source_mask)
    else: 
        print("\nSince no mask was provided, a source mask will be obtained "+
              "now...")
        source_mask = make_source_mask(image_file, astrom_sigma)
        mask_file = image_file.replace(".fits", "_sourcemask.fits")
    
    filt = image_header["FILTER"][0]
    date_isot = image_header["DATE"] # full ISOT time
    t_isot = Time(date_isot, format='isot', scale='utc')
    t_MJD = t_isot.mjd # convert ISOT in UTC to MJD

    if error_file:
        image_error = fits.getdata(error_file)
    else:
        print("\nComputing the image error array...")
        image_error = error_array(image_file, mask_file, astrom_sigma)
        
    # initialize table of sources found by aperture photometry if needed
    cols = ["xcenter","ycenter", "ra","dec", "aperture_sum", 
            "aperture_sum_err", "aper_r", "annulus_r1", "annulus_r2",
            "annulus_median", "aper_bkg", "aper_bkg_std", 
            "aper_sum_bkgsub", "aper_sum_bkgsub_err", "mag_fit", 
            "mag_unc", "mag_calib", "mag_calib_unc", "sigma"]
    aperture_sources = Table(names=cols)
    filt_col = Column([], "filter", dtype='S2') # specify
    mjd_col = Column([], "MJD")
    aperture_sources.add_column(filt_col)
    aperture_sources.add_column(mjd_col)
            
    # convert to lists if needed 
    if (type(ra_list) in [float, int]):
        ra_list = [ra_list]
    if (type(dec_list) in [float, int]):
        dec_list = [dec_list]
    
    # compute background-subtracted flux for the input aperture(s) 
    # add these to the list of sources found by aperture photometry 
    print("\nAttemtping to perform aperture photometry...")
    for i in range(0, len(ra_list)):
        phot_table = __drop_aperture(image_data, image_error, image_header,
                                     source_mask, ra_list[i], dec_list[i],
                                     ap_radius, r1, r2, plot_annulus, 
                                     plot_aperture, ann_output, ap_output,
                                     bkgsub_verify)
        if phot_table: # if a valid flux (non-negative) is found
            
            # compute error on bkg-subtracted aperture sum 
            # dominated by aperture_sum_err
            phot_table["aper_sum_bkgsub_err"] = np.sqrt(
                    phot_table["aperture_sum_err"]**2+
                    phot_table["aper_bkg_std"]**2)
            
            # compute instrumental magnitude
            flux = phot_table["aper_sum_bkgsub"]
            phot_table["mag_fit"] = -2.5*np.log10(flux)
            
            # compute error on instrumental magnitude 
            phot_table["mag_unc"] = 2.5/(phot_table['aper_sum_bkgsub']*
                                        np.log(10))
            phot_table["mag_unc"] *= phot_table['aper_sum_bkgsub_err']
            
            # obtain calibrated magnitudes, propagate errors
            try:
                zp_mean = image_header["ZP_MEAN"]
                zp_std = image_header["ZP_STD"]
                
                mag_calib = phot_table['mag_fit'] + zp_mean
                mag_calib.name = 'mag_calib'
                mag_calib_unc = np.sqrt(phot_table['mag_unc']**2 + zp_std**2)
                mag_calib_unc.name = 'mag_calib_unc'
                phot_table['mag_calib'] = mag_calib
                phot_table['mag_calib_unc'] = mag_calib_unc
            except KeyError:
                print("\nWarning: The zero point has not yet been obtained "+
                      "for the input image, so calibrated magnitudes cannot "+
                      "be obtained. Proceeding.")
             
            # compute sigma 
            phot_table["sigma"] = phot_table['aper_sum_bkgsub']
            phot_table["sigma"] /= phot_table['aper_sum_bkgsub_err']
            
            # other useful columns 
            col_filt = Column(filt, "filter", dtype = np.dtype("U2"))
            col_mjd = Column(t_MJD, "MJD")
            phot_table["filter"] = col_filt
            phot_table["MJD"] = col_mjd
            phot_table.remove_column("id") # id is not accurate 
            
            aperture_sources.add_row(phot_table[-1])
        
        else:
            continue
        
    # if only one value computed, print it neatly
    if len(aperture_sources) == 1:
        a = aperture_sources
        s = "\n"+filt
        s += " = %.2f ± %.2f"%(a["mag_calib"],a["mag_calib_unc"])
        s += ", %.1f"%a["sigma"]+"sigma"
        print(s)
        
    return aperture_sources


def limiting_magnitude(image_file, ra, dec, mask_file=None, astrom_sigma=8.0, 
                       error_file=None, sigma=5.0):
    """
    Input: 
        - filename for **NOT background-subtracted** image
        - ra, dec of interest
        - filename for source mask (optional; default None, in which case a 
          source mask is made)
        - the sigma for source detection in astrometry.net (optional; default 
          8.0, only relevant if a source mask is not provided)
        - filename for image error array (optional; default None; in which case 
          an error will be computed)
        - the sigma to use when computing the limiting magnitude (optional; 
          default 5.0)
    
    Output: the limiting magnitude 
    """
    #from photutils import SkyCircularAperture
    
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    if mask_file:
        source_mask = fits.getdata(mask_file)
        source_mask = np.logical_not(source_mask)
    else: 
        print("\nSince no mask was provided, a source mask will be obtained "+
              "now...")
        source_mask = make_source_mask(image_file, astrom_sigma)
    
    if error_file:
        image_error = fits.getdata(error_file)
    else: 
        print("\nComputing the image error array...")
        image_error = error_array(image_file, mask_file, astrom_sigma)
        
    #w = wcs.WCS(image_header) # wcs object 
    #position = SkyCoord(ra, dec, unit="deg", frame="icrs") # desired posn
    #ap = SkyCircularAperture(position, r=1.2*u.arcsec) 
    #ap_pix = ap.to_pixel(w)
    
    # do aperture photometry on region of interest
    # use a large annulus
    phot_table = __drop_aperture(image_data, image_error, image_header,
                                       source_mask, ra, dec, 
                                       ap_radius=1.2, r1=2.0, r2=20.0, 
                                       plot_annulus=False,
                                       plot_aperture=False,
                                       ann_output=None,
                                       ap_output=None,
                                       bkgsub_verify=False)
    
    #bkg_std_total = phot_table["aper_bkg_std"]*ap_pix.area()
 
    phot_table["aper_sum_bkgsub_err"] = np.sqrt(
            phot_table["aperture_sum_err"]**2 +
            phot_table["aper_bkg_std"]**2)
    
    # compute limit below which we can't make a detection
    limit = sigma*phot_table["aper_sum_bkgsub_err"][0]
    try:
        zp_mean = image_header["ZP_MEAN"]
    except KeyError:
        print("\nCannot compute a calibrated limiting magnitude because the "+
              "zero point has not yet been obtained. Please perform PSF "+
              "photometry before calling this function. Exiting.")
        return
    
    limiting_mag = -2.5*np.log10(limit) + zp_mean
    
    #limit = sigma*bkg_std_total
    #self.limiting_mag = -2.5*np.log10(limit) + self.zp_mean
    
    filt = image_header["FILTER"][0]
    print("\n"+filt+" > %.1f (%d sigma)"%(limiting_mag,sigma))
    return limiting_mag


def __drop_aperture(image_data, image_error, image_header, source_mask, 
                    ra, dec, ap_radius, r1, r2, plot_annulus, plot_aperture, 
                    ann_output, ap_output, bkgsub_verify):
    """    
    Input: image data, the RMS deviation error array of the image, the image 
    header, a mask for the sources, the ra, dec of a source of interest, the 
    aperture radius (in arcsec), the inner and outer radii for the annulus (in
    arcsec), whether to plot the annulus, whether to plot the aperture, the 
    name for the output annulus plot, the name for the output aperture plot, 
    and whether to verify that the background-subtracted flux is non-negative 
    
    This method finds the total flux in a defined aperture, computes the 
    background in an annulus around this aperture, and computes the 
    background-subtracted flux of the "source" defined by the aperture.
    
    Output: a table containing the pix coords, ra, dec, aperture flux, 
    aperture radius, annulus inner and outer radii, the median background, 
    total background in aperture, standard deviation in this background, 
    and background-subtracted aperture flux 
    """
            
    # wcs object
    w = wcs.WCS(image_header)
    
    # lay down the aperture 
    position = SkyCoord(ra, dec, unit="deg", frame="icrs") # source posn
    ap = SkyCircularAperture(position, r=ap_radius*u.arcsec) # aperture 
    ap_pix = ap.to_pixel(w) # aperture in pix
    
    # table of the source's x, y, and total flux in aperture
    phot_table = aperture_photometry(image_data, ap_pix, image_error)
    # ra, dec of source
    ra_col = Column([ra], "ra")
    dec_col = Column([dec], "dec")
    phot_table.add_column(ra_col, 3)
    phot_table.add_column(dec_col, 4)
    
    # lay down the annulus 
    annulus_apertures = SkyCircularAnnulus(position, r_in=r1*u.arcsec, 
                                           r_out=r2*u.arcsec)
    annulus_apertures = annulus_apertures.to_pixel(w)
    annulus_masks = annulus_apertures.to_mask(method='center')
    
    # mask out sources 
    image_data_masked = np.ma.masked_where(np.logical_not(source_mask), 
                                           image_data)
    image_data_masked.fill_value = 0
    image_data_masked = image_data_masked.filled()
    annulus_data = annulus_masks[0].multiply(image_data_masked)
    mask = annulus_data <= 0 # mask invalid data 
    annulus_data = np.ma.masked_where(mask, annulus_data)
    
    # estimate background as median in the annulus 
    annulus_data_1d = annulus_data[annulus_data>0]
    bkg_mean, bkg_med, bkg_std = sigma_clipped_stats(annulus_data_1d)
    bkg_total = bkg_med*ap_pix.area()
    
    # add aperture radius, annuli radii, background (median), total 
    # background in aperture, and stdev in background to table 
    # subtract total background in aperture from total flux in aperture 
    phot_table['aper_r'] = ap_radius
    phot_table['annulus_r1'] = r1
    phot_table['annulus_r2'] = r2
    phot_table['annulus_median'] = bkg_med
    phot_table['aper_bkg'] = bkg_total
    ### SHOULD THE NEXT LINE BE MULTIPLIED BY THE AREA?
    phot_table['aper_bkg_std'] = np.std(annulus_data_1d) #NOT sigma-clipped
    phot_table['aper_sum_bkgsub'] = (
            phot_table['aperture_sum']-phot_table['aper_bkg'])
    
    if (phot_table['aper_sum_bkgsub'] < 0) and bkgsub_verify:
        print("Warning: the background-subtracted flux at this aperture "+
              "is negative. It cannot be used to compute a magnitude. "+
              "Consider using a different radius for the aperture/annuli "+
              "and make sure that no sources remain in the annulus. "+
              "Alternatively, get a limiting magnitude at these coordinates "+
              "instead. Exiting.")
        return         
    
    if plot_annulus:
        __plot_annulus(image_header, annulus_data, ra, dec, r1, r2, ann_output)   
    if plot_aperture:
        __plot_aperture(image_data, image_header, annulus_apertures, ra, dec, 
                        ap_pix, r1, r2, ap_output)  
        
    return phot_table


def __plot_annulus(image_header, annulus_data, ra, dec, r1, r2, ann_output):
    """        
    Input: the image header, annulus data, the ra, dec of the source of 
    interest, the inner and outer radii for the annuli (in pixels), and the 
    name for the output plot
    
    Plots an image of the pixels in the annulus drawn around a source of 
    interest for aperture photometry.
    
    Output: None
    """
    
    pixscale = image_header["PIXSCAL1"]
    
    # plotting
    fig, ax = plt.subplots(figsize=(10,10)) 
    plt.imshow(annulus_data, origin="lower", cmap="magma")
    cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cb.set_label(label="ADU",fontsize=15)
    plt.title('Annulus around %.5f, %.5f (1 pixel = %s")'%(ra, dec,
              str(pixscale)), fontsize=15)
    plt.xlabel("Pixels", fontsize=15)
    plt.ylabel("Pixels", fontsize=15)
    
    # textbox indicating inner/outer radii of annulus 
    textstr = r'$r_{in} = %.1f$"'%r1+'\n'+r'$r_{out} = %.1f$"'%r2
    box = dict(boxstyle="square", facecolor="white", 
       alpha=0.6)
    plt.text(0.81, 0.91, transform=ax.transAxes, s=textstr, bbox=box,
             fontsize=14)
    
    if not(ann_output):
        ann_output = "annulus_RA%.5f_DEC%.5f"%(ra, dec)+".png"
        
    plt.savefig(ann_output, bbox_inches="tight")
    plt.close()
        
        
def __plot_aperture(image_data, image_header, annulus_pix, ra, dec, ap_pix, r1, 
                    r2, ap_output):
    """
    Input: the image data, image header, annulus data, the ra, dec of the 
    source of interest, the aperture radius (in pixels), the inner and 
    outer radii for the annuli (in pixels), and the name for the output plot
        
    Plots an image of the aperture and annuli drawn around a source of 
    interest for aperture photometry.
    
    Output: None
    """
    
    pixscale = image_header["PIXSCAL1"]
    # wcs object
    w = wcs.WCS(image_header)

    # update wcs object and image to span a box around the aperture
    xpix, ypix = ap_pix.positions[0] # pix coords of aper. centre 
    boxsize = int(annulus_pix.r_out)+5 # size of box around aperture 
    idx_x = [int(xpix-boxsize), int(xpix+boxsize)]
    idx_y = [int(ypix-boxsize), int(ypix+boxsize)]
    w.wcs.crpix = w.wcs.crpix - [idx_x[0], idx_y[0]] 
    image_data_temp = image_data[idx_y[0]:idx_y[1], idx_x[0]:idx_x[1]] 
    
    # update aperture/annuli positions 
    ap_pix.positions -= [idx_x[0], idx_y[0]] 
    annulus_pix.positions -= [idx_x[0], idx_y[0]] 
    
    # plotting
    plt.figure(figsize=(10,10))
    ax = plt.subplot(projection=w) # show wcs 
    plt.imshow(image_data_temp, origin="lower", cmap="magma")
    ap_pix.plot(color='white', lw=2) # aperture as white cirlce
    annulus_pix.plot(color='red', lw=2) # annuli as red circles 
    cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cb.set_label(label="ADU", fontsize=15)
    plt.title("Aperture photometry around %.5f, %.5f"%(ra, dec), 
              fontsize=15)
    textstr = r'$r_{aper} = %.1f$"'%(ap_pix.r*pixscale)+'\n'
    textstr += r'$r_{in} = %.1f$"'%r1+'\n'
    textstr += r'$r_{out} = %.1f$"'%r2
    box = dict(boxstyle="square", facecolor="white", alpha=0.6)
    plt.text(0.83, 0.88, transform=ax.transAxes, s=textstr, bbox=box, 
             fontsize=14)
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(ap_output):
        ap_output = "aperture_RA%.5f_DEC%.5f"%(ra, dec)+".png"
    
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    plt.savefig(ap_output, bbox_inches="tight")
    plt.close()
