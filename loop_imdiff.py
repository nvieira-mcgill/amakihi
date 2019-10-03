#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:06:15 2019
@author: Nicholas Vieira
@loop_imdiff.py
"""

import amakihi 
import os
import sys
from subprocess import run
import glob 
import re
import pandas as pd 
import numpy as np
from astropy.io import fits
from astropy import wcs
from timeit import default_timer as timer

# enable/disable plots popping up
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg') # plots pop up
plt.switch_backend('agg') # plots don't pop up

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

### ARGUMENTS #############################################################
## NOTE: csv must be a text csv
CSV = "GW190814_50_GLADE.csv" # text csv of RAs, DECs, Ranks 
## directories
#BASEDIR = "/media/nvieira/OS/Users/nviei/Documents/Storage/"
#SCI_DIR = BASEDIR+"isci_50region" # science
#TMP_DIR = BASEDIR+"itmp_50region" # template 
#WORKDIR = BASEDIR+"workdir" # working directory
#PLOTDIR = BASEDIR+"plotdir" # plotting directory 
#CROPALDIR = BASEDIR+"cropaldir" # cropped and aligned images
#BKGSUBDIR = BASEDIR+"bkgsubdir" # cropped, aligned, background-subtracted
#SATMASKDIR = BASEDIR+"satmaskdir" # saturation masks
#SUBDIR = BASEDIR+"subdir" # difference images
#SUBMASKDIR = BASEDIR+"submaskdir" # hotpants subtraction mask
#PSFDIR = BASEDIR+"psfdir" # ePSFs (proper image subtraction)
#PROPSUBDIR = BASEDIR+"propersubdir" # proper image subtractions
# or, if directories are 1:1 and just working in cwd 
BASEDIR = os.getcwd()+"/"
SCI_DIR = BASEDIR+"isci" 
TMP_DIR = BASEDIR+"itmp" 
WORKDIR = BASEDIR+"workdir"
PLOTDIR = BASEDIR+"plotdir" # plotting directory 
CROPALDIR = BASEDIR+"cropaldir" # cropped and aligned images
BKGSUBDIR = BASEDIR+"bkgsubdir" # cropped, aligned, background-subtracted
SATMASKDIR = BASEDIR+"satmaskdir" # saturation masks
SUBDIR = BASEDIR+"subdir" # difference images
SUBMASKDIR = BASEDIR+"submaskdir" # hotpants subtraction mask
PSFDIR = BASEDIR+"psfdir" # ePSFs (proper image subtraction)
PROPSUBDIR = BASEDIR+"propersubdir" # proper image subtractions

## RA, DEC determination
RANKS = False # use the .csv to obtain Ranks
TARGET_CROP = False # use the .csv to obtain RAs, DECs for every Rank 
OCTANT_CROP = False # crop according to which octant contains some target
MANUAL = True # manually supply a RA, DEC below 
## which subtraction to use 
PROPERSUB = True # use proper image subtraction (if False, use hotpants)
## other 
CROPMIN = 1000 # minimum pixel dimensions of cropped image 
ALIGNSIGMA = 5.0 # sigma for source detection with image_align
SIGMA = 8.0 # sigma for source detection when building ePSF, if applicable 
PSFSIGMA = 5.0 # estimated PSF width for astrometry.net 
ALIM = 1500 # maximum allowed area in pix**2 for astrometry.net
NTHREADS = 8 # number of threads to use in FFTs
 
## clean up directories
#run("rm -rf "+SCI_DIR+"/*crop* ", shell=True)
#run("rm -rf "+TMP_DIR+"/*crop* ", shell=True) 
#run("rm -rf "+SCI_DIR+"/*.png ", shell=True)
#run("rm -rf "+TMP_DIR+"/*.png ", shell=True)
#run("rm -rf "+WORKDIR+"/*.fits ", shell=True)
#run("rm -rf "+WORKDIR+"/*.txt ", shell=True)
#run("rm -rf "+WORKDIR+"/*.png ", shell=True)
#run("rm -rf "+WORKDIR+"/*conv ", shell=True)
#run("rm -rf "+WORKDIR+"/*.all ", shell=True)
#run("rm -rf "+WORKDIR+"/*.skipped ", shell=True)
#run("rm -rf "+SATMASKDIR+"/*.fits ", shell=True)
#run("rm -rf "+PLOTDIR+"/*.png ", shell=True)
#run("rm -rf "+PSFDIR+"/*.fits ", shell=True)
#run("rm -rf "+PROPSUBDIR+"/*.fits ", shell=True)

### LOAD IN DATA #########################################################
df = pd.read_csv(CSV) # get coordinates 
ranks = df["Rank"] # get ranks for each object 

cwd = os.getcwd() # get science/template filenames 
sci_files = sorted(glob.glob(SCI_DIR+"/*.fits"))#[10:13]
# NOTE: 14-18 == rank04 (incl. i1)
#       10:13 == rank04 otherwise 

if RANKS: # if using rankXX.fits files
    tmp_files = []
    for s in sci_files:
        s = s.replace(".fits","")
        if s[-1].isnumeric():
            s = s[:-1]
        topfile = re.sub(".*/", "", s) # from /a/b/c, extract c
        tmp_files.append(TMP_DIR+"/"+topfile+"final.fits")
else: # if looping over many images of some object with manually input coords
    tmp_files = sorted(glob.glob(TMP_DIR+"/*.fits")) # dirs must be 1:1
    # check if there is a corresponding template for every science image 
    if len(sci_files) != len(tmp_files):
        print("The number of science files and template files do not match. "+
              "Exiting.")
        sys.exit()

#### ITERATE THROUGH ALL OF THE FILES #####################################
nfiles = len(sci_files)
for i in range(nfiles):
    start = timer()
    # open the files 
    sci = fits.getdata(sci_files[i])
    sci_hdr = fits.getheader(sci_files[i])
    tmp = fits.getdata(tmp_files[i])
    tmp_hdr = fits.getheader(tmp_files[i])

    source_file, template_file = sci_files[i], tmp_files[i]
    print("\n*******************************************************")
    print("ORIGINAL IMAGES:")
    print("*******************************************************")
    print("science file: "+re.sub(".*/", "",source_file))
    print("template file: "+re.sub(".*/", "",template_file)+"\n")

    ### DETERMINE RA, DEC OF SOURCE OF INTEREST ########################### 
    if RANKS: # use csv to get RA, Dec for each ranked object 
        rank_sci = re.sub(".*/", "",sci_files[i])[4:6] # determine rank 
        rank_tmp = re.sub(".*/", "",tmp_files[i])[4:6]
        if int(rank_sci) != int(rank_tmp): 
            print("The images being compared are not of the same field. "+
                  "Exiting.")
            sys.exit()
        
        ra = float((df.loc[df["Rank"] == int(rank_sci)])["RAJ2000"])
        dec = float((df.loc[df["Rank"] == int(rank_sci)])["DEJ2000"])
        
    elif MANUAL: # or just manually supply an RA, Dec
        # current values are for AT02019ntp
        ra_gal, dec_gal = 12.55052905, -26.19831783
        #ra, dec = 12.5503, -26.1979
        ra, dec = 12.5502516722, -26.1980040852
 
    ### DETERMINE CROPPING PARAMETERS ####################################
    w = wcs.WCS(sci_hdr)
    x_sci, y_sci = w.all_world2pix(ra, dec, 1)
    x_sci = float(x_sci) # x, y coords of target in science image
    y_sci = float(y_sci)
    w = wcs.WCS(tmp_hdr)
    x_tmp, y_tmp = w.all_world2pix(ra, dec, 1)
    x_tmp = float(x_tmp) # x, y coords of target in template image 
    y_tmp = float(y_tmp)
    
    if TARGET_CROP: # construct largest cutout possible which is still a square 
                    # centered on the target source                
        pot_sizes = [x_sci, abs(x_sci-sci.shape[1]), y_sci, 
                     abs(y_sci-sci.shape[0]), x_tmp, abs(x_tmp-tmp.shape[1]),
                     y_tmp, abs(y_tmp-tmp.shape[0])]
        size = min(pot_sizes) # size of largest possible square
        
        if size < CROPMIN: 
            # if size is too small, alignment is difficult
            if OCTANT_CROP:
                # use octant-based cropping 
                oct_hdu = amakihi.crop_octant(source_file, ra, dec, 
                                              write=False)
                # get the RA, DEC of its center 
                w = wcs.WCS(oct_hdu.header)
                x = oct_hdu.data.shape[1]/2.0
                y = oct_hdu.data.shape[0]/2.0
                crop_center = w.all_pix2world(x, y, 1)
                
                ra_crop = float(crop_center[0])
                dec_crop = float(crop_center[1])
                size = float(max(oct_hdu.data.shape)) # size
            else:
                # grab a square of the image in the top/bottom half or middle
                # based on where source is found 
                # get the half but don't write it 
                half_hdu = amakihi.crop_half(source_file, ra, dec, write=False)
                # get the RA, DEC of its center 
                w = wcs.WCS(half_hdu.header)
                x = half_hdu.data.shape[1]/2.0
                y = half_hdu.data.shape[0]/2.0
                crop_center = w.all_pix2world(x, y, 1)
            
                ra_crop = float(crop_center[0])
                dec_crop = float(crop_center[1])
                size = float(max(half_hdu.data.shape)) # size
        
            ra = float((df.loc[df["Rank"] == int(rank_sci)])["RAJ2000"])
            dec = float((df.loc[df["Rank"] == int(rank_sci)])["DEJ2000"])
            
        else: # size is sufficiently large 
            ra_crop = ra
            dec_crop = dec
    
    elif OCTANT_CROP:
        # grab a square of the image based on the octant where source is found
        # get the octant but don't write it 
        oct_hdu = amakihi.crop_octant(source_file, ra, dec, write=False)
        # get the RA, DEC of its center 
        w = wcs.WCS(oct_hdu.header)
        x = oct_hdu.data.shape[1]/2.0
        y = oct_hdu.data.shape[0]/2.0
        crop_center = w.all_pix2world(x, y, 1)
        
        ra_crop = float(crop_center[0])
        dec_crop = float(crop_center[1])
        size = float(min(oct_hdu.data.shape)) # size        
    else: 
        # grab a square of the image in the top half, bottom half, or middle
        # based on where source is found 
        # get the half but don't write it 
        half_hdu = amakihi.crop_half(source_file, ra, dec, write=False)
        # get the RA, DEC of its center 
        w = wcs.WCS(half_hdu.header)
        x = half_hdu.data.shape[1]/2.0
        y = half_hdu.data.shape[0]/2.0
        crop_center = w.all_pix2world(x, y, 1)

        ra_crop = float(crop_center[0])
        dec_crop = float(crop_center[1])
        size = float(min(half_hdu.data.shape)) # size
    
    ### CROPPING #########################################################
    print("*******************************************************")
    print("Cropping images...")
    print("*******************************************************")
    print("RA_CROP = %.5f"%ra_crop)
    print("DEC_CROP = %.5f"%dec_crop)
    print("SIZE = "+str(size)+" pix\n")
    amakihi.crop_WCS(source_file, ra_crop, dec_crop, size)
    if RANKS: # both science and template are from CFHT
        amakihi.crop_WCS(template_file, ra_crop, dec_crop, size)
    else: # template from PS1/DECaLS, which has pixels twice as large
        amakihi.crop_WCS(template_file, ra_crop, dec_crop, size/2.0)        
    source_file = source_file.replace(".fits", "_crop.fits")
    template_file = template_file.replace(".fits", "_crop.fits")
    
    ### IMAGE ALIGNMENT ##################################################
    # align with astroalign and then use image_registration for fine alignment 
    # if not possible, use image_registration only 
    print("*******************************************************")
    print("Image alignment...")#\n")
    print("*******************************************************")
    ret = amakihi.image_align(source_file, template_file, 
                              thresh_sigma=ALIGNSIGMA)
    
    if ret == None: # if astroalign fails
        # align them with image_registration
        ret = amakihi.image_align_fine(source_file, template_file)
        if ret == None: # if image_registration fails too
            sys.exit() # can't do anything 
        else: # if image_registration succeeds
            source_file = source_file.replace(".fits", "_align.fits")
            mask_file = source_file.replace(".fits", "_mask.fits")
    else: # if astroalign succeeds
        # fine alignment with image_registration
        source_file = source_file.replace(".fits", "_align.fits")
        mask_file = source_file.replace(".fits", "_mask.fits")
        ret = amakihi.image_align_fine(source_file, template_file)
        if ret: # if image_registration succeeds
            source_file = source_file.replace(".fits", "_align.fits")
            mask_file = source_file.replace(".fits", "_mask.fits")
    source_crop = source_file # cropped and aligned 
    template_crop = template_file # cropped and aligned
        
    ### BACKGROUND SUBTRACTION ###########################################
    print("*******************************************************")
    print("Background subtraction...")#\n")
    print("*******************************************************\n")
    amakihi.bkgsub(source_file, crreject=False, bkg_filt=(1,1))
    if RANKS: # both science and template are from CFHT
        amakihi.bkgsub(template_file, crreject=False, bkg_filt=(1,1))
    else:
        amakihi.bkgsub(template_file, crreject=False, bkg_filt=(1,1))
    source_file = source_file.replace(".fits", "_bkgsub.fits")
    template_file = template_file.replace(".fits", "_bkgsub.fits")
    source_bkgsub = source_file # cropped, aligned, bkg-subtracted
    template_bkgsub = template_file # cropped, aligned, bkg-subtracted
    
    ### GET SUBSTAMPS X, Y FOR GOOD SOURCES ##############################
    ## not sure if this is working right now 
    #ret = amakihi.get_substamps(source_file, template_file, mask_file)
    #substamps_file = source_file.replace(".fits", "_substamps.txt")
    
    ### BUILDING A MASK OF SATURATED STARS ###############################
    print("*******************************************************")
    print("Building a mask of saturated stars...")#\n")
    print("*******************************************************\n")
    amakihi.saturation_mask(source_file, mask_file=None, dilation_its=7,
                            ra_safe=ra, dec_safe=dec, rad_safe=10.0, 
                            plot=True)
    mask_file = source_file.replace(".fits", "_satmask.fits")
    
    
    ### MAKE AN IMAGE WITH THE SATURATION MASK APPLIED ###################
    print("*******************************************************")
    print("Plotting the aligned, background-subtracted image...")#\n")
    print("*******************************************************\n") 
    amakihi.make_image(source_bkgsub, mask_file=mask_file, scale="asinh",
                       cmap="viridis", target=[ra,dec])

    ### PROPER IMAGE SUBTRACTION #########################################
    if PROPERSUB:
        print("*******************************************************")
        print("Building ePSF for both science and reference image... ")#\n")
        print("*******************************************************\n")
        amakihi.build_ePSF(source_bkgsub, sigma=SIGMA, psfsigma=PSFSIGMA,
                           alim=ALIM, plot=True)
        amakihi.build_ePSF(template_bkgsub, sigma=SIGMA, psfsigma=PSFSIGMA,
                           alim=ALIM, plot=True)
        pn_file = source_bkgsub.replace(".fits", "_ePSF.fits")
        pr_file = template_bkgsub.replace(".fits", "_ePSF.fits")
    
        print("*******************************************************")
        print("Performing proper image subtraction...")#\n")
        print("*******************************************************")
        d, s = amakihi.proper_subtraction(new_file=source_file, 
                                          ref_file=template_file, 
                                          new_og=source_crop,
                                          ref_og=template_crop,
                                          pn_file=pn_file,
                                          pr_file=pr_file,
                                          mask_file=mask_file, 
                                          odd=False,
  
                                          bpix_fill=True,
                                          plot_bpixfill=True,

                                          findbeta=False,
                                          betaits=20, 
                                          analyticbeta=True,
                                        
                                          sigma=SIGMA,
                                          psfsigma=PSFSIGMA,
                                          alim=ALIM, 
                                          
                                          nthreads=NTHREADS,
                                          pad=True,
                                          maskborder=False,
                                          zeroborder=True,
                                          
                                          plot="S",
                                          target=[ra,dec])
        # move and copy files
        if type(s) == np.ndarray: # if successful subtraction 
            S_file = source_file.replace(".fits", "_propersub_S.fits")
            D_file = source_file.replace(".fits", "_propersub_D.fits")
            print("*******************************************************")
            print("Performing transient detection...")
            print("*******************************************************")
            tab = amakihi.transient_detect_proper(S_file, source_crop, 
                                                  elongation_lim=1.5,
                                                  og_scale="asinh",
                                                  crosshair_sub="#fe01b1",
                                                  toi=[ra,dec],
                                                  toi_sep_min=0.0,
                                                  toi_sep_max=100.0)
            run("cp "+source_crop+" "+CROPALDIR, shell=True) # crop
            run("cp "+template_crop+" "+CROPALDIR, shell=True) 
            run("cp "+source_bkgsub+" "+BKGSUBDIR, shell=True) # +align, bgsub
            run("cp "+template_bkgsub+" "+BKGSUBDIR, shell=True) 
            run("cp "+mask_file+" "+SATMASKDIR, shell=True) # saturation mask
            run("cp "+pn_file+" "+PSFDIR, shell=True) # new image ePSF
            run("cp "+pr_file+" "+PSFDIR, shell=True) # ref image ePSF
            run("cp "+S_file+" "+PROPSUBDIR, shell=True) # S statistic 
            run("cp "+D_file+" "+PROPSUBDIR, shell=True) # D statistic 
        
    ### IMAGE DIFFERENCING WITH HOTPANTS #################################
    else: 
        print("*******************************************************")
        print("Performing image subtraction with hotpants...")#\n")
        print("*******************************************************")
        if RANKS: # both science and template are from CFHT
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   iu=50000, il=-100.0, tu=50000, tl=-100.0,
                                   ng="3 6 2.5 4 5.0 2 10.0", 
                                   bgo=0, ko=0, v=0, target=[ra,dec], 
                                   rkernel=2.5*5.0)
        else:
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   iu=50000, il=-500.0, bgo=0, ko=0,
                                   #ng="3 6 2.5 4 5.0 2 10.0", 
                                   #rkernel=2.5*5.0,
                                   convi=True, v=0, target=[ra_gal,dec_gal],
                                   target_small=[ra,dec])
        
        # move and copy files
        if (type(ret) == np.ndarray): # if successful subtraction 
            subfile = source_file.replace(".fits", "_subtracted.fits")
            submask = source_file.replace(".fits", "_submask.fits")
            print("*******************************************************")
            print("Performing transient detection...")
            print("*******************************************************")
            tab = amakihi.transient_detect_hotpants(subfile, source_crop, 
                                                    elongation_lim=2.0,
                                                    og_scale="asinh",
                                                    crosshair_sub="#fe01b1",
                                                    toi=[ra,dec],
                                                    toi_sep_min=0.0,
                                                    toi_sep_max=100.0)
            run("cp "+source_crop+" "+CROPALDIR, shell=True) # crop
            run("cp "+template_crop+" "+CROPALDIR, shell=True) 
            run("cp "+source_bkgsub+" "+BKGSUBDIR, shell=True) # +align, bgsub
            run("cp "+template_bkgsub+" "+BKGSUBDIR, shell=True) 
            run("cp "+mask_file+" "+SATMASKDIR, shell=True) # saturation mask
            run("cp "+subfile+" "+SUBDIR, shell=True) # difference
            run("cp "+submask+" "+SUBMASKDIR, shell=True) # difference mask

    # finally, move everything 
    run("mv "+SCI_DIR+"/*crop* "+WORKDIR, shell=True)
    run("mv "+TMP_DIR+"/*crop* "+WORKDIR, shell=True)
    run("mv "+WORKDIR+"/*.png "+PLOTDIR, shell=True)
    
    end = timer()
    print("\nTIME = %.4f s"%(end-start))


    
    
    
