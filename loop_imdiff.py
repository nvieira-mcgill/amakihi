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

# enable/disable plots popping up
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg') # plots pop up
plt.switch_backend('agg') # plots don't pop up

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

### ARGUMENTS #############################################################

## directories
BASEDIR = "/media/nvieira/OS/Users/nviei/Documents/Storage/"
SCI_DIR = BASEDIR+"isci_50region" # science
TMP_DIR = BASEDIR+"itmp_50region" # template 
WORKDIR = BASEDIR+"workdir" # working directory
CROPALDIR = BASEDIR+"cropaldir" # cropped and aligned images
BKGSUBDIR = BASEDIR+"bkgsubdir" # cropped, aligned, background-subtracted
SATMASKDIR = BASEDIR+"satmaskdir" # saturation masks
SUBDIR = BASEDIR+"subdir" # difference images
SUBMASKDIR = BASEDIR+"submaskdir" # hotpants subtraction mask
PSFDIR = BASEDIR+"psfdir" # ePSFs (proper image subtraction)
PROPSUBDIR = BASEDIR+"propersubdir" # proper image subtractions
# or, if directories are 1:1
#BASEDIR = os.getcwd()+"/"
#SCI_DIR = BASEDIR+"isci" 
#TMP_DIR = BASEDIR+"itmp" 
#WORKDIR = BASEDIR+"workdir"
#SUBDIR = BASEDIR+"subdir"

## RA, DEC determination
RANKS = True # use the .csv to obtain Ranks
TARGET_CROP = True # use the .csv to obtain RAs, DECs for every Rank 
MANUAL = False # manually supply a RA, DEC below 
## which subtraction to use 
PROPERSUB = True # use proper image subtraction (if False, use hotpants)
## other 
CROPMIN = 800 # minimum pixel dimensions of cropped image 
SIGMA = 8.0 # sigma for source detection when building ePSF, if applicable 
 
# clean up directories
run("rm -rf "+SCI_DIR+"/*crop* ", shell=True)
run("rm -rf "+TMP_DIR+"/*crop* ", shell=True) 
run("rm -rf "+SCI_DIR+"/*.png ", shell=True)
run("rm -rf "+TMP_DIR+"/*.png ", shell=True)
run("rm -rf "+WORKDIR+"/*.fits ", shell=True)
run("rm -rf "+WORKDIR+"/*.txt ", shell=True)
run("rm -rf "+WORKDIR+"/*.png ", shell=True)
run("rm -rf "+WORKDIR+"/*conv ", shell=True)
run("rm -rf "+WORKDIR+"/*.all ", shell=True)
run("rm -rf "+WORKDIR+"/*.skipped ", shell=True)

### LOAD IN DATA #########################################################
# NOTE: csv must be a text csv
CSV = "GW190814_50_GLADE.csv" # text csv of RAs, DECs, Ranks 
df = pd.read_csv(CSV) # get coordinates 
ranks = df["Rank"] # get ranks for each object 

cwd = os.getcwd() # get science/template filenames 
sci_files = sorted(glob.glob(SCI_DIR+"/*.fits"))[3:]
# NOTE: 14-19 == rank04

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
    # open the files 
    sci = fits.getdata(sci_files[i])
    sci_hdr = fits.getheader(sci_files[i])
    tmp = fits.getdata(tmp_files[i])
    tmp_hdr = fits.getheader(tmp_files[i])

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
        ra, dec = 12.5503, -26.1979
 
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
        
        if size < CROPMIN: # if size is too small, alignment is difficult
            w = wcs.WCS(sci_hdr)
            size = sci.shape[1] # x dimension of CCD
            if y_sci > sci.shape[0]*0.75: # if target in top quarter 
                crop_center = w.all_pix2world(sci.shape[1]/2.0, 
                                              sci.shape[0]*0.75, 1)
            elif y_sci < sci.shape[0]*0.25: # if target in bottom quarter
                crop_center = w.all_pix2world(sci.shape[1]/2.0, 
                                              sci.shape[0]*0.25, 1)
            else: 
                crop_center = w.all_pix2world(sci.shape[1]/2.0, 
                                              sci.shape[0]/2.0, 1)
            ra_crop = float(crop_center[0])
            dec_crop = float(crop_center[1])
            ra = float((df.loc[df["Rank"] == int(rank_sci)])["RAJ2000"])
            dec = float((df.loc[df["Rank"] == int(rank_sci)])["DEJ2000"])
            
        else: # size is sufficiently large 
            ra_crop = ra
            dec_crop = dec
    
    else: # grab a square of the image in the top half, bottom half, or middle
          # depending on the location of the source of interest
        w = wcs.WCS(sci_hdr)
        size = sci.shape[1] # x dimension of CCD
        if y_sci > sci.shape[0]*0.75: # if target in top quarter 
            crop_center = w.all_pix2world(sci.shape[1]/2.0, 
                                          sci.shape[0]*0.75, 1)
        elif y_sci < sci.shape[0]*0.25: # if target in bottom quarter
            crop_center = w.all_pix2world(sci.shape[1]/2.0, 
                                          sci.shape[0]*0.25, 1)
        else: 
            crop_center = w.all_pix2world(sci.shape[1]/2.0, 
                                          sci.shape[0]/2.0, 1)
        ra_crop = float(crop_center[0])
        dec_crop = float(crop_center[1])
    
    source_file, template_file = sci_files[i], tmp_files[i]
    print("\nOriginal images:")
    print("science file: "+re.sub(".*/", "",source_file))
    print("template file: "+re.sub(".*/", "",template_file)+"\n")
    
    ### CROPPING #########################################################
    print("Cropping images...")
    print("RA_CROP = %.5f"%ra_crop)
    print("DEC_CROP = %.5f"%dec_crop)
    print("SIZE = "+str(size)+" pix\n")
    amakihi.crop_WCS(source_file, ra_crop, dec_crop, size)
    if RANKS: # both science and template are from CFHT
        amakihi.crop_WCS(template_file, ra_crop, dec_crop, size)
    else: # templat from PS1/DECaLS, which has pixels twice as large
        amakihi.crop_WCS(template_file, ra_crop, dec_crop, size/2.0)        
    source_file = source_file.replace(".fits", "_crop.fits")
    template_file = template_file.replace(".fits", "_crop.fits")
    
    ### IMAGE ALIGNMENT ##################################################
    # align with astroalign and then use image_registration for fine alignment 
    # if not possible, use image_registration only 
    print("Image alignment...\n")
    ret = amakihi.image_align(source_file, template_file, thresh_sigma=3.0)
    
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
    print("Background subtraction...\n")
    amakihi.bkgsub(source_file, crreject=False)
    if RANKS: # both science and template are from CFHT
        amakihi.bkgsub(template_file, crreject=False)
    else:
        amakihi.bkgsub(template_file, crreject=False)
    source_file = source_file.replace(".fits", "_bkgsub.fits")
    template_file = template_file.replace(".fits", "_bkgsub.fits")
    source_bkgsub = source_file # cropped, aligned, bkg-subtracted
    template_bkgsub = template_file # cropped, aligned, bkg-subtracted
    
    ### GET SUBSTAMPS X, Y FOR GOOD SOURCES ##############################
    ## not sure if this is working right now 
    #ret = amakihi.get_substamps(source_file, template_file, mask_file)
    #substamps_file = source_file.replace(".fits", "_substamps.txt")
    
    ### BUILDING A MASK OF SATURATED STARS ###############################
    print("Building a mask of saturated stars to be ignored during image "+
          "differencing...\n")
    amakihi.saturation_mask(source_file, mask_file=mask_file, ra_safe=ra, 
                            dec_safe=dec, rad_safe=10.0)
    mask_file = source_file.replace(".fits", "_satmask.fits")

    ### PROPER IMAGE SUBTRACTION #########################################
    if PROPERSUB:
        print("Obtaining the ePSF for both the science and reference image to"+
              " be used in proper image subtraction...\n")
        amakihi.derive_ePSF(source_crop, sigma=SIGMA)
        amakihi.derive_ePSF(template_crop, sigma=SIGMA)
        pn_file = source_crop.replace(".fits", "_ePSF.fits")
        pr_file = template_crop.replace(".fits", "_ePSF.fits")
    
        print("Performing proper image subtraction...\n")
        d, s = amakihi.proper_subtraction(new_file=source_file, 
                                          ref_file=template_file, 
                                          new_og=source_crop,
                                          ref_og=template_crop,
                                          pn_file=pn_file,
                                          pr_file=pr_file,
                                          mask_file=mask_file, 
                                          findbeta=False)
        # move and copy files
        if type(s) == np.ndarray: # if successful subtraction 
            print("Image subtraction successful.\n")
            S_file = source_file.replace(".fits", "_propersub_S.fits")
            D_file = source_file.replace(".fits", "_propersub_D.fits")
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
        print("Performing image subtraction with hotpants...\n")
        if RANKS: # both science and template are from CFHT
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   iu=50000, il=-100.0, tu=50000, tl=-100.0,
                                   ng="3 6 2.5 4 5.0 2 10.0", 
                                   bgo=0, ko=0, v=0, target=[ra,dec], 
                                   rkernel=2.5*5.0)
        else:
            ret = amakihi.hotpants(source_file, template_file, mask_file, iu=50000, 
                                   il=-500.0, bgo=0, ko=0,
                                   #ng="3 6 2.5 4 5.0 2 10.0", 
                                   #rkernel=2.5*5.0,
                                   convi=True, v=0, target=[ra_gal,dec_gal],
                                   target_small=[ra,dec])
        
        # move and copy files
        if (type(ret) == np.ndarray): # if successful subtraction 
            print("Image subtraction successful.\n")
            subfile = source_file.replace(".fits", "_subtracted.fits")
            submask = source_file.replace(".fits", "_submask.fits")
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


    
    
    
