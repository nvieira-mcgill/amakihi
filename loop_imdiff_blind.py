#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:46:30 2019
@author: Nicholas Vieira
@loop_imdiff_blind.py 
"""

import amakihi 
import os
from subprocess import run
import glob 
import re
import numpy as np
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

## directories
BASEDIR = os.getcwd()+"/"
SCI_DIR = BASEDIR+"GW190814_coadds_science_cropped" # (PRE-CROPPED) sci ims
TMP_DIR = BASEDIR+"GW190814_coadds_templates_cropped" # (PRE-CROPPED) tmpl ims
TOPWORKDIR = BASEDIR+"GW190814_imdiff_workdir/" # top working directory 
WORKDIR = TOPWORKDIR+"workdir" # working directory
PLOTDIR = TOPWORKDIR+"plotdir" # plotting directory 
CANDIDATEDIR = TOPWORKDIR+"candidatedir" # plots of potential transients 
BKGSUBDIR = TOPWORKDIR+"bkgsubdir" # aligned, background-subtracted images
SATMASKDIR = TOPWORKDIR+"satmaskdir" # saturation masks
SUBDIR = TOPWORKDIR+"subdir" # difference images
SUBMASKDIR = TOPWORKDIR+"submaskdir" # hotpants subtraction mask
PSFDIR = TOPWORKDIR+"psfdir" # ePSFs (proper image subtraction)
PROPSUBDIR = TOPWORKDIR+"propersubdir" # proper image subtractions

## general subtraction arguments
TMP_CFHT = True # are templates also from CFHT? 
ALIGNSIGMA = 5.0 # sigma for source detection with image_align
SEPLIM = 5.0 # maximum allowed separation between TOI and transients 
TRANSIENTSIGMA = 3.0 # subtraction image transient detection sigma 

## which subtraction to use 
PROPERSUB = False # use proper image subtraction (if False, use hotpants)

## proper image subtraction arguments 
SIGMA = 8.0 # sigma for source detection when building ePSF, if applicable 
PSFSIGMA = 5.0 # estimated PSF width for astrometry.net 
ALIM = 1500 # maximum allowed area in pix**2 for astrometry.net
NTHREADS = 8 # number of threads to use in FFTs
 
## clean up directories
run("rm -rf "+SCI_DIR+"/*crop* ", shell=True)
run("rm -rf "+TMP_DIR+"/*crop* ", shell=True) 
run("rm -rf "+SCI_DIR+"/*.png ", shell=True)
run("rm -rf "+TMP_DIR+"/*.png ", shell=True)
run("rm -rf "+WORKDIR+"/*.fits ", shell=True)
run("rm -rf "+WORKDIR+"/*.txt ", shell=True)
run("rm -rf "+WORKDIR+"/*.png ", shell=True)
run("rm -rf "+WORKDIR+"/*conv ", shell=True)
run("rm -rf "+WORKDIR+"/*.all ", shell=True)
run("rm -rf "+SATMASKDIR+"/*.fits ", shell=True)
#run("rm -rf "+PLOTDIR+"/*.png ", shell=True)
#run("rm -rf "+PSFDIR+"/*.fits ", shell=True)
#run("rm -rf "+PROPSUBDIR+"/*.fits ", shell=True)

### LOAD IN DATA ######################################################### 
# data should be pre-cropped/not in need of cropping 
sci_files = sorted(glob.glob(SCI_DIR+"/*.fits"))
tmp_files = sorted(glob.glob(TMP_DIR+"/*.fits")) # dirs must be 1:1

#### ITERATE THROUGH ALL OF THE FILES #####################################
nfiles = len(sci_files)
for i in range(nfiles):
    start = timer()
    
    source_file, template_file = sci_files[i], tmp_files[i]
    print("\n*******************************************************")
    print("ORIGINAL IMAGES:")
    print("*******************************************************")
    print("science file: "+re.sub(".*/", "",source_file))
    print("template file: "+re.sub(".*/", "",template_file)+"\n")

    ### IMAGE ALIGNMENT ##################################################
    # align with astroalign and then use image_registration for fine alignment 
    # if not possible, use image_registration only 
    print("*******************************************************")
    print("Image alignment...")
    print("*******************************************************")
    ret = amakihi.image_align(source_file, template_file, 
                              thresh_sigma=ALIGNSIGMA)
    
    if ret == None: # if astroalign fails
        # align them with image_registration
        ret = amakihi.image_align_fine(source_file, template_file)
        if ret == None: # if image_registration fails too
            continue # can't do anything 
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
    source_align = source_file # cropped and aligned 
    template_align = template_file # cropped and aligned

    ### BACKGROUND SUBTRACTION ###########################################
    print("*******************************************************")
    print("Background subtraction...")
    print("*******************************************************\n")
    amakihi.bkgsub(source_file, crreject=False, bkg_filt=(1,1))
    if TMP_CFHT: # both science and template are from CFHT
        amakihi.bkgsub(template_file, crreject=False, bkg_filt=(1,1))
    else:
        amakihi.bkgsub(template_file, crreject=False, bkg_filt=(1,1))
    source_file = source_file.replace(".fits", "_bkgsub.fits")
    template_file = template_file.replace(".fits", "_bkgsub.fits")
    source_bkgsub = source_file # cropped, aligned, bkg-subtracted
    template_bkgsub = template_file # cropped, aligned, bkg-subtracted

    ### BUILDING A MASK OF SATURATED STARS ###############################
    print("*******************************************************")
    print("Building a mask of saturated stars...")
    print("*******************************************************\n")
    amakihi.saturation_mask(source_file, mask_file=None, dilation_its=7, 
                            plot=True)
    mask_file = source_file.replace(".fits", "_satmask.fits")

    ### MAKE AN IMAGE WITH THE SATURATION MASK APPLIED ###################
    print("*******************************************************")
    print("Plotting the aligned, background-subtracted image...")
    print("*******************************************************\n") 
    amakihi.make_image(source_bkgsub, mask_file=mask_file, scale="asinh",
                       cmap="viridis")

    ### PROPER IMAGE SUBTRACTION #########################################
    if PROPERSUB:
        print("*******************************************************")
        print("Building ePSF for both science and reference image... ")
        print("*******************************************************\n")
        amakihi.build_ePSF(source_bkgsub, sigma=SIGMA, psfsigma=PSFSIGMA,
                           alim=ALIM, plot=True)
        amakihi.build_ePSF(template_bkgsub, sigma=SIGMA, psfsigma=PSFSIGMA,
                           alim=ALIM, plot=True)
        pn_file = source_bkgsub.replace(".fits", "_ePSF.fits")
        pr_file = template_bkgsub.replace(".fits", "_ePSF.fits")
    
        print("*******************************************************")
        print("Performing proper image subtraction...")
        print("*******************************************************")
        d, s = amakihi.proper_subtraction(new_file=source_file, 
                                          ref_file=template_file, 
                                          new_og=source_align,
                                          ref_og=template_align,
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
                                          
                                          plot="S")
        # move and copy files
        if type(s) == np.ndarray: # if successful subtraction 
            S_file = source_file.replace(".fits", "_propersub_S.fits")
            D_file = source_file.replace(".fits", "_propersub_D.fits")
            print("*******************************************************")
            print("Performing transient detection...")
            print("*******************************************************")
            tab = amakihi.transient_detect_proper(S_file, source_align, 
                                                  elongation_lim=1.5,
                                                  og_scale="asinh",
                                                  crosshair_sub="#fe01b1",
                                                  sigma=TRANSIENTSIGMA)
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
        print("Performing image subtraction with hotpants...")
        print("*******************************************************")
        if TMP_CFHT: # both science and template are from CFHT
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   iu=50000, il=-100.0, tu=50000, tl=-100.0,
                                   ng="3 6 2.5 4 5.0 2 10.0", 
                                   bgo=0, ko=0, v=0, 
                                   rkernel=2.5*5.0)
        else:
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   iu=50000, il=-500.0, bgo=0, ko=0,
                                   #ng="3 6 2.5 4 5.0 2 10.0", 
                                   #rkernel=2.5*5.0,
                                   convi=True, v=0)
        
        # move and copy files
        if (type(ret) == np.ndarray): # if successful subtraction 
            subfile = source_file.replace(".fits", "_subtracted.fits")
            submask = source_file.replace(".fits", "_submask.fits")
            print("*******************************************************")
            print("Performing transient detection...")
            print("*******************************************************")
            tab = amakihi.transient_detect_hotpants(subfile, source_align, 
                                                    elongation_lim=2.0,
                                                    og_scale="asinh",
                                                    crosshair_sub="#fe01b1",
                                                    sigma=TRANSIENTSIGMA)
            run("cp "+source_bkgsub+" "+BKGSUBDIR, shell=True) # +align, bgsub
            run("cp "+template_bkgsub+" "+BKGSUBDIR, shell=True) 
            run("cp "+mask_file+" "+SATMASKDIR, shell=True) # saturation mask
            run("cp "+subfile+" "+SUBDIR, shell=True) # difference
            run("cp "+submask+" "+SUBMASKDIR, shell=True) # difference mask

    # finally, move everything 
    run("mv "+SCI_DIR+"/*crop* "+WORKDIR, shell=True)
    run("mv "+TMP_DIR+"/*crop* "+WORKDIR, shell=True)
    run("mv "+WORKDIR+"/*candidate*.png "+CANDIDATEDIR, shell=True)
    run("mv "+WORKDIR+"/*.png "+PLOTDIR, shell=True)
    
    end = timer()
    print("\nTIME = %.4f s"%(end-start))

