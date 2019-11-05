#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:46:30 2019
@author: Nicholas Vieira
@loop_imdiff_blind.py 
"""

import amakihi 
import os
import sys
from subprocess import run
import glob 
import re
import numpy as np
from timeit import default_timer as timer
from astropy.io import fits 

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
# base data directories:
BASEDIR = os.getcwd()+"/"
SCI_DIR = BASEDIR+"GW190814_coadds_science_cropped" # science (CROPPED)
TMP_DIR = BASEDIR+"GW190814_coadds_templates_cropped" # template (CROPPED)
WORKDIR = BASEDIR+"GW190814_imdiff_workdir/" # top working directory 

# base data directories (no template)
#SCI_DIR = BASEDIR+"GW190814_coadds_science_notemplate_cropped" # science 
#TMP_DIR = BASEDIR+"GW190814_coadds_templates_DECaLS" # template 
#WORKDIR = BASEDIR+"GW190814_imdiff_notemplate_workdir/" # working directory 

# FOR BUGTESTING
#WORKDIR = WORKDIR[:-1]+"_TEST/"

# general directories:
PLOTDIR = WORKDIR+"plotdir" # plotting directory 
CANDIDATEDIR = WORKDIR+"candidatedir" # tables of potential transients
CANDIDATEPLOTDIR = CANDIDATEDIR+"/plots" 
ALIGNDIR = WORKDIR+"aligndir" # aligned only 
ALIGNMASKDIR = WORKDIR+"alignmaskdir" # alignment masks
BKGSUBDIR = WORKDIR+"bkgsubdir" # aligned, background-subtracted images
SATMASKDIR = WORKDIR+"satmaskdir" # saturation masks
SATMASKPLOTDIR = SATMASKDIR+"/plots"
SUBDIR = WORKDIR+"subdir" # difference images
SUBPLOTDIR = SUBDIR+"/plots"
SUBMASKDIR = WORKDIR+"submaskdir" # hotpants subtraction mask
#PSFDIR = WORKDIR+"psfdir" # ePSFs (proper image subtraction)
#PROPSUBDIR = WORKDIR+"propersubdir" # proper image subtractions
# where failed files go:
ALIGNFAILDIR = WORKDIR+"alignfaildir" # failed both alignments 
HOTPANTSFAILDIR = WORKDIR+"hotpantsfaildir" # failed hotpants images

## look for existing files or just overwrite them?
OVERWRITE = True # alignments, background subtractions, masks
OVERWRITEHOTPANTS = True # hotpants difference images

## templates
TMP_CFHT = True # are templates also from CFHT? 
SURVEY = "DECaLS" # if not from CFHT, which survey (PS1 or DECaLS)?

## alignment
DOUBLEALIGN = True # do double alignment (if possible)?
ASTROMSIGMA = [5, 8] # sigma(s) for source detection with astrometry.net
SEGMSIGMA = 3.0 # sigma for source detect with img segmentation in image_align
MAXOFFSET = 100.0 # maximum allowed pixel offset for image_align_fine

## transient detection
ELONGLIM = 1.5 # maximum allowed source elongation
TRANSIENTSIGMA = 5.0 # subtraction image transient detection sigma

## pointing arguments
if len(sys.argv) > 1 and len(str(sys.argv[1])) == 2: # pointing given, use it
    POINTINGS = str(sys.argv[1])
else:
    POINTINGS = "50"
 
## what to do with found candidates
# ["full", "zoom og", "zoom diff"] --> plot all
# None OR False--> plot none 
PLOTCANDS = False # plot candidate transients?

## which subtraction to use 
#PROPERSUB = False # use proper image subtraction (if False, use hotpants)

## proper image subtraction arguments 
#SIGMA = 8.0 # sigma for source detection when building ePSF, if applicable 
#PSFSIGMA = 5.0 # estimated PSF width for astrometry.net 
#ALIM = 1500 # maximum allowed area in pix**2 for astrometry.net
#NTHREADS = 8 # number of threads to use in FFTs
 
## clean up directories
run("rm -rf "+SCI_DIR+"/*xy.fits", shell=True)
run("rm -rf "+TMP_DIR+"/*xy.fits", shell=True) 

### LOAD IN DATA ######################################################### 
# data should be pre-cropped/not in need of cropping 

if len(sys.argv) > 2 and len(str(sys.argv[2])) == 1: # region supplied, use it
    REGION = str(sys.argv[2])
    sci_files = sorted(glob.glob(SCI_DIR+"/"+POINTINGS+"region"+REGION+
                                 "*.fits"))
else:
    sci_files = sorted(glob.glob(SCI_DIR+"/"+POINTINGS+"region*.fits"))

if not(TMP_CFHT) and SURVEY == "DECaLS":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] in ["g","r","z"])]
elif not(TMP_CFHT) and SURVEY == "PS1":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] == "i")]   

tmp_files = []
for s in sci_files:
    s = s.replace(".fits","")
    if (not(TMP_CFHT) and SURVEY=="DECaLS"):
        toapp = s[-5:] # "_crXX"
        s = s[:-5]
    else:
        toapp = s[-4:] # "_top" or "_bot"
        s = s[:-4]
    if POINTINGS == "90" and ("20190819" in s):
        s = re.sub("glade","_glade",s) # 90regionNglade --> 90regionN_glade
    while s[-1].isnumeric():
        s = s[:-1]
    topfile = re.sub(".*/", "", s) # from /a/b/c, extract c
    tmp_files.append(TMP_DIR+"/"+topfile+"template"+toapp+".fits")

#### ITERATE THROUGH ALL OF THE FILES #####################################
nfiles = len(sci_files)
for i in range(nfiles):
    start = timer()
    
    source_file, template_file = sci_files[i], tmp_files[i]
    
    print("\n*******************************************************")
    print("                     ORIGINAL IMAGES:")
    print("*******************************************************")
    print("science file: "+re.sub(".*/", "",source_file))
    print("template file: "+re.sub(".*/", "",template_file)+"\n")
    
    if not(OVERWRITEHOTPANTS):
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        subfiles = glob.glob(SUBDIR+"/"+filename+"*") # update
        if len(subfiles) > 0:
            print("\nA difference image already exists for this science file."+
                  " Skipping to next science file.\n")
            continue

    ### IMAGE ALIGNMENT ##################################################
    # align with astroalign and then use image_registration for fine alignment 
    # if not possible, use image_registration only 
    print("*******************************************************")
    print("Image alignment...")
    print("*******************************************************")
    
    alignfiles = []
    if not(OVERWRITE):
        filename = (re.sub(".*/", "",source_file)).replace(".fits", "")
        alignfiles = glob.glob(ALIGNDIR+"/"+filename+"*")
        if len(alignfiles) > 0:
            print("An aligned image was already produced. Skipping to next "+
                  "step in image differencing.")
            source_file = alignfiles[0]
            mask_file = (re.sub(".*/", "",source_file)).replace(".fits", 
                                                                "_mask.fits")
            mask_file = ALIGNMASKDIR+"/"+mask_file
            source_align = source_file
            template_align = template_file
            print("\nALIGNED FILE:\n"+re.sub(".*/", "",source_file))
            print("ALIGNMENT MASK:\n"+re.sub(".*/", "",mask_file)+"\n")
            
    if OVERWRITE or (len(alignfiles) == 0):
        # set output names
        source_align = ALIGNDIR+"/"
        source_align += (re.sub(".*/", "",source_file)).replace(".fits",
                         "_align.fits")
        mask_file = ALIGNMASKDIR+"/"
        mask_file += (re.sub(".*/", "",source_align)).replace(".fits", 
                      "_mask.fits")
        
        # first, try astroalign
        ret = amakihi.image_align(source_file, template_file, 
                                  astrometry=True, # use astrometry.net
                                  astrom_sigma=ASTROMSIGMA,
                                  thresh_sigma=SEGMSIGMA,
                                  output_im=source_align,
                                  output_mask=mask_file)
        
        # if astroalign fails        
        if ret == None: 
            # try image_registration
            ret = amakihi.image_align_fine(source_file, template_file,
                                           output_im=source_align,
                                           output_mask=mask_file,
                                           maxoffset=MAXOFFSET)
            # if image_registration fails too, can't do anything
            if ret == None: 
                print("\nCopying the science file to the directory holding "
                      "files which could not be properly aligned.\n")
                run("cp "+sci_files[i]+" "+ALIGNFAILDIR, shell=True)
                continue 
            # if image_registration succeeds
            else: 
                source_file = source_align # update
                mask_file = mask_file 
                
        # if astroalign succeeds        
        elif DOUBLEALIGN: 
            # set doubly-aligned output names
            source_align_align = ALIGNDIR+"/"
            source_align_align += (re.sub(".*/", "",source_file)).replace(
                                   ".fits", "_align_align.fits")
            mask_file_align = ALIGNMASKDIR+"/"
            mask_file_align += (re.sub(".*/", "",source_align_align)).replace(
                               ".fits", "_mask.fits")
            # fine alignment with image_registration
            ret = amakihi.image_align_fine(source_align, template_file,
                                           mask_file=mask_file,
                                           output_im=source_align_align,
                                           output_mask=mask_file_align,
                                           maxoffset=MAXOFFSET)
            if ret: # if image_registration succeeds
                run("rm "+source_align, shell=True) # get rid of singly aligned
                run("rm "+mask_file, shell=True)
                source_file = source_align_align # update
                mask_file = mask_file_align 
            else: # if image_registration fails
                source_file = source_align # update
                
        source_align = source_file # cropped and aligned 
        template_align = template_file # cropped and aligned (no change)

        print("WRITING TO:\n"+source_file+"\n"+mask_file+"\n")

    ### BACKGROUND SUBTRACTION ###########################################
    print("*******************************************************")
    print("Background subtraction...")
    print("*******************************************************\n")
    
    bkgsubfiles_source = []
    bkgsubfiles_temp = []
    if not(OVERWRITE):
        filename_source = (re.sub(".*/", "",source_file)).replace(".fits", "")
        bkgsubfiles_source = glob.glob(BKGSUBDIR+"/"+filename_source+"*")
        filename_temp = (re.sub(".*/", "",template_file)).replace(".fits", "")
        bkgsubfiles_temp = glob.glob(BKGSUBDIR+"/"+filename_temp+"*")
        if (len(bkgsubfiles_source) == 1) and (len(bkgsubfiles_temp) == 1):
            print("A background-subtracted image was already produced. "+
                  "Skipping to next step in image differencing.")
            source_file = bkgsubfiles_source[0]
            template_file = bkgsubfiles_temp[0]
            source_bkgsub = source_file
            template_bkgsub = template_file
            print("\nBKG-SUBTRACTED SOURCE:\n"+re.sub(".*/", "",source_file))
            print("BKG-SUBTRACTED TEMPLATE:\n"+re.sub(".*/", "",template_file))
            print("\n")
    
    if OVERWRITE or (len(bkgsubfiles_source)<1) or (len(bkgsubfiles_temp)<1):
        # set output names
        source_bkgsub = BKGSUBDIR+"/"
        source_bkgsub += (re.sub(".*/", "",source_file)).replace(".fits",
                         "_bkgsub.fits")
        template_bkgsub = BKGSUBDIR+"/"
        template_bkgsub += (re.sub(".*/", "",template_file)).replace(".fits",
                           "_bkgsub.fits")
        
        # background subtraction
        amakihi.bkgsub(source_file, mask_file=mask_file, bkg_filt=(1,1),
                       output=source_bkgsub)       
        amakihi.bkgsub(template_file, mask_file=mask_file, bkg_filt=(1,1),
                       output=template_bkgsub)
            
        source_file = source_bkgsub # cropped, aligned, bkg-subtracted
        template_file = template_bkgsub # cropped, aligned, bkg-subtracted
        
        print("WRITING TO:\n"+source_file+"\n"+template_file+"\n")
    
    ### BUILDING A MASK OF SATURATED STARS ###############################
    print("*******************************************************")
    print("Building a mask of saturated stars...")
    print("*******************************************************\n")
    
    satmasks = []
    if not(OVERWRITE):
        filename = (re.sub(".*/", "",source_file)).replace(".fits", "")
        satmasks = glob.glob(SATMASKDIR+"/"+filename+"*")
        if len(satmasks) > 0:
            print("A saturation mask for the image was already produced. "+
                  "Skipping to next step in image differencing.")
            mask_file = satmasks[0]
            satmask_plot = (re.sub(".*/", "",mask_file)).replace(".fits", "")
            satmask_plot = SATMASKPLOTDIR+"/"+satmask_plot+".png"
            print("\nSATURATION MASK:\n"+re.sub(".*/", "",mask_file)+"\n")

    if OVERWRITE or (len(satmasks) == 0):
        # set output name
        mask_file = (re.sub(".*/", "",mask_file)).replace(".fits", 
                    "_satmask.fits")
        mask_file = SATMASKDIR+"/"+mask_file
        # build saturation mask
        amakihi.saturation_mask(source_file, mask_file=None, dilation_its=7, 
                                plot=True, output=mask_file)       
        satmask_plot = source_file.replace(".fits","_satmask.png")
    
        # move over plot
        run("mv "+satmask_plot+" "+SATMASKPLOTDIR, shell=True) # move
        print("WRITING TO:\n"+mask_file+"\n")

    ### MAKE AN IMAGE WITH THE SATURATION MASK APPLIED ###################
    print("*******************************************************")
    print("Plotting the aligned, background-subtracted image...")
    print("*******************************************************\n")
    
    # set output name
    implot = PLOTDIR+"/"+(re.sub(".*/", "",source_bkgsub)).replace(".fits",
                         "_asinh.png")
    # plotting
    amakihi.make_image(source_bkgsub, mask_file=mask_file, output=implot, 
                       scale="asinh", cmap="viridis")
    print("WRITING TO:\n"+implot+"\n")
    
    """
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
                                                  elongation_lim=ELONGLIM,
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
    """

    ### IMAGE DIFFERENCING WITH HOTPANTS #################################
    print("*******************************************************")
    print("Performing image subtraction with hotpants...")
    print("*******************************************************")
    
    # set output names
    subfile = (re.sub(".*/", "",source_file)).replace(
               ".fits", "_subtracted.fits")
    subfile = SUBDIR+"/"+subfile
    submask = (re.sub(".*/", "",source_file)).replace(
               ".fits", "_submask.fits")
    submask = SUBMASKDIR+"/"+submask

    # hotpants
    if TMP_CFHT: # both science and template are from CFHT?
        ret = amakihi.hotpants(source_file, template_file, mask_file, 
                               iu=50000, il=-100.0, tu=50000, tl=-100.0,
                               ng="3 6 2.5 4 5.0 2 10.0", 
                               bgo=0, ko=0, v=0, 
                               rkernel=2.5*5.0,
                               output=subfile,
                               mask_write=False,
                               maskout=submask)
    else:
        ret = amakihi.hotpants(source_file, template_file, mask_file, 
                               iu=50000, il=-500.0, bgo=0, ko=0,
                               #ng="3 6 2.5 4 5.0 2 10.0", 
                               #rkernel=2.5*5.0,
                               convi=True, v=0,
                               output=subfile,
                               mask_write=False,
                               maskout=submask)
    
    if (type(ret) == np.ndarray): # if successful subtraction 
        run("mv "+BASEDIR+"*.png "+SUBPLOTDIR, shell=True) # diffim plot
        print("\nWRITING TO:\n"+subfile+"\n")
        
        print("*******************************************************")
        print("Performing transient detection...")
        print("*******************************************************")
        try:
            tabfile = CANDIDATEDIR+"/"+re.sub(".*/", "",source_align)
            tabfile = tabfile.replace(".fits", "_candidates.fits")
            tab = amakihi.transient_detect_hotpants(subfile, source_align, 
                                                elongation_lim=ELONGLIM,
                                                output=tabfile,
                                                plots=PLOTCANDS,
                                                og_scale="asinh",
                                                crosshair_sub="#fe01b1",
                                                sigma=TRANSIENTSIGMA)
                
        except: # except any errors
            print("\nSome error occurred during transient detection. "+
                  "No transients will be reported.")
    else:
        print("\nCopying the science file to the directory holding files "+
              "which caused some error for hotpants.\n")
        run("cp "+sci_files[i]+" "+HOTPANTSFAILDIR, shell=True)
    
    end = timer()
    print("\nTIME = %.4f s"%(end-start))

