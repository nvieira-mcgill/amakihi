#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:46:30 2019
@author: Nicholas Vieira
@loop_imdiff_blind.py 
"""

import amakihi 
import sys
from subprocess import run
import glob 
import re
import numpy as np
from timeit import default_timer as timer
from astropy.io import fits 
from astropy.table import QTable

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
# base data directories (for files w CFHT templates)
BASEDIR = "/data/irulan/cfht/"
#SCI_DIR = BASEDIR+"GW190814_coadds_science_cropped" # science (CROPPED)
#TMP_DIR = BASEDIR+"GW190814_coadds_templates_cropped" # template (CROPPED)
#WORKDIR = BASEDIR+"GW190814_imdiff_workdir/" # top working directory 

# base data directories (for files w DECaLS templates)
SCI_DIR = BASEDIR+"GW190814_coadds_science_notemplate_cropped50" # science 
TMP_DIR = BASEDIR+"GW190814_templates_DECaLS" # template 
WORKDIR = BASEDIR+"GW190814_imdiff_DECaLS/" # working directory 

# base data directories (for files w PS1 templates)
#SCI_DIR = BASEDIR+"GW190814_coadds_science_notemplate_croppedhalf" # science 
#TMP_DIR = BASEDIR+"GW190814_templates_PS1" # template 
#WORKDIR = BASEDIR+"GW190814_imdiff_PS1/" # working directory 

# image differencing directories:
ALIGNDIR = WORKDIR+"aligndir" # aligned only 
ALIGNMASKDIR = WORKDIR+"alignmaskdir" # alignment masks
BKGSUBDIR = WORKDIR+"bkgsubdir" # aligned, background-subtracted images
SATMASKDIR = WORKDIR+"satmaskdir" # saturation masks
SATMASKPLOTDIR = SATMASKDIR+"/plots"
PLOTDIR = WORKDIR+"plotdir" # plotting directory 
SUBDIR = WORKDIR+"subdir" # difference images
SUBPLOTDIR = SUBDIR+"/plots"
SUBMASKDIR = WORKDIR+"submaskdir" # hotpants subtraction mask
# transient directories:
CANDIDATEDIR = WORKDIR+"candidatedir" # tables of potential transients
CANDIDATEPLOTDIR = CANDIDATEDIR+"/plots" 
TRIPDIR = WORKDIR+"tripletdir" # tables of triplets
TRIPPLOTDIR = TRIPDIR+"/plots" 
# where failed files go:
ALIGNFAILDIR = WORKDIR+"alignfaildir" # failed both alignments 
HOTPANTSFAILDIR = WORKDIR+"hotpantsfaildir" # failed hotpants images

## look for existing files or just overwrite them?
OVERWRITE = False # overwrite alignments, background subtractions, masks?
OVERWRITEHOTPANTS = True # overwrite hotpants difference images?
OVERWRITETRANSIENTS = True # overwrite transient detections?

## templates
TMP_CFHT = False # are templates also from CFHT? 
SURVEY = "DECaLS" # if not from CFHT, which survey (PS1 or DECaLS)?

## alignment
DOUBLEALIGN = False # do double alignment (if possible)?
ASTROMSIGMA = [3, 5] # sigma(s) for source detection with astrometry.net
SEGMSIGMA = 3.0 # sigma for source detect with img segmentation in image_align
MAXOFFSET = 100.0 # maximum allowed pixel offset for image_align_fine

## transient detection
TRANSIENTSIGMA = 5.0 # subtraction image transient detection sigma
PIXELMIN = 20 # minimum required isophotal sq. pix. area of transient
# set any of the following to None to impose no limit
DIPOLEWIDTH = 1.0 # maximum dipole width to look for (in arcsec)
ELONGLIM = 1.8 # maximum allowed source elongation
NSOURCE_LIM = 40 # maximum allowed total no. of transients

## what to do with found candidates
# ["full", "zoom og", "zoom ref", "zoom diff"] --> plot all
# ["zoom og", "zoom ref", "zoom diff"] --> plot only postage stamps
# None OR False--> plot none 
PLOTCANDS = False # plot candidate transients?

## triplets
TRIPSIZE = 128 # triplet size
PLOTTRIP = True # plot triplets?

## pointing arguments
if len(sys.argv) > 1 and len(str(sys.argv[1])) == 2: # pointing given, use it
    POINTINGS = str(sys.argv[1])
else:
    POINTINGS = "50"

### LOAD IN DATA ######################################################### 
# data should be pre-cropped/not in need of cropping 

if len(sys.argv) > 2 and len(str(sys.argv[2])) == 1: # region supplied, use it
    REGION = str(sys.argv[2])
    sci_files = sorted(glob.glob(SCI_DIR+"/"+POINTINGS+"region"+REGION+
                                 "*.fits"))
else:
    sci_files = sorted(glob.glob(SCI_DIR+"/"+POINTINGS+"region*.fits"))
    
# ignore any xy files, if present
sci_files = [s for s in sci_files.copy() if not(".xy.fits" in s)]

if not(TMP_CFHT) and SURVEY == "DECaLS":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] in "grz")]
#elif not(TMP_CFHT) and SURVEY == "PS1":
#    sci_files = [s for s in sci_files.copy() if (
#            fits.getheader(s)["FILTER"][0] in "griz")]   

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
    
    if not(OVERWRITETRANSIENTS):
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        candfiles = glob.glob(CANDIDATEDIR+"/"+filename+"*") 
        if len(candfiles) > 0:
            print("\nImage differencing and transient detection has alread "+
                  "been performed on this science file. Skipping to next "
                  "science file.\n")
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
        template_align = template_file
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
                
        # if astroalign succeeds + double alignment    
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
        
        # astroalign succeeds but no double alignment
        else:
            source_file = source_align

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
    
    old_mask_file = mask_file
    
    satmasks = []
    if not(OVERWRITE):
        filename = (re.sub(".*/", "",mask_file)).replace(".fits", "")
        satmasks = glob.glob(SATMASKDIR+"/"+filename+"*")
        if len(satmasks) > 0:
            print("A saturation mask for the image was already produced. "+
                  "Skipping to next step in image differencing.")
            mask_file = satmasks[0]
            satmask_plot = (re.sub(".*/", "",mask_file)).replace(".fits", "")
            satmask_plot = SATMASKPLOTDIR+"/"+satmask_plot+".png"
            print("\nSATURATION MASK:\n"+re.sub(".*/", "",mask_file)+"\n")

    if OVERWRITE or (len(satmasks) == 0):
        # set output names
        mask_file = (re.sub(".*/", "",mask_file)).replace(".fits", 
                    "_satmask.fits")
        mask_file = SATMASKDIR+"/"+mask_file
        satmask_plot = (re.sub(".*/", "",source_file)).replace(".fits", 
                        "_satmask.png")
        satmask_plot = SATMASKPLOTDIR+"/"+satmask_plot
        
        # build saturation mask
        amakihi.saturation_mask(source_file, mask_file=old_mask_file, 
                                dilation_its=7, output=mask_file,
                                plot=True, plotname=satmask_plot)       

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
    

    ### IMAGE DIFFERENCING WITH HOTPANTS #################################
    print("*******************************************************")
    print("Performing image subtraction with hotpants...")
    print("*******************************************************")

    # for later
    ret = None
    subfiles = []

    if not(OVERWRITEHOTPANTS): # check that files do not exist
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        subfiles = glob.glob(SUBDIR+"/"+filename+"*") 
        if len(subfiles) > 0:
            print("A difference image already exists for this science file."+
                  " Skipping to transient detection.\n")
            subfile = subfiles[0]
            print("\nDIFFERENCE IMAGE:\n"+re.sub(".*/", "",subfile)+"\n")
    
    # set output names
    subfile = SUBDIR+"/"+(re.sub(".*/", "",source_file)).replace(".fits", 
                         "_subtracted.fits")
    submask = SUBMASKDIR+"/"+(re.sub(".*/", "",source_file)).replace(".fits", 
                             "_submask.fits")
    subplot = SUBPLOTDIR+"/"+(re.sub(".*/", "",source_file)).replace(".fits", 
                         "_subtracted_hotpants.png")

    ## hotpants
    if OVERWRITEHOTPANTS or (len(subfiles) == 0):        
        if TMP_CFHT: # both science and template are from CFHT?
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   iu=50000, il=-100.0, tu=50000, tl=-100.0,
                                   ng="3 6 2.5 4 5.0 2 10.0", 
                                   bgo=0, ko=0, v=0, 
                                   rkernel=2.5*5.0,
                                   output=subfile,
                                   mask_write=False,
                                   maskout=submask,
                                   plot=True,
                                   plotname=subplot)
        else:
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   iu=50000, il=-500.0, 
                                   ng="3 6 2.5 4 5.0 2 10.0", 
                                   bgo=0, ko=0, v=0, 
                                   rkernel=2.5*5.0,
                                   convi=True, 
                                   output=subfile,
                                   mask_write=False,
                                   maskout=submask,
                                   plot=True,
                                   plotname=subplot)
            
    # if successful subtraction on this loop specifically
    if (type(ret) == np.ndarray):
        print("\nWRITING TO:\n"+subfile+"\n")
        
    # if existing subtraction was found OR successful subtraction on this loop       
    if (not(OVERWRITEHOTPANTS) and len(subfiles)>0) or (type(ret)==np.ndarray): 
        
        print("*******************************************************")
        print("Performing transient detection...")
        print("*******************************************************")
        try:
            tabfile = CANDIDATEDIR+"/"+re.sub(".*/", "",source_align)
            tabfile = tabfile.replace(".fits", "_candidates.fits")
            tab = amakihi.transient_detect(subfile, source_align, 
                                           template_align,
                                           #astrom_sigma=ASTROMSIGMA,
                                           thresh_sigma=TRANSIENTSIGMA,
                                           pixelmin=PIXELMIN,
                                           dipole_width=DIPOLEWIDTH,
                                           elongation_lim=ELONGLIM,
                                           nsource_lim=NSOURCE_LIM,
                                           output=tabfile)#, 
                                           #plots=PLOTCANDS,
                                           #og_scale="asinh",
                                           #crosshair_sub="#fe01b1",
                                           #plotdir=CANDIDATEPLOTDIR)
            
            if (type(tab) == QTable): # if any transients are found 
                print("\n****************************************************"+
                      "***")
                print("Creating triplets...")
                print("*****************************************************"+
                      "**")
                trip = TRIPDIR+"/"+re.sub(".*/", "",source_align)
                trip = trip.replace(".fits", "_candidates_triplets.npy")
                t = amakihi.transient_triplets(subfile, source_align, 
                                               template_align, tabfile,
                                               size=TRIPSIZE, crosshair=None,
                                               output=trip, plot=PLOTTRIP,
                                               plotdir=TRIPPLOTDIR)                   
                
        except: # except any errors
            e = sys.exc_info()
            print("\nSome error occurred during transient detection/triplet "+
                  "writing: \n"+str(e[0])+"\n"+str(e[1])+
                  "\nNo transients will be reported.")
            
    else: # if hotpants did not work
        print("\nCopying the science file to the directory holding files "+
              "which caused some error for hotpants.\n")
        run("cp "+sci_files[i]+" "+HOTPANTSFAILDIR, shell=True)
        run("rm "+subfile, shell=True)
    
    end = timer()
    print("\nTIME = %.4f s"%(end-start))

