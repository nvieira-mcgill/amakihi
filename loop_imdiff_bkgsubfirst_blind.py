#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:50:00 2019
@author: Nicholas Vieira
@loop_imdiff_bkgsubfirst_blind.py
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
BASEDIR = "/data/irulan/cfht/"

# base data directories (for files w CFHT templates)
#SCI_DIR = f"{BASEDIR}GW190814_coadds_science_cropped" # science (CROPPED)
#TMP_DIR = f"{BASEDIR}GW190814_coadds_templates_cropped" # template (CROPPED)
#WORKDIR = f"{BASEDIR}GW190814_imdiff_workdir/" # top working directory 

# base data directories (for files w DECaLS templates)
#SCI_DIR = f"{BASEDIR}GW190814_coadds_science_notemplate_cropped50" # science 
#TMP_DIR = f"{BASEDIR}GW190814_templates_DECaLS" # template 
#WORKDIR = f"{BASEDIR}GW190814_imdiff_DECaLS/" # working directory 

# base data directories (for files w PS1 templates)
SCI_DIR = f"{BASEDIR}GW190814_coadds_science_notemplate_croppedhalf" # science 
TMP_DIR = f"{BASEDIR}GW190814_templates_PS1" # template 
WORKDIR = f"{BASEDIR}GW190814_imdiff_PS1/" # working directory 

# image differencing directories:
ALIGNDIR = f"{WORKDIR}aligndir" # aligned only 
ALIGNMASKDIR = f"{WORKDIR}alignmaskdir" # alignment masks
BKGSUBDIR = f"{WORKDIR}bkgsubdir" # aligned, background-subtracted images
SATMASKDIR = f"{WORKDIR}satmaskdir" # saturation masks
SATMASKPLOTDIR = f"{SATMASKDIR}/plots"
PLOTDIR = f"{WORKDIR}plotdir" # plotting directory 
SUBDIR = f"{WORKDIR}subdir" # difference images
SUBPLOTDIR = f"{SUBDIR}/plots"
SUBMASKDIR = f"{WORKDIR}submaskdir" # hotpants subtraction mask
NOISEDIR = f"{WORKDIR}noisedir" # noise
NOISEPLOTDIR = f"{NOISEDIR}/plots"
NOISESCALEDIR = f"{WORKDIR}noisescaledir" # scaled noise
NOISESCALEPLOTDIR = f"{NOISESCALEDIR}/plots"
# transient directories:
CANDIDATEDIR = f"{WORKDIR}candidatedir" # tables of potential transients
CANDIDATEPLOTDIR = f"{CANDIDATEDIR}/plots" 
TRIPDIR = f"{WORKDIR}tripletdir" # tables of triplets
TRIPPLOTDIR = f"{TRIPDIR}/plots" 
# failed files:
ALIGNFAILDIR = f"{WORKDIR}alignfaildir" # failed both alignments 
HOTPANTSFAILDIR = f"{WORKDIR}hotpantsfaildir" # failed hotpants images
# transient detection diagnostics: 
ELONGDIR = f"{WORKDIR}elongation_histograms"
AREADIR = f"{WORKDIR}area_histograms"
ELONGAREASCATTERS = f"{WORKDIR}elongation_area_scatters"
REJECTIONDIR = f"{WORKDIR}rejection_plots"

## look for existing files or just overwrite them?
OVERWRITE = False # overwrite alignments, background subtractions, masks?
OVERWRITEHOTPANTS = False # overwrite hotpants difference images?
OVERWRITETRANSIENTS = False # overwrite transient detections?

## templates
TMP_CFHT = False # are templates also from CFHT? 
SURVEY = "PS1" # if not from CFHT, which survey (PS1 or DECaLS)?

if SURVEY == "PS1": TU = 100000; TL = -500 # hotpants tu/tl

## alignment
DOUBLEALIGN = False # do double alignment (if possible)?
PLOTSOURCES = True # plot sources input to astroalign? 
SEGMSIGMA = 3.0 # sigma for source detect with img segmentation in image_align
MAXOFFSET = 200.0 # maximum allowed pixel offset for image_align_morph

## sigma(s)/nsources for source detection with astrometry.net
ASTROMSIGMA = 5.0 # will be overriden if gz band observations or i20190817
PSFSIGMA = [5.0, 5.0]
NSOURCES = 100 # limit on no. of sources to use in image_align

## transient detection
TRANSIENTSIGMA = 5.0 # subtraction image transient detection sigma
PIXELMIN = 15 # minimum required isophotal sq. pix. area of transient
# set any of the following to None to impose no limit
DIPOLEWIDTH = 2.0 # maximum dipole width to look for (in arcsec)
ETAMAX = 2.0 # maximum allowed source elongation
AREAMAX = 400.0 # maximum allowed sq. pix area 
NSOURCE_MAX = 100 # maximum allowed total no. of transients

## what to do with found candidates
# ["full", "zoom og", "zoom ref", "zoom diff"] --> plot all
# ["zoom og", "zoom ref", "zoom diff"] --> plot only postage stamps
# None OR False--> plot none 
PLOTCANDS = None # plot candidate transients?

## triplets
TRIPSIZE = 63 # triplet size
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
    sci_files = glob.glob(f"{SCI_DIR}/{POINTINGS}region{REGION}*.fits")
    sci_files.sort()
else:
    sci_files = glob.glob(f"{SCI_DIR}/{POINTINGS}region*.fits")
    sci_files.sort()
    
# ignore any xy files, if present
sci_files = [s for s in sci_files.copy() if not(".xy.fits" in s)]

if not(TMP_CFHT) and SURVEY == "DECaLS":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] in "grz")]
elif not(TMP_CFHT) and SURVEY == "PS1":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] in "gri")]   

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
    tmp_files.append(f"{TMP_DIR}/{topfile}template{toapp}.fits")

#### ITERATE THROUGH ALL OF THE FILES #####################################
al_success = 0 # keep track of alignment success
al_fail = 0
sub_success = 0 # keep track of subtraction success
sub_fail = 0
nfiles = len(sci_files)
for i in range(nfiles):
    start = timer()
    
    source_file, template_file = sci_files[i], tmp_files[i]

    if not(OVERWRITETRANSIENTS):
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        candfiles = glob.glob(f"{CANDIDATEDIR}/{filename}*.fits") 
        if len(candfiles) > 0:
            print("\nImage differencing and transient detection has already "+
                  "been performed on this science file. Skipping to next "
                  "science file.\n")
            continue
    
    print("\n*******************************************************")
    print("                     ORIGINAL IMAGES:")
    print("*******************************************************")
    print(f'science file: {re.sub(".*/", "",source_file)}')
    print(f'template file: {re.sub(".*/", "",template_file)}\n')

    if not(TMP_CFHT) and SURVEY=="DECaLS":
        ASTROMSIGMA = [3.0, 3.0] # z band
    elif not(TMP_CFHT) and SURVEY=="PS1": # g or i band
        # g band, 20190816 or i band, 20190817
        if ("0816" in source_file) or ("0817" in source_file): 
            ASTROMSIGMA = [3.0, 3.0]

    ### BACKGROUND SUBTRACTION ###########################################    
    bkgsubfiles_source = []
    bkgsubfiles_temp = []
    if not(OVERWRITE):
        filename_source = (re.sub(".*/", "",source_file)).replace(".fits", "")
        bkgsubfiles_source = glob.glob(f"{BKGSUBDIR}/{filename_source}*.fits")
        filename_temp = (re.sub(".*/", "",template_file)).replace(".fits", "")
        bkgsubfiles_temp = glob.glob(f"{BKGSUBDIR}/{filename_temp}*.fits")
        if (len(bkgsubfiles_source) == 1) and (len(bkgsubfiles_temp) == 1):
            print("A background-subtracted image was already produced. "+
                  "Skipping to next step in image differencing.")
            source_file = bkgsubfiles_source[0]
            template_file = bkgsubfiles_temp[0]
            source_bkgsub = source_file
            template_bkgsub = template_file
            print(f'\nBKG-SUB. SOURCE:\n{re.sub(".*/","",source_file)}')
            print(f'BKG-SUB. TEMPLATE:\n{re.sub(".*/","",template_file)}')
            print("\n")
    
    if OVERWRITE or (len(bkgsubfiles_source)<1) or (len(bkgsubfiles_temp)<1):
        print("*******************************************************")
        print("Background subtraction...")
        print("*******************************************************")
        # set output names
        source_bkgsub = f"{BKGSUBDIR}/"
        source_bkgsub += (re.sub(".*/", "",source_file)).replace(".fits",
                         "_bkgsub.fits")
        template_bkgsub = f"{BKGSUBDIR}/"
        template_bkgsub += (re.sub(".*/", "",template_file)).replace(".fits",
                           "_bkgsub.fits")
        
        # background subtraction
        amakihi.bkgsub(source_file, bkg_filt=(5,5), output=source_bkgsub)       
        amakihi.bkgsub(template_file, bkg_filt=(5,5), output=template_bkgsub)
            
        source_file = source_bkgsub # cropped, bkg-subtracted
        template_file = template_bkgsub # cropped, bkg-subtracted
        
        print("WRITING TO:\n"+source_file+"\n"+template_file+"\n")

    ### IMAGE ALIGNMENT ##################################################
    # align with astroalign and then use image_registration for alignment 
    # based on morphology
    # if not possible, use image_registration only 
    
    alignfiles = []
    if not(OVERWRITE):
        filename = (re.sub(".*/", "",source_file)).replace(".fits", "")
        alignfiles = glob.glob(f"{ALIGNDIR}/{filename}*.fits")
        if len(alignfiles) > 0:
            print("An aligned image was already produced. Skipping to next "+
                  "step in image differencing.")
            source_file = alignfiles[0]
            mask_file = (re.sub(".*/", "",source_file)).replace(".fits", 
                                                                "_mask.fits")
            mask_file = f"{ALIGNMASKDIR}/{mask_file}"
            source_align = source_file
            template_align = template_file
            print(f'\nALIGNED FILE:\n{re.sub(".*/", "",source_file)}')
            print(f'ALIGNMENT MASK:\n{re.sub(".*/", "",mask_file)}\n')
            
    if OVERWRITE or (len(alignfiles) == 0):
        print("*******************************************************")
        print("Image alignment...")
        print("*******************************************************")
        # set output names
        source_align = f"{ALIGNDIR}/"
        source_align += (re.sub(".*/", "",source_file)).replace(".fits",
                         "_align.fits")
        template_align = template_file
        mask_file = f"{ALIGNMASKDIR}/"
        mask_file += (re.sub(".*/", "",source_align)).replace(".fits", 
                      "_mask.fits")
        
        # first, try astroalign
        ret = amakihi.image_align(source_file, template_file, 
                                  bkgsubbed=True, # already bkg-subtracted
                                  astrometry=True, # use astrometry.net
                                  astrom_sigma=ASTROMSIGMA,
                                  psf_sigma=PSFSIGMA,
                                  nsources=NSOURCES,
                                  thresh_sigma=SEGMSIGMA,
                                  plot_sources=PLOTSOURCES,
                                  output_im=source_align,
                                  output_mask=mask_file)        
        # if astroalign fails        
        if ret == None: 
            # try image_registration
            ret = amakihi.image_align_morph(source_file, template_file,
                                            output_im=source_align,
                                            output_mask=mask_file,
                                            maxoffset=MAXOFFSET)
            # if image_registration fails too, can't do anything
            if ret == None: 
                print("\nCopying the science file and template to the "+
                      "directory holding files which could not be properly "+
                      "aligned.\n")
                run(f"cp {source_file} {ALIGNFAILDIR}", shell=True)
                run(f"cp {template_file} {ALIGNFAILDIR}", shell=True)
                al_fail += 1
                print("Alignment success so far: "+
                      f"{al_success}/{al_success+al_fail}"+
                      f" = {al_success/(al_success+al_fail)*100}%\n")
                continue 
            # if image_registration succeeds
            else: 
                source_file = source_align # update
                mask_file = mask_file 
                al_success += 1 
                print("Alignment success so far: "+
                      f"{al_success}/{al_success+al_fail}"+
                      f" = {al_success/(al_success+al_fail)*100}%\n")
                
        # if astroalign succeeds + double alignment    
        elif DOUBLEALIGN:
            # set doubly-aligned output names
            source_align_align = f"{ALIGNDIR}/"
            source_align_align += (re.sub(".*/", "",source_file)).replace(
                                   ".fits", "_align_align.fits")
            mask_file_align = f"{ALIGNMASKDIR}/"
            mask_file_align += (re.sub(".*/", "",source_align_align)).replace(
                               ".fits", "_mask.fits")
            # alignment with image_registration
            ret = amakihi.image_align_morph(source_align, template_file,
                                            mask_file=mask_file,
                                            output_im=source_align_align,
                                            output_mask=mask_file_align,
                                            maxoffset=MAXOFFSET)
            if ret: # if image_registration succeeds
                run(f"rm {source_align}", shell=True) # get rid of singly align
                run(f"rm {mask_file}", shell=True)
                source_file = source_align_align # update
                mask_file = mask_file_align 
            else: # if image_registration fails
                source_file = source_align # update
            al_success += 1
            print("Alignment success so far: "+
                  f"{al_success}/{al_success+al_fail}"+
                  f" = {al_success/(al_success+al_fail)*100}%\n")
        
        # astroalign succeeds but no double alignment
        else:
            source_file = source_align
            al_success += 1
            print("Alignment success so far: "+
                  f"{al_success}/{al_success+al_fail}"+
                  f" = {al_success/(al_success+al_fail)*100}%\n")

        # files are now cropped, bkg-subtracted, and aligned
        print(f"WRITING TO:\n{source_file}\n{mask_file}\n")
        
        
    ### BUILDING A MASK OF SATURATED STARS ###############################    
    old_mask_file = mask_file
    
    satmasks = []
    if not(OVERWRITE):
        filename = (re.sub(".*/", "",mask_file)).replace(".fits", "")
        satmasks = glob.glob(f"{SATMASKDIR}/{filename}*.fits")
        if len(satmasks) > 0:
            print("A saturation mask for the image was already produced. "+
                  "Skipping to next step in image differencing.")
            mask_file = satmasks[0]
            satmask_plot = (re.sub(".*/", "",mask_file)).replace(".fits", "")
            satmask_plot = f"{SATMASKPLOTDIR}/{satmask_plot}.png"
            print(f'SATURATION MASK:\n{re.sub(".*/", "",mask_file)}\n')

    if OVERWRITE or (len(satmasks) == 0):
        print("*******************************************************")
        print("Building a mask of saturated stars...")
        print("*******************************************************")
        # set output names
        mask_file = (re.sub(".*/", "",mask_file)).replace(".fits", 
                    "_satmask.fits")
        mask_file = f"{SATMASKDIR}/{mask_file}"
        satmask_plot = (re.sub(".*/", "",source_file)).replace(".fits", 
                        "_satmask.png")
        satmask_plot = f"{SATMASKPLOTDIR}/{satmask_plot}"
        
        # build saturation mask
        amakihi.saturation_mask(source_file, mask_file=old_mask_file, 
                                dilation_its=7, output=mask_file,
                                plot=True, plotname=satmask_plot)       

        print(f"WRITING TO:\n{mask_file}\n")

    ### MAKE AN IMAGE WITH THE SATURATION MASK APPLIED ###################
#    print("*******************************************************")
#    print("Plotting the aligned, background-subtracted image...")
#    print("*******************************************************\n")
#    
#    # set output name
#    implot = f"{PLOTDIR}/"+(re.sub(".*/", "",source_bkgsub)).replace(".fits",
#                         "_asinh.png")
#    # plotting
#    amakihi.make_image(source_bkgsub, mask_file=mask_file, output=implot, 
#                       scale="asinh", cmap="viridis")
#    print(f"WRITING TO:\n{implot}\n")
    

    ### IMAGE DIFFERENCING WITH HOTPANTS #################################
    # for later
    ret = None
    subfiles = []

    if not(OVERWRITEHOTPANTS): # check that files do not exist
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        subfiles = glob.glob(f"{SUBDIR}/{filename}*.fits") 
        if len(subfiles) > 0:
            print("A difference image already exists for this science file."+
                  " Skipping to transient detection.\n")
            subfile = subfiles[0]
            print(f'\nDIFFERENCE IMAGE:\n{re.sub(".*/", "",subfile)}\n')
    
    # set output names
    subfile = f"{SUBDIR}/"+(re.sub(".*/", "",source_file)).replace(".fits", 
                         "_subtracted.fits")
    submask = f"{SUBMASKDIR}/"+(re.sub(".*/", "",source_file)).replace(".fits", 
                             "_submask.fits")
    subplot = f"{SUBPLOTDIR}/"+(re.sub(".*/", "",source_file)).replace(".fits", 
                         "_subtracted_hotpants.png")
    subnoise = f"{NOISEDIR}/"+(re.sub(".*/", "",source_file)).replace(".fits", 
                            "_subnoise.fits")
    subnoisescale = f"{NOISESCALEDIR}/"
    subnoisescale += (re.sub(".*/", "",source_file)).replace(".fits", 
                     "_subnoise_scaled.fits") 

    ## hotpants
    if OVERWRITEHOTPANTS or (len(subfiles) == 0):  
        print("*******************************************************")
        print("Performing image subtraction with hotpants...")
        print("*******************************************************")
        if TMP_CFHT: 
            ret = amakihi.hotpants(source_file, 
                                   template_file, 
                                   mask_file, 
                                   param_estimate=False,
                                   iu=50000, il=-200.0, tu=50000, tl=-200.0,
                                   ng="3 6 0.95 4 1.9 2 3.8", 
                                   bgo=0, ko=0, v=0, 
                                   rkernel=2.5*4.42,
                                   norm="i",
                                   output=subfile,
                                   mask_write=True,
                                   maskout=submask,
                                   noise_write=True, 
                                   noiseout=subnoise,
                                   noise_scale_write=True,
                                   noisescaleout=subnoisescale,
                                   plot=True,
                                   plotname=subplot)
        else: # need to update this?
            if SURVEY == "DECaLS":
                TU = 1.01*np.max(np.ravel(fits.getdata(source_file)))
                TL = -0.1
            ret = amakihi.hotpants(source_file, 
                                   template_file, 
                                   mask_file, 
                                   param_estimate=False,
                                   iu=50000, il=-200.0,
                                   tu=TU, tl=TL,
                                   ng="3 6 0.95 4 1.9 2 3.8", 
                                   bgo=0, ko=1, v=0, 
                                   rkernel=2.5*4.42,
                                   convi=True, 
                                   norm="i",
                                   output=subfile,
                                   mask_write=True,
                                   maskout=submask,
                                   noise_write=True, 
                                   noiseout=subnoise,
                                   noise_scale_write=True,
                                   noisescaleout=subnoisescale,
                                   plot=True,
                                   plotname=subplot)
            
    # if successful subtraction on this loop specifically
    if (type(ret) == np.ndarray):
        print(f"\nWRITING TO:\n{subfile}\n")
        
    # if existing subtraction was found OR successful subtraction on this loop       
    if (not(OVERWRITEHOTPANTS) and len(subfiles)>0) or (type(ret)==np.ndarray): 
        sub_success += 1
        print("Subtraction success so far: "+
              f"{sub_success}/{sub_success+sub_fail}"+
              f" = {sub_success/(sub_success+sub_fail)*100}%\n")     
        
#        print("*******************************************************")
#        print("Plotting unscaled/scaled noise images...")
#        print("*******************************************************")
#
#        # set output names
#        subnoiseplot = f"{NOISEPLOTDIR}/"
#        subnoiseplot += (re.sub(".*/", "",source_file)).replace(".fits", 
#                        "_subnoise.png") 
#        subnoisescaleplot = f"{NOISESCALEPLOTDIR}/"
#        subnoisescaleplot += (re.sub(".*/", "",source_file)).replace(".fits", 
#                              "_subnoise_scaled.png") 
#
#        # make images 
#        amakihi.make_image(subnoise, mask_file=submask, output=subnoiseplot, 
#                           cmap="viridis")
#        amakihi.make_image(subnoisescale, mask_file=submask, 
#                           output=subnoisescaleplot, 
#                           cmap="viridis")       
#        print(f"WRITING TO:\n{subnoiseplot}\n{subnoisescaleplot}\n")
        
        
        print("*******************************************************")
        print("Performing transient detection...")
        print("*******************************************************")
        
        try:
            tabfile = f'{CANDIDATEDIR}/{re.sub(".*/", "",source_align)}'
            tabfile = tabfile.replace(".fits", "_candidates.fits")
            tab = amakihi.transient_detect(subfile, 
                                           source_align, 
                                           template_align,
                                           mask_file=mask_file, # sat mask
                                           thresh_sigma=TRANSIENTSIGMA,
                                           pixelmin=PIXELMIN,
                                           dipole_width=DIPOLEWIDTH,
                                           etamax=ETAMAX,
                                           area_max=AREAMAX,
                                           nsource_max=NSOURCE_MAX,
                                           output=tabfile,
                                           plot_distributions=True,
                                           plot_rejections=True,
                                           plots=PLOTCANDS)
            
            if (type(tab) == QTable): # if any transients are found 
                print("\n****************************************************"+
                      "***")
                print("Creating triplets...")
                print("*****************************************************"+
                      "**")
                trip = f'{TRIPDIR}/{re.sub(".*/", "",source_align)}'
                trip = trip.replace(".fits", "_candidates_triplets.npy")
                t = amakihi.transient_triplets(subfile, 
                                               source_align, 
                                               template_align, 
                                               tab,
                                               size=TRIPSIZE,
                                               output=trip, plot=PLOTTRIP,
                                               plotdir=TRIPPLOTDIR)                   
                
        except: # except any errors
            e = sys.exc_info()
            print("\nSome error occurred during transient detection/triplet "+
                  f"writing: \n{str(e[0])}\n{str(e[1])}"+
                  "\nNo transients will be reported.")

        # move plots produced for bugtesting
        run(f"mv {ALIGNDIR}/*elongs_areas.png {ELONGAREASCATTERS}", shell=True)
        run(f"mv {ALIGNDIR}/*elongs.png {ELONGDIR}", shell=True)
        run(f"mv {ALIGNDIR}/*areas.png {AREADIR}", shell=True)
        run(f"mv {SUBDIR}/*rejections.png {REJECTIONDIR}", shell=True)
        
            
    else: # if hotpants did not work
        print("\nCopying the science file to the directory holding files "+
              "which caused some error for hotpants.\n")
        run(f"cp {sci_files[i]} {HOTPANTSFAILDIR}", shell=True)
        run(f"rm {subfile}", shell=True)
        sub_fail += 1
        print("Subtraction success so far: "+
              f"{sub_success}/{sub_success+sub_fail}"+
              f" = {sub_success/(sub_success+sub_fail)*100}%\n")
    
    end = timer()
    print(f"\nTIME = {(end-start):.4f} s")

