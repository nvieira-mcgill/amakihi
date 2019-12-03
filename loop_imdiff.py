#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:06:15 2019
@author: Nicholas Vieira
@loop_imdiff.py
"""

import amakihi 
import sys
from subprocess import run
import glob 
import re
import pandas as pd 
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.table import QTable
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
#CSV = "GW190814_50_GLADE.csv" 
CSV = "AT2019candidatesCFHT.csv" 

## directories
# base data directories:
BASEDIR = "/media/nvieira/OS/Users/nviei/Documents/Storage/" # base data dir

# i transients with CFHT templates:
WORKDIR = BASEDIR+"workdir_CFHTtmps/" # top working directory 
SCI_DIR = BASEDIR+"isci_transient" # science
TMP_DIR = BASEDIR+"itmp_transient" # template
SCI_NOTMP_DIR = BASEDIR+"isci_transient_notmp" # sci files without templates

# gi transients with PS1 templates:
#WORKDIR = BASEDIR+"workdir_PS1tmps/" # top working directory 
#SCI_DIR = BASEDIR+"gisci_transient_notmp_forPS1" # science
#TMP_DIR = BASEDIR+"gitmp_transient_PS1" # template
#SCI_NOTMP_DIR = BASEDIR+"" # sci files without templates

# z transients with DECaLS templates:
#WORKDIR = BASEDIR+"workdir_DECaLStmps/" # top working directory 
#SCI_DIR = BASEDIR+"zsci_transient_notmp_forDECaLS" # science
#TMP_DIR = BASEDIR+"ztmp_transient_DECaLS" # template
#SCI_NOTMP_DIR = BASEDIR+"" # sci files without templates

# general directories:
CROPDIR = WORKDIR+"cropdir" # cropped only
ALIGNDIR = WORKDIR+"aligndir" # cropped + aligned only 
ALIGNMASKDIR = WORKDIR+"alignmaskdir" # alignment masks
BKGSUBDIR = WORKDIR+"bkgsubdir" # cropped, aligned, background-subtracted 
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
# failed files:
ALIGNFAILDIR = WORKDIR+"alignfaildir" # failed both alignments 
HOTPANTSFAILDIR = WORKDIR+"hotpantsfaildir" # failed hotpants images
# transient detection diagnostics: 
ELONGDIR=WORKDIR+"elongation_histograms"
AREADIR=WORKDIR+"area_histograms"
ELONGAREASCATTERS=WORKDIR+"elongation_area_scatters"
REJECTIONDIR=WORKDIR+"rejection_plots"

## look for existing files or just overwrite them?
OVERWRITE = False # overwrite alignments, background subtractions, masks?
OVERWRITEHOTPANTS = True # overwrite hotpants difference images?
OVERWRITETRANSIENTS = True # overwrite transient detections?

## RA, DEC determination
RANKS = False # use the .csv to obtain RANKS
TRANSIENTS = True # use the .csv to obtain TRANSIENT NAMES
MANUAL = False # manually supply a RA, DEC below 

## cropping parameters
CROP = False # do any sort of cropping?
CROPMIN = 1000.0 # minimum crop size (pix)
SCIPRECROP = True # are science files pre-cropped?
TMPPRECROP = True # are template files pre-cropped?
TARGET_CROP = False # use the rank/transient .csv to set crop
OCTANT_CROP = False # crop according to which octant contains some target

## general subtraction arguments
## templates
TMP_CFHT = True # are templates also from CFHT? 
SURVEY = "DECaLS" # if not from CFHT, which survey (PS1 or DECaLS)?

## alignment
DOUBLEALIGN = False # do double alignment (if possible)?
SEGMSIGMA = 3.0 # sigma for source detect with img segmentation in image_align
MAXOFFSET = 100.0 # maximum allowed pixel offset for image_align_morph

## sigma(s)/nsources for source detection with astrometry.net
ASTROMSIGMA = 5.0 # will be overriden if gz band observations or i20190817
PSFSIGMA = [5.0, 5.0]
NSOURCES = 100 # limit on no. of sources to use in image_align

## transient detection
TRANSIENTSIGMA = 3.0 # subtraction image transient detection sigma
PIXELMIN = 10 # minimum required sq. pix. area of transient
# set any of the following to 0 to impose no limit
DIPOLEWIDTH = 1.0 # maximum dipole width to look for (")
ETAMAX = 2.0 # maximum allowed source elongation
AREAMAX = 500.0 # maximum allowed sq. pix area of transient
NSOURCE_MAX = 50 # maximum allowed total no. of transients
SEPLIM = 5.0 # max allowed sep between transient and target of interest (")

## what to do with found candidates
# ["full", "zoom og", "zoom ref", "zoom diff"] --> plot all
# ["zoom og", "zoom ref", "zoom diff"] --> plot only postage stamps
# None OR False--> plot none 
#PLOTCANDS = False # plot candidate transients?
PLOTCANDS = False #["zoom og", "zoom ref", "zoom diff"]

## triplets
TRIPSIZE = 63 # triplet size
PLOTTRIP = True # plot triplets?

## path to hotpants (needed when running on my machine)
amakihi.hotpants_path("~/hotpants/hotpants")

### LOAD IN DATA #########################################################
df = pd.read_csv(CSV) # get coordinates 
sci_files = sorted(glob.glob(SCI_DIR+"/*.fits"))

# ignore any xy files, if present
sci_files = [s for s in sci_files.copy() if not(".xy.fits" in s)]

if not(TMP_CFHT) and SURVEY == "DECaLS":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] in "grz")]
#elif not(TMP_CFHT) and SURVEY == "PS1":
#    sci_files = [s for s in sci_files.copy() if (
#            fits.getheader(s)["FILTER"][0] == "griz")] 

if RANKS: # if using rankXX.fits files
    ranks = df["Rank"] # get ranks for each object 
    tmp_files = []
    for s in sci_files:
        s = s.replace(".fits","")
        while s[-1].isnumeric():
            s = s[:-1]
        topfile = re.sub(".*/", "", s) # from /a/b/c, extract c
        tmp_files.append(TMP_DIR+"/"+topfile+"final.fits")
        
elif TRANSIENTS:
    names = df["ID"]
    tmp_files = []
    for s in sci_files:
        if SCIPRECROP:
            s = s.replace(".fits", "")[:-13] # remove "YYYYMMDD_crop"
        else:
            s = s.replace(".fits", "")[:-8] # remove "YYYYMMDD"
        topfile = re.sub(".*/", "", s) # from /a/b/c, extract c
        
        if TMPPRECROP:
            tmp_files.append(TMP_DIR+"/"+topfile+"template_crop.fits")        
        else:
            tmp_files.append(TMP_DIR+"/"+topfile+"template.fits")
    
    
else: # if looping over many images of some object with MANUALLY input coords
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
    
    try: # load in data and header from SCIENCE files
        sci = fits.getdata(sci_files[i])
        sci_hdr = fits.getheader(sci_files[i])
    except FileNotFoundError:
        print("Did not find the requested science files. Skipping.")
        continue
    
    try: # load in data and header from TEMPLATE files 
        tmp = fits.getdata(tmp_files[i])
        tmp_hdr = fits.getheader(tmp_files[i])
    except FileNotFoundError: # template not found --> look for similar files
        try:
            if RANKS and ("_50_" in CSV): # determine rank, update
                rank_sci = re.sub(".*/", "",sci_files[i])[4:6] 
                tmp_files[i] = glob.glob(TMP_DIR+"/rank"+rank_sci+"*")[0] 
                
            elif RANKS and ("_90_" in CSV): # determine rank, update
                rank_sci = re.sub(".*/", "",sci_files[i])[6:8] 
                tmp_files[i] = glob.glob(TMP_DIR+"/90rank"+rank_sci+"*")[0] 
                
            elif TRANSIENTS: # determine transient name, update
                name_sci = re.sub(".*/", "",sci_files[i])[:7] 
                tmp_files[i] = glob.glob(TMP_DIR+"/"+name_sci+"*")[0] 
                
                tmp = fits.getdata(tmp_files[i])
                tmp_hdr = fits.getheader(tmp_files[i])
        except:
            print("Did not find a corresponding template for the science "+
                  "file. Moving the science file to a separate directory "+
                  " and skipping to the next science file.")
            run("mv "+sci_files[i]+" "+SCI_NOTMP_DIR, shell=True)
            continue

    source_file, template_file = sci_files[i], tmp_files[i]
    
    if not(OVERWRITETRANSIENTS):
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        candfiles = glob.glob(CANDIDATEDIR+"/"+filename+"*") 
        if len(candfiles) > 0:
            print("\nImage differencing and transient detection has already "+
                  "been performed on this science file. Skipping to next "
                  "science file.\n")
            continue
        
    print("\n*******************************************************")
    print("                     ORIGINAL IMAGES:")
    print("*******************************************************")
    print("science file: "+re.sub(".*/", "",source_file))
    print("template file: "+re.sub(".*/", "",template_file)+"\n")
    
    if not(TMP_CFHT) and SURVEY=="DECaLS":
        ASTROMSIGMA = [3.0, 5.0] # z band
    elif not(TMP_CFHT) and SURVEY=="PS1": # g or i band
        # g band, 20190816 or i band, 20190817
        if ("0816" in source_file) or ("0817" in source_file): 
            ASTROMSIGMA = [3.0, 5.0]

    ### DETERMINE RA, DEC OF SOURCE OF INTEREST ########################### 
    if RANKS and ("_50_" in CSV): # use csv to get RA, Dec for each rank 
        rank_sci = re.sub(".*/", "",sci_files[i])[4:6] # determine rank 
        rank_tmp = re.sub(".*/", "",tmp_files[i])[4:6]
        if int(rank_sci) != int(rank_tmp): 
            print("The images being compared are not of the same field. "+
                  "Exiting.")
            sys.exit()      
        ra = float((df.loc[df["Rank"] == int(rank_sci)])["RAJ2000"])
        dec = float((df.loc[df["Rank"] == int(rank_sci)])["DEJ2000"])
        
    if RANKS and ("_90_" in CSV): # use csv to get RA, Dec for each rank 
        rank_sci = re.sub(".*/", "",sci_files[i])[6:8] # determine rank 
        rank_tmp = re.sub(".*/", "",tmp_files[i])[6:8]
        if int(rank_sci) != int(rank_tmp): 
            print("The images being compared are not of the same field. "+
                  "Exiting.")
            sys.exit()      
        ra = float((df.loc[df["Rank"] == int(rank_sci)])["RAJ2000"])
        dec = float((df.loc[df["Rank"] == int(rank_sci)])["DEJ2000"])
    
    
    elif TRANSIENTS: # use csv to get RA, Dec for each transient ID 
        name_sci = re.sub(".*/", "",sci_files[i])[:7] # determine name
        name_tmp = re.sub(".*/", "",tmp_files[i])[:7]
        if name_sci != name_tmp: 
            print("The images being compared are not of the same field. "+
                  "Exiting.")
            sys.exit()
            
        ra = float((df.loc[df["ID"] == name_sci])["RA"])
        dec = float((df.loc[df["ID"] == name_sci])["DEC"])
        
    elif MANUAL: # or just manually supply an RA, Dec
        # current values are for AT02019ntp
        ra_gal, dec_gal = 12.55052905, -26.19831783
        ra, dec = 12.5502516722, -26.1980040852
 
    ### DETERMINE CROPPING PARAMETERS ####################################
    if CROP:    
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
            pot_sizes = [x_sci, abs(x_sci-sci.shape[1]), 
                         y_sci, abs(y_sci-sci.shape[0]), 
                         x_tmp, abs(x_tmp-tmp.shape[1]), 
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
                    # grab a square of the image in the top/bottom half or mid
                    # based on where source is found 
                    # get the half but don't write it 
                    half_hdu = amakihi.crop_half(source_file, ra, dec, 
                                                 write=False)
                    # get the RA, DEC of its center 
                    w = wcs.WCS(half_hdu.header)
                    x = half_hdu.data.shape[1]/2.0
                    y = half_hdu.data.shape[0]/2.0
                    crop_center = w.all_pix2world(x, y, 1)
                
                    ra_crop = float(crop_center[0])
                    dec_crop = float(crop_center[1])
                    size = float(max(half_hdu.data.shape)) # size
                
            else: # size is sufficiently large 
                ra_crop = ra
                dec_crop = dec
        
        elif OCTANT_CROP:
            # grab a square of the image in octant where source is found
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
    
        ### CROPPING ######################################################
        print("*******************************************************")
        print("Cropping images...")
        print("*******************************************************")
        print("RA_CROP = %.5f"%ra_crop)
        print("DEC_CROP = %.5f"%dec_crop)
        print("SIZE = "+str(size)+" pix\n")
        
        # set output names
        source_crop = CROPDIR+"/"+re.sub(".*/", "",source_file)
        source_crop = source_crop.replace(".fits", "_crop.fits")
        template_crop = CROPDIR+"/"+re.sub(".*/", "",template_file)
        template_crop = template_crop.replace(".fits", "_crop.fits")
        
        amakihi.crop_WCS(source_file, ra_crop, dec_crop, size, 
                         output=source_crop)
        if TMP_CFHT: # both science and template are from CFHT
            amakihi.crop_WCS(template_file, ra_crop, dec_crop, size,
                             output=template_crop)
        else: # template from PS1/DECaLS, which has pixels twice as large
            amakihi.crop_WCS(template_file, ra_crop, dec_crop, size/2.0,
                             output=template_crop)    
            
        source_file = source_crop
        template_file = template_crop
   
    ### IMAGE ALIGNMENT ##################################################
    # align with astroalign and then use image_registration for alignment 
    # based on morphology
    # if not possible, use image_registration only 
    
    alignfiles = []
    if not(OVERWRITE):
        filename = (re.sub(".*/", "",source_file)).replace(".fits", "")
        alignfiles = glob.glob(ALIGNDIR+"/"+filename+"*.fits")
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
        print("*******************************************************")
        print("Image alignment...")
        print("*******************************************************")
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
                                  psf_sigma=PSFSIGMA,
                                  thresh_sigma=SEGMSIGMA,
                                  nsources=NSOURCES,
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
            # alignment with image_registration
            ret = amakihi.image_align_morph(source_align, template_file,
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

        # if astroalign succeeds but no double alignment
        else:
            source_file = source_align

        print("WRITING TO:\n"+source_file+"\n"+mask_file+"\n")
        
    ### BACKGROUND SUBTRACTION ##########################################    
    bkgsubfiles_source = []
    bkgsubfiles_temp = []
    if not(OVERWRITE):
        filename_source = (re.sub(".*/", "",source_file)).replace(".fits", "")
        bkgsubfiles_source = glob.glob(BKGSUBDIR+"/"+filename_source+"*.fits")
        filename_temp = (re.sub(".*/", "",template_file)).replace(".fits", "")
        bkgsubfiles_temp = glob.glob(BKGSUBDIR+"/"+filename_temp+"*.fits")
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
        print("*******************************************************")
        print("Background subtraction...")
        print("*******************************************************")
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

    old_mask_file = mask_file
    
    satmasks = []
    if not(OVERWRITE):
        filename = (re.sub(".*/", "",mask_file)).replace(".fits", "")
        satmasks = glob.glob(SATMASKDIR+"/"+filename+"*.fits")
        if len(satmasks) > 0:
            print("A saturation mask for the image was already produced. "+
                  "Skipping to next step in image differencing.")
            mask_file = satmasks[0]
            satmask_plot = (re.sub(".*/", "",mask_file)).replace(".fits", "")
            satmask_plot = SATMASKPLOTDIR+"/"+satmask_plot+".png"
            print("\nSATURATION MASK:\n"+re.sub(".*/", "",mask_file)+"\n")

    if OVERWRITE or (len(satmasks) == 0):
        print("*******************************************************")
        print("Building a mask of saturated stars...")
        print("*******************************************************")
        # set output names
        mask_file = (re.sub(".*/", "",mask_file)).replace(".fits", 
                    "_satmask.fits")
        mask_file = SATMASKDIR+"/"+mask_file
        satmask_plot = (re.sub(".*/", "",source_file)).replace(".fits", 
                        "_satmask.png")
        satmask_plot = SATMASKPLOTDIR+"/"+satmask_plot
        
        # build saturation mask
        amakihi.saturation_mask(source_file, mask_file=old_mask_file, 
                                dilation_its=7, 
                                ra_safe=ra, dec_safe=dec, rad_safe=10.0,
                                output=mask_file,
                                plot=True, plotname=satmask_plot)       
        satmask_plot = source_file.replace(".fits","_satmask.png")
    
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
                       scale="asinh", cmap="viridis", target=[ra,dec])
    print("WRITING TO:\n"+implot+"\n")
 
    
    ### IMAGE DIFFERENCING WITH HOTPANTS #################################    
    # for later
    ret = None
    subfiles = []

    if not(OVERWRITEHOTPANTS):
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        subfiles = glob.glob(SUBDIR+"/"+filename+"*.fits") # update
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
        print("*******************************************************")
        print("Performing image subtraction with hotpants...")
        print("*******************************************************")
        if TMP_CFHT: # both science and template are from CFHT
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   param_estimate=False,
                                   iu=50000, il=-200.0, tu=50000, tl=-200.0,
                                   ssig=5.0,
                                   ng="3 6 0.95 4 1.9 2 3.8", 
                                   bgo=0, ko=1, v=0, target=[ra,dec], 
                                   rkernel=2.5*4.42,
                                   output=subfile,
                                   mask_write=True,
                                   maskout=submask,
                                   plot=True,
                                   plotname=subplot)
        else: # need to update 
            ret = amakihi.hotpants(source_file, template_file, mask_file, 
                                   param_estimate=False,
                                   iu=50000, il=-200.0, 
                                   lsigma=10, hsigma=10,
                                   ng="3 6 2.5 4 5.0 2 10.0", 
                                   bgo=0, ko=1, v=0, target_small=[ra,dec],
                                   rkernel=2.5*5.0,
                                   convi=True, 
                                   norm="i", 
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
                                           thresh_sigma=TRANSIENTSIGMA,
                                           pixelmin=PIXELMIN,
                                           dipole_width=DIPOLEWIDTH,
                                           etamax=ETAMAX,
                                           area_max=AREAMAX,
                                           nsource_max=NSOURCE_MAX,
                                           #toi=[ra,dec], toi_sep_min=0.0,
                                           #toi_sep_max=SEPLIM,
                                           output=tabfile,                                            
                                           plots=PLOTCANDS,
                                           plot_rejections=True,
                                           og_scale="asinh",
                                           crosshair_sub="#fe01b1",
                                           plotdir=CANDIDATEPLOTDIR)
                
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

        # move plots produced for bugtesting
        run("mv "+ALIGNDIR+"/*elongs_areas.png "+ELONGAREASCATTERS, shell=True)
        run("mv "+ALIGNDIR+"/*elongs.png "+ELONGDIR, shell=True)
        run("mv "+ALIGNDIR+"/*areas.png "+AREADIR, shell=True)
        run("mv "+SUBDIR+"/*rejections.png "+REJECTIONDIR, shell=True)
        
    else: # if hotpants did not work
        print("\nCopying the science file to the directory holding files "+
              "which caused some error for hotpants.\n")
        run("cp "+sci_files[i]+" "+HOTPANTSFAILDIR, shell=True)
        run("rm "+subfile, shell=True)
    
    end = timer()
    print("\nTIME = %.4f s"%(end-start))


    
    
    
