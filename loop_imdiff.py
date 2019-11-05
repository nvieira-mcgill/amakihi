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
#CSV = "GW190814_50_GLADE.csv" 
CSV = "AT2019candidatesCFHT.csv" 

## directories
# base data directories:
BASEDIR = "/media/nvieira/OS/Users/nviei/Documents/Storage/" # base data dir
WORKDIR = BASEDIR+"workdir/" # top working directory 
SCI_DIR = BASEDIR+"isci_transient" # science
TMP_DIR = BASEDIR+"itmp_transient" # template
SCI_NOTMP_DIR = BASEDIR+"isci_transient_notmp" # sci files w/o CFHT template

# general directories:
PLOTDIR = WORKDIR+"plotdir" # plotting directory 
CANDIDATEDIR = WORKDIR+"candidatedir" # tables of potential transients
CANDIDATEPLOTDIR = CANDIDATEDIR+"/plots" 
CROPDIR = WORKDIR+"cropdir"
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
OVERWRITE = False # alignments, background subtractions, masks
OVERWRITEHOTPANTS = False # hotpants difference images

## RA, DEC determination
RANKS = False # use the .csv to obtain RANKS
TRANSIENTS = True # use the .csv to obtain TRANSIENT NAMES
MANUAL = False # manually supply a RA, DEC below 

## cropping parameters
CROP = True # do any sort of cropping?
CROPMIN = 1000.0 # minimum crop size (pix)
PRECROP = False # are files pre-cropped?
TARGET_CROP = True # use the rank/transient .csv to set crop
OCTANT_CROP = False # crop according to which octant contains some target

## which subtraction to use 
PROPERSUB = False # use proper image subtraction (if False, use hotpants)

## general subtraction arguments
## templates
TMP_CFHT = False # are templates also from CFHT? 
SURVEY = "DECaLS" # if not from CFHT, which survey (PS1 or DECaLS)?

## alignment
DOUBLEALIGN = True # do double alignment (if possible)?
ASTROMSIGMA = [5, 8] # sigma(s) for source detection with astrometry.net
SEGMSIGMA = 3.0 # sigma for source detect with img segmentation in image_align
MAXOFFSET = 100.0 # maximum allowed pixel offset for image_align_fine

## transient detection
ELONGLIM = 1.5 # maximum allowed source elongation
TRANSIENTSIGMA = 5.0 # subtraction image transient detection sigma
SEPLIM = 5.0 # max allowed sep between transient and target of interest

## what to do with found candidates
# ["full", "zoom og", "zoom diff"] --> plot all
# None --> plot none 
PLOTCANDS = ["zoom og", "zoom diff"] # plot candidate transients?

## proper image subtraction arguments 
#SIGMA = 8.0 # sigma for source detection when building ePSF, if applicable 
#PSFSIGMA = 5.0 # estimated PSF width for astrometry.net 
#ALIM = 1500 # maximum allowed area in pix**2 for astrometry.net
#NTHREADS = 8 # number of threads to use in FFTs

## path to hotpants (needed when running on my machine)
amakihi.hotpants_path("~/hotpants/hotpants")
 
## clean up directories
run("rm -rf "+SCI_DIR+"/*xy.fits", shell=True)
run("rm -rf "+TMP_DIR+"/*xy.fits", shell=True) 

### LOAD IN DATA #########################################################
df = pd.read_csv(CSV) # get coordinates 
cwd = os.getcwd() # get science/template filenames 
sci_files = sorted(glob.glob(SCI_DIR+"/*.fits"))

if not(TMP_CFHT) and SURVEY == "DECaLS":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] in ["g","r","z"])]
elif not(TMP_CFHT) and SURVEY == "PS1":
    sci_files = [s for s in sci_files.copy() if (
            fits.getheader(s)["FILTER"][0] == "i")] 

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
        if PRECROP:
            s = s.replace(".fits", "")[:-13] # "YYYYMMDD_crop"
        else:
            s = s.replace(".fits", "")[:-8] # "YYYYMMDD"
        topfile = re.sub(".*/", "", s) # from /a/b/c, extract c
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
    
    if not(OVERWRITEHOTPANTS):
        filename = re.sub(".*/", "",sci_files[i])
        filename = filename.replace(".fits", "")
        subfiles = glob.glob(SUBDIR+"/"+filename+"*") # update
        if len(subfiles) > 0:
            print("\nA difference image already exists for this science file."+
                  " Skipping to next science file.\n")
            continue
        
    print("\n*******************************************************")
    print("                     ORIGINAL IMAGES:")
    print("*******************************************************")
    print("science file: "+re.sub(".*/", "",source_file))
    print("template file: "+re.sub(".*/", "",template_file)+"\n")

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
                                ra_safe=ra, dec_safe=dec, rad_safe=10.0,
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
                       scale="asinh", cmap="viridis", target=[ra,dec])
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
                                          
                                          plot="S",
                                          target=[ra,dec])
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
                                                  toi=[ra,dec],
                                                  toi_sep_min=0.0,
                                                  toi_sep_max=SEPLIM,
                                                  sigma=TRANSIENTSIGMA)
            run("cp "+source_align+" "+CROPALDIR, shell=True) # cropped, align
            run("cp "+template_align+" "+CROPALDIR, shell=True) 
            run("cp "+source_bkgsub+" "+BKGSUBDIR, shell=True) # +bkgsub
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

    if TMP_CFHT: # both science and template are from CFHT
        ret = amakihi.hotpants(source_file, template_file, mask_file, 
                               iu=50000, il=-100.0, tu=50000, tl=-100.0,
                               ng="3 6 2.5 4 5.0 2 10.0", 
                               bgo=0, ko=0, v=0, target=[ra,dec], 
                               rkernel=2.5*5.0,
                               output=subfile,
                               mask_write=False,
                               maskout=submask)
    else:
        ret = amakihi.hotpants(source_file, template_file, mask_file, 
                               iu=50000, il=-500.0, bgo=0, ko=0,
                               ng="3 6 2.5 4 5.0 2 10.0", 
                               rkernel=2.5*5.0,
                               convi=True, v=0, #target=[ra_gal,dec_gal],
                               target_small=[ra,dec],
                               output=subfile,
                               mask_write=False,
                               maskout=submask)
    
    # move and copy files
    if (type(ret) == np.ndarray): # if successful subtraction 
        run("mv "+os.getcwd()+"/*.png "+SUBPLOTDIR, shell=True) # diffim plot
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
                                                toi=[ra,dec],
                                                toi_sep_min=0.0,
                                                toi_sep_max=SEPLIM,
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


    
    
    
