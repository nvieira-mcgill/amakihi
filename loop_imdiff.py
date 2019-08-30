#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:06:15 2019
@author: Nicholas Vieira
@loop_imdiff.py
"""

import amakihi 
import os
from subprocess import run
import glob 
import re
import pandas as pd 
from astropy.io import fits
from astropy import wcs


# ARGUMENTS
# note: csv must be a text csv
COORDS = "GW190814_50_GLADE.csv" # a csv of RAs, DECs, Ranks 
SCI_DIR = "sci_files" # a directory for science images
TMP_DIR = "tmp_files" # a directory for templates
WORKDIR = "workdir" # directory for cropped, aligned, and differenced files 
SUBDIR = "subdir" # directory for differences only 
 
# load in data 
df = pd.read_csv(COORDS) # get coordinates 
ranks = df["Rank"] # get ranks for each object 
cwd = os.getcwd() # get science/template filenames 
sci_files = sorted(glob.glob(cwd+"/"+SCI_DIR+"/*.fits"))
tmp_files = sorted(glob.glob(cwd+"/"+TMP_DIR+"/*.fits"))

# check if there is a corresponding template for every science image 
if len(sci_files) != len(tmp_files):
    print("The number of science files and template files do not match. "+
          "Exiting.")
    exit
    
nfiles = len(sci_files)

# clean up directories
run("rm -rf "+SCI_DIR+"/*crop* ", shell=True)
run("rm -rf "+TMP_DIR+"/*crop* ", shell=True) 
run("rm -rf "+WORKDIR+"/*.fits ", shell=True)

for i in range(nfiles):
    
    # load in data 
    sci = fits.getdata(sci_files[i])
    sci_hdr = fits.getheader(sci_files[i])
    tmp = fits.getdata(tmp_files[i])
    tmp_hdr = fits.getheader(tmp_files[i])

    ### DETERMINE RA, DEC OF SOURCE OF INTEREST 
    rank_sci = re.sub(".*/", "",sci_files[i])[4:6] # determine rank of target
    rank_tmp = re.sub(".*/", "",tmp_files[i])[4:6]
    if int(rank_sci) != int(rank_tmp): # why is this not triggered?
        print("The images being compared are not of the same field. Exiting.")
        exit
    
    ra = (df.loc[df["Rank"] == int(rank_sci)])["RAJ2000"]
    dec = (df.loc[df["Rank"] == int(rank_sci)])["DEJ2000"]
    
    ### DETERMINE CROPPING SIZE 
    # construct the largest cutout possible which is still a square 
    w = wcs.WCS(sci_hdr)
    x_sci, y_sci = w.all_world2pix(ra, dec, 1)
    x_sci = float(x_sci) # x, y coords of RA, DEC in science image
    y_sci = float(y_sci)
    w = wcs.WCS(tmp_hdr)
    x_tmp, y_tmp = w.all_world2pix(ra, dec, 1)
    x_tmp = float(x_tmp) # x, y coords of RA, DEC in template image 
    y_tmp = float(y_tmp)
    
    pot_sizes = [x_sci, abs(x_sci-sci.shape[1]), y_sci, 
                 abs(y_sci-sci.shape[0]), x_tmp, abs(x_tmp-tmp.shape[1]),
                 y_tmp, abs(y_tmp-tmp.shape[0])]
    size = min(pot_sizes) # size of largest possible square centered on coords
    
    # source/template/coordinate info 
    source_file, template_file = sci_files[i], tmp_files[i]
    
    print("\nOriginal images:")
    print("science file: "+re.sub(".*/", "",source_file))
    print("template file: "+re.sub(".*/", "",template_file)+"\n")
    
    ### CROPPING 
    print("Cropping images...\n")
    amakihi.crop_WCS(source_file, ra, dec, size)
    amakihi.crop_WCS(template_file, ra, dec, size)
    source_file = source_file.replace(".fits", "_crop.fits")
    template_file = template_file.replace(".fits", "_crop.fits")
    
    ### BACKGROUND SUBTRACTION
    print("Background subtraction...\n")
    amakihi.bkgsub(source_file, crreject=True)
    amakihi.bkgsub(template_file, crreject=True)
    source_file = source_file.replace(".fits", "_bkgsub.fits")
    template_file = template_file.replace(".fits", "_bkgsub.fits")
    
    ### IMAGE ALIGNMENT 
    print("Image alignment...\n")
    # align the images with astroalign
    ret = amakihi.image_align(source_file, template_file, thresh_sigma=3)
    
    if ret == None: # if astroalign fails
        # align them with image_registration
        ret = amakihi.image_align_fine(source_file, template_file)
        
    source_file = source_file.replace(".fits", "_align.fits")
    mask_file = source_file.replace(".fits", "_mask.fits")
    
    ### IMAGE DIFFERENCING WITH HOTPANTS 
    amakihi.hotpants(source_file, template_file, mask_file, target=[ra,dec],
                     v=0, bgo=0, convt=True)
    
    # move and copy files files
    run("mv "+SCI_DIR+"/*crop* "+WORKDIR, shell=True)
    run("mv "+TMP_DIR+"/*crop* "+WORKDIR, shell=True)
    run("cp "+WORKDIR+"/*subtracted* "+SUBDIR, shell=True)
    
    
    
