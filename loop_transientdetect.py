#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:48:24 2019
@author: Nicholas Vieira
@loop_transientdetect.py
"""

import amakihi 
import psfphotom
import apphotom
from subprocess import run
import glob 
import re
import pandas as pd 
from astropy.table import Table

# enable/disable plots popping up
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg') # plots pop up
plt.switch_backend('agg') # plots don't pop up

# disable annoying warnings 
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

### ARGUMENTS
# note: csv must be a text csv
CSV = "GW190814_50_GLADE.csv" # csv of RAs, DECs, Ranks 
BASEDIR = "/media/nvieira/OS/Users/nviei/Documents/Storage/"
SUBDIR = BASEDIR+"subdir"
OGDIR = BASEDIR+"ogdir"
OGCROPDIR = BASEDIR+"ogcropdir"
# args for transient_detect
SIGMA = 5.0 # sigma for image segmentation
PIXELMIN = 8 # pixel area minimum 
ELONGLIM = 2.0 # maximum elongation
PLOT=True # whether to plot
SUBSCALE="asinh" # scale to apply to subtraction images
OGSCALE="asinh" # scale to apply to science images
STAMPSIZE = 200.0 # transient stamp size
CROSSHAIR = "#0485d1" # crosshair colour
SEPMIN = 0.5 # minimum separation from TOI in arcsec
SEPMAX = 30.0  # maximum separation from TOI in arcsec

# load in data 
sub_files = sorted(glob.glob(SUBDIR+"/*.fits"))
og_files = sorted(glob.glob(OGDIR+"/*.fits"))
ogcrop_files = sorted(glob.glob(OGCROPDIR+"/*.fits"))
df = pd.read_csv(CSV)

# clean up directories
run("rm -f "+SUBDIR+"/*.png ", shell=True)
run("rm -f "+SUBDIR+"/*error* ", shell=True)
run("rm -f "+SUBDIR+"/*sourcemask* ", shell=True)
run("rm -f "+SUBDIR+"/*sourcelist* ", shell=True)
run("rm -f "+OGDIR+"/*.png ", shell=True)
run("rm -f "+OGDIR+"/*error* ", shell=True)
run("rm -f "+OGDIR+"/*sourcemask* ", shell=True)
run("rm -f "+OGDIR+"/*sourcelist* ", shell=True)
run("rm -f "+OGCROPDIR+"/*.png ", shell=True)
run("rm -f "+OGCROPDIR+"/*error* ", shell=True)
run("rm -f "+OGCROPDIR+"/*sourcemask* ", shell=True)
run("rm -f "+OGCROPDIR+"/*sourcelist* ", shell=True)

# for storing results
trans_ras = []
trans_decs = []
trans_seps = []

nfiles = len(sub_files)
for i in range(nfiles):
    og = og_files[i]
    og_crop = ogcrop_files[i]
    sub = sub_files[i]
    
    # determine target of interest (potential host galaxy)
    rank_sub = re.sub(".*/", "",sub_files[i])[4:6] # determine rank 
    ra = float((df.loc[df["Rank"] == int(rank_sub)])["RAJ2000"])
    dec = float((df.loc[df["Rank"] == int(rank_sub)])["DEJ2000"])
    
    print("\nLooking for transients using the following files:")
    print("cropped, background-subtracted file: "+
          re.sub(".*/", "",og))
    print("image difference file: "+re.sub(".*/", "",sub)+"\n")
    
    # run it 
    t = amakihi.transient_detect_hotpants(sub, og, 
                                          sigma=SIGMA,
                                          pixelmin=PIXELMIN, 
                                          elongation_lim=ELONGLIM, 
                                          plots=PLOT,
                                          sub_scale=SUBSCALE,
                                          og_scale=OGSCALE,
                                          stampsize=STAMPSIZE,
                                          crosshair=CROSSHAIR,
                                          toi=[ra,dec],
                                          toi_sep_min=SEPMIN,
                                          toi_sep_max=SEPMAX)
    if not(t == None): # if at least 1 candidate is found 
        
        ### this section is buggy ####################################

        ras = t["ra"].data # RAs of all candidate transients
        decs = t["dec"].data # DECs of all candidate transients
        
        # PSF photometry
        print("\nPerforming PSF photometry on the cropped, background-"+
              "subtracted file to obtain the photometric zero point...")
        psfphotom.PSF_photometry(og)
        
        # aperture photometry
        print("\nPerforming aperture photometry on all of the candidates...")
        apphotom.make_source_mask(og_crop) # make source mask
        mask_file = (og_crop).replace(".fits", "_sourcemask.fits")
        og_crop = ogcrop_files[i]
        apphotom.error_array(og_crop, mask_file) # make error array
        error_file = (og_crop).replace(".fits", "_error.fits")
        og_crop = ogcrop_files[i]
        # actual aperture photometry
        ap = apphotom.aperture_photom(og, ras, decs, mask_file, error_file)
        # write the resulting file 
        ap.write(og_crop.replace(".fits","_apphotom.csv"), overwrite=True)
        og_crop = ogcrop_files[i]
        
        ##############################################################
        
        for row in t: # collect important results
            trans_ras.append(row["ra"])
            trans_decs.append(row["dec"])
            trans_seps.append(row["sep"])
        del t # get rid of t 
    
    # move resultant plots and fits files 
    run("mv "+SUBDIR+"/*.png "+SUBDIR+"/../RESULTS/candidates_30as",shell=True)
    run("mv "+OGDIR+"/*.png "+OGDIR+"/../RESULTS/candidates_30as", shell=True)
    run("mv "+OGDIR+"/*.csv "+OGDIR+"/../RESULTS/candidates_30as", shell=True)
    run("mv "+OGDIR+"/*source* "+OGDIR+"/../RESULTS/candidates_30as",
        shell=True)
    run("mv "+OGDIR+"/*error* "+OGDIR+"/../RESULTS/candidates_30as", 
        shell=True)
   

    

# write the RA, DEC and separation from the target of interest of each source 
tbl = Table(data=[trans_ras, trans_decs, trans_seps], names=["ra","dec","sep"])
tbl.write("candidates30as.csv", overwrite=True)

