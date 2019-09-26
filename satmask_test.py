#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:41:39 2019
@author: Nicholas Vieira
@satmask_test.py
"""

import glob
import re
import amakihi

# enable/disable plots popping up
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg') # plots pop up
#plt.switch_backend('agg') # plots don't pop up

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

BASEDIR = "/media/nvieira/OS/Users/nviei/Documents/Storage/"
OGDIR = BASEDIR+"ogdir"
TMP_DIR = BASEDIR+"itmp_50region"

og_files = sorted(glob.glob(OGDIR+"/*.fits"))[110:]

for i in range(len(og_files)):
    og_file = og_files[i]
    print("file: "+re.sub(".*/", "",og_file))
    amakihi.saturation_mask(og_file, plot=True)
    #sat_file = og_file.replace(".fits", "_satmask.fits")
    #amakihi.make_image(og_file, mask_file=sat_file)

