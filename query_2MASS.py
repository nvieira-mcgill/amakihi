#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:31:33 2019
@author: Nicholas Vieira
@query_2MASS.py
"""

import requests
import re

def geturl(ra, dec, size=150, filters="A"):
    """
    Input: a RA, Dec of interest, a size for the cutout image in pixels (1 
    pixel == 4.0" in 2MASS), and filter(s) (A for all or J, H, K) for the image
    Output: a list of urls for the relevant fits files
    """
    
    # define region of interest (ROI), bands, and which survey to use 
    pos = f"{ra},{dec}"
    size = size*4.0/3600.0 # size must be passed in degrees
    band = filters
    survey = "asky" # all sky survey
    
    # url for querying 
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?"
    url = f"{url}POS={pos}&SIZE={size}&band={band}&ds={survey}"
    url = f"{url}&INTERSECT=CENTER" # returned files must overlap center of ROI 
    
    # get content of webpage at this url and extract all links to FITS files
    r = requests.get(url) 
    f_messy = re.findall("(?P<url>https?://[^\s]+)*.fits", r.content.decode())
    f_urls = []
    for fil in f_messy:
        if len(fil) != 0: # files_messy is sometimes pop. with empty strings 
            f_urls.append(f"{fil}.fits") # .fits needs to be appended to all files
            
    return f_urls
    
        


