#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Tue Aug 13 10:31:33 2019
.. @author: Nicholas Vieira
.. @query_2MASS.py
"""

import requests
import re

def geturl_2MASS(ra, dec, size=150, bands="A"):
    """Get the URL(s) for some reference image(s) to download from the 2-Micron
    All Sky Survey (2MASS). 
    
    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : float, optional
        Size of the cutout image in pixels (1 pix == 4.0" in 2MASS; default
        150)
    bands : str, optional
        Photometric band(s) of choice (default "A" --> all; options are "A",
        "J", "H", "K" - see notes for details)
    
    Returns
    -------
    list
        List of URL(s) to download the relevant fits files
        
    Notes
    -----
    Options for bands are "A" -->  all, **OR** "J" for only J-band, **OR** "H" 
    for only H-band, **OR** "K" for only K-band. Combinations such as e.g. "JH" 
    are not allowed.
    
    See: https://irsa.ipac.caltech.edu/applications/2MASS/IM/docs/siahelp.html
    """
    
    # define region of interest (ROI), bands, and which survey to use 
    pos = f"{ra},{dec}"
    size = size*4.0/3600.0 # size must be passed in degrees
    survey = "asky" # all sky survey
    
    # url for querying 
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?"
    url = f"{url}POS={pos}&SIZE={size}&band={bands}&ds={survey}"
    url = f"{url}&INTERSECT=CENTER" # returned files must overlap center of ROI 
    
    # get content of webpage at this url and extract all links to FITS files
    r = requests.get(url) 
    f_messy = re.findall("(?P<url>https?://[^\s]+)*.fits", r.content.decode())
    f_urls = []
    for fil in f_messy:
        if len(fil) != 0: # files_messy is sometimes pop. with empty strings 
            f_urls.append(f"{fil}.fits") # needs to be appended to all files
            
    return f_urls
    
        


