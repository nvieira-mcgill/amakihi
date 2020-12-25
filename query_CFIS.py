#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:56:17 2019
@author: Nicholas Vieira
@query_CFIS.py
"""

import requests
from lxml import html

def geturl_CFIS(ra, dec, size=1600, bands="ur"):
    """Get the URL(s) for some reference image(s) to download from the Canada-
    France Imaging Survey (CFIS). 

    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : float, optional
        Size of the cutout image in pixels (1 pix == 0.185" in CFIS; default
        1600)
    bands : str, optional
        Photometric band(s) of choice (default "ur" --> u- and r-band; 
        options are "u", "r", "ur")
    
    Returns
    -------
    list
        List of URL(s) to download the relevant fits files
        
    Notes
    -----    
    See: https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/cfis/csky.html
    """
    
    bands_upd = ""
    for bs in bands:
        bands_upd = f"{bands_upd}&fils={bs}"
    url = "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/community/"
    url = f"{url}cfis/mcut.pl?&ra={ra}&dec={dec}&tiles=true"
    url = f"{url}{bands_upd}&cutout={size}"
    
    r = requests.get(url)
    webpage = html.fromstring(r.content)
    hrefs = webpage.xpath('//a/@href') # extract hyperlinks
    
    fits_url = []
    for h in hrefs:
        if ".fits[" in h: # ignore .cat and weights.fits.fz files 
            fits_url.append(h)
    
    return fits_url

