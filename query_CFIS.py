#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:56:17 2019
@author: Nicholas Vieira
@query_CFIS.py
"""

import requests
from lxml import html

def geturl(ra,dec,size=1600,filters="ur"):
    """
    Input: a RA, Dec of interest, a size for the cutout image in pixels (1 
    pixel == 0.185" in CFIS), and filter(s) (u, r) for the image 
    Output: a list of the urls for the relevant fits files 
    """
    
    filts = ""
    for fil in filters:
        filts = f"{filts}&fils={fil}"
    url = "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/community/"
    url = f"{url}cfis/mcut.pl?&ra={ra}&dec={dec}&tiles=true"
    url = f"{url}{filts}&cutout={size}"
    
    r = requests.get(url)
    webpage = html.fromstring(r.content)
    hrefs = webpage.xpath('//a/@href') # extract hyperlinks
    
    fits_url = []
    for h in hrefs:
        if ".fits[" in h: # ignore .cat and weights.fits.fz files 
            fits_url.append(h)
    
    return fits_url

