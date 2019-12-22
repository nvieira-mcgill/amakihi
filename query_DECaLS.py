#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:41:59 2019
@author: Nicholas Vieira
@query_DECaLS.py
"""

def geturl(ra, dec, size=400, pixscale=0.262, filters="grz"):
    """
    Input: a RA, Dec of interest, a size for the cutout image in pixels, the 
    scale of the image in arcseconds per pixel, and the filters of interest 
    (g, r, z)
    Output: a list of urls for the relevant fits files
    """    
    # list of urls
    urls = []
    # build the url
    url = "http://legacysurvey.org/viewer/fits-cutout?"
    url = f"{url}ra={ra}&dec={dec}&layer=dr8-south"
    url = f"{url}&size={size}&pixscale={pixscale}"
    for fil in filters:
        urls.append(f"{url}&bands={fil}")
    
    if len(urls) == 1:
        urls = urls[0]
    return urls
    
        


