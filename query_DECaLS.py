#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Mon Aug 26 19:41:59 2019
.. @author: Nicholas Vieira
.. @query_DECaLS.py
"""

def geturl_DECaLS(ra, dec, size=512, pixscale=0.262, bands="grz"):
    """Get the URL(s) for some reference image(s) to download from the Dark 
    Energy Camera Legacy Survey (DECaLS). 

    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : int, optional
        Size of the cutout image in pixels (1 pix == 0.262" in native DECam
        resolution, but pixel scale can be changed; default 512)
    pixscale : float, optional
        Pixel scale of the images in arcsec per pixel, (default 0.262"/pix, 
        which is the native DECam resolution)
    bands : str, optional
        Photometric band(s) of choice (default "grz" --> g-, r-, and z-band; 
        options are "g", "r", "z", or any combination of them)
    
    Returns
    -------
    list or str
        List of URL(s) to download the relevant fits files **OR** a single str
        if only one URL
        
    Notes
    -----    
    See: https://www.legacysurvey.org/dr8/files/#image-stacks-region-coadd

    """    
    # check that the size is an int
    if not(type(size) == int):
        raise TypeError("size must be an integer, argument supplied was "+
                        f"{size}")
    
    # list of urls
    urls = []
    # build the url
    url = "http://legacysurvey.org/viewer/fits-cutout?"
    url = f"{url}ra={ra}&dec={dec}&layer=dr8-south"
    url = f"{url}&size={size}&pixscale={pixscale}"
    for bs in bands:
        urls.append(f"{url}&bands={bs}")
    
    if len(urls) == 1:
        urls = urls[0]
    return urls
    
        


