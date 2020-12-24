#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:33:50 2019
@author: Nicholas Vieira
Taken from https://ps1images.stsci.edu/ps1image.html
@query_PS1.py
"""

import numpy as np
from astropy.table import Table

###############################################################################
#### HELPER FUNCTIONS ####
#### Taken from https://ps1images.stsci.edu/ps1image.html ####

def getimages(ra, dec, size=2400, filters="grizy"):
    """Query the online ps1filenames.py service, which can provide a list of 
    URL(s) for some reference image(s) to download from the Panoramic Survey 
    Telescope and Rapid Response System (Pan-STARRS) archive (specifically, 
    the 3PI PS1 archive).

    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : int, optional
        Size of the cutout image in pixels (1 pix == 0.25" in PS1; default 
        2400)
    filters : str, optional
        Photometric filter of choice (default 'grizy' --> g-, r-, i-, z-, and 
        y-band; options are 'g', 'r', 'i', 'z', 'y' or any combination of them)
    
    Returns
    -------
    list or str
        List of URL(s) to download the relevant fits files **OR** a single str
        if only one URL
        
    Notes
    -----    
    See: http://ps1images.stsci.edu/cgi-bin/ps1cutouts

    """
    # check that the size is an int
    if not(type(size) == int):
        raise TypeError("size must be an integer, argument supplied was "+
                        f"{size}")
        
    # build the URLs
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&size={size}&format=fits"
    url = f"{url}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table 


def geturl(ra, dec, size=2400, output_size=None, filters="grizy", fmt="jpg", 
           color=False):  
    """Get URL for images in the table

    Arguments
    ---------
    ra, dec : float
        RA and Dec of interest
    size : int, optional
        Size of the cutout image in pixels (1 pix == 0.25" in PS1; default 
        2400)
    output_size : int, optional
        Size of the output image, **only relevant if requesting a .jpg or 
        .png** (default is `output_size = size`)
    filters : str, optional
        Photometric filter of choice (default 'grizy' --> g-, r-, i-, z-, and 
        y-band; options are 'g', 'r', 'i', 'z', 'y' or any combination of them)
    fmt : {'jpg', 'png', 'fits'}, optional
        Data format (default 'jpg')
    color : bool, optional
        Whether to create a colour image (default False; only relevant if 
        downloading a jpg or png)

    Returns
    -------
    list or str
        List of URL(s) to download the relevant fits files **OR** a single str
        if only one URL
        
    Notes
    -----    
    See: http://ps1images.stsci.edu/cgi-bin/ps1cutouts       
    
    """
    # check that the size is an int
    if not(type(size) == int):
        raise TypeError("size must be an integer, argument supplied was "+
                        f"{size}")
        
    # check that the otuput size is an int
    if not(type(output_size) == int):
        raise TypeError("output_size must be an integer, argument supplied "+
                        f"was {output_size}")

    # check that desired filetype is available
    if color and fmt == "fits":
        raise ValueError("color images are available only for jpg/png formats")
    if not (fmt in ("jpg", "png", "fits")):
        raise ValueError("format must be one of (jpg, png, fits)")
    
    # build the URLs
    table = getimages(ra,dec,size=size,filters=filters)
    url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
    url = f"{url}ra={ra}&dec={dec}&size={size}"
    url = f"{url}&format={format}"
    if output_size: # only relevant for jpgs, pngs
        url = url + "&output_size={}".format(output_size)
        
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    
    if color: # if plotting a jpg, png
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        urls = []
        for filename in table['filename']:
            urls.append(urlbase+filename)
        
    if len(urls) == 1:
        urls = urls[0]
    return urls
    

        
        
