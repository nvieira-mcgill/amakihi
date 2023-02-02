#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Tue Jan 31 21:47:35 2023
.. @author: Nicholas Vieira
.. @crossmatch.py

Functions for cross-matching candidate transients yielded by `amakihi` with 
lists of known transients, transients from the Transient Name Server (TNS), 
catalogues of variable star, quasars, or active galactic nuclei, etc.

"""

import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.table import Table, setdiff, unique
from astropy.coordinates import SkyCoord, Angle
from astroquery.vizier import Vizier


###############################################################################
### CROSSMATCHING #############################################################
 
### GENERAL LISTS/TABLES ######################################################
       
def crossmatch_list(tabfile,  
                    crossmatch_ra, crossmatch_dec, 
                    tripfile=None,
                    sep_max=2.0, 
                    output_tab=None, 
                    output_tab_crossmatch=None, 
                    output_trips=None):
    """Given some candidate transients, search for matches within `sep_max` 
    arcseconds of some `(crossmatch_ra, crossmatch_dec)`
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    crossmatch_ra, crossmatch_dec : array_like
        Lists of RA and Dec to crossmatch to; must be in degrees
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_crossmatch : str, optional
        Name for output table with RA, Dec which were successfully 
        crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)

    Notes
    -----
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, looks for candidates which
    are within `sep_max` arcseconds of pairs (ra, dec) given by the input 
    lists `crossmatch_ra` and `crossmatch_dec`. If M>0 matches are found, 
    writes a table containing the M crossmatched transient candidates, another 
    tablecontaining just the RA, Dec of the M crossmatched sources, and a 
    (M,3,Y,X) .npy file containing the M relevant triplets.
    
    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")
    # check input RA, Dec
    if not(len(crossmatch_ra) == len(crossmatch_dec)):
        raise ValueError("The length of the input RAs and input Decs do "+
                         "not match")
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    cat_skycoords = SkyCoord(crossmatch_ra*u.deg, crossmatch_dec*u.deg, 
                             frame="icrs")
    trns_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                                   frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_trns, idx_cat, d2d, d3d = cat_skycoords.search_around_sky(
            trns_skycoords, sep_max*u.arcsec)

    print(f"{len(idx_trns)} match(es) found for {len(triplets)} input "+
          "transient candidate(s) cross-matched against a catalog with "+
          f"{len(crossmatch_dec)} source(s)")

    # tables/triplets with only those crossmatched sources    
    tbl_match = tbl[idx_trns]
    triplets_matched = triplets[idx_trns]
    cat_match = Table(data=[crossmatch_ra, crossmatch_dec], 
                      names=["ra","dec"])[idx_cat]

    # write files    
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_crossmatched_list{filext}")    
    if type(output_tab_crossmatch) == type(None):
        output_tab_crossmatch = tabfile.replace(filext, 
                                        f"_catalog_crossmatched_list{filext}")
    if filext == ".fits":
        tbl_match.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, format="ascii.csv", overwrite=True)
    
    # always write crossmatched transients to a .csv 
    cat_match.write(output_tab_crossmatch, format="ascii.csv", overwrite=True)
    
    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", "_crossmatched_list.npy")
        np.save(output_trips, triplets_matched)


def crossmatch_table(tabfile, transient_tabfile, 
                     tripfile=None, 
                     sep_max=2.0, 
                     match=True, 
                     output_tab=None, 
                     output_tab_transients=None, 
                     output_trips=None):
    """Given some tables of candidate transients and known transients, search 
    for matches within `sep_max` arcseconds of each other
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    transient_tabfile : str
        Table of known transients from whatever source (must have, at 
        minimum, 'RA'/'ra'/'RAJ2000' and 'DEC'/'dec'/'DEJ2000' columns; 
        must be .csv or .fits file)
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    match : bool, optional
        Look for matches? or look for non-matches (default True)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_transients : str, optional
        Name for output table with **known** transients which were 
        successfully crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)

    Notes
    -----
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, looks for candidates which
    are within `sep_max` arcsec of known transients provided via the .fits / 
    .csv table `transient_tabfile`. If M>0 matches are found, writes a table 
    containing the M crossmatched transient *candidates*, another table
    containing the M *known* transients, and a (M,3,Y,X) .npy file containing 
    the M relevant triplets.    

    Bug: files are overwritten if tabfile and known_transient_table have 
    different file extensions?
    
    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
        
    # load in known transients
    if ".fits" in transient_tabfile:
        trns = Table.read(transient_tabfile, format="ascii")
        filext_trns = ".fits"
    elif ".csv" in transient_tabfile:
        trns = Table.read(transient_tabfile, format="ascii.csv")
        filext_trns = ".csv"
    else:
        raise ValueError("transient_tabfile must be of filetype .csv or "+
                         f".fits, did not recognize {transient_tabfile}")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")

    # informative print 
    trns.pprint() 

    # find out name for ra, dec columns in the known transients table 
    # get RAs, DECs
    cols = trns.colnames
    ra_col = [c for c in cols if (("ra" in c) or ("RA" in c))]
    if "RAJ2000" in ra_col: ra_col = "RAJ2000"
    else: ra_col = ra_col[0]
        
    dec_col = [c for c in cols if (("dec" in c) or ("DEC" in c) or (
            "DEJ" in c) or ("de" in c))]
    if "DEJ2000" in dec_col: dec_col = "DEJ2000"
    else: dec_col = dec_col[0]
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    trns_skycoords = SkyCoord(trns[ra_col]*u.deg, trns[dec_col]*u.deg, 
                             frame="icrs")
    cand_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                                   frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_cand, idx_trns, d2d, d3d = trns_skycoords.search_around_sky(
            cand_skycoords, sep_max*u.arcsec)
    
    if len(idx_cand) == 0:
        print('\nNo matches with input sources found for maximum allowed '+
              f'separation of {sep_max}". Exiting.')
        return

    if match: # tables/triplets with only those crossmatched sources
        print("\nSearching for matching sources...")
        tbl_match = tbl[idx_cand]
        trns_match = trns[idx_trns]
        if not(type(tripfile) == type(None)): 
            triplets_match = triplets[idx_cand]
    else: # tables with only NON-matched sources 
        print("\nSearching for NON-matching sources...")
        tbl_match = setdiff(tbl, tbl[idx_cand])
        trns_match = setdiff(trns, trns[trns_match], keys=[ra_col,dec_col])
        idx_nomatch = [i for i in range(len(tbl)) if not(i in idx_cand)]
        if not(type(tripfile) == type(None)): 
            triplets_match = triplets[idx_nomatch]  


    print(f'\n{len(idx_cand)} match(es) found for {len(tbl)} input '+
          f'transient candidate(s) cross-matched against {len(trns)} known '+
          f'transient source(s) for maximum allowed separation of {sep_max}"')      

    # write files    
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, 
                                     f"_crossmatch_og{filext}")    
    if type(output_tab_transients) == type(None):
        output_tab_transients = tabfile.replace(filext_trns, 
                                f"_crossmatch_knowntrns{filext_trns}")
    if filext == ".fits":
        tbl_match.write(output_tab, 
                        format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, 
                        format="ascii.csv", overwrite=True)
        
    if filext_trns == ".fits":
        trns_match.write(output_tab_transients, 
                         format="ascii", overwrite=True)
    elif filext_trns == ".csv":
        trns_match.write(output_tab_transients, 
                         format="ascii.csv", overwrite=True)
    
    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", "_crossmatched_og.npy")
        np.save(output_trips, triplets_match)       



### TRANSIENT NAME SERVER (TNS) ###############################################

def crossmatch_TNS(tabfile, TNS_tabfile, 
                   tripfile=None, 
                   sep_max=2.0, 
                   match=True,
                   output_tab=None,
                   output_tab_TNS=None,
                   output_trips=None):
    """Given some tables of candidate transients and known treansients from 
    the Transient Name Server (TNS), search for matches within `sep_max` 
    arcseconds of each other

    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    TNS_tabfile : str
        Table of known transients from Transient Name Server (must be a .csv)
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    match : bool, optional
        Look for matches? or look for non-matches (default True)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_TNS : str, optional
        Name for output table with **TNS** transients which were successfully 
        crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)    

    Notes
    -----
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, looks for candidates which
    are within `sep_max` arcsec of known transients provided via the .csv 
    table `TNS_tabfile`, which must come the Transient Name Server's querying 
    page. If M>0 matches are found, writes a table containing the M 
    crossmatched transient *candidates*, another table containing the M 
    *known* TNS sources, and a `(M,3,Y,X)` .npy file containing the M relevant 
    triplets. 
    
    The .csv of TNS files must have been previously downloaded. In the future, 
    will write a function which queries the TNS in real-time.
    

    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
        
    # load in TNS transients (always a .csv)
    tns = Table.read(TNS_tabfile, format="ascii.csv")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")
    
    # convert RA, DEC to degrees 
    tns["RA"] = [float(Angle(str(t["RA"])+" hours").degree) for t in tns]
    tns["DEC"] = [float(Angle(str(t["DEC"])+" degrees").degree) for t in tns]

    # informative print 
    tns["ID","Name","RA","DEC","Obj. Type", "Disc. Internal Name"].pprint() 
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    tns_skycoords = SkyCoord(tns["RA"]*u.deg, tns["DEC"]*u.deg, 
                             frame="icrs")
    cand_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                                   frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_cand, idx_tns, d2d, d3d = tns_skycoords.search_around_sky(
            cand_skycoords, sep_max*u.arcsec)
    
    if len(idx_cand) == 0:
        print('\nNo matches with input sources found for maximum allowed '+
              f'separation of {sep_max}". Exiting.')
        return

    if match: # tables/triplets with only those crossmatched sources
        print("\nSearching for matching sources...")
        tbl_match = tbl[idx_cand]
        tns_match = tns[idx_tns]
        if not(type(tripfile) == type(None)): 
            triplets_match = triplets[idx_cand]
    else: # tables with only NON-matched sources 
        print("\nSearching for NON-matching sources...")
        tbl_match = setdiff(tbl, tbl[idx_cand])
        tns_match = setdiff(tns, tns[tns_match], keys="Name")
        idx_nomatch = [i for i in range(len(tbl)) if not(i in idx_cand)]
        if not(type(tripfile) == type(None)): 
            triplets_match = triplets[idx_nomatch]  

    print(f'\n{len(idx_cand)} match(es) found for {len(tbl)} input '+
          f'transient candidate(s) cross-matched against {len(tns)} known '+
          'source(s) from the Transient Name Server for a maximum allowed '+
          f'separation of {sep_max}"')

    if match: 
        print("\n\nTable of matched sources:")
    else:
        print("\n\nTable of NON-matched sources:")
    tns_match["Name","RA","DEC","Obj. Type","Disc. Internal Name"].pprint() 

    # write the CANDIDATE file, with matches    
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_TNS_crossmatched{filext}")    
    if filext == ".fits":
        tbl_match.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, format="ascii.csv", overwrite=True)
    
    # always write known TNS transients to a .csv 
    if type(output_tab_TNS) == type(None):
        output_tab_TNS = tabfile.replace(filext, 
                                         "_TNS_crossmatched_catalog.csv")
    tns_match.write(output_tab_TNS, format="ascii.csv", overwrite=True)
    
    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", "_TNS_crossmatched.npy")
        np.save(output_trips, triplets_match)



### VARIABLE STARS ############################################################

def crossmatch_GCVS(tabfile, 
                    tripfile=None, 
                    sep_max=2.0, 
                    match=True, 
                    output_tab=None,
                    output_tab_GCVS=None,
                    output_trips=None):
    """Given a table of candidate transients, search for cross-matches in the 
    General Catalogue of Variable Stars (GCVS), within `sep_max` arcseconds of 
    each other
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    match : bool, optional
        Look for matches? or look for non-matches (default True)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_GCVS : str, optional
        Name for output table with **GCVS** transients which were successfully 
        crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)

    Notes
    -----
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, queries the General 
    Catalog of Variable Stars (~55,000 stars as of 22 November 2019) for 
    sources within `sep_max` arcsec of the candidate transients. If M>0 
    matches are found, writes a table containing the M crossmatched transient 
    *candidates*, another table containing the M *known* GCVS variable stars, 
    and a `(M,3,Y,X)` .npy file containing the M relevant triplets.  

    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")
    
    # find the footprint of the input candidates table
    ra_min, ra_max = np.min(tbl["ra"]), np.max(tbl["ra"]) 
    dec_min, dec_max = np.min(tbl["dec"]), np.max(tbl["dec"])
    ra_width, dec_height = ra_max - ra_min, dec_max - dec_min
    ra_centre, dec_centre = (ra_max+ra_min)/2.0, (dec_max+dec_min)/2.0
    
    # load in the catalogue
    # RAJ2000, DEJ2000 cols cannot be empty
    v = Vizier(columns=["*"], column_filters={"RAJ2000":"!=", "DEJ2000":"!="}, 
               row_limit=-1) # no row limit 
    
    
    # query in the footprint of the candidates table 
    Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                                unit = (u.deg, u.deg)), 
                       width = f'{ra_width}d',
                       height = f'{dec_height}d',
                       catalog="B/gcvs/gcvs_cat", cache=False)
    
    if len(Q) == 0:
        print("\nThere are no variable stars in the General Catalogue of "+
              "Variable Stars in the region of "+
              f"\nRA = [{ra_min}, {ra_max}]"+
              f"\nDec = [{dec_min}, {dec_max}]"+
              "\nExiting.")
        return
    
    cat = Q[0]
    print(f"\nFound {len(cat)} variable stars in the General "+
          "Catalogue of Variable Stars "+cat.meta["description"][13:]+
          " in the region of "+
          f"\nRA = [{ra_min}, {ra_max}]"+
          f"\nDec = [{dec_min}, {dec_max}]")
    
    # convert RAJ2000, DEJ2000 to degrees 
    cat["RAJ2000"] = [f"{c[0:2]}:{c[3:5]}:{c[6:]}" for c in cat["RAJ2000"]]
    cat["DEJ2000"] = [f"{c[0:3]}:{c[4:6]}:{c[7:]}" for c in cat["DEJ2000"]]
    cat["RAJ2000"] = [Angle(c["RAJ2000"]+" hours").degree for c in cat]
    cat["DEJ2000"] = [Angle(c["DEJ2000"]+" degrees").degree for c in cat]

    # informative print 
    cat["GCVS","RAJ2000","DEJ2000","VarType","Period"].pprint() 
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    cat_skycoords = SkyCoord(cat["RAJ2000"]*u.deg, cat["DEJ2000"]*u.deg, 
                             frame="icrs")
    trns_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                              frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_trns, idx_cat, d2d, d3d = cat_skycoords.search_around_sky(
            trns_skycoords, sep_max*u.arcsec)
    
    if len(idx_trns) == 0:
        print('\nNo matches with input sources found for maximum allowed '+
              f'separation of {sep_max}". Exiting.')
        return

    if match: # tables/triplets with only those crossmatched sources
        print("\nSearching for matching sources...")
        tbl_match = tbl[idx_trns]
        cat_match = cat[idx_cat]
        if not(type(tripfile) == type(None)):
            triplets_match = triplets[idx_trns]
    else: # tables with only NON-matched sources 
        print("\nSearching for NON-matching sources...") 
        tbl_match = setdiff(tbl, tbl[idx_trns])
        cat_match = setdiff(cat, cat[idx_cat], keys="GCVS")
        idx_nomatch = [i for i in range(len(tbl)) if not(i in idx_trns)]
        if not(type(tripfile) == type(None)):
            triplets_match = triplets[idx_nomatch] 
    
    print(f'\n{len(idx_trns)} match(es) found for {len(tbl)} input '+
          f'transient candidate(s) cross-matched against {len(cat)} variable '+
          'star(s) in the General Catalogue of Variable Stars '+
          cat.meta["description"][13:]+
          f', for a maximum allowed separation of {sep_max}"')
    if match: 
        print("\n\nTable of matched sources:")
    else:
        print("\n\nTable of NON-matched sources:")        
    cat_match["GCVS","RAJ2000","DEJ2000","VarType","Period"].pprint() 

    # write the CANDIDATE file, with matches     
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_GCVS_crossmatched{filext}")    
    if filext == ".fits":
        tbl_match.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, format="ascii.csv", overwrite=True)
    
    # always write known GCVS stars to a .csv
    if type(output_tab_GCVS) == type(None):
        output_tab_GCVS = tabfile.replace(filext, 
                                          "_GCVS_crossmatched_catalog.csv")
    cat_match.write(output_tab_GCVS, format="ascii.csv", overwrite=True)
    
    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", "_GCVS_crossmatched.npy")
        np.save(output_trips, triplets_match)


def crossmatch_AAVSO(tabfile, 
                     tripfile=None, 
                     sep_max=2.0, 
                     match=True, 
                     output_tab=None,
                     output_tab_AAVSO=None,
                     output_trips=None):
    """Given a table of candidate transients, search for cross-matches in the 
    American Association of Variable Star Observers (AAVSO) International 
    Variable Star Index (VSX) within `sep_max` arcseconds of each other
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    match : bool, optional
        Look for matches? or look for non-matches (default True)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_AAVSO : str, optional
        Name for output table with **AAVSO** transients which were 
        successfully crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)
    
    Notes
    -----
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, queries the American 
    Association of Variable Star Observers (AAVSO) International Variable Star
    indeX (VSX, ~2 million stars as of 22 November 2019) for sources within
    `sep_max` arcsec of the candidate transients. If M>0 matches are found, 
    writes a table containing the M crossmatched transient *candidates*, 
    another table containing the M *known* AAVSO variable stars, and a 
    `(M,3,Y,X)` .npy file containing the M relevant triplets.  
    
    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")
    
    # find the footprint of the input candidates table
    ra_min, ra_max = np.min(tbl["ra"]), np.max(tbl["ra"]) 
    dec_min, dec_max = np.min(tbl["dec"]), np.max(tbl["dec"])
    ra_width, dec_height = ra_max - ra_min, dec_max - dec_min
    ra_centre, dec_centre = (ra_max+ra_min)/2.0, (dec_max+dec_min)/2.0
    
    # load in the catalogue
    # RAJ2000, DEJ2000 cols cannot be empty
    v = Vizier(columns=["*"], column_filters={"RAJ2000":"!=", "DEJ2000":"!="}, 
               row_limit=-1) # no row limit 
    
    # query in the footprint of the candidates table 
    Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                                unit = (u.deg, u.deg)), 
                       width = f'{ra_width}d',
                       height = f'{dec_height}d',
                       catalog="B/vsx/vsx", cache=False)
    
    if len(Q) == 0:
        print("\nThere are no variable stars in the AAVSO International "+
              "Variable Star Index in the region of "+
              f"\nRA = [{ra_min}, {ra_max}]"+
              f"\nDec = [{dec_min}, {dec_max}]"+
              "\nExiting.")
        return
    
    cat = Q[0]

    print(f"\nFound {len(cat)} variable stars in the AAVSO International "+
          cat.meta["description"][:19]+" ("+cat.meta["description"][21:]+") "+
          "in the region of "+
          f"\nRA = [{ra_min}, {ra_max}]"+
          f"\nDec = [{dec_min}, {dec_max}]")

    # informative print 
    cat["Name","RAJ2000","DEJ2000","Type","Period"].pprint() 
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    cat_skycoords = SkyCoord(cat["RAJ2000"], cat["DEJ2000"], 
                             frame="icrs")
    trns_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                                   frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_trns, idx_cat, d2d, d3d = cat_skycoords.search_around_sky(
            trns_skycoords, sep_max*u.arcsec)
    
    if len(idx_trns) == 0:
        print('\nNo matches with input sources found for maximum allowed '+
              f'separation of {sep_max}". Exiting.')
        return

    if match: # tables/triplets with only those crossmatched sources
        print("\nSearching for matching sources...")
        tbl_match = tbl[idx_trns]
        cat_match = cat[idx_cat]
        if not(type(tripfile) == type(None)):
            triplets_match = triplets[idx_trns]
    else: # tables with only NON-matched sources 
        print("\nSearching for NON-matching sources...")
        tbl_match = setdiff(tbl, tbl[idx_trns])
        cat_match = setdiff(cat, cat[idx_cat], keys="Name")
        idx_nomatch = [i for i in range(len(tbl)) if not(i in idx_trns)]
        if not(type(tripfile) == type(None)):
            triplets_match = triplets[idx_nomatch] 
   
    print(f'\n{len(idx_trns)} match(es) found for {len(tbl)} input '+
          f'transient candidate(s) cross-matched against {len(cat)} variable '+
          'star(s) in the AAVSO International '+
          cat.meta["description"][:19]+' ('+cat.meta["description"][21:]+'),'+
          f' for maximum allowed separation of {sep_max}"')

    if match: 
        print("\n\nTable of matched sources:")
    else:
        print("\n\nTable of NON-matched sources:")
    cat_match["Name","RAJ2000","DEJ2000","Type","Period"].pprint() 

    # write the CANDIDATE file with matches    
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, 
                                     f"_AAVSO-VSX_crossmatched{filext}")    
    if filext == ".fits":
        tbl_match.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, format="ascii.csv", overwrite=True)
    
    # always write known AAVSO stars to a .csv
    if type(output_tab_AAVSO) == type(None):
        output_tab_AAVSO = tabfile.replace(filext, 
                                     "_AAVSO-VSX_crossmatched_catalog.csv")
    cat_match.write(output_tab_AAVSO, format="ascii.csv", overwrite=True)
    
    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", 
                                            "_AAVSO-VSX_crossmatched.npy")
        np.save(output_trips, triplets_match)



### QUASARS / ACTIVE GALACTIC NUCLEI ##########################################

def crossmatch_VeronCettyVeron(tabfile, 
                               tripfile=None, 
                               sep_max=2.0, 
                               match=True,
                               output_tab=None,
                               output_tab_VCV=None,
                               output_trips=None):
    """Given a table of candidate transients, search for cross-matches in the 
    Veron-Cetty and Veron catalogue of quasars, within `sep_max` arcseconds of 
    each other
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    match : bool, optional
        Look for matches? or look for non-matches (default True)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_VCV : str, optional
        Name for output table with **Veron-Cetty and Veron** quasars which 
        were successfully crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)
    
    Notes
    -----
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, queries the Veron-Cetty & 
    Veron catalogue of quasars and active galactic nuclei (13ed, Veron-Cetty & 
    Veron 2010) for sources within `sep_max` arcsec of the candidate 
    transients. If M>0 matches are found, writes a table containing the M 
    crossmatched transient *candidates*, another table containing the M 
    *known* quasars/AGN, and a `(M,3,Y,X)` .npy file containing the M relevant 
    triplets.  
    
    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")
    
    # find the footprint of the input candidates table
    ra_min, ra_max = np.min(tbl["ra"]), np.max(tbl["ra"]) 
    dec_min, dec_max = np.min(tbl["dec"]), np.max(tbl["dec"])
    ra_width, dec_height = ra_max - ra_min, dec_max - dec_min
    ra_centre, dec_centre = (ra_max+ra_min)/2.0, (dec_max+dec_min)/2.0
    
    # load in the catalogue
    # RAJ2000, DEJ2000 cols cannot be empty
    v = Vizier(columns=["*"], column_filters={"RAJ2000":"!=", "DEJ2000":"!="}, 
               row_limit=-1) # no row limit 
    
    
    Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                                unit = (u.deg, u.deg)), 
                       width = f'{ra_width}d',
                       height = f'{dec_height}d',
                       catalog="VII/258/vv10", cache=False)
    
    if len(Q) == 0:
        print("\nThere are no quasars/AGN in the Veron-Cetty & Veron 2010 "+
              "catalogue in the region "+
              f"\nRA = [{ra_min}, {ra_max}]"+
              f"\nDec = [{dec_min}, {dec_max}]"+
              "\nExiting.")
        return
    
    cat = Q[0]
    
    print(f"\nFound {len(cat)} quasars/AGN in the Veron-Cetty & Veron "+
          "2010 catalogue in the region of "+
          f"\nRA = [{ra_min}, {ra_max}]"+
          f"\nDec = [{dec_min}, {dec_max}]")
    
    # convert RAJ2000, DEJ2000 to degrees 
    cat["RAJ2000"] = [f"{c[0:2]}:{c[3:5]}:{c[6:]}" for c in cat["RAJ2000"]]
    cat["DEJ2000"] = [f"{c[0:3]}:{c[4:6]}:{c[7:]}" for c in cat["DEJ2000"]]
    cat["RAJ2000"] = [Angle(c["RAJ2000"]+" hours").degree for c in cat]
    cat["DEJ2000"] = [Angle(c["DEJ2000"]+" degrees").degree for c in cat]

    # informative print 
    cat["Cl", "Name","RAJ2000","DEJ2000"].pprint() 
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    cat_skycoords = SkyCoord(cat["RAJ2000"]*u.deg, cat["DEJ2000"]*u.deg, 
                             frame="icrs")
    trns_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                              frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_trns, idx_cat, d2d, d3d = cat_skycoords.search_around_sky(
            trns_skycoords, sep_max*u.arcsec)
    
    if len(idx_trns) == 0:
        print('\nNo matches with input sources found for maximum allowed '+
              f'separation of {sep_max}". Exiting.')
        return

    if match: # tables/triplets with only those crossmatched sources
        print("\nSearching for matching sources...")
        tbl_match = tbl[idx_trns]
        cat_match = cat[idx_cat]
        if not(type(tripfile) == type(None)):
            triplets_match = triplets[idx_trns]
    else: # tables with only NON-matched sources 
        print("\nSearching for NON-matching sources...") 
        tbl_match = setdiff(tbl, tbl[idx_trns])
        cat_match = setdiff(cat, cat[idx_cat], keys="Name")
        idx_nomatch = [i for i in range(len(tbl)) if not(i in idx_trns)]
        if not(type(tripfile) == type(None)):
            triplets_match = triplets[idx_nomatch] 

    print(f'\n{len(idx_trns)} MATCH(ES) representing '+
          f'{len(unique(cat[idx_cat],keys="Name"))} intrinsic SOURCE(S) '+
          f'found for {len(tbl)} input transient candidate(s) '+
          f'cross-matched against {len(cat)} quasars/AGN in the '+
          'Veron-Cetty & Veron 2010 catalogue, for a maximum '+
          f'allowed separation of {sep_max}"')

    if match: 
        print("\n\nTable of matched sources:")
    else:
        print("\n\nTable of NON-matched sources:")
    cat_match["Cl", "Name","RAJ2000","DEJ2000"].pprint() 

    # write the CANDIDATE file, with matches     
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, 
                                     f"_VeronCettyVeron_crossmatched{filext}")    
    if filext == ".fits":
        tbl_match.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, format="ascii.csv", overwrite=True)
    
    # always write known Veron-Cetty & Veron quasars/AGN to a .csv
    if type(output_tab_VCV) == type(None):
        output_tab_VCV = tabfile.replace(filext, 
                                "_VeronCettyVeron_crossmatched_catalog.csv")
    cat_match.write(output_tab_VCV, format="ascii.csv", overwrite=True)
    
    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", 
                                        "_VeronCettyVeron_crossmatched.npy")
        np.save(output_trips, triplets_match)


def crossmatch_MILLIQUAS(tabfile, 
                         tripfile=None, 
                         sep_max=2.0, 
                         match=True,
                         output_tab=None,
                         output_tab_MILLIQUAS=None, 
                         output_trips=None):
    """Given a table of candidate transients, search for cross-matches in the 
    Million Quasars (MILLIQUAS) catalogue of quasars, within `sep_max` 
    arcseconds of each other
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    match : bool, optional
        Look for matches? or look for non-matches (default True)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_MILLIQUAS : str, optional
        Name for output table with **MILLIQUAS** quasars which were 
        successfully crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)
    
    Notes
    -----
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, queries the Million 
    Quasars (MILLIQUAS) catalogue of quasars (v6.3, Flesch 2019, updated 16 
    June 2019) for sources within `sep_max` arcsec of the candidate 
    transients. If M>0 matches are found, writes a table containing the M 
    crossmatched transient *candidates*, another table containing the M 
    *known* quasars, and a `(M,3,Y,X)` .npy file containing the M relevant 
    triplets.  

    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")
    
    # find the footprint of the input candidates table
    ra_min, ra_max = np.min(tbl["ra"]), np.max(tbl["ra"]) 
    dec_min, dec_max = np.min(tbl["dec"]), np.max(tbl["dec"])
    ra_width, dec_height = ra_max - ra_min, dec_max - dec_min
    ra_centre, dec_centre = (ra_max+ra_min)/2.0, (dec_max+dec_min)/2.0
    
    # load in the catalogue
    # RAJ2000, DEJ2000 cols cannot be empty
    v = Vizier(columns=["*"], column_filters={"RAJ2000":"!=", "DEJ2000":"!="}, 
               row_limit=-1) # no row limit 
    
    
    Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                                unit = (u.deg, u.deg)), 
                       width = f'{ra_width}d',
                       height = f'{dec_height}d',
                       catalog="VII/283/catalog", cache=False)
    
    if len(Q) == 0:
        print("\nThere are no quasars in the MILLIQUAS v6.3 (16 June 2019) "+
              "catalogue in the region "+
              f"\nRA = [{ra_min}, {ra_max}]"+
              f"\nDec = [{dec_min}, {dec_max}]"+
              "\nExiting.")
        return
    
    cat = Q[0]
    
    print(f"\nFound {len(cat)} quasars in the MILLIQUAS v6.3 (16 June 2019)"+
          " catalogue in the region of "+
          f"\nRA = [{ra_min}, {ra_max}]"+
          f"\nDec = [{dec_min}, {dec_max}]")

    # informative print 
    cat["Cl_Ass", "Name","RAJ2000","DEJ2000"].pprint() 
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    cat_skycoords = SkyCoord(cat["RAJ2000"], cat["DEJ2000"], 
                             frame="icrs")
    trns_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                              frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_trns, idx_cat, d2d, d3d = cat_skycoords.search_around_sky(
            trns_skycoords, sep_max*u.arcsec)
    
    if len(idx_trns) == 0:
        print('\nNo matches with input sources found for maximum allowed '+
              f'separation of {sep_max}". Exiting.')
        return

    if match: # tables/triplets with only those crossmatched sources
        print("\nSearching for matching sources...")
        tbl_match = tbl[idx_trns]
        cat_match = cat[idx_cat]
        if tripfile:
            triplets_match = triplets[idx_trns]
    else: # tables with only NON-matched sources 
        print("\nSearching for NON-matching sources...")
        tbl_match = setdiff(tbl, tbl[idx_trns])
        cat_match = setdiff(cat, cat[idx_cat], keys="Name")
        idx_nomatch = [i for i in range(len(tbl)) if not(i in idx_trns)]
        if tripfile:
            triplets_match = triplets[idx_nomatch] 
    
    print(f'\n{len(idx_trns)} MATCH(ES) representing '+
          f'{len(unique(cat[idx_cat],keys="Name"))} intrinsic SOURCE(S) '+
          f'found for {len(tbl)} input transient candidate(s) '+
          f'cross-matched against {len(cat)} quasars '+
          'in the MILLIQUAS v6.3 (16 June 2019) catalogue, for a maximum '+
          f'allowed separation of {sep_max}"')

    if match: 
        print("\n\nTable of matched sources:")
    else:
        print("\n\nTable of NON-matched sources:")
    cat_match["Cl_Ass", "Name","RAJ2000","DEJ2000"].pprint() 

    # write the CANDIDATE file, with matches     
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, 
                                 f"_MILLIQUAS_crossmatched{filext}")    
    if filext == ".fits":
        tbl_match.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, format="ascii.csv", overwrite=True)
    
    # always write known MILLIQUAS quasars to a .csv
    if type(output_tab_MILLIQUAS) == type(None):
        output_tab_MILLIQUAS = tabfile.replace(filext, 
                                "_MILLIQUAS_crossmatched_catalog.csv")
    cat_match.write(output_tab_MILLIQUAS, format="ascii.csv", overwrite=True)
    
    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", 
                                            "_MILLIQUAS_crossmatched.npy")
        np.save(output_trips, triplets_match)



### Pan-STARRS1 3pi STELLAR SOURCES ###########################################
    
def crossmatch_PS1_stellar(tabfile, 
                           tripfile=None, 
                           sep_max=2.0, 
                           match=True,
                           output_tab=None,
                           output_tab_PS1=None, 
                           output_trips=None):
    """Given a table of candidate transients, search for cross-matches among 
    the stellar sources in the Pan-STARSS1 3pi catalogue, within `sep_max` 
    arcseconds of each other

    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str, optional
        .npy file containing triplets corresponding to `tabfile` (default 
        None; if provided, will write matches to a new file)
    sep_max : float, optional
        Maximum separation to consider two sources to be the same, in 
        arcseconds (default 2.0)
    match : bool, optional
        Look for matches? or look for non-matches (default True)
    output_tab : str, optional
        Name for output table with candidate transients which were 
        successfully crossmatched (default set by function)
    output_tab_PS1 : str, optional
        Name for output table with **Pan-STARSS1** stars which were 
        successfully crossmatched (default set by function)
    output_trips : str, optional
        Name for output .npy file with crossmatched triplets (default set by 
        function)
    
    Arguments
    ---------
    For a single .fits or .csv table containing N transient candidates and a 
    single triplet .npy file containing N triplets, queries the Pan-STARRS 1 
    3pi survey (Chambers et al 2016) for **stellar** sources within `sep_max` 
    arcseconds of the candidate transients. If M>0 matches are found, writes a 
    table containing the M crossmatched transient  *candidates*, another table
    containing the M *known* quasars, and a `(M,3,Y,X)` .npy file containing 
    the M relevant triplets.  

    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    # load in triplets, if given
    if not(type(tripfile) == type(None)):
        triplets = np.load(tripfile, mmap_mode="r")
        if not(len(tbl) == len(triplets)):
            raise ValueError("Length of table with candidate transients "+
                             "does not match number of input triplets")
    
    # load in the catalogue
    # RAJ2000, DEJ2000 cols cannot be empty, must have >=5 multi-epoch 
    # detections, and cannot be extended
    v = Vizier(columns=["objID", "Qual", "RAJ2000","DEJ2000", "Nd"], 
               column_filters={"RAJ2000":"!=", "DEJ2000":"!=", "Nd":">=5"},
               row_limit=-1, # no row limit    
               timeout=600) # 600 s timeout  

    tbl_temp = tbl.copy()
    tbl_temp["_RAJ2000"] = tbl["ra"]*u.deg # need columns with these names
    tbl_temp["_DEJ2000"] = tbl["ra"]*u.deg # need columns with these names    
    
    Q = v.query_region(tbl_temp, radius=f"{sep_max}s",
                       catalog="II/349/ps1", cache=False)
    
    if len(Q) == 0:
        print('\nThere are no stellar sources in the Pan-STARRS1 3pi survey '+
              f'within {sep_max:.2f}" of the input sources. \nExiting.')
        return 
    
    cat = Q[0]
    
    ## get rid of extended sources 
    # convert Qual number to binary
    # if last (rightmost) bit or second-to-last bit is 1, then Qual 
    # includes 2**0 = 1 or 2**1 = 2, which means it is extended
    from timeit import default_timer as timer
    start = timer()
    qual_bin = [bin(cat["Qual"].tolist()[i]) for i in range(len(cat))]
    cat_mask = [not(q[-1] == '1' or q[-2] == '1') for q in qual_bin]
    print(f"\nBefore removing extended sources, N = {len(cat)}")
    cat = cat[cat_mask]
    print(f"After removing extended sources, N = {len(cat)}")
    end = timer()
    print(f"t = {end-start:.2f} s")
    
    print(f'\nFound {len(cat)} stellar (non-extended) sources in the '+
         f'Pan-STARRS1 3pi survey within {sep_max:.2f}" of the input sources.')

    # informative print 
    cat["objID", "Qual", "RAJ2000","DEJ2000", "Nd"].pprint() 
    
    # build skycoord objects out of table of candidates and candidate RA, Decs
    # to crossmatch to
    cat_skycoords = SkyCoord(cat["RAJ2000"], cat["DEJ2000"], 
                             frame="icrs")
    trns_skycoords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, 
                              frame="icrs")   

    # look for sources within <sep_max> arcsec of each other 
    idx_trns, idx_cat, d2d, d3d = cat_skycoords.search_around_sky(
            trns_skycoords, sep_max*u.arcsec)
    
    if len(idx_trns) == 0:
        print('\nNo matches with input sources found for maximum allowed '+
              f'separation of {sep_max}". Exiting.')
        return  
    
    if match: # tables/triplets with only those crossmatched sources
        print("\nSearching for matching sources...")
        tbl_match = tbl[idx_trns]
        cat_match = cat[idx_cat]
        if tripfile:
            triplets_match = triplets[idx_trns]
    else: # tables with only NON-matched sources 
        print("\nSearching for NON-matching sources...")
        tbl_match = setdiff(tbl, tbl[idx_trns])
        cat_match = setdiff(cat, cat[idx_cat], keys="Name")
        idx_nomatch = [i for i in range(len(tbl)) if not(i in idx_trns)]
        if tripfile:
            triplets_match = triplets[idx_nomatch] 
    
    print(f'\n{len(idx_trns)} MATCH(ES) representing '+
          f'{len(unique(cat[idx_cat],keys="objID"))} intrinsic SOURCE(S) '+
          f'found for {len(tbl)} input transient candidate(s) '+
          f'cross-matched against {len(cat)} stellar (non-extended) sources '+
          'in the Pan-STARRS 1 3pi survey, for a maximum '+
          f'allowed separation of {sep_max}"')
    cat["objID", "Qual", "RAJ2000","DEJ2000", "Nd"].pprint() 

    # write the CANDIDATE file, with matches     
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, 
                                 f"_PS1_crossmatched{filext}")    
    if filext == ".fits":
        tbl_match.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        tbl_match.write(output_tab, format="ascii.csv", overwrite=True)

    # always write PS1 matches to a .csv
    if type(output_tab_PS1) == type(None):
        output_tab_PS1 = tabfile.replace(filext, 
                                "_PS1_crossmatched_catalog.csv")
    cat_match.write(output_tab_PS1, format="ascii.csv", overwrite=True)

    # write triplets?
    if not(type(tripfile) == type(None)): # were input triplets provided?
        if type(output_trips) == type(None):
            output_trips = tripfile.replace(".npy", "_PS1_crossmatched.npy")
        np.save(output_trips, triplets_match)



### MATCHING TO MULTI-ORDER COVERAGE MAPS (MOCs) ##############################

def MOC_crossmatch_TNS(moc_file, TNS_tabfile, output_tab_TNS=None):
    """Given a Multi-Order Coverage (MOC) map and table of transients from the 
    Transient Name Server (TNS), identify TNS transients which lie in the MOC 
    footprint

    Arguments
    ---------
    moc_file : str
        .fits file containing MOC
    TNS_tabfile : str
        Table of known transients from Transient Name Server (must be a .csv)
    output_tab_TNS : str, optional
        Name for output table with **TNS** transients which were successfully 
        crossmatched (default set by function)

    Returns
    -------
    astropy.table.Table
        Table of TNS sources in the MOC footprint
            
    Given some table of sources from TNS, produces a table of only those 
    sources which are also in the footprint defined by the MOC. 
    
    """
    from mocpy import MOC
    
    # load in the MOC 
    moc = MOC.from_fits(moc_file)

    # load in TNS transients (always a csv)
    tns = Table.read(TNS_tabfile, format="ascii.csv")
    # convert RA, DEC to degrees 
    tns["RA"] = [float(Angle(str(t["RA"])+" hours").degree) for t in tns]
    tns["DEC"] = [float(Angle(str(t["DEC"])+" degrees").degree) for t in tns]
    ra = tns["RA"]
    dec = tns["DEC"]
    
    # get only the sources which lie in the footprint 
    contains = moc.contains(ra*u.deg, dec*u.deg)
    tns_valid = tns[contains]    
    
    # write
    if type(output_tab_TNS) == type(None):
        output_tab_TNS = moc_file.replace(".fits","_crossmatch_TNS.fits")
        tns_valid.write(output_tab_TNS, format="ascii", overwrite=True)
    elif ".fits" in output_tab_TNS:
        tns_valid.write(output_tab_TNS, format="ascii", overwrite=True)
    elif ".csv" in output_tab_TNS:        
        tns_valid.write(output_tab_TNS, format="ascii.csv", overwrite=True)
        
    return tns_valid


def MOC_crossmatch_AAVSO(moc_file, output_tab_AAVSO=None):
    """Find AAVSO sources in the region defined by some Multi-Order Coverage 
    (MOC) map 
    
    Arguments
    ---------
    moc_file : str
        .fits file containing MOC
    output_tab_AAVSO : str, optional
        Name for output table with **AAVSO** transients which were successfully 
        crossmatched (default set by function)
    
    Returns
    -------
    astropy.table.Table
        Table of American Association of Variable Star Observers (AAVSO) 
        international Variable Source indeX (VSC) sources in the MOC 
        footprint
        
    Notes
    -----
    Gets a very barebones table of all AAVSO sources present in an input MOC 
    file by querying Vizier.
    
    """
    from mocpy import MOC
    
    moc = MOC.from_fits(moc_file)
    
    # query Vizier for sources in the footprint, keep relevant columns
    tab = moc.query_vizier_table("B/vsx/vsx")
    tab = tab["Name","RAJ2000","DEJ2000","Type","Period"]
    
    # informative print
    tab.pprint()
    
    # write
    if type(output_tab_AAVSO) == type(None):
        output_tab_AAVSO = moc_file.replace(".fits","_crossmatch_AAVSO.fits")
        tab.write(output_tab_AAVSO, format="ascii", overwrite=True)
    elif ".fits" in output_tab_AAVSO:
        tab.write(output_tab_AAVSO, format="ascii", overwrite=True)
    elif ".csv" in output_tab_AAVSO:        
        tab.write(output_tab_AAVSO, format="ascii.csv", overwrite=True)
        
    return tab


