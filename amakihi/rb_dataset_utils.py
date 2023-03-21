#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Thu Feb  2 03:23:35 2023
.. @author: Nicholas Vieira
.. @rb_dataset_utils.py

Utility functions for (pre-)processing the tables, triplets, etc. which go 
into training the bogus-real adversarial artificial intelligence (braai) 
convolutional neural network. 

"""

import glob
import re
import sys
import numpy as np

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import Angle, SkyCoord

from scipy.ndimage import rotate

### MERGING DATASETS ##########################################################

def merge_triplets(direc, filt=None, output_trips=None):
    """Search a directory for .npy files containing triplets and merge into 
    one large array, for use in preparing a training set. 
    
    Arguments
    ---------
    direc : str
        Directory full of .npy files
    filt : str, optional
        String filter to use when searching for files by name (default None; 
        e.g., "2019")
    output_trips : str, optional
        Name for output file (default None --> do not write)
        
    Returns
    -------
    np.ndarray
        Array of merged triplets
        
    Notes
    -----
    For a directory containing N .npy triplet files, each with shape 
    `(t_i, 3, Y, X)`, combines them into a single merged ndarray with shape
    `(sum(t_i) i=1...N, 3, Y, X)`
    
    """
    
    if direc[-1] == "/":
        direc = direc[:-1]    
    if not(filt):
        filt=""
    files = glob.glob(f"{direc}/*{filt}*.npy")
    files.sort()
    for f in files: print(f)
    
    triplets = [np.load(f) for f in files] # load in all triplets
    merge = np.concatenate(triplets) # merge them
    
    if not(type(output_trips) == type(None)):
        np.save(output_trips, merge)    
    
    return merge


def merge_tables(direc, filext=".fits", filt=None, output_tab=None, *cols):
    """Search a directory for tables of candidate transients generated using 
    `amakihi` and merge into one large table, for use in preparing a training 
    set.
    
    Arguments
    ---------
    direc : str
        Directory containing tables 
    filext : {".fits", ".csv"}, optional
        File extension to search for (".csv" or ".fits", default ".fits")
    filt : str, optional
        String filter to use when searching for files by name (default None; 
        e.g., "2019")
    output_tab : str, optional
        Name for output file (default None --> do not write)
    *cols : strs, optional
        Names of columns to use from the input tables   
    
    Returns
    -------
    astropy.table.Table
        Merged table
    
    Notes
    -----
    For a directory of N astropy tables of transient candidates in the .fits 
    **or** .csv format, each with `r_i` rows, merges them into a single table 
    with `sum(r_i) i=1...N` rows.
    
    Currently does not allow for merging tables with different columns.
    
    """

    # load in data
    if direc[-1] == "/":
        direc = direc[:-1]       
    if not(filt):
        filt=""
    
    if not filext in (".fits", ".csv"):
        raise ValueError("Expected filext one of '.csv.' or '.fits', did not "+
                         f"recognize {filext}")
    files = glob.glob(f"{direc}/*{filt}*.{filext}")
    
    if filext == ".fits":
        tab0 = Table.read(files[0], format="ascii")
    else:
        tab0 = Table.read(files[0], format="ascii.csv")
    files.sort()

    for f in files: 
        print(f, flush=True)
    
    try: tab0.remove_column("id") # remove id column if it's present
    except KeyError: pass
    
    # decide which table columns to use, initialize the table         
    if cols: # merge these columns
        colnames = [str(c) for c in cols]
    else: # merge only the columns present in the first table
        colnames = tab0.colnames
        # get rid of any empty columns
        colnames = [c for c in colnames.copy() if not(tab0[c][0] == 'None')]
        
    cand_table = Table.read(files[0], format="ascii")[colnames]

    # add each row 
    for f in files[1:]:
        if filext == ".fits":
            tab = Table.read(f, format="ascii")[colnames]
        else:
            tab = Table.read(f, format="ascii.csv")[colnames]
        for row in tab:
            try:
                cand_table.add_row(row)
            except:
                e = sys.exc_info()
                print(f"\nError: \n{str(e[0])}\n{str(e[1])}", flush=True)
                continue

    if not(type(output_tab) == type(None)):
        if ".fits" in output_tab:
            cand_table.write(output_tab, format="ascii", overwrite=True)
        elif ".csv" in output_tab:
            cand_table.write(output_tab, format="ascii.csv", overwrite=True)
        else:
            raise ValueError("\nInvalid file extension chosen for the merged "+
                             f"table {output_tab}; valid choices are .csv "+
                             "or .fits")
            
    return cand_table


### MERGING TABLES WITH DIFFERENT COLUMNS ########################################################

def merge_catalogs(*catalogs, output=None):
    """Merge catalogs/tables, looking only for the essential columns
    
    Arguments
    ---------
    *catalogs : str
        Filenames for as many catalogs (tables of sources; must be .fits or 
        .csv files) with (at minimum) columns giving the RA, Dec, and a name
    output : str, optional
        Filename for output table (default set by function)
        
    Returns
    -------
    astropy.table.Table
        Merged table

    Notes
    -----
    Valid keys for the RA are: 'ra', 'RA', 'RAJ2000'; valid keys for the 
    Declination are 'dec', 'Dec', 'DEJ2000'. Must have a `Name` column.
    
    """
    
    rows = []
    for c in catalogs:
        
        # load in table
        if ".fits" in c:
            tbl = Table.read(c, format="ascii")
        elif ".csv" in c:
            tbl = Table.read(c, format="ascii.csv")
        else:
            raise ValueError("all catalogs must be of filetype .csv or "+
                             f".fits, did not recognize {c}")
        
        # get RAs, DECs
        cols = tbl.colnames
        ra_col = [c for c in cols if (("ra" in c) or ("RA" in c))]
        if "RAJ2000" in ra_col: ra_col = "RAJ2000"
        else: ra_col = ra_col[0]
            
        dec_col = [c for c in cols if (("dec" in c) or ("DEC" in c) or (
                "DEJ" in c) or ("de" in c))]
        if "DEJ2000" in dec_col: dec_col = "DEJ2000"
        else: dec_col = dec_col[0]
        
        # if RA, DEC are in sexagesimal coords
        if 'TNS AT' in cols and ":" in tbl[ra_col][0]:
            tbl[ra_col] = [float(Angle(str(t[ra_col])+" hours").degree) for 
               t in tbl]
            tbl[dec_col] = [float(Angle(str(t[dec_col])+" degrees").degree) for 
               t in tbl]        

        ras = tbl[ra_col].data.tolist()
        decs = tbl[dec_col].data.tolist()
        
        newtbl = Table(data=[tbl["Name"], ras, decs], 
                       names=["Name", "ra", "dec"])
        
        for row in newtbl: rows.append(row) # add to row list

    final_table = Table(rows=rows, names=["Name", "ra", "dec"])

    if type(output) == type(None):
        output = "catalogs_merged.fits"
        final_table.write(output, format="ascii", overwrite=True)
    elif ".fits" in output:
        final_table.write(output, format="ascii", overwrite=True)
    elif ".csv" in output:        
        final_table.write(output, format="ascii.csv", overwrite=True)
        
    return final_table



### AUGMENTING DATASETS #######################################################

def augment_dataset(tab_dir, triplet_dir, outdir, factor=4, filt=None, 
                    real_only=False, *cols):
    """Augment a dataset for use in training a braai neural network, by 
    rotating triplets.
    
    Arguments
    ---------
    candidate_dir : str
        Directory containing candidate transient tables
    triplet_dir : str
        Directory containing candidatre triplets (in .npy files)
    outdir : str
        Directory in which to write the augmented dataset (tables and triplets)
    factor : {2,3,4}, optional
        Factor by which to augment the dataset (default 4)
    filt : str, optional
        String filter to use when searching for files by name (default None; 
        e.g., "2019")
    real_only : bool, optional
        Apply the augmentation factor to only candidates with the label 1 
        (real)? or all candidates (default False)
    *cols : str, optional
        Names for the columns to take from the tables (default save all 
        columns for which the first row in the table is populated)

    Notes
    -----
    For a directory of N astropy tables of transient candidates in the .fits 
    format and another directory containing N .npy files, augments this 
    dataset by rotating each of the three sub-arrays in the triplets in 
    `factor` increments of 90 degrees, where `factor` can be 2, 3, or 4. 
    
    E.g., for 100 tables of candidates, each with 10 candidates, and 100 
    triplet files, each with 10 triplets, a factor of 3 will produce a new 
    dataset containing 100 tables, each with 30 candidates, and 100 triplet 
    files, each with 30 triplets. These new tables and triplets will then be 
    written to the directory `outdir`.
    
    The purpose of this function is to augment the size of a dataset to be used
    in training a braai neural network. 
    
    """
    
    # check the factor 
    if not(int(factor) in [2,3,4]):
        raise ValueError("Expected factor to be one of 2, 3, 4; did not "+
                         f"recognize {factor}")
    factor = int(factor)
    
    ## load in data
    if tab_dir[-1] == "/":
        tab_dir = tab_dir[:-1]
    if triplet_dir[-1] == "/":
        triplet_dir = triplet_dir[:-1]                                     
    if type(filt) == type(None):
        filt=""
    tabfiles = glob.glob(f"{tab_dir}/*{filt}*.csv")
    tabfiles.sort()
    tripfiles = glob.glob(f"{triplet_dir}/*{filt}*.npy")
    tripfiles.sort()
    
    if not(len(tabfiles) == len(tripfiles)):
        raise ValueError(f"Number of table files [{len(tripfiles)}] does not "+
                         f"match number of triplet files [{len(tripfiles)}]")

    # load in a table and decide which columns to use    
    tab0 = Table.read(tabfiles[0], format="ascii.csv")
    try: tab0.remove_column("id") # remove id column if it's present       
    except KeyError: pass

    if cols:
        colnames = [str(c) for c in cols]
    else:
        colnames = tab0.colnames
        colnames = [c for c in colnames.copy() if not(tab0[c][0] == 'None')]
    
    ## loop through all of the triplets and candidate tables
    for i in range(len(tabfiles)):
        f = tabfiles[i]
        t = tripfiles[i]

        ## for the i'th table 
        tab = Table.read(f, format="ascii.csv")[colnames]
        newtab = Table(names=colnames)
        for row in tab: # for each row 
            try:
                for n in range(factor): # add row <factor> times
                    ### SHOULD THERE BE SOME CHANGE IN X, Y HERE OR NOT NEEDED?
                    newtab.add_row(row)
            except:
                e = sys.exc_info()
                print(f"\nError: \n{str(e[0])}\n{str(e[1])}", flush=True)
                continue
        
        newf = f.replace(".fits", f"_augment{factor}.fits")
        newf = f"{outdir}/"+re.sub(".*/", "", 
                                 newf) # for a file /a/b/c, extract the "c"
        newtab.write(newf, format="ascii.csv", overwrite=True) # write it
        
        ## for the i'th triplet
        trip = np.load(t)
        newtrip = []
        
        for tr in trip:
            for n in range(factor): 
                # add triplet <factor> times
                # for each addition, rotate all 3 sub-arrays by 90.0 degrees 
                # more than the previous rotation, where the first rotation is 
                # 0.0 degrees
                newtrip.append(np.array([rotate(tr[0], 90.0*n),
                                         rotate(tr[1], 90.0*n),
                                         rotate(tr[2], 90.0*n)])) 

        newtrip = np.stack(newtrip) # (3, size, size) --> (N, 3, size, size)
        
        newf = t.replace(".npy", f"_augment{factor}.npy")
        newf = f"{outdir}/"+re.sub(".*/", "", 
                                 newf) # for a file /a/b/c, extract the "c"
        np.save(newf, newtrip) # write it


def augment_dataset_single(tabfile, tripfile, 
                           factor=4, real_only=False, 
                           output_tab=None, output_trips=None, *cols):
    """Same as :func:`augment_dataset`, but for a single `tabfile` and 
    `tripfile`.

    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    factor : {2,3,4}, optional
        Factor by which to augment the dataset (default 4)
    real_only : bool, optional
        Apply the augmentation factor to only candidates with the label 1 
        (real)? or all candidates (default False)
    output_tab : str, optional
        Name for output table with augmented set of candidate transients 
        (default set by function)
    output_trips : str, optional
        Name for output .npy file with augmented set of triplets (default 
        set by function)
    *cols : str, optional
        Names for the columns to take from the tables (default save all 
        columns for which the first row in the table is populated)

    Notes
    -----
    For a single table of N candidates and a corresponding .npy file of N 
    triplets, augments this dataset by rotating each of the three sub-arrays 
    in the triplets in `factor` increments of 90 degrees, where `factor` can 
    be 2, 3, or 4.
    
    E.g., for 100 candidates and 100 triplets, a factor of 3 will produce a 
    new dataset containing 300 candidates 300 triplets. The new table and 
    triplets will then be written to `output_tab` and `output_trips`, 
    respectively. 
    
    The purpose of this function is to augment the size of a dataset to be 
    used in training a neural network. 

    """
    
    # check the factor 
    if not(int(factor) in [2,3,4]):
        raise ValueError("Expected factor to be one of 2, 3, 4; did not "+
                         f"recognize {factor}")
    factor = int(factor)

    ## load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")

    # decide which columns to use   
    # temporarily get rid of the science, ref, diff columns 
    sci = list(tbl["science"])
    template = list(tbl["template"])
    diff = list(tbl["difference"])
    tbl.remove_columns(["science","template","difference"])    
    try: tbl.remove_column("id") # remove id column if it's present
    except KeyError: pass

    if cols:
        colnames = [str(c) for c in cols]
    else:
        colnames = tbl.colnames
        colnames = [c for c in colnames.copy() if not(tbl[0][c] == 'None')]
    
    ## build up the new table 
    newtab = Table(names=colnames)
    for i in range(len(tbl)):
        row = tbl[i] # loop through each row
        # if we only want to augment REAL candidates, and the row in question
        # is a bogus candidate, skip to the next row:
        if real_only and int(row["label"]) != 1: 
            newtab.add_row(row)
            continue
        try:
            for n in range(factor): # add row <factor> times
                ### SHOULD THERE BE SOME CHANGE IN X, Y HERE OR NOT NEEDED?
                newtab.add_row(row)
            #for m in range(factor):
                sci.insert(i+1, sci[i])
                template.insert(i+1, template[i])
                diff.insert(i+1, diff[i])
        except:
            e = sys.exc_info()
            print(f"\nError: \n{str(e[0])}\n{str(e[1])}", flush=True)
            continue 
        
    # save the new table
    if type(output_tab) == type(None):
        if real_only:
            output_tab = tabfile.replace(filext, f"_augment{factor}real{filext}")
        else:
            output_tab = tabfile.replace(filext, f"_augment{factor}all{filext}") 
            
    newtab["science"] = sci
    newtab["template"] = template
    newtab["difference"] = diff            
    if filext == ".fits":
        newtab.write(output_tab, format="ascii", overwrite=True) 
    elif filext == ".csv":
        newtab.write(output_tab, format="ascii.csv", overwrite=True)

    ## load in the triplets
    triplets = np.load(tripfile, mmap_mode="r")  
    
    ## build up the new triplets
    newtrip = []
    #print(len(triplets))
    for i in range(len(triplets)): 
        tr = triplets[i] # loop through each triplet
        # if we only want to augment REAL candidates, and the triplet in 
        # question is a bogus candidate, skip to the next triplet
        if real_only and int(tbl[i]["label"]) != 1: 
            newtrip.append(tr) # add triplet only once
            continue
        for n in range(factor): 
            # add triplet <factor> times
            # for each addition, rotate all 3 sub-arrays by 90.0 degrees 
            # more than the previous rotation, where the first rotation is 
            # 0.0 degrees
            newtrip.append(np.array([rotate(tr[0], 90.0*n),
                                     rotate(tr[1], 90.0*n),
                                     rotate(tr[2], 90.0*n)])) 
    newtrip = np.stack(newtrip) # (3, size, size) --> (N, 3, size, size)
    
    # save the new triplets
    if type(output_trips) == type(None):
        if real_only:
            output_trips = tripfile.replace(".npy", f"_augment{factor}real.npy")
        else:
            output_trips = tripfile.replace(".npy", f"_augment{factor}all.npy")
    np.save(output_trips, newtrip) # write it
    
    # helpful prints
    print(f"Original table length: {len(tbl)}", flush=True)
    print(f"Augmented table length: {len(newtab)}", flush=True)


def augment_SMOTE(tabfile, tripfile, newfrac=0.5, output_trips=None):
    """Apply Synthetic Minority Over-sampling Technique (SMOTE) to obtain a 
    more class-balanced training set.

    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    newfrac : float, optional
        Fraction (between 0.0, 1.0) of the data which should belong to the 
        minority class after SMOTE (default 0.5, i.e., augmented dataset 
        will be half real and half bogus)
    output_trips : str, optional
        Name for output .npy file with augmented set of triplets (default 
        set by function)

    Notes
    -----
    **Work in progress: does not yet write an updated table.**
    
    See https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html 
    for details and a link to the paper introducing SMOTE. 
    
    """

    from imblearn.over_sampling import SMOTE 
    
    ## load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        #filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        #filext = ".csv"
    else:
        raise ValueError("tabfile must be of filetype .csv or .fits, did not "+
                         f"recognize {tabfile}")
    
    labels = np.array(list(tbl["label"]))
    triplets = np.load(tripfile, mmap_mode='r')
    
    random_state = 13
    
    print(f'Original dataset shape: (bogus: {len(triplets[labels==0])}, '+
          f'real: {len(triplets[labels==1])})', flush=True)  
    
    sm = SMOTE(sampling_strategy=newfrac, random_state=random_state)
    triplets_res, labels_res = sm.fit_resample([np.ravel(t) for t in triplets], 
                                                labels)
    triplets_res = np.reshape(triplets_res, (triplets_res.shape[0], 3, 63, 63))
    
    print('Resampled dataset shape: ('+
          f'bogus: {len(triplets_res[labels_res==0])}, '+
          f'real: {len(triplets_res[labels_res==1])})', flush=True)

    # save the new triplets
    if type(output_trips) == type(None):
        output_trips = tripfile.replace(".npy", 
                            f"_augment_SMOTE_newfrac-{newfrac}-real.npy")
    np.save(output_trips, triplets_res) # write it



### REPEATERS (FOR TRAINING SETS WITH TRIPLETS ACROSS MULTIPLE EPOCHS) ########

def __get_idx_repeaters(tbl, sep_max=1.0, dt_min=0.1):
    """Get indices of repeaters in a table
    
    Arguments
    ---------
    tbl : astropy.table.Table
        Table of candidate transients
    sep_max : float, optional
        *Maximum* allowed angular separation to consider two sources to be 
        the same, in arcseconds (default 1.0)
    dt_min : float, optional
        *Minimum* time separation between potential matches, in days 
        (default 0.1)
        
    Returns
    -------
    idx_og : list
        Indices for any sources which were matched at least once
    idx_final : list
        Indices for repeated sources, with respect to their row number in the 
        original input table

    """
    
    tbl["idx"] = [i for i in range(len(tbl))] # temporary idx column 
    
    # build skycoord objects out of table of candidates 
    tbl_coords = SkyCoord(tbl["ra"]*u.deg, tbl["dec"]*u.deg, frame="icrs") 
    
    # crossmatch catalog with itself 
    idx, idx, d2d, d3d = tbl_coords.search_around_sky(tbl_coords, 
                                                      sep_max*u.arcsec)     
    idx = idx[d2d>0] # remove indices which matched themselves 

    # verify that the sources are not temporally coincident
    idx_cop = idx.copy()
    idx = []
    for i in range(len(idx_cop)//2): 
        dt = np.abs(tbl["MJD"][idx_cop[2*i]] - tbl["MJD"][idx_cop[2*i+1]])  
        if dt > dt_min: # if not temporally coincident 
            idx.append(idx_cop[2*i])
            idx.append(idx_cop[2*i+1])
    idx_og = np.unique(idx.copy()).tolist() # idx for sources matched >=1 times
    
    tbl_matched = tbl[idx_og]
    tbl_matched.sort(["ra","dec","MJD"]) # sort the candidates 
    idx_sorted = tbl_matched["idx"].data.tolist()
    
    # build a new table containing only ONE of each group of matched sources 
    newtbl = Table(rows=tbl_matched[0]) # first row only
    coord = SkyCoord(newtbl[0]["ra"]*u.deg, newtbl[0]["dec"]*u.deg, 
                     frame="icrs")    
    for i in range(1, len(idx_sorted)): 
        row = tbl_matched[i]
        rcoord = SkyCoord(row["ra"]*u.deg, row["dec"]*u.deg, frame="icrs")
        sep = rcoord.separation(coord)
        if (sep.arcsecond < sep_max): # if a matched source
            if row["MJD"] < newtbl[-1]["MJD"]: 
                newtbl[-1] = row
            else: 
                continue
        else: # if a new source, add to table and continue 
            newtbl.add_row(row)
            coord = rcoord
            
    idx_final = newtbl["idx"].data.tolist()
            
    return idx_og, idx_final

 
def repeaters_pick(tabfile, tripfile, #sep_max=1.0, dt_min=0.5,
                   output_tab=None, output_trips=None):
    """Pick out sources which are repeated in a given dataset 
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    output_tab : str, optional
        Name for output table with only sources which were repeated (default 
        set by function)
    output_trips : str, optional
        Name for output .npy file with only triplets which were repeated 
        (default set by function)

    Notes
    -----
    Given some table of sources, find sources within 0.001 degrees = 3.6 
    arcseconds of each other and write a table/corresponding triplet file 
    including only these repeated sources.
        
    E.g., for a table of 1000 candidates where 50 sources were detected 
    exactly twice, 30 sources detected exactly three times, and 10 sources 
    detected exactly four times, the final table will contain 
    50 + 30 + 10 = 90 remaining candidates. 

    """
    from astropy.table import setdiff, unique

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

    # load in triplets
    triplets = np.load(tripfile, mmap_mode="r")  

#    ## use the __get_idx_repeaters() function
#    # crossmatch catalog with itself 
#    idx_og, idx_final = __get_idx_repeaters(tbl, sep_max, dt_min)
#    
#    # new table/triplets of ONLY the repeated sources
#    newtbl = tbl[idx_final]
#    newtrips = []
#    for i in idx_final: newtrips.append(triplets[i])
#    newtrips = np.stack(newtrips)
#    newtbl.remove_column("idx")
#    
#    # informative prints
#    print(f'\nfound {len(idx_final)} sources for which >=1 other source(s) '+
#          f'were found within angular separation {sep_max:.2f}" '+
#          f'and detections were made more than {dt_min:.2f} days apart')
#    newtbl["ra","dec","MJD"].pprint() # print      

    # temporary idx column 
    tblc = tbl.copy()
    tblc["idx"] = [i for i in range(len(tbl))] # temporary idx column 

    # round the ra, dec rows
    tblc["ra"] = np.round(tblc["ra"], decimals=3)
    tblc["dec"] = np.round(tblc["dec"], decimals=3)

    # get rows which do NOT repeat 
    tblc_norep = unique(tblc, keys=["ra","dec"], keep="none")

    # for rows which do repeat, keep only one (the first)
    tblc_unique = unique(tblc, keys=["ra","dec"], keep="first")
    nrepeaters = len(tblc_unique) - len(tblc_norep)
    
    # table of ONLY the repeaters
    repeaters = setdiff(tblc_unique, tblc_norep)
    repeaters.sort(["ra","dec","MJD"])
    #print("\nrepeaters:\n", flush=True)
    #repeaters["ra","dec","MJD"].pprint()

    # informative prints
    print(f'\nfound {nrepeaters} intrinsic source(s) detected across'+
          ' multiple epochs', flush=True)
    # print(f'\nfound {len(tblc_norep)} sources which were only detected in a '+
    #       'single observation epoch')

    # indices of repeaters only 
    repeaters.sort(["ra","dec","MJD"])
    idx_sorted = repeaters["idx"].tolist() # sorted
    
    # make the table and triplets
    newtbl = tbl[idx_sorted]
    newtrips = []
    for i in idx_sorted: newtrips.append(triplets[i])
    newtrips = np.stack(newtrips)    
    
    # informative prints    
    print(f'\n--> {len(newtbl)} source(s) written to final table:\n', 
          flush=True)
    newtbl["ra","dec","MJD"].pprint() # print 

    # write the table
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_repeatpick{filext}")
    if filext == ".fits":
        newtbl.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        newtbl.write(output_tab, format="ascii.csv", overwrite=True)
        
    # write the triplets    
    if type(output_trips) == type(None):
        output_trips = tripfile.replace(".npy", "_repeatpick.npy")
    np.save(output_trips, newtrips)
    

def repeaters_remove(tabfile, tripfile, #sep_max=1.0, dt_min=0.5, 
                     output_tab=None, output_trips=None):
    """Pick out sources which are **not** repeated in a given dataset 
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    output_tab : str, optional
        Name for output table with only sources which were **not** repeated 
        (default set by function)
    output_trips : str, optional
        Name for output .npy file with only triplets which were **not** 
        repeated (default set by function)

    Notes
    -----
    Given some table of sources, find sources within 0.001 degrees = 3.6 
    arcseconds of each other and write a table/corresponding triplet file 
    which does not include repeats.

    E.g., for a table of 1000 candidates where 50 sources were detected 
    exactly twice, 30 sources detected exactly three times, and 10 sources 
    detected exactly four times, the final table will contain 
    1000 - (2-1)*50 - (3-1)*30 - (4-1)*10 = 860 remaining candidates.

    """

    from astropy.table import setdiff, unique
    
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

    # load in triplets
    triplets = np.load(tripfile, mmap_mode="r")
    
#    # crossmatch catalog with itself 
#    # get indices of repeated sources 
#    idx_og, idx_final = __get_idx_repeaters(tbl, sep_max, dt_min)
#    
#    # informative prints
#    print(f'\nfound {len(idx_final)} sources for which >=1 other source(s) '+
#          f'were found within angular separation {sep_max:.2f}" '+
#          f'and detections were made more than {dt_min:.2f} days apart')
#    print(f'\nfound {len(tbl)-len(np.unique(idx_og))} sources which were '+
#          'only detected in a single observation epoch')
#    
#    # get indices of sources which did NOT repeat 
#    for i in range(len(tbl)):
#        if not(i in np.unique(idx_og)): idx_final.append(i)
#
#    # make the table and triplets
#    newtbl = tbl[idx_final]
#    newtbl.sort(["ra","dec","MJD"])
#    idx_sorted = newtbl["idx"] # idx, sorted
#    newtrips = []
#    for i in idx_sorted: newtrips.append(triplets[i])
#    newtrips = np.stack(newtrips)
#    newtbl.remove_column("idx")
#
#    # informative prints
#    print(f'\n--> {len(idx_final)} sources written to final table:\n')
#    newtbl["ra","dec","MJD"].pprint() # print 

    # temporary idx column 
    tblc = tbl.copy()
    tblc["idx"] = [i for i in range(len(tbl))] # temporary idx column 

    # round the ra, dec, MJD rows
    tblc["ra"] = np.round(tblc["ra"], decimals=3)
    tblc["dec"] = np.round(tblc["dec"], decimals=3)

    # get rows which do NOT repeat 
    tblc_norep = unique(tblc, keys=["ra","dec"], keep="none")

    # for rows which do repeat, keep only one (the first)
    tblc_unique = unique(tblc, keys=["ra","dec"], keep="first")
    nrepeaters = len(tblc_unique) - len(tblc_norep)
    
    # table of only the repeaters
    repeaters = setdiff(tblc, tblc_norep)
    repeaters.sort(["ra","dec","MJD"])
    print("\nrepeaters:\n", flush=True)
    repeaters["ra","dec","MJD"].pprint()

    # informative prints
    print(f'\nfound {nrepeaters} intrinsic source(s) detected across'+
          ' multiple epochs', flush=True)
    print(f'\nfound {len(tblc_norep)} source(s) only detected in a '+
          'single observation epoch', flush=True)

    # indices of non-repeaters + sources which only repeated once
    tblc_unique.sort(["ra","dec","MJD"])
    idx_sorted = tblc_unique["idx"].tolist() # sorted
    
    # make the table and triplets
    newtbl = tbl[idx_sorted]
    newtrips = []
    for i in idx_sorted: newtrips.append(triplets[i])
    newtrips = np.stack(newtrips)    
    
    # informative prints    
    print(f'\n--> {len(newtbl)} source(s) written to final table:\n', 
          flush=True)
    newtbl["ra","dec","MJD"].pprint() # print 

    # write the table
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_repeatremove{filext}")
    if filext == ".fits":
        newtbl.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        newtbl.write(output_tab, format="ascii.csv", overwrite=True)
        
    # write the triplets    
    if type(output_trips) == type(None):
        output_trips = tripfile.replace(".npy", "_repeatremove.npy")
    np.save(output_trips, newtrips)



### PREPARING TRIPLETS FOR TRAINING ###########################################
    
def triplets_normalize(tripfile, output_trips=None):
    """Normalize triplets for training.
    
    Arguments
    ---------
    tripfile : str
        .npy file containing triplets
    output_trips : str, optional
        Name for output .npy file with normalized set of triplets (default 
        set by function) 
    
    Returns
    -------
    np.ndarray
        Array of normalized triplets
    
    Notes
    -----
    For an array of N triplets with shape `(N, 3, X, Y)`, normalizes each 
    science, reference and difference image **independently** using L2 
    normalization. Should be called **BEFORE** rearranging the triplets into 
    shape `(N, X, Y, 3)` with :func:`triplets_rearrange`.

    """
    
    from sklearn.preprocessing import normalize

    triplets = np.load(tripfile, mmap_mode="r")
    new_triplets = []
    
    for i in range(len(triplets)):
        new_triplets.append([normalize(triplets[i][0], norm="l2", axis=0), 
                             normalize(triplets[i][1], norm="l2", axis=0), 
                             normalize(triplets[i][2], norm="l2", axis=0)]) 
    new_triplets = np.stack(new_triplets)
    
    # write triplets
    if type(output_trips) == type(None):
        output_trips = tripfile.replace(".npy", "_L2norm.npy")    
    np.save(output_trips, new_triplets)
    
    return new_triplets
    
       
def triplets_rearrange(tripfile, output_trips=None):
    """Re-arrange triplets to a shape compatible with bogus-real adversarial 
    artifiical intelligence (braai).

    Arguments
    ---------
    tripfile : str
        .npy file containing triplets
    output_trips : str, optional
        Name for output .npy file with re-arranged set of triplets (default 
        set by function) 
    
    Returns
    -------
    np.ndarray
        Array of re-arranged triplets
    
    Notes
    -----
    For an array of N triplets with shape `(N, 3, X, Y)`, rearranges the array 
    to have shape `(N, X, Y, 3)`. This is the required format for training the 
    braai. Should be used as the **FINAL** step of pre-processing, after 
    merging, augmenting, normalizing, etc. Calling this function on several 
    triplet files and then trying to merge, augment, etc. will **NOT** work. 
    
    """
    
    triplets = np.load(tripfile, mmap_mode="r",
                       allow_pickel=True)
    new_triplets = []
    
    # (N, 3, size, size) --> (N, size, size, 3)    
    for i in range(len(triplets)):
        temp = np.ndarray((63,63,3))
        for j in range(63):
            for k in range(63):
                temp[j,k] = [triplets[i][0][j,k], triplets[i][1][j,k], 
                     triplets[i][2][j,k]]
        new_triplets.append(temp.copy())
    new_triplets = np.stack(new_triplets)
    
    # write triplets
    if type(output_trips) == type(None):
        output_trips = tripfile.replace(".npy", "_braaiready.npy")    
    np.save(output_trips, new_triplets)
    
    return new_triplets
