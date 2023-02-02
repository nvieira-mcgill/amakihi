#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Thu Feb  2 03:29:02 2023
.. @author: Nicholas Vieira
.. @CNN_model_utils.py

"""

import numpy as np

from astropy.table import Table

###############################################################################
### SAVING/LOADING KERAS MODELS ###############################################

def write_model(model, json_filename=None, weights_filename=None):
    """
    Input:
        - Keras model to be saved
        - name of the json file in which to save the model architecture 
          (optional; default set below, MUST be .json file)
        - name for h5 file in which to save the weights (optional; default 
          set below, MUST be .h5 file)
    
    Saves a trained Keras model.
    
    Output: None
    """
    
    # set output names
    if not(json_filename):
        json_filename = "model.json"
    if not(weights_filename):
        weights_filename = "model.h5"
        
    # check input filenames
    if not(".json" in json_filename):
        print("\nThe <json_filename> MUST be a .json file. Exiting")
        return
    if not(".h5" in weights_filename):
        print("\nThe <weights_filename> MUST be a .h5 file. Exiting.")
        return
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model(json_filename, weights_filename):
    """
    Input:
        - name of the .json file containing the architecture
        - name of the .h5 file containing the weights
    
    Loads in a Keras model.
    
    Output: the model
    """
    
    from tensorflow.keras.models import model_from_json
    
    with open(json_filename, 'r') as json_file:
        loaded_model_json = json_file.read()
    m = model_from_json(loaded_model_json)
    m.load_weights(weights_filename)    
    
    return m

###############################################################################
### USE A TRAINED MODEL ON SOME DATASET + DIAGNOSTICS #########################
    
def use_model(tabfile, tripfile, json_filename, weights_filename, 
              rb_thresh=0.5, output=None):
    """
    Input: 
        - single table of transient candidates (must be of form .csv or .fits)
        - single .npy file containing corresponding triplets (triplets should 
          be NORMALIZED and REARRANGED using functions triplet_normalize() and
          triplet_rearrange() )
        - name for the .json file containing the keras model
        - name for the .h5 file containing the weights for the model
        - RB score threshold for a source to be considered "real" (optional; 
          default 0.5; only used when printing info about the results; can be 
          an array/list or an int/float)
        - name for the output table of candidates with their RB scores 
          (optional; default set below)
    
    Uses a previously trained Keras model on a new dataset (i.e., new table of 
    candidates and corresponding triplets.)
    
    Output: an array containing the RB scores for each triplet in the dataset
    """
    
    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        print("\nInput table must be of filetype .csv or .fits. Exiting.")
        return
    # load in triplets    
    triplets = np.load(tripfile, mmap_mode="r")

    if not(len(tbl) == len(triplets)):
        print("\nThe number of candidate transients in the input table does "+
              "not match the number of triplets. Exiting.")
        return
    # load in the keras model
    model = load_model(json_filename, weights_filename)
    
    ## get predictions !
    label_preds = model.predict(triplets, verbose=1)
    
    # add predictions to a new table 
    newtbl = tbl
    newtbl["label_preds"] = label_preds.flatten()   
    
    # informative prints 
    if type(rb_thresh) in (float, int):
        rb_thresh = [rb_thresh]
    for t in rb_thresh:
        nreal = len(newtbl[np.array(label_preds.flatten()) > t])
        nbogus = len(newtbl[np.array(label_preds.flatten()) < t])
        print(f"\nAt RB threshold = {t}:")
        print(f"N_REAL = {nreal} [{(100*nreal/(nbogus+nreal)):2f}%]")
        print(f"N_BOGUS = {nbogus} [{(100*nbogus/(nbogus+nreal)):2f}%]") 
    
    if not(output):
        output = tabfile.replace(filext, f"_braai_preds{filext}")
    if filext == ".fits":
        newtbl.write(output, format="ascii", overwrite=True)
    elif filext == ".csv":
        newtbl.write(output, format="ascii.csv", overwrite=True)
    
    return label_preds


def write_reals(tabfile, tripfile, key="label_preds", rb_thresh=0.5, 
                tabout=None, tripout=None):
    """
    Input:
        - single table of transient candidates which have been assigned an RB 
          score using a braai model (must be of form .csv or .fits)
        - single .npy file containing corresponding triplets (triplets need not
          be normalized/rearranged)
        - the key (column name) to search for the predicted RB score (optional; 
          default "label_preds")
        - the RB score to set as the threshold for real sources (optional; 
          default 0.5)
        - name for the output table of candidates with RB scores ABOVE the RB 
          threshold such that they are considered real (optional; default set 
          below)
        - name for output .npy array of triplets with RB scores ABOVE the RB 
          threshold such that they are considered real (optional; default set 
          below)
    
    Given some dataset of candidates labelled by a neural net, writes a new 
    table/triplet file containing only those with an assigned RB score ABOVE 
    <rb_thresh>.
    
    Can also be used to isolate real sources from a labelled dataset designated
    for training by setting key="label". 
    
    Output: None
    """

    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        print("\nInput table must be of filetype .csv or .fits. Exiting.")
        return

    # load in triplets    
    triplets = np.load(tripfile, mmap_mode="r")

    if not(len(tbl) == len(triplets)):
        print("\nThe number of candidate transients in the input table does "+
              "not match the number of triplets. Exiting.")
        return

    # select the sources above the given RB threshold
    realmask = (tbl[key] > rb_thresh)
    realtbl = tbl[realmask]
    realtriplets = triplets[realmask]

    # write the table
    if not(tabout):
        tabout = tabfile.replace(filext, f"_rba{rb_thresh}{filext}")
    if filext == ".fits":
        realtbl.write(tabout, format="ascii", overwrite=True)
    elif filext == ".csv":
        realtbl.write(tabout, format="ascii.csv", overwrite=True)

    # write the triplets    
    if not(tripout):
        tripout = tripfile.replace(".npy", f"_rba{rb_thresh}.npy")
    np.save(tripout, realtriplets)


def write_bogus(tabfile, tripfile, key="label_preds", rb_thresh=0.5, 
                tabout=None, tripout=None):
    """
    Input:
        - single table of transient candidates which have been assigned an RB 
          score using a braai model (must be of form .csv or .fits)
        - single .npy file containing corresponding triplets (triplets need not
          be normalized/rearranged)
        - the key (column name) to search for the predicted RB score (optional; 
          default "label_preds")
        - the RB score to set as the threshold for real sources (optional; 
          default 0.5)
        - name for the output table of candidates with RB scores BELOW the RB 
          threshold such that they are considered real (optional; default set 
          below)
        - name for output .npy array of triplets with RB scores BELOW the RB 
          threshold such that they are considered real (optional; default set 
          below)

    Given some dataset of candidates labelled by a neural net, writes a new 
    table/triplet file containing only those with an assigned RB score BELOW 
    <rb_thresh>.
         
    Output: None
    """

    # load in table
    if ".fits" in tabfile:
        tbl = Table.read(tabfile, format="ascii")
        filext = ".fits"
    elif ".csv" in tabfile:
        tbl = Table.read(tabfile, format="ascii.csv")
        filext = ".csv"
    else:
        print("\nInput table must be of filetype .csv or .fits. Exiting.")
        return

    # load in triplets    
    triplets = np.load(tripfile, mmap_mode="r")

    if not(len(tbl) == len(triplets)):
        print("\nThe number of candidate transients in the input table does "+
              "not match the number of triplets. Exiting.")
        return

    # select the sources above the given RB threshold
    bogusmask = (tbl[key] < rb_thresh)
    bogustbl = tbl[bogusmask]
    bogustriplets = triplets[bogusmask]

    # write the table
    if not(tabout):
        tabout = tabfile.replace(filext, f"_rbb{rb_thresh}{filext}")
    if filext == ".fits":
        bogustbl.write(tabout, format="ascii", overwrite=True)
    elif filext == ".csv":
        bogustbl.write(tabout, format="ascii.csv", overwrite=True)

    # write the triplets    
    if not(tripout):
        tripout = tripfile.replace(".npy", f"_rbb{rb_thresh}.npy")
    np.save(tripout, bogustriplets)