#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Thu Feb  2 03:29:02 2023
.. @author: Nicholas Vieira
.. @rb_model_utils.py

Utility functions for interacting with (real-bogus) Keras models: writing, 
reading, and applying to candidate transients. 

"""

import numpy as np

from astropy.table import Table

###############################################################################
### SAVING/LOADING KERAS MODELS ###############################################

def write_model(model, output_json=None, output_weights=None):
    """Save a Keras model by writing the architecture to a .json file and 
    writing the weights to a .h5 file
    
    Arguments
    ---------
    model : tf.keras.model
        Keras model to save
    output_json : str, optional
        Name for .json file containing the architecture (default 'model.json')
    output_weights : str, optional
        Name for .h5 file containing the model weights (default 'model.h5')

    """
    
    # set output names
    if type(output_json) == type(None):
        output_json = "model.json"
    if type(output_weights) == type(None):
        output_weights = "model.h5"
        
    # check input filenames
    if not(".json" in output_json):
        raise ValueError("output_json must be a .json file, did not "+
                         f"recognize {output_json}")
    if not(".h5" in output_weights):
        raise ValueError("output_weights must be a .h5 file, did not "+
                         f"recognize {output_weights}")
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights("model.h5")



def load_model(input_json, input_weights):
    """Load in a Keras model from the architecture (.json file) and weights 
    (.h5 file)
    
    Arguments
    ---------
    input_json : str
        .json file containing the architecture
    input_weights : str
        .h5 file containing the model weights
        
    Returns
    -------
    tf.keras.model
        Keras model

    """
    
    from tensorflow.keras.models import model_from_json
    
    with open(input_json, 'r') as json_file:
        loaded_model_json = json_file.read()
    m = model_from_json(loaded_model_json)
    
    m.load_weights(input_weights)    
    
    return m



###############################################################################
### USE A TRAINED MODEL ON SOME DATASET + DIAGNOSTICS #########################

def use_model(tabfile, tripfile, input_json, input_weights, 
              rb_thresh=0.5, output_tab=None):
    """Apply some trained model to a set of candidate transients contained in 
    a table and array of triplets. 
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    input_json : str
        .json file containing the architecture
    input_weights : str
        .h5 file containing the model weights
    rb_thresh : float, array_like, optional
        RB score(s) to use as threshold for a source to be considered real, 
        between 0.0 and 1.0 (default 0.5 --> sources with RB score above 0.5 
        considered real; only used for printing the aggregate results; can 
        provide an array to print results for multiple scores)
    output_tab : str, optional
        Name for output table with their RB scores (default set by function)
        
    Returns
    -------
    np.ndarray
        Array of RB scores for each triplet
    
    Notes
    -----
    Input triplets must already be normalized 
    (:func:`rb_dataset_utils.triplets_normalize`) and re-arranged 
    (:func:`rb_dataset_utils.triplets_rearrange`) to be compatible with 
    braai.

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
    
    # load in triplets    
    triplets = np.load(tripfile, mmap_mode="r")
    if not(len(tbl) == len(triplets)):
        raise ValueError("Length of table with candidate transients "+
                         "does not match number of input triplets")

    # load in the keras model
    model = load_model(input_json, input_weights)
    
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
        print(f"\nRB threshold = {t}", flush=True)
        print(f"N_REAL = {nreal} [{(100*nreal/(nbogus+nreal)):2f}%]",
              flush=True)
        print(f"N_BOGUS = {nbogus} [{(100*nbogus/(nbogus+nreal)):2f}%]",
              flush=True) 
    
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_braai_preds{filext}")
    if filext == ".fits":
        newtbl.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        newtbl.write(output_tab, format="ascii.csv", overwrite=True)
    
    return label_preds


def __apply_threshold(tabfile, tripfile, rb_thresh, real=True):
    """Apply the RB score threshold to some table and triplets. 
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    rb_thresh : float
        RB score(s) to use as threshold for a source to be considered real, 
        between 0.0 and 1.0
    real : bool, optional
        Use the threshold to select real sources above some threshold? or 
        sources below (default True)
    
    Returns
    -------
    filext : str
        File extension (.fits or .csv) of the input table 
    new_tbl : astropy.table.Table
        Table of sources with threshold applied
    new_triplets : np.ndarray
        Triplets with threshold applied
    
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

    # load in triplets    
    triplets = np.load(tripfile, mmap_mode="r")
    if not(len(tbl) == len(triplets)):
        raise ValueError("Length of table with candidate transients "+
                         "does not match number of input triplets")
    
    # get real or bogus sources only
    if real:
        mask = (tbl["label_preds"] > rb_thresh)
    else:
        mask = (tbl["label_preds"] < rb_thresh)
        
    new_tbl = tbl[mask]
    new_triplets = triplets[mask]
    
    return filext, new_tbl, new_triplets



def write_reals(tabfile, tripfile, rb_thresh=0.5, 
                output_tab=None, output_trips=None):
    """Given a table and triplets which have been labelled by some model 
    predictions, write only the sources with scores **ABOVE** `rb_thresh`.
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    rb_thresh : float, array_like, optional
        RB score(s) to use as threshold for a source to be considered real, 
        between 0.0 and 1.0 (default 0.5 --> sources with RB score above 0.5 
        will be considered real)
    output_tab : str, optional
        Name for output table with their RB scores (default set by function)
    output_trips : str, optional
        Name for output .npy file with triplets labeled as real (default set 
        by function)
    
    """

    # select the sources **above** the given RB threshold
    filext, realtbl, realtriplets = __apply_threshold(tabfile, tripfile, 
                                                      rb_thresh, 
                                                      real=True)

    # write the table
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_rb-above-{rb_thresh}{filext}")
    if filext == ".fits":
        realtbl.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        realtbl.write(output_tab, format="ascii.csv", overwrite=True)

    # write the triplets    
    if type(output_trips) == type(None):
        output_trips = tripfile.replace(".npy", f"_rb-above-{rb_thresh}.npy")
    np.save(output_trips, realtriplets)



def write_bogus(tabfile, tripfile, rb_thresh=0.5, 
                output_tab=None, output_trips=None):
    """Given a table and triplets which have been labelled by some model 
    predictions, write only the sources with scores **BELOW** `rb_thresh`.
    
    Arguments
    ---------
    tabfile : str
        Table of candidate transients (must be a .csv or .fits file)
    tripfile : str
        .npy file containing triplets corresponding to `tabfile`
    rb_thresh : float, array_like, optional
        RB score(s) to use as threshold for a source to be considered real, 
        between 0.0 and 1.0 (default 0.5 --> sources with RB score below 0.5 
        will be considered bogus)
    output_tab : str, optional
        Name for output table with their RB scores (default set by function)
    output_trips : str, optional
        Name for output .npy file with triplets labeled as bogus (default set 
        by function)
    
    """

    # select the sources **below** the given RB threshold
    filext, bogustbl, bogustriplets = __apply_threshold(tabfile, tripfile, 
                                                        rb_thresh, 
                                                        real=False)

    # write the table
    if type(output_tab) == type(None):
        output_tab = tabfile.replace(filext, f"_rb-below-{rb_thresh}{filext}")
    if filext == ".fits":
        bogustbl.write(output_tab, format="ascii", overwrite=True)
    elif filext == ".csv":
        bogustbl.write(output_tab, format="ascii.csv", overwrite=True)

    # write the triplets    
    if type(output_trips) == type(None):
        output_trips = tripfile.replace(".npy", f"_rb-below-{rb_thresh}.npy")
    np.save(output_trips, bogustriplets)
    


###############################################################################
### CONVOLUTIONAL NEURAL NETWORK MODEL ########################################

def vgg6(input_shape=(63, 63, 3), n_classes: int = 1):
    """VGG6 convolutional neural network. 
    
    Notes
    -----
    Notes taken from braai: 
    
    "VGG-like sequential model (VGG6; this architecture was first proposed by 
    the Visual Geometry Group of the Department of Engineering Science, 
    University of Oxford, UK)

    The model has six layers with trainable parameters: four convolutional and 
    two fully-connected. The first two convolutional layers use 16 3x3 pixel 
    filters each while in the second pair, 32 3x3 pixel filters are used. To 
    prevent over-fitting, a dropout rate of 0.25 is applied after each max-
    pooling layer and a dropout rate of 0.5 is applied after the second fully-
    connected layer. ReLU activation functions (Rectified Linear Unit -- a 
    function defined as the positive part of its argument) are used for all 
    five hidden trainable layers; a sigmoid activation function is used for 
    the output layer."
    
    """
    
    import tensorflow as tf

    model = tf.keras.models.Sequential(name='VGG6')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    
    # first layer (convolutional)
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 
                                     input_shape=input_shape, name='conv1'))
    # second layer (convolutional)
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 
                                     name='conv2'))
    # maxpool 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # dropout of 0.25
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # third layer (convolutional)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                     name='conv3'))
    # fourth layer (convolutional)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                     name='conv4'))
    
    # maxpool 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
    # dropout of 0.25
    model.add(tf.keras.layers.Dropout(rate=0.25))
    # what does Flatten do ????
    model.add(tf.keras.layers.Flatten())
    # what does Dense do ???? (this layer: fully connected)
    model.add(tf.keras.layers.Dense(256, activation='relu', 
                                    name='fc_1'))
    # dropout of 0.5
    model.add(tf.keras.layers.Dropout(rate=0.5))
        
    # output layer (fully connected)
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(tf.keras.layers.Dense(n_classes, activation=activation, 
                                    name='fc_out'))
    return model
