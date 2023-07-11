# amakihi

[![image](https://img.shields.io/pypi/v/amakihi.svg)](https://pypi.python.org/pypi/amakihi)

[![image](https://img.shields.io/travis/nvieira-mcgill/amakihi.svg)](https://travis-ci.com/nvieira-mcgill/amakihi)

[![Documentation Status](https://readthedocs.org/projects/amakihi/badge/?version=latest)](https://amakihi.readthedocs.io/en/latest/?version=latest)

End-to-end image differencing and transient detection software,
including:

-   Downloading reference images (a.k.a. templates) from various surveys
-   Image background subtraction/estimation
-   Image masking, including masking of saturated pixels
-   Building effective Point-Spread Functions (ePSFs)
-   Image alignment (a.k.a. image registration)
-   Image differencing (a.k.a. image subtraction)
-   Transient detection and basic vetting of candidate transients

The end products of a pipeline constructed from `amakihi` are
\"triplets\", i.e., [N]{.title-ref} length-3 arrays of the (science,
reference, difference) images cropped around [N]{.title-ref} candidate
transient sources. These candidate transients can then be further vetted
with e.g. your favourite machine learning algorithm. (I use
[braai](https://github.com/dmitryduev/braai)).

Some modules for interfacing with `braai` are included here as well.

This software was developed to serve in a pipeline for the MegaCam
instrument of the Canada-France-Hawaii Telescope (CFHT). This pipeline
was used for all image differencing and transient detection in the
following paper describing our CFHT MegaCam follow-up of the
gravitational wave event GW190814:

[Vieira, N., Ruan, J.J, Haggard, D., Drout, M.R. et al. 2020, ApJ, 895,
96, 2. \*A Deep CFHT Optical Search for a Counterpart to the Possible
Neutron Star - Black Hole Merger
GW190814.](https://ui.adsabs.harvard.edu/abs/2020arXiv200309437V/abstract)

## Documentation

Detailed documentation for all modules can be found
[here](https://amakihi.readthedocs.io/en/latest/).. In the future,
example scripts/notebooks will be added.

## Installation

Currently, needs to be installed directly from github. May be
installable with `conda` and/or `pip` in the future.

Dependencies must be installed manually for the time being. Dependencies
are:

**Essential dependencies:**

-   numpy
-   scipy
-   matplotlib
-   [astropy](https://docs.astropy.org/en/stable/)
-   [photutils](https://photutils.readthedocs.io/en/stable/)
-   [scikit-image](https://scikit-image.org/)
-   [scikit-learn](https://scikit-learn.org/stable/install.html)
-   [astroquery](https://astroquery.readthedocs.io/en/latest/) (for
    [crossmatch]{.title-ref} module)
-   [lmxl](https://lxml.de/) (for [query_CFIS]{.title-ref} module)

**Semi-optional dependencies:**

-   [tensorflow](https://www.tensorflow.org/install) (if using
    [rb_model_utils.write_model()]{.title-ref},
    [.load_model()]{.title-ref}, [.use_model()]{.title-ref}, or
    [.vgg6()]{.title-ref})

**Really optional dependencies:**

-   [imblearn](https://imbalanced-learn.org/stable/index.html) (if using
    [rb_dataset_utils.augment_dataset_SMOTE()]{.title-ref})

**Non-Python:**

-   [astrometry.net](http://astrometry.net/use.html) (can however be
    ignored in favour of source detection with the image segmentation
    methods of `photutils` instead)
-   [hotpants](https://github.com/acbecker/hotpants)

Once you have the dependencies, to install `amakihi`, simply run :

    python setup.py install

## Contact

<nicholas.vieira@mail.mcgill.ca>

## Credits

Free software: MIT license

This package was created in part with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
