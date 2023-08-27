# helpers.py
#
# This file contains helpers function for PyXspec 
# 
# Author: Simon van Eeden

import os
from xspec import *

# Set calibrated energy range of instrument and grating
HEG_REGION = [1., 10.] # High Energy Grating (HEG) of Chandra
MEG_REGION = [0.4, 5.] # Medium Energy Grating (MEG) of Chandra

def load_spectrum(spectrum_path, calib_region):
    """
    Loads spectrum into Xspec and returns the spectrum object.

    Parameters:
    spectrum_path (str): Path to the PHA2 spectrum file
    calib_region (list): Calibrated energy range of the instrument

    Returns:
    s (Spectrum): Spectrum object in Xspec
    """

    # Extract filename
    filename = spectrum_path.split('/')[-1]

    # Add the directory of the spectrum to the system paths,
    # in order to read the arf and rmf files
    sdir = spectrum_path.replace(filename, '')
    os.chdir(sdir) 

    # Make sure earlier loaded spectra are removed from memory
    AllData.clear()

    # Load spectrum in Xspec
    s = Spectrum(spectrum_path)

    # Ignore bad pixels
    AllData.ignore("bad") 

    # Ignore uncalibrated spectral regions
    AllData.ignore("{0:.1f}-** **-{1:.1f}".format(calib_region[1], calib_region[0]))  

    print("Spectrum loaded in Xspec")
    print("\t Filename: {0:s}".format(spectrum_path))
    print("\t Object name: {0:s}".format(s.fileinfo('OBJECT').strip()))
    print("\t Obs ID: {0:s}".format(s.fileinfo('OBS_ID').strip()))
    print("")

    return s