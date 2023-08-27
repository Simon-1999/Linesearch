# grid.py
#
# This file contains the Grid class which is used as a framework to perform the linesearch,
# load the output of a linesearch and to analyse the output.
#
# Author: Simon van Eeden

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from xspec import *
import numpy as np
import os
from scipy.constants import c, h  

class Grid():
    """
    Object class to run and analyse a grid search. 
    """

    def __init__(self):
        """
        Sets all parameters to None.
        """

        # Grid parameters
        self.start = None
        self.end = None
        self.step_size = None
        self.energies = None
        self.wavelengths = None
        self.num_grid_points = None

        # Search parameters
        self.line_width = None
        self.spectrum_path = None
        self.search_mode = None
        self.window_size = None
        
        # Search output parameters
        self.significance = None
        self.delta_fitstat = None
        self.initial_continuum = None
        self.optimized_continuum = None
        self.const_factor = None
        self.fit_err = None
        self.fit_stat = None
        self.runtime = None
        self.output_path = None
        self.fit_model = None
        self.date = None
        
        # Observation details
        self.object_name = None
        self.obs_id = None
        self.obs_mode = None
        self.instrument = None
        self.telescope = None

    # Import methods
    from ._search import run_search, run_custom_search
    from ._plots import plot_search, plot_significance, plot_gridpoint

    def set_grid_points(self, start=1., end=10., line_width=500., step_size=0.3):
        """
        Creates and set energy grid into Grid class 

        Parameters
        start (float): Energy where linesearch will start (keV)
        end (float): Energy where linesearch will end (keV)
        line_width (float): Width of gaussian line feature (km/s)
        step_size (float): Stepsize, as a fraction of the gaussian line width
        """

        # Set grid parameters
        self.start = start
        self.end = end
        self.step_size = step_size

        # Create energy grid
        energies = []
        e_center = start
        while e_center < end:
            # Save energy in grid
            energies.append(e_center)

            # Get line width in energy (keV)
            line_energy_width = line_width*1000/c * e_center

            # Calculate next energy gridpoint
            e_center += line_energy_width*step_size

        self.energies = np.array(energies)
        self.wavelengths = np.array([(h*c)/(kev*1.6022e-16)*1e10 for kev in self.energies])
        self.num_grid_points = len(self.energies)

        print("Initialized energy grid")
        print("\t Grid start: \t {0:.2f} keV".format(self.start))
        print("\t Grid end: \t {0:.2f} keV".format(self.end))
        print("\t Step fraction:  {0:.2f} (fraction of line width)".format(self.step_size))
        print("\t Line width: \t {0:.0f} km/s".format(line_width))
        print("\t Grid points: \t {0:d}".format(self.num_grid_points))
        print("")

    def load_search(self, output_path, object_name=None):
        """
        Reads header and search information from the linesearch output text file.

        Parameters
        output_path (str): Path to the linesearch output file
        object_name (str): Manually specified object name. Defaults to None
        """

        # Read in header information
        header = {}
        with open(output_path) as f:
            # Iterate through the file until the table starts
            for line in f:
                if not line.startswith('#'):
                    break

                stripped = line.strip('# \n').split('=')
                header[stripped[0]] = stripped[1]

        # Read in data
        dat = np.genfromtxt(output_path, names=header["COLUMN_NAMES"])

        # Set search parameters
        self.start = float(header["GRID_START"])
        self.end = float(header["GRID_END"])
        self.step_size = float(header["GRID_STEP_SIZE"])
        self.energies = dat["ENERGY"]
        self.wavelengths = dat["WAVELENGTH"]
        self.num_grid_points = int(header["NUM_GRID_POINTS"])
        self.line_width = float(header["LINE_VELOCITY_WIDTH"])
        self.spectrum_path = header["SPECTRUM_PATH"]
        self.search_mode = header["SEARCH_MODE"]

        # Set output parameters
        self.window_size = float(header["WINDOW_SIZE"])
        self.significance = dat["SIGNIFICANCE"]
        self.norm_pm = dat["NORM_SIGN"]
        self.delta_fitstat = dat["DELTA_FITSTAT"]
        self.initial_continuum = dat["INITIAL_CONTINUUM"]
        self.optimized_continuum = dat["OPTIMIZED_CONTINUUM"]
        self.const_factor = dat["CONSTANT_FACTOR"]
        self.fit_err = dat["FIT_ERR"]
        self.runtime = float(header["RUNTIME"])
        self.fit_stat = header["XSPEC_FITSTAT"]
        self.output_path = output_path
        self.fit_model = header["XSPEC_FIT_MODEL"]
        self.date = header["SEARCH_DATE"]

        # Additonal parameters
        self.obs_id = header["OBS_ID"]
        self.obs_mode = header["OBS_MODE"]
        self.grating = header["GRATING"]
        self.telescope = header["TELESCOPE"]

        # Allow user to specify object name manually
        if object_name is not None:
            self.object_name = object_name
        else:
            self.object_name = header["OBJECT_NAME"]

        print("Linesearch loaded")
        print("\t Object name: {0:s}".format(self.object_name))
        print("\t Output path: {0:s}".format(self.output_path))
        print("")

    def calc_line(self, line_center, line_region):
        """
        Calculates line significance and shift for a given line center and region

        Parameters
        line_center (float): Line center in keV
        line_region (list): Region around line center in keV

        Returns
        line_significance (float): Line significance
        line_shift (float): Line shift in km/s
        fit_error_in_region (bool): True if fit error occured in selected region
        """

        # Select significances around line center
        significance_line_region = self.significance[(self.energies > line_region[0]) & (self.energies < line_region[1])]
        energies_line_region = self.energies[(self.energies > line_region[0]) & (self.energies < line_region[1])]
        fit_errors_in_region = np.sum(self.fit_err[(self.energies > line_region[0]) & (self.energies < line_region[1])])

        # Calculate significance
        line_significance = np.min(significance_line_region)

        # Calculate line shift
        line_energy = energies_line_region[np.argmin(significance_line_region)]
        line_shift = (line_energy - line_center)/line_center * (c/1000)

        # Check if fit error occured in selected region
        if fit_errors_in_region > 0:
            print('Warning: somewhere in the selected energy range a fit error has occured: ' + self.output_path)
        
        return line_significance, line_shift, line_energy, fit_errors_in_region > 0
    
    def check():
        """
        Checks fit errors and constant factor.

        Returns
        validity (bool): True if fit errors and constant factor are valid
        """

        # Check if fit errors occured
        if np.sum(self.fit_err) > 0:
            print("Warning: fit errors occured")
            return False

        # Check if constant factor is valid
        if np.max(self.const_factor) > 2:
            print("Warning: constant factor is higher then 2")
            return False
        
        # Checks for unreal line significances
        if np.min(self.significance) < -50:
            print("Warning: line significance is lower then -50")
            return False
        
        # Checks for unreal line significances
        if np.min(self.significance) < -3:
            print("Note: line significance lower then -3")
        
        return True

    def __str__(self):
        return '{0:s}, {1:s}'.format(self.object_name, self.obs_id)
