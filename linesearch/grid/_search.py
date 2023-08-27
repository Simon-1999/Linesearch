# _search.py
#
# This file contains functions to run a line search.
#
# Author: Simon van Eeden

from __future__ import print_function
import time
import numpy as np
from xspec import *
from scipy.constants import c
import sys

# Import internal helper functions for PyXspec
from _spectrum import *

def run_search(self, spectrum_path, output_dir, search_mode="sliding window", \
               line_width=500, window_size=30, plot_fits=False):
    """
    Run a linesearch with a sliding window and save the output in a text file.

    Parameters
    spectrum_path (str): Path to the PHA2 spectrum file. RMF and ARF files must be
        located in the same folder
    output_dir (str): Path to the folder where the output text file will be saved
    search_mode (str): 'sliding window' or 'freeze continuum', defaults to 'sliding window'
    line_width (float): Width of the line in km/s, defaults to 500km/s
    window_size (float): Determines the size of sliding window, fraction of the line width
    plot_fits (bool): If True, opens Xspec and display the fit at each grid point,
        defaults to False

    Returns
    output_path (str): Path to the output text file.
    """

    self.spectrum_path = spectrum_path
    self.line_width = line_width
    self.search_mode = search_mode
    self.window_size = window_size

    ###################################
    ### --- Fit continuum model --- ###
    ###################################
    
    # Xspec settings
    Xset.chatter = 0 # Make Xspec console more quiet
    Xset.abund = "wilm" # Set solar abundance, latest version  
    Fit.nIterations =  1000 # Set maximum number of iterations when fitting
    Fit.query = "yes" # Always continue fitting
    Plot.xAxis = "keV" # Set plot unit

    # Opens Xspec plotting window 
    if plot_fits:
        Plot.device = "/xs" 
    
    # Load spectrum into Xspec
    self.grating = "HEG"
    if self.grating == "HEG":
        calib_region = HEG_REGION # see _spectrum.py
    if self.grating == "MEG":
        calib_region = MEG_REGION # see _spectrum.py
    s = load_spectrum(spectrum_path, calib_region) 

    print("Fitting continuum model in Xspec...")
    print("")

    # Make model in Xspec
    fit_model = "tbabs*(bbodyrad + pow)*constant + gauss"
    m = Model(fit_model)

    # Set initial values
    m.TBabs.nH = 0.3
    m.bbodyrad.kT = 1
    m.bbodyrad.norm = 10
    m.powerlaw.PhoIndex = 2
    m.powerlaw.norm = 1 
    m.constant.factor = 1
    m.gaussian.LineE = 0
    m.gaussian.Sigma = 0

    # Allow normalization to be positive and negative by changing the hard
    # and soft fit limits: value, delta, hard lo, soft lo, soft hi, hard hi
    m.gaussian.norm.values = "0.0,0.01,-1e2,-1e1,1e1,1e2" 

    # Freeze gauss and constant factor
    m.gaussian.LineE.frozen = True
    m.gaussian.norm.frozen = True
    m.gaussian.Sigma.frozen = True
    m.constant.factor.frozen = True

    # Stop if continuum model could not be fitted
    try:
        # Fit continuum with Chi squared
        Fit.statMethod = 'chi'
        Fit.renorm()
        Fit.perform()

        # Fit continuum with cstat
        Fit.statMethod = 'cstat'
        Fit.perform()

        # Plot fit in Xspec
        Plot("ld ratio")

    except:
        print("\nError: could not fit continuum model in Xspec")
        print("")
    
        return None
    
    # Freeze continuum
    m.TBabs.nH.frozen = True
    m.bbodyrad.kT.frozen = True
    m.bbodyrad.norm.frozen = True
    m.powerlaw.PhoIndex.frozen = True
    m.powerlaw.norm.frozen = True

    # Allow continuum shifts when search mode is set to sliding window
    if search_mode == 'sliding window':
        m.constant.factor.frozen = False
    
    # Do not allow continuum shifts when search mode is set to 
    # freeze continuum
    if search_mode == 'freeze continuum':
        m.constant.factor.frozen = True
    
    ##########################################
    ### --- Perform linesearch on grid --- ###
    ##########################################

    print("\nRunning linesearch...")
    print("\t Search mode: \t\t {0:s}".format(search_mode))
    print("\t Line width: \t\t {0:.0f} km/s".format(line_width))
    print("\t Window size: \t\t {0:.0f} (fraction of line width)".format(window_size))

    # Calculate estimated runtim
    estimated_runtime = self.num_grid_points*0.20 # seconds
    print("\t Estimated runtime: \t {0:.0f} minutes and {1:.0f} seconds".format(estimated_runtime//60, estimated_runtime%60))

    # Start timer
    start_time = time.time()

    # Initialize arrays for saving output
    self.significance, self.delta_fitstat = \
        np.zeros(self.num_grid_points), np.zeros(self.num_grid_points)
    self.norm, self.norm_err_lo, self.norm_err_hi = \
        np.zeros(self.num_grid_points), np.zeros(self.num_grid_points), \
        np.zeros(self.num_grid_points)
    self.norm_pm = np.zeros(self.num_grid_points)
    self.const_factor, self.const_factor_err_lo, self.const_factor_err_hi = \
        np.zeros(self.num_grid_points), np.zeros(self.num_grid_points), \
        np.zeros(self.num_grid_points)
    self.fit_err = np.zeros(self.num_grid_points) 
    self.initial_continuum, self.optimized_continuum = \
        np.zeros(self.num_grid_points), np.zeros(self.num_grid_points)

    # Fix gauss on each energy grid point and fit normalization and constant
    # factor
    for i, e_center in enumerate(self.energies):
        # Set gaussian line center, normalization and line width
        m.gaussian.LineE = e_center
        m.gaussian.norm.frozen = False
        m.gaussian.norm = 0
        m.gaussian.Sigma = self.line_width*1000/c * e_center # keV

        # Calculate width of the window
        region_width = window_size * m.gaussian.Sigma.values[0] # keV

        # Select spectral region to fit
        AllData.notice("{0:.5f}-{1:.5f}".format(calib_region[0], calib_region[1]))
        ignore_below = e_center - region_width # keV
        ignore_above = e_center + region_width # keV
        AllData.ignore("{0:.5f}-** **-{1:.5f}".format(ignore_above, ignore_below))

        # Set continuum shift to default value
        m.constant.factor = 1   

        # Get continuum at line center
        Plot("ldata")
        continuum = Plot.model()

        # Get spectral index at line center
        energies_lo = np.array([bin[0] for bin in s.energies])
        energies_hi = np.array([bin[1] for bin in s.energies]) 
        energies_center = (energies_lo + energies_hi)/2
        idx_line_center = np.argmin(np.abs(energies_center - (e_center)))
        self.initial_continuum[i] = continuum[idx_line_center]

        # Fit normalization of the gaussian and the constant factor
        try:
            Fit.perform()

            # Show fit on XSPEC plot
            if plot_fits:
                Plot("ldata")
        
            try:
                # Calculate 1-sigma error on normalization and constant factor,
                # which are the 9th and 6th parameter of the model
                Fit.error("nonew 1. 9 6")

                # Save fit values
                self.norm[i] = m.gaussian.norm.values[0] # normalization fit value
                self.norm_err_lo[i] = m.gaussian.norm.error[0] # lower bound error
                self.norm_err_hi[i] = m.gaussian.norm.error[1] # upper bound error
                self.const_factor[i] = m.constant.factor.values[0] # constant fit value
                self.const_factor_err_lo[i] = m.constant.factor.error[0] # lower bound error
                self.const_factor_err_hi[i] = m.constant.factor.error[1] # upper bound error

                # Calculate line significance for absorption lines (normalization < 0)
                # and for emission lines seperately (normalization > 0)
                if self.norm[i] < 0:
                    self.significance[i] = self.norm[i]/(self.norm[i] - self.norm_err_lo[i])
                    self.norm_pm[i] = -1
                else:
                    self.significance[i] = self.norm[i]/(self.norm_err_hi[i] - self.norm[i])
                    self.norm_pm[i] = 1

                # Get fitstatistic of fit with continuum and gauss
                fitstat_line = Fit.statistic  

                # Get continuum model without gauss
                m.gaussian.norm = 0   

                # Save optimized continuum at line center
                Plot("ldata")
                continuum = Plot.model()
                self.optimized_continuum[i] = continuum[idx_line_center]  

                # Save delta fitstat
                fitstat_cont = Fit.statistic
                self.delta_fitstat[i] = fitstat_cont - fitstat_line 

            except:  
                # When an Xspec error occured during error calculation, only save
                # fit values of normalization and constant factor
                self.significance[i] = 0
                self.norm_pm[i] = 0
                self.delta_fitstat[i] = 0
                self.norm[i] = m.gaussian.norm.values[0] # fit value
                self.norm_err_lo[i] = 0 # lower bound error
                self.norm_err_hi[i] = 0 # upper bound error
                self.const_factor[i] = m.constant.factor.values[0] # fit value
                self.const_factor_err_lo[i] = 0 # lower bound error
                self.const_factor_err_hi[i] = 0 # upper bound error

                # Xspec error occured, set to one
                self.fit_err[i] = 1

        except:
            # When an Xspec error occured during the fit of the normalization and
            # the constant factor only save zeros

            self.significance[i] = 0
            self.norm_pm[i] = 0
            self.delta_fitstat[i] = 0 
            self.norm[i] = 0 # fit value
            self.norm_err_lo[i] = 0 # lower bound error
            self.norm_err_hi[i] = 0 # upper bound error
            self.const_factor[i] = 0 # fit value
            self.const_factor_err_lo[i] = 0 # lower bound error
            self.const_factor_err_hi[i] = 0 # upper bound error

            # Xspec error occured, set to one
            self.fit_err[i] = 1

        # Write estimated time on terminal
        if i > 1:
            runtime = time.time() - start_time
            amtDone = i/float(self.num_grid_points)

            print("\t Progress: at {0:.1f}keV, {1:.1f}% (estimated time left {2:.0f}s)  ".format(e_center, amtDone * 100, (runtime/amtDone - runtime)), end="\r")
            sys.stdout.flush() # Clear terminal line

    print("\t Progress: 100%" + " "*80)

    # Calculate total runtime
    self.runtime = time.time() - start_time

    ###########################################
    ### --- Save linesearch in textfile --- ###
    ###########################################
   
    self.date = time.ctime(start_time)

    # Get spectrum information
    self.object_name = s.fileinfo('OBJECT').strip()
    self.obs_id = s.fileinfo('OBS_ID').strip()
    self.obs_mode = s.fileinfo('DATAMODE').strip()
    self.telescope = s.fileinfo('TELESCOP').strip()

    # Make filename
    fname = "{0:s}_{1:s}_{2:.0f}_{3:.2f}_{4:s}.txt".format(self.object_name, \
        self.obs_id, self.line_width, self.step_size, self.search_mode)
    savepath = "{0:s}/{1:s}".format(output_dir, fname)
    self.output_path = savepath

    # Convert data into table
    table = np.column_stack([self.energies, self.wavelengths, self.norm, self.norm_pm, \
        self.norm_err_lo, self.norm_err_hi, self.delta_fitstat, self.const_factor, \
        self.const_factor_err_lo, self.const_factor_err_hi, self.fit_err, self.significance, \
        self.initial_continuum, self.optimized_continuum])

    # Set columns names of table
    column_names = "ENERGY,WAVELENGTH,NORM,NORM_SIGN,NORM_ERR_LO," + \
        "NORM_ERR_HI,DELTA_FITSTAT,CONSTANT_FACTOR,CONSTANT_FACTOR_ERR_LO," + \
        "CONSTANT_FACTOR_ERR_HI,FIT_ERR,SIGNIFICANCE,INITIAL_CONTINUUM,OPTIMIZED_CONTINUUM"
    
    # Create header with grid search and observation details
    header = "OBJECT_NAME={0:s}\n".format(self.object_name) + \
        "OBS_ID={0:s}\n".format(self.obs_id) + \
        "OBS_MODE={0:s}\n".format(self.obs_mode) + \
        "TELESCOPE={0:s}\n".format(self.telescope) + \
        "GRATING={0:s}\n".format(self.grating) + \
        "SPECTRUM_PATH={0:s}\n".format(self.spectrum_path) + \
        "GRID_START={0:.2f}\n".format(self.start) + \
        "GRID_END={0:.2f}\n".format(self.end) + \
        "GRID_STEP_SIZE={0:.2f}\n".format(self.step_size) + \
        "NUM_GRID_POINTS={0:d}\n".format(self.num_grid_points) + \
        "LINE_VELOCITY_WIDTH={0:.0f}\n".format(self.line_width) + \
        "SEARCH_MODE={0:s}\n".format(self.search_mode) + \
        "WINDOW_SIZE={0:.2f}\n".format(self.window_size) + \
        "RUNTIME={0:.0f}\n".format(self.runtime) + \
        "XSPEC_FITSTAT={0:s}\n".format("cstat") + \
        "XSPEC_FIT_MODEL={0:s}\n".format(fit_model) + \
        "XSPEC_ABUN={0:s}\n".format(Xset.abund) + \
        "XSPEC_XSECT={0:s}\n".format(Xset.xsect) + \
        "XSPEC_VERSION={0:s}\n".format(Xset.version[1]) + \
        "PYXSPEC_VERSION={0:s}\n".format(Xset.version[0]) + \
        "SEARCH_DATE={0:s}\n".format(self.date) + \
        "COLUMN_NAMES={0:s}".format(column_names)

    # Save linesearch in textfile
    np.savetxt(self.output_path, table, header=header)

    print("")
    print("Linesearch finished")
    print("\t Total runtime: {0:.0f}s".format(self.runtime))
    print("\t Number of Xspec fit errors: {0:d}".format(int(np.sum(self.fit_err))))
    print("\t Saved output in textfile: " + savepath)
    print("")

    return self.output_path

def run_custom_search(self, spectrum_path, output_dir):
    """
    Run a linesearch with a custom frozen continuum and save the output in a text file.

    Parameters:
    spectrum_path (str): Path to the PHA2 spectrum file. RMF and ARF files must be
        located in the same folder.
    output_dir (str): Path to the folder where the output text file will be saved.

    Returns:
    output_path (str): Path to the output text file.
    """

    self.search_mode = "custom"
    self.line_width = None

    # Global Xspec settings
    Xset.chatter = 0 # Make Xspec console more quiet
    Xset.abund = "wilm" # Set solar abundance, latest version  
    Fit.nIterations =  1000 # Set maximum number of iterations when fitting
    Fit.query = "yes" # Always continue fitting
    Plot.xAxis = "keV" # Set plot unit
   
    # Load spectrum into Xspec
    self.instrument = "HEG" 
    if self.grating == "HEG":
        calib_region = HEG_REGION # see _spectrum.py
    if self.grating == "MEG":
        calib_region = MEG_REGION # see _spectrum.py
    s = load_spectrum(spectrum_path, calib_region) 

    ##########################################
    ### --- Perform linesearch on grid --- ###
    ##########################################

    print("Running linesearch...")
    print("")

    # Start timer
    start_time = time.time()

    # Required arrays (for plotting functions)
    self.initial_continuum, self.optimized_continuum = \
        np.zeros(self.num_grid_points), np.zeros(self.num_grid_points)
    self.significance, self.delta_fitstat = \
        np.zeros(self.num_grid_points), np.zeros(self.num_grid_points)
    
    # ADD YOUR CODE HERE

    # Loop over grid points
    for i, e_center in enumerate(self.energies):

        # ADD YOUR CODE HERE
        continue

    # Calculate total runtime
    self.runtime = time.time() - start_time

    ###########################################
    ### --- Save linesearch in textfile --- ###
    ###########################################
   
    self.date = time.ctime(start_time)

    # Get spectrum information
    self.object_name = s.fileinfo('OBJECT').strip()
    self.obs_id = s.fileinfo('OBS_ID').strip()
    self.obs_mode = s.fileinfo('DATAMODE').strip()
    self.telescope = s.fileinfo('TELESCOP').strip()

    # Make filename
    self.output_path = "{0:s}_{1:s}_{2:.0f}_{3:.2f}_{4:s}.txt".format(self.object_name, \
        self.obs_id, self.line_width, self.step_size, self.search_mode)
    savepath = "{0:s}/{1:s}".format(output_dir, self.output_path)

    # Convert data into table
    table = np.column_stack([self.energies, self.wavelengths, self.significance])

    # Set columns names of table
    column_names = "ENERGY,WAVELENGTH,SIGNIFICANCE"
    
    # Create header with grid search and observation details
    header = "OBJECT_NAME={0:s}\n".format(self.object_name) + \
        "OBS_ID={0:s}\n".format(self.obs_id) + \
        "OBS_MODE={0:s}\n".format(self.obs_mode) + \
        "TELESCOPE={0:s}\n".format(self.telescope) + \
        "INSTRUMENT={0:s}\n".format(self.instrument) + \
        "SPECTRUM_PATH={0:s}\n".format(self.spectrum_path) + \
        "GRID_START={0:.2f}\n".format(self.start) + \
        "GRID_END={0:.2f}\n".format(self.end) + \
        "GRID_STEP_SIZE={0:.2f}\n".format(self.step_size) + \
        "NUM_GRID_POINTS={0:d}\n".format(self.num_grid_points) + \
        "LINE_VELOCITY_WIDTH={0:.0f}\n".format(self.line_width) + \
        "SEARCH_MODE={0:s}\n".format(self.search_mode) + \
        "RUNTIME={0:.0f}\n".format(self.runtime) + \
        "XSPEC_FITSTAT={0:s}\n".format("cstat") + \
        "XSPEC_ABUN={0:s}\n".format(Xset.abund) + \
        "XSPEC_XSECT={0:s}\n".format(Xset.xsect) + \
        "XSPEC_VERSION={0:s}\n".format(Xset.version[1]) + \
        "PYXSPEC_VERSION={0:s}\n".format(Xset.version[0]) + \
        "SEARCH_DATE={0:s}\n".format(self.date) + \
        "COLUMN_NAMES={0:s}".format(column_names)

    # Save linesearch in textfile
    np.savetxt(self.output_path, table, header=header)

    print("")
    print("Linesearch finished")
    print("\t Total runtime: {0:.0f}s".format(self.runtime))
    print("\t Saved output in textfile: " + savepath)
    print("")

    print("Custom search mode not yet implemented")
    quit()

    return self.output_path
