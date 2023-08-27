# _plots.py
#
# This file contains plotting function for the Grid class 
# 
# Author: Simon van Eeden

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.constants import c

# Import internal helper functions for PyXspec
from _spectrum import *

# A dictionary with line center lab energies, structure:
# line label (string) and line center in keV (float)
LINES = {
    'Fe XXVI': 6.9662,
    'Fe XXV': 6.70040,
}

# Sets automatic x-axis scaling to log for larger regions
XSCALE_TRESSHOLD = 3

def plot_significance(self, save_path=None, lines=LINES):
    """
    Plots the line significance of a linesearch.

    Parameters
    save_path (str): Path to save the plot to.
    lines (dict): A dictionary with restwavelenghts of lines to be plotted, the 
        default line list is defined on top of this file, a custom line
        dictionary can be provided where the key is the label (string) and 
        the value is the restwavelength (float).
    """

    # Make figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    plt.rc("font", family='serif', size=8)
    plt.grid(False)

    # Title
    ax1.set_title(self.object_name)

    # Significance C-stat and N/error
    ax1.plot(self.energies, np.sqrt(self.delta_fitstat)*self.norm_pm, '-', label=r"$\sqrt{\Delta C}$", alpha=0.9)
    ax1.plot(self.energies, self.significance, '-', label="$N/\sigma_N$", alpha=0.9)

    # Confidence region 1, 2, 3 sigma
    ax1.fill_between([self.energies[0], self.energies[-1]], [-3, -3], [3, 3], facecolor='gray', alpha=0.2)
    ax1.fill_between([self.energies[0], self.energies[-1]], [-2, -2], [2, 2], facecolor='gray', alpha=0.2)
    ax1.fill_between([self.energies[0], self.energies[-1]], [-1, -1], [1, 1], facecolor='gray', alpha=0.2)
    ax1.axhline(y = 0, color = 'k', linestyle = '-', lw=1)
    ax1.set_xlim(self.energies[0], self.energies[-1])

    # Automaticly select log scale for large energy ranges
    if self.energies[-1] - self.energies[0] > XSCALE_TRESSHOLD:
        ax1.set_xscale('log')

    # Plot lines with label
    for name, wavelength in lines.items():
        # Do not plot lines outside grid range
        if wavelength < self.energies[0] or wavelength > self.energies[-1]:
            continue

        ax1.axvline(x=wavelength, color='k', linestyle='--', linewidth=0.5)
        ax1.text(wavelength - 0.0001, -1, name, rotation=90, ha='right')

    # Reformat axis labels (for fancy log scale labels)
    ax1.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())

    ax1.legend()
    ax1.set_xlabel("Energy [keV]")
    ax1.set_ylabel("Line significance")

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()

def plot_search(self, save_path=None, lines=LINES):
    """
    Plots the output of a linesearch and the spectrum with the continuum model.

    Parameters
    save_path (str): Path to save the plot to.
    lines (dict): A dictionary with restwavelenghts of lines to be plotted, the 
        default line list is defined on top of this file, a custom line
        dictionary can be provided where the key is the label (string) and 
        the value is the restwavelength (float).
    """
    
    # Set default plotstyle
    plt.rc("font", family='serif', size=8)

    # Xspec settings
    Xset.chatter = 0 # Make Xspec console more quiet
    Plot.xAxis = "keV" # Set plot unit

    # Make figure
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(2, 2, (1,2))
    ax3 = fig.add_subplot(2, 2, 3, sharex = ax1)
    ax2 = fig.add_subplot(2, 2, 4)
    ax1.set_title("Line search %s" %(self.object_name))
    ax1.tick_params(which='both', axis='both', direction='in', top=True, right=True)
    ax3.tick_params(which='both', axis='both', direction='in', top=True, right=True)

    # Line significance
    ax1.plot(self.energies, self.significance, '-', label="$N/\sigma_N$")
    ax1.set_xlabel("Energy [keV]")
    ax1.set_ylabel("Line significance")
    ax1.set_xlim(self.energies[0], self.energies[-1])

    # Confidence region
    ax1.fill_between([self.energies[0], self.energies[-1]], [-3, -3], [3, 3], facecolor='gray', alpha=0.2)
    ax1.fill_between([self.energies[0], self.energies[-1]], [-2, -2], [2, 2], facecolor='gray', alpha=0.2)
    ax1.fill_between([self.energies[0], self.energies[-1]], [-1, -1], [1, 1], facecolor='gray', alpha=0.2)
    ax1.axhline(y = 0, color = 'k', linestyle = '-', lw=1)

    # Indicate grid points with fit errors
    if np.sum(self.fit_err) > 0:
        for energy, fit_err in zip(self.energies, self.fit_err):
            if fit_err == 1:
                ax1.plot(energy, 0, 'ro', ms=1.5)

        ax1.plot([], [], 'ro', ms=1.5, label="Xspec error")

    ax1.legend()

    # Automaticly select log scale for large energy ranges
    if self.energies[-1] - self.energies[0] > 3:
        ax1.set_xscale('log')

    # Plot lines with label
    for name, wavelength in lines.items():
        # Do not plot lines outside grid range
        if wavelength < self.energies[0] or wavelength > self.energies[-1]:
            continue

        ax1.axvline(x=wavelength, color='k', linestyle='--', linewidth=0.5)
        ax1.text(wavelength - 0.0001, -1, name, rotation=90, ha='right')

    # Load spectrum into Xspec
    s = load_spectrum(self.spectrum_path, [self.energies[0], self.energies[-1]]) 

    # Spectrum and background
    Plot.background = True
    Plot("ldata")
    ax3.plot(Plot.x(), Plot.y(), 'o', color='gray', ms=1, label="Spectrum")
    ax3.errorbar(Plot.x(), Plot.y(), yerr=Plot.yErr(), ecolor="gray", elinewidth=0.5, marker="", ls="")
    ax3.step(Plot.x(), Plot.backgroundVals(), '-', where="mid", lw=1, label="Background")

    # Initial and optimized continuum
    ax3.plot(self.energies, self.initial_continuum, '-', label="Intial continuum")
    ax3.plot(self.energies, self.optimized_continuum, '-', label="Optimized continuum")

    ax3.legend()
    ax3.set_xlabel('Energy [keV]')
    ax3.set_ylabel('Normalized counts')
    ax3.set_yscale('log')
    ax3.set_xlim(self.energies[0], self.energies[-1])

    # Automaticly select log scale for large energy ranges
    if self.energies[-1] - self.energies[0] > XSCALE_TRESSHOLD:
        ax3.set_xscale('log')   

    # Grid search info
    text_bottom = "# Line search details \n" + \
        "Runtime: {0:.0f} minutes and {1:.0f} seconds\n".format(self.runtime//60, self.runtime%60) + \
        "Search mode: {0:s}\n".format(self.search_mode) + \
        "Window size: {0:.1f}\n".format(self.window_size) + \
        "Line width: {0:.0f} km/s\n".format(self.line_width) + \
        "Grid stepsize: {0:.1f} ({1:d} grid points)\n".format(self.step_size, self.num_grid_points) + \
        "Number of Xspec fit errors: {0:d}\n".format(int(np.sum(self.fit_err))) + \
        "Search date: {0:s}\n".format(self.date) + \
        "\n# Observation details \n" + \
        "Obs-ID: {0:s}\n".format(self.obs_id) + \
        "Instrument: {0:s} - {1:s}\n".format(self.telescope, self.grating) + \
        "Mode: {0:s}\n".format(self.obs_mode)       
    ax2.text(0, 0, text_bottom, ha='left', fontsize='small')
    ax2.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()

def plot_gridpoint(self, e_center, save_path=None):
    """
    Plots sliding window fit at a certain energy.

    Parameters
    e_center (float): Energy of the grid point
    save_path (str): Path to save the plot to
    """

    if self.search_mode != "sliding window":
        print("ERROR: plot_gridpoint() can only be used with the sliding window search mode")
        return None

    # Make figure
    plt.rc("font", family='serif', size=8)
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    ax1.set_title("Sliding window fit {0:s} at {1:.2f}".format(self.object_name, e_center))
    ax1.tick_params(which='both', axis='both', direction='in', top=True, right=True)

    # Xspec settings
    Xset.chatter = 0 # Make Xspec console more quiet
    Xset.abund = "wilm" # Set solar abundance, latest version  
    Fit.nIterations =  1000 # Set maximum number of iterations when fitting
    Fit.query = "yes" # Always continue fitting
    Plot.xAxis = "keV" # Set plot unit
    
    # Load spectrum into Xspec
    if self.grating == "HEG":
        calib_region = HEG_REGION # see _spectrum.py
    if self.grating == "MEG":
        calib_region = MEG_REGION # see _spectrum.py
    s = load_spectrum(self.spectrum_path, calib_region)

    print("Fitting continuum model in Xspec...\n") 

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
    m.constant.factor.frozen = False
        
    # Set gaussian line center, normalization and line width
    m.gaussian.LineE = e_center
    m.gaussian.norm.frozen = False
    m.gaussian.norm = 0
    m.gaussian.Sigma = self.line_width*1000/c * e_center # keV

    # Calculate width of the window
    region_width = self.window_size * m.gaussian.Sigma.values[0] # keV

    # Select spectral region to fit
    AllData.notice("{0:.5f}-{1:.5f}".format(calib_region[0], calib_region[1]))
    ignore_below = e_center - region_width # keV
    ignore_above = e_center + region_width # keV
    AllData.ignore("{0:.5f}-** **-{1:.5f}".format(ignore_above, ignore_below))

    # Set continuum shift to default value
    m.constant.factor = 1   

    # Plot spectrum
    Plot("ldata")
    ax1.plot(Plot.x(), Plot.y(), 'o', color='gray', ms=1, label="Spectrum")
    ax1.errorbar(Plot.x(), Plot.y(), yerr=Plot.yErr(), ecolor="gray", elinewidth=0.5, marker="", ls="")

    # Plot intial continuum
    ax1.plot(Plot.x(), Plot.model(), '-', label="Initial continuum")

    # Fit normalization of the gaussian and the constant factor
    try:
        Fit.perform()

        # Show fit on XSPEC plot
        Plot("ldata")

        # Save fit values
        ax1.plot(Plot.x(), Plot.model(), '-', label="Continuum + gaussian")
    except:
        print("\nError: could not fit line in Xspec")

    # Line significance
    ax1.legend()
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Normalized counts")
    ax1.set_xlim(ignore_below, ignore_above)

    # Automaticly select log scale for large energy ranges
    if self.energies[-1] - self.energies[0] > XSCALE_TRESSHOLD:
        ax1.set_xscale('log')

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()

