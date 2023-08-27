# example.py
#
# File to demenostrate how to use the package
#
# Author: Simon van Eeden, simonveeden@hotmail.com

import linesearch as ls

# Create grid object
g = ls.Grid()

# Run search
spec_file = "/home/simon/Downloads/tgcat/obs_3823_tgid_3851/pm_heg_abs1.pha"
output_dir = "/home/simon/thesis/output/example"
g.set_grid_points(6, 8, 500, 0.3)
output_file = g.run_search(spec_file, output_dir, search_mode="sliding window", \
               line_width=500, window_size=30)

# Plot search
g.load_search(output_file)
g.plot_search(save_path=output_file[:-4] + ".png")
g.plot_significance()

# Calculate line significance of Fe XXVI
line_center = 6.9961
line_region = [6.95, 7.4]
line_significance, line_shift, line_energy, error = g.calc_line(line_center, line_region)

# Plot fit at Fe XXVI line
g.plot_gridpoint(line_energy, save_path=output_file[:-4] + "_fe-xxvi.png")

# Calculate line significance of Fe XXV
line_center = 6.698
line_region = [6.67, 6.85]
line_significance, line_shift, line_energy, error = g.calc_line(line_center, line_region)

# Plot fit at Fe XXVI line
g.plot_gridpoint(line_energy, save_path=output_file[:-4] + "_fe-xxv.png")

