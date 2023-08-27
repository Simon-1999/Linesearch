print("")
print("Linesearch python package")
print("* Documentation: https://github.com/Simon-1999/linesearch")
print("* Author: Simon van Eeden")
print("* Email: simonveeden@hotmail.com")

# Check if PyXspec is accessible
try:
    from xspec import *
except:
    print("Error: could not import PyXspec")
    print("\t 1. Make sure you have installed Xspec")
    print("\t 2. Make sure you have installed PyXspec in Python 2.7")
    print("\t 3. Make sure you have imported Xspec with headas-init")
    print("\t   or used the 'setup_chandra' command in the terminal")
    print("\t   (when using the linux computer at UVA science park)")
    print("")

    # Stop program
    quit()

print("* Using PyXspec version: " + Xset.version[0])
print("* Using Xspec version: " + Xset.version[1])
print("")

# Import grid module
from grid import *