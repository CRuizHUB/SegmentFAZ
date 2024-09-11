# SegmentFAZ
OCT-A Scanning utility for FAZ segmentation and further analysis.

Instructions:

Move main.py to the folder where the .tif OCT-A scans are, as well as the binary masks in .JPG format are. (If error checkings are not desired, simply create black .jpg images with the same names as the .tif files).

The calibration file is meant to be run with a large set of files. Its output is very similar to 'main.py', but, for each image, the optimal set of parameters is shown. Requires to have valid manually-segmented binary masks to work from. The parameters used in 'main.py' are the average of a calibartion run of 177 images.
