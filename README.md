# Overview
TODO

# Functions
## splitDoubleDicom.py
- Function to split bruker dicoms by double sampling, coil, and real/imaginary
- should only be needed when cfmm custom export splitting double sampling into two volumes is used, which causes dcm2niix to fail.
- usage: `python splitDoubleDicom.py \path\to\directory\with\input\dicoms \path\to\output\directory`
