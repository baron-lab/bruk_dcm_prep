# Overview
Python code to convert enhanced dicoms from bruker to nifti.
It can handle very large dicoms that are split across multiple files due to file size limits.
Also exports diffusion gradient shape files for baron lab custom dMRI sequence, dtiEpiFreeShapes
  - the shape files also list frequency centroids of q-spectra and b-matrix rank
For complex data, additional processing is performed:
  - coil combination if multiple coils exist in the data
  - combination of double sampling double samples have been saved separately. This is a custom recon that can be used for dtiEpiFreeShapes
  - centering of k=0 in k-space (may be helpful for subsequent complex denoising algorithms). Pretty benign change to images.
  - characterization of signal magnitude drift if multiple repetitions exist, or across b0's for dwi
  - characterization and correction of signal phase drift for EPI (manifesting as image shifts along PE) if multiple repetitions exist, or across b0's for dwi

# Setup
- Install required libraries via pixi.toml that is included in this repository. 
  - for module package management on your system, may need to use `module load pixi`
  - then use `pixi install` within the main directory that contains the pixi.toml file.
- dcm2niix must be installed

# Usage:
`pixi run python /path/to/bruk_dcm_prep.py /path/to/directory/containing/dicoms /path/to/directory/for/nifti/output`
  - note that if the output directory does not exit, it will be created

# Other files in repository, which are needed by the main script bruk_dcm_prep.py
You should rarely need to call these independently, but some have been written to be able to be called from command line.

## brukHead.py
- parses enhanced dicom header and saves dMRI related files (.bval, shapes for dtiEpiFreeShapes sequence)
- usage: `pixi run python /path/to/dicom.dcm /path/to/output/directory`

## splitDoubleDicom.py
- Function to split bruker dicoms by double sampling, coil, and real/imaginary
- should only be needed when cfmm custom export splitting double sampling into two volumes is used, which causes dcm2niix to fail.
- usage: `pixi run python /path/to/splitDoubleDicom.py /path/to/directory/with/input/dicoms /path/to/output/directory`

## rcvr_coil_combine.py
- library of functions for processing complex data
