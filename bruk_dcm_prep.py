import os
import sys
import argparse
import subprocess
import nibabel as nib
import numpy as np
import re
import brukHead
import splitDoubleDicom
import rcvr_coil_combine

def main():
    parser = argparse.ArgumentParser(description="Convert bruker dcm to nii, accounting for custom baronlab sequences.")
    parser.add_argument("input_dir", help="Directory containing the input DICOM files for a single scan")
    parser.add_argument("output_dir", help="Directory to save the final files")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.output_dir

    # Automatically find and sort only .dcm files in the input directory
    input_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".dcm")
    ])
    if not input_files:
        print(f"No DICOM (.dcm) files found in {input_dir}")
        return

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Just use one file for .bval extraction, and to check if multiple coils are in the data file
    SeriesDescription, hasCoilComplexDoubleDims, transmit_coil_name, acqDim, im_matrix_xyz_diff_echo_rep, num_concat, bval = brukHead.extract_info(input_files[0],out_dir)
    if num_concat != len(input_files):
        print(f"Error: number of files does not match file concatenation identified in header of file {input_files[0]}", file=sys.stderr)
        sys.exit(1)

    # Create intermediate dicoms. Needed for double sampling. 
    # It may be needed for files that are split across two dicoms, but that has not been encountered yet in the absence of double sampling, so we split to be safe there
    if not (any(hasCoilComplexDoubleDims) or num_concat>1):
        # Nothing custom to do without complex data
        print("\nConverting to NIFTI...")
        command = ['dcm2niix', '-f', '%d', '-o', out_dir, input_dir]
        subprocess.run(command, capture_output=True, text=True, check=True)
        return

    # TODO: newer versions of dcm2niix collapse slices and DWI, seeminly leaving cycles intact. 
    # To make matters worse, when complex it incorrectly indexes real/imaginary, because in my test file I'm getting twice as many volumes in the _real file compared to _imaginary
      # So, I think I need to use splitDoubleDicom even if it is just complex, then uncollapse slices and DWI myself.
      # It seems that splitDoubleDicom correctly separates real/imag in these cases, and the collapsing is still left.
      # So, I just need to get the proper size from brukhead or something. At this point I'm wondering if I should have brukhead output the whole header H.
    made_dcm_dir = False
    if any(hasCoilComplexDoubleDims) or num_concat>1:
        # Split apart double sampling, coils, and complex dims into separate dicoms
        dir_dcm2convert = f"{out_dir}{os.sep}tempDicom"
        os.makedirs(dir_dcm2convert, exist_ok=True)
        splitDoubleDicom.split_bruker_dicoms(input_dir,dir_dcm2convert) # TODO: uncomment
        made_dcm_dir = True

    # Run dcm2niix
    print("\nConverting to NIFTI...")
    dir_nifti_temp = f"{out_dir}{os.sep}tempNifti"
    os.makedirs(dir_nifti_temp, exist_ok=True)
    if made_dcm_dir:
        command = ['dcm2niix', '-f', '%d', '-d', '0', '-o', dir_nifti_temp, dir_dcm2convert]
    else:
        command = ['dcm2niix', '-f', '%d', '-d', '0', '-o', dir_nifti_temp, input_dir]
    subprocess.run(command, capture_output=True, text=True, check=True)  # TODO: uncomment

    # Check for quad coil, which has simpler coil combination
    combineCoils = 0
    if hasCoilComplexDoubleDims[0] and hasCoilComplexDoubleDims[1]:
        if transmit_coil_name == 'RF RES 650 1H 02': 
            print('Quad coil detected. Combining coils assuming DC phase difference...')
            combineCoils = 1
        else:
            print('No quad coil detected. Combining coils using sense1...')
            combineCoils = 2

    # Initialize image using header from a sample nifti
    # im dims: [x y z Ndiff Nrep Nds]. 
    nii_files = [ # get all nifti filenames
        f for f in os.listdir(dir_nifti_temp)
        if os.path.isfile(os.path.join(dir_nifti_temp, f)) and f.endswith('.nii')
    ]
    im_t = nib.load(dir_nifti_temp + os.sep + nii_files[0])
    num_dim3 = np.prod(np.array(im_t.shape)) / np.prod(np.array(im_matrix_xyz_diff_echo_rep[0:3])) / im_matrix_xyz_diff_echo_rep[5] 
    if np.abs(np.round(num_dim3) - num_dim3) > 0.01:
        print("ERROR: image matrix shape mismatch")
        sys.exit(1)
    num_dim3 = np.round(num_dim3).astype('int')
    new_shape = im_matrix_xyz_diff_echo_rep[0:3] + [num_dim3,] + [im_matrix_xyz_diff_echo_rep[5],] + [1+hasCoilComplexDoubleDims[2],]
    if hasCoilComplexDoubleDims[1]:
        new_dtype = np.complex64
    else:
        new_dtype = np.float32
    im = np.zeros(new_shape,dtype=new_dtype)

    # Combine receivers 
    ds_all = [0]
    if hasCoilComplexDoubleDims[2]:
        ds_all = [0, 1]
    for dsi in ds_all:
        im_a = load_nii_files(dir_nifti_temp,nii_files,hasCoilComplexDoubleDims,im.shape,dsi)
        if combineCoils > 0:
            im_a = rcvr_coil_combine.combine_coils(
                im_a, combineCoils, acqDim) 
        elif im_a.ndim > 5 and im_a.shape[5] == 1:
            im_a = im_a[:,:,:,:,:,0]
        else:
            print("ERROR: unexpected image size")
            sys.exit(1)
        if im_a.ndim < 5:
            im_a = np.expand_dims(im_a,tuple(range(im_a.ndim,5)))
        im[:,:,:,:,:,dsi] = im_a
        del im_a

    # Combine double sampling
    if hasCoilComplexDoubleDims[2]:
        print('Combining double sampling using SENSE1...')
        im = rcvr_coil_combine.combine_coils(im, 3, acqDim)
    else:
        im = im[:,:,:,:,:,0]
    
    # Perform the following:
    #    Assess signal drift
    #    Align k-space center locations across all volumes (if complex)
    #    Assess and correct shifts caused by drift (if complex)
    # TODO: should keep all repetitions, and replicate the bval arrays. Do the replication in brukhead.py, which does the writing anyway
    im = rcvr_coil_combine.process_repetitions(im,acqDim,bval)
    # TODO: reshape im so that all reps are in dim 4. Order all be {all dwi rep 1}, {all dwi rep 2}, ...

    # TODO: need to save QA figures

    # Remove singleton dimensions from end
    if im.shape[4] == 1:
        im = im[:,:,:,:,0]
    if im.shape[3] == 1:
        im = im[:,:,:,0]

    # Save the final nifti
    print("Saving final nifti...")
    if hasCoilComplexDoubleDims[1]:
        im_p = np.angle(im)
        im = np.abs(im)
    im_out_a = nib.Nifti1Image(im, affine=im_t.affine, header=im_t.header)
    im_out_a.set_data_dtype(im.dtype)
    im_out_a.update_header()
    nib.save(im_out_a, out_dir + os.sep + SeriesDescription + ".nii.gz")
    if hasCoilComplexDoubleDims[1]:
        im_out_p = nib.Nifti1Image(im_p, affine=im_t.affine, header=im_t.header)
        im_out_p.set_data_dtype(im.dtype)
        im_out_p.update_header()
        nib.save(im_out_p, out_dir + os.sep + SeriesDescription + "_phase.nii.gz")

    # TODO: need to copy over .json from dcm2niix

    # Remove temp directories
    command = ['rm', '-r', dir_dcm2convert]
    subprocess.run(command, capture_output=True, text=True, check=True)
    command = ['rm', '-r', dir_nifti_temp]
    subprocess.run(command, capture_output=True, text=True, check=True)

    return


def load_nii_files(nii_dir,nii_files,hasCoilComplexDoubleDims,im_shape,dsi):    
    # Find number of coils
    if hasCoilComplexDoubleDims[0]:
        coil_matches = [re.search(r'coil(\d+)', f, re.IGNORECASE) for f in nii_files]
        coil_nums = sorted(list({int(m.group(1)) for m in coil_matches if m}))
        Ncoil = len(coil_nums)
    else:
        Ncoil = 1
    
    # Initialize data array
    # dims: [x y z Ndiff Nrep Ncoil]
    new_shape = im_shape[0:5] + (Ncoil,)
    if hasCoilComplexDoubleDims[1]:
        new_dtype = 'complex64'
    else:
        new_dtype = 'float32'
    im = np.zeros(new_shape,dtype=new_dtype)

    # Loop through coils
    for i_coil in range(0,Ncoil):
        # Find real part for this coil and ds
        matches = nii_files
        if hasCoilComplexDoubleDims[2]:
            matches = [item for item in matches if f'ds{dsi+1}' in item]  # accounts for 1 based indexing in filenames
        if hasCoilComplexDoubleDims[0]:
            matches = [item for item in matches if f'coil{i_coil+1}' in item] # accounts for 1 based indexing in filenames
        if hasCoilComplexDoubleDims[1]:
            fname_r = next(item for item in matches if 'real' in item)
            fname_i = next(item for item in matches if 'imaginary' in item)
            nii_r = nib.load(nii_dir + os.sep + fname_r)
            nii_i = nib.load(nii_dir + os.sep + fname_i)
            im_a = nii_r.get_fdata(dtype=np.float32) + 1j*nii_i.get_fdata(dtype=np.float32)
            if im_a.shape[2] == im.shape[2]*im.shape[3]:
                # Accounts for dcm2niix collapsing slices and dwi
                im_a = np.reshape(im_a, im.shape[0:2] + (im.shape[3],) + (im.shape[2],) + (im.shape[4],))
                im_a = im_a.swapaxes(2,3)
            im[:,:,:,:,:,i_coil] = np.reshape(im_a, im.shape[0:5])
            del im_a
        else:
            fname_r = matches[0]
            nii_r = nib.load(nii_dir + os.sep + fname_r)
            im[:,:,:,:,:,i_coil] = np.reshape(nii_r.get_fdata(dtype=np.float32), im.shape[0:5])

    return im    


if __name__ == "__main__":
    main()