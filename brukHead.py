import os
import argparse
import json
import pydicom
from pybruker import jcamp
import numpy as np
from scipy import fft

# Define function for expanding arrays with repeated values in methods json
def expandArray(vec_in):
    vec_out = []
    for item in vec_in:
        if not(isinstance(item,float)) and not(isinstance(item,int)):
            nval = item.split("*")
            nval = nval[0]
            nval = nval[1:]
            val = item.split("(")
            val = val[1]
            val = val[0:len(val)-1] 
            val = [float(val)] * int(nval)
            vec_out = vec_out + val
        else:
            vec_out = vec_out + [float(item)]
    return vec_out

def extract_info(fname, out_dir, do_write=True):
    # read dicom header. It can be huge, so instead of pydicom.dcmread(fname, stop_before_pixels=True), we set a custom stopping point
    #H = pydicom.dcmread(fname, stop_before_pixels=True)
    stop_tag = pydicom.tag.Tag(
        0x5200, 0x9230
    )  # Stop right at PerFrameFunctionalGroupsSequence, which can be huge
    with open(fname, "rb") as fp:
        H = pydicom.filereader.read_partial(
            fp, stop_when=lambda tag, vr, length: tag >= stop_tag
        )

    # Check total number of concatenated files
    num_concat = 1
    if hasattr(H,'InConcatenationTotalNumber'):
        num_concat = int(H.InConcatenationTotalNumber)

    # Set output filename
    fname_out = out_dir + os.sep + H.SeriesDescription

    #read bruker headers
    method=jcamp.jcamp_parse(H[0x0177,0x1100].value.decode('utf-8').splitlines())
    #visu_pars=jcamp.jcamp_parse(H[0x0177,0x1101].value.decode('utf-8').splitlines())

    # Check dimension of acquisition (i.e., 2D or 3D k-space)
    acqDim = expandArray(method["$PVM_Matrix"]["dim"])
    acqDim = round(acqDim[0])

    # Find phase encode axis
    pe_enc_dicom = H.SharedFunctionalGroupsSequence[0].MRFOVGeometrySequence[0].InPlanePhaseEncodingDirection
    if pe_enc_dicom.lower() == 'row':
        pe_axis = 0
    elif pe_enc_dicom.lower() == 'col' or pe_enc_dicom.lower() == 'column': 
        pe_axis = 1
    else:
        pe_axis = -1

    # Get some dimensions
    im_matrix_xyz_diff_echo_rep = expandArray(method["$PVM_Matrix"]["value"])
    if acqDim < 3:
        im_matrix_xyz_diff_echo_rep = im_matrix_xyz_diff_echo_rep + expandArray(method["$PVM_SPackArrNSlices"]["value"])
    if method["$PVM_NSPacks"] > 1:
        print("WARNING: brukHead found PVM_NSPacks>1. Downstream code may not account for this")
    im_matrix_xyz_diff_echo_rep = im_matrix_xyz_diff_echo_rep + [1,1,1]
    if "$PVM_DwNDiffDir" in method:
        im_matrix_xyz_diff_echo_rep[3] = method["$PVM_DwNDiffDir"]
    im_matrix_xyz_diff_echo_rep[4] = method["$PVM_NEchoImages"]
    im_matrix_xyz_diff_echo_rep[5] = method["$PVM_NRepetitions"]
    im_matrix_xyz_diff_echo_rep = [round(x) for x in im_matrix_xyz_diff_echo_rep]

    # Check for coil dimension
    hasCoilComplexDoubleDims = [False, False, False]
    if "DimensionIndexSequence" in H:
        for i, dim in enumerate(H.DimensionIndexSequence):
            desc = getattr(dim, "DimensionDescriptionLabel", "No Label")
            if desc.lower() == 'coil':
                hasCoilComplexDoubleDims[0] = True
            if desc.lower() == 'complex':
                hasCoilComplexDoubleDims[1] = True
            if desc.lower() == 'double samples':
                hasCoilComplexDoubleDims[2] = True
            #print(desc)

    # Read transmit coil name
    transmit_coil_name = "none"
    if "SharedFunctionalGroupsSequence" in H:
        if "MRTransmitCoilSequence" in H.SharedFunctionalGroupsSequence[0]:
            transmit_coil_name = H.SharedFunctionalGroupsSequence[0].MRTransmitCoilSequence[0].get("TransmitCoilName", "none")
    #print(transmit_coil_name)

    # Create output directory
    if do_write:
        os.makedirs(out_dir, exist_ok=True)

        # save method.json file
        with open(fname_out+'_method.json', 'w') as fp:
            json.dump(method,  fp, indent=4)

    #with open(fname_out+'_visu_pars.json', 'w') as fp:
    #    json.dump(visu_pars,  fp, indent=4)

    # Bvalue information 
    bval = []
    if "$PVM_DwEffBval" in method and do_write:
        bval = expandArray(method["$PVM_DwEffBval"]["value"])
        bval = np.array(bval)
        bvec = expandArray(method["$PVM_DwDir"]["value"])
        bvec = np.reshape(bvec, (int(len(bvec)/3+0.5), 3)).T
        bmat = expandArray(method["$PVM_DwBMat"]["value"])
        bmat = np.reshape(bmat, (int(len(bmat)/9+0.5), 9)).T

        if im_matrix_xyz_diff_echo_rep[5] > 1:
            # Replicate bvalues etc for repetitions
            bval = np.tile(bval,(im_matrix_xyz_diff_echo_rep[5]))
            bvec = np.tile(bvec,(1,im_matrix_xyz_diff_echo_rep[5]))
            bmat = np.tile(bmat,(1,im_matrix_xyz_diff_echo_rep[5]))

        with open(fname_out+'.bval', 'w') as fp:
            for item in bval:
                fp.write("%s " % item)
            fp.write("\n")

        with open(fname_out+'.bmat', 'w') as fp:
            for row in bmat:
                for item in row:
                    fp.write("%s " % item)
                fp.write("\n")
                
        with open(fname_out+'.bvec_in', 'w') as fp:
            for row in bvec:
                for item in row:
                    fp.write("%s " % item)
                fp.write("\n")

        # Determine bvec from bmat
        bmat = np.transpose(bmat)
        bvecFromMat = np.zeros((len(bval), 3))
        for idx, row in enumerate(bmat):
            row = np.reshape(row,(3,3))
            u, s, vh = np.linalg.svd(row, full_matrices=True)
            bvecFromMat[idx] = vh[0]

        # Get polarity of bvec correct based on input vectors (since polarity is arbitrary after svd)
        bvec = np.transpose(bvec)
        if len(bval) > len(bvec):
            # Fill in b0 acquisitions that were not in input dir vector
            for n in range(len(bval) - len(bvec)):
                idx = np.argsort(bval)
                bvec = np.insert(bvec,idx[0],0,axis=0)
        for idx, dir_n in enumerate(bvec):
            mind1 = np.argmax(np.abs(dir_n))
            mind2 = np.argmax(np.abs(bvecFromMat[idx]))
            fact = np.sign(dir_n[mind1]*bvecFromMat[idx][mind2])
            if fact < 0:
                bvecFromMat[idx] = fact*bvecFromMat[idx]

        bvecFromMat = np.transpose(bvecFromMat)
        with open(fname_out+'.bvec', 'w') as fp:
            for row in bvecFromMat:
                for item in row:
                    fp.write("%s " % item)
                fp.write("\n")

        # Export shape info, and the shapes themselves
        if "$DwShapesOn" in method and method["$DwNShapes"]>0:
            # Shapes per direction
            shape_per_dir= np.array(expandArray(method["$DwShapePerDir"]["value"]),dtype=int) - 1
            if im_matrix_xyz_diff_echo_rep[5] > 1:
                shape_per_dir = np.tile(shape_per_dir,(im_matrix_xyz_diff_echo_rep[5]))
            with open(fname_out+'.shapeperdir', 'w') as fp:
                for item in shape_per_dir:
                    fp.write("%s " % item)
                fp.write("\n")

            # The actual shapes
            shape_dt = ( np.array(expandArray(method["$DwShapeDurs"]["value"])) /
                np.array(expandArray(method["$DwShapePoints"]["value"])) )
            shape_gaps = ( np.array(expandArray(method["$DwShapeSeps"]["value"])) -
                np.array(expandArray(method["$DwShapeDurs"]["value"])) )
            Nzeros_gap = np.round(shape_gaps // shape_dt).astype(int)
            # for $DwGradShapeArray:
                # axis 0 is shape number
                # axis 1 is the diffusion gradient shape
                # asis 2 is 6 elements. First three are gx,gy,gz for before 180 pulse, while next three are gx,gy,gz for after
            shape_all = np.reshape( np.array(expandArray(method["$DwGradShapeArray"]["value"])),
                np.array(expandArray(method["$DwGradShapeArray"]["dim"]),dtype=int) )
            for ns in range(method["$DwNShapes"]):
                # write files, including a header
                # also build a unified shape that we can calculate q-spectra and centroids from
                shp_a = shape_all[ns,:,:].T
                shp_net = np.concat((-shp_a[0:3,:], np.zeros((3,Nzeros_gap[ns])), shp_a[3:6,:]),axis=1)
                q = np.cumsum(shp_net,axis=1)
                # Find centroids
                Q = fft.fft(q,axis=1)
                freq = fft.fftfreq(shp_net.shape[1], d=shape_dt[ns]/1000)
                Q = Q[:,0:Q.shape[1]//2]
                freq = freq[0:Q.shape[1]]
                Q[0] = Q[0]/2  # needed for accurate 1-sided fft.
                Q = np.abs(Q)**2 # spectral power
                # Find rank of b-matrix
                bm = np.zeros((3,3))
                for n0 in range(3):
                    for n1 in range(n0+1):
                        bm[n0,n1] = np.sum(q[n0,:]*q[n1,:])
                        bm[n1,n0] = bm[n0,n1]
                eigenvals = np.abs(np.linalg.eigvals(bm))
                eigenvals = eigenvals/eigenvals.max()
                brank = round(sum(eigenvals),2)
                with np.errstate(invalid='ignore'): # Don't throw nan warnings, which can happen for a gradient with all zeros
                    cents = np.sum(Q*freq[None,:],axis=1) / np.sum(Q,axis=1)
                with open(fname_out+f".shp{ns}", 'w') as fp:
                    fp.write(f"# DWI Gradient Shape {ns}\n")
                    fp.write("# the rows after this header correspond to:\n")
                    fp.write("#  temporal spacing between points in ms\n")
                    fp.write("#  dead time between gradient shapes before and after 180 RF pulse, in ms\n")
                    fp.write("#  q-spectra centroids for each of gx, gy, gz, in Hz. Will be nan if g is all zeros\n")
                    fp.write("#  rank of bmatrix\n")
                    fp.write("#  gradient amplitudes for gx of pre 180 shape\n")
                    fp.write("#  gradient amplitudes for gy of pre 180 shape\n")
                    fp.write("#  gradient amplitudes for gz of pre 180 shape\n")
                    fp.write("#  gradient amplitudes for gx of post 180 shape\n")
                    fp.write("#  gradient amplitudes for gx of post 180 shape\n")
                    fp.write("#  gradient amplitudes for gx of post 180 shape\n")
                    fp.write("# Note that the gradient amplitudes are normalized to max of 1; actual magnitudes can be calculated with help from .bval table and .shapeperdir file\n")
                    fp.write(f"{shape_dt[ns]} \n")
                    fp.write(f"{shape_gaps[ns]} \n")
                    for nax in range(3):
                        if np.isnan(cents[nax]):
                            fp.write("nan ")
                        else:
                            fp.write(f"{cents[nax]} ")
                    fp.write("\n")
                    fp.write(f"{brank} \n")
                    for row in shp_a:
                        for item in row:
                            fp.write("%s " % item)
                        fp.write("\n")

    # Return the series description and transmit info
    return H.SeriesDescription, hasCoilComplexDoubleDims, transmit_coil_name, acqDim, pe_axis, im_matrix_xyz_diff_echo_rep, num_concat, bval

def main():
    parser = argparse.ArgumentParser(description="Extract *_method.json, bmat, bval, and bvec information from bruker DICOM.")
    parser.add_argument("input_file", help="DICOM file (.dcm extension) containing the input DICOM files")
    parser.add_argument("output_dir", nargs="?", default=None, help="Directory to save the output to (defaults to <input_dir>/brukhead_out)")
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else f"{os.path.dirname(args.input_file)}/brukhead_out"
    extract_info(args.input_file, out_dir)


if __name__ == "__main__":
    main()