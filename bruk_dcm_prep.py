import os
import argparse
import brukHead
import splitDoubleDicom

def main():
    parser = argparse.ArgumentParser(description="Convert bruker dcm to nii, accounting for custom baronlab sequences.")
    parser.add_argument("input_dir", help="Directory containing the input DICOM files for a single scan")
    parser.add_argument("output_dir", help="Directory to save the final files")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

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
    SeriesDescription, hasCoilComplexDoubleDims, transmit_coil_name = brukHead.extract_info(input_files[0],out_dir)

    # Create a temp folder for intermediate files
    if any(hasCoilComplexDoubleDims):
        temp_dir = f"{input_dir}{os.sep}temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Split apart double sampling, coils, and complex dims into separate dicoms
        splitDoubleDicom.split_bruker_dicoms(input_dir,temp_dir)

        #if hasCoilComplexDoubleDims[0] and hasCoilComplexDoubleDims[1]:
            #if transmit_coil_name == 'RF RES 650 1H 02': # only combine coils here for quadrature coil, which has fairly trivial combination
                # TODO: combine coils
            #else:
                #print(f"Unknown coil, so skipping quadrature coil combination. coil = {transmit_coil_name}")

        # TODO: remove temp directory


if __name__ == "__main__":
    main()