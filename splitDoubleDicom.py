import os
import argparse
import copy
import numpy as np
import pydicom
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

# Function to split bruker dicoms by double sampling, coil, and real/imaginary
# Should only be needed when cfmm custom export splitting double sampling into two volumes is used, which causes dcm2niix to fail.


def get_pixel_data_offset(fp, is_implicit_vr, is_little_endian):
    """Finds the exact byte offset where raw PixelData values begin, Starting from fp.tell()."""
    endian = "<" if is_little_endian else ">"
    tag_bytes = fp.read(4)
    if len(tag_bytes) < 4:
        raise RuntimeError("Reached EOF before finding (7FE0, 0010) PixelData tag.")

    if is_implicit_vr:
        # Implicit VR: 4-byte tag + 4-byte length
        fp.seek(4, os.SEEK_CUR)
    else:
        # Explicit VR: 4-byte tag + 2-byte VR
        vr = fp.read(2)
        if vr in (b"OB", b"OW", b"OF", b"OD", b"SQ", b"UN", b"UC", b"UR", b"UT"):
            # 2-byte reserved + 4-byte length
            fp.seek(6, os.SEEK_CUR)
        else:
            # 2-byte length
            fp.seek(2, os.SEEK_CUR)

    return fp.tell()


def index_bruker_file(filepath, file_idx):
    """Indexes all frames in a Bruker Enhanced DICOM using pydicom object parsing."""
    with open(filepath, "rb") as fp:
        # Read all metadata up to (7FE0, 0010) PixelData
        ds_header = pydicom.dcmread(fp, stop_before_pixels=True)

        is_implicit_vr = ds_header.file_meta.TransferSyntaxUID.is_implicit_VR
        is_little_endian = ds_header.file_meta.TransferSyntaxUID.is_little_endian
        pixel_data_offset = get_pixel_data_offset(fp, is_implicit_vr, is_little_endian)

    n_frames = int(getattr(ds_header, "NumberOfFrames", 1))
    rows = int(ds_header.Rows)
    cols = int(ds_header.Columns)
    samples_per_pixel = int(getattr(ds_header, "SamplesPerPixel", 1))
    bits_alloc = int(ds_header.BitsAllocated)
    bytes_per_frame = rows * cols * samples_per_pixel * (bits_alloc // 8)

    # Extract and remove the PerFrameFunctionalGroupsSequence from the base header
    # so we can cheaply copy ds_header later without duplicating thousands of frame items
    per_frame_seq = ds_header.PerFrameFunctionalGroupsSequence
    del ds_header.PerFrameFunctionalGroupsSequence

    if len(per_frame_seq) != n_frames:
        raise RuntimeError(
            f"Expected {n_frames} frame items in {filepath}, found {len(per_frame_seq)}"
        )

    frames_meta = []
    for f_i, frame_item in enumerate(per_frame_seq):
        # 1. Extract DimensionIndexValues (0020, 9157) inside FrameContentSequence (0020, 9111)
        frame_content_seq = frame_item.FrameContentSequence[0]
        vals = [int(x) for x in frame_content_seq.DimensionIndexValues]
        # vals = [COMPLEX, COIL, diffusion, double_samples, SLICE_PACKAGE, SLICE]

        # 2. Extract ComplexImageComponent (0008, 9208) inside MRImageFrameTypeSequence (0018, 9226)
        comp_str = None
        if "MRImageFrameTypeSequence" in frame_item:
            mr_type_seq = frame_item.MRImageFrameTypeSequence[0]
            if "ComplexImageComponent" in mr_type_seq:
                comp_str = str(mr_type_seq.ComplexImageComponent).strip()

        if not comp_str:
            comp_str = "REAL" if vals[0] == 1 else "IMAGINARY"

        # 3. Normalize COMPLEX (vals[0]), COIL (vals[1]), and double_samples (vals[3]) to 1
        # so dcm2niix sees a clean 1-indexed array in every split file
        patched_vals = list(vals)
        patched_vals[0] = 1
        if len(patched_vals) > 1:
            patched_vals[1] = 1
        if len(patched_vals) > 3:
            patched_vals[3] = 1
        frame_content_seq.DimensionIndexValues = patched_vals

        frames_meta.append({
            "file_idx": file_idx,
            "frame_idx": f_i,
            "frame_item": frame_item,
            "div": tuple(vals),
            "comp_str": comp_str,
            "complex_idx": vals[0],
            "coil_idx": vals[1] if len(vals) > 1 else 1,
            "diff_idx": vals[2] if len(vals) > 2 else 1,
            "double_sample_idx": vals[3] if len(vals) > 3 else 1,
            "pkg_idx": vals[4] if len(vals) > 4 else 1,
            "slice_idx": vals[5] if len(vals) > 5 else 1,
        })

    return ds_header, frames_meta, pixel_data_offset, bytes_per_frame


def write_split_dicom(out_path, base_header, selected_frames, pixel_memmaps):
    """Writes a standalone Enhanced DICOM file using standard pydicom serialization."""
    ds_out = copy.deepcopy(base_header)
    n_out = len(selected_frames)
    ds_out.NumberOfFrames = n_out

    # Generate unique UIDs so dcm2niix treats each file as a distinct series
    ds_out.SeriesInstanceUID = generate_uid()
    ds_out.SOPInstanceUID = generate_uid()
    ds_out.file_meta.MediaStorageSOPInstanceUID = ds_out.SOPInstanceUID

    # Attach the subset of PerFrameFunctionalGroupsSequence items
    ds_out.PerFrameFunctionalGroupsSequence = Sequence(
        [f["frame_item"] for f in selected_frames]
    )

    # Assemble the contiguous PixelData byte payload from the memory maps
    ds_out.PixelData = b"".join(
        pixel_memmaps[f["file_idx"]][f["frame_idx"]].tobytes()
        for f in selected_frames
    )

    # Set VR for PixelData appropriately based on BitsAllocated
    ds_out["PixelData"].VR = "OW" if ds_out.BitsAllocated > 8 else "OB"

    pydicom.dcmwrite(out_path, ds_out, write_like_original=False)


def split_bruker_dicoms(input_dir, out_dir):
    """Splits Bruker Enhanced DICOMs by Complex, Coil, and Double Samples."""
    os.makedirs(out_dir, exist_ok=True)

    input_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".dcm")
    ])

    if not input_files:
        print(f"No DICOM (.dcm) files found in {input_dir}")
        return

    all_frames = []
    pixel_memmaps = []
    base_header = None

    for idx, fpath in enumerate(input_files):
        print(f"Indexing {os.path.basename(fpath)}...")
        ds_hdr, frames, pix_offset, bpf = index_bruker_file(fpath, idx)
        if idx == 0:
            base_header = ds_hdr
        all_frames.extend(frames)
        print(f"  -> Indexed {len(frames)} frames.")

        # Memory-map the pixel data directly from disk using the exact parsed offset and frame size
        n_f = len(frames)
        mmap = np.memmap(
            fpath,
            dtype=np.uint8,
            mode="r",
            offset=pix_offset,
            shape=(n_f, bpf),
        )
        pixel_memmaps.append(mmap)

    # Summarize unique dimension counts dynamically across however many dimensions exist
    dim_names = ["COMPLEX", "COIL", "diffusion", "double_samples", "SLICE_PKG", "SLICE"]
    num_dims = len(all_frames[0]["div"]) if all_frames else 0
    for dim_idx in range(num_dims):
        dim_name = dim_names[dim_idx] if dim_idx < len(dim_names) else f"DIM_{dim_idx}"
        unique_vals = sorted(list(set(f["div"][dim_idx] for f in all_frames)))
        print(
            f"Dimension {dim_idx} ({dim_name}): {len(unique_vals)} unique values -> "
            f"{unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}"
        )

    # Save original descriptions safely outside the loop
    orig_desc = getattr(base_header, "SeriesDescription", "dMRI")
    orig_num = int(getattr(base_header, "SeriesNumber", 1))
    orig_typelen = 0
    orig_imgtypelist = []
    if "ImageType" in base_header:
        orig_typelen = len(base_header.ImageType)
        orig_imgtypelist = list(base_header.ImageType)

    # Group frames by (comp_str, coil_idx, double_sample_idx)
    groups = {}
    for f in all_frames:
        key = (f["comp_str"], f["coil_idx"], f["double_sample_idx"])
        groups.setdefault(key, []).append(f)

    print(f"\nSplitting into {len(groups)} files...")
    for idx, ((comp_str, coil_idx, ds_idx), flist) in enumerate(
        sorted(groups.items())
    ):
        # Ensure frames are sorted cleanly by (diffusion, slice_pkg, slice)
        flist.sort(key=lambda f: (f["diff_idx"], f["pkg_idx"], f["slice_idx"]))

        out_name = f"dMRI_{comp_str.lower()}_coil{coil_idx}_ds{ds_idx}.dcm"
        out_path = os.path.join(out_dir, out_name)

        # Modify a copy of the base header (which excludes PerFrameFunctionalGroupsSequence)
        hdr_copy = copy.deepcopy(base_header)
        hdr_copy.SeriesDescription = (
            f"{orig_desc}_{comp_str[:4].lower()}_c{coil_idx}_ds{ds_idx}"
        )
        hdr_copy.SeriesNumber = orig_num * 100 + idx + 1

        # Set top-level ImageType[3] to REAL or IMAGINARY to help dcm2niix
        if "ImageType" in hdr_copy and orig_typelen >= 4:
            img_type = list(orig_imgtypelist)
            img_type[3] = comp_str
            hdr_copy.ImageType = img_type

        print(f"Writing {out_name} ({len(flist)} frames)...")
        write_split_dicom(out_path, hdr_copy, flist, pixel_memmaps)

    print(f"\nDone! You can now run dcm2niix on: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split Bruker Enhanced DICOMs by Complex, Coil, and Double Samples."
    )
    parser.add_argument(
        "input_dir", help="Directory containing the input DICOM files"
    )
    parser.add_argument(
        "output_dir", help="Directory to save the split DICOM files"
    )
    args = parser.parse_args()

    split_bruker_dicoms(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()