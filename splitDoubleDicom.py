import os
import argparse
import copy
import numpy as np
import pydicom
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid


def get_pixel_data_offset(fp, is_implicit_vr, is_little_endian):
    """Finds the exact byte offset where raw PixelData values begin, starting from fp.tell()."""
    tag_bytes = fp.read(4)
    if len(tag_bytes) < 4:
        raise RuntimeError("Reached EOF before finding (7FE0, 0010) PixelData tag.")

    if is_implicit_vr:
        fp.seek(4, os.SEEK_CUR)
    else:
        vr = fp.read(2)
        if vr in (b"OB", b"OW", b"OF", b"OD", b"SQ", b"UN", b"UC", b"UR", b"UT"):
            fp.seek(6, os.SEEK_CUR)
        else:
            fp.seek(2, os.SEEK_CUR)

    return fp.tell()


def parse_dimension_organization(ds_header):
    """
    Parses DimensionIndexSequence (0020, 9222) to dynamically map each dimension index
    to a standardized semantic role.
    """
    dim_seq = getattr(ds_header, "DimensionIndexSequence", [])
    dim_map = []

    for i, item in enumerate(dim_seq):
        ptr = getattr(item, "DimensionIndexPointer", None)
        ptr_tuple = (ptr.group, ptr.element) if ptr is not None else (0, 0)
        label = str(getattr(item, "DimensionDescriptionLabel", "")).strip()
        label_up = label.upper()

        role = f"UNKNOWN_{i}"
        if ptr_tuple == (0x0008, 0x9208) or "COMPLEX" in label_up:
            role = "COMPLEX"
        elif ptr_tuple == (0x0018, 0x9045) or "COIL" in label_up:
            role = "COIL"
        elif ptr_tuple == (0x0018, 0x9117) or "DIFFUSION" in label_up:
            role = "DIFFUSION"
        elif ptr_tuple == (0x0020, 0x9056) or "PACKAGE" in label_up:
            role = "SLICE_PACKAGE"
        elif ptr_tuple == (0x0020, 0x9057) or label_up == "SLICE":
            role = "SLICE"
        elif ptr_tuple == (0x0020, 0x9453):
            # (0020, 9453) FrameLabel is used by Bruker for both 'double samples' and 'CYCLE'
            if "CYCLE" in label_up:
                role = "CYCLE"
            elif "DOUBLE" in label_up or "SAMPLE" in label_up:
                role = "DOUBLE_SAMPLES"
            else:
                role = label_up if label_up else "FRAME_LABEL"
        else:
            if "CYCLE" in label_up:
                role = "CYCLE"
            elif "DOUBLE" in label_up:
                role = "DOUBLE_SAMPLES"

        dim_map.append({
            "idx": i,
            "role": role,
            "label": label if label else role,
            "pointer": ptr_tuple,
        })

    return dim_map


def ensure_cycle_outer_loop_in_header(ds_header, dim_map):
    """
    Ensures that if both CYCLE and DIFFUSION exist in DimensionIndexSequence,
    CYCLE appears before DIFFUSION so dcm2niix always loops over CYCLE as the outer loop
    and DIFFUSION as the inner loop. Returns True if their indices were swapped.
    """
    role_to_idx = {d["role"]: d["idx"] for d in dim_map}
    if "CYCLE" in role_to_idx and "DIFFUSION" in role_to_idx:
        c_pos = role_to_idx["CYCLE"]
        d_pos = role_to_idx["DIFFUSION"]
        if c_pos > d_pos:
            # Swap in DimensionIndexSequence so CYCLE is evaluated before DIFFUSION
            dim_seq = ds_header.DimensionIndexSequence
            dim_seq[c_pos], dim_seq[d_pos] = dim_seq[d_pos], dim_seq[c_pos]
            return True, c_pos, d_pos
    return False, None, None


def index_bruker_file(filepath, file_idx):
    """Indexes all frames in a Bruker Enhanced DICOM using dynamic dimension mapping."""
    with open(filepath, "rb") as fp:
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

    # Parse dynamic dimension organization
    dim_map = parse_dimension_organization(ds_header)
    role_to_idx = {d["role"]: d["idx"] for d in dim_map}

    # Check if we need to swap CYCLE and DIFFUSION in DimensionIndexValues
    swapped, c_pos, d_pos = ensure_cycle_outer_loop_in_header(ds_header, dim_map)

    # Extract and remove PerFrameFunctionalGroupsSequence for fast header copying later
    per_frame_seq = ds_header.PerFrameFunctionalGroupsSequence
    del ds_header.PerFrameFunctionalGroupsSequence

    if len(per_frame_seq) != n_frames:
        raise RuntimeError(
            f"Expected {n_frames} frame items in {filepath}, found {len(per_frame_seq)}"
        )

    # Dimensions we split into separate DICOM files (and normalize to 1 inside each file)
    split_roles = {"COMPLEX", "COIL", "DOUBLE_SAMPLES"}

    frames_meta = []
    for f_i, frame_item in enumerate(per_frame_seq):
        frame_content_seq = frame_item.FrameContentSequence[0]
        div_elem = frame_content_seq.DimensionIndexValues
        vals = [div_elem] if isinstance(div_elem, int) else [int(x) for x in div_elem]

        def get_dim_val(role, default=1):
            idx = role_to_idx.get(role)
            return vals[idx] if (idx is not None and idx < len(vals)) else default

        complex_idx = get_dim_val("COMPLEX", None)
        coil_idx = get_dim_val("COIL", None)
        ds_idx = get_dim_val("DOUBLE_SAMPLES", None)
        cycle_idx = get_dim_val("CYCLE", 1)
        diff_idx = get_dim_val("DIFFUSION", 1)
        pkg_idx = get_dim_val("SLICE_PACKAGE", 1)
        slice_idx = get_dim_val("SLICE", 1)

        # Determine ComplexImageComponent (REAL / IMAGINARY / MAGNITUDE)
        comp_str = None
        if "MRImageFrameTypeSequence" in frame_item:
            mr_type_seq = frame_item.MRImageFrameTypeSequence[0]
            if "ComplexImageComponent" in mr_type_seq:
                comp_str = str(mr_type_seq.ComplexImageComponent).strip()

        if not comp_str and complex_idx is not None:
            comp_str = "REAL" if complex_idx == 1 else "IMAGINARY"

        # Normalize split dimensions (COMPLEX, COIL, DOUBLE_SAMPLES) to 1 in the frame metadata
        patched_vals = list(vals)
        for d in dim_map:
            if d["role"] in split_roles and d["idx"] < len(patched_vals):
                patched_vals[d["idx"]] = 1

        # If CYCLE appeared after DIFFUSION in the original header, swap their values too
        if swapped:
            patched_vals[c_pos], patched_vals[d_pos] = patched_vals[d_pos], patched_vals[c_pos]

        frame_content_seq.DimensionIndexValues = patched_vals

        frames_meta.append({
            "file_idx": file_idx,
            "frame_idx": f_i,
            "frame_item": frame_item,
            "div": tuple(vals),
            "comp_str": comp_str,
            "complex_idx": complex_idx,
            "coil_idx": coil_idx,
            "double_sample_idx": ds_idx,
            "cycle_idx": cycle_idx,
            "diff_idx": diff_idx,
            "pkg_idx": pkg_idx,
            "slice_idx": slice_idx,
        })

    return ds_header, frames_meta, dim_map, pixel_data_offset, bytes_per_frame


def write_split_dicom(out_path, base_header, selected_frames, pixel_memmaps):
    """Writes a standalone Enhanced DICOM file using standard pydicom serialization."""
    ds_out = copy.deepcopy(base_header)
    n_out = len(selected_frames)
    ds_out.NumberOfFrames = n_out

    ds_out.SeriesInstanceUID = generate_uid()
    ds_out.SOPInstanceUID = generate_uid()
    ds_out.file_meta.MediaStorageSOPInstanceUID = ds_out.SOPInstanceUID

    ds_out.PerFrameFunctionalGroupsSequence = Sequence(
        [f["frame_item"] for f in selected_frames]
    )

    ds_out.PixelData = b"".join(
        pixel_memmaps[f["file_idx"]][f["frame_idx"]].tobytes()
        for f in selected_frames
    )

    ds_out["PixelData"].VR = "OW" if ds_out.BitsAllocated > 8 else "OB"

    pydicom.dcmwrite(out_path, ds_out, write_like_original=False)


def split_bruker_dicoms(input_dir, out_dir):
    """Splits Bruker Enhanced DICOMs dynamically by whichever of Complex, Coil, or Double Samples exist."""
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
    dim_map = []

    for idx, fpath in enumerate(input_files):
        print(f"Indexing {os.path.basename(fpath)}...")
        ds_hdr, frames, d_map, pix_offset, bpf = index_bruker_file(fpath, idx)
        if idx == 0:
            base_header = ds_hdr
            dim_map = d_map
        all_frames.extend(frames)
        print(f"  -> Indexed {len(frames)} frames.")

        n_f = len(frames)
        mmap = np.memmap(
            fpath,
            dtype=np.uint8,
            mode="r",
            offset=pix_offset,
            shape=(n_f, bpf),
        )
        pixel_memmaps.append(mmap)

    # Summarize detected dimensions dynamically
    print("\nDetected Dimension Organization:")
    for d in dim_map:
        dim_idx = d["idx"]
        unique_vals = sorted(list(set(f["div"][dim_idx] for f in all_frames)))
        ptr_str = f"({d['pointer'][0]:04X}, {d['pointer'][1]:04X})"
        print(
            f"  Dim {dim_idx}: Pointer={ptr_str}, Role={d['role']} ('{d['label']}') -> "
            f"{len(unique_vals)} unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}"
        )

    roles_present = {d["role"] for d in dim_map}
    has_complex = "COMPLEX" in roles_present or any(f["comp_str"] is not None for f in all_frames)
    has_coil = "COIL" in roles_present
    has_ds = "DOUBLE_SAMPLES" in roles_present

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

    print(f"\nSplitting into {len(groups)} file(s)...")
    for idx, ((comp_str, coil_idx, ds_idx), flist) in enumerate(
        sorted(groups.items(), key=lambda x: tuple(v if v is not None else "" for v in x[0]))
    ):
        # Sort frames with cycle_idx on the outer loop and diff_idx on the inner loop:
        # (cycle 1: diff 1..N), (cycle 2: diff 1..N), ...
        flist.sort(key=lambda f: (f["cycle_idx"], f["diff_idx"], f["pkg_idx"], f["slice_idx"]))

        # Dynamically build filename and SeriesDescription based on which dimensions exist
        name_parts = [f"mri_{base_header.SeriesNumber}"]
        desc_parts = [orig_desc]

        if has_coil and coil_idx is not None:
            name_parts.append(f"coil{coil_idx}")
            desc_parts.append(f"coil{coil_idx}")
        if has_ds and ds_idx is not None:
            name_parts.append(f"ds{ds_idx}")
            desc_parts.append(f"ds{ds_idx}")
        if has_complex and comp_str:
            name_parts.append(comp_str.lower())
            #desc_parts.append(comp_str[:4].lower()) # Dont need this, because main use for desc is dcm2niix naming, and dcm2niix appends real/imaginary anyway because of below code setting img_type

        if len(name_parts) == 1:
            name_parts.append("split")

        out_name = "_".join(name_parts) + ".dcm"
        out_path = os.path.join(out_dir, out_name)

        hdr_copy = copy.deepcopy(base_header)
        hdr_copy.SeriesDescription = "_".join(desc_parts)
        hdr_copy.SeriesNumber = orig_num * 100 + idx + 1

        if has_complex and comp_str and "ImageType" in hdr_copy and orig_typelen >= 4:
            img_type = list(orig_imgtypelist)
            img_type[3] = comp_str
            hdr_copy.ImageType = img_type

        print(f"Writing {out_name} ({len(flist)} frames)...")
        write_split_dicom(out_path, hdr_copy, flist, pixel_memmaps)

    print(f"\nDone! Output files located in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Dynamically split Bruker Enhanced DICOMs by Complex, Coil, and Double Samples."
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