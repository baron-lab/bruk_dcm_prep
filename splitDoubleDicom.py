import os
import argparse
import struct
import numpy as np
import pydicom
from pydicom.filereader import read_partial
from pydicom.tag import Tag
from pydicom.uid import generate_uid

# Function to split bruker dicoms by double sampling, coil, and real/imaginary
# Should only be needed when cfmm custom export splitting double sampling into two volumes is used, which causes dcm2niix to fail.

# Exact binary signatures verified from your hex dump:
FRAME_ITEM_START = (
    b"\xfe\xff\x00\xe0\xff\xff\xff\xff\x18\x00\x42\x90"  # Top-level Frame Item start
)
SEQ_DELIM = b"\xfe\xff\xdd\xe0\x00\x00\x00\x00"  # (FFFE, E0DD) Sequence Delimiter
DIV_SIG = b"\x20\x00\x57\x91UL"  # (0020, 9157) DimensionIndexValues
COMP_SIG = b"\x08\x00\x08\x92CS"  # (0008, 9208) ComplexImageComponent


def index_bruker_file(filepath, file_idx):
  """Indexes all frames in a Bruker Enhanced DICOM in ~1-2 seconds using raw byte offsets."""
  stop_tag = Tag(0x5200, 0x9230)
  file_size = os.path.getsize(filepath)

  with open(filepath, "rb") as fp:
    ds_header = read_partial(
        fp, stop_when=lambda tag, vr, length: tag >= stop_tag
    )
    seq_header_pos = fp.tell()

    n_frames = int(ds_header.NumberOfFrames)
    rows = int(ds_header.Rows)
    cols = int(ds_header.Columns)
    bytes_per_frame = rows * cols * (int(ds_header.BitsAllocated) // 8)
    total_pixel_bytes = n_frames * bytes_per_frame

    # Read from start of (5200, 9230) up to just after the start of (7FE0, 0010) PixelData
    fp.seek(seq_header_pos)
    read_len = (file_size - total_pixel_bytes) - seq_header_pos + 64
    raw_seq_buf = bytearray(fp.read(read_len))

  # Locate exact (7FE0, 0010) PixelData header at the end of raw_seq_buf
  pix_tag_idx = raw_seq_buf.rfind(b"\xe0\x7f\x10\x00")
  if pix_tag_idx == -1:
    raise RuntimeError(f"Could not find (7FE0, 0010) PixelData in {filepath}")

  vr = bytes(raw_seq_buf[pix_tag_idx + 4 : pix_tag_idx + 6])
  pix_header_len = 12 if vr in (b"OW", b"OB") else 8
  pixel_data_offset = seq_header_pos + pix_tag_idx + pix_header_len

  # Trim buffer to end right before (7FE0, 0010)
  del raw_seq_buf[pix_tag_idx:]
  # Also strip the final 8-byte Sequence Delimiter (FFFE, E0DD) of (5200, 9230)
  if raw_seq_buf.endswith(SEQ_DELIM):
    del raw_seq_buf[-8:]

  # Find all top-level frame starts using FRAME_ITEM_START
  item_starts = []
  pos = 12  # Skip the 12-byte (5200, 9230) SQ header
  while True:
    idx = raw_seq_buf.find(FRAME_ITEM_START, pos)
    if idx == -1:
      break
    item_starts.append(idx)
    pos = idx + 900  # Each frame item is ~954 bytes; jump ahead for speed

  if len(item_starts) != n_frames:
    raise RuntimeError(
        f"Expected {n_frames} frame items in {filepath}, found"
        f" {len(item_starts)}"
    )

  frames_meta = []
  for f_i in range(n_frames):
    start_b = item_starts[f_i]
    end_b = item_starts[f_i + 1] if (f_i + 1) < n_frames else len(raw_seq_buf)

    # Find DimensionIndexValues (0020, 9157) inside this frame
    div_idx = raw_seq_buf.find(DIV_SIG, start_b, end_b)
    val_len = struct.unpack("<H", raw_seq_buf[div_idx + 6 : div_idx + 8])[0]
    vals = list(
        struct.unpack(
            f"<{val_len//4}I", raw_seq_buf[div_idx + 8 : div_idx + 8 + val_len]
        )
    )
    # vals = [COMPLEX, COIL, diffusion, double_samples, SLICE_PACKAGE, SLICE]

    # Find ComplexImageComponent (0008, 9208) string (e.g., 'REAL' or 'IMAGINARY')
    comp_idx = raw_seq_buf.find(COMP_SIG, start_b, end_b)
    if comp_idx != -1:
      c_len = struct.unpack("<H", raw_seq_buf[comp_idx + 6 : comp_idx + 8])[0]
      comp_str = (
          raw_seq_buf[comp_idx + 8 : comp_idx + 8 + c_len]
          .decode("ascii")
          .strip()
      )
    else:
      comp_str = "REAL" if vals[0] == 1 else "IMAGINARY"

    # Extract a mutable copy of this frame's metadata bytes and normalize
    # COMPLEX (vals[0]), COIL (vals[1]), and double_samples (vals[3]) to 1
    # so dcm2niix sees a clean 1-indexed array in every split file!
    item_bytes = bytearray(raw_seq_buf[start_b:end_b])
    rel_div_idx = div_idx - start_b
    patched_vals = [1, 1, vals[2], 1, vals[4], vals[5]]
    item_bytes[rel_div_idx + 8 : rel_div_idx + 8 + val_len] = struct.pack(
        f"<{len(patched_vals)}I", *patched_vals
    )

    frames_meta.append({
        "file_idx": file_idx,
        "frame_idx": f_i,
        "item_bytes": bytes(item_bytes),
        "div": tuple(vals),
        "comp_str": comp_str,
        "complex_idx": vals[0],
        "coil_idx": vals[1],
        "diff_idx": vals[2],
        "double_sample_idx": vals[3],
        "pkg_idx": vals[4],
        "slice_idx": vals[5],
    })

  return ds_header, frames_meta, pixel_data_offset, bytes_per_frame


def write_split_dicom(
    out_path, base_header, selected_frames, pixel_memmaps, bytes_per_frame
):
  """Writes a standalone Enhanced DICOM file for the selected subset of frames."""
  ds_out = base_header.copy()
  n_out = len(selected_frames)
  ds_out.NumberOfFrames = n_out

  # Generate unique UIDs so dcm2niix treats each file as a distinct series
  ds_out.SeriesInstanceUID = generate_uid()
  ds_out.SOPInstanceUID = generate_uid()
  ds_out.file_meta.MediaStorageSOPInstanceUID = ds_out.SOPInstanceUID

  with open(out_path, "wb") as fp:
    # 1. Write top-level DICOM header
    pydicom.dcmwrite(fp, ds_out, write_like_original=False)

    # 2. Write (5200, 9230) PerFrameFunctionalGroupsSequence header
    fp.write(b"\x00\x52\x30\x92SQ\x00\x00\xff\xff\xff\xff")
    for f in selected_frames:
      fp.write(f["item_bytes"])
    fp.write(SEQ_DELIM)

    # 3. Write (7FE0, 0010) PixelData header + raw frame pixels
    total_pix_bytes = n_out * bytes_per_frame
    fp.write(b"\xe0\x7f\x10\x00OW\x00\x00")
    fp.write(struct.pack("<I", total_pix_bytes))

    for f in selected_frames:
      mmap_arr = pixel_memmaps[f["file_idx"]]
      fp.write(mmap_arr[f["frame_idx"]].tobytes())


def main():
  parser = argparse.ArgumentParser(description="Split Bruker Enhanced DICOMs by Complex, Coil, and Double Samples.")
  parser.add_argument("input_dir", help="Directory containing the input DICOM files")
  parser.add_argument("output_dir", help="Directory to save the split DICOM files")
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

  all_frames = []
  pixel_memmaps = []
  base_header = None
  bytes_per_frame = 0

  for idx, fpath in enumerate(input_files):
    print(f"Indexing {os.path.basename(fpath)}...")
    ds_hdr, frames, pix_offset, bpf = index_bruker_file(fpath, idx)
    if idx == 0:
      base_header = ds_hdr
      bytes_per_frame = bpf
    all_frames.extend(frames)
    print(f"  -> Indexed {len(frames)} frames.")

    # Memory-map the pixel data directly from disk (0 RAM overhead)
    n_f = int(ds_hdr.NumberOfFrames)
    mmap = np.memmap(
        fpath,
        dtype=np.uint8,
        mode="r",
        offset=pix_offset,
        shape=(n_f, bpf),
    )
    pixel_memmaps.append(mmap)

  # Summarize unique dimension counts across the combined dataset
  for dim_idx, dim_name in enumerate(
      ["COMPLEX", "COIL", "diffusion", "double_samples", "SLICE_PKG", "SLICE"]
  ):
    unique_vals = sorted(list(set(f["div"][dim_idx] for f in all_frames)))
    print(
        f"Dimension {dim_idx} ({dim_name}): {len(unique_vals)} unique values ->"
        f" {unique_vals[:10]}{'...' if len(unique_vals)>10 else ''}"
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

    # Use copy() to duplicate the header for modification
    hdr_copy = base_header.copy()
    hdr_copy.SeriesDescription = f"{orig_desc}_{comp_str[:4].lower()}_c{coil_idx}_ds{ds_idx}"
    hdr_copy.SeriesNumber = orig_num * 100 + idx + 1

    # Also set top-level ImageType[2] to REAL or IMAGINARY to help dcm2niix
    if "ImageType" in hdr_copy and orig_typelen >= 4:
      img_type = list(orig_imgtypelist)  # Create a fresh copy of the list for this iteration
      img_type[3] = comp_str
      hdr_copy.ImageType = img_type

    print(f"Writing {out_name} ({len(flist)} frames)...")
    write_split_dicom(out_path, hdr_copy, flist, pixel_memmaps, bytes_per_frame)

  print("\nDone! You can now run dcm2niix on:", out_dir)


if __name__ == "__main__":
  main()