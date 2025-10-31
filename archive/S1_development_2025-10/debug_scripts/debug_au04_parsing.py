#!/usr/bin/env python3
"""Debug AU04 static model parsing"""
import struct
from pathlib import Path

dat_path = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/svr_combined/AU_4_static_intensity_comb.dat")

print(f"File size: {dat_path.stat().st_size} bytes")
print(f"\nReading as STATIC model (no marker/cutoff header):\n")

with open(dat_path, 'rb') as f:
    # Read first 50 bytes as hex
    first_bytes = f.read(50)
    print("First 50 bytes (hex):")
    print(' '.join(f'{b:02x}' for b in first_bytes))
    print()

    # Reset and read as static model
    f.seek(0)

    # Means matrix header
    means_rows = struct.unpack('<i', f.read(4))[0]
    means_cols = struct.unpack('<i', f.read(4))[0]
    means_dtype = struct.unpack('<i', f.read(4))[0]

    print(f"Means matrix header:")
    print(f"  rows: {means_rows}")
    print(f"  cols: {means_cols}")
    print(f"  dtype: {means_dtype}")

    # Read means data
    if means_rows > 0 and means_cols > 0:
        means_size = means_rows * means_cols * 8
        print(f"  data size: {means_size} bytes")
        f.read(means_size)
    else:
        print(f"  Empty means matrix")

    print(f"\nCurrent position: {f.tell()} bytes")

    # Support vectors header
    sv_rows = struct.unpack('<i', f.read(4))[0]
    sv_cols = struct.unpack('<i', f.read(4))[0]
    sv_dtype = struct.unpack('<i', f.read(4))[0]

    print(f"\nSupport vectors matrix header:")
    print(f"  rows: {sv_rows}")
    print(f"  cols: {sv_cols}")
    print(f"  dtype: {sv_dtype}")

    sv_size = sv_rows * sv_cols * 8
    print(f"  expected data size: {sv_size} bytes")

    # Check remaining bytes
    current_pos = f.tell()
    f.seek(0, 2)  # Seek to end
    end_pos = f.tell()
    remaining = end_pos - current_pos
    print(f"  remaining bytes in file: {remaining}")

    if sv_size + 8 == remaining:  # +8 for bias
        print(f"  ✓ Dimensions look correct!")
    else:
        print(f"  ✗ Dimension mismatch! Expected {sv_size + 8}, have {remaining}")
