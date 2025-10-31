#!/usr/bin/env python3
"""Compare dynamic vs static model byte structure"""
import struct
from pathlib import Path

dynamic_path = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/svr_combined/AU_1_dynamic_intensity_comb.dat")
static_path = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/svr_combined/AU_4_static_intensity_comb.dat")

print("=" * 80)
print("DYNAMIC MODEL (AU_1)")
print("=" * 80)
print(f"File size: {dynamic_path.stat().st_size} bytes\n")

with open(dynamic_path, 'rb') as f:
    first_40 = f.read(40)
    print("First 40 bytes (hex):")
    print(' '.join(f'{b:02x}' for b in first_40))
    print()

    f.seek(0)
    marker = struct.unpack('<i', f.read(4))[0]
    cutoff = struct.unpack('<d', f.read(8))[0]
    means_rows = struct.unpack('<i', f.read(4))[0]
    means_cols = struct.unpack('<i', f.read(4))[0]
    means_dtype = struct.unpack('<i', f.read(4))[0]

    print(f"marker: {marker}")
    print(f"cutoff: {cutoff}")
    print(f"means: rows={means_rows}, cols={means_cols}, dtype={means_dtype}")
    print(f"Position after means header: {f.tell()} bytes")

print("\n" + "=" * 80)
print("STATIC MODEL (AU_4)")
print("=" * 80)
print(f"File size: {static_path.stat().st_size} bytes\n")

with open(static_path, 'rb') as f:
    first_40 = f.read(40)
    print("First 40 bytes (hex):")
    print(' '.join(f'{b:02x}' for b in first_40))
    print()

    f.seek(0)
    means_rows = struct.unpack('<i', f.read(4))[0]
    means_cols = struct.unpack('<i', f.read(4))[0]
    means_dtype = struct.unpack('<i', f.read(4))[0]

    print(f"means: rows={means_rows}, cols={means_cols}, dtype={means_dtype}")
    print(f"Position after means header: {f.tell()} bytes")

    # If cols=1, maybe it's not empty?
    if means_rows == 0 and means_cols == 1:
        print("\n⚠️  Unusual: rows=0 but cols=1")
        print("This might mean:")
        print("  1. The means matrix is actually 1x4702 (transposed)?")
        print("  2. Or dtype is wrong and we're misaligned?")

print("\n" + "=" * 80)
print("File size comparison:")
print("=" * 80)
print(f"Dynamic: {dynamic_path.stat().st_size} bytes")
print(f"Static:  {static_path.stat().st_size} bytes")
print(f"Difference: {dynamic_path.stat().st_size - static_path.stat().st_size} bytes")
print(f"(Expected difference if only cutoff missing: 8 bytes for marker + cutoff = 12 bytes)")
