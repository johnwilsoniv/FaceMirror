#!/usr/bin/env python3
"""
Inspect OpenFace 2.2 binary .dat file format to understand structure
"""
import struct
from pathlib import Path

def inspect_binary_file(dat_path, max_bytes=100):
    """Inspect first N bytes of binary file"""
    with open(dat_path, 'rb') as f:
        data = f.read(max_bytes)

    print(f"\n{'='*80}")
    print(f"File: {dat_path.name}")
    print(f"Size: {dat_path.stat().st_size} bytes")
    print(f"{'='*80}")

    # Print raw hex
    print("\nFirst 100 bytes (hex):")
    hex_str = ' '.join(f'{b:02x}' for b in data)
    for i in range(0, len(hex_str), 80):
        print(f"  {hex_str[i:i+80]}")

    # Try different interpretations
    print("\n\nInterpretation attempts:")
    print("-" * 80)

    offset = 0

    # Try reading as sequence of int32
    print("\nAs int32 sequence (little-endian):")
    for i in range(0, min(40, len(data)), 4):
        if i + 4 <= len(data):
            val = struct.unpack('<i', data[i:i+4])[0]
            print(f"  Bytes {i:2d}-{i+3:2d}: {val:12d} (0x{val:08x})")

    # Try reading mixed int32 and float64
    print("\n\nAs mixed format:")
    print("  Assuming: int32, int32, int32, float64_data...")
    if len(data) >= 12:
        v1 = struct.unpack('<i', data[0:4])[0]
        v2 = struct.unpack('<i', data[4:8])[0]
        v3 = struct.unpack('<i', data[8:12])[0]
        print(f"  [0-3]  int32:  {v1}")
        print(f"  [4-7]  int32:  {v2}")
        print(f"  [8-11] int32:  {v3}")

        # If v2 is 0 (empty matrix), next should be support vectors header
        if v2 == 0:
            print(f"\n  Means matrix is EMPTY (cols=0)")
            print(f"  Next should be support_vectors header at byte 12:")
            if len(data) >= 24:
                sv_rows = struct.unpack('<i', data[12:16])[0]
                sv_cols = struct.unpack('<i', data[16:20])[0]
                sv_dtype = struct.unpack('<i', data[20:24])[0]
                print(f"    [12-15] SV rows:  {sv_rows}")
                print(f"    [16-19] SV cols:  {sv_cols}")
                print(f"    [20-23] SV dtype: {sv_dtype}")
        else:
            # Has means data
            print(f"\n  Means matrix has data (rows={v1}, cols={v2}, dtype={v3})")
            means_size = v1 * v2 * 8  # float64 = 8 bytes
            print(f"  Means data size: {means_size} bytes")
            print(f"  Support vectors header should start at byte {12 + means_size}")

def main():
    """Compare multiple AU files"""

    models_dir = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/svr_disfa")

    # Test multiple AU files
    test_aus = [1, 2, 4, 5, 6, 9, 12]

    print("="*80)
    print("OpenFace 2.2 Binary Format Investigation")
    print("="*80)
    print(f"\nModels directory: {models_dir}")

    for au_num in test_aus:
        dat_path = models_dir / f"AU_{au_num}_dynamic_intensity.dat"
        if dat_path.exists():
            inspect_binary_file(dat_path)
        else:
            print(f"\n{'='*80}")
            print(f"File: AU_{au_num}_dynamic_intensity.dat - NOT FOUND")
            print(f"{'='*80}")

if __name__ == "__main__":
    main()
