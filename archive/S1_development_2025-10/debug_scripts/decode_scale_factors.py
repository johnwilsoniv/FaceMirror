#!/usr/bin/env python3
"""
Decode the scale factors in bytes 4-11 of AU model files
"""
import struct
from pathlib import Path

def read_scale_factor(dat_path):
    """Read the scale factor from bytes 4-11"""
    with open(dat_path, 'rb') as f:
        marker = struct.unpack('<i', f.read(4))[0]
        scale = struct.unpack('<d', f.read(8))[0]  # Read as double (float64)
    return marker, scale

def main():
    models_dir = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/svr_disfa")

    print("="*80)
    print("AU Model Scale Factors")
    print("="*80)

    # Check all AUs 1-26
    for au_num in [1, 2, 4, 5, 6, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26]:
        dat_path = models_dir / f"AU_{au_num}_dynamic_intensity.dat"
        if dat_path.exists():
            marker, scale = read_scale_factor(dat_path)
            print(f"AU_{au_num:2d}: marker={marker}, scale={scale:.4f}")
        else:
            print(f"AU_{au_num:2d}: FILE NOT FOUND")

if __name__ == "__main__":
    main()
