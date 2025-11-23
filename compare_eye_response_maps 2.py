#!/usr/bin/env python3
"""
Compare C++ vs Python eye model response maps.
"""

import numpy as np
import re

def parse_python_response_maps():
    """Parse Python response maps from debug file."""
    try:
        with open('/tmp/python_eye_response_maps.txt', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("Python response map file not found")
        return {}

    # Find all landmark response maps
    maps = {}
    pattern = r'Eye landmark (\d+) response map:\s*\n\s*min: ([\d.]+), max: ([\d.]+), mean: ([\d.]+)\s*\n\s*(\d+)x\d+ response map:\s*\n((?:\s+[\d.]+\s*)+)'

    matches = re.findall(pattern, content)
    for match in matches:
        lm_idx = int(match[0])
        min_val = float(match[1])
        max_val = float(match[2])
        mean_val = float(match[3])
        size = int(match[4])

        # Parse the matrix values
        values = [float(v) for v in match[5].split()]
        matrix = np.array(values).reshape(size, size)

        maps[lm_idx] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'matrix': matrix
        }

    return maps

def parse_cpp_response_maps_from_detail():
    """Parse C++ response maps from detailed debug file if available."""
    # The C++ eye model debug doesn't save response maps in binary format
    # Let me check the detailed text output
    try:
        with open('/tmp/cpp_eye_model_detailed.txt', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return {}

    # Look for any response map data in the C++ output
    # C++ doesn't output the actual response map values, only mean-shifts
    return {}

def main():
    print("=" * 70)
    print("COMPARING EYE MODEL RESPONSE MAPS")
    print("=" * 70)

    py_maps = parse_python_response_maps()

    if not py_maps:
        print("\nNo Python response maps found. Running comparison script first...")
        return

    print(f"\nPython response maps found for landmarks: {sorted(py_maps.keys())}")

    for lm_idx in sorted(py_maps.keys()):
        data = py_maps[lm_idx]
        print(f"\n--- Eye Landmark {lm_idx} ---")
        print(f"  Shape: {data['matrix'].shape}")
        print(f"  Min: {data['min']:.6f}")
        print(f"  Max: {data['max']:.6f}")
        print(f"  Mean: {data['mean']:.6f}")

        # Find peak
        matrix = data['matrix']
        peak_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        center = (matrix.shape[0] - 1) / 2.0
        peak_offset_x = peak_idx[1] - center
        peak_offset_y = peak_idx[0] - center

        print(f"  Peak position: ({peak_idx[1]}, {peak_idx[0]})")
        print(f"  Peak offset from center: dx={peak_offset_x:.1f}, dy={peak_offset_y:.1f}")

        # Compute mean-shift using KDE
        sigma = 1.0  # Same as eye model
        ws = matrix.shape[0]

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(range(ws), range(ws), indexing='ij')

        # KDE weighted mean
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        for i in range(ws):
            for j in range(ws):
                dist_sq = (i - center)**2 + (j - center)**2
                kde_weight = np.exp(-0.5 * dist_sq / (sigma**2))
                response_weight = matrix[i, j] * kde_weight
                weighted_x += j * response_weight
                weighted_y += i * response_weight
                total_weight += response_weight

        if total_weight > 0:
            mean_x = weighted_x / total_weight - center
            mean_y = weighted_y / total_weight - center
            print(f"  KDE mean-shift (ref space): dx={mean_x:.4f}, dy={mean_y:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compare with what we expect
    print("\nThe Python response maps show the patch expert responses.")
    print("Key question: Are these response maps computed at the correct positions?")
    print("\nFor Eye_8 (main landmark 36):")
    if 8 in py_maps:
        data = py_maps[8]
        print(f"  Max response: {data['max']:.4f}")
        peak_idx = np.unravel_index(np.argmax(data['matrix']), data['matrix'].shape)
        center = (data['matrix'].shape[0] - 1) / 2.0
        print(f"  Peak at: ({peak_idx[1]}, {peak_idx[0]}), offset from center: ({peak_idx[1]-center:.1f}, {peak_idx[0]-center:.1f})")

        # The peak should be near the center if landmarks are at correct position
        # A peak offset means the landmarks need to move in that direction

if __name__ == '__main__':
    main()
