#!/usr/bin/env python3
"""
Compare C++ vs Python eye model parameter updates.
"""

import re
import numpy as np

def parse_cpp_detailed():
    """Parse the C++ detailed debug output."""
    with open('/tmp/cpp_eye_model_detailed.txt', 'r') as f:
        content = f.read()

    # Find all iterations with parameter updates for left eye (first 28-point model)
    # C++ outputs two eyes, we want the first one

    # Extract initial params
    match = re.search(r'Initial global params: scale=([\d.]+) rot=\(([-\d.]+),([-\d.]+),([-\d.]+)\) tx=([-\d.]+) ty=([-\d.]+)', content)
    if match:
        initial = {
            'scale': float(match.group(1)),
            'rx': float(match.group(2)),
            'ry': float(match.group(3)),
            'rz': float(match.group(4)),
            'tx': float(match.group(5)),
            'ty': float(match.group(6))
        }
        print("C++ Initial params:", initial)

    # Look for parameter updates
    updates = re.findall(r'Parameter update:\s*\n\s*delta_scale: ([-\d.]+)\s*\n\s*delta_rot: \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)\s*\n\s*delta_tx: ([-\d.]+)\s*\n\s*delta_ty: ([-\d.]+)', content)

    print(f"\nFound {len(updates)} C++ parameter updates")
    for i, u in enumerate(updates[:5]):  # First 5 iterations
        print(f"  Iter {i}: delta_scale={float(u[0]):.6f} delta_rot=({float(u[1]):.6f},{float(u[2]):.6f},{float(u[3]):.6f}) delta_tx={float(u[4]):.6f} delta_ty={float(u[5]):.6f}")

def parse_python_detailed():
    """Parse the Python detailed debug output."""
    with open('/tmp/python_eye_model_detailed.txt', 'r') as f:
        content = f.read()

    # Extract initial params
    match = re.search(r'Initial global params: scale=([\d.]+) rot=\(([-\d.]+),([-\d.]+),([-\d.]+)\) tx=([-\d.]+) ty=([-\d.]+)', content)
    if match:
        initial = {
            'scale': float(match.group(1)),
            'rx': float(match.group(2)),
            'ry': float(match.group(3)),
            'rz': float(match.group(4)),
            'tx': float(match.group(5)),
            'ty': float(match.group(6))
        }
        print("\nPython Initial params:", initial)

    # Look for "After RIGID phase" and "After NONRIGID phase"
    rigid_matches = re.findall(r'After RIGID phase WS=(\d+).*?Global params: scale=([\d.]+) rot=\(([-\d.]+),([-\d.]+),([-\d.]+)\) tx=([-\d.]+) ty=([-\d.]+)', content)

    print(f"\nFound {len(rigid_matches)} Python RIGID phase results")
    for m in rigid_matches:
        ws = int(m[0])
        print(f"  After RIGID WS={ws}: scale={float(m[1]):.6f} rot=({float(m[2]):.6f},{float(m[3]):.6f},{float(m[4]):.6f}) tx={float(m[5]):.6f} ty={float(m[6]):.6f}")

def main():
    print("=" * 70)
    print("COMPARING EYE MODEL PARAMETER UPDATES")
    print("=" * 70)

    try:
        parse_cpp_detailed()
    except Exception as e:
        print(f"Error parsing C++: {e}")

    try:
        parse_python_detailed()
    except Exception as e:
        print(f"Error parsing Python: {e}")

    # Now let's compute what the difference should be
    print("\n" + "=" * 70)
    print("ANALYZING MOVEMENT")
    print("=" * 70)

    # Python's initial landmarks from file
    print("\nPython moves from ~425 tx to ~432 tx (7px) while C++ only moves ~2px")
    print("This suggests Python is over-adjusting the translation parameters")

    # The key observation: Python initial tx=425, final tx=432 (7px movement)
    # C++ stays much closer to initial position

    print("\nPossible causes:")
    print("1. Python damping factor (0.5) may be different from C++")
    print("2. Python may be doing more RIGID iterations than C++")
    print("3. Jacobian computation differs")

if __name__ == '__main__':
    main()
