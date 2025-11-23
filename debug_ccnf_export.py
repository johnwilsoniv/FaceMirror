#!/usr/bin/env python3
"""
Debug CCNF export by directly reading C++ binary and comparing with exported Python.
"""

import numpy as np
from pathlib import Path
import struct

def read_int32(f):
    """Read 4-byte integer (little-endian)."""
    return struct.unpack('<i', f.read(4))[0]

def read_float64(f):
    """Read 8-byte double (little-endian)."""
    return struct.unpack('<d', f.read(8))[0]

def read_matrix_bin(f):
    """Read binary matrix in OpenFace format."""
    rows = read_int32(f)
    cols = read_int32(f)
    cv_type = read_int32(f)

    if cv_type == 4:  # CV_32SC1
        dtype, elem_size = np.int32, 4
    elif cv_type == 5:  # CV_32FC1
        dtype, elem_size = np.float32, 4
    elif cv_type == 6:  # CV_64FC1
        dtype, elem_size = np.float64, 8
    else:
        raise ValueError(f"Unsupported type: {cv_type}")

    data = np.frombuffer(f.read(rows * cols * elem_size), dtype=dtype)
    return data.reshape(rows, cols)

def main():
    print("=" * 70)
    print("CCNF EXPORT DEBUG - Comparing C++ binary with Python export")
    print("=" * 70)

    # C++ binary file for left eye
    cpp_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/model_eye/patch_experts/left_ccnf_patches_1.00_synth_lid_.txt"

    # Python exported directory
    py_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_ccnf_left/scale_1.00"

    print(f"\nC++ source: {cpp_file}")
    print(f"Python dir: {py_dir}")

    # Read C++ binary header
    print("\n" + "=" * 70)
    print("READING C++ BINARY FILE")
    print("=" * 70)

    with open(cpp_file, 'rb') as f:
        patch_scaling = read_float64(f)
        num_views = read_int32(f)

        print(f"patch_scaling: {patch_scaling}")
        print(f"num_views: {num_views}")

        # Read view centers
        centers = []
        for view_idx in range(num_views):
            center = read_matrix_bin(f)
            center_deg = center * 180.0 / np.pi
            centers.append(center_deg)
            print(f"View {view_idx} center: pitch={center_deg[0,0]:.1f}°, yaw={center_deg[1,0]:.1f}°")

        # Read visibilities
        visibilities = []
        for view_idx in range(num_views):
            visibility = read_matrix_bin(f)
            visibilities.append(visibility)
            print(f"View {view_idx} visibility: {np.sum(visibility)}/28 visible")

        # Read window sizes and sigmas
        num_win_sizes = read_int32(f)
        window_sizes = []
        sigma_components = []

        for w in range(num_win_sizes):
            ws = read_int32(f)
            n_sigmas = read_int32(f)
            window_sizes.append(ws)

            sigmas = []
            for s in range(n_sigmas):
                sigma_mat = read_matrix_bin(f)
                sigmas.append(sigma_mat)
            sigma_components.append(sigmas)
            print(f"Window {ws}: {n_sigmas} sigma components")

        # Read patch expert for landmark 8 in view 0 (frontal)
        # Landmark 8 is the outer left eye corner in 28-point model
        target_landmark = 8

        print(f"\n" + "=" * 70)
        print(f"READING LANDMARK {target_landmark} PATCH EXPERT (View 0)")
        print("=" * 70)

        # Read all patches until we get to landmark 8
        for view_idx in range(num_views):
            if view_idx > 0:
                break  # Only read view 0

            for lm_idx in range(28):
                # Check visibility
                if visibilities[view_idx][lm_idx, 0] != 1:
                    continue

                # Read patch expert header
                read_type = read_int32(f)
                if read_type != 5:
                    raise ValueError(f"Expected type 5, got {read_type}")

                width = read_int32(f)
                height = read_int32(f)
                num_neurons = read_int32(f)

                if num_neurons == 0:
                    read_int32(f)  # Empty marker
                    continue

                # Read neurons
                neurons = []
                for n in range(num_neurons):
                    n_type = read_int32(f)
                    if n_type != 2:
                        raise ValueError(f"Expected neuron type 2, got {n_type}")

                    neuron_type = read_int32(f)
                    norm_weights = read_float64(f)
                    bias = read_float64(f)
                    alpha = read_float64(f)
                    weights = read_matrix_bin(f)

                    neurons.append({
                        'neuron_type': neuron_type,
                        'norm_weights': norm_weights,
                        'bias': bias,
                        'alpha': alpha,
                        'weights': weights
                    })

                # Read betas
                n_betas = len(sigma_components[0])
                betas = [read_float64(f) for _ in range(n_betas)]

                # Read confidence
                confidence = read_float64(f)

                if lm_idx == target_landmark:
                    print(f"\nLandmark {lm_idx}:")
                    print(f"  Size: {height}x{width}")
                    print(f"  Num neurons: {num_neurons}")
                    print(f"  Betas: {betas}")
                    print(f"  Confidence: {confidence}")

                    for i, neuron in enumerate(neurons):
                        if neuron['alpha'] > 0.1:  # Only print significant neurons
                            print(f"\n  Neuron {i}:")
                            print(f"    alpha: {neuron['alpha']:.6f}")
                            print(f"    bias: {neuron['bias']:.6f}")
                            print(f"    norm_weights: {neuron['norm_weights']:.6f}")
                            print(f"    weights shape: {neuron['weights'].shape}")
                            print(f"    weights[0,:5]: {neuron['weights'].flatten()[:5]}")

                    break

    # Now load Python exported version
    print("\n" + "=" * 70)
    print("LOADING PYTHON EXPORTED MODEL")
    print("=" * 70)

    py_path = Path(py_dir)

    # Load global metadata
    global_meta = np.load(py_path / 'global_metadata.npz')
    print(f"patch_scaling: {global_meta['patch_scaling']}")
    print(f"num_views: {global_meta['num_views']}")
    print(f"num_landmarks: {global_meta['num_landmarks']}")

    # Load patch for landmark 8 in view 0
    patch_dir = py_path / 'view_00' / f'patch_{target_landmark:02d}'

    if not patch_dir.exists():
        print(f"ERROR: Patch directory not found: {patch_dir}")
        return

    meta = np.load(patch_dir / 'metadata.npz')
    print(f"\nLandmark {target_landmark}:")
    print(f"  Size: {meta['height']}x{meta['width']}")
    print(f"  Betas: {meta['betas']}")
    print(f"  Confidence: {meta['patch_confidence']}")

    # Load neurons
    neuron_files = sorted(patch_dir.glob('neuron_*.npz'))
    print(f"  Num neurons: {len(neuron_files)}")

    for nf in neuron_files:
        neuron = np.load(nf)
        alpha = float(neuron['alpha'])
        if alpha > 0.1:
            n_idx = int(nf.stem.split('_')[1])
            print(f"\n  Neuron {n_idx}:")
            print(f"    alpha: {alpha:.6f}")
            print(f"    bias: {float(neuron['bias']):.6f}")
            print(f"    norm_weights: {float(neuron['norm_weights']):.6f}")
            print(f"    weights shape: {neuron['weights'].shape}")
            print(f"    weights[0,:5]: {neuron['weights'].flatten()[:5]}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print("\nCompare the neuron values above to find the mismatch.")
    print("C++ uses alpha=3.44 for neuron 0, Python shows alpha=6.82")

if __name__ == '__main__':
    main()
