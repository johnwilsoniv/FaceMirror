#!/usr/bin/env python3
"""
Manually parse CCNF model file to understand structure.
"""
import struct
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "~/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/patch_experts/ccnf_patches_0.25_general.txt"
model_path = model_path.replace("~", "/Users/johnwilsoniv")

with open(model_path, "rb") as f:
    # Read patch scaling (double, 8 bytes)
    patch_scaling = struct.unpack('d', f.read(8))[0]
    print(f"Patch scaling: {patch_scaling}")

    # Read number of views (int, 4 bytes)
    num_views = struct.unpack('i', f.read(4))[0]
    print(f"Number of views: {num_views}")
    print()

    # Read centers for each view (cv::Mat format via ReadMatBin)
    print(f"View centers:")
    for i in range(num_views):
        # ReadMatBin: rows, cols, type (each 4 bytes), then data
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        mat_type = struct.unpack('i', f.read(4))[0]

        # Read center data (3 doubles = 24 bytes for Vec3d)
        x, y, z = struct.unpack('ddd', f.read(24))
        print(f"  View {i}: [{x:.2f}, {y:.2f}, {z:.2f}] (mat: {rows}x{cols} type={mat_type})")
    print()

    # Read visibility matrices (cv::Mat format via ReadMatBin)
    print(f"Visibility matrices:")
    for i in range(num_views):
        # OpenFace ReadMatBin: read rows, cols, type (each 4 bytes)
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        mat_type = struct.unpack('i', f.read(4))[0]

        print(f"  View {i}: visibility={rows}x{cols} type={mat_type}", end="")

        # Calculate data size based on type
        if rows > 0 and cols > 0 and rows < 1000 and cols < 1000:
            # CV_32SC1 = 4, CV_32FC1 = 5, CV_32FC2 = 6
            if mat_type == 4:  # CV_32SC1
                elem_size = 4
            elif mat_type == 5:  # CV_32FC1
                elem_size = 4
            elif mat_type == 6:  # CV_32FC2
                elem_size = 8
            else:
                elem_size = 4

            data_size = rows * cols * elem_size
            f.read(data_size)  # Skip the data
            print(f" (skipped {data_size} bytes)")
        else:
            print(f" ERROR: invalid dimensions!")
    print()

    # Current file position
    pos_before_sigma = f.tell()
    print(f"File position before sigma components: {pos_before_sigma}")
    print()

    # Read sigma components
    num_win_sizes = struct.unpack('i', f.read(4))[0]
    print(f"Number of window sizes: {num_win_sizes}")
    print()

    # Create output directory
    import os
    import numpy as np
    output_dir = "/tmp/sigma_export"
    os.makedirs(output_dir, exist_ok=True)

    # Save window sizes
    windows = []
    sigma_data = {}

    if num_win_sizes < 10:  # Sanity check
        for w in range(num_win_sizes):
            window_size = struct.unpack('i', f.read(4))[0]
            num_sigma_comp = struct.unpack('i', f.read(4))[0]
            windows.append(window_size)

            print(f"Window size {window_size}: {num_sigma_comp} sigma components")

            sigma_data[window_size] = []

            if num_sigma_comp < 100:  # Sanity check
                for s in range(num_sigma_comp):
                    rows = struct.unpack('i', f.read(4))[0]
                    cols = struct.unpack('i', f.read(4))[0]
                    mat_type = struct.unpack('i', f.read(4))[0]

                    # CV_32FC1 = 5, each element is 4 bytes
                    elem_size = 4
                    data_size = rows * cols * elem_size

                    # Read the actual sigma component data
                    sigma_bytes = f.read(data_size)
                    sigma_array = np.frombuffer(sigma_bytes, dtype=np.float32).reshape(rows, cols)

                    sigma_data[window_size].append(sigma_array)

                    print(f"  Sigma[{s}]: {rows}x{cols} type={mat_type}")

                    # Save to file
                    filename = f"{output_dir}/sigma_w{window_size}_c{s}.npy"
                    np.save(filename, sigma_array)
                    print(f"    Saved: {filename}")
            else:
                print(f"  ERROR: num_sigma_comp={num_sigma_comp} is too large!")
                break
    else:
        print(f"ERROR: num_win_sizes={num_win_sizes} seems wrong!")
        print("File reading might be misaligned")
        sys.exit(1)

    # Save window sizes
    np.save(f"{output_dir}/window_sizes.npy", np.array(windows))
    print()
    print(f"Exported {num_win_sizes} window sizes with sigma components to {output_dir}")
