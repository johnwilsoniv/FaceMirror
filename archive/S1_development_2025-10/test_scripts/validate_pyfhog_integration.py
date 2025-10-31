#!/usr/bin/env python3
"""
Validate pyfhog Integration with OpenFace 2.2

This script validates that pyfhog produces identical FHOG features to OpenFace 2.2
when given the same aligned face images.

Strategy:
1. Use OpenFace C++ to extract features (produces .hog files from aligned faces)
2. Use OpenFace C++ to also save aligned face images
3. Run pyfhog on those aligned face images
4. Compare pyfhog output with OpenFace .hog files

Expected result: r > 0.9999 correlation (near-perfect match)
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import shutil
from scipy.stats import pearsonr
import pyfhog
from openface22_hog_parser import OF22HOGParser


def extract_features_with_aligned_images(video_path: str, output_dir: Path):
    """
    Run OpenFace C++ to extract features AND save aligned face images

    Returns:
        hog_file: Path to .hog file
        csv_file: Path to .csv file
        aligned_dir: Path to directory with aligned face images
    """
    openface_binary = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        openface_binary,
        "-f", str(video_path),
        "-out_dir", str(output_dir),
        "-hogalign",      # Extract HOG from aligned faces
        "-simalign",      # Save aligned face images
        "-pdmparams",     # Extract PDM parameters
        "-2Dfp",          # Extract 2D landmarks
        "-q"              # Quiet mode
    ]

    print(f"Running OpenFace C++ to extract features and aligned faces...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"OpenFace extraction failed:\n{result.stderr}")

    video_stem = Path(video_path).stem
    hog_file = output_dir / f"{video_stem}.hog"
    csv_file = output_dir / f"{video_stem}.csv"
    aligned_dir = output_dir / "aligned"

    if not hog_file.exists():
        raise FileNotFoundError(f"HOG file not created: {hog_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not created: {csv_file}")
    if not aligned_dir.exists():
        print(f"Warning: Aligned face directory not found: {aligned_dir}")
        print("OpenFace may not have saved aligned faces. Checking for processed directory...")
        aligned_dir = output_dir / f"{video_stem}_aligned"
        if not aligned_dir.exists():
            print(f"Also not found: {aligned_dir}")
            print("Will need to reconstruct aligned faces from landmarks.")
            aligned_dir = None

    print(f"✓ HOG file: {hog_file}")
    print(f"✓ CSV file: {csv_file}")
    if aligned_dir:
        print(f"✓ Aligned faces: {aligned_dir}")

    return hog_file, csv_file, aligned_dir


def load_aligned_face_from_openface(aligned_dir: Path, frame_num: int) -> np.ndarray:
    """
    Load aligned face image produced by OpenFace

    OpenFace saves aligned faces as: frame_det_00_000001.bmp (for frame 1)
    OpenFace saves them as 112x112, but FHOG extraction uses 96x96
    """
    # Try different naming patterns OpenFace might use
    patterns = [
        f"frame_det_00_{frame_num:06d}.bmp",
        f"frame_det_00_{frame_num:06d}.png",
        f"{frame_num:06d}.bmp",
        f"{frame_num:06d}.png",
    ]

    for pattern in patterns:
        img_path = aligned_dir / pattern
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                # Convert BGR to RGB for pyfhog
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # OpenFace uses 112x112 for FHOG extraction (produces 4464 features)
                # No resizing needed - use the 112x112 aligned face directly!
                return img_rgb

    raise FileNotFoundError(f"Could not find aligned face image for frame {frame_num}")


def compare_hog_features(openface_hog: np.ndarray, pyfhog_hog: np.ndarray, frame_idx: int):
    """Compare FHOG features from OpenFace and pyfhog"""

    # Check dimensions
    if openface_hog.shape != pyfhog_hog.shape:
        print(f"Frame {frame_idx}: Shape mismatch!")
        print(f"  OpenFace: {openface_hog.shape}")
        print(f"  pyfhog:   {pyfhog_hog.shape}")
        return None

    # Compute correlation
    corr, _ = pearsonr(openface_hog.flatten(), pyfhog_hog.flatten())

    # Compute statistics
    diff = openface_hog - pyfhog_hog
    mean_diff = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))

    return {
        'frame': frame_idx,
        'correlation': corr,
        'mean_abs_diff': mean_diff,
        'max_abs_diff': max_diff
    }


def main():
    print("="*80)
    print("PYFHOG INTEGRATION VALIDATION")
    print("="*80)

    # Configuration
    video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

    # Create output directory (permanent for diagnostics)
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/pyfhog_validation_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Step 1: Extract features with OpenFace C++
    print("\n" + "="*80)
    print("STEP 1: Extract features with OpenFace C++")
    print("="*80)

    hog_file, csv_file, aligned_dir = extract_features_with_aligned_images(
        video_path, output_dir
    )

    # Step 2: Load OpenFace HOG features
    print("\n" + "="*80)
    print("STEP 2: Load OpenFace HOG features")
    print("="*80)

    hog_parser = OF22HOGParser(str(hog_file))
    frame_indices, openface_hog_features = hog_parser.parse()
    print(f"✓ Loaded {len(frame_indices)} frames")
    print(f"✓ HOG dimensions: {openface_hog_features.shape}")

    # Step 3: Extract FHOG with pyfhog
    print("\n" + "="*80)
    print("STEP 3: Extract FHOG with pyfhog")
    print("="*80)

    if aligned_dir is None:
        print("ERROR: No aligned face images available!")
        print("OpenFace did not save aligned faces.")
        print("\nTo fix this, we need to either:")
        print("1. Modify OpenFace command to save aligned faces")
        print("2. Or reconstruct aligned faces from landmarks + original frames")
        return

    pyfhog_features = []
    comparison_results = []

    # Test on first 10 frames for quick validation
    num_test_frames = min(10, len(frame_indices))
    print(f"Testing on first {num_test_frames} frames...")

    for i in range(num_test_frames):
        frame_num = i + 1  # Aligned face images are 1-indexed (frame_det_00_000001.bmp, etc.)

        try:
            # Load aligned face
            aligned_face = load_aligned_face_from_openface(aligned_dir, frame_num)

            # Extract FHOG with pyfhog
            pyfhog_feat = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
            pyfhog_features.append(pyfhog_feat)

            # Compare
            result = compare_hog_features(
                openface_hog_features[i],
                pyfhog_feat,
                frame_num
            )

            if result:
                comparison_results.append(result)
                print(f"Frame {frame_num:4d}: r={result['correlation']:.6f}, "
                      f"mean_diff={result['mean_abs_diff']:.6f}, "
                      f"max_diff={result['max_abs_diff']:.6f}")

        except Exception as e:
            print(f"Frame {frame_num}: Error - {e}")

    # Step 4: Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    if comparison_results:
        correlations = [r['correlation'] for r in comparison_results]
        mean_corr = np.mean(correlations)
        min_corr = np.min(correlations)

        print(f"\nCorrelation Statistics:")
        print(f"  Mean correlation: r = {mean_corr:.6f}")
        print(f"  Min correlation:  r = {min_corr:.6f}")
        print(f"  Max correlation:  r = {np.max(correlations):.6f}")

        if mean_corr > 0.9999:
            print(f"\n✅ SUCCESS! pyfhog produces near-identical features to OpenFace 2.2")
            print(f"   Average correlation: r = {mean_corr:.6f} (> 0.9999)")
        elif mean_corr > 0.999:
            print(f"\n⚠️  GOOD: pyfhog is very close to OpenFace 2.2")
            print(f"   Average correlation: r = {mean_corr:.6f}")
            print(f"   Minor differences may exist but should not affect AU prediction")
        else:
            print(f"\n❌ ISSUE: pyfhog differs significantly from OpenFace 2.2")
            print(f"   Average correlation: r = {mean_corr:.6f} (< 0.999)")
            print(f"   This needs investigation!")
    else:
        print("\n❌ No comparisons completed!")

    # Keep output directory for diagnostics
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == "__main__":
    main()
