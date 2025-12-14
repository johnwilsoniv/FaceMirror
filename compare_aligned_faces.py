#!/usr/bin/env python3
"""Compare Python vs C++ aligned faces."""

import numpy as np
import cv2
import sys

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

print("=" * 60)
print("FACE ALIGNMENT COMPARISON: Python vs C++")
print("=" * 60)

# Load C++ aligned face
cpp_aligned_path = '/tmp/cpp_aligned_face_for_au.png'
cpp_aligned = cv2.imread(cpp_aligned_path)
print(f"C++ aligned: {cpp_aligned.shape}, range [{cpp_aligned.min()}, {cpp_aligned.max()}]")

# Load first frame of video
video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0422.MOV'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
print(f"Frame: {frame.shape}")

# Get landmarks using pyclnf
from pyclnf import CLNF
clnf = CLNF(convergence_profile='video', detector='pymtcnn',
            use_validator=False, use_eye_refinement=False)
landmarks, info = clnf.detect_and_fit(frame)
print(f"Landmarks shape: {landmarks.shape}")

# Get pose info for alignment
pose_tx = info['pose_tx']
pose_ty = info['pose_ty']
print(f"Pose: tx={pose_tx:.2f}, ty={pose_ty:.2f}")

# Align face using pyfaceau
from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.features.triangulation import TriangulationParser

# Load aligner with same settings as pipeline
aligner = OpenFace22FaceAligner(
    pdm_file='/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/weights/In-the-wild_aligned_PDM_68.txt',
    sim_scale=0.7,
    output_size=(112, 112)
)

# Load triangulation for masking
triangulation = TriangulationParser(
    '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/weights/tris_68_full.txt'
)

# Align with mask (like C++ AlignFaceMask)
py_aligned = aligner.align_face(frame, landmarks, pose_tx, pose_ty,
                                 apply_mask=True, triangulation=triangulation)
print(f"Python aligned: {py_aligned.shape}, range [{py_aligned.min()}, {py_aligned.max()}]")

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

# Resize if needed
if cpp_aligned.shape != py_aligned.shape:
    print(f"Shape mismatch! C++: {cpp_aligned.shape}, Py: {py_aligned.shape}")
    # Resize Python to match C++
    py_aligned = cv2.resize(py_aligned, (cpp_aligned.shape[1], cpp_aligned.shape[0]))

# Compute difference
diff = np.abs(cpp_aligned.astype(float) - py_aligned.astype(float))
print(f"\nPixel difference:")
print(f"  Mean: {diff.mean():.2f}")
print(f"  Max: {diff.max():.2f}")
print(f"  Std: {diff.std():.2f}")

# Per-channel
for i, ch in enumerate(['Blue', 'Green', 'Red']):
    ch_diff = diff[:,:,i]
    print(f"  {ch}: mean={ch_diff.mean():.2f}, max={ch_diff.max():.2f}")

# Correlation
cpp_flat = cpp_aligned.flatten().astype(float)
py_flat = py_aligned.flatten().astype(float)
corr = np.corrcoef(cpp_flat, py_flat)[0, 1]
print(f"\nPixel correlation: {corr:.4f}")

# Check non-zero (masked) regions
cpp_nonzero = (cpp_aligned.sum(axis=2) > 0)
py_nonzero = (py_aligned.sum(axis=2) > 0)
mask_match = (cpp_nonzero == py_nonzero).mean() * 100
print(f"Mask match: {mask_match:.1f}%")

# Count non-zero pixels
cpp_nz_count = cpp_nonzero.sum()
py_nz_count = py_nonzero.sum()
print(f"C++ non-zero pixels: {cpp_nz_count}")
print(f"Py  non-zero pixels: {py_nz_count}")

# Save for visual comparison
cv2.imwrite('/tmp/py_aligned_face.png', py_aligned)
cv2.imwrite('/tmp/alignment_diff.png', (diff * 2).clip(0, 255).astype(np.uint8))

# Create side-by-side comparison
comparison = np.hstack([cpp_aligned, py_aligned, (diff * 2).clip(0, 255).astype(np.uint8)])
cv2.imwrite('/tmp/alignment_comparison.png', comparison)
print(f"\nSaved comparison to /tmp/alignment_comparison.png")

# If significant difference, analyze where
if corr < 0.99:
    print("\n" + "=" * 60)
    print("ANALYZING DIFFERENCES")
    print("=" * 60)

    # Find regions with largest differences
    diff_gray = diff.mean(axis=2)

    # Top/bottom/left/right quadrants
    h, w = diff_gray.shape
    quadrants = {
        'Top-Left': diff_gray[:h//2, :w//2],
        'Top-Right': diff_gray[:h//2, w//2:],
        'Bottom-Left': diff_gray[h//2:, :w//2],
        'Bottom-Right': diff_gray[h//2:, w//2:]
    }

    print("\nDifference by quadrant:")
    for name, q in quadrants.items():
        print(f"  {name}: mean={q.mean():.2f}, max={q.max():.2f}")

    # Check if it's a translation issue
    # Try shifting Python image to match C++
    best_shift = (0, 0)
    best_corr = corr

    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx == 0 and dy == 0:
                continue
            # Shift Python image
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(py_aligned, M, (w, h))
            shifted_corr = np.corrcoef(cpp_flat, shifted.flatten().astype(float))[0, 1]
            if shifted_corr > best_corr:
                best_corr = shifted_corr
                best_shift = (dx, dy)

    if best_shift != (0, 0):
        print(f"\nBest shift to improve correlation: dx={best_shift[0]}, dy={best_shift[1]}")
        print(f"Correlation after shift: {best_corr:.4f} (was {corr:.4f})")
