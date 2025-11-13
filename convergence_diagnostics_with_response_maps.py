#!/usr/bin/env python3
"""
Convergence diagnostics with response map visualization.

Tests pyMTCNN bbox correction and analyzes CLNF convergence with visual debugging.
"""

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
from pyclnf import CLNF

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50
OUTPUT_DIR = Path('test_output/convergence_diagnostics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def visualize_bbox_comparison(frame, pymtcnn_bbox, corrected_bbox):
    """Visualize pyMTCNN bbox before/after correction."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Original pyMTCNN bbox
    ax = axes[0]
    ax.imshow(frame_rgb)
    x1, y1, x2, y2 = pymtcnn_bbox
    w, h = x2 - x1, y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='red',
                            facecolor='none', label='Raw pyMTCNN')
    ax.add_patch(rect)
    ax.set_title(f'Raw pyMTCNN bbox\n(x={x1}, y={y1}, w={w}, h={h})', fontsize=12)
    ax.axis('off')

    # Corrected bbox
    ax = axes[1]
    ax.imshow(frame_rgb)
    x1, y1, x2, y2 = corrected_bbox
    w, h = x2 - x1, y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='green',
                            facecolor='none', label='Corrected (pyMTCNN-specific)')
    ax.add_patch(rect)
    ax.set_title(f'Corrected bbox (pyMTCNN-specific)\n(x={x1}, y={y1}, w={w}, h={h})', fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'bbox_correction_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def test_convergence_with_bbox_correction(frame):
    """Test convergence with and without bbox correction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("="*80)
    print("TESTING pyMTCNN WITH DERIVED BBOX CORRECTION")
    print("="*80)
    print()

    # Initialize pyMTCNN detector
    pyfaceau_dir = Path("pyfaceau/pyfaceau/detectors")
    mtcnn_weights = pyfaceau_dir / "openface_mtcnn_weights.pth"

    detector = OpenFaceMTCNN(
        weights_path=str(mtcnn_weights),
        min_face_size=60,
        thresholds=[0.3, 0.4, 0.4]
    )

    # Detect face with pyMTCNN
    bboxes, _ = detector.detect(frame, return_landmarks=False)

    if bboxes is None or len(bboxes) == 0:
        print("ERROR: No face detected!")
        return None

    pymtcnn_bbox = bboxes[0].astype(int)  # [x1, y1, x2, y2]
    print(f"Raw pyMTCNN bbox: {pymtcnn_bbox}")

    # The bbox correction is applied internally by OpenFaceMTCNN.detect()
    # For testing, we'll also get the internal corrected bbox
    x1, y1, x2, y2 = pymtcnn_bbox
    w, h = x2 - x1, y2 - y1

    # Apply correction manually to show the difference
    corrected_x = int(x1 + w * -0.0082)
    corrected_y = int(y1 + h * 0.2239)
    corrected_w = int(w * 0.9807)
    corrected_h = int(h * 0.7571)
    corrected_bbox = np.array([corrected_x, corrected_y,
                               corrected_x + corrected_w, corrected_y + corrected_h])

    print(f"Corrected bbox (pyMTCNN-specific): {corrected_bbox}")
    print()

    # Visualize bbox comparison
    bbox_viz_file = visualize_bbox_comparison(frame, pymtcnn_bbox, corrected_bbox)
    print(f"Saved bbox comparison: {bbox_viz_file}")
    print()

    # Test convergence with both bboxes
    print("-" * 80)
    print("CONVERGENCE TEST: Raw vs Corrected bbox")
    print("-" * 80)
    print()

    # Convert bboxes to (x, y, w, h) format for CLNF
    raw_bbox_xywh = (pymtcnn_bbox[0], pymtcnn_bbox[1],
                     pymtcnn_bbox[2] - pymtcnn_bbox[0],
                     pymtcnn_bbox[3] - pymtcnn_bbox[1])
    corrected_bbox_xywh = (corrected_bbox[0], corrected_bbox[1],
                           corrected_bbox[2] - corrected_bbox[0],
                           corrected_bbox[3] - corrected_bbox[1])

    # Test with raw bbox
    print("Testing with RAW pyMTCNN bbox...")
    clnf1 = CLNF(model_dir='pyclnf/models', max_iterations=20)
    landmarks1, info1 = clnf1.fit(gray, raw_bbox_xywh, return_params=True)
    print(f"  Converged: {info1['converged']}")
    print(f"  Iterations: {info1['iterations']}")
    print(f"  Final update: {info1['final_update']:.6f}")
    print(f"  Ratio vs target (0.005): {info1['final_update'] / 0.005:.1f}x")
    print()

    # Test with corrected bbox
    print("Testing with CORRECTED pyMTCNN bbox (pyMTCNN-specific coefficients)...")
    clnf2 = CLNF(model_dir='pyclnf/models', max_iterations=20)
    landmarks2, info2 = clnf2.fit(gray, corrected_bbox_xywh, return_params=True)
    print(f"  Converged: {info2['converged']}")
    print(f"  Iterations: {info2['iterations']}")
    print(f"  Final update: {info2['final_update']:.6f}")
    print(f"  Ratio vs target (0.005): {info2['final_update'] / 0.005:.1f}x")
    print()

    # Compare
    improvement = (info1['final_update'] - info2['final_update']) / info1['final_update'] * 100
    print("-" * 80)
    print("RESULT:")
    print("-" * 80)
    print(f"Convergence improvement with correction: {improvement:+.1f}%")

    if improvement > 5:
        print("✓ pyMTCNN-specific correction SIGNIFICANTLY improves convergence!")
    elif improvement > 0:
        print("~ pyMTCNN-specific correction slightly improves convergence")
    else:
        print("✗ Correction does not improve convergence")
    print()

    return {
        'raw_bbox': pymtcnn_bbox,
        'corrected_bbox': corrected_bbox,
        'raw_info': info1,
        'corrected_info': info2,
        'improvement_pct': improvement
    }


def analyze_convergence_internals(frame, bbox):
    """Analyze CLNF convergence internals and response maps."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("="*80)
    print("ANALYZING CONVERGENCE INTERNALS")
    print("="*80)
    print()

    # Convert bbox to (x, y, w, h) format
    bbox_xywh = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])

    # Run CLNF with debug info
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=10)
    landmarks, info = clnf.fit(gray, bbox_xywh, return_params=True)

    print(f"Bbox: {bbox_xywh}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final update: {info['final_update']:.6f}")
    print(f"Target: 0.005")
    print(f"Ratio: {info['final_update'] / 0.005:.1f}x target")
    print()

    # Visualize landmarks
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)

    # Draw bbox
    x, y, w, h = bbox_xywh
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow',
                            facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Draw landmarks
    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=20, alpha=0.8)

        # Draw landmark indices for reference
        for i, (lx, ly) in enumerate(landmarks):
            if i % 5 == 0:  # Label every 5th landmark to avoid clutter
                ax.text(lx, ly, str(i), fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

    ax.set_title(f'CLNF Landmarks (converged={info["converged"]}, '
                f'final_update={info["final_update"]:.3f})', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'convergence_landmarks.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved landmark visualization: {output_file}")
    print()

    return info


def main():
    print("\n" + "="*80)
    print("CONVERGENCE DIAGNOSTICS WITH pyMTCNN BBOX CORRECTION")
    print("="*80)
    print()

    # Load test frame
    print(f"Loading frame {FRAME_NUM} from {VIDEO_PATH}...")
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    print(f"Frame shape: {frame.shape}")
    print()

    # Test bbox correction and convergence
    results = test_convergence_with_bbox_correction(frame)

    if results:
        # Analyze convergence internals with corrected bbox
        print()
        analyze_convergence_internals(frame, results['corrected_bbox'])

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()
    print("Files generated:")
    print("  - bbox_correction_comparison.png: Shows raw vs corrected bbox")
    print("  - convergence_landmarks.png: CLNF landmark results")
    print()

    if results:
        print(f"Convergence improvement: {results['improvement_pct']:+.1f}%")
        print(f"Raw bbox final_update: {results['raw_info']['final_update']:.6f}")
        print(f"Corrected bbox final_update: {results['corrected_info']['final_update']:.6f}")

    print()
    print("="*80)
    print("NEXT STEPS FOR CONVERGENCE DEBUGGING:")
    print("="*80)
    print()
    print("1. Response map analysis - Compare peak locations with OpenFace C++")
    print("2. Mean-shift vector analysis - Check if magnitudes decrease < 1px")
    print("3. Jacobian verification - Ensure no zero rows/columns")
    print("4. Parameter update tracking - Should decrease monotonically to < 0.005")
    print()
    print("See README_CONVERGENCE_DEBUG.md for detailed instrumentation points.")
    print()


if __name__ == "__main__":
    main()
