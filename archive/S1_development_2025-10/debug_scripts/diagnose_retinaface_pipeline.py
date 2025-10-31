#!/usr/bin/env python3
"""
Diagnostic script to investigate RetinaFace behavior during AU extraction.

This script will:
1. Process a mirrored video with RetinaFace ENABLED (current behavior)
2. Process the same video with RetinaFace DISABLED (intended behavior)
3. Visualize bboxes and face crops being sent to MTL
4. Compare AU outputs from both modes
5. Log detailed pipeline information
"""

import os
import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openface_integration import OpenFace3Processor


def visualize_pipeline_comparison(video_path, output_dir, max_frames=30):
    """
    Run both pipeline modes and visualize the differences

    Args:
        video_path: Path to mirrored video
        output_dir: Directory for output visualizations
        max_frames: Number of frames to analyze
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RETINAFACE PIPELINE DIAGNOSTIC")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Output: {output_dir}")
    print()

    # ============================================================================
    # MODE 1: RetinaFace ENABLED (current behavior)
    # ============================================================================
    print("MODE 1: RetinaFace ENABLED (current pipeline)")
    print("-" * 80)

    processor_with_retinaface = OpenFace3Processor(
        device='cpu',
        skip_face_detection=False,  # CURRENT BEHAVIOR: RetinaFace runs
        debug_mode=True
    )

    # ============================================================================
    # MODE 2: RetinaFace DISABLED (intended behavior for mirrored videos)
    # ============================================================================
    print("\nMODE 2: RetinaFace DISABLED (intended for mirrored videos)")
    print("-" * 80)

    processor_without_retinaface = OpenFace3Processor(
        device='cpu',
        skip_face_detection=True,  # INTENDED BEHAVIOR: Skip RetinaFace
        debug_mode=True
    )

    # ============================================================================
    # Process frames and collect data
    # ============================================================================
    print(f"\nProcessing first {max_frames} frames...")
    print("=" * 80)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Storage for comparison
    results = {
        'frames': [],
        'mode1_bboxes': [],
        'mode1_crops': [],
        'mode1_aus': [],
        'mode2_bboxes': [],
        'mode2_crops': [],
        'mode2_aus': [],
    }

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Store original frame
        results['frames'].append(frame.copy())

        # ========================================================================
        # MODE 1: Process with RetinaFace
        # ========================================================================
        print(f"\nFrame {frame_idx} - MODE 1 (RetinaFace ENABLED):")

        # Manually call the detection logic to see what's happening
        if processor_with_retinaface.skip_face_detection:
            print("  ERROR: skip_face_detection is True, but should be False!")
            mode1_bbox = None
            mode1_crop = frame.copy()
        else:
            # Run RetinaFace
            dets = processor_with_retinaface.preprocess_image(frame)

            if dets is None or len(dets) == 0:
                print("  ⚠ RetinaFace: NO FACE DETECTED")
                mode1_bbox = None
                mode1_crop = None
            else:
                det = dets[0]
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                confidence = float(det[4])

                print(f"  ✓ RetinaFace: Face detected at [{x1}, {y1}, {x2}, {y2}]")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Bbox size: {x2-x1}x{y2-y1}")

                # Get frame dimensions for comparison
                h, w = frame.shape[:2]
                coverage = ((x2-x1) * (y2-y1)) / (w * h) * 100
                print(f"    Frame coverage: {coverage:.1f}%")

                mode1_bbox = [x1, y1, x2, y2, confidence]
                mode1_crop = frame[y1:y2, x1:x2]

        # Extract AUs with MODE 1
        if mode1_crop is not None and mode1_crop.size > 0:
            emotion, gaze, au_output = processor_with_retinaface.multitask_model.predict(mode1_crop)
            mode1_aus = au_output.cpu().numpy().flatten()
            print(f"  AUs (8D): [{', '.join([f'{x:.3f}' for x in mode1_aus])}]")
        else:
            mode1_aus = np.full(8, np.nan)
            print(f"  AUs: [ALL NaN - no face detected]")

        results['mode1_bboxes'].append(mode1_bbox)
        results['mode1_crops'].append(mode1_crop)
        results['mode1_aus'].append(mode1_aus)

        # ========================================================================
        # MODE 2: Process without RetinaFace (full frame)
        # ========================================================================
        print(f"Frame {frame_idx} - MODE 2 (RetinaFace DISABLED):")

        if not processor_without_retinaface.skip_face_detection:
            print("  ERROR: skip_face_detection is False, but should be True!")

        # Use full frame
        h, w = frame.shape[:2]
        mode2_bbox = [0, 0, w, h, 1.0]
        mode2_crop = frame.copy()

        print(f"  ✓ Full frame: [{0}, {0}, {w}, {h}]")
        print(f"    Frame size: {w}x{h}")

        # Extract AUs with MODE 2
        emotion, gaze, au_output = processor_without_retinaface.multitask_model.predict(mode2_crop)
        mode2_aus = au_output.cpu().numpy().flatten()
        print(f"  AUs (8D): [{', '.join([f'{x:.3f}' for x in mode2_aus])}]")

        results['mode2_bboxes'].append(mode2_bbox)
        results['mode2_crops'].append(mode2_crop)
        results['mode2_aus'].append(mode2_aus)

        # Print comparison
        if mode1_aus is not None and not np.any(np.isnan(mode1_aus)):
            au_diff = np.abs(mode1_aus - mode2_aus)
            print(f"\n  AU Difference (L1): {np.sum(au_diff):.3f}")
            print(f"    Max difference: {np.max(au_diff):.3f}")
            print(f"    Mean difference: {np.mean(au_diff):.3f}")

    cap.release()

    # ============================================================================
    # Visualize Results
    # ============================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Create comparison visualizations for select frames
    visualization_frames = [0, max_frames // 4, max_frames // 2, 3 * max_frames // 4]

    for frame_idx in visualization_frames:
        if frame_idx >= len(results['frames']):
            continue

        print(f"Creating visualization for frame {frame_idx}...")

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Original frame with bboxes
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        frame = results['frames'][frame_idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Original frame
        ax1.imshow(frame_rgb)
        ax1.set_title(f"Original Frame {frame_idx}", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Frame with MODE 1 bbox (RetinaFace)
        frame_mode1 = frame_rgb.copy()
        if results['mode1_bboxes'][frame_idx] is not None:
            bbox = results['mode1_bboxes'][frame_idx]
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(frame_mode1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            ax2.text(x1, y1-10, f"Conf: {conf:.2f}", color='red', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.imshow(frame_mode1)
        ax2.set_title("MODE 1: RetinaFace Bbox", fontsize=12, fontweight='bold', color='red')
        ax2.axis('off')

        # Frame with MODE 2 bbox (full frame)
        frame_mode2 = frame_rgb.copy()
        bbox = results['mode2_bboxes'][frame_idx]
        x1, y1, x2, y2, conf = bbox
        cv2.rectangle(frame_mode2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        ax3.imshow(frame_mode2)
        ax3.set_title("MODE 2: Full Frame Bbox", fontsize=12, fontweight='bold', color='green')
        ax3.axis('off')

        # Row 2: Face crops sent to MTL
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        ax4.axis('off')  # Empty for symmetry

        # MODE 1 crop
        if results['mode1_crops'][frame_idx] is not None:
            crop1 = cv2.cvtColor(results['mode1_crops'][frame_idx], cv2.COLOR_BGR2RGB)
            ax5.imshow(crop1)
            ax5.set_title(f"MODE 1 Crop\n{crop1.shape[1]}x{crop1.shape[0]}",
                         fontsize=11, color='red')
        else:
            ax5.text(0.5, 0.5, "NO FACE\nDETECTED", ha='center', va='center',
                    fontsize=14, color='red', transform=ax5.transAxes)
            ax5.set_title("MODE 1 Crop", fontsize=11, color='red')
        ax5.axis('off')

        # MODE 2 crop
        crop2 = cv2.cvtColor(results['mode2_crops'][frame_idx], cv2.COLOR_BGR2RGB)
        ax6.imshow(crop2)
        ax6.set_title(f"MODE 2 Crop\n{crop2.shape[1]}x{crop2.shape[0]}",
                     fontsize=11, color='green')
        ax6.axis('off')

        # Row 3: AU comparison bar chart
        ax7 = fig.add_subplot(gs[2, :])

        au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25']
        x_pos = np.arange(len(au_labels))
        width = 0.35

        mode1_aus = results['mode1_aus'][frame_idx]
        mode2_aus = results['mode2_aus'][frame_idx]

        bars1 = ax7.bar(x_pos - width/2, mode1_aus, width, label='MODE 1 (RetinaFace)',
                       color='red', alpha=0.7)
        bars2 = ax7.bar(x_pos + width/2, mode2_aus, width, label='MODE 2 (Full Frame)',
                       color='green', alpha=0.7)

        ax7.set_xlabel('Action Units', fontsize=12, fontweight='bold')
        ax7.set_ylabel('AU Intensity', fontsize=12, fontweight='bold')
        ax7.set_title(f'AU Comparison - Frame {frame_idx}', fontsize=13, fontweight='bold')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(au_labels)
        ax7.legend(fontsize=11)
        ax7.grid(axis='y', alpha=0.3)

        # Add difference annotations
        for i, (au1, au2) in enumerate(zip(mode1_aus, mode2_aus)):
            if not np.isnan(au1) and not np.isnan(au2):
                diff = abs(au1 - au2)
                if diff > 0.1:  # Only annotate significant differences
                    max_val = max(au1, au2)
                    ax7.text(i, max_val + 0.05, f'Δ{diff:.2f}',
                            ha='center', va='bottom', fontsize=9, color='red')

        plt.suptitle(f"RetinaFace Pipeline Diagnostic - Frame {frame_idx}",
                    fontsize=14, fontweight='bold')

        # Save figure
        output_path = output_dir / f"diagnostic_frame_{frame_idx:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

    # ============================================================================
    # Create temporal AU comparison plot
    # ============================================================================
    print("\nCreating temporal AU comparison plot...")

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Temporal AU Comparison: RetinaFace vs Full Frame',
                fontsize=14, fontweight='bold')

    au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25']

    mode1_aus_array = np.array(results['mode1_aus'])
    mode2_aus_array = np.array(results['mode2_aus'])

    frames = np.arange(len(results['frames']))

    for idx, (ax, au_label) in enumerate(zip(axes.flat, au_labels)):
        mode1_values = mode1_aus_array[:, idx]
        mode2_values = mode2_aus_array[:, idx]

        ax.plot(frames, mode1_values, 'r-', label='MODE 1 (RetinaFace)', linewidth=2, alpha=0.7)
        ax.plot(frames, mode2_values, 'g-', label='MODE 2 (Full Frame)', linewidth=2, alpha=0.7)

        ax.set_title(au_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Intensity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Compute statistics
        if not np.all(np.isnan(mode1_values)):
            mean_diff = np.nanmean(np.abs(mode1_values - mode2_values))
            ax.text(0.98, 0.98, f'Mean Δ: {mean_diff:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / "temporal_au_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # ============================================================================
    # Print Summary Statistics
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nMODE 1 (RetinaFace ENABLED):")
    print("-" * 40)

    mode1_failed = sum(1 for bbox in results['mode1_bboxes'] if bbox is None)
    print(f"  Frames with no detection: {mode1_failed}/{len(results['frames'])}")

    if mode1_failed < len(results['frames']):
        valid_bboxes = [b for b in results['mode1_bboxes'] if b is not None]
        bbox_sizes = [(b[2]-b[0]) * (b[3]-b[1]) for b in valid_bboxes]
        print(f"  Average bbox area: {np.mean(bbox_sizes):.0f} pixels")

        frame_area = results['frames'][0].shape[0] * results['frames'][0].shape[1]
        coverages = [size / frame_area * 100 for size in bbox_sizes]
        print(f"  Average frame coverage: {np.mean(coverages):.1f}%")
        print(f"  Coverage range: {np.min(coverages):.1f}% - {np.max(coverages):.1f}%")

    print("\nMODE 2 (RetinaFace DISABLED):")
    print("-" * 40)
    print(f"  Frames processed: {len(results['frames'])}/{len(results['frames'])}")
    print(f"  Frame coverage: 100.0% (always uses full frame)")

    print("\nAU COMPARISON:")
    print("-" * 40)

    mode1_aus_array = np.array(results['mode1_aus'])
    mode2_aus_array = np.array(results['mode2_aus'])

    for idx, au_label in enumerate(au_labels):
        mode1_values = mode1_aus_array[:, idx]
        mode2_values = mode2_aus_array[:, idx]

        # Skip if MODE 1 has all NaN
        if np.all(np.isnan(mode1_values)):
            print(f"  {au_label}: MODE 1 all NaN (no faces detected)")
            continue

        mean_diff = np.nanmean(np.abs(mode1_values - mode2_values))
        max_diff = np.nanmax(np.abs(mode1_values - mode2_values))

        mode1_mean = np.nanmean(mode1_values)
        mode2_mean = np.nanmean(mode2_values)

        print(f"  {au_label}:")
        print(f"    MODE 1 mean: {mode1_mean:.3f}")
        print(f"    MODE 2 mean: {mode2_mean:.3f}")
        print(f"    Mean absolute difference: {mean_diff:.3f}")
        print(f"    Max absolute difference: {max_diff:.3f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(f"Visualizations saved to: {output_dir}")
    print("\nCONCLUSION:")
    print("  If MODE 1 shows:")
    print("    - Small bboxes (low frame coverage) → RetinaFace is cropping too tight")
    print("    - Different AU values → Face crop quality affects AU extraction")
    print("    - Failed detections → RetinaFace can't handle mirrored videos")
    print("\n  Solution: Use skip_face_detection=True for mirrored videos")
    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose RetinaFace pipeline behavior')
    parser.add_argument('video_path', type=str,
                       help='Path to mirrored video file')
    parser.add_argument('--output-dir', type=str,
                       default='./retinaface_diagnostic_output',
                       help='Output directory for visualizations')
    parser.add_argument('--max-frames', type=int, default=30,
                       help='Number of frames to analyze (default: 30)')

    args = parser.parse_args()

    visualize_pipeline_comparison(args.video_path, args.output_dir, args.max_frames)
