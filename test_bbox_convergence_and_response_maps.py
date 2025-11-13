#!/usr/bin/env python3
"""
Comprehensive test comparing convergence AND response map quality across bbox sources.

Tests three bbox sources:
1. Hardcoded test bbox (241, 555, 532, 532)
2. RetinaFace bbox (production detector)
3. pyMTCNN bbox (OpenFace MTCNN weights)

For each bbox source, measures:
- Convergence performance (final update magnitude)
- Response map quality (peak offsets, SNR)
- Initial vs final peak alignment
"""

import cv2
import numpy as np
from pyclnf import CLNF
import pandas as pd
from pathlib import Path

def analyze_response_map_quality(response_map, window_size):
    """Analyze response map sharpness and quality."""
    center = (window_size - 1) / 2.0
    peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
    peak_y, peak_x = peak_idx

    peak_value = response_map[peak_y, peak_x]
    offset_x = peak_x - center
    offset_y = peak_y - center
    peak_dist = np.sqrt(offset_x**2 + offset_y**2)

    variance = np.var(response_map)

    mask = np.ones_like(response_map, dtype=bool)
    mask[peak_y, peak_x] = False
    background_mean = response_map[mask].mean()
    snr = peak_value / (background_mean + 1e-10)

    return {
        'peak_value': peak_value,
        'peak_offset_x': offset_x,
        'peak_offset_y': offset_y,
        'peak_dist': peak_dist,
        'variance': variance,
        'snr': snr
    }

def analyze_all_response_maps(gray, face_bbox, clnf, window_size=11):
    """Analyze response map quality for all landmarks given a bbox."""
    params = clnf.pdm.init_params(face_bbox)
    landmarks_init = clnf.pdm.params_to_landmarks_2d(params)

    scale_idx = clnf.window_to_scale[window_size]
    patch_scale = clnf.patch_scaling[scale_idx]
    patch_experts = clnf._get_patch_experts(view_idx=0, scale=patch_scale)

    qualities = []
    for lm_idx in range(68):
        if lm_idx not in patch_experts:
            continue

        patch_expert = patch_experts[lm_idx]
        lm_x, lm_y = landmarks_init[lm_idx]

        response_map = clnf.optimizer._compute_response_map(
            gray, lm_x, lm_y, patch_expert, window_size,
            sim_img_to_ref=None,
            sim_ref_to_img=None,
            sigma_components=None
        )

        quality = analyze_response_map_quality(response_map, window_size)
        quality['landmark'] = lm_idx
        qualities.append(quality)

    return pd.DataFrame(qualities)

def test_bbox_source(gray, face_bbox, bbox_name):
    """Test convergence and response map quality for a single bbox source."""
    print(f"\n{'='*80}")
    print(f"TESTING: {bbox_name}")
    print(f"{'='*80}")
    print(f"Bbox: {face_bbox}")
    x, y, w, h = face_bbox
    aspect_ratio = w / h
    print(f"  x={x}, y={y}, width={w}, height={h}")
    print(f"  Aspect ratio: {aspect_ratio:.3f}")
    print()

    # Analyze INITIAL response map quality
    print("Analyzing INITIAL response map quality...")
    clnf_init = CLNF(model_dir='pyclnf/models', max_iterations=1)
    df_initial = analyze_all_response_maps(gray, face_bbox, clnf_init)

    initial_mean_offset = df_initial['peak_dist'].mean()
    initial_median_offset = df_initial['peak_dist'].median()
    initial_large_offset_pct = (df_initial['peak_dist'] > 2.0).sum() / len(df_initial) * 100

    print(f"  Initial peak offsets:")
    print(f"    Mean:   {initial_mean_offset:.2f} pixels")
    print(f"    Median: {initial_median_offset:.2f} pixels")
    print(f"    >2px:   {initial_large_offset_pct:.1f}%")
    print()

    # Test convergence
    print("Running convergence test (20 iterations)...")
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=20)
    landmarks, info = clnf.fit(gray, face_bbox, return_params=True)

    converged = info['converged']
    iterations = info['iterations']
    final_update = info['final_update']

    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Final update: {final_update:.6f} (target: 0.005)")
    print(f"  Ratio to target: {final_update / 0.005:.1f}x")
    print()

    # Analyze FINAL response map quality
    print("Analyzing FINAL response map quality...")
    params_final = info['params']
    landmarks_final = clnf.pdm.params_to_landmarks_2d(params_final)

    # Create new instance to avoid cached state
    clnf_final = CLNF(model_dir='pyclnf/models', max_iterations=1)

    # Manually compute response maps at final positions
    window_size = 11
    scale_idx = clnf_final.window_to_scale[window_size]
    patch_scale = clnf_final.patch_scaling[scale_idx]
    patch_experts = clnf_final._get_patch_experts(view_idx=0, scale=patch_scale)

    final_qualities = []
    for lm_idx in range(68):
        if lm_idx not in patch_experts:
            continue

        patch_expert = patch_experts[lm_idx]
        lm_x, lm_y = landmarks_final[lm_idx]

        response_map = clnf_final.optimizer._compute_response_map(
            gray, lm_x, lm_y, patch_expert, window_size,
            sim_img_to_ref=None,
            sim_ref_to_img=None,
            sigma_components=None
        )

        quality = analyze_response_map_quality(response_map, window_size)
        quality['landmark'] = lm_idx
        final_qualities.append(quality)

    df_final = pd.DataFrame(final_qualities)

    final_mean_offset = df_final['peak_dist'].mean()
    final_median_offset = df_final['peak_dist'].median()
    final_large_offset_pct = (df_final['peak_dist'] > 2.0).sum() / len(df_final) * 100

    print(f"  Final peak offsets:")
    print(f"    Mean:   {final_mean_offset:.2f} pixels")
    print(f"    Median: {final_median_offset:.2f} pixels")
    print(f"    >2px:   {final_large_offset_pct:.1f}%")
    print()

    # Compare improvement
    offset_improvement = initial_mean_offset - final_mean_offset
    offset_improvement_pct = (offset_improvement / initial_mean_offset) * 100

    print(f"  Peak alignment improvement:")
    print(f"    Initial → Final: {initial_mean_offset:.2f} → {final_mean_offset:.2f} pixels")
    print(f"    Improvement: {offset_improvement:.2f} pixels ({offset_improvement_pct:+.1f}%)")
    print()

    return {
        'bbox_name': bbox_name,
        'bbox': face_bbox,
        'aspect_ratio': aspect_ratio,
        'converged': converged,
        'iterations': iterations,
        'final_update': final_update,
        'ratio_to_target': final_update / 0.005,
        'initial_mean_offset': initial_mean_offset,
        'initial_median_offset': initial_median_offset,
        'initial_large_offset_pct': initial_large_offset_pct,
        'final_mean_offset': final_mean_offset,
        'final_median_offset': final_median_offset,
        'final_large_offset_pct': final_large_offset_pct,
        'offset_improvement': offset_improvement,
        'offset_improvement_pct': offset_improvement_pct,
    }

def main():
    print("="*80)
    print("COMPREHENSIVE BBOX SOURCE COMPARISON")
    print("Convergence + Response Map Quality")
    print("="*80)

    # Load test frame
    video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define bbox sources
    bboxes = {
        'Hardcoded (test)': (241, 555, 532, 532),
        'RetinaFace (prod)': (288, 553, 424, 546),
        'pyMTCNN (OpenFace)': (270, 706, 437, 414),
    }

    # Test each bbox source
    results = []
    for bbox_name, bbox in bboxes.items():
        result = test_bbox_source(gray, bbox, bbox_name)
        results.append(result)

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print()

    print("CONVERGENCE PERFORMANCE:")
    print(f"{'Bbox Source':<25} {'Final Update':>15} {'Ratio to Target':>18} {'Iterations':>12}")
    print("-"*80)
    for r in results:
        print(f"{r['bbox_name']:<25} {r['final_update']:>15.6f} {r['ratio_to_target']:>17.1f}x {r['iterations']:>12d}")
    print()

    print("INITIAL RESPONSE MAP QUALITY:")
    print(f"{'Bbox Source':<25} {'Mean Offset':>15} {'Median Offset':>15} {'>2px %':>12}")
    print("-"*80)
    for r in results:
        print(f"{r['bbox_name']:<25} {r['initial_mean_offset']:>14.2f}px {r['initial_median_offset']:>14.2f}px {r['initial_large_offset_pct']:>11.1f}%")
    print()

    print("FINAL RESPONSE MAP QUALITY:")
    print(f"{'Bbox Source':<25} {'Mean Offset':>15} {'Median Offset':>15} {'>2px %':>12}")
    print("-"*80)
    for r in results:
        print(f"{r['bbox_name']:<25} {r['final_mean_offset']:>14.2f}px {r['final_median_offset']:>14.2f}px {r['final_large_offset_pct']:>11.1f}%")
    print()

    print("PEAK ALIGNMENT IMPROVEMENT:")
    print(f"{'Bbox Source':<25} {'Improvement':>15} {'% Change':>15}")
    print("-"*80)
    for r in results:
        print(f"{r['bbox_name']:<25} {r['offset_improvement']:>14.2f}px {r['offset_improvement_pct']:>14.1f}%")
    print()

    # Find best performer
    best_convergence = min(results, key=lambda r: r['final_update'])
    best_initial_alignment = min(results, key=lambda r: r['initial_mean_offset'])
    best_final_alignment = min(results, key=lambda r: r['final_mean_offset'])
    best_improvement = max(results, key=lambda r: r['offset_improvement_pct'])

    print("="*80)
    print("RANKINGS")
    print("="*80)
    print(f"Best convergence:        {best_convergence['bbox_name']} ({best_convergence['final_update']:.6f})")
    print(f"Best initial alignment:  {best_initial_alignment['bbox_name']} ({best_initial_alignment['initial_mean_offset']:.2f}px)")
    print(f"Best final alignment:    {best_final_alignment['bbox_name']} ({best_final_alignment['final_mean_offset']:.2f}px)")
    print(f"Best improvement:        {best_improvement['bbox_name']} ({best_improvement['offset_improvement_pct']:+.1f}%)")
    print()

    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    # Check if hardcoded is best (which would be suspicious)
    if best_convergence['bbox_name'] == 'Hardcoded (test)':
        print("⚠️  WARNING: Hardcoded bbox performs best for convergence")
        print("   This confirms our tests have been overly optimistic!")
        print("   Production performance will be worse.")
        print()

    # Compare production detectors
    retinaface_result = next(r for r in results if r['bbox_name'] == 'RetinaFace (prod)')
    pymtcnn_result = next(r for r in results if r['bbox_name'] == 'pyMTCNN (OpenFace)')

    print("PRODUCTION DETECTOR COMPARISON (RetinaFace vs pyMTCNN):")
    print()
    print(f"Convergence:")
    print(f"  RetinaFace:  {retinaface_result['final_update']:.6f} ({retinaface_result['ratio_to_target']:.1f}x target)")
    print(f"  pyMTCNN:     {pymtcnn_result['final_update']:.6f} ({pymtcnn_result['ratio_to_target']:.1f}x target)")

    if retinaface_result['final_update'] < pymtcnn_result['final_update']:
        winner = 'RetinaFace'
        diff_pct = (pymtcnn_result['final_update'] / retinaface_result['final_update'] - 1) * 100
        print(f"  → RetinaFace converges {diff_pct:.1f}% better")
    else:
        winner = 'pyMTCNN'
        diff_pct = (retinaface_result['final_update'] / pymtcnn_result['final_update'] - 1) * 100
        print(f"  → pyMTCNN converges {diff_pct:.1f}% better")
    print()

    print(f"Initial peak alignment:")
    print(f"  RetinaFace:  {retinaface_result['initial_mean_offset']:.2f}px mean offset")
    print(f"  pyMTCNN:     {pymtcnn_result['initial_mean_offset']:.2f}px mean offset")

    if retinaface_result['initial_mean_offset'] < pymtcnn_result['initial_mean_offset']:
        print(f"  → RetinaFace has better initial alignment")
    else:
        print(f"  → pyMTCNN has better initial alignment")
    print()

    print("RECOMMENDATION:")
    if winner == 'RetinaFace':
        print(f"✓ Continue using RetinaFace detector (current production)")
        print(f"  - Better convergence than pyMTCNN")
        print(f"  - CoreML accelerated for performance")
    else:
        print(f"✓ Consider switching to pyMTCNN detector")
        print(f"  - Better convergence than RetinaFace")
        print(f"  - Closer to OpenFace C++ behavior")
        print(f"  - May require performance optimization")
    print()

    # Check if poor initial alignment is the bottleneck
    for r in results:
        if r['initial_large_offset_pct'] > 80:
            print(f"⚠️  {r['bbox_name']}: {r['initial_large_offset_pct']:.1f}% landmarks misaligned >2px initially")
            print(f"   → Poor initialization is PRIMARY convergence bottleneck")
            print(f"   → Need to fix initialization formula for this bbox format")
            print()

    print("="*80)

if __name__ == "__main__":
    main()
