#!/usr/bin/env python3
"""
Test aspect-ratio-aware initialization formula.

Current formula (empirical):
  scale = width / 200.0
  ty = y + height/2 + 54

This works for square bboxes (aspect ~1.0) but fails for:
  - Portrait bboxes (aspect <1.0, like RetinaFace 0.777)
  - Wider bboxes (aspect >1.0, like pyMTCNN 1.056)

OpenFace formula (from PDM.cpp:219-231):
  model_width = max_x - min_x
  model_height = max_y - min_y
  scale = (width/model_width + height/model_height) / 2.0
  tx = x + width/2 - scale * (min_x + max_x)/2
  ty = y + height/2 - scale * (min_y + max_y)/2

Hypothesis:
  OpenFace formula should work better for all aspect ratios.
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core.pdm import PDM

def openface_style_init(pdm, face_bbox):
    """
    Initialize parameters using OpenFace-style formula.

    Based on OpenFace PDM.cpp:193-231.
    """
    x, y, width, height = face_bbox

    # Get mean shape from PDM
    mean_shape_3d = pdm.mean_shape.reshape(-1, 3).T  # Shape: (3, 68)

    # With zero rotation, shape is just mean_shape rotated by identity
    rotation = np.array([0.0, 0.0, 0.0])
    R = cv2.Rodrigues(rotation)[0]  # 3x3 rotation matrix

    # Rotate shape (identity rotation doesn't change it)
    rotated_shape = R @ mean_shape_3d  # (3, 68)

    # Find bounding box of model
    min_x = rotated_shape[0, :].min()
    max_x = rotated_shape[0, :].max()
    min_y = rotated_shape[1, :].min()
    max_y = rotated_shape[1, :].max()

    model_width = abs(max_x - min_x)
    model_height = abs(max_y - min_y)
    model_aspect = model_width / model_height
    bbox_aspect = width / height

    # OpenFace formula: average of width and height scaling
    scale = ((width / model_width) + (height / model_height)) / 2.0

    # Translation with correction for model center offset
    tx = x + width / 2.0 - scale * (min_x + max_x) / 2.0
    ty = y + height / 2.0 - scale * (min_y + max_y) / 2.0

    # Create parameter vector
    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[1:4] = rotation
    params[4] = tx
    params[5] = ty

    return params, {
        'model_width': model_width,
        'model_height': model_height,
        'model_aspect': model_aspect,
        'bbox_aspect': bbox_aspect,
        'scale': scale,
        'tx': tx,
        'ty': ty
    }

def empirical_init(pdm, face_bbox):
    """Current empirical initialization (only works for square bboxes)."""
    x, y, width, height = face_bbox

    scale = width / 200.0
    tx = x + width / 2.0
    ty = y + height / 2.0 + 54.0

    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[4] = tx
    params[5] = ty

    return params, {
        'scale': scale,
        'tx': tx,
        'ty': ty
    }

def test_initialization(gray, face_bbox, bbox_name):
    """Test both initialization methods and compare convergence."""
    print(f"\n{'='*80}")
    print(f"TESTING: {bbox_name}")
    print(f"Bbox: {face_bbox}")
    x, y, w, h = face_bbox
    aspect = w / h
    print(f"  Aspect ratio: {aspect:.3f} ({'square' if abs(aspect-1.0)<0.1 else 'portrait' if aspect<1.0 else 'wide'})")
    print(f"{'='*80}\n")

    # Test empirical initialization
    print("1. EMPIRICAL INITIALIZATION (current)")
    clnf_emp = CLNF(model_dir='pyclnf/models', max_iterations=20)

    emp_params, emp_info = empirical_init(clnf_emp.pdm, face_bbox)
    landmarks_emp, info_emp = clnf_emp.fit(gray, face_bbox, return_params=True)

    print(f"  Scale: {emp_info['scale']:.6f}")
    print(f"  Translation: ({emp_info['tx']:.2f}, {emp_info['ty']:.2f})")
    print(f"  Convergence: final_update={info_emp['final_update']:.6f} ({info_emp['final_update']/0.005:.1f}x target)")
    print()

    # Test OpenFace-style initialization
    print("2. OPENFACE-STYLE INITIALIZATION")
    clnf_of = CLNF(model_dir='pyclnf/models', max_iterations=20)

    of_params, of_info = openface_style_init(clnf_of.pdm, face_bbox)

    # Run optimization starting from OpenFace init
    # Need to manually call the full optimization cycle
    view_idx = 0
    current_params = of_params.copy()

    # Run through window sizes [11, 9, 7]
    for window_size in [11, 9, 7]:
        scale_idx = clnf_of.window_to_scale[window_size]
        patch_scale = clnf_of.patch_scaling[scale_idx]
        patch_experts = clnf_of._get_patch_experts(view_idx, patch_scale)

        current_params, info = clnf_of.optimizer.optimize(
            clnf_of.pdm,
            current_params,
            patch_experts,
            gray,
            weights=None,
            window_size=window_size,
            patch_scaling=patch_scale,
            sigma_components=clnf_of.ccnf.sigma_components
        )

    landmarks_of = clnf_of.pdm.params_to_landmarks_2d(current_params)
    final_update = info['iteration_history'][-1]['update_magnitude']

    print(f"  Model dimensions: {of_info['model_width']:.2f} x {of_info['model_height']:.2f}")
    print(f"  Model aspect: {of_info['model_aspect']:.3f}")
    print(f"  Bbox aspect:  {of_info['bbox_aspect']:.3f}")
    print(f"  Aspect match: {'GOOD' if abs(of_info['model_aspect']-of_info['bbox_aspect'])<0.2 else 'MISMATCH'}")
    print(f"  Scale: {of_info['scale']:.6f}")
    print(f"  Translation: ({of_info['tx']:.2f}, {of_info['ty']:.2f})")
    print(f"  Convergence: final_update={final_update:.6f} ({final_update/0.005:.1f}x target)")
    print()

    # Compare
    print("COMPARISON:")
    improvement = (info_emp['final_update'] - final_update) / info_emp['final_update'] * 100
    if final_update < info_emp['final_update']:
        print(f"  OpenFace-style is BETTER by {improvement:.1f}%")
        print(f"  Improvement: {info_emp['final_update']:.6f} → {final_update:.6f}")
    elif final_update > info_emp['final_update']:
        print(f"  Empirical is BETTER by {-improvement:.1f}%")
        print(f"  Degradation: {info_emp['final_update']:.6f} → {final_update:.6f}")
    else:
        print(f"  Similar performance (difference <1%)")
    print()

    return {
        'bbox_name': bbox_name,
        'bbox': face_bbox,
        'aspect': aspect,
        'empirical_final': info_emp['final_update'],
        'openface_final': final_update,
        'improvement_pct': improvement
    }

def main():
    print("="*80)
    print("ASPECT-RATIO-AWARE INITIALIZATION TEST")
    print("="*80)

    # Load test frame
    video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Test all three bbox sources
    bboxes = {
        'Hardcoded (square)': (241, 555, 532, 532),
        'RetinaFace (portrait)': (288, 553, 424, 546),
        'pyMTCNN (portrait)': (270, 706, 437, 414),
    }

    results = []
    for bbox_name, bbox in bboxes.items():
        result = test_initialization(gray, bbox, bbox_name)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()

    print(f"{'Bbox Source':<25} {'Aspect':<10} {'Empirical':<15} {'OpenFace':<15} {'Improvement':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['bbox_name']:<25} {r['aspect']:<10.3f} {r['empirical_final']:<15.6f} "
              f"{r['openface_final']:<15.6f} {r['improvement_pct']:>14.1f}%")
    print()

    # Analysis
    print("ANALYSIS:")
    print()

    all_improved = all(r['improvement_pct'] > 0 for r in results)
    if all_improved:
        print("SUCCESS: OpenFace-style initialization improves ALL bbox sources!")
        print("  → Should replace empirical formula in pdm.py")
        avg_improvement = np.mean([r['improvement_pct'] for r in results])
        print(f"  → Average improvement: {avg_improvement:.1f}%")
    else:
        degraded = [r for r in results if r['improvement_pct'] < 0]
        if degraded:
            print("MIXED RESULTS: OpenFace-style is worse for some bboxes:")
            for r in degraded:
                print(f"  - {r['bbox_name']}: {r['improvement_pct']:.1f}% worse")
            print()
            print("  → Need hybrid approach or further tuning")

    print()
    print("="*80)

if __name__ == "__main__":
    main()
