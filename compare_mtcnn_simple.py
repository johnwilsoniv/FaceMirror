"""
Simple MTCNN Comparison: C++ vs PyMTCNN (CoreML and ONNX)

Compares box counts at each stage and calculates IoU
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add pymtcnn to path
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn import MTCNN

# Test image
TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

# C++ baseline results (from manual run)
CPP_RESULTS = {
    'pnet': 95,   # After cross-scale NMS
    'rnet': 95,   # Going to ONet
    'onet': 1,    # Final output
    'final': 1,
    'bbox': [301.938, 782.149, 400.586, 400.585],  # x, y, w, h
    'confidence': 1.0
}

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]"""
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def test_pymtcnn(backend):
    """Test PyMTCNN with specified backend"""
    print(f"\n{'='*60}")
    print(f"Testing PyMTCNN - {backend.upper()} Backend")
    print(f"{'='*60}")

    # Initialize detector
    detector = MTCNN(backend=backend, debug_mode=True)

    # Load image
    img = cv2.imread(TEST_IMAGE)

    # Detect faces
    result = detector.detect(img)
    if len(result) == 3:
        bboxes, landmarks, debug_info = result
    else:
        bboxes, landmarks = result
        debug_info = {}

    if len(bboxes) == 0:
        print("⚠ No faces detected!")
        return None

    bbox = bboxes[0]

    # Extract stage counts from debug_info
    pnet_count = debug_info.get('pnet', {}).get('num_boxes', 'N/A')
    rnet_count = debug_info.get('rnet', {}).get('num_boxes', 'N/A')
    onet_count = debug_info.get('onet', {}).get('num_boxes', 'N/A')
    final_count = len(bboxes)

    # Calculate IoU
    iou = compute_iou(CPP_RESULTS['bbox'], bbox)

    return {
        'pnet': pnet_count,
        'rnet': rnet_count,
        'onet': onet_count,
        'final': final_count,
        'bbox': bbox,
        'iou': iou,
        'landmarks': landmarks[0] if landmarks is not None and len(landmarks) > 0 else None
    }

def main():
    print("\n" + "="*80)
    print("MTCNN Validation: C++ vs PyMTCNN")
    print("="*80)
    print(f"Test image: {TEST_IMAGE}\n")

    # Test CoreML
    coreml_results = test_pymtcnn('coreml')

    # Test ONNX
    onnx_results = test_pymtcnn('onnx')

    # Display results
    print("\n" + "="*80)
    print("Stage-by-Stage Box Counts")
    print("="*80)

    print(f"\n{'Stage':<15} {'C++':<12} {'CoreML':<12} {'ONNX':<12}")
    print("-"*51)

    stages = [
        ('PNet', 'pnet'),
        ('RNet', 'rnet'),
        ('ONet', 'onet'),
        ('Final', 'final')
    ]

    for stage_name, stage_key in stages:
        cpp_val = CPP_RESULTS[stage_key]
        coreml_val = coreml_results[stage_key] if coreml_results else 'N/A'
        onnx_val = onnx_results[stage_key] if onnx_results else 'N/A'
        print(f"{stage_name:<15} {cpp_val:<12} {coreml_val:<12} {onnx_val:<12}")

    # Bounding Box Comparison
    print("\n" + "="*80)
    print("Bounding Box Comparison")
    print("="*80)

    cpp_bbox = CPP_RESULTS['bbox']
    print(f"\nC++ bbox:     [{cpp_bbox[0]:.1f}, {cpp_bbox[1]:.1f}, {cpp_bbox[2]:.1f}, {cpp_bbox[3]:.1f}]")

    if coreml_results:
        coreml_bbox = coreml_results['bbox']
        print(f"CoreML bbox:  [{coreml_bbox[0]:.1f}, {coreml_bbox[1]:.1f}, {coreml_bbox[2]:.1f}, {coreml_bbox[3]:.1f}]")
        print(f"  IoU (C++ vs CoreML): {coreml_results['iou']:.4f}")

    if onnx_results:
        onnx_bbox = onnx_results['bbox']
        print(f"ONNX bbox:    [{onnx_bbox[0]:.1f}, {onnx_bbox[1]:.1f}, {onnx_bbox[2]:.1f}, {onnx_bbox[3]:.1f}]")
        print(f"  IoU (C++ vs ONNX):   {onnx_results['iou']:.4f}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    if coreml_results and onnx_results:
        print(f"\n✓ Both backends detected faces")
        print(f"  CoreML IoU: {coreml_results['iou']:.4f} {'✓ PASS' if coreml_results['iou'] > 0.9 else '⚠ REVIEW'}")
        print(f"  ONNX IoU:   {onnx_results['iou']:.4f} {'✓ PASS' if onnx_results['iou'] > 0.9 else '⚠ REVIEW'}")

        print(f"\nStage Box Count Matching:")
        for stage_name, stage_key in stages:
            cpp = CPP_RESULTS[stage_key]
            coreml = coreml_results[stage_key]
            onnx = onnx_results[stage_key]

            coreml_match = "✓" if cpp == coreml else f"✗ ({coreml} vs {cpp})"
            onnx_match = "✓" if cpp == onnx else f"✗ ({onnx} vs {cpp})"

            print(f"  {stage_name:<8} CoreML: {coreml_match:<20} ONNX: {onnx_match}")

if __name__ == "__main__":
    main()
