"""
MTCNN Stage-by-Stage Validation

Compares C++ MTCNN vs PyMTCNN (CoreML and ONNX backends):
1. Box counts at each stage (PNet, RNet, ONet, Final)
2. Bounding box IoU comparison
3. 5-point landmark comparison
"""

import os
import sys
import cv2
import numpy as np
import subprocess
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "pymtcnn"))

TEST_IMAGE = PROJECT_ROOT / "calibration_frames" / "patient1_frame1.jpg"
CPP_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
OUTPUT_DIR = PROJECT_ROOT / "mtcnn_validation"
OUTPUT_DIR.mkdir(exist_ok=True)

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes
    box format: [x, y, w, h]
    """
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    # Intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def run_cpp_mtcnn():
    """Run C++ MTCNN and parse debug output"""
    print("\n" + "="*60)
    print("Running C++ MTCNN")
    print("="*60)

    # Clean up temp files
    for tmp_file in ["/tmp/mtcnn_debug.csv", "/tmp/cpp_pnet_all_boxes.txt",
                     "/tmp/cpp_rnet_output.txt", "/tmp/cpp_before_onet.txt"]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    # Run C++ OpenFace
    cmd = [
        CPP_BINARY,
        "-f", str(TEST_IMAGE),
        "-out_dir", str(OUTPUT_DIR),
        "-of", "cpp_output"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse stdout for stage counts
    pnet_count = None
    rnet_count = None
    onet_count = None

    for line in result.stdout.split('\n'):
        if 'DEBUG_PNET_BOXES:' in line:
            pnet_count = int(line.split(':')[1].strip())
        elif 'DEBUG_RNET_BOXES:' in line:
            rnet_count = int(line.split(':')[1].strip())
        elif 'DEBUG_ONET_BOXES:' in line:
            onet_count = int(line.split(':')[1].strip())

    # Load final MTCNN output
    import pandas as pd
    if os.path.exists("/tmp/mtcnn_debug.csv"):
        df = pd.read_csv("/tmp/mtcnn_debug.csv")
        if len(df) > 0:
            row = df.iloc[0]
            bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
            confidence = row['confidence']
            landmarks = np.array([
                [row['lm1_x'], row['lm1_y']],
                [row['lm2_x'], row['lm2_y']],
                [row['lm3_x'], row['lm3_y']],
                [row['lm4_x'], row['lm4_y']],
                [row['lm5_x'], row['lm5_y']]
            ])

            return {
                'pnet_boxes': pnet_count,
                'rnet_boxes': rnet_count,
                'onet_boxes': onet_count,
                'final_boxes': 1,
                'bbox': bbox,
                'confidence': confidence,
                'landmarks': landmarks
            }

    return None

def run_python_mtcnn(backend='coreml'):
    """Run Python MTCNN with specified backend"""
    print(f"\n" + "="*60)
    print(f"Running PyMTCNN ({backend.upper()} backend)")
    print("="*60)

    from pymtcnn import MTCNN

    # Initialize detector
    detector = MTCNN(
        backend=backend,
        debug_mode=True  # Enable debug mode to capture stage info
    )

    # Load image
    img = cv2.imread(str(TEST_IMAGE))

    # Detect faces (debug_mode returns bboxes, landmarks, debug_info)
    result = detector.detect(img)
    if len(result) == 3:
        bboxes, landmarks, debug_info = result
    else:
        bboxes, landmarks = result
        debug_info = {}

    if len(bboxes) > 0:
        bbox = bboxes[0]  # [x, y, w, h]
        landmark = landmarks[0] if landmarks is not None else None

        return {
            'pnet_boxes': debug_info.get('pnet', {}).get('num_boxes', 0),
            'rnet_boxes': debug_info.get('rnet', {}).get('num_boxes', 0),
            'onet_boxes': debug_info.get('onet', {}).get('num_boxes', 0),
            'final_boxes': debug_info.get('final', {}).get('num_boxes', len(bboxes)),
            'bbox': bbox,  # [x, y, w, h]
            'confidence': 1.0,  # PyMTCNN doesn't return confidence in bbox
            'landmarks': landmark
        }

    return None

def main():
    """Run MTCNN validation"""
    print("\n" + "="*80)
    print("MTCNN Stage-by-Stage Validation")
    print("="*80)
    print(f"Test image: {TEST_IMAGE}")

    # Run C++ MTCNN
    cpp_results = run_cpp_mtcnn()

    # Run PyMTCNN with CoreML
    coreml_results = run_python_mtcnn('coreml')

    # Run PyMTCNN with ONNX
    onnx_results = run_python_mtcnn('onnx')

    # Display results
    print("\n" + "="*80)
    print("MTCNN Stage Box Counts")
    print("="*80)

    print(f"\n{'Stage':<15} {'C++':<12} {'CoreML':<12} {'ONNX':<12}")
    print("-"*51)

    if cpp_results and coreml_results and onnx_results:
        stages = ['pnet_boxes', 'rnet_boxes', 'onet_boxes', 'final_boxes']
        stage_names = ['PNet', 'RNet', 'ONet', 'Final']

        for stage, name in zip(stages, stage_names):
            cpp_count = cpp_results.get(stage, 'N/A')
            coreml_count = coreml_results.get(stage, 'N/A')
            onnx_count = onnx_results.get(stage, 'N/A')
            print(f"{name:<15} {cpp_count:<12} {coreml_count:<12} {onnx_count:<12}")

    # IoU Comparison
    print("\n" + "="*80)
    print("Bounding Box Comparison")
    print("="*80)

    if cpp_results and coreml_results and onnx_results:
        cpp_bbox = cpp_results['bbox']
        coreml_bbox = coreml_results['bbox']
        onnx_bbox = onnx_results['bbox']

        print(f"\nC++ bbox:     [{cpp_bbox[0]:.1f}, {cpp_bbox[1]:.1f}, {cpp_bbox[2]:.1f}, {cpp_bbox[3]:.1f}]")
        print(f"CoreML bbox:  [{coreml_bbox[0]:.1f}, {coreml_bbox[1]:.1f}, {coreml_bbox[2]:.1f}, {coreml_bbox[3]:.1f}]")
        print(f"ONNX bbox:    [{onnx_bbox[0]:.1f}, {onnx_bbox[1]:.1f}, {onnx_bbox[2]:.1f}, {onnx_bbox[3]:.1f}]")

        # Calculate IoU
        iou_coreml = compute_iou(cpp_bbox, coreml_bbox)
        iou_onnx = compute_iou(cpp_bbox, onnx_bbox)

        print(f"\nIoU (C++ vs CoreML): {iou_coreml:.4f}")
        print(f"IoU (C++ vs ONNX):   {iou_onnx:.4f}")

        # Confidence comparison
        print(f"\nConfidence:")
        print(f"  C++:    {cpp_results['confidence']:.4f}")
        print(f"  CoreML: {coreml_results['confidence']:.4f}")
        print(f"  ONNX:   {onnx_results['confidence']:.4f}")

    # Landmark comparison
    print("\n" + "="*80)
    print("5-Point Landmark Comparison")
    print("="*80)

    if cpp_results and coreml_results and onnx_results:
        cpp_lm = cpp_results['landmarks']
        coreml_lm = coreml_results['landmarks']
        onnx_lm = onnx_results['landmarks']

        # Convert relative landmarks to absolute if needed
        if cpp_lm is not None and np.max(cpp_lm) <= 1.0:
            # C++ landmarks are relative, convert to absolute
            bbox = cpp_results['bbox']
            cpp_lm_abs = np.zeros_like(cpp_lm)
            cpp_lm_abs[:, 0] = bbox[0] + cpp_lm[:, 0] * bbox[2]
            cpp_lm_abs[:, 1] = bbox[1] + cpp_lm[:, 1] * bbox[3]
        else:
            cpp_lm_abs = cpp_lm

        if coreml_lm is not None and onnx_lm is not None:
            # Calculate landmark errors
            coreml_error = np.mean(np.linalg.norm(cpp_lm_abs - coreml_lm, axis=1))
            onnx_error = np.mean(np.linalg.norm(cpp_lm_abs - onnx_lm, axis=1))

            print(f"\nMean landmark error:")
            print(f"  C++ vs CoreML: {coreml_error:.2f} pixels")
            print(f"  C++ vs ONNX:   {onnx_error:.2f} pixels")

            # Per-landmark errors
            print(f"\nPer-landmark error (C++ vs CoreML):")
            for i in range(5):
                err = np.linalg.norm(cpp_lm_abs[i] - coreml_lm[i])
                print(f"  Landmark {i+1}: {err:.2f}px")

            print(f"\nPer-landmark error (C++ vs ONNX):")
            for i in range(5):
                err = np.linalg.norm(cpp_lm_abs[i] - onnx_lm[i])
                print(f"  Landmark {i+1}: {err:.2f}px")

    # Save results
    results = {
        'cpp': cpp_results,
        'coreml': coreml_results,
        'onnx': onnx_results,
        'iou': {
            'coreml': iou_coreml if cpp_results and coreml_results else None,
            'onnx': iou_onnx if cpp_results and onnx_results else None
        }
    }

    with open(OUTPUT_DIR / 'mtcnn_validation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        json.dump(convert(results), f, indent=2)

    print(f"\n✓ Results saved to: {OUTPUT_DIR / 'mtcnn_validation_results.json'}")

if __name__ == "__main__":
    main()
