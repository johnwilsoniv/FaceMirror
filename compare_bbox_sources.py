#!/usr/bin/env python3
"""
Compare bbox sources: OpenFace C++ MTCNN, pyMTCNN, RetinaFace, and hardcoded test bbox.

Shows which bbox source is closest to OpenFace C++ behavior.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'pyfaceau')
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector

def compute_bbox_metrics(bbox1, bbox2):
    """Compute similarity metrics between two bboxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Center difference
    cx1, cy1 = x1 + w1/2, y1 + h1/2
    cx2, cy2 = x2 + w2/2, y2 + h2/2
    center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    # Size difference
    size_diff = abs(w1 - w2) + abs(h1 - h2)

    # Aspect ratio difference
    aspect1 = w1 / h1
    aspect2 = w2 / h2
    aspect_diff = abs(aspect1 - aspect2)

    # IoU (Intersection over Union)
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        iou = 0.0
    else:
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        iou = intersection / union

    return {
        'center_dist': center_dist,
        'size_diff': size_diff,
        'aspect_diff': aspect_diff,
        'iou': iou
    }

def main():
    print("="*80)
    print("BBOX SOURCE COMPARISON")
    print("="*80)
    print()

    # Load test frame
    video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    # OpenFace C++ MTCNN bbox (from debug output)
    # DEBUG_BBOX: 293.145,702.034,418.033,404.659
    openface_cpp_bbox = (int(293), int(702), int(418), int(405))

    # Hardcoded test bbox
    hardcoded_bbox = (241, 555, 532, 532)

    # Get RetinaFace bbox
    print("Detecting with RetinaFace...")
    retinaface = ONNXRetinaFaceDetector(
        'S1 Face Mirror/weights/retinaface_mobilenet025_coreml.onnx',
        use_coreml=True,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    detections, _ = retinaface.detect_faces(frame)
    det = detections[0]
    x1, y1, x2, y2 = det[:4]
    retinaface_bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))

    # pyMTCNN bbox (from earlier test_bbox_convergence_and_response_maps.py)
    pymtcnn_bbox = (270, 706, 437, 414)

    print()
    print("="*80)
    print("BBOX COORDINATES")
    print("="*80)
    print()

    print(f"OpenFace C++ MTCNN: {openface_cpp_bbox}")
    x, y, w, h = openface_cpp_bbox
    print(f"  x={x}, y={y}, width={w}, height={h}")
    print(f"  Aspect ratio: {w/h:.3f}")
    print()

    print(f"pyMTCNN:            {pymtcnn_bbox}")
    x, y, w, h = pymtcnn_bbox
    print(f"  x={x}, y={y}, width={w}, height={h}")
    print(f"  Aspect ratio: {w/h:.3f}")
    print()

    print(f"RetinaFace:         {retinaface_bbox}")
    x, y, w, h = retinaface_bbox
    print(f"  x={x}, y={y}, width={w}, height={h}")
    print(f"  Aspect ratio: {w/h:.3f}")
    print()

    print(f"Hardcoded (test):   {hardcoded_bbox}")
    x, y, w, h = hardcoded_bbox
    print(f"  x={x}, y={y}, width={w}, height={h}")
    print(f"  Aspect ratio: {w/h:.3f}")
    print()

    # Compare each bbox to OpenFace C++ MTCNN
    print("="*80)
    print("SIMILARITY TO OPENFACE C++ MTCNN")
    print("="*80)
    print()

    sources = [
        ('pyMTCNN', pymtcnn_bbox),
        ('RetinaFace', retinaface_bbox),
        ('Hardcoded', hardcoded_bbox)
    ]

    results = []
    for name, bbox in sources:
        metrics = compute_bbox_metrics(openface_cpp_bbox, bbox)
        results.append((name, bbox, metrics))

        print(f"{name}:")
        print(f"  Center distance:  {metrics['center_dist']:.1f} pixels")
        print(f"  Size difference:  {metrics['size_diff']:.1f} pixels (|Δw| + |Δh|)")
        print(f"  Aspect diff:      {metrics['aspect_diff']:.3f}")
        print(f"  IoU:              {metrics['iou']:.3f} (1.0 = perfect match)")
        print()

    # Find best match
    print("="*80)
    print("RANKING (by IoU with OpenFace C++ MTCNN)")
    print("="*80)
    print()

    results_sorted = sorted(results, key=lambda r: r[2]['iou'], reverse=True)
    for i, (name, bbox, metrics) in enumerate(results_sorted, 1):
        print(f"{i}. {name:<15} IoU={metrics['iou']:.3f}, Center dist={metrics['center_dist']:.1f}px")

    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()

    best_name = results_sorted[0][0]
    best_iou = results_sorted[0][2]['iou']

    if best_iou < 0.7:
        print(f"⚠️  WARNING: Best match ({best_name}) has IoU={best_iou:.3f} < 0.7")
        print(f"   None of the detectors match OpenFace C++ closely!")
        print()
        print(f"   Possible reasons:")
        print(f"   - pyMTCNN weights differ from OpenFace MTCNN")
        print(f"   - Different NMS thresholds or confidence thresholds")
        print(f"   - Different preprocessing (scaling, image pyramid)")
    elif best_iou < 0.85:
        print(f"✓ {best_name} is closest match (IoU={best_iou:.3f})")
        print(f"  But still significant difference from OpenFace C++.")
        print(f"  OpenFace-style initialization should help compensate.")
    else:
        print(f"✓ {best_name} matches OpenFace C++ well (IoU={best_iou:.3f})")
        print(f"  This bbox source is a good proxy for C++ behavior.")

    print()
    print("RECOMMENDATION:")
    print()

    # Check if hardcoded was accidentally best
    if best_name == 'Hardcoded':
        print(f"⚠️  CRITICAL: Hardcoded bbox is the closest match to OpenFace C++!")
        print(f"   This confirms the hardcoded bbox WAS from C++ output.")
        print(f"   All testing has been using this 'golden' bbox.")
        print(f"   Production performance will be WORSE with real detectors.")
        print()

        # Compare pyMTCNN vs RetinaFace
        pymtcnn_metrics = next(r[2] for r in results if r[0] == 'pyMTCNN')
        retinaface_metrics = next(r[2] for r in results if r[0] == 'RetinaFace')

        if pymtcnn_metrics['iou'] > retinaface_metrics['iou']:
            print(f"   For production: pyMTCNN is closer to C++ (IoU={pymtcnn_metrics['iou']:.3f})")
            print(f"   vs RetinaFace (IoU={retinaface_metrics['iou']:.3f})")
        else:
            print(f"   For production: RetinaFace is closer to C++ (IoU={retinaface_metrics['iou']:.3f})")
            print(f"   vs pyMTCNN (IoU={pymtcnn_metrics['iou']:.3f})")
    elif best_name == 'pyMTCNN':
        print(f"✓ pyMTCNN is the closest to OpenFace C++ MTCNN")
        print(f"  Use pyMTCNN for testing/validation against C++ output")
        print(f"  However, RetinaFace may still be preferred for production due to CoreML speed")
    else:  # RetinaFace
        print(f"✓ RetinaFace is the closest to OpenFace C++ MTCNN")
        print(f"  Continue using RetinaFace for production")
        print(f"  Its bbox format works well with the landmark detector")

    print()
    print("NEXT STEPS:")
    print()
    print(f"1. Implement OpenFace-style initialization (fixes aspect ratio issues)")
    print(f"2. Test with {best_name} bbox to ensure improvements are real")
    print(f"3. Remove hardcoded bbox from all test scripts")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
