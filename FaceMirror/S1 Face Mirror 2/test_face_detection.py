#!/usr/bin/env python3
"""
Diagnostic script to test RetinaFace detection on problematic patient videos.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory and pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector
import config_paths

def test_face_detection(image_path, confidence_thresholds=[0.5, 0.3, 0.2, 0.1]):
    """
    Test face detection with different confidence thresholds.

    Args:
        image_path: Path to test image
        confidence_thresholds: List of thresholds to try
    """
    print(f"\n{'='*70}")
    print(f"Testing: {Path(image_path).name}")
    print(f"{'='*70}")

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"Image size: {w}x{h}")

    # Test with different confidence thresholds
    for threshold in confidence_thresholds:
        print(f"\n--- Testing with confidence threshold: {threshold} ---")

        try:
            # Create detector with this threshold
            from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
            model_dir = config_paths.get_weights_dir()

            face_detector = ONNXRetinaFaceDetector(
                str(model_dir / 'retinaface_mobilenet025_coreml.onnx'),
                use_coreml=True,
                confidence_threshold=threshold,
                nms_threshold=0.4
            )

            # Detect faces
            detections, _ = face_detector.detect_faces(frame)

            if detections is None or len(detections) == 0:
                print(f"  ❌ NO FACES DETECTED")
            else:
                print(f"  ✓ Detected {len(detections)} face(s)")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det[:4].astype(int)
                    confidence = det[4] if len(det) > 4 else 0.0
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    print(f"    Face {i+1}: bbox=[{x1},{y1},{x2},{y2}] size={bbox_w}x{bbox_h} confidence={confidence:.3f}")

                # Save debug image with detections
                debug_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det[:4].astype(int)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                output_path = f"/tmp/{Path(image_path).stem}_detected_{threshold}.jpg"
                cv2.imwrite(output_path, debug_frame)
                print(f"    Debug image saved: {output_path}")

            del face_detector

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    # Test full landmark detection pipeline
    print(f"\n--- Testing full landmark detection pipeline ---")
    try:
        detector = PyFaceAU68LandmarkDetector(
            debug_mode=True,
            skip_face_detection=False,
            use_clnf_refinement=True
        )

        landmarks, _ = detector.get_face_mesh(frame)

        if landmarks is None:
            print(f"  ❌ LANDMARK DETECTION FAILED")
        else:
            print(f"  ✓ Successfully detected {len(landmarks)} landmarks")

            # Save debug image with landmarks
            debug_frame = frame.copy()
            for (x, y) in landmarks:
                cv2.circle(debug_frame, (x, y), 3, (0, 255, 0), -1)

            output_path = f"/tmp/{Path(image_path).stem}_landmarks.jpg"
            cv2.imwrite(output_path, debug_frame)
            print(f"  Debug image saved: {output_path}")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test both problematic patient videos
    test_images = [
        "/tmp/IMG_8401_frame_1.jpg",
        "/tmp/IMG_9330_frame_1.jpg"
    ]

    for image_path in test_images:
        if Path(image_path).exists():
            test_face_detection(image_path)
        else:
            print(f"WARNING: Image not found: {image_path}")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}\n")
