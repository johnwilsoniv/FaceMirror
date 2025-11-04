#!/usr/bin/env python3
"""
Comprehensive test of Python MTCNN + CLNF pipeline
Tests the pure Python landmark detection pathway
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

print("="*80)
print("Python MTCNN + CLNF Pipeline Test")
print("="*80)

# Step 1: Test MTCNN
print("\n[Step 1] Testing MTCNN Detector")
print("-" * 80)

try:
    from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
    print("✓ MTCNN module imported successfully")

    mtcnn = OpenFaceMTCNN()
    print("✓ MTCNN detector initialized")
    print(f"  Device: {mtcnn.device}")
    print(f"  Min face size: {mtcnn.min_face_size}px")
    print(f"  Thresholds: {mtcnn.thresholds}")

except Exception as e:
    print(f"✗ MTCNN initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Test MTCNN on a real video frame
print("\n[Step 2] Testing MTCNN on Real Video")
print("-" * 80)

test_video = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Paralysis Cohort/IMG_1837.MOV"
if not Path(test_video).exists():
    print(f"✗ Test video not found: {test_video}")
    print("  Trying alternative video...")
    test_video = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Paralysis Cohort/20250213_193056000_iOS.MOV"

if Path(test_video).exists():
    print(f"  Video: {Path(test_video).name}")

    # Extract a frame
    cap = cv2.VideoCapture(test_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()

    if ret:
        print(f"✓ Extracted frame: {frame.shape[1]}x{frame.shape[0]}px")

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MTCNN detection
        try:
            import time
            start = time.time()
            bboxes, landmarks = mtcnn.detect(frame_rgb, return_landmarks=True)
            elapsed = time.time() - start

            print(f"✓ MTCNN detection completed in {elapsed:.3f}s")
            print(f"  Detected {len(bboxes)} face(s)")

            if len(bboxes) > 0:
                bbox = bboxes[0]
                print(f"  BBox: ({bbox[0]:.0f}, {bbox[1]:.0f}) -> ({bbox[2]:.0f}, {bbox[3]:.0f})")
                print(f"  Size: {bbox[2]-bbox[0]:.0f}x{bbox[3]-bbox[1]:.0f}px")

                if landmarks is not None:
                    print(f"  5-point landmarks: {landmarks[0].shape}")

                    # Save visualization
                    vis = frame_rgb.copy()
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    for i, (x, y) in enumerate(landmarks[0]):
                        cv2.circle(vis, (int(x), int(y)), 5, (255, 0, 0), -1)

                    output_path = "/tmp/mtcnn_test_detection.jpg"
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, vis_bgr)
                    print(f"✓ Saved visualization: {output_path}")

                    mtcnn_works = True
            else:
                print("⚠ No faces detected (but MTCNN ran without errors)")
                mtcnn_works = True

        except Exception as e:
            print(f"✗ MTCNN detection failed: {e}")
            import traceback
            traceback.print_exc()
            mtcnn_works = False
    else:
        print("✗ Could not extract frame")
        mtcnn_works = False
else:
    print(f"✗ Test video not found")
    mtcnn_works = False

# Step 3: Test CLNF
print("\n[Step 3] Testing CLNF Detector")
print("-" * 80)

try:
    from pyfaceau.clnf.clnf_detector import CLNFDetector
    print("✓ CLNF module imported successfully")

    # Check for model files
    model_dir = Path(__file__).parent / "pyfaceau" / "pyfaceau" / "detectors" / "weights" / "clnf"
    if not model_dir.exists():
        model_dir = Path(__file__).parent / "S1 Face Mirror" / "weights" / "clnf"

    print(f"  Model directory: {model_dir}")

    if model_dir.exists():
        pdm_path = model_dir / "In-the-wild_aligned_PDM_68.txt"
        if pdm_path.exists():
            print(f"✓ PDM file found: {pdm_path.name}")

            # Try to initialize CLNF
            try:
                clnf = CLNFDetector(
                    model_dir=model_dir,
                    max_iterations=5,
                    convergence_threshold=0.01
                )
                print("✓ CLNF detector initialized")
                clnf_works = True
            except Exception as e:
                print(f"✗ CLNF initialization failed: {e}")
                import traceback
                traceback.print_exc()
                clnf_works = False
        else:
            print(f"✗ PDM file not found: {pdm_path}")
            clnf_works = False
    else:
        print(f"✗ Model directory not found: {model_dir}")
        clnf_works = False

except Exception as e:
    print(f"✗ CLNF import failed: {e}")
    import traceback
    traceback.print_exc()
    clnf_works = False

# Step 4: Test CLNF refinement (if both work)
if mtcnn_works and clnf_works and len(bboxes) > 0:
    print("\n[Step 4] Testing MTCNN + CLNF Integration")
    print("-" * 80)

    try:
        # Create simple 68-point initialization from bbox
        bbox = bboxes[0]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Simple initialization: place landmarks in a grid
        init_landmarks = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            # Distribute landmarks roughly in face region
            if i < 17:  # Jaw
                t = i / 16.0
                init_landmarks[i, 0] = x1 + t * w
                init_landmarks[i, 1] = y2 - h * 0.1
            elif i < 27:  # Eyebrows
                t = (i - 17) / 9.0
                init_landmarks[i, 0] = x1 + 0.2 * w + t * 0.6 * w
                init_landmarks[i, 1] = y1 + 0.3 * h
            elif i < 36:  # Nose
                init_landmarks[i, 0] = cx
                init_landmarks[i, 1] = y1 + 0.4 * h + (i - 27) * 0.04 * h
            elif i < 48:  # Eyes
                t = (i - 36) / 11.0
                init_landmarks[i, 0] = x1 + 0.25 * w + t * 0.5 * w
                init_landmarks[i, 1] = y1 + 0.45 * h
            else:  # Mouth
                t = (i - 48) / 19.0
                init_landmarks[i, 0] = x1 + 0.3 * w + t * 0.4 * w
                init_landmarks[i, 1] = y1 + 0.75 * h

        print("✓ Created 68-point initialization from bbox")

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run CLNF refinement
        print("  Running CLNF refinement...")
        refined_landmarks, converged, num_iters = clnf.refine_landmarks(
            gray, init_landmarks, scale_idx=2, regularization=0.5, multi_scale=False
        )

        print(f"✓ CLNF refinement completed")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {num_iters}")

        # Save visualization
        vis = frame.copy()

        # Draw initial landmarks (orange)
        for i, (x, y) in enumerate(init_landmarks):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 165, 255), -1)

        # Draw refined landmarks (cyan)
        for i, (x, y) in enumerate(refined_landmarks):
            cv2.circle(vis, (int(x), int(y)), 4, (255, 255, 0), -1)
            cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), 1)

        cv2.putText(vis, "Orange=Init, Cyan=CLNF", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        output_path = "/tmp/clnf_refinement_test.jpg"
        cv2.imwrite(output_path, vis)
        print(f"✓ Saved visualization: {output_path}")

        integration_works = True

    except Exception as e:
        print(f"✗ CLNF refinement failed: {e}")
        import traceback
        traceback.print_exc()
        integration_works = False
else:
    integration_works = False
    print("\n[Step 4] MTCNN + CLNF Integration")
    print("-" * 80)
    print("⚠ Skipped (prerequisites not met)")

# Final Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

results = {
    "MTCNN Import": "✓" if 'mtcnn' in locals() else "✗",
    "MTCNN Detection": "✓" if mtcnn_works else "✗",
    "CLNF Import": "✓" if 'clnf' in locals() else "✗",
    "CLNF Initialization": "✓" if clnf_works else "✗",
    "MTCNN + CLNF Integration": "✓" if integration_works else "✗"
}

for test, status in results.items():
    print(f"  {status} {test}")

print("\n" + "="*80)
if all(status == "✓" for status in results.values()):
    print("SUCCESS: Pure Python landmark detection pathway is functional!")
    print("="*80)
    print("\nVisualizations saved:")
    print("  /tmp/mtcnn_test_detection.jpg - MTCNN face detection")
    print("  /tmp/clnf_refinement_test.jpg - CLNF landmark refinement")
else:
    print("ISSUES FOUND: Some components are not working properly")
    print("="*80)
    print("\nProblems:")
    for test, status in results.items():
        if status == "✗":
            print(f"  • {test}")

print()
