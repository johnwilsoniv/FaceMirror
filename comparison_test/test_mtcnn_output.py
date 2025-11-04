#!/usr/bin/env python3
"""
Test MTCNN output structure and values.
"""

import sys
import numpy as np
import cv2

print("="*80)
print("MTCNN OUTPUT TEST")
print("="*80)

TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"

# Load image
print(f"\nLoading image: {TEST_IMAGE}")
image = cv2.imread(TEST_IMAGE)
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")

# Try to import and run MTCNN
print(f"\n{'='*80}")
print("IMPORTING MTCNN")
print(f"{'='*80}")

try:
    sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')
    from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN
    print("✓ OpenFaceMTCNN imported successfully")
except Exception as e:
    print(f"✗ Failed to import MTCNN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*80}")
print("INITIALIZING MTCNN")
print(f"{'='*80}")

try:
    mtcnn = OpenFaceMTCNN()
    print("✓ MTCNN initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize MTCNN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*80}")
print("RUNNING MTCNN DETECTION")
print(f"{'='*80}")

try:
    # Run detection
    # detect() returns (bboxes, landmarks)
    # bboxes: [N, 4] (x1, y1, x2, y2)
    # landmarks: [N, 5, 2] (x, y) for 5 facial points
    print("About to call mtcnn.detect()...")
    sys.stdout.flush()

    bboxes, landmarks = mtcnn.detect(image, return_landmarks=True)

    print("✓ MTCNN detection completed successfully")
    sys.stdout.flush()

    # Show output structure
    print(f"\n{'='*80}")
    print("OUTPUT STRUCTURE")
    print(f"{'='*80}")

    print(f"\n1. Bounding Boxes:")
    if bboxes is None or len(bboxes) == 0:
        print("   None (no faces detected)")
    else:
        print(f"   Type: {type(bboxes)}")
        print(f"   Shape: {bboxes.shape}")
        print(f"   Dtype: {bboxes.dtype}")
        print(f"   Number of faces: {len(bboxes)}")
        if len(bboxes) > 0:
            print(f"   First box: {bboxes[0]}")
            print(f"   Box format: [x1, y1, x2, y2]")
            print(f"   Box values:")
            print(f"      x1={bboxes[0, 0]:.2f}, y1={bboxes[0, 1]:.2f}")
            print(f"      x2={bboxes[0, 2]:.2f}, y2={bboxes[0, 3]:.2f}")
            print(f"      width={bboxes[0, 2] - bboxes[0, 0]:.2f}")
            print(f"      height={bboxes[0, 3] - bboxes[0, 1]:.2f}")

    print(f"\n2. Landmarks (5-point facial landmarks):")
    if landmarks is None or len(landmarks) == 0:
        print("   None")
    else:
        print(f"   Type: {type(landmarks)}")
        print(f"   Shape: {landmarks.shape}")
        print(f"   Dtype: {landmarks.dtype}")
        print(f"   Format: (n_faces={landmarks.shape[0]}, n_landmarks={landmarks.shape[1]}, xy={landmarks.shape[2]})")
        if len(landmarks) > 0:
            print(f"\n   First face 5-point landmarks:")
            landmark_names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
            for i in range(5):
                name = landmark_names[i] if i < len(landmark_names) else f"point_{i}"
                print(f"      {name:12s}: ({landmarks[0, i, 0]:7.2f}, {landmarks[0, i, 1]:7.2f})")

    # Visualize
    print(f"\n{'='*80}")
    print("VISUALIZATION")
    print(f"{'='*80}")

    if bboxes is not None and len(bboxes) > 0:
        vis = image.copy()

        # Draw first box
        box = bboxes[0]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        print(f"   Drew box: ({x1}, {y1}) to ({x2}, {y2})")

        # Draw 5-point landmarks
        if landmarks is not None and len(landmarks) > 0:
            lm = landmarks[0]  # Shape: (5, 2)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            landmark_names = ["L_eye", "R_eye", "Nose", "L_mouth", "R_mouth"]

            for i in range(5):
                x, y = int(lm[i, 0]), int(lm[i, 1])
                cv2.circle(vis, (x, y), 5, colors[i], -1)
                cv2.putText(vis, landmark_names[i], (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
            print(f"   Drew {len(lm)} landmarks with labels")

        # Save visualization
        out_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/mtcnn_output.jpg"
        cv2.imwrite(out_path, vis)
        print(f"\n✓ Saved visualization to: {out_path}")
    else:
        print("   No faces detected - cannot visualize")

except Exception as e:
    print(f"✗ MTCNN detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
