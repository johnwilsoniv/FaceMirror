#!/usr/bin/env python3
"""
Comprehensive diagnosis of why Python CLNF fails vs OpenFace C++ success.
Investigates all 5 hypotheses.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyfaceau"))

import cv2
import numpy as np
import pandas as pd
from pyfaceau_detector import PyFaceAU68LandmarkDetector

print("="*80)
print("CLNF FAILURE DIAGNOSIS")
print("="*80)

# Load test videos
video_8401 = '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV'
video_9330 = '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV'

# Load OpenFace ground truth
openface_8401 = pd.read_csv('/tmp/openface_test_8401_rotated/IMG_8401_source.csv')
openface_9330 = pd.read_csv('/tmp/openface_test_9330_rotated/IMG_9330_source.csv')

# Test frame indices
test_frames = [50, 100, 150, 200]

def extract_openface_landmarks(df, frame_idx):
    """Extract 68 landmarks from OpenFace CSV."""
    row = df[df['frame'] == frame_idx + 1].iloc[0]
    landmarks = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks[i, 0] = row[f'x_{i}']
        landmarks[i, 1] = row[f'y_{i}']
    return landmarks

def compute_landmark_error(pred, gt):
    """Compute average L2 distance between predicted and ground truth landmarks."""
    return np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=1)))

def visualize_comparison(frame, pfld_lms, clnf_lms, openface_lms, title, save_path):
    """Visualize PFLD, CLNF, and OpenFace landmarks side by side."""
    vis = frame.copy()

    # Draw PFLD landmarks in red
    for x, y in pfld_lms:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Draw CLNF landmarks in blue
    if clnf_lms is not None:
        for x, y in clnf_lms:
            cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 0), -1)

    # Draw OpenFace landmarks in green
    for x, y in openface_lms:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Add legend
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, "Red: PFLD Init | Blue: CLNF | Green: OpenFace GT",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(save_path, vis)
    return vis

# Initialize our detector
print("\nInitializing PyfaceAU detector...")
detector = PyFaceAU68LandmarkDetector(
    model_dir=Path(__file__).parent / 'weights',
    debug_mode=True
)

print("\n" + "="*80)
print("HYPOTHESIS 1: Face Detection & Bounding Box Quality")
print("="*80)

for video_path, video_name in [(video_8401, 'IMG_8401'), (video_9330, 'IMG_9330')]:
    print(f"\n### Testing {video_name} ###")
    cap = cv2.VideoCapture(video_path)

    # Test frame 100
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()

    if ret:
        # Detect face with RetinaFace
        faces, _, _ = detector.face_detector.detect_faces(frame)

        if len(faces) > 0:
            bbox = faces[0]
            x1, y1, x2, y2 = bbox

            print(f"  RetinaFace bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
            print(f"  Face size: {x2-x1:.0f}x{y2-y1:.0f} pixels")

            # Visualize bbox
            vis = frame.copy()
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis, f"{video_name} - RetinaFace Detection",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show face crop
            face_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            output_path = f'/tmp/diagnosis_{video_name}_bbox.jpg'
            cv2.imwrite(output_path, vis)
            print(f"  Saved bbox visualization: {output_path}")

            output_crop = f'/tmp/diagnosis_{video_name}_face_crop.jpg'
            cv2.imwrite(output_crop, face_crop)
            print(f"  Saved face crop: {output_crop}")
        else:
            print(f"  ❌ No face detected!")

    cap.release()

print("\n" + "="*80)
print("HYPOTHESIS 2: PFLD Initialization Quality")
print("="*80)

for video_path, video_name, openface_df in [
    (video_8401, 'IMG_8401', openface_8401),
    (video_9330, 'IMG_9330', openface_9330)
]:
    print(f"\n### Testing {video_name} ###")
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()

    if ret:
        # Detect with PFLD
        faces, _, _ = detector.face_detector.detect_faces(frame)

        if len(faces) > 0:
            bbox = faces[0]
            pfld_landmarks, _ = detector.landmark_detector.detect_landmarks(frame, bbox)

            # Get OpenFace ground truth
            openface_landmarks = extract_openface_landmarks(openface_df, 100)

            # Compute error
            pfld_error = compute_landmark_error(pfld_landmarks, openface_landmarks)

            print(f"  PFLD initialization error: {pfld_error:.2f} pixels")

            if pfld_error > 10:
                print(f"  ⚠️  PFLD error is HIGH (>{10}px) - poor initialization!")
            else:
                print(f"  ✅ PFLD error is reasonable (<{10}px)")

            # Visualize
            vis = frame.copy()
            for x, y in pfld_landmarks:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            for x, y in openface_landmarks:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

            cv2.putText(vis, f"{video_name} - PFLD (Red) vs OpenFace (Green)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"Error: {pfld_error:.2f} pixels",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            output_path = f'/tmp/diagnosis_{video_name}_pfld_init.jpg'
            cv2.imwrite(output_path, vis)
            print(f"  Saved comparison: {output_path}")

    cap.release()

print("\n" + "="*80)
print("HYPOTHESIS 3: CLNF Activation & Quality Check")
print("="*80)

for video_path, video_name, openface_df in [
    (video_8401, 'IMG_8401', openface_8401),
    (video_9330, 'IMG_9330', openface_9330)
]:
    print(f"\n### Testing {video_name} ###")
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()

    if ret:
        # Process through full detector
        results = detector.detect(frame)

        if results['success']:
            pfld_landmarks = results['landmarks_68']

            # Check if CLNF was activated
            print(f"  CLNF activated: {detector.clnf_fallback_activated}")

            # Get OpenFace ground truth
            openface_landmarks = extract_openface_landmarks(openface_df, 100)

            # Compute errors
            final_error = compute_landmark_error(pfld_landmarks, openface_landmarks)

            print(f"  Final landmark error: {final_error:.2f} pixels")

            # Check landmark quality assessment
            is_poor, reason = detector.assess_landmark_quality(pfld_landmarks, frame)
            print(f"  Quality assessment: is_poor={is_poor}, reason='{reason}'")

            if not is_poor:
                print(f"  ⚠️  CLNF NOT triggered! Quality check thinks landmarks are good.")
                print(f"  This means our quality thresholds may be too lenient!")

    cap.release()

print("\n" + "="*80)
print("HYPOTHESIS 4: CLNF Refinement Testing")
print("="*80)

if detector.clnf_fallback is not None:
    for video_path, video_name, openface_df in [
        (video_8401, 'IMG_8401', openface_8401),
        (video_9330, 'IMG_9330', openface_9330)
    ]:
        print(f"\n### Testing {video_name} ###")
        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()

        if ret:
            faces, _, _ = detector.face_detector.detect_faces(frame)

            if len(faces) > 0:
                bbox = faces[0]
                pfld_landmarks, _ = detector.landmark_detector.detect_landmarks(frame, bbox)

                # Get OpenFace ground truth
                openface_landmarks = extract_openface_landmarks(openface_df, 100)

                pfld_error = compute_landmark_error(pfld_landmarks, openface_landmarks)

                # Force CLNF refinement
                print(f"  Testing CLNF refinement (forcing it to run)...")
                clnf_landmarks, converged, num_iters = detector.clnf_fallback.refine_landmarks(
                    frame, pfld_landmarks, scale_idx=0, regularization=0.5, multi_scale=True
                )

                clnf_error = compute_landmark_error(clnf_landmarks, openface_landmarks)

                print(f"  PFLD error: {pfld_error:.2f} pixels")
                print(f"  CLNF error: {clnf_error:.2f} pixels")
                print(f"  Converged: {converged}, Iterations: {num_iters}")

                if clnf_error < pfld_error:
                    improvement = pfld_error - clnf_error
                    print(f"  ✅ CLNF improved by {improvement:.2f} pixels ({improvement/pfld_error*100:.1f}%)")
                else:
                    degradation = clnf_error - pfld_error
                    print(f"  ❌ CLNF MADE IT WORSE by {degradation:.2f} pixels!")

                # Visualize
                vis = visualize_comparison(
                    frame, pfld_landmarks, clnf_landmarks, openface_landmarks,
                    f"{video_name} - Frame 100",
                    f'/tmp/diagnosis_{video_name}_clnf_comparison.jpg'
                )
                print(f"  Saved comparison: /tmp/diagnosis_{video_name}_clnf_comparison.jpg")

        cap.release()
else:
    print("  ❌ CLNF fallback not available!")

print("\n" + "="*80)
print("HYPOTHESIS 5: Temporal Tracking")
print("="*80)
print("OpenFace uses previous frame landmarks as initialization for next frame.")
print("We reinitialize from PFLD on every frame.")
print("\nTesting sequential frame tracking...")

for video_path, video_name, openface_df in [
    (video_8401, 'IMG_8401', openface_8401)
]:
    print(f"\n### Testing {video_name} (frames 100-110) ###")
    cap = cv2.VideoCapture(video_path)

    prev_landmarks = None
    errors_with_tracking = []
    errors_without_tracking = []

    for frame_idx in range(100, 110):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            faces, _, _ = detector.face_detector.detect_faces(frame)

            if len(faces) > 0 and detector.clnf_fallback is not None:
                bbox = faces[0]
                pfld_landmarks, _ = detector.landmark_detector.detect_landmarks(frame, bbox)

                openface_landmarks = extract_openface_landmarks(openface_df, frame_idx)

                # Test 1: Without temporal tracking (reinit from PFLD)
                clnf_no_tracking, _, _ = detector.clnf_fallback.refine_landmarks(
                    frame, pfld_landmarks, multi_scale=True
                )
                error_no_tracking = compute_landmark_error(clnf_no_tracking, openface_landmarks)
                errors_without_tracking.append(error_no_tracking)

                # Test 2: With temporal tracking (use previous frame as init)
                if prev_landmarks is not None:
                    clnf_with_tracking, _, _ = detector.clnf_fallback.refine_landmarks(
                        frame, prev_landmarks, multi_scale=True
                    )
                    error_with_tracking = compute_landmark_error(clnf_with_tracking, openface_landmarks)
                    errors_with_tracking.append(error_with_tracking)
                    prev_landmarks = clnf_with_tracking
                else:
                    prev_landmarks = clnf_no_tracking

    cap.release()

    if len(errors_with_tracking) > 0:
        avg_without = np.mean(errors_without_tracking)
        avg_with = np.mean(errors_with_tracking)

        print(f"  Average error WITHOUT temporal tracking: {avg_without:.2f} pixels")
        print(f"  Average error WITH temporal tracking: {avg_with:.2f} pixels")

        if avg_with < avg_without:
            improvement = avg_without - avg_with
            print(f"  ✅ Temporal tracking improves by {improvement:.2f} pixels!")
        else:
            print(f"  ❌ Temporal tracking doesn't help (or makes it worse)")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print("\nCheck the following images:")
print("  /tmp/diagnosis_IMG_8401_bbox.jpg")
print("  /tmp/diagnosis_IMG_8401_face_crop.jpg")
print("  /tmp/diagnosis_IMG_8401_pfld_init.jpg")
print("  /tmp/diagnosis_IMG_8401_clnf_comparison.jpg")
print("  /tmp/diagnosis_IMG_9330_bbox.jpg")
print("  /tmp/diagnosis_IMG_9330_face_crop.jpg")
print("  /tmp/diagnosis_IMG_9330_pfld_init.jpg")
print("  /tmp/diagnosis_IMG_9330_clnf_comparison.jpg")
