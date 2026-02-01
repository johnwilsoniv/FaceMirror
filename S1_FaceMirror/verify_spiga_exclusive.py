#!/usr/bin/env python3
"""
Verify SPIGA is being used exclusively in the AU extraction pipeline.

This test directly inspects the pipeline's landmark_detector to confirm
it's our SPIGALandmarkWrapper, not pyclnf CLNF.
"""

import os
import sys
sys.path.insert(0, '.')

# Monkey-patch onnxruntime to force CPU-only provider BEFORE any other imports
# This avoids CoreML compatibility issues on some systems
import onnxruntime as ort
_original_InferenceSession = ort.InferenceSession

def _patched_InferenceSession(model_path, *args, **kwargs):
    """Force CPU-only execution provider."""
    # Force providers to CPU only
    kwargs['providers'] = ['CPUExecutionProvider']
    return _original_InferenceSession(model_path, *args, **kwargs)

ort.InferenceSession = _patched_InferenceSession
print("[PATCH] Forced onnxruntime to use CPUExecutionProvider only")

import cv2
import numpy as np
from pathlib import Path
import traceback

# Test video
TEST_VIDEO = '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_3324.MOV'
OUTPUT_DIR = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/S1_FaceMirror/test_output')
OUTPUT_DIR.mkdir(exist_ok=True)


def verify_spiga_integration():
    """Verify that SPIGA wrapper is actually being used."""
    print("\n" + "="*70)
    print("VERIFYING SPIGA EXCLUSIVE INTEGRATION")
    print("="*70)

    # Step 1: Check config
    import importlib.util
    config_path = Path(__file__).parent / 'config.py'
    spec = importlib.util.spec_from_file_location("local_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    print(f"\n[1] Config check:")
    print(f"    LANDMARK_DETECTOR = '{config.LANDMARK_DETECTOR}'")
    if config.LANDMARK_DETECTOR != 'spiga':
        print("    ERROR: Config is not set to 'spiga'!")
        return False
    print("    ✓ Config is set to 'spiga'")

    # Step 2: Initialize processor and check detector type
    print(f"\n[2] Initializing OpenFace3Processor...")
    from openface_integration import OpenFace3Processor, SPIGALandmarkWrapper

    processor = OpenFace3Processor(debug_mode=True)

    print(f"    processor._using_spiga = {processor._using_spiga}")
    if not processor._using_spiga:
        print("    ERROR: processor._using_spiga is False!")
        return False
    print("    ✓ Processor reports using SPIGA")

    # Step 3: Check if pipeline's landmark_detector is SPIGALandmarkWrapper
    print(f"\n[3] Checking pipeline.landmark_detector type...")

    # Force initialization if not done
    if not processor.pipeline._components_initialized:
        print("    Warning: Components not initialized, forcing initialization...")
        processor.pipeline._initialize_components()

    detector = processor.pipeline.landmark_detector
    detector_type = type(detector).__name__
    print(f"    Detector type: {detector_type}")

    if detector_type != 'SPIGALandmarkWrapper':
        print(f"    ERROR: Expected SPIGALandmarkWrapper, got {detector_type}!")
        print(f"    This means the detector replacement FAILED!")
        return False
    print("    ✓ Detector is SPIGALandmarkWrapper")

    # Step 4: Test detector.fit() directly
    print(f"\n[4] Testing detector.fit() directly...")

    cap = cv2.VideoCapture(TEST_VIDEO)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"    ERROR: Could not read video: {TEST_VIDEO}")
        return False

    print(f"    Frame shape: {frame.shape}")

    # Create a dummy bbox (using approximate face location)
    h, w = frame.shape[:2]
    bbox = (int(w * 0.25), int(h * 0.1), int(w * 0.5), int(h * 0.7))  # (x, y, w, h)
    print(f"    Test bbox: {bbox}")

    # Reset call counter
    SPIGALandmarkWrapper._call_count = 0

    try:
        landmarks, info = detector.fit(frame, bbox)
        print(f"    ✓ fit() returned landmarks shape: {landmarks.shape}")
        print(f"    ✓ fit() call count: {SPIGALandmarkWrapper._call_count}")
        print(f"    ✓ info keys: {list(info.keys())}")

        if SPIGALandmarkWrapper._call_count == 0:
            print("    ERROR: fit() call count is still 0!")
            return False

    except Exception as e:
        print(f"    ERROR during fit(): {e}")
        traceback.print_exc()
        return False

    # Step 5: Process 10 frames through pipeline
    print(f"\n[5] Processing 10 frames through full pipeline...")

    SPIGALandmarkWrapper._call_count = 0

    cap = cv2.VideoCapture(TEST_VIDEO)
    frames_processed = 0

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        result = processor.pipeline._process_frame(frame, i, i / 30.0)
        if result['success']:
            frames_processed += 1

    cap.release()

    print(f"    Frames processed successfully: {frames_processed}/10")
    print(f"    SPIGALandmarkWrapper.fit() call count: {SPIGALandmarkWrapper._call_count}")

    if SPIGALandmarkWrapper._call_count == 0:
        print("    ERROR: SPIGALandmarkWrapper.fit() was never called during pipeline!")
        print("    This means pyfaceau is using a different landmark detector!")
        return False

    print(f"    ✓ SPIGA wrapper was called {SPIGALandmarkWrapper._call_count} times")

    # Step 6: Verify face detection is using FaceNet MTCNN
    print(f"\n[6] Checking face detector...")
    face_detector = processor.pipeline.face_detector
    detector_type = type(face_detector).__name__
    print(f"    Face detector type: {detector_type}")

    if hasattr(face_detector, 'backend'):
        print(f"    Face detector backend: {face_detector.backend}")

    # Note: pyfaceau uses PyMTCNN (not facenet-pytorch MTCNN)
    # For fully exclusive SPIGA+FaceNet, we'd need to also replace face detector

    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"  Config: SPIGA enabled ✓")
    print(f"  Landmark Detector: SPIGALandmarkWrapper ✓")
    print(f"  fit() calls during processing: {SPIGALandmarkWrapper._call_count} ✓")
    print(f"  Face Detector: {detector_type} (pyfaceau's PyMTCNN)")
    print()
    print("NOTE: Face detection still uses pyfaceau's PyMTCNN (CoreML/CUDA).")
    print("SPIGA landmarks are computed using FaceNet MTCNN internally for bbox,")
    print("but the pipeline face tracking uses PyMTCNN for initial detection.")
    print()

    return True


def process_test_videos():
    """Process test videos with verified SPIGA integration."""
    print("\n" + "="*70)
    print("PROCESSING TEST VIDEOS WITH SPIGA")
    print("="*70)

    test_videos = [
        '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_3324.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_9330.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_7540.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_8270.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_8401.MOV',
    ]

    from openface_integration import OpenFace3Processor, SPIGALandmarkWrapper

    processor = OpenFace3Processor(debug_mode=False)

    results = {}

    for video_path in test_videos:
        video_name = Path(video_path).stem
        output_csv = OUTPUT_DIR / f"{video_name}_spiga_exclusive.csv"

        print(f"\nProcessing: {video_name}")
        print(f"  Output: {output_csv}")

        SPIGALandmarkWrapper._call_count = 0

        def progress_callback(current, total, fps):
            if current % 100 == 0 or current == total:
                print(f"  Frame {current}/{total} ({fps:.1f} fps, SPIGA calls: {SPIGALandmarkWrapper._call_count})")

        try:
            frame_count = processor.process_video(
                str(video_path),
                str(output_csv),
                progress_callback=progress_callback
            )

            results[video_name] = {
                'success': True,
                'frames': frame_count,
                'spiga_calls': SPIGALandmarkWrapper._call_count,
                'csv_size': output_csv.stat().st_size if output_csv.exists() else 0
            }

            print(f"  ✓ Processed {frame_count} frames, SPIGA called {SPIGALandmarkWrapper._call_count} times")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[video_name] = {'success': False, 'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)

    for video_name, result in results.items():
        if result['success']:
            print(f"  {video_name}: {result['frames']} frames, {result['spiga_calls']} SPIGA calls, {result['csv_size']} bytes")
        else:
            print(f"  {video_name}: FAILED - {result.get('error', 'unknown')}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true', help='Process all test videos after verification')
    args = parser.parse_args()

    print("SPIGA Exclusive Integration Verification")
    print("="*70)

    # First verify SPIGA is properly integrated
    if verify_spiga_integration():
        print("\n✓ SPIGA integration verified!\n")

        if args.process:
            process_test_videos()
        else:
            print("Run with --process to process all 5 test videos")
    else:
        print("\n✗ SPIGA integration verification FAILED!")
        print("Please check the errors above and fix the integration.")
