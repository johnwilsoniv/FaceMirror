#!/usr/bin/env python3
"""
Test SPIGA/FaceNet AU extraction pipeline directly without GUI.
"""

import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from pathlib import Path
import traceback

# Test videos
TEST_VIDEOS = [
    '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_3324.MOV',
    '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_9330.MOV',
    '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_7540.MOV',
    '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_8270.MOV',
    '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort/IMG_8401.MOV',
]

OUTPUT_DIR = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/S1_FaceMirror/test_output')
OUTPUT_DIR.mkdir(exist_ok=True)

def test_spiga_detector():
    """Test SPIGA detector on a single frame."""
    print("\n" + "="*60)
    print("TEST 1: SPIGA Detector")
    print("="*60)

    from spiga_detector import SPIGALandmarkDetector

    # Load first frame from first video
    video_path = TEST_VIDEOS[0]
    print(f"Loading frame from: {video_path}")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read video: {video_path}")
        return False

    print(f"Frame shape: {frame.shape}")

    # Test SPIGA
    print("\nInitializing SPIGA detector...")
    detector = SPIGALandmarkDetector(debug_mode=True)

    print(f"Device: {detector._spiga_device}")
    print(f"Native landmarks: {detector.native_landmark_count}")
    print(f"Output landmarks: {detector.landmark_count}")

    print("\nDetecting landmarks...")
    landmarks, info = detector.get_face_mesh(frame)

    if landmarks is None:
        print(f"ERROR: No landmarks detected. Info: {info}")
        return False

    print(f"SUCCESS: Got {landmarks.shape} landmarks")
    print(f"Info: {info}")
    return True


def test_spiga_wrapper():
    """Test SPIGA wrapper with CalcParams."""
    print("\n" + "="*60)
    print("TEST 2: SPIGA Wrapper with CalcParams")
    print("="*60)

    from spiga_detector import SPIGALandmarkDetector
    from openface_integration import SPIGALandmarkWrapper
    from pyfaceau.features.pdm import PDMParser
    from pathlib import Path

    weights_dir = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/S1_FaceMirror/weights')
    pdm_file = weights_dir / 'In-the-wild_aligned_PDM_68.txt'

    print(f"Loading PDM from: {pdm_file}")
    pdm_parser = PDMParser(str(pdm_file))

    print("\nInitializing SPIGA detector...")
    spiga = SPIGALandmarkDetector(debug_mode=False)

    print("Creating SPIGA wrapper...")
    wrapper = SPIGALandmarkWrapper(spiga, pdm_parser)

    # Load frame
    video_path = TEST_VIDEOS[0]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read video")
        return False

    # Simulate bbox (x, y, w, h) format as pyfaceau would pass
    h, w = frame.shape[:2]
    bbox = (w * 0.1, h * 0.1, w * 0.8, h * 0.8)

    print(f"\nTesting wrapper.fit() with bbox: {bbox}")
    try:
        landmarks, info = wrapper.fit(frame, bbox)
        print(f"SUCCESS: Got landmarks shape: {landmarks.shape}")
        print(f"Info keys: {info.keys()}")
        print(f"Params shape: {info['params'].shape if 'params' in info else 'N/A'}")
        print(f"Converged: {info.get('converged', 'N/A')}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test full AU extraction pipeline with SPIGA."""
    print("\n" + "="*60)
    print("TEST 3: Full AU Pipeline with SPIGA")
    print("="*60)

    from openface_integration import OpenFace3Processor
    import importlib.util

    # Load local config
    config_path = Path(__file__).parent / 'config.py'
    spec = importlib.util.spec_from_file_location("local_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    print(f"Config LANDMARK_DETECTOR: {config.LANDMARK_DETECTOR}")

    print("\nInitializing OpenFace3Processor...")
    try:
        processor = OpenFace3Processor(debug_mode=True)
        print(f"Using SPIGA: {processor._using_spiga}")
        print(f"Landmark detector type: {type(processor.pipeline.landmark_detector).__name__}")
    except Exception as e:
        print(f"ERROR initializing processor: {e}")
        traceback.print_exc()
        return False

    if not processor._using_spiga:
        print("WARNING: Not using SPIGA! Check config and wrapper initialization.")
        return False

    # Test on first video
    video_path = TEST_VIDEOS[0]
    output_csv = OUTPUT_DIR / f"{Path(video_path).stem}_spiga.csv"

    print(f"\nProcessing video: {video_path}")
    print(f"Output CSV: {output_csv}")

    def progress_callback(current, total, fps):
        if current % 50 == 0 or current == total:
            print(f"  Frame {current}/{total} ({fps:.1f} fps)")

    try:
        frame_count = processor.process_video(
            str(video_path),
            str(output_csv),
            progress_callback=progress_callback
        )
        print(f"\nSUCCESS: Processed {frame_count} frames")
        print(f"CSV saved: {output_csv}")

        # Check CSV was created
        if output_csv.exists():
            print(f"CSV file size: {output_csv.stat().st_size} bytes")
            return True
        else:
            print("ERROR: CSV file was not created")
            return False

    except Exception as e:
        print(f"ERROR processing video: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("SPIGA/FaceNet Pipeline Test")
    print("="*60)

    results = {}

    # Test 1: SPIGA detector
    try:
        results['spiga_detector'] = test_spiga_detector()
    except Exception as e:
        print(f"TEST 1 FAILED: {e}")
        traceback.print_exc()
        results['spiga_detector'] = False

    # Test 2: SPIGA wrapper
    try:
        results['spiga_wrapper'] = test_spiga_wrapper()
    except Exception as e:
        print(f"TEST 2 FAILED: {e}")
        traceback.print_exc()
        results['spiga_wrapper'] = False

    # Test 3: Full pipeline (only if wrapper works)
    if results.get('spiga_wrapper'):
        try:
            results['full_pipeline'] = test_full_pipeline()
        except Exception as e:
            print(f"TEST 3 FAILED: {e}")
            traceback.print_exc()
            results['full_pipeline'] = False
    else:
        print("\nSkipping full pipeline test (wrapper failed)")
        results['full_pipeline'] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")
