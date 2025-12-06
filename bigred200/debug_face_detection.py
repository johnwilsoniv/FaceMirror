#!/usr/bin/env python3
"""
Debug script for face detection issues on Big Red 200.

This script tests face detection with different frame orientations
to diagnose issues with portrait videos from iPhones.
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pyclnf"))
sys.path.insert(0, str(project_root / "pyfaceau"))
sys.path.insert(0, str(project_root / "pymtcnn"))

# Set ONNX thread count before import
os.environ['ORT_NUM_THREADS'] = '1'

import cv2
import numpy as np


def get_video_rotation(video_path):
    """Get rotation metadata from video using ffprobe-like approach."""
    cap = cv2.VideoCapture(str(video_path))

    # Try to get rotation from video properties
    # Note: OpenCV doesn't expose rotation metadata directly
    # We'll check the frame dimensions instead

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.release()

    if ret:
        actual_h, actual_w = frame.shape[:2]
        return {
            'reported_width': width,
            'reported_height': height,
            'actual_width': actual_w,
            'actual_height': actual_h,
            'frame_shape': frame.shape,
            'is_portrait': actual_h > actual_w,
            'frame': frame
        }
    return None


def test_detection_with_rotation(frame, rotation=0, detector=None):
    """Test face detection with different rotations."""
    if detector is None:
        from pymtcnn import MTCNN
        detector = MTCNN()

    # Apply rotation
    if rotation == 90:
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == -90 or rotation == 270:
        rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        rotated = cv2.rotate(frame, cv2.ROTATE_180)
    else:
        rotated = frame

    # Detect faces
    bboxes, landmarks = detector.detect(rotated)

    return {
        'rotation': rotation,
        'shape': rotated.shape,
        'faces_found': 0 if bboxes is None else len(bboxes),
        'bboxes': bboxes,
        'detector': detector
    }


def test_jpeg_encode_decode(frame, detector):
    """Test if JPEG encoding/decoding affects detection."""
    # JPEG encode/decode like HPC pipeline does
    _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    decoded = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)

    bboxes, landmarks = detector.detect(decoded)

    return {
        'original_shape': frame.shape,
        'decoded_shape': decoded.shape,
        'faces_found': 0 if bboxes is None else len(bboxes),
        'bboxes': bboxes
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug face detection")
    parser.add_argument('--video', default="test_data/IMG_0942.MOV",
                        help='Test video path')
    parser.add_argument('--frame-num', type=int, default=0,
                        help='Frame number to test')
    args = parser.parse_args()

    video_path = project_root / args.video
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("Face Detection Debug")
    print("=" * 60)
    print(f"Video: {video_path}")
    print()

    # Get video info
    print("Video Information:")
    info = get_video_rotation(video_path)
    if info is None:
        print("  ERROR: Could not read video")
        sys.exit(1)

    print(f"  Reported dimensions: {info['reported_width']}x{info['reported_height']}")
    print(f"  Actual frame shape: {info['frame_shape']}")
    print(f"  Is portrait: {info['is_portrait']}")
    print()

    frame = info['frame']

    # Read specific frame if requested
    if args.frame_num > 0:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  ERROR: Could not read frame {args.frame_num}")
            sys.exit(1)
        print(f"Testing frame {args.frame_num}")
        print()

    # Initialize detector once
    from pymtcnn import MTCNN
    detector = MTCNN()

    # Test different rotations
    print("Testing face detection with different rotations:")
    print("-" * 60)

    rotations = [0, 90, 180, 270]
    results = []

    for rot in rotations:
        result = test_detection_with_rotation(frame, rot, detector)
        results.append(result)
        status = "FOUND" if result['faces_found'] > 0 else "NONE"
        print(f"  Rotation {rot:3d}°: shape={result['shape']}, faces={status}")
        if result['faces_found'] > 0:
            bbox = result['bboxes'][0]
            # bbox has 4 elements [x1, y1, x2, y2] when no confidence returned
            if len(bbox) >= 5:
                print(f"    Bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}], conf={bbox[4]:.3f}")
            else:
                print(f"    Bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    print()

    # Test JPEG encode/decode
    print("Testing JPEG encode/decode (like HPC pipeline):")
    print("-" * 60)
    jpeg_result = test_jpeg_encode_decode(frame, detector)
    status = "FOUND" if jpeg_result['faces_found'] > 0 else "NONE"
    print(f"  Original shape: {jpeg_result['original_shape']}")
    print(f"  Decoded shape: {jpeg_result['decoded_shape']}")
    print(f"  Faces after JPEG: {status}")
    if jpeg_result['faces_found'] > 0:
        bbox = jpeg_result['bboxes'][0]
        if len(bbox) >= 5:
            print(f"    Bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}], conf={bbox[4]:.3f}")
        else:
            print(f"    Bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    print()

    # Determine best rotation
    best = max(results, key=lambda r: r['faces_found'])
    if best['faces_found'] > 0:
        print(f"RESULT: Best rotation is {best['rotation']}° ({best['faces_found']} faces found)")

        if best['rotation'] == 90:
            print("\nTo fix: Add cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) after reading frames")
        elif best['rotation'] == 270:
            print("\nTo fix: Add cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) after reading frames")
        elif best['rotation'] == 180:
            print("\nTo fix: Add cv2.rotate(frame, cv2.ROTATE_180) after reading frames")

        # Check if JPEG encoding breaks it
        if jpeg_result['faces_found'] == 0:
            print("\n*** CRITICAL: JPEG encoding breaks face detection! ***")
            print("The issue is that JPEG encode/decode (used by HPC pipeline) removes faces.")
    else:
        print("RESULT: No faces found with any rotation!")
        print("\nPossible issues:")
        print("  - Face may be too small in frame")
        print("  - Image quality issue after encoding")
        print("  - MTCNN model loading issue")

        # Try with scaled frame
        print("\nTrying with scaled frame (2x)...")
        scaled = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        result = test_detection_with_rotation(scaled, 0, detector)
        if result['faces_found'] > 0:
            print(f"  SUCCESS: Found {result['faces_found']} faces after 2x scaling!")
        else:
            print("  Still no faces found")

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
