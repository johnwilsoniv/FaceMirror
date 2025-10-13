#!/usr/bin/env python3
"""
Test script to inspect OpenFace 3.0 Action Unit output format
"""
import cv2
import numpy as np
import torch
import os
from pathlib import Path

# Set environment variables (same as openface3_detector.py)
os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
os.environ.setdefault('TMPDIR', os.path.expanduser('~/tmp'))

from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

def test_au_output():
    """Test OpenFace 3.0 AU extraction and inspect output format"""

    # Get weights directory
    script_dir = Path(__file__).parent
    weights_dir = script_dir / 'weights'

    print("="*60)
    print("OpenFace 3.0 Action Unit Output Test")
    print("="*60)

    # Initialize models
    print("\n1. Initializing models...")
    device = 'cpu'  # Use CPU for testing

    face_detector = FaceDetector(
        model_path=str(weights_dir / 'Alignment_RetinaFace.pth'),
        device=device,
        confidence_threshold=0.9
    )
    print("   ✓ Face detector loaded")

    multitask_model = MultitaskPredictor(
        model_path=str(weights_dir / 'MTL_backbone.pth'),
        device=device
    )
    print("   ✓ Multitask model loaded")

    # Create a test image with a face (or load from file if available)
    print("\n2. Loading test image...")

    # Try to find a test image in output directory
    output_dir = Path.cwd() / 'output'
    test_image_path = None

    if output_dir.exists():
        # Look for any video file we can extract a frame from
        for video_file in output_dir.glob('*_debug.mp4'):
            cap = cv2.VideoCapture(str(video_file))
            ret, frame = cap.read()
            cap.release()
            if ret:
                test_image = frame
                # Save to temp file for face detector
                test_image_path = script_dir / 'temp_test_frame.jpg'
                cv2.imwrite(str(test_image_path), test_image)
                print(f"   ✓ Loaded frame from {video_file.name}")
                print(f"   ✓ Saved to {test_image_path}")
                break
        else:
            print("   ! No debug videos found, creating synthetic test image")
            test_image = create_synthetic_test_image()
            test_image_path = script_dir / 'temp_test_frame.jpg'
            cv2.imwrite(str(test_image_path), test_image)
    else:
        print("   ! No output directory found, creating synthetic test image")
        test_image = create_synthetic_test_image()
        test_image_path = script_dir / 'temp_test_frame.jpg'
        cv2.imwrite(str(test_image_path), test_image)

    print(f"   Image shape: {test_image.shape}")

    # Detect face
    print("\n3. Detecting face...")
    cropped_face, dets = face_detector.get_face(str(test_image_path))

    if cropped_face is None:
        print("   ✗ No face detected!")
        print("   Creating a simple test case with dummy data...")
        # Create dummy cropped face for testing (as numpy array, not tensor)
        cropped_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print("   Using dummy face image for AU extraction test")
    else:
        print(f"   ✓ Face detected and cropped")
        print(f"   Cropped face shape: {cropped_face.shape}")

    # Extract AUs
    print("\n4. Extracting Action Units...")
    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

    print(f"   ✓ AU extraction complete")
    print(f"\n   AU Output Details:")
    print(f"   - Type: {type(au_output)}")
    print(f"   - Shape: {au_output.shape}")
    print(f"   - Device: {au_output.device}")
    print(f"   - Dtype: {au_output.dtype}")

    # Convert to numpy for inspection
    au_values = au_output.detach().cpu().numpy()
    print(f"\n   AU Values (first sample):")
    print(f"   - Min: {au_values.min():.4f}")
    print(f"   - Max: {au_values.max():.4f}")
    print(f"   - Mean: {au_values.mean():.4f}")
    print(f"   - Number of AUs: {au_values.shape[-1]}")

    print(f"\n   Individual AU values:")
    for i, val in enumerate(au_values[0]):
        print(f"   AU[{i:2d}]: {val:8.4f}")

    # Also test emotion and gaze outputs for completeness
    print(f"\n5. Other outputs:")
    print(f"   Emotion logits shape: {emotion_logits.shape}")
    print(f"   Gaze output shape: {gaze_output.shape}")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

    # Cleanup temp file
    if test_image_path and test_image_path.exists():
        test_image_path.unlink()
        print(f"\nCleaned up temp file: {test_image_path}")

    return au_output

def create_synthetic_test_image():
    """Create a simple test image (gray square)"""
    # Create a 640x480 gray image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    return img

if __name__ == "__main__":
    try:
        test_au_output()
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
