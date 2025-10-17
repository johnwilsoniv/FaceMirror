#!/usr/bin/env python3
"""
Quick test to verify ONNX integration for both RetinaFace and STAR detectors
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_onnx_detectors():
    """Test that both ONNX detectors load correctly"""
    print("="*60)
    print("TESTING ONNX INTEGRATION")
    print("="*60)

    # Test RetinaFace ONNX detector
    print("\n1. Testing RetinaFace ONNX Detector...")
    try:
        from onnx_retinaface_detector import OptimizedFaceDetector

        model_dir = os.path.join(os.path.dirname(__file__), 'weights')
        face_detector = OptimizedFaceDetector(
            model_path=f'{model_dir}/Alignment_RetinaFace.pth',
            onnx_model_path=f'{model_dir}/retinaface_mobilenet025_coreml.onnx',
            device='cpu'
        )

        print(f"   ✓ RetinaFace detector loaded successfully")
        print(f"   Backend: {face_detector.backend}")

        if hasattr(face_detector.detector, 'backend'):
            print(f"   ONNX backend: {face_detector.detector.backend}")

    except Exception as e:
        print(f"   ✗ RetinaFace detector failed: {e}")
        import traceback
        traceback.print_exc()

    # Test STAR ONNX detector
    print("\n2. Testing STAR ONNX Detector...")
    try:
        from onnx_star_detector import OptimizedLandmarkDetector

        landmark_detector = OptimizedLandmarkDetector(
            model_path=f'{model_dir}/Landmark_98.pkl',
            onnx_model_path=f'{model_dir}/star_landmark_98_coreml.onnx',
            device='cpu'
        )

        print(f"   ✓ STAR detector loaded successfully")
        print(f"   Backend: {landmark_detector.backend}")

        if hasattr(landmark_detector.detector, 'backend'):
            print(f"   ONNX backend: {landmark_detector.detector.backend}")

    except Exception as e:
        print(f"   ✗ STAR detector failed: {e}")
        import traceback
        traceback.print_exc()

    # Test integrated detector (as used in main application)
    print("\n3. Testing Integrated OpenFace3LandmarkDetector...")
    try:
        from openface3_detector import OpenFace3LandmarkDetector

        detector = OpenFace3LandmarkDetector(debug_mode=False, device='cpu', model_dir=model_dir)

        print("   ✓ Integrated detector loaded successfully")

        if hasattr(detector.face_detector, 'backend'):
            print(f"   Face detector backend: {detector.face_detector.backend}")

        if hasattr(detector.landmark_detector, 'backend'):
            print(f"   Landmark detector backend: {detector.landmark_detector.backend}")

    except Exception as e:
        print(f"   ✗ Integrated detector failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == '__main__':
    test_onnx_detectors()
