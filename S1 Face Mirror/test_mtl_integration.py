#!/usr/bin/env python3
"""
Quick test to verify MTL ONNX integration works correctly
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mtl_integration():
    """Test that MTL ONNX detector loads and works correctly"""
    print("="*60)
    print("TESTING MTL ONNX INTEGRATION")
    print("="*60)

    # Test MTL ONNX detector directly
    print("\n1. Testing MTL ONNX Detector...")
    try:
        from onnx_mtl_detector import OptimizedMultitaskPredictor

        model_dir = os.path.join(os.path.dirname(__file__), 'weights')
        predictor = OptimizedMultitaskPredictor(
            model_path=f'{model_dir}/MTL_backbone.pth',
            onnx_model_path=f'{model_dir}/mtl_efficientnet_b0_coreml.onnx',
            device='cpu'
        )

        print(f"   ✓ MTL predictor loaded successfully")
        print(f"   Backend: {predictor.backend}")

        if hasattr(predictor.predictor, 'backend'):
            print(f"   ONNX backend: {predictor.predictor.backend}")

        # Test prediction with dummy face
        import cv2
        dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emotion, gaze, au = predictor.predict(dummy_face)
        print(f"   ✓ Prediction successful")
        print(f"     Emotion shape: {emotion.shape}")
        print(f"     Gaze shape: {gaze.shape}")
        print(f"     AU shape: {au.shape}")

    except Exception as e:
        print(f"   ✗ MTL predictor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test integration in OpenFace3Processor (used for AU extraction)
    print("\n2. Testing OpenFace3Processor with MTL integration...")
    try:
        from openface_integration import OpenFace3Processor

        processor = OpenFace3Processor(device='cpu', weights_dir=model_dir)

        print("   ✓ OpenFace3Processor loaded successfully")

        if hasattr(processor.multitask_model, 'backend'):
            print(f"   MTL backend: {processor.multitask_model.backend}")

    except Exception as e:
        print(f"   ✗ OpenFace3Processor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_mtl_integration()
    sys.exit(0 if success else 1)
