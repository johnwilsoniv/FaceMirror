"""Test PyFaceAU debug mode implementation"""

import cv2
import numpy as np
from pyfaceau.pipeline import FullPythonAUPipeline

# Test image
test_image_path = "test_images/test_frame.jpg"
img = cv2.imread(test_image_path)

if img is None:
    print(f"❌ Could not load test image: {test_image_path}")
    exit(1)

print(f"✓ Loaded test image: {img.shape}")
print()

# Initialize PyFaceAU pipeline with debug mode
print("=" * 60)
print("Initializing PyFaceAU Pipeline with Debug Mode")
print("=" * 60)

pipeline = FullPythonAUPipeline(
    pfld_model='weights/pfld_model.onnx',
    pdm_file='weights/pdm_68_multi_pie.txt',
    au_models_dir='weights/',
    triangulation_file='weights/tris_68_full.txt',
    mtcnn_backend='coreml',  # Use CoreML for speed
    use_coreml_pfld=True,
    debug_mode=True,  # Enable debug mode
    track_faces=False,  # Disable tracking for debug test
    verbose=True  # Enable verbose to see what's happening
)

print("✓ Pipeline initialized")
print()

# Initialize components (lazy initialization)
print("Initializing pipeline components...")
pipeline._initialize_components()
print("✓ Components initialized")
print()

# Process a single frame with debug mode
print("=" * 60)
print("Processing Frame with Debug Mode")
print("=" * 60)

result = pipeline._process_frame(img, frame_idx=0, timestamp=0.0, return_debug=True)

print(f"\n✓ Frame Processing Result:")
print(f"  Success: {result['success']}")

if result['success'] and 'debug_info' in result:
    debug_info = result['debug_info']

    print("\nDebug Info - Component Timing:")
    print("-" * 60)

    components = [
        'face_detection',
        'landmark_detection',
        'pose_estimation',
        'alignment',
        'hog_extraction',
        'geometric_extraction',
        'running_median',
        'au_prediction'
    ]

    total_time = 0.0
    for component in components:
        if component in debug_info:
            info = debug_info[component]
            time_ms = info.get('time_ms', 0)
            total_time += time_ms

            print(f"\n{component.upper().replace('_', ' ')}:")
            print(f"  Time: {time_ms:.2f}ms")

            # Print component-specific info
            if component == 'face_detection':
                print(f"  Faces detected: {info.get('num_faces', 0)}")
                if info.get('bbox') is not None:
                    print(f"  Bbox: {info['bbox']}")
                print(f"  Cached: {info.get('cached', False)}")

            elif component == 'landmark_detection':
                print(f"  Landmarks: {info.get('num_landmarks', 0)} points")
                print(f"  CLNF refined: {info.get('clnf_refined', False)}")

            elif component == 'pose_estimation':
                print(f"  Scale: {info.get('scale', 0):.3f}")
                print(f"  Rotation (rx, ry, rz): {info.get('rotation', [])}")
                print(f"  Translation (tx, ty): {info.get('translation', [])}")

            elif component == 'alignment':
                print(f"  Aligned face shape: {info.get('aligned_face_shape', 'N/A')}")

            elif component == 'hog_extraction':
                print(f"  HOG features: {info.get('hog_shape', 'N/A')}")

            elif component == 'geometric_extraction':
                print(f"  Geometric features: {info.get('geom_shape', 'N/A')}")

            elif component == 'running_median':
                print(f"  Median shape: {info.get('median_shape', 'N/A')}")
                print(f"  Updated histogram: {info.get('update_histogram', False)}")

            elif component == 'au_prediction':
                print(f"  AUs predicted: {info.get('num_aus', 0)}")

    print()
    print("=" * 60)
    print(f"Total Pipeline Time: {total_time:.2f}ms")
    print("=" * 60)

    # Show a few AU predictions
    print("\nSample AU Predictions:")
    au_keys = [k for k in result.keys() if k.startswith('AU') and '_r' in k][:5]
    for au_key in au_keys:
        print(f"  {au_key}: {result[au_key]:.3f}")

    print()
    print("✓ PyFaceAU debug mode test PASSED!")

else:
    print("\n❌ Frame processing failed or debug info not available")
    exit(1)
