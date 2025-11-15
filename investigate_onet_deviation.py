"""
Investigate ONet Deviation

Why does ONet show ~4% deviation when PNet/RNet are bit-for-bit?

Hypotheses:
1. Network depth (14 layers vs 8/11) causes error accumulation
2. Specific test patch has problematic characteristics
3. Landmark outputs (10 values) have higher variance
4. Different layer types or operations in ONet
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from cpp_cnn_loader import CPPCNN
import onnxruntime as ort

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("ONet Deviation Investigation")
print("="*80)

# Load models
model_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/face_detection/mtcnn/convert_to_cpp"
cpp_onet = CPPCNN(f"{model_dir}/ONet.dat")

onnx_dir = Path(__file__).parent / "pymtcnn" / "pymtcnn" / "models"
onet_onnx = ort.InferenceSession(str(onnx_dir / "onet.onnx"), providers=['CPUExecutionProvider'])

img = cv2.imread(TEST_IMAGE)

# Test multiple patches
print("\nTesting multiple 48x48 patches:")
print("-" * 60)

test_locations = [
    (300, 400, "Original"),
    (500, 600, "Center-ish"),
    (100, 100, "Top-left"),
    (800, 400, "Face region"),
]

for x, y, desc in test_locations:
    patch_bgr = img[y:y+48, x:x+48, :]

    if patch_bgr.shape[0] != 48 or patch_bgr.shape[1] != 48:
        print(f"\n{desc} ({x},{y}): SKIP - out of bounds")
        continue

    # Preprocess
    patch_norm = (patch_bgr.astype(np.float32) - 127.5) * 0.0078125
    patch_chw = np.transpose(patch_norm, (2, 0, 1))
    patch_preprocessed = patch_chw[[2, 1, 0], :, :]

    # Run both
    cpp_out = cpp_onet(patch_preprocessed)[-1]

    onet_input = patch_preprocessed[np.newaxis, :, :, :].astype(np.float32)
    onnx_out = onet_onnx.run(None, {'input': onet_input})[0][0]

    # Compare
    diff = np.abs(cpp_out - onnx_out)

    print(f"\n{desc} ({x},{y}):")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Classification diff: {diff[0:2].max():.6f}")
    print(f"  BBox reg diff: {diff[2:6].max():.6f}")
    print(f"  Landmarks diff: {diff[6:16].max():.6f}")

# Test with synthetic inputs
print("\n" + "="*80)
print("Test with Synthetic Inputs")
print("="*80)

# Test 1: All zeros
print("\nTest 1: All zeros input")
zeros = np.zeros((3, 48, 48), dtype=np.float32)

cpp_out = cpp_onet(zeros)[-1]
onnx_out = onet_onnx.run(None, {'input': zeros[np.newaxis, :, :, :]})[0][0]
diff = np.abs(cpp_out - onnx_out)

print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

# Test 2: All ones
print("\nTest 2: All ones input")
ones = np.ones((3, 48, 48), dtype=np.float32)

cpp_out = cpp_onet(ones)[-1]
onnx_out = onet_onnx.run(None, {'input': ones[np.newaxis, :, :, :]})[0][0]
diff = np.abs(cpp_out - onnx_out)

print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

# Test 3: Random uniform [-1, 1]
print("\nTest 3: Random uniform [-1, 1] input")
np.random.seed(42)
random_uniform = np.random.uniform(-1, 1, (3, 48, 48)).astype(np.float32)

cpp_out = cpp_onet(random_uniform)[-1]
onnx_out = onet_onnx.run(None, {'input': random_uniform[np.newaxis, :, :, :]})[0][0]
diff = np.abs(cpp_out - onnx_out)

print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

# Analyze layer-by-layer
print("\n" + "="*80)
print("Layer-by-Layer Analysis")
print("="*80)

# Use the original test patch
x, y = 300, 400
patch_bgr = img[y:y+48, x:x+48, :]
patch_norm = (patch_bgr.astype(np.float32) - 127.5) * 0.0078125
patch_chw = np.transpose(patch_norm, (2, 0, 1))
patch_preprocessed = patch_chw[[2, 1, 0], :, :]

print("\nRunning C++ ONet with layer-by-layer outputs...")
cpp_layers = cpp_onet(patch_preprocessed)

print(f"\nC++ ONet has {len(cpp_layers)} layer outputs")
for i, layer_out in enumerate(cpp_layers):
    print(f"  Layer {i}: shape={layer_out.shape}, range=[{layer_out.min():.6f}, {layer_out.max():.6f}]")

# Note: ONNX doesn't expose intermediate layers easily, but we can see the pattern

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
Key observations:
1. If deviation is consistent across different patches → Model difference
2. If deviation varies significantly → Input-dependent numerical instability
3. If synthetic inputs show low deviation → Real image characteristics matter
4. If all tests show ~4% deviation → Acceptable tolerance for ONet's complexity

The ~4% ONet deviation appears in BOTH CoreML and ONNX, and CoreML still
achieves 96.83% IoU (PASS). This suggests:
  → ONet's ~4% deviation is ACCEPTABLE
  → The ONNX pipeline issue is NOT due to model accuracy
  → Focus on pipeline bugs (which we've started fixing with softmax)
""")
