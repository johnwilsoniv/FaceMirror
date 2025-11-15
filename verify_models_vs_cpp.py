"""
Verify ONNX and CoreML Models vs C++ Gold Standard

Compare model outputs when given identical inputs:
- C++ (loaded from .dat files) - GOLD STANDARD
- CoreML (FP32)
- ONNX

This isolates whether issues are in the models or the pipeline.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add cpp_cnn_loader to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from cpp_cnn_loader import CPPCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("Model Verification: C++ vs CoreML vs ONNX")
print("="*80)

# Load test image
img = cv2.imread(TEST_IMAGE)
print(f"\nTest image: {img.shape}, BGR format")

# Load C++ models (gold standard)
model_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/face_detection/mtcnn/convert_to_cpp"

print("\nLoading C++ models (gold standard)...")
cpp_pnet = CPPCNN(f"{model_dir}/PNet.dat")
cpp_rnet = CPPCNN(f"{model_dir}/RNet.dat")
cpp_onet = CPPCNN(f"{model_dir}/ONet.dat")
print("✓ C++ models loaded")

# Load ONNX models
import onnxruntime as ort

onnx_dir = Path(__file__).parent / "pymtcnn" / "pymtcnn" / "models"
print(f"\nLoading ONNX models from {onnx_dir}...")
pnet_onnx = ort.InferenceSession(str(onnx_dir / "pnet.onnx"), providers=['CPUExecutionProvider'])
rnet_onnx = ort.InferenceSession(str(onnx_dir / "rnet.onnx"), providers=['CPUExecutionProvider'])
onet_onnx = ort.InferenceSession(str(onnx_dir / "onet.onnx"), providers=['CPUExecutionProvider'])
print("✓ ONNX models loaded")

# Load CoreML models
from pymtcnn.backends.coreml_backend import CoreMLMTCNN

print("\nLoading CoreML models...")
coreml_detector = CoreMLMTCNN()
print("✓ CoreML models loaded")

print("\n" + "="*80)
print("Test 1: PNet Model Comparison")
print("="*80)

# Prepare PNet input
h, w = img.shape[:2]
min_face_size = 60
m = 12.0 / min_face_size
scale = m

hs = int(np.ceil(h * scale))
ws = int(np.ceil(w * scale))

print(f"\nScale: {scale:.6f}, Target size: {hs}x{ws}")

# Preprocess (C++ method: BGR -> normalize -> transpose -> flip to RGB)
img_resized = cv2.resize(img.astype(np.float32), (ws, hs), interpolation=cv2.INTER_LINEAR)
img_norm = (img_resized - 127.5) * 0.0078125
img_chw = np.transpose(img_norm, (2, 0, 1))
img_preprocessed = img_chw[[2, 1, 0], :, :]  # Flip BGR to RGB

print(f"Preprocessed: shape={img_preprocessed.shape}, range=[{img_preprocessed.min():.4f}, {img_preprocessed.max():.4f}]")

# Run C++ PNet
print("\n--- C++ PNet (Gold Standard) ---")
cpp_pnet_out = cpp_pnet(img_preprocessed)[-1]  # Get final output
cpp_pnet_out_hwc = cpp_pnet_out.transpose(1, 2, 0)  # CHW -> HWC

print(f"Output shape: {cpp_pnet_out.shape} (CHW)")
print(f"Output range: [{cpp_pnet_out.min():.6f}, {cpp_pnet_out.max():.6f}]")
print(f"Logit not-face [0,:,:] range: [{cpp_pnet_out[0,:,:].min():.6f}, {cpp_pnet_out[0,:,:].max():.6f}]")
print(f"Logit face [1,:,:] range: [{cpp_pnet_out[1,:,:].min():.6f}, {cpp_pnet_out[1,:,:].max():.6f}]")

# Run CoreML PNet
print("\n--- CoreML PNet ---")
coreml_pnet_out = coreml_detector._run_pnet(img_preprocessed)
coreml_pnet_out_chw = coreml_pnet_out[0]  # Remove batch dim
coreml_pnet_out_hwc = coreml_pnet_out_chw.transpose(1, 2, 0)  # CHW -> HWC

print(f"Output shape: {coreml_pnet_out_chw.shape} (CHW)")
print(f"Output range: [{coreml_pnet_out_chw.min():.6f}, {coreml_pnet_out_chw.max():.6f}]")
print(f"Logit not-face [0,:,:] range: [{coreml_pnet_out_chw[0,:,:].min():.6f}, {coreml_pnet_out_chw[0,:,:].max():.6f}]")
print(f"Logit face [1,:,:] range: [{coreml_pnet_out_chw[1,:,:].min():.6f}, {coreml_pnet_out_chw[1,:,:].max():.6f}]")

# Run ONNX PNet
print("\n--- ONNX PNet ---")
pnet_input_batch = img_preprocessed[np.newaxis, :, :, :].astype(np.float32)
onnx_pnet_out = pnet_onnx.run(None, {'input': pnet_input_batch})[0]
onnx_pnet_out_chw = onnx_pnet_out[0]  # Remove batch dim
onnx_pnet_out_hwc = onnx_pnet_out_chw.transpose(1, 2, 0)  # CHW -> HWC

print(f"Output shape: {onnx_pnet_out_chw.shape} (CHW)")
print(f"Output range: [{onnx_pnet_out_chw.min():.6f}, {onnx_pnet_out_chw.max():.6f}]")
print(f"Logit not-face [0,:,:] range: [{onnx_pnet_out_chw[0,:,:].min():.6f}, {onnx_pnet_out_chw[0,:,:].max():.6f}]")
print(f"Logit face [1,:,:] range: [{onnx_pnet_out_chw[1,:,:].min():.6f}, {onnx_pnet_out_chw[1,:,:].max():.6f}]")

# Compare
print("\n--- PNet Comparison ---")
diff_coreml = np.abs(cpp_pnet_out - coreml_pnet_out_chw)
diff_onnx = np.abs(cpp_pnet_out - onnx_pnet_out_chw)

print(f"\nC++ vs CoreML:")
print(f"  Mean diff: {diff_coreml.mean():.10f}")
print(f"  Max diff:  {diff_coreml.max():.10f}")

print(f"\nC++ vs ONNX:")
print(f"  Mean diff: {diff_onnx.mean():.10f}")
print(f"  Max diff:  {diff_onnx.max():.10f}")

if diff_coreml.max() < 1e-5:
    print("  ✓ CoreML: BIT-FOR-BIT ACCURACY")
elif diff_coreml.max() < 0.03:
    print(f"  ✓ CoreML: HIGH ACCURACY (within ~2% tolerance)")
else:
    print(f"  ✗ CoreML: SIGNIFICANT DIFFERENCES")

if diff_onnx.max() < 1e-5:
    print("  ✓ ONNX: BIT-FOR-BIT ACCURACY")
elif diff_onnx.max() < 1e-3:
    print("  ✓ ONNX: HIGH ACCURACY")
else:
    print(f"  ✗ ONNX: SIGNIFICANT DIFFERENCES")

print("\n" + "="*80)
print("Test 2: RNet Model Comparison")
print("="*80)

# Extract 24x24 patch
patch_size = 24
y_start, x_start = 400, 300
patch_bgr = img[y_start:y_start+patch_size, x_start:x_start+patch_size, :]

if patch_bgr.shape[0] == patch_size and patch_bgr.shape[1] == patch_size:
    print(f"\nTest patch: {patch_bgr.shape} from ({x_start}, {y_start})")

    # Preprocess
    patch_norm = (patch_bgr.astype(np.float32) - 127.5) * 0.0078125
    patch_chw = np.transpose(patch_norm, (2, 0, 1))
    patch_preprocessed = patch_chw[[2, 1, 0], :, :]  # BGR to RGB

    # Run C++ RNet
    print("\n--- C++ RNet (Gold Standard) ---")
    cpp_rnet_out = cpp_rnet(patch_preprocessed)[-1]
    print(f"Output: {cpp_rnet_out}")

    # Run CoreML RNet
    print("\n--- CoreML RNet ---")
    coreml_rnet_out = coreml_detector._run_rnet(patch_preprocessed)
    print(f"Output: {coreml_rnet_out}")

    # Run ONNX RNet
    print("\n--- ONNX RNet ---")
    rnet_input_batch = patch_preprocessed[np.newaxis, :, :, :].astype(np.float32)
    onnx_rnet_out = rnet_onnx.run(None, {'input': rnet_input_batch})[0][0]
    print(f"Output: {onnx_rnet_out}")

    # Compare
    print("\n--- RNet Comparison ---")
    diff_coreml = np.abs(cpp_rnet_out - coreml_rnet_out)
    diff_onnx = np.abs(cpp_rnet_out - onnx_rnet_out)

    print(f"\nC++ vs CoreML max diff: {diff_coreml.max():.10f}")
    print(f"C++ vs ONNX max diff:   {diff_onnx.max():.10f}")

    if diff_coreml.max() < 1e-5:
        print("✓ CoreML: BIT-FOR-BIT ACCURACY")
    elif diff_coreml.max() < 0.03:
        print("✓ CoreML: HIGH ACCURACY (within ~2% tolerance)")
    else:
        print("✗ CoreML: SIGNIFICANT DIFFERENCES")

    if diff_onnx.max() < 1e-5:
        print("✓ ONNX: BIT-FOR-BIT ACCURACY")
    elif diff_onnx.max() < 1e-3:
        print("✓ ONNX: HIGH ACCURACY")
    else:
        print("✗ ONNX: SIGNIFICANT DIFFERENCES")

print("\n" + "="*80)
print("Test 3: ONet Model Comparison")
print("="*80)

# Extract 48x48 patch
patch_size = 48
patch_bgr = img[y_start:y_start+patch_size, x_start:x_start+patch_size, :]

if patch_bgr.shape[0] == patch_size and patch_bgr.shape[1] == patch_size:
    print(f"\nTest patch: {patch_bgr.shape} from ({x_start}, {y_start})")

    # Preprocess
    patch_norm = (patch_bgr.astype(np.float32) - 127.5) * 0.0078125
    patch_chw = np.transpose(patch_norm, (2, 0, 1))
    patch_preprocessed = patch_chw[[2, 1, 0], :, :]  # BGR to RGB

    # Run C++ ONet
    print("\n--- C++ ONet (Gold Standard) ---")
    cpp_onet_out = cpp_onet(patch_preprocessed)[-1]
    print(f"Output: {cpp_onet_out}")

    # Run CoreML ONet
    print("\n--- CoreML ONet ---")
    coreml_onet_out = coreml_detector._run_onet(patch_preprocessed)
    print(f"Output: {coreml_onet_out}")

    # Run ONNX ONet
    print("\n--- ONNX ONet ---")
    onet_input_batch = patch_preprocessed[np.newaxis, :, :, :].astype(np.float32)
    onnx_onet_out = onet_onnx.run(None, {'input': onet_input_batch})[0][0]
    print(f"Output: {onnx_onet_out}")

    # Compare
    print("\n--- ONet Comparison ---")
    diff_coreml = np.abs(cpp_onet_out - coreml_onet_out)
    diff_onnx = np.abs(cpp_onet_out - onnx_onet_out)

    print(f"\nC++ vs CoreML max diff: {diff_coreml.max():.10f}")
    print(f"C++ vs ONNX max diff:   {diff_onnx.max():.10f}")

    if diff_coreml.max() < 1e-5:
        print("✓ CoreML: BIT-FOR-BIT ACCURACY")
    elif diff_coreml.max() < 0.03:
        print("✓ CoreML: HIGH ACCURACY (within ~2% tolerance)")
    else:
        print("✗ CoreML: SIGNIFICANT DIFFERENCES")

    if diff_onnx.max() < 1e-5:
        print("✓ ONNX: BIT-FOR-BIT ACCURACY")
    elif diff_onnx.max() < 1e-3:
        print("✓ ONNX: HIGH ACCURACY")
    else:
        print("✗ ONNX: SIGNIFICANT DIFFERENCES")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
If all models show bit-for-bit or high accuracy vs C++:
  → Models are CORRECT
  → Issue is in the ONNX PIPELINE (bbox generation, NMS, coordinate mapping, etc.)

If any model shows significant differences:
  → That model needs investigation/regeneration
""")
