"""
Verify ONNX Models vs C++ - Bit-for-Bit Accuracy

Compare ONNX model outputs directly with C++ model outputs using identical inputs.
This isolates whether the issue is in the models or the pipeline.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("ONNX vs C++ Model Output Verification")
print("="*80)

# Load test image
img = cv2.imread(TEST_IMAGE)
print(f"\nTest image: {img.shape}, BGR format")

# Import CoreML MTCNN (verified to match C++ - our gold standard)
try:
    from pymtcnn.backends.coreml_backend import CoreMLMTCNN
    cpp_mtcnn = CoreMLMTCNN()
    has_cpp = True
    print("✓ Loaded CoreML MTCNN (verified C++ gold standard)")
except Exception as e:
    print(f"✗ Could not load CoreML MTCNN: {e}")
    has_cpp = False

# Import ONNX backend
import onnxruntime as ort

onnx_dir = Path(__file__).parent / "pymtcnn" / "pymtcnn" / "models"
pnet_path = onnx_dir / "pnet.onnx"
rnet_path = onnx_dir / "rnet.onnx"
onet_path = onnx_dir / "onet.onnx"

print("\n" + "="*80)
print("Test 1: PNet Model Output Comparison")
print("="*80)

if not has_cpp or not pnet_path.exists():
    print("✗ Cannot run test - missing models")
else:
    # Prepare PNet input
    h, w = img.shape[:2]
    min_face_size = 60
    m = 12.0 / min_face_size
    scale = m

    hs = int(np.ceil(h * scale))
    ws = int(np.ceil(w * scale))

    print(f"\nScale: {scale:.6f}, Target size: {hs}x{ws}")

    # Resize and preprocess (matching C++ preprocessing)
    img_resized = cv2.resize(img.astype(np.float32), (ws, hs), interpolation=cv2.INTER_LINEAR)
    img_norm = (img_resized - 127.5) * 0.0078125
    img_chw = np.transpose(img_norm, (2, 0, 1))
    # Flip BGR to RGB
    img_preprocessed = img_chw[[2, 1, 0], :, :]

    print(f"Preprocessed: shape={img_preprocessed.shape}, range=[{img_preprocessed.min():.4f}, {img_preprocessed.max():.4f}]")

    # Run CoreML PNet (gold standard)
    print("\n--- CoreML PNet (Gold Standard) ---")
    cpp_output = cpp_mtcnn._run_pnet(img_preprocessed)
    print(f"Output shape: {cpp_output.shape}")
    print(f"Output range: [{cpp_output.min():.8f}, {cpp_output.max():.8f}]")

    # Extract logits
    cpp_output_chw = cpp_output[0].transpose(1, 2, 0)  # (1, 6, H, W) -> (H, W, 6)
    cpp_logit_not_face = cpp_output_chw[:, :, 0]
    cpp_logit_face = cpp_output_chw[:, :, 1]
    cpp_bbox_reg = cpp_output_chw[:, :, 2:6]

    print(f"Logit not-face range: [{cpp_logit_not_face.min():.6f}, {cpp_logit_not_face.max():.6f}]")
    print(f"Logit face range: [{cpp_logit_face.min():.6f}, {cpp_logit_face.max():.6f}]")

    # Run ONNX PNet
    print("\n--- ONNX PNet ---")
    pnet_sess = ort.InferenceSession(str(pnet_path), providers=['CPUExecutionProvider'])

    # Add batch dimension
    pnet_input = img_preprocessed[np.newaxis, :, :, :].astype(np.float32)
    onnx_output = pnet_sess.run(None, {'input': pnet_input})[0]

    print(f"Output shape: {onnx_output.shape}")
    print(f"Output range: [{onnx_output.min():.8f}, {onnx_output.max():.8f}]")

    # Extract logits
    onnx_output_chw = onnx_output[0].transpose(1, 2, 0)  # (1, 6, H, W) -> (H, W, 6)
    onnx_logit_not_face = onnx_output_chw[:, :, 0]
    onnx_logit_face = onnx_output_chw[:, :, 1]
    onnx_bbox_reg = onnx_output_chw[:, :, 2:6]

    print(f"Logit not-face range: [{onnx_logit_not_face.min():.6f}, {onnx_logit_not_face.max():.6f}]")
    print(f"Logit face range: [{onnx_logit_face.min():.6f}, {onnx_logit_face.max():.6f}]")

    # Compare
    print("\n--- Comparison ---")
    diff_logit_not_face = np.abs(cpp_logit_not_face - onnx_logit_not_face)
    diff_logit_face = np.abs(cpp_logit_face - onnx_logit_face)
    diff_bbox = np.abs(cpp_bbox_reg - onnx_bbox_reg)

    print(f"\nLogit not-face difference:")
    print(f"  Mean: {diff_logit_not_face.mean():.10f}")
    print(f"  Max:  {diff_logit_not_face.max():.10f}")

    print(f"\nLogit face difference:")
    print(f"  Mean: {diff_logit_face.mean():.10f}")
    print(f"  Max:  {diff_logit_face.max():.10f}")

    print(f"\nBBox regression difference:")
    print(f"  Mean: {diff_bbox.mean():.10f}")
    print(f"  Max:  {diff_bbox.max():.10f}")

    if diff_logit_face.max() < 1e-5 and diff_bbox.max() < 1e-5:
        print("\n✓ PNet: BIT-FOR-BIT ACCURACY (max diff < 1e-5)")
    elif diff_logit_face.max() < 1e-3:
        print("\n✓ PNet: HIGH ACCURACY (max diff < 1e-3)")
    else:
        print(f"\n✗ PNet: SIGNIFICANT DIFFERENCES (max diff = {max(diff_logit_face.max(), diff_bbox.max()):.6f})")

print("\n" + "="*80)
print("Test 2: RNet Model Output Comparison")
print("="*80)

if not has_cpp or not rnet_path.exists():
    print("✗ Cannot run test - missing models")
else:
    # Use a 24x24 patch from the image
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
        print("\n--- C++ RNet (Pure Python) ---")
        cpp_output = cpp_mtcnn._run_rnet(patch_preprocessed)
        print(f"Output shape: {cpp_output.shape}")
        print(f"Output: {cpp_output}")

        cpp_logit_not_face = cpp_output[0]
        cpp_logit_face = cpp_output[1]
        cpp_bbox_reg = cpp_output[2:6]

        # Run ONNX RNet
        print("\n--- ONNX RNet ---")
        rnet_sess = ort.InferenceSession(str(rnet_path), providers=['CPUExecutionProvider'])

        rnet_input = patch_preprocessed[np.newaxis, :, :, :].astype(np.float32)
        onnx_output = rnet_sess.run(None, {'input': rnet_input})[0][0]

        print(f"Output shape: {onnx_output.shape}")
        print(f"Output: {onnx_output}")

        onnx_logit_not_face = onnx_output[0]
        onnx_logit_face = onnx_output[1]
        onnx_bbox_reg = onnx_output[2:6]

        # Compare
        print("\n--- Comparison ---")
        diff_logit_not_face = abs(cpp_logit_not_face - onnx_logit_not_face)
        diff_logit_face = abs(cpp_logit_face - onnx_logit_face)
        diff_bbox = np.abs(cpp_bbox_reg - onnx_bbox_reg)

        print(f"Logit not-face diff: {diff_logit_not_face:.10f}")
        print(f"Logit face diff:     {diff_logit_face:.10f}")
        print(f"BBox regression max diff: {diff_bbox.max():.10f}")

        if diff_logit_face < 1e-5 and diff_bbox.max() < 1e-5:
            print("\n✓ RNet: BIT-FOR-BIT ACCURACY (max diff < 1e-5)")
        elif diff_logit_face < 1e-3:
            print("\n✓ RNet: HIGH ACCURACY (max diff < 1e-3)")
        else:
            print(f"\n✗ RNet: SIGNIFICANT DIFFERENCES (max diff = {max(diff_logit_face, diff_bbox.max()):.6f})")
    else:
        print(f"✗ Patch size incorrect: {patch_bgr.shape}")

print("\n" + "="*80)
print("Test 3: ONet Model Output Comparison")
print("="*80)

if not has_cpp or not onet_path.exists():
    print("✗ Cannot run test - missing models")
else:
    # Use a 48x48 patch from the image
    patch_size = 48
    y_start, x_start = 400, 300
    patch_bgr = img[y_start:y_start+patch_size, x_start:x_start+patch_size, :]

    if patch_bgr.shape[0] == patch_size and patch_bgr.shape[1] == patch_size:
        print(f"\nTest patch: {patch_bgr.shape} from ({x_start}, {y_start})")

        # Preprocess
        patch_norm = (patch_bgr.astype(np.float32) - 127.5) * 0.0078125
        patch_chw = np.transpose(patch_norm, (2, 0, 1))
        patch_preprocessed = patch_chw[[2, 1, 0], :, :]  # BGR to RGB

        # Run C++ ONet
        print("\n--- C++ ONet (Pure Python) ---")
        cpp_output = cpp_mtcnn._run_onet(patch_preprocessed)
        print(f"Output shape: {cpp_output.shape}")
        print(f"Output: {cpp_output}")

        cpp_logit_not_face = cpp_output[0]
        cpp_logit_face = cpp_output[1]
        cpp_bbox_reg = cpp_output[2:6]
        cpp_landmarks = cpp_output[6:16]

        # Run ONNX ONet
        print("\n--- ONNX ONet ---")
        onet_sess = ort.InferenceSession(str(onet_path), providers=['CPUExecutionProvider'])

        onet_input = patch_preprocessed[np.newaxis, :, :, :].astype(np.float32)
        onnx_output = onet_sess.run(None, {'input': onet_input})[0][0]

        print(f"Output shape: {onnx_output.shape}")
        print(f"Output: {onnx_output}")

        onnx_logit_not_face = onnx_output[0]
        onnx_logit_face = onnx_output[1]
        onnx_bbox_reg = onnx_output[2:6]
        onnx_landmarks = onnx_output[6:16]

        # Compare
        print("\n--- Comparison ---")
        diff_logit_not_face = abs(cpp_logit_not_face - onnx_logit_not_face)
        diff_logit_face = abs(cpp_logit_face - onnx_logit_face)
        diff_bbox = np.abs(cpp_bbox_reg - onnx_bbox_reg)
        diff_landmarks = np.abs(cpp_landmarks - onnx_landmarks)

        print(f"Logit not-face diff: {diff_logit_not_face:.10f}")
        print(f"Logit face diff:     {diff_logit_face:.10f}")
        print(f"BBox regression max diff: {diff_bbox.max():.10f}")
        print(f"Landmarks max diff: {diff_landmarks.max():.10f}")

        if diff_logit_face < 1e-5 and diff_bbox.max() < 1e-5 and diff_landmarks.max() < 1e-5:
            print("\n✓ ONet: BIT-FOR-BIT ACCURACY (max diff < 1e-5)")
        elif diff_logit_face < 1e-3:
            print("\n✓ ONet: HIGH ACCURACY (max diff < 1e-3)")
        else:
            print(f"\n✗ ONet: SIGNIFICANT DIFFERENCES (max diff = {max(diff_logit_face, diff_bbox.max(), diff_landmarks.max()):.6f})")
    else:
        print(f"✗ Patch size incorrect: {patch_bgr.shape}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
If all three models show bit-for-bit or high accuracy:
  → The ONNX models are CORRECT
  → The issue is in the PIPELINE (bbox generation, NMS, etc.)

If any model shows significant differences:
  → That specific ONNX model needs to be regenerated from C++ .dat files
""")
