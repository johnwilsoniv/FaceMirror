"""
Validate ONNX Model Outputs - Layer-by-Layer

Test PNet, RNet, ONet model inference to confirm near bit-for-bit accuracy.
Focus on:
1. Input preprocessing (BGR vs RGB, normalization)
2. Model output validation
3. Output tensor shape and transpose/flatten ordering
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add pymtcnn to path
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("ONNX Model Output Validation")
print("="*80)

# Load test image
img = cv2.imread(TEST_IMAGE)
print(f"\nOriginal image: {img.shape}, dtype={img.dtype}, BGR format")

# Import both backends
from pymtcnn.backends.onnx_backend import ONNXMTCNN
from pymtcnn.backends.coreml_backend import CoreMLMTCNN

# Skip archive detector - focus on current vs base comparison
has_working = False
print("Skipping archive detector - comparing current ONNX vs CoreML/Base")

print("\n" + "="*80)
print("Test 1: PNet Input Preprocessing")
print("="*80)

# Prepare PNet input - first scale
h, w = img.shape[:2]
min_face_size = 60
factor = 0.709
m = 12.0 / min_face_size
scale = m

hs = int(np.ceil(h * scale))
ws = int(np.ceil(w * scale))

print(f"\nFirst pyramid scale: {scale:.6f}")
print(f"Scaled image size: {hs} x {ws}")

# Method 1: Current ONNX backend preprocessing
print("\n--- Current ONNX Backend Preprocessing ---")
img_resized = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)
print(f"After resize: shape={img_resized.shape}, dtype={img_resized.dtype}, range=[{img_resized.min()}, {img_resized.max()}]")

# BGR -> RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
print(f"After BGR→RGB: range=[{img_rgb.min()}, {img_rgb.max()}]")

# Normalize to [-1, 1]
img_norm_current = (img_rgb.astype(np.float32) - 127.5) / 128.0
print(f"After normalize: range=[{img_norm_current.min():.4f}, {img_norm_current.max():.4f}]")

# Transpose HWC -> CHW
pnet_input_current = img_norm_current.transpose(2, 0, 1)
print(f"After transpose: shape={pnet_input_current.shape}")

# Add batch dimension
pnet_input_current_batch = pnet_input_current[np.newaxis, :, :, :].astype(np.float32)
print(f"After add batch: shape={pnet_input_current_batch.shape}")

# Method 2: Working archive ONNX preprocessing
if has_working:
    print("\n--- Archive Working ONNX Preprocessing ---")
    working_detector = WorkingONNXMTCNN()
    img_resized_working = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)

    # Their _preprocess method: normalize, transpose, flip BGR->RGB
    img_norm_working = (img_resized_working.astype(np.float32) - 127.5) * 0.0078125
    img_chw_working = np.transpose(img_norm_working, (2, 0, 1))
    # Flip BGR to RGB
    img_preprocessed_working = img_chw_working[[2, 1, 0], :, :]

    print(f"After full preprocess: shape={img_preprocessed_working.shape}, range=[{img_preprocessed_working.min():.4f}, {img_preprocessed_working.max():.4f}]")

    # Add batch dimension
    pnet_input_working_batch = img_preprocessed_working[np.newaxis, :, :, :].astype(np.float32)
    print(f"After add batch: shape={pnet_input_working_batch.shape}")

    # Compare preprocessing
    print("\n--- Preprocessing Comparison ---")
    # Compare current (RGB in HWC) vs working (flipped BGR->RGB in CHW)
    # Current: RGB in CHW order
    # Working: BGR->RGB flipped in CHW order

    # They should be identical if done correctly
    diff = np.abs(pnet_input_current_batch - pnet_input_working_batch)
    print(f"Absolute difference:")
    print(f"  Mean: {diff.mean():.8f}")
    print(f"  Max:  {diff.max():.8f}")
    print(f"  Median: {np.median(diff):.8f}")

    if diff.max() < 1e-6:
        print("✓ Preprocessing methods are identical")
    elif diff.max() < 1e-3:
        print("⚠ Small preprocessing differences (acceptable)")
    else:
        print("✗ SIGNIFICANT preprocessing mismatch!")

# Method 3: CoreML/Base preprocessing (ground truth)
print("\n--- CoreML/Base Preprocessing (Ground Truth) ---")
img_resized_base = cv2.resize(img.astype(np.float32), (ws, hs), interpolation=cv2.INTER_LINEAR)
img_norm_base = (img_resized_base - 127.5) * 0.0078125
img_chw_base = np.transpose(img_norm_base, (2, 0, 1))
# Flip BGR to RGB
img_preprocessed_base = img_chw_base[[2, 1, 0], :, :]
print(f"After full preprocess: shape={img_preprocessed_base.shape}, range=[{img_preprocessed_base.min():.4f}, {img_preprocessed_base.max():.4f}]")

pnet_input_base = img_preprocessed_base[np.newaxis, :, :, :].astype(np.float32)

# Compare current vs base
print("\n--- Current ONNX vs Base Comparison ---")
diff_vs_base = np.abs(pnet_input_current_batch - pnet_input_base)
print(f"Absolute difference:")
print(f"  Mean: {diff_vs_base.mean():.8f}")
print(f"  Max:  {diff_vs_base.max():.8f}")

if diff_vs_base.max() > 1e-3:
    print("✗ CRITICAL: Current ONNX preprocessing differs from Base!")
    print("\nDEBUG: Channel order check")
    print(f"Current ONNX input[0,0,0:5,0]: {pnet_input_current_batch[0,0,0:5,0]}")
    print(f"Base input[0,0,0:5,0]:         {pnet_input_base[0,0,0:5,0]}")
    print(f"Current ONNX input[0,2,0:5,0]: {pnet_input_current_batch[0,2,0:5,0]}")
    print(f"Base input[0,2,0:5,0]:         {pnet_input_base[0,2,0:5,0]}")
else:
    print("✓ Current ONNX preprocessing matches Base")

print("\n" + "="*80)
print("Test 2: PNet Model Inference")
print("="*80)

# Load ONNX PNet model
import onnxruntime as ort

model_path = Path(__file__).parent / "pymtcnn" / "pymtcnn" / "models" / "pnet.onnx"
if not model_path.exists():
    print(f"✗ ONNX model not found: {model_path}")
else:
    print(f"Loading: {model_path}")
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    # Get input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    print(f"Input shape: {sess.get_inputs()[0].shape}")
    print(f"Output shape: {sess.get_outputs()[0].shape}")

    # Test with current preprocessing
    print("\n--- Running PNet with Current Preprocessing ---")
    output_current = sess.run(None, {input_name: pnet_input_current_batch})[0]
    print(f"Output shape: {output_current.shape}")
    print(f"Output range: [{output_current.min():.6f}, {output_current.max():.6f}]")
    print(f"Output[0,1,:,:] (face prob) range: [{output_current[0,1,:,:].min():.6f}, {output_current[0,1,:,:].max():.6f}]")

    # Apply softmax to get probabilities
    logit_not_face = output_current[0, 0, :, :]
    logit_face = output_current[0, 1, :, :]
    prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    print(f"Face probability range: [{prob_face.min():.6f}, {prob_face.max():.6f}]")
    print(f"Faces above 0.6 threshold: {(prob_face > 0.6).sum()}")

    # Test with base preprocessing
    print("\n--- Running PNet with Base Preprocessing ---")
    output_base = sess.run(None, {input_name: pnet_input_base})[0]
    print(f"Output shape: {output_base.shape}")
    print(f"Output range: [{output_base.min():.6f}, {output_base.max():.6f}]")

    # Compare outputs
    print("\n--- PNet Output Comparison ---")
    diff_output = np.abs(output_current - output_base)
    print(f"Absolute difference:")
    print(f"  Mean: {diff_output.mean():.8f}")
    print(f"  Max:  {diff_output.max():.8f}")

    if diff_output.max() < 1e-5:
        print("✓ PNet outputs are identical (bit-for-bit accuracy)")
    elif diff_output.max() < 1e-3:
        print("✓ PNet outputs match within tolerance (<1e-3)")
    else:
        print("✗ SIGNIFICANT PNet output mismatch!")

print("\n" + "="*80)
print("Test 3: Check ONNX Model Input Requirements")
print("="*80)

# Check if models expect specific preprocessing
print("\nChecking ONNX model metadata...")
if model_path.exists():
    import onnx
    model = onnx.load(str(model_path))

    print(f"Model IR version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")

    # Check for preprocessing metadata
    if model.metadata_props:
        print("\nMetadata properties:")
        for prop in model.metadata_props:
            print(f"  {prop.key}: {prop.value}")
    else:
        print("No metadata properties found")

    # Check input tensor value info
    if model.graph.input:
        input_tensor = model.graph.input[0]
        print(f"\nInput tensor: {input_tensor.name}")
        print(f"  Type: {input_tensor.type.tensor_type.elem_type}")

print("\n" + "="*80)
print("Summary")
print("="*80)

print("""
Next steps:
1. If preprocessing differs: Fix current ONNX backend preprocessing order
2. If model outputs differ: Check ONNX model conversion (weights/architecture)
3. If everything matches: Issue is in post-processing (NMS, bbox generation, etc.)
""")
