#!/usr/bin/env python3
"""
Compare raw PNet outputs between C++ and Python to identify calibration issue.
Extract same input patches, run through both networks, compare logits.
"""

import cv2
import numpy as np
import subprocess
import re
import os
from cpp_mtcnn_detector import CPPMTCNNDetector

def extract_pnet_patch_python(img, scale_factor):
    """
    Extract PNet input patch at specific scale from Python implementation.

    Returns:
        preprocessed_patch: The actual tensor fed to PNet (1, 3, H, W)
        scale: The scale factor used
    """
    # Resize image to scale
    h, w = img.shape[:2]
    hs = int(np.ceil(h * scale_factor))
    ws = int(np.ceil(w * scale_factor))

    img_scaled = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)

    # Preprocess (normalize to [-1, 1] and convert to CHW)
    img_preprocessed = img_scaled.astype(np.float32)
    img_preprocessed = (img_preprocessed - 127.5) * 0.0078125
    img_preprocessed = np.transpose(img_preprocessed, (2, 0, 1))
    img_preprocessed = np.expand_dims(img_preprocessed, 0)

    return img_preprocessed, scale_factor

def run_pnet_python(detector, img, scale_factor):
    """
    Run Python PNet at specific scale and return raw outputs.

    Returns:
        prob_map: Raw probability map (face class)
        reg_map: Raw regression map
    """
    preprocessed, _ = extract_pnet_patch_python(img, scale_factor)

    # Run PNet
    output = detector.pnet.run(None, {detector.pnet.get_inputs()[0].name: preprocessed})[0]

    # Output shape: (1, 6, H_out, W_out)
    # Channels: [face_prob, non_face_prob, reg_x1, reg_y1, reg_x2, reg_y2]
    prob_map = output[0, 1, :, :]  # Face probability (logit, not softmaxed)
    reg_map = output[0, 2:, :, :]  # Regression offsets

    return prob_map, reg_map, preprocessed

def add_cpp_pnet_logging():
    """
    Modify C++ code to dump raw PNet outputs to file.
    """
    cpp_file = os.path.expanduser("~/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/FaceDetectorMTCNN.cpp")

    # Check if logging already exists
    with open(cpp_file, 'r') as f:
        content = f.read()
        if "PNET_RAW_OUTPUT_DEBUG" in content:
            print("C++ PNet logging already exists")
            return

    print("Adding C++ PNet raw output logging...")

    # Find the line after PNet forward pass (around line 618-630)
    # We want to dump the raw output tensor after network execution
    insertion_code = """
    // DEBUG: Dump raw PNet outputs
    #ifdef PNET_RAW_OUTPUT_DEBUG
    {
        std::ofstream debug_file("/tmp/cpp_pnet_raw_output.txt");
        debug_file << "PNet Raw Output Debug\\n";
        debug_file << "Scale factor: " << scale << "\\n";
        debug_file << "Output shape: " << output_tensor.size[1] << " x "
                   << output_tensor.size[2] << " x " << output_tensor.size[3] << "\\n";

        // Dump probability map (channel 1 = face class)
        debug_file << "\\nProbability map (face class, logits):\\n";
        int h = output_tensor.size[2];
        int w = output_tensor.size[3];
        float* prob_data = output_tensor.ptr<float>(0, 1);

        for (int y = 0; y < std::min(h, 10); ++y) {
            for (int x = 0; x < std::min(w, 10); ++x) {
                debug_file << prob_data[y * w + x] << " ";
            }
            debug_file << "\\n";
        }

        // Dump first few raw scores before softmax
        debug_file << "\\nFirst 20 face probability logits:\\n";
        for (int i = 0; i < std::min(20, h * w); ++i) {
            debug_file << prob_data[i] << " ";
        }
        debug_file << "\\n";

        debug_file.close();
    }
    #endif
"""

    # For now, just document what we'd add
    print("\nC++ logging code prepared (manual insertion needed):")
    print(insertion_code)
    print("\nTo enable, add -DPNET_RAW_OUTPUT_DEBUG to CMake flags and rebuild")

def compare_pnet_outputs(img_path, scale=0.5):
    """
    Compare PNet outputs between C++ and Python at same scale.
    """
    img = cv2.imread(img_path)

    # Run Python PNet
    print(f"\n{'='*80}")
    print(f"Comparing PNet outputs at scale {scale}")
    print(f"{'='*80}")

    detector = CPPMTCNNDetector()
    prob_map_py, reg_map_py, preprocessed_py = run_pnet_python(detector, img, scale)

    print(f"\nPython PNet outputs:")
    print(f"  Probability map shape: {prob_map_py.shape}")
    print(f"  Probability map (logits) stats:")
    print(f"    Min: {prob_map_py.min():.6f}")
    print(f"    Max: {prob_map_py.max():.6f}")
    print(f"    Mean: {prob_map_py.mean():.6f}")
    print(f"    Median: {np.median(prob_map_py):.6f}")

    # Show top 10 locations by raw logit value
    print(f"\n  Top 10 locations by face probability (raw logit):")
    flat_probs = prob_map_py.flatten()
    top_indices = np.argsort(flat_probs)[-10:][::-1]

    h, w = prob_map_py.shape
    for rank, idx in enumerate(top_indices):
        y = idx // w
        x = idx % w
        logit = flat_probs[idx]

        # Convert back to image coordinates
        # Each PNet output pixel corresponds to a 12x12 region with stride 2
        stride = 2
        cellsize = 12
        img_x = x * stride
        img_y = y * stride

        # Scale back to original image
        orig_x = img_x / scale
        orig_y = img_y / scale

        print(f"    #{rank+1}: cell({x:3d},{y:3d}) → img({orig_x:5.0f},{orig_y:5.0f}) logit={logit:8.5f}")

    # Show distribution of logits
    print(f"\n  Logit distribution:")
    print(f"    < -5: {np.sum(prob_map_py < -5)}")
    print(f"    -5 to 0: {np.sum((prob_map_py >= -5) & (prob_map_py < 0))}")
    print(f"    0 to 5: {np.sum((prob_map_py >= 0) & (prob_map_py < 5))}")
    print(f"    > 5: {np.sum(prob_map_py > 5)}")

    # Save preprocessed patch for C++ comparison
    patch_path = "/tmp/python_pnet_patch.bin"
    preprocessed_py.astype(np.float32).tofile(patch_path)
    print(f"\n  Saved preprocessed patch to: {patch_path}")
    print(f"  Shape: {preprocessed_py.shape}")

    # Sample pixels from preprocessed patch
    print(f"\n  Sample preprocessed pixels (CHW format):")
    print(f"    Channel 0 (B): [0,0]={preprocessed_py[0,0,0,0]:.6f}, [0,1]={preprocessed_py[0,0,0,1]:.6f}")
    print(f"    Channel 1 (G): [0,0]={preprocessed_py[0,1,0,0]:.6f}, [0,1]={preprocessed_py[0,1,0,1]:.6f}")
    print(f"    Channel 2 (R): [0,0]={preprocessed_py[0,2,0,0]:.6f}, [0,1]={preprocessed_py[0,2,0,1]:.6f}")

    return prob_map_py, preprocessed_py

if __name__ == "__main__":
    test_image = "calibration_frames/patient1_frame1.jpg"

    if not os.path.exists(test_image):
        print(f"Error: Test image not found: {test_image}")
        exit(1)

    # Compare at scale 0.5 (typical scale where face is detected)
    prob_map, preprocessed = compare_pnet_outputs(test_image, scale=0.5)

    print(f"\n{'='*80}")
    print(f"NEXT STEPS:")
    print(f"{'='*80}")
    print(f"1. Add C++ logging to dump raw PNet outputs")
    print(f"2. Run C++ FeatureExtraction with same image")
    print(f"3. Compare logit distributions to find calibration difference")
    print(f"4. Check if ONNX export changed activation functions or scaling")
