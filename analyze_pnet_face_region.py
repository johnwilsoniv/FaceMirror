#!/usr/bin/env python3
"""
Analyze PNet logit distribution specifically in the face region vs artifacts.
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

def analyze_pnet_regions(img_path, scale=0.5):
    """
    Compare PNet logits in face region vs bottom artifact region.
    """
    img = cv2.imread(img_path)
    detector = CPPMTCNNDetector()

    # Resize to scale
    h, w = img.shape[:2]
    hs = int(np.ceil(h * scale))
    ws = int(np.ceil(w * scale))
    img_scaled = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)

    # Preprocess
    img_preprocessed = img_scaled.astype(np.float32)
    img_preprocessed = (img_preprocessed - 127.5) * 0.0078125
    img_preprocessed = np.transpose(img_preprocessed, (2, 0, 1))
    img_preprocessed = np.expand_dims(img_preprocessed, 0)

    # Run PNet
    output = detector.pnet.run(None, {detector.pnet.get_inputs()[0].name: img_preprocessed})[0]

    # Output shape: (1, 6, H_out, W_out)
    # Let me check what channels 0 and 1 are
    print(f"PNet output shape: {output.shape}")

    channel_0 = output[0, 0, :, :]  # Non-face class logit
    channel_1 = output[0, 1, :, :]  # Face class logit

    print(f"\nChannel 0 (presumably non-face) stats:")
    print(f"  Min: {channel_0.min():.6f}, Max: {channel_0.max():.6f}, Mean: {channel_0.mean():.6f}")

    print(f"\nChannel 1 (presumably face) stats:")
    print(f"  Min: {channel_1.min():.6f}, Max: {channel_1.max():.6f}, Mean: {channel_1.mean():.6f}")

    # Compute softmax probabilities
    prob_face = np.exp(channel_1) / (np.exp(channel_0) + np.exp(channel_1))

    print(f"\nProbability (after softmax) stats:")
    print(f"  Min: {prob_face.min():.6f}, Max: {prob_face.max():.6f}, Mean: {prob_face.mean():.6f}")

    # Define regions in ORIGINAL image coordinates
    # Face region: y=300-800 (from C++ debug, face was around y=623)
    # Artifact region: y=1200-1920 (bottom of image)

    # Convert to scaled image coordinates
    face_y_min_scaled = int(300 * scale)
    face_y_max_scaled = int(800 * scale)
    artifact_y_min_scaled = int(1200 * scale)
    artifact_y_max_scaled = int(1920 * scale)

    # Convert to PNet cell coordinates (stride 2, cellsize 12)
    stride = 2
    face_cell_y_min = face_y_min_scaled // stride
    face_cell_y_max = face_y_max_scaled // stride
    artifact_cell_y_min = artifact_y_min_scaled // stride
    artifact_cell_y_max = artifact_y_max_scaled // stride

    # Ensure within bounds
    face_cell_y_max = min(face_cell_y_max, prob_face.shape[0])
    artifact_cell_y_max = min(artifact_cell_y_max, prob_face.shape[0])

    print(f"\n{'='*80}")
    print(f"REGION ANALYSIS")
    print(f"{'='*80}")

    # Face region
    face_region_probs = prob_face[face_cell_y_min:face_cell_y_max, :]
    face_region_logits = channel_1[face_cell_y_min:face_cell_y_max, :]

    print(f"\nFACE REGION (orig_y={300}-{800}, cell_y={face_cell_y_min}-{face_cell_y_max}):")
    print(f"  Probability stats:")
    print(f"    Min: {face_region_probs.min():.6f}, Max: {face_region_probs.max():.6f}")
    print(f"    Mean: {face_region_probs.mean():.6f}, Median: {np.median(face_region_probs):.6f}")
    print(f"  Logit stats:")
    print(f"    Min: {face_region_logits.min():.6f}, Max: {face_region_logits.max():.6f}")
    print(f"    Mean: {face_region_logits.mean():.6f}, Median: {np.median(face_region_logits):.6f}")

    # Show top 5 in face region
    face_flat = face_region_probs.flatten()
    face_logits_flat = face_region_logits.flatten()
    top_face_indices = np.argsort(face_flat)[-5:][::-1]

    print(f"\n  Top 5 locations in face region:")
    h_face, w_face = face_region_probs.shape
    for rank, idx in enumerate(top_face_indices):
        local_y = idx // w_face
        local_x = idx % w_face
        cell_y = face_cell_y_min + local_y
        cell_x = local_x

        # Convert to original image coordinates
        orig_x = (cell_x * stride) / scale
        orig_y = (cell_y * stride) / scale

        prob = face_flat[idx]
        logit = face_logits_flat[idx]

        print(f"    #{rank+1}: cell({cell_x:3d},{cell_y:3d}) → orig({orig_x:5.0f},{orig_y:5.0f}) "
              f"prob={prob:.6f} logit={logit:.6f}")

    # Artifact region
    artifact_region_probs = prob_face[artifact_cell_y_min:artifact_cell_y_max, :]
    artifact_region_logits = channel_1[artifact_cell_y_min:artifact_cell_y_max, :]

    print(f"\nARTIFACT REGION (orig_y={1200}-{1920}, cell_y={artifact_cell_y_min}-{artifact_cell_y_max}):")
    print(f"  Probability stats:")
    print(f"    Min: {artifact_region_probs.min():.6f}, Max: {artifact_region_probs.max():.6f}")
    print(f"    Mean: {artifact_region_probs.mean():.6f}, Median: {np.median(artifact_region_probs):.6f}")
    print(f"  Logit stats:")
    print(f"    Min: {artifact_region_logits.min():.6f}, Max: {artifact_region_logits.max():.6f}")
    print(f"    Mean: {artifact_region_logits.mean():.6f}, Median: {np.median(artifact_region_logits):.6f}")

    # Show top 5 in artifact region
    artifact_flat = artifact_region_probs.flatten()
    artifact_logits_flat = artifact_region_logits.flatten()
    top_artifact_indices = np.argsort(artifact_flat)[-5:][::-1]

    print(f"\n  Top 5 locations in artifact region:")
    h_artifact, w_artifact = artifact_region_probs.shape
    for rank, idx in enumerate(top_artifact_indices):
        local_y = idx // w_artifact
        local_x = idx % w_artifact
        cell_y = artifact_cell_y_min + local_y
        cell_x = local_x

        # Convert to original image coordinates
        orig_x = (cell_x * stride) / scale
        orig_y = (cell_y * stride) / scale

        prob = artifact_flat[idx]
        logit = artifact_logits_flat[idx]

        print(f"    #{rank+1}: cell({cell_x:3d},{cell_y:3d}) → orig({orig_x:5.0f},{orig_y:5.0f}) "
              f"prob={prob:.6f} logit={logit:.6f}")

    # Key finding
    print(f"\n{'='*80}")
    print(f"KEY FINDING:")
    print(f"{'='*80}")
    print(f"  Face region max probability: {face_region_probs.max():.6f} (logit={face_region_logits.max():.6f})")
    print(f"  Artifact region max probability: {artifact_region_probs.max():.6f} (logit={artifact_region_logits.max():.6f})")

    if artifact_region_probs.max() > face_region_probs.max():
        diff = artifact_region_probs.max() - face_region_probs.max()
        print(f"\n  ⚠️  ARTIFACT scores {diff:.4f} higher than FACE!")
        print(f"      This explains why NMS suppresses face boxes.")
    else:
        print(f"\n  ✓ Face scores higher than artifacts (expected behavior)")

if __name__ == "__main__":
    test_image = "calibration_frames/patient1_frame1.jpg"
    analyze_pnet_regions(test_image, scale=0.5)
