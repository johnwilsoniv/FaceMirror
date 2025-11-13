#!/usr/bin/env python3
"""
Direct comparison of PyCLNF vs OpenFace C++ patch expert responses.

Test if patch experts produce identical responses for the same image patch.
This will identify if the issue is:
1. Patch expert weights loaded incorrectly
2. Image preprocessing mismatch
3. Neuron response computation bug
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'pyclnf')
from pyclnf.core.patch_expert import CCNFModel

# Test configuration
VIDEO_PATH = 'Patient Data/Normal Cohort/IMG_0433.MOV'
FRAME_NUM = 50

# Known landmark positions from OpenFace C++ (frame 50, converged)
# These are from the OpenFace CSV output
OPENFACE_LANDMARKS = {
    30: (450, 731),  # Nose tip (typically well-detected)
    36: (385, 683),  # Left eye outer corner
    45: (509, 683),  # Right eye outer corner
}


def extract_frame(video_path, frame_num):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

    return frame


def extract_patch_around_landmark(image, lm_x, lm_y, patch_width, patch_height):
    """
    Extract patch centered at landmark position.

    Matches OpenFace's patch extraction:
    - Center patch at (lm_x, lm_y)
    - Patch spans [lm_x - w/2, lm_x + w/2) × [lm_y - h/2, lm_y + h/2)
    """
    half_w = patch_width // 2
    half_h = patch_height // 2

    start_x = int(lm_x) - half_w
    start_y = int(lm_y) - half_h
    end_x = start_x + patch_width
    end_y = start_y + patch_height

    # Check bounds
    if start_x < 0 or start_y < 0 or end_x > image.shape[1] or end_y > image.shape[0]:
        print(f"  WARNING: Patch out of bounds")
        return None

    patch = image[start_y:end_y, start_x:end_x]
    return patch


def main():
    print("=" * 80)
    print("COMPARING PYCLNF VS OPENFACE C++ PATCH EXPERT RESPONSES")
    print("=" * 80)
    print()
    print("This test extracts the SAME image patch and compares responses.")
    print("If responses differ significantly, the issue is in patch expert weights/preprocessing.")
    print()

    # Load frame
    frame = extract_frame(VIDEO_PATH, FRAME_NUM)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    print(f"Loaded frame {FRAME_NUM} from {VIDEO_PATH}")
    print(f"Frame shape: {gray.shape}")
    print()

    # Load PyCLNF model
    print("Loading PyCLNF CCNF model...")
    ccnf = CCNFModel('pyclnf/models', scales=[0.25])
    print()

    # Test each landmark
    scale = 0.25
    view_idx = 0  # Frontal view

    print(f"Testing patch experts at scale={scale}, view={view_idx}")
    print()

    results = []

    for landmark_idx, (lm_x, lm_y) in OPENFACE_LANDMARKS.items():
        print(f"{'─' * 80}")
        print(f"LANDMARK {landmark_idx} at ({lm_x}, {lm_y})")
        print(f"{'─' * 80}")

        # Get patch expert
        patch_expert = ccnf.get_patch_expert(scale, view_idx, landmark_idx)
        if patch_expert is None:
            print(f"  ✗ Patch expert not found")
            continue

        info = patch_expert.get_info()
        print(f"  Patch size: {info['width']}×{info['height']}")
        print(f"  Neurons: {info['num_neurons']}")
        print()

        # Extract patch centered at landmark
        patch = extract_patch_around_landmark(gray, lm_x, lm_y, info['width'], info['height'])

        if patch is None:
            print(f"  ✗ Could not extract patch (out of bounds)")
            continue

        print(f"  Extracted patch shape: {patch.shape}")
        print(f"  Patch intensity: min={patch.min():.1f}, max={patch.max():.1f}, mean={patch.mean():.1f}")
        print()

        # Convert to uint8 for patch expert (expects 0-255)
        patch_uint8 = np.clip(patch, 0, 255).astype(np.uint8)

        # Compute PyCLNF response
        response = patch_expert.compute_response(patch_uint8)
        print(f"  PyCLNF response: {response:.6f}")
        print()

        # Analyze neuron contributions
        print(f"  Neuron breakdown:")
        neuron_responses = []
        for i, neuron in enumerate(patch_expert.neurons):
            if abs(neuron['alpha']) < 1e-4:
                continue

            # Manually compute this neuron's response
            features = patch_uint8.astype(np.float32) / 255.0
            weights = neuron['weights']

            # Resize if needed
            if features.shape != weights.shape:
                features = cv2.resize(features, (weights.shape[1], weights.shape[0]))

            # Compute correlation (TM_CCOEFF_NORMED)
            weight_mean = np.mean(weights)
            feature_mean = np.mean(features)
            weights_centered = weights - weight_mean
            features_centered = features - feature_mean
            weight_norm = np.linalg.norm(weights_centered)
            feature_norm = np.linalg.norm(features_centered)

            if weight_norm > 1e-10 and feature_norm > 1e-10:
                correlation = np.sum(weights_centered * features_centered) / (weight_norm * feature_norm)
            else:
                correlation = 0.0

            # Apply sigmoid
            sigmoid_input = correlation * neuron['norm_weights'] + neuron['bias']
            neuron_resp = (2.0 * neuron['alpha']) / (1.0 + np.exp(-sigmoid_input))

            neuron_responses.append((i, neuron_resp, neuron['alpha']))

            if i < 3:  # Show first 3 neurons
                print(f"    Neuron {i:2d}: response={neuron_resp:8.4f}, alpha={neuron['alpha']:.4f}, corr={correlation:.4f}")

        if len(neuron_responses) > 3:
            print(f"    ... ({len(neuron_responses) - 3} more neurons)")

        print()

        # Check if response is reasonable
        if response < 0.001:
            print(f"  ⚠️  Very low response ({response:.6f}) - patch expert may not match this region")
        elif response > 10.0:
            print(f"  ⚠️  Very high response ({response:.6f}) - possible overflow or incorrect scaling")
        else:
            print(f"  ✓ Response in reasonable range")

        results.append({
            'landmark_idx': landmark_idx,
            'position': (lm_x, lm_y),
            'response': response,
            'num_neurons': len(neuron_responses)
        })

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if not results:
        print("✗ No patch experts could be tested")
        return

    responses = [r['response'] for r in results]
    print(f"Tested {len(results)} landmarks")
    print(f"Response range: [{min(responses):.4f}, {max(responses):.4f}]")
    print(f"Mean response: {np.mean(responses):.4f}")
    print()

    print("NEXT STEPS:")
    print("1. Compare these responses with OpenFace C++ on the same patches")
    print("2. If responses differ → patch expert weights are incorrect")
    print("3. If responses match but response MAPS differ → Sigma transformation issue")
    print("4. If both match → issue is in mean-shift optimization")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
