"""
Debug patch expert internals to find why responses are zero/constant.

Check:
1. Are features (gradient magnitudes) being computed correctly?
2. Are neuron weights loaded properly?
3. Are alpha/bias values reasonable?
4. Is the response computation numerically stable?
"""

import cv2
import numpy as np
from pathlib import Path
from pyclnf.core.patch_expert import CCNFModel
import subprocess
import tempfile
import pandas as pd


def run_openface_get_landmarks(image_path: str, output_dir: str):
    """Get OpenFace ground truth landmarks."""
    openface_bin = Path("~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction").expanduser()
    cmd = [str(openface_bin), "-f", image_path, "-out_dir", output_dir, "-2Dfp"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    csv_file = Path(output_dir) / (Path(image_path).stem + ".csv")
    df = pd.read_csv(csv_file)

    landmarks = np.zeros((68, 2))
    for i in range(68):
        x_col = f'x_{i}'
        y_col = f'y_{i}'
        if x_col in df.columns:
            landmarks[i, 0] = df[x_col].iloc[0]
            landmarks[i, 1] = df[y_col].iloc[0]

    return landmarks


def main():
    print("=" * 80)
    print("Debugging Patch Expert Internals")
    print("=" * 80)

    # Load CCNF model
    print("\nLoading CCNF model...")
    ccnf = CCNFModel("pyclnf/models", scales=[0.25])

    # Get a patch expert
    scale = 0.25
    view_idx = 0
    landmark_idx = 30  # Nose tip

    patch_expert = ccnf.scale_models[scale]['views'][view_idx]['patches'][landmark_idx]
    print(f"\nPatch Expert {landmark_idx} ({patch_expert.width}x{patch_expert.height}):")
    print(f"  Num neurons: {patch_expert.num_neurons}")
    print(f"  Num betas: {len(patch_expert.betas)}")
    print(f"  Betas: {patch_expert.betas}")

    # Load test image and get ground truth landmark
    video_path = "Patient Data/Normal Cohort/IMG_0434.MOV"
    frame_num = 50

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_path = "/tmp/test_frame.jpg"
    cv2.imwrite(frame_path, frame)

    with tempfile.TemporaryDirectory() as tmpdir:
        landmarks = run_openface_get_landmarks(frame_path, tmpdir)

    gt_x, gt_y = landmarks[landmark_idx]
    print(f"\nGround truth position: ({gt_x:.1f}, {gt_y:.1f})")

    # Extract patch at ground truth
    half_w = patch_expert.width // 2
    half_h = patch_expert.height // 2
    x1 = int(gt_x) - half_w
    y1 = int(gt_y) - half_h
    x2 = x1 + patch_expert.width
    y2 = y1 + patch_expert.height

    patch = gray[y1:y2, x1:x2]
    print(f"\nPatch shape: {patch.shape}")
    print(f"Patch pixel range: [{patch.min()}, {patch.max()}]")

    # Extract features
    features = patch_expert._extract_features(patch)
    print(f"\nFeatures (gradient magnitude):")
    print(f"  Shape: {features.shape}")
    print(f"  Range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"  Mean: {features.mean():.6f}")
    print(f"  Std: {features.std():.6f}")

    # Check a few neurons
    print(f"\nFirst 3 neurons:")
    for i in range(min(3, len(patch_expert.neurons))):
        neuron = patch_expert.neurons[i]
        print(f"\nNeuron {i}:")
        print(f"  Weights shape: {neuron['weights'].shape}")
        print(f"  Weights range: [{neuron['weights'].min():.6f}, {neuron['weights'].max():.6f}]")
        print(f"  Weights mean: {neuron['weights'].mean():.6f}")
        print(f"  Bias: {neuron['bias']:.6f}")
        print(f"  Alpha: {neuron['alpha']:.6f}")

        # Compute response for this neuron
        response_value = np.sum(neuron['weights'] * features) + neuron['bias']
        print(f"  Raw response (before sigmoid): {response_value:.6f}")
        print(f"  Alpha * response: {neuron['alpha'] * response_value:.6f}")

        response = patch_expert._sigmoid(neuron['alpha'] * response_value)
        print(f"  After sigmoid: {response:.6f}")

    # Compute full response
    response = patch_expert.compute_response(patch)
    print(f"\nFull patch expert response: {response:.6f}")

    # Try a patch offset by a few pixels
    x1_offset = int(gt_x) - half_w + 5
    y1_offset = int(gt_y) - half_h
    x2_offset = x1_offset + patch_expert.width
    y2_offset = y1_offset + patch_expert.height

    patch_offset = gray[y1_offset:y2_offset, x1_offset:x2_offset]
    response_offset = patch_expert.compute_response(patch_offset)
    print(f"Response at +5px offset: {response_offset:.6f}")
    print(f"Response difference: {response - response_offset:.6f}")

    # Expected dynamic range check
    print(f"\n{'='*80}")
    if abs(response - response_offset) < 0.001:
        print("⚠️  WARNING: Response is essentially constant!")
        print("This explains why landmarks don't track properly.")

        # Check why
        if features.std() < 0.01:
            print("\nPossible cause: Features have very low variance")
            print("  → Image preprocessing may be wrong")
        elif abs(patch_expert.betas).sum() < 0.001:
            print("\nPossible cause: Betas are near zero")
            print("  → Model loading issue")
        else:
            print("\nPossible cause: Neuron weights may be incorrect")
            print("  → Check model export/loading")
    else:
        print("✓ Response has some dynamic range")


if __name__ == "__main__":
    main()
