"""
Validate Training Data Generation

This script:
1. Processes a few frames from a patient video
2. Saves sample images showing aligned faces with landmarks
3. Displays AU values
4. Creates a small test HDF5 to verify the pipeline

Run this BEFORE generating full training data to verify quality.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=1):
    """Draw landmarks on image."""
    img = image.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return img


def validate_training_data():
    """Validate training data generation on a few frames."""

    # Import here after path setup
    from pyfaceau.data.training_data_generator import TrainingDataGenerator, GeneratorConfig
    from pyfaceau.data.hdf5_dataset import TrainingDataWriter, TrainingDataset, AU_NAMES

    # Configuration
    video_path = Path("Patient Data/Normal Cohort/IMG_0422.MOV")
    output_dir = Path("training_validation")
    output_dir.mkdir(exist_ok=True)

    test_h5_path = output_dir / "test_data.h5"

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return False

    print("=" * 70)
    print("TRAINING DATA VALIDATION")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_dir}")

    # Initialize generator
    print("\n1. Initializing TrainingDataGenerator...")
    config = GeneratorConfig(verbose=True)
    generator = TrainingDataGenerator(config)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Video: {total_frames} frames @ {fps:.1f} FPS")

    # Process sample frames
    test_frame_indices = [0, 50, 100, 150, 200]
    samples = []

    print(f"\n2. Processing {len(test_frame_indices)} sample frames...")

    for target_idx in test_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"   Frame {target_idx}: FAILED to read")
            continue

        result = generator.process_frame(frame)

        if result is None:
            print(f"   Frame {target_idx}: FAILED to process")
            continue

        samples.append({
            'frame_idx': target_idx,
            'original_frame': frame,
            'result': result
        })

        # Print AU values for first few
        if len(samples) <= 3:
            print(f"\n   Frame {target_idx} - AU Values:")
            aus = result['au_intensities']
            for i, name in enumerate(AU_NAMES):
                if aus[i] > 0.1:  # Only show non-zero AUs
                    print(f"      {name}: {aus[i]:.2f}")

    cap.release()

    if len(samples) == 0:
        print("\nERROR: No frames processed successfully!")
        return False

    print(f"\n   Successfully processed {len(samples)}/{len(test_frame_indices)} frames")

    # Create visualization
    print("\n3. Creating visualization...")

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    if len(samples) == 1:
        axes = axes.reshape(2, 1)

    for i, sample in enumerate(samples):
        result = sample['result']
        frame_idx = sample['frame_idx']

        # Top row: Original frame with landmarks
        original = sample['original_frame'].copy()
        landmarks = result['landmarks']
        for x, y in landmarks:
            cv2.circle(original, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Draw bounding box
        bbox = result['bbox']
        cv2.rectangle(original,
                     (int(bbox[0]), int(bbox[1])),
                     (int(bbox[2]), int(bbox[3])),
                     (255, 0, 0), 2)

        axes[0, i].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Frame {frame_idx}\nOriginal + Landmarks")
        axes[0, i].axis('off')

        # Bottom row: Aligned face (already RGB)
        aligned = result['image']  # Already RGB
        axes[1, i].imshow(aligned)
        axes[1, i].set_title(f"Aligned 112x112\n(RGB)")
        axes[1, i].axis('off')

    plt.tight_layout()
    viz_path = output_dir / "sample_frames.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {viz_path}")

    # Save individual aligned faces for inspection
    print("\n4. Saving individual aligned faces...")
    for sample in samples:
        result = sample['result']
        frame_idx = sample['frame_idx']

        # Save RGB version (correct colors)
        aligned_rgb = result['image']
        rgb_path = output_dir / f"aligned_frame_{frame_idx}_rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))

    print(f"   Saved {len(samples)} aligned face images")

    # Create test HDF5 file
    print("\n5. Creating test HDF5 file...")

    with TrainingDataWriter(test_h5_path, expected_samples=len(samples)) as writer:
        for sample in samples:
            result = sample['result']
            writer.add_sample(
                image=result['image'],
                hog_features=result['hog_features'],
                landmarks=result['landmarks'],
                global_params=result['global_params'],
                local_params=result['local_params'],
                au_intensities=result['au_intensities'],
                bbox=result['bbox'],
                video_name=video_path.name,
                frame_index=sample['frame_idx'],
                quality_score=1.0
            )

    print(f"   Saved: {test_h5_path}")

    # Verify HDF5 file
    print("\n6. Verifying HDF5 file...")
    dataset = TrainingDataset(test_h5_path)
    print(f"   Samples: {len(dataset)}")
    print(f"   AU names: {dataset.au_names[:5]}...")

    # Check a sample
    sample = dataset[0]
    print(f"\n   Sample 0 shapes:")
    print(f"      image: {sample['image'].shape}, dtype={sample['image'].dtype}")
    print(f"      landmarks: {sample['landmarks'].shape}")
    print(f"      global_params: {sample['global_params'].shape}")
    print(f"      local_params: {sample['local_params'].shape}")
    print(f"      au_intensities: {sample['au_intensities'].shape}")
    print(f"      hog_features: {sample['hog_features'].shape}")

    # Check color format
    img = sample['image']
    print(f"\n   Image color check (should be RGB - R > B for skin):")
    print(f"      R mean: {img[:,:,0].mean():.1f}")
    print(f"      G mean: {img[:,:,1].mean():.1f}")
    print(f"      B mean: {img[:,:,2].mean():.1f}")

    dataset.close()

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {output_dir.absolute()}")
    print(f"  - sample_frames.png: Visual comparison")
    print(f"  - aligned_frame_*.png: Individual aligned faces")
    print(f"  - test_data.h5: Test HDF5 file")
    print("\nPlease visually inspect:")
    print("  1. Landmarks are correctly placed on faces")
    print("  2. Aligned faces look properly centered/aligned")
    print("  3. Colors look correct (skin should be reddish, not blue)")

    print("\n" + "=" * 70)
    print("NEXT STEP: Generate Full Training Data")
    print("=" * 70)
    print("""
To generate training data from all patient videos:

    python generate_training_data.py

This will create: training_data.h5

Then train the models:

    # Train landmark/pose model
    python -m pyfaceau.nn.train_landmark_pose \\
        --data training_data.h5 \\
        --output models/landmark_pose \\
        --epochs 100 \\
        --batch-size 32

    # Train AU prediction model
    python -m pyfaceau.nn.train_au_prediction \\
        --data training_data.h5 \\
        --output models/au_prediction \\
        --epochs 100 \\
        --batch-size 32
""")

    return True


if __name__ == "__main__":
    success = validate_training_data()
    sys.exit(0 if success else 1)
