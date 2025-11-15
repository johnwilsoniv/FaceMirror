#!/usr/bin/env python3
"""
Direct test of OpenFace 3.0 using the cloned repository.
This bypasses the pip package's hardcoded paths.
"""

import sys
import os
import cv2
import tempfile
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add OpenFace-3.0 to path
sys.path.insert(0, str(Path(__file__).parent / 'OpenFace-3.0'))

from model.MLT import MLT

# Transform for AU model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_first_n_frames(video_path, n_frames=30):
    """Extract first N frames to a temporary video"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return temp_video.name

def simple_face_crop(frame):
    """Simple center crop as fallback if face detection fails"""
    h, w = frame.shape[:2]
    # Assume face is centered, take middle 60% of frame
    crop_size = min(h, w)
    y1 = (h - crop_size) // 2
    x1 = (w - crop_size) // 2
    return frame[y1:y1+crop_size, x1:x1+crop_size]

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load AU model
    print("\nLoading AU detection model...")
    weights_path = Path(__file__).parent / "weights" / "MTL_backbone.pth"

    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return

    model = MLT()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✓ AU model loaded")

    # Test video
    video_path = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0422.MOV")

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"\nProcessing video: {video_path.name}")

    # Get total frame count
    cap_check = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_check.release()

    # Process all frames (or cap at 500 for reasonable processing time)
    frames_to_process = min(total_video_frames, 500)
    print(f"Video has {total_video_frames} total frames")
    print(f"Extracting {frames_to_process} frames...")
    test_video = extract_first_n_frames(video_path, n_frames=frames_to_process)

    # Process video
    cap = cv2.VideoCapture(test_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames...")

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simple center crop for face region
        face_region = simple_face_crop(frame)

        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Get AU predictions
        with torch.no_grad():
            emotion_output, gaze_output, au_output = model(image_tensor)

        # au_output shape should be (1, 8) for 8 AUs
        au_values = au_output.cpu().numpy()[0]

        results.append({
            'frame': frame_idx,
            'au_values': au_values,
            'emotion': emotion_output.cpu().numpy()[0],
            'gaze': gaze_output.cpu().numpy()[0]
        })

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    # Save to CSV
    output_csv = Path(__file__).parent / "openface3_output.csv"

    print(f"\n{'='*70}")
    print("CREATING CSV OUTPUT")
    print(f"{'='*70}")

    # Check AU output shape
    if len(results) > 0:
        num_aus = len(results[0]['au_values'])
        print(f"\nAU output dimension: {num_aus}")
        print(f"Note: OpenFace 3.0 model has au_numbers={num_aus} parameter")

        # Create CSV with header
        with open(output_csv, 'w') as f:
            # Write header
            header_parts = ['frame']
            for i in range(num_aus):
                header_parts.append(f'AU{i+1:02d}_r')  # AU output indices
            header_parts.extend(['emotion_0', 'emotion_1', 'emotion_2', 'emotion_3',
                               'emotion_4', 'emotion_5', 'emotion_6', 'emotion_7',
                               'gaze_x', 'gaze_y'])
            f.write(','.join(header_parts) + '\n')

            # Write data
            for result in results:
                row = [str(result['frame'])]
                row.extend([f"{val:.6f}" for val in result['au_values']])
                row.extend([f"{val:.6f}" for val in result['emotion']])
                row.extend([f"{val:.6f}" for val in result['gaze']])
                f.write(','.join(row) + '\n')

        print(f"\n✓ CSV saved to: {output_csv}")

        # Analyze the output
        print(f"\n{'='*70}")
        print("ANALYZING AU OUTPUT")
        print(f"{'='*70}")

        # Show header
        with open(output_csv, 'r') as f:
            header = f.readline().strip()
            print(f"\nCSV Header:")
            print(header)

            au_columns = [col for col in header.split(',') if col.startswith('AU')]
            print(f"\n\nAU columns found: {len(au_columns)}")
            for col in au_columns:
                print(f"  - {col}")

            # Read first frame data
            first_row = f.readline().strip().split(',')

            print(f"\n\nSample data (Frame 0):")
            header_cols = header.split(',')
            for i, (col, val) in enumerate(zip(header_cols, first_row)):
                if col.startswith('AU'):
                    print(f"  {col}: {val}")

        print(f"\n{'='*70}")
        print(f"Total frames processed: {len(results)}")
        print(f"Output saved to: {output_csv}")
        print(f"{'='*70}")

if __name__ == '__main__':
    main()
