#!/usr/bin/env python3
"""
Test script to run OpenFace 3.0 on a video and examine AU CSV output.
"""

import cv2
import tempfile
from pathlib import Path
from openface import demo

def extract_first_n_frames(video_path, n_frames=30):
    """Extract first N frames to a temporary video for quick testing"""
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

def main():
    # Test video path
    video_path = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0422.MOV")

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    print("Creating 30-frame test clip...")
    test_video = extract_first_n_frames(video_path, n_frames=30)
    print(f"Test video: {test_video}")

    print("\n" + "="*70)
    print("PROCESSING VIDEO WITH OPENFACE 3.0")
    print("="*70)

    # Process the video
    output_dir = Path(__file__).parent / "openface3_output"
    output_dir.mkdir(exist_ok=True)

    try:
        csv_path = demo.process_video(test_video, output_dir=str(output_dir), device='cpu')

        print("\n" + "="*70)
        print("EXAMINING CSV OUTPUT")
        print("="*70)
        print(f"\nCSV file: {csv_path}")

        # Read and analyze the CSV
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
            all_columns = header.split(',')

            print(f"\nTotal columns: {len(all_columns)}")
            print("\nAll columns:")
            for i, col in enumerate(all_columns, 1):
                print(f"  {i}. {col}")

            # Find AU columns
            au_columns = [col for col in all_columns if 'AU' in col.upper()]
            print(f"\n\nAU-related columns ({len(au_columns)}):")
            for col in au_columns:
                print(f"  - {col}")

            # Show first data row
            first_row = f.readline().strip().split(',')

            print("\n" + "="*70)
            print("SAMPLE DATA (First frame)")
            print("="*70)

            # Show all column values for first frame
            print("\nAll values:")
            for col, val in zip(all_columns, first_row):
                if 'AU' in col.upper():
                    print(f"  {col}: {val}")

        print(f"\n\nFull output saved to: {csv_path}")

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
