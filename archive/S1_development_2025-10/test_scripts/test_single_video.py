"""
Simple test: Process one OF2.2-mirrored video with OF3.0
"""

import config
config.apply_environment_settings()

import sys
from pathlib import Path
from openface_integration import OpenFace3Processor

def main():
    # Test with left video only
    video_path = Path("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_left_mirrored.mp4")
    output_csv = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22/IMG_0942_left_mirroredOP22_processedONNXv3.csv")

    print(f"Input: {video_path}")
    print(f"Output: {output_csv}")
    print(f"Video exists: {video_path.exists()}")

    if not video_path.exists():
        print("ERROR: Video file not found!")
        return

    # Create output directory
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("\nInitializing processor...")
    processor = OpenFace3Processor(device='cpu', calculate_landmarks=False, debug_mode=False)

    print("\nProcessing video...")

    def progress(current, total, fps):
        if current % 100 == 0 or current == total:
            print(f"  Frame {current}/{total} ({100*current/total:.1f}%) @ {fps:.1f} fps")

    try:
        frame_count = processor.process_video(video_path, output_csv, progress_callback=progress)
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ CSV saved: {output_csv}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
