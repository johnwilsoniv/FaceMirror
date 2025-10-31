"""
Process OpenFace 2.2 mirrored videos with OpenFace 3.0 ONNX models

This script tests if the mirroring quality is affecting OF3.0 AU predictions.
We'll run OF3.0 AU extraction on videos that were mirrored by OF2.2 (higher quality).
"""

import config
config.apply_environment_settings()

import sys
from pathlib import Path
from openface_integration import OpenFace3Processor
import time

def main():
    """Process OF2.2-mirrored videos with OF3.0 AU extraction"""

    # Input videos (mirrored by OpenFace 2.2)
    left_video = Path("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_left_mirrored.mp4")
    right_video = Path("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror/output/IMG_0942_right_mirrored.mp4")

    # Output directory
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify input files exist
    if not left_video.exists():
        print(f"ERROR: Left video not found: {left_video}")
        sys.exit(1)
    if not right_video.exists():
        print(f"ERROR: Right video not found: {right_video}")
        sys.exit(1)

    print("=" * 80)
    print("Processing OF2.2-Mirrored Videos with OF3.0 ONNX Models")
    print("=" * 80)
    print(f"\nInput videos:")
    print(f"  Left:  {left_video.name}")
    print(f"  Right: {right_video.name}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nPurpose: Test if OF2.2's higher-quality mirroring improves OF3.0 AU predictions")
    print("\n" + "=" * 80 + "\n")

    # Initialize OpenFace 3.0 processor
    print("Initializing OpenFace 3.0 ONNX processor...")
    try:
        processor = OpenFace3Processor(
            device='cpu',  # Use ONNX + CoreML on Apple Silicon
            calculate_landmarks=False  # Don't need landmarks, just AUs
        )
        print("✓ OpenFace 3.0 processor initialized\n")
    except Exception as e:
        print(f"ERROR: Failed to initialize processor: {e}")
        sys.exit(1)

    # Process both videos
    videos = [
        (left_video, "IMG_0942_left_mirroredOP22_processedONNXv3.csv"),
        (right_video, "IMG_0942_right_mirroredOP22_processedONNXv3.csv")
    ]

    for video_path, csv_name in videos:
        output_csv = output_dir / csv_name
        side = "LEFT" if "left" in video_path.name else "RIGHT"

        print(f"\n{'=' * 80}")
        print(f"Processing {side} side: {video_path.name}")
        print(f"Output CSV: {csv_name}")
        print(f"{'=' * 80}\n")

        # Progress callback
        def progress_callback(current_frame, total_frames, processing_fps):
            if current_frame > 0 and current_frame % 50 == 0:
                percentage = (current_frame / total_frames * 100) if total_frames > 0 else 0
                if processing_fps > 0:
                    remaining_frames = total_frames - current_frame
                    eta_seconds = remaining_frames / processing_fps
                    eta_str = f" | ETA: {int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                    fps_str = f" | {processing_fps:.1f} fps"
                else:
                    eta_str = ""
                    fps_str = ""
                print(f"  Frame {current_frame:>6}/{total_frames:>6} ({percentage:>5.1f}%){fps_str}{eta_str}", end='\r')
            elif current_frame == total_frames and total_frames > 0:
                print(f"  Frame {current_frame:>6}/{total_frames:>6} (100.0%) - Complete          ")

        # Process video
        start_time = time.time()
        try:
            frame_count = processor.process_video(
                video_path,
                output_csv,
                progress_callback=progress_callback
            )
            elapsed = time.time() - start_time

            if frame_count > 0:
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"\n✓ Successfully processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")
                print(f"✓ CSV saved: {output_csv.name}\n")
            else:
                print(f"\n✗ No frames processed for {video_path.name}\n")
        except Exception as e:
            print(f"\n✗ Error processing {video_path.name}: {e}\n")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Processing Complete")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Compare original OF2.2 CSVs with these new OF3.0 CSVs")
    print("2. Check if correlations improved (indicating mirroring was the issue)")
    print("3. If correlations still poor, confirms OF3.0 AU models are the problem\n")

if __name__ == "__main__":
    main()
