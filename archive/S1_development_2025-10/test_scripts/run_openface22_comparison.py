#!/usr/bin/env python3
"""
OpenFace 2.2 Comparison Script

Processes mirrored videos through OpenFace 2.2 binary for comparison with OpenFace 3.0 results.
This is a temporary debugging tool to identify differences between versions.

Usage:
    python run_openface22_comparison.py

Input:
    - Scans: ~/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/
    - Finds: All *_mirrored.mp4 files (excluding debug videos)

Output:
    - CSVs: ~/Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test/
    - Format: Same filename as input (e.g., video_left_mirrored.csv)
"""

import subprocess
import shutil
import os
import sys
from pathlib import Path
import time

# OpenFace 2.2 binary location
OPENFACE2_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

# Input/Output directories
INPUT_DIR = Path.home() / "Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output"
OUTPUT_DIR = Path.home() / "Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test"

# Temporary directory for OpenFace 2.2 processing
# (OpenFace 2.2 creates a 'processed/' subdirectory where it's run)
TEMP_DIR = Path.home() / "Documents/SplitFace/S1O Processed Files/.openface22_temp"


def find_mirrored_videos():
    """
    Scan Face Mirror output directory for mirrored videos

    Returns:
        list: List of Path objects for mirrored videos
    """
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return []

    # Find all video files with 'mirrored' in filename (excluding 'debug')
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.MOV']
    mirrored_videos = []

    for ext in video_extensions:
        for video_path in INPUT_DIR.glob(f'*{ext}'):
            filename = video_path.name.lower()
            if 'mirrored' in filename and 'debug' not in filename:
                mirrored_videos.append(video_path)

    return sorted(mirrored_videos)


def process_video_with_openface22(video_path):
    """
    Process a single video through OpenFace 2.2 binary

    Args:
        video_path: Path to mirrored video file

    Returns:
        tuple: (success: bool, output_csv_path: Path or None, error_msg: str or None)
    """
    video_name = video_path.name
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    # Verify binary exists
    if not Path(OPENFACE2_BINARY).exists():
        error_msg = f"OpenFace 2.2 binary not found: {OPENFACE2_BINARY}"
        print(f"ERROR: {error_msg}")
        return False, None, error_msg

    # Create temp directory for processing
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    processed_subdir = TEMP_DIR / "processed"

    # Clean up any previous processed files
    if processed_subdir.exists():
        shutil.rmtree(processed_subdir)

    # Construct OpenFace 2.2 command
    # -aus: Extract Action Units
    # -verbose: Show detailed progress
    # -tracked: Use tracked model (better for videos)
    # -f: Input video file
    command = [
        OPENFACE2_BINARY,
        "-aus",
        "-verbose",
        "-tracked",
        "-f", str(video_path)
    ]

    print(f"Running OpenFace 2.2...")
    print(f"Command: {' '.join(command)}")

    try:
        # Run OpenFace 2.2 binary from temp directory
        # (so it creates processed/ subdirectory there)
        result = subprocess.run(
            command,
            cwd=str(TEMP_DIR),
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Print stdout for debugging
        if result.stdout:
            print("\nOpenFace 2.2 Output:")
            # Print last 20 lines only (avoid spam)
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:
                print(f"  {line}")

        # Find the generated CSV file
        # OpenFace 2.2 outputs to: processed/<video_stem>.csv
        expected_csv_name = video_path.stem + ".csv"
        source_csv = processed_subdir / expected_csv_name

        # Wait for CSV file to appear (up to 5 seconds)
        for _ in range(50):
            if source_csv.exists():
                break
            time.sleep(0.1)

        if not source_csv.exists():
            error_msg = f"OpenFace 2.2 did not create expected CSV: {source_csv}"
            print(f"ERROR: {error_msg}")
            print(f"Contents of {processed_subdir}:")
            if processed_subdir.exists():
                for f in processed_subdir.iterdir():
                    print(f"  - {f.name}")
            return False, None, error_msg

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Copy CSV to output directory
        dest_csv = OUTPUT_DIR / expected_csv_name
        shutil.copy2(source_csv, dest_csv)

        # Remove hidden flag (OpenFace 2.2 output files get marked as hidden)
        try:
            subprocess.run(['chflags', 'nohidden', str(dest_csv)], check=True, capture_output=True)
        except Exception as e:
            print(f"  Warning: Could not remove hidden flag: {e}")

        print(f"✓ Successfully processed: {video_name}")
        print(f"  Output CSV: {dest_csv}")

        # Clean up processed directory
        shutil.rmtree(processed_subdir)

        return True, dest_csv, None

    except subprocess.TimeoutExpired:
        error_msg = f"OpenFace 2.2 timed out after 10 minutes"
        print(f"ERROR: {error_msg}")
        return False, None, error_msg

    except subprocess.CalledProcessError as e:
        error_msg = f"OpenFace 2.2 failed with exit code {e.returncode}"
        print(f"ERROR: {error_msg}")
        if e.stderr:
            print(f"Error output:\n{e.stderr}")
        return False, None, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, None, error_msg


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("OpenFace 2.2 Comparison Script")
    print("="*60)
    print(f"\nInput Directory:  {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"OpenFace Binary:  {OPENFACE2_BINARY}")

    # Find mirrored videos
    print(f"\nScanning for mirrored videos...")
    videos = find_mirrored_videos()

    if not videos:
        print("No mirrored videos found!")
        print(f"Expected location: {INPUT_DIR}")
        return

    print(f"Found {len(videos)} mirrored video(s):")
    for video in videos:
        print(f"  - {video.name}")

    # Confirm processing
    print(f"\n{'='*60}")
    response = input(f"Process {len(videos)} video(s) through OpenFace 2.2? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return

    # Process each video
    print(f"\n{'='*60}")
    print("Starting batch processing...")
    print(f"{'='*60}")

    results = []
    success_count = 0
    fail_count = 0

    start_time = time.time()

    for i, video_path in enumerate(videos, 1):
        print(f"\n[Video {i}/{len(videos)}]")

        success, output_csv, error = process_video_with_openface22(video_path)

        results.append({
            'video': video_path.name,
            'success': success,
            'output_csv': output_csv,
            'error': error
        })

        if success:
            success_count += 1
        else:
            fail_count += 1

    end_time = time.time()
    total_seconds = end_time - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos:    {len(videos)}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {fail_count}")

    # Format processing time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"

    print(f"Processing time: {time_str}")

    if success_count > 0:
        print(f"\nOutput CSVs saved to:")
        print(f"  {OUTPUT_DIR}")

    # Print detailed results
    if fail_count > 0:
        print(f"\nFailed videos:")
        for result in results:
            if not result['success']:
                print(f"  ✗ {result['video']}")
                if result['error']:
                    print(f"    Error: {result['error']}")

    # Clean up temp directory
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
    except Exception as e:
        print(f"\nWarning: Could not clean up temp directory: {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
