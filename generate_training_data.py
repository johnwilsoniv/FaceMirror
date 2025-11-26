"""
Generate Training Data from Patient Videos

Processes all patient videos and creates an HDF5 dataset for training
neural network models.

Usage:
    python generate_training_data.py

Options:
    --videos-dir: Directory containing videos (default: Patient Data)
    --output: Output HDF5 file (default: training_data.h5)
    --max-frames: Max frames per video (default: all)
    --skip-frames: Skip every N frames (default: 0, no skipping)
    --min-quality: Minimum quality threshold (default: 0.5)
    --recursive: Search for videos recursively in subdirectories (default: True)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Setup paths - add all local packages
_root = Path(__file__).parent
for _pkg in ['pyclnf', 'pymtcnn', 'pyfaceau', 'pyfhog', '.']:
    _path = _root / _pkg if _pkg != '.' else _root
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main():
    parser = argparse.ArgumentParser(description="Generate training data from videos")
    parser.add_argument("--videos-dir", type=str,
                       default="Patient Data",
                       help="Directory containing video files (searches recursively)")
    parser.add_argument("--output", type=str,
                       default="training_data.h5",
                       help="Output HDF5 file path")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames per video (default: all)")
    parser.add_argument("--skip-frames", type=int, default=0,
                       help="Skip every N frames (0 = no skipping)")
    parser.add_argument("--min-quality", type=float, default=0.5,
                       help="Minimum quality threshold")
    parser.add_argument("--no-recursive", action="store_true",
                       help="Don't search subdirectories for videos")

    args = parser.parse_args()

    # Import after path setup
    from pyfaceau.data.training_data_generator import TrainingDataGenerator, GeneratorConfig

    # Find videos
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"ERROR: Videos directory not found: {videos_dir}")
        sys.exit(1)

    # Find all video files (both .MOV and .mov)
    if args.no_recursive:
        # Non-recursive: only in specified directory
        video_paths = list(videos_dir.glob("*.MOV")) + list(videos_dir.glob("*.mov"))
    else:
        # Recursive: search all subdirectories
        video_paths = list(videos_dir.glob("**/*.MOV")) + list(videos_dir.glob("**/*.mov"))

    # Remove duplicates (in case of case-insensitive filesystem) and sort
    video_paths = sorted(set(video_paths))

    if not video_paths:
        print(f"ERROR: No .MOV/.mov videos found in {videos_dir}")
        sys.exit(1)

    print("=" * 70)
    print("TRAINING DATA GENERATION")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Videos directory: {videos_dir}")
    print(f"Recursive search: {not args.no_recursive}")
    print(f"Videos found: {len(video_paths)}")

    # Group videos by parent directory for cleaner display
    from collections import defaultdict
    videos_by_dir = defaultdict(list)
    for vp in video_paths:
        rel_parent = vp.parent.relative_to(videos_dir) if vp.parent != videos_dir else Path(".")
        videos_by_dir[str(rel_parent)].append(vp.name)

    for dir_name, video_names in sorted(videos_by_dir.items()):
        print(f"\n  {dir_name}/ ({len(video_names)} videos):")
        for vn in sorted(video_names)[:5]:  # Show first 5
            print(f"    - {vn}")
        if len(video_names) > 5:
            print(f"    ... and {len(video_names) - 5} more")
    print(f"\nOutput: {args.output}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Min quality: {args.min_quality}")
    if args.max_frames:
        print(f"Max frames per video: {args.max_frames}")

    # Configure generator
    config = GeneratorConfig(
        skip_frames=args.skip_frames,
        min_quality=args.min_quality,
        verbose=True,
    )

    generator = TrainingDataGenerator(config)

    # Process all videos
    print("\n" + "=" * 70)
    print("PROCESSING VIDEOS")
    print("=" * 70)

    all_stats = generator.process_multiple_videos(
        video_paths=video_paths,
        output_path=args.output,
        max_frames_per_video=args.max_frames,
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {args.output}")

    # File size
    output_path = Path(args.output)
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")

    print("\n" + "-" * 70)
    print("PER-VIDEO STATISTICS")
    print("-" * 70)
    for video_name, stats in all_stats.items():
        print(f"\n{video_name}:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Saved: {stats['processed_frames']}")
        print(f"  Filtered: {stats['filtered_frames']}")
        print(f"  Failed: {stats['failed_frames']}")

    # Total stats
    total_processed = sum(s['processed_frames'] for s in all_stats.values())
    total_filtered = sum(s['filtered_frames'] for s in all_stats.values())
    total_failed = sum(s['failed_frames'] for s in all_stats.values())
    total_read = sum(s['total_frames'] for s in all_stats.values())

    print("\n" + "-" * 70)
    print("TOTALS")
    print("-" * 70)
    print(f"Videos processed: {len(all_stats)}")
    print(f"Frames read: {total_read}")
    print(f"Frames saved: {total_processed}")
    print(f"Frames filtered (quality): {total_filtered}")
    print(f"Frames failed (detection): {total_failed}")
    if total_read > 0:
        print(f"Success rate: {100 * total_processed / total_read:.1f}%")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"""
1. Verify the data:
   python -c "
from pyfaceau.data.hdf5_dataset import TrainingDataset
ds = TrainingDataset('{args.output}')
print(f'Total samples: {{len(ds)}}')
print(f'Sample image shape: {{ds[0][\"image\"].shape}}')
ds.close()
"

2. Train landmark/pose model:
   python -m pyfaceau.nn.train_landmark_pose \\
       --data {args.output} \\
       --output models/landmark_pose \\
       --epochs 100 \\
       --batch-size 32

3. Train AU prediction model:
   python -m pyfaceau.nn.train_au_prediction \\
       --data {args.output} \\
       --output models/au_prediction \\
       --epochs 100 \\
       --batch-size 32
""")


if __name__ == "__main__":
    main()
