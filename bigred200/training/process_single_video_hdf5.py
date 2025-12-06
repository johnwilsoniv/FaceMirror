#!/usr/bin/env python3
"""
Process Single Video to HDF5

Worker script for SLURM array jobs. Loads video path from manifest using
SLURM_ARRAY_TASK_ID and generates HDF5 training data.

Usage:
    python process_single_video_hdf5.py <video_path> --output <output.h5>

Or with manifest:
    python process_single_video_hdf5.py --manifest manifest.txt --task-id 0 --output-dir output/
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add parent paths for imports
_root = Path(__file__).parent.parent.parent
for _pkg in ['pyclnf', 'pymtcnn', 'pyfaceau', 'pyfhog']:
    _path = _root / _pkg
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def load_video_path_from_manifest(manifest_path: str, task_id: int) -> str:
    """Load video path from manifest file using task ID."""
    with open(manifest_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if task_id >= len(lines):
        raise ValueError(f"Task ID {task_id} exceeds manifest size {len(lines)}")

    return lines[task_id]


def main():
    parser = argparse.ArgumentParser(description='Process single video to HDF5')

    # Input options (either direct path or manifest)
    parser.add_argument('video_path', nargs='?', type=str, default=None,
                        help='Path to video file (direct mode)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to video manifest file')
    parser.add_argument('--task-id', type=int, default=None,
                        help='Task ID (default: from SLURM_ARRAY_TASK_ID)')

    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output HDF5 path (for direct mode)')
    parser.add_argument('--output-dir', type=str, default='bigred200/output/per_video',
                        help='Output directory (for manifest mode)')

    # Processing options
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Skip every N frames (0 = no skipping)')
    parser.add_argument('--min-quality', type=float, default=0.5,
                        help='Minimum quality threshold (0.0-1.0)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Get task ID
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # Determine video path
    if args.video_path:
        video_path = Path(args.video_path)
    elif args.manifest:
        video_path = Path(load_video_path_from_manifest(args.manifest, task_id))
    else:
        parser.error("Either video_path or --manifest is required")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"video_{task_id:05d}.h5"

    # Print header
    print("=" * 60)
    print(f"TASK {task_id}: Processing {video_path.name}")
    print("=" * 60)
    print(f"Input:  {video_path}")
    print(f"Output: {output_path}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Min quality: {args.min_quality}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")

    # Verify input exists
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    # Import after arg parsing (heavy imports)
    from pyfaceau.data.training_data_generator import TrainingDataGenerator, GeneratorConfig

    # Configure generator
    config = GeneratorConfig(
        skip_frames=args.skip_frames,
        min_quality=args.min_quality,
        verbose=not args.quiet,
    )

    generator = TrainingDataGenerator(config)

    # Process video
    start_time = time.time()

    try:
        stats = generator.process_video(
            video_path=video_path,
            output_path=output_path,
            max_frames=args.max_frames,
        )
    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print(f"TASK {task_id}: COMPLETE")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Frames read: {stats['total_frames']}")
    print(f"Frames saved: {stats['processed_frames']}")
    print(f"Frames filtered: {stats['filtered_frames']}")
    print(f"Frames failed: {stats['failed_frames']}")
    print(f"Time: {elapsed:.1f}s")
    if stats['total_frames'] > 0:
        print(f"Processing rate: {stats['total_frames']/elapsed:.1f} fps")

    # Output file size
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Output size: {size_mb:.1f} MB")

    # Exit success if we got at least some frames
    if stats['processed_frames'] > 0:
        return 0
    else:
        print("WARNING: No frames processed successfully")
        return 1


if __name__ == '__main__':
    sys.exit(main())
