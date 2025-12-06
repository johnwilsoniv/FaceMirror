#!/usr/bin/env python3
"""
Generate Video Manifest for SLURM Array Job

Discovers all video files in the data directory and creates a manifest file
that maps SLURM_ARRAY_TASK_ID to video paths.

Usage:
    python generate_video_manifest.py --videos-dir "S Data" --output manifest.txt
"""

import argparse
from pathlib import Path
import sys


def discover_videos(videos_dir: Path, patterns: list = None) -> list:
    """
    Discover all video files in directory.

    Args:
        videos_dir: Root directory to search
        patterns: File patterns to match (default: MOV, mov, mp4, MP4)

    Returns:
        List of video paths, sorted by name
    """
    if patterns is None:
        patterns = ['*.MOV', '*.mov', '*.mp4', '*.MP4']

    video_paths = []
    for pattern in patterns:
        video_paths.extend(videos_dir.glob(f'**/{pattern}'))

    # Remove duplicates and sort
    video_paths = sorted(set(video_paths))

    return video_paths


def main():
    parser = argparse.ArgumentParser(description='Generate video manifest for SLURM array job')
    parser.add_argument('--videos-dir', default='S Data',
                        help='Root directory containing videos')
    parser.add_argument('--output', default='bigred200/config/video_manifest.txt',
                        help='Output manifest file path')
    parser.add_argument('--patterns', nargs='+', default=None,
                        help='File patterns to match (e.g., *.MOV *.mp4)')
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_path = Path(args.output)

    if not videos_dir.exists():
        print(f"ERROR: Videos directory not found: {videos_dir}")
        sys.exit(1)

    # Discover videos
    print(f"Searching for videos in: {videos_dir}")
    video_paths = discover_videos(videos_dir, args.patterns)

    if not video_paths:
        print("ERROR: No videos found!")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write manifest
    with open(output_path, 'w') as f:
        for vp in video_paths:
            f.write(f"{vp}\n")

    # Print summary
    print(f"\nFound {len(video_paths)} videos:")

    # Group by subdirectory
    subdirs = {}
    for vp in video_paths:
        subdir = vp.parent.name
        subdirs[subdir] = subdirs.get(subdir, 0) + 1

    for subdir, count in sorted(subdirs.items()):
        print(f"  {subdir}: {count} videos")

    print(f"\nManifest saved to: {output_path}")
    print(f"\nFor SLURM array job, use:")
    print(f"  #SBATCH --array=0-{len(video_paths)-1}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
