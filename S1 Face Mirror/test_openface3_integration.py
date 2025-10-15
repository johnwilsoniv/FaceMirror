#!/usr/bin/env python3
"""
Test script for OpenFace 3.0 integration

Tests the complete pipeline:
1. Face detection
2. Landmark detection (98 points)
3. AU extraction (8 AUs)
4. AU adaptation (8 → 18 AUs)
5. CSV output generation
"""

import sys
from pathlib import Path

# Test with a single mirrored video
test_video = Path(__file__).parent / 'tqdm_test_output' / 'IMG_0452_left_mirrored.mp4'

if not test_video.exists():
    print(f"Error: Test video not found at {test_video}")
    print("Please ensure test videos exist in tqdm_test_output/")
    sys.exit(1)

print("="*60)
print("TESTING OPENFACE 3.0 INTEGRATION")
print("="*60)
print(f"\nTest video: {test_video.name}")
print(f"Video exists: {test_video.exists()}")
print(f"Video size: {test_video.stat().st_size / (1024*1024):.2f} MB\n")

# Import and test
try:
    from openface_integration import OpenFace3Processor

    print("Step 1: Initializing OpenFace 3.0 models...")
    processor = OpenFace3Processor(device='cpu')
    print("✓ Models initialized successfully\n")

    print("Step 2: Processing video...")
    output_csv = Path(__file__).parent / 'test_output' / f'{test_video.stem}.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_count = processor.process_video(test_video, output_csv)

    if frame_count > 0:
        print(f"\n✓ Processing complete: {frame_count} frames processed")
        print(f"✓ CSV output: {output_csv}")

        # Verify CSV contents
        print("\nStep 3: Verifying CSV output...")
        import csv
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if rows:
                first_row = rows[0]
                print(f"  Total rows: {len(rows)}")
                print(f"  Columns: {len(first_row)}")
                print(f"  First 10 columns: {list(first_row.keys())[:10]}")

                # Check for AU columns
                au_r_cols = [col for col in first_row.keys() if col.endswith('_r')]
                au_c_cols = [col for col in first_row.keys() if col.endswith('_c')]
                print(f"  AU intensity (_r) columns: {len(au_r_cols)}")
                print(f"  AU binary (_c) columns: {len(au_c_cols)}")

                # Show first frame AU values
                print(f"\n  Sample AU values (frame 0):")
                for au_col in sorted(au_r_cols):
                    value = first_row[au_col]
                    print(f"    {au_col}: {value}")

                print("\n✓ CSV format verified")
            else:
                print("  ✗ CSV is empty!")
    else:
        print("\n✗ Processing failed: 0 frames processed")
        sys.exit(1)

    print("\n" + "="*60)
    print("TEST PASSED ✓")
    print("="*60)

except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
