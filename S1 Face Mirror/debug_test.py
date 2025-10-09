#!/usr/bin/env python
"""Quick debug test for progress monitoring"""

from face_splitter import StableFaceSplitter
from pathlib import Path

# Test with one video
test_video = Path("../D Normal Pts/IMG_0452.MOV")
output_dir = Path("./debug_output")
output_dir.mkdir(exist_ok=True)

print("Starting debug test with progress monitoring...")
print(f"Test video: {test_video}")
print("="*60)

splitter = StableFaceSplitter(debug_mode=True)
result = splitter.process_video(str(test_video), str(output_dir))

print("="*60)
print("Debug test complete!")
print(f"Output files: {result}")
