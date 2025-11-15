#!/usr/bin/env python3
"""Quick test to see shape_change values during CLNF optimization."""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from pyclnf import CLNF

# Test image
test_img = Path(__file__).parent / "calibration_frames" / "patient1_frame1.jpg"
img = cv2.imread(str(test_img))

# Initialize CLNF
clnf = CLNF(regularization=35, max_iterations=10)

# Detect landmarks (this will print the iteration output)
landmarks, success = clnf.detect_landmarks(img, (296, 778, 405, 407))

print(f"\n{'='*80}")
print(f"Final result: {'Success' if success else 'Failed'}")
print(f"{'='*80}")
