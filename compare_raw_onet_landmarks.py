#!/usr/bin/env python3
"""Compare raw ONet landmark outputs (indices 6-15) before denormalization"""

import numpy as np
import cv2
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN

# Create a debug version that captures raw ONet output
class DebugCoreMLMTCNN(CoreMLMTCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_onet_output = None
        self.bbox_before_calibration = None
        
    def detect(self, img):
        # Run normal detection but capture ONet output
        result = super().detect(img)
        return result

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

# Load C++ landmarks from CSV
df = pd.read_csv("/tmp/mtcnn_debug.csv")
row = df.iloc[0]

print("="*80)
print("C++ Raw ONet Landmark Values (indices 6-15)")
print("="*80)

cpp_raw_landmarks = []
for i in range(1, 6):
    lm_x = row[f'lm{i}_x']
    lm_y = row[f'lm{i}_y']
    cpp_raw_landmarks.append([lm_x, lm_y])
    print(f"  Point {i-1}: ({lm_x:.6f}, {lm_y:.6f})")

cpp_raw_landmarks = np.array(cpp_raw_landmarks)

# Now I need to extract the same from PyMTCNN
# The issue is that PyMTCNN's detect() method doesn't expose the raw ONet output
# Let me check what the ONet output looks like by adding debug code

img = cv2.imread(TEST_IMAGE)
detector = CoreMLMTCNN(verbose=False)

# I need to patch the ONet stage to capture raw output
# Let me manually run through the stages

print("\n" + "="*80)
print("Need to add debug output to PyMTCNN to capture raw ONet values")
print("="*80)
print("\nThe raw ONet output (indices 6-15) contains normalized landmark coordinates")
print("These should be identical between C++ and PyMTCNN if the models match")
print("\nC++ raw values shown above - need to compare with PyMTCNN ONet output")
