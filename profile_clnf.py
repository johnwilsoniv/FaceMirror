#!/usr/bin/env python3
"""Profile CLNF to find remaining bottlenecks."""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import time
import cProfile
import pstats
from io import StringIO

from pyclnf.clnf import CLNF

# Load video
video_path = 'Patient Data/Normal Cohort/Shorty.mov'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print('Failed to load frame')
    exit()

# Initialize CLNF
print('Initializing CLNF...')
clnf = CLNF('pyclnf/pyclnf/models', regularization=40)

# Warmup
print('Warmup...')
_ = clnf.detect_and_fit(frame)

# Profile with timing
print('Profiling 3 frames...')
pr = cProfile.Profile()
pr.enable()

for _ in range(3):
    result = clnf.detect_and_fit(frame)

pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())
