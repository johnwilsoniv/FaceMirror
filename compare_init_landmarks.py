#!/usr/bin/env python3
"""
Compare C++ and Python initial landmarks to find the source of divergence.
"""
import numpy as np

# Load C++ initial landmarks
cpp_landmarks = []
with open('/tmp/cpp_init_landmarks_68.txt', 'r') as f:
    for line in f:
        if line.startswith('Landmark_'):
            parts = line.split(': (')[1].strip(')\n').split(', ')
            x, y = float(parts[0]), float(parts[1])
            cpp_landmarks.append([x, y])

cpp_landmarks = np.array(cpp_landmarks)
print(f"C++ landmarks: {cpp_landmarks.shape}")

# Load Python initial landmarks from earlier test
# We need to run our Python CLNF and save initial landmarks
# For now, let's check landmark 36 which we know differs

print("\nC++ Landmark 36:")
print(f"  Position: ({cpp_landmarks[36, 0]:.3f}, {cpp_landmarks[36, 1]:.3f})")

print("\nPrevious Python Landmark 36 (from earlier test):")
print(f"  Position: (377.862, 848.968)")

print("\nDifference:")
print(f"  dx: {377.862 - cpp_landmarks[36, 0]:.3f}")
print(f"  dy: {848.968 - cpp_landmarks[36, 1]:.3f}")
print(f"  distance: {np.sqrt((377.862 - cpp_landmarks[36, 0])**2 + (848.968 - cpp_landmarks[36, 1])**2):.3f} pixels")

print("\n" + "="*60)
print("HYPOTHESIS: Initialization differs between C++ and Python")
print("="*60)
print("\nThis would explain why response maps differ:")
print("- Different initial landmarks → different patch locations")
print("- Different patch locations → different CEN response maps")
print("- Same CEN model, but different inputs")

print("\nNext step: Save Python initial landmarks and compare all 68 points")
