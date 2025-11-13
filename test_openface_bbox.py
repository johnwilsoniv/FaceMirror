"""
Test PyCLNF with the exact bbox that OpenFace C++ detected.

This isolates bbox initialization from face detector differences.
"""
import cv2
import numpy as np
from pyclnf import CLNF

# Load test video and extract frame 50 (same as previous tests)
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read frame")
    exit(1)

print(f"Frame shape: {frame.shape}")

# OpenFace C++ detected bbox (from DEBUG_BBOX output)
# Format: x, y, width, height
openface_bbox = (293.145, 702.034, 418.033, 404.659)
print(f"\nOpenFace C++ bbox: x={openface_bbox[0]:.1f}, y={openface_bbox[1]:.1f}, "
      f"w={openface_bbox[2]:.1f}, h={openface_bbox[3]:.1f}")

# (Skipping Python Haar cascade comparison due to cv2.data compatibility)

# Now test PyCLNF with OpenFace's bbox
print("\n" + "="*60)
print("Testing PyCLNF with OpenFace C++ bbox")
print("="*60)

clnf = CLNF(model_dir="pyclnf/models", scale=0.25)
landmarks, info = clnf.fit(frame, openface_bbox, return_params=True)

print(f"\nPyCLNF Results:")
print(f"  Converged: {info['converged']}")
print(f"  Iterations: {info['iterations']}")
print(f"  Scale: {info['params'][0]:.3f}")
print(f"  Rotation: wx={info['params'][1]:.3f}, wy={info['params'][2]:.3f}, wz={info['params'][3]:.3f}")
print(f"  Translation: tx={info['params'][4]:.1f}, ty={info['params'][5]:.1f}")
print(f"\nLandmark range:")
print(f"  x: [{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}]")
print(f"  y: [{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")

# Compare to OpenFace C++ landmarks (from previous test)
print("\n" + "="*60)
print("Comparison to OpenFace C++ (from previous test)")
print("="*60)
print("OpenFace C++ landmarks:")
print("  x: [300.7, 703.6]")
print("  y: [691.4, 1087.6]")
print("\nPyCLNF vs OpenFace:")
print(f"  x range diff: [{landmarks[:, 0].min() - 300.7:.1f}, {landmarks[:, 0].max() - 703.6:.1f}]")
print(f"  y range diff: [{landmarks[:, 1].min() - 691.4:.1f}, {landmarks[:, 1].max() - 1087.6:.1f}]")

# Visualize
vis = frame.copy()
for i, (x, y) in enumerate(landmarks):
    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

# Draw OpenFace bbox
x, y, w, h = openface_bbox
cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
cv2.putText(vis, "OpenFace C++ bbox", (int(x), int(y-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imwrite("pyclnf_openface_bbox_test.jpg", vis)
print("\nVisualization saved to: pyclnf_openface_bbox_test.jpg")
