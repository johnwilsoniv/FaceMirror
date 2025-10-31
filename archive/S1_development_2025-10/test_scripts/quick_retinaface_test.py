#!/usr/bin/env python3
import cv2
import sys
sys.path.insert(0, ".")
from onnx_retinaface_detector import OptimizedFaceDetector

# Load detector
detector = OptimizedFaceDetector(
    model_path="weights/Alignment_RetinaFace.pth",
    onnx_model_path="weights/retinaface_mobilenet025_coreml.onnx",
    device="cpu",
    confidence_threshold=0.02,  # Very low threshold
    nms_threshold=0.4,
    vis_threshold=0.5
)

# Load mirrored frame
cap = cv2.VideoCapture('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4')
ret, frame = cap.read()
cap.release()

print(f"Frame shape: {frame.shape}")
print()

# Test detection
dets = detector.detect_faces(frame, resize=1.0)

print(f"Detections found: {len(dets) if dets is not None and len(dets) > 0 else 0}")

if dets is not None and len(dets) > 0:
    print(f"\\nTop {min(5, len(dets))} detections:")
    for i, det in enumerate(dets[:5]):
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        w, h = x2-x1, y2-y1
        print(f"  {i+1}. bbox=[{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}]  size={w}x{h}  conf={conf:.4f}")
else:
    print("\\nNO FACES DETECTED - even with threshold=0.02!")
    print("This suggests RetinaFace fundamentally cannot detect faces in mirrored videos.")
