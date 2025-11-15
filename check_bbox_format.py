import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
import cv2

img = cv2.imread("calibration_frames/patient1_frame1.jpg")
detector = CoreMLMTCNN(verbose=False)
bboxes, landmarks = detector.detect(img)

print("BBox array contents:")
print(f"bbox[0] = {bboxes[0]}")
print(f"\nIndividual values:")
print(f"  bbox[0][0] = {bboxes[0][0]}")
print(f"  bbox[0][1] = {bboxes[0][1]}")
print(f"  bbox[0][2] = {bboxes[0][2]}")
print(f"  bbox[0][3] = {bboxes[0][3]}")

print(f"\nLandmarks array shape: {landmarks.shape}")
print(f"landmarks[0] =\n{landmarks[0]}")
