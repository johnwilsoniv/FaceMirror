import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
import cv2
import pandas as pd

img = cv2.imread("calibration_frames/patient1_frame1.jpg")

print("C++ Raw ONet Landmarks:")
df = pd.read_csv("/tmp/mtcnn_debug.csv")
for i in range(1, 6):
    print(f"  Point {i-1}: ({df.iloc[0][f'lm{i}_x']:.6f}, {df.iloc[0][f'lm{i}_y']:.6f})")

print("\n")

detector = CoreMLMTCNN(verbose=False)
bboxes, landmarks = detector.detect(img)
