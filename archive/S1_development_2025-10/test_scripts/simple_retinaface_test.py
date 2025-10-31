#!/usr/bin/env python3
"""Simple RetinaFace test without profiler dependencies"""

import cv2
import pandas as pd
import numpy as np
import sys

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Try direct ONNX approach without the full detector wrapper
import onnxruntime as ort

# RetinaFace post-processing
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from openface.Pytorch_Retinaface.data import cfg_mnet
import torch

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
ONNX_MODEL = "weights/retinaface_mobilenet025_coreml.onnx"

print("Loading RetinaFace ONNX model...")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(ONNX_MODEL, sess_options=sess_options, providers=['CPUExecutionProvider'])
print("Model loaded")
print()

print("Loading CSV baseline...")
df = pd.read_csv(CSV_PATH)
print(f"CSV frames: {len(df)}, C++ detections: {df['success'].sum()}/{len(df)}")
print()

print("Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open {VIDEO_PATH}")
    sys.exit(1)
print("Video opened")
print()

print("Testing first 10 frames...")
cfg = cfg_mnet
confidence_threshold = 0.02
nms_threshold = 0.4

detections = []

for frame_idx in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = np.float32(frame)
    img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    # Inference
    outputs = session.run(None, {'input': img})
    loc, conf, landms = outputs

    # Post-process
    im_height, im_width = frame.shape[:2]

    loc = torch.from_numpy(loc)
    conf = torch.from_numpy(conf)
    landms = torch.from_numpy(landms)

    scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms_decoded = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5)
    landms_decoded = landms_decoded * scale1
    landms_decoded = landms_decoded.cpu().numpy()

    inds = np.where(scores > confidence_threshold)[0]
    boxes, landms_decoded, scores = boxes[inds], landms_decoded[inds], scores[inds]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep]

    if len(dets) > 0:
        confidence = dets[0][4]
        detections.append((frame_idx + 1, 1, confidence))
        print(f"Frame {frame_idx+1}: DETECTED (conf={confidence:.3f})")
    else:
        detections.append((frame_idx + 1, 0, 0.0))
        print(f"Frame {frame_idx+1}: NOT DETECTED")

cap.release()

print()
print(f"RetinaFace: {sum(d[1] for d in detections)}/10 frames detected")
print(f"C++ OpenFace: {df['success'].iloc[:10].sum()}/10 frames detected")
print()

if sum(d[1] for d in detections) == df['success'].iloc[:10].sum() == 10:
    print("✅ SUCCESS: RetinaFace matches C++ (100% detection on first 10 frames)")
else:
    print("⚠️  MISMATCH: Review results")
