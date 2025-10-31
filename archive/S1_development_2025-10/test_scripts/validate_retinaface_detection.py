#!/usr/bin/env python3
"""
Validate RetinaFace detection against C++ OpenFace baseline
Tests detection success rate and confidence matching
"""

import cv2
import pandas as pd
import numpy as np
import sys
import onnxruntime as ort
import torch
import warnings
warnings.filterwarnings('ignore')

# RetinaFace post-processing utilities
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from openface.Pytorch_Retinaface.data import cfg_mnet

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
ONNX_MODEL = "weights/retinaface_mobilenet025_coreml.onnx"

class SimpleRetinaFace:
    """Minimal RetinaFace detector without profiler dependency"""

    def __init__(self, onnx_path, confidence_threshold=0.02, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.cfg = cfg_mnet

        # Load ONNX model with CPU provider only (fast enough for validation)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

    def detect_faces(self, frame):
        """Detect faces in frame"""
        # Preprocess
        img = np.float32(frame)
        img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # Inference
        outputs = self.session.run(None, {'input': img})
        loc, conf, landms = outputs

        # Post-process
        im_height, im_width = frame.shape[:2]

        loc = torch.from_numpy(loc)
        conf = torch.from_numpy(conf)
        landms = torch.from_numpy(landms)

        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms_decoded = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5)
        landms_decoded = landms_decoded * scale1
        landms_decoded = landms_decoded.cpu().numpy()

        # Filter by confidence
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes, landms_decoded, scores = boxes[inds], landms_decoded[inds], scores[inds]

        # NMS
        if len(boxes) > 0:
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            dets = dets[keep]
            landms_decoded = landms_decoded[keep]

            # Concatenate boxes and landmarks
            dets = np.concatenate((dets, landms_decoded), axis=1)
            return dets
        else:
            return None

def main():
    print("=" * 80)
    print("COMPONENT 2 VALIDATION: RetinaFace vs C++ OpenFace")
    print("=" * 80)
    print()

    # Load C++ baseline
    print(f"Loading C++ baseline: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    total_frames = len(df)
    cpp_detections = df['success'].sum()
    cpp_mean_conf = df[df['success'] == 1]['confidence'].mean()

    print(f"  Total frames: {total_frames}")
    print(f"  C++ detections: {cpp_detections}/{total_frames} ({100*cpp_detections/total_frames:.1f}%)")
    print(f"  C++ mean confidence: {cpp_mean_conf:.3f}")
    print()

    # Load RetinaFace
    print(f"Loading RetinaFace: {ONNX_MODEL}")
    detector = SimpleRetinaFace(ONNX_MODEL)
    print("  Model loaded (CPU)")
    print()

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total_video_frames}")
    print()

    # Test on first 50 frames for quick validation
    TEST_FRAMES = 50
    print(f"Testing RetinaFace on first {TEST_FRAMES} frames (quick validation)...")
    print()

    rf_results = []
    frame_idx = 0

    while frame_idx < TEST_FRAMES:  # Quick test
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        dets = detector.detect_faces(frame)

        if dets is not None and len(dets) > 0:
            confidence = dets[0][4]
            rf_results.append({
                'frame': frame_idx + 1,
                'detected': 1,
                'confidence': float(confidence)
            })
        else:
            rf_results.append({
                'frame': frame_idx + 1,
                'detected': 0,
                'confidence': 0.0
            })

        frame_idx += 1

        if (frame_idx % 10) == 0:
            print(f"  Processed {frame_idx}/{TEST_FRAMES} frames...")

    cap.release()
    print(f"  Completed: {frame_idx} frames")
    print()

    # Convert to DataFrame
    rf_df = pd.DataFrame(rf_results)

    # Compare results
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()

    rf_detections = rf_df['detected'].sum()
    rf_mean_conf = rf_df[rf_df['detected'] == 1]['confidence'].mean()

    print(f"RetinaFace:")
    print(f"  Detections: {rf_detections}/{frame_idx} ({100*rf_detections/frame_idx:.1f}%)")
    print(f"  Mean confidence: {rf_mean_conf:.3f}")
    print()

    print(f"C++ OpenFace:")
    print(f"  Detections: {cpp_detections}/{total_frames} ({100*cpp_detections/total_frames:.1f}%)")
    print(f"  Mean confidence: {cpp_mean_conf:.3f}")
    print()

    # Frame-by-frame comparison
    min_frames = min(len(rf_df), len(df))
    rf_success = rf_df['detected'].values[:min_frames]
    cpp_success = df['success'].values[:min_frames]

    agreement = (rf_success == cpp_success).sum()
    agreement_pct = 100 * agreement / min_frames

    print(f"Frame-by-Frame Agreement:")
    print(f"  Matching: {agreement}/{min_frames} ({agreement_pct:.2f}%)")
    print()

    # Analyze disagreements
    disagreements = np.where(rf_success != cpp_success)[0]
    if len(disagreements) > 0:
        rf_yes_cpp_no = np.sum((rf_success == 1) & (cpp_success == 0))
        rf_no_cpp_yes = np.sum((rf_success == 0) & (cpp_success == 1))

        print(f"  Disagreements: {len(disagreements)} frames")
        print(f"    RetinaFace detected, C++ missed: {rf_yes_cpp_no}")
        print(f"    C++ detected, RetinaFace missed: {rf_no_cpp_yes}")

        if len(disagreements) <= 10:
            print(f"    Frame numbers: {disagreements + 1}")
    print()

    # Confidence correlation (for successfully detected frames)
    both_detected = (rf_success == 1) & (cpp_success == 1)
    if both_detected.sum() > 0:
        rf_conf_subset = rf_df[rf_df['detected'] == 1]['confidence'].values
        cpp_conf_subset = df[df['success'] == 1]['confidence'].values[:len(rf_conf_subset)]

        if len(rf_conf_subset) == len(cpp_conf_subset):
            conf_corr = np.corrcoef(rf_conf_subset, cpp_conf_subset)[0, 1]
            conf_diff = np.abs(rf_conf_subset - cpp_conf_subset).mean()

            print(f"Confidence Comparison (detected frames only):")
            print(f"  Correlation: r={conf_corr:.4f}")
            print(f"  Mean absolute difference: {conf_diff:.4f}")
            print()

    # Verdict
    print("=" * 80)
    print("VALIDATION VERDICT")
    print("=" * 80)
    print()

    if agreement_pct == 100.0 and rf_detections == cpp_detections:
        print("✅ PERFECT MATCH: RetinaFace exactly matches C++ OpenFace")
        print()
        print("   Detection rate: 100% agreement")
        print("   Frame-by-frame: 100% match")
        print("   Component 2: VALIDATED ✅")
    elif agreement_pct >= 99.0:
        print("✅ EXCELLENT MATCH: RetinaFace nearly identical to C++ OpenFace")
        print()
        print(f"   Detection rate: {100*rf_detections/frame_idx:.1f}% vs {100*cpp_detections/total_frames:.1f}%")
        print(f"   Frame-by-frame: {agreement_pct:.2f}% agreement")
        print("   Component 2: VALIDATED ✅")
        print()
        print(f"   Note: {len(disagreements)} frame(s) differ - acceptable for production")
    elif agreement_pct >= 95.0:
        print("✓ GOOD MATCH: RetinaFace performs well")
        print()
        print(f"   Detection rate: {100*rf_detections/frame_idx:.1f}% vs {100*cpp_detections/total_frames:.1f}%")
        print(f"   Frame-by-frame: {agreement_pct:.2f}% agreement")
        print("   Component 2: USABLE ✓")
        print()
        print(f"   Note: {len(disagreements)} frame(s) differ - review if critical")
    else:
        print("⚠️  SIGNIFICANT DIFFERENCES: Review required")
        print()
        print(f"   Detection rate: {100*rf_detections/frame_idx:.1f}% vs {100*cpp_detections/total_frames:.1f}%")
        print(f"   Frame-by-frame: {agreement_pct:.2f}% agreement")
        print("   Component 2: NEEDS REVIEW ⚠️")
    print()

    # Save results
    output_file = "retinaface_validation_results.csv"
    rf_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
