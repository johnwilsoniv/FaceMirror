#!/usr/bin/env python3
"""
Validate PFLD 68-Point Landmark Detection (Component 3)

Tests PFLD landmark detector against C++ OpenFace baseline CSV.
Compares landmark positions and calculates RMSE error.

Expected result: RMSE < 3 pixels for high-quality videos
"""

import cv2
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from pfld_landmark_detector import PFLDLandmarkDetector, visualize_landmarks

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PFLD_MODEL = "weights/pfld_68_landmarks.onnx"
RETINAFACE_MODEL = "weights/retinaface_mobilenet025_coreml.onnx"


def load_csv_landmarks(csv_path):
    """Load 68 landmarks from CSV baseline."""
    df = pd.read_csv(csv_path)

    # Extract landmark columns (x_0 through x_67, y_0 through y_67)
    landmark_cols_x = [f'x_{i}' for i in range(68)]
    landmark_cols_y = [f'y_{i}' for i in range(68)]

    if not all(col in df.columns for col in landmark_cols_x + landmark_cols_y):
        print("ERROR: CSV does not contain landmark columns (x_0...x_67, y_0...y_67)")
        return None

    # Convert to (num_frames, 68, 2) array
    landmarks_x = df[landmark_cols_x].values  # (num_frames, 68)
    landmarks_y = df[landmark_cols_y].values  # (num_frames, 68)

    landmarks = np.stack([landmarks_x, landmarks_y], axis=2)  # (num_frames, 68, 2)

    return landmarks, df['success'].values


def calculate_landmark_rmse(pred_landmarks, gt_landmarks):
    """
    Calculate RMSE between predicted and ground truth landmarks.

    Args:
        pred_landmarks: (68, 2) predicted landmarks
        gt_landmarks: (68, 2) ground truth landmarks

    Returns:
        rmse: Root mean square error in pixels
    """
    if pred_landmarks is None or gt_landmarks is None:
        return None

    diff = pred_landmarks - gt_landmarks
    squared_errors = np.sum(diff ** 2, axis=1)  # (68,)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return rmse


def main():
    print("=" * 80)
    print("COMPONENT 3 VALIDATION: PFLD 68-Point Landmarks vs C++ OpenFace")
    print("=" * 80)
    print()

    # Load CSV baseline
    print(f"Loading C++ baseline: {CSV_PATH}")
    csv_landmarks, csv_success = load_csv_landmarks(CSV_PATH)

    if csv_landmarks is None:
        print("ERROR: Failed to load CSV landmarks")
        return

    total_frames = len(csv_landmarks)
    cpp_detections = csv_success.sum()

    print(f"  Total frames: {total_frames}")
    print(f"  C++ successful detections: {cpp_detections}/{total_frames} ({100*cpp_detections/total_frames:.1f}%)")
    print()

    # Load PFLD detector
    print(f"Loading PFLD landmark detector: {PFLD_MODEL}")
    landmark_detector = PFLDLandmarkDetector(PFLD_MODEL)
    print("  PFLD model loaded (CPU)")
    print()

    # Load RetinaFace for face detection
    print(f"Loading RetinaFace detector: {RETINAFACE_MODEL}")
    import onnxruntime as ort
    import torch
    from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
    from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
    from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
    from openface.Pytorch_Retinaface.data import cfg_mnet

    # Simple RetinaFace wrapper
    class SimpleRetinaFace:
        def __init__(self, onnx_path, confidence_threshold=0.02, nms_threshold=0.4):
            self.confidence_threshold = confidence_threshold
            self.nms_threshold = nms_threshold
            self.cfg = cfg_mnet

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

        def detect_faces(self, frame):
            img = np.float32(frame)
            img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)

            outputs = self.session.run(None, {'input': img})
            loc, conf, landms = outputs

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

            inds = np.where(scores > self.confidence_threshold)[0]
            boxes, landms_decoded, scores = boxes[inds], landms_decoded[inds], scores[inds]

            if len(boxes) > 0:
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, self.nms_threshold)
                dets = dets[keep]
                landms_decoded = landms_decoded[keep]

                dets = np.concatenate((dets, landms_decoded), axis=1)
                return dets
            else:
                return None

    face_detector = SimpleRetinaFace(RETINAFACE_MODEL)
    print("  RetinaFace loaded (CPU)")
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
    print(f"Testing PFLD on first {TEST_FRAMES} frames (quick validation)...")
    print()

    results = []
    frame_idx = 0

    while frame_idx < TEST_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face
        dets = face_detector.detect_faces(frame)

        if dets is not None and len(dets) > 0:
            # Use largest face
            bbox = dets[0][:4]  # [x1, y1, x2, y2]

            # Detect landmarks
            pfld_landmarks = landmark_detector.detect_landmarks(frame, bbox)

            if pfld_landmarks is not None:
                # Compare with CSV baseline
                csv_frame_landmarks = csv_landmarks[frame_idx]
                rmse = calculate_landmark_rmse(pfld_landmarks, csv_frame_landmarks)

                results.append({
                    'frame': frame_idx + 1,
                    'detected': 1,
                    'rmse': rmse
                })
            else:
                results.append({
                    'frame': frame_idx + 1,
                    'detected': 0,
                    'rmse': None
                })
        else:
            results.append({
                'frame': frame_idx + 1,
                'detected': 0,
                'rmse': None
            })

        frame_idx += 1

        if (frame_idx % 10) == 0:
            print(f"  Processed {frame_idx}/{TEST_FRAMES} frames...")

    cap.release()
    print(f"  Completed: {frame_idx} frames")
    print()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate statistics
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()

    pfld_detections = results_df['detected'].sum()
    pfld_detection_rate = 100 * pfld_detections / frame_idx

    print(f"PFLD Detector:")
    print(f"  Detections: {pfld_detections}/{frame_idx} ({pfld_detection_rate:.1f}%)")

    # RMSE statistics (for successfully detected frames)
    detected_results = results_df[results_df['detected'] == 1]
    if len(detected_results) > 0:
        mean_rmse = detected_results['rmse'].mean()
        median_rmse = detected_results['rmse'].median()
        max_rmse = detected_results['rmse'].max()
        min_rmse = detected_results['rmse'].min()

        print(f"  Mean RMSE: {mean_rmse:.2f} pixels")
        print(f"  Median RMSE: {median_rmse:.2f} pixels")
        print(f"  Min RMSE: {min_rmse:.2f} pixels")
        print(f"  Max RMSE: {max_rmse:.2f} pixels")
    else:
        print("  No successful detections")

    print()

    print(f"C++ OpenFace Baseline:")
    cpp_detections_subset = csv_success[:frame_idx].sum()
    cpp_detection_rate = 100 * cpp_detections_subset / frame_idx
    print(f"  Detections: {cpp_detections_subset}/{frame_idx} ({cpp_detection_rate:.1f}%)")
    print()

    # Frame-by-frame agreement
    pfld_success = results_df['detected'].values
    cpp_success_subset = csv_success[:frame_idx]
    agreement = (pfld_success == cpp_success_subset).sum()
    agreement_pct = 100 * agreement / frame_idx

    print(f"Detection Agreement:")
    print(f"  Matching: {agreement}/{frame_idx} ({agreement_pct:.2f}%)")
    print()

    # Validation verdict
    print("=" * 80)
    print("VALIDATION VERDICT")
    print("=" * 80)
    print()

    if len(detected_results) > 0 and mean_rmse < 3.0:
        if pfld_detection_rate >= 95.0:
            print("✅ EXCELLENT: PFLD landmarks highly accurate")
            print()
            print(f"   Mean RMSE: {mean_rmse:.2f} pixels (< 3 pixel target)")
            print(f"   Detection rate: {pfld_detection_rate:.1f}%")
            print("   Component 3: VALIDATED ✅")
        elif pfld_detection_rate >= 90.0:
            print("✅ GOOD: PFLD landmarks accurate")
            print()
            print(f"   Mean RMSE: {mean_rmse:.2f} pixels (< 3 pixel target)")
            print(f"   Detection rate: {pfld_detection_rate:.1f}%")
            print("   Component 3: USABLE ✓")
        else:
            print("⚠️  ACCEPTABLE: PFLD landmarks accurate but detection rate low")
            print()
            print(f"   Mean RMSE: {mean_rmse:.2f} pixels (< 3 pixel target)")
            print(f"   Detection rate: {pfld_detection_rate:.1f}% (review)")
            print("   Component 3: NEEDS REVIEW ⚠️")
    elif len(detected_results) > 0:
        print("⚠️  SIGNIFICANT ERROR: PFLD landmarks need tuning")
        print()
        print(f"   Mean RMSE: {mean_rmse:.2f} pixels (target: < 3 pixels)")
        print(f"   Detection rate: {pfld_detection_rate:.1f}%")
        print("   Component 3: NEEDS IMPROVEMENT ⚠️")
    else:
        print("❌ FAILURE: PFLD not detecting faces")
        print()
        print("   No successful detections")
        print("   Component 3: FAILED ❌")

    print()

    # Save results
    output_file = "pfld_validation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
