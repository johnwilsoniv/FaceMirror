#!/usr/bin/env python3
"""
Pure Python CNN MTCNN - Version 2
Standalone implementation using Pure Python CNN.
"""

import numpy as np
import cv2
from cpp_cnn_loader import CPPCNN
import os


class PurePythonMTCNN_V2:
    """
    Pure Python CNN MTCNN - standalone implementation.
    """

    def __init__(self, model_dir=None):
        """
        Initialize using Pure Python CNN instead of ONNX.

        Args:
            model_dir: Directory containing PNet.dat, RNet.dat, ONet.dat
        """
        # Don't call parent __init__ (we don't need ONNX models)
        if model_dir is None:
            model_dir = os.path.expanduser(
                "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
                "face_detection/mtcnn/convert_to_cpp/"
            )

        print(f"Loading Pure Python CNN MTCNN V2...")
        print(f"Model directory: {model_dir}")

        # Load Pure Python CNNs instead of ONNX
        self.pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))
        self.rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))
        self.onet = CPPCNN(os.path.join(model_dir, "ONet.dat"))

        # Official MTCNN thresholds matching C++
        self.thresholds = [0.6, 0.7, 0.7]  # PNet, RNet, ONet
        self.min_face_size = 60  # CRITICAL FIX: Match C++ (was 40, caused 140px bbox differences!)
        self.factor = 0.709

        print(f"✓ Pure Python CNN MTCNN V2 loaded!")

    def _preprocess(self, img: np.ndarray, flip_bgr_to_rgb: bool = False) -> np.ndarray:
        """
        Preprocess for Pure Python CNN (no batch dimension).

        Args:
            img: BGR image (H, W, 3)
            flip_bgr_to_rgb: If True, flip BGR to RGB (for PNet only, matching C++ line 151-155)

        Returns:
            Normalized image (3, H, W) in BGR or RGB order
        """
        # Same normalization as ONNX version
        img_norm = (img.astype(np.float32) - 127.5) * 0.0078125

        # Transpose to (C, H, W)
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # C++ flips BGR->RGB for ALL networks inside CNN::Inference (line 147-155)
        if flip_bgr_to_rgb:
            # Flip BGR to RGB (matching C++ line 147-155 inside Inference)
            flipped = img_chw[[2, 1, 0], :, :]
            # DEBUG: Verify flip happened
            if hasattr(self, '_debug_flip_once'):
                pass
            else:
                self._debug_flip_once = True
                print(f"[DEBUG] Flipping BGR→RGB: before [B={img_chw[0,0,0]:.4f}, G={img_chw[1,0,0]:.4f}, R={img_chw[2,0,0]:.4f}]")
                print(f"[DEBUG]               after  [R={flipped[0,0,0]:.4f}, G={flipped[1,0,0]:.4f}, B={flipped[2,0,0]:.4f}]")
            return flipped
        else:
            # Keep BGR order (not used - all networks flip)
            return img_chw

    # Override the network inference to use Pure Python CNN
    # The rest of the detect() method is inherited from CPPMTCNNDetector

    def _run_pnet(self, img_data, debug=False):
        """Run PNet using Pure Python CNN."""
        # Pure Python CNN returns a list of outputs
        outputs = self.pnet.forward(img_data, debug=debug)
        output = outputs[-1]  # Extract final output: (6, H, W)

        # Add batch dimension to match ONNX format: (1, 6, H, W)
        output = output[np.newaxis, :, :, :]

        return output

    def _run_rnet(self, img_data):
        """Run RNet using Pure Python CNN."""
        outputs = self.rnet(img_data)
        output = outputs[-1]  # Extract final output: (6,)
        return output

    def _run_onet(self, img_data):
        """Run ONet using Pure Python CNN."""
        outputs = self.onet(img_data)
        output = outputs[-1]  # Extract final output: (16,)
        return output

    # Now patch the detect() method to use our _run_* methods
    # We need to override the part that calls self.pnet.run()

    def detect(self, img: np.ndarray, debug=False):
        """
        Detect faces using Pure Python CNN MTCNN.
        Uses the exact pipeline from CPPMTCNNDetector but with Pure Python CNN inference.

        Args:
            img: Input image (H, W, 3) in BGR format
            debug: If True, print detailed logging
        """
        # Import here to avoid circular dependency
        from typing import Tuple
        import cv2

        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        if debug:
            print("\n" + "=" * 80)
            print("PURE PYTHON MTCNN V2 DEBUG")
            print("=" * 80)
            print(f"Input image: {img.shape}")
            print(f"Thresholds: {self.thresholds}")

        # Build image pyramid (same as ONNX version)
        min_size = self.min_face_size
        m = 12.0 / min_size
        min_l = min(img_h, img_w) * m

        scales = []
        scale = m
        while min_l >= 12:
            scales.append(scale)
            scale *= self.factor
            min_l *= self.factor

        # ===== Stage 1: PNet =====
        if debug:
            print(f"\n--- STAGE 1: PNet ---")
            print(f"Image pyramid: {len(scales)} scales")

        total_boxes = []

        for i, scale in enumerate(scales):
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))

            # Resize image (CRITICAL: Use INTER_LINEAR to match C++ cv::resize default)
            img_scaled = cv2.resize(img_float, (ws, hs), interpolation=cv2.INTER_LINEAR)
            # PNet: Flip BGR to RGB (matching C++ line 151-155)
            img_data = self._preprocess(img_scaled, flip_bgr_to_rgb=True)

            # DEBUG: Save scale 0 data for comparison with C++
            if i == 0 and debug:
                print(f"\n[PYTHON PNET DEBUG] Scale {i}: {scale:.6f}, Input: {ws}x{hs}")
                # Save preprocessed input (matching C++ format: HWC BGR)
                img_scaled_norm = (img_scaled - 127.5) * 0.0078125
                img_scaled_norm.astype(np.float32).tofile('/tmp/python_pnet_input_scale0.bin')
                print(f"  Saved input to /tmp/python_pnet_input_scale0.bin")
                print(f"  Input shape: {img_scaled_norm.shape} (H, W, C)")
                print(f"  img_data shape fed to CNN: {img_data.shape} (C, H, W)")

            # Run PNet using Pure Python CNN
            # Pass debug=True only for first scale (i==0)
            output = self._run_pnet(img_data, debug=(debug and i == 0))

            # Rest is same as ONNX version
            output = output[0].transpose(1, 2, 0)  # (H, W, 6)

            logit_not_face = output[:, :, 0]
            logit_face = output[:, :, 1]
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            # DEBUG: Save scale 0 PNet logits for comparison with C++
            if i == 0 and debug:
                # Save logits in same format as C++ (H, W) arrays
                logit_not_face.astype(np.float32).tofile('/tmp/python_pnet_logit0_scale0.bin')
                logit_face.astype(np.float32).tofile('/tmp/python_pnet_logit1_scale0.bin')
                print(f"  Saved logits to /tmp/python_pnet_logit0_scale0.bin and logit1")
                print(f"  Logit shape: {logit_not_face.shape} (H, W)")
                print(f"  Logit0 range: [{logit_not_face.min():.3f}, {logit_not_face.max():.3f}]")
                print(f"  Logit1 range: [{logit_face.min():.3f}, {logit_face.max():.3f}]")

                # Show sample values for verification
                num_detections = np.sum(prob_face > self.thresholds[0])
                print(f"  Detections at threshold {self.thresholds[0]}: {num_detections}")
                print(f"  Prob range: [{prob_face.min():.3f}, {prob_face.max():.3f}]")

            score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
            reg_map = output[:, :, 2:6]

            # Generate bboxes
            boxes = self._generate_bboxes(score_map, reg_map, scale, self.thresholds[0])

            if debug:
                print(f"\n  Scale {i+1} BEFORE NMS: {boxes.shape[0]} boxes generated")
                # Debug Scale 7 specifically
                if i == 6:  # Scale 7 (0-indexed)
                    print(f"  [Scale 7 Debug] Scale factor: {scale:.6f}")
                    print(f"  [Scale 7 Debug] Scaled image size: {ws}x{hs}")
                    # Save ALL probabilities to file for analysis
                    prob_face.astype(np.float32).tofile('/tmp/python_scale7_probs.bin')
                    print(f"  [Scale 7 Debug] Saved all probabilities to /tmp/python_scale7_probs.bin")
                    # Show probabilities near threshold
                    probs_above_thresh = prob_face[prob_face > self.thresholds[0]]
                    if len(probs_above_thresh) > 0:
                        print(f"  [Scale 7 Debug] Probabilities above {self.thresholds[0]}: {len(probs_above_thresh)}")
                        print(f"  [Scale 7 Debug] Top 5 probs: {np.sort(probs_above_thresh)[-5:][::-1]}")
                        # Show probabilities just below threshold
                        probs_close = prob_face[(prob_face > 0.55) & (prob_face <= self.thresholds[0])]
                        if len(probs_close) > 0:
                            print(f"  [Scale 7 Debug] Probs between 0.55-0.60: {len(probs_close)}")
                            print(f"  [Scale 7 Debug] Values: {np.sort(probs_close)[-5:][::-1]}")
                        # Find values very close to threshold
                        probs_very_close = prob_face[(prob_face > 0.595) & (prob_face < 0.605)]
                        if len(probs_very_close) > 0:
                            print(f"  [Scale 7 Debug] Probs within 0.005 of threshold: {len(probs_very_close)}")
                            for idx, val in enumerate(np.sort(probs_very_close)[::-1]):
                                print(f"    Value {idx+1}: {val:.10f}")
                if i == 0 and boxes.shape[0] > 0:
                    print(f"  Top 5 boxes:")
                    for j in range(min(5, boxes.shape[0])):
                        x1, y1, x2, y2, score = boxes[j, 0:5]
                        print(f"    #{j+1}: x={x1:.0f}, y={y1:.0f}, w={x2-x1:.0f}, h={y2-y1:.0f}, score={score:.6f}")

            if boxes.shape[0] > 0:
                keep = self._nms(boxes, 0.5, 'Union')
                boxes = boxes[keep]
                total_boxes.append(boxes)
                if debug and boxes.shape[0] > 0:
                    print(f"  Scale {i+1}/{len(scales)}: {boxes.shape[0]} boxes after NMS")

        if len(total_boxes) == 0:
            if debug:
                print(f"✗ PNet: No boxes detected")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.vstack(total_boxes)
        if debug:
            print(f"\nPNet: {total_boxes.shape[0]} boxes before cross-scale NMS")

        # NMS across scales
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]

        if debug:
            print(f"PNet: {total_boxes.shape[0]} boxes after cross-scale NMS")
            # Dump all boxes to file for comparison with C++
            with open('/tmp/python_pnet_all_boxes.txt', 'w') as f:
                f.write(f"Python PNet: {total_boxes.shape[0]} boxes after cross-scale NMS\n")
                for i in range(min(total_boxes.shape[0], 110)):  # Dump all boxes
                    x1, y1, x2, y2, score = total_boxes[i, 0:5]
                    f.write(f"  Box {i}: x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f} w={x2-x1:.1f} h={y2-y1:.1f} score={score:.6f}\n")

        if total_boxes.shape[0] == 0:
            if debug:
                print(f"✗ PNet: No boxes after NMS")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply PNet bbox regression (matching C++ line 1009)
        # This is the CRITICAL missing step!
        if debug:
            print(f"\nPNet: Before bbox regression:")
            for i in range(min(5, total_boxes.shape[0])):
                x1, y1, x2, y2 = total_boxes[i, 0:4]
                w = x2 - x1
                h = y2 - y1
                dx1, dy1, dx2, dy2 = total_boxes[i, 5:9]
                print(f"  Box {i+1}: ({x1:.1f}, {y1:.1f}) {w:.1f}×{h:.1f}, reg=[{dx1:.3f}, {dy1:.3f}, {dx2:.3f}, {dy2:.3f}]")

        total_boxes = self._apply_bbox_regression(total_boxes)

        if debug:
            print(f"\nPNet: After bbox regression:")
            for i in range(min(5, total_boxes.shape[0])):
                x1, y1, x2, y2 = total_boxes[i, 0:4]
                w = x2 - x1
                h = y2 - y1
                print(f"  Box {i+1}: ({x1:.1f}, {y1:.1f}) {w:.1f}×{h:.1f}")

        # ===== Stage 2: RNet =====
        if debug:
            print(f"\n--- STAGE 2: RNet ---")
            print(f"RNet input: {total_boxes.shape[0]} boxes from PNet")

        total_boxes = self._square_bbox(total_boxes)

        rnet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            # Match C++ RNet cropping with +1 padding (FaceDetectorMTCNN.cpp:1053-1076)
            bbox_x = total_boxes[i, 0]
            bbox_y = total_boxes[i, 1]
            bbox_w = total_boxes[i, 2] - total_boxes[i, 0]
            bbox_h = total_boxes[i, 3] - total_boxes[i, 1]

            # C++: width_target = width + 1, height_target = height + 1
            width_target = int(bbox_w + 1)
            height_target = int(bbox_h + 1)

            # C++: start_x_in = max(x - 1, 0), start_y_in = max(y - 1, 0)
            start_x_in = max(int(bbox_x - 1), 0)
            start_y_in = max(int(bbox_y - 1), 0)
            end_x_in = min(int(bbox_x + width_target - 1), img_w)
            end_y_in = min(int(bbox_y + height_target - 1), img_h)

            # Output coordinates in the zero-padded image
            start_x_out = max(int(-bbox_x + 1), 0)
            start_y_out = max(int(-bbox_y + 1), 0)
            end_x_out = min(int(width_target - (bbox_x + bbox_w - img_w)), width_target)
            end_y_out = min(int(height_target - (bbox_y + bbox_h - img_h)), height_target)

            # Create zero-padded temp image
            tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)

            # Copy the valid region
            tmp[start_y_out:end_y_out, start_x_out:end_x_out] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            # Resize and preprocess
            face = cv2.resize(tmp, (24, 24))

            # DEBUG: Save first crop for comparison with C++
            if i == 0 and debug:
                print(f"\n  [PYTHON RNet Crop Debug for Box 0]")
                print(f"    Original bbox: x={bbox_x:.0f}, y={bbox_y:.0f}, w={bbox_w:.0f}, h={bbox_h:.0f}")
                print(f"    Padded size: {width_target}x{height_target}")
                print(f"    Input region: [{start_y_in}:{end_y_in}, {start_x_in}:{end_x_in}]")
                print(f"    Output region: [{start_y_out}:{end_y_out}, {start_x_out}:{end_x_out}]")
                print(f"    Tmp shape: {tmp.shape}")

                # Save the preprocessed crop
                preprocessed = self._preprocess(face)
                print(f"    Sample preprocessed pixels (after (x-127.5)/127.5):")
                print(f"      [0,0]: {preprocessed[0:3, 0, 0]}")
                print(f"      [0,1]: {preprocessed[0:3, 0, 1]}")

            # RNet: Flip BGR to RGB (matching C++ line 147-155 inside Inference)
            rnet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
            valid_indices.append(i)

        if len(rnet_input) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        # Run RNet
        rnet_outputs = []
        for face_data in rnet_input:
            output = self._run_rnet(face_data)
            rnet_outputs.append(output)

        output = np.vstack(rnet_outputs)

        # Calculate scores
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        if debug:
            print(f"RNet: Tested {len(scores)} faces")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Scores > {self.thresholds[1]}: {(scores > self.thresholds[1]).sum()}")
            if len(scores) > 0:
                top_5_idx = np.argsort(scores)[-5:][::-1]
                print(f"  Top 5 scores: {scores[top_5_idx]}")

            # DEBUG: Dump ALL RNet scores for comparison with C++
            with open('/tmp/python_rnet_scores.txt', 'w') as f:
                f.write(f"Python RNet: {len(scores)} boxes with CNN scores (BEFORE threshold filter)\n")
                for i in range(len(scores)):
                    f.write(f"  Box {i}: x1={total_boxes[i,0]:.3f} y1={total_boxes[i,1]:.3f} ")
                    f.write(f"x2={total_boxes[i,2]:.3f} y2={total_boxes[i,3]:.3f} ")
                    f.write(f"score={scores[i]:.6f}\n")

        # Filter by threshold
        keep = scores > self.thresholds[1]

        if not keep.any():
            if debug:
                print(f"✗ RNet: No faces passed threshold {self.thresholds[1]}")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]

        # NMS
        if debug:
            print(f"  Before RNet NMS: {total_boxes.shape[0]} boxes")

        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = reg[keep]

        if debug:
            print(f"  After RNet NMS: {total_boxes.shape[0]} boxes")

        if total_boxes.shape[0] == 0:
            if debug:
                print(f"✗ RNet: No boxes after NMS")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply regression (FIXED to match C++ FaceDetectorMTCNN.cpp:832-836)
        if debug and total_boxes.shape[0] > 0:
            print(f"\n  [BEFORE RNet regression]")
            print(f"    Box 0: x={total_boxes[0,0]:.0f}, y={total_boxes[0,1]:.0f}, "
                  f"w={total_boxes[0,2] - total_boxes[0,0]:.0f}, h={total_boxes[0,3] - total_boxes[0,1]:.0f}")
            print(f"    x2={total_boxes[0,2]:.0f}, y2={total_boxes[0,3]:.0f}")
            print(f"    Reg: dx1={reg[0,0]:.4f}, dy1={reg[0,1]:.4f}, dx2={reg[0,2]:.4f}, dy2={reg[0,3]:.4f}")

            # Also write to file for comparison with C++
            with open('/tmp/python_before_rnet_regression.txt', 'w') as f:
                f.write("Python Boxes BEFORE RNet regression (after RNet NMS):\n")
                f.write(f"Total boxes: {total_boxes.shape[0]}\n")
                for i in range(min(5, total_boxes.shape[0])):
                    w = total_boxes[i,2] - total_boxes[i,0]
                    h = total_boxes[i,3] - total_boxes[i,1]
                    f.write(f"  Box {i}: x={total_boxes[i,0]:.3f} y={total_boxes[i,1]:.3f} "
                           f"w={w:.3f} h={h:.3f}\n")
                    f.write(f"    Regression offsets: dx1={reg[i,0]:.6f} dy1={reg[i,1]:.6f} "
                           f"dx2={reg[i,2]:.6f} dy2={reg[i,3]:.6f}\n")

        w = total_boxes[:, 2] - total_boxes[:, 0]
        h = total_boxes[:, 3] - total_boxes[:, 1]
        x1 = total_boxes[:, 0].copy()  # CRITICAL: Must be a copy, not a view!
        y1 = total_boxes[:, 1].copy()  # CRITICAL: Must be a copy, not a view!
        # C++: new_min_x = x + dx * w, new_max_x = x + w + w * dx2
        total_boxes[:, 0] = x1 + reg[:, 0] * w
        total_boxes[:, 1] = y1 + reg[:, 1] * h
        total_boxes[:, 2] = x1 + w + w * reg[:, 2]  # NOT x2 + dx2*w!
        total_boxes[:, 3] = y1 + h + h * reg[:, 3]  # NOT y2 + dy2*h!
        total_boxes[:, 4] = scores

        if debug and total_boxes.shape[0] > 0:
            print(f"  [AFTER RNet regression]")
            print(f"    Box 0: x={total_boxes[0,0]:.0f}, y={total_boxes[0,1]:.0f}, "
                  f"w={total_boxes[0,2] - total_boxes[0,0]:.0f}, h={total_boxes[0,3] - total_boxes[0,1]:.0f}")
            print(f"    x2={total_boxes[0,2]:.0f}, y2={total_boxes[0,3]:.0f}")

        # ===== Stage 3: ONet =====
        if debug:
            print(f"\n--- STAGE 3: ONet ---")
            print(f"ONet input: {total_boxes.shape[0]} boxes from RNet (BEFORE square)")
            if total_boxes.shape[0] > 0:
                for i in range(min(5, total_boxes.shape[0])):
                    print(f"  Box {i}: x={total_boxes[i,0]:.0f}, y={total_boxes[i,1]:.0f}, "
                          f"w={total_boxes[i,2] - total_boxes[i,0]:.0f}, h={total_boxes[i,3] - total_boxes[i,1]:.0f}")
                    print(f"    x2={total_boxes[i,2]:.0f}, y2={total_boxes[i,3]:.0f}")

            # DEBUG: Write RNet output boxes to file for comparison with C++
            with open('/tmp/python_rnet_output.txt', 'w') as f:
                f.write(f"Python RNet: {total_boxes.shape[0]} boxes OUTPUT from RNet (BEFORE square for ONet)\n")
                for i in range(total_boxes.shape[0]):
                    w = total_boxes[i,2] - total_boxes[i,0]
                    h = total_boxes[i,3] - total_boxes[i,1]
                    f.write(f"  Box {i}: x1={total_boxes[i,0]:.3f} y1={total_boxes[i,1]:.3f} "
                           f"x2={total_boxes[i,2]:.3f} y2={total_boxes[i,3]:.3f} "
                           f"w={w:.3f} h={h:.3f} score={total_boxes[i,4]:.6f}\n")

        total_boxes = self._square_bbox(total_boxes)

        if debug and total_boxes.shape[0] > 0:
            print(f"\nAfter RNet square (input to ONet):")
            for i in range(min(5, total_boxes.shape[0])):
                print(f"  Box {i}: x={total_boxes[i,0]:.0f}, y={total_boxes[i,1]:.0f}, "
                      f"w={total_boxes[i,2] - total_boxes[i,0]:.0f}, h={total_boxes[i,3] - total_boxes[i,1]:.0f}")
                print(f"    x2={total_boxes[i,2]:.0f}, y2={total_boxes[i,3]:.0f}")
        x1, y1, x2, y2, dx1, dy1, dx2, dy2 = self._pad_bbox(total_boxes, img_h, img_w)

        onet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            # Simplified cropping (no zero-padding)
            x1 = int(max(0, total_boxes[i, 0]))
            y1 = int(max(0, total_boxes[i, 1]))
            x2 = int(min(img_w, total_boxes[i, 2]))
            y2 = int(min(img_h, total_boxes[i, 3]))

            if x2 <= x1 or y2 <= y1:
                continue

            face = img_float[y1:y2, x1:x2]
            face = cv2.resize(face, (48, 48))
            # ONet: Flip BGR to RGB (matching C++ line 147-155 inside Inference)
            onet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
            valid_indices.append(i)

        if len(onet_input) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        # Run ONet
        onet_outputs = []
        for face_data in onet_input:
            output = self._run_onet(face_data)
            onet_outputs.append(output)

        output = np.vstack(onet_outputs)

        # Calculate scores
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        if debug:
            print(f"ONet: Tested {len(scores)} faces")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Scores > {self.thresholds[2]}: {(scores > self.thresholds[2]).sum()}")
            if len(scores) > 0:
                print(f"  All scores: {scores}")

        # Filter by threshold
        keep = scores > self.thresholds[2]

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]
        landmarks = output[keep, 6:16]

        if debug:
            print(f"  After ONet threshold filter: {total_boxes.shape[0]} boxes")

        if total_boxes.shape[0] == 0:
            if debug:
                print(f"✗ ONet: No faces passed threshold {self.thresholds[2]}")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        if debug and total_boxes.shape[0] > 0:
            print(f"\n  [BEFORE ONet regression]")
            print(f"    Box 0: x1={total_boxes[0,0]:.2f}, y1={total_boxes[0,1]:.2f}, "
                  f"x2={total_boxes[0,2]:.2f}, y2={total_boxes[0,3]:.2f}")
            print(f"    Width={total_boxes[0,2] - total_boxes[0,0]:.2f}, "
                  f"Height={total_boxes[0,3] - total_boxes[0,1]:.2f}")
            print(f"    Reg offsets: dx1={reg[0,0]:.4f}, dy1={reg[0,1]:.4f}, "
                  f"dx2={reg[0,2]:.4f}, dy2={reg[0,3]:.4f}")

        # Apply bbox regression (with +1 for ONet)
        # FIXED to match C++ FaceDetectorMTCNN.cpp:832-836
        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        x1 = total_boxes[:, 0].copy()  # CRITICAL: Must be a copy, not a view!
        y1 = total_boxes[:, 1].copy()  # CRITICAL: Must be a copy, not a view!

        if debug and total_boxes.shape[0] > 0:
            print(f"    Width+1={w[0]:.2f}, Height+1={h[0]:.2f}")
            print(f"    Computing: x1_new = {x1[0]:.2f} + {reg[0,0]:.4f} * {w[0]:.2f} = {x1[0] + reg[0,0] * w[0]:.2f}")
            print(f"    Computing: x2_new = {x1[0]:.2f} + {w[0]:.2f} + {w[0]:.2f} * {reg[0,2]:.4f} = {x1[0] + w[0] + w[0] * reg[0,2]:.2f}")

        # C++: new_min_x = x + dx * w, new_max_x = x + w + w * dx2
        total_boxes[:, 0] = x1 + reg[:, 0] * w
        total_boxes[:, 1] = y1 + reg[:, 1] * h
        total_boxes[:, 2] = x1 + w + w * reg[:, 2]  # NOT x2 + dx2*w!
        total_boxes[:, 3] = y1 + h + h * reg[:, 3]  # NOT y2 + dy2*h!
        total_boxes[:, 4] = scores

        if debug and total_boxes.shape[0] > 0:
            print(f"  [AFTER ONet regression]")
            print(f"    Box 0: x1={total_boxes[0,0]:.2f}, y1={total_boxes[0,1]:.2f}, "
                  f"x2={total_boxes[0,2]:.2f}, y2={total_boxes[0,3]:.2f}")
            print(f"    Width={total_boxes[0,2] - total_boxes[0,0]:.2f}, "
                  f"Height={total_boxes[0,3] - total_boxes[0,1]:.2f}")

        # Denormalize landmarks
        for i in range(5):
            landmarks[:, 2*i] = total_boxes[:, 0] + landmarks[:, 2*i] * w
            landmarks[:, 2*i+1] = total_boxes[:, 1] + landmarks[:, 2*i+1] * h

        landmarks = landmarks.reshape(-1, 5, 2)

        # DEBUG: Dump boxes BEFORE NMS (after threshold filter and regression)
        if debug:
            with open('/tmp/python_onet_before_nms.txt', 'w') as f:
                f.write(f"Python ONet: {total_boxes.shape[0]} boxes BEFORE NMS (after threshold + regression)\n")
                for i in range(total_boxes.shape[0]):
                    f.write(f"  Box {i}: x1={total_boxes[i,0]:.3f} y1={total_boxes[i,1]:.3f} "
                           f"x2={total_boxes[i,2]:.3f} y2={total_boxes[i,3]:.3f} "
                           f"w={total_boxes[i,2]-total_boxes[i,0]:.3f} "
                           f"h={total_boxes[i,3]-total_boxes[i,1]:.3f} "
                           f"score={total_boxes[i,4]:.6f}\n")

        # Final NMS
        keep = self._nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[keep]
        landmarks = landmarks[keep]

        if debug and total_boxes.shape[0] > 0:
            print(f"\n  [BEFORE final calibration]")
            print(f"    Box 0: x1={total_boxes[0,0]:.2f}, y1={total_boxes[0,1]:.2f}, "
                  f"x2={total_boxes[0,2]:.2f}, y2={total_boxes[0,3]:.2f}")
            print(f"    Width={total_boxes[0,2] - total_boxes[0,0]:.2f}, "
                  f"Height={total_boxes[0,3] - total_boxes[0,1]:.2f}")

            # Dump ONet boxes to file for comparison with C++
            with open('/tmp/python_onet_boxes.txt', 'w') as f:
                f.write(f"Python ONet: {total_boxes.shape[0]} boxes after final NMS (BEFORE bbox correction)\n")
                for i in range(total_boxes.shape[0]):
                    w = total_boxes[i, 2] - total_boxes[i, 0]
                    h = total_boxes[i, 3] - total_boxes[i, 1]
                    f.write(f"  Box {i}: x={total_boxes[i,0]:.3f} y={total_boxes[i,1]:.3f} "
                           f"w={w:.3f} h={h:.3f} score={total_boxes[i,4]:.6f}\n")

        # Apply C++ bbox correction
        for k in range(total_boxes.shape[0]):
            w = total_boxes[k, 2] - total_boxes[k, 0]
            h = total_boxes[k, 3] - total_boxes[k, 1]
            new_x1 = total_boxes[k, 0] + w * -0.0075
            new_y1 = total_boxes[k, 1] + h * 0.2459
            new_width = w * 1.0323
            new_height = h * 0.7751
            total_boxes[k, 0] = new_x1
            total_boxes[k, 1] = new_y1
            total_boxes[k, 2] = new_x1 + new_width
            total_boxes[k, 3] = new_y1 + new_height

        if debug and total_boxes.shape[0] > 0:
            print(f"  [AFTER final calibration]")
            print(f"    Box 0: x1={total_boxes[0,0]:.2f}, y1={total_boxes[0,1]:.2f}, "
                  f"x2={total_boxes[0,2]:.2f}, y2={total_boxes[0,3]:.2f}")
            print(f"    Width={total_boxes[0,2] - total_boxes[0,0]:.2f}, "
                  f"Height={total_boxes[0,3] - total_boxes[0,1]:.2f}")

        # Convert to (x, y, width, height) format
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        return bboxes, landmarks

    # Helper methods

    def _generate_bboxes(self, score_map, reg_map, scale, threshold):
        """Generate bounding boxes from PNet output.

        Matches C++ generate_bounding_boxes (FaceDetectorMTCNN.cpp:756-792).
        CRITICAL: C++ uses int() truncation, not rounding!
        CRITICAL: C++ formula is (stride*x + cellsize), NOT (stride*x + 1 + cellsize)!
        """
        stride = 2
        cellsize = 12

        # CRITICAL FIX: Must use >= to match C++ (line 773 of FaceDetectorMTCNN.cpp)
        t_index = np.where(score_map[:, :, 1] >= threshold)

        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg_map[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = score_map[t_index[0], t_index[1], 1]

        # CRITICAL FIX: Match C++ exactly!
        # C++: min_x = int((stride * x + 1) / scale)
        # C++: max_x = int((stride * x + face_support) / scale)
        # Must use floor+int, NOT round!
        # Must NOT add extra +1 to max calculation!
        boundingbox = np.vstack([
            np.floor((stride * t_index[1] + 1) / scale).astype(int),
            np.floor((stride * t_index[0] + 1) / scale).astype(int),
            np.floor((stride * t_index[1] + cellsize) / scale).astype(int),
            np.floor((stride * t_index[0] + cellsize) / scale).astype(int),
            score,
            reg
        ])

        return boundingbox.T

    def _nms(self, boxes, threshold, method):
        """
        Non-Maximum Suppression.

        CRITICAL: Matches C++ OpenCV implementation (no +1 for area calculation).
        C++ uses geometric distance (rect.area() = width * height), NOT pixel counting.
        """
        if boxes.shape[0] == 0:
            return np.array([])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        # CRITICAL: NO +1 to match C++ OpenCV rect.area()
        # C++ uses geometric distance, not boundary-inclusive pixel counting
        area = (x2 - x1) * (y2 - y1)
        sorted_s = np.argsort(s)

        pick = []
        while sorted_s.shape[0] > 0:
            i = sorted_s[-1]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[sorted_s[:-1]])
            yy1 = np.maximum(y1[i], y1[sorted_s[:-1]])
            xx2 = np.minimum(x2[i], x2[sorted_s[:-1]])
            yy2 = np.minimum(y2[i], y2[sorted_s[:-1]])

            # CRITICAL: NO +1 to match C++ intersection calculation
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            inter = w * h

            if method == 'Min':
                o = inter / np.minimum(area[i], area[sorted_s[:-1]])
            else:
                o = inter / (area[i] + area[sorted_s[:-1]] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)[0]]

        return pick

    def _apply_bbox_regression(self, bboxes, add1=False):
        """
        Apply bbox regression corrections.

        Matches C++ apply_correction() function (FaceDetectorMTCNN.cpp:819-841).

        CRITICAL: C++ uses different logic for ONet vs PNet/RNet:
        - PNet/RNet: apply_correction(boxes, corrections, add1=false)
        - ONet:      apply_correction(boxes, corrections, add1=true)

        The add1 parameter adds 1 to width/height before applying regression!

        Args:
            bboxes: Array with shape (N, 9) where:
                - Columns 0-3: x1, y1, x2, y2 (current bbox coordinates)
                - Column 4: score
                - Columns 5-8: dx1, dy1, dx2, dy2 (regression offsets)
            add1: If True, adds 1 to w and h before regression (ONet only!)

        Returns:
            Array with shape (N, 9) with updated bbox coordinates
        """
        result = bboxes.copy()

        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, 0:4]
            dx1, dy1, dx2, dy2 = bboxes[i, 5:9]

            # Current box dimensions
            w = x2 - x1
            h = y2 - y1

            # CRITICAL FIX: Add 1 to dimensions when add1=True (ONet stage)
            # C++ code (lines 826-830):
            #   if (add1) {
            #       curr_box.width++;
            #       curr_box.height++;
            #   }
            if add1:
                w = w + 1
                h = h + 1

            # Apply regression (matching C++ lines 832-836 EXACTLY!)
            # C++: new_min_x = curr_box.x + corrections[i].x * curr_box.width;
            # C++: new_max_x = curr_box.x + curr_box.width + curr_box.width * corrections[i].width;
            # CRITICAL: C++ uses starting x for BOTH min and max!
            new_x1 = x1 + dx1 * w
            new_y1 = y1 + dy1 * h
            new_x2 = x1 + w + w * dx2  # NOT x2 + dx2 * w!
            new_y2 = y1 + h + h * dy2  # NOT y2 + dy2 * h!

            result[i, 0] = new_x1
            result[i, 1] = new_y1
            result[i, 2] = new_x2
            result[i, 3] = new_y2

        return result

    def _square_bbox(self, bboxes):
        """Convert bounding boxes to squares.

        CRITICAL: Matches C++ rectify() which casts to int!
        C++ code (FaceDetectorMTCNN.cpp:797-817):
            total_bboxes[i].x = (int)new_min_x;
            total_bboxes[i].y = (int)new_min_y;
            total_bboxes[i].width = (int)max_side;
            total_bboxes[i].height = (int)max_side;
        """
        square_bboxes = bboxes.copy()
        h = bboxes[:, 3] - bboxes[:, 1]
        w = bboxes[:, 2] - bboxes[:, 0]
        max_side = np.maximum(h, w)

        # CRITICAL FIX: Use np.trunc() to match C++ (int) casting!
        # C++ (int) truncates toward zero, NOT floor!
        new_x1 = np.trunc(bboxes[:, 0] + w * 0.5 - max_side * 0.5).astype(int)
        new_y1 = np.trunc(bboxes[:, 1] + h * 0.5 - max_side * 0.5).astype(int)
        max_side_int = np.trunc(max_side).astype(int)

        square_bboxes[:, 0] = new_x1
        square_bboxes[:, 1] = new_y1
        square_bboxes[:, 2] = new_x1 + max_side_int
        square_bboxes[:, 3] = new_y1 + max_side_int
        return square_bboxes

    def _pad_bbox(self, bboxes, img_h, img_w):
        """Calculate padding for bboxes (not used in simplified version but kept for compatibility)."""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        dx1 = np.zeros_like(x1)
        dy1 = np.zeros_like(y1)
        dx2 = np.zeros_like(x2)
        dy2 = np.zeros_like(y2)

        return x1, y1, x2, y2, dx1, dy1, dx2, dy2
