#!/usr/bin/env python3
"""
ONNX Runtime MTCNN Face Detector
Optimized implementation using ONNX Runtime with hardware acceleration.

Performance targets:
- Apple Silicon: Use CoreMLExecutionProvider for Neural Engine acceleration
- NVIDIA CUDA: Use TensorrtExecutionProvider + CUDAExecutionProvider
- CPU fallback: Use CPUExecutionProvider

Expected speedup: 8-10x vs pure Python implementation
"""

import numpy as np
import cv2
import onnxruntime as ort
import os
from pathlib import Path
import time


class ONNXMTCNNDetector:
    """
    MTCNN Face Detector using ONNX Runtime for hardware acceleration.

    Inherits detection logic from pure_python_mtcnn_v2.py but uses ONNX Runtime
    for inference to achieve 8-10x speedup.
    """

    def __init__(self, model_dir=None, use_coreml=True, use_cuda=True):
        """
        Initialize ONNX MTCNN with hardware-accelerated execution providers.

        Args:
            model_dir: Directory containing pnet.onnx, rnet.onnx, onet.onnx
            use_coreml: Enable CoreML provider for Apple Silicon (default: True)
            use_cuda: Enable CUDA/TensorRT providers for NVIDIA GPUs (default: True)
        """
        if model_dir is None:
            # Use archived ONNX models
            model_dir = Path(__file__).parent / "archive" / "onnx_implementations" / "cpp_mtcnn_onnx"
        else:
            model_dir = Path(model_dir)

        print(f"Loading ONNX MTCNN...")
        print(f"Model directory: {model_dir}")

        # Configure execution providers based on platform
        providers = self._get_execution_providers(use_coreml, use_cuda)
        print(f"Execution providers: {providers}")

        # Load ONNX models
        pnet_path = str(model_dir / "pnet.onnx")
        rnet_path = str(model_dir / "rnet.onnx")
        onet_path = str(model_dir / "onet.onnx")

        # Create inference sessions with optimized providers
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.pnet_session = ort.InferenceSession(pnet_path, sess_options, providers=providers)
        self.rnet_session = ort.InferenceSession(rnet_path, sess_options, providers=providers)
        self.onet_session = ort.InferenceSession(onet_path, sess_options, providers=providers)

        # Print actual providers being used
        print(f"PNet using: {self.pnet_session.get_providers()}")
        print(f"RNet using: {self.rnet_session.get_providers()}")
        print(f"ONet using: {self.onet_session.get_providers()}")

        # MTCNN parameters (matching pure Python implementation)
        self.thresholds = [0.6, 0.7, 0.7]  # PNet, RNet, ONet
        self.min_face_size = 60
        self.factor = 0.709

        print(f"✓ ONNX MTCNN loaded with hardware acceleration!")

    def _get_execution_providers(self, use_coreml, use_cuda):
        """
        Get optimal execution providers based on platform and preferences.

        Priority order:
        1. CoreMLExecutionProvider (Apple Silicon Neural Engine)
        2. TensorrtExecutionProvider (NVIDIA GPUs)
        3. CUDAExecutionProvider (NVIDIA GPUs)
        4. CPUExecutionProvider (fallback)
        """
        providers = []
        available_providers = ort.get_available_providers()

        # Apple Silicon: CoreML provider for Neural Engine
        if use_coreml and 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
            print("  ✓ CoreML provider enabled (Apple Neural Engine)")

        # NVIDIA: TensorRT + CUDA providers
        if use_cuda:
            if 'TensorrtExecutionProvider' in available_providers:
                providers.append(('TensorrtExecutionProvider', {
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache'
                }))
                print("  ✓ TensorRT provider enabled (NVIDIA GPU with FP16)")

            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                print("  ✓ CUDA provider enabled (NVIDIA GPU)")

        # Always include CPU as fallback
        providers.append('CPUExecutionProvider')

        return providers

    def _preprocess(self, img: np.ndarray, flip_bgr_to_rgb: bool = True) -> np.ndarray:
        """
        Preprocess image for ONNX MTCNN.

        Args:
            img: BGR image (H, W, 3)
            flip_bgr_to_rgb: If True, convert BGR to RGB (default: True)

        Returns:
            Normalized image tensor (1, 3, H, W) in RGB order
        """
        # Normalize to [-1, 1]
        img_norm = (img.astype(np.float32) - 127.5) * 0.0078125

        # Transpose to (C, H, W)
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # Convert BGR to RGB
        if flip_bgr_to_rgb:
            img_chw = img_chw[[2, 1, 0], :, :]

        # Add batch dimension: (1, 3, H, W)
        img_batch = img_chw[np.newaxis, :, :, :]

        return img_batch

    def _run_pnet(self, img_data):
        """Run PNet using ONNX Runtime."""
        input_name = self.pnet_session.get_inputs()[0].name
        output = self.pnet_session.run(None, {input_name: img_data})[0]
        return output

    def _run_rnet(self, img_data):
        """Run RNet using ONNX Runtime."""
        input_name = self.rnet_session.get_inputs()[0].name
        output = self.rnet_session.run(None, {input_name: img_data})[0]
        return output

    def _run_onet(self, img_data):
        """Run ONet using ONNX Runtime."""
        input_name = self.onet_session.get_inputs()[0].name
        output = self.onet_session.run(None, {input_name: img_data})[0]
        return output

    def _generate_bboxes(self, confidence_map, reg_map, scale, threshold):
        """
        Generate bounding boxes from PNet output.

        Args:
            confidence_map: (H, W) confidence scores
            reg_map: (4, H, W) bbox regression
            scale: Image scale factor
            threshold: Confidence threshold

        Returns:
            Bounding boxes [x1, y1, x2, y2, score]
        """
        stride = 2
        cellsize = 12

        # Find locations where confidence > threshold
        t_index = np.where(confidence_map > threshold)

        # No detections
        if t_index[0].size == 0:
            return np.array([])

        # Get bbox offsets
        dx1, dy1, dx2, dy2 = reg_map[:, t_index[0], t_index[1]]

        # Calculate bbox coordinates
        reg = np.array([dx1, dy1, dx2, dy2])
        score = confidence_map[t_index[0], t_index[1]]

        # Map to original image
        boundingbox = np.vstack([
            np.round((stride * t_index[1] + 1) / scale),
            np.round((stride * t_index[0] + 1) / scale),
            np.round((stride * t_index[1] + 1 + cellsize) / scale),
            np.round((stride * t_index[0] + 1 + cellsize) / scale),
            score,
            reg
        ])

        return boundingbox.T

    def _nms(self, boxes, threshold, method='Union'):
        """
        Non-Maximum Suppression.

        Args:
            boxes: (N, 5+) array of [x1, y1, x2, y2, score, ...]
            threshold: IoU threshold
            method: 'Union' or 'Min'

        Returns:
            Indices of boxes to keep
        """
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            if method == 'Min':
                ovr = inter / np.minimum(areas[i], areas[order[1:]])
            else:
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def _calibrate_box(self, bbox, reg):
        """Apply bbox regression offsets."""
        w = bbox[:, 2] - bbox[:, 0] + 1
        h = bbox[:, 3] - bbox[:, 1] + 1

        bbox[:, 0] = bbox[:, 0] + reg[:, 0] * w
        bbox[:, 1] = bbox[:, 1] + reg[:, 1] * h
        bbox[:, 2] = bbox[:, 2] + reg[:, 2] * w
        bbox[:, 3] = bbox[:, 3] + reg[:, 3] * h

        return bbox

    def _convert_to_square(self, bbox):
        """Convert bboxes to square."""
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1]
        w = bbox[:, 2] - bbox[:, 0]
        max_side = np.maximum(h, w)

        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side
        square_bbox[:, 3] = square_bbox[:, 1] + max_side

        return square_bbox

    def _pad(self, bboxes, w, h):
        """Calculate padding for bboxes that extend beyond image boundaries."""
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)

        dx = np.zeros((len(bboxes),))
        dy = np.zeros((len(bboxes),))
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = bboxes[:, 0].copy().astype(np.int32)
        y = bboxes[:, 1].copy().astype(np.int32)
        ex = bboxes[:, 2].copy().astype(np.int32)
        ey = bboxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx[tmp] = (-ex[tmp] + w + tmpw[tmp]).astype(np.int32)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy[tmp] = (-ey[tmp] + h + tmph[tmp]).astype(np.int32)
        ey[tmp] = h

        tmp = np.where(x < 0)
        dx[tmp] = 0 - x[tmp]
        x[tmp] = 0

        tmp = np.where(y < 0)
        dy[tmp] = 0 - y[tmp]
        y[tmp] = 0

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    def detect(self, img, debug=False):
        """
        Detect faces in image using ONNX-accelerated MTCNN.

        Args:
            img: BGR image (H, W, 3)
            debug: Enable debug output

        Returns:
            bounding_boxes: (N, 5) array of [x, y, w, h, confidence]
        """
        if debug:
            print("\n" + "="*80)
            print("ONNX MTCNN DETECTION")
            print("="*80)

        h, w, _ = img.shape
        if debug:
            print(f"Input image size: {w}x{h}")

        # Stage 1: PNet (Proposal Network)
        if debug:
            print("\n[Stage 1: PNet]")

        # Build image pyramid
        min_size = self.min_face_size
        m = 12.0 / min_size
        min_l = np.amin([h, w])
        min_l = min_l * m

        scales = []
        factor_count = 0
        while min_l >= 12:
            scales.append(m * np.power(self.factor, factor_count))
            min_l = min_l * self.factor
            factor_count += 1

        if debug:
            print(f"  Image pyramid scales: {scales}")

        total_boxes = np.empty((0, 9))

        for scale in scales:
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))

            im_resized = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)
            im_data = self._preprocess(im_resized, flip_bgr_to_rgb=True)

            # Run PNet
            output = self._run_pnet(im_data)

            # Extract outputs: (1, 6, H, W) -> prob (H, W), reg (4, H, W)
            boxes = self._generate_bboxes(
                output[0, 1, :, :],  # Confidence for face class
                output[0, 2:6, :, :],  # Bbox regression
                scale,
                self.thresholds[0]
            )

            if boxes.size != 0:
                # NMS within scale
                pick = self._nms(boxes, 0.5, 'Union')
                if boxes.size != 0 and pick.size != 0:
                    boxes = boxes[pick, :]
                    total_boxes = np.append(total_boxes, boxes, axis=0)

        if debug:
            print(f"  Total proposals after PNet: {len(total_boxes)}")

        if total_boxes.size == 0:
            if debug:
                print("  No faces detected by PNet")
            return np.array([])

        # NMS across scales
        pick = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]

        # Calibrate bboxes
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        total_boxes = self._calibrate_box(total_boxes, total_boxes[:, 5:9])
        total_boxes = self._convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        if debug:
            print(f"  Proposals after NMS and calibration: {len(total_boxes)}")

        # Stage 2: RNet (Refinement Network)
        if debug:
            print("\n[Stage 2: RNet]")

        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self._pad(total_boxes, w, h)
        num_boxes = total_boxes.shape[0]

        if num_boxes > 0:
            # Crop and resize patches to 24x24
            tempimg = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
            for k in range(num_boxes):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3), dtype=np.uint8)
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1, :] = \
                    img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1, :]

                if tmp.shape[0] > 0 and tmp.shape[1] > 0:
                    img_resized = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
                    tempimg[k, :, :, :] = self._preprocess(img_resized, flip_bgr_to_rgb=True)[0]

            # Run RNet
            output = self._run_rnet(tempimg)

            # Filter by threshold
            passed_t = np.where(output[:, 1] > self.thresholds[1])[0]
            total_boxes = total_boxes[passed_t, :]
            total_boxes[:, 4] = output[passed_t, 1].reshape((-1,))
            reg = output[passed_t, 2:6]

            if debug:
                print(f"  Boxes passing RNet threshold: {len(total_boxes)}")

            if total_boxes.size == 0:
                if debug:
                    print("  No faces passed RNet")
                return np.array([])

            # NMS
            pick = self._nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = self._calibrate_box(total_boxes, reg[pick, :])
            total_boxes = self._convert_to_square(total_boxes)
            total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

            if debug:
                print(f"  Boxes after RNet NMS: {len(total_boxes)}")

        # Stage 3: ONet (Output Network)
        if debug:
            print("\n[Stage 3: ONet]")

        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self._pad(total_boxes, w, h)
        num_boxes = total_boxes.shape[0]

        if num_boxes > 0:
            # Crop and resize patches to 48x48
            tempimg = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
            for k in range(num_boxes):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3), dtype=np.uint8)
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1, :] = \
                    img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1, :]

                if tmp.shape[0] > 0 and tmp.shape[1] > 0:
                    img_resized = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
                    tempimg[k, :, :, :] = self._preprocess(img_resized, flip_bgr_to_rgb=True)[0]

            # Run ONet
            output = self._run_onet(tempimg)

            # Filter by threshold
            passed_t = np.where(output[:, 1] > self.thresholds[2])[0]
            total_boxes = total_boxes[passed_t, :]
            total_boxes[:, 4] = output[passed_t, 1].reshape((-1,))
            reg = output[passed_t, 2:6]

            if debug:
                print(f"  Boxes passing ONet threshold: {len(total_boxes)}")

            if total_boxes.size == 0:
                if debug:
                    print("  No faces passed ONet")
                return np.array([])

            # Calibrate and NMS
            total_boxes = self._calibrate_box(total_boxes, reg)
            pick = self._nms(total_boxes, 0.7, 'Min')
            total_boxes = total_boxes[pick, :]

            if debug:
                print(f"  Final boxes after ONet NMS: {len(total_boxes)}")

        # Apply final calibration (matching pure Python implementation)
        if debug:
            print("\n[Applying final calibration]")

        for k in range(total_boxes.shape[0]):
            w_box = total_boxes[k, 2] - total_boxes[k, 0]
            h_box = total_boxes[k, 3] - total_boxes[k, 1]
            new_x1 = total_boxes[k, 0] + w_box * -0.0075
            new_y1 = total_boxes[k, 1] + h_box * 0.2459
            new_width = w_box * 1.0323
            new_height = h_box * 0.7751
            total_boxes[k, 0] = new_x1
            total_boxes[k, 1] = new_y1
            total_boxes[k, 2] = new_x1 + new_width
            total_boxes[k, 3] = new_y1 + new_height

        # Convert to [x, y, w, h, confidence] format
        result = np.zeros((len(total_boxes), 5))
        result[:, 0] = total_boxes[:, 0]  # x
        result[:, 1] = total_boxes[:, 1]  # y
        result[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]  # width
        result[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]  # height
        result[:, 4] = total_boxes[:, 4]  # confidence

        if debug:
            print(f"\n✓ Detection complete: {len(result)} faces found")
            print("="*80)

        return result


if __name__ == "__main__":
    # Test the ONNX MTCNN detector
    print("Testing ONNX MTCNN Detector...")

    detector = ONNXMTCNNDetector()

    # Test with a sample frame
    test_image_path = "Patient Data/IMG_0422.MOV"
    if os.path.exists(test_image_path):
        import cv2
        cap = cv2.VideoCapture(test_image_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            print(f"\nTesting on frame from {test_image_path}")
            start_time = time.time()
            bboxes = detector.detect(frame, debug=True)
            elapsed = (time.time() - start_time) * 1000

            print(f"\nResults:")
            print(f"  Detected faces: {len(bboxes)}")
            print(f"  Latency: {elapsed:.2f} ms")
            print(f"  FPS: {1000/elapsed:.2f}")

            if len(bboxes) > 0:
                print(f"\nFirst bbox: x={bboxes[0,0]:.2f}, y={bboxes[0,1]:.2f}, "
                      f"w={bboxes[0,2]:.2f}, h={bboxes[0,3]:.2f}, conf={bboxes[0,4]:.3f}")
    else:
        print(f"Test image not found: {test_image_path}")
