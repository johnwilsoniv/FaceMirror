"""
Visualize ALL raw MTCNN detections before face selection.

Shows:
1. Python MTCNN: All detected faces + largest face (by width)
2. C++ MTCNN: Final selected face (from debug output)
3. Comparison between Python and C++ selected faces
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector
import csv
from pathlib import Path


def visualize_all_mtcnn_detections():
    """Visualize all MTCNN detections and face selection."""

    # Load test image
    test_image = "calibration_frames/patient1_frame1.jpg"
    img = cv2.imread(test_image)

    if img is None:
        print(f"Error: Could not load {test_image}")
        return

    print("=" * 80)
    print("MTCNN Raw Detections Visualization")
    print("=" * 80)

    # ========================================================================
    # PYTHON MTCNN: Get ALL raw detections
    # ========================================================================
    print("\n[1] Python MTCNN - All Raw Detections:")
    print("-" * 80)

    detector = CPPMTCNNDetector()
    bboxes, landmarks = detector.detect(img)

    print(f"Total faces detected: {len(bboxes)}")

    # Display all detections
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        area = w * h
        print(f"  Face {i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, area={area:.0f}")

    # Select largest face (by width) - matching C++ behavior
    if len(bboxes) > 0:
        widths = [bbox[2] for bbox in bboxes]
        largest_idx = np.argmax(widths)
        selected_python_bbox = bboxes[largest_idx]

        print(f"\nFace Selection (by largest width):")
        print(f"  Selected: Face {largest_idx + 1} (width={selected_python_bbox[2]:.1f})")
    else:
        selected_python_bbox = None
        largest_idx = -1

    # ========================================================================
    # C++ MTCNN: Get final selected face from debug output
    # ========================================================================
    print("\n[2] C++ MTCNN - Final Selected Face:")
    print("-" * 80)

    cpp_debug_file = Path("/tmp/mtcnn_debug.csv")
    cpp_bbox = None

    if cpp_debug_file.exists():
        with open(cpp_debug_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['bbox_x'])
                y = float(row['bbox_y'])
                w = float(row['bbox_w'])
                h = float(row['bbox_h'])
                conf = float(row['confidence'])
                cpp_bbox = (x, y, w, h)
                print(f"  Selected: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, conf={conf:.4f}")
                break
    else:
        print("  Warning: C++ debug output not found at /tmp/mtcnn_debug.csv")
        print("  Run C++ FeatureExtraction first to generate debug output")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    if selected_python_bbox is not None and cpp_bbox is not None:
        print("\n[3] Comparison: Python vs C++ Selected Face:")
        print("-" * 80)

        # Calculate IoU
        def calculate_iou(bbox1, bbox2):
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2

            x1_max = x1 + w1
            y1_max = y1 + h1
            x2_max = x2 + w2
            y2_max = y2 + h2

            # Intersection
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1_max, x2_max)
            yi2 = min(y1_max, y2_max)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            # Union
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area if union_area > 0 else 0

        iou = calculate_iou(selected_python_bbox, cpp_bbox)

        print(f"  Python selected: Face {largest_idx + 1}")
        print(f"  Python bbox: x={selected_python_bbox[0]:.1f}, y={selected_python_bbox[1]:.1f}, "
              f"w={selected_python_bbox[2]:.1f}, h={selected_python_bbox[3]:.1f}")
        print(f"  C++ bbox:    x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}, "
              f"w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")
        print(f"  IoU: {iou:.1%}")

        if iou > 0.5:
            print(f"  ✓ MATCH: Python selects same face as C++ (IoU={iou:.1%})")
        else:
            print(f"  ✗ MISMATCH: Python selects different face (IoU={iou:.1%})")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n[4] Creating Visualization...")
    print("-" * 80)

    # Create visualization with all detections
    vis = img.copy()
    h_vis, w_vis = vis.shape[:2]

    # Draw all Python MTCNN detections
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox

        # Different colors for different faces
        if i == largest_idx:
            # Largest face (selected) - GREEN
            color = (0, 255, 0)
            thickness = 3
            label = f"Python Selected (Face {i+1})"
        else:
            # Other faces - YELLOW
            color = (0, 255, 255)
            thickness = 2
            label = f"Python Face {i+1}"

        # Draw bbox
        cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)

        # Add label
        label_y = int(y - 10) if y > 30 else int(y + h + 20)
        cv2.putText(vis, label, (int(x), label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw landmarks
        if landmarks is not None and i < len(landmarks):
            for pt in landmarks[i]:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, color, -1)

        # Add size info
        size_text = f"w={w:.0f}"
        cv2.putText(vis, size_text, (int(x), int(y+h+40)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw C++ selected face (if available)
    if cpp_bbox is not None:
        x, y, w, h = cpp_bbox
        color = (255, 0, 255)  # MAGENTA
        thickness = 3

        # Draw bbox
        cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)

        # Add label
        label = "C++ Selected"
        label_y = int(y + h + 60) if y + h + 60 < h_vis - 20 else int(y - 30)
        cv2.putText(vis, label, (int(x), label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add title and legend
    title = f"MTCNN Raw Detections: {len(bboxes)} faces detected"
    cv2.putText(vis, title, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Legend
    legend_y = 70
    cv2.putText(vis, "GREEN = Python Selected (Largest Width)", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, "YELLOW = Python Other Detections", (10, legend_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, "MAGENTA = C++ Selected", (10, legend_y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Save visualization
    output_path = "mtcnn_all_detections_comparison.jpg"
    cv2.imwrite(output_path, vis)

    print(f"✓ Saved visualization to: {output_path}")
    print(f"  Image size: {w_vis}x{h_vis}")
    print(f"  Total detections shown: {len(bboxes)}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Python MTCNN: {len(bboxes)} faces detected")
    if selected_python_bbox is not None:
        print(f"Python selected: Face {largest_idx + 1} (largest width={selected_python_bbox[2]:.1f})")
    print(f"C++ MTCNN: {'1 face selected' if cpp_bbox is not None else 'No debug output'}")

    if selected_python_bbox is not None and cpp_bbox is not None:
        if iou > 0.5:
            print(f"✓ Face selection MATCHES (IoU={iou:.1%})")
        else:
            print(f"✗ Face selection MISMATCH (IoU={iou:.1%})")

    print("\n✓ Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    visualize_all_mtcnn_detections()
