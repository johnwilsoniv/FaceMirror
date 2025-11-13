#!/usr/bin/env python3
"""
Phase 1: Visual inspection of RNet inputs

Saves all 24×24 crops fed to RNet with their scores to see what's passing.
Are they full faces or just features (eyes, noses)?
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2
import os

print("=" * 80)
print("PHASE 1: INSPECT RNET INPUTS")
print("=" * 80)

# Create output directory
output_dir = "rnet_crops_inspection"
os.makedirs(output_dir, exist_ok=True)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print(f"\nTest image: {img_w}×{img_h}")
print(f"C++ Gold Standard: x=331.6, y=753.5, w=367.9, h=422.8")
print(f"Output directory: {output_dir}/")

# We'll modify the detection to save crops
# First, run PNet to get boxes
detector = PurePythonMTCNN_V2()

# Run PNet stage only to get the boxes before RNet
print("\n" + "-" * 80)
print("Running PNet stage...")
print("-" * 80)

# Build pyramid
min_face_size = 40
factor = 0.709
m = 12.0 / min_face_size
min_l = min(img_h, img_w) * m

scales = []
scale = m
while min_l >= 12:
    scales.append(scale)
    scale *= factor
    min_l *= factor

print(f"Image pyramid: {len(scales)} scales")

# Run PNet on all scales
total_boxes = []
for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    img_scaled = cv2.resize(img_float, (ws, hs))
    img_data = detector._preprocess(img_scaled)

    # Run PNet
    output = detector._run_pnet(img_data)
    output = output[0].transpose(1, 2, 0)

    logit_not_face = output[:, :, 0]
    logit_face = output[:, :, 1]
    prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
    reg_map = output[:, :, 2:6]

    boxes = detector._generate_bboxes(score_map, reg_map, scale, 0.6)

    if boxes.shape[0] > 0:
        keep = detector._nms(boxes, 0.5, 'Union')
        boxes = boxes[keep]
        total_boxes.append(boxes)

if len(total_boxes) == 0:
    print("No boxes from PNet!")
    exit(1)

total_boxes = np.vstack(total_boxes)
print(f"PNet: {total_boxes.shape[0]} boxes before cross-scale NMS")

# Cross-scale NMS
keep = detector._nms(total_boxes, 0.7, 'Union')
total_boxes = total_boxes[keep]
print(f"PNet: {total_boxes.shape[0]} boxes after cross-scale NMS")

# Apply PNet regression
total_boxes = detector._apply_bbox_regression(total_boxes)
print(f"PNet: Applied bbox regression")

# Square the boxes
total_boxes = detector._square_bbox(total_boxes)
print(f"PNet: Squared bboxes")

print(f"\n{total_boxes.shape[0]} boxes will be fed to RNet")

# Now extract and save all RNet crops
print("\n" + "-" * 80)
print("Extracting RNet crops...")
print("-" * 80)

crops = []
crop_info = []

for i in range(total_boxes.shape[0]):
    x1 = int(max(0, total_boxes[i, 0]))
    y1 = int(max(0, total_boxes[i, 1]))
    x2 = int(min(img_w, total_boxes[i, 2]))
    y2 = int(min(img_h, total_boxes[i, 3]))

    if x2 <= x1 or y2 <= y1:
        continue

    # Extract crop
    face = img_float[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (24, 24))

    # Run RNet on this crop
    face_data = detector._preprocess(face_resized)
    output = detector._run_rnet(face_data)

    # Calculate score (output is 1D: [6 values])
    score = 1.0 / (1.0 + np.exp(output[0] - output[1]))

    crops.append(face_resized.astype(np.uint8))
    crop_info.append({
        'idx': i,
        'bbox': (x1, y1, x2, y2),
        'size': (x2 - x1, y2 - y1),
        'score': score,
        'crop': face_resized.astype(np.uint8)
    })

print(f"Extracted {len(crops)} crops (24×24)")

# Sort by score descending
crop_info.sort(key=lambda x: x['score'], reverse=True)

# Save individual crops
print("\n" + "-" * 80)
print("Saving individual crops...")
print("-" * 80)

for i, info in enumerate(crop_info[:50]):  # Top 50
    filename = f"{output_dir}/crop_{i:03d}_score{info['score']:.4f}_size{info['size'][0]}x{info['size'][1]}.jpg"
    cv2.imwrite(filename, info['crop'])

print(f"Saved top 50 crops to {output_dir}/")

# Create visual summary grid
print("\n" + "-" * 80)
print("Creating visual summary...")
print("-" * 80)

# Grid: 10 columns, as many rows as needed
cols = 10
rows = (len(crop_info) + cols - 1) // cols

# Each cell: 24×24 crop + 20px text below = 24×44
cell_w = 100
cell_h = 120
grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255

for i, info in enumerate(crop_info):
    row = i // cols
    col = i % cols

    if row >= rows:
        break

    # Resize crop to 100×100 for visibility
    crop_large = cv2.resize(info['crop'], (100, 100))

    y1 = row * cell_h
    y2 = y1 + 100
    x1 = col * cell_w
    x2 = x1 + 100

    grid[y1:y2, x1:x2] = crop_large

    # Add score text
    score_text = f"{info['score']:.3f}"
    size_text = f"{info['size'][0]}x{info['size'][1]}"

    cv2.putText(grid, score_text, (x1 + 5, y1 + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.putText(grid, size_text, (x1 + 5, y1 + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)

cv2.imwrite(f"{output_dir}/SUMMARY_all_crops_by_score.jpg", grid)
print(f"Saved visual summary: {output_dir}/SUMMARY_all_crops_by_score.jpg")

# Analyze score distribution
print("\n" + "=" * 80)
print("RNET SCORE ANALYSIS")
print("=" * 80)

scores = [info['score'] for info in crop_info]
sizes = [info['size'][0] for info in crop_info]

print(f"\nTotal crops: {len(scores)}")
print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
print(f"Mean score: {np.mean(scores):.4f}")

# Score thresholds
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
for thresh in thresholds:
    count = sum(1 for s in scores if s >= thresh)
    print(f"  Scores >= {thresh}: {count} ({count*100/len(scores):.1f}%)")

print(f"\nSize range: [{min(sizes)}, {max(sizes)}] px")
print(f"Mean size: {np.mean(sizes):.1f} px")

# Show top 10 by score
print(f"\nTop 10 crops by RNet score:")
for i in range(min(10, len(crop_info))):
    info = crop_info[i]
    x1, y1, x2, y2 = info['bbox']
    w, h = info['size']
    score = info['score']
    print(f"  #{i+1}: Score={score:.4f}, Size={w}×{h}px, Bbox=({x1}, {y1})")

# Analyze what passes threshold 0.7
passing = [info for info in crop_info if info['score'] >= 0.7]
if passing:
    print(f"\n{len(passing)} crops pass RNet threshold (0.7):")
    for i, info in enumerate(passing):
        w, h = info['size']
        x1, y1, x2, y2 = info['bbox']
        print(f"  #{i+1}: Score={info['score']:.4f}, Size={w}×{h}px, Location=({x1}, {y1})")

        # Check if this overlaps with gold standard face
        gold_x1, gold_y1 = 331.6, 753.5
        gold_x2, gold_y2 = gold_x1 + 367.9, gold_y1 + 422.8

        # Calculate IoU
        inter_x1 = max(x1, gold_x1)
        inter_y1 = max(y1, gold_y1)
        inter_x2 = min(x2, gold_x2)
        inter_y2 = min(y2, gold_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        gold_area = (gold_x2 - gold_x1) * (gold_y2 - gold_y1)
        crop_area = (x2 - x1) * (y2 - y1)
        union_area = gold_area + crop_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        if iou > 0.01:
            print(f"       → IoU with gold face: {iou:.4f} ({iou*100:.1f}%)")
            if iou < 0.3:
                print(f"       → This is likely a FEATURE (eye/nose), not full face!")
        else:
            print(f"       → Does NOT overlap with gold face - wrong location!")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print(f"""
1. Open: {output_dir}/SUMMARY_all_crops_by_score.jpg
   - Look at the top-scoring crops
   - Are they full faces or just features (eyes, noses, eyebrows)?

2. Check individual crops in {output_dir}/
   - Filenames show score and original size
   - Compare high-scoring small crops vs low-scoring large crops

3. Key Question to Answer:
   - Are the high-scoring (>0.7) crops FULL FACES or PARTIAL FEATURES?
   - If they're features, that explains why final detections are tiny!
""")
