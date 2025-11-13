#!/usr/bin/env python3
"""
Debug PNet proposals to understand why the face detection is getting filtered out.
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

print("="*80)
print("PNET PROPOSAL DEBUG")
print("="*80)

# Load test image
img = cv2.imread('cpp_mtcnn_test.jpg')
print(f"\nTest image shape: {img.shape}")
img_h, img_w = img.shape[:2]

# Create detector
detector = CPPMTCNNDetector()
detector.min_face_size = 40  # Match C++

# Manually run just the PNet stage
img_float = img.astype(np.float32)

# Build scales (matching C++ FaceDetectorMTCNN.cpp:849-856)
min_size = detector.min_face_size
m = 12.0 / min_size
min_l = min(img_h, img_w) * m
pyramid_factor = 0.709

scales = []
scale = m
while min_l >= 12:
    scales.append(scale)
    scale *= pyramid_factor
    min_l *= pyramid_factor

print(f"\nScales: {scales}")
print(f"Number of scales: {len(scales)}")

# Run PNet for each scale and collect all proposals
all_proposals = []

for i, scale in enumerate(scales):
    hs = int(np.ceil(img_h * scale))
    ws = int(np.ceil(img_w * scale))

    # Resize and preprocess
    img_scaled = cv2.resize(img_float, (ws, hs))
    img_data = detector._preprocess(img_scaled)

    # Run PNet
    output = detector.pnet.run(None, {'input': img_data})[0]
    output = output[0].transpose(1, 2, 0)  # (H, W, 6)

    # Calculate probabilities
    logit_not_face = output[:, :, 0]
    logit_face = output[:, :, 1]
    prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

    # Generate bboxes (matching C++ generate_bounding_boxes)
    stride = 2
    face_support = 12
    threshold = detector.thresholds[0]

    for x in range(prob_face.shape[1]):
        for y in range(prob_face.shape[0]):
            if prob_face[y, x] >= threshold:
                # Match C++ FaceDetectorMTCNN.cpp:771-777
                min_x = int((stride * x + 1) / scale)
                max_x = int((stride * x + face_support) / scale)
                min_y = int((stride * y + 1) / scale)
                max_y = int((stride * y + face_support) / scale)

                w = max_x - min_x
                h = max_y - min_y

                # Store as [x1, y1, x2, y2, score, scale_idx]
                all_proposals.append([min_x, min_y, max_x, max_y, prob_face[y, x], i])

    print(f"Scale {i}: {scale:.3f}, Size: {ws}x{hs}, Proposals: {np.sum(prob_face >= threshold)}")

all_proposals = np.array(all_proposals)

print(f"\n{'='*80}")
print(f"BEFORE ANY NMS:")
print(f"{'='*80}")
print(f"Total proposals: {len(all_proposals)}")

# Find face region proposals (y=300-900, w>200)
face_mask = ((all_proposals[:, 1] >= 300) & (all_proposals[:, 1] <= 900) &
             ((all_proposals[:, 2] - all_proposals[:, 0]) > 200))
face_proposals = all_proposals[face_mask]

print(f"\nProposals in FACE REGION (y=300-900, w>200): {len(face_proposals)}")
if len(face_proposals) > 0:
    # Sort by score
    sorted_idx = np.argsort(face_proposals[:, 4])[::-1]
    for idx in sorted_idx[:5]:
        box = face_proposals[idx]
        w = box[2] - box[0]
        h = box[3] - box[1]
        print(f"  x1={box[0]:.0f}, y1={box[1]:.0f}, w={w:.0f}, h={h:.0f}, "
              f"score={box[4]:.6f}, scale={int(box[5])}")

# Check top 20 overall proposals
print(f"\nTop 20 proposals OVERALL (sorted by score):")
sorted_idx = np.argsort(all_proposals[:, 4])[::-1][:20]
for rank, idx in enumerate(sorted_idx):
    box = all_proposals[idx]
    w = box[2] - box[0]
    h = box[3] - box[1]
    is_face = '← FACE!' if face_mask[idx] else ''
    print(f"  #{rank+1}: x1={box[0]:.0f}, y1={box[1]:.0f}, w={w:.0f}, h={h:.0f}, "
          f"score={box[4]:.6f}, scale={int(box[5])} {is_face}")

# Now apply within-scale NMS
print(f"\n{'='*80}")
print(f"AFTER WITHIN-SCALE NMS (threshold=0.5):")
print(f"{'='*80}")

proposals_by_scale = [[] for _ in scales]
for prop in all_proposals:
    scale_idx = int(prop[5])
    proposals_by_scale[scale_idx].append(prop[:5])  # Remove scale_idx

all_after_scale_nms = []
for i, props in enumerate(proposals_by_scale):
    if len(props) > 0:
        props = np.array(props)
        keep = detector._nms(props, 0.5, 'Union')
        props_kept = props[keep]
        all_after_scale_nms.extend(props_kept)
        print(f"Scale {i}: {len(props)} → {len(props_kept)} boxes")

all_after_scale_nms = np.array(all_after_scale_nms)
print(f"Total after within-scale NMS: {len(all_after_scale_nms)}")

# Check face proposals after within-scale NMS
face_mask = ((all_after_scale_nms[:, 1] >= 300) & (all_after_scale_nms[:, 1] <= 900) &
             ((all_after_scale_nms[:, 2] - all_after_scale_nms[:, 0]) > 200))
print(f"Face region proposals remaining: {np.sum(face_mask)}")

# Now apply cross-scale NMS
print(f"\n{'='*80}")
print(f"AFTER CROSS-SCALE NMS (threshold=0.7):")
print(f"{'='*80}")

keep = detector._nms(all_after_scale_nms, 0.7, 'Union')
final_proposals = all_after_scale_nms[keep]

print(f"Total after cross-scale NMS: {len(final_proposals)}")

# Check if face is still there
face_mask = ((final_proposals[:, 1] >= 300) & (final_proposals[:, 1] <= 900) &
             ((final_proposals[:, 2] - final_proposals[:, 0]) > 200))
print(f"Face region proposals remaining: {np.sum(face_mask)}")

if np.sum(face_mask) > 0:
    face_final = final_proposals[face_mask]
    print(f"\nFace proposals that survived:")
    for box in face_final:
        w = box[2] - box[0]
        h = box[3] - box[1]
        print(f"  x1={box[0]:.0f}, y1={box[1]:.0f}, w={w:.0f}, h={h:.0f}, score={box[4]:.6f}")
else:
    print(f"\n⚠ WARNING: Face was FILTERED OUT during NMS!")
    print(f"\nChecking top 10 final proposals:")
    sorted_idx = np.argsort(final_proposals[:, 4])[::-1][:10]
    for rank, idx in enumerate(sorted_idx):
        box = final_proposals[idx]
        w = box[2] - box[0]
        h = box[3] - box[1]
        print(f"  #{rank+1}: x1={box[0]:.0f}, y1={box[1]:.0f}, w={w:.0f}, h={h:.0f}, score={box[4]:.6f}")

print(f"\n{'='*80}")
print("DIAGNOSIS:")
print(f"{'='*80}")
print("""
If face was filtered out, possible reasons:
1. IoU overlap with higher-scoring but wrong detections
2. NMS threshold too aggressive for this image
3. Bbox generation coordinates differ from C++
4. Scale calculation differs from C++
""")
