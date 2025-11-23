#!/usr/bin/env python3
"""
Test different flatten orders for eye CCNF to find the correct one.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING

def compute_response_with_order(patch_expert, image_patch, order='F'):
    """Compute response with specified flatten order."""
    if image_patch.shape != (patch_expert.height, patch_expert.width):
        image_patch = cv2.resize(image_patch, (patch_expert.width, patch_expert.height))

    features = image_patch.astype(np.float32)

    total_response = 0.0
    sum_alphas = 0.0

    for neuron in patch_expert.neurons:
        alpha = neuron['alpha']
        if abs(alpha) < 1e-4:
            continue
        sum_alphas += alpha

        weights = neuron['weights']
        if features.shape != weights.shape:
            features_resized = cv2.resize(features, (weights.shape[1], weights.shape[0]))
        else:
            features_resized = features

        # Use specified flatten order
        features_flat = features_resized.flatten(order)
        feature_mean = np.mean(features_flat)
        features_centered = features_flat - feature_mean
        feature_norm = np.linalg.norm(features_centered)

        if feature_norm > 1e-10:
            normalized_features = features_centered / feature_norm
        else:
            normalized_features = features_centered

        weights_flat = weights.flatten(order) * neuron['norm_weights']

        sigmoid_input = neuron['bias'] + np.dot(weights_flat, normalized_features)
        response = (2.0 * neuron['alpha']) / (1.0 + np.exp(-np.clip(sigmoid_input, -500, 500)))
        total_response += response

    if sum_alphas > 1e-10:
        return total_response / (2.0 * sum_alphas)
    return total_response

def compute_mean_shift(response_map, sigma=1.0):
    ws = response_map.shape[0]
    center = (ws - 1) / 2.0
    a_kde = -0.5 / (sigma * sigma)

    total_weight = mx = my = 0.0

    for ii in range(ws):
        for jj in range(ws):
            dist_sq = (ii - center)**2 + (jj - center)**2
            kde_weight = np.exp(a_kde * dist_sq)
            weight = response_map[ii, jj] * kde_weight
            total_weight += weight
            mx += weight * jj
            my += weight * ii

    if total_weight > 1e-10:
        return (mx / total_weight) - center, (my / total_weight) - center
    return 0.0, 0.0

def main():
    print("=" * 70)
    print("TEST FLATTEN ORDER FOR EYE CCNF")
    print("=" * 70)

    video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
    ret, frame = video.read()
    video.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model_dir = 'pyclnf/models'
    eye_model = HierarchicalEyeModel(model_dir)
    pdm = eye_model.pdm['left']

    CPP_LEFT_EYE_INPUT = {
        36: (392.1590, 847.6613), 37: (410.0039, 828.3166),
        38: (436.9223, 826.1841), 39: (461.9583, 842.8420),
        40: (438.4380, 850.4288), 41: (411.4089, 853.9998)
    }

    target_points = np.array([CPP_LEFT_EYE_INPUT[i] for i in [36, 37, 38, 39, 40, 41]])
    params = eye_model._fit_eye_shape(target_points, LEFT_EYE_MAPPING, 'left', main_rotation=None)
    eye_landmarks = pdm.params_to_landmarks_2d(params)

    ccnf = eye_model.ccnf['left']
    patch_experts = ccnf.get_all_patch_experts(eye_model.patch_scale)

    ws = eye_model.window_sizes[0]
    half_ws = ws // 2

    CPP_DELTA = {
        36: (+0.4725, +2.5932), 37: (+0.4878, +2.8680),
        38: (+0.2048, +2.5201), 39: (-0.5040, +1.1696),
        40: (-0.0311, +1.9948), 41: (+0.4704, +2.0950)
    }

    eye_to_main = {8: 36, 10: 37, 12: 38, 14: 39, 16: 40, 18: 41}

    print("\nTesting flatten orders: F (column-major) vs C (row-major)")
    print("With Y-flip fix applied to response map storage\n")

    for eye_idx in [8, 10, 12, 14, 16, 18]:
        if eye_idx not in patch_experts:
            continue

        main_idx = eye_to_main[eye_idx]
        cpp_dy = CPP_DELTA[main_idx][1]
        lm = eye_landmarks[eye_idx]
        x_int, y_int = int(round(lm[0])), int(round(lm[1]))
        pe = patch_experts[eye_idx]

        # Compute response maps with different orders
        f_map = np.zeros((ws, ws), dtype=np.float32)
        c_map = np.zeros((ws, ws), dtype=np.float32)

        for dy in range(-half_ws, half_ws + 1):
            for dx in range(-half_ws, half_ws + 1):
                px, py = x_int + dx, y_int + dy
                patch_size = pe.height
                half_patch = patch_size // 2

                y1 = max(0, py - half_patch)
                y2 = min(gray.shape[0], py + half_patch + 1)
                x1 = max(0, px - half_patch)
                x2 = min(gray.shape[1], px + half_patch + 1)

                if y2 - y1 < patch_size or x2 - x1 < patch_size:
                    patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
                    patch[:y2-y1, :x2-x1] = gray[y1:y2, x1:x2]
                else:
                    patch = gray[y1:y2, x1:x2]

                # Use -dy for Y-flip
                f_map[-dy + half_ws, dx + half_ws] = compute_response_with_order(pe, patch, 'F')
                c_map[-dy + half_ws, dx + half_ws] = compute_response_with_order(pe, patch, 'C')

        f_ms_x, f_ms_y = compute_mean_shift(f_map)
        c_ms_x, c_ms_y = compute_mean_shift(c_map)

        f_match = (f_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True
        c_match = (c_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True

        print(f"Eye_{eye_idx} (C++ Δy: {cpp_dy:+.4f}):")
        print(f"  flatten('F'): ms_y={f_ms_y:+.4f} {'✓' if f_match else '✗'}")
        print(f"  flatten('C'): ms_y={c_ms_y:+.4f} {'✓' if c_match else '✗'}")

        if c_match and not f_match:
            print(f"  *** ROW-MAJOR (C) ORDER BETTER ***")
        elif f_match and not c_match:
            print(f"  *** COLUMN-MAJOR (F) ORDER BETTER ***")
        print()

if __name__ == '__main__':
    main()
